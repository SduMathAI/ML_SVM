from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVC


FILENAME_PATTERN = (
    r"^(?P<source>[^+]+)\+(?P<subject>[^+]+)\+(?P<label>[A-Z]+)-m(?P<month>\d+)-"
    r"(?P<scanner>.+?)\.nii\.gz$"
)


@dataclass
class ExperimentConfig:
    """描述一次 MRI PCA+SVM 实验所需配置的对象。"""

    data_dir: Path
    output_dir: Path
    cache_dir: Path
    labels: list[str]
    month: str
    target_shape: tuple[int, int, int]
    pca_components: int
    test_size: float
    random_state: int
    cv_folds: int
    clip_percentile: float
    n_jobs: int


def parse_args() -> ExperimentConfig:
    """解析命令行参数，并整理成实验配置对象。"""

    parser = argparse.ArgumentParser(description="Run MRI PCA+SVM experiments.")
    parser.add_argument("--data-dir", default="processed", help="Directory containing .nii.gz MRI files.")
    parser.add_argument("--output-dir", required=True, help="Directory used to save results.")
    parser.add_argument("--cache-dir", default="artifacts/feature_cache", help="Feature cache directory.")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels to include, e.g. AD NORMAL.")
    parser.add_argument("--month", default="00", help="Follow-up month, e.g. 00, 06, 12.")
    parser.add_argument("--target-shape", nargs=3, type=int, default=[32, 32, 32], help="Downsampled MRI shape.")
    parser.add_argument("--pca-components", type=int, default=100, help="Number of PCA components.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size at subject level.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for grid search.")
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="Upper percentile used for intensity clipping on non-zero voxels.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for grid search.")
    args = parser.parse_args()

    return ExperimentConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        labels=list(args.labels),
        month=str(args.month).zfill(2),
        target_shape=tuple(args.target_shape),
        pca_components=args.pca_components,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        clip_percentile=args.clip_percentile,
        n_jobs=args.n_jobs,
    )


def ensure_dir(path: Path) -> None:
    """如果目录不存在，就先创建出来。"""

    path.mkdir(parents=True, exist_ok=True)


def index_dataset(data_dir: Path) -> pd.DataFrame:
    """把 MRI 文件名解析成结构化的数据索引表。"""

    records: list[dict[str, str]] = []
    for path in sorted(data_dir.glob("*.nii.gz")):
        match = re.match(FILENAME_PATTERN, path.name)
        if not match:
            continue
        record = match.groupdict()
        record["path"] = str(path.resolve())
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No .nii.gz files parsed from {data_dir}")
    df = pd.DataFrame(records)
    df["month"] = df["month"].astype(str).str.zfill(2)
    return df


def select_subset(df: pd.DataFrame, labels: Iterable[str], month: str) -> pd.DataFrame:
    """筛选出满足标签组合和时间点要求的样本子集。"""

    labels = list(labels)
    subset = df[df["label"].isin(labels) & (df["month"] == month)].copy()
    if subset.empty:
        raise ValueError(f"No samples found for labels={labels} and month={month}")
    subset = subset.sort_values(["label", "subject", "path"]).reset_index(drop=True)
    return subset


def preprocess_volume(path: Path, target_shape: tuple[int, int, int], clip_percentile: float) -> np.ndarray:
    """读取单个 MRI，并完成标准化、缩放和展平。"""

    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    mask = data != 0

    if mask.any():
        nonzero = data[mask]
        low = np.percentile(nonzero, 0.5)
        high = np.percentile(nonzero, clip_percentile)
        if high <= low:
            high = float(nonzero.max())
        data = np.clip(data, low, high)
        mean = float(nonzero.mean())
        std = float(nonzero.std())
        if std < 1e-6:
            std = 1.0
        data = (data - mean) / std
        data[~mask] = 0.0

    factors = tuple(t / s for t, s in zip(target_shape, data.shape))
    resized = zoom(data, zoom=factors, order=1)
    return resized.astype(np.float32).ravel()


def feature_cache_path(cache_dir: Path, subset: pd.DataFrame, config: ExperimentConfig) -> Path:
    """根据样本列表和预处理设置生成特征缓存文件路径。"""

    ensure_dir(cache_dir)
    digest_source = "|".join(subset["path"].tolist())
    digest_source += f"|shape={config.target_shape}|clip={config.clip_percentile}"
    digest = hashlib.md5(digest_source.encode("utf-8")).hexdigest()[:12]
    label_name = "_".join(config.labels).lower()
    return cache_dir / f"{label_name}_m{config.month}_{digest}.npz"


def build_feature_matrix(subset: pd.DataFrame, config: ExperimentConfig) -> np.ndarray:
    """把当前样本子集中的 MRI 全部转换成特征矩阵 X。"""

    cache_path = feature_cache_path(config.cache_dir, subset, config)
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["X"]

    features = []
    total = len(subset)
    for index, row in enumerate(subset.itertuples(index=False), start=1):
        if index == 1 or index % 25 == 0 or index == total:
            print(f"[feature] {index}/{total} {Path(row.path).name}")
        feature = preprocess_volume(Path(row.path), config.target_shape, config.clip_percentile)
        features.append(feature)

    X = np.vstack(features).astype(np.float32)
    np.savez_compressed(cache_path, X=X)
    return X


def split_subjects(subset: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.Series, pd.Series]:
    """按受试者而不是按图像划分训练集和测试集，避免数据泄漏。"""

    subject_table = subset[["subject", "label"]].drop_duplicates().reset_index(drop=True)
    train_subjects, test_subjects = train_test_split(
        subject_table["subject"],
        test_size=test_size,
        random_state=random_state,
        stratify=subject_table["label"],
    )
    return train_subjects, test_subjects


def safe_pca_components(requested: int, X_train: np.ndarray) -> int:
    """把 PCA 维度限制在当前训练矩阵允许的合法范围内。"""

    max_components = min(X_train.shape[0] - 1, X_train.shape[1])
    if max_components < 1:
        raise ValueError("Training set is too small for PCA.")
    return min(requested, max_components)


def pipeline_for(kernel: str, pca_components: int, random_state: int) -> Pipeline:
    """构建统一使用的 StandardScaler -> PCA -> SVM 流水线。"""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components, svd_solver="randomized", random_state=random_state)),
            ("svm", SVC(kernel=kernel, class_weight="balanced", decision_function_shape="ovr")),
        ]
    )


def param_grid_for(kernel: str) -> list[dict[str, list[object]]]:
    """返回指定 SVM 核函数对应的超参数搜索空间。"""

    if kernel == "linear":
        return [{"svm__C": [0.1, 1, 10]}]
    return [{"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.1, 0.01]}]


def decision_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    positive_label: int | None = None,
) -> float | None:
    """根据二分类或多分类场景计算 ROC-AUC。"""

    if len(class_names) == 2:
        if positive_label is None:
            raise ValueError("positive_label must be provided for binary AUC.")
        binary_scores = scores if positive_label == 1 else -scores
        return float(roc_auc_score(y_true == positive_label, binary_scores))

    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    if y_bin.ndim == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])

    try:
        return float(roc_auc_score(y_bin, scores, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def save_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path, title: str) -> None:
    """把混淆矩阵保存成热力图图片。"""

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_roc_curve(y_true: np.ndarray, scores: np.ndarray, output_path: Path, positive_label: int, label_name: str) -> None:
    """为指定正类保存二分类 ROC 曲线图。"""

    binary_scores = scores if positive_label == 1 else -scores
    fpr, tpr, _ = roc_curve(y_true == positive_label, binary_scores)
    auc_value = roc_auc_score(y_true == positive_label, binary_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (positive={label_name})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def fit_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    config: ExperimentConfig,
    output_dir: Path,
) -> pd.DataFrame:
    """训练线性核和 RBF 核 SVM，评估并保存全部结果。"""

    results = []
    pca_components = safe_pca_components(config.pca_components, X_train)
    cv = StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    for kernel in ["linear", "rbf"]:
        print(f"[train] kernel={kernel}")
        pipeline = pipeline_for(kernel, pca_components, config.random_state)
        grid = GridSearchCV(
            estimator=pipeline,  # 标准化 + PCA + SVM 的完整流水线
            param_grid=param_grid_for(kernel),  # 当前核函数对应的搜索空间
            scoring="balanced_accuracy",  # 用类不平衡更稳健的指标选参
            cv=cv,  # 训练集内部的分组交叉验证器
            n_jobs=config.n_jobs,  # 并行搜索时的 worker 数
            refit=True,  # 用最佳参数在全部训练集上重训
            verbose=1,  # 输出搜索进度，便于观察运行过程
        )
        grid.fit(X_train, y_train, groups=groups_train)
        best_model: Pipeline = grid.best_estimator_
        predictions = best_model.predict(X_test)
        scores = best_model.decision_function(X_test)
        positive_label = 0 if len(class_names) == 2 else None
        if len(class_names) == 2:
            precision = float(precision_score(y_test, predictions, pos_label=positive_label, average="binary", zero_division=0))
            recall = float(recall_score(y_test, predictions, pos_label=positive_label, average="binary", zero_division=0))
            f1 = float(f1_score(y_test, predictions, pos_label=positive_label, average="binary", zero_division=0))
        else:
            precision = float(precision_score(y_test, predictions, average="macro", zero_division=0))
            recall = float(recall_score(y_test, predictions, average="macro", zero_division=0))
            f1 = float(f1_score(y_test, predictions, average="macro", zero_division=0))

        metrics = {
            "model": kernel,
            "best_params": grid.best_params_,
            "cv_best_score": float(grid.best_score_),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": decision_auc(y_test, scores, class_names, positive_label=positive_label),
            "support_vectors_total": int(np.sum(best_model.named_steps["svm"].n_support_)),
            "support_vectors_per_class": best_model.named_steps["svm"].n_support_.tolist(),
            "pca_components": pca_components,
            "pca_explained_variance": float(best_model.named_steps["pca"].explained_variance_ratio_.sum()),
        }
        results.append(metrics)

        report = classification_report(y_test, predictions, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, predictions)

        (output_dir / f"best_params_{kernel}.json").write_text(
            json.dumps(grid.best_params_, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / f"classification_report_{kernel}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / f"confusion_matrix_{kernel}.csv")
        save_confusion_matrix(cm, class_names, output_dir / f"confusion_matrix_{kernel}.png", f"Confusion Matrix ({kernel})")

        if len(class_names) == 2:
            save_roc_curve(
                y_test,
                scores,
                output_dir / f"roc_curve_{kernel}.png",
                positive_label=positive_label,
                label_name=class_names[positive_label],
            )

        joblib.dump(best_model, output_dir / f"best_model_{kernel}.joblib")

    summary = pd.DataFrame(results)
    summary["best_params"] = summary["best_params"].apply(lambda item: json.dumps(item, ensure_ascii=False))
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    return summary


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """执行完整实验流程：从数据索引到最终指标保存。"""

    ensure_dir(config.output_dir)
    ensure_dir(config.cache_dir)

    dataset_index = index_dataset(config.data_dir)
    dataset_index.to_csv(config.output_dir / "dataset_index.csv", index=False)

    subset = select_subset(dataset_index, config.labels, config.month)
    subset.to_csv(config.output_dir / "selected_samples.csv", index=False)

    X = build_feature_matrix(subset, config)

    class_names = list(config.labels)
    label_to_index = {label: index for index, label in enumerate(class_names)}
    y = subset["label"].map(label_to_index).to_numpy()

    train_subjects, test_subjects = split_subjects(subset, config.test_size, config.random_state)
    split_column = np.where(subset["subject"].isin(train_subjects), "train", "test")
    subset_with_split = subset.copy()
    subset_with_split["split"] = split_column
    subset_with_split.to_csv(config.output_dir / "train_test_split.csv", index=False)

    train_mask = subset_with_split["split"] == "train"
    test_mask = ~train_mask

    X_train = X[train_mask.to_numpy()]
    X_test = X[test_mask.to_numpy()]
    y_train = y[train_mask.to_numpy()]
    y_test = y[test_mask.to_numpy()]
    groups_train = subset_with_split.loc[train_mask, "subject"].to_numpy()

    summary = fit_and_evaluate(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        config=config,
        output_dir=config.output_dir,
    )

    metadata = {
        "labels": config.labels,
        "month": config.month,
        "target_shape": config.target_shape,
        "pca_components_requested": config.pca_components,
        "test_size": config.test_size,
        "random_state": config.random_state,
        "cv_folds": config.cv_folds,
        "clip_percentile": config.clip_percentile,
        "n_samples": int(len(subset)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "subjects_train": int(train_subjects.nunique()),
        "subjects_test": int(test_subjects.nunique()),
        "class_distribution": subset["label"].value_counts().sort_index().to_dict(),
    }
    (config.output_dir / "experiment_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    return summary


def main() -> None:
    """通用 MRI PCA+SVM 实验脚本的命令行入口。"""

    config = parse_args()
    run_experiment(config)


if __name__ == "__main__":
    main()
