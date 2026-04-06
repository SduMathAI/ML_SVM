from __future__ import annotations

"""第 6 章讲课脚本共用的底层工具函数。

这个文件的存在目的，是让 `lecture_topic01-06.py` 尽量短、尽量好讲。
如果把所有数据处理、训练、评估代码都写进每个 topic 脚本里，
那课堂脚本会非常长，也不利于理解。

因此，我们把重复步骤统一放到这里：

1. 从 `processed/` 读取并索引 MRI 文件；
2. 为某个课题选择标签子集；
3. 把 MRI 预处理成特征向量；
4. 按受试者级别划分训练集和测试集；
5. 训练 SVM 并计算指标；
6. 保存讲课需要的表格和图像。

课堂上你一般不需要逐行解释这个文件。
它更像是 `lecture_topic01-06` 背后的“工具箱”。
"""

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


FILENAME_PATTERN = re.compile(
    r"^(?P<source>[^+]+)\+(?P<subject>[^+]+)\+(?P<label>[A-Z]+)-m(?P<month>\d+)-(?P<scanner>.+?)\.nii\.gz$"
)


@dataclass
class LectureConfig:
    """讲课脚本共用的小型配置对象。

    这里的默认值尽量与主项目实验保持一致，
    这样讲稿中的数字和主实验结果就不会打架。
    """

    data_dir: Path = Path("processed")
    cache_dir: Path = Path("artifacts/feature_cache")
    month: str = "00"
    target_shape: tuple[int, int, int] = (32, 32, 32)
    pca_components: int = 100
    clip_percentile: float = 99.5
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1


def ensure_dir(path: Path) -> None:
    """如果目录不存在，就先创建出来。"""

    path.mkdir(parents=True, exist_ok=True)


def index_dataset(data_dir: Path) -> pd.DataFrame:
    """扫描 `processed/` 并把文件名解析成结构化表格。

    每一行对应一个 MRI 文件。
    由于文件名中本身已经包含受试者编号、标签、时间点等信息，
    所以我们先统一解析一次，后面所有课题都直接复用。
    """

    records: list[dict[str, str]] = []
    for path in sorted(data_dir.glob("*.nii.gz")):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            continue
        record = match.groupdict()
        record["path"] = str(path.resolve())
        records.append(record)

    if not records:
        raise FileNotFoundError(f"No MRI files found in {data_dir}")

    df = pd.DataFrame(records)
    df["month"] = df["month"].astype(str).str.zfill(2)
    return df


def select_subset(df: pd.DataFrame, labels: Iterable[str], month: str) -> pd.DataFrame:
    """只保留当前课题真正需要的数据行。

    例如：
    - labels = ["AD", "NORMAL"]
    - month = "00"

    这就表示：只保留 AD 和 NORMAL 两类的 m00 基线 MRI。
    """

    labels = list(labels)
    subset = df[df["label"].isin(labels) & (df["month"] == month)].copy()
    if subset.empty:
        raise ValueError(f"No samples found for labels={labels}, month={month}")
    return subset.sort_values(["label", "subject", "path"]).reset_index(drop=True)


def preprocess_volume(path: Path, target_shape: tuple[int, int, int], clip_percentile: float) -> np.ndarray:
    """把一个 MRI 体数据转换成一维特征向量。

    这里的预处理故意设计得比较轻量，
    因为这个项目的重点是讲 SVM，不是做工业级医学影像系统。

    主要步骤是：
    1. 读取 MRI；
    2. 忽略背景零值体素；
    3. 裁剪极端强度值；
    4. 对单个受试者做标准化；
    5. 下采样到统一的 3D 尺寸；
    6. 展平成一个特征向量。
    """

    # 这里只保留一个足够简单的预处理流程，方便课堂上解释：
    # 1. 读取 MRI
    # 2. 裁剪特别大的强度值
    # 3. 对单个受试者做标准化
    # 4. 下采样到一个较小的 3D 网格
    # 5. 展平成特征向量
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    mask = data != 0

    if mask.any():
        # 背景通常都是 0，所以强度范围只从脑区的非零体素里估计。
        nonzero = data[mask]
        low = np.percentile(nonzero, 0.5)
        high = np.percentile(nonzero, clip_percentile)
        if high <= low:
            high = float(nonzero.max())

        # 裁剪异常值，避免少数极端体素把整体尺度带偏。
        data = np.clip(data, low, high)
        mean = float(nonzero.mean())
        std = float(nonzero.std())
        if std < 1e-6:
            std = 1.0

        # 每个受试者单独标准化，然后再把背景位置恢复成 0。
        data = (data - mean) / std
        data[~mask] = 0.0

    # 把所有 MRI 统一到同一个尺寸，这样进入 PCA 和 SVM 前，
    # 每个样本的特征长度才会一致。
    factors = tuple(target / size for target, size in zip(target_shape, data.shape))
    resized = zoom(data, zoom=factors, order=1)
    return resized.astype(np.float32).ravel()


def feature_cache_path(cache_dir: Path, labels: list[str], month: str, target_shape: tuple[int, int, int]) -> Path:
    """生成预处理特征缓存文件的路径。"""

    ensure_dir(cache_dir)
    label_name = "_".join(labels).lower()
    shape_name = "x".join(str(v) for v in target_shape)
    return cache_dir / f"lecture_{label_name}_m{month}_{shape_name}.npz"


def build_feature_matrix(subset: pd.DataFrame, labels: list[str], config: LectureConfig) -> np.ndarray:
    """把 `subset` 中所有 MRI 预处理后堆叠成特征矩阵 X。

    这里用了缓存，因为 MRI 预处理通常比模型训练更慢。
    如果文件列表没变，就直接复用缓存结果。
    """

    cache_path = feature_cache_path(config.cache_dir, labels, config.month, config.target_shape)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_paths = cached["paths"].tolist()
        current_paths = subset["path"].tolist()
        if cached_paths == current_paths:
            return cached["X"]

    features = []
    for row in subset.itertuples(index=False):
        # 一个 MRI 文件最终对应特征矩阵中的一行。
        features.append(preprocess_volume(Path(row.path), config.target_shape, config.clip_percentile))

    X = np.vstack(features).astype(np.float32)
    np.savez_compressed(cache_path, X=X, paths=np.array(subset["path"].tolist(), dtype=str))
    return X


def make_subject_split(
    subset: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.Series, pd.Series]:
    """按受试者而不是按图片划分训练集和测试集。

    这是整个项目里最重要的设计之一。
    如果同一个受试者同时出现在训练集和测试集，
    那么评估结果就会被“数据泄漏”人为抬高。
    """

    subject_table = subset[["subject", "label"]].drop_duplicates().reset_index(drop=True)
    train_subjects, test_subjects = train_test_split(
        subject_table["subject"],
        test_size=test_size,
        random_state=random_state,
        stratify=subject_table["label"],
    )
    return train_subjects, test_subjects


def encode_labels(subset: pd.DataFrame, labels: list[str]) -> np.ndarray:
    """把 AD、NORMAL 这样的类别名映射成整数标签。"""

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    return subset["label"].map(label_to_index).to_numpy()


def safe_pca_components(requested: int, X_train: np.ndarray) -> int:
    """给 PCA 选择一个始终合法的维度数。

    PCA 的主成分个数不能超过训练矩阵允许的范围，
    所以这里会把请求值裁剪到一个安全区间。
    """

    return max(1, min(requested, X_train.shape[0] - 1, X_train.shape[1]))


def build_pipeline(kernel: str, pca_components: int, random_state: int, c_value: float | None = None) -> Pipeline:
    """构建讲课统一使用的流水线：标准化 -> PCA -> SVM。"""

    svm_kwargs: dict[str, object] = {
        "kernel": kernel,
        "class_weight": "balanced",
        "decision_function_shape": "ovr",
    }
    if c_value is not None:
        svm_kwargs["C"] = c_value

    return Pipeline(
        [
            # 先做标准化，让不同维度的特征处在可比较的尺度上。
            ("scaler", StandardScaler()),
            # 再做 PCA，既能降维提速，也更方便课堂上解释。
            ("pca", PCA(n_components=pca_components, svd_solver="randomized", random_state=random_state)),
            # 最后一步才是真正的分类器：线性核或 RBF 核的 SVM。
            ("svm", SVC(**svm_kwargs)),
        ]
    )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, positive_label: int = 0) -> dict[str, float]:
    """计算课堂上二分类任务常用的主要指标。"""

    # scikit-learn 的 decision score 对一类为正、另一类为负。
    # 如果正类不是默认方向，就需要翻转符号，保证 ROC-AUC 的含义正确。
    auc_value = float(roc_auc_score(y_true == positive_label, scores if positive_label == 1 else -scores))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "roc_auc": auc_value,
    }


def multiclass_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """计算多分类任务的宏平均 one-vs-rest ROC-AUC。"""

    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    if y_bin.ndim == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])
    try:
        return float(roc_auc_score(y_bin, scores, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def save_confusion_figure(cm: np.ndarray, labels: list[str], output_path: Path, title: str) -> None:
    """保存混淆矩阵热力图，方便放进讲稿或 PPT。"""

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_binary_roc(y_true: np.ndarray, scores: np.ndarray, output_path: Path, positive_label: int = 0) -> float:
    """保存二分类 ROC 曲线，并返回对应的 AUC。"""

    binary_scores = scores if positive_label == 1 else -scores
    fpr, tpr, _ = roc_curve(y_true == positive_label, binary_scores)
    auc_value = float(roc_auc_score(y_true == positive_label, binary_scores))

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return auc_value


def save_json(path: Path, payload: object) -> None:
    """用 UTF-8 和缩进格式写出 JSON 文件。"""

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fit_grid_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    config: LectureConfig,
    output_dir: Path,
    kernels: list[str],
) -> pd.DataFrame:
    """训练一个或多个 SVM 版本，并保存讲课需要的结果。

    这是多个 lecture 脚本背后的核心训练函数。
    对于每一种 kernel，它都会完成：

    1. 构建流水线；
    2. 只在训练集上做 GridSearchCV；
    3. 在留出的测试集上评估最佳模型；
    4. 保存指标、图像和训练好的模型。
    """

    ensure_dir(output_dir)
    pca_components = safe_pca_components(config.pca_components, X_train)
    # 这里用分组交叉验证，是为了防止同一个受试者的数据泄漏到不同折里。
    cv = StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    rows: list[dict[str, object]] = []

    for kernel in kernels:
        # 预处理流程保持不变，只替换核函数。
        pipeline = build_pipeline(kernel=kernel, pca_components=pca_components, random_state=config.random_state)
        if kernel == "linear":
            param_grid = [{"svm__C": [0.1, 1, 10]}]
        else:
            # RBF 核同时需要搜索 C 和 gamma，所以参数空间更大。
            param_grid = [{"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.1, 0.01]}]

        search = GridSearchCV(
            estimator=pipeline,  # 标准化 + PCA + SVM 流水线
            param_grid=param_grid,  # 当前 kernel 的超参数搜索空间
            scoring="balanced_accuracy",  # 以 balanced accuracy 作为选模标准
            cv=cv,  # 受试者分组交叉验证
            n_jobs=config.n_jobs,  # 并行 worker 数
            refit=True,  # 用最佳参数在整个训练集上重训
        )
        # 课堂上要重点强调：
        # 参数选择必须发生在训练集内部，不能拿测试集调参。
        search.fit(X_train, y_train, groups=groups_train)
        model: Pipeline = search.best_estimator_
        y_pred = model.predict(X_test)
        scores = model.decision_function(X_test)

        if len(labels) == 2:
            # 二分类任务：使用二分类指标，并绘制 ROC 曲线。
            metric_row = binary_metrics(y_test, y_pred, scores, positive_label=0)
            save_binary_roc(y_test, scores, output_dir / f"roc_curve_{kernel}.png", positive_label=0)
        else:
            # 多分类任务：使用宏平均指标。
            metric_row = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "roc_auc": multiclass_auc(y_test, scores),
            }

        cm = confusion_matrix(y_test, y_pred)
        # 同时保存表格版结果和适合放到 PPT 的图像版结果。
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / f"confusion_matrix_{kernel}.csv")
        save_confusion_figure(cm, labels, output_dir / f"confusion_matrix_{kernel}.png", f"Confusion Matrix ({kernel})")
        save_json(output_dir / f"best_params_{kernel}.json", search.best_params_)
        joblib.dump(model, output_dir / f"best_model_{kernel}.joblib")

        rows.append(
            {
                "model": kernel,
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
                "cv_best_score": float(search.best_score_),
                **metric_row,
                "support_vectors_total": int(np.sum(model.named_steps["svm"].n_support_)),
                "support_vectors_per_class": json.dumps(model.named_steps["svm"].n_support_.tolist()),
                "pca_components": pca_components,
                "pca_explained_variance": float(model.named_steps["pca"].explained_variance_ratio_.sum()),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    return summary


def prepare_split_data(labels: list[str], config: LectureConfig) -> dict[str, object]:
    """在模型训练前，把后续要用的一切都准备好。

    返回的是一个字典，目的是让 lecture 脚本保持尽量短。
    它里面包含：

    1. 索引后的完整数据表；
    2. 当前课题选出的子集；
    3. 特征矩阵；
    4. 标签向量；
    5. 训练/测试划分结果；
    6. 训练集和测试集数组。
    """

    dataset_index = index_dataset(config.data_dir)
    subset = select_subset(dataset_index, labels=labels, month=config.month)
    X = build_feature_matrix(subset, labels=labels, config=config)
    y = encode_labels(subset, labels=labels)

    train_subjects, test_subjects = make_subject_split(subset, config.test_size, config.random_state)
    split_column = np.where(subset["subject"].isin(train_subjects), "train", "test")
    train_mask = split_column == "train"

    return {
        "dataset_index": dataset_index,
        "subset": subset,
        "X": X,
        "y": y,
        "split_column": split_column,
        "X_train": X[train_mask],
        "X_test": X[~train_mask],
        "y_train": y[train_mask],
        "y_test": y[~train_mask],
        "groups_train": subset.loc[train_mask, "subject"].to_numpy(),
    }


def save_split_tables(
    output_dir: Path,
    dataset_index: pd.DataFrame,
    subset: pd.DataFrame,
    split_column: np.ndarray,
) -> None:
    """保存数据索引、所选样本表和训练/测试划分表。"""

    ensure_dir(output_dir)
    dataset_index.to_csv(output_dir / "dataset_index.csv", index=False)
    subset.to_csv(output_dir / "selected_samples.csv", index=False)
    split_df = subset.copy()
    split_df["split"] = split_column
    split_df.to_csv(output_dir / "train_test_split.csv", index=False)
