from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib import pyplot as plt


@dataclass
class TrainConfig:
    ad_npz: Path
    normal_npz: Path
    output_dir: Path
    test_size: float
    random_state: int
    cv_folds: int
    n_jobs: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train linear and RBF SVM models from two .npz feature files (AD vs NORMAL)."
    )
    parser.add_argument("--ad-npz", required=True, help="Path to AD feature .npz file.")
    parser.add_argument("--normal-npz", required=True, help="Path to NORMAL feature .npz file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save results.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for grid search.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for grid search.")
    args = parser.parse_args()

    return TrainConfig(
        ad_npz=Path(args.ad_npz),
        normal_npz=Path(args.normal_npz),
        output_dir=Path(args.output_dir),
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
    )


def load_features(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "X" in data:
        X = data["X"]
    else:
        keys = data.files
        if not keys:
            raise ValueError(f"no arrays found in {path}")
        X = data[keys[0]]
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X.astype(np.float32)


def build_dataset(ad_npz: Path, normal_npz: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_ad = load_features(ad_npz)
    X_normal = load_features(normal_npz)
    if X_ad.shape[1] != X_normal.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: AD={X_ad.shape[1]} NORMAL={X_normal.shape[1]}"
        )
    X = np.vstack([X_ad, X_normal]).astype(np.float32)
    y = np.array([0] * len(X_ad) + [1] * len(X_normal))
    class_names = ["AD", "NORMAL"]
    return X, y, class_names


def pipeline_for(kernel: str) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=kernel, class_weight="balanced")),
        ]
    )


def param_grid_for(kernel: str) -> list[dict[str, list[object]]]:
    if kernel == "linear":
        return [{"svm__C": [0.1, 1, 10]}]
    return [{"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.1, 0.01]}]


def save_roc_curve(y_true: np.ndarray, scores: np.ndarray, output_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true == 0, -scores)
    auc_value = roc_auc_score(y_true == 0, -scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (positive=AD)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return float(auc_value)


def train_and_eval(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    config: TrainConfig,
) -> pd.DataFrame:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    results = []

    for kernel in ["linear", "rbf"]:
        print(f"[train] kernel={kernel}")
        pipeline = pipeline_for(kernel)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid_for(kernel),
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=config.n_jobs,
            refit=True,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        best_model: Pipeline = grid.best_estimator_
        preds = best_model.predict(X_test)
        scores = best_model.decision_function(X_test)

        auc_value = save_roc_curve(y_test, scores, output_dir / f"roc_curve_{kernel}.png")
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)

        metrics = {
            "model": kernel,
            "best_params": grid.best_params_,
            "cv_best_score": float(grid.best_score_),
            "accuracy": float(accuracy_score(y_test, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, pos_label=0, average="binary", zero_division=0)),
            "recall": float(recall_score(y_test, preds, pos_label=0, average="binary", zero_division=0)),
            "f1": float(f1_score(y_test, preds, pos_label=0, average="binary", zero_division=0)),
            "roc_auc": auc_value,
            "support_vectors_total": int(np.sum(best_model.named_steps["svm"].n_support_)),
            "support_vectors_per_class": best_model.named_steps["svm"].n_support_.tolist(),
        }
        results.append(metrics)

        (output_dir / f"best_params_{kernel}.json").write_text(
            json.dumps(grid.best_params_, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / f"classification_report_{kernel}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / f"confusion_matrix_{kernel}.csv")

        joblib.dump(best_model, output_dir / f"best_model_{kernel}.joblib")

    summary = pd.DataFrame(results)
    summary["best_params"] = summary["best_params"].apply(lambda item: json.dumps(item, ensure_ascii=False))
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    metadata = {
        "ad_npz": str(config.ad_npz.resolve()),
        "normal_npz": str(config.normal_npz.resolve()),
        "n_samples_ad": int((y == 0).sum()),
        "n_samples_normal": int((y == 1).sum()),
        "test_size": config.test_size,
        "random_state": config.random_state,
        "cv_folds": config.cv_folds,
        "class_names": class_names,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    return summary


def main() -> None:
    config = parse_args()
    X, y, class_names = build_dataset(config.ad_npz, config.normal_npz)
    train_and_eval(X, y, class_names, config)


if __name__ == "__main__":
    main()
'''
用 argparse 接收命令行参数。用户需要提供：
--ad-npz, --normal-npz：两个特征文件的路径。
--output-dir：结果输出路径。
--test-size：测试集比例（默认 0.2，即 80% 训练，20% 测试）。
--cv-folds：交叉验证的折数（默认 5 折）。
② 数据准备 (load_features & build_dataset)
load_features：读取 .npz 文件中的 NumPy 数组（通常键名为 "X"）。
build_dataset：
将 AD 数据和 NORMAL 数据上下拼接 (np.vstack)。
打标签：给 AD 生成 0，给 NORMAL 生成 1。注意：在医学影像中，通常把患病（AD）视为正例（Positive），这里的代码在后续计算中也是把 0 作为正例来处理的。
③ 建立模型流水线 (pipeline_for & param_grid_for)
Pipeline：非常重要的一步。SVM 对数据的尺度非常敏感，所以代码先用 StandardScaler() 将特征标准化（均值为0，方差为1），然后再送入 SVC()。并且设置了 class_weight="balanced"，自动处理可能出现的两类数据不平衡问题。
param_grid：定义了网格搜索的范围。
Linear 核：尝试不同的惩罚系数 C（0.1, 1, 10）。
RBF 核：除了 C，还尝试不同的核函数参数 gamma。
④ 核心训练与评估循环 (train_and_eval)
这是代码的主体部分，它循环两次，分别训练 Linear 和 RBF 模型：
train_test_split(stratify=y)：按标签比例划分训练测试集，确保训练集和测试集里 AD 和 NORMAL 的比例一致。
GridSearchCV：使用网格搜索和 5 折交叉验证，自动穷举所有的参数组合，找到在验证集上 balanced_accuracy（平衡准确率）最高的参数。
计算指标：
拿到最佳模型 best_model 后，对测试集进行预测。
调用 save_roc_curve 绘制 ROC 曲线并计算 AUC 值。
计算并记录 Accuracy、Precision、Recall、F1 等指标。注意代码中显式指定了 pos_label=0，确保它是以识别 AD 为核心目标来计算这些指标的。
文件保存：将结果固化到硬盘上。
3. 运行后会生成什么？
假设你的 --output-dir 设置为 results/svm_models，程序运行完毕后，该文件夹下会生成以下丰富的文件：
模型文件：
best_model_linear.joblib / best_model_rbf.joblib：可以直接用于未来新数据预测的模型文件。
图表文件：
roc_curve_linear.png / roc_curve_rbf.png：直观展示模型分类性能的 ROC 曲线图。
数据与报告：
metrics_summary.csv：汇总两种模型的各项得分（准确率、AUC等），方便对比哪个模型更好。
confusion_matrix_linear.csv / rbf.csv：混淆矩阵，能看清模型把多少 AD 错判成了 NORMAL，或者反过来。
classification_report_*.json：详细的分类报告。
best_params_*.json：网格搜索找到的最佳超参数。
experiment_config.json：记录了本次实验的所有配置（路径、随机种子等），保证实验的可复现性（Reproducibility）。
'''