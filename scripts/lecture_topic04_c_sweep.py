from __future__ import annotations

"""课题四教学脚本：解释软间隔与参数 C。

这个脚本用于第 6.4 节。
它固定任务、固定预处理、固定模型结构，
只改变一个量：C。

这样课堂上最容易讲清楚：
C 控制的是“间隔宽度”和“错分惩罚”之间的平衡。
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt

from lecture_experiments.common import (
    LectureConfig,
    binary_metrics,
    build_pipeline,
    ensure_dir,
    prepare_split_data,
    safe_pca_components,
)


def parse_args() -> argparse.Namespace:
    """解析“参数 C 扫描”脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="Topic 4: show how C changes the linear SVM.")
    parser.add_argument("--output-dir", default="results/lecture_topic04_c_sweep")
    return parser.parse_args()


def main() -> None:
    """扫描不同的 C 值，展示软间隔中的权衡关系。"""

    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    config = LectureConfig()  # 其他实验条件全部保持默认
    prepared = prepare_split_data(labels=["AD", "NORMAL"], config=config)  # 固定主线二分类任务
    pca_components = safe_pca_components(config.pca_components, prepared["X_train"])  # 防止 PCA 维数超过训练集允许范围

    rows = []
    for c_value in [0.01, 0.1, 1.0, 10.0, 100.0]:
        # 这里只改 C，其他所有条件都保持不变。
        # 这样实验现象才容易解释，不会把多个因素混在一起。
        model = build_pipeline(
            kernel="linear",  # 固定线性核，只观察 C 的影响
            pca_components=pca_components,  # 维持统一降维维数
            random_state=config.random_state,  # 固定随机性，保证可复现
            c_value=c_value,  # 当前软间隔惩罚系数
        )
        model.fit(prepared["X_train"], prepared["y_train"])
        y_pred = model.predict(prepared["X_test"])
        scores = model.decision_function(prepared["X_test"])
        metrics = binary_metrics(prepared["y_test"], y_pred, scores, positive_label=0)

        rows.append(
            {
                "C": c_value,
                **metrics,
                "support_vectors_total": int(model.named_steps["svm"].n_support_.sum()),
                "support_vectors_per_class": str(model.named_steps["svm"].n_support_.tolist()),
                "pca_explained_variance": float(model.named_steps["pca"].explained_variance_ratio_.sum()),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "c_sweep_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(summary["C"], summary["balanced_accuracy"], marker="o", label="Balanced accuracy")
    plt.plot(summary["C"], summary["f1"], marker="s", label="F1")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.title("Topic 4: effect of C on AD vs NORMAL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "c_sweep_scores.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summary["C"], summary["support_vectors_total"], marker="o")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("Number of support vectors")
    plt.title("Topic 4: support vector count under different C")
    plt.tight_layout()
    plt.savefig(output_dir / "c_sweep_support_vectors.png", dpi=200)
    plt.close()
    print(f"Saved C-sweep tables and figures to: {output_dir}")


if __name__ == "__main__":
    main()
