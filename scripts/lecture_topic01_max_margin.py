from __future__ import annotations

"""课题一教学脚本：最大间隔的直观理解。

这个脚本只服务于第 6.1 节课堂讲解。
它会把高维 MRI 特征压到二维 PCA 空间里，
让学生直观看到三件事：

1. 什么是分类超平面；
2. 什么是间隔；
3. 哪些点是支持向量。

注意：
这个脚本是为了讲几何直觉，不是为了报告最终最优精度。
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lecture_experiments.common import LectureConfig, ensure_dir, prepare_split_data, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topic 1: maximum margin demo on AD vs NORMAL @ m00.")
    parser.add_argument("--output-dir", default="results/lecture_topic01_max_margin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # 这里故意把 PCA 降到二维，只是为了把图画出来给课堂展示。
    config = LectureConfig(pca_components=2)
    prepared = prepare_split_data(labels=["AD", "NORMAL"], config=config)

    # 这里的目标不是追求最优性能，而是让分类边界能够被看见。
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=config.random_state)),
            ("svm", SVC(kernel="linear", C=1.0, class_weight="balanced")),
        ]
    )
    model.fit(prepared["X_train"], prepared["y_train"])

    X_plot = model.named_steps["pca"].transform(model.named_steps["scaler"].transform(prepared["X_train"]))
    clf = model.named_steps["svm"]
    support_indices = clf.support_

    x_min, x_max = X_plot[:, 0].min() - 1.0, X_plot[:, 0].max() + 1.0
    y_min, y_max = X_plot[:, 1].min() - 1.0, X_plot[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # decision_function 可以理解为“点到分类面的有符号距离”。
    # 轮廓线中的 -1、0、1 分别对应下边界、分类面、上边界。
    zz = clf.decision_function(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contour(xx, yy, zz, levels=[-1, 0, 1], colors=["#999999", "#d95f02", "#999999"], linestyles=["--", "-", "--"])
    plt.scatter(X_plot[prepared["y_train"] == 0, 0], X_plot[prepared["y_train"] == 0, 1], label="AD", alpha=0.75, s=35)
    plt.scatter(X_plot[prepared["y_train"] == 1, 0], X_plot[prepared["y_train"] == 1, 1], label="NORMAL", alpha=0.75, s=35)
    plt.scatter(
        X_plot[support_indices, 0],
        X_plot[support_indices, 1],
        s=140,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Support vectors",
    )
    plt.title("Topic 1: Linear SVM in 2D PCA space")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "max_margin_demo.png", dpi=200)
    plt.close()

    summary = {
        "train_samples": int(len(prepared["y_train"])),
        "test_samples": int(len(prepared["y_test"])),
        "support_vectors_total": int(len(support_indices)),
        "support_vectors_per_class": clf.n_support_.tolist(),
        "note": "This figure is for teaching geometry in 2D PCA space, not for reporting the best final accuracy.",
    }
    save_json(output_dir / "summary.json", summary)
    print(f"Saved teaching figure to: {output_dir / 'max_margin_demo.png'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
