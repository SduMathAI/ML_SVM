from __future__ import annotations

"""课题三教学脚本：比较线性核和 RBF 核。

这个脚本用于第 6.3 节“核函数”的讲解。
它会在同一个 AD vs NORMAL 任务上分别训练：

1. 线性核 SVM；
2. RBF 核 SVM。

这样你就可以直接比较：

1. 指标差异；
2. 混淆矩阵差异；
3. 支持向量数量差异；
4. 更复杂的核函数是否真的更好。
"""

import argparse
from pathlib import Path

from lecture_experiments.common import (
    LectureConfig,
    fit_grid_models,
    prepare_split_data,
    save_split_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topic 3: compare linear and RBF kernels on AD vs NORMAL @ m00.")
    parser.add_argument("--output-dir", default="results/lecture_topic03_kernel_compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    config = LectureConfig()
    # 这里的随机种子写在配置里，所以这份结果是可复现的，
    # 也便于和讲稿中的数字保持一致。
    prepared = prepare_split_data(labels=["AD", "NORMAL"], config=config)
    save_split_tables(output_dir, prepared["dataset_index"], prepared["subset"], prepared["split_column"])
    fit_grid_models(
        X_train=prepared["X_train"],
        y_train=prepared["y_train"],
        groups_train=prepared["groups_train"],
        X_test=prepared["X_test"],
        y_test=prepared["y_test"],
        labels=["AD", "NORMAL"],
        config=config,
        output_dir=output_dir,
        kernels=["linear", "rbf"],
    )
    print(f"Saved kernel-comparison results to: {output_dir}")


if __name__ == "__main__":
    main()
