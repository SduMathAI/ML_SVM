from __future__ import annotations

"""课题六教学脚本：把 SVM 从二分类扩展到三分类。

这个脚本适合放在整章后半段或结尾。
它想说明两件事：

1. 同一套 MRI + PCA + SVM 流程可以扩展到 AD / MCI / NORMAL；
2. 任务一旦从二分类扩展到三分类，难度会明显提高。
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
    """解析“三分类 SVM”脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="Topic 6: multiclass SVM on AD/MCI/NORMAL @ m00.")
    parser.add_argument("--output-dir", default="results/lecture_topic06_multiclass")
    return parser.parse_args()


def main() -> None:
    """把 MRI PCA+SVM 流程从二分类扩展到三分类。"""

    args = parse_args()
    output_dir = Path(args.output_dir)

    config = LectureConfig()  # 保持和主实验相同的默认预处理设置
    # 这是整章的收束任务：
    # 仍然是同一条 MRI 建模流水线，但现在标签变成三类。
    labels = ["AD", "MCI", "NORMAL"]
    prepared = prepare_split_data(labels=labels, config=config)  # 三分类数据准备
    save_split_tables(output_dir, prepared["dataset_index"], prepared["subset"], prepared["split_column"])
    fit_grid_models(
        X_train=prepared["X_train"],
        y_train=prepared["y_train"],
        groups_train=prepared["groups_train"],
        X_test=prepared["X_test"],
        y_test=prepared["y_test"],
        labels=labels,  # AD / MCI / NORMAL 三类标签
        config=config,  # 统一实验配置
        output_dir=output_dir,  # 输出三分类结果目录
        kernels=["linear", "rbf"],  # 继续比较线性核和 RBF 核
    )
    print(f"Saved multiclass results to: {output_dir}")


if __name__ == "__main__":
    main()
