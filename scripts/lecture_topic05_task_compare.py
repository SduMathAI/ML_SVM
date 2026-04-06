from __future__ import annotations

"""课题五教学脚本：比较三个二分类任务的难度。

这个脚本对应“同一套方法，不同任务难度”的讲解。
它的作用是帮助学生理解：
模型表现不仅取决于算法，也取决于任务本身是否容易分开。
"""

import argparse
from pathlib import Path

import pandas as pd

from lecture_experiments.common import (
    LectureConfig,
    ensure_dir,
    fit_grid_models,
    prepare_split_data,
    save_split_tables,
)


def parse_args() -> argparse.Namespace:
    """解析“二分类任务难度比较”脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="Topic 5: compare binary task difficulty.")
    parser.add_argument("--output-dir", default="results/lecture_topic05_task_compare")
    return parser.parse_args()


def main() -> None:
    """在多个二分类任务上运行同一流程并比较难度。"""

    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    config = LectureConfig()  # 保持统一实验设定，只改变任务标签
    # 保持同一个 m00 设置、同一套预处理和同一类模型，
    # 只改变标签组合，这样更容易比较“任务本身的难度差异”。
    tasks = [
        ["AD", "NORMAL"],
        ["MCI", "NORMAL"],
        ["AD", "MCI"],
    ]
    all_rows: list[pd.DataFrame] = []

    for labels in tasks:
        task_name = "_vs_".join(label.lower() for label in labels)
        task_dir = output_dir / task_name
        prepared = prepare_split_data(labels=labels, config=config)  # 当前标签组合对应的数据子集
        save_split_tables(task_dir, prepared["dataset_index"], prepared["subset"], prepared["split_column"])
        summary = fit_grid_models(
            X_train=prepared["X_train"],
            y_train=prepared["y_train"],
            groups_train=prepared["groups_train"],
            X_test=prepared["X_test"],
            y_test=prepared["y_test"],
            labels=labels,  # 当前任务，如 AD vs MCI
            config=config,  # 统一预处理和数据切分
            output_dir=task_dir,  # 每个任务单独保存结果
            kernels=["linear", "rbf"],  # 每个任务都比较两种核函数
        )
        summary.insert(0, "task", f"{labels[0]} vs {labels[1]}")
        all_rows.append(summary)

    pd.concat(all_rows, ignore_index=True).to_csv(output_dir / "task_comparison_summary.csv", index=False)
    print(f"Saved task-comparison results to: {output_dir}")


if __name__ == "__main__":
    main()
