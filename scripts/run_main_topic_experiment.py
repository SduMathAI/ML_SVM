from __future__ import annotations

"""主课题唯一推荐入口脚本。

如果你现在只想完成一个最核心、最适合汇报的实验，
直接运行这个脚本即可。

主课题：
AD vs NORMAL 在基线时间点 m00 上的分类任务。
"""

from pathlib import Path

import pandas as pd

from run_mri_svm_experiment import ExperimentConfig, run_experiment


def main() -> None:
    # 这是当前唯一推荐的主课题入口：
    # 只做 AD vs NORMAL，并且只使用 m00 基线数据。
    #
    # 这里的划分策略也固定下来，方便你汇报时直接说明：
    # 1. 只取 m00，所以每个受试者只贡献一个基线 MRI；
    # 2. 按受试者做 80% 训练集、20% 测试集划分；
    # 3. 在训练集内部做 5 折分组交叉验证来选参数。
    config = ExperimentConfig(
        data_dir=Path("processed"),
        output_dir=Path("results/main_topic_ad_vs_normal_m00"),
        cache_dir=Path("artifacts/feature_cache"),
        labels=["AD", "NORMAL"],
        month="00",
        target_shape=(32, 32, 32),
        pca_components=100,
        test_size=0.2,
        random_state=42,
        cv_folds=5,
        clip_percentile=99.5,
        n_jobs=-1,
    )

    summary = run_experiment(config)

    print("\n=== Main topic summary ===")
    print("Task: AD vs NORMAL @ m00")
    print("Train/test split: subject-level 80/20")
    print("Model selection: 5-fold StratifiedGroupKFold on the training set")

    for row in summary.itertuples(index=False):
        print(
            f"{row.model}: "
            f"test_accuracy={row.accuracy:.4f}, "
            f"balanced_accuracy={row.balanced_accuracy:.4f}, "
            f"f1={row.f1:.4f}, "
            f"roc_auc={row.roc_auc:.4f}"
        )

    metrics_path = config.output_dir / "metrics_summary.csv"
    split_path = config.output_dir / "train_test_split.csv"
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved split table to: {split_path}")


if __name__ == "__main__":
    main()
