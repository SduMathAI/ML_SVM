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
    """运行当前唯一推荐的主课题实验入口。"""

    # 这是当前唯一推荐的主课题入口：
    # 只做 AD vs NORMAL，并且只使用 m00 基线数据。
    #
    # 这里的划分策略也固定下来，方便你汇报时直接说明：
    # 1. 只取 m00，所以每个受试者只贡献一个基线 MRI；
    # 2. 按受试者做 80% 训练集、20% 测试集划分；
    # 3. 在训练集内部做 5 折分组交叉验证来选参数。
    config = ExperimentConfig(
        data_dir=Path("processed"),  # MRI 数据目录
        output_dir=Path("results/main_topic_ad_vs_normal_m00"),  # 主课题结果输出目录
        cache_dir=Path("artifacts/feature_cache"),  # 预处理特征缓存目录
        labels=["AD", "NORMAL"],  # 当前二分类标签
        month="00",  # 只使用基线时间点 m00
        target_shape=(32, 32, 32),  # MRI 下采样后的统一尺寸
        pca_components=100,  # PCA 主成分上限
        test_size=0.2,  # 受试者级别测试集比例
        random_state=42,  # 固定随机种子，保证可复现
        cv_folds=5,  # 训练集内部 5 折交叉验证
        clip_percentile=99.5,  # 强度裁剪上界分位数
        n_jobs=-1,  # 并行使用全部 CPU 核心
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
