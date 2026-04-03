# Script Quick Guide

如果你只想完成主课题，请只记住下面这一个脚本：

```powershell
python scripts\run_main_topic_experiment.py
```

它负责：

1. 从 `processed/` 中选择 `AD vs NORMAL @ m00`
2. 自动做受试者级别的 `80%/20%` 训练测试划分
3. 在训练集内部做 `5-fold StratifiedGroupKFold` 交叉验证
4. 在测试集上输出最终准确率、Balanced Accuracy、F1、ROC-AUC
5. 把结果保存到 `results/main_topic_ad_vs_normal_m00/`

## 1. 你现在最需要的脚本

### `scripts/run_main_topic_experiment.py`

用途：

- 当前最推荐的主入口
- 对应主课题：`AD vs NORMAL @ m00`
- 最适合汇报和讲课

### `scripts/run_mri_svm_experiment.py`

用途：

- 通用训练脚本
- 如果以后要改标签、改时间点、改输出目录，就用它

## 2. 讲课用脚本

这些脚本不是主训练入口，而是配合第 6 章讲稿使用的：

- `scripts/lecture_topic01_max_margin.py`
- `scripts/lecture_topic02_support_vectors.py`
- `scripts/lecture_topic03_kernel_compare.py`
- `scripts/lecture_topic04_c_sweep.py`
- `scripts/lecture_topic05_task_compare.py`
- `scripts/lecture_topic06_multiclass.py`

## 3. 主课题统一表述

当前主课题建议统一表述为：

**基于 ADNI 脑 MRI 基线数据的 AD 与 NORMAL 分类研究：以 PCA-SVM 为主线分析最大间隔、支持向量、核函数与软间隔方法。**

更短的版本可以直接说：

**主课题：`AD vs NORMAL @ m00` 的 MRI 分类。**

## 4. 现在只要关注这些文件

- `scripts/run_main_topic_experiment.py`
- `scripts/run_mri_svm_experiment.py`
- `scripts/lecture_topic01_max_margin.py`
- `scripts/lecture_topic02_support_vectors.py`
- `scripts/lecture_topic03_kernel_compare.py`
- `scripts/lecture_topic04_c_sweep.py`
- `scripts/lecture_topic05_task_compare.py`
- `scripts/lecture_topic06_multiclass.py`
- `scripts/lecture_experiments/common.py`
