# Scripts README

这个目录现在只保留主线需要的脚本。

## 1. 主课题入口

### `run_main_topic_experiment.py`

这是你现在最应该使用的脚本。

运行命令：

```powershell
python scripts\run_main_topic_experiment.py
```

它会自动完成：

1. 选择主课题数据：`AD vs NORMAL @ m00`
2. 按受试者做 `80%` 训练集、`20%` 测试集划分
3. 在训练集内部做 `5-fold StratifiedGroupKFold` 交叉验证
4. 在测试集上输出最终指标
5. 保存结果到 `results/main_topic_ad_vs_normal_m00/`

## 2. 通用训练底层

### `run_mri_svm_experiment.py`

这是底层通用训练脚本。

它适合在下面场景使用：

1. 你想修改标签组合
2. 你想修改时间点
3. 你想修改输出目录或参数

如果你只是做主课题，可以先不直接碰它。

## 3. 讲课脚本

### `lecture_topic01_max_margin.py`

讲 `6.1` 最大间隔与支持向量的二维示意图。

### `lecture_topic02_support_vectors.py`

讲 `6.2` 支持向量、对偶问题、KKT 的直观意义。

### `lecture_topic03_kernel_compare.py`

讲 `6.3` 线性核 vs RBF 核。

### `lecture_topic04_c_sweep.py`

讲 `6.4` 参数 `C` 如何影响间隔与泛化。

### `lecture_topic05_task_compare.py`

讲不同二分类任务的难度差异。

### `lecture_topic06_multiclass.py`

讲三分类扩展 `AD/MCI/NORMAL`。

## 4. 公共工具

### `lecture_experiments/common.py`

这是讲课脚本共用的底层工具文件。  
你只需要知道它负责数据读取、划分、训练和评估，不需要上课逐行讲。

## 5. 现在的最推荐顺序

1. 先跑 `run_main_topic_experiment.py`
2. 看 `results/main_topic_ad_vs_normal_m00/metrics_summary.csv`
3. 再看 `Lecture_Note/chapter6_project_driven_script.md`
4. 如果要上课展示，再按顺序跑 `lecture_topic01-06`

## 6. 一句话总结

当前目录里：

- 真正必须会用的：`run_main_topic_experiment.py`
- 真正必须会讲的：`lecture_topic01-06.py`
- 不必展开源码细节的：`lecture_experiments/common.py`
