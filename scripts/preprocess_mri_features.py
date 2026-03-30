#解决6.2数据预处理问题  # 记录任务对应的章节需求
# ------------------------------  # 模块级分隔行
from __future__ import annotations  # 启用延期注解支持
# ------------------------------  # 导入标准库依赖的分隔
import argparse  # 解析命令行参数
from dataclasses import dataclass  # 提供数据类装饰器
from pathlib import Path  # 处理文件路径
# ------------------------------  # 导入第三方库的分隔
import nibabel as nib  # 读取与处理NIfTI医学影像
import numpy as np  # 数值运算基础库
from scipy.ndimage import zoom  # 图像缩放插值函数
# ------------------------------  # 数据类定义区域分隔

@dataclass  # 将配置定义为不可变的数据结构
class PreprocessConfig:
    input_dir: Path  # 输入NIfTI体数据所在目录
    output_path: Path  # 特征输出npz路径
    target_shape: tuple[int, int, int]  # 目标采样形状
    low_percentile: float  # 裁剪低百分位阈值
    high_percentile: float  # 裁剪高百分位阈值
# ------------------------------  # 函数定义分隔


def parse_args() -> PreprocessConfig:  # 解析CLI参数并构造配置
    parser = argparse.ArgumentParser(  # 初始化参数解析器
        description=(  # 提供命令描述
            "Preprocess .nii.gz MRI volumes: nonzero percentile clipping, z-score, "  # 描述第一部分
            "downsample, and flatten into feature vectors."  # 描述第二部分
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing .nii.gz volumes.")  # 必填输入目录
    parser.add_argument(  # 配置输出路径参数
        "--output",  # 参数名
        default="artifacts/mri_features.npz",  # 默认输出文件
        help="Output .npz file to store features and file paths.",  # 参数帮助
    )
    parser.add_argument(  # 配置目标尺寸参数
        "--target-shape",  # 参数名
        nargs=3,  # 接收三个维度
        type=int,  # 解析为整数
        default=[32, 32, 32],  # 默认体素尺寸
        metavar=("X", "Y", "Z"),  # 帮助文本中维度标签
        help="Target shape after downsampling.",  # 参数说明
    )
    parser.add_argument(  # 配置低百分位参数
        "--low-percentile",  # 参数名
        type=float,  # 解析为浮点数
        default=0.5,  # 默认值
        help="Lower percentile for clipping, computed on nonzero voxels.",  # 参数说明
    )
    parser.add_argument(  # 配置高百分位参数
        "--high-percentile",  # 参数名
        type=float,  # 解析为浮点数
        default=99.5,  # 默认值
        help="Upper percentile for clipping, computed on nonzero voxels.",  # 参数说明
    )
    args = parser.parse_args()  # 执行解析得到命名空间

    return PreprocessConfig(  # 构造并返回配置数据类
        input_dir=Path(args.input_dir),  # 标准化输入路径
        output_path=Path(args.output),  # 标准化输出路径
        target_shape=tuple(args.target_shape),  # 存储目标体素尺寸
        low_percentile=args.low_percentile,  # 保存低百分位阈值
        high_percentile=args.high_percentile,  # 保存高百分位阈值
    )


def preprocess_volume(  # 单个NIfTI体数据预处理
    path: Path,  # 当前体数据路径
    target_shape: tuple[int, int, int],  # 缩放后目标尺寸
    low_percentile: float,  # 低裁剪阈值
    high_percentile: float,  # 高裁剪阈值
) -> np.ndarray:  # 返回扁平化特征
    data = nib.load(str(path)).get_fdata(dtype=np.float32)  # 读取体数据并转为float32
    mask = data != 0  # 构建非零体素掩码

    if mask.any():  # 若存在非零体素
        nonzero = data[mask]  # 提取非零体素数组
        low = float(np.percentile(nonzero, low_percentile))  # 计算低百分位
        high = float(np.percentile(nonzero, high_percentile))  # 计算高百分位
        if high <= low:  # 防止高低阈值逆序
            high = float(nonzero.max())  # 强制使用最大值
        data = np.clip(data, low, high)  # 应用裁剪

        mean = float(nonzero.mean())  # 计算均值
        std = float(nonzero.std())  # 计算标准差
        if std < 1e-6:  # 防止除零
            std = 1.0  # 回退为1
        data = (data - mean) / std  # 执行z-score归一化
        data[~mask] = 0.0  # 将零体素填零
    else:  # 若全为零
        data = np.zeros_like(data, dtype=np.float32)  # 保持同形状零数组

    factors = tuple(t / s for t, s in zip(target_shape, data.shape))  # 计算缩放因子
    resized = zoom(data, zoom=factors, order=1)  # 线性插值缩放
    return resized.astype(np.float32).ravel()  # 转换类型并扁平化返回


def collect_paths(input_dir: Path) -> list[Path]:  # 收集所有待处理文件路径
    if not input_dir.exists():  # 输入路径校验
        raise FileNotFoundError(f"input dir not found: {input_dir}")  # 抛错提醒
    paths = sorted(input_dir.glob("*.nii.gz"))  # 查找并排序所有NIfTI文件
    if not paths:  # 若无文件
        raise FileNotFoundError(f"no .nii.gz files found in {input_dir}")  # 抛错提醒
    return paths  # 返回路径列表


def main() -> None:  # 程序入口
    config = parse_args()  # 读取参数配置
    paths = collect_paths(config.input_dir)  # 收集输入文件
    config.output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    features = []  # 收集特征的列表
    total = len(paths)  # 记录样本总数
    for index, path in enumerate(paths, start=1):  # 遍历所有体数据
        if index == 1 or index % 25 == 0 or index == total:  # 间隔日志输出
            print(f"[preprocess] {index}/{total} {path.name}")  # 打印进度
        feature = preprocess_volume(  # 处理并获取特征
            path,  # 当前文件路径
            target_shape=config.target_shape,  # 期望输出形状
            low_percentile=config.low_percentile,  # 低裁剪阈值
            high_percentile=config.high_percentile,  # 高裁剪阈值
        )
        features.append(feature)  # 保存特征向量

    X = np.vstack(features).astype(np.float32)  # 堆叠特征为矩阵并转为float32
    np.savez_compressed(config.output_path, X=X, paths=np.array([str(p) for p in paths]))  # 保存压缩npz
    print(f"saved features: {config.output_path}")  # 提示已保存
    print(f"feature matrix shape: {X.shape}")  # 输出矩阵形状


if __name__ == "__main__":  # 确保脚本直接运行时执行main
    main()  # 调用主函数
