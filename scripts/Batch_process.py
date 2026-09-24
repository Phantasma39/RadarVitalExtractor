# ====================== 这里修改路径 ======================
# 也可以从命令行覆盖：python scripts/Batch_process.py <数据文件夹> [输出目录]
data_folder = r"F:\data_new"
output_root = r"F:\my_output_new_DC"
# ==========================================================

import os
import sys

import _bootstrap  # noqa: F401  把 src/ 加入 sys.path，免安装即可 import

if len(sys.argv) >= 2:
    data_folder = sys.argv[1]
if len(sys.argv) >= 3:
    output_root = sys.argv[2]

import numpy as np
from tqdm import tqdm
from radar_project.main import process_single_bin

if not os.path.isdir(data_folder):
    print(f"错误: 数据文件夹不存在 - {data_folder}")
    print("提示: python scripts/Batch_process.py <数据文件夹> [输出目录]")
    sys.exit(1)

# ===== 获取所有 bin 文件 =====
file_list = [f for f in os.listdir(data_folder) if f.endswith(".bin")]
total_files = len(file_list)

print(f"✅ 找到 {total_files} 个 .bin 文件，开始处理...\n")

# ===== 带进度条遍历 =====
# 注意：这里直接复用 process_single_bin，保证批量处理与单文件处理走同一条管线。
# 旧版本在批量循环里另写了一份实现，FFT 点数写成 512（单文件入口是 1024），
# 且 `signal - np.mean(signal)` 对 (12, Frame) 复数矩阵算的是全局标量而非逐通道去直流，
# 导致批量和单文件结果不一致。现在不会再漂移。
for file in tqdm(file_list, desc="整体进度", unit="文件"):

    file_path = os.path.join(data_folder, file)
    name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"正在处理 {name}\n")
    try:
        process_single_bin(file_path, output_root=output_root)
        print(f"✅ 处理完成: {name}\n")

    except Exception as e:
        print(f"\n❌ 处理失败: {file}  | 错误: {e}")

print("\n🎉 所有文件处理完成！")