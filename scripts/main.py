# 便捷运行脚本（兼容两种用法）
#
# 用法一（推荐）：带参数，走通用入口，不再依赖下面的硬编码路径
#     python scripts/main.py <bin文件路径> [输出目录]
#
# 用法二（保留原行为）：直接 `python scripts/main.py`，使用下面配置的默认路径
#     —— 适合在 IDE 里直接 F5 运行
import _bootstrap  # noqa: F401  把 src/ 加入 sys.path，免安装即可 import

import os
import sys

# ====================== 用法二的默认路径（只在没传命令行参数时生效）======================
file_path = r"E:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\adc_data_Raw_0.bin"
output_root = r"C:\Users\Phantasma\Desktop\RADAR"
# ========================================================================================

from radar_project.main import process_single_bin


def run_with_defaults():
    if not os.path.exists(file_path):
        print(f"错误: 默认路径不存在 - {file_path}")
        print("提示: 可以改用 `python scripts/main.py <bin文件路径> [输出目录]` 显式指定文件")
        sys.exit(1)
    process_single_bin(file_path, output_root=output_root)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # 带参数：直接交给包内的标准命令行入口
        from radar_project.main import main as cli_main
        cli_main()
    else:
        run_with_defaults()
