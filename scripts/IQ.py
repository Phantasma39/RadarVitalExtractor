# IQ 星座图分析脚本（薄包装，实际逻辑在 src/radar_project/IQ.py）
#
# 用法：python scripts/IQ.py <bin文件路径> [range_bin] [输出目录]
import _bootstrap  # noqa: F401  把 src/ 加入 sys.path，免安装即可 import

import sys

from radar_project.IQ import main as iq_main

if __name__ == "__main__":
    iq_main()
