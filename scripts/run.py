# 统一入口：处理单个 .bin，输出 12 个通道的位移 CSV。
#
# 用法:
#   python scripts/run.py <bin文件> [输出目录] [低截止Hz] [高截止Hz]
#   python scripts/run.py data/TEST.bin
#   python scripts/run.py data/TEST.bin my_output 0.8 2.0     # 只看心跳频带
#
# 不带参数时: 列出可用的 .bin 文件（不会自动开跑，避免误处理大数据）
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  加入 src/ 到 sys.path + 修正控制台编码

from radar_project.main import process_single_bin

ROOT = Path(__file__).resolve().parent.parent

USAGE = """用法:
  python scripts/run.py <bin文件> [输出目录] [低截止Hz] [高截止Hz]

示例:
  python scripts/run.py data/TEST.bin
  python scripts/run.py data/TEST.bin output_TEST
  python scripts/run.py data/TEST.bin output_TEST 0.8 2.0   # 只看心跳频带
"""


def main(argv):
    if len(argv) < 2:
        print(USAGE)
        bins = sorted((ROOT / "data").glob("*.bin")) if (ROOT / "data").is_dir() else []
        if bins:
            print("data/ 目录下可用的 .bin 文件:")
            for b in bins:
                print(f"  data/{b.name}   ({b.stat().st_size / 1024 / 1024:.0f} MB)")
        else:
            print("data/ 目录下没有找到 .bin 文件。")
        return 0

    file_path = Path(argv[1])
    if not file_path.is_absolute():
        # 先按当前目录找，再按仓库根目录找
        if not file_path.exists():
            cand = ROOT / file_path
            if cand.exists():
                file_path = cand
    if not file_path.exists():
        print(f"错误: 文件不存在 - {file_path}")
        return 1

    output_root = argv[2] if len(argv) >= 3 else "output"
    lowcut = float(argv[3]) if len(argv) >= 4 else 0.5
    highcut = float(argv[4]) if len(argv) >= 5 else 5.0

    print(f"输入      : {file_path}")
    print(f"输出根目录: {output_root}")
    print(f"带通      : {lowcut} - {highcut} Hz")
    print()

    process_single_bin(
        str(file_path),
        output_root=output_root,
        fft_len=1024,
        do_dc_eliminate=True,
        lowcut=lowcut,
        highcut=highcut,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)
