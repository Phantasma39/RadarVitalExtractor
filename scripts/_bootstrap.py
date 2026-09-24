# scripts/ 下脚本的公共引导：加入 src/ 到 sys.path + 修正控制台编码。
#
# 用法：在脚本顶部 `import _bootstrap`（依赖脚本以自身目录为 sys.path[0]）。
import os
import sys


def add_src_to_path():
    """
    把仓库的 src/ 加入 sys.path，使 `from radar_project... import ...` 生效。

    这样 `python scripts/xxx.py` 在未执行 `pip install -e .` 的环境下也能直接跑，
    避免 README 里的命令因为 ModuleNotFoundError 而失败。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(os.path.dirname(here), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    return src


def fix_console_encoding():
    """
    中文 Windows 控制台默认 GBK，打印 emoji / 特殊符号会抛 UnicodeEncodeError，
    直接中断整个数据处理流程（例如日志里的 ✅⚠️）。这里切到 UTF-8；
    若终端不支持则退化为替换不可编码字符，保证不会因为一行日志把程序跑挂。
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


SRC_DIR = add_src_to_path()
fix_console_encoding()
