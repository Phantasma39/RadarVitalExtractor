# 把雷达数据处理界面打包成 exe（Windows）。
#
# 用法（在仓库根目录）:
#     python scripts/build_exe.py                 # 目录版（启动快，推荐）
#     python scripts/build_exe.py --onefile       # 单文件版
#     python scripts/build_exe.py --clean         # 先清掉旧产物再打包
#     python scripts/build_exe.py --console       # 保留控制台窗口（方便看报错）
#
# 产物:
#     dist/RadarVitalExtractor/RadarVitalExtractor.exe   （目录版）
#     dist/RadarVitalExtractor.exe                       （单文件版，--onefile）
#
# 排错提示
# --------
# 如果打包后运行报类似
#     ModuleNotFoundError: No module named 'scipy._cyutility'
#     ImportError: The `scipy` install you are using seems to be broken
# 那是 PyInstaller 漏收了 scipy 的编译扩展（.pyd）。本项目改用 packaging/radar_gui.spec，
# 里面用 collect_all + collect_dynamic_libs 把它们收全，不应再出现该问题。
# 打包时加 --console 可以看到运行时真实报错。
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "packaging", "radar_gui.spec")
NAME = "RadarVitalExtractor"


def check_pyinstaller():
    try:
        import PyInstaller
        return PyInstaller.__version__
    except ImportError:
        return None


def main(argv):
    onefile = "--onefile" in argv
    clean = "--clean" in argv
    console = "--console" in argv

    ver = check_pyinstaller()
    if ver is None:
        print("错误: 没有安装 PyInstaller。")
        print("请先执行:  pip install pyinstaller")
        return 1
    print(f"PyInstaller 版本: {ver}")

    if not os.path.isfile(SPEC):
        print(f"错误: 找不到 spec 文件 {SPEC}")
        return 1

    if clean:
        for d in ("build", "dist"):
            p = os.path.join(ROOT, d)
            if os.path.isdir(p):
                print(f"清理 {p}")
                shutil.rmtree(p, ignore_errors=True)

    env = dict(os.environ)
    env["RADAR_ONEFILE"] = "1" if onefile else "0"
    env["RADAR_CONSOLE"] = "1" if console else "0"

    args = [sys.executable, "-m", "PyInstaller", "--noconfirm",
            "--distpath", os.path.join(ROOT, "dist"),
            "--workpath", os.path.join(ROOT, "build"), SPEC]

    print("=" * 70)
    print("开始打包" + ("（单文件）" if onefile else "（目录版）"))
    print("=" * 70)

    r = subprocess.run(args, cwd=ROOT, env=env)
    if r.returncode != 0:
        print(f"\n打包失败，退出码 {r.returncode}")
        return r.returncode

    if onefile:
        out_dir = os.path.join(ROOT, "dist")
        exe = os.path.join(out_dir, NAME + (".exe" if os.name == "nt" else ""))
    else:
        out_dir = os.path.join(ROOT, "dist", NAME)
        exe = os.path.join(out_dir, NAME + (".exe" if os.name == "nt" else ""))

    # 在产物旁放一个 data/ 目录，方便直接把 bin 丢进去
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "把bin文件放到这里.txt"), "w",
              encoding="utf-8") as f:
        f.write("把要处理的 .bin 文件放到这个 data 目录里。\n"
                "程序启动后，「数据文件」下方的下拉框会自动列出这里的文件。\n")

    print()
    print("=" * 70)
    print("打包完成")
    print("=" * 70)
    print(f"  可执行文件: {exe}")
    print(f"  存在: {os.path.isfile(exe)}")
    if not onefile and os.path.isdir(out_dir):
        total = 0
        n_files = 0
        for dp, _dn, fn in os.walk(out_dir):
            for x in fn:
                total += os.path.getsize(os.path.join(dp, x))
                n_files += 1
        print(f"  文件数: {n_files}   总大小: {total / 1024 / 1024:.0f} MB")
    print(f"  数据目录: {data_dir}")
    print()
    print("  如果运行时报 ImportError / ModuleNotFoundError，")
    print("  重新打包时加上 --console，就能看到具体缺什么：")
    print("      python scripts/build_exe.py --console")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
