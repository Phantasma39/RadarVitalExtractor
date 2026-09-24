# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置（由 scripts/build_exe.py 调用）。

为什么需要这个文件而不是纯命令行参数
------------------------------------
命令行里的 `--collect-submodules scipy` **只收集 Python 模块**，不会带上
编译扩展（.pyd）。scipy 有大量"只有 .pyd、没有对应 .py"的模块，例如
`scipy._cyutility`，一旦漏掉，运行打包出来的程序就会报：

    ModuleNotFoundError: No module named 'scipy._cyutility'
    ImportError: The `scipy` install you are using seems to be broken

所以这里用 collect_dynamic_libs() 把编译扩展一起收进来，
scipy / sklearn / matplotlib 都按同样方式处理。
"""
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# ---- 收集数据文件、二进制、隐藏导入 ----
datas = []
binaries = []
hiddenimports = []

# collect_all 会同时收集：数据文件 + 编译扩展 + 隐藏导入。
# 注意不要加 pandas：它很大且本项目并不需要（GUI 里已改用 numpy）。
for pkg in ("scipy", "sklearn", "matplotlib", "joblib"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
        print(f"[spec] collect_all({pkg}): datas={len(d)} binaries={len(b)} "
              f"hiddenimports={len(h)}")
    except Exception as e:                                   # noqa: BLE001
        print(f"[spec] collect_all({pkg}) 失败: {e}")

# 关键：把编译扩展（.pyd / .so）显式收进来。
# collect_all 一般能覆盖，但某些 scipy 子包会漏，这里再兜一次底。
for pkg in ("scipy", "sklearn"):
    try:
        extra = collect_dynamic_libs(pkg)
        binaries += extra
        print(f"[spec] collect_dynamic_libs({pkg}): {len(extra)} 个二进制")
    except Exception as e:                                   # noqa: BLE001
        print(f"[spec] collect_dynamic_libs({pkg}) 失败: {e}")

# 显式点名这些容易被漏掉的扩展/模块
hiddenimports += [
    "scipy._cyutility",
    "scipy._lib._ccallback_c",
    "scipy._lib.messagestream",
    "scipy._lib._fpumode",
    "scipy.special._ufuncs_cxx",
    "scipy.special._cdflib",
    "scipy.linalg.cython_blas",
    "scipy.linalg.cython_lapack",
    "scipy.sparse.linalg._propack._spropack",
    "scipy.sparse.linalg._propack._dpropack",
    "scipy.sparse.linalg._propack._cpropack",
    "scipy.sparse.linalg._propack._zpropack",
    "scipy.optimize._highs._highs_wrapper",
    "scipy.optimize._highs._highs_constants",
    "sklearn.ensemble._forest",
    "sklearn.tree._tree",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.neighbors._partition_nodes",
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends._backend_tk",
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.ttk",
    "radar_project.resources",
]

# 模型文件打进包里（resources.py 会在 _MEIPASS 下找它们）
models_dir = os.path.join(ROOT, "models")
if os.path.isdir(models_dir):
    datas.append((models_dir, "models"))

# 用不到的 GUI 框架，排除掉以缩小体积
excludes = [
    "PyQt5", "PyQt6", "PySide2", "PySide6", "wx",
    "IPython", "notebook", "jupyter", "pytest", "sphinx",
]

block_cipher = None

a = Analysis(
    [os.path.join(ROOT, "src", "radar_project", "gui.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

ONEFILE = os.environ.get("RADAR_ONEFILE", "0") == "1"
# 需要看运行时报错时，用 `python scripts/build_exe.py --console` 打开控制台
CONSOLE = os.environ.get("RADAR_CONSOLE", "0") == "1"

if ONEFILE:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.datas, [],
        name="RadarVitalExtractor",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        runtime_tmpdir=None,
        console=CONSOLE,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="RadarVitalExtractor",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=CONSOLE,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe, a.binaries, a.zipfiles, a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="RadarVitalExtractor",
    )
