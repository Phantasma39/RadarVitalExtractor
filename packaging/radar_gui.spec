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

# =====================================================================
# conda 环境提供的运行库 DLL
# =====================================================================
# 问题现象（打包后运行直接退出）：
#     ImportError: DLL load failed while importing _ctypes: 找不到指定的模块。
#
# 原因：本项目用的 Python 是 conda 环境（sys.prefix 指向 base_prefix 下的
# D:\anaconda3\envs\phantasma，在其上层又建了 .venv）。
# conda 把 libffi / liblzma / openssl / sqlite3 等运行库放在
# <conda 环境>\Library\bin\ 下，而 PyInstaller 的依赖解析器只在标准位置找，
# 于是这些 DLL 没被收集，_ctypes / _lzma / _bz2 等扩展运行时加载失败。
#
# 打包日志里的对应警告：
#     WARNING: Library not found: could not resolve 'ffi.dll',
#              dependency of '...\DLLs\_ctypes.pyd'
#
# 定位方式：从标准库扩展模块（_ctypes.pyd）的实际所在目录反推，
# 这样必然指向"真正被加载的那个 Python 环境"，而不是 .venv。
# 该 DLL 与本项目代码无关，属于 Python 自身的运行库。
def _find_conda_dll_dir():
    import sysconfig

    cands = []
    # 0) conda 环境根（.venv 建立在 conda 之上时，base_prefix 指向 conda 环境）
    bp = getattr(sys, "base_prefix", None)
    if bp:
        cands.append(bp)
    # 1) 从 _ctypes.pyd 的位置反推（最可靠）
    try:
        import _ctypes
        cands.append(os.path.dirname(os.path.abspath(_ctypes.__file__)))
    except Exception:                                       # noqa: BLE001
        pass
    # 2) 标准库安装目录
    for key in ("DLLs", "stdlib", "platstdlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            cands.append(p)
    # 3) 解释器所在目录
    cands.append(os.path.dirname(os.path.abspath(sys.executable)))

    for base in cands:
        cur = base
        for _ in range(4):                                  # 向上找若干层
            d = os.path.join(cur, "Library", "bin")
            if os.path.isdir(d):
                return d, cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    # 兜底：conda 的 base 安装目录
    for guess in (r"D:\anaconda3", r"C:\ProgramData\anaconda3"):
        d = os.path.join(guess, "Library", "bin")
        if os.path.isdir(d):
            return d, guess
    return None, None


_NEEDED_DLLS = [
    "ffi.dll",                    # _ctypes 依赖
    "liblzma.dll",                # _lzma
    "libbz2.dll",                 # _bz2
    "LIBBZ2.dll",                 # Windows 上大小写可能不同
    "libcrypto-3-x64.dll",        # _ssl / _hashlib
    "libssl-3-x64.dll",
    "libexpat.dll",               # pyexpat
    "sqlite3.dll",                # _sqlite3
]

_conda_dll_dir, _conda_root = _find_conda_dll_dir()
if _conda_dll_dir:
    print(f"[spec] conda DLL 目录: {_conda_dll_dir}")

    def _force_dll(name, note=""):
        """
        把某个 DLL 强制设为 conda 版本。

        PyInstaller 可能已经从别处收集了同名 DLL（例如 Tcl/Tk），
        这时必须"替换"而不是"追加"——追加会留下两个同名文件，
        运行时加载到哪个取决于搜索路径，可能仍是旧版本。
        """
        src = os.path.join(_conda_dll_dir, name)
        if not os.path.isfile(src):
            return False
        # 移除已有的同名项（不论来源）
        for i in range(len(binaries) - 1, -1, -1):
            dst_name = os.path.basename(binaries[i][0])
            if dst_name.lower() == name.lower():
                old = binaries.pop(i)
                print(f"[spec] 替换 {name}: {os.path.basename(old[0])} "
                      f"<- {src} {note}")
        binaries.append((src, "."))
        return True

    _added = []
    for _name in _NEEDED_DLLS:
        if _force_dll(_name):
            _added.append(_name)
    print(f"[spec] 已加入 conda 运行库 DLL ({len(_added)} 个): {_added}")
    if len(_added) < 4:
        print("[spec] 警告: 找到的 DLL 偏少，打包后可能仍缺依赖")

    # ---------------- Tcl / Tk 运行时 ----------------
    # 另一个必须处理的问题：PyInstaller 自动收集的 Tcl/Tk 可能与本环境的
    # Python 不匹配。实测本环境 _tkinter.pyd 来自 conda（Tcl 8.6.15），
    # 但自动收集到的是 Tcl 8.6.5，运行时直接报：
    #     version conflict for package "Tcl": have 8.6.5, need exactly 8.6.15
    #     _tkinter.TclError: Can't find a usable init.tcl ...
    # 因此这里把 tcl86t.dll / tk86t.dll 也强制换成 conda 版本，
    # 与 _tcl_data / _tk_data（同样来自 conda）保持一致。
    _tcl_added = []
    for _name in ("tcl86t.dll", "tk86t.dll", "tcl86.dll", "tk86.dll"):
        if _force_dll(_name, "(Tcl/Tk)"):
            _tcl_added.append(_name)
    if _tcl_added:
        print(f"[spec] Tcl/Tk 运行时已对齐为 conda 版本: {_tcl_added}")
    else:
        print("[spec] 未替换 Tcl/Tk DLL（可能非 conda 环境）")
else:
    print("[spec] 未找到 conda Library\\bin，跳过（非 conda 环境属正常）")

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
