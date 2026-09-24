"""
资源路径定位：同时支持「源码运行」和「PyInstaller 打包成 exe」。

打包后 `__file__` 指向临时解包目录（sys._MEIPASS），项目目录结构不再可用，
因此凡是需要读取随程序分发的文件（模型、配置、样例数据）都要走这里。
"""
from __future__ import annotations

import os
import sys


def is_frozen():
    """是否运行在 PyInstaller 打包出来的可执行文件里。"""
    return bool(getattr(sys, "frozen", False))


def resource_path(*parts):
    """
    返回随程序分发的资源路径（只读）。

    优先级：
      1) PyInstaller 解包目录 (sys._MEIPASS)
      2) 源码树根目录（本文件的上上级）
      3) 可执行文件所在目录
    """
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(meipass)
    # 源码结构: <root>/src/radar_project/resources.py -> <root>
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.dirname(os.path.dirname(here)))
    if getattr(sys, "frozen", False):
        candidates.append(os.path.dirname(os.path.abspath(sys.executable)))
    candidates.append(os.getcwd())

    for base in candidates:
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    # 都不存在时返回第一个候选，便于报错信息里看到期望位置
    return os.path.join(candidates[0], *parts)


def app_dir():
    """
    应用所在目录（用于放"输出"这类可写文件）。

    打包后是 exe 所在目录；源码运行时是仓库根目录。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def default_data_dir():
    """
    默认的数据目录。

    打包后优先用 exe 同级的 data/（方便别人把 bin 放进去）；
    源码运行时用仓库的 data/。
    """
    d = os.path.join(app_dir(), "data")
    if os.path.isdir(d):
        return d
    return resource_path("data")
