"""
雷达数据处理可视化界面（Tkinter）。

功能
----
- 可视化选择要处理的 .bin 文件
- 选择输出目录、结果命名方式、CSV 分隔符、是否带表头
- 处理参数：FFT 点数、DC 消除开关、带通上下限、TX 数、RX 数、帧数、chirp 数
- 处理完成后在界面内直接展示 12 个通道的位移曲线（3x4 网格）
- 显示汇总表（通道 / TX / RX / 目标 bin / 距离 / RANSAC / 概率 / 峰峰值）
- 可单独查看某个通道的放大曲线
- 可通过 "显示频带" 在不重跑的情况下切换渲染频带
- 左侧有一块空的 "收发天线处理（预留区）" Frame，后续要加 TX/RX 天线相关的
  处理时，把控件加进 `self.antenna_frame` 即可，界面其余部分不用动

启动
----
    python scripts/gui.py
或
    python -m radar_project.gui
"""
from __future__ import annotations

import gc
import os
import queue
import threading
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from scipy.signal import butter, detrend, filtfilt

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure  # noqa: E402

from radar_project.utils import (  # noqa: E402
    LAYOUT_CHOICES,
    LAYOUT_DEFAULT,
    read_and_decode,
)
from radar_project.resources import app_dir, default_data_dir  # noqa: E402
from radar_project.range_fft import range_fft, final_signal  # noqa: E402
from radar_project.DC_Eliminate import fit_circle_ransac_iq  # noqa: E402
from radar_project.Judge import judge_channel  # noqa: E402

# ------------------------- 中文字体 -------------------------
# matplotlib 默认字体不含中文，图上的中文会变成方框。这里自动挑一个可用的中文字体。
_CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
    "PingFang SC", "Heiti SC",
]


def setup_cjk_font():
    """把 matplotlib 的字体设成系统中可用的中文字体，找不到就保持默认。"""
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = next((n for n in _CJK_CANDIDATES if n in available), None)
    if picked:
        matplotlib.rcParams["font.sans-serif"] = [picked] + \
            list(matplotlib.rcParams.get("font.sans-serif", []))
    # 用中文字体时负号常显示为方块，改用 ASCII 减号
    matplotlib.rcParams["axes.unicode_minus"] = False
    return picked


_CJK_FONT = setup_cjk_font()


# =========================================================================
# 显示/滤波频带预设
# =========================================================================
# 脉搏波基频约 0.8~2 Hz（48~120 bpm）；含二次谐波时到 ~3 Hz。
# 呼吸 0.1~0.5 Hz 与脉搏波相邻，所以带通选择直接决定能不能把心跳分离出来。
BAND_PRESETS = {
    "脉搏波 0.8-2.0 Hz (48-120bpm)": (0.8, 2.0),
    "脉搏波+谐波 0.8-3.0 Hz": (0.8, 3.0),
    "脉搏波宽 0.7-4.0 Hz": (0.7, 4.0),
    "脉搏波窄 1.0-1.8 Hz": (1.0, 1.8),
    "呼吸 0.1-0.5 Hz": (0.1, 0.5),
    "呼吸+脉搏 0.1-2.0 Hz": (0.1, 2.0),
    "低频全带 0.5-5.0 Hz": (0.5, 5.0),
    "宽带 0.5-10 Hz": (0.5, 10.0),
}
BAND_CUSTOM = "自定义 (用下方带通低/高)"
BAND_CHOICES = list(BAND_PRESETS.keys()) + [BAND_CUSTOM]
BAND_DEFAULT = "低频全带 0.5-5.0 Hz"

# 目标 bin 选择策略
BIN_MODE_PER_CH = "各自选峰 (每通道各选最强，默认)"
BIN_MODE_ESTIMATE = "估计范围 (在距离范围内估一个共同点)"
BIN_MODE_CHOICES = [BIN_MODE_PER_CH, BIN_MODE_ESTIMATE]
BIN_MODE_DEFAULT = BIN_MODE_PER_CH


# =========================================================================
# 参数预设
# =========================================================================
# 每个预设就是一份参数字典，键名与界面上的变量名一一对应。
# 用户自定义预设保存在 config/presets.json，与内置预设合并显示。
# 打包后 _MEIPASS 是只读的临时解包目录，必须写到 exe 同级，否则保存会失败。
CONFIG_DIR = os.path.join(app_dir(), "config")
PRESET_FILE = os.path.join(CONFIG_DIR, "presets.json")

BUILTIN_PRESETS = {
    "★ 我的配置 (IWR1843 4RX Complex IIQQ)": {
        "num_frames": "6250",
        "num_chirps": "24",
        "num_rx": "4",
        "num_tx": "3",
        "num_samples": "256",
        "fft_len": "1024",
        "frame_rate": "250",
        "fc": "77e9",
        "slope": "80e12",
        "fs_adc": "1e7",
        "layout": "IIQQ",
        "do_dc_eliminate": True,
        "lowcut": "0.5",
        "highcut": "5.0",
        "seed": "",
        "note": "当前正在用的采集配置。IIQQ 是实测正确的 I/Q 排布。",
    },
    "IWR1843 4RX - 只看心跳 0.8-2Hz": {
        "num_frames": "6250",
        "num_chirps": "24",
        "num_rx": "4",
        "num_tx": "3",
        "num_samples": "256",
        "fft_len": "1024",
        "frame_rate": "250",
        "fc": "77e9",
        "slope": "80e12",
        "fs_adc": "1e7",
        "layout": "IIQQ",
        "do_dc_eliminate": True,
        "lowcut": "0.8",
        "highcut": "2.0",
        "seed": "",
        "note": "脉搏波基频带，滤掉呼吸分量。",
    },
    "IWR1843 4RX - 脉搏波+谐波 0.8-3Hz": {
        "num_frames": "6250",
        "num_chirps": "24",
        "num_rx": "4",
        "num_tx": "3",
        "num_samples": "256",
        "fft_len": "1024",
        "frame_rate": "250",
        "fc": "77e9",
        "slope": "80e12",
        "fs_adc": "1e7",
        "layout": "IIQQ",
        "do_dc_eliminate": True,
        "lowcut": "0.8",
        "highcut": "3.0",
        "seed": "",
        "note": "含脉搏波二次谐波，波形更完整。",
    },
    "IWR1843 4RX - 只看呼吸 0.1-0.5Hz": {
        "num_frames": "6250",
        "num_chirps": "24",
        "num_rx": "4",
        "num_tx": "3",
        "num_samples": "256",
        "fft_len": "1024",
        "frame_rate": "250",
        "fc": "77e9",
        "slope": "80e12",
        "fs_adc": "1e7",
        "layout": "IIQQ",
        "do_dc_eliminate": True,
        "lowcut": "0.1",
        "highcut": "0.5",
        "seed": "",
        "note": "呼吸频带。",
    },
    "2RX Complex (只用 lane1)": {
        "num_frames": "6250",
        "num_chirps": "24",
        "num_rx": "2",
        "num_tx": "3",
        "num_samples": "256",
        "fft_len": "1024",
        "frame_rate": "250",
        "fc": "77e9",
        "slope": "80e12",
        "fs_adc": "1e7",
        "layout": "IIQQ",
        "do_dc_eliminate": True,
        "lowcut": "0.5",
        "highcut": "5.0",
        "seed": "",
        "note": "只开了 2 个接收通道时用（每个采样点 4 个 int16）。",
    },
}


def load_user_presets():
    """读取用户自定义预设；文件不存在或损坏时返回空字典。"""
    try:
        import json
        with open(PRESET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def save_user_presets(presets):
    """写入用户自定义预设。"""
    import json
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(PRESET_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)

# ------------------------- 常量 -------------------------
C_LIGHT = 3e8
FC_DEFAULT = 77e9
SLOPE_DEFAULT = 80e12
FS_ADC_DEFAULT = 1e7


# =========================================================================
# 计算部分（与 process_single_bin 同一条管线，便于在后台线程里跑）
# =========================================================================
def compute_all_channels(
    bin_path,
    *,
    num_frames=6250,
    num_chirps=24,
    num_rx=4,
    num_samples=256,
    num_tx=3,
    fft_len=1024,
    do_dc_eliminate=True,
    frame_rate=250.0,
    fc=FC_DEFAULT,
    slope=SLOPE_DEFAULT,
    fs_adc=FS_ADC_DEFAULT,
    layout=LAYOUT_DEFAULT,
    display_lowcut=0.5,
    display_highcut=5.0,
    filter_order=4,
    seed=None,
    min_dist=None,
    max_dist=None,
    bin_mode="per_channel",
    despike=False,
    despike_threshold=15.0,
    despike_window=51,
    despike_max_width=3,
    fast_preview=False,
    preview_frames=1500,
    progress=None,
    should_stop=None,
):
    """
    跑完整管线，返回一个结果字典。

    progress : callable(str, float) or None
        进度回调 (消息, 0~1)。
    should_stop : callable() -> bool or None
        返回 True 时提前中止，抛 KeyboardInterrupt。
    fast_preview : bool
        只处理文件前 preview_frames 帧，并降低 RANSAC 迭代次数，用于快速预览。
    """
    def report(msg, frac=None):
        if progress:
            progress(msg, frac)

    def check_stop():
        if should_stop and should_stop():
            raise KeyboardInterrupt("用户中止")

    # ---- 快速预览：限制帧数 + 降低 RANSAC 迭代 ----
    n_iter = 20000
    max_samples = None
    if fast_preview:
        orig_frames = num_frames
        num_frames = min(num_frames, int(preview_frames))
        max_samples = num_frames * num_chirps * num_samples
        n_iter = 2000
        report(f"快速预览: 只处理前 {num_frames} 帧 "
               f"(原 {orig_frames} 帧), RANSAC 迭代降为 {n_iter}", 0.01)

    report("读取并解码 bin 文件 ...", 0.02)
    virtual = read_and_decode(
        bin_path,
        num_frames=num_frames,
        num_chirps=num_chirps,
        num_rx=num_rx,
        num_samples=num_samples,
        num_tx=num_tx,
        max_samples=max_samples,
        layout=layout,
    )
    check_stop()
    virtual_shape = virtual.shape

    # range_fft 会返回 complex128，full-scale 时单个数组就要 2.4GB。
    # 先把输入降到 complex64，中间量减半，峰值内存显著下降。
    if virtual.dtype != np.complex64:
        virtual = virtual.astype(np.complex64)

    report("Range FFT ...", 0.20)
    range_data = range_fft(
        virtual, axis=-1, fft_len=fft_len, window_type="hann",
        remove_dc=True, keep_positive=True, output="complex",
    )
    # virtual 已经不再需要，立刻释放（full-scale 约 0.6GB）
    del virtual
    gc.collect()
    check_stop()

    report("选取目标 range bin ...", 0.35)
    power = np.mean(np.abs(range_data), axis=(1, 2))
    n_bin = power.shape[1]

    # 距离轴: R(bin) = bin * fs * c / (2 * slope * fft_len)
    per_bin = fs_adc * C_LIGHT / (2 * slope * fft_len)
    dist_axis = np.arange(n_bin, dtype=np.float64) * per_bin

    # ---- 把「距离范围」(米) 换算成 bin 范围 ----
    # 用户按实际距离给范围更自然，bin 由程序算，不必手动换算。
    def _dist_to_bin(d):
        return int(round(d / per_bin))

    lo_b = 0 if min_dist is None else max(0, _dist_to_bin(min_dist))
    hi_b = n_bin if max_dist is None else min(n_bin, _dist_to_bin(max_dist) + 1)
    if hi_b <= lo_b:
        # 范围太窄被压成一个 bin，退化为单 bin
        hi_b = min(n_bin, lo_b + 1)
    if hi_b <= lo_b:
        raise ValueError(
            f"目标距离范围换算后无效: "
            f"[{min_dist}, {max_dist}] m -> bin [{lo_b}, {hi_b})")
    check_stop()

    sub = power[:, lo_b:hi_b]

    # ---- 选目标 bin ----
    # "各自选最强峰"在多反射体场景下会让不同通道选到不同距离
    # （实测：前 4 通道 0.46m、后 8 通道 1.10m）。策略：
    #   per_channel : 每个通道各自选（默认）
    #   estimate    : 在距离范围内用 12 通道平均谱估一个共同点，全部通道共用
    sub = power[:, lo_b:hi_b]
    mean_sub = sub.mean(axis=0)
    est_bin = int(np.argmax(mean_sub)) + lo_b
    est_dist = float(dist_axis[est_bin])

    if bin_mode == "estimate":
        target_bins = np.full(power.shape[0], est_bin, dtype=int)
    else:  # per_channel
        target_bins = np.argmax(sub, axis=1) + lo_b

    # 全通道平均谱里的候选峰（供界面参考，避免选错）
    mean_spec = power.mean(axis=0)
    cand_idx = np.argsort(-mean_spec)[:8]
    cand = []
    for b in sorted(int(x) for x in cand_idx):
        cand.append((b, dist_axis[b], float(mean_spec[b])))

    signal_raw = final_signal(range_data, target_bins)     # (12, F) 复数
    check_stop()

    # 到这里 range_data 已经用完，可以释放（full-scale complex64 约 1.2GB）。
    # 只留 3 条小曲线给"距离谱"视图，避免处理完还一直占着内存。
    spec_all = np.abs(range_data).mean(axis=(1, 2))       # (12, n_bin)
    range_info = {
        "dist_axis": dist_axis,
        "prof_mean": spec_all.mean(axis=0),
        "prof_min": spec_all.min(axis=0),
        "prof_max": spec_all.max(axis=0),
    }
    del range_data, spec_all
    gc.collect()

    n_ch = signal_raw.shape[0]

    # ---- 相位 -> 位移（原始，未做 DC 消除）----
    phase_raw = np.unwrap(np.angle(signal_raw), axis=1)[:, 1:]
    disp_no_dc = np.zeros((n_ch, phase_raw.shape[1]))
    disp_no_dc[:] = (C_LIGHT / (4 * np.pi * fc)) * (
        phase_raw - phase_raw[:, [0]]
    )
    disp_no_dc = detrend(disp_no_dc, axis=1)

    # ---- DC 消除 ----
    fit_info = [None] * n_ch
    signal = signal_raw.copy()
    if do_dc_eliminate:
        for ch in range(n_ch):
            check_stop()
            report(f"RANSAC 圆拟合 通道 {ch + 1}/{n_ch} ...",
                   0.35 + 0.35 * (ch + 1) / n_ch)
            xc, yc, R = fit_circle_ransac_iq(
                signal[ch], n_iter=n_iter, random_state=seed, verbose=False
            )
            if xc is None:
                fit_info[ch] = None
            else:
                fit_info[ch] = (xc, yc, R)
                signal[ch] = signal[ch] - xc - yc * 1j

    # ---- 相位 -> 位移（DC 消除后、滤波前）----
    report("相位解缠与位移换算 ...", 0.72)
    phase = np.unwrap(np.angle(signal), axis=1)[:, 1:]
    disp_raw = np.zeros((n_ch, phase.shape[1]))
    disp_raw[:] = (C_LIGHT / (4 * np.pi * fc)) * (phase - phase[:, [0]])
    disp_raw = detrend(disp_raw, axis=1)
    check_stop()

    # ---- 带通滤波 ----
    # 关键：把「未滤波位移」disp_raw 一并返回，界面就能在不重跑的情况下
    # 任意改变滤波范围（带通只花几十毫秒）。
    report("带通滤波 ...", 0.80)
    disp = _bandpass(disp_raw, frame_rate, display_lowcut, display_highcut,
                     filter_order)

    # ---- 大波动检查（可选）----
    spike_report = None
    if despike:
        report("大波动检查 ...", 0.86)
        disp, spike_report = despike_matrix(
            disp, threshold=despike_threshold, window=despike_window,
            max_width=despike_max_width)
        if progress and spike_report:
            n_rep = sum(r["n_replaced"] for r in spike_report)
            n_skip = sum(r["skipped_runs"] for r in spike_report)
            report(f"  大波动检查: 替换 {n_rep} 点, "
                   f"跳过 {n_skip} 段连续波形", 0.88)

    # ---- 质量分 ----
    report("通道质量评估 ...", 0.92)
    probs = []
    good = []
    for ch in range(n_ch):
        ok, p = judge_channel(disp[ch], frame_rate)
        probs.append(float(p))
        good.append(bool(ok))

    # 每条曲线对应的时间轴
    n_samp = disp.shape[1]
    t = np.arange(n_samp) / frame_rate

    report("完成", 1.0)
    return {
        "bin_path": bin_path,
        "virtual_shape": tuple(virtual_shape),
        "target_bins": np.asarray(target_bins),
        "bin_candidates": cand,
        "bin_range": (lo_b, hi_b),
        "bin_mode": bin_mode,
        "bin_estimate": est_bin,
        "dist_estimate": est_dist,
        "per_bin": per_bin,
        "dist_range": (dist_axis[lo_b], dist_axis[min(hi_b, n_bin) - 1]),
        "range_info": range_info,      # 距离谱汇总（不保留完整 range_data）
        "fit_info": fit_info,
        "signal_raw": signal_raw,
        "disp_no_dc": disp_no_dc,
        "disp_raw": disp_raw,          # 未滤波，供界面任意改滤波范围
        "disp": disp,                  # 当前频带滤波后（可能已做大波动替换）
        "spike_report": spike_report,
        "despike_cfg": dict(enabled=bool(despike),
                            threshold=despike_threshold,
                            window=despike_window,
                            max_width=despike_max_width),
        "band": (display_lowcut, display_highcut),
        "bands": {(display_lowcut, display_highcut): disp},
        "probs": np.asarray(probs),
        "good": np.asarray(good),
        "t": t,
        "n_ch": n_ch,
        "num_rx": num_rx,
        "num_tx": num_tx,
        "frame_rate": frame_rate,
        "fft_len": fft_len,
        "sample_rate": fs_adc,
        "slope": slope,
        "params": dict(
            num_frames=num_frames, num_chirps=num_chirps, num_rx=num_rx,
            num_samples=num_samples, num_tx=num_tx, fft_len=fft_len,
            do_dc_eliminate=do_dc_eliminate, frame_rate=frame_rate, fc=fc,
            slope=slope, fs_adc=fs_adc, lowcut=display_lowcut,
            highcut=display_highcut, filter_order=filter_order,
            layout=layout, min_dist=min_dist, max_dist=max_dist,
            bin_mode=bin_mode,
        ),
    }


def build_filename(ch, prob, prefix="channel", suffix="", ext=".csv",
                   tag_prob=True):
    """按命名规则生成一个通道的 CSV 文件名。"""
    prefix = (prefix or "channel").strip()
    suffix = (suffix or "").strip()
    ext = (ext or ".csv").strip()
    if not ext.startswith("."):
        ext = "." + ext
    name = f"{prefix}_{ch}"
    if suffix:
        name += f"_{suffix}"
    if tag_prob:
        name += f"_prob_{int(prob * 100):02d}"
    return name + ext


def write_displacement_csvs(res, outdir, disp=None,
                            prefix="channel", suffix="", ext=".csv",
                            delim=",", header=True, tag_prob=True):
    """
    把 12 个通道的位移写成 CSV。

    返回写出的文件路径列表。
    """
    os.makedirs(outdir, exist_ok=True)
    if disp is None:
        disp = res["disp"]
    t = res["t"]
    probs = res["probs"]
    paths = []
    for ch in range(res["n_ch"]):
        fn = os.path.join(outdir, build_filename(
            ch, probs[ch], prefix=prefix, suffix=suffix, ext=ext,
            tag_prob=tag_prob))
        arr = np.column_stack((t, disp[ch]))
        if header:
            np.savetxt(fn, arr, delimiter=delim,
                       header=f"time(s){delim}displacement(m)", comments="")
        else:
            np.savetxt(fn, arr, delimiter=delim)
        paths.append(fn)
    return paths


def _process_one_file(path, outdir, compute_kwargs, export_opts):
    """
    在**子进程**里处理单个文件：跑完整管线 + 写 CSV。

    必须是模块级函数（不能是闭包/局部函数），否则 multiprocessing
    无法用 pickle 把它传给子进程。

    返回 (path, out_dir, None) 或 (path, None, 错误信息)。
    """
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        # 子进程里不需要逐步骤回调（回调对象无法跨进程），静默跑完即可
        res = compute_all_channels(path, **compute_kwargs)
        sub_out = os.path.join(outdir, name)
        write_displacement_csvs(res, sub_out, **export_opts)
        # 结果里有大数组，处理完立刻释放，避免子进程常驻占用
        res = None
        gc.collect()
        return (path, sub_out, None)
    except Exception as e:                             # noqa: BLE001
        gc.collect()
        return (path, None, f"{type(e).__name__}: {e}")


def output_dir_for(bin_path, outdir):
    """某个 bin 文件对应的输出子目录。"""
    return os.path.join(outdir, os.path.splitext(os.path.basename(bin_path))[0])


def is_already_done(bin_path, outdir, export_opts=None, n_ch=12):
    """
    判断某个 bin 是否已经处理完成（用于批处理的断点续处理）。

    判定标准（必须同时满足）：
      1) 输出子目录存在；
      2) 目录里有 n_ch 个 CSV 文件（12 个通道一个不少）；
      3) 这些文件名符合当前的命名规则（前缀 / 后缀 / 扩展名）。

    第 3 条很重要：如果你改了前缀就重新跑，旧结果不该被当成"已完成"，
    否则会得到一批命名规则混杂的结果目录。

    返回 (bool, 说明字符串)。
    """
    opts = dict(export_opts or {})
    prefix = (opts.get("prefix") or "channel").strip()
    suffix = (opts.get("suffix") or "").strip()
    ext = (opts.get("ext") or ".csv").strip()
    if not ext.startswith("."):
        ext = "." + ext

    sub = output_dir_for(bin_path, outdir)
    if not os.path.isdir(sub):
        return False, "无输出目录"

    try:
        names = os.listdir(sub)
    except OSError as e:
        return False, f"无法读取输出目录: {e}"

    # 只统计符合当前命名规则的文件：<前缀>_<通道>[ _<后缀>]...<ext>
    head = f"{prefix}_"
    tail = f"_{suffix}" if suffix else ""
    found = set()
    for fn in names:
        if not fn.lower().endswith(ext.lower()):
            continue
        stem = fn[: len(fn) - len(ext)]
        if not stem.startswith(head):
            continue
        rest = stem[len(head):]
        if tail:
            if not rest.endswith(tail):
                continue
            rest = rest[: len(rest) - len(tail)]
        else:
            # 无后缀时，通道号后面可能还跟着 _prob_XX，需要切掉
            rest = rest.split("_", 1)[0]
        if rest.isdigit():
            found.add(int(rest))

    want = set(range(n_ch))
    if len(found) == n_ch:
        return True, f"{n_ch} 个通道齐全"
    missing = sorted(want - found)
    if not found:
        return False, "无符合命名规则的输出"
    return False, f"只有 {len(found)}/{n_ch} 个通道（缺 {missing}）"


def scan_done(files, outdir, export_opts=None):
    """
    扫描哪些文件已经处理完成。

    返回 (done_map, todo_files, detail_list)
      done_map  : {path: 说明}
      todo_files: 还需要处理的文件列表（保持输入顺序）
      detail_list: [(path, 是否完成, 说明), ...]
    """
    done_map = {}
    todo = []
    detail = []
    for p in files:
        ok, why = is_already_done(p, outdir, export_opts)
        detail.append((p, ok, why))
        if ok:
            done_map[p] = why
        else:
            todo.append(p)
    return done_map, todo, detail


def default_worker_count(n_files=None):
    """
    默认并行进程数。

    每个进程处理一个 bin 时会占用一两 GB 内存（range FFT 的中间数组很大），
    所以不能简单用满所有核心，要按内存约束压一压。
    """
    cpus = os.cpu_count() or 2
    # 留一个核心给界面，且上限 4：再多内存容易吃不消
    w = max(1, min(4, cpus - 1))
    if n_files is not None:
        w = max(1, min(w, n_files))
    return w


def process_folder(folder, outdir, *, file_glob="*.bin", recursive=False,
                   progress=None, should_stop=None, export_opts=None,
                   workers=None, files=None, resume=False, **compute_kwargs):
    """
    批量处理一个文件夹里的所有 .bin 文件，每个文件输出 12 个通道的 CSV。

    **多进程并行**：每个文件交给一个子进程处理，默认开
    min(4, CPU核数-1) 个进程。

    为什么用多进程而不是多线程：处理耗时几乎全在 RANSAC 和 FFT 上，
    是纯 CPU 计算；Python 的多线程受 GIL 限制，同一时刻只有一个线程
    能执行 Python 字节码，用线程池基本不会提速，必须用进程池。

    参数
    ----
    folder : str
        输入文件夹。
    outdir : str
        输出根目录；每个 bin 会在其下建一个以文件名命名的子目录。
    progress : callable(str, float) or None
        进度回调 (消息, 0~1)。并行时按"文件"粒度汇报。
    workers : int or None
        并行进程数。None = 自动；1 = 串行（不起子进程）。
    export_opts : dict or None
        CSV 命名/格式选项，见 write_displacement_csvs。
    resume : bool
        断点续处理。为 True 时先扫描输出目录，
        跳过已经处理完成的文件，只处理剩下的。
        "已完成"的判定见 is_already_done（12 个通道齐全 且 命名规则一致）。

    返回
    ----
    dict: {'ok': [(path, out_dir), ...], 'failed': [(path, err), ...],
           'skipped': [path, ...], 'resumed': [(path, 说明), ...],
           'workers': int, 'seconds': float}
    """
    import glob as _glob
    import time

    t_start = time.perf_counter()
    folder = os.path.abspath(folder)
    outdir = os.path.abspath(outdir)
    if not os.path.isdir(folder):
        raise ValueError(f"输入文件夹不存在: {folder}")

    pattern = os.path.join(folder, "**", file_glob) if recursive \
        else os.path.join(folder, file_glob)
    if files is None:
        files = sorted(_glob.glob(pattern, recursive=recursive))
        files = [f for f in files if os.path.isfile(f)]
    else:
        files = [f for f in files if os.path.isfile(f)]

    n = len(files)
    result = {"ok": [], "failed": [], "skipped": [],
              "resumed": [], "workers": 1, "seconds": 0.0}
    if n == 0:
        if progress:
            progress(f"文件夹里没有匹配 {file_glob} 的文件", 1.0)
        return result

    opts = dict(export_opts or {})

    # ---------- 断点续处理：先扫描哪些已经做完了 ----------
    if resume:
        done_map, todo, detail = scan_done(files, outdir, opts)
        result["resumed"] = [(p, done_map[p]) for p in files if p in done_map]
        if progress:
            progress(f"断点续处理：已扫描 {n} 个文件，"
                     f"已完成 {len(done_map)} 个，待处理 {len(todo)} 个", 0.0)
            # 已完成的逐条报告，便于确认
            for p, ok, why in detail:
                if ok:
                    progress(f"  [跳过] {os.path.basename(p)}: {why}", None)
            for p, ok, why in detail:
                if not ok and why != "无输出目录":
                    # 只报告"开始做了但没做完"的，纯新的文件不啰嗦
                    progress(f"  [待处理] {os.path.basename(p)}: {why}", None)
        if not todo:
            result["seconds"] = time.perf_counter() - t_start
            if progress:
                progress(f"断点续处理：{n} 个文件全部已完成，无需处理", 1.0)
            return result
        files = todo
        n = len(files)

    n_workers = default_worker_count(n) if workers is None else int(workers)
    n_workers = max(1, min(n_workers, n))

    # 串行路径：workers=1 时直接走；也作为并行失败时的回退实现
    def _serial_run(only_files):
        result["workers"] = 1
        if progress:
            progress(f"单进程处理 {len(only_files)} 个文件 ...", 0.0)
        for i, path in enumerate(only_files):
            name = os.path.splitext(os.path.basename(path))[0]

            def sub(msg, frac=None, _i=i, _name=name):
                if progress:
                    base = _i / max(1, len(only_files))
                    overall = base if frac is None else base + frac / max(1, len(only_files))
                    progress(f"[{_i+1}/{len(only_files)}] {_name}: {msg}",
                             min(overall, 1.0))

            if should_stop and should_stop():
                result["skipped"].append(path)
                continue

            res = None
            try:
                res = compute_all_channels(path, progress=sub,
                                           should_stop=should_stop,
                                           **compute_kwargs)
                sub("写出 CSV ...", 0.98)
                sub_out = os.path.join(outdir, name)
                write_displacement_csvs(res, sub_out, **opts)
                result["ok"].append((path, sub_out))
                sub("完成", 1.0)
            except KeyboardInterrupt:
                result["skipped"].append(path)
                break
            except Exception as e:                   # noqa: BLE001
                result["failed"].append((path, f"{type(e).__name__}: {e}"))
                if progress:
                    progress(
                        f"[{i+1}/{len(only_files)}] {name}: "
                        f"失败 - {type(e).__name__}: {e}",
                        (i + 1) / max(1, len(only_files)))
            finally:
                res = None
                gc.collect()

    if n_workers <= 1:
        _serial_run(files)
        result["seconds"] = time.perf_counter() - t_start
        if progress:
            progress(f"批量处理结束: 成功 {len(result['ok'])}, "
                     f"失败 {len(result['failed'])}, "
                     f"跳过 {len(result['skipped'])} / 共 {n}，"
                     f"耗时 {result['seconds']:.1f}s", 1.0)
        return result

    # ---------- 多进程并行 ----------
    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed
    except ImportError:
        return process_folder(folder, outdir, file_glob=file_glob,
                              recursive=recursive, progress=progress,
                              should_stop=should_stop, export_opts=opts,
                              workers=1, **compute_kwargs)

    result["workers"] = n_workers
    if progress:
        progress(f"多进程处理 {n} 个文件（{n_workers} 个进程并行）...", 0.0)

    done = 0
    executor = None
    try:
        # fork 在 Windows 上不可用，spawn 会重新 import 本模块；
        # 因此 worker 必须是模块级函数、参数必须是可 pickle 的普通对象。
        executor = ProcessPoolExecutor(max_workers=n_workers)
        futures = {
            executor.submit(_process_one_file, p, outdir,
                            dict(compute_kwargs), opts): p
            for p in files
        }
        # 用超时轮询，保证「中止」按钮在并行时也能及时生效
        pending = set(futures)
        while pending:
            if should_stop and should_stop():
                for f in pending:
                    f.cancel()
                result["skipped"].extend(futures[f] for f in pending)
                pending.clear()
                break
            finished = set()
            try:
                for f in as_completed(pending, timeout=0.3):
                    finished.add(f)
                    path, sub_out, err = f.result()
                    name = os.path.splitext(os.path.basename(path))[0]
                    done += 1
                    if err is None:
                        result["ok"].append((path, sub_out))
                        if progress:
                            progress(f"[{done}/{n}] {name}: 完成",
                                     done / n)
                    else:
                        result["failed"].append((path, err))
                        if progress:
                            progress(f"[{done}/{n}] {name}: 失败 - {err}",
                                     done / n)
                    if len(finished) >= 1:
                        break          # 每轮只取一个，界面才能及时刷新
            except TimeoutError:
                pass
            pending -= finished
    except Exception as e:                             # noqa: BLE001
        # 进程池起不来（沙箱限制、打包环境异常等）时退回串行，保证功能可用
        if progress:
            progress(f"多进程不可用（{type(e).__name__}: {e}），改用单进程", 0.0)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:                          # noqa: BLE001
                pass
        # 只处理还没成功的文件，避免重复处理（传 files 而不是重新 glob）
        already = ({p for p, _ in result["ok"]}
                   | {p for p, _ in result["failed"]}
                   | set(result["skipped"]))
        remain = [p for p in files if p not in already]
        if remain:
            _serial_run(remain)
        result["workers"] = 1
    finally:
        if executor is not None:
            try:
                executor.shutdown(wait=True)
            except Exception:                          # noqa: BLE001
                pass
        gc.collect()

    result["seconds"] = time.perf_counter() - t_start
    if progress:
        progress(f"批量处理结束: 成功 {len(result['ok'])}, "
                 f"失败 {len(result['failed'])}, "
                 f"跳过 {len(result['skipped'])} / 共 {n}，"
                 f"耗时 {result['seconds']:.1f}s（{result['workers']} 进程）", 1.0)
    return result


def _make_compact_toolbar(canvas, master):
    """
    在 master 里放一条紧凑的 matplotlib 标准导航栏。

    自带这些按钮（与 matplotlib 默认一致）：
        Home  : 返回原始视野（就是你要的"返回"）
        Back / Forward : 在历史视野之间前后切换
        Pan   : 按住拖动平移
        Zoom  : 拖出一个矩形框，松开即放大框中区域
        Save  : 保存图片

    用 classic 模式让按钮更小；按钮太小会挤，所以只去掉 Subplots
    （它会另开窗口，用不到）。返回构造好的 toolbar。
    """
    try:
        tb = NavigationToolbar2Tk(canvas, master, pack_toolbar=False)
        tb.update()
        # 去掉用不到的 Subplots 按钮（会弹独立窗口）
        btn = tb._buttons.pop("Subplots", None)
        if btn is not None:
            btn.destroy()
        # 关键：把当前视野压入导航栈作为"原始视野"，
        # 否则 Home（返回）按钮没有可回退的目标，点了没反应。
        tb.push_current()
        tb.pack(side=tk.LEFT, fill=tk.X)
        return tb
    except Exception:                                # noqa: BLE001
        # 工具栏构造失败也不该让整个界面起不来
        return None


def _bandpass(data, fs, lowcut, highcut, order=4):
    """
    零相位 Butterworth 带通（1D/2D 均可，沿最后一维）。

    低频带（如呼吸 0.1-0.5 Hz）直接以 250 Hz 采样率滤波时，归一化截止频率
    低至 0.0008，Butterworth 的数值条件极差，filtfilt 会溢出（实测出现 1e58
    量级的荒谬值）。因此当采样率远高于频带上限时，先把信号抗混叠降采样到
    ~8 倍上限频率再滤，既数值稳定又快得多。

    抽取有下限保护：不能让抽取后的样本数少于 filtfilt 的 padlen 要求，
    否则会报 "length of the input vector x must be greater than padlen"。
    """
    data = np.asarray(data)
    nyq = 0.5 * fs

    if not 0 < lowcut < highcut < nyq:
        raise ValueError(f"非法频带: {lowcut}-{highcut} Hz (fs={fs})")

    n = data.shape[-1]
    fs_use = fs
    work = data

    # ---- 计算安全的抽取倍数 ----
    # 目标: 抽取后采样率 ≈ 8 x 上限频率，但同时
    #   1) 不低于 4 x 上限频率（留足过渡带）
    #   2) 抽取后样本数 >= 256（远大于 filtfilt 的 padlen，且低频滤波稳定）
    target_fs = 8.0 * highcut
    q = int(fs // target_fs) if target_fs > 0 else 1
    q = max(q, 1)
    if q > 1:
        min_len = 256
        max_q = max(1, n // min_len)
        q = min(q, max_q)
        # 抽取后采样率不能低于 4 倍上限频率
        while q > 1 and fs / q < 4.0 * highcut:
            q -= 1
    if q > 1:
        try:
            from scipy.signal import decimate
            work = decimate(data, q, axis=-1, ftype="iir", zero_phase=True)
            fs_use = fs / q
        except Exception:
            work = data
            fs_use = fs

    nyq_use = 0.5 * fs_use
    lo = max(lowcut / nyq_use, 1e-9)
    hi = min(highcut / nyq_use, 0.999999)
    if not 0 < lo < hi < 1:
        raise ValueError(
            f"非法频带: {lowcut}-{highcut} Hz (实际采样率 {fs_use:g} Hz)")

    b, a = butter(order, [lo, hi], btype="band")

    # ---- 长度不足时的降级：逐通道用单程滤波，并去掉直流 ----
    padlen = 3 * max(len(a), len(b))
    if work.shape[-1] <= padlen + 1:
        out = _safe_filter(data, fs, lowcut, highcut, order)
        return out

    out = filtfilt(b, a, work, axis=-1)

    # 抽取过的需要插值回原长度，方便和原始时间轴一起画
    if out.shape[-1] != data.shape[-1]:
        from scipy.signal import resample
        out = resample(out, data.shape[-1], axis=-1)
    return np.asarray(out)


def _safe_filter(data, fs, lowcut, highcut, order=4):
    """
    数据太短/频带太低时的降级滤波。

    用单程 IIR（lfilter）替代 filtfilt，避免 padlen 限制；
    这在数值上不如零相位稳定，但至少不会报错或溢出到天文数字。
    """
    from scipy.signal import lfilter
    nyq = 0.5 * fs
    lo = max(lowcut / nyq, 1e-9)
    hi = min(highcut / nyq, 0.999999)
    if not 0 < lo < hi < 1:
        return np.zeros_like(data)
    b, a = butter(max(1, order - 2), [lo, hi], btype="band")
    out = lfilter(b, a, data, axis=-1)
    # 简单去趋势，抑制单程滤波引入的低频偏移
    out = out - np.mean(out, axis=-1, keepdims=True)
    return out


def despike_channel(x, threshold=8.0, window=51, max_width=3):
    """
    检测并替换单个通道里**孤立的**大峰/大谷（峰谷都处理）。

    为什么不能只看"离开局部基线多远"
    --------------------------------
    实测发现：按"残差 > k·σ"筛出来的点有 390/423 是**相邻连续**的，
    也就是一整段波形/起始暂态，而不是孤立尖峰。直接替换会把真实波形
    一起抹掉（误伤数据）。而且同一个 σ 阈值在不同采集上差异极大
    （一个文件替换 7.6% 的点，另一个一个都不替换）。

    因此这里改成判断"**尖**"而不是"**高**"：
    一个正常的波峰，左右邻居是逐渐升上来的；而一个异常尖峰，
    它和左右邻居之间是突变的。用二阶差分（曲率）衡量：

        spike = |2·x[i] - x[i-1] - x[i+1]|

    再用局部 MAD 稳健估计噪声 σ，只有 spike > threshold·σ 的点才算
    异常。最后还会检查异常点是否连成片：连续长度超过 max_width 的
    一组点不算尖峰（是真实波形），予以保留。

    返回
    ----
    (clean, mask, info)
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    info = {"sigma": float("nan"), "n": 0, "max_ratio": 0.0, "skipped_runs": 0}
    if n < 8:
        return x.copy(), np.zeros(n, bool), info

    win = int(window)
    if win < 5:
        win = 5
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win < 5:
        return x.copy(), np.zeros(n, bool), info

    from scipy.ndimage import median_filter, uniform_filter1d

    # ---- 1. 局部中值基线，先扣掉 DC 与低频起伏 ----
    base = median_filter(x, size=win, mode="nearest")
    d = x - base

    # ---- 2. 二阶差分 = 尖锐程度 ----
    curv = np.zeros(n)
    curv[1:-1] = np.abs(2.0 * d[1:-1] - d[:-2] - d[2:])

    # ---- 3. 局部 MAD 估计噪声 σ（滑动中值绝对偏差）----
    #     用滑动窗口的 MAD，避免整段信号幅值差异导致阈值不合适。
    #     注意：局部 MAD 可能是 0（该窗口内曲率完全一致），会让比值爆炸到
    #     1e23 这种荒谬值，所以必须加"地板"。
    abs_curv = np.abs(curv - np.median(curv))
    local_mad = median_filter(abs_curv, size=win, mode="nearest")
    g_mad = float(np.median(abs_curv))
    g_sigma = 1.4826 * g_mad
    # 地板取 全局稳健 σ 的一小部分；若全局 σ 也是 0，则用信号自身的尺度
    if not np.isfinite(g_sigma) or g_sigma <= 0:
        g_sigma = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    if not np.isfinite(g_sigma) or g_sigma <= 0:
        g_sigma = float(np.abs(x).max()) * 1e-6
    if not np.isfinite(g_sigma) or g_sigma <= 0:
        return x.copy(), np.zeros(n, bool), info

    floor = max(g_sigma * 0.1, np.finfo(np.float64).tiny)
    sigma = np.maximum(1.4826 * local_mad, floor)

    ratio = curv / sigma
    info["sigma"] = float(np.median(sigma))
    info["max_ratio"] = float(np.max(ratio))
    mask = ratio > float(threshold)
    mask[0] = mask[-1] = False

    if not mask.any():
        info["n"] = 0
        return x.copy(), mask, info

    # ---- 4. 剔除"连成片"的：连续长度 > max_width 视为真实波形 ----
    idx = np.nonzero(mask)[0]
    keep = np.zeros(n, bool)
    start = 0
    runs = []
    for k in range(1, idx.size + 1):
        if k == idx.size or idx[k] != idx[k - 1] + 1:
            runs.append((idx[start], idx[k - 1]))
            start = k
    for a, b in runs:
        if (b - a + 1) <= int(max_width):
            keep[a:b + 1] = True
        else:
            info["skipped_runs"] += 1
    mask = keep
    info["n"] = int(mask.sum())

    if not mask.any():
        return x.copy(), mask, info

    # ---- 5. 用局部均值替换（只统计未被替换的点）----
    good = (~mask).astype(np.float64)
    num = uniform_filter1d(np.where(mask, 0.0, x), size=win, mode="nearest")
    den = uniform_filter1d(good, size=win, mode="nearest")
    repl = np.where(den > 0, num / np.maximum(den, 1e-30), x)

    clean = x.copy()
    clean[mask] = repl[mask]
    return clean, mask, info


def despike_matrix(disp, threshold=8.0, window=51, max_width=3):
    """
    对多通道位移矩阵逐通道做孤立尖峰检查与替换。

    返回 (clean, report)，report 为每个通道的统计信息列表。
    """
    disp = np.asarray(disp, dtype=np.float64)
    clean = disp.copy()
    report = []
    for ch in range(disp.shape[0]):
        c, m, info = despike_channel(disp[ch], threshold=threshold,
                                     window=window, max_width=max_width)
        clean[ch] = c
        report.append({
            "ch": ch,
            "n_replaced": int(m.sum()),
            "sigma": info["sigma"],
            "max_ratio": info["max_ratio"],
            "skipped_runs": info["skipped_runs"],
            "pkpk_before": float(disp[ch].max() - disp[ch].min()),
            "pkpk_after": float(c.max() - c.min()),
        })
    return clean, report


# =========================================================================
# 界面
# =========================================================================
class RadarGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("雷达数据处理 - 12 通道可视化")

        # 按屏幕实际大小开窗，避免在有缩放/小屏上放不下
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        w = max(900, min(1500, int(sw * 0.92)))
        h = max(600, min(940, int(sh * 0.90)))
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 3)
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.minsize(900, 600)

        self.result = None                 # compute_all_channels 的返回值
        self.worker = None
        self.msg_q = queue.Queue()
        self._stop_flag = threading.Event()
        self._busy = False

        # 竖排视图复用的图/卡片引用
        self.channel_canvases = []
        self.channel_figs = []
        self.channel_cards = []
        self.channel_toolbars = []
        self._cv = None
        self._tb = None
        self._list_sig = None

        self._build_vars()
        self._build_layout()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<F5>", lambda e: self.on_run())
        self._pump_queue()

    def on_close(self):
        """关窗时先让后台线程停下，避免进程挂住。"""
        if self._busy:
            if not messagebox.askyesno("正在处理",
                                       "还有任务在跑，确定要退出吗？"):
                return
            self._stop_flag.set()
            try:
                if self.worker is not None:
                    self.worker.join(timeout=3.0)
            except RuntimeError:
                pass
        self.root.destroy()

    # ------------------------------------------------------------------
    def _build_vars(self):
        def v(value="", cast=str):
            """构造一个绑定到主窗口的 Tk 变量。"""
            return tk.StringVar(master=self.root, value=str(value))

        self.var_bin = v()
        self.var_outdir = v(os.path.join(os.getcwd(), "output_gui"))
        self.var_prefix = v("channel")
        self.var_suffix = v("")
        self.var_ext = v(".csv")
        self.var_delim = v("comma")
        self.var_header = tk.BooleanVar(value=True)
        self.var_tag_prob = tk.BooleanVar(value=True)

        self.var_fftlen = v("1024")
        self.var_dc = tk.BooleanVar(value=True)
        self.var_frames = v("6250")
        self.var_chirps = v("24")
        self.var_rx = v("4")
        self.var_tx = v("3")
        self.var_samples = v("256")
        self.var_framerate = v("250")
        self.var_fc = v("77e9")
        self.var_slope = v("80e12")
        self.var_fsadc = v("1e7")
        self.var_lowcut = v("0.5")
        self.var_highcut = v("5.0")
        self.var_seed = v("")
        self.var_layout = v(LAYOUT_DEFAULT)

        # 大波动检查（可选项）
        self.var_spike_on = tk.BooleanVar(value=False)
        self.var_spike_thr = v("15")
        self.var_spike_win = v("51")
        self.var_spike_width = v("3")

        # 目标距离范围（按距离输入，bin 由程序换算）
        self.var_dist_min = v("")
        self.var_dist_max = v("")
        self.var_bin_mode = v(BIN_MODE_DEFAULT)

        # 批处理
        self.var_batch_in = v("")
        self.var_batch_out = v("")
        self.var_batch_rec = tk.BooleanVar(value=False)
        self.var_batch_ext = v(".bin")
        self.var_batch_workers = v(str(default_worker_count()))
        self.var_batch_resume = tk.BooleanVar(value=False)

        self.var_band = v(BAND_DEFAULT)
        self.var_view = v("list")
        self.var_status = v("就绪")
        self.var_progress = tk.DoubleVar(value=0.0)

        # 预设
        self.presets = {}
        self.var_preset = v("")

    # ------------------------------------------------------------------
    def _build_layout(self):
        # 左右可拖动分隔，用户可自行调整两侧宽度
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned, padding=4)
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        self.paned = paned

        # ---- 左侧：操作按钮固定在顶部，参数区可滚动 ----
        act = ttk.Frame(left, padding=(8, 8, 8, 2))
        act.pack(fill=tk.X)
        self.btn_run = ttk.Button(act, text="开始处理 (F5)", command=self.on_run)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.btn_stop = ttk.Button(act, text="中止", width=6,
                                   command=self.on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=3)
        self.btn_export = ttk.Button(act, text="导出CSV", width=8,
                                     command=self.on_export, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT)

        # 可滚动容器：左侧内容比窗口高时可以滚动查看，不会再把控件挤出屏幕
        body, inner = self._make_scrollable(left, width=330)
        body.pack(fill=tk.BOTH, expand=True, padx=(2, 0), pady=(0, 4))
        self._left_canvas = self._scroll_widgets[-1][1]

        self.right = right
        self.left = inner
        self._build_left(inner)
        self._build_right(right)

    def _make_scrollable(self, parent, width=330, wheel_on_enter=True):
        """
        返回 (外层容器, 内部可放控件的 Frame)。

        内部 Frame 高度超过容器时出现纵向滚动条；
        wheel_on_enter=True 时鼠标进入该区域可用滚轮滚动。
        """
        outer = ttk.Frame(parent)
        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0,
                           width=width)
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas, padding=(6, 2, 6, 6))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_inner_configure(_e=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(e):
            # 让内部 Frame 宽度跟随画布，避免横向滚动
            canvas.itemconfigure(win, width=e.width)

        inner.bind("<Configure>", on_inner_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def on_wheel(e):
            # 仅在内容高于可视区时滚动，避免无谓的抖动
            if inner.winfo_reqheight() > canvas.winfo_height():
                canvas.yview_scroll(int(-e.delta / 120), "units")

        def bind_wheel(_e):
            canvas.bind_all("<MouseWheel>", on_wheel)

        def unbind_wheel(_e):
            canvas.unbind_all("<MouseWheel>")

        if wheel_on_enter:
            # 注意：要连子控件一起绑定，否则鼠标在图/按钮上时收不到
            for w in (canvas, inner):
                w.bind("<Enter>", bind_wheel)
                w.bind("<Leave>", unbind_wheel)
            # 内部后加入的控件在 <Enter> 冒泡时仍会命中 canvas 的绑定

        outer.bind("<Configure>", lambda e: on_inner_configure())
        self._scroll_widgets = getattr(self, "_scroll_widgets", [])
        self._scroll_widgets.append((outer, canvas, inner))
        return outer, inner

    # ------------------------- 左侧控制面板 -------------------------
    def _build_left(self, parent):
        # ---- 参数预设 ----
        f0 = ttk.LabelFrame(parent, text="0. 参数预设", padding=6)
        f0.pack(fill=tk.X, pady=4)
        self.cmb_preset = ttk.Combobox(f0, textvariable=self.var_preset,
                                       state="readonly", width=34)
        self.cmb_preset.pack(fill=tk.X)
        self.cmb_preset.bind("<<ComboboxSelected>>", self._on_preset_selected)

        r = ttk.Frame(f0)
        r.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(r, text="应用", width=7,
                   command=self.on_preset_apply).pack(side=tk.LEFT)
        ttk.Button(r, text="另存为预设", width=11,
                   command=self.on_preset_save).pack(side=tk.LEFT, padx=3)
        ttk.Button(r, text="删除", width=7,
                   command=self.on_preset_delete).pack(side=tk.LEFT)

        self.lbl_preset_note = ttk.Label(f0, text="", foreground="#555",
                                         wraplength=300, justify=tk.LEFT)
        self.lbl_preset_note.pack(anchor=tk.W, pady=(4, 0))

        # 说明标签建好之后才能刷新列表（刷新过程会更新说明）
        self._refresh_preset_list()

        # 选中项一旦变化就更新说明
        self.var_preset.trace_add("write", lambda *a: self._update_preset_note())

        # ---- 文件与输出 ----
        f1 = ttk.LabelFrame(parent, text="1. 输入 / 输出", padding=6)
        f1.pack(fill=tk.X, pady=4)
        self._file_row(f1)

        row = ttk.Frame(f1)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="输出目录", width=9).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_outdir, width=26).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="...", width=3,
                   command=self._pick_outdir).pack(side=tk.LEFT, padx=2)

        row = ttk.Frame(f1)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="命名方式", width=9).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_prefix, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="<ch>").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_suffix, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="后缀").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_ext, width=6).pack(side=tk.LEFT)

        row = ttk.Frame(f1)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="分隔符", width=9).pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="逗号", variable=self.var_delim,
                        value="comma").pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="Tab", variable=self.var_delim,
                        value="tab").pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="空格", variable=self.var_delim,
                        value="space").pack(side=tk.LEFT)
        ttk.Checkbutton(f1, text="包含表头", variable=self.var_header).pack(anchor=tk.W)
        ttk.Checkbutton(f1, text="文件名附带概率(prob_xx)",
                        variable=self.var_tag_prob).pack(anchor=tk.W)

        ttk.Label(f1, text="预览: channel_3_prob_88.csv",
                  foreground="#555").pack(anchor=tk.W, pady=(2, 0))

        # ---- 处理参数 ----
        f2 = ttk.LabelFrame(parent, text="2. 处理参数", padding=6)
        f2.pack(fill=tk.X, pady=4)

        def prow(label, var, width=9, hint=""):
            r = ttk.Frame(f2)
            r.pack(fill=tk.X, pady=0)
            ttk.Label(r, text=label, width=10).pack(side=tk.LEFT)
            ttk.Entry(r, textvariable=var, width=width).pack(side=tk.LEFT)
            if hint:
                ttk.Label(r, text=hint, foreground="#777").pack(side=tk.LEFT, padx=4)

        prow("FFT 点数", self.var_fftlen)
        prow("帧数", self.var_frames)
        prow("每帧 chirp", self.var_chirps)
        prow("RX 数", self.var_rx)
        prow("TX 数", self.var_tx)
        prow("adc 采样", self.var_samples)
        prow("帧率 Hz", self.var_framerate)
        prow("fc", self.var_fc)
        prow("slope", self.var_slope)
        prow("fs_adc", self.var_fsadc)
        prow("带通低 Hz", self.var_lowcut)
        prow("带通高 Hz", self.var_highcut)
        prow("RANSAC 种子", self.var_seed, hint="留空=每次不同")

        # 原始数据的 I/Q 排布
        r = ttk.Frame(f2)
        r.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(r, text="I/Q 排布", width=10).pack(side=tk.LEFT)
        self.cmb_layout = ttk.Combobox(
            r, textvariable=self.var_layout, state="readonly", width=11,
            values=list(LAYOUT_CHOICES))
        self.cmb_layout.pack(side=tk.LEFT)
        ttk.Label(r, text="(IIQQ = 本雷达实测)",
                  foreground="#777").pack(side=tk.LEFT, padx=3)

        ttk.Checkbutton(f2, text="执行 DC 消除 (RANSAC 圆拟合)",
                        variable=self.var_dc).pack(anchor=tk.W, pady=(2, 0))

        # 快速预览：只取前若干个采样点，用于快速看效果
        r = ttk.Frame(f2)
        r.pack(fill=tk.X, pady=(2, 0))
        self.var_preview = tk.BooleanVar(value=False)
        ttk.Checkbutton(r, text="快速预览 前", variable=self.var_preview).pack(
            side=tk.LEFT)
        self.var_preview_n = tk.StringVar(master=self.root, value="1500")
        ttk.Entry(r, textvariable=self.var_preview_n, width=7).pack(side=tk.LEFT, padx=3)
        ttk.Label(r, text="帧 (秒出图)", foreground="#777").pack(side=tk.LEFT)

        # ---- 目标距离范围 ----
        f7 = ttk.LabelFrame(parent, text="目标距离范围与选峰方式", padding=6)
        f7.pack(fill=tk.X, pady=4)

        r7 = ttk.Frame(f7)
        r7.pack(fill=tk.X)
        ttk.Label(r7, text="距离").pack(side=tk.LEFT)
        ttk.Entry(r7, textvariable=self.var_dist_min, width=7).pack(
            side=tk.LEFT, padx=2)
        ttk.Label(r7, text="~").pack(side=tk.LEFT)
        ttk.Entry(r7, textvariable=self.var_dist_max, width=7).pack(
            side=tk.LEFT, padx=2)
        ttk.Label(r7, text="cm   (留空=不限)").pack(side=tk.LEFT)

        r8 = ttk.Frame(f7)
        r8.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(r8, text="选峰方式").pack(side=tk.LEFT)
        self.cmb_binmode = ttk.Combobox(r8, textvariable=self.var_bin_mode,
                                        state="readonly", width=28,
                                        values=BIN_MODE_CHOICES)
        self.cmb_binmode.pack(side=tk.LEFT, padx=3)

        ttk.Label(f7,
                  text="「各自选峰」在多反射体场景下会让不同通道选到不同距离，\n"
                       "出现「前几个通道 0.46m、后面 1.10m」这种分裂。\n"
                       "填上目标的距离范围，选「估计范围」即可锁到同一距离。",
                  foreground="#777", justify=tk.LEFT).pack(anchor=tk.W, pady=(3, 0))
        # 距离 -> bin 的实时换算显示
        self.lbl_binpreview = ttk.Label(f7, text="", foreground="#c60",
                                        wraplength=310, justify=tk.LEFT)
        self.lbl_binpreview.pack(anchor=tk.W, pady=(3, 0))
        for var in (self.var_dist_min, self.var_dist_max, self.var_fftlen,
                    self.var_fc, self.var_slope, self.var_fsadc,
                    self.var_framerate, self.var_samples):
            var.trace_add("write", lambda *a: self._update_bin_preview())
        self._update_bin_preview()

        self.lbl_bincand = ttk.Label(f7, text="", foreground="#06c",
                                     wraplength=310, justify=tk.LEFT)
        self.lbl_bincand.pack(anchor=tk.W, pady=(3, 0))

        # ---- 大波动检查（可选）----
        f6 = ttk.LabelFrame(parent, text="大波动检查（可选）", padding=6)
        f6.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(
            f6, text="启用：把孤立的大峰/大谷用局部均值替换",
            variable=self.var_spike_on).pack(anchor=tk.W)
        rr = ttk.Frame(f6)
        rr.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(rr, text="阈值").pack(side=tk.LEFT)
        ttk.Entry(rr, textvariable=self.var_spike_thr, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(rr, text="σ   窗口").pack(side=tk.LEFT)
        ttk.Entry(rr, textvariable=self.var_spike_win, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(rr, text="点").pack(side=tk.LEFT)
        rr2 = ttk.Frame(f6)
        rr2.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(rr2, text="最大连续宽度").pack(side=tk.LEFT)
        ttk.Entry(rr2, textvariable=self.var_spike_width, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(rr2, text="点（超过则视为真实波形，不动）",
                  foreground="#777").pack(side=tk.LEFT)
        ttk.Label(f6,
                  text="只替换\"尖\"的单点异常，连续波形不会被误伤。\n"
                       "阈值越小替换越多；建议 10~20σ。",
                  foreground="#777", justify=tk.LEFT).pack(anchor=tk.W, pady=(3, 0))
        self.lbl_spike = ttk.Label(f6, text="", foreground="#0a5",
                                   wraplength=310, justify=tk.LEFT)
        self.lbl_spike.pack(anchor=tk.W, pady=(3, 0))

        # 距离标定实时显示：参数改动时自动重算，方便核对距离对不对
        self.lbl_calib = ttk.Label(f2, text="", foreground="#0a5",
                                   justify=tk.LEFT, wraplength=310)
        self.lbl_calib.pack(anchor=tk.W, pady=(4, 0))
        for var in (self.var_fftlen, self.var_fc, self.var_slope,
                    self.var_fsadc, self.var_samples):
            var.trace_add("write", lambda *a: self._update_calib())
        self._update_calib()

        # ---- 收发天线处理（预留区）----
        # 只保留一个空 Frame，后续要加 TX/RX 天线相关的功能时，
        # 把控件加进 self.antenna_frame 即可，界面其余部分不用动。
        f3 = ttk.LabelFrame(parent, text="3. 收发天线处理（预留区）", padding=6)
        f3.pack(fill=tk.X, pady=4)
        self.antenna_frame = f3

        # ---- 批处理 ----
        fb = ttk.LabelFrame(parent, text="批处理（整个文件夹）", padding=6)
        fb.pack(fill=tk.X, pady=4)

        rb1 = ttk.Frame(fb)
        rb1.pack(fill=tk.X)
        ttk.Label(rb1, text="输入文件夹", width=10).pack(side=tk.LEFT)
        ttk.Entry(rb1, textvariable=self.var_batch_in, width=20).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(rb1, text="...", width=3,
                   command=self._pick_batch_in).pack(side=tk.LEFT, padx=2)

        rb2 = ttk.Frame(fb)
        rb2.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(rb2, text="输出根目录", width=10).pack(side=tk.LEFT)
        ttk.Entry(rb2, textvariable=self.var_batch_out, width=20).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(rb2, text="...", width=3,
                   command=self._pick_batch_out).pack(side=tk.LEFT, padx=2)

        rb3 = ttk.Frame(fb)
        rb3.pack(fill=tk.X, pady=(3, 0))
        ttk.Checkbutton(rb3, text="包含子文件夹",
                        variable=self.var_batch_rec).pack(side=tk.LEFT)
        ttk.Label(rb3, text="  扩展名").pack(side=tk.LEFT)
        ttk.Entry(rb3, textvariable=self.var_batch_ext, width=6).pack(side=tk.LEFT)

        rb4 = ttk.Frame(fb)
        rb4.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(rb4, text="并行进程数").pack(side=tk.LEFT)
        ttk.Spinbox(rb4, from_=1, to=16, width=5,
                    textvariable=self.var_batch_workers).pack(side=tk.LEFT, padx=3)
        ttk.Label(rb4, text=f"（自动={default_worker_count()}）",
                  foreground="#777").pack(side=tk.LEFT)

        rb5 = ttk.Frame(fb)
        rb5.pack(fill=tk.X, pady=(3, 0))
        ttk.Checkbutton(rb5, text="断点续处理",
                        variable=self.var_batch_resume).pack(side=tk.LEFT)
        ttk.Button(rb5, text="扫描", width=6,
                   command=self.on_batch_scan).pack(side=tk.LEFT, padx=4)
        ttk.Label(rb5, text="勾选后跳过输出目录里已完成的文件",
                  foreground="#777").pack(side=tk.LEFT)

        ttk.Label(fb,
                  text="多进程并行：每个文件一个进程。CPU 核数 - 1，上限 4，\n"
                       "填 1 则串行。进程越多越快，但内存占用也成倍上升。",
                  foreground="#777", justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))

        self.btn_batch = ttk.Button(fb, text="开始批处理", command=self.on_batch)
        self.btn_batch.pack(fill=tk.X, pady=(4, 0))
        self.lbl_batch = ttk.Label(fb, text="", foreground="#0a5",
                                   wraplength=310, justify=tk.LEFT)
        self.lbl_batch.pack(anchor=tk.W, pady=(3, 0))
        ttk.Label(fb,
                  text="每个 bin 处理后在输出目录下建同名子目录，\n"
                       "写 12 个通道的 CSV（命名用上面第 1 节的规则）。",
                  foreground="#777", justify=tk.LEFT).pack(anchor=tk.W, pady=(3, 0))

        # ---- 日志 ----
        # 注意：「开始处理 / 中止 / 导出CSV」三个按钮已移到左侧面板顶部固定显示，
        # 这里不再重复放置，避免被挤出屏幕。
        f5 = ttk.LabelFrame(parent, text="日志", padding=4)
        f5.pack(fill=tk.X, pady=4)
        self.log = tk.Text(f5, width=40, height=8, wrap=tk.WORD,
                           font=("Consolas", 8))
        sb = ttk.Scrollbar(f5, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _file_row(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="数据文件", width=9).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_bin, width=26).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="...", width=3,
                   command=self._pick_binfile).pack(side=tk.LEFT, padx=2)

        # 快速选择 data/ 目录下已有的 bin
        self.cmb_bin = ttk.Combobox(parent, state="readonly", width=40,
                                    values=self._list_data_bins())
        self.cmb_bin.pack(fill=tk.X, pady=(2, 0))
        self.cmb_bin.set("—— 从 data/ 目录快速选择 ——")
        self.cmb_bin.bind("<<ComboboxSelected>>", self._on_pick_from_data)

    def _list_data_bins(self):
        """列出可选的数据文件：优先 exe/仓库同级的 data/，其次程序内置资源。"""
        for d in (default_data_dir(), app_dir(), os.getcwd()):
            if d and os.path.isdir(d):
                bins = [os.path.join(d, f)
                        for f in sorted(os.listdir(d))
                        if f.lower().endswith(".bin")]
                if bins:
                    return bins
        return []

    def _on_pick_from_data(self, _evt):
        p = self.cmb_bin.get()
        if os.path.isfile(p):
            self.var_bin.set(p)
            self._log(f"已选择: {p}")

    # ------------------------- 右侧显示区 -------------------------
    def _build_right(self, parent):
        # 第一行：视图切换
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(bar, text="视图:").pack(side=tk.LEFT)
        for txt, val in [("1-12通道竖排", "list"), ("12通道网格", "grid"),
                         ("单通道放大", "single"), ("IQ 星座图", "iq"),
                         ("距离谱", "range")]:
            ttk.Radiobutton(bar, text=txt, variable=self.var_view, value=val,
                            command=self.render).pack(side=tk.LEFT, padx=3)
        ttk.Label(bar, text="单通道:").pack(side=tk.LEFT, padx=(8, 0))
        self.var_single = tk.IntVar(value=0)
        self.spin_single = ttk.Spinbox(bar, from_=0, to=11, width=4,
                                       textvariable=self.var_single,
                                       command=self.render)
        self.spin_single.pack(side=tk.LEFT)

        # 第二行：滤波频带（脉搏波/呼吸等，可自定义）
        bar2 = ttk.LabelFrame(parent, text="滤波范围（切换不需重跑）", padding=4)
        bar2.pack(fill=tk.X, pady=(2, 2))

        row = ttk.Frame(bar2)
        row.pack(fill=tk.X)
        ttk.Label(row, text="频带:").pack(side=tk.LEFT)
        self.cmb_band = ttk.Combobox(row, textvariable=self.var_band,
                                     state="readonly", width=26,
                                     values=BAND_CHOICES)
        self.cmb_band.pack(side=tk.LEFT, padx=4)
        self.cmb_band.bind("<<ComboboxSelected>>", self._on_band_selected)

        row2 = ttk.Frame(bar2)
        row2.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(row2, text="低").pack(side=tk.LEFT)
        self.ent_band_lo = ttk.Entry(row2, textvariable=self.var_lowcut, width=7)
        self.ent_band_lo.pack(side=tk.LEFT, padx=2)
        ttk.Label(row2, text="Hz   高").pack(side=tk.LEFT)
        self.ent_band_hi = ttk.Entry(row2, textvariable=self.var_highcut, width=7)
        self.ent_band_hi.pack(side=tk.LEFT, padx=2)
        ttk.Label(row2, text="Hz").pack(side=tk.LEFT)
        ttk.Button(row2, text="应用滤波", width=9,
                   command=self.on_apply_band).pack(side=tk.LEFT, padx=6)
        # 回车即应用
        for e in (self.ent_band_lo, self.ent_band_hi):
            e.bind("<Return>", lambda _ev: self.on_apply_band())

        self.lbl_band_info = ttk.Label(bar2, text="", foreground="#0a5")
        self.lbl_band_info.pack(anchor=tk.W, pady=(3, 0))

        # 进度条
        pbar = ttk.Frame(parent)
        pbar.pack(fill=tk.X)
        ttk.Progressbar(pbar, variable=self.var_progress, maximum=100.0).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(pbar, textvariable=self.var_status, width=32,
                  anchor=tk.W).pack(side=tk.LEFT, padx=6)

        # 主画布（仅用于"请先处理"的占位提示）
        self.fig = Figure(figsize=(11, 6.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 竖向滚动容器：竖排通道图、IQ 图、距离谱都放这里
        self.body, self.body_inner = self._make_scrollable(parent, width=800)
        self._body_canvas = self._scroll_widgets[-1][1]

        # 汇总表（固定在底部，始终可见）
        cols = ("ch", "tx", "rx", "bin", "dist", "ransac", "prob", "pkpk")
        heads = ("通道", "TX", "RX", "目标bin", "距离(m)", "RANSAC",
                 "质量分", "峰峰值(um)")
        widths = (48, 40, 40, 70, 80, 70, 70, 100)
        box = ttk.Frame(parent)
        box.pack(fill=tk.X, pady=4)
        self.tree = ttk.Treeview(box, columns=cols, show="headings", height=5)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor=tk.CENTER)
        tsb = ttk.Scrollbar(box, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        tsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        self._draw_placeholder()

    # ------------------------------------------------------------------
    def _draw_placeholder(self):
        try:
            self.body.pack_forget()
        except (tk.TclError, AttributeError):
            pass
        self._ensure_main_canvas()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "请选择 .bin 文件并点击「开始处理」\n\n"
                "处理完成后这里会显示 12 个通道的结果",
                ha="center", va="center", fontsize=14, color="#888")
        ax.set_axis_off()
        self.canvas.draw()

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{ts}] {msg}\n")
        self.log.see(tk.END)

    # ------------------------------------------------------------------
    # 参数预设
    # ------------------------------------------------------------------
    _PRESET_KEYS = (
        "num_frames", "num_chirps", "num_rx", "num_tx", "num_samples",
        "fft_len", "frame_rate", "fc", "slope", "fs_adc", "layout",
        "do_dc_eliminate", "lowcut", "highcut", "seed",
    )

    def _refresh_preset_list(self):
        """合并内置预设与用户预设，刷新下拉列表。"""
        self.presets = dict(BUILTIN_PRESETS)
        self.presets.update(load_user_presets())
        names = list(self.presets.keys())
        self.cmb_preset.configure(values=names)
        if names and not self.var_preset.get():
            self.var_preset.set(names[0])
        self._update_preset_note()

    # ------------------------------------------------------------------
    # 距离标定
    # ------------------------------------------------------------------
    def _update_calib(self):
        """
        根据当前参数实时算出距离换算率，显示在界面上。

            R(bin) = bin x fs_adc x c / (2 x slope x fft_len)
            分辨率 = c / (2 x slope x adc_samples / fs_adc) = c x fs_adc / (2 x slope x N)
        """
        try:
            fs = float(self.var_fsadc.get())
            slope = float(self.var_slope.get())
            fft_len = int(float(self.var_fftlen.get()))
            n_adc = int(float(self.var_samples.get()))
            if fs <= 0 or slope <= 0 or fft_len <= 0 or n_adc <= 0:
                raise ValueError
        except (TypeError, ValueError):
            txt = "标定: 请填写有效的 slope / fs_adc / fft_len / adc采样"
            try:
                self.lbl_calib.configure(text=txt, foreground="#a00")
            except tk.TclError:
                pass
            return

        per_bin = fs * C_LIGHT / (2 * slope * fft_len)
        bandwidth = slope * n_adc / fs
        res = C_LIGHT / (2 * bandwidth)
        max_r = (fs / 2) * C_LIGHT / (2 * slope)
        # 中频链 0.4*fs 限制
        r_if = (0.4 * fs) * C_LIGHT / (2 * slope)
        bin_if = 0.4 * fft_len

        txt = (f"距离标定: 每 bin = {per_bin*100:.3f} cm\n"
               f"分辨率 ΔR = c/(2B) = {res*100:.2f} cm   (B={bandwidth/1e9:.3f} GHz)\n"
               f"最大不模糊 = {max_r:.2f} m,  受中频链限制 ≈ {r_if:.2f} m\n"
               f"有效 bin 上限 ≈ {bin_if:.0f} (超出不可信)")
        try:
            self.lbl_calib.configure(text=txt, foreground="#0a5")
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    def _update_preset_note(self):
        p = self.presets.get(self.var_preset.get())
        note = ""
        if p:
            note = p.get("note", "")
            layout = p.get("layout", "")
            if layout:
                note = f"[{layout}] {note}" if note else f"[{layout}]"
        try:
            lbl = getattr(self, "lbl_preset_note", None)
            if lbl is not None:
                lbl.configure(text=note)
        except tk.TclError:
            pass

    def _apply_preset(self, preset, log=True):
        """把预设字典写入界面变量。未知键忽略。"""
        mapping = {
            "num_frames": self.var_frames,
            "num_chirps": self.var_chirps,
            "num_rx": self.var_rx,
            "num_tx": self.var_tx,
            "num_samples": self.var_samples,
            "fft_len": self.var_fftlen,
            "frame_rate": self.var_framerate,
            "fc": self.var_fc,
            "slope": self.var_slope,
            "fs_adc": self.var_fsadc,
            "layout": self.var_layout,
            "lowcut": self.var_lowcut,
            "highcut": self.var_highcut,
            "seed": self.var_seed,
        }
        for key, var in mapping.items():
            if key in preset and preset[key] is not None:
                var.set(str(preset[key]))
        if "do_dc_eliminate" in preset:
            self.var_dc.set(bool(preset["do_dc_eliminate"]))
        # 应用预设后同步一下频带名与滤波范围显示
        self._sync_band_name_from_cutoffs()
        if log:
            self._log(f"已应用预设「{self.var_preset.get()}」")

    def _current_preset(self):
        """从界面收集当前参数，用于保存成预设。"""
        return {
            "num_frames": self.var_frames.get(),
            "num_chirps": self.var_chirps.get(),
            "num_rx": self.var_rx.get(),
            "num_tx": self.var_tx.get(),
            "num_samples": self.var_samples.get(),
            "fft_len": self.var_fftlen.get(),
            "frame_rate": self.var_framerate.get(),
            "fc": self.var_fc.get(),
            "slope": self.var_slope.get(),
            "fs_adc": self.var_fsadc.get(),
            "layout": self.var_layout.get(),
            "do_dc_eliminate": bool(self.var_dc.get()),
            "lowcut": self.var_lowcut.get(),
            "highcut": self.var_highcut.get(),
            "seed": self.var_seed.get(),
        }

    def _on_preset_selected(self, _evt=None):
        self._update_preset_note()
        # 选中即应用，减少一次点击
        self.on_preset_apply()

    def on_preset_apply(self):
        name = self.var_preset.get()
        preset = self.presets.get(name)
        if not preset:
            return
        self._apply_preset(preset)
        self._update_preset_note()

    def on_preset_save(self):
        """把当前界面参数存成一个新的用户预设。"""
        current = self._current_preset()
        dlg = tk.Toplevel(self.root)
        dlg.title("另存为预设")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        ttk.Label(dlg, text="预设名称:", padding=8).pack(anchor=tk.W)
        name_var = tk.StringVar(master=dlg, value="我的预设")
        ent = ttk.Entry(dlg, textvariable=name_var, width=34)
        ent.pack(padx=8, fill=tk.X)
        ent.focus_set()

        ttk.Label(dlg, text="说明 (可选):", padding=(8, 4, 8, 0)).pack(anchor=tk.W)
        note_var = tk.StringVar(master=dlg, value="")
        ttk.Entry(dlg, textvariable=note_var, width=34).pack(padx=8, fill=tk.X)

        def do_save():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("名称为空", "请填写预设名称。", parent=dlg)
                return
            if name in BUILTIN_PRESETS:
                if not messagebox.askyesno(
                        "覆盖内置预设",
                        f"「{name}」是内置预设，要覆盖它吗？\n"
                        f"（同名用户预设会优先于内置预设生效）", parent=dlg):
                    return
            user = load_user_presets()
            entry = dict(current)
            entry["note"] = note_var.get().strip()
            user[name] = entry
            try:
                save_user_presets(user)
            except OSError as e:
                messagebox.showerror("保存失败", str(e), parent=dlg)
                return
            dlg.destroy()
            self.var_preset.set(name)
            self._refresh_preset_list()
            self.var_preset.set(name)
            self._log(f"已保存预设「{name}」到 {PRESET_FILE}")

        btns = ttk.Frame(dlg)
        btns.pack(pady=8)
        ttk.Button(btns, text="保存", command=do_save).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="取消", command=dlg.destroy).pack(side=tk.LEFT)

    def on_preset_delete(self):
        name = self.var_preset.get()
        if name in BUILTIN_PRESETS and name not in load_user_presets():
            messagebox.showinfo("无法删除",
                                f"「{name}」是内置预设，不能删除。\n"
                                f"可以「另存为预设」保存自己的版本。")
            return
        if not messagebox.askyesno("确认删除", f"删除用户预设「{name}」？"):
            return
        user = load_user_presets()
        user.pop(name, None)
        try:
            save_user_presets(user)
        except OSError as e:
            messagebox.showerror("删除失败", str(e))
            return
        self._log(f"已删除预设「{name}」")
        self.var_preset.set("")
        self._refresh_preset_list()

    # ------------------------------------------------------------------
    # 文件选择
    # ------------------------------------------------------------------
    def _pick_binfile(self):
        path = filedialog.askopenfilename(
            title="选择雷达原始数据",
            filetypes=[("雷达原始数据", "*.bin"), ("所有文件", "*.*")],
            initialdir=self._initial_dir(),
        )
        if path:
            self.var_bin.set(path)
            self._log(f"已选择: {path}")
            if not self.var_outdir.get().strip():
                self.var_outdir.set(os.path.join(os.path.dirname(path), "output"))

    def _initial_dir(self):
        for d in (default_data_dir(), app_dir()):
            if d and os.path.isdir(d):
                return d
        p = self.var_bin.get().strip()
        if p and os.path.dirname(p):
            return os.path.dirname(p)
        return os.getcwd()

    def _pick_outdir(self):
        path = filedialog.askdirectory(title="选择输出目录",
                                       initialdir=self.var_outdir.get() or os.getcwd())
        if path:
            self.var_outdir.set(path)

    def _pick_batch_in(self):
        path = filedialog.askdirectory(
            title="选择要批量处理的文件夹",
            initialdir=self.var_batch_in.get() or self._initial_dir())
        if path:
            self.var_batch_in.set(path)
            if not self.var_batch_out.get().strip():
                self.var_batch_out.set(os.path.join(path, "output"))

    def _pick_batch_out(self):
        path = filedialog.askdirectory(
            title="选择输出根目录",
            initialdir=self.var_batch_out.get() or os.getcwd())
        if path:
            self.var_batch_out.set(path)

    def _export_opts(self):
        """当前界面上的 CSV 命名/格式选项（批处理与导出共用）。"""
        return dict(
            prefix=self.var_prefix.get(),
            suffix=self.var_suffix.get(),
            ext=self.var_ext.get(),
            delim={"comma": ",", "tab": "\t", "space": " "}[
                self.var_delim.get()],
            header=bool(self.var_header.get()),
            tag_prob=bool(self.var_tag_prob.get()))

    def on_batch_scan(self):
        """扫描输入文件夹，报告哪些已处理完成、哪些待处理（不真正处理）。"""
        folder = self.var_batch_in.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("输入文件夹无效",
                                 f"请选择一个有效的输入文件夹。\n当前: {folder!r}")
            return
        outdir = self.var_batch_out.get().strip()
        if not outdir:
            outdir = os.path.join(folder, "output")
            self.var_batch_out.set(outdir)

        ext = self.var_batch_ext.get().strip() or ".bin"
        glob_pat = "*" + (ext if ext.startswith(".") else "." + ext)
        rec = bool(self.var_batch_rec.get())
        opts = self._export_opts()

        import glob as _glob
        pattern = os.path.join(os.path.abspath(folder), "**", glob_pat) \
            if rec else os.path.join(os.path.abspath(folder), glob_pat)
        files = sorted(f for f in _glob.glob(pattern, recursive=rec)
                       if os.path.isfile(f))
        if not files:
            self._log(f"[扫描] {folder} 下没有匹配 {glob_pat} 的文件")
            try:
                self.lbl_batch.configure(text="扫描: 没有找到文件")
            except tk.TclError:
                pass
            return

        done_map, todo, detail = scan_done(files, os.path.abspath(outdir), opts)
        self._log("=" * 40)
        self._log(f"[扫描] 输入: {folder}")
        self._log(f"[扫描] 输出: {outdir}")
        self._log(f"[扫描] 共 {len(files)} 个文件，"
                  f"已完成 {len(done_map)} 个，待处理 {len(todo)} 个")
        for p, ok, why in detail:
            tag = "[完成]" if ok else "[待处理]"
            self._log(f"  {tag} {os.path.basename(p)}: {why}")

        try:
            self.lbl_batch.configure(
                text=f"扫描: 共 {len(files)} / 已完成 {len(done_map)} / "
                     f"待处理 {len(todo)}")
        except tk.TclError:
            pass

        if todo:
            messagebox.showinfo(
                "扫描结果",
                f"共 {len(files)} 个文件\n\n"
                f"已完成: {len(done_map)} 个（可跳过）\n"
                f"待处理: {len(todo)} 个\n\n"
                f"勾选「断点续处理」后点「开始批处理」，"
                f"就只会处理这 {len(todo)} 个。")
        else:
            messagebox.showinfo("扫描结果",
                                f"共 {len(files)} 个文件，已全部处理完成，"
                                f"无需再处理。")

    def on_batch(self):
        """按当前参数批量处理一个文件夹里的所有 bin。"""
        if self._busy:
            messagebox.showinfo("正忙", "已有任务在运行，请先等待或中止。")
            return
        folder = self.var_batch_in.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("输入文件夹无效",
                                 f"请选择一个有效的输入文件夹。\n当前: {folder!r}")
            return
        outdir = self.var_batch_out.get().strip()
        if not outdir:
            messagebox.showerror("缺少输出目录", "请选择输出根目录。")
            return
        try:
            params = self._read_params()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        rec = bool(self.var_batch_rec.get())
        ext = self.var_batch_ext.get().strip() or ".bin"
        glob_pat = "*" + (ext if ext.startswith(".") else "." + ext)
        try:
            workers = max(1, int(float(self.var_batch_workers.get())))
        except (TypeError, ValueError):
            workers = default_worker_count()
        opts = self._export_opts()
        resume = bool(self.var_batch_resume.get())

        self._busy = True
        self._stop_flag.clear()
        self.btn_run.configure(state=tk.DISABLED)
        self.btn_batch.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.var_progress.set(0.0)
        self.var_status.set("批处理中 ...")
        self._log("=" * 40)
        self._log(f"批处理开始")
        self._log(f"  输入: {folder}")
        self._log(f"  输出: {outdir}")
        self._log(f"  匹配: {glob_pat}  包含子目录: {rec}")
        self._log(f"  并行进程数: {workers}")
        self._log(f"  断点续处理: {'开' if resume else '关'}")
        try:
            self.lbl_batch.configure(text="批处理中 ...")
        except tk.TclError:
            pass

        def report(msg, frac=None):
            self.msg_q.put(("progress", (msg, frac)))

        def work():
            try:
                r = process_folder(
                    folder, outdir, file_glob=glob_pat, recursive=rec,
                    progress=report, should_stop=self._stop_flag.is_set,
                    export_opts=opts, workers=workers, resume=resume,
                    **params)
                self.msg_q.put(("batch_done", r))
            except KeyboardInterrupt:
                self.msg_q.put(("stopped", None))
            except Exception as e:                       # noqa: BLE001
                self.msg_q.put(("error", (e, traceback.format_exc())))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------
    def _dist_per_bin(self):
        """每个 bin 对应多少米。参数不合法时返回 None。"""
        try:
            fs = float(self.var_fsadc.get())
            slope = float(self.var_slope.get())
            fft_len = int(float(self.var_fftlen.get()))
            if fs <= 0 or slope <= 0 or fft_len <= 0:
                return None
            return fs * C_LIGHT / (2 * slope * fft_len)
        except (TypeError, ValueError):
            return None

    def _update_bin_preview(self):
        """把输入的距离范围实时换算成 bin 显示出来。"""
        per_bin = self._dist_per_bin()
        try:
            lbl = getattr(self, "lbl_binpreview", None)
        except tk.TclError:
            return
        if lbl is None:
            return
        if per_bin is None:
            try:
                lbl.configure(text="换算: 请先填好 slope / fs_adc / FFT 点数")
            except tk.TclError:
                pass
            return

        def parse(var):
            s = var.get().strip()
            if not s:
                return None
            try:
                return float(s) / 100.0        # 输入单位 cm -> m
            except ValueError:
                return None

        dmin = parse(self.var_dist_min)
        dmax = parse(self.var_dist_max)
        txt = [f"换算: 每 bin = {per_bin*100:.2f} cm"]
        if dmin is not None:
            txt.append(f"下限 {dmin*100:.1f}cm -> bin {int(round(dmin/per_bin))}")
        if dmax is not None:
            txt.append(f"上限 {dmax*100:.1f}cm -> bin {int(round(dmax/per_bin))}")
        if dmin is None and dmax is None:
            txt.append("（未限制，全部 bin 都参与选峰）")
        elif dmin is not None and dmax is not None:
            n = int(round(dmax / per_bin)) - int(round(dmin / per_bin)) + 1
            txt.append(f"共 {max(n,1)} 个 bin")
        try:
            lbl.configure(text="\n".join(txt))
        except tk.TclError:
            pass

    def _read_params(self):
        """从界面收集参数，返回 (params_dict, 错误信息)。"""
        def num(var, name, cast=float):
            s = var.get().strip()
            if s == "":
                raise ValueError(f"{name} 不能为空")
            try:
                return cast(s)
            except ValueError:
                raise ValueError(f"{name} 不是合法数字: {s!r}")

        p = dict(
            num_frames=num(self.var_frames, "帧数", int),
            num_chirps=num(self.var_chirps, "每帧 chirp", int),
            num_rx=num(self.var_rx, "RX 数", int),
            num_samples=num(self.var_samples, "adc 采样", int),
            num_tx=num(self.var_tx, "TX 数", int),
            fft_len=num(self.var_fftlen, "FFT 点数", int),
            do_dc_eliminate=bool(self.var_dc.get()),
            frame_rate=num(self.var_framerate, "帧率"),
            fc=num(self.var_fc, "fc"),
            slope=num(self.var_slope, "slope"),
            fs_adc=num(self.var_fsadc, "fs_adc"),
            display_lowcut=num(self.var_lowcut, "带通低"),
            display_highcut=num(self.var_highcut, "带通高"),
            layout=self.var_layout.get(),
        )
        if p["num_rx"] not in (2, 4):
            raise ValueError("RX 数只支持 2 或 4")
        if p["layout"] not in LAYOUT_CHOICES:
            raise ValueError(f"I/Q 排布只能是 {LAYOUT_CHOICES}")
        if p["layout"] != "IIQQ" and p["num_rx"] != 4:
            raise ValueError(f"排布 {p['layout']} 只支持 RX=4（IIQQ 支持 2 或 4）")
        if not 0 < p["display_lowcut"] < p["display_highcut"] < p["frame_rate"] / 2:
            raise ValueError("带通频率需满足 0 < 低 < 高 < 帧率/2")
        s = self.var_seed.get().strip()
        p["seed"] = int(s) if s else None

        # 目标距离范围（界面输入 cm，内部换算成 m 后由 compute 换算 bin）
        def _dist(var, label):
            s = var.get().strip()
            if not s:
                return None
            try:
                v = float(s) / 100.0        # cm -> m
            except ValueError:
                raise ValueError(f"{label} 必须是数字（单位 cm）")
            if v <= 0:
                raise ValueError(f"{label} 必须为正数")
            return v

        p["min_dist"] = _dist(self.var_dist_min, "距离下限")
        p["max_dist"] = _dist(self.var_dist_max, "距离上限")
        if (p["min_dist"] is not None and p["max_dist"] is not None
                and p["max_dist"] <= p["min_dist"]):
            raise ValueError("距离上限必须大于下限")

        mode = self.var_bin_mode.get()
        p["bin_mode"] = ("estimate" if mode == BIN_MODE_ESTIMATE
                         else "per_channel")

        # 大波动检查
        p["despike"] = bool(self.var_spike_on.get())
        if p["despike"]:
            thr = num(self.var_spike_thr, "大波动阈值")
            win = num(self.var_spike_win, "大波动窗口", int)
            mw = num(self.var_spike_width, "最大连续宽度", int)
            if thr <= 0:
                raise ValueError("大波动阈值须为正数")
            if win < 5 or win % 2 == 0:
                raise ValueError("大波动窗口须为不小于 5 的奇数（如 51）")
            if mw < 1:
                raise ValueError("最大连续宽度须 ≥ 1")
            p["despike_threshold"] = thr
            p["despike_window"] = win
            p["despike_max_width"] = mw

        # 快速预览：只用前 N 帧，RANSAC 迭代次数也随之降低，避免等太久
        p["fast_preview"] = bool(self.var_preview.get())
        if p["fast_preview"]:
            n = num(self.var_preview_n, "预览帧数", int)
            # judge_channel 对长度 < 1000 的信号直接返回 0，
            # 帧数太少会导致质量分全为 0 而看不出版本差异，这里兜底。
            if n < 1000:
                raise ValueError(
                    "预览帧数至少 1000（质量模型要求信号长度 ≥ 1000，"
                    "否则质量分会全为 0）"
                )
            p["preview_frames"] = n
        return p

    def on_run(self):
        if self._busy:
            return
        bin_path = self.var_bin.get().strip()
        if not bin_path:
            messagebox.showwarning("缺少输入", "请先选择 .bin 数据文件。")
            return
        if not os.path.isfile(bin_path):
            messagebox.showerror("文件不存在", bin_path)
            return
        try:
            params = self._read_params()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        self._busy = True
        self._stop_flag.clear()
        self.btn_run.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_export.configure(state=tk.DISABLED)
        self.var_progress.set(0.0)
        self.var_status.set("开始处理 ...")
        self._log(f"开始处理: {os.path.basename(bin_path)}")
        self._log(f"  参数: {params}")

        def report(msg, frac=None):
            self.msg_q.put(("progress", (msg, frac)))

        def work():
            try:
                res = compute_all_channels(
                    bin_path, progress=report,
                    should_stop=self._stop_flag.is_set, **params
                )
                self.msg_q.put(("done", res))
            except KeyboardInterrupt:
                self.msg_q.put(("stopped", None))
            except Exception as e:
                self.msg_q.put(("error", (e, traceback.format_exc())))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def on_stop(self):
        if self._busy:
            self._stop_flag.set()
            self.var_status.set("正在中止 ...")
            self._log("收到中止请求，等待当前步骤结束 ...")

    def _pump_queue(self):
        try:
            while True:
                kind, payload = self.msg_q.get_nowait()
                if kind == "progress":
                    msg, frac = payload
                    self.var_status.set(msg)
                    if frac is not None:
                        self.var_progress.set(frac * 100.0)
                    self._log(msg)
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "batch_done":
                    self._on_batch_done(payload)
                elif kind == "stopped":
                    self._busy = False
                    self.btn_run.configure(state=tk.NORMAL)
                    self.btn_stop.configure(state=tk.DISABLED)
                    self.var_status.set("已中止")
                    self._log("处理已中止。")
                elif kind == "error":
                    e, tb = payload
                    self._busy = False
                    self.btn_run.configure(state=tk.NORMAL)
                    self.btn_stop.configure(state=tk.DISABLED)
                    self.var_status.set("处理失败")
                    self._log(f"错误: {e}")
                    self._log(tb)
                    messagebox.showerror("处理失败", f"{type(e).__name__}: {e}")
        except queue.Empty:
            pass
        self.root.after(80, self._pump_queue)

    # ------------------------------------------------------------------
    def _on_batch_done(self, r):
        self._busy = False
        self.btn_run.configure(state=tk.NORMAL)
        self.btn_batch.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.var_progress.set(100.0)
        n_ok, n_fail, n_skip = len(r["ok"]), len(r["failed"]), len(r["skipped"])
        n_resume = len(r.get("resumed") or [])
        secs = float(r.get("seconds") or 0.0)
        nw = int(r.get("workers") or 1)
        per = secs / n_ok if n_ok else 0.0
        self.var_status.set(f"批处理完成: 成功 {n_ok}, 失败 {n_fail}"
                            + (f", 跳过已完成 {n_resume}" if n_resume else ""))
        self._log("-" * 40)
        self._log(f"批处理完成: 成功 {n_ok}, 失败 {n_fail}, 跳过 {n_skip}"
                  + (f", 断点跳过已完成 {n_resume}" if n_resume else ""))
        self._log(f"  耗时 {secs:.1f}s，{nw} 个进程并行，"
                  f"平均 {per:.1f}s/文件")
        if n_resume:
            self._log(f"  断点续处理跳过的文件（{n_resume} 个）:")
            for p, why in r["resumed"]:
                self._log(f"    [跳过] {os.path.basename(p)}: {why}")
        for p, o in r["ok"]:
            self._log(f"  [OK]   {os.path.basename(p)} -> {o}")
        for p, err in r["failed"]:
            self._log(f"  [FAIL] {os.path.basename(p)}: {err}")
        if r["skipped"]:
            self._log(f"  跳过: {[os.path.basename(p) for p in r['skipped']]}")
        try:
            txt = (f"完成: 成功 {n_ok} / 失败 {n_fail} / 跳过 {n_skip}\n"
                   f"{secs:.1f}s，{nw} 进程，{per:.1f}s/文件")
            if n_resume:
                txt += f"\n断点跳过 {n_resume} 个已完成"
            self.lbl_batch.configure(text=txt)
        except tk.TclError:
            pass
        msg = (f"批处理完成。\n\n成功 {n_ok} 个\n失败 {n_fail} 个\n\n"
               f"耗时 {secs:.1f}s（{nw} 进程并行，平均 {per:.1f}s/文件）")
        if n_resume:
            msg += f"\n\n断点续处理跳过了 {n_resume} 个已完成的文件。"
        if n_skip:
            msg += f"\n跳过 {n_skip} 个"
        if r["failed"]:
            msg += "\n\n失败的文件见日志。"
        messagebox.showinfo("批处理完成", msg)

    def _on_done(self, res):
        self.result = res
        self._busy = False
        self.btn_run.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_export.configure(state=tk.NORMAL)
        self.var_progress.set(100.0)
        self.var_status.set("处理完成")
        self._log("处理完成。")
        self._log(f"  virtual shape = {res['virtual_shape']}")
        dr = res.get("dist_range")
        if dr:
            self._log(f"  目标距离范围 = [{dr[0]*100:.1f}, {dr[1]*100:.1f}] cm "
                      f"(bin [{res.get('bin_range')})")
        self._log(f"  目标 bin = {res['target_bins'].tolist()}")
        if res.get("dist_estimate") is not None:
            self._log(f"  估计目标距离 = {res['dist_estimate']*100:.1f} cm "
                      f"(bin {res.get('bin_estimate')})")
        cand = res.get("bin_candidates") or []
        if cand:
            txt = "  ".join(f"bin{b}({d:.2f}m)" for b, d, _ in cand)
            self._log(f"  全通道平均谱候选峰: {txt}")
            try:
                self.lbl_bincand.configure(text="候选峰: " + txt)
            except tk.TclError:
                pass
        # 检测通道是否分裂到不同距离
        tb = np.asarray(res["target_bins"])
        u = np.unique(tb)
        if u.size > 1:
            detail = " ".join(
                f"bin{int(b)}:{int((tb == b).sum())}个通道" for b in u)
            self._log(f"  [注意] 各通道选到了不同距离: {detail}")
            self._log("  [注意] 建议设置「目标范围」把搜索限制在你目标的距离内")
        self._fill_table()
        self.render()

    def _fill_table(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        r = self.result
        disp = self._current_disp()
        for ch in range(r["n_ch"]):
            tx = ch // r["num_rx"]
            rx = ch % r["num_rx"]
            binv = int(r["target_bins"][ch])
            freq = binv * (r["sample_rate"] / r["fft_len"])
            dist = freq * C_LIGHT / (2 * r["slope"])
            fit = r["fit_info"][ch]
            d = disp[ch]
            pkpk = (d.max() - d.min()) * 1e6
            self.tree.insert("", tk.END, values=(
                ch, tx, rx, binv, f"{dist:.2f}",
                "OK" if fit is not None else "FAIL",
                f"{r['probs'][ch]:.3f}", f"{pkpk:.1f}",
            ))

    def _on_tree_select(self, _evt):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        try:
            ch = int(vals[0])
        except (ValueError, IndexError):
            return
        self.var_single.set(ch)
        self.var_view.set("single")
        self.render()

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self):
        if self.result is None:
            self._draw_placeholder()
            return
        view = self.var_view.get()
        if view == "list":
            self._render_list()
        elif view == "grid":
            self._render_grid()
        elif view == "single":
            self._render_single()
        elif view == "iq":
            self._render_iq()
        elif view == "range":
            self._render_range()
        self.canvas.draw()

    def _clear_body(self):
        """清空可滚动区里的内容控件（保留 canvas / inner 本身）。"""
        for w in self.body_inner.winfo_children():
            w.destroy()
        # 释放旧的嵌入式图表与工具栏引用
        self.channel_canvases = []
        self.channel_figs = []
        self.channel_cards = []
        self.channel_toolbars = []
        self._cv = None
        self._tb = None
        self._list_sig = None

    def _blank_for_main(self):
        """主画布只用作"请先处理"的提示，其余视图都用 body 里的独立图。"""
        self.fig.clear()
        self.canvas.get_tk_widget().pack_forget()

    def _ensure_main_canvas(self):
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _embed_figure(self, parent, fig, draw=True, zoom=True,
                      toolbar=True, switch_target=None):
        """
        把一个 Figure 嵌进 parent，返回 (canvas, 工具栏容器)。

        toolbar=True 时在图下方放一条紧凑的标准导航栏，自带：
        Home（返回原始视野）、Back/Forward、Pan、Zoom（框选放大）、Save。

        switch_target 用于多张图共用一条工具栏的场景：传入一个
        callable，用户点这张图时会被调用，用来把共享工具栏切到本图。
        """
        box = ttk.Frame(parent)
        box.pack(fill=tk.X, expand=True)
        c = FigureCanvasTkAgg(fig, master=box)
        w = c.get_tk_widget()
        w.pack(fill=tk.X, expand=True)

        tb = None
        if toolbar:
            tb = _make_compact_toolbar(c, box)

        # 滚轮在图上用于 matplotlib 交互，不要再冒泡给外层滚动容器，
        # 否则缩放的同时整个列表也会跟着滚。
        def eat(_e):
            return "break"
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                w.bind(seq, eat)
            except tk.TclError:
                pass

        if switch_target is not None:
            def on_click(_e, fn=switch_target):
                fn()
            w.bind("<Button-1>", on_click)
            if tb is not None:
                tb.bind("<Button-1>", on_click)

        if draw:
            c.draw_idle()
        return c, tb

    def _use_body(self):
        """切换到可滚动的 body 区，并隐藏主画布。"""
        try:
            self.canvas.get_tk_widget().pack_forget()
        except tk.TclError:
            pass
        self.body.pack(fill=tk.BOTH, expand=True)
        self._clear_body()

    # ------------------------------------------------------------------
    def _render_list(self):
        """
        1-12 通道从上到下竖排，整列可滚动。

        每个通道一张独立图（占满宽度、约 1.7 英寸高），
        比塞进 3x4 网格里看得清楚得多。

        切换滤波频带时复用已建好的 Figure，只更新曲线数据，
        避免重复创建 12 个 Figure 带来的几百毫秒开销。
        """
        r = self.result
        disp = self._current_disp()
        band = self._describe_band()
        t = r["t"]

        # 复用条件必须同时满足：
        #   1) 还是同一份结果、通道数没变、时间轴长度没变
        #      （长度变了还 set_ydata 会因长度不符而画错）
        #   2) 每张图的子图还在、且只有 1 个子图、曲线条数正确
        # 只要有一条不满足就重建，避免出现"图没更新"的假象。
        figs = getattr(self, "channel_figs", []) or []
        cvs = getattr(self, "channel_canvases", []) or []
        sig = (id(r), r["n_ch"], int(t.size))

        ymin = float(disp.min()) * 1e6
        ymax = float(disp.max()) * 1e6
        pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.1

        reuse = (
            getattr(self, "_list_sig", None) == sig
            and len(figs) == r["n_ch"] == len(cvs)
            and all(c.get_tk_widget().winfo_exists() for c in cvs)
            and all(len(f.axes) == 1 and f.axes[0].lines
                    and f.axes[0].lines[0].get_xdata().size == t.size
                    for f in figs)
        )

        if reuse:
            # ---- 复用路径：只重画曲线 ----
            for ch in range(r["n_ch"]):
                ax = self.channel_figs[ch].axes[0]
                ax.lines[0].set_ydata(disp[ch] * 1e6)
                ax.set_ylim(ymin - pad, ymax + pad)
                tx, rx = ch // r["num_rx"], ch % r["num_rx"]
                fit = r["fit_info"][ch]
                pkpk = (disp[ch].max() - disp[ch].min()) * 1e6
                self.channel_cards[ch].configure(
                    text=f"ch{ch}   (TX{tx} x RX{rx})   "
                         f"bin={int(r['target_bins'][ch])}   "
                         f"RANSAC={'OK' if fit is not None else 'FAIL'}   "
                         f"质量分={r['probs'][ch]:.3f}   "
                         f"峰峰值={pkpk:.1f} um")
            self._list_head.configure(
                text=f"{os.path.basename(r['bin_path'])}   "
                     f"12 通道位移（从上到下 ch0 → ch11）   {band}")
            for c in self.channel_canvases:
                c.draw_idle()
            self.root.update_idletasks()
            return

        # ---- 首次构建 ----
        self._blank_for_main()
        self._use_body()

        head = ttk.Frame(self.body_inner)
        head.pack(fill=tk.X, pady=(2, 6))
        self._list_head = ttk.Label(
            head,
            text=f"{os.path.basename(r['bin_path'])}   "
                 f"12 通道位移（从上到下 ch0 → ch11）   {band}",
            font=("", 10, "bold"))
        self._list_head.pack(side=tk.LEFT)
        ttk.Label(head, text="   每张图下方工具栏: Zoom 框选放大  Home 返回",
                  foreground="#777").pack(side=tk.LEFT)

        self.channel_canvases = []
        self.channel_figs = []
        self.channel_cards = []
        self.channel_toolbars = []
        for ch in range(r["n_ch"]):
            tx, rx = ch // r["num_rx"], ch % r["num_rx"]
            fit = r["fit_info"][ch]
            d = disp[ch]
            pkpk = (d.max() - d.min()) * 1e6

            card = ttk.LabelFrame(
                self.body_inner,
                text=f"ch{ch}   (TX{tx} x RX{rx})   "
                     f"bin={int(r['target_bins'][ch])}   "
                     f"RANSAC={'OK' if fit is not None else 'FAIL'}   "
                     f"质量分={r['probs'][ch]:.3f}   "
                     f"峰峰值={pkpk:.1f} um")
            card.pack(fill=tk.X, expand=False, pady=3, padx=2)

            fig = Figure(figsize=(9.5, 1.7), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(t, d * 1e6, linewidth=0.7)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            ax.set_ylabel("um", fontsize=7)
            if ch == r["n_ch"] - 1:
                ax.set_xlabel("time (s)", fontsize=8)
            # 固定边距代替 fig.tight_layout()：后者在 12 张图上要花 300ms+
            fig.subplots_adjust(left=0.075, right=0.995, top=0.97,
                                bottom=0.20 if ch == r["n_ch"] - 1 else 0.13)
            cv, tb = self._embed_figure(card, fig)
            self.channel_cards.append(card)
            self.channel_figs.append(fig)
            self.channel_canvases.append(cv)
            self.channel_toolbars.append(tb)

        self._list_sig = sig
        self.root.update_idletasks()
        self._body_canvas.yview_moveto(0.0)

        # 回到顶部
        self.body_inner.update_idletasks()
        self._body_canvas.yview_moveto(0.0)

    def _band_key(self):
        """当前生效的滤波范围 (lowcut, highcut)。"""
        return (float(self.var_lowcut.get()), float(self.var_highcut.get()))

    def _filter_order(self):
        if self.result:
            return int(self.result["params"].get("filter_order", 4))
        return 4

    def _describe_band(self):
        lo, hi = self._band_key()
        return (f"当前滤波 {lo:g} - {hi:g} Hz"
                f"（对应脉率 {lo*60:.0f} - {hi*60:.0f} bpm）")

    def _current_disp(self):
        """
        取当前滤波范围对应的位移矩阵。

        结果里缓存了「未滤波位移」disp_raw，所以这里改滤波范围只花几十毫秒，
        不需要重新读取 bin / 重跑 FFT / 重跑 RANSAC。
        若启用了大波动检查，缓存的是"替换后"的结果。
        """
        r = self.result
        if r is None:
            return None
        key = self._band_key()
        cache = r.setdefault("bands", {})
        if key not in cache:
            y = _bandpass(r["disp_raw"], r["frame_rate"],
                          key[0], key[1], self._filter_order())
            cfg = self._despike_cfg()
            if cfg["enabled"]:
                y, rep = despike_matrix(y, threshold=cfg["threshold"],
                                        window=cfg["window"],
                                        max_width=cfg["max_width"])
                cache[key + ("spike",)] = rep
            cache[key] = y
        return cache[key]

    def _despike_cfg(self):
        """从界面收集大波动检查的设置。"""
        try:
            thr = float(self.var_spike_thr.get())
        except (ValueError, AttributeError):
            thr = 15.0
        try:
            win = int(float(self.var_spike_win.get()))
        except (ValueError, AttributeError):
            win = 51
        try:
            mw = int(float(self.var_spike_width.get()))
        except (ValueError, AttributeError):
            mw = 3
        try:
            en = bool(self.var_spike_on.get())
        except AttributeError:
            en = False
        return dict(enabled=en, threshold=thr, window=win, max_width=mw)

    def _current_spike_report(self):
        """取当前频带的大波动检查报告（未启用则返回 None）。"""
        r = self.result
        if r is None:
            return None
        key = self._band_key() + ("spike",)
        return r.setdefault("bands", {}).get(key)

    def _on_band_selected(self, _evt=None):
        """选中预设频带：把上下限填进输入框并立即应用。"""
        if getattr(self, "_syncing_band", False):
            return                    # 防止 _sync_band_name_from_cutoffs 触发递归
        name = self.var_band.get()
        if name in BAND_PRESETS:
            lo, hi = BAND_PRESETS[name]
            self.var_lowcut.set(f"{lo:g}")
            self.var_highcut.set(f"{hi:g}")
        self.on_apply_band()

    def _sync_band_name_from_cutoffs(self):
        """根据当前上下限反查预设名；没有匹配的就显示"自定义"。"""
        try:
            cur = (float(self.var_lowcut.get()), float(self.var_highcut.get()))
        except ValueError:
            return
        self._syncing_band = True
        try:
            for name, edges in BAND_PRESETS.items():
                if abs(edges[0] - cur[0]) < 1e-9 and abs(edges[1] - cur[1]) < 1e-9:
                    self.var_band.set(name)
                    return
            self.var_band.set(BAND_CUSTOM)
        finally:
            self._syncing_band = False

    def on_apply_band(self):
        """应用当前滤波范围（不重跑处理流程）。"""
        if self.result is None:
            return
        try:
            lo = float(self.var_lowcut.get())
            hi = float(self.var_highcut.get())
        except ValueError:
            messagebox.showerror("频率无效", "带通低/高必须是数字。")
            return
        fs = float(self.result.get("frame_rate", 250))
        if not 0 < lo < hi < fs / 2:
            messagebox.showerror(
                "频率无效",
                f"需满足 0 < 低 < 高 < {fs/2:g} Hz（帧率 {fs:g} 的一半）。\n"
                f"当前: {lo:g} - {hi:g} Hz")
            return

        self._current_disp()          # 触发计算并缓存
        self._sync_band_name_from_cutoffs()   # 下拉框同步为预设名或"自定义"
        self._fill_table()            # 峰峰值列随频带变化，需重算
        self.render()
        self._log(self._describe_band())
        self._report_spikes()
        try:
            self.lbl_band_info.configure(text=self._describe_band())
        except tk.TclError:
            pass

    def _report_spikes(self):
        """把大波动检查的结果写到日志和界面标签。"""
        rep = self._current_spike_report()
        if rep is None:
            try:
                self.lbl_spike.configure(text="")
            except tk.TclError:
                pass
            return
        n_rep = sum(r["n_replaced"] for r in rep)
        n_skip = sum(r["skipped_runs"] for r in rep)
        txt = (f"已替换 {n_rep} 个孤立点；跳过 {n_skip} 段连续波形\n"
               + " ".join(f"ch{r['ch']}={r['n_replaced']}" for r in rep))
        try:
            self.lbl_spike.configure(text=txt)
        except tk.TclError:
            pass
        self._log(f"大波动检查: 替换 {n_rep} 点, 跳过 {n_skip} 段连续波形")
        self._log("  " + " ".join(f"ch{r['ch']}:{r['n_replaced']}" for r in rep))

    # ------------------------------------------------------------------
    def _render_grid(self):
        r = self.result
        disp = self._current_disp()
        band = self._describe_band()
        self.fig = Figure(figsize=(11, 8.5), dpi=100)
        n = r["n_ch"]
        ncol = 4
        nrow = (n + ncol - 1) // ncol
        ymin = float(disp.min()) * 1e6
        ymax = float(disp.max()) * 1e6
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
        t = r["t"]
        for ch in range(n):
            ax = self.fig.add_subplot(nrow, ncol, ch + 1)
            ax.plot(t, disp[ch] * 1e6, linewidth=0.7)
            tx, rx = ch // r["num_rx"], ch % r["num_rx"]
            ax.set_title(
                f"ch{ch} (TX{tx}xRX{rx}) bin={int(r['target_bins'][ch])} "
                f"p={r['probs'][ch]:.2f}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(ymin - pad, ymax + pad)
            if ch // ncol == nrow - 1:
                ax.set_xlabel("time (s)", fontsize=7)
            if ch % ncol == 0:
                ax.set_ylabel("disp (um)", fontsize=7)
        self.fig.suptitle(f"{os.path.basename(r['bin_path'])}  -  "
                          f"12 通道位移  [{band}]", fontsize=11)
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._cv, self._tb = self._embed_figure(self.body_inner, self.fig)
    def _render_single(self):
        r = self.result
        disp = self._current_disp()
        ch = int(self.var_single.get()) % r["n_ch"]
        self.fig = Figure(figsize=(11, 5.0), dpi=100)
        ax = self.fig.add_subplot(111)
        ax.plot(r["t"], disp[ch] * 1e6, linewidth=0.8)
        tx, rx = ch // r["num_rx"], ch % r["num_rx"]
        fit = r["fit_info"][ch]
        ax.set_title(
            f"{os.path.basename(r['bin_path'])}  -  ch{ch} (TX{tx}xRX{rx})  "
            f"bin={int(r['target_bins'][ch])}  "
            f"RANSAC={'OK' if fit is not None else 'FAIL'}  "
            f"prob={r['probs'][ch]:.3f}  "
            f"pk-pk={(disp[ch].max()-disp[ch].min())*1e6:.1f}um  "
            f"[{self._describe_band()}]", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("displacement (um)")
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self._cv, self._tb = self._embed_figure(self.body_inner, self.fig)

    def _render_iq(self):
        r = self.result
        self.fig = Figure(figsize=(11, 8.5), dpi=100)
        n = r["n_ch"]
        ncol = 4
        nrow = (n + ncol - 1) // ncol
        for ch in range(n):
            ax = self.fig.add_subplot(nrow, ncol, ch + 1)
            z = r["signal_raw"][ch]
            ax.scatter(z.real, z.imag, s=0.4, alpha=0.35, edgecolors="none")
            fit = r["fit_info"][ch]
            if fit is not None:
                xc, yc, R = fit
                th = np.linspace(0, 2 * np.pi, 200)
                ax.plot(xc + R * np.cos(th), yc + R * np.sin(th), "r-", lw=1.0)
                ax.scatter([xc], [yc], color="red", s=12)
            tx, rx = ch // r["num_rx"], ch % r["num_rx"]
            ax.set_title(f"ch{ch} (TX{tx}xRX{rx}) "
                         f"{'OK' if fit is not None else 'FAIL'}", fontsize=8)
            ax.set_aspect("equal", adjustable="datalim")
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
        self.fig.suptitle(f"{os.path.basename(r['bin_path'])}  -  "
                          f"IQ 星座图（DC 消除前）", fontsize=11)
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._cv, self._tb = self._embed_figure(self.body_inner, self.fig)

    def _render_range(self):
        r = self.result
        ri = r.get("range_info")
        if not ri:
            return
        self.fig = Figure(figsize=(11, 5.0), dpi=100)
        ax = self.fig.add_subplot(111)
        d = ri["dist_axis"]
        mean_p = 20 * np.log10(ri["prof_mean"] + 1e-9)
        lo_p = 20 * np.log10(ri["prof_min"] + 1e-9)
        hi_p = 20 * np.log10(ri["prof_max"] + 1e-9)
        ax.fill_between(d, lo_p, hi_p, alpha=0.25, color="steelblue",
                        label="12 通道范围")
        ax.plot(d, mean_p, linewidth=1.4, color="C0", label="12 通道平均")

        # 各通道选中的 bin 用竖线标出（按出现次数去重，避免 12 条线糊成一片）
        tb = np.asarray(r["target_bins"])
        vals, counts = np.unique(tb, return_counts=True)
        for b, c in zip(vals, counts):
            x = ri["dist_axis"][int(b)]
            ax.axvline(x, color="crimson", alpha=0.55, linewidth=1.0,
                       linestyle="--")
            ax.text(x, ax.get_ylim()[1], f" bin{b}×{c}",
                    rotation=90, fontsize=7, color="crimson",
                    va="top", ha="left")

        lo_b, hi_b = r.get("bin_range", (0, len(d)))
        ax.axvspan(d[lo_b], d[min(hi_b, len(d) - 1)], color="orange", alpha=0.10)
        ax.set_xlabel("distance (m)")
        ax.set_ylabel("dB")
        ax.set_title(f"{os.path.basename(r['bin_path'])}  -  距离谱"
                     f"（红虚线=各通道选中的 bin，橙色区=目标范围）", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        self.fig.tight_layout()
        self._cv, self._tb = self._embed_figure(self.body_inner, self.fig)

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def _naming(self, ch, prob):
        prefix = self.var_prefix.get().strip() or "channel"
        suffix = self.var_suffix.get().strip()
        ext = self.var_ext.get().strip() or ".csv"
        if not ext.startswith("."):
            ext = "." + ext
        name = f"{prefix}_{ch}"
        if suffix:
            name += f"_{suffix}"
        if self.var_tag_prob.get():
            name += f"_prob_{int(prob * 100):02d}"
        return name + ext

    def on_export(self):
        if self.result is None:
            return
        outdir = self.var_outdir.get().strip()
        if not outdir:
            messagebox.showwarning("缺少输出目录", "请先选择输出目录。")
            return
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as e:
            messagebox.showerror("无法创建输出目录", str(e))
            return

        disp = self._current_disp()
        r = self.result
        try:
            saved = write_displacement_csvs(
                r, outdir, disp=disp, **self._export_opts())
        except OSError as e:
            messagebox.showerror("保存失败", str(e))
            return

        self._log(f"已导出 {len(saved)} 个文件到 {outdir}")
        for f in saved:
            self._log(f"  {os.path.basename(f)}")
        messagebox.showinfo("导出完成",
                            f"已导出 {len(saved)} 个文件到:\n{outdir}")


# =========================================================================
def main():
    # 打包成 exe 后必须调用：否则子进程启动时会重新执行整个脚本，
    # 导致无限递归地 spawn 新进程（Windows 上表现为程序卡死/刷屏）。
    import multiprocessing
    multiprocessing.freeze_support()

    # 中文 Windows 控制台/界面编码兜底
    import sys
    for s in (sys.stdout, sys.stderr):
        if s is not None and hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    RadarGUI(root)
    root.mainloop()


if __name__ == "__main__":
    # 这个判断在打包后同样需要保留（配合 freeze_support）
    import multiprocessing
    multiprocessing.freeze_support()
    main()
