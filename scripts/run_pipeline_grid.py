# 完整管线 + 12 通道全景输出（位移图 / IQ 星座图 / 距离谱 / CSV）。
#
# 用法: python scripts/run_pipeline_grid.py [bin文件] [输出目录]
#      不带参数时默认处理 data/TEST.bin，输出到 output_TEST/
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  把 src/ 加入 sys.path，免安装即可 import

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radar_project.utils import read_and_decode
from radar_project.range_fft import range_fft, final_signal
from radar_project.DC_Eliminate import fit_circle_ransac_iq
from radar_project.Judge import judge_channel

# 仓库根目录 = 本文件所在目录(scripts/)的上一级
ROOT = Path(__file__).resolve().parent.parent

BIN = Path(sys.argv[1]) if len(sys.argv) >= 2 else ROOT / "data" / "TEST.bin"
OUT = Path(sys.argv[2]) if len(sys.argv) >= 3 else ROOT / "output_TEST"

FFT_LEN = 1024
FC = 77e9
FRAME_RATE = 250
SLOPE = 80e12
SAMPLE_RATE = 1e7
N_TX = 3
N_RX = 4

if not BIN.exists():
    print(f"错误: 找不到 bin 文件 - {BIN}")
    print("用法: python scripts/run_pipeline_grid.py [bin文件] [输出目录]")
    sys.exit(1)

OUT.mkdir(parents=True, exist_ok=True)
CHIRP_PER_TX = 8

print(f"读取 {BIN} ...")
virtual = read_and_decode(BIN, num_frames=6250, num_chirps=24,
                          num_rx=N_RX, num_samples=256, num_tx=N_TX)
print(f"virtual shape = {virtual.shape}  (12 通道, 6250 帧, 8 chirp/TX, 256 采样)")

print("Range FFT ...")
rd = range_fft(virtual, axis=-1, fft_len=FFT_LEN, window_type="hann",
               remove_dc=True, keep_positive=True, output="complex")

power = np.mean(np.abs(rd), axis=(1, 2))
bins = np.argmax(power, axis=1)
print(f"目标 bin = {bins.tolist()}")

sig = final_signal(rd, bins)          # (12, 6250) 复数
signals_raw = sig.copy()              # 保留原始，用于画 IQ

print("DC 消除 (RANSAC) ...")
fit_info = []
sig_dc = sig.copy()
for ch in range(12):
    xc, yc, R = fit_circle_ransac_iq(sig_dc[ch], verbose=False)
    if xc is None:
        fit_info.append((ch, None, None, None))
    else:
        fit_info.append((ch, xc, yc, R))
        sig_dc[ch] = sig_dc[ch] - xc - yc * 1j

print("计算微位移 ...")
phase = np.unwrap(np.angle(sig_dc), axis=1)
disp = (3e8 / (4 * np.pi * FC)) * (phase - phase[:, [0]])
from scipy.signal import detrend  # noqa: E402
disp = detrend(disp, axis=1)
from radar_project.displacement_processing import bandpass_filter  # noqa: E402
disp = bandpass_filter(disp, fs=FRAME_RATE, lowcut=0.5, highcut=5.0, order=4, axis=1)

t = np.arange(disp.shape[1]) / FRAME_RATE

# ---- 质量分 ----
print("评估通道质量 ...")
probs = []
for ch in range(12):
    is_good, p = judge_channel(disp[ch], FRAME_RATE)
    probs.append(p)

# ---- 汇总表 ----
print("\n" + "=" * 84)
print(f"{'ch':>3} {'TX':>3} {'RX':>3} {'bin':>5} {'距离(m)':>9} {'RANSAC':>8} "
      f"{'概率':>7} {'峰峰值(um)':>12}")
print("=" * 84)
for ch in range(12):
    tx = ch // N_RX
    rx = ch % N_RX
    f = bins[ch] * (SAMPLE_RATE / FFT_LEN)
    dist = f * 3e8 / (2 * SLOPE)
    ok = "OK" if fit_info[ch][1] is not None else "FAIL"
    pp = (disp[ch].max() - disp[ch].min()) * 1e6
    print(f"{ch:>3} {tx:>3} {rx:>3} {bins[ch]:>5} {dist:>9.2f} {ok:>8} "
          f"{probs[ch]:>7.3f} {pp:>12.2f}")
print("=" * 84)

# ---- 1. 12 通道位移图 (3x4) ----
fig, axes = plt.subplots(3, 4, figsize=(24, 12))
ymin = disp.min() * 1e6
ymax = disp.max() * 1e6
for ch in range(12):
    ax = axes[ch // 4, ch % 4]
    ax.plot(t, disp[ch] * 1e6, linewidth=0.7)
    tx, rx = ch // N_RX, ch % N_RX
    ax.set_title(f"ch{ch}  (TX{tx}×RX{rx})  bin={bins[ch]}  prob={probs[ch]:.2f}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("displacement (um)")
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
fig.suptitle("TEST.bin  -  12 virtual channels displacement (0.5-5 Hz)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
p1 = OUT / "displacement_12ch_grid.svg"
fig.savefig(p1, format="svg", bbox_inches="tight")
plt.close(fig)
print(f"\n已保存: {p1}")

# ---- 2. 每通道单独图 ----
for ch in range(12):
    fig = plt.figure(figsize=(16, 4))
    plt.plot(t, disp[ch] * 1e6, linewidth=0.8)
    tx, rx = ch // N_RX, ch % N_RX
    plt.title(f"Channel {ch} (TX{tx}×RX{rx})  -  bin={bins[ch]}  prob={probs[ch]:.2f}  "
              f"pk-pk={((disp[ch].max()-disp[ch].min())*1e6):.1f}um")
    plt.xlabel("time (s)")
    plt.ylabel("displacement (um)")
    plt.grid(True, alpha=0.3)
    p = OUT / f"displacement_ch{ch:02d}.svg"
    fig.savefig(p, format="svg", bbox_inches="tight")
    plt.close(fig)
print(f"已保存 12 张单通道图: {OUT}/displacement_ch00.svg ... ch11.svg")

# ---- 3. IQ 星座图 (DC 消除前) ----
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for ch in range(12):
    ax = axes[ch // 4, ch % 4]
    z = signals_raw[ch]
    ax.scatter(z.real, z.imag, s=0.5, alpha=0.4, edgecolors="none")
    if fit_info[ch][1] is not None:
        xc, yc, R = fit_info[ch][1], fit_info[ch][2], fit_info[ch][3]
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(xc + R * np.cos(th), yc + R * np.sin(th), "r-", linewidth=1.2)
        ax.scatter([xc], [yc], color="red", s=20)
    tx, rx = ch // N_RX, ch % N_RX
    ax.set_title(f"ch{ch} (TX{tx}×RX{rx})  {'circle OK' if fit_info[ch][1] is not None else 'fit FAILED'}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
fig.suptitle("TEST.bin - IQ constellation per channel (before DC removal)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
p3 = OUT / "iq_constellation_grid.svg"
fig.savefig(p3, format="svg", bbox_inches="tight")
plt.close(fig)
print(f"已保存: {p3}")

# ---- 4. 距离谱 ----
fig = plt.figure(figsize=(14, 6))
rb = np.arange(rd.shape[-1])
for ch in range(12):
    prof = np.abs(rd[ch]).mean(axis=(0, 1))
    plt.plot(rb, 20 * np.log10(prof + 1e-9), linewidth=1, label=f"ch{ch}")
plt.xlabel("range bin")
plt.ylabel("dB")
plt.title("TEST.bin - mean range profile (12 channels)")
plt.grid(True, alpha=0.3)
plt.legend(ncol=4, fontsize=8)
p4 = OUT / "range_profile_12ch.svg"
fig.savefig(p4, format="svg", bbox_inches="tight")
plt.close(fig)
print(f"已保存: {p4}")

# ---- 5. CSV ----
for ch in range(12):
    np.savetxt(OUT / f"channel_{ch:02d}_prob_{int(probs[ch]*100):02d}.csv",
               np.column_stack((t, disp[ch])), delimiter=",",
               header="time(s),displacement(m)", comments="")
print(f"已保存 12 个 CSV 到 {OUT}/")
print("\n完成。")
