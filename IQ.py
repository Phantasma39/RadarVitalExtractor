import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft


# ================= 参数 =================
bin_file = r"DATA/adc_data_Raw_12.bin"

num_frames = 6250
num_chirps = 24
num_rx = 4
num_adc_samples = 256

fft_len = 512


# ================= 1. 正确读取 =================
raw = np.fromfile(bin_file, dtype=np.int16)

raw = raw.reshape(-1, 8)

iq_rx0 = raw[:, 0] + 1j * raw[:, 2]
iq_rx1 = raw[:, 1] + 1j * raw[:, 3]
iq_rx2 = raw[:, 4] + 1j * raw[:, 6]
iq_rx3 = raw[:, 5] + 1j * raw[:, 7]

iq = np.stack([iq_rx0, iq_rx1, iq_rx2, iq_rx3], axis=1)  # [N, 4]


# ================= 2. reshape =================
expected = num_frames * num_chirps * num_adc_samples

if iq.shape[0] != expected:
    raise ValueError(f"数据长度不对: {iq.shape[0]} vs {expected}")

# [frame, chirp, sample, rx]
data = iq.reshape(num_frames, num_chirps, num_adc_samples, num_rx)

# → [chirp, rx, sample, frame]
adcData = np.transpose(data, (1, 3, 2, 0))


# ================= 3. Range FFT =================
sig = np.mean(adcData, axis=0)   # [rx, sample, frame]

# ⭐ 去慢时间均值（关键！！！）
sig = sig - np.mean(sig, axis=2, keepdims=True)

sig_fft = fft(sig, n=fft_len, axis=1)

power = np.mean(np.abs(sig_fft)**2, axis=(0, 2))
target_bin = np.argmax(power)

print("新bin:", target_bin)

var_power = np.var(np.abs(sig_fft), axis=2)  # [rx, range_bin]
var_power = np.mean(var_power, axis=0)

target_bin = np.argmax(var_power)
print("方差选bin:", target_bin)

# ================= 5. 画12图 =================
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for i in range(12):
    r = i // 4
    c = i % 4

    rx = c

    frames_per_block = num_frames // 3
    start = r * frames_per_block
    end = (r + 1) * frames_per_block

    iq_target = sig_fft[rx, target_bin, start:end]

    # 去直流（关键）
    iq_target = iq_target - np.mean(iq_target)

    iq_plot = iq_target[::]

    axes[r, c].scatter(iq_plot.real, iq_plot.imag, s=3)
    axes[r, c].set_title(f"Rx{rx} Block{r}")
    axes[r, c].axis("equal")
    axes[r, c].grid()

plt.tight_layout()
plt.show()