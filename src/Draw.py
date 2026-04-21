# 这是一个画图用的文件，我想到什么就画什么，我又记不住

import numpy as np
from scipy.signal import butter, filtfilt
from Judge import judge_channel
from utils import read_and_decode
from range_fft import range_fft, final_signal
from DC_Eliminate import fit_circle_ransac_iq
from displacement_processing import compute_displacement, bandpass_filter
import os
from scipy.signal import butter, filtfilt, detrend
import matplotlib.pyplot as plt

file_path = r"F:\data_new\adc_data_Raw_sujunwei_1.bin"
name = os.path.splitext(os.path.basename(file_path))[0]

print(name)
c_v = 3e8  # 光速
FFT_len = 1024
num_chirps = 24
frequency_slope = 80e12  # 斜率
sample_rate = 1e7  # ADC采样率
fc = 77e9  # 基频
frame_rate = 250  # 帧率
lam = c_v / fc
d = lam / 2  # 天线间距

adc_data = read_and_decode(file_path)

range_data = range_fft(
    adc_data,
    axis=-1,
    fft_len=FFT_len,
    window_type="hann",
    remove_dc=True,
    keep_positive=True,
    output="complex"
)

# ===== 计算功率并选最大bin =====
#power = np.mean(np.abs(range_data)**2, axis=(1, 2))
power = 10 * np.log10(np.mean(np.abs(range_data)**2, axis=(1, 2)) + 1e-6)
target_bins = np.argmax(power, axis=1)  # (12,)

for i in range(len(target_bins)):
    frequency = target_bins[i] * (sample_rate / FFT_len)
    R = (frequency * 3e8) / (2 * frequency_slope)
    print(f"通道{i}选取频率为{frequency}Hz,对应的距离为{R}m.")

signal = final_signal(range_data, target_bins)

disp = compute_displacement(
    signal,
    fc=fc,
    frame_rate=frame_rate,
    do_detrend=True,
    do_filter=True,
    lowcut=0.5,
    highcut=5.0,
    filter_order=4,
    save_csv=True,
    save_dir="output_" + name  # ✅ 每个文件自动分文件夹
)