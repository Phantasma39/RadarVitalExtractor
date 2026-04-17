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

file_path = r"F:\data_new\adc_data_Raw_mayuning_8.bin"
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
    fft_len=512,
    window_type="hann",
    remove_dc=True,
    keep_positive=True,
    output="complex"
)

# ===== 计算功率并选最大bin =====
power = np.mean(np.abs(range_data), axis=(1, 2))  # (12, RangeBin)

target_bins = np.argmax(power, axis=1)  # (12,)

print("Target bins:", target_bins)




for i in range(len(target_bins)):
    frequency = target_bins[i] * (sample_rate / FFT_len)
    R = (frequency * 3e8) / (2 * frequency_slope)
    print(f"通道{i}选取频率为{frequency}Hz,对应的距离为{R}m.")

signal = final_signal(range_data, target_bins)  # 得到最终结果

signal = signal - np.mean(signal)

N = target_bins.shape[0]  # 12通道

beamformed = []

angles = np.linspace(-60, 60, 121)  # 角度范围

for theta in angles:

    theta_rad = np.deg2rad(theta)

    w = np.exp(-1j * 2 * np.pi * d * np.arange(N) * np.sin(theta_rad) / lam)

    y = np.dot(w.conj(), signal)  # 合成

    beamformed.append(y)

beamformed = np.array(beamformed)

best_signal=compute_displacement(
        beamformed,
        fc=77e9,
        frame_rate=250,  # Hz (4ms → 250Hz)
        do_detrend=True,
        do_filter=True,
        lowcut=0.5,
        highcut=5.0,
        filter_order=4,
        save_csv=True,
        save_dir="output"
)

# # 去直流偏置看看效果
# for i in range(len(target_bins)):
#     xc, yc, R = fit_circle_ransac_iq(signal[i])
#     if xc is None:
#
#         print(f"通道{i}拟合失败，取平均值处理")
#     else:
#         signal[i] = signal[i] - xc - yc * 1j
#         print(f"通道{i}拟合成功")

plt.figure()
plt.imshow(np.abs(beamformed), aspect='auto', origin='lower')
plt.title("Angle-Time Map")
plt.xlabel("Frame")
plt.ylabel("Angle Index")
plt.colorbar()
plt.show()

disp = compute_displacement(
    signal,
    fc=fc,
    frame_rate=frame_rate,  # 4ms一帧
    do_detrend=True,
    do_filter=True,
    lowcut=0.5,  # 呼吸
    highcut=5.0,
    filter_order=4,
    save_csv=True,
    save_dir="output_" + name)
