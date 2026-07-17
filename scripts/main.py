# ====================== 这里修改路径 ======================
file_path = r"F:\data_new\adc_data_Raw_sujunwei_13.bin"
output_root = r"F:\my_output"
# ==========================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from radar_project.Judge import judge_channel
from radar_project.utils import read_and_decode
from radar_project.range_fft import range_fft, final_signal
from radar_project.DC_Eliminate import fit_circle_ransac_iq
from radar_project.displacement_processing import compute_displacement, bandpass_filter

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
power = np.mean(np.abs(range_data), axis=(1, 2))  # (12, RangeBin)

target_bins = np.argmax(power, axis=1)  # (12,)

print("Target bins:", target_bins)

for i in range(len(target_bins)):
    frequency = target_bins[i] * (sample_rate / FFT_len)
    R = (frequency * 3e8) / (2 * frequency_slope)
    print(f"通道{i}选取频率为{frequency}Hz,对应的距离为{R}m.")

signal = final_signal(range_data, target_bins)  # 得到最终结果

# 去直流偏置看看效果
for i in range(len(target_bins)):
    xc, yc, R = fit_circle_ransac_iq(signal[i])
    if xc is None:
        print(f"通道{i}拟合失败，取平均值处理")
    else:
        print(f"{xc},{yc}")
        signal[i] = signal[i] - xc - yc * 1j
        print(f"通道{i}拟合成功")

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
    save_root=output_root,
    save_dir="output_" + name,
    draw=False
)

print(f"完成！结果保存到: {os.path.join(output_root, 'output_' + name)}")