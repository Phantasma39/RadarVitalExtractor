import numpy as np
from utils import read_and_decode
from range_fft import range_fft, final_signal
from DC_Eliminate import fit_circle_ransac_iq
from displacement_processing import compute_displacement
import os

# ===== 数据文件夹 =====
data_folder = r"F:\data_new"

# ===== 遍历所有 bin 文件 =====
for file in os.listdir(data_folder):

    if not file.endswith(".bin"):
        continue

    file_path = os.path.join(data_folder, file)
    name = os.path.splitext(os.path.basename(file_path))[0]

    print("\n======================")
    print("正在处理:", name)

    try:
        # ===== 你原来的 main 内容（一行不改）=====

        print(name)
        c_v = 3e8
        FFT_len = 1024
        num_chirps = 24
        frequency_slope = 80e12
        sample_rate = 1e7
        fc = 77e9
        frame_rate = 250

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

        power = np.mean(np.abs(range_data), axis=(1, 2))
        target_bins = np.argmax(power, axis=1)

        print("Target bins:", target_bins)

        for i in range(len(target_bins)):
            frequency = target_bins[i] * (sample_rate / FFT_len)
            R = (frequency * 3e8) / (2 * frequency_slope)
            print(f"通道{i}选取频率为{frequency}Hz,对应的距离为{R}m.")

        signal = final_signal(range_data, target_bins)
        signal = signal - np.mean(signal)

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
            save_dir="output_" + name   # ✅ 每个文件自动分文件夹
        )

    except Exception as e:
        print("❌ 出错:", file, e)