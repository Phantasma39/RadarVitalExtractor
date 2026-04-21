import numpy as np
import os
from tqdm import tqdm 

from src.utils import read_and_decode
from src.range_fft import range_fft, final_signal
from src.DC_Eliminate import fit_circle_ransac_iq
from src.displacement_processing import compute_displacement 

# ===== 数据文件夹 =====
data_folder = r"F:\data_new"

# ===== 获取所有 bin 文件 =====
file_list = [f for f in os.listdir(data_folder) if f.endswith(".bin")]
total_files = len(file_list)

print(f"✅ 找到 {total_files} 个 .bin 文件，开始处理...\n")

# ===== 带进度条遍历 =====
for file in tqdm(file_list, desc="整体进度", unit="文件"):

    file_path = os.path.join(data_folder, file)
    name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"正在处理{name}\n")
    try:
        # ===== 你原来的 main 内容 =====
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
            save_root = r"F:\\my_output_new",
            save_dir="output_" + name
        )

    except Exception as e:
        print(f"\n❌ 处理失败: {file}  | 错误: {e}")

print("\n🎉 所有文件处理完成！")