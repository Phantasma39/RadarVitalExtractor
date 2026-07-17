# 这是一个画图用的文件，我想到什么就画什么
# 使用方式: python -m radar_project.Draw <bin文件路径> [输出目录]

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from radar_project.utils import read_and_decode
from radar_project.range_fft import range_fft, final_signal
from radar_project.displacement_processing import compute_displacement


def draw_process(file_path, output_root="output", fft_len=1024):
    """
    读取 bin 文件，执行完整处理流程并保存 SVG 图表。
    """
    name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"正在处理: {name}")

    c_v = 3e8
    sample_rate = 1e7
    frequency_slope = 80e12
    fc = 77e9
    frame_rate = 250

    adc_data = read_and_decode(file_path)

    range_data = range_fft(
        adc_data,
        axis=-1,
        fft_len=fft_len,
        window_type="hann",
        remove_dc=True,
        keep_positive=True,
        output="complex"
    )

    # 计算功率并选最大 bin
    power = 10 * np.log10(np.mean(np.abs(range_data) ** 2, axis=(1, 2)) + 1e-6)
    target_bins = np.argmax(power, axis=1)

    for i in range(len(target_bins)):
        frequency = target_bins[i] * (sample_rate / fft_len)
        R = (frequency * 3e8) / (2 * frequency_slope)
        print(f"通道{i} 选取频率为 {frequency:.1f}Hz, 对应距离 {R:.2f}m")

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
        save_dir="output_" + name,
        save_root=output_root,
        draw=True  # 生成 SVG 图
    )

    print(f"完成！结果保存到: {os.path.join(output_root, 'output_' + name)}")


def main():
    if len(sys.argv) < 2:
        print("用法: python -m radar_project.Draw <bin文件路径> [输出目录]")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        sys.exit(1)

    output_root = sys.argv[2] if len(sys.argv) >= 3 else "output"
    draw_process(file_path, output_root=output_root)


if __name__ == "__main__":
    main()