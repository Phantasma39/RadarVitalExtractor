import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免弹窗
import matplotlib.pyplot as plt

from radar_project.Judge import judge_channel
from radar_project.utils import read_and_decode
from radar_project.range_fft import range_fft, final_signal
from radar_project.DC_Eliminate import fit_circle_ransac_iq
from radar_project.displacement_processing import compute_displacement, bandpass_filter


def process_single_bin(file_path, output_root="output", fft_len=1024,
                       do_dc_eliminate=True, lowcut=0.5, highcut=5.0):
    """
    处理单个 .bin 文件，输出微位移数据。

    参数:
        file_path: .bin 文件路径
        output_root: 输出根目录
        fft_len: FFT 点数
        do_dc_eliminate: 是否执行 DC 消除（RANSAC 圆拟合）
        lowcut: 带通滤波器下截止频率 (Hz)
        highcut: 带通滤波器上截止频率 (Hz)

    返回:
        best_disp: 最佳通道的微位移信号
    """
    name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"正在处理: {name}")

    c_v = 3e8  # 光速
    frequency_slope = 80e12  # 斜率
    sample_rate = 1e7  # ADC采样率
    fc = 77e9  # 基频
    frame_rate = 250  # 帧率

    # ===== 1. 读取数据 =====
    print("  读取 bin 文件...")
    adc_data = read_and_decode(file_path)

    # ===== 2. Range FFT =====
    print("  执行 Range FFT...")
    range_data = range_fft(
        adc_data,
        axis=-1,
        fft_len=fft_len,
        window_type="hann",
        remove_dc=True,
        keep_positive=True,
        output="complex"
    )

    # ===== 3. 计算功率并选最大 bin =====
    power = np.mean(np.abs(range_data), axis=(1, 2))  # (12, RangeBin)
    target_bins = np.argmax(power, axis=1)  # (12,)
    print(f"  Target bins: {target_bins}")

    for i in range(len(target_bins)):
        frequency = target_bins[i] * (sample_rate / fft_len)
        R = (frequency * 3e8) / (2 * frequency_slope)
        print(f"  通道{i}: 频率={frequency:.1f}Hz, 距离={R:.2f}m")

    # ===== 4. 提取目标 bin 信号 =====
    signal = final_signal(range_data, target_bins)  # (12, Frame)

    # ===== 5. DC 消除（RANSAC 圆拟合）=====
    if do_dc_eliminate:
        print("  执行 DC 消除（RANSAC 圆拟合）...")
        for i in range(len(target_bins)):
            xc, yc, R = fit_circle_ransac_iq(signal[i])
            if xc is None:
                print(f"  通道{i} 拟合失败，取平均值处理")
            else:
                signal[i] = signal[i] - xc - yc * 1j
                print(f"  通道{i} 拟合成功 (xc={xc:.3f}, yc={yc:.3f})")

    # ===== 6. 计算微位移 =====
    print("  计算微位移...")
    disp = compute_displacement(
        signal,
        fc=fc,
        frame_rate=frame_rate,
        do_detrend=True,
        do_filter=True,
        lowcut=lowcut,
        highcut=highcut,
        filter_order=4,
        save_csv=True,
        save_dir="output_" + name,
        save_root=output_root,
        draw=False
    )

    print(f"  完成！结果保存到: {os.path.join(output_root, 'output_' + name)}")
    return disp


def main():
    """命令行入口：python -m radar_project.main <bin文件路径> [输出目录]"""
    if len(sys.argv) < 2:
        print("用法: python -m radar_project.main <bin文件路径> [输出目录]")
        print("示例: python -m radar_project.main F:/data_new/adc_data_Raw_xxx.bin")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        sys.exit(1)

    output_root = sys.argv[2] if len(sys.argv) >= 3 else "output"

    process_single_bin(file_path, output_root=output_root)


if __name__ == "__main__":
    main()