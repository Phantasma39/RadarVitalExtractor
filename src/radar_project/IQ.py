# IQ 星座图分析工具
# 使用方式: python -m radar_project.IQ <bin文件路径>

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from radar_project.utils import read_and_decode
from radar_project.DC_Eliminate import fit_circle_ransac_iq


def extract_12ch_iq(virtual, range_bin=50):
    """
    virtual: [12, Frame, Chirp, Sample]
    return:  [12, N] complex IQ
    """
    num_ch = virtual.shape[0]
    iq_list = []

    for ch in range(num_ch):
        # 取固定距离bin
        sig = virtual[ch, :, :, range_bin]
        # 展平 slow time
        sig = sig.reshape(-1)
        iq_list.append(sig)

    return np.stack(iq_list, axis=0)


def plot_12_iq(iq_12ch, max_points=6000, save_dir="figuress"):
    """
    绘制12个通道的IQ星座图，若拟合圆成功则画出圆，
    保存为三种矢量格式: pdf, svg, eps
    """
    os.makedirs(save_dir, exist_ok=True)

    for ch in range(12):
        # 获取该通道数据（全部点，用于画散点图）
        full_signal = iq_12ch[ch]                     # 复数数组
        complex_points = full_signal                  # 全部点
        # 可选抽样（若点太多，加速绘图）
        if len(complex_points) > max_points:
            idx = np.linspace(0, len(complex_points) - 1, max_points).astype(int)
            complex_points = complex_points[idx]

        # 创建新图形
        fig, ax = plt.subplots(figsize=(6, 6))

        # 画 IQ 散点图
        ax.scatter(complex_points.real, complex_points.imag,
                   s=6, alpha=0.7, edgecolors='none')

        # 调用圆拟合
        xc, yc, R = fit_circle_ransac_iq(iq_12ch[ch, 0:10000])

        # 如果拟合成功，画出圆
        if xc is not None:
            theta = np.linspace(0, 2 * np.pi, 300)
            circle_x = xc + R * np.cos(theta)
            circle_y = yc + R * np.sin(theta)
            ax.plot(circle_x, circle_y, color='red', linewidth=2, label='Fitted circle')
            ax.scatter(xc, yc, color='red', s=30, zorder=5)
            ax.text(xc, yc, f"({xc:.2f}, {yc:.2f})", fontsize=8,
                    ha='center', va='bottom', color='red')
        else:
            ax.text(0.05, 0.95, "Circle fitting failed", transform=ax.transAxes,
                    fontsize=10, color='red', verticalalignment='top')

        # 装饰图形
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"IQ Constellation - CH{ch}")
        ax.set_xlabel("I (real)")
        ax.set_ylabel("Q (imag)")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        if xc is not None:
            ax.legend()

        # 保存三种矢量格式
        base = os.path.join(save_dir, f"iq_ch{ch}")
        plt.savefig(f"{base}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(f"{base}.svg", format='svg', bbox_inches='tight')
        plt.savefig(f"{base}.eps", format='eps', bbox_inches='tight')
        print(f"已保存: {base}.pdf, {base}.svg, {base}.eps")

        plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("用法: python -m radar_project.IQ <bin文件路径> [range_bin]")
        print("示例: python -m radar_project.IQ F:/data_new/adc_data_Raw_xxx.bin 50")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        sys.exit(1)

    range_bin = int(sys.argv[2]) if len(sys.argv) >= 3 else 50

    print(f"读取 bin 文件: {file_path}")
    virtual = read_and_decode(file_path)

    iq_12ch = extract_12ch_iq(virtual, range_bin=range_bin)
    plot_12_iq(iq_12ch)
    print("IQ 星座图已保存到 figuress/ 目录")


if __name__ == "__main__":
    main()