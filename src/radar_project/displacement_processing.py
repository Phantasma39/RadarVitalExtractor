import os

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免无显示环境下弹窗
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, detrend

from radar_project.Judge import judge_channel


# ===== 1. 带通滤波器 =====
def bandpass_filter(data, fs, lowcut, highcut, order=4, axis=-1):
    """
    零相位 Butterworth 带通滤波。

    参数
    ----------
    data : np.ndarray
        输入数据，一维或任意维。
    fs : float
        采样率（Hz）。
    lowcut, highcut : float
        带通上下截止频率（Hz）。
    order : int
        滤波器阶数。
    axis : int
        沿哪一维滤波。默认 -1（最后一维），对 (通道, 帧) 与一维数组都正确。
        旧版本硬编码 axis=1，传入一维数组会直接 IndexError。

    返回
    ----------
    np.ndarray
        与输入同形状的滤波结果。
    """
    nyq = 0.5 * fs

    if not 0 < lowcut < highcut:
        raise ValueError(f"要求 0 < lowcut < highcut，当前 lowcut={lowcut}, highcut={highcut}")
    if highcut >= nyq:
        raise ValueError(
            f"highcut={highcut}Hz 超过奈奎斯特频率 {nyq}Hz（fs={fs}），滤波器设计无意义"
        )

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=axis)


def _resolve_save_root(save_root):
    """
    解析输出根目录。

    旧版本 save_root 默认值是硬编码的 r"D:\\my_output"（只有作者本机存在），
    不显式传参时会在 D 盘建目录甚至直接失败。现在默认落在当前工作目录下的 output/。
    """
    if save_root is None:
        return os.path.join(os.getcwd(), "output")
    return save_root


# ===== 2. 微位移计算主函数 =====
def compute_displacement(
        final_signal,
        fc=77e9,
        frame_rate=250,  # Hz (4ms → 250Hz)
        do_detrend=True,
        do_filter=True,
        lowcut=0.1,
        highcut=2.0,
        filter_order=4,
        save_csv=True,
        save_dir="output",
        save_root=None,
        draw=False,
        svg_dir=None,
        verbose=True
):
    """
    final_signal: (12, Frame) 复数信号
    """

    c = 3e8

    # ===== 1. 相位 =====
    phase = np.angle(final_signal)

    # ===== 2. 相位解缠 =====
    phase_unwrap = np.unwrap(phase, axis=1)

    # ===== 3. 转微位移 =====
    disp = (c / (4 * np.pi * fc)) * (phase_unwrap - phase_unwrap[:, [0]])

    # ===== 4. 去趋势 =====
    if do_detrend:
        disp = detrend(disp, axis=1)

    # ===== 5. 滤波 =====
    if do_filter:
        disp = bandpass_filter(
            disp,
            fs=frame_rate,
            lowcut=lowcut,
            highcut=highcut,
            order=filter_order,
            axis=1
        )

    save_root = _resolve_save_root(save_root)

    # ===== 6. 保存CSV（带时间）=====
    scores = None
    if save_csv:

        # ===== 在总目录下再建子文件夹 =====
        final_save_dir = os.path.join(save_root, save_dir)

        os.makedirs(final_save_dir, exist_ok=True)

        num_frames = disp.shape[1]
        t = np.arange(num_frames) / frame_rate

        scores = []

        for ch in range(disp.shape[0]):
            is_good, prob = judge_channel(disp[ch], frame_rate)
            scores.append(prob)

            filename = os.path.join(final_save_dir, f"channel_{ch}_prob_{int(prob * 100)}.csv")

            data_to_save = np.column_stack((t, disp[ch]))
            if verbose:
                print(filename)
            np.savetxt(
                filename,
                data_to_save,
                delimiter=',',
                header="time(s),displacement(m)",
                comments=''
            )

        scores = np.array(scores)

    # ===== 7. 画图 =====
    if draw:
        # 计算所有通道的统一纵坐标范围
        all_disp = disp.flatten()
        y_min = np.min(all_disp)
        y_max = np.max(all_disp)
        margin = 0.05 * (y_max - y_min) if y_max != y_min else 0.1
        y_min -= margin
        y_max += margin

        # 保存到 save_root 下的 SVG 文件夹（例如 output/SVG_2）
        svg_root = svg_dir if svg_dir is not None else os.path.join(save_root, "SVG_2")
        os.makedirs(svg_root, exist_ok=True)

        for ch in range(disp.shape[0]):
            fig = plt.figure(figsize=(16, 4))
            plt.plot(disp[ch])
            plt.title(f"Channel {ch} - Displacement")
            plt.xlabel("Frame")
            plt.ylabel("Displacement (m)")
            plt.grid(True)
            plt.ylim(y_min, y_max)

            svg_path = os.path.join(svg_root, f"displacement_ch{ch}.svg")
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            if verbose:
                print(f"已保存: {svg_path}")
            plt.close(fig)

    # ===== 8. 返回最佳通道的微位移 =====
    # 说明：这里返回的是 judge_channel 打分最高的那一维，而不是全通道矩阵。
    # 旧版本在 save_csv=False 时 scores/best_idx 未定义会直接 UnboundLocalError。
    if scores is not None and len(scores) > 0:
        best_idx = int(np.argmax(scores))
        return disp[best_idx]

    # 未保存 CSV 时没有打分依据，退回第 0 通道（行为可预期，不再抛异常）
    return disp[0]
