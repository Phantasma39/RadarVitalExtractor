import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
import os
from Judge import judge_channel


# ===== 1. 带通滤波器 =====
def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)


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
        save_root = r"F:\\my_output",
        draw=False
):
    """
    final_signal: (12, Frame) 复数信号
    """

    c = 3e8

    # ===== 1. 相位 =====
    phase = np.angle(final_signal)

    # ===== 2. 相位解缠 =====
    phase_unwrap =np.unwrap(phase, axis=1)

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
            order=filter_order
        )

    # ===== 6. 保存CSV（带时间）=====
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
            if True:
                filename = os.path.join(final_save_dir, f"channel_{ch}_prob_{int(prob*100)}.csv")

                data_to_save = np.column_stack((t, disp[ch]))

                np.savetxt(
                    filename,
                    data_to_save,
                    delimiter=',',
                    header="time(s),displacement(mm)",
                    comments=''
                )

                #print(f"通道{ch}结果较好，已经保存,概率为{prob}")

            else:
                print(f"通道{ch}结果较差，噪声过大或没有对准，概率为{prob}未保存")

    scores = np.array(scores)
    best_idx = np.argmax(scores)
    if draw:
        for ch in range(disp.shape[0]):
            # #===== 7. 绘图（12个通道分开）=====
            plt.figure(figsize=(16, 4))
            plt.plot(disp[ch])
            plt.title(f"Channel{ch}")
            plt.xlabel("Frame")
            plt.ylabel("Displacement (m)")
            plt.grid()
            plt.show()

    return disp[best_idx]
