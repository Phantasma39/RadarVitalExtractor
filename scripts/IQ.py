import numpy as np
import matplotlib.pyplot as plt

from src.utils import read_and_decode
from src.DC_Eliminate import fit_circle_ransac_iq


# =========================
# 1. 读取 bin
# =========================
import numpy as np

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


def plot_12_iq(iq_12ch, max_points=600):

    for ch in range(12):

        x = iq_12ch[ch]
        fit_circle_ransac_iq(iq_12ch[ch,0:10000])
        # # 抽样
        # if len(x) > max_points:
        #     idx = np.linspace(0, len(x)-1, max_points).astype(int)
        #     x = x[idx]
        #
        # plt.figure()
        # plt.scatter(x.real, x.imag, s=6)
        #
        # plt.axhline(0)
        # plt.axvline(0)
        #
        # plt.title(f"IQ Constellation - CH{ch}")
        # plt.axis("equal")
        # plt.grid(True)
        #
        # plt.show()

virtual = read_and_decode(r"F:\data_new\adc_data_Raw_sujunwei_1.bin")

iq_12ch = extract_12ch_iq(virtual, range_bin=50)

plot_12_iq(iq_12ch)