import numpy as np


def read_bin_complex2x_4lane(bin_file):
    raw = np.fromfile(bin_file, dtype=np.int16)

    # 每4个数一个lane组
    data = raw.reshape(-1, 4)

    lane0 = data[:, 0]
    lane1 = data[:, 1]
    lane2 = data[:, 2]
    lane3 = data[:, 3]

    # 构造复数（已验证正确）
    rx0 = lane0 + 1j * lane2
    rx1 = lane1 + 1j * lane3

    # 合并
    adc = np.stack([rx0, rx1], axis=1)  # [N, 2]

    return adc


def reshape_adc(adc, num_frames, num_chirps, num_rx, num_samples):
    """
    adc: [total_samples, rx]
    输出: [Frame, Chirp, Rx, Sample]
    """

    total = num_frames * num_chirps * num_rx * num_samples

    adc = adc[:total * 2]  # 防止越界

    adc = adc.reshape(num_frames, num_chirps, num_rx, num_samples)

    return adc


def build_virtual_channels(adc):
    """
    输入: [Frame, Chirp, Rx, Sample]
    输出: [12, Frame, Chirp_per_tx, Sample]
    """

    num_frames, num_chirps, num_rx, num_samples = adc.shape

    num_tx = 3
    chirp_per_tx = num_chirps // num_tx

    # 初始化
    virtual = np.zeros(
        (num_tx * num_rx, num_frames, chirp_per_tx, num_samples),
        dtype=np.complex64
    )

    for tx in range(num_tx):
        chirp_idx = np.arange(tx, num_chirps, num_tx)

        for rx in range(num_rx):
            ch = tx * num_rx + rx

            virtual[ch] = adc[:, chirp_idx, rx, :]

    return virtual


def read_and_decode(bin_file):

    num_frames = 6250
    num_chirps = 24
    num_rx = 4
    num_samples = 256

    # ========= 1. 读数据 =========
    adc_raw = read_bin_complex2x_4lane(bin_file)

    #print("Raw shape:", adc_raw.shape)

    # ========= 2. reshape =========
    adc = reshape_adc(
        adc_raw,
        num_frames,
        num_chirps,
        num_rx,
        num_samples
    )

    # print("ADC shape:", adc.shape)

    # ========= 3. 构建12通道 =========
    virtual = build_virtual_channels(adc)

    # print("Virtual shape:", virtual.shape)
    # 应该是 (12, 6250, 8, 256)
    return virtual

