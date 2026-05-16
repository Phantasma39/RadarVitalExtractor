import numpy as np
import matplotlib.pyplot as plt


# ================== 1. 读取bin ==================
def read_bin_complex2x_4lane(bin_file):
    raw = np.fromfile(bin_file, dtype=np.int16)

    data = raw.reshape(-1, 4)

    lane0 = data[:, 0]
    lane1 = data[:, 1]
    lane2 = data[:, 2]
    lane3 = data[:, 3]

    # IQ组合（你已经验证过这个是对的）
    rx0 = lane0 + 1j * lane2
    rx1 = lane1 + 1j * lane3

    adc = np.stack([rx0, rx1], axis=1)  # [N, 2]

    return adc


# ================== 2. reshape ==================
def reshape_adc(adc, num_frames, num_chirps, num_rx, num_samples):
    adc = adc[:num_frames * num_chirps * num_rx * num_samples]

    adc = adc.reshape(num_frames, num_chirps, num_rx, num_samples)

    return adc


# ================== 3. 构建12通道 ==================
def build_virtual_channels(adc):
    num_frames, num_chirps, num_rx, num_samples = adc.shape

    num_tx = 3
    chirp_per_tx = num_chirps // num_tx

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


# ================== 4. Range FFT ==================
def range_fft(data):
    # 去直流
    data = data - np.mean(data, axis=-1, keepdims=True)

    # 加窗
    window = np.hanning(data.shape[-1])
    data = data * window

    # FFT
    fft_out = np.fft.fft(data, axis=-1)

    # 取正频率
    fft_out = fft_out[..., :data.shape[-1] // 2]

    return fft_out


# ================== 5. 选目标bin ==================
def select_target_bin(range_data):
    # 平均 chirp + frame
    power = np.mean(np.abs(range_data), axis=(1, 2))  # [12, 128]

    target_bin = np.argmax(power, axis=1)

    return target_bin


# ================== 6. 微位移 ==================
def compute_displacement(range_data, target_bin):
    c = 3e8
    fc = 77e9

    lam = c / fc

    num_ch, num_frames, num_chirps, _ = range_data.shape

    disp_all = []

    for ch in range(num_ch):
        bin_idx = target_bin[ch]

        # 取该bin
        sig = range_data[ch, :, :, bin_idx]  # [Frame, Chirp]

        # chirp平均（这里才可以平均！）
        sig = np.mean(sig, axis=1)  # [Frame]

        # 相位
        phase = np.angle(sig)

        # 相位展开
        phase = np.unwrap(phase)

        # 位移
        disp = lam / (4 * np.pi) * (phase - phase[0])

        disp_all.append(disp)

    return np.array(disp_all)  # [12, Frame]


# ================== 7. 主函数 ==================
def main():
    # ===== 参数 =====
    bin_file = "DATA/miaoyuxin_004_Raw_0.bin"

    num_frames = 6250
    num_chirps = 24
    num_rx = 4
    num_samples = 256

    # ===== 1. 读数据 =====
    adc_raw = read_bin_complex2x_4lane(bin_file)
    print("Raw:", adc_raw.shape)

    # ===== 2. reshape =====
    adc = reshape_adc(adc_raw, num_frames, num_chirps, num_rx, num_samples)
    print("ADC:", adc.shape)

    # ===== 3. 12通道 =====
    virtual = build_virtual_channels(adc)
    print("Virtual:", virtual.shape)  # (12, 6250, 8, 256)

    # ===== 4. FFT =====
    range_data = range_fft(virtual)
    print("Range FFT:", range_data.shape)  # (12, 6250, 8, 128)

    # ===== 5. 选目标 =====
    target_bin = select_target_bin(range_data)
    print("Target bins:", target_bin)

    # ===== 6. 微位移 =====
    disp = compute_displacement(range_data, target_bin)
    print("Disp shape:", disp.shape)  # (12, 6250)

    # ===== 7. 验证图1：Range Profile =====
    ch = 0
    frame = 0

    profile = np.mean(np.abs(range_data[ch, frame]), axis=0)

    plt.figure()
    plt.plot(20 * np.log10(profile + 1e-6))
    plt.title("Range Profile")
    plt.xlabel("Range Bin")
    plt.ylabel("dB")
    plt.grid()

    # ===== 8. 验证图2：Range-Time =====
    plt.figure()

    rt_map = np.mean(np.abs(range_data[ch]), axis=1)

    plt.imshow(
        20 * np.log10(rt_map + 1e-6),
        aspect='auto',
        cmap='jet'
    )
    plt.colorbar()
    plt.title("Range-Time Map")

    # ===== 9. 微位移图 =====
    plt.figure()

    for i in range(12):
        plt.plot(disp[i], label=f"ch{i}")

    plt.title("Displacement")
    plt.xlabel("Frame")
    plt.ylabel("Meter")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()