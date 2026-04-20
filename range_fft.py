import numpy as np


def range_fft(
    data,
    axis=-1,
    fft_len=None,
    window_type="hann",
    remove_dc=True,
    keep_positive=True,
    output="complex"
):
    """
    Range FFT 模块（工程级封装）

    参数：
    ----------
    data : np.ndarray
        输入数据，例如 [12, Frame, Chirp, Sample]

    axis : int
        FFT维度（默认最后一维）

    fft_len : int or None
        FFT点数（None = 使用原始长度）

    window_type : str
        窗函数类型：
        - "hann"
        - "hamming"
        - None

    remove_dc : bool
        是否去直流

    keep_positive : bool
        是否只保留正频率

    output : str
        输出类型：
        - "complex"
        - "magnitude"
        - "power_db"

    返回：
    ----------
    fft_out : np.ndarray
        与输入结构一致，仅最后一维变为 RangeBin
    """

    x = data.copy()

    # ===== 1. 去直流 =====
    if remove_dc:
        x = x - np.mean(x, axis=axis, keepdims=True)

    # ===== 2. FFT长度 =====
    if fft_len is None:
        fft_len = x.shape[axis]

    # ===== 3. 加窗 =====
    if window_type is not None:
        if window_type == "hann":
            window = np.hanning(x.shape[axis])
        elif window_type == "hamming":
            window = np.hamming(x.shape[axis])
        else:
            raise ValueError("Unsupported window type")

        # reshape window 以匹配维度
        shape = [1] * x.ndim
        shape[axis] = -1
        window = window.reshape(shape)

        x = x * window

    # ===== 4. FFT =====
    fft_out = np.fft.fft(x, n=fft_len, axis=axis)
    fft_out = fft_out / fft_len
    print("fuck you")

    # ===== 5. 保留正频率 =====
    if keep_positive:
        slicer = [slice(None)] * fft_out.ndim
        slicer[axis] = slice(0, fft_len // 2)
        fft_out = fft_out[tuple(slicer)]

    # ===== 6. 输出格式 =====
    if output == "complex":
        return fft_out

    elif output == "magnitude":
        return np.abs(fft_out)

    elif output == "power_db":
        return 20 * np.log10(np.abs(fft_out) + 1e-6)

    else:
        raise ValueError("Unsupported output type")


def final_signal(range_data,target_bins):
    # ===== 输入 =====
    # range_data: (12, Frame, Chirp, RangeBin)
    # target_bins: (12,)

    num_ch, num_frames, num_chirps, _ = range_data.shape

    # ===== 1. 先提取目标bin（保留chirp）=====
    target_signal = np.zeros((num_ch, num_frames, num_chirps), dtype=np.complex64)

    for ch in range(num_ch):
        target_signal[ch] = range_data[ch, :, :, target_bins[ch]]

    # target_signal: (12, Frame, Chirp)

    # ===== 2. chirp平均（核心步骤）=====
    final_signal = np.mean(target_signal, axis=2)

    # final_signal: (12, Frame)

    print(final_signal.shape)
    return final_signal