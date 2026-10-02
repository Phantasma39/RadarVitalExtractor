import numpy as np

# ===================== 支持的 I/Q 排布方案 =====================
# 每个方案定义: (该方案一个采样点占多少个 int16, 实部在组内的下标, 虚部在组内的下标)
#
#   IIQQ     : [I0 I1 Q0 Q1] 或 [I0 I1 Q0 Q1 I2 I3 Q2 Q3]
#              两条 lane 各承载 2 个 RX，每 lane 内先 2 个实部、再 2 个虚部。
#              IWR1843 + Complex + lane1/lane2 实测为该方案。
# 下标含义: 实部取 [0,1,4,5]，虚部取 [2,3,6,7]。
#
#   IQIQ     : [I0 Q0 I1 Q1] 或 [I0 Q0 I1 Q1 I2 Q2 I3 Q3]
#              每路 RX 的 I/Q 相邻，实部取 [0,2,4,6]，虚部取 [1,3,5,7]。
#
#   IIIIQQQQ : [I0 I1 I2 I3 Q0 Q1 Q2 Q3]（先所有实部、再所有虚部）
#              实部取 [0,1,2,3]，虚部取 [4,5,6,7]。
#
#   REAL     : 实数采样，每个 int16 就是一个采样点，无虚部。
LAYOUTS = {
    "IIQQ":     [(0, 1, 4, 5), (2, 3, 6, 7)],
    "IQIQ":     [(0, 2, 4, 6), (1, 3, 5, 7)],
    "IIIIQQQQ": [(0, 1, 2, 3), (4, 5, 6, 7)],
}

LAYOUT_CHOICES = ["IIQQ", "IQIQ", "IIIIQQQQ"]
LAYOUT_DEFAULT = "IIQQ"


def _layout_plan(layout, num_rx):
    """
    把排布方案 + RX 数解析成 (每组 int16 数, 实部下标, 虚部下标)。

    IIQQ 支持 2/4 RX（2 RX 时每个采样点 4 个 int16，只用 lane1）；
    IQIQ / IIIIQQQQ 只支持 4 RX。
    """
    if layout == "REAL":
        return num_rx, tuple(range(num_rx)), None

    if layout not in LAYOUTS:
        raise ValueError(
            f"不支持的排布方案: {layout!r}，可选 {LAYOUT_CHOICES}"
        )

    if layout == "IIQQ":
        # 2 RX: [I0 I1 Q0 Q1] -> 实部 [0,1]，虚部 [2,3]
        # 4 RX: [I0 I1 Q0 Q1 I2 I3 Q2 Q3] -> 实部 [0,1,4,5]，虚部 [2,3,6,7]
        if num_rx == 2:
            return 4, (0, 1), (2, 3)
        if num_rx == 4:
            return 8, (0, 1, 4, 5), (2, 3, 6, 7)
        raise ValueError(f"排布 IIQQ 只支持 num_rx=2 或 4，收到 {num_rx}")

    if num_rx != 4:
        raise ValueError(f"排布 {layout} 只支持 num_rx=4，收到 {num_rx}")

    i_idx, q_idx = LAYOUTS[layout]
    return 8, i_idx, q_idx


def read_bin_complex2x_4lane(bin_file, num_rx=4, max_samples=None,
                             layout=LAYOUT_DEFAULT):
    """
    解析 DCA1000 采集的 int16 原始 ADC 文件，还原出各接收通道的复数采样。

    数据格式（IWR1843 + DCA1000 + Complex 复数模式 + lane1/lane2 两条 LVDS lane）:

        每个采样点占 8 个 int16，排列为::

            [ I0 I1 Q0 Q1 I2 I3 Q2 Q3 ]
              └─ lane1 ──┘ └─ lane2 ──┘

        即两条 lane 各承载 2 个 RX 的复数数据，先 2 个实部、再 2 个虚部。
        TX 侧仍是 3 发 TDM，故 3TX x 4RX = 12 个虚拟通道由
        build_virtual_channels() 完成。

    参数
    ----------
    bin_file : str
        .bin 文件路径。
    num_rx : int
        解出几个接收通道，只能是 2 或 4。
        - 4: 每个采样点 8 个 int16
        - 2: 每个采样点 4 个 int16（只用 lane1）
    max_samples : int or None
        只解析前这么多个采样点，用于快速预览大文件（None = 全部）。
    layout : str
        I/Q 排布方案，取值见 LAYOUT_CHOICES：
        - "IIQQ"     先 2 实部后 2 虚部（本雷达实测方案，默认）
        - "IQIQ"     每路 RX 的 I/Q 相邻
        - "IIIIQQQQ" 先所有实部、再所有虚部
        若排布选错，I 与 Q 会来自不同接收通道，解出的相位无意义。
        不确定时可以看 IQ 星座图：方案正确时轨迹应接近圆。

    返回
    ----------
    np.ndarray
        (N, num_rx) 的复数数组，N 为采样点总数。

    说明
    ----------
    旧实现硬编码 4 个 int16 一组且只取前两个 lane，等于把
    RX0/RX2 交替混进同一个流、RX1/RX3 混进另一个流，
    再交给 reshape_adc 按 4 个 RX 切分，导致第 3、4 路 RX 样本错位。
    """
    if num_rx not in (2, 4):
        raise ValueError(f"num_rx 只支持 2 或 4，收到 {num_rx}")

    group, i_idx, q_idx = _layout_plan(layout, num_rx)

    if max_samples is None:
        raw = np.fromfile(bin_file, dtype=np.int16)
    else:
        # 只读文件前 max_samples 个采样点对应的字节，避免整文件载入
        count = int(max_samples) * group
        raw = np.fromfile(bin_file, dtype=np.int16, count=count)

    if raw.size < group:
        raise ValueError(
            f"文件过短，无法解析: {bin_file} (仅 {raw.size} 个 int16，"
            f"每采样点需要 {group} 个)"
        )

    data = raw[: raw.size // group * group].reshape(-1, group)

    if layout == "REAL":
        # 实数采样：每个 int16 直接作为一个采样点，无虚部
        return data.astype(np.complex64)          # [N, num_rx]

    # 显式按下标取实部/虚部：不同方案里 I 与 Q 的相对位置不同，
    # 不能简单 reshape 成 (...,2)（IIQQ 下 I 与 Q 相隔 2 个位置）。
    # 原始数据是 int16，用 complex128 存储没有意义且 4 倍占内存，这里用 complex64。
    i = data[:, i_idx]
    q = data[:, q_idx]
    return (i + 1j * q).astype(np.complex64)      # [N, num_rx]


def reshape_adc(adc, num_frames, num_chirps, num_rx, num_samples):
    """
    adc: [N, num_rx] 复数数组（每个复数算 1 个元素）
    输出: [Frame, Chirp, Rx, Sample]

    关键点（旧实现出错的地方）
    --------------------------
    不能把 [N, num_rx] 直接 reshape 成 [F, C, num_rx, S]。
    因为内存顺序是「所有采样点的 RX0 RX1 RX2 RX3 | 下一个采样点的 RX0 RX1 RX2 RX3 | ...」，
    直接重切会把**相邻采样点的同一位**凑成一个采样点，
    使 RX1/RX2/RX3 的样本整体错位（RX0 因为偏移为 0 而凑巧正确，
    所以问题不容易被发现）。

    正确做法: 先按 [采样点总数, num_rx] 分组（每个采样点的各 RX），
    再把采样点总数展开成 [Frame, Chirp, Sample]，最后换轴到 [Frame, Chirp, Rx, Sample]。
    即 reshape 必须写成 (F*C*S, num_rx) -> (F, C, S, num_rx)，
    而不是 (F, C, num_rx, S)。

    同时校验 rx 维是否与传入的 num_rx 一致，避免上游解错通道数时静默错位。
    """

    if adc.ndim != 2 or adc.shape[1] != num_rx:
        raise ValueError(
            f"adc 形状 {adc.shape} 与 num_rx={num_rx} 不一致。"
            f"请检查 read_bin_complex2x_4lane(..., num_rx=...) 的取值。"
        )

    n_samples_total = num_frames * num_chirps * num_samples
    total = n_samples_total * num_rx

    n = adc.shape[0] * adc.shape[1]
    if n < total:
        raise ValueError(
            f"数据长度不足：需要 {total} 个复采样点，实际只有 {n} 个。"
            f"请检查 num_frames({num_frames}) / num_chirps({num_chirps}) / "
            f"num_rx({num_rx}) / num_samples({num_samples}) 是否与采集配置一致，"
            f"或 bin 文件是否被截断。"
        )

    adc = adc[:total]

    # step 1: 按 [采样点总数, num_rx] 分组 —— 每个采样点的各接收通道
    adc = adc.reshape(n_samples_total, num_rx)
    # step 2: 再把采样点总数展开成 [Frame, Chirp, Sample]
    adc = adc.reshape(num_frames, num_chirps, num_samples, num_rx)
    # step 3: 换到 [Frame, Chirp, Rx, Sample]
    adc = np.ascontiguousarray(adc.transpose(0, 1, 3, 2))

    return adc


def build_virtual_channels(adc, num_tx=3):
    """
    由物理接收通道合成 MIMO 虚拟通道。

    TDM-MIMO: num_tx 个发射天线轮流发射，每个 TX 对应 num_chirps/num_tx 个 chirp。
    因此虚拟通道数 = num_tx * num_rx（本雷达 3TX x 4RX = 12）。

    输入: [Frame, Chirp, Rx, Sample]
    输出: [num_tx * num_rx, Frame, Chirp_per_tx, Sample]
    """

    num_frames, num_chirps, num_rx, num_samples = adc.shape

    if num_chirps % num_tx != 0:
        raise ValueError(
            f"num_chirps={num_chirps} 不能被 TX 数 {num_tx} 整除，"
            f"无法按 TDM 结构拆分到各个发射天线。"
            f"请检查 num_chirps 与使能的 TX 数是否匹配。"
        )

    chirp_per_tx = num_chirps // num_tx

    # 初始化
    virtual = np.zeros(
        (num_tx * num_rx, num_frames, chirp_per_tx, num_samples),
        dtype=np.complex64
    )

    for tx in range(num_tx):
        # TDM: 第 tx 个发射天线对应 chirp 下标 tx, tx+num_tx, tx+2*num_tx, ...
        chirp_idx = np.arange(tx, num_chirps, num_tx)

        for rx in range(num_rx):
            ch = tx * num_rx + rx

            virtual[ch] = adc[:, chirp_idx, rx, :]

    return virtual


def read_and_decode(bin_file,
                    num_frames=6250,
                    num_chirps=24,
                    num_rx=4,
                    num_samples=256,
                    num_tx=3,
                    max_samples=None,
                    layout=LAYOUT_DEFAULT):
    """
    读取并解码一个 .bin 采集文件，输出 MIMO 虚拟通道。

    参数都可以显式覆盖——采集配置变化时不必再改源码。

    max_samples : int or None
        最多读取多少个**采样点**（不是帧数！）。用于只读文件开头一段。
        注意它必须与 num_frames 配套：若只截断读取却不降低 num_frames，
        reshape 会因长度不足而报错。GUI 的「快速预览」就是这样成对设置的：
            num_frames  = min(num_frames, preview_frames)
            max_samples = num_frames * num_chirps * num_samples
    layout : str
        原始数据的 I/Q 排布，见 LAYOUT_CHOICES。

    返回
    ----------
    virtual : np.ndarray
        (num_tx * num_rx, num_frames, num_chirps // num_tx, num_samples)
        默认配置下为 (12, 6250, 8, 256)
    """

    # ========= 1. 读数据 =========
    adc_raw = read_bin_complex2x_4lane(bin_file, num_rx=num_rx,
                                       max_samples=max_samples,
                                       layout=layout)

    # ========= 2. reshape =========
    adc = reshape_adc(
        adc_raw,
        num_frames,
        num_chirps,
        num_rx,
        num_samples
    )

    # ========= 3. 构建虚拟通道 =========
    virtual = build_virtual_channels(adc, num_tx=num_tx)

    return virtual

