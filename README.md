# RadarVitalExtractor

基于 TI-IWR1843 毫米波雷达的**生命体征信号处理项目**。

**当前版本：v0.3.0** · [更新记录](#更新记录)

> 📖 **完整说明文档（97 页 PDF）**：`docs/latex/main.pdf`
>
> 涵盖雷达参数与 FMCW 原理、原始数据格式（IIQQ）的来龙去脉、
> 处理流水线每一步的数学推导与代码实现、图形界面、多进程批处理、
> 断点续处理、exe 打包、故障排查、API 参考与缺陷档案（19 个 bug）。
> LaTeX 源码在 `docs/latex/`，用 `latexmk -xelatex main.tex` 编译。
> 该目录未纳入版本管理（含数 MB 的 PDF 产物）。

---

## 项目简介
本项目为北京理工大学毫米波雷达测量大创项目，旨在利用 TI-IWR1843 雷达传感器，实现对脉搏波信号的采集与分析。

> **项目状态：开发中**
>
> 目前已完成基础数据读取与预处理模块，后续功能仍在持续迭代，欢迎提出建议与反馈。

---

## 快速开始

### 0. 图形界面（最省事）

```bash
python scripts/gui.py
```

界面里可以可视化地选择 `.bin` 文件、设置输出目录与结果命名方式、调处理参数，
处理完直接在界面内看到 **12 个通道的位移曲线**。

**每张图都带 matplotlib 标准导航栏**（图下方一小条，共 6 个按钮）：

| 按钮 | 作用 |
|---|---|
| **Home** | **返回原始视野**（放大/平移后点它复位） |
| Back / Forward | 在历史视野之间前后切换 |
| Pan | 按住拖动平移 |
| **Zoom** | **拖出一个矩形框，松开即放大框中区域** |
| Save | 把当前图保存成图片 |

竖排视图里每张图各有自己的一条工具栏（点哪张的 Zoom 就放大哪张）；
网格视图 / IQ 星座图 / 距离谱用一条共用工具栏，Home 会复位该图全部子图。

**默认视图是「1-12通道竖排」**：ch0 → ch11 从上到下依次排列，整列**可滚动**，
每个通道一张占满宽度的独立图（比挤在 3×4 网格里清楚得多）。
另有「12通道网格」「单通道放大」「IQ 星座图」「距离谱」四个视图可切换。

**批处理**（界面左侧「批处理（整个文件夹）」）：

- 选一个文件夹 → 选择是否包含子文件夹、扩展名、**并行进程数**
- 每个 `.bin` 的输出写到「输出根目录/<文件名>/」下，12 个通道各一个 CSV
- 命名规则沿用第 1 节设置的前缀/后缀/分隔符/表头/是否带概率
- 可随时「中止」；单个文件失败不影响其他文件
- **每个文件处理完会立即释放内存**，不会因为文件多而累积

**断点续处理**（勾选「断点续处理」）：

跑到一半中断了、或者崩了，重跑时不必从头再来——勾上这个选项，
程序会先扫描输出目录，**只处理还没做完的文件**。

- 点「**扫描**」按钮可以**先看看**状态，不会真正处理：
  会列出每个文件是「已完成」还是「待处理」，以及待处理的原因
- 「已完成」的判定标准（三条同时满足）：
  1. 输出子目录存在
  2. 里面有 **12 个通道的 CSV**（一个都不能少）
  3. 文件名**符合当前的命名规则**（前缀 / 后缀 / 扩展名一致）
- 第 3 条很重要：如果你改了前缀再跑，旧结果**不会**被误判成"已完成"，
  避免得到一个命名规则混杂的结果目录
- 只做了一半的文件（比如只写了 4 个通道）会被识别为**待处理**，重算补全
- 不勾选时就是原来的行为：全部重算

> 实测：续处理产出的结果与全新处理**逐位一致**
> （CSV 最大数值差为 0），不会因为跳过了前面的文件而引入差异。

**多进程并行**：批处理默认开 `min(4, CPU核数-1)` 个进程，每个文件一个进程。

> 为什么用**多进程**而不是多线程：处理耗时几乎全在 RANSAC 和 FFT 上，是纯 CPU 计算。
> Python 的多线程受 GIL 限制，同一时刻只有一个线程能执行字节码，线程池基本不会提速。

- 进程数可在界面填（1 = 串行）
- 进程越多越快，但内存占用成倍上升（每个进程处理时会占一两 GB）
- 完成后会显示「耗时 / 进程数 / 平均每个文件耗时」
- 如果运行环境不允许创建子进程，会自动退回单进程串行，功能仍然可用

**内存说明**：一次完整处理（6250 帧 × 24 chirp × 4 RX）会生成很大的中间数组，
所以代码里做了几件事避免爆内存：

- 输入先降到 `complex64` 再算 FFT，中间量减半
- `range_fft` 之后立刻释放不再需要的解码结果
- 距离谱只保留 3 条小曲线（均值/最小/最大），**不保留完整的 range_data**
- 结果字典总大小约 2 MB，处理后内存可以回落
- 批处理在 `finally` 里显式释放每个文件的结果并 `gc.collect()`

**目标选择（多反射体场景）**：默认「各自选峰」是每个通道各自选最强峰。
如果场景里有多个反射体，不同通道可能选到不同距离（实测出现过
「前 4 个通道 0.46m、后面 8 个通道 1.10m」的分裂）。此时：

- 在 **「目标距离范围」** 里按**实际距离（cm）**填入你目标所在的范围
  （例如 `20 ~ 40`），程序会**自动换算成对应的 bin**，并在下方实时显示
  「每 bin = x cm / 下限 -> bin N / 上限 -> bin M / 共 K 个 bin」
- 把「选峰方式」改成 **「估计范围」**，会在该距离范围内估一个共同点，12 通道统一

> 距离换算公式：`每 bin = fs_adc · c / (2 · slope · fft_len)`
> 本雷达默认参数下为 **1.83 cm/bin**。所以输入距离比输入 bin 号直观得多，
> 不用自己算。留空 = 不限制，全部 bin 参与选峰。

**距离标定实时显示**：改动 `slope` / `fs_adc` / `FFT 点数` / `adc 采样` 时，
界面会立刻算出并显示每 bin 距离、分辨率、最大不模糊距离、有效 bin 上限——
**距离算得对不对，可以直接对着这几行核对**。

> 距离公式：`R(bin) = bin · fs_adc · c / (2 · slope · fft_len)`
> 分辨率：`ΔR = c / (2B)`，其中 `B = slope · N_adc / fs_adc`
>
> 若感觉距离偏差较大，先核对 **`fs_adc` 是否与实际 config 一致**：
> `fs_adc = adc_samples / Ramp_End_Time`。填错会让所有距离整体差同一倍数
> （bin 位置不变，只是换算比例错）：

**参数预设**（界面左上「0. 参数预设」）：

内置了 4 个预设，点一下就把所有参数填好：

| 预设 | 说明 |
|---|---|
| **★ 我的配置 (IWR1843 4RX Complex IIQQ)** | 当前在用的采集配置，IIQQ 排布 |
| IWR1843 4RX - 只看心跳 0.8-2Hz | 带通改为脉搏波频带 |
| IWR1843 4RX - 只看呼吸 0.1-0.5Hz | 带通改为呼吸频带 |
| 2RX Complex (只用 lane1) | 只开了 2 个接收通道时用 |

自己调好参数后点 **「另存为预设」** 就能存下来，保存在 `config/presets.json`
（该目录已加入 `.gitignore`，属于个人配置）。同名用户预设会覆盖内置预设。

> 左侧有一块 **「收发天线处理（预留区）」**，后续要加 TX/RX 天线相关的处理时，
> 控件直接加进 `self.antenna_frame` 即可，不用改别的结构。
>
> 处理大文件时建议先勾选 **「快速预览」**（只取前 N 帧、并降低 RANSAC 迭代），
> 秒级出图；确认参数无误后再取消勾选跑完整数据（6250 帧约需几分钟，RANSAC 是瓶颈）。

**I/Q 排布可选**（界面「2. 处理参数」里的 `I/Q 排布`）：
原始 int16 数据里 I 和 Q 怎么排，取决于雷达的 ADC 模式与 LVDS lane 配置，
排错了会导致 I 与 Q 来自不同接收通道、相位完全无意义。界面支持三种：

| 方案 | 排列 | 适用 |
|---|---|---|
| **`IIQQ`** | `[I0 I1 Q0 Q1 I2 I3 Q2 Q3]` | **本雷达实测正确**（默认），支持 2/4 RX |
| `IQIQ` | `[I0 Q0 I1 Q1 I2 Q2 I3 Q3]` | 每路 RX 的 I/Q 相邻 |
| `IIIIQQQQ` | `[I0 I1 I2 I3 Q0 Q1 Q2 Q3]` | 先所有实部、再所有虚部 |

判别方法：切到「IQ 星座图」视图，**方案正确时轨迹接近圆**。实测本雷达：

```
IIQQ      |corr(I,Q)| = 0.023   <- 正确（复数信号 I/Q 本就无关）
IQIQ      |corr(I,Q)| = 0.992   <- 错误
IIIIQQQQ  |corr(I,Q)| = 0.975   <- 错误
```

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/Phantasma39/RadarVitalExtractor.git
cd RadarVitalExtractor

# 安装包（会自动安装所有依赖）
pip install -e .
```

> **关于 tqdm**：只有**命令行版**的批量脚本 `scripts/Batch_process.py`
> 需要 `tqdm`（显示进度条）。它在 `requirements.txt` 里，装包时会一起装上。
>
> 如果没装 tqdm 又想批量处理，用图形界面或 Python 调用即可，
> 它们**不依赖 tqdm**：
>
> ```bash
> python scripts/gui.py                       # 图形界面批处理（推荐）
> ```
>
> ```python
> from radar_project.gui import process_folder
> process_folder("data", "output", file_glob="*.bin")   # 多进程 + 断点续处理
> ```

### 1b. 打包成 exe（给别人用，对方不用装 Python）

```powershell
# 先装打包工具
pip install pyinstaller

# 打包（目录版，启动快，推荐）
python scripts/build_exe.py

# 首次打包建议加 --clean 清掉旧产物
python scripts/build_exe.py --clean

# 单文件版（一个 exe，启动稍慢）
python scripts/build_exe.py --onefile
```

产物：

| 模式 | 位置 |
|---|---|
| 目录版（默认） | `dist/RadarVitalExtractor/RadarVitalExtractor.exe` |
| 单文件版 | `dist/RadarVitalExtractor.exe` |

**打包后运行报错怎么办**

打包用的是 `packaging/radar_gui.spec`（而不是纯命令行参数），这是有原因的：
`--collect-submodules` **只收集 Python 模块，不会带上编译扩展（`.pyd`）**。
scipy 有大量"只有 `.pyd`、没有对应 `.py`"的模块，漏掉就会在运行时崩：

```
ModuleNotFoundError: No module named 'scipy._cyutility'
ImportError: The `scipy` install you are using seems to be broken
```

本 spec 用 `collect_all()` + `collect_dynamic_libs()` 把 scipy / sklearn /
matplotlib 的编译扩展一起收进来，可以避免这个问题。

**如果你的 Python 来自 conda**（`sys.base_prefix` 指向 anaconda 目录，
在其上又建了 venv），打包还会遇到两个额外问题，本 spec 也已处理：

| 报错 | 原因 | spec 里的处理 |
|---|---|---|
| `ImportError: DLL load failed while importing _ctypes` | conda 把 `ffi.dll`/`liblzma.dll`/openssl/sqlite3 等放在 `Library\bin\`，PyInstaller 找不到 | 从 `_ctypes.pyd` 反推 conda 目录，显式加入 8 个 DLL |
| `version conflict for package "Tcl": have 8.6.5, need 8.6.15` | 自动收集的 Tcl 版本与环境不一致 | 把 `tcl86t.dll`/`tk86t.dll` **替换**为 conda 版本 |

**如果 exe 双击后「闪一下就退出」、没有任何提示**：

```powershell
# 第一步：用 --console 打包，让报错显示出来
python scripts/build_exe.py --clean --console

# 第二步：若仍无输出，开启启动日志（默认关闭，不影响正常使用）
$env:RADAR_STARTUP_LOG=1
dist\RadarVitalExtractor\RadarVitalExtractor.exe
type dist\RadarVitalExtractor\_startup.log
```

启动日志会记录：冻结状态、每个模块的导入完成情况、字体选择、
窗口创建、进入 mainloop，以及**任何逃逸到顶层的异常和完整 traceback**。

如果还遇到别的 `ImportError` / `ModuleNotFoundError`，同样用 `--console` 即可。

其他常见问题：

- **删不掉 `dist/`**：说明 exe 还开着（文件被占用），先把程序关掉
- **打包慢**：PyInstaller 需要分析 numpy/scipy/sklearn 的全部依赖，首次几分钟正常
- **体积大**：目录版约 250 MB，因为 numpy/scipy/sklearn/matplotlib 都打进去了；
  单文件版更小一些但启动更慢

**其他要点**：

- 模型文件（`models/rf_model.pkl`、`threshold.pkl`）会被打进包里，
  靠 `src/radar_project/resources.py` 在 `sys._MEIPASS` 下定位，打包后照样能加载
- 打包完成后会在产物旁自动建一个 **`data/`** 目录，把 `.bin` 丢进去，
  程序启动时下拉框就会列出来
- 界面里的「另存为预设」写到 **exe 同级** 的 `config/presets.json`
  （不会写进只读的解包目录）
- 分发时把整个文件夹（目录版）或单个 exe（单文件版）拷给别人即可

> **不想安装也可以跑**：`scripts/` 下的脚本会通过 `scripts/_bootstrap.py` 自动把
> `src/` 加入 `sys.path`，所以 `python scripts/main.py <bin路径>` 在没执行
> `pip install -e .` 的环境里同样能直接运行。
> 但 `python -m radar_project.xxx` 这种模块方式**必须先安装**，否则会报
> `No module named 'radar_project'`。

### 2. 运行

**只记一个入口就够了：**

```bash
python scripts/run.py data/TEST.bin                    # 处理单个 bin
python scripts/run.py data/TEST.bin output_TEST        # 指定输出目录
python scripts/run.py data/TEST.bin output_TEST 0.8 2.0  # 只看心跳频带(0.8-2Hz)
python scripts/run.py                                  # 不带参数会列出 data/ 下的 bin
```

输出到 `<输出目录>/output_<文件名>/channel_<通道>_prob_<概率>.csv`。

**想看 12 通道的图**（位移总览 / IQ 星座图 / 距离谱，输出 SVG）：

```bash
python scripts/run_pipeline_grid.py                    # 默认 data/TEST.bin -> output_TEST/
python scripts/run_pipeline_grid.py data/TEST.bin output_TEST
```

**其他入口：**

```bash
# 批量处理整个文件夹（命令行版，需要 tqdm）
python scripts/Batch_process.py <数据文件夹> [输出目录]

# IQ 星座图分析
python scripts/IQ.py <bin文件路径> [range_bin]

# 画图（含 draw=True，输出位移 SVG）
python -m radar_project.Draw <bin文件路径> [输出目录] [fft_len]

# 端到端独立测试（自带一份实现，可用于交叉验证）
python scripts/TEST.py <bin文件路径>

# 模块方式（需要先 pip install -e .）
python -m radar_project.main <bin文件路径> [输出目录]
```

**关于安装**：`scripts/` 下的脚本都会 `import _bootstrap`，它会自动把 `src/` 加入
`sys.path`，所以**不装包也能直接跑**。但 `python -m radar_project.xxx` 这种模块方式
**必须先 `pip install -e .`**，否则会报 `No module named 'radar_project'`。

**脚本的两种用法**：`scripts/main.py`、`scripts/Batch_process.py`、`scripts/rename.py`
同时支持"带参数走通用入口"和"不带参数用脚本顶部硬编码的默认路径"，方便在 IDE 里直接 F5。

---

## 项目结构

```
RADAR/
├── data/                       # 原始采集数据（.bin，不入库，体积大）
├── docs/                       # 文档与手册
│   ├── 我的论文部分.pdf
│   └── mmwaveSensing-FMCW-offlineviewing_0.pdf
├── models/                     # 预训练模型
│   ├── rf_model.pkl            # 随机森林通道质量判断模型
│   └── threshold.pkl           # 判断阈值
├── src/radar_project/          # 核心包
│   ├── __init__.py             # 公共 API 导出
│   ├── gui.py                  # ★ 可视化界面（Tkinter）
│   ├── resources.py            # 资源路径定位（源码/exe 通用）
│   ├── main.py                 # 主入口：process_single_bin()
│   ├── utils.py                # bin 文件读取与解码（IQ 配对 / reshape / 12 虚拟通道）
│   ├── range_fft.py            # Range FFT 处理
│   ├── DC_Eliminate.py         # RANSAC 圆拟合 DC 消除
│   ├── displacement_processing.py # 微位移计算
│   ├── Judge.py                # 通道质量判断（机器学习）
│   ├── Draw.py                 # 画图工具
│   ├── IQ.py                   # IQ 星座图分析
│   ├── select_file.py          # 按概率筛选文件
│   ├── Select.py               # 模型训练脚本
│   ├── rename.py               # 批量重命名
│   └── TEST.py                 # 测试代码
├── scripts/                    # 运行脚本
│   ├── gui.py                  # ★ 启动可视化界面
│   ├── build_exe.py            # ★ 打包成 exe
│   ├── run.py                  # ★ 命令行入口：处理单个 bin
│   ├── run_pipeline_grid.py    # 12 通道图 + 汇总表
│   ├── main.py                 # 单文件（脚本版）
│   ├── Batch_process.py        # 批量处理
│   ├── IQ.py                   # IQ 星座图
│   ├── TEST.py                 # 端到端独立测试
│   ├── rename.py               # 批量重命名
│   ├── Delete.py               # 本地工具：按概率清理结果（默认试运行，不入库）
│   └── _bootstrap.py           # 公共引导：sys.path + 控制台编码
├── output_*/                   # 处理结果（不入库）
├── pyproject.toml
├── requirements.txt
└── README.md
```

> 注意：`python -m radar_project.TEST` 之类的模块方式不可用，`TEST.py` / `Delete.py`
> 只在 `scripts/` 下提供脚本入口。

---

## 雷达信号测量参数

> 考虑到本雷达信号处理目前仅适用于我的雷达参数，以后将会考虑对所有参数雷达信号进行处理

- 3TX4RX，共计 12 个通道
- adc_samples: 256
- num_chirps: 24（3 TX × 8 chirp，即每帧 8 个 chirp loop）
- frequency_slope: 80e12
- sample_rate: 1e7
- fc: 77e9
- frame_rate: 250
- ADC 模式: Complex（复数）
- LVDS lane: lane1 + lane2（两条 lane 承载 4 个 RX）

### 原始 bin 数据格式（重要）

DCA1000 采集的 int16 数据，**每个采样点占 8 个 int16**：

```
[ I0 I1 Q0 Q1 I2 I3 Q2 Q3 ]
  └─ lane1 ──┘ └─ lane2 ──┘
```

- 两条 lane 各承载 **2 个 RX** 的复数数据
- **每 lane 内先 2 个实部、再 2 个虚部**（即 IIQQ，不是 IQIQ）
- 正确配对：`RX0=I0+jQ0`、`RX1=I1+jQ1`、`RX2=I2+jQ2`、`RX3=I3+jQ3`
- 内存推进顺序：同一采样点内跨 4 个 RX 连续 8 个 int16，然后才进入下一个采样点

因此 `reshape` 必须写成 `(F*C*S, num_rx)` 而不是 `(F, C, num_rx, S)`——
后者会把**相邻采样点的同一位**凑成一个采样点，导致 RX1/RX2/RX3 样本整体错位
（RX0 因偏移为 0 凑巧正确，所以这个 bug 很难被发现）。

> 排布方案在代码里是可配置的（`read_bin_complex2x_4lane(..., layout="IIQQ")`，
> 也可在图形界面的「I/Q 排布」下拉里切换），默认 `IIQQ`。见上文排布对照表。
>
> 参考：TI `readDCA1000.m` 的复数配对公式 `adcData(i) + j*adcData(i+2)`，
> 以及 SWRA581B 中 "complex data ... beginning with the real part" 的说明。

---

## 当前已实现功能

- ✅ 雷达原始数据（ADC/IQ）读取与解析（支持 IIQQ / IQIQ / IIIIQQQQ 三种排布）
- ✅ 基础 DC 消除（RANSAC 圆拟合）
- ✅ Range FFT 与目标 bin 选取（支持「各自选峰」「估计范围」）
- ✅ 微位移计算（相位解缠 + 带通滤波）
- ✅ 通道质量自动判断（随机森林模型）
- ✅ 图形界面（12 通道竖排显示、图表可框选放大）
- ✅ 多进程并行批处理 + **断点续处理**
- ✅ 打包成 Windows exe（含 scipy 编译扩展的正确收集）

---

## 更新记录

### v0.3.0

**新增**

- 批处理**断点续处理**：扫描输出目录，跳过已完成的文件
  - 「已完成」判定：12 个通道 CSV 齐全 **且** 命名规则与当前设置一致
  - 界面新增「断点续处理」勾选框与「扫描」按钮（扫描只报告，不处理）
  - 实测续处理结果与全新处理**逐位一致**（最大数值差 0）

**修正**

- `read_and_decode` 的 `max_samples` 文档说明：它是「最多读取的采样点数」
  而非帧数，且必须与 `num_frames` 配套降低，否则 reshape 会报长度不足
- **修复打包后的 exe 无法启动**（三个独立问题，详见打包章节的排错说明）：
  1. conda 提供的运行库 DLL（`ffi.dll` 等）没被打进包，导致
     `ImportError: DLL load failed while importing _ctypes`；
     现在从 `_ctypes.pyd` 反推 conda 目录并显式收集
  2. 自动收集的 Tcl 版本（8.6.5）与环境（8.6.15）不一致，
     导致 Tcl 初始化失败；现在强制替换为 conda 版本
  3. `main()` 内 `import sys` 造成变量遮蔽，在它之前引用 `sys` 会抛
     `UnboundLocalError`；窗口版 exe 看不到该异常，
     表现为「双击闪一下就退出」。改用模块级别名 `_sys`

**新增**

- **启动日志**（默认关闭）：设置 `RADAR_STARTUP_LOG=1` 后，
  程序把启动过程与顶层异常写入 `_startup.log`，
  用于排查窗口版 exe「闪退且无提示」的问题

### v0.2.0

**核心修复**

- **修正 IIQQ 排布下的解码错位**（最严重的 bug）：原实现把相邻采样点的
  同一位凑成一个采样点，导致 RX1/RX2/RX3 样本整体错位
  （RX0 因偏移为 0 恰好正确，所以极难发现）
- `bandpass_filter` 支持任意维度（原来硬编码 `axis=1`，传一维数组会 IndexError）
- `displacement_processing` 去掉硬编码的 `D:\my_output` 路径
- 修复 `save_csv=False` 时 `best_idx` 未定义导致的崩溃
- `DC_Eliminate` 把强制写 SVG 的副作用改为可选
- `Judge` 模型路径改用 `resources.resource_path`，打包后也能找到
- 修复 GBK 控制台输出 emoji 导致 `UnicodeEncodeError` 中断整个流程

**新增**

- **图形界面**：12 通道竖排可滚动显示；每张图带 matplotlib 标准导航栏
  （Home 返回、**Zoom 框选放大**）；滤波频带可选可自定义且切换不需重跑；
  目标距离**按 cm 输入、自动换算 bin**；大波动检查；参数预设；五种视图
- **多进程并行批处理**：每个文件一个进程；逐文件释放内存；可中止
- **打包成 exe**：`packaging/radar_gui.spec` 用 `collect_all` +
  `collect_dynamic_libs` 收集 scipy/sklearn 的编译扩展
- `resources.py` 资源路径定位，兼容源码运行与 PyInstaller 打包

### v0.1.0

首个版本：数据读取与解码、RANSAC 直流消除、Range FFT、
微位移计算、带通滤波、通道质量判断。

---

## 为什么做这个项目

第一次写 GitHub 开源项目，参加了大创项目后，考虑到我总是把自己的信号预处理问题弄得一团乱，所以打算上传到 GitHub 上便于管理。

考虑到本科生可能并未学习雷达信号处理知识（我也没有），难以弄到相关有效的开源处理文件（谁能想到 bin 文件居然是 IIQQ 这样排列的！），所以我打算把自己的项目处理代码上传。

目前这个项目处于能跑就行的状态，后面会慢慢完善，做完一个完整的数据处理流程。

后续会加上一些更高级的信号处理功能，后续更加模块化一点，慢慢来。

<span style="color:#39C5BB;">待って わかってよ 何でもないから</span>
<span style="color:#39C5BB;">僕の歌を笑わないで</span>