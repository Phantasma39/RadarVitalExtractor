# RadarVitalExtractor

基于 TI-IWR1843 毫米波雷达的**生命体征信号处理项目**。

---

## 项目简介
本项目为北京理工大学毫米波雷达测量大创项目，旨在利用 TI-IWR1843 雷达传感器，实现对脉搏波信号的采集与分析。

> **项目状态：开发中**
>
> 目前已完成基础数据读取与预处理模块，后续功能仍在持续迭代，欢迎提出建议与反馈。

---

## 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/Phantasma39/RadarVitalExtractor.git
cd RadarVitalExtractor

# 安装包（会自动安装所有依赖）
pip install -e .
```

### 2. 运行

```bash
# 方式一：使用命令行模块（推荐）
python -m radar_project.main <bin文件路径> [输出目录]

# 方式二：使用脚本
python scripts/main.py <bin文件路径> [输出目录]

# 批量处理
python scripts/Batch_process.py <数据文件夹路径> [输出目录]

# IQ 星座图分析
python scripts/IQ.py <bin文件路径> [range_bin]

# 画图分析
python -m radar_project.Draw <bin文件路径> [输出目录]

# 端到端独立测试（不依赖包安装也能跑）
python scripts/TEST.py <bin文件路径>
```

---

## 项目结构

```
RADAR/
├── docs/                       # 文档
├── models/                     # 预训练模型
│   ├── rf_model.pkl            # 随机森林通道质量判断模型
│   └── threshold.pkl           # 判断阈值
├── src/radar_project/          # 核心包
│   ├── __init__.py             # 公共 API 导出
│   ├── main.py                 # 主入口（命令行）
│   ├── utils.py                # bin 文件读取与解码
│   ├── range_fft.py            # Range FFT 处理
│   ├── DC_Eliminate.py         # RANSAC 圆拟合 DC 消除
│   ├── displacement_processing.py # 微位移计算
│   ├── Judge.py                # 通道质量判断（机器学习）
│   ├── Draw.py                 # 画图工具
│   ├── IQ.py                   # IQ 星座图分析
│   ├── select_file.py          # 按概率筛选文件
│   ├── Select.py               # 模型训练脚本
│   ├── Delete.py               # 删除低质量文件
│   ├── rename.py               # 批量重命名
│   └── TEST.py                 # 测试代码
├── scripts/                    # 便捷运行脚本
│   ├── main.py
│   ├── Batch_process.py
│   ├── IQ.py
│   └── TEST.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 雷达信号测量参数

> 考虑到本雷达信号处理目前仅适用于我的雷达参数，以后将会考虑对所有参数雷达信号进行处理

- 3TX4RX，共计 12 个通道
- adc_samples: 256
- num_chirps: 24
- frequency_slope: 80e12
- sample_rate: 1e7
- fc: 77e9
- frame_rate: 250

---

## 当前已实现功能

- ✅ 雷达原始数据（ADC/IQ）读取与解析
- ✅ 基础 DC 消除（RANSAC 圆拟合）
- ✅ Range FFT 与目标 bin 选取
- ✅ 微位移计算（相位解缠 + 带通滤波）
- ✅ 通道质量自动判断（随机森林模型）
- ✅ 批量处理脚本

---

## 为什么做这个项目

第一次写 GitHub 开源项目，参加了大创项目后，考虑到我总是把自己的信号预处理问题弄得一团乱，所以打算上传到 GitHub 上便于管理。

考虑到本科生可能并未学习雷达信号处理知识（我也没有），难以弄到相关有效的开源处理文件（谁能想到 bin 文件居然是 IIQQ 这样排列的！），所以我打算把自己的项目处理代码上传。

目前这个项目处于能跑就行的状态，后面会慢慢完善，做完一个完整的数据处理流程。

后续会加上一些更高级的信号处理功能，后续更加模块化一点，慢慢来。

<span style="color:#39C5BB;">待って わかってよ 何でもないから</span>
<span style="color:#39C5BB;">僕の歌を笑わないで</span>