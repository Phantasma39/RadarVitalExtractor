# RadarVitalExtractor

基于 TI-IWR1843 毫米波雷达的**生命体征信号处理项目**，目前仍在开发与完善中。

---

## 项目简介
本项目为北京理工大学毫米波雷达测量大创项目，旨在利用 TI-IWR1843 雷达传感器，实现对脉搏波信号的采集与分析。

>  **项目状态：开发中**
> 
> 目前已完成基础数据读取与预处理模块，后续功能仍在持续迭代，欢迎提出建议与反馈。

---

## 雷达信号测量参数
>考虑到本雷达信号处理目前仅适用于我的雷达参数,以后将会考虑对所有参数雷达信号进行处理

- 3TX4RX  共计12个通道
- adc_samples 256
- num_chirps = 24
- frequency_slope = 80e12  # 斜率
- sample_rate = 1e7  # ADC采样率
- fc = 77e9  # 基频
- frame_rate = 250  # 帧率
- lam = c_v / fc
- d = lam / 2  # 天线间距

## 当前已实现功能
- ✅ 雷达原始数据（ADC/IQ）读取与解析
- ✅ 基础 DC 消除与批处理脚本

---
## 为什么做这个项目
第一次写github开源项目,什么也不会,只能慢慢来

参加了大创项目后,考虑到我总是把自己的信号预处理问题弄的一团乱,所以打算上传到github上便于管理

考虑到本科生可能并未学习雷达信号处理知识
__(我也没有)__,难以弄到相关有效的开源处理文件(谁能想到bin文件居然是IIQQ这样排列的!)

所以我打算把自己的项目处理代码上传

目前这个项目处于能跑就行的状态,后面会慢慢完善,做完一个完整的数据处理流程

后续会加上一些更高级一点的信号处理功能,后续更加模块化一点,慢慢来

最后是"海百合海底谭"的第一句歌词,我想作为结尾

__<span style="color:#39C5BB;">待って わかってよ 何でもないから</span>__

__<span style="color:#39C5BB;">等一等 我知道哦 没什么大不了的</span>__

__<span style="color:#39C5BB;">僕の歌を笑わないで</span>__

__<span style="color:#39C5BB;">请不要嘲笑我的歌啊</span>__



---
## 使用说明
```bash
# 克隆仓库
git clone https://github.com/Phantasma39/RadarVitalExtractor.git
cd RadarVitalExtractor

## 环境依赖

pip install -r requirements.txt

# 程序运行
python main.py
