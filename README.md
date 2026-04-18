# RadarVitalExtractor

基于 TI-IWR1843 毫米波雷达的**生命体征信号处理项目**，目前仍在开发与完善中。

---

## 项目简介
本项目为北京理工大学相关课程/实验项目，旨在利用 TI-IWR1843 雷达传感器，实现呼吸、心跳等生命体征信号的采集与分析。

> 🔧 **项目状态：开发中**
> 
> 目前已完成基础数据读取与预处理模块，后续功能仍在持续迭代，欢迎提出建议与反馈。

---

## 当前已实现功能
- ✅ 雷达原始数据（ADC/IQ）读取与解析
- ✅ 基础 DC 消除与批处理脚本
- ⏳ 呼吸/心跳信号提取（开发中）
- ⏳ 可视化与结果分析（开发中）

---

## 使用说明
```bash
# 克隆仓库
git clone https://github.com/Phantasma39/RadarVitalExtractor.git
cd RadarVitalExtractor

# 运行示例（后续补充具体命令）
python Batch_process.py
