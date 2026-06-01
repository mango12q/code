# Code Repository

[![MATLAB](https://img.shields.io/badge/MATLAB-100%25-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](LICENSE)

个人代码仓库，包含学术论文复现、工具脚本和技能开发等项目。

---

## 📁 项目结构

```
code/
├── 2510matlab/          # MATLAB 工具脚本集合
├── 2601SST_Prediction/  # 南海海表面温度预测研究复现
└── 2605deskmate.skill/  # Desk Mate 技能开发
```

---

## 🌊 2601SST_Prediction - 南海海表面温度智能化预测

基于深度学习的南海海表面温度（SST）预测研究 - MATLAB复现代码。

### 简介

本项目复现了论文《基于深度学习的南海海表面温度的智能化预测研究》（作者：谢博闻等，发表于《海洋与湖沼》2024年第55卷第5期）中的研究方法。

### 核心功能

- 🌡️ **多变量SST预测**：利用3D U-Net模型预测南海未来30天海表面温度
- 🌊 **海洋热浪检测**：识别和预测海洋热浪（MHW）事件
- 📊 **模型对比**：3D U-Net vs ConvLSTM性能对比分析
- 📈 **完整评估**：RMSE、R、SMAPE、MedAE等多维度评估指标

### 技术特点

| 特性 | 说明 |
|------|------|
| **研究区域** | 105°~122.5°E, 0°~23°N（南海） |
| **空间分辨率** | 0.25° |
| **输入变量** | SST、SSHA、ESSW、NSSW |
| **预测时长** | 历史64天 → 未来30天 |
| **模型架构** | 3D U-Net（3层编码器+3层解码器） |

### 快速开始

```matlab
cd('2601SST_Prediction');
main;  % 运行主程序
```

📖 [查看详细文档](2601SST_Prediction/README.md)

---

## 🛠️ 2510matlab - MATLAB 工具集

MATLAB实用脚本和工具函数集合，用于数据处理和科学计算。

### 内容

- 数据处理工具
- 可视化脚本
- 数学计算函数

---

## 🎨 2605deskmate.skill - Desk Mate 技能

自定义Desk Mate技能开发，扩展AI助手功能。

---

## 📊 论文复现结果

### 模型性能对比

| 模型 | RMSE (°C) | R | SMAPE | MedAE (°C) |
|------|-----------|---|-------|------------|
| 3D U-Net | 0.53 | 0.96 | 1.90% | 0.44 |
| ConvLSTM | 0.68 | 0.94 | 2.02% | 0.45 |

*3D U-Net在各项评估指标上均优于ConvLSTM模型*

---

## 🔧 环境要求

### MATLAB 依赖
- MATLAB R2020b 或更高版本
- Deep Learning Toolbox（深度学习工具箱）
- 可选：NetCDF支持（用于读取.nc数据文件）

### 硬件建议
- **内存**: 16GB+（处理高分辨率时空数据）
- **GPU**: NVIDIA GPU + CUDA（加速训练，可选）

---

## 📖 使用指南

### 1. 克隆仓库

```bash
git clone https://github.com/mango12q/code.git
cd code
```

### 2. 选择项目

```bash
# SST预测项目
cd 2601SST_Prediction

# MATLAB工具集
cd 2510matlab

# Desk Mate技能
cd 2605deskmate.skill
```

### 3. 运行代码

在MATLAB中打开对应项目文件夹，运行主程序。

---

## 📁 数据准备

对于SST预测项目，需要准备以下数据：

```
data/
├── SST/          # 海表面温度数据 (NOAA OISST v2.1)
├── SSHA/         # 海表面高度异常 (CMEMS)
└── SSW/          # 海表面风场 (CCMP v2.0)
    ├── essw/     # 东向风
    └── nssw/     # 北向风
```

支持格式：NetCDF (.nc) 或 MATLAB (.mat)



---

## 📝 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-06-01 | 更新春节期间完成的论文复现代码 |
| 2026-05-16 | 补充完善Desk Mate技能细节 |
| 2026-05-09 | 添加MATLAB工具集备注 |

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📄 许可证

本仓库代码仅供学术研究使用。

---

## 📧 联系方式

如有问题或建议，欢迎通过GitHub Issues联系。

---

<p align="center">Made with ❤️ by mango12q</p>
