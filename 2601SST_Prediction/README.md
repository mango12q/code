# 基于深度学习的南海海表面温度智能化预测研究 - MATLAB复现代码

本代码库复现了论文《基于深度学习的南海海表面温度的智能化预测研究》（作者：谢博闻等，发表于《海洋与湖沼》2024年第55卷第5期）中的研究方法。

## 论文核心内容

### 研究目标
- 利用深度学习模型（3D U-Net）预测南海海表面温度（SST）
- 预测未来30天的SST变化
- 检测海洋热浪（MHW）事件

### 研究区域
- **经度范围**: 105°~122.5°E
- **纬度范围**: 0°~23°N
- **空间分辨率**: 0.25°

### 输入数据
1. **SST**（海表面温度）- NOAA OISST v2.1
2. **SSHA**（海表面高度异常）- CMEMS
3. **ESSW**（东向海表面风）- CCMP v2.0
4. **NSSW**（北向海表面风）- CCMP v2.0

### 时间范围
- **训练集**: 1993年1月1日 - 2020年12月31日
- **测试集**: 2021年1月1日 - 2021年12月31日

### 模型架构
- **3D U-Net**: 3层编码器 + 3层解码器，隐藏层维度[64, 128, 256]
- **ConvLSTM**: 对比模型（3层，隐藏层维度[64, 64, 30]）

### 评估指标
- **RMSE**: 均方根误差
- **R**: 皮尔逊相关系数
- **SMAPE**: 对称平均绝对百分比误差
- **MedAE**: 中值绝对误差

## 代码文件说明

### 主程序
- `main.m` - 主程序入口，包含完整的训练和评估流程

### 数据预处理
- `load_and_preprocess_data.m` - 加载和预处理数据（支持NetCDF和MAT格式）
- `split_train_test.m` - 划分训练集和测试集
- `build_sequences.m` - 构建输入输出序列样本
- `denormalize.m` - 数据反标准化

### 模型定义
- `build_3d_unet.m` - 构建3D U-Net模型
- `build_convlstm.m` - 构建ConvLSTM对比模型

### 训练
- `train_3d_unet.m` - 训练3D U-Net模型（含2D替代方案）
- `train_convlstm.m` - 训练ConvLSTM模型

### 评估和可视化
- `evaluate_models.m` - 计算评估指标
- `visualize_results.m` - 生成可视化图表

## 使用说明

### 1. 环境要求
- MATLAB R2020b或更高版本
- Deep Learning Toolbox（深度学习工具箱）
- 可选：NetCDF支持（ncread函数）

### 2. 数据准备

将数据文件放置在以下目录结构中：
```
data/
├── SST/
│   ├── sst_data.nc  或  sst_data.mat
├── SSHA/
│   ├── ssha_data.nc  或  ssha_data.mat
└── SSW/
    ├── essw_data.nc  或  ssw_data.mat
    └── nssw_data.nc
```

数据文件格式：
- **NetCDF (.nc)**: 包含变量 'sst', 'ssha', 'essw', 'nssw' 以及 'lon', 'lat', 'time'
- **MAT (.mat)**: 包含变量 sst, ssha, essw, nssw, lon, lat, time

### 3. 运行代码

在MATLAB中运行：
```matlab
cd('SST_Prediction');  % 进入代码目录
main;  % 运行主程序
```

### 4. 参数配置

在`main.m`中修改以下参数：
```matlab
params.paths.sst = 'data/SST/';      % SST数据路径
params.paths.ssha = 'data/SSHA/';    % SSHA数据路径
params.paths.ssw = 'data/SSW/';      % SSW数据路径

params.model.input_days = 64;        % 输入历史天数
params.model.output_days = 30;       % 预测未来天数
params.model.batch_size = 12;        % 批次大小
params.model.learning_rate = 0.01;   % 学习率
params.model.epochs = 1000;          % 训练轮数
```

## 重要说明

### 关于3D卷积的实现
MATLAB原生对5D数据（3D卷积）的支持有限，代码中提供了两种方案：

1. **2D U-Net替代方案**（默认）: 将时间维度合并到通道维度，使用2D卷积处理
2. **Python实现建议**: 如需完整的3D U-Net，建议使用Python + TensorFlow/PyTorch

### 关于ConvLSTM
MATLAB原生不支持ConvLSTM层，代码中提供了LSTM + 全连接层的简化替代方案。

### 示例数据
如果没有真实数据，代码会自动生成示例数据用于测试。示例数据包含：
- 季节性变化（正弦波）
- 纬度梯度
- 随机噪声

## 输出结果

程序运行后会生成以下结果：

### 控制台输出
- 模型性能指标（RMSE, R, SMAPE, MedAE）
- 不同预测超前时间的性能对比

### 图表文件（保存在results/目录）
- `performance_comparison.png` - RMSE和R随预测时间变化曲线
- `scatter_plots.png` - 观测值vs预测值散点图
- `spatial_distribution.png` - 空间分布对比图
- `error_distribution.png` - 误差分布直方图

## 论文结果参考

论文中3D U-Net模型的性能（30天平均）：
- **RMSE**: 0.53°C
- **R**: 0.96
- **SMAPE**: 1.90%
- **MedAE**: 0.44°C

## 注意事项

1. **内存要求**: 处理高分辨率时空数据需要较大内存，建议至少16GB RAM
2. **GPU加速**: 如有NVIDIA GPU并安装了CUDA，MATLAB会自动使用GPU加速
3. **训练时间**: 完整训练可能需要数小时，取决于硬件配置
4. **早停机制**: 默认设置连续15轮验证损失不下降则停止训练

## 引用

如果您使用了本代码，请引用原始论文：

```
谢博闻, 张聪, 杨树国, 等. 基于深度学习的南海海表面温度的智能化预测研究[J]. 
海洋与湖沼, 2024, 55(5): 1083-1095.
```

## 联系方式

如有问题或建议，欢迎提出Issue或Pull Request。

## 许可证

本代码仅供学术研究使用。
