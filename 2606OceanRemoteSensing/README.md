# 海洋遥感数据处理实验 - Python改进版

基于《海洋遥感数据处理实践初级教程》改进的Python实验代码，使用Python 3.13+和Cartopy等现代库实现。本套代码包含4个完整的海洋遥感数据处理实验，涵盖水色遥感、海表温度、散射计风场和高度计波高验证等核心内容。

---

## 目录结构

```
d:\trae\
├── 实验1_水色遥感_叶绿素可视化.py      # 实验1：MODIS叶绿素可视化
├── 实验2_海表温度变化分析.py            # 实验2：OISST海表温度分析
├── 实验3_散射计风场数据处理.py          # 实验3：HY-2B风场处理
├── 实验4_高度计波高验证与校准.py        # 实验4：Jason-3波高验证
├── requirements.txt                     # Python依赖库列表
├── README.md                            # 本说明文档
├── data\                                # 实验数据目录
│   ├── A20160652016072.L3m_8D_CHL_chlor_a_9km.nc    # 实验1：MODIS叶绿素
│   ├── sst.wkmean.1990-present.nc                   # 实验2：OISST海温
│   ├── H2B_OPER_SCA_L2B_OR_*.h5                    # 实验3：HY-2B风场(多文件)
│   ├── altimeter_data\                              # 实验4：高度计数据
│   │   └── wm_20120901.nc ~ wm_20120930.nc         # 30个NetCDF文件
│   └── buoy_data\                                   # 实验4：浮标数据
│       ├── 32012h2012.txt
│       ├── 41001h2012.txt
│       ├── 42002h2012.txt
│       ├── 46059h2012.txt
│       └── 51003h2012.txt
└── results\                             # 输出结果目录（自动创建）
    ├── 实验1_水色遥感\                  # 实验1输出
    ├── 实验2_海表温度\                  # 实验2输出
    ├── 实验3_散射计风场\                # 实验3输出
    └── 实验4_高度计波高\                # 实验4输出
```

---

## 环境要求

- **Python版本**: 3.13+
- **操作系统**: Windows/Linux/macOS
- **内存要求**: 建议8GB以上（实验2和实验4数据量较大）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖库说明

| 库名 | 版本 | 用途 |
|------|------|------|
| numpy | >=1.26.0 | 数值计算、数组操作 |
| matplotlib | >=3.8.0 | 数据可视化、绘图 |
| cartopy | >=0.23.0 | 地图投影（替代已弃用的Basemap） |
| netCDF4 | >=1.6.0 | NetCDF格式数据读写 |
| h5py | >=3.10.0 | HDF5格式数据读写 |
| scipy | >=1.12.0 | 科学计算、统计分析 |
| pandas | >=2.2.0 | 数据处理（可选） |
| tqdm | >=4.66.0 | 进度条显示（可选） |

---

## 实验详情

### 实验1：水色遥感数据的可视化

**文件名**: `实验1_水色遥感_叶绿素可视化.py`

**数据内容**:
- **来源**: MODIS Aqua卫星L3级产品
- **文件名**: `A20160652016072.L3m_8D_CHL_chlor_a_9km.nc`
- **时间范围**: 2016年第65-72天（3月5日-3月12日，8天合成）
- **空间范围**: 全球（-180°~180°E，-90°~90°N）
- **空间分辨率**: 9km
- **变量**: `chlor_a`（叶绿素a浓度，单位mg/m³）

**处理方式**:
1. 读取NetCDF格式叶绿素数据
2. 处理缺失值（`_FillValue`）
3. 使用对数刻度显示（叶绿素浓度分布通常呈对数正态）
4. 绘制全球分布图和中国海区域放大图
5. 对比线性刻度与对数刻度的显示效果

**输出结果**:
- `chlorophyll_global_log.png` - 全球叶绿素分布（对数刻度）
- `chlorophyll_global_linear.png` - 全球叶绿素分布（线性刻度）
- `chlorophyll_china_sea.png` - 中国海区域放大图

---

### 实验2：全球海表温度变化分析

**文件名**: `实验2_海表温度变化分析.py`

**数据内容**:
- **来源**: NOAA OISST V2（最优插值海表温度）
- **文件名**: `sst.wkmean.1990-present.nc`
- **时间范围**: 1990年至今，周平均数据
- **空间范围**: 全球（0°~360°E，-90°~90°N）
- **空间分辨率**: 1°
- **变量**: `sst`（海表温度，单位°C）
- **时间单位**: days since 1800-1-1 00:00:00

**处理方式**:
1. 读取OISST周平均海温数据
2. 计算年平均海表温度
3. 分析海温季节变化（气候态）
4. 对比El Niño年和La Niña年的海温异常
5. 计算海温长期变化趋势

**输出结果**:
- `sst_yearly_mean_YYYY.png` - 指定年份年平均海温分布
- `sst_climatology_season.png` - 四季气候态海温分布
- `sst_elnino_vs_lanina.png` - El Niño与La Niña年对比
- `sst_trend.png` - 海温长期变化趋势图
- `sst_time_series.png` - 指定区域海温时间序列

---

### 实验3：海洋风场遥感数据处理

**文件名**: `实验3_散射计风场数据处理.py`

**数据内容**:
- **来源**: HY-2B卫星微波散射计L2B级产品
- **文件名格式**: `H2B_OPER_SCA_L2B_OR_YYYYMMDDTHHMMSS_*.h5`
- **时间范围**: 2020年4月15日（多轨数据）
- **数据格式**: HDF5
- **空间分辨率**: 25km
- **主要变量**:
  - `wvc_lon` / `wvc_lat`: 风矢量单元经纬度（二维数组）
  - `wind_speed_selection`: 最佳风速解（m/s）
  - `wind_dir_selection`: 最佳风向解（度）
  - `wvc_quality_flag`: 质量标志
- **填充值**: 经纬度=1.7e38，风速/风向=-32767

**处理方式**:
1. 批量读取多个HDF5风场数据文件
2. 处理填充值和质量控制标志
3. 将二维扫描数据展平为一维点集
4. 绘制风场矢量图（quiver图）
5. 多文件数据叠加显示
6. 绘制风速玫瑰图
7. 风速频率统计分析

**输出结果**:
- `wind_field_map.png` - 风场空间分布矢量图
- `wind_speed_distribution.png` - 风速频率分布直方图
- `wind_rose.png` - 风向风速玫瑰图
- `wind_multi_overlay.png` - 多文件风场叠加图

---

### 实验4：卫星高度计波高数据验证与校准

**文件名**: `实验4_高度计波高验证与校准.py`

**数据内容**:

**高度计数据**:
- **来源**: Jason-3卫星雷达高度计（多星融合产品）
- **文件位置**: `data/altimeter_data/`
- **文件名**: `wm_20120901.nc` ~ `wm_20120930.nc`（30个文件）
- **时间范围**: 2012年9月1日-30日
- **数据格式**: NetCDF3 CLASSIC
- **空间覆盖**: 全球轨道数据
- **总数据点数**: 约429万点（30天合并）
- **主要变量**:
  - `swh`: 有效波高（significant wave height，单位m）
  - `lon`: 经度（度）
  - `lat`: 纬度（度）
  - `time`: 时间（days since 1900-1-1）
- **数据特性**: `int16`存储，`scale_factor=0.01`，NetCDF4自动缩放

**浮标数据**:
- **来源**: NDBC（美国国家数据浮标中心）
- **文件位置**: `data/buoy_data/`
- **文件名**: `32012h2012.txt`, `41001h2012.txt`, `42002h2012.txt`, `46059h2012.txt`, `51003h2012.txt`
- **时间范围**: 2012年全年
- **数据格式**: 标准NDBC文本格式（18列）
- **主要列**:
  - 第1-5列: 年、月、日、时、分
  - 第6列: 风向（WDIR，度）
  - 第7列: 风速（WSPD，m/s）
  - 第9列: 波高（WVHT，m）
- **填充值**: WVHT=99.0表示缺失，WSPD=99.0表示缺失

**浮标位置信息**:

| 浮标编号 | 纬度 | 经度 | 位置描述 |
|----------|------|------|----------|
| 32012 | -19.50° | 85.00° | 印度洋 |
| 41001 | 34.68° | -72.66° | 美国东海岸外海 |
| 42002 | 25.93° | -93.65° | 墨西哥湾 |
| 46059 | 38.05° | -129.95° | 东北太平洋 |
| 51003 | 19.30° | -160.70° | 夏威夷附近 |

**处理方式**:
1. **批量读取**: 自动读取所有高度计NetCDF文件和所有浮标文本文件
2. **数据合并**: 使用`np.concatenate`合并多文件数据
3. **质量控制**:
   - 高度计：过滤`_FillValue`和NaN值；注意NetCDF4库会自动应用`scale_factor`和`add_offset`
   - 浮标：过滤WVHT>50（填充值99.0）和WSPD>90的异常值
4. **时空匹配**（优化版，经纬度粗筛加速）:
   - 先按经纬度矩形粗筛（50km ≈ 0.45°），从429万点缩减到几千候选点
   - 再按时间窗口筛选（±3小时）
   - 最后用Haversine公式计算精确距离，空间窗口±50公里
   - 对每个浮标时间点，在候选点中找到时空最近邻点
   - 预先将卫星时间转为小时数，避免逐个datetime计算
5. **验证统计**:
   - 偏差（BIAS）
   - 均方根误差（RMSE）
   - 平均绝对误差（MAE）
   - 相关系数（R）
   - 相对偏差和相对RMSE
6. **线性回归校准**: y = ax + b
7. **校准效果评估**: 对比校准前后的统计指标
8. **诊断输出**: 运行时打印`[DEBUG]`标记的关键数据，便于排查问题

**输出结果**:
- `satellite_swh_map.png` - 卫星波高全球空间分布图
- `swh_scatter_comparison.png` - 卫星vs浮标散点对比图
- `swh_time_series.png` - 波高时间序列对比图
- `swh_residuals.png` - 残差分析图（散点+直方图）
- `swh_calibration_comparison.png` - 校准前后对比图

---

## 主要改进点

相比原教程代码，本改进版有以下提升：

1. **使用Cartopy替代Basemap**
   - Basemap已弃用，Cartopy是官方推荐替代方案
   - 支持更多投影方式和更现代的API

2. **函数化封装**
   - 每个功能封装为独立函数
   - 提高代码复用性和可读性
   - 完整的文档字符串（docstring）

3. **增加数据验证**
   - 自动检查数据文件是否存在
   - 验证变量名和数据维度
   - 处理多种异常情况和填充值

4. **自动创建输出目录**
   - 使用`pathlib.Path.mkdir(parents=True, exist_ok=True)`
   - 结果自动保存到`results/`目录下的对应子文件夹

5. **pathlib路径处理**
   - 替代字符串路径操作
   - 提高跨平台可移植性（Windows/Linux/macOS）

6. **多文件批量处理**
   - 实验3和实验4支持自动读取多个数据文件
   - 使用`glob`模式匹配文件
   - 数据合并与叠加显示

7. **时空匹配算法**
   - 实验4实现基于Haversine距离的最近邻匹配
   - 支持时间和空间双窗口筛选

8. **质量控制（QC）**
   - 实验3增加质量标志筛选
   - 实验4增加填充值和异常值过滤

---

## 运行代码

### 单个实验运行

```bash
# 实验1：水色遥感
python 实验1_水色遥感_叶绿素可视化.py

# 实验2：海表温度
python 实验2_海表温度变化分析.py

# 实验3：散射计风场
python 实验3_散射计风场数据处理.py

# 实验4：高度计波高
python 实验4_高度计波高验证与校准.py
```

### 批量运行所有实验

```bash
# Windows
for %f in (实验*.py) do python %f

# Linux/macOS
for f in 实验*.py; do python "$f"; done
```

---

## 数据准备

### 数据目录结构

请将实验数据按以下结构放置：

```
data/
├── A20160652016072.L3m_8D_CHL_chlor_a_9km.nc    # 实验1
├── sst.wkmean.1990-present.nc                   # 实验2
├── H2B_OPER_SCA_L2B_OR_*.h5                    # 实验3（多个文件）
├── altimeter_data/                               # 实验4
│   └── wm_201209*.nc                            # 高度计数据
└── buoy_data/                                    # 实验4
    └── *h2012.txt                               # 浮标数据
```

### 修改数据路径

如果数据存放位置不同，请修改各实验代码中的`BASE_DIR`变量：

```python
# 默认路径
BASE_DIR = Path(r"E:\Python")

# 修改为自定义路径
BASE_DIR = Path(r"D:\Your\Data\Path")
```

---

## 常见问题

### 1. 中文字体显示问题

如果图表中中文显示为方框，请修改代码中的字体设置：

```python
# Windows（默认）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']

# macOS
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']

# Linux
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
```

### 2. Cartopy安装问题

Cartopy依赖GEOS和PROJ库，安装时可能遇到问题：

```bash
# 推荐使用conda安装（自动处理依赖）
conda install -c conda-forge cartopy

# 或使用pip（需要预先安装系统库）
pip install cartopy
```

### 3. 内存不足

实验2和实验4数据量较大，如遇内存不足：
- 实验2：修改代码只读取部分年份数据
- 实验4：减少同时读取的高度计文件数量

### 4. 浮标位置不准确

实验4中的浮标位置为预设值，如需更精确的位置：
1. 查询NDBC官网获取最新位置
2. 修改代码中的`BUOY_LOCATIONS`字典

---

## 输出结果说明

所有实验结果保存在`results/`目录下，按实验分子目录：

```
results/
├── 实验1_水色遥感/
│   ├── chlorophyll_global_log.png
│   ├── chlorophyll_global_linear.png
│   └── chlorophyll_china_sea.png
├── 实验2_海表温度/
│   ├── sst_yearly_mean_*.png
│   ├── sst_climatology_season.png
│   ├── sst_elnino_vs_lanina.png
│   ├── sst_trend.png
│   └── sst_time_series.png
├── 实验3_散射计风场/
│   ├── wind_field_map.png
│   ├── wind_speed_distribution.png
│   ├── wind_rose.png
│   └── wind_multi_overlay.png
└── 实验4_高度计波高/
    ├── satellite_swh_map.png
    ├── swh_scatter_comparison.png
    ├── swh_time_series.png
    ├── swh_residuals.png
    └── swh_calibration_comparison.png
```

---

## 技术细节

### NetCDF数据读取

所有实验使用`netCDF4`库读取NetCDF数据：

```python
import netCDF4 as nc

with nc.Dataset(filepath, 'r') as dataset:
    # 读取变量
    data = dataset.variables['variable_name'][:]
    
    # 获取属性
    units = getattr(dataset.variables['variable_name'], 'units', 'N/A')
    scale_factor = getattr(dataset.variables['variable_name'], 'scale_factor', 1.0)
```

**注意**: netCDF4在读取时会自动应用`scale_factor`和`add_offset`，通常不需要手动缩放。

### HDF5数据读取

实验3使用`h5py`库读取HDF5数据：

```python
import h5py

with h5py.File(filepath, 'r') as f:
    # 查看变量
    for key in f.keys():
        print(f"{key}: shape={f[key].shape}")
    
    # 读取数据
    data = f['variable_name'][:]
```

### 地图投影

使用Cartopy创建地图投影：

```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 创建Robinson投影
ax = plt.axes(projection=ccrs.Robinson())

# 添加地图要素
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# 绘制数据
ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree())
```

---

## 参考资源

- [Cartopy官方文档](https://scitools.org.uk/cartopy/docs/latest/)
- [netCDF4 Python接口](https://unidata.github.io/netcdf4-python/)
- [h5py文档](https://docs.h5py.org/)
- [NDBC浮标数据](https://www.ndbc.noaa.gov/)
- [NOAA OISST数据](https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/)
- [MODIS Ocean Color](https://oceancolor.gsfc.nasa.gov/)

---

## 作者信息

- **基础教材**: 《海洋遥感数据处理实践初级教程》
- **改进版本**: 基于Python 3.13+和Cartopy的现代实现
- **改进内容**: 替代已弃用的Basemap，增加函数化封装、数据验证、多文件处理等功能

---

## 许可证

本代码仅供学习和研究使用，数据版权归原始数据提供方所有。
