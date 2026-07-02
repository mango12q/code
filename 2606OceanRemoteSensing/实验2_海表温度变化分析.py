"""
实验2：全球海表温度变化分析（改进版）
========================================
基于NOAA OISST V2数据，分析全球海表温度(SST)的时空分布和变化趋势。

改进点：
1. 使用Cartopy替代Basemap进行地图投影
2. 函数化封装，提高代码复用性
3. 增加时间序列分析和趋势计算功能
4. 自动创建结果输出目录
5. 详细的注释和文档字符串
6. 使用pathlib处理路径，提高跨平台可移植性
7. 增加多时间点对比分析功能

依赖库：
    - numpy: 数值计算
    - matplotlib: 绘图
    - cartopy: 地图投影
    - netCDF4: NetCDF文件读写
    - scipy: 科学计算（可选，用于更高级的趋势分析）

作者：基于《海洋遥感数据处理实践初级教程》改进
Python版本：3.13+
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import netCDF4 as nc

# 使用Cartopy替代Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# 配置区域
# =============================================================================

# 数据文件路径（使用绝对路径避免工作目录问题）
# 修改此处为您的实际数据路径
BASE_DIR = Path(r"E:\Python")
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "sst.wkmean.1990-present.nc"

# 结果输出目录
RESULTS_DIR = BASE_DIR / "results" / "实验2_海表温度"

# 时间基准（OISST数据的时间单位：days since 1800-1-1 00:00:00）
TIME_BASE = datetime(1800, 1, 1)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 工具函数
# =============================================================================

def setup_directories():
    """创建必要的结果输出目录。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 结果将保存至: {RESULTS_DIR}")


def check_data_file(filepath: Path) -> bool:
    """检查数据文件是否存在。"""
    if not filepath.exists():
        print(f"[ERROR] 数据文件不存在: {filepath}")
        return False
    print(f"[INFO] 数据文件已找到: {filepath}")
    return True


def days_to_date(days: np.ndarray) -> np.ndarray:
    """
    将距离1800-01-01的天数转换为datetime对象。
    
    Args:
        days: 距离基准时间的天数数组
        
    Returns:
        np.ndarray: datetime对象数组
    """
    return np.array([TIME_BASE + timedelta(days=float(d)) for d in days])


def date_to_days(date: datetime) -> float:
    """
    将datetime对象转换为距离1800-01-01的天数。
    
    Args:
        date: datetime对象
        
    Returns:
        float: 距离基准时间的天数
    """
    return (date - TIME_BASE).total_seconds() / 86400.0


def read_sst_data(filepath: Path) -> dict:
    """
    读取OISST海表温度数据。
    
    Args:
        filepath: NetCDF文件路径
        
    Returns:
        dict: 包含SST数据、经纬度、时间等信息的字典
    """
    print("[INFO] 正在读取SST数据...")
    
    with nc.Dataset(filepath, 'r') as dataset:
        # 打印数据基本信息
        print("\n" + "="*60)
        print("数据文件基本信息")
        print("="*60)
        print(f"文件格式: {dataset.data_model}")
        print(f"标题: {getattr(dataset, 'title', 'N/A')}")
        print(f"来源: {getattr(dataset, 'source', 'N/A')}")
        
        # 读取维度
        lat = dataset.variables['lat'][:]
        lon = dataset.variables['lon'][:]
        time = dataset.variables['time'][:]
        
        # 读取SST数据（注意：大文件可能需要较多内存）
        print("[INFO] 正在读取SST变量（可能需要一些时间）...")
        sst = dataset.variables['sst'][:]
        
        # 获取变量属性
        sst_attrs = {
            'units': getattr(dataset.variables['sst'], 'units', 'N/A'),
            'long_name': getattr(dataset.variables['sst'], 'long_name', 'N/A'),
            'scale_factor': getattr(dataset.variables['sst'], 'scale_factor', 1.0),
            'add_offset': getattr(dataset.variables['sst'], 'add_offset', 0.0),
            '_FillValue': getattr(dataset.variables['sst'], '_FillValue', -32767)
        }
        
        # 时间属性
        time_units = getattr(dataset.variables['time'], 'units', 'days since 1800-1-1')
        
        print(f"\n变量信息:")
        print(f"  SST 形状: {sst.shape} (time, lat, lon)")
        print(f"  时间范围: {len(time)} 个时间点")
        print(f"  经度范围: {lon.min():.1f}° ~ {lon.max():.1f}°")
        print(f"  纬度范围: {lat.min():.1f}° ~ {lat.max():.1f}°")
        print(f"  单位: {sst_attrs['units']}")
        print(f"  时间单位: {time_units}")
        print("="*60 + "\n")
        
        # 转换时间
        dates = days_to_date(time)
        
        return {
            'sst': np.array(sst),
            'lon': np.array(lon),
            'lat': np.array(lat),
            'time': np.array(time),
            'dates': dates,
            'sst_attrs': sst_attrs
        }


def select_time_range(data: dict, start_year: int, end_year: int) -> dict:
    """
    选择指定年份范围的数据。
    
    Args:
        data: 完整数据集
        start_year: 起始年份
        end_year: 结束年份
        
    Returns:
        dict: 包含筛选后数据的字典
    """
    dates = data['dates']
    mask = (dates >= datetime(start_year, 1, 1)) & (dates <= datetime(end_year, 12, 31))
    indices = np.where(mask)[0]
    
    print(f"[INFO] 选择 {start_year}-{end_year} 年数据，共 {len(indices)} 个时间点")
    
    return {
        'sst': data['sst'][indices, :, :],
        'time': data['time'][indices],
        'dates': data['dates'][indices],
        'lon': data['lon'],
        'lat': data['lat']
    }


def calculate_yearly_mean(data: dict, year: int) -> np.ndarray:
    """
    计算指定年份的平均SST。
    
    Args:
        data: 完整数据集
        year: 目标年份
        
    Returns:
        np.ndarray: 年平均SST二维数组 (lat, lon)
    """
    year_data = select_time_range(data, year, year)
    return np.mean(year_data['sst'], axis=0)


def calculate_period_mean(data: dict, start_year: int, end_year: int) -> np.ndarray:
    """
    计算指定时间段的平均SST。
    
    Args:
        data: 完整数据集
        start_year: 起始年份
        end_year: 结束年份
        
    Returns:
        np.ndarray: 时段平均SST二维数组 (lat, lon)
    """
    period_data = select_time_range(data, start_year, end_year)
    return np.mean(period_data['sst'], axis=0)


def calculate_trend(data: dict, start_year: int, end_year: int) -> np.ndarray:
    """
    计算每个网格点的SST变化趋势（线性回归斜率）。
    
    Args:
        data: 完整数据集
        start_year: 起始年份
        end_year: 结束年份
        
    Returns:
        np.ndarray: 趋势数组 (lat, lon)，单位：°C/年
    """
    print(f"[INFO] 正在计算 {start_year}-{end_year} 年SST变化趋势...")
    
    period_data = select_time_range(data, start_year, end_year)
    sst = period_data['sst']
    time_days = period_data['time']
    
    n_lat, n_lon = len(period_data['lat']), len(period_data['lon'])
    trend = np.zeros((n_lat, n_lon))
    
    # 逐点计算线性趋势
    total = n_lat * n_lon
    count = 0
    
    for i in range(n_lat):
        for j in range(n_lon):
            sst_series = sst[:, i, j]
            
            # 排除无效值
            valid_mask = sst_series > -100  # 排除明显无效值
            if np.sum(valid_mask) < 10:  # 需要足够的数据点
                trend[i, j] = np.nan
                continue
            
            # 线性回归
            x = time_days[valid_mask]
            y = sst_series[valid_mask]
            
            # 使用numpy的polyfit
            coeffs = np.polyfit(x, y, 1)
            # 转换为 °C/年
            trend[i, j] = coeffs[0] * 365.25
            
            count += 1
            if count % 10000 == 0:
                print(f"[PROGRESS] 已处理 {count}/{total} 个网格点 ({count/total*100:.1f}%)")
    
    print(f"[INFO] 趋势计算完成")
    return trend


# =============================================================================
# 可视化函数
# =============================================================================

def plot_sst_map(sst_data: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                 title: str = "SST Distribution", cmap: str = 'jet',
                 vmin: float = -2, vmax: float = 35,
                 save_path: Optional[Path] = None) -> None:
    """
    绘制SST空间分布图。
    
    Args:
        sst_data: SST二维数组 (lat, lon)
        lon: 经度数组
        lat: 纬度数组
        title: 图标题
        cmap: 颜色映射
        vmin: 最小值
        vmax: 最大值
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    # 创建网格
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # 绘制数据
    im = ax.pcolormesh(lon_grid, lat_grid, sst_data,
                       transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto')
    
    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    
    # 网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('Sea Surface Temperature (°C)', fontsize=12)
    
    # 标题
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


def plot_sst_comparison(data: dict, year1: int, year2: int) -> None:
    """
    对比两个年份的SST分布。
    
    Args:
        data: 完整数据集
        year1: 第一个年份
        year2: 第二个年份
    """
    print(f"[INFO] 正在对比 {year1} 年和 {year2} 年的SST...")
    
    sst1 = calculate_yearly_mean(data, year1)
    sst2 = calculate_yearly_mean(data, year2)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                            subplot_kw={'projection': ccrs.Robinson()})
    
    lon = data['lon']
    lat = data['lat']
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # 年份1
    im1 = axes[0].pcolormesh(lon_grid, lat_grid, sst1,
                             transform=ccrs.PlateCarree(),
                             cmap='jet', vmin=-2, vmax=35, shading='auto')
    axes[0].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[0].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[0].set_title(f'{year1} Mean SST', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.05,
                shrink=0.8, label='SST (°C)')
    
    # 年份2
    im2 = axes[1].pcolormesh(lon_grid, lat_grid, sst2,
                             transform=ccrs.PlateCarree(),
                             cmap='jet', vmin=-2, vmax=35, shading='auto')
    axes[1].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[1].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[1].set_title(f'{year2} Mean SST', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.05,
                shrink=0.8, label='SST (°C)')
    
    # 差值
    diff = sst2 - sst1
    im3 = axes[2].pcolormesh(lon_grid, lat_grid, diff,
                             transform=ccrs.PlateCarree(),
                             cmap='RdBu_r', vmin=-2, vmax=2, shading='auto')
    axes[2].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[2].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[2].set_title(f'{year2}-{year1} SST Difference', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.05,
                shrink=0.8, label='ΔSST (°C)')
    
    fig.suptitle(f'Sea Surface Temperature Comparison: {year1} vs {year2}',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_file = RESULTS_DIR / f'sst_comparison_{year1}_{year2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()


def plot_trend_map(trend: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                   start_year: int, end_year: int) -> None:
    """
    绘制SST变化趋势图。
    
    Args:
        trend: 趋势数组 (lat, lon)
        lon: 经度数组
        lat: 纬度数组
        start_year: 起始年份
        end_year: 结束年份
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # 使用对称的颜色范围
    vmax = np.nanmax(np.abs(trend))
    
    im = ax.pcolormesh(lon_grid, lat_grid, trend,
                       transform=ccrs.PlateCarree(),
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='auto')
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('SST Trend (°C/year)', fontsize=12)
    
    ax.set_title(f'Global SST Trend ({start_year}-{end_year})\n'
                f'Linear Regression Slope',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_file = RESULTS_DIR / f'sst_trend_{start_year}_{end_year}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()


def plot_time_series(data: dict, lon_target: float, lat_target: float,
                     start_year: int, end_year: int) -> None:
    """
    绘制指定位置的SST时间序列和趋势。
    
    Args:
        data: 完整数据集
        lon_target: 目标经度
        lat_target: 目标纬度
        start_year: 起始年份
        end_year: 结束年份
    """
    print(f"[INFO] 正在分析位置 ({lon_target}°E, {lat_target}°N) 的SST时间序列...")
    
    lon = data['lon']
    lat = data['lat']
    
    # 找到最近的网格点
    lon_idx = np.argmin(np.abs(lon - lon_target))
    lat_idx = np.argmin(np.abs(lat - lat_target))
    
    actual_lon = lon[lon_idx]
    actual_lat = lat[lat_idx]
    
    print(f"[INFO] 最近网格点: ({actual_lon}°E, {actual_lat}°N)")
    
    # 选择时间范围
    period_data = select_time_range(data, start_year, end_year)
    sst_series = period_data['sst'][:, lat_idx, lon_idx]
    dates = period_data['dates']
    time_days = period_data['time']
    
    # 线性回归
    valid_mask = sst_series > -100
    coeffs = np.polyfit(time_days[valid_mask], sst_series[valid_mask], 1)
    trend_line = np.polyval(coeffs, time_days)
    
    # 计算趋势
    trend_per_year = coeffs[0] * 365.25
    
    # 绘图
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(dates[valid_mask], sst_series[valid_mask], 'b-', 
           linewidth=0.8, alpha=0.7, label='Weekly SST')
    ax.plot(dates[valid_mask], trend_line[valid_mask], 'r-',
           linewidth=2, label=f'Trend: {trend_per_year:.4f} °C/year')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Sea Surface Temperature (°C)', fontsize=12)
    ax.set_title(f'SST Time Series at ({actual_lon}°E, {actual_lat}°N)\n'
                f'{start_year}-{end_year}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = RESULTS_DIR / f'sst_timeseries_{actual_lon}E_{actual_lat}N.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()
    
    print(f"[RESULT] 该位置SST变化趋势: {trend_per_year:.4f} °C/年")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主函数：执行实验2的所有步骤。"""
    print("="*70)
    print("实验2：全球海表温度变化分析")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # 步骤1：创建输出目录
    setup_directories()
    
    # 步骤2：检查数据文件
    if not check_data_file(DATA_FILE):
        print("\n[ERROR] 数据文件检查失败，程序退出。")
        sys.exit(1)
    
    # 步骤3：读取数据
    try:
        data = read_sst_data(DATA_FILE)
    except Exception as e:
        print(f"\n[ERROR] 数据读取失败: {e}")
        sys.exit(1)
    
    # 步骤4：绘制1998年平均SST
    print("\n" + "-"*60)
    print("任务1: 绘制1998年平均SST分布")
    print("-"*60)
    sst_1998 = calculate_yearly_mean(data, 1998)
    plot_sst_map(sst_1998, data['lon'], data['lat'],
                title='Global Mean SST in 1998',
                save_path=RESULTS_DIR / 'sst_mean_1998.png')
    
    # 步骤5：绘制1990-2019年平均SST
    print("\n" + "-"*60)
    print("任务2: 绘制1990-2019年平均SST分布")
    print("-"*60)
    sst_climatology = calculate_period_mean(data, 1990, 2019)
    plot_sst_map(sst_climatology, data['lon'], data['lat'],
                title='Global Mean SST (1990-2019 Climatology)',
                save_path=RESULTS_DIR / 'sst_climatology_1990_2019.png')
    
    # 步骤6：对比1998和1999年（El Nino vs La Nina）
    print("\n" + "-"*60)
    print("任务3: 对比1998年(El Nino)和1999年(La Nina)的SST")
    print("-"*60)
    plot_sst_comparison(data, 1998, 1999)
    
    # 步骤7：绘制某点时间序列和趋势
    print("\n" + "-"*60)
    print("任务4: 分析特定位置的SST时间序列")
    print("-"*60)
    plot_time_series(data, lon_target=173.5, lat_target=20.5,
                    start_year=1990, end_year=2019)
    
    # 步骤8：计算并绘制全球SST变化趋势
    print("\n" + "-"*60)
    print("任务5: 计算全球SST变化趋势")
    print("-"*60)
    trend = calculate_trend(data, start_year=1990, end_year=2019)
    plot_trend_map(trend, data['lon'], data['lat'], 1990, 2019)
    
    print("\n" + "="*70)
    print("实验2完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果保存在: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
