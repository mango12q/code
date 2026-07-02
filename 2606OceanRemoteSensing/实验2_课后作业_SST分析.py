"""
实验2 课后作业：全球海表温度变化分析
====================================
基于NOAA OISST V2数据，完成教材第3.4节要求的课后作业。

作业要求：
1. 画出1998和1999年的SST的差值，主要的差异集中在哪些区域，试分析原因
2. 绘制1990年-2019年全球不同位置SST最大和最小值，以及最大最小值差值的分布
3. 绘制1990年-2019年全球不同位置SST的标准差，并描述分布特征
4. 选取140.5°W经线上5个不同纬度（0.5°N, 20.5°N, 40.5°N, 60.5°N, 80.5°N）的点，
   在同一张图上画出他们1990年-2000年温度变化的曲线

依赖库：
    - numpy, matplotlib, cartopy, netCDF4

作者：基于《海洋遥感数据处理实践初级教程》改进
Python版本：3.13+
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# 配置区域
# =============================================================================

BASE_DIR = Path(r"E:\Python")
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "sst.wkmean.1990-present.nc"

RESULTS_DIR = BASE_DIR / "results" / "实验2_课后作业"

# 时间基准（OISST数据的时间单位：days since 1800-1-1 00:00:00）
TIME_BASE = datetime(1800, 1, 1)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 工具函数
# =============================================================================

def setup_directories():
    """创建必要的结果输出目录。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 结果将保存至: {RESULTS_DIR}")


def days_to_date(days: np.ndarray) -> np.ndarray:
    """将距离1800-01-01的天数转换为datetime对象。"""
    return np.array([TIME_BASE + timedelta(days=float(d)) for d in days])


def read_sst_data(filepath: Path) -> dict:
    """读取OISST海表温度数据。"""
    print("[INFO] 正在读取SST数据...")
    
    with nc.Dataset(filepath, 'r') as dataset:
        lat = dataset.variables['lat'][:]
        lon = dataset.variables['lon'][:]
        time = dataset.variables['time'][:]
        sst = dataset.variables['sst'][:]
        
        dates = days_to_date(time)
        
        print(f"  SST 形状: {sst.shape} (time, lat, lon)")
        print(f"  时间范围: {len(time)} 个时间点")
        print(f"  经度范围: {lon.min():.1f}° ~ {lon.max():.1f}°")
        print(f"  纬度范围: {lat.min():.1f}° ~ {lat.max():.1f}°")
        
        return {
            'sst': np.array(sst),
            'lon': np.array(lon),
            'lat': np.array(lat),
            'time': np.array(time),
            'dates': dates
        }


def select_time_range(data: dict, start_year: int, end_year: int) -> dict:
    """选择指定年份范围的数据。"""
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
    """计算指定年份的平均SST。"""
    year_data = select_time_range(data, year, year)
    return np.mean(year_data['sst'], axis=0)


def plot_sst_map(sst_data: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                 title: str = "SST Distribution", cmap: str = 'jet',
                 vmin: float = -2, vmax: float = 35,
                 save_path: Path = None) -> None:
    """绘制SST空间分布图。"""
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    im = ax.pcolormesh(lon_grid, lat_grid, sst_data,
                       transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto')
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('Sea Surface Temperature (°C)', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


# =============================================================================
# 作业1: 1998年与1999年SST差值
# =============================================================================

def homework_1_sst_difference(data: dict) -> None:
    """
    作业1：画出1998和1999年的SST差值。
    1998年为强El Niño年，1999年为La Niña年。
    """
    print("\n" + "-"*60)
    print("作业1: 1998年与1999年SST差值分析")
    print("-"*60)
    
    sst_1998 = calculate_yearly_mean(data, 1998)
    sst_1999 = calculate_yearly_mean(data, 1999)
    diff = sst_1999 - sst_1998
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                            subplot_kw={'projection': ccrs.Robinson()})
    
    lon = data['lon']
    lat = data['lat']
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # 1998年
    im1 = axes[0].pcolormesh(lon_grid, lat_grid, sst_1998,
                             transform=ccrs.PlateCarree(),
                             cmap='jet', vmin=-2, vmax=35, shading='auto')
    axes[0].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[0].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[0].set_title('1998 Mean SST (El Niño)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.05,
                shrink=0.8, label='SST (°C)')
    
    # 1999年
    im2 = axes[1].pcolormesh(lon_grid, lat_grid, sst_1999,
                             transform=ccrs.PlateCarree(),
                             cmap='jet', vmin=-2, vmax=35, shading='auto')
    axes[1].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[1].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[1].set_title('1999 Mean SST (La Niña)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.05,
                shrink=0.8, label='SST (°C)')
    
    # 差值
    vmax = np.nanmax(np.abs(diff))
    im3 = axes[2].pcolormesh(lon_grid, lat_grid, diff,
                             transform=ccrs.PlateCarree(),
                             cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    axes[2].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[2].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[2].set_title('1999-1998 SST Difference', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.05,
                shrink=0.8, label='ΔSST (°C)')
    
    fig.suptitle('SST Comparison: El Niño (1998) vs La Niña (1999)',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_file = RESULTS_DIR / 'sst_diff_1998_1999.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()
    
    print("\n[分析] 1998年为强El Niño年，赤道东太平洋海温异常偏高；")
    print("       1999年为La Niña年，赤道东太平洋海温异常偏低。")
    print("       差值图显示东太平洋区域（尤其是赤道附近）差异最为显著。")


# =============================================================================
# 作业2: SST极值和差值分布
# =============================================================================

def homework_2_sst_extremes(data: dict) -> None:
    """
    作业2：绘制1990-2019年全球不同位置SST最大值、最小值及差值分布。
    """
    print("\n" + "-"*60)
    print("作业2: SST极值和差值分布 (1990-2019)")
    print("-"*60)
    
    period_data = select_time_range(data, 1990, 2019)
    sst = period_data['sst']
    
    # 计算每个网格点的最大值、最小值和差值
    sst_max = np.max(sst, axis=0)
    sst_min = np.min(sst, axis=0)
    sst_range = sst_max - sst_min
    
    lon = data['lon']
    lat = data['lat']
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                            subplot_kw={'projection': ccrs.Robinson()})
    
    # 最大值分布
    im1 = axes[0].pcolormesh(lon_grid, lat_grid, sst_max,
                             transform=ccrs.PlateCarree(),
                             cmap='jet', vmin=-2, vmax=35, shading='auto')
    axes[0].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[0].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[0].set_title('SST Maximum (1990-2019)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.05,
                shrink=0.8, label='SST (°C)')
    
    # 最小值分布
    im2 = axes[1].pcolormesh(lon_grid, lat_grid, sst_min,
                             transform=ccrs.PlateCarree(),
                             cmap='jet', vmin=-2, vmax=35, shading='auto')
    axes[1].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[1].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[1].set_title('SST Minimum (1990-2019)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.05,
                shrink=0.8, label='SST (°C)')
    
    # 差值分布
    im3 = axes[2].pcolormesh(lon_grid, lat_grid, sst_range,
                             transform=ccrs.PlateCarree(),
                             cmap='jet', vmin=0, vmax=15, shading='auto')
    axes[2].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[2].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[2].set_title('SST Range (Max - Min)', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.05,
                shrink=0.8, label='ΔSST (°C)')
    
    fig.suptitle('SST Extremes Analysis (1990-2019)',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_file = RESULTS_DIR / 'sst_extremes_1990_2019.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()
    
    print("\n[分析] 最大值分布：热带区域SST最高，可达30°C以上；")
    print("       最小值分布：极地和高纬度区域SST最低；")
    print("       差值分布：中高纬度区域季节变化最大，赤道区域变化较小。")


# =============================================================================
# 作业3: SST标准差分布
# =============================================================================

def homework_3_sst_std(data: dict) -> None:
    """
    作业3：绘制1990-2019年全球不同位置SST的标准差分布。
    """
    print("\n" + "-"*60)
    print("作业3: SST标准差分布 (1990-2019)")
    print("-"*60)
    
    period_data = select_time_range(data, 1990, 2019)
    sst = period_data['sst']
    
    # 计算每个网格点的标准差
    sst_std = np.std(sst, axis=0)
    
    lon = data['lon']
    lat = data['lat']
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    im = ax.pcolormesh(lon_grid, lat_grid, sst_std,
                       transform=ccrs.PlateCarree(),
                       cmap='jet', vmin=0, vmax=5, shading='auto')
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('SST Standard Deviation (°C)', fontsize=12)
    
    ax.set_title('Global SST Standard Deviation (1990-2019)\n'
                'Temporal Variability',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_file = RESULTS_DIR / 'sst_std_1990_2019.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()
    
    print("\n[分析] 标准差分布反映了SST的时间变异性：")
    print("       中高纬度区域（如北太平洋、北大西洋）标准差较大，")
    print("       表明这些区域SST季节变化显著；")
    print("       赤道区域标准差较小，SST相对稳定。")


# =============================================================================
# 作业4: 140.5°W经线上5个点的SST时间序列
# =============================================================================

def homework_4_time_series(data: dict) -> None:
    """
    作业4：选取140.5°W经线上5个不同纬度的点，
    在同一张图上画出1990-2000年温度变化曲线。
    
    5个点坐标：
    - 0.5°N, 140.5°W
    - 20.5°N, 140.5°W
    - 40.5°N, 140.5°W
    - 60.5°N, 140.5°W
    - 80.5°N, 140.5°W
    """
    print("\n" + "-"*60)
    print("作业4: 140.5°W经线5个纬度点SST时间序列 (1990-2000)")
    print("-"*60)
    
    # 选择时间范围
    period_data = select_time_range(data, 1990, 2000)
    sst = period_data['sst']
    dates = period_data['dates']
    lon = data['lon']
    lat = data['lat']
    
    # 目标坐标
    target_lon = 360 - 140.5  # 140.5°W = 219.5°E (在0-360°范围内)
    target_lats = [0.5, 20.5, 40.5, 60.5, 80.5]
    colors = ['red', 'black', 'yellow', 'blue', 'green']
    labels = ['0°N', '20°N', '40°N', '60°N', '80°N']
    
    # 找到经度索引
    lon_idx = np.argmin(np.abs(lon - target_lon))
    actual_lon = lon[lon_idx]
    print(f"[INFO] 目标经度: 140.5°W, 实际使用: {actual_lon:.1f}°")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for target_lat, color, label in zip(target_lats, colors, labels):
        lat_idx = np.argmin(np.abs(lat - target_lat))
        actual_lat = lat[lat_idx]
        
        # 提取该点的SST时间序列
        sst_series = sst[:, lat_idx, lon_idx]
        
        # 排除无效值
        valid_mask = sst_series > -100
        
        ax.plot(dates[valid_mask], sst_series[valid_mask], 
               color=color, linewidth=1.5, label=f'{label} ({actual_lat:.1f}°N)')
        
        print(f"  {label}: 纬度索引={lat_idx}, 实际纬度={actual_lat:.1f}°")
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Sea Surface Temperature (°C)', fontsize=12)
    ax.set_title('SST Time Series at 140.5°W (1990-2000)\n'
                'Different Latitudes Comparison',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = RESULTS_DIR / 'sst_timeseries_1405W_1990_2000.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()
    
    print("\n[分析] 不同纬度SST变化特征：")
    print("       低纬度（0°N, 20°N）：SST较高且季节变化较小；")
    print("       中纬度（40°N, 60°N）：SST季节变化明显，夏季高冬季低；")
    print("       高纬度（80°N）：SST最低，且受海冰影响变化复杂。")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主函数：执行实验2所有课后作业。"""
    print("="*70)
    print("实验2 课后作业：全球海表温度变化分析")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # 步骤1：创建输出目录
    setup_directories()
    
    # 步骤2：检查数据文件
    if not DATA_FILE.exists():
        print(f"[ERROR] 数据文件不存在: {DATA_FILE}")
        sys.exit(1)
    print(f"[INFO] 数据文件已找到: {DATA_FILE}")
    
    # 步骤3：读取数据
    try:
        data = read_sst_data(DATA_FILE)
    except Exception as e:
        print(f"[ERROR] 数据读取失败: {e}")
        sys.exit(1)
    
    # 作业1：SST差值分析
    homework_1_sst_difference(data)
    
    # 作业2：SST极值分布
    homework_2_sst_extremes(data)
    
    # 作业3：SST标准差分布
    homework_3_sst_std(data)
    
    # 作业4：时间序列
    homework_4_time_series(data)
    
    print("\n" + "="*70)
    print("实验2 课后作业全部完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果保存在: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
