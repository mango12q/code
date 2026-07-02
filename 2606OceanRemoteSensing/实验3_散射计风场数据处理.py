"""
实验3：海洋风场遥感数据处理（改进版）
========================================
基于HY-2B散射计沿轨风场数据，进行数据读取、处理和可视化分析。

改进点：
1. 使用Cartopy替代Basemap进行地图投影
2. 函数化封装，提高代码复用性
3. 增加数据质量控制（QC）筛选功能
4. 自动创建结果输出目录
5. 详细的注释和文档字符串
6. 使用pathlib处理路径，提高跨平台可移植性
7. 增加风速玫瑰图、时间序列等高级分析功能
8. 支持多文件批量处理

HY-2B散射计L2B数据格式说明：
- wvc_lon/wvc_lat: 风矢量单元经纬度 (n_scans, n_cells)
- wind_speed/wind_dir: 所有解的风速/风向 (n_scans, n_cells, 4)
- wind_speed_selection/wind_dir_selection: 选择的最佳解 (n_scans, n_cells)
- wvc_quality_flag: 质量标志 (n_scans, n_cells)
- 填充值: 经纬度为1.7e38, 风速/风向为-32767

依赖库：
    - numpy: 数值计算
    - matplotlib: 绘图
    - cartopy: 地图投影
    - h5py: HDF5文件读写

作者：基于《海洋遥感数据处理实践初级教程》改进
Python版本：3.13+
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py

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

# 结果输出目录
RESULTS_DIR = BASE_DIR / "results" / "实验3_散射计风场"

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


def find_h5_files(directory: Path) -> List[Path]:
    """
    查找目录中的所有HDF5文件。
    
    Args:
        directory: 数据目录
        
    Returns:
        List[Path]: HDF5文件路径列表
    """
    files = sorted(directory.glob("*.h5"))
    print(f"[INFO] 找到 {len(files)} 个HDF5文件")
    return files


def read_hy2b_wind_data(filepath: Path) -> dict:
    """
    读取HY-2B散射计风场数据。
    
    HY-2B L2B数据格式：
    - 二维数组: (n_scans, n_cells)
    - 三维数组: (n_scans, n_cells, 4) - 4个模糊解
    - 填充值: 经纬度=1.7e38, 风速/风向=-32767
    
    Args:
        filepath: HDF5文件路径
        
    Returns:
        dict: 包含风场数据的字典（展平后的一维数组）
    """
    print(f"[INFO] 正在读取: {filepath.name}")
    
    with h5py.File(filepath, 'r') as f:
        # 打印文件基本信息
        print("\n" + "="*60)
        print(f"文件: {filepath.name}")
        print("="*60)
        print("文件中的变量:")
        for key in f.keys():
            print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")
        
        # 读取数据
        data = {}
        
        # 读取经纬度（二维数组）
        lon_raw = np.array(f['wvc_lon'][:])
        lat_raw = np.array(f['wvc_lat'][:])
        
        # 读取选择的最佳风速和风向（二维数组）
        ws_select = np.array(f['wind_speed_selection'][:])
        wd_select = np.array(f['wind_dir_selection'][:])
        
        # 读取质量标志（二维数组）
        quality = np.array(f['wvc_quality_flag'][:])
        
        # 处理填充值
        # 经纬度填充值: 1.7e38
        FILL_LONLAT = 1.7e38
        # 风速/风向填充值: -32767
        FILL_WIND = -32767
        
        # 创建有效数据掩码
        valid_mask = (lon_raw < FILL_LONLAT) & (lat_raw < FILL_LONLAT) & \
                     (ws_select != FILL_WIND) & (wd_select != FILL_WIND) & \
                     (quality >= 0)  # 质量标志非负
        
        print(f"\n数据信息:")
        print(f"  原始数据形状: {lon_raw.shape} (scans, cells)")
        print(f"  有效数据点数: {np.sum(valid_mask)}/{valid_mask.size}")
        
        # 展平并筛选有效数据
        data['lon'] = lon_raw[valid_mask]
        data['lat'] = lat_raw[valid_mask]
        data['wind_speed'] = ws_select[valid_mask]
        data['wind_dir'] = wd_select[valid_mask]
        data['quality'] = quality[valid_mask]
        
        # 转换单位（如果数据有scale factor）
        # 检查wind_speed_selection的属性
        if 'scale_factor' in f['wind_speed_selection'].attrs:
            scale = f['wind_speed_selection'].attrs['scale_factor']
            data['wind_speed'] = data['wind_speed'] * scale
            print(f"  风速缩放因子: {scale}")
        
        if len(data['lon']) > 0:
            print(f"  经度范围: {data['lon'].min():.2f}° ~ {data['lon'].max():.2f}°")
            print(f"  纬度范围: {data['lat'].min():.2f}° ~ {data['lat'].max():.2f}°")
            print(f"  风速范围: {data['wind_speed'].min():.2f} ~ {data['wind_speed'].max():.2f} m/s")
            print(f"  风向范围: {data['wind_dir'].min():.2f}° ~ {data['wind_dir'].max():.2f}°")
        else:
            print(f"  [WARNING] 无有效数据")
        
        print("="*60 + "\n")
        
        return data


def filter_by_quality(data: dict, quality_threshold: int = 0) -> dict:
    """
    根据质量控制标志筛选数据。
    
    Args:
        data: 原始数据字典（一维数组）
        quality_threshold: 质量阈值
        
    Returns:
        dict: 筛选后的数据字典
    """
    if 'quality' not in data:
        print("[WARNING] 数据中无质量控制标志，跳过筛选")
        return data
    
    quality = data['quality']
    valid_mask = quality <= quality_threshold
    
    print(f"[INFO] 质量控制筛选: {np.sum(valid_mask)}/{len(quality)} 数据点通过")
    
    filtered_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) == len(quality):
            filtered_data[key] = value[valid_mask]
        else:
            filtered_data[key] = value
    
    return filtered_data


def filter_by_region(data: dict, 
                     lon_range: Tuple[float, float],
                     lat_range: Tuple[float, float]) -> dict:
    """
    根据经纬度范围筛选数据。
    
    Args:
        data: 原始数据字典
        lon_range: (lon_min, lon_max)
        lat_range: (lat_min, lat_max)
        
    Returns:
        dict: 筛选后的数据字典
    """
    lon = data['lon']
    lat = data['lat']
    
    valid_mask = ((lon >= lon_range[0]) & (lon <= lon_range[1]) &
                  (lat >= lat_range[0]) & (lat <= lat_range[1]))
    
    print(f"[INFO] 区域筛选: {np.sum(valid_mask)}/{len(lon)} 数据点在区域内")
    
    filtered_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) == len(lon):
            filtered_data[key] = value[valid_mask]
        else:
            filtered_data[key] = value
    
    return filtered_data


def filter_by_speed(data: dict, 
                    speed_range: Tuple[float, float] = (0, 50)) -> dict:
    """
    根据风速范围筛选数据。
    
    Args:
        data: 原始数据字典
        speed_range: (speed_min, speed_max)
        
    Returns:
        dict: 筛选后的数据字典
    """
    if 'wind_speed' not in data:
        return data
    
    speed = data['wind_speed']
    valid_mask = (speed >= speed_range[0]) & (speed <= speed_range[1])
    
    print(f"[INFO] 风速筛选: {np.sum(valid_mask)}/{len(speed)} 数据点通过")
    
    filtered_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) == len(speed):
            filtered_data[key] = value[valid_mask]
        else:
            filtered_data[key] = value
    
    return filtered_data


# =============================================================================
# 可视化函数
# =============================================================================

def plot_wind_scatter(data: dict, 
                      region: str = 'global',
                      title: str = "Wind Field",
                      save_path: Optional[Path] = None) -> None:
    """
    绘制风场散点图。
    
    Args:
        data: 风场数据字典
        region: 区域名称
        title: 图标题
        save_path: 保存路径
    """
    print(f"[INFO] 正在绘制风场散点图...")
    
    lon = data['lon']
    lat = data['lat']
    speed = data['wind_speed']
    
    if len(lon) == 0:
        print("[WARNING] 无数据可绘制")
        return
    
    # 定义区域范围
    regions = {
        'global': {
            'extent': [-180, 180, -80, 80],
            'projection': ccrs.Robinson()
        },
        'china_sea': {
            'extent': [102, 132, 0, 45],
            'projection': ccrs.LambertConformal(central_latitude=22.5, 
                                               central_longitude=117)
        },
        'north_pacific': {
            'extent': [120, 260, 0, 60],
            'projection': ccrs.PlateCarree(central_longitude=180)
        }
    }
    
    region_info = regions.get(region, regions['global'])
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=region_info['projection'])
    
    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    
    # 网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    # 绘制散点（风速用颜色表示）
    scatter = ax.scatter(lon, lat, c=speed, s=5,
                        cmap='jet', vmin=0, vmax=25,
                        transform=ccrs.PlateCarree(),
                        alpha=0.6)
    
    # 设置显示范围
    if 'extent' in region_info:
        ax.set_extent(region_info['extent'], crs=ccrs.PlateCarree())
    
    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('Wind Speed (m/s)', fontsize=12)
    
    # 标题
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


def plot_wind_quiver(data: dict,
                     title: str = "Wind Field",
                     save_path: Optional[Path] = None,
                     stride: int = 10) -> None:
    """
    绘制风场矢量图（箭头图）。
    
    Args:
        data: 风场数据字典
        title: 图标题
        save_path: 保存路径
        stride: 采样间隔
    """
    print(f"[INFO] 正在绘制风场矢量图...")
    
    lon = data['lon']
    lat = data['lat']
    speed = data['wind_speed']
    direction = data['wind_dir']
    
    if len(lon) == 0:
        print("[WARNING] 无数据可绘制")
        return
    
    # 风向转UV分量
    # 气象风向：风的来向，0°=北，90°=东
    dir_rad = np.deg2rad(direction)
    u = -speed * np.sin(dir_rad)
    v = -speed * np.cos(dir_rad)
    
    # 采样
    lon_sample = lon[::stride]
    lat_sample = lat[::stride]
    u_sample = u[::stride]
    v_sample = v[::stride]
    speed_sample = speed[::stride]
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    
    # 网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    # 绘制矢量
    quiver = ax.quiver(lon_sample, lat_sample, u_sample, v_sample,
                       speed_sample,
                       cmap='jet', scale=500, width=0.002,
                       transform=ccrs.PlateCarree(),
                       clim=[0, 25])
    
    # 颜色条
    cbar = plt.colorbar(quiver, ax=ax, orientation='horizontal',
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('Wind Speed (m/s)', fontsize=12)
    
    # 标题
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


def plot_wind_speed_histogram(data: dict,
                              bins: int = 50,
                              save_path: Optional[Path] = None) -> None:
    """
    绘制风速频率分布直方图。
    
    Args:
        data: 风场数据字典
        bins: 直方图分箱数
        save_path: 保存路径
    """
    print(f"[INFO] 正在绘制风速频率分布...")
    
    speed = data['wind_speed']
    
    if len(speed) == 0:
        print("[WARNING] 无数据可绘制")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 直方图
    n, bins_edges, patches = ax.hist(speed, bins=bins, 
                                     color='steelblue', alpha=0.7,
                                     edgecolor='black', linewidth=0.5)
    
    # 统计信息
    mean_speed = np.mean(speed)
    median_speed = np.median(speed)
    std_speed = np.std(speed)
    max_speed = np.max(speed)
    
    ax.axvline(mean_speed, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_speed:.2f} m/s')
    ax.axvline(median_speed, color='green', linestyle='--', linewidth=2,
              label=f'Median: {median_speed:.2f} m/s')
    
    ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Wind Speed Distribution\n'
                f'Mean={mean_speed:.2f}, Std={std_speed:.2f}, Max={max_speed:.2f} m/s',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


def plot_wind_rose(data: dict,
                   bins: int = 16,
                   save_path: Optional[Path] = None) -> None:
    """
    绘制风向玫瑰图。
    
    Args:
        data: 风场数据字典
        bins: 风向分箱数
        save_path: 保存路径
    """
    print(f"[INFO] 正在绘制风向玫瑰图...")
    
    direction = data['wind_dir']
    speed = data['wind_speed']
    
    if len(direction) == 0:
        print("[WARNING] 无数据可绘制")
        return
    
    # 风向分箱
    dir_bins = np.linspace(0, 360, bins + 1)
    dir_centers = (dir_bins[:-1] + dir_bins[1:]) / 2
    
    # 计算每个方向的风速统计
    speed_means = []
    counts = []
    
    for i in range(bins):
        mask = (direction >= dir_bins[i]) & (direction < dir_bins[i + 1])
        if np.sum(mask) > 0:
            speed_means.append(np.mean(speed[mask]))
            counts.append(np.sum(mask))
        else:
            speed_means.append(0)
            counts.append(0)
    
    speed_means = np.array(speed_means)
    
    # 绘制极坐标图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    
    theta = np.deg2rad(dir_centers)
    
    bars = ax.bar(theta, speed_means, width=np.deg2rad(360/bins),
                  bottom=0.0, alpha=0.7, color='steelblue',
                  edgecolor='black', linewidth=0.5)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(range(0, 360, 45), 
                     ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax.set_rlabel_position(90)
    ax.set_ylabel('Mean Wind Speed (m/s)', fontsize=12)
    ax.set_title('Wind Rose (Mean Speed by Direction)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


def plot_multiple_scatter(files: List[Path],
                          max_files: int = 4,
                          save_path: Optional[Path] = None) -> None:
    """
    绘制多个文件的风场散点图叠加对比。
    
    将所有文件的数据叠加显示在同一个地图上，用不同颜色区分不同文件。
    
    Args:
        files: HDF5文件列表
        max_files: 最大显示文件数
        save_path: 保存路径
    """
    print(f"[INFO] 正在绘制多文件风场叠加对比...")
    
    n_files = min(len(files), max_files)
    
    # 创建单个图形
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    
    # 网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    # 颜色方案：为每个文件分配不同颜色
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    # 存储所有数据用于颜色条
    all_lons = []
    all_lats = []
    all_speeds = []
    all_colors = []
    
    for i in range(n_files):
        print(f"[INFO] 读取文件 {i+1}/{n_files}: {files[i].name}")
        data = read_hy2b_wind_data(files[i])
        
        if len(data.get('lon', [])) == 0:
            print(f"[WARNING] 文件 {files[i].name} 无有效数据，跳过")
            continue
        
        lon = data['lon']
        lat = data['lat']
        speed = data['wind_speed']
        
        # 使用风速作为颜色映射
        scatter = ax.scatter(lon, lat, c=speed, s=2,
                           cmap='jet', vmin=0, vmax=25,
                           transform=ccrs.PlateCarree(),
                           alpha=0.6)
        
        all_lons.extend(lon)
        all_lats.extend(lat)
        all_speeds.extend(speed)
    
    if len(all_lons) == 0:
        print("[WARNING] 所有文件均无有效数据")
        return
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('Wind Speed (m/s)', fontsize=12)
    
    # 设置标题
    ax.set_title(f'Multi-File Wind Field Overlay ({n_files} files)\n'
                f'Total Points: {len(all_lons)}',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主函数：执行实验3的所有步骤。"""
    print("="*70)
    print("实验3：海洋风场遥感数据处理")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # 步骤1：创建输出目录
    setup_directories()
    
    # 步骤2：查找数据文件
    h5_files = find_h5_files(DATA_DIR)
    
    if not h5_files:
        print("\n[ERROR] 未找到HDF5数据文件，程序退出。")
        print(f"[HINT] 请确认数据文件位于: {DATA_DIR}")
        sys.exit(1)
    
    # 步骤3：读取第一个文件
    print("\n" + "-"*60)
    print("任务1: 读取并分析风场数据")
    print("-"*60)
    data = read_hy2b_wind_data(h5_files[0])
    
    if len(data.get('lon', [])) == 0:
        print("[ERROR] 无有效数据，程序退出")
        sys.exit(1)
    
    # 步骤4：质量控制筛选
    print("\n" + "-"*60)
    print("任务2: 质量控制筛选")
    print("-"*60)
    data_qc = filter_by_quality(data, quality_threshold=0)
    
    # 步骤5：风速筛选
    print("\n" + "-"*60)
    print("任务3: 风速范围筛选")
    print("-"*60)
    data_filtered = filter_by_speed(data_qc, speed_range=(0, 30))
    
    # 步骤6：绘制全球风场散点图
    print("\n" + "-"*60)
    print("任务4: 绘制全球风场散点图")
    print("-"*60)
    plot_wind_scatter(data_filtered, region='global',
                     title='Global Wind Field (HY-2B Scatterometer)',
                     save_path=RESULTS_DIR / 'wind_scatter_global.png')
    
    # 步骤7：绘制中国海区域风场
    print("\n" + "-"*60)
    print("任务5: 绘制中国海区域风场")
    print("-"*60)
    data_china = filter_by_region(data_filtered, 
                                  lon_range=(102, 132), 
                                  lat_range=(0, 45))
    if len(data_china.get('lon', [])) > 0:
        plot_wind_scatter(data_china, region='china_sea',
                         title='China Sea Wind Field (HY-2B)',
                         save_path=RESULTS_DIR / 'wind_scatter_china_sea.png')
    else:
        print("[WARNING] 中国海区域内无数据")
    
    # 步骤8：绘制风场矢量图
    print("\n" + "-"*60)
    print("任务6: 绘制风场矢量图")
    print("-"*60)
    plot_wind_quiver(data_filtered, 
                    title='Wind Vector Field (HY-2B)',
                    save_path=RESULTS_DIR / 'wind_quiver_global.png',
                    stride=20)
    
    # 步骤9：绘制风速频率分布
    print("\n" + "-"*60)
    print("任务7: 绘制风速频率分布")
    print("-"*60)
    plot_wind_speed_histogram(data_filtered,
                             save_path=RESULTS_DIR / 'wind_speed_histogram.png')
    
    # 步骤10：绘制风向玫瑰图
    print("\n" + "-"*60)
    print("任务8: 绘制风向玫瑰图")
    print("-"*60)
    plot_wind_rose(data_filtered,
                  save_path=RESULTS_DIR / 'wind_rose.png')
    
    # 步骤11：多文件对比
    if len(h5_files) > 1:
        print("\n" + "-"*60)
        print("任务9: 多文件风场对比")
        print("-"*60)
        plot_multiple_scatter(h5_files, max_files=4,
                             save_path=RESULTS_DIR / 'wind_multiple_comparison.png')
    
    print("\n" + "="*70)
    print("实验3完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果保存在: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
