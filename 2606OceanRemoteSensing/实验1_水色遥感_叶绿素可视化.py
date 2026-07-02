"""
实验1：水色遥感数据的可视化（改进版）
========================================
基于MODIS Aqua叶绿素浓度数据，进行全球和中国海区域的可视化分析。

改进点：
1. 使用Cartopy替代已弃用的Basemap进行地图投影
2. 函数化封装，提高代码复用性
3. 增加数据验证和错误处理
4. 自动创建结果输出目录
5. 详细的注释和文档字符串
6. 使用pathlib处理路径，提高跨平台可移植性

依赖库：
    - numpy: 数值计算
    - matplotlib: 绘图
    - cartopy: 地图投影（替代Basemap）
    - netCDF4: NetCDF文件读写

作者：基于《海洋遥感数据处理实践初级教程》改进
Python版本：3.13+
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import netCDF4 as nc

# 使用Cartopy替代Basemap（Basemap已弃用）
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# 配置区域
# =============================================================================

# 数据文件路径（使用绝对路径避免工作目录问题）
# 修改此处为您的实际数据路径
BASE_DIR = Path(r"E:\Python")
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "A20160652016072.L3m_8D_CHL_chlor_a_9km.nc"

# 结果输出目录
RESULTS_DIR = BASE_DIR / "results" / "实验1_水色遥感"

# 设置中文字体支持（Windows系统）
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
    """
    检查数据文件是否存在。
    
    Args:
        filepath: 数据文件路径
        
    Returns:
        bool: 文件是否存在
    """
    if not filepath.exists():
        print(f"[ERROR] 数据文件不存在: {filepath}")
        print(f"[HINT] 请确认数据文件位于: {filepath}")
        return False
    print(f"[INFO] 数据文件已找到: {filepath}")
    return True


def read_chlorophyll_data(filepath: Path) -> dict:
    """
    读取NetCDF格式的叶绿素浓度数据。
    
    Args:
        filepath: NetCDF文件路径
        
    Returns:
        dict: 包含叶绿素数据、经纬度等信息的字典
        
    Raises:
        FileNotFoundError: 文件不存在
        KeyError: 变量名不匹配
    """
    print("[INFO] 正在读取数据...")
    
    # 调试：打印路径信息
    print(f"[DEBUG] 传入路径: {filepath}")
    print(f"[DEBUG] 路径类型: {type(filepath)}")
    print(f"[DEBUG] 绝对路径: {filepath.absolute()}")
    print(f"[DEBUG] 路径是否存在: {filepath.exists()}")
    print(f"[DEBUG] 是否为文件: {filepath.is_file()}")
    
    # 转换为字符串路径（netCDF4可能需要字符串）
    path_str = str(filepath.absolute())
    print(f"[DEBUG] 字符串路径: {path_str}")
    
    with nc.Dataset(path_str, 'r') as dataset:
        # 打印数据基本信息
        print("\n" + "="*60)
        print("数据文件基本信息")
        print("="*60)
        print(f"文件格式: {dataset.data_model}")
        print(f"全局属性:")
        for attr in ['title', 'instrument', 'platform', 'temporal_range']:
            if hasattr(dataset, attr):
                print(f"  {attr}: {getattr(dataset, attr)}")
        
        # 读取变量
        chlor_a = dataset.variables['chlor_a'][:]
        lon = dataset.variables['lon'][:]
        lat = dataset.variables['lat'][:]
        
        # 获取变量属性
        chlor_attrs = {attr: getattr(dataset.variables['chlor_a'], attr, None) 
                      for attr in ['long_name', 'units', 'valid_min', 'valid_max', '_FillValue']}
        
        print(f"\n变量信息:")
        print(f"  chlor_a 形状: {chlor_a.shape}")
        print(f"  经度范围: {lon.min():.2f} ~ {lon.max():.2f}")
        print(f"  纬度范围: {lat.min():.2f} ~ {lat.max():.2f}")
        print(f"  单位: {chlor_attrs.get('units', 'N/A')}")
        print("="*60 + "\n")
        
        return {
            'chlor_a': np.array(chlor_a),
            'lon': np.array(lon),
            'lat': np.array(lat),
            'attrs': chlor_attrs
        }


def create_map_projection(projection_type: str = 'global', 
                         central_longitude: float = 0,
                         extent: list = None) -> tuple:
    """
    创建地图投影。
    
    Args:
        projection_type: 投影类型 ('global', 'regional')
        central_longitude: 中心经度
        extent: 区域范围 [lon_min, lon_max, lat_min, lat_max]
        
    Returns:
        tuple: (fig, ax) matplotlib图形对象
    """
    if projection_type == 'global':
        proj = ccrs.Robinson(central_longitude=central_longitude)
    else:
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    
    # 添加网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                      linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    # 设置显示范围
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    return fig, ax


def plot_chlorophyll_global(data: dict, log_scale: bool = True, 
                            cmap: str = 'jet', save: bool = True) -> None:
    """
    绘制全球叶绿素浓度分布图。
    
    Args:
        data: 包含chlor_a, lon, lat的字典
        log_scale: 是否使用对数刻度（叶绿素浓度通常用对数显示）
        cmap: 颜色映射方案
        save: 是否保存图片
    """
    print("[INFO] 正在绘制全球叶绿素分布图...")
    
    chlor = data['chlor_a']
    lons = data['lon']
    lats = data['lat']
    
    # 创建经纬度网格
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 处理数据：对数转换（叶绿素浓度范围大，对数显示更清晰）
    if log_scale:
        # 处理无效值和负数
        chlor_plot = np.where(chlor > 0, chlor, np.nan)
        chlor_plot = np.log10(chlor_plot)
        vmin, vmax = -2, 1.5  # log10 scale
        label = 'Log10 Chlorophyll-a (mg/m³)'
        title_suffix = ' (Log Scale)'
    else:
        chlor_plot = np.where(chlor > 0, chlor, np.nan)
        vmin, vmax = 0.01, 20
        label = 'Chlorophyll-a (mg/m³)'
        title_suffix = ''
    
    # 创建图形
    fig, ax = create_map_projection(projection_type='global', 
                                    central_longitude=0)
    
    # 绘制数据
    im = ax.pcolormesh(lon_grid, lat_grid, chlor_plot, 
                       transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label(label, fontsize=12)
    
    # 设置标题
    ax.set_title(f'Global Distribution of Chlorophyll-a Concentration{title_suffix}\n'
                f'MODIS-Aqua 8-Day Composite (2016)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save:
        output_file = RESULTS_DIR / 'global_chlorophyll_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white')
        print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()


def plot_chlorophyll_regional(data: dict, region: str = 'china_sea',
                              log_scale: bool = True, cmap: str = 'jet',
                              save: bool = True) -> None:
    """
    绘制区域叶绿素浓度分布图。
    
    Args:
        data: 包含chlor_a, lon, lat的字典
        region: 区域名称 ('china_sea', 'north_pacific'等)
        log_scale: 是否使用对数刻度
        cmap: 颜色映射方案
        save: 是否保存图片
    """
    print(f"[INFO] 正在绘制{region}区域叶绿素分布图...")
    
    # 定义区域范围
    regions = {
        'china_sea': {
            'extent': [102, 132, 0, 45],
            'title': 'China Sea',
            'projection': ccrs.LambertConformal(central_latitude=22.5, 
                                               central_longitude=117)
        },
        'north_pacific': {
            'extent': [120, 260, 0, 60],
            'title': 'North Pacific',
            'projection': ccrs.PlateCarree(central_longitude=180)
        },
        'global_pacific': {
            'extent': [0, 360, -80, 80],
            'title': 'Global (Pacific Centered)',
            'projection': ccrs.PlateCarree(central_longitude=180)
        }
    }
    
    if region not in regions:
        raise ValueError(f"未知区域: {region}. 可用区域: {list(regions.keys())}")
    
    region_info = regions[region]
    
    chlor = data['chlor_a']
    lons = data['lon']
    lats = data['lat']
    
    # 处理经度范围（支持0-360度）
    if region_info['extent'][1] > 180:
        # 需要转换经度范围
        lons_shifted = np.where(lons < 0, lons + 360, lons)
        sort_idx = np.argsort(lons_shifted)
        lons_plot = lons_shifted[sort_idx]
        chlor_plot = chlor[:, sort_idx]
    else:
        lons_plot = lons
        chlor_plot = chlor
    
    # 创建经纬度网格
    lon_grid, lat_grid = np.meshgrid(lons_plot, lats)
    
    # 数据处理
    if log_scale:
        chlor_plot = np.where(chlor_plot > 0, chlor_plot, np.nan)
        chlor_plot = np.log10(chlor_plot)
        vmin, vmax = -2, 1.5
        label = 'Log10 Chlorophyll-a (mg/m³)'
        title_suffix = ' (Log Scale)'
    else:
        chlor_plot = np.where(chlor_plot > 0, chlor_plot, np.nan)
        vmin, vmax = 0.01, 20
        label = 'Chlorophyll-a (mg/m³)'
        title_suffix = ''
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=region_info['projection'])
    
    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    
    # 添加网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                      linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    # 绘制数据
    im = ax.pcolormesh(lon_grid, lat_grid, chlor_plot, 
                       transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto')
    
    # 设置显示范围
    ax.set_extent(region_info['extent'], crs=ccrs.PlateCarree())
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label(label, fontsize=12)
    
    # 设置标题
    ax.set_title(f'{region_info["title"]} Chlorophyll-a Distribution{title_suffix}\n'
                f'MODIS-Aqua 8-Day Composite (2016)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save:
        output_file = RESULTS_DIR / f'{region}_chlorophyll_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white')
        print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()


def plot_chlorophyll_comparison(data: dict, save: bool = True) -> None:
    """
    绘制线性刻度与对数刻度的对比图。
    
    Args:
        data: 包含chlor_a, lon, lat的字典
        save: 是否保存图片
    """
    print("[INFO] 正在绘制线性/对数刻度对比图...")
    
    chlor = data['chlor_a']
    lons = data['lon']
    lats = data['lat']
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), 
                            subplot_kw={'projection': ccrs.Robinson()})
    
    # 线性刻度
    chlor_linear = np.where(chlor > 0, chlor, np.nan)
    im1 = axes[0].pcolormesh(lon_grid, lat_grid, chlor_linear, 
                             transform=ccrs.PlateCarree(),
                             cmap='jet', vmin=0.01, vmax=20,
                             shading='auto')
    axes[0].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[0].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[0].set_title('Linear Scale', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.05, 
                shrink=0.8, label='Chlorophyll-a (mg/m³)')
    
    # 对数刻度
    chlor_log = np.where(chlor > 0, chlor, np.nan)
    chlor_log = np.log10(chlor_log)
    im2 = axes[1].pcolormesh(lon_grid, lat_grid, chlor_log, 
                             transform=ccrs.PlateCarree(),
                             cmap='jet', vmin=-2, vmax=1.5,
                             shading='auto')
    axes[1].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[1].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[1].set_title('Log10 Scale', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.05, 
                shrink=0.8, label='Log10 Chlorophyll-a (mg/m³)')
    
    fig.suptitle('Chlorophyll-a Concentration: Linear vs Log Scale\n'
                'MODIS-Aqua 8-Day Composite (2016)', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save:
        output_file = RESULTS_DIR / 'chlorophyll_linear_vs_log.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white')
        print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主函数：执行实验1的所有步骤。"""
    print("="*70)
    print("实验1：水色遥感数据的可视化")
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
        data = read_chlorophyll_data(DATA_FILE)
    except Exception as e:
        print(f"\n[ERROR] 数据读取失败: {e}")
        sys.exit(1)
    
    # 步骤4：绘制全球分布图（对数刻度）
    plot_chlorophyll_global(data, log_scale=True, cmap='jet', save=True)
    
    # 步骤5：绘制线性/对数对比图
    plot_chlorophyll_comparison(data, save=True)
    
    # 步骤6：绘制中国海区域
    plot_chlorophyll_regional(data, region='china_sea', 
                             log_scale=True, cmap='jet', save=True)
    
    # 步骤7：绘制北太平洋区域（以太平洋为中心）
    plot_chlorophyll_regional(data, region='north_pacific', 
                             log_scale=True, cmap='jet', save=True)
    
    print("\n" + "="*70)
    print("实验1完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果保存在: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
