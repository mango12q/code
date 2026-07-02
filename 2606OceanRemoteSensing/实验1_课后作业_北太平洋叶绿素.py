"""
实验1 课后作业：北太平洋叶绿素浓度分布
========================================
基于MODIS Aqua叶绿素浓度数据，绘制北太平洋区域的可视化分析图。

作业要求（教材第2.4节）：
1. 理解本章中每一句代码的含义，对画图代码进行整理
2. 在此基础上绘制北太平洋叶绿素浓度分布

依赖库：
    - numpy: 数值计算
    - matplotlib: 绘图
    - cartopy: 地图投影
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
import netCDF4 as nc

# 使用Cartopy替代Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# 配置区域
# =============================================================================

# 数据文件路径
BASE_DIR = Path(r"E:\Python")
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "A20160652016072.L3m_8D_CHL_chlor_a_9km.nc"

# 结果输出目录
RESULTS_DIR = BASE_DIR / "results" / "实验1_课后作业"

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


def read_chlorophyll_data(filepath: Path) -> dict:
    """
    读取NetCDF格式的叶绿素浓度数据。
    
    Args:
        filepath: NetCDF文件路径
        
    Returns:
        dict: 包含叶绿素数据、经纬度等信息的字典
    """
    print("[INFO] 正在读取叶绿素数据...")
    
    path_str = str(filepath.absolute())
    
    with nc.Dataset(path_str, 'r') as dataset:
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
        
        return {
            'chlor_a': np.array(chlor_a),
            'lon': np.array(lon),
            'lat': np.array(lat),
            'attrs': chlor_attrs
        }


def plot_north_pacific_chlorophyll(data: dict, log_scale: bool = True,
                                   cmap: str = 'jet', save: bool = True) -> None:
    """
    绘制北太平洋区域叶绿素浓度分布图。
    
    北太平洋范围：经度 120°E ~ 100°W (120°~260°E)，纬度 0°N ~ 60°N
    使用 PlateCarree 投影，中心经度为 180°
    
    Args:
        data: 包含chlor_a, lon, lat的字典
        log_scale: 是否使用对数刻度
        cmap: 颜色映射方案
        save: 是否保存图片
    """
    print("[INFO] 正在绘制北太平洋叶绿素分布图...")
    
    chlor = data['chlor_a']
    lons = data['lon']
    lats = data['lat']
    
    # 创建经纬度网格
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 处理数据：对数转换
    if log_scale:
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
    
    # 创建图形 - 使用PlateCarree投影，中心经度180°
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    
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
    
    # 设置显示范围：北太平洋
    # 经度 120°E ~ 100°W，纬度 0°N ~ 60°N
    ax.set_extent([120, 260, 0, 60], crs=ccrs.PlateCarree())
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label(label, fontsize=12)
    
    # 设置标题
    ax.set_title(f'North Pacific Chlorophyll-a Distribution{title_suffix}\n'
                f'MODIS-Aqua 8-Day Composite (2016)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save:
        output_file = RESULTS_DIR / 'north_pacific_chlorophyll.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white')
        print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主函数：执行实验1课后作业。"""
    print("="*70)
    print("实验1 课后作业：北太平洋叶绿素浓度分布")
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
        data = read_chlorophyll_data(DATA_FILE)
    except Exception as e:
        print(f"[ERROR] 数据读取失败: {e}")
        sys.exit(1)
    
    # 步骤4：绘制北太平洋叶绿素分布图（对数刻度）
    plot_north_pacific_chlorophyll(data, log_scale=True, cmap='jet', save=True)
    
    print("\n" + "="*70)
    print("实验1 课后作业完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果保存在: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
