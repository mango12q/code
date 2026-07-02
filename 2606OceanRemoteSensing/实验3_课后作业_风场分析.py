"""
实验3 课后作业：海洋风场遥感数据处理
====================================
基于HY-2B散射计数据和NCEP-CFSR风场数据，完成教材第4.4节要求的课后作业。

作业要求：
1. 删去所转换CSV文件中的所有无效值，或重新存储一个无无效值的CSV文件
2. 结合本章所学内容，对代码略作修改，画出文件夹中所有数据的风速、风向的分布
3. 利用NCEP-CFSR数据产品"ww3.201001_wnd.nc"，画出2010年1月1日UTC-03:00时刻的全球风场分布

依赖库：
    - numpy, matplotlib, cartopy, h5py, netCDF4, pandas

作者：基于《海洋遥感数据处理实践初级教程》改进
Python版本：3.13+
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import h5py
import netCDF4 as nc
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# 配置区域
# =============================================================================

BASE_DIR = Path(r"E:\Python")
DATA_DIR = BASE_DIR / "data"
H5_DATA_DIR = DATA_DIR  # HY-2B HDF5文件目录
NCEP_FILE = DATA_DIR / "ww3.201001_wnd.nc"

RESULTS_DIR = BASE_DIR / "results" / "实验3_课后作业"

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
    """查找目录中的所有HDF5文件。"""
    files = sorted(directory.glob("*.h5"))
    print(f"[INFO] 找到 {len(files)} 个HDF5文件")
    return files


def read_hy2b_wind_data(filepath: Path) -> dict:
    """
    读取HY-2B散射计风场数据。
    返回展平后的一维数组。
    """
    print(f"[INFO] 正在读取: {filepath.name}")
    
    with h5py.File(filepath, 'r') as f:
        # 读取数据
        lon_raw = np.array(f['wvc_lon'][:])
        lat_raw = np.array(f['wvc_lat'][:])
        ws_select = np.array(f['wind_speed_selection'][:])
        wd_select = np.array(f['wind_dir_selection'][:])
        quality = np.array(f['wvc_quality_flag'][:])
        
        # 处理填充值
        FILL_LONLAT = 1.7e38
        FILL_WIND = -32767
        
        valid_mask = (lon_raw < FILL_LONLAT) & (lat_raw < FILL_LONLAT) & \
                     (ws_select != FILL_WIND) & (wd_select != FILL_WIND) & \
                     (quality >= 0)
        
        print(f"  原始数据形状: {lon_raw.shape} (scans, cells)")
        print(f"  有效数据点数: {np.sum(valid_mask)}/{valid_mask.size}")
        
        # 展平并筛选有效数据
        data = {
            'lon': lon_raw[valid_mask],
            'lat': lat_raw[valid_mask],
            'wind_speed': ws_select[valid_mask],
            'wind_dir': wd_select[valid_mask],
            'quality': quality[valid_mask]
        }
        
        # 检查是否有scale factor
        if 'scale_factor' in f['wind_speed_selection'].attrs:
            scale = f['wind_speed_selection'].attrs['scale_factor']
            data['wind_speed'] = data['wind_speed'] * scale
            print(f"  风速缩放因子: {scale}")
        
        if len(data['lon']) > 0:
            print(f"  风速范围: {data['wind_speed'].min():.2f} ~ {data['wind_speed'].max():.2f} m/s")
            print(f"  风向范围: {data['wind_dir'].min():.2f}° ~ {data['wind_dir'].max():.2f}°")
        
        return data


# =============================================================================
# 作业1: 清理CSV文件中的无效值
# =============================================================================

def homework_1_clean_csv() -> None:
    """
    作业1：读取HY-2B HDF5数据，转换为无无效值的CSV文件。
    """
    print("\n" + "-"*60)
    print("作业1: 清理CSV文件中的无效值")
    print("-"*60)
    
    h5_files = find_h5_files(H5_DATA_DIR)
    
    if not h5_files:
        print("[WARNING] 未找到HDF5文件，跳过作业1")
        return
    
    # 读取第一个文件作为示例
    data = read_hy2b_wind_data(h5_files[0])
    
    if len(data.get('lon', [])) == 0:
        print("[WARNING] 无有效数据")
        return
    
    # 保存为CSV文件（仅有效数据）
    output_csv = RESULTS_DIR / 'clean_wind_data.csv'
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['Longitude', 'Latitude', 'Wind_Speed(m/s)', 'Wind_Direction(deg)', 'Quality_Flag'])
        
        # 写入数据
        for i in range(len(data['lon'])):
            writer.writerow([
                f"{data['lon'][i]:.4f}",
                f"{data['lat'][i]:.4f}",
                f"{data['wind_speed'][i]:.2f}",
                f"{data['wind_dir'][i]:.2f}",
                f"{data['quality'][i]}"
            ])
    
    print(f"[INFO] 已保存清理后的CSV文件: {output_csv}")
    print(f"[INFO] 共 {len(data['lon'])} 条有效记录")
    
    # 同时保存所有文件的合并数据
    if len(h5_files) > 1:
        all_data = {
            'lon': [], 'lat': [], 'wind_speed': [], 'wind_dir': [], 'quality': []
        }
        
        for h5_file in h5_files:
            d = read_hy2b_wind_data(h5_file)
            if len(d.get('lon', [])) > 0:
                all_data['lon'].extend(d['lon'])
                all_data['lat'].extend(d['lat'])
                all_data['wind_speed'].extend(d['wind_speed'])
                all_data['wind_dir'].extend(d['wind_dir'])
                all_data['quality'].extend(d['quality'])
        
        # 转换为numpy数组
        for key in all_data:
            all_data[key] = np.array(all_data[key])
        
        output_csv_all = RESULTS_DIR / 'clean_wind_data_all_files.csv'
        with open(output_csv_all, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Longitude', 'Latitude', 'Wind_Speed(m/s)', 'Wind_Direction(deg)', 'Quality_Flag'])
            
            for i in range(len(all_data['lon'])):
                writer.writerow([
                    f"{all_data['lon'][i]:.4f}",
                    f"{all_data['lat'][i]:.4f}",
                    f"{all_data['wind_speed'][i]:.2f}",
                    f"{all_data['wind_dir'][i]:.2f}",
                    f"{all_data['quality'][i]}"
                ])
        
        print(f"[INFO] 已保存所有文件的合并CSV: {output_csv_all}")
        print(f"[INFO] 共 {len(all_data['lon'])} 条有效记录")


# =============================================================================
# 作业2: 批量绘制所有文件的风速、风向分布
# =============================================================================

def homework_2_batch_distribution() -> None:
    """
    作业2：绘制文件夹中所有数据的风速、风向分布。
    """
    print("\n" + "-"*60)
    print("作业2: 批量绘制风速/风向分布")
    print("-"*60)
    
    h5_files = find_h5_files(H5_DATA_DIR)
    
    if not h5_files:
        print("[WARNING] 未找到HDF5文件，跳过作业2")
        return
    
    # 合并所有文件数据
    all_speeds = []
    all_directions = []
    
    for h5_file in h5_files:
        data = read_hy2b_wind_data(h5_file)
        if len(data.get('lon', [])) > 0:
            all_speeds.extend(data['wind_speed'])
            all_directions.extend(data['wind_dir'])
    
    all_speeds = np.array(all_speeds)
    all_directions = np.array(all_directions)
    
    if len(all_speeds) == 0:
        print("[WARNING] 无有效数据")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 风速频率分布
    mean_speed = np.mean(all_speeds)
    median_speed = np.median(all_speeds)
    std_speed = np.std(all_speeds)
    
    axes[0].hist(all_speeds, bins=50, color='steelblue', alpha=0.7,
                edgecolor='black', linewidth=0.5)
    axes[0].axvline(mean_speed, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_speed:.2f} m/s')
    axes[0].axvline(median_speed, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_speed:.2f} m/s')
    axes[0].set_xlabel('Wind Speed (m/s)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Wind Speed Distribution (All Files)\n'
                     f'Mean={mean_speed:.2f}, Std={std_speed:.2f}, N={len(all_speeds)}',
                     fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 风向频率分布
    axes[1].hist(all_directions, bins=36, color='coral', alpha=0.7,
                edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Wind Direction (°)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Wind Direction Distribution (All Files)\n'
                     f'N={len(all_directions)}',
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 360)
    
    fig.suptitle(f'HY-2B Wind Field Statistics ({len(h5_files)} files)',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = RESULTS_DIR / 'wind_distribution_batch.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()


# =============================================================================
# 作业3: NCEP-CFSR全球风场分布
# =============================================================================

def homework_3_ncep_wind_field() -> None:
    """
    作业3：使用ww3.201001_wnd.nc绘制2010年1月1日UTC-03:00时刻的全球风场分布。
    
    时间说明：
    - 文件时间单位：days since 1990-01-01T00:00:00Z
    - 2010-01-01 00:00 UTC = 7305.0 days
    - 2010-01-01 03:00 UTC = 7305.125 days
    - 数据为逐小时，时间步长约0.0416667天（1小时）
    - 第3个时间步（索引2）对应 2010-01-01 02:00，接近03:00
    """
    print("\n" + "-"*60)
    print("作业3: NCEP-CFSR全球风场分布 (2010-01-01 UTC-03:00)")
    print("-"*60)
    
    if not NCEP_FILE.exists():
        print(f"[WARNING] NCEP数据文件不存在: {NCEP_FILE}")
        print("[HINT] 请确认数据文件已放置到正确位置")
        return
    
    print(f"[INFO] 读取NCEP数据: {NCEP_FILE}")
    
    with nc.Dataset(NCEP_FILE, 'r') as ds:
        # 读取变量
        lon = ds.variables['longitude'][:]
        lat = ds.variables['latitude'][:]
        time = ds.variables['time'][:]
        uwnd = ds.variables['uwnd'][:]  # 东向风分量
        vwnd = ds.variables['vwnd'][:]  # 北向风分量
        
        print(f"  时间维度: {len(time)} 个时间步")
        print(f"  空间维度: {len(lat)} x {len(lon)}")
        print(f"  时间范围: {time.min():.4f} ~ {time.max():.4f} days")
        
        # 计算风速和风向
        wind_speed = np.sqrt(uwnd**2 + vwnd**2)
        wind_dir = np.degrees(np.arctan2(-uwnd, -vwnd))  # 气象风向（风的来向）
        wind_dir = np.where(wind_dir < 0, wind_dir + 360, wind_dir)
        
        # 找到2010-01-01 03:00对应的时间索引
        # 7305.0 = 2010-01-01 00:00, 7305.125 = 2010-01-01 03:00
        target_time = 7305.125
        time_idx = np.argmin(np.abs(time - target_time))
        actual_time = time[time_idx]
        
        print(f"[INFO] 目标时间: {target_time:.4f} days (2010-01-01 03:00)")
        print(f"[INFO] 实际使用: {actual_time:.4f} days (索引 {time_idx})")
        
        # 提取该时刻的数据
        speed_2d = wind_speed[time_idx, :, :]
        u_2d = uwnd[time_idx, :, :]
        v_2d = vwnd[time_idx, :, :]
        
        # 创建经纬度网格
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # 绘制全球风场
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
        
        # 绘制风速背景
        im = ax.pcolormesh(lon_grid, lat_grid, speed_2d,
                          transform=ccrs.PlateCarree(),
                          cmap='jet', vmin=0, vmax=25,
                          shading='auto')
        
        # 绘制风向矢量（采样显示）
        stride = 10  # 采样间隔
        ax.quiver(lon_grid[::stride, ::stride], lat_grid[::stride, ::stride],
                 u_2d[::stride, ::stride], v_2d[::stride, ::stride],
                 scale=500, width=0.002,
                 transform=ccrs.PlateCarree(),
                 color='white', alpha=0.7)
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                           pad=0.05, shrink=0.8, aspect=30)
        cbar.set_label('Wind Speed (m/s)', fontsize=12)
        
        # 标题
        ax.set_title(f'Global Wind Field (NCEP-CFSR)\n'
                    f'2010-01-01 UTC 03:00\n'
                    f'Wind Speed & Direction',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_file = RESULTS_DIR / 'ncep_global_wind_20100101_0300.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {output_file}")
        
        plt.show()
        
        # 同时绘制风速和风向的单独分布图
        fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                                subplot_kw={'projection': ccrs.Robinson()})
        
        # 风速分布
        im1 = axes[0].pcolormesh(lon_grid, lat_grid, speed_2d,
                                transform=ccrs.PlateCarree(),
                                cmap='jet', vmin=0, vmax=25, shading='auto')
        axes[0].add_feature(cfeature.COASTLINE, linewidth=0.8)
        axes[0].add_feature(cfeature.LAND, facecolor='lightgray')
        axes[0].set_title('Wind Speed Distribution', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.05,
                    shrink=0.8, label='Wind Speed (m/s)')
        
        # 风向分布
        im2 = axes[1].pcolormesh(lon_grid, lat_grid, wind_dir[time_idx, :, :],
                                transform=ccrs.PlateCarree(),
                                cmap='hsv', vmin=0, vmax=360, shading='auto')
        axes[1].add_feature(cfeature.COASTLINE, linewidth=0.8)
        axes[1].add_feature(cfeature.LAND, facecolor='lightgray')
        axes[1].set_title('Wind Direction Distribution', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.05,
                    shrink=0.8, label='Wind Direction (°)')
        
        fig.suptitle(f'NCEP-CFSR Wind Field Components (2010-01-01 03:00 UTC)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_file2 = RESULTS_DIR / 'ncep_wind_components_20100101_0300.png'
        plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {output_file2}")
        
        plt.show()


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主函数：执行实验3所有课后作业。"""
    print("="*70)
    print("实验3 课后作业：海洋风场遥感数据处理")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # 步骤1：创建输出目录
    setup_directories()
    
    # 作业1：清理CSV
    homework_1_clean_csv()
    
    # 作业2：批量分布
    homework_2_batch_distribution()
    
    # 作业3：NCEP风场
    homework_3_ncep_wind_field()
    
    print("\n" + "="*70)
    print("实验3 课后作业全部完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果保存在: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
