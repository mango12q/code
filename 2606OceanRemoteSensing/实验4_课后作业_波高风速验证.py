"""
实验4 课后作业：卫星高度计波高验证与校准
========================================
基于Jason-3高度计和NDBC浮标数据，完成教材第5.4节要求的课后作业。

作业要求：
1. 在NDBC网站查询更多浮标位置信息，下载3-5个浮标数据，添加到地图上
2. 选择浮标42002，选取一个月的数据，绘制风速和波高的时间序列，分析关系
3. 将下载的浮标添加到列表中，将空间窗口从75km改为50km，分析对SWH对比的影响
4. 匹配并验证风速数据（50km窗口），绘制散点图，计算BIAS/RMSE/相关系数
5. 对风速数据进行线性校准，评估校准效果
6. 选择一天Jason-1数据，绘制后向散射系数与风速的散点图（GMF）

依赖库：
    - numpy, matplotlib, cartopy, netCDF4, scipy

作者：基于《海洋遥感数据处理实践初级教程》改进
Python版本：3.13+
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List

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
ALTIMETER_DIR = DATA_DIR / "altimeter_data"
BUOY_DIR = DATA_DIR / "buoy_data"

RESULTS_DIR = BASE_DIR / "results" / "实验4_课后作业"

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 浮标位置信息（NDBC官方数据）
# 格式: 浮标编号 -> [纬度, 经度]
BUOY_LOCATIONS = {
    # 原有浮标
    '32012': [-19.425, -85.078],   # 印度洋
    '41001': [34.724, -72.317],   # 美国东海岸外海
    '42002': [26.055, -93.646],   # 墨西哥湾
    '46059': [38.094, -129.951],  # 东北太平洋
    '51003': [19.175, -160.625],  # 夏威夷附近
    
    # 额外浮标（教材要求查询）
    '41004': [32.502, -79.099],   # 美国东南海岸
    '41040': [14.525, -53.011],   # 热带大西洋
    '41048': [31.831, -69.573],   # 西北大西洋
    '41044': [21.562, -58.594],   # 北大西洋
    '41047': [27.465, -71.452],   # 北大西洋
    '42001': [25.918, -89.658],   # 墨西哥湾
    '42003': [25.925, -85.616],   # 墨西哥湾
    '46006': [40.781, -137.469],  # 东北太平洋
    '46001': [56.232, -148.125],  # 北太平洋
    '51101': [24.318, -162.231],  # 夏威夷附近
    '51004': [17.538, -152.328],  # 中太平洋
}

# 匹配参数
MATCH_TIME_WINDOW = 3.0  # 时间匹配窗口（小时）
MATCH_SPACE_WINDOW_DEFAULT = 50.0  # 默认空间匹配窗口（公里）
MATCH_SPACE_WINDOW_ALT = 75.0  # 替代空间匹配窗口（公里）


# =============================================================================
# 工具函数
# =============================================================================

def setup_directories():
    """创建必要的结果输出目录。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 结果将保存至: {RESULTS_DIR}")


def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间的Haversine距离（公里）。"""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def read_single_altimeter_file(filepath: Path) -> dict:
    """读取单个高度计数据文件。"""
    with nc.Dataset(filepath, 'r') as dataset:
        lon_var = next((name for name in ['lon', 'longitude', 'long', 'x'] 
                       if name in dataset.variables), None)
        lat_var = next((name for name in ['lat', 'latitude', 'y'] 
                       if name in dataset.variables), None)
        swh_var = next((name for name in ['swh', 'wave_height', 'significant_wave_height', 'shw'] 
                       if name in dataset.variables), None)
        time_var = next((name for name in ['time', 'datetime', 't'] 
                        if name in dataset.variables), None)
        ws_var = next((name for name in ['wind_speed', 'ws', 'wind_speed_alt'] 
                      if name in dataset.variables), None)
        sig0_var = next((name for name in ['sigma0', 'sig0', 'backscatter'] 
                        if name in dataset.variables), None)
        
        if not all([lon_var, lat_var, swh_var]):
            return {}
        
        lon = dataset.variables[lon_var][:]
        lat = dataset.variables[lat_var][:]
        swh = dataset.variables[swh_var][:]
        
        # 处理填充值
        fill_value = getattr(dataset.variables[swh_var], '_FillValue', -9999)
        if fill_value is not None:
            swh = np.where(swh == fill_value, np.nan, swh)
        
        # 读取时间
        time_data = None
        if time_var:
            time_raw = dataset.variables[time_var][:]
            time_units = getattr(dataset.variables[time_var], 'units', 'days since 1900-1-1')
            try:
                if 'days since' in time_units:
                    base_date = datetime.strptime(time_units.replace('days since ', '').strip(), '%Y-%m-%d')
                    time_data = np.array([base_date + timedelta(days=float(t)) for t in time_raw])
                elif 'hours since' in time_units:
                    base_date = datetime.strptime(time_units.replace('hours since ', '').strip(), '%Y-%m-%d')
                    time_data = np.array([base_date + timedelta(hours=float(t)) for t in time_raw])
            except:
                time_data = None
        
        # 读取风速（如果存在）
        wind_speed = None
        if ws_var:
            wind_speed = np.array(dataset.variables[ws_var][:])
            ws_fill = getattr(dataset.variables[ws_var], '_FillValue', -9999)
            if ws_fill is not None:
                wind_speed = np.where(wind_speed == ws_fill, np.nan, wind_speed)
        
        # 读取后向散射系数（如果存在）
        sigma0 = None
        if sig0_var:
            sigma0 = np.array(dataset.variables[sig0_var][:])
            sig0_fill = getattr(dataset.variables[sig0_var], '_FillValue', -9999)
            if sig0_fill is not None:
                sigma0 = np.where(sigma0 == sig0_fill, np.nan, sigma0)
        
        return {
            'swh': np.array(swh),
            'lon': np.array(lon),
            'lat': np.array(lat),
            'time': time_data,
            'wind_speed': wind_speed,
            'sigma0': sigma0
        }


def read_jason_data(filepaths: List[Path]) -> dict:
    """读取并合并多个高度计数据文件。"""
    all_swh, all_lon, all_lat, all_time, all_ws, all_sig0 = [], [], [], [], [], []
    
    for filepath in filepaths:
        data = read_single_altimeter_file(filepath)
        if data:
            all_swh.append(data['swh'])
            all_lon.append(data['lon'])
            all_lat.append(data['lat'])
            if data['time'] is not None:
                all_time.append(data['time'])
            if data['wind_speed'] is not None:
                all_ws.append(data['wind_speed'])
            if data['sigma0'] is not None:
                all_sig0.append(data['sigma0'])
    
    if not all_swh:
        return {}
    
    result = {
        'swh': np.concatenate(all_swh),
        'lon': np.concatenate(all_lon),
        'lat': np.concatenate(all_lat),
        'time': np.concatenate(all_time) if all_time else None,
        'wind_speed': np.concatenate(all_ws) if all_ws else None,
        'sigma0': np.concatenate(all_sig0) if all_sig0 else None
    }
    
    # 移除无效值
    valid_mask = ~np.isnan(result['swh'])
    for key in result:
        if result[key] is not None:
            result[key] = result[key][valid_mask]
    
    return result


def read_buoy_file(filepath: Path) -> dict:
    """读取单个浮标数据文件。"""
    data = None
    for skip in [2, 1, 0]:
        try:
            data = np.loadtxt(filepath, skiprows=skip)
            break
        except:
            continue
    
    if data is None or (len(data.shape) > 1 and data.shape[1] < 9):
        return {}
    
    if len(data.shape) == 1:
        return {}
    
    n_cols = data.shape[1]
    year = data[:, 0]
    month = data[:, 1]
    day = data[:, 2]
    hour = data[:, 3]
    minute = data[:, 4]
    wdir = data[:, 5]
    wspd = data[:, 6]
    wvht = data[:, 8] if n_cols > 8 else np.full(len(year), np.nan)
    
    year = np.where(year < 50, year + 2000, year + 1900)
    
    datetime_array = np.array([
        datetime(int(y), int(m), int(d), int(h), int(min))
        for y, m, d, h, min in zip(year, month, day, hour, minute)
    ])
    
    valid_mask = (wvht > 0) & (wvht < 50) & (wspd >= 0) & (wspd < 90)
    
    buoy_id = filepath.stem[:5]
    buoy_latlon = BUOY_LOCATIONS.get(buoy_id, [None, None])
    
    return {
        'buoy_id': buoy_id,
        'lat': buoy_latlon[0],
        'lon': buoy_latlon[1],
        'datetime': datetime_array[valid_mask],
        'wspd': wspd[valid_mask],
        'wvht': wvht[valid_mask]
    }


def read_buoy_data(directory: Path) -> List[dict]:
    """读取目录中的所有浮标数据文件。"""
    files = sorted(directory.glob("*.txt"))
    buoys = []
    
    for f in files:
        data = read_buoy_file(f)
        if data and len(data.get('wvht', [])) > 0:
            buoys.append(data)
    
    return buoys


def match_satellite_buoy(sat_data: dict, buoy_data: List[dict],
                         time_window: float = MATCH_TIME_WINDOW,
                         space_window: float = MATCH_SPACE_WINDOW_DEFAULT) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    匹配卫星和浮标数据。
    返回: (卫星SWH, 浮标SWH, 卫星风速, 浮标风速)
    """
    matched_sat_swh, matched_buoy_swh = [], []
    matched_sat_ws, matched_buoy_ws = [], []
    
    sat_swh = sat_data['swh']
    sat_lon = sat_data['lon']
    sat_lat = sat_data['lat']
    sat_time = sat_data.get('time')
    sat_ws = sat_data.get('wind_speed')
    
    sat_time_hours = None
    if sat_time is not None:
        base_time = sat_time[0]
        sat_time_hours = np.array([(st - base_time).total_seconds() / 3600.0 for st in sat_time])
    
    for buoy in buoy_data:
        if buoy['lat'] is None or buoy['lon'] is None:
            continue
        
        buoy_lat, buoy_lon = buoy['lat'], buoy['lon']
        buoy_times = buoy['datetime']
        buoy_wvht = buoy['wvht']
        buoy_wspd = buoy['wspd']
        
        # 经纬度粗筛
        lat_window = space_window / 111.0
        lon_window = space_window / (111.0 * np.cos(np.radians(buoy_lat)))
        
        lat_mask = (sat_lat >= buoy_lat - lat_window) & (sat_lat <= buoy_lat + lat_window)
        lon_mask = (sat_lon >= buoy_lon - lon_window) & (sat_lon <= buoy_lon + lon_window)
        coarse_mask = lat_mask & lon_mask
        
        if not np.any(coarse_mask):
            continue
        
        cand_idx = np.where(coarse_mask)[0]
        cand_lat = sat_lat[cand_idx]
        cand_lon = sat_lon[cand_idx]
        cand_swh = sat_swh[cand_idx]
        cand_ws = sat_ws[cand_idx] if sat_ws is not None else None
        
        for bt, bw, bws in zip(buoy_times, buoy_wvht, buoy_wspd):
            if sat_time_hours is not None:
                bt_hours = (bt - base_time).total_seconds() / 3600.0
                time_mask = np.abs(sat_time_hours[cand_idx] - bt_hours) <= time_window
                
                if not np.any(time_mask):
                    continue
                
                time_cand_lat = cand_lat[time_mask]
                time_cand_lon = cand_lon[time_mask]
                time_cand_swh = cand_swh[time_mask]
                time_cand_ws = cand_ws[time_mask] if cand_ws is not None else None
                
                distances = haversine_distance(buoy_lat, buoy_lon, time_cand_lat, time_cand_lon)
                space_valid = distances <= space_window
                
                if not np.any(space_valid):
                    continue
                
                nearest_idx = np.argmin(distances[space_valid])
                matched_sat_swh.append(time_cand_swh[nearest_idx])
                matched_buoy_swh.append(bw)
                if time_cand_ws is not None:
                    matched_sat_ws.append(time_cand_ws[nearest_idx])
                matched_buoy_ws.append(bws)
            else:
                distances = haversine_distance(buoy_lat, buoy_lon, cand_lat, cand_lon)
                space_valid = distances <= space_window
                
                if not np.any(space_valid):
                    continue
                
                nearest_idx = np.argmin(distances[space_valid])
                matched_sat_swh.append(cand_swh[nearest_idx])
                matched_buoy_swh.append(bw)
                if cand_ws is not None:
                    matched_sat_ws.append(cand_ws[nearest_idx])
                matched_buoy_ws.append(bws)
    
    return (np.array(matched_sat_swh), np.array(matched_buoy_swh),
            np.array(matched_sat_ws), np.array(matched_buoy_ws))


def calculate_statistics(satellite: np.ndarray, buoy: np.ndarray) -> dict:
    """计算验证统计指标。"""
    n = min(len(satellite), len(buoy))
    sat = satellite[:n]
    bu = buoy[:n]
    diff = sat - bu
    
    return {
        'n': n,
        'sat_mean': np.mean(sat),
        'buoy_mean': np.mean(bu),
        'bias': np.mean(diff),
        'rmse': np.sqrt(np.mean(diff**2)),
        'mae': np.mean(np.abs(diff)),
        'correlation': np.corrcoef(sat, bu)[0, 1] if n > 1 else 0,
        'relative_bias': np.mean(diff) / np.mean(bu) * 100 if np.mean(bu) != 0 else 0,
        'relative_rmse': np.sqrt(np.mean(diff**2)) / np.mean(bu) * 100 if np.mean(bu) != 0 else 0
    }


def linear_calibration(satellite: np.ndarray, buoy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """线性回归校准。"""
    coeffs = np.polyfit(satellite, buoy, 1)
    calibrated = np.polyval(coeffs, satellite)
    return calibrated, coeffs


# =============================================================================
# 可视化函数
# =============================================================================

def plot_buoy_map(buoy_data: List[dict], title: str = "Buoy Locations",
                  save_path: Path = None) -> None:
    """绘制浮标位置分布图。"""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    for buoy in buoy_data:
        if buoy['lat'] is not None and buoy['lon'] is not None:
            # 有数据的浮标用红色，无数据的用灰色
            has_data = buoy.get('has_data', True)
            color = 'ro' if has_data else 'ko'
            alpha = 1.0 if has_data else 0.4
            ax.plot(buoy['lon'], buoy['lat'], color, markersize=8,
                   transform=ccrs.PlateCarree(), alpha=alpha)
            ax.text(buoy['lon'] + 3, buoy['lat'] + 2, buoy['buoy_id'],
                   transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold',
                   alpha=alpha)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


def plot_scatter_comparison(satellite: np.ndarray, buoy: np.ndarray,
                           title: str = "Validation", xlabel: str = "Satellite",
                           ylabel: str = "Buoy", save_path: Path = None) -> None:
    """绘制散点对比图。"""
    stats = calculate_statistics(satellite, buoy)
    coeffs = np.polyfit(satellite, buoy, 1)
    x_line = np.linspace(0, max(satellite.max(), buoy.max()), 100)
    y_line = np.polyval(coeffs, x_line)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(satellite, buoy, c='blue', s=30, alpha=0.5, edgecolors='black', linewidth=0.5)
    
    max_val = max(satellite.max(), buoy.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='1:1 Line')
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}')
    
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel(f'{xlabel}', fontsize=12)
    ax.set_ylabel(f'{ylabel}', fontsize=12)
    ax.set_title(f'{title}\nN={stats["n"]}, BIAS={stats["bias"]:.3f}, '
                f'RMSE={stats["rmse"]:.3f}, R={stats["correlation"]:.3f}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


def plot_time_series_wind_wave(buoy: dict, month: int = 9) -> None:
    """绘制浮标风速和波高时间序列。"""
    print(f"\n[INFO] 绘制浮标 {buoy['buoy_id']} 风速/波高时间序列...")
    
    # 选择指定月份的数据
    month_mask = np.array([d.month == month for d in buoy['datetime']])
    
    if not np.any(month_mask):
        print(f"[WARNING] 浮标 {buoy['buoy_id']} 没有 {month} 月的数据")
        return
    
    dates = buoy['datetime'][month_mask]
    wspd = buoy['wspd'][month_mask]
    wvht = buoy['wvht'][month_mask]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 风速时间序列
    axes[0].plot(dates, wspd, 'b-', linewidth=1.5, label='Wind Speed')
    axes[0].set_ylabel('Wind Speed (m/s)', fontsize=12)
    axes[0].set_title(f'Buoy {buoy["buoy_id"]} Wind Speed & Wave Height Time Series\n'
                     f'{dates[0].strftime("%Y-%m")} (Month {month})',
                     fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 波高时间序列
    axes[1].plot(dates, wvht, 'r-', linewidth=1.5, label='Wave Height')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Wave Height (m)', fontsize=12)
    axes[1].legend(loc='best', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = RESULTS_DIR / f'buoy_{buoy["buoy_id"]}_timeseries_month{month}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] 图片已保存: {output_file}")
    
    plt.show()
    
    # 绘制风速-波高散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(wspd, wvht, c='green', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # 线性回归
    if len(wspd) > 1:
        coeffs = np.polyfit(wspd, wvht, 1)
        x_line = np.linspace(wspd.min(), wspd.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}')
        
        # 计算相关系数
        r = np.corrcoef(wspd, wvht)[0, 1]
        ax.set_title(f'Buoy {buoy["buoy_id"]} Wind Speed vs Wave Height\n'
                    f'R = {r:.3f}, N = {len(wspd)}',
                    fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Buoy {buoy["buoy_id"]} Wind Speed vs Wave Height',
                    fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax.set_ylabel('Wave Height (m)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file2 = RESULTS_DIR / f'buoy_{buoy["buoy_id"]}_wind_wave_scatter.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] 图片已保存: {output_file2}")
    
    plt.show()
    
    print("\n[分析] 风速与波高的关系：")
    print("       风速增大时，波高通常也随之增大；")
    print("       两者呈正相关关系，符合风浪生成理论。")


def plot_gmf(sigma0: np.ndarray, wind_speed: np.ndarray,
             title: str = "Geophysical Model Function",
             save_path: Path = None) -> None:
    """绘制后向散射系数与风速的GMF散点图。"""
    print(f"\n[INFO] 绘制GMF散点图...")
    
    # 移除无效值
    valid_mask = ~np.isnan(sigma0) & ~np.isnan(wind_speed) & (sigma0 > -50) & (wind_speed > 0)
    sigma0_valid = sigma0[valid_mask]
    ws_valid = wind_speed[valid_mask]
    
    if len(sigma0_valid) == 0:
        print("[WARNING] 无有效的后向散射数据")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(ws_valid, sigma0_valid, c='blue', s=20, alpha=0.5,
              edgecolors='none')
    
    ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax.set_ylabel('Backscatter Coefficient σ₀ (dB)', fontsize=12)
    ax.set_title(f'{title}\nN = {len(sigma0_valid)}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()
    
    print("\n[分析] GMF关系：")
    print("       后向散射系数σ₀随风速增大而减小（dB值更负）；")
    print("       这是高度计测风的基本物理原理。")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主函数：执行实验4所有课后作业。"""
    print("="*70)
    print("实验4 课后作业：卫星高度计波高验证与校准")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    setup_directories()
    
    # 读取高度计数据
    jason_files = sorted(ALTIMETER_DIR.glob("*.nc"))
    if not jason_files:
        print("[ERROR] 未找到高度计数据")
        sys.exit(1)
    
    jason_data = read_jason_data(jason_files)
    if not jason_data:
        print("[ERROR] 高度计数据读取失败")
        sys.exit(1)
    
    # 读取浮标数据
    buoy_data = read_buoy_data(BUOY_DIR)
    if not buoy_data:
        print("[WARNING] 未找到浮标数据")
    
    # 作业1：绘制所有浮标位置（包括额外浮标）
    print("\n" + "-"*60)
    print("作业1: 浮标位置分布图（含额外浮标）")
    print("-"*60)
    
    # 创建包含所有【有数据文件】浮标位置的列表
    available_buoy_ids = {b['buoy_id'] for b in buoy_data}
    all_buoys_for_map = []
    for bid, (lat, lon) in BUOY_LOCATIONS.items():
        all_buoys_for_map.append({
            'buoy_id': bid,
            'lat': lat,
            'lon': lon,
            'has_data': bid in available_buoy_ids  # 标记是否有数据
        })
    
    plot_buoy_map(all_buoys_for_map,
                 title=f'NDBC Buoy Locations (N={len(available_buoy_ids)} with data)',
                 save_path=RESULTS_DIR / 'buoy_map_updated.png')
    
    print(f"[INFO] 地图中浮标总数: {len(BUOY_LOCATIONS)}")
    print(f"[INFO] 有实际数据的浮标: {len(available_buoy_ids)} 个")
    print(f"[INFO] 数据浮标列表: {sorted(available_buoy_ids)}")
    
    # 作业2：浮标42002风速/波高时间序列
    print("\n" + "-"*60)
    print("作业2: 浮标42002风速/波高时间序列")
    print("-"*60)
    
    buoy_42002 = next((b for b in buoy_data if b['buoy_id'] == '42002'), None)
    if buoy_42002:
        plot_time_series_wind_wave(buoy_42002, month=9)
    else:
        print("[WARNING] 未找到浮标42002的数据")
    
    # 作业3：空间窗口对比（50km vs 75km）
    print("\n" + "-"*60)
    print("作业3: 空间窗口对比 (50km vs 75km)")
    print("-"*60)
    
    if buoy_data:
        # 50km窗口
        sat_swh_50, buoy_swh_50, sat_ws_50, buoy_ws_50 = match_satellite_buoy(
            jason_data, buoy_data, space_window=50.0
        )
        
        # 75km窗口
        sat_swh_75, buoy_swh_75, sat_ws_75, buoy_ws_75 = match_satellite_buoy(
            jason_data, buoy_data, space_window=75.0
        )
        
        print(f"\n50km窗口匹配点数: {len(sat_swh_50)}")
        print(f"75km窗口匹配点数: {len(sat_swh_75)}")
        
        if len(sat_swh_50) > 0 and len(sat_swh_75) > 0:
            stats_50 = calculate_statistics(sat_swh_50, buoy_swh_50)
            stats_75 = calculate_statistics(sat_swh_75, buoy_swh_75)
            
            print("\n50km窗口统计:")
            print(f"  BIAS: {stats_50['bias']:.3f} m, RMSE: {stats_50['rmse']:.3f} m, R: {stats_50['correlation']:.3f}")
            print("\n75km窗口统计:")
            print(f"  BIAS: {stats_75['bias']:.3f} m, RMSE: {stats_75['rmse']:.3f} m, R: {stats_75['correlation']:.3f}")
            
            # 绘制对比图
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            
            # 50km
            axes[0].scatter(sat_swh_50, buoy_swh_50, c='blue', s=30, alpha=0.5)
            max_val = max(sat_swh_50.max(), buoy_swh_50.max())
            axes[0].plot([0, max_val], [0, max_val], 'k--', linewidth=2)
            axes[0].set_xlim(0, max_val)
            axes[0].set_ylim(0, max_val)
            axes[0].set_xlabel('Satellite SWH (m)', fontsize=12)
            axes[0].set_ylabel('Buoy SWH (m)', fontsize=12)
            axes[0].set_title(f'50km Window\nBIAS={stats_50["bias"]:.3f}m, '
                             f'RMSE={stats_50["rmse"]:.3f}m, R={stats_50["correlation"]:.3f}',
                             fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_aspect('equal')
            
            # 75km
            axes[1].scatter(sat_swh_75, buoy_swh_75, c='red', s=30, alpha=0.5)
            max_val = max(sat_swh_75.max(), buoy_swh_75.max())
            axes[1].plot([0, max_val], [0, max_val], 'k--', linewidth=2)
            axes[1].set_xlim(0, max_val)
            axes[1].set_ylim(0, max_val)
            axes[1].set_xlabel('Satellite SWH (m)', fontsize=12)
            axes[1].set_ylabel('Buoy SWH (m)', fontsize=12)
            axes[1].set_title(f'75km Window\nBIAS={stats_75["bias"]:.3f}m, '
                             f'RMSE={stats_75["rmse"]:.3f}m, R={stats_75["correlation"]:.3f}',
                             fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_aspect('equal')
            
            fig.suptitle('SWH Validation: Spatial Window Comparison',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            output_file = RESULTS_DIR / 'swh_comparison_spatial_window.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"[INFO] 图片已保存: {output_file}")
            
            plt.show()
    else:
        print("[WARNING] 无浮标数据，跳过空间窗口对比")
    
    # 作业4：风速验证（50km窗口）
    print("\n" + "-"*60)
    print("作业4: 风速验证 (50km窗口)")
    print("-"*60)
    
    if buoy_data and sat_ws_50 is not None and len(sat_ws_50) > 0:
        plot_scatter_comparison(sat_ws_50, buoy_ws_50,
                               title='Wind Speed Validation (50km Window)',
                               xlabel='Satellite Wind Speed (m/s)',
                               ylabel='Buoy Wind Speed (m/s)',
                               save_path=RESULTS_DIR / 'wind_validation_50km.png')
    else:
        print("[WARNING] 无风速匹配数据，跳过风速验证")
    
    # 作业5：风速线性校准
    print("\n" + "-"*60)
    print("作业5: 风速线性校准")
    print("-"*60)
    
    if buoy_data and sat_ws_50 is not None and len(sat_ws_50) > 0:
        calibrated_ws, coeffs = linear_calibration(sat_ws_50, buoy_ws_50)
        
        stats_before = calculate_statistics(sat_ws_50, buoy_ws_50)
        stats_after = calculate_statistics(calibrated_ws, buoy_ws_50)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        max_val = max(sat_ws_50.max(), buoy_ws_50.max())
        
        # 校准前
        axes[0].scatter(sat_ws_50, buoy_ws_50, c='blue', s=30, alpha=0.5)
        axes[0].plot([0, max_val], [0, max_val], 'k--', linewidth=2)
        axes[0].set_xlim(0, max_val)
        axes[0].set_ylim(0, max_val)
        axes[0].set_xlabel('Satellite Wind Speed (m/s)', fontsize=12)
        axes[0].set_ylabel('Buoy Wind Speed (m/s)', fontsize=12)
        axes[0].set_title(f'Before Calibration\nBIAS={stats_before["bias"]:.3f}m/s, '
                         f'RMSE={stats_before["rmse"]:.3f}m/s, R={stats_before["correlation"]:.3f}',
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal')
        
        # 校准后
        axes[1].scatter(calibrated_ws, buoy_ws_50, c='green', s=30, alpha=0.5)
        axes[1].plot([0, max_val], [0, max_val], 'k--', linewidth=2)
        axes[1].set_xlim(0, max_val)
        axes[1].set_ylim(0, max_val)
        axes[1].set_xlabel('Calibrated Satellite Wind Speed (m/s)', fontsize=12)
        axes[1].set_ylabel('Buoy Wind Speed (m/s)', fontsize=12)
        axes[1].set_title(f'After Calibration\nBIAS={stats_after["bias"]:.3f}m/s, '
                         f'RMSE={stats_after["rmse"]:.3f}m/s, R={stats_after["correlation"]:.3f}',
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal')
        
        fig.suptitle('Wind Speed Calibration: Before vs After',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = RESULTS_DIR / 'wind_calibration_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {output_file}")
        
        plt.show()
        
        print(f"\n校准系数: y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}")
        print(f"RMSE改进: {(1 - stats_after['rmse']/stats_before['rmse'])*100:.1f}%")
    else:
        print("[WARNING] 无风速匹配数据，跳过风速校准")
    
    # 作业6：GMF散点图
    print("\n" + "-"*60)
    print("作业6: 后向散射系数与风速GMF")
    print("-"*60)
    
    if jason_data.get('sigma0') is not None and jason_data.get('wind_speed') is not None:
        plot_gmf(jason_data['sigma0'], jason_data['wind_speed'],
                title='Geophysical Model Function (σ₀ vs Wind Speed)',
                save_path=RESULTS_DIR / 'gmf_sigma0_windspeed.png')
    else:
        print("[WARNING] 高度计数据中无后向散射系数或风速数据，跳过GMF绘图")
        print("[HINT] 请确认数据文件中包含sigma0和wind_speed变量")
    
    print("\n" + "="*70)
    print("实验4 课后作业全部完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果保存在: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
