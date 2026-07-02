"""
实验4：卫星高度计波高数据验证与校准（改进版）
================================================
基于Jason-3高度计和NDBC浮标数据，进行有效波高(SWH)的验证与校准分析。

改进点：
1. 使用Cartopy替代Basemap进行地图投影
2. 函数化封装，提高代码复用性
3. 增加多种统计指标（RMSE、BIAS、相关系数等）
4. 自动创建结果输出目录
5. 详细的注释和文档字符串
6. 使用pathlib处理路径，提高跨平台可移植性
7. 增加线性回归校准功能
8. 支持多种数据格式（CSV、NetCDF）
9. 支持多文件批量读取与合并
10. 基于时空位置的最近邻匹配算法

依赖库：
    - numpy: 数值计算
    - matplotlib: 绘图
    - cartopy: 地图投影
    - netCDF4: NetCDF文件读写
    - scipy: 科学计算（可选，用于高级统计分析）

作者：基于《海洋遥感数据处理实践初级教程》改进
Python版本：3.13+
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List

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
ALTIMETER_DIR = BASE_DIR / "data" / "altimeter_data"
BUOY_DIR = BASE_DIR / "data" / "buoy_data"

# 结果输出目录
RESULTS_DIR = BASE_DIR / "results" / "实验4_高度计波高"

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 浮标位置信息（NDBC浮标编号 -> [纬度, 经度]）
# 根据NDBC官方数据，这些是常见浮标位置
BUOY_LOCATIONS = {
    '32012': [-19.425, -85.078],   # 印度洋
    '41001': [34.724, -72.317],   # 美国东海岸外海
    '42002': [26.055, -93.646],   # 墨西哥湾
    '46059': [38.094, -129.951],  # 东北太平洋
    '51003': [19.175, -160.625],  # 夏威夷附近
}

# 匹配参数
MATCH_TIME_WINDOW = 3.0  # 时间匹配窗口（小时）
MATCH_SPACE_WINDOW = 50.0  # 空间匹配窗口（公里）


# =============================================================================
# 工具函数
# =============================================================================

def setup_directories():
    """创建必要的结果输出目录。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 结果将保存至: {RESULTS_DIR}")


def find_data_files():
    """
    查找指定目录中的所有高度计和浮标数据文件。
    
    Returns:
        tuple: (jason_files_list, ndbc_files_list) 或 ([], [])
    """
    jason_files = []
    ndbc_files = []
    
    # 查找高度计数据（altimeter_data目录）
    if ALTIMETER_DIR.exists():
        jason_files = sorted(list(ALTIMETER_DIR.glob("*.nc")))
        if jason_files:
            print(f"[INFO] 找到 {len(jason_files)} 个高度计数据文件")
            for f in jason_files:
                print(f"  - {f.name}")
        else:
            print(f"[WARNING] {ALTIMETER_DIR} 中未找到NetCDF文件")
    else:
        print(f"[WARNING] 高度计数据目录不存在: {ALTIMETER_DIR}")
    
    # 查找浮标数据（buoy_data目录）
    if BUOY_DIR.exists():
        ndbc_files = sorted(list(BUOY_DIR.glob("*.txt")))
        if ndbc_files:
            print(f"[INFO] 找到 {len(ndbc_files)} 个浮标数据文件")
            for f in ndbc_files:
                print(f"  - {f.name}")
        else:
            print(f"[WARNING] {BUOY_DIR} 中未找到文本文件")
    else:
        print(f"[WARNING] 浮标数据目录不存在: {BUOY_DIR}")
    
    return jason_files, ndbc_files


def read_single_altimeter_file(filepath: Path) -> dict:
    """
    读取单个高度计数据文件。
    
    Args:
        filepath: NetCDF文件路径
        
    Returns:
        dict: 包含高度计数据的字典
    """
    print(f"[INFO] 正在读取: {filepath.name}")
    
    with nc.Dataset(filepath, 'r') as dataset:
        # 尝试读取常见变量名
        lon_var = None
        lat_var = None
        swh_var = None
        time_var = None
        
        for name in ['lon', 'longitude', 'long', 'x']:
            if name in dataset.variables:
                lon_var = name
                break
        
        for name in ['lat', 'latitude', 'y']:
            if name in dataset.variables:
                lat_var = name
                break
        
        for name in ['swh', 'wave_height', 'significant_wave_height', 'shw']:
            if name in dataset.variables:
                swh_var = name
                break
        
        for name in ['time', 'datetime', 't']:
            if name in dataset.variables:
                time_var = name
                break
        
        if not all([lon_var, lat_var, swh_var]):
            print(f"[WARNING] {filepath.name} 未找到必要变量，跳过")
            return {}
        
        # 读取数据
        lon = dataset.variables[lon_var][:]
        lat = dataset.variables[lat_var][:]
        swh = dataset.variables[swh_var][:]
        
        # 获取变量属性
        swh_attrs = {
            'units': getattr(dataset.variables[swh_var], 'units', 'm'),
            'scale_factor': getattr(dataset.variables[swh_var], 'scale_factor', 1.0),
            'add_offset': getattr(dataset.variables[swh_var], 'add_offset', 0.0),
            '_FillValue': getattr(dataset.variables[swh_var], '_FillValue', -9999)
        }
        
        # 处理填充值
        fill_value = swh_attrs['_FillValue']
        if fill_value is not None:
            swh = np.where(swh == fill_value, np.nan, swh)
        
        # NetCDF4库在读取int16类型变量时会自动应用scale_factor和add_offset
        # 因此不需要手动缩放。如果数据范围异常（如>100），可能是数据本身问题
        # 不再进行二次缩放，避免数值错误
        
        # 读取时间数据
        time_data = None
        if time_var:
            time_raw = dataset.variables[time_var][:]
            time_units = getattr(dataset.variables[time_var], 'units', 'days since 1900-1-1')
            # 转换时间为datetime
            try:
                if 'days since' in time_units:
                    base_date_str = time_units.replace('days since ', '').strip()
                    base_date = datetime.strptime(base_date_str, '%Y-%m-%d')
                    time_data = np.array([base_date + timedelta(days=float(t)) for t in time_raw])
                elif 'hours since' in time_units:
                    base_date_str = time_units.replace('hours since ', '').strip()
                    base_date = datetime.strptime(base_date_str, '%Y-%m-%d')
                    time_data = np.array([base_date + timedelta(hours=float(t)) for t in time_raw])
            except:
                time_data = None
        
        return {
            'swh': np.array(swh),
            'lon': np.array(lon),
            'lat': np.array(lat),
            'time': time_data,
            'attrs': swh_attrs
        }


def read_jason3_data(filepaths: List[Path]) -> dict:
    """
    读取并合并多个高度计数据文件。
    
    Args:
        filepaths: NetCDF文件路径列表
        
    Returns:
        dict: 合并后的高度计数据字典
    """
    print(f"[INFO] 正在读取 {len(filepaths)} 个高度计数据文件...")
    
    all_swh = []
    all_lon = []
    all_lat = []
    all_time = []
    
    for filepath in filepaths:
        data = read_single_altimeter_file(filepath)
        if data:
            all_swh.append(data['swh'])
            all_lon.append(data['lon'])
            all_lat.append(data['lat'])
            if data['time'] is not None:
                all_time.append(data['time'])
    
    if not all_swh:
        print("[ERROR] 未成功读取任何高度计数据")
        return {}
    
    # 合并数据
    merged_swh = np.concatenate(all_swh)
    merged_lon = np.concatenate(all_lon)
    merged_lat = np.concatenate(all_lat)
    merged_time = np.concatenate(all_time) if all_time else None
    
    # 移除无效值
    valid_mask = ~np.isnan(merged_swh)
    
    print("\n" + "="*60)
    print("高度计数据合并完成")
    print("="*60)
    print(f"  总数据点数: {len(merged_swh)}")
    print(f"  有效数据点: {np.sum(valid_mask)}")
    print(f"  经度范围: {merged_lon.min():.2f}° ~ {merged_lon.max():.2f}°")
    print(f"  纬度范围: {merged_lat.min():.2f}° ~ {merged_lat.max():.2f}°")
    print(f"  SWH范围: {np.nanmin(merged_swh):.2f} ~ {np.nanmax(merged_swh):.2f} m")
    print("="*60 + "\n")
    
    # 获取第一个成功读取文件的属性
    first_attrs = {}
    for filepath in filepaths:
        data = read_single_altimeter_file(filepath)
        if data:
            first_attrs = data.get('attrs', {})
            break
    
    return {
        'swh': merged_swh,
        'lon': merged_lon,
        'lat': merged_lat,
        'time': merged_time,
        'attrs': first_attrs
    }


def read_single_buoy_file(filepath: Path) -> dict:
    """
    读取单个浮标数据文件。
    
    Args:
        filepath: 文本文件路径
        
    Returns:
        dict: 包含浮标数据的字典
    """
    print(f"[INFO] 正在读取: {filepath.name}")
    
    # 尝试多种读取方式
    data = None
    
    for skip in [2, 1, 0]:
        try:
            data = np.loadtxt(filepath, skiprows=skip)
            break
        except:
            continue
    
    if data is None:
        print(f"[WARNING] 无法读取 {filepath.name}")
        return {}
    
    n_cols = data.shape[1] if len(data.shape) > 1 else 0
    
    if n_cols >= 9:
        year = data[:, 0]
        month = data[:, 1]
        day = data[:, 2]
        hour = data[:, 3]
        minute = data[:, 4]
        wdir = data[:, 5]
        wspd = data[:, 6]
        wvht = data[:, 8]
        
        # 处理年份（NDBC格式：两位数年份）
        year = np.where(year < 50, year + 2000, year + 1900)
        
        # 构建datetime数组
        datetime_array = np.array([
            datetime(int(y), int(m), int(d), int(h), int(min))
            for y, m, d, h, min in zip(year, month, day, hour, minute)
        ])
        
        # 筛选有效数据
        # NDBC填充值：WVHT=99.0表示缺失，WSPD=99.0表示缺失
        # 保留wspd < 90的合理范围（真实大风可能超过此值，但99是填充值）
        valid_mask = (wvht > 0) & (wvht < 50) & (wspd >= 0) & (wspd < 90)
        
        # 获取浮标位置（从文件名提取编号）
        buoy_id = filepath.stem[:5]  # 例如 "42002h2012" -> "42002"
        buoy_latlon = BUOY_LOCATIONS.get(buoy_id, [None, None])
        
        return {
            'buoy_id': buoy_id,
            'lat': buoy_latlon[0],
            'lon': buoy_latlon[1],
            'datetime': datetime_array[valid_mask],
            'year': year[valid_mask],
            'month': month[valid_mask],
            'day': day[valid_mask],
            'hour': hour[valid_mask],
            'minute': minute[valid_mask],
            'wdir': wdir[valid_mask],
            'wspd': wspd[valid_mask],
            'wvht': wvht[valid_mask]
        }
    else:
        print(f"[WARNING] {filepath.name} 列数不足（{n_cols}列）")
        return {}


def read_ndbc_data(filepaths: List[Path]) -> List[dict]:
    """
    读取多个浮标数据文件，返回列表（每个浮标一个字典）。
    
    Args:
        filepaths: 文本文件路径列表
        
    Returns:
        List[dict]: 每个浮标的数据字典列表
    """
    print(f"[INFO] 正在读取 {len(filepaths)} 个浮标数据文件...")
    
    all_buoys = []
    
    for filepath in filepaths:
        data = read_single_buoy_file(filepath)
        if data:
            all_buoys.append(data)
    
    if not all_buoys:
        print("[ERROR] 未成功读取任何浮标数据")
        return []
    
    total_points = sum(len(b['wvht']) for b in all_buoys)
    print("\n" + "="*60)
    print("浮标数据读取完成")
    print("="*60)
    print(f"  浮标数量: {len(all_buoys)}")
    print(f"  总数据点数: {total_points}")
    for b in all_buoys:
        if len(b['wvht']) > 0:
            print(f"  浮标 {b['buoy_id']}: {len(b['wvht'])} 个点, "
                  f"波高 {b['wvht'].min():.2f}~{b['wvht'].max():.2f} m")
        else:
            print(f"  浮标 {b['buoy_id']}: {len(b['wvht'])} 个点, "
                  f"[无有效波高数据]")
    print("="*60 + "\n")
    
    # 过滤掉没有有效数据的浮标
    all_buoys = [b for b in all_buoys if len(b['wvht']) > 0]
    
    if not all_buoys:
        print("[WARNING] 所有浮标均无有效波高数据")
    
    return all_buoys


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两点间的Haversine距离（公里）。
    
    Args:
        lat1, lon1: 第一点经纬度（度）
        lat2, lon2: 第二点经纬度（度）
        
    Returns:
        float: 距离（公里）
    """
    R = 6371.0  # 地球半径（公里）
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def match_satellite_buoy(sat_data: dict, buoy_data: List[dict],
                         time_window_hours: float = MATCH_TIME_WINDOW,
                         space_window_km: float = MATCH_SPACE_WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于时空最近邻匹配卫星和浮标数据（优化版，使用经纬度粗筛加速）。
    
    Args:
        sat_data: 卫星数据字典
        buoy_data: 浮标数据列表
        time_window_hours: 时间匹配窗口（小时）
        space_window_km: 空间匹配窗口（公里）
        
    Returns:
        Tuple: (匹配后的卫星SWH数组, 匹配后的浮标SWH数组)
    """
    print("[INFO] 开始时空匹配卫星和浮标数据...")
    print(f"[INFO] 时间窗口: ±{time_window_hours}小时")
    print(f"[INFO] 空间窗口: ±{space_window_km}公里")
    
    matched_sat_swh = []
    matched_buoy_swh = []
    
    sat_swh = sat_data['swh']
    sat_lon = sat_data['lon']
    sat_lat = sat_data['lat']
    sat_time = sat_data.get('time')
    
    # 预计算：将卫星时间转换为小时数（相对于某个基准）
    sat_time_hours = None
    if sat_time is not None:
        base_time = sat_time[0]
        sat_time_hours = np.array([(st - base_time).total_seconds() / 3600.0 
                                   for st in sat_time])
    
    for buoy in buoy_data:
        if buoy['lat'] is None or buoy['lon'] is None:
            print(f"[WARNING] 浮标 {buoy['buoy_id']} 无位置信息，跳过")
            continue
        
        buoy_lat = buoy['lat']
        buoy_lon = buoy['lon']
        buoy_times = buoy['datetime']
        buoy_wvht = buoy['wvht']
        
        print(f"[INFO] 匹配浮标 {buoy['buoy_id']} (位置: {buoy_lat:.2f}°N, {buoy_lon:.2f}°E)...")
        
        # ===== 优化1: 经纬度粗筛 =====
        # 50km ≈ 0.45°纬度，经度随纬度变化
        lat_degree_window = space_window_km / 111.0  # 1°纬度 ≈ 111km
        lon_degree_window = space_window_km / (111.0 * np.cos(np.radians(buoy_lat)))
        
        lat_mask = (sat_lat >= buoy_lat - lat_degree_window) & \
                   (sat_lat <= buoy_lat + lat_degree_window)
        lon_mask = (sat_lon >= buoy_lon - lon_degree_window) & \
                   (sat_lon <= buoy_lon + lon_degree_window)
        coarse_mask = lat_mask & lon_mask
        
        n_candidates = np.sum(coarse_mask)
        if n_candidates == 0:
            continue
        
        # 提取候选点（大幅减少数据量）
        cand_idx = np.where(coarse_mask)[0]
        cand_lat = sat_lat[cand_idx]
        cand_lon = sat_lon[cand_idx]
        cand_swh = sat_swh[cand_idx]
        
        if sat_time_hours is not None:
            cand_time_hours = sat_time_hours[cand_idx]
        
        # 对于每个浮标时间点，在候选点中找到最近邻
        for bt, bw in zip(buoy_times, buoy_wvht):
            # ===== 优化2: 时间粗筛 =====
            if sat_time_hours is not None:
                bt_hours = (bt - base_time).total_seconds() / 3600.0
                time_mask = np.abs(cand_time_hours - bt_hours) <= time_window_hours
                
                if not np.any(time_mask):
                    continue
                
                # 在时间和空间候选点中计算精确距离
                time_cand_idx = np.where(time_mask)[0]
                time_cand_lat = cand_lat[time_mask]
                time_cand_lon = cand_lon[time_mask]
                time_cand_swh = cand_swh[time_mask]
                
                # 计算精确距离
                distances = haversine_distance(buoy_lat, buoy_lon, 
                                               time_cand_lat, time_cand_lon)
                
                # 精确空间筛选
                space_valid = distances <= space_window_km
                if not np.any(space_valid):
                    continue
                
                # 找到最近的点
                nearest_local_idx = np.argmin(distances[space_valid])
                nearest_idx = cand_idx[time_cand_idx[space_valid][nearest_local_idx]]
            else:
                # 仅空间匹配
                distances = haversine_distance(buoy_lat, buoy_lon, cand_lat, cand_lon)
                space_valid = distances <= space_window_km
                
                if not np.any(space_valid):
                    continue
                
                nearest_local_idx = np.argmin(distances[space_valid])
                nearest_idx = cand_idx[space_valid][nearest_local_idx]
            
            matched_sat_swh.append(sat_swh[nearest_idx])
            matched_buoy_swh.append(bw)
    
    if len(matched_sat_swh) == 0:
        print("[WARNING] 未找到任何匹配的数据点")
        return np.array([]), np.array([])
    
    matched_sat_swh = np.array(matched_sat_swh)
    matched_buoy_swh = np.array(matched_buoy_swh)
    
    # 移除NaN值
    valid_mask = ~np.isnan(matched_sat_swh) & ~np.isnan(matched_buoy_swh)
    matched_sat_swh = matched_sat_swh[valid_mask]
    matched_buoy_swh = matched_buoy_swh[valid_mask]
    
    print(f"\n[INFO] 匹配完成: {len(matched_sat_swh)} 对数据点")
    print(f"[INFO] 卫星SWH范围: {matched_sat_swh.min():.2f} ~ {matched_sat_swh.max():.2f} m")
    print(f"[INFO] 浮标SWH范围: {matched_buoy_swh.min():.2f} ~ {matched_buoy_swh.max():.2f} m")
    
    return matched_sat_swh, matched_buoy_swh


def calculate_statistics(satellite_swh: np.ndarray, 
                        buoy_swh: np.ndarray) -> dict:
    """
    计算验证统计指标。
    
    Args:
        satellite_swh: 卫星波高数据
        buoy_swh: 浮标波高数据
        
    Returns:
        dict: 统计指标字典
    """
    # 确保数据长度一致
    n = min(len(satellite_swh), len(buoy_swh))
    sat = satellite_swh[:n]
    buoy = buoy_swh[:n]
    
    # 计算统计指标
    diff = sat - buoy
    
    stats = {
        'n': n,
        'sat_mean': np.mean(sat),
        'buoy_mean': np.mean(buoy),
        'sat_std': np.std(sat),
        'buoy_std': np.std(buoy),
        'bias': np.mean(diff),
        'rmse': np.sqrt(np.mean(diff**2)),
        'mae': np.mean(np.abs(diff)),
        'correlation': np.corrcoef(sat, buoy)[0, 1],
        'relative_bias': np.mean(diff) / np.mean(buoy) * 100,
        'relative_rmse': np.sqrt(np.mean(diff**2)) / np.mean(buoy) * 100
    }
    
    return stats


def linear_calibration(satellite_swh: np.ndarray, 
                      buoy_swh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    线性回归校准。
    
    Args:
        satellite_swh: 卫星波高数据
        buoy_swh: 浮标波高数据
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (校准后的波高, 回归系数)
    """
    # 线性回归: y = a*x + b
    coeffs = np.polyfit(satellite_swh, buoy_swh, 1)
    calibrated = np.polyval(coeffs, satellite_swh)
    
    return calibrated, coeffs


# =============================================================================
# 可视化函数
# =============================================================================

def plot_swh_map(data: dict, 
                 title: str = "Significant Wave Height",
                 save_path: Optional[Path] = None) -> None:
    """
    绘制波高空间分布图。
    
    Args:
        data: 包含swh, lon, lat的字典
        title: 图标题
        save_path: 保存路径
    """
    print(f"[INFO] 正在绘制波高空间分布图...")
    
    swh = data['swh']
    lon = data['lon']
    lat = data['lat']
    
    # 处理数据（展平）
    swh_flat = np.array(swh).flatten()
    lon_flat = np.array(lon).flatten()
    lat_flat = np.array(lat).flatten()
    
    # 移除无效值
    valid_mask = ~np.isnan(swh_flat)
    swh_flat = swh_flat[valid_mask]
    lon_flat = lon_flat[valid_mask]
    lat_flat = lat_flat[valid_mask]
    
    if len(swh_flat) == 0:
        print("[WARNING] 无有效数据可绘制")
        return
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    
    # 网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    # 绘制散点
    scatter = ax.scatter(lon_flat, lat_flat, c=swh_flat, s=10,
                        cmap='jet', vmin=0, vmax=10,
                        transform=ccrs.PlateCarree(),
                        alpha=0.7)
    
    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('Significant Wave Height (m)', fontsize=12)
    
    # 标题
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


def plot_scatter_comparison(satellite_swh: np.ndarray,
                           buoy_swh: np.ndarray,
                           title: str = "SWH Validation",
                           save_path: Optional[Path] = None) -> None:
    """
    绘制卫星与浮标波高散点对比图。
    
    Args:
        satellite_swh: 卫星波高数据
        buoy_swh: 浮标波高数据
        title: 图标题
        save_path: 保存路径
    """
    print(f"[INFO] 正在绘制散点对比图...")
    
    # 计算统计指标
    stats = calculate_statistics(satellite_swh, buoy_swh)
    
    # 线性回归
    coeffs = np.polyfit(satellite_swh, buoy_swh, 1)
    x_line = np.linspace(0, max(satellite_swh.max(), buoy_swh.max()), 100)
    y_line = np.polyval(coeffs, x_line)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 散点图
    ax.scatter(satellite_swh, buoy_swh, c='blue', s=30, alpha=0.5, 
              edgecolors='black', linewidth=0.5)
    
    # 1:1线
    max_val = max(satellite_swh.max(), buoy_swh.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='1:1 Line')
    
    # 回归线
    ax.plot(x_line, y_line, 'r-', linewidth=2, 
           label=f'y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}')
    
    # 设置坐标轴
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel('Satellite SWH (m)', fontsize=12)
    ax.set_ylabel('Buoy SWH (m)', fontsize=12)
    
    # 标题和统计信息
    ax.set_title(f'{title}\n'
                f'N={stats["n"]}, BIAS={stats["bias"]:.3f}m, '
                f'RMSE={stats["rmse"]:.3f}m, R={stats["correlation"]:.3f}',
                fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("验证统计结果")
    print("="*60)
    print(f"样本数 (N): {stats['n']}")
    print(f"卫星平均波高: {stats['sat_mean']:.3f} m")
    print(f"浮标平均波高: {stats['buoy_mean']:.3f} m")
    print(f"卫星标准差: {stats['sat_std']:.3f} m")
    print(f"浮标标准差: {stats['buoy_std']:.3f} m")
    print(f"偏差 (BIAS): {stats['bias']:.3f} m")
    print(f"均方根误差 (RMSE): {stats['rmse']:.3f} m")
    print(f"平均绝对误差 (MAE): {stats['mae']:.3f} m")
    print(f"相关系数 (R): {stats['correlation']:.3f}")
    print(f"相对偏差: {stats['relative_bias']:.2f}%")
    print(f"相对RMSE: {stats['relative_rmse']:.2f}%")
    print(f"回归系数: y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}")
    print("="*60 + "\n")


def plot_time_series(satellite_swh: np.ndarray,
                    buoy_swh: np.ndarray,
                    time_index: Optional[np.ndarray] = None,
                    title: str = "SWH Time Series",
                    save_path: Optional[Path] = None) -> None:
    """
    绘制波高时间序列对比图。
    
    Args:
        satellite_swh: 卫星波高数据
        buoy_swh: 浮标波高数据
        time_index: 时间索引
        title: 图标题
        save_path: 保存路径
    """
    print(f"[INFO] 正在绘制时间序列对比图...")
    
    if time_index is None:
        time_index = np.arange(len(satellite_swh))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制时间序列
    ax.plot(time_index, satellite_swh, 'b-', linewidth=1.5, 
           alpha=0.7, label='Satellite')
    ax.plot(time_index, buoy_swh, 'r-', linewidth=1.5,
           alpha=0.7, label='Buoy')
    
    # 设置坐标轴
    ax.set_xlabel('Time Index', fontsize=12)
    ax.set_ylabel('Significant Wave Height (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


def plot_residuals(satellite_swh: np.ndarray,
                  buoy_swh: np.ndarray,
                  title: str = "SWH Residuals",
                  save_path: Optional[Path] = None) -> None:
    """
    绘制残差分析图。
    
    Args:
        satellite_swh: 卫星波高数据
        buoy_swh: 浮标波高数据
        title: 图标题
        save_path: 保存路径
    """
    print(f"[INFO] 正在绘制残差分析图...")
    
    diff = satellite_swh - buoy_swh
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 残差散点图
    axes[0].scatter(buoy_swh, diff, c='blue', s=30, alpha=0.5,
                   edgecolors='black', linewidth=0.5)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[0].axhline(y=mean_diff, color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_diff:.3f}m')
    axes[0].axhline(y=mean_diff + std_diff, color='g', linestyle='--', 
                   linewidth=1, alpha=0.7, label=f'±1 STD: {std_diff:.3f}m')
    axes[0].axhline(y=mean_diff - std_diff, color='g', linestyle='--',
                   linewidth=1, alpha=0.7)
    axes[0].set_xlabel('Buoy SWH (m)', fontsize=12)
    axes[0].set_ylabel('Satellite - Buoy (m)', fontsize=12)
    axes[0].set_title('Residuals vs Buoy SWH', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 残差直方图
    axes[1].hist(diff, bins=50, color='steelblue', alpha=0.7,
                edgecolor='black', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linestyle='-', linewidth=1)
    axes[1].axvline(x=mean_diff, color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_diff:.3f}m')
    axes[1].set_xlabel('Residual (m)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()


def plot_calibration_comparison(satellite_swh: np.ndarray,
                               buoy_swh: np.ndarray,
                               calibrated_swh: np.ndarray,
                               title: str = "Calibration Comparison",
                               save_path: Optional[Path] = None) -> None:
    """
    绘制校准前后的对比图。
    
    Args:
        satellite_swh: 原始卫星波高
        buoy_swh: 浮标波高
        calibrated_swh: 校准后的卫星波高
        title: 图标题
        save_path: 保存路径
    """
    print(f"[INFO] 正在绘制校准对比图...")
    
    # 计算校准前后的统计指标
    stats_before = calculate_statistics(satellite_swh, buoy_swh)
    stats_after = calculate_statistics(calibrated_swh, buoy_swh)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    max_val = max(satellite_swh.max(), buoy_swh.max(), calibrated_swh.max())
    
    # 校准前
    axes[0].scatter(satellite_swh, buoy_swh, c='blue', s=30, alpha=0.5,
                   edgecolors='black', linewidth=0.5)
    axes[0].plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='1:1 Line')
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].set_xlabel('Satellite SWH (m)', fontsize=12)
    axes[0].set_ylabel('Buoy SWH (m)', fontsize=12)
    axes[0].set_title(f'Before Calibration\n'
                     f'BIAS={stats_before["bias"]:.3f}m, '
                     f'RMSE={stats_before["rmse"]:.3f}m, '
                     f'R={stats_before["correlation"]:.3f}',
                     fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # 校准后
    axes[1].scatter(calibrated_swh, buoy_swh, c='green', s=30, alpha=0.5,
                   edgecolors='black', linewidth=0.5)
    axes[1].plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='1:1 Line')
    axes[1].set_xlim(0, max_val)
    axes[1].set_ylim(0, max_val)
    axes[1].set_xlabel('Calibrated Satellite SWH (m)', fontsize=12)
    axes[1].set_ylabel('Buoy SWH (m)', fontsize=12)
    axes[1].set_title(f'After Calibration\n'
                     f'BIAS={stats_after["bias"]:.3f}m, '
                     f'RMSE={stats_after["rmse"]:.3f}m, '
                     f'R={stats_after["correlation"]:.3f}',
                     fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图片已保存: {save_path}")
    
    plt.show()
    
    # 打印改进信息
    print("\n" + "="*60)
    print("校准效果对比")
    print("="*60)
    print(f"校准前 BIAS: {stats_before['bias']:.3f} m")
    print(f"校准后 BIAS: {stats_after['bias']:.3f} m")
    print(f"校准前 RMSE: {stats_before['rmse']:.3f} m")
    print(f"校准后 RMSE: {stats_after['rmse']:.3f} m")
    print(f"校准前 R: {stats_before['correlation']:.3f}")
    print(f"校准后 R: {stats_after['correlation']:.3f}")
    if stats_before['rmse'] > 0:
        print(f"RMSE改进: {(1 - stats_after['rmse']/stats_before['rmse'])*100:.1f}%")
    print("="*60 + "\n")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主函数：执行实验4的所有步骤。"""
    print("="*70)
    print("实验4：卫星高度计波高数据验证与校准")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # 步骤1：创建输出目录
    setup_directories()
    
    # 步骤2：自动查找数据文件
    jason_files, ndbc_files = find_data_files()
    
    if not jason_files:
        print("\n[ERROR] 未找到高度计数据文件，程序退出。")
        print(f"[HINT] 请确认NetCDF文件位于: {ALTIMETER_DIR}")
        sys.exit(1)
    
    # 步骤3：读取并合并高度计数据
    print("\n" + "-"*60)
    print("任务1: 读取高度计数据")
    print("-"*60)
    jason_data = read_jason3_data(jason_files)
    
    if not jason_data:
        print("[ERROR] 高度计数据读取失败，程序退出")
        sys.exit(1)
    
    # 步骤4：绘制高度计波高空间分布
    print("\n" + "-"*60)
    print("任务2: 绘制波高空间分布")
    print("-"*60)
    plot_swh_map(jason_data, 
                title='Satellite Significant Wave Height',
                save_path=RESULTS_DIR / 'satellite_swh_map.png')
    
    # 步骤5：读取浮标数据
    if ndbc_files:
        print("\n" + "-"*60)
        print("任务3: 读取浮标数据")
        print("-"*60)
        buoy_data_list = read_ndbc_data(ndbc_files)
        
        if buoy_data_list:
            # 步骤6：时空匹配
            print("\n" + "-"*60)
            print("任务4: 匹配卫星和浮标数据")
            print("-"*60)
            
            satellite_swh, buoy_swh = match_satellite_buoy(
                jason_data, buoy_data_list,
                time_window_hours=MATCH_TIME_WINDOW,
                space_window_km=MATCH_SPACE_WINDOW
            )
            
            if len(satellite_swh) == 0:
                print("[WARNING] 时空匹配失败，使用模拟数据进行演示")
                np.random.seed(42)
                n_points = min(len(jason_data['swh']), 1000)
                satellite_swh = np.array(jason_data['swh']).flatten()[:n_points]
                satellite_swh = satellite_swh[~np.isnan(satellite_swh)]
                buoy_swh = satellite_swh + np.random.normal(0, 0.3, len(satellite_swh))
                buoy_swh = np.maximum(buoy_swh, 0.1)
        else:
            print("[WARNING] 浮标数据读取失败，使用模拟数据")
            np.random.seed(42)
            n_points = min(len(jason_data['swh']), 1000)
            satellite_swh = np.array(jason_data['swh']).flatten()[:n_points]
            satellite_swh = satellite_swh[~np.isnan(satellite_swh)]
            buoy_swh = satellite_swh + np.random.normal(0, 0.3, len(satellite_swh))
            buoy_swh = np.maximum(buoy_swh, 0.1)
    else:
        print("[WARNING] 未找到浮标数据，使用模拟数据进行演示")
        np.random.seed(42)
        n_points = min(len(jason_data['swh']), 1000)
        satellite_swh = np.array(jason_data['swh']).flatten()[:n_points]
        satellite_swh = satellite_swh[~np.isnan(satellite_swh)]
        buoy_swh = satellite_swh + np.random.normal(0, 0.3, len(satellite_swh))
        buoy_swh = np.maximum(buoy_swh, 0.1)
    
    # 步骤7：绘制散点对比图
    print("\n" + "-"*60)
    print("任务5: 绘制散点对比图")
    print("-"*60)
    plot_scatter_comparison(satellite_swh, buoy_swh,
                           title='Satellite vs Buoy SWH Validation',
                           save_path=RESULTS_DIR / 'swh_scatter_comparison.png')
    
    # 步骤8：绘制时间序列对比
    print("\n" + "-"*60)
    print("任务6: 绘制时间序列对比")
    print("-"*60)
    plot_time_series(satellite_swh, buoy_swh,
                    title='Satellite vs Buoy SWH Time Series',
                    save_path=RESULTS_DIR / 'swh_time_series.png')
    
    # 步骤9：绘制残差分析
    print("\n" + "-"*60)
    print("任务7: 绘制残差分析")
    print("-"*60)
    plot_residuals(satellite_swh, buoy_swh,
                  title='SWH Residual Analysis',
                  save_path=RESULTS_DIR / 'swh_residuals.png')
    
    # 步骤10：线性回归校准
    print("\n" + "-"*60)
    print("任务8: 线性回归校准")
    print("-"*60)
    calibrated_swh, coeffs = linear_calibration(satellite_swh, buoy_swh)
    
    plot_calibration_comparison(satellite_swh, buoy_swh, calibrated_swh,
                               title='SWH Calibration: Before vs After',
                               save_path=RESULTS_DIR / 'swh_calibration_comparison.png')
    
    print("\n" + "="*70)
    print("实验4完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果保存在: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
