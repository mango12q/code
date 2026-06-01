function [data_norm, data_mean, data_std, lon, lat, time_all] = load_and_preprocess_data(params)
%% 加载和预处理数据
% 输入: params - 参数结构体
% 输出: data_norm - 标准化后的数据
%       data_mean, data_std - 均值和标准差
%       lon, lat - 经纬度网格
%       time_all - 时间序列

fprintf('加载数据中...\n');

% 这里假设数据为NetCDF格式，根据实际情况修改
% 如果数据是其他格式（如.mat, .nc, .txt），请相应调整

%% 1. 加载SST数据
try
    % 尝试加载NetCDF数据
    sst_file = fullfile(params.paths.sst, 'sst_data.nc');
    if exist(sst_file, 'file')
        sst = ncread(sst_file, 'sst');
        lon = ncread(sst_file, 'lon');
        lat = ncread(sst_file, 'lat');
        time_all = ncread(sst_file, 'time');
    else
        % 如果没有NetCDF文件，尝试加载MAT文件
        mat_file = fullfile(params.paths.sst, 'sst_data.mat');
        if exist(mat_file, 'file')
            data = load(mat_file);
            sst = data.sst;
            lon = data.lon;
            lat = data.lat;
            time_all = data.time;
        else
            error('未找到SST数据文件，请检查路径设置');
        end
    end
catch ME
    fprintf('警告: %s\n', ME.message);
    fprintf('使用示例数据生成...\n');
    % 生成示例数据用于测试
    [sst, lon, lat, time_all] = generate_sample_data(params);
end

%% 2. 裁剪研究区域
lon_idx = find(lon >= params.region.lon_range(1) & lon <= params.region.lon_range(2));
lat_idx = find(lat >= params.region.lat_range(1) & lat <= params.region.lat_range(2));

lon = lon(lon_idx);
lat = lat(lat_idx);
sst = sst(lon_idx, lat_idx, :);

fprintf('区域网格大小: %d x %d\n', length(lon), length(lat));

%% 3. 加载SSHA数据
try
    ssha_file = fullfile(params.paths.ssha, 'ssha_data.nc');
    if exist(ssha_file, 'file')
        ssha = ncread(ssha_file, 'ssha');
        ssha = ssha(lon_idx, lat_idx, :);
    else
        mat_file = fullfile(params.paths.ssha, 'ssha_data.mat');
        if exist(mat_file, 'file')
            data = load(mat_file);
            ssha = data.ssha(lon_idx, lat_idx, :);
        else
            fprintf('警告: 未找到SSHA数据，使用零值填充\n');
            ssha = zeros(size(sst));
        end
    end
catch
    fprintf('警告: SSHA数据加载失败，使用零值填充\n');
    ssha = zeros(size(sst));
end

%% 4. 加载SSW数据（ESSW和NSSW）
try
    essw_file = fullfile(params.paths.ssw, 'essw_data.nc');
    nssw_file = fullfile(params.paths.ssw, 'nssw_data.nc');
    if exist(essw_file, 'file') && exist(nssw_file, 'file')
        essw = ncread(essw_file, 'essw');
        nssw = ncread(nssw_file, 'nssw');
        essw = essw(lon_idx, lat_idx, :);
        nssw = nssw(lon_idx, lat_idx, :);
    else
        mat_file = fullfile(params.paths.ssw, 'ssw_data.mat');
        if exist(mat_file, 'file')
            data = load(mat_file);
            essw = data.essw(lon_idx, lat_idx, :);
            nssw = data.nssw(lon_idx, lat_idx, :);
        else
            fprintf('警告: 未找到SSW数据，使用零值填充\n');
            essw = zeros(size(sst));
            nssw = zeros(size(sst));
        end
    end
catch
    fprintf('警告: SSW数据加载失败，使用零值填充\n');
    essw = zeros(size(sst));
    nssw = zeros(size(sst));
end

%% 5. 数据质量控制 - 填充陆地值为0
sst(isnan(sst)) = 0;
ssha(isnan(ssha)) = 0;
essw(isnan(essw)) = 0;
nssw(isnan(nssw)) = 0;

%% 6. 标准化处理（Z-score标准化）
fprintf('进行数据标准化...\n');

% 计算每个变量的均值和标准差
[data_norm.sst, data_mean.sst, data_std.sst] = zscore_normalize(sst);
[data_norm.ssha, data_mean.ssha, data_std.ssha] = zscore_normalize(ssha);
[data_norm.essw, data_mean.essw, data_std.essw] = zscore_normalize(essw);
[data_norm.nssw, data_mean.nssw, data_std.nssw] = zscore_normalize(nssw);

fprintf('数据标准化完成\n');
fprintf('SST - 均值: %.4f, 标准差: %.4f\n', data_mean.sst, data_std.sst);

end

%% 辅助函数：Z-score标准化
function [data_norm, data_mean, data_std] = zscore_normalize(data)
    % 只对非零值（海洋区域）进行标准化
    mask = data ~= 0;
    data_mean = mean(data(mask));
    data_std = std(data(mask));
    
    data_norm = zeros(size(data));
    data_norm(mask) = (data(mask) - data_mean) / data_std;
end

%% 辅助函数：生成示例数据
function [sst, lon, lat, time] = generate_sample_data(params)
    % 生成示例数据用于测试代码
    fprintf('生成示例数据...\n');
    
    % 创建经纬度网格（0.25°分辨率）
    lon = params.region.lon_range(1):0.25:params.region.lon_range(2);
    lat = params.region.lat_range(1):0.25:params.region.lat_range(2);
    
    % 创建时间序列（1993-2021年，每日）
    time = params.time.train_start:days(1):params.time.test_end;
    nt = length(time);
    
    fprintf('示例数据时间长度: %d天\n', nt);
    
    % 生成模拟SST数据（带季节性变化和趋势）
    [LON, LAT] = meshgrid(lon, lat);
    LON = LON';
    LAT = LAT';
    
    nx = length(lon);
    ny = length(lat);
    
    sst = zeros(nx, ny, nt);
    
    for t = 1:nt
        % 基础温度（随纬度变化）
        base_temp = 28 - 0.3 * abs(LAT - 10);
        
        % 季节性变化
        day_of_year = day(time(t), 'dayofyear');
        seasonal = 2 * sin(2 * pi * day_of_year / 365 - pi/2);
        
        % 随机波动
        noise = 0.5 * randn(nx, ny);
        
        % 合成SST
        sst(:,:,t) = base_temp + seasonal + noise;
    end
    
    % 设置陆地值为0（简单假设：纬度高于22度或经度小于106度为陆地）
    land_mask = (LAT > 22) | (LON < 106);
    for t = 1:nt
        sst(:,:,t) = sst(:,:,t) .* ~land_mask;
    end
    
    fprintf('示例数据生成完成\n');
end
