%% ========================================================================
% ProcessSeaLevel_Complete_Optimized.m
%
% 功能全集：
%   1. 复杂 Excel 数据读取与清洗
%   2. ICEEMDAN 信号分解（含镜像延拓消除端点效应，支持并行计算）
%   3. 纯 ARIMA 线性预测（高阶模型）
%   4. ICEEMDAN + ARIMA 非线性融合预测（低阶模型，防止过拟合）
%   5. 完整的绘图与误差分析
%
% 优化说明：
%   - 代码已模块化拆分 (runICEEMDAN, fitARIMAModel, runICEEMDAN_ARIMA)
%   - ICEEMDAN 引入 parfor 并行计算加速
%   - 保持了原有的低阶化策略防止过拟合
%% ========================================================================

clear; clc; close all;

%% ---------------------------- 0. 路径设置 --------------------------------
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% 尝试开启并行池 (如果安装了 Parallel Computing Toolbox)
try
    if isempty(gcp('nocreate'))
        parpool; 
    end
catch
    warning('无法开启并行池，将使用串行模式或单核运行。');
end

%% ----------------------- 0.1 参数配置 (关键优化区) -----------------------
config = struct();

% [数据范围]
config.DataRange = [1 6524];        
% [训练范围]
config.TrainRange = [1 3500];    
% [预测步长]
config.ForecastLength = 24;         

% [ICEEMDAN 设置]
iceConfig = struct(...
    'NumRealizations', 500, ...     % [优化] 增加到 500 以获得更平滑的 IMF，利用并行计算加速
    'NoiseStd', 0.2, ...            
    'MaxIMFs', 6, ...               % 保持为 6，防止过度分解
    'ResidualEnergyTol', 1e-5, ...  
    'ExtensionRatio', 0.15, ...     % 镜像延拓比例
    'Verbose', true);               

% [ARIMA 设置 - 分离策略]
% 1. 纯 ARIMA 配置
arimaConfig_Pure = struct('P', 12, 'D', 1, 'Q', 12);     

% 2. 混合模型 IMF 配置
% [优化] 提高阶数上限，允许模型捕捉 12h/24h 潮汐周期
% 之前 P=2 导致欠拟合，现在放宽到 12，依靠 BIC 准则自动防止过拟合
% [关键修改] IMFs 本身是平稳的震荡分量，不应进行差分 (D=0)，否则会导致过差分引起振幅衰减
arimaConfig_Hybrid = struct('P', 12, 'D', 0, 'Q', 12);      

%% ------------------------ 1. 数据加载与预处理 ---------------------------
dataFile = '201300.xlsx'; 
defaultYear = 2013;

fprintf('正在加载数据: %s ...\n', dataFile);
if ~isfile(fullfile(thisDir, dataFile))
    error('未找到数据文件: %s。请确认文件位于脚本同目录下。', dataFile);
end

% 1.1 读取数据
opts = detectImportOptions(fullfile(thisDir, dataFile), 'VariableNamingRule', 'preserve');
rawTable = readtable(fullfile(thisDir, dataFile), opts);

% 1.2 提取数值矩阵
numCols = width(rawTable);
timeRaw = rawTable{:, 1};
lastDataCol = min(25, numCols); 
selectedVars = rawTable(:, 2:lastDataCol);

dataMatrix = zeros(height(selectedVars), width(selectedVars));
for c = 1:width(selectedVars)
    colData = selectedVars{:, c};
    if iscell(colData), colData = str2double(colData); end
    if isstring(colData), colData = str2double(colData); end
    dataMatrix(:, c) = colData;
end

% 1.3 清洗无效行
validRowMask = ~all(ismissing(dataMatrix), 2);
timeRaw = timeRaw(validRowMask);
dataMatrix = dataMatrix(validRowMask, :);
if ~isnumeric(dataMatrix), dataMatrix = str2double(dataMatrix); end

% 1.4 异常值处理
dataMatrix(dataMatrix <= -9999) = NaN;

% 1.5 构造时间轴
numRows = size(dataMatrix, 1);
numColsData = size(dataMatrix, 2);
hourSlots = 0:(numColsData-1);

try
    timeDaily = datetime(timeRaw, 'InputFormat', 'yyyy/M/d', 'Format', 'yyyy-MM-dd');
catch
    if isnumeric(timeRaw)
        timeDaily = NaT(numel(timeRaw), 1);
        curMonth = 1; prevDay = NaN;
        for r = 1:numel(timeRaw)
            d = timeRaw(r);
            if ~isnan(d)
                if ~isnan(prevDay) && d < prevDay, curMonth = curMonth + 1; end
                prevDay = d;
                try timeDaily(r) = datetime(defaultYear, curMonth, d); catch, end
            end
        end
    else
        timeDaily = (1:numel(timeRaw))'; 
    end
end

if isdatetime(timeDaily)
    timeMatrix = repmat(timeDaily, 1, numColsData) + hours(repmat(hourSlots, numRows, 1));
else
    timeMatrix = zeros(numRows, numColsData);
    for r = 1:numRows, timeMatrix(r, :) = (r-1)*numColsData + hourSlots; end
end

% 1.6 展平数据
seaLevelSeries = dataMatrix.'; seaLevelSeries = seaLevelSeries(:);
timeAxisSeries = timeMatrix.'; timeAxisSeries = timeAxisSeries(:);

validMask = ~isnan(seaLevelSeries);
seaLevel = seaLevelSeries(validMask);
timeAxis = timeAxisSeries(validMask);
totalSamples = numel(seaLevel);

% 1.7 应用数据范围
rangeIdx = config.DataRange(1):min(config.DataRange(2), totalSamples);
seaLevel = seaLevel(rangeIdx);
timeAxis = timeAxis(rangeIdx);
totalSamples = numel(seaLevel);

% 1.8 划分训练集与预测集
trainIdx = config.TrainRange(1):config.TrainRange(2);
trainIdx = trainIdx(trainIdx <= totalSamples);
forecastStart = trainIdx(end) + 1;
forecastIdx = forecastStart : min(forecastStart + config.ForecastLength - 1, totalSamples);

if isempty(forecastIdx)
    error('预测区间无效，请检查数据长度或 TrainRange 设置。');
end

trainSeries = seaLevel(trainIdx);
trainTime = timeAxis(trainIdx);
groundTruth = seaLevel(forecastIdx); 
testTime = timeAxis(forecastIdx);
forecastHorizon = numel(groundTruth);

fprintf('数据准备完毕: 训练样本 %d 个, 预测样本 %d 个。\n', numel(trainSeries), forecastHorizon);

%% ------------------------- 2. ICEEMDAN 分解 (含端点效应优化) -----------------------------
fprintf('\n[步骤 1] 执行改进版 ICEEMDAN 分解...\n');
fprintf('  > 策略优化: 使用 ARIMA 初步预测值对信号进行"预延拓"，以消除分解的端点效应。\n');

% 1. 构造延拓信号：[训练数据; ARIMA预测数据]
% 注意：这里利用了后面步骤计算的 arimaPredVals，因此我们需要先运行 ARIMA 预测，或者在此处先跑一个简单的 ARIMA
% 为了代码逻辑顺畅，我们将步骤顺序微调：先跑 ARIMA (步骤2)，再跑分解 (步骤1的优化版)

% --- 临时调整执行顺序 ---
% 我们先执行 ARIMA 预测 (原步骤3)，获取用于延拓的数据
fprintf('\n[预处理] 先执行纯 ARIMA 预测以获取延拓数据...\n');
[arimaModel, ~, ~] = fitARIMAModel(trainSeries, 'MaxOrder', arimaConfig_Pure, 'Display', false);
[yPred, ~] = forecast(arimaModel, forecastHorizon, 'Y0', trainSeries);
arimaPredVals = yPred;

% 2. 实施预延拓
trainSeries_Extended = [trainSeries; arimaPredVals];
fprintf('  > 原始长度: %d, 延拓后长度: %d (延拓了 %d 个点)\n', ...
    numel(trainSeries), numel(trainSeries_Extended), numel(arimaPredVals));

% 3. 对延拓后的信号进行分解
% 注意：这里 ExtensionRatio 可以设为 0 了，因为我们已经手动延拓了最关键的右侧
[imfsExt, residueExt, iceInfo] = runICEEMDAN(trainSeries_Extended, ...
    'NumRealizations', iceConfig.NumRealizations, ...
    'NoiseStd', iceConfig.NoiseStd, ...
    'MaxIMFs', iceConfig.MaxIMFs, ...
    'ResidualEnergyTol', iceConfig.ResidualEnergyTol, ...
    'Verbose', iceConfig.Verbose);

% 4. 截断回原始训练集长度 (去除延拓部分)
% 这样得到的 IMF 在 t=end 处就没有端点效应了！
validLen = numel(trainSeries);
imfs = imfsExt(1:validLen, :);
residue = residueExt(1:validLen);

fprintf('ICEEMDAN 分解完成 (已消除端点效应)，共提取 %d 个 IMF 分量。\n', size(imfs, 2));

% --- 绘图：分解结果 ---
numIMFsToPlot = min(4, size(imfs, 2));
figure('Name', 'ICEEMDAN 分解结果 (端点优化版)', 'Color', 'w');
for k = 1:numIMFsToPlot
    subplot(numIMFsToPlot+1, 1, k);
    plot(trainTime, imfs(:, k), 'b', 'LineWidth', 1);
    ylabel(['IMF ', num2str(k)]); grid on;
    if k==1, title('ICEEMDAN 分解 (前4层 & 残差) - 端点优化后'); end
end
subplot(numIMFsToPlot+1, 1, numIMFsToPlot+1);
plot(trainTime, residue, 'k', 'LineWidth', 1);
ylabel('Residue'); xlabel('时间'); grid on;

%% ------------------------- 3. 纯 ARIMA 预测 (结果展示) -----------------------------
fprintf('\n[步骤 2] 纯 ARIMA 预测结果 (已在预处理阶段计算)...\n');
% arimaPredVals 已经在上面计算过了，这里直接计算误差
mseA = mean((arimaPredVals - groundTruth).^2);
rmseA = sqrt(mseA);
meA = mean(arimaPredVals - groundTruth);
fprintf('>> 纯 ARIMA 结果: RMSE=%.4f, ME=%.4f\n', rmseA, meA);

% (原有的 fitARIMAModel 调用代码块已移除，因为挪到了前面)

%% ------------------ 4. ICEEMDAN + ARIMA 融合预测 ------------------------
fprintf('\n[步骤 3] 执行 ICEEMDAN + ARIMA 融合预测 (使用低阶配置)...\n');

% 调用外部函数 runICEEMDAN_ARIMA
hybridResults = runICEEMDAN_ARIMA(trainSeries, ...
    'ICEOptions', iceConfig, ...        
    'PrecomputedIMFs', {imfs, residue}, ... 
    'ForecastHorizon', forecastHorizon, ...
    'MaxOrder', arimaConfig_Hybrid, ...  
    'GroundTruth', groundTruth, ...
    'Display', true);

hybridPredVals = hybridResults.FinalForecast;
rmseH = hybridResults.Metrics.RMSE;
meH = hybridResults.Metrics.ME;
rH = hybridResults.Metrics.R;

fprintf('>> 融合模型结果: RMSE=%.4f, ME=%.4f, R=%.4f\n', rmseH, meH, rH);

%% ------------------------- 5. 结果可视化与对比 --------------------------
figure('Name', '预测结果对比', 'Color', 'w');
hold on;
plot(testTime, groundTruth, 'k-o', 'LineWidth', 1.5, 'DisplayName', '真实值');
plot(testTime, arimaPredVals, 'r--', 'LineWidth', 1.2, 'DisplayName', sprintf('ARIMA (RMSE=%.3f)', rmseA));
plot(testTime, hybridPredVals, 'b-^', 'LineWidth', 1.5, 'DisplayName', sprintf('ICEEMDAN+ARIMA (RMSE=%.3f)', rmseH));

title('潮位预测效果对比');
xlabel('时间'); ylabel('潮位 (cm)');
legend('Location', 'best');
grid on; box on;

% 绘图：各分量贡献
figure('Name', '各分量预测贡献', 'Color', 'w');
hold on;
colors = lines(size(hybridResults.ComponentForecasts, 2));
for i = 1:size(hybridResults.ComponentForecasts, 2)
    plot(testTime, hybridResults.ComponentForecasts(:, i), ...
        'Color', colors(i,:), 'LineWidth', 1, 'DisplayName', hybridResults.Components{i});
end
title('各 IMF 分量及残差的预测贡献');
xlabel('时间'); ylabel('贡献值');
legend('Location', 'bestoutside');
grid on;

fprintf('\n============= 运行结束 =============\n');
