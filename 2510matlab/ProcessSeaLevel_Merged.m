%% ========================================================================
% ProcessSeaLevel_Complete_Optimized.m
%
% 功能全集：
%   1. 复杂 Excel 数据读取与清洗
%   2. ICEEMDAN 信号分解（含镜像延拓消除端点效应）
%   3. 纯 ARIMA 线性预测（高阶模型）
%   4. ICEEMDAN + ARIMA 非线性融合预测（低阶模型，防止过拟合）
%   5. 完整的绘图与误差分析
%
% 修复说明：
%   针对混合模型 RMSE 过大的问题，实施了“分量低阶化”策略，
%   并减少了分解层数，防止噪声拟合导致的预测发散。
%% ========================================================================

clear; clc; close all;

%% ---------------------------- 0. 路径设置 --------------------------------
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

%% ----------------------- 0.1 参数配置 (关键优化区) -----------------------
config = struct();

% [数据范围]
config.DataRange = [1 6524];        
% [训练范围]
config.TrainRange = [1 6500];    
% [预测步长]
config.ForecastLength = 24;         

% [ICEEMDAN 设置]
iceConfig = struct(...
    'NumRealizations', 100, ...     % 稍微减少次数以提高速度
    'NoiseStd', 0.2, ...            
    'MaxIMFs', 6, ...               % <--- [关键修改] 从 10 改为 6，防止过度分解噪声
    'ResidualEnergyTol', 1e-5, ...  
    'ExtensionRatio', 0.15, ...     % 镜像延拓比例，消除端点效应
    'Verbose', true);               

% [ARIMA 设置 - 分离策略]
% 1. 纯 ARIMA 配置：需要高阶来独自拟合复杂波形
arimaConfig_Pure = struct(...
    'P', 12, ...  
    'D', 1, ...   
    'Q', 12);     

% 2. 混合模型 IMF 配置：强制低阶！
% IMF 分量通常很简单，如果用高阶模型会过拟合导致预测发散（RMSE爆炸的原因）
arimaConfig_Hybrid = struct(...
    'P', 2, ...   % <--- [关键修改] 限制为低阶
    'D', 1, ...   
    'Q', 2);      

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

%% ------------------------- 2. ICEEMDAN 分解 -----------------------------
fprintf('\n[步骤 1] 执行改进版 ICEEMDAN 分解 (层数: %d, 延拓: %.2f)...\n', ...
    iceConfig.MaxIMFs, iceConfig.ExtensionRatio);

[imfs, residue, iceInfo] = runICEEMDAN(trainSeries, ...
    'NumRealizations', iceConfig.NumRealizations, ...
    'NoiseStd', iceConfig.NoiseStd, ...
    'MaxIMFs', iceConfig.MaxIMFs, ...
    'ResidualEnergyTol', iceConfig.ResidualEnergyTol, ...
    'ExtensionRatio', iceConfig.ExtensionRatio, ... 
    'Verbose', iceConfig.Verbose);

fprintf('ICEEMDAN 分解完成，共提取 %d 个 IMF 分量。\n', size(imfs, 2));

% --- 绘图：分解结果 ---
numIMFsToPlot = min(4, size(imfs, 2));
figure('Name', 'ICEEMDAN 分解结果', 'Color', 'w');
for k = 1:numIMFsToPlot
    subplot(numIMFsToPlot+1, 1, k);
    plot(trainTime, imfs(:, k), 'b', 'LineWidth', 1);
    ylabel(['IMF ', num2str(k)]); grid on;
    if k==1, title('ICEEMDAN 分解 (前4层 & 残差)'); end
end
subplot(numIMFsToPlot+1, 1, numIMFsToPlot+1);
plot(trainTime, residue, 'k', 'LineWidth', 1);
ylabel('Residue'); xlabel('时间'); grid on;

%% ------------------------- 3. 纯 ARIMA 预测 -----------------------------
fprintf('\n[步骤 2] 执行纯 ARIMA 预测 (使用高阶配置)...\n');

% 使用 arimaConfig_Pure (高阶)
[arimaModel, chosenOrder, arimaDiag] = fitARIMAModel(trainSeries, ...
    'MaxOrder', arimaConfig_Pure, 'Display', true);

arimaFcParams = forecastARIMA(arimaModel, trainSeries, forecastHorizon);
arimaPredVals = arimaFcParams.Forecast;

mseA = mean((arimaPredVals - groundTruth).^2);
rmseA = sqrt(mseA);
meA = mean(arimaPredVals - groundTruth);
fprintf('>> 纯 ARIMA 结果: RMSE=%.4f, ME=%.4f\n', rmseA, meA);

%% ------------------ 4. ICEEMDAN + ARIMA 融合预测 ------------------------
fprintf('\n[步骤 3] 执行 ICEEMDAN + ARIMA 融合预测 (使用低阶配置)...\n');

% 使用 arimaConfig_Hybrid (低阶)
hybridResults = runICEEMDAN_ARIMA(trainSeries, ...
    'ICEOptions', iceConfig, ...        
    'PrecomputedIMFs', {imfs, residue}, ... 
    'ForecastHorizon', forecastHorizon, ...
    'MaxOrder', arimaConfig_Hybrid, ...  % <--- 这里传入低阶配置
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


%% ========================================================================
%                             本地函数定义区
% ========================================================================

function [imfs, residue, info] = runICEEMDAN(signal, varargin)
    % RUNICEEMDAN: 改进版，包含镜像延拓处理
    signal = signal(:);
    n = numel(signal);
    
    parser = inputParser;
    addParameter(parser, 'NumRealizations', 200, @isnumeric);
    addParameter(parser, 'NoiseStd', 0.2, @isnumeric);
    addParameter(parser, 'MaxIMFs', 10, @isnumeric);
    addParameter(parser, 'ResidualEnergyTol', 1e-5, @isnumeric);
    addParameter(parser, 'Verbose', false, @islogical);
    addParameter(parser, 'ExtensionRatio', 0.0, @isnumeric); 
    addParameter(parser, 'BetaSchedule', [], @(x) true);
    parse(parser, varargin{:});
    opts = parser.Results;
    
    if exist('emd', 'file') ~= 2
        error('需要 Signal Processing Toolbox (emd 函数)。');
    end

    % --- [核心改进] 镜像延拓处理端点效应 ---
    nExt = floor(n * opts.ExtensionRatio);
    if nExt > 0
        leftExt = flipud(signal(2 : nExt+1));   
        rightExt = flipud(signal(end-nExt : end-1));
        signalProc = [leftExt; signal; rightExt];
        if opts.Verbose
            fprintf('[ICEEMDAN] 启用镜像延拓 (Ratio=%.2f): %d -> %d 点\n', ...
                opts.ExtensionRatio, n, numel(signalProc));
        end
    else
        signalProc = signal;
    end
    
    nProc = numel(signalProc);
    
    noiseIMFs = cell(opts.NumRealizations, 1);
    for i = 1:opts.NumRealizations
        omega = randn(nProc, 1);
        noiseIMFs{i} = computeImfs(omega, opts.MaxIMFs);
    end
    
    imfsProc = zeros(nProc, opts.MaxIMFs);
    residueProc = signalProc;
    
    betaFun = opts.BetaSchedule;
    if isempty(betaFun), betaFun = @(k) opts.NoiseStd / k; end
    
    info = struct('NumIMFs', 0);
    
    for k = 1:opts.MaxIMFs
        beta_k = betaFun(k);
        ensembleIMFs = zeros(nProc, opts.NumRealizations);
        
        if opts.Verbose
            fprintf('    计算 IMF %d ...\n', k);
        end
        
        for i = 1:opts.NumRealizations
            noiseComp = getImfColumn(noiseIMFs{i}, k);
            noisySig = residueProc + beta_k * noiseComp;
            tmpImf = computeImfs(noisySig, 1); 
            ensembleIMFs(:, i) = tmpImf(:, 1);
        end
        
        currentIMF = mean(ensembleIMFs, 2);
        imfsProc(:, k) = currentIMF;
        residueProc = residueProc - currentIMF;
        info.NumIMFs = info.NumIMFs + 1;
        
        % 检查停止条件 (仅基于原始数据段)
        if nExt > 0
            currResSegment = residueProc(nExt+1 : end-nExt);
        else
            currResSegment = residueProc;
        end
        
        % 使用简单的能量判断
        if norm(currResSegment) < opts.ResidualEnergyTol || isMonotonic(currResSegment)
            imfsProc = imfsProc(:, 1:k);
            break;
        end
    end
    
    % --- [核心改进] 截断回原始长度 ---
    if nExt > 0
        startIdx = numel(leftExt) + 1;
        endIdx = startIdx + n - 1;
        imfs = imfsProc(startIdx:endIdx, :);
        residue = residueProc(startIdx:endIdx);
    else
        imfs = imfsProc;
        residue = residueProc;
    end
end

function results = runICEEMDAN_ARIMA(signal, varargin)
    % 混合预测主控函数
    parser = inputParser;
    addParameter(parser, 'ICEOptions', struct(), @isstruct);
    addParameter(parser, 'PrecomputedIMFs', {}, @iscell); 
    addParameter(parser, 'ForecastHorizon', 24, @isnumeric);
    addParameter(parser, 'Order', [], @isnumeric);
    addParameter(parser, 'MaxOrder', struct('P',3,'D',1,'Q',3), @isstruct);
    addParameter(parser, 'GroundTruth', [], @isnumeric);
    addParameter(parser, 'Display', true, @islogical);
    parse(parser, varargin{:});
    opts = parser.Results;
    
    if ~isempty(opts.PrecomputedIMFs)
        imfs = opts.PrecomputedIMFs{1};
        residue = opts.PrecomputedIMFs{2};
        iceInfo = struct('Note', 'Precomputed');
    else
        iceArgs = struct2args(opts.ICEOptions);
        [imfs, residue, iceInfo] = runICEEMDAN(signal, iceArgs{:});
    end
    
    numIMFs = size(imfs, 2);
    numComponents = numIMFs + 1;
    componentNames = [arrayfun(@(k) sprintf('IMF%d',k), 1:numIMFs, 'Un',0), {'Residue'}];
    
    forecasts = zeros(opts.ForecastHorizon, numComponents);
    
    for idx = 1:numComponents
        if idx <= numIMFs
            compData = imfs(:, idx);
        else
            compData = residue;
        end
        
        if opts.Display
            fprintf('[Hybrid] 正在拟合组件 %s ... ', componentNames{idx});
        end
        
        if isempty(opts.Order)
            [mdl, order] = fitARIMAModel(compData, ...
                'MaxOrder', opts.MaxOrder, 'Display', false);
        else
            [mdl, order] = fitARIMAModel(compData, ...
                'Order', opts.Order, 'Display', false);
        end
        
        if opts.Display
            fprintf('最佳阶数: [%d %d %d]\n', order);
        end
        
        fcStruct = forecastARIMA(mdl, compData, opts.ForecastHorizon);
        forecasts(:, idx) = fcStruct.Forecast;
    end
    
    finalForecast = sum(forecasts, 2);
    
    metrics = struct();
    if ~isempty(opts.GroundTruth)
        gt = opts.GroundTruth(:);
        pred = finalForecast;
        len = min(numel(gt), numel(pred));
        metrics.R = corr(gt(1:len), pred(1:len));
        metrics.RMSE = sqrt(mean((gt(1:len) - pred(1:len)).^2));
        metrics.ME = mean(pred(1:len) - gt(1:len));
    end
    
    results.ICEEMDAN.IMFs = imfs;
    results.ICEEMDAN.Residue = residue;
    results.ICEEMDAN.Info = iceInfo;
    results.Components = componentNames;
    results.ComponentForecasts = forecasts;
    results.FinalForecast = finalForecast;
    results.Metrics = metrics;
end

function [model, order, diagnostics] = fitARIMAModel(data, varargin)
    parser = inputParser;
    addParameter(parser, 'Order', [], @isnumeric);
    addParameter(parser, 'MaxOrder', struct('P',3,'D',1,'Q',3), @isstruct);
    addParameter(parser, 'Display', false, @islogical);
    parse(parser, varargin{:});
    opts = parser.Results;
    
    if exist('arima', 'file') ~= 2
        error('需要 Econometrics Toolbox (arima 函数)。');
    end
    
    if isempty(opts.Order)
        order = autoSelectARIMAOrder(data, opts.MaxOrder, opts.Display);
    else
        order = opts.Order;
    end
    
    mdl = arima(order(1), order(2), order(3));
    
    try
        [model, ~, logL, info] = estimate(mdl, data, 'Display', 'off');
    catch
        warning('复杂模型估计失败，回退到 ARIMA(1,1,1)');
        model = estimate(arima(1,1,1), data, 'Display', 'off');
        logL = 0; info = struct();
        order = [1 1 1];
    end
    
    diagnostics.LogLikelihood = logL;
    diagnostics.Info = info;
end

function forecastStruct = forecastARIMA(model, data, horizon)
    [yPred, yMSE] = forecast(model, horizon, 'Y0', data);
    forecastStruct = struct('Forecast', yPred(:), 'MSE', yMSE(:));
end

function order = autoSelectARIMAOrder(data, maxOrder, ~)
    bestAIC = inf;
    bestOrder = [1 1 0]; 
    
    d_val = maxOrder.D; 
    
    % 限制搜索范围，防止过慢
    % 如果 MaxOrder.P 很大 (如12)，只搜索偶数阶以加快速度
    if maxOrder.P > 6
        p_range = 0:2:maxOrder.P;
    else
        p_range = 0:maxOrder.P;
    end
    
    q_range = 0:min(maxOrder.Q, 4); 
    
    for p = p_range
        for q = q_range
            try
                mdl = arima(p, d_val, q);
                [~, ~, logL] = estimate(mdl, data, 'Display', 'off');
                k = p + q + 1;
                aic = 2*k - 2*logL; 
                
                if aic < bestAIC
                    bestAIC = aic;
                    bestOrder = [p d_val q];
                end
            catch
                continue;
            end
        end
    end
    order = bestOrder;
end

function imfMatrix = computeImfs(signal, maxImfs)
    [c, ~] = emd(signal, 'MaxNumIMF', maxImfs, 'Display', 0);
    if isempty(c)
        imfMatrix = zeros(numel(signal), maxImfs); return;
    end
    cols = min(size(c, 2), maxImfs);
    imfMatrix = zeros(numel(signal), maxImfs);
    imfMatrix(:, 1:cols) = c(:, 1:cols);
end

function imf = getImfColumn(imfMatrix, k)
    if size(imfMatrix, 2) >= k, imf = imfMatrix(:, k);
    else, imf = zeros(size(imfMatrix, 1), 1); end
end

function tf = isMonotonic(x)
    d = diff(x); tf = all(d >= 0) || all(d <= 0);
end

function args = struct2args(s)
    names = fieldnames(s);
    args = cell(1, 2*numel(names));
    for i = 1:numel(names)
        args{2*i-1} = names{i}; args{2*i} = s.(names{i});
    end
end