function results = runICEEMDAN_ARIMA(signal, varargin)
    % RUNICEEMDAN_ARIMA: Hybrid forecasting using ICEEMDAN and ARIMA
    % Supports 'PrecomputedIMFs' to allow external decomposition (e.g. for predictive extension).
    
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
    
    % --- 1. Decomposition (or use precomputed) ---
    if ~isempty(opts.PrecomputedIMFs)
        imfs = opts.PrecomputedIMFs{1};
        residue = opts.PrecomputedIMFs{2};
        iceInfo = struct('Note', 'Precomputed IMFs used');
        if opts.Display
            fprintf('[Hybrid] Using precomputed IMFs (skipping internal decomposition)...\n');
        end
    else
        iceArgs = struct2args(opts.ICEOptions);
        [imfs, residue, iceInfo] = runICEEMDAN(signal, iceArgs{:});
    end
    
    numIMFs = size(imfs, 2);
    numComponents = numIMFs + 1;
    componentNames = [arrayfun(@(k) sprintf('IMF%d',k), 1:numIMFs, 'Un',0), {'Residue'}];
    
    forecasts = zeros(opts.ForecastHorizon, numComponents);
    
    % --- 2. Component Forecasting ---
    % Initialize models and orders storage
    models = cell(numComponents, 1);
    orders = zeros(numComponents, 3);
    diagnostics = repmat(struct('Order', [], 'LogLikelihood', []), numComponents, 1);

    for idx = 1:numComponents
        if idx <= numIMFs
            compData = imfs(:, idx);
        else
            compData = residue;
        end
        
        if opts.Display
            fprintf('[Hybrid] Fitting component %s ... ', componentNames{idx});
        end
        
        if isempty(opts.Order)
            [mdl, order, diag] = fitARIMAModel(compData, ...
                'MaxOrder', opts.MaxOrder, 'Display', false);
        else
            [mdl, order, diag] = fitARIMAModel(compData, ...
                'Order', opts.Order, 'Display', false);
        end
        
        models{idx} = mdl;
        orders(idx, :) = order;
        diagnostics(idx).Order = order;
        diagnostics(idx).LogLikelihood = diag.LogLikelihood;
        
        if opts.Display
            fprintf('Best Order: [%d %d %d]\n', order);
        end
        
        fcStruct = forecastARIMA(mdl, compData, opts.ForecastHorizon);
        forecasts(:, idx) = fcStruct.Forecast;
    end
    
    finalForecast = sum(forecasts, 2);
    
    % --- 3. Metrics ---
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
    results.Models = models;
    results.Orders = orders;
    results.ComponentForecasts = forecasts;
    results.FinalForecast = finalForecast;
    results.Diagnostics = diagnostics;
    results.Metrics = metrics;
end

function forecastStruct = forecastARIMA(model, data, horizon)
    [yPred, yMSE] = forecast(model, horizon, 'Y0', data);
    forecastStruct = struct('Forecast', yPred(:), 'MSE', yMSE(:));
end

function args = struct2args(s)
    names = fieldnames(s);
    args = cell(1, 2*numel(names));
    for i = 1:numel(names)
        args{2*i-1} = names{i}; args{2*i} = s.(names{i});
    end
end
