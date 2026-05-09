function [model, order, diagnostics] = fitARIMAModel(data, varargin)
    % FITARIMAMODEL: Fit ARIMA model with automatic or specified order
    
    parser = inputParser;
    addParameter(parser, 'Order', [], @isnumeric);
    addParameter(parser, 'MaxOrder', struct('P',3,'D',1,'Q',3), @isstruct);
    addParameter(parser, 'Display', false, @islogical);
    parse(parser, varargin{:});
    opts = parser.Results;
    
    if exist('arima', 'file') ~= 2
        error('Econometrics Toolbox (arima function) is required.');
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
        warning('Complex model estimation failed, falling back to ARIMA(1,1,1)');
        model = estimate(arima(1,1,1), data, 'Display', 'off');
        logL = 0; info = struct();
        order = [1 1 1];
    end
    
    diagnostics.LogLikelihood = logL;
    diagnostics.Info = info;
end

function order = autoSelectARIMAOrder(data, maxOrder, ~)
    bestBIC = inf;
    bestOrder = [1 1 0]; 
    
    d_val = maxOrder.D; 
    
    % Optimize search space
    if maxOrder.P > 6
        p_range = 0:2:maxOrder.P; % Search even orders for high P
    else
        p_range = 0:maxOrder.P;
    end
    
    q_range = 0:maxOrder.Q; 
    
    % Flatten the loop for parfor
    [P, Q] = meshgrid(p_range, q_range);
    combinations = [P(:), Q(:)];
    numCombs = size(combinations, 1);
    
    bics = inf(numCombs, 1);
    
    % Use parfor for parallel grid search
    parfor i = 1:numCombs
        p = combinations(i, 1);
        q = combinations(i, 2);
        try
            mdl = arima(p, d_val, q);
            [~, ~, logL] = estimate(mdl, data, 'Display', 'off');
            
            % Calculate AIC: 2k - 2ln(L)
            k = p + q + 1;
            aic = 2*k - 2*logL; 
            
            aics(i) = aic;
        catch
            aics(i) = inf;
        end
    end
    
    [minAIC, idx] = min(aics);
    if ~isinf(minAIC)
        bestOrder = [combinations(idx, 1), d_val, combinations(idx, 2)];
    end
    
    order = bestOrder;
end
