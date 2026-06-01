function evaluate_models(Y_true, Y_pred_unet, Y_pred_convlstm, params)
%% 评估模型性能
% 计算论文中提到的评估指标：RMSE, R, SMAPE, MedAE

fprintf('\n========== 模型评估结果 ==========\n');

output_days = params.model.output_days;

% 评估不同预测超前时间的性能
lead_times = [1, 7, 14, 30];  % 1天, 7天, 14天, 30天

fprintf('\n--- 3D U-Net模型性能 ---\n');
for i = 1:length(lead_times)
    day = lead_times(i);
    if day <= output_days
        y_true = Y_true(:, :, day, :);
        y_pred = Y_pred_unet(:, :, day, :);
        
        [rmse, r, smape, medae] = calculate_metrics(y_true, y_pred);
        
        fprintf('第%2d天 - RMSE: %.4f°C, R: %.4f, SMAPE: %.2f%%, MedAE: %.4f°C\n', ...
            day, rmse, r, smape*100, medae);
    end
end

fprintf('\n--- ConvLSTM模型性能 ---\n');
for i = 1:length(lead_times)
    day = lead_times(i);
    if day <= output_days
        y_true = Y_true(:, :, day, :);
        y_pred = Y_pred_convlstm(:, :, day, :);
        
        [rmse, r, smape, medae] = calculate_metrics(y_true, y_pred);
        
        fprintf('第%2d天 - RMSE: %.4f°C, R: %.4f, SMAPE: %.2f%%, MedAE: %.4f°C\n', ...
            day, rmse, r, smape*100, medae);
    end
end

% 计算整体性能（所有预测天数的平均）
fprintf('\n--- 整体性能（30天平均） ---\n');
[rmse_u, r_u, smape_u, medae_u] = calculate_metrics(Y_true, Y_pred_unet);
[rmse_c, r_c, smape_c, medae_c] = calculate_metrics(Y_true, Y_pred_convlstm);

fprintf('3D U-Net  - RMSE: %.4f°C, R: %.4f, SMAPE: %.2f%%, MedAE: %.4f°C\n', ...
    rmse_u, r_u, smape_u*100, medae_u);
fprintf('ConvLSTM  - RMSE: %.4f°C, R: %.4f, SMAPE: %.2f%%, MedAE: %.4f°C\n', ...
    rmse_c, r_c, smape_c*100, medae_c);

end

%% 计算评估指标
function [rmse, r, smape, medae] = calculate_metrics(y_true, y_pred)
    % 展平数据
    y_true_flat = y_true(:);
    y_pred_flat = y_pred(:);
    
    % 只考虑非零值（海洋区域）
    mask = y_true_flat ~= 0;
    y_true_valid = y_true_flat(mask);
    y_pred_valid = y_pred_flat(mask);
    
    n = length(y_true_valid);
    
    % RMSE (均方根误差)
    rmse = sqrt(mean((y_true_valid - y_pred_valid).^2));
    
    % R (皮尔逊相关系数)
    y_mean = mean(y_true_valid);
    y_pred_mean = mean(y_pred_valid);
    numerator = sum((y_true_valid - y_mean) .* (y_pred_valid - y_pred_mean));
    denominator = sqrt(sum((y_true_valid - y_mean).^2) * sum((y_pred_valid - y_pred_mean).^2));
    if denominator > 0
        r = numerator / denominator;
    else
        r = 0;
    end
    
    % SMAPE (对称平均绝对百分比误差)
    smape = mean(abs(y_true_valid - y_pred_valid) ./ ((abs(y_true_valid) + abs(y_pred_valid)) / 2));
    
    % MedAE (中值绝对误差)
    medae = median(abs(y_true_valid - y_pred_valid));
end
