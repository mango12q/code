function visualize_results(Y_true, Y_pred_unet, Y_pred_convlstm, lon, lat, params)
%% 可视化结果
% 生成论文中的各种图表

output_days = params.model.output_days;

%% 1. 绘制RMSE和R随预测超前时间的变化（类似论文图3）
figure('Name', '模型性能对比', 'Position', [100 100 1200 500]);

lead_times = 1:output_days;
rmse_unet = zeros(size(lead_times));
rmse_convlstm = zeros(size(lead_times));
r_unet = zeros(size(lead_times));
r_convlstm = zeros(size(lead_times));

for i = 1:length(lead_times)
    day = lead_times(i);
    y_true = Y_true(:, :, day, :);
    y_pred_u = Y_pred_unet(:, :, day, :);
    y_pred_c = Y_pred_convlstm(:, :, day, :);
    
    [rmse_unet(i), r_unet(i), ~, ~] = calculate_metrics_simple(y_true, y_pred_u);
    [rmse_convlstm(i), r_convlstm(i), ~, ~] = calculate_metrics_simple(y_true, y_pred_c);
end

% 子图1: RMSE
subplot(1, 2, 1);
plot(lead_times, rmse_unet, 'b-', 'LineWidth', 2, 'DisplayName', '3D U-Net');
hold on;
plot(lead_times, rmse_convlstm, 'r--', 'LineWidth', 2, 'DisplayName', 'ConvLSTM');
hold off;
xlabel('预测超前时间 (天)');
ylabel('RMSE (°C)');
title('均方根误差对比');
legend('Location', 'best');
grid on;

% 子图2: R
subplot(1, 2, 2);
plot(lead_times, r_unet, 'b-', 'LineWidth', 2, 'DisplayName', '3D U-Net');
hold on;
plot(lead_times, r_convlstm, 'r--', 'LineWidth', 2, 'DisplayName', 'ConvLSTM');
hold off;
xlabel('预测超前时间 (天)');
ylabel('皮尔逊相关系数 R');
title('相关系数对比');
legend('Location', 'best');
grid on;

saveas(gcf, 'results/performance_comparison.png');

%% 2. 绘制散点图（类似论文图4）
figure('Name', '散点图对比', 'Position', [100 100 1200 600]);

% 选择第30天的预测结果
day = 30;
y_true_flat = Y_true(:, :, day, :);
y_pred_u_flat = Y_pred_unet(:, :, day, :);
y_pred_c_flat = Y_pred_convlstm(:, :, day, :);

% 展平并过滤非零值
mask = y_true_flat ~= 0;
y_true_valid = y_true_flat(mask);
y_pred_u_valid = y_pred_u_flat(mask);
y_pred_c_valid = y_pred_c_flat(mask);

% 随机采样一部分点以便可视化（避免点太多）
n_samples = min(5000, length(y_true_valid));
idx = randperm(length(y_true_valid), n_samples);

subplot(1, 2, 1);
scatter(y_true_valid(idx), y_pred_u_valid(idx), 10, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot([min(y_true_valid), max(y_true_valid)], [min(y_true_valid), max(y_true_valid)], 'r--', 'LineWidth', 2);
hold off;
xlabel('观测 SST (°C)');
ylabel('预测 SST (°C)');
title('3D U-Net 散点图');
grid on;

[rmse, r] = calculate_metrics_simple(y_true_flat, y_pred_u_flat);
text(0.05, 0.95, sprintf('RMSE = %.2f°C\nR = %.2f', rmse, r), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'BackgroundColor', 'w');

subplot(1, 2, 2);
scatter(y_true_valid(idx), y_pred_c_valid(idx), 10, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot([min(y_true_valid), max(y_true_valid)], [min(y_true_valid), max(y_true_valid)], 'r--', 'LineWidth', 2);
hold off;
xlabel('观测 SST (°C)');
ylabel('预测 SST (°C)');
title('ConvLSTM 散点图');
grid on;

[rmse, r] = calculate_metrics_simple(y_true_flat, y_pred_c_flat);
text(0.05, 0.95, sprintf('RMSE = %.2f°C\nR = %.2f', rmse, r), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'BackgroundColor', 'w');

saveas(gcf, 'results/scatter_plots.png');

%% 3. 绘制空间分布图（类似论文图5）
figure('Name', '空间分布对比', 'Position', [100 100 1500 500]);

% 选择一个样本和预测天数
sample_idx = 1;
day = 15;

% 观测值
subplot(1, 3, 1);
imagesc(lon, lat, Y_true(:, :, day, sample_idx)');
axis xy;
colorbar;
clim([min(Y_true(:)), max(Y_true(:))]);
xlabel('经度 (°E)');
ylabel('纬度 (°N)');
title('观测 SST');

% 3D U-Net预测
subplot(1, 3, 2);
imagesc(lon, lat, Y_pred_unet(:, :, day, sample_idx)');
axis xy;
colorbar;
clim([min(Y_true(:)), max(Y_true(:))]);
xlabel('经度 (°E)');
ylabel('纬度 (°N)');
title('3D U-Net 预测');

% 误差
subplot(1, 3, 3);
error = Y_true(:, :, day, sample_idx) - Y_pred_unet(:, :, day, sample_idx);
imagesc(lon, lat, error');
axis xy;
colorbar;
clim([-2, 2]);
xlabel('经度 (°E)');
ylabel('纬度 (°N)');
title('预测误差 (观测-预测)');

colormap(jet);
saveas(gcf, 'results/spatial_distribution.png');

%% 4. 绘制误差分布直方图（类似论文图6）
figure('Name', '误差分布', 'Position', [100 100 1000 600]);

% 计算30天预测的误差
errors = [];
for day = 1:output_days
    error_day = Y_true(:, :, day, :) - Y_pred_unet(:, :, day, :);
    error_flat = error_day(:);
    error_flat = error_flat(error_flat ~= 0);  % 过滤陆地
    errors = [errors; error_flat];
end

histogram(errors, 50, 'Normalization', 'pdf');
hold on;

% 拟合高斯分布
x_range = linspace(min(errors), max(errors), 100);
mu = mean(errors);
sigma = std(errors);
gauss_fit = normpdf(x_range, mu, sigma);
plot(x_range, gauss_fit, 'r-', 'LineWidth', 2);
hold off;

xlabel('预测误差 (°C)');
ylabel('概率密度');
title('预测误差分布（高斯核密度估计）');
legend('误差直方图', '高斯拟合');
grid on;

text(0.05, 0.95, sprintf('均值 = %.2f°C\n标准差 = %.2f°C', mu, sigma), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'BackgroundColor', 'w');

saveas(gcf, 'results/error_distribution.png');

fprintf('可视化结果已保存到 results/ 目录\n');

end

%% 简化的评估指标计算函数
function [rmse, r, smape, medae] = calculate_metrics_simple(y_true, y_pred)
    y_true_flat = y_true(:);
    y_pred_flat = y_pred(:);
    
    mask = y_true_flat ~= 0;
    y_true_valid = y_true_flat(mask);
    y_pred_valid = y_pred_flat(mask);
    
    % RMSE
    rmse = sqrt(mean((y_true_valid - y_pred_valid).^2));
    
    % R
    y_mean = mean(y_true_valid);
    y_pred_mean = mean(y_pred_valid);
    numerator = sum((y_true_valid - y_mean) .* (y_pred_valid - y_pred_mean));
    denominator = sqrt(sum((y_true_valid - y_mean).^2) * sum((y_pred_valid - y_pred_mean).^2));
    if denominator > 0
        r = numerator / denominator;
    else
        r = 0;
    end
    
    % SMAPE
    smape = mean(abs(y_true_valid - y_pred_valid) ./ ((abs(y_true_valid) + abs(y_pred_valid)) / 2));
    
    % MedAE
    medae = median(abs(y_true_valid - y_pred_valid));
end
