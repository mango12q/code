%% 基于深度学习的南海海表面温度智能化预测研究 - 主程序
% 复现论文《基于深度学习的南海海表面温度的智能化预测研究》
% 作者：谢博闻等
% 期刊：海洋与湖沼 2024

clear; clc; close all;

%% 1. 参数设置
params = struct();
params.region.lon_range = [105, 122.5];  % 经度范围 105°~122.5°E
params.region.lat_range = [0, 23];       % 纬度范围 0°~23°N
params.time.train_start = datetime(1993, 1, 1);
params.time.train_end = datetime(2020, 12, 31);
params.time.test_start = datetime(2021, 1, 1);
params.time.test_end = datetime(2021, 12, 31);
params.model.input_days = 64;   % 输入历史64天
params.model.output_days = 30;  % 预测未来30天
params.model.batch_size = 12;
params.model.learning_rate = 0.01;
params.model.epochs = 1000;
params.model.patience = 15;  % 早停耐心值

% 数据路径（请根据实际路径修改）
params.paths.sst = 'data/SST/';
params.paths.ssha = 'data/SSHA/';
params.paths.ssw = 'data/SSW/';

fprintf('=== 南海海表面温度预测模型 ===\n');
fprintf('研究区域: %.1f°~%.1f°E, %.1f°~%.1f°N\n', ...
    params.region.lon_range(1), params.region.lon_range(2), ...
    params.region.lat_range(1), params.region.lat_range(2));
fprintf('输入时间长度: %d天\n', params.model.input_days);
fprintf('预测时间长度: %d天\n', params.model.output_days);

%% 2. 数据加载与预处理
fprintf('\n--- 数据加载与预处理 ---\n');
[data_norm, data_mean, data_std, lon, lat, time_all] = load_and_preprocess_data(params);

% 划分训练集和测试集
[train_data, test_data, train_time, test_time] = split_train_test(data_norm, time_all, params);

fprintf('训练集样本数: %d\n', size(train_data.sst, 4) - params.model.input_days - params.model.output_days + 1);
fprintf('测试集样本数: %d\n', size(test_data.sst, 4) - params.model.input_days - params.model.output_days + 1);

%% 3. 构建训练样本
fprintf('\n--- 构建训练样本 ---\n');
[X_train, Y_train] = build_sequences(train_data, params.model.input_days, params.model.output_days);
[X_test, Y_test] = build_sequences(test_data, params.model.input_days, params.model.output_days);

fprintf('训练样本: X=%s, Y=%s\n', mat2str(size(X_train)), mat2str(size(Y_train)));
fprintf('测试样本: X=%s, Y=%s\n', mat2str(size(X_test)), mat2str(size(Y_test)));

%% 4. 训练3D U-Net模型
fprintf('\n--- 训练3D U-Net模型 ---\n');
unet_model = build_3d_unet(params);
unet_model = train_3d_unet(unet_model, X_train, Y_train, X_test, Y_test, params);

%% 5. 训练ConvLSTM模型（对比）
fprintf('\n--- 训练ConvLSTM模型 ---\n');
convlstm_model = build_convlstm(params);
convlstm_model = train_convlstm(convlstm_model, X_train, Y_train, X_test, Y_test, params);

%% 6. 模型预测
fprintf('\n--- 模型预测 ---\n');
Y_pred_unet = predict(unet_model, X_test);
Y_pred_convlstm = predict(convlstm_model, X_test);

%% 7. 反标准化
Y_test_denorm = denormalize(Y_test, data_mean.sst, data_std.sst);
Y_pred_unet_denorm = denormalize(Y_pred_unet, data_mean.sst, data_std.sst);
Y_pred_convlstm_denorm = denormalize(Y_pred_convlstm, data_mean.sst, data_std.sst);

%% 8. 评估模型性能
fprintf('\n--- 模型评估 ---\n');
evaluate_models(Y_test_denorm, Y_pred_unet_denorm, Y_pred_convlstm_denorm, params);

%% 9. 可视化结果
fprintf('\n--- 生成可视化结果 ---\n');
visualize_results(Y_test_denorm, Y_pred_unet_denorm, Y_pred_convlstm_denorm, lon, lat, params);

fprintf('\n=== 程序运行完成 ===\n');
