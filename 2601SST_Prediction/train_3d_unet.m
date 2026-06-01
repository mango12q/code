function trained_net = train_3d_unet(net, X_train, Y_train, X_val, Y_val, params)
%% 训练3D U-Net模型
% 输入: net - 网络结构
%       X_train, Y_train - 训练数据
%       X_val, Y_val - 验证数据
%       params - 训练参数
% 输出: trained_net - 训练好的网络

fprintf('开始训练3D U-Net模型...\n');

% 数据准备 - 调整数据维度以适应MATLAB深度学习框架
% 输入: [height, width, channels, time, samples] -> [height, width, channels, time, samples]
% 输出: [height, width, time, samples] -> [height, width, 1, time, samples]

[nx, ny, n_channels, n_time_in, n_samples] = size(X_train);
[~, ~, n_time_out, ~] = size(Y_train);

fprintf('训练数据维度: X=%s, Y=%s\n', mat2str(size(X_train)), mat2str(size(Y_train)));

% 由于MATLAB的3D卷积对5D数据支持有限，这里使用自定义训练循环
% 或者转换为适合MATLAB的格式

% 训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', params.model.learning_rate, ...
    'MaxEpochs', params.model.epochs, ...
    'MiniBatchSize', params.model.batch_size, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', params.model.patience, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 10, ...
    'ExecutionEnvironment', 'auto');

% 注意：由于3D U-Net在MATLAB中的实现较为复杂
% 这里提供一个简化的实现方案
% 实际使用时可能需要使用Python的TensorFlow/PyTorch

fprintf('警告: MATLAB原生对5D数据（3D卷积）的支持有限\n');
fprintf('建议使用MATLAB的深度学习工具箱或转换为Python实现\n');
fprintf('这里提供一个基于2D卷积的替代方案...\n');

% 替代方案：使用2D U-Net + 时间维度作为通道
trained_net = train_2d_unet_alternative(X_train, Y_train, X_val, Y_val, params);

end

%% 替代方案：基于2D U-Net的实现
function net = train_2d_unet_alternative(X_train, Y_train, X_val, Y_val, params)
%% 使用2D U-Net处理时空数据（将时间维度合并到通道维度）

fprintf('使用2D U-Net替代方案...\n');

[nx, ny, n_channels, n_time_in, n_samples] = size(X_train);
[~, ~, n_time_out, ~] = size(Y_train);

% 重塑数据：将时间维度合并到通道维度
% 新维度: [nx, ny, n_channels * n_time_in, n_samples]
X_train_2d = reshape(X_train, nx, ny, n_channels * n_time_in, n_samples);
Y_train_2d = reshape(Y_train, nx, ny, n_time_out, n_samples);
X_val_2d = reshape(X_val, nx, ny, n_channels * n_time_in, size(X_val, 5));
Y_val_2d = reshape(Y_val, nx, ny, n_time_out, size(Y_val, 4));

fprintf('2D数据维度: X=%s, Y=%s\n', mat2str(size(X_train_2d)), mat2str(size(Y_train_2d)));

% 构建2D U-Net
num_input_channels = n_channels * n_time_in;
num_output_channels = n_time_out;

layers = [
    imageInputLayer([nx ny num_input_channels], 'Name', 'input', 'Normalization', 'none')
    
    % 编码器
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc_conv1_1')
    eluLayer('Name', 'enc_elu1_1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc_conv1_2')
    eluLayer('Name', 'enc_elu1_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc_pool1')
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'enc_conv2_1')
    eluLayer('Name', 'enc_elu2_1')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'enc_conv2_2')
    eluLayer('Name', 'enc_elu2_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc_pool2')
    
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'enc_conv3_1')
    eluLayer('Name', 'enc_elu3_1')
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'enc_conv3_2')
    eluLayer('Name', 'enc_elu3_2')
    
    % 解码器
    transposedConv2dLayer(2, 128, 'Stride', 2, 'Name', 'dec_up2')
    eluLayer('Name', 'dec_elu_up2')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'dec_conv2_1')
    eluLayer('Name', 'dec_elu2_1')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'dec_conv2_2')
    eluLayer('Name', 'dec_elu2_2')
    
    transposedConv2dLayer(2, 64, 'Stride', 2, 'Name', 'dec_up1')
    eluLayer('Name', 'dec_elu_up1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'dec_conv1_1')
    eluLayer('Name', 'dec_elu1_1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'dec_conv1_2')
    eluLayer('Name', 'dec_elu1_2')
    
    % 输出层
    convolution2dLayer(1, num_output_channels, 'Padding', 'same', 'Name', 'output_conv')
    regressionLayer('Name', 'output')
];

% 训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', params.model.learning_rate, ...
    'MaxEpochs', params.model.epochs, ...
    'MiniBatchSize', params.model.batch_size, ...
    'ValidationData', {X_val_2d, Y_val_2d}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', params.model.patience, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 10);

% 训练网络
fprintf('开始训练2D U-Net...\n');
net = trainNetwork(X_train_2d, Y_train_2d, layers, options);

fprintf('训练完成\n');

end
