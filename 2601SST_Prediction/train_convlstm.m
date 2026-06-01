function trained_net = train_convlstm(net, X_train, Y_train, X_val, Y_val, params)
%% 训练ConvLSTM模型
% 输入: net - 网络结构
%       X_train, Y_train - 训练数据
%       X_val, Y_val - 验证数据
%       params - 训练参数
% 输出: trained_net - 训练好的网络

fprintf('开始训练ConvLSTM模型...\n');

% 由于MATLAB原生不支持ConvLSTM，这里提供一个简化的替代方案
% 使用序列到序列的LSTM模型

[nx, ny, n_channels, n_time_in, n_samples] = size(X_train);
[~, ~, n_time_out, ~] = size(Y_train);

% 将空间维度展平，作为特征
num_features = nx * ny * n_channels;
num_responses = nx * ny * n_time_out;

% 重塑数据
X_train_seq = reshape(X_train, num_features, n_samples);
Y_train_seq = reshape(Y_train, num_responses, n_samples);
X_val_seq = reshape(X_val, num_features, size(X_val, 5));
Y_val_seq = reshape(Y_val, num_responses, size(Y_val, 4));

% 转置以适应LSTM输入格式 [samples, features]
X_train_seq = X_train_seq';
Y_train_seq = Y_train_seq';
X_val_seq = X_val_seq';
Y_val_seq = Y_val_seq';

fprintf('序列数据维度: X=%s, Y=%s\n', mat2str(size(X_train_seq)), mat2str(size(Y_train_seq)));

% 构建简单的LSTM网络
layers = [
    sequenceInputLayer(num_features, 'Name', 'input')
    lstmLayer(128, 'Name', 'lstm1', 'OutputMode', 'sequence')
    dropoutLayer(0.2, 'Name', 'drop1')
    lstmLayer(128, 'Name', 'lstm2', 'OutputMode', 'last')
    dropoutLayer(0.2, 'Name', 'drop2')
    fullyConnectedLayer(num_responses, 'Name', 'fc')
    regressionLayer('Name', 'output')
];

% 训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', params.model.learning_rate, ...
    'MaxEpochs', params.model.epochs, ...
    'MiniBatchSize', params.model.batch_size, ...
    'ValidationData', {X_val_seq', Y_val_seq'}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', params.model.patience, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 10);

% 训练网络
fprintf('开始训练LSTM网络...\n');
trained_net = trainNetwork(X_train_seq', Y_train_seq', layers, options);

fprintf('训练完成\n');

end
