function net = build_3d_unet(params)
%% 构建3D U-Net模型
% 根据论文描述，构建用于南海SST预测的3D U-Net模型
% 输入: params - 参数结构体
% 输出: net - 3D U-Net网络

fprintf('构建3D U-Net模型...\n');

% 获取输入尺寸
input_days = params.model.input_days;
output_days = params.model.output_days;

% 注意：这里使用占位符尺寸，实际训练时会根据数据调整
% 假设输入尺寸为 [height, width, channels, time]

% 3D U-Net参数
encoder_layers = 3;  % 编码器层数
decoder_layers = 3;  % 解码器层数
hidden_dims = [64, 128, 256];  % 隐藏层维度
conv_kernel_size = 3;  % 3D卷积核大小
pool_size = 2;  % 池化大小

% 创建层图
layers = [
    % 输入层
    image3dInputLayer([NaN NaN 4 input_days], 'Name', 'input', 'Normalization', 'none')
    
    % ========== 编码器部分 ==========
    % 第1层编码器
    convolution3dLayer(conv_kernel_size, hidden_dims(1), 'Padding', 'same', 'Name', 'conv3d_1_1')
    eluLayer('Name', 'elu_1_1')
    convolution3dLayer(conv_kernel_size, hidden_dims(1), 'Padding', 'same', 'Name', 'conv3d_1_2')
    eluLayer('Name', 'elu_1_2')
    maxPooling3dLayer([pool_size pool_size pool_size], 'Stride', [pool_size pool_size pool_size], 'Name', 'pool_1')
    
    % 第2层编码器
    convolution3dLayer(conv_kernel_size, hidden_dims(2), 'Padding', 'same', 'Name', 'conv3d_2_1')
    eluLayer('Name', 'elu_2_1')
    convolution3dLayer(conv_kernel_size, hidden_dims(2), 'Padding', 'same', 'Name', 'conv3d_2_2')
    eluLayer('Name', 'elu_2_2')
    maxPooling3dLayer([pool_size pool_size pool_size], 'Stride', [pool_size pool_size pool_size], 'Name', 'pool_2')
    
    % 第3层编码器（瓶颈层）
    convolution3dLayer(conv_kernel_size, hidden_dims(3), 'Padding', 'same', 'Name', 'conv3d_3_1')
    eluLayer('Name', 'elu_3_1')
    convolution3dLayer(conv_kernel_size, hidden_dims(3), 'Padding', 'same', 'Name', 'conv3d_3_2')
    eluLayer('Name', 'elu_3_2')
    
    % ========== 解码器部分 ==========
    % 第3层解码器
    transposedConv3dLayer([pool_size pool_size pool_size], hidden_dims(2), 'Stride', [pool_size pool_size pool_size], 'Name', 'deconv3d_3')
    eluLayer('Name', 'elu_de_3')
    % 跳跃连接将在训练时通过层连接实现
    convolution3dLayer(conv_kernel_size, hidden_dims(2), 'Padding', 'same', 'Name', 'conv3d_de_3_1')
    eluLayer('Name', 'elu_de_3_1')
    convolution3dLayer(conv_kernel_size, hidden_dims(2), 'Padding', 'same', 'Name', 'conv3d_de_3_2')
    eluLayer('Name', 'elu_de_3_2')
    
    % 第2层解码器
    transposedConv3dLayer([pool_size pool_size pool_size], hidden_dims(1), 'Stride', [pool_size pool_size pool_size], 'Name', 'deconv3d_2')
    eluLayer('Name', 'elu_de_2')
    convolution3dLayer(conv_kernel_size, hidden_dims(1), 'Padding', 'same', 'Name', 'conv3d_de_2_1')
    eluLayer('Name', 'elu_de_2_1')
    convolution3dLayer(conv_kernel_size, hidden_dims(1), 'Padding', 'same', 'Name', 'conv3d_de_2_2')
    eluLayer('Name', 'elu_de_2_2')
    
    % 第1层解码器
    transposedConv3dLayer([pool_size pool_size pool_size], 32, 'Stride', [pool_size pool_size pool_size], 'Name', 'deconv3d_1')
    eluLayer('Name', 'elu_de_1')
    convolution3dLayer(conv_kernel_size, 32, 'Padding', 'same', 'Name', 'conv3d_de_1_1')
    eluLayer('Name', 'elu_de_1_1')
    
    % 输出层 - 预测SST
    convolution3dLayer(1, output_days, 'Padding', 'same', 'Name', 'output_conv')
    regressionLayer('Name', 'output')
];

% 创建网络
net = layerGraph(layers);

% 添加跳跃连接（Skip Connections）
% 编码器第1层 -> 解码器第1层
net = addLayers(net, [
    convolution3dLayer(1, hidden_dims(1), 'Name', 'skip_conv_1', 'Padding', 'same')
    ]);
net = connectLayers(net, 'elu_1_2', 'skip_conv_1');

% 编码器第2层 -> 解码器第2层
net = addLayers(net, [
    convolution3dLayer(1, hidden_dims(2), 'Name', 'skip_conv_2', 'Padding', 'same')
    ]);
net = connectLayers(net, 'elu_2_2', 'skip_conv_2');

fprintf('3D U-Net模型构建完成\n');
fprintf('编码器层数: %d\n', encoder_layers);
fprintf('解码器层数: %d\n', decoder_layers);
fprintf('隐藏层维度: [%s]\n', num2str(hidden_dims));

end
