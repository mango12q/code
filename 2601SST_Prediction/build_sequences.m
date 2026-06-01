function [X, Y] = build_sequences(data, input_days, output_days)
%% 构建输入输出序列
% 输入: data - 数据（包含sst, ssha, essw, nssw）
%       input_days - 输入历史天数
%       output_days - 预测未来天数
% 输出: X - 输入序列 [lon, lat, channels, time, samples]
%       Y - 输出序列 [lon, lat, output_days, samples]

fprintf('构建序列样本...\n');

[nx, ny, nt] = size(data.sst);
num_channels = 4;  % SST, SSHA, ESSW, NSSW

% 计算样本数
num_samples = nt - input_days - output_days + 1;

% 初始化数组
% X: [空间x, 空间y, 通道数, 时间步长, 样本数]
X = zeros(nx, ny, num_channels, input_days, num_samples);
% Y: [空间x, 空间y, 预测天数, 样本数]
Y = zeros(nx, ny, output_days, num_samples);

for i = 1:num_samples
    % 构建输入序列 (input_days天)
    for t = 1:input_days
        X(:, :, 1, t, i) = data.sst(:, :, i + t - 1);      % SST
        X(:, :, 2, t, i) = data.ssha(:, :, i + t - 1);     % SSHA
        X(:, :, 3, t, i) = data.essw(:, :, i + t - 1);     % ESSW
        X(:, :, 4, t, i) = data.nssw(:, :, i + t - 1);     % NSSW
    end
    
    % 构建输出序列 (output_days天) - 只预测SST
    for t = 1:output_days
        Y(:, :, t, i) = data.sst(:, :, i + input_days + t - 1);
    end
end

fprintf('样本构建完成: 共%d个样本\n', num_samples);

end
