function [train_data, test_data, train_time, test_time] = split_train_test(data_norm, time_all, params)
%% 划分训练集和测试集
% 按照论文要求：
% 训练集：1993年1月1日 - 2020年12月31日
% 测试集：2021年1月1日 - 2021年12月31日

fprintf('划分训练集和测试集...\n');

% 找到训练集和测试集的索引
train_idx = find(time_all >= params.time.train_start & time_all <= params.time.train_end);
test_idx = find(time_all >= params.time.test_start & time_all <= params.time.test_end);

% 划分数据
train_data.sst = data_norm.sst(:, :, train_idx);
train_data.ssha = data_norm.ssha(:, :, train_idx);
train_data.essw = data_norm.essw(:, :, train_idx);
train_data.nssw = data_norm.nssw(:, :, train_idx);

test_data.sst = data_norm.sst(:, :, test_idx);
test_data.ssha = data_norm.ssha(:, :, test_idx);
test_data.essw = data_norm.essw(:, :, test_idx);
test_data.nssw = data_norm.nssw(:, :, test_idx);

% 时间序列
train_time = time_all(train_idx);
test_time = time_all(test_idx);

fprintf('训练集时间范围: %s 至 %s (%d天)\n', ...
    datestr(train_time(1)), datestr(train_time(end)), length(train_time));
fprintf('测试集时间范围: %s 至 %s (%d天)\n', ...
    datestr(test_time(1)), datestr(test_time(end)), length(test_time));

end
