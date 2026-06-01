function data_denorm = denormalize(data_norm, data_mean, data_std)
%% 反标准化
% 将标准化后的数据转换回原始尺度

% 处理不同维度的输入
if isstruct(data_norm)
    % 如果是结构体，对每个字段进行反标准化
    data_denorm = struct();
    fields = fieldnames(data_norm);
    for i = 1:length(fields)
        field = fields{i};
        data_denorm.(field) = data_norm.(field) * data_std.(field) + data_mean.(field);
    end
else
    % 如果是数组
    data_denorm = data_norm * data_std + data_mean;
end

end
