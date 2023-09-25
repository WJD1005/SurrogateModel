function [PV, MSE] = krigingPredict(krigingModel, X)
%KRIGINGPREDICT 使用克里金模型输出预测。
% 输入：
% krigingModel：克里金模型结构体，
% X：未知点，m×n矩阵，代表m个需要预测的点，每个点有n个变量
% 输出：
% PV：预测值，m×1向量
% MSE：均方误差，m×1向量

m = size(X, 1);
n = size(X, 2);

PV = zeros(m, 1);
MSE = zeros(m, 1);

if n == krigingModel.n
    for i = 1 : m
        rx = krigingModel.r(krigingModel.theta, krigingModel.S, repmat(X(i, :), krigingModel.m, 1));
        PV(i) = krigingModel.mu + rx' * inv(krigingModel.R) * (krigingModel.Y - krigingModel.I * krigingModel.mu);
        MSE(i) = krigingModel.sigma2 * (1 - rx' * inv(krigingModel.R) * rx + (1 - krigingModel.I' * inv(krigingModel.R) * rx)^2 / (krigingModel.I' * inv(krigingModel.R) * krigingModel.I));
    end
else
    PV = NaN;
    MSE = NaN;
end

end

