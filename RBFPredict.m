function PV = RBFPredict(RBFModel, X)
%RBFPREDICT 使用RBF模型输出预测。
% 输入：
% RBFModel：RBF模型结构体，
% X：未知点，m×n矩阵，代表m个需要预测的点，每个点有n个变量
% 输出：
% PV：预测值，m×1向量

m = size(X, 1);
n = size(X, 2);

PV = zeros(m, 1);

if n == RBFModel.n
    for i = 1 : m
        PV(i) = sum(RBFModel.w .* RBFModel.kernel(RBFModel.S, repmat(X(i, :), RBFModel.m, 1), RBFModel.sigma2));
    end
else
    PV = NaN;
end

end

