function RBFModel = RBFTrain(S, Y)
%RBFTRAIN 训练RBF模型。
% 使用高斯核函数，sigma^2=max(||xj-xi||)(nm)^(-1/n)。
% 输入：
% S：训练数据，m×n矩阵，代表m个数据，每个数据有n个变量
% Y：训练数据目标值，m×1向量，代表m个数据的目标值
% 输出：
% RBFModel：RBF模型结构体

m = size(S, 1);
n = size(S, 2);

% 高斯核函数（可输入矩阵，每行为一个向量）
gaussianKernel = @(x1, x2, sigma2)exp(-(sum((x1 - x2) .^ 2, 2) .^ (1 / 2)) / (2 * sigma2));

% sigma对拟合结果影响较大，采用一种自适应策略
sigma2 = max(pdist(S)) * (n * m)^(-1 / n);

% 求解权重
phi = zeros(m, m);
for i = 1 : m
    phi(:, i) = gaussianKernel(S, repmat(S(i, :), m, 1), sigma2);
end
w = inv(phi' * phi) * phi' * Y;

% 保存模型
RBFModel.m = m;
RBFModel.n = n;
RBFModel.S = S;
RBFModel.kernel = gaussianKernel;
RBFModel.sigma2 = sigma2;
RBFModel.w = w;

end

