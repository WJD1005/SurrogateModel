function [krigingModel] = krigingTrain(S, Y)
%KRIGINGTRIAN 训练克里金模型。
% 该模型为常用形式，基函数为0次函数即f(x)=1，无运算优化。
% 输入：
% S：训练数据，m×n矩阵，代表m个数据，每个数据有n个变量
% Y：训练数据目标值，m×1向量，代表m个数据的目标值
% 输出：
% krigingModel：克里金模型结构体

m = size(S, 1);
n = size(S, 2);

% 相关函数（可输入矩阵，每行为一个向量）
r = @(theta, x1, x2) exp(-sum(theta .* abs(x1 - x2) .^ 2, 2));

% 最大似然估计
func = @(theta)-thetaMLF(theta, S, Y, r);  % 最大似然函数，只让theta作为参数，并取负变为最小化问题
theta = fmincon(func, 0.5 * ones(1, n), [], [], [], [], 0.00001 * ones(1, n), 200 * ones(1, n));  % 最小化

% 相关矩阵
R = zeros(m, m);
for i = 1 : m
    R(:, i) = r(theta, S, repmat(S(i, :), m, 1));  % 单列相关矩阵
end

% 模型变量
I = ones(m, 1);
mu = (I' * inv(R) * Y) / (I' * inv(R) * I);
sigma2 = (Y - I * mu)' * inv(R) * (Y - I * mu) / m;

% 保存模型
krigingModel.S = S;
krigingModel.Y = Y;
krigingModel.n = n;
krigingModel.m = m;
krigingModel.r = r;
krigingModel.theta = theta;
krigingModel.R = R;
krigingModel.I = I;
krigingModel.mu = mu;
krigingModel.sigma2 = sigma2;

end


%% theta的最大似然函数
function fval = thetaMLF(theta, S, Y, r)
m = size(S, 1);
% 相关矩阵
R = zeros(m, m);
for i = 1 : m
    R(:, i) = r(theta, S, repmat(S(i, :), m, 1));  % 单列相关矩阵
end
% 中间变量
I = ones(m, 1);
mu = (I' * inv(R) * Y) / (I' * inv(R) * I);
sigma2 = (Y - I * mu)' * inv(R) * (Y - I * mu) / m;
% 最大似然函数
fval = -(m * log(sigma2) + log(det(R))) / 2;
end
