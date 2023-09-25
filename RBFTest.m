clear
clc

% 二维正态分布函数
ux = 0;
uy = 0;
dx = 4;
dy = 4;
r = 0;
func = @(x, y)(1 / (2 * pi * dx * dy * sqrt(1 - r^2))) * exp((-1 / (2 * (1 - r^2))) * ((x - ux) .^ 2 / dx^2) - (2 * r * (x - ux) .* (y - uy) / (dx * dy) + (y - uy) .^2 / dy^2));

% 二维正态分布图
figure(1);
ub = 10;
step = 0.1;
x = -ub : step : ub;
y = -ub : step : ub;
[xx, yy] = meshgrid(x, y);
zz = func(xx, yy);
surf(xx, yy, zz);
title('真实函数');
hold on;

% 随机取样
num = 100;  % 样本数
sx = (rand(num, 1) - 0.5) .* 2 .* ub;
sy = (rand(num, 1) - 0.5) .* 2 .* ub;
sz = func(sx, sy);
scatter3(sx, sy, sz, 'filled', 'r');  % 标注样本点

% 训练RBF模型
RBFModel = RBFTrain([sx, sy], sz);

% 使用克里金模型的输出画图
figure(2);
X = [reshape(xx, [], 1), reshape(yy, [], 1)];
zz = RBFPredict(RBFModel, X);
zz = reshape(zz, size(xx, 1), size(xx, 2));
surf(xx, yy, zz);
title('RBF模型拟合函数');
