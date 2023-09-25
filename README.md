# SurrogateModel

一些常用代理模型的简单实现。

## 1. Kriging 模型

#### 原始形式

Kriging 模型的原始形式为：

$$
\begin{cases}
\hat{y}(x) = f^T(x)\beta^* + r^T(x)\gamma^* \\
s^2(x) = \sigma^2(1 + u^T(F^TR^{-1}F)^{-1}u - r^T(x)R^{-1}r(x))
\end{cases}
$$

其中

$$
\begin{cases}
\beta^* = (F^TR^{-1}F)^{-1}F^TR^{-1}Y \\
R \gamma^* = Y - F\beta^* \\
u = F^TR^{-1}r - f(x) \\
\sigma^2 = \frac{1}{m}(Y - F\beta^*)^TR^{-1}(Y - F\beta^*)
\end{cases}
$$

各矩阵表示如下：

- 数据输入 $S = \begin{pmatrix} s_{11} & s_{12} & \cdots & s_{1n} \\ s_{21} & s_{22} & \cdots & s_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ s_{m1} & s_{m2} & \cdots & s_{mn} \end{pmatrix}$
- 数据输出 $Y = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{pmatrix}$
- 数据点之间的协方差矩阵 $R = \begin{bmatrix} R(\theta, s_1, s_1) & R(\theta, s_1, s_2) & \cdots & R(\theta, s_1, s_m) \\ R(\theta, s_2, s_1) & R(\theta, s_2, s_2) & \cdots & R(\theta, s_2, s_m) \\ \vdots & \vdots & \ddots & \vdots \\ R(\theta, s_m, s_1) & R(\theta, s_m, s_2) & \cdots & R(\theta, s_m, s_m) \end{bmatrix}$
- 数据点的基函数矩阵 $F = \begin{bmatrix} f_1(s_1) & f_2(s_1) & \cdots & f_p(s_1) \\ f_1(s_2) & f_2(s_2) & \cdots & f_p(s_2) \\ \vdots & \vdots & \ddots & \vdots \\ f_1(s_m) & f_2(s_m) & \cdots & f_p(s_m) \end{bmatrix}$
- 未知点与已知点的协方差矩阵 $r(x) = \begin{bmatrix} R(\theta, s_1, x) \\ R(\theta, s_2, x) \\ \vdots \\ R(\theta, s_m, x) \end{bmatrix}$

### 常用形式

常用形式是原始形式在基函数为 0 次函数即 $f(x) = 1$ 的特例，大多使用这种作为代理模型。

$$
\begin{cases}
\hat{y}(x) = \hat{\mu} + r^TR^{-1}(Y - \mathbf{1}\hat{\mu}) \\
s^2(x) = \hat{\sigma}^2[1 - r^TR^{-1}r + \frac{(1 - \mathbf{1}^TR^{-1}r)^2}{\mathbf{1}^TR^{-1}\mathbf{1}}]
\end{cases}
$$

其中

$$
\begin{cases}
\hat{\mu} = \frac{\mathbf{1}^TR^{-1}Y}{\mathbf{1}^TR^{-1}\mathbf{1}} \\
\hat{\sigma}^2 = \frac{(Y - \mathbf{1}\hat{\mu})^TR^{-1}(Y - \mathbf{1}\hat{\mu})}{m}
\end{cases}
$$

通常使用 $R(\theta, s_i, s_j) =  \exp\left( -\sum_{k=1}^{n}\theta_k |x_{i,k} - x_{j,k}|^2 \right)$，超参数 $\theta$ 由下式给出：

$$
\max_\theta \left( -\frac{m\ln\hat{\sigma}^2 + \ln|R|}{2} \right)
$$

## 2. RBF 模型

对于一个训练集 $D = \{ (\mathbf{x}_i, y_i) | i = 1, \ldots, N \}$，它近似为下述连续函数：

$$
\hat{f}_\text{rbf}(\mathbf{x}) = \sum_{i=1}^{N}w_i \psi(\mathbf{x}_i, \mathbf{x})
$$

其中 $w_i$ 是权重，$\psi(\cdot, \cdot)$ 是基函数。权重向量由下式计算：

$$
\mathbf{w} = (\Psi^T\Psi)^{-1}\Psi^T\mathbf{y}
$$

其中

$$
\Psi = \begin{bmatrix}
\psi(\mathbf{x}_1, \mathbf{x}_1) & \cdots & \psi(\mathbf{x}_1, \mathbf{x}_N) \\
\vdots & \ddots & \vdots \\
\psi(\mathbf{x}_N, \mathbf{x}_1) & \cdots & \psi(\mathbf{x}_N, \mathbf{x}_n)
\end{bmatrix}
$$

常用的基函数有高斯核函数 $\psi(||x - x_i||) = e^{-\frac{||x - x_i||^2}{2\sigma^2}}$ 等。