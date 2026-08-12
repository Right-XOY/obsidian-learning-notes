# ==一、协方差：==

协方差反映了变量之间的线性关系强度和方向
$$
\mathrm{Cov}(X,Y)=\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)(y_i-\bar y)
$$
其中：
- $\mathrm{Cov}>0$：X 增大，Y 倾向增大，**正相关**
- $\mathrm{Cov}<0$：X 增大，Y 倾向减小，**负相关**
- $\mathrm{Cov}=0$：两个变量**线性无关**

设数据集共有 p 个特征，数据的协方差矩阵为 $p×p$ 的对称方阵：
$$
\boldsymbol{\Sigma}= \begin{bmatrix} \mathrm{Cov}(X_1,X_1) & \mathrm{Cov}(X_1,X_2) & \dots & \mathrm{Cov}(X_1,X_p)\\ \mathrm{Cov}(X_2,X_1) & \mathrm{Cov}(X_2,X_2) & \dots & \mathrm{Cov}(X_2,X_p)\\ \vdots & \vdots & \ddots & \vdots\\ \mathrm{Cov}(X_p,X_1) & \mathrm{Cov}(X_p,X_2) & \dots & \mathrm{Cov}(X_p,X_p) \end{bmatrix}
$$
但是协方差的值受量纲影响，大小不能直接代表强度，怎么办？
***
# ==二、Z‑score 标准化：==
 $$
 x^*=\frac{x-\bar x}{\sigma}
 $$
 $\bar x$ 为特征均值，$\sigma$ 为特征标准差。标准化之后每个特征满足：均值为 0，标准差为 1。

标准化之后再计算变量间协方差，此时的协方差等价于**皮尔逊相关系数**

相关系数取值范围 \([-1,1]\)：

- $\rho=1$：完全正线性相关
- $\rho=-1$：完全负线性相关
- $\rho=0$：无线性相关
***
# ==三、特征分解：==

对协方差矩阵做特征分解：
设协方差矩阵为 $A$
接下来对矩阵做正交对角化，他的充要条件是矩阵为实对称矩阵
$$\boldsymbol A=\boldsymbol V\boldsymbol{\Lambda}\boldsymbol V^\mathrm{T}$$
得到：
$$
\boldsymbol{\Lambda}= \begin{bmatrix} \lambda_1 & & \\ & \lambda_2 & \\ & & \ddots \\ & & & \lambda_p \end{bmatrix}
$$
$\boldsymbol A\boldsymbol v_i=\lambda_i \boldsymbol v_i$，把全部特征向量按列拼接，就得到矩阵形式 $\boldsymbol A\boldsymbol V=\boldsymbol V\boldsymbol{\Lambda}$
***
# ==三、构造投影矩阵：==

取前 n 个最大特征值对应的特征向量，构成投影矩阵 $\boldsymbol W\in\mathbb R^{p\times n}$：
$$\boldsymbol W=\big[\boldsymbol v_1,\ \boldsymbol v_2,\ \dots,\boldsymbol v_n\big]$$
设原始数据集有 $N$ 个样本，$p$ 个特征，假设标准化之后的数据矩阵是 $\boldsymbol X^*$

我们的目标是把 $p$ 个特征降维成 $n$ 个特征，降维后的数据矩阵变成 $N$ 行 $n$ 列：
$$\boldsymbol X_\text{pca}= \boldsymbol X^* \boldsymbol W$$
***
# ==四、方差贡献率：==

$\mathrm{tr}(\boldsymbol{\Lambda})$ 为矩阵的迹，等于全部特征值之和：$\mathrm{tr}(\boldsymbol{\Lambda})=\sum_{i=1}^p\lambda_i$。

第 $k$ 个主成分方差贡献率：
$$
\mathrm{Contrib}_k=\frac{\lambda_k}{\mathrm{tr}(\boldsymbol{\Lambda})}
$$
前 $n$ 个主成分累计方差贡献率：
$$
\mathrm{CumContrib}_n=\frac{\sum_{k=1}^n\lambda_k}{\mathrm{tr}(\boldsymbol{\Lambda})}
$$