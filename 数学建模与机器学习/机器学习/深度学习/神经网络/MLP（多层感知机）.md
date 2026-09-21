# ==一、从感知机讲起==

多层感知机（MLP）是神经网络的基础，要理解它，先从最原始的**感知机（Perceptron）**开始。

## （1）感知机的结构

感知机只有一个神经元，输入若干特征，输出一个结果：

$$
\hat y = f\left(\sum_{i=1}^n w_i x_i + b\right)
$$

其中：
- $\boldsymbol{x}=(x_1,x_2,\dots,x_n)$：输入特征向量，共 $n$ 维
- $\boldsymbol{w}=(w_1,w_2,\dots,w_n)$：权重向量，每个输入对应一个权重
- $b$：偏置（bias），相当于给决策加一个平移量
- $f(\cdot)$：激活函数，感知机里常用**阶跃函数**（大于 0 输出 1，否则输出 0）
- $\hat y$：模型输出

***
## （2）决策边界

感知机的核心是一个**线性判别**：
$$
\sum_{i=1}^n w_i x_i + b = 0
$$
这个方程在特征空间里是一个**超平面**，把空间一分为二。

所以感知机只能解决**线性可分**问题，面对非线性问题无能为力。

***
## （3）XOR 异或问题

最经典的例子是 XOR 异或：

| $x_1$ | $x_2$ | XOR 输出 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

这四个点在平面上没法用一条直线把“输出 1”和“输出 0”的点分开，所以**单层感知机无法表达 XOR**。

解决办法：**多加一层（隐藏层）**，这就是多层感知机。

***
# ==二、多层感知机（MLP）结构==

多层感知机 = 在输入层和输出层之间**插入若干隐藏层**。

- **输入层**：接收原始特征，节点数 = 特征维数，不参与计算
- **隐藏层**：一层或多层，负责提取特征的非线性组合
- **输出层**：输出最终预测值，节点数由任务决定（回归 1 个、分类 $k$ 个）

相邻两层之间**全连接**（fully connected）：前一层的每个神经元都与后一层的每个神经元相连。

## （1）为什么叫“多层”

- 只有输入层 + 输出层（无隐藏层）→ 就是感知机，本质是线性模型
- 有 1 层隐藏层 → 已具备表达任意连续函数的能力（见通用近似定理）
- 隐藏层越多、越宽 → 表达能力越强，但更容易过拟合

***
# ==三、前向传播（Forward Propagation）==

前向传播就是数据从输入层逐层流向输出层的过程。

每一层只做两件事：**先线性变换，再过激活函数**。

## （1）单个神经元

$$
z = \sum_{i=1}^n w_i x_i + b = \boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b
$$
$$
a = f(z)
$$
其中：
- $z$：线性变换的结果，称为**净输入 / 加权和**
- $a$：经过激活函数后的输出，称为**激活值**
- $f(\cdot)$：激活函数

***
## （2）矩阵形式（第 $l$ 层）

写成矩阵，方便批量计算：

$$
\boldsymbol{z}^{(l)} = \boldsymbol{W}^{(l)}\boldsymbol{a}^{(l-1)} + \boldsymbol{b}^{(l)}
$$
$$
\boldsymbol{a}^{(l)} = f\big(\boldsymbol{z}^{(l)}\big)
$$
其中：
- $\boldsymbol{a}^{(l-1)}$：第 $l-1$ 层的输出（第 $l$ 层的输入）
- $\boldsymbol{W}^{(l)}$：第 $l$ 层的权重矩阵，形状为 `[本层节点数 × 上一层节点数]`
- $\boldsymbol{b}^{(l)}$：第 $l$ 层的偏置向量
- $\boldsymbol{z}^{(l)}$：第 $l$ 层的净输入
- $\boldsymbol{a}^{(l)}$：第 $l$ 层的激活输出

约定 $\boldsymbol{a}^{(0)}=\boldsymbol{x}$（第 0 层就是输入特征），$\boldsymbol{a}^{(L)}=\hat{\boldsymbol y}$（最后一层就是最终输出）。

***
## （3）完整前向传播（共 $L$ 层）

$$
\begin{aligned}
\boldsymbol{a}^{(0)} &= \boldsymbol{x}\\
\boldsymbol{z}^{(1)} &= \boldsymbol{W}^{(1)}\boldsymbol{a}^{(0)} + \boldsymbol{b}^{(1)}, & \boldsymbol{a}^{(1)} &= f(\boldsymbol{z}^{(1)})\\
\boldsymbol{z}^{(2)} &= \boldsymbol{W}^{(2)}\boldsymbol{a}^{(1)} + \boldsymbol{b}^{(2)}, & \boldsymbol{a}^{(2)} &= f(\boldsymbol{z}^{(2)})\\
&\vdots\\
\boldsymbol{z}^{(L)} &= \boldsymbol{W}^{(L)}\boldsymbol{a}^{(L-1)} + \boldsymbol{b}^{(L)}, & \hat{\boldsymbol y} &= \boldsymbol{a}^{(L)}
\end{aligned}
$$

注意：输出层有时**不加激活函数**（回归任务），或加 softmax（多分类任务）。

***
# ==四、激活函数的作用==

## （1）为什么必须是非线性？

如果每一层只做线性变换（$f(z)=z$，即恒等映射），那么无论叠多少层，整体仍然是一个线性变换：

$$
\boldsymbol{a}^{(L)} = \boldsymbol{W}^{(L)}\cdots\boldsymbol{W}^{(2)}\boldsymbol{W}^{(1)}\boldsymbol{x}
$$

多个线性矩阵相乘，最终可以合并成**一个**矩阵，等价于**单层线性模型**，多层就失去意义。

所以激活函数必须是非线性的，才能让 MLP 逼近任意非线性函数。

## （2）常见激活函数

- **Sigmoid**：$\sigma(z)=\dfrac{1}{1+e^{-z}}$，输出 $(0,1)$
- **Tanh**：$\tanh(z)$，输出 $(-1,1)$
- **ReLU**：$\mathrm{ReLU}(z)=\max(0,z)$，输出 $[0,\infty)$
- **Leaky ReLU**：$\max(\alpha z, z)$，解决 ReLU 的“神经元死亡”问题

> 详见 [[激活函数]]

***
# ==五、损失函数==

损失函数衡量模型预测 $\hat{\boldsymbol y}$ 与真实标签 $\boldsymbol y$ 的差距，是优化的目标。

> 详见 [[损失函数]]

## （1）回归任务：均方误差 MSE

$$
L = \frac{1}{2}\big(y - \hat y\big)^2
$$
（系数 $\tfrac12$ 只是为了求导时消掉平方的 2，方便书写）

## （2）分类任务：交叉熵 CE

$$
L = -\sum_{k=1}^K y_k \log \hat y_k
$$
其中：
- $K$：类别总数
- $y_k$：真实标签的 one-hot 编码（只有正确类别为 1，其余为 0）
- $\hat y_k$：模型输出的第 $k$ 类概率（通常由 softmax 得到）

***
# ==六、反向传播（Back Propagation，BP）==

反向传播是 MLP 训练的核心算法，用**链式法则**从输出层往回逐层计算梯度。

> 详细推导见 [[前向与反向传播]]

## （1）目标

我们要计算损失 $L$ 对每一层参数的梯度：
$$
\frac{\partial L}{\partial \boldsymbol{W}^{(l)}},\qquad \frac{\partial L}{\partial \boldsymbol{b}^{(l)}}
$$
拿到梯度之后，就可以用梯度下降更新参数。

***
## （2）定义误差项 $\delta$

先定义第 $l$ 层的**误差项**（对净输入 $\boldsymbol{z}^{(l)}$ 的梯度）：
$$
\boldsymbol{\delta}^{(l)} = \frac{\partial L}{\partial \boldsymbol{z}^{(l)}}
$$
这个量是反向传播的关键：算出每层的 $\boldsymbol{\delta}^{(l)}$，就能推出参数的梯度。

***
## （3）输出层的 $\delta$

设输出层净输入为 $\boldsymbol{z}^{(L)}$，输出 $\hat{\boldsymbol y}=\boldsymbol{a}^{(L)}=f(\boldsymbol{z}^{(L)})$。

由链式法则：
$$
\boldsymbol{\delta}^{(L)} = \frac{\partial L}{\partial \boldsymbol{z}^{(L)}} = \frac{\partial L}{\partial \boldsymbol{a}^{(L)}}\odot f'\big(\boldsymbol{z}^{(L)}\big)
$$
其中 $\odot$ 表示逐元素相乘（Hadamard 积）。

两个常用结论：
- **回归 + MSE + 恒等输出**：$\boldsymbol{\delta}^{(L)}=\hat{\boldsymbol y}-\boldsymbol y$
- **分类 + 交叉熵 + softmax**：$\boldsymbol{\delta}^{(L)}=\hat{\boldsymbol y}-\boldsymbol y$

> 神奇的是，两种任务下输出层误差项形式一致，都是“预测值减真实值”。

***
## （4）隐藏层的 $\delta$（误差反向递推）

由第 $l+1$ 层递推第 $l$ 层：

$$
\boldsymbol{\delta}^{(l)} = \left(\boldsymbol{W}^{(l+1)}\right)^\mathrm{T}\boldsymbol{\delta}^{(l+1)}\ \odot\ f'\big(\boldsymbol{z}^{(l)}\big)
$$
其中：
- $(\boldsymbol{W}^{(l+1)})^\mathrm{T}$：第 $l+1$ 层权重的转置，把误差“反传”回上一层
- $f'(\boldsymbol{z}^{(l)})$：第 $l$ 层激活函数的导数
- 含义：本层误差 = 后一层误差经权重反传后，再乘本层激活函数的导数

从输出层开始，一层层套这个公式，就能算出所有层的 $\boldsymbol{\delta}^{(l)}$。

***
## （5）参数梯度

拿到 $\boldsymbol{\delta}^{(l)}$ 后，参数梯度直接得出：

$$
\frac{\partial L}{\partial \boldsymbol{W}^{(l)}} = \boldsymbol{\delta}^{(l)} \left(\boldsymbol{a}^{(l-1)}\right)^\mathrm{T}
$$
$$
\frac{\partial L}{\partial \boldsymbol{b}^{(l)}} = \boldsymbol{\delta}^{(l)}
$$
其中：
- $\boldsymbol{a}^{(l-1)}$：第 $l-1$ 层的输出（本层输入）
- 直观理解：参数的梯度 = 本层误差 × 本层的输入

***
# ==七、参数更新（梯度下降）==

拿到所有梯度后，沿负梯度方向更新参数：

> 详见 [[梯度下降]]

$$
\boldsymbol{W}^{(l)} \leftarrow \boldsymbol{W}^{(l)} - \eta\,\frac{\partial L}{\partial \boldsymbol{W}^{(l)}}
$$
$$
\boldsymbol{b}^{(l)} \leftarrow \boldsymbol{b}^{(l)} - \eta\,\frac{\partial L}{\partial \boldsymbol{b}^{(l)}}
$$
其中：
- $\eta$：学习率（learning rate），控制每步更新的幅度
- 学习率太大 → 震荡甚至发散；太小 → 收敛慢

实际常用 SGD、Adam 等优化器（在梯度下降基础上改进）。

***
# ==八、通用近似定理==

**通用近似定理（Universal Approximation Theorem）**：

> 只要隐藏层神经元足够多，含**一个隐藏层**的前馈网络就能以任意精度逼近任意连续函数。

这解释了为什么 MLP 理论上能拟合任意复杂的映射。但定理只保证“存在”，不保证“训练能学到”，实际仍受数据、优化、过拟合等因素制约。

***
# ==九、MLP 的优缺点==

**优点：**
- 结构简单，是各种复杂网络（CNN、RNN、Transformer）的基础
- 通用近似能力强，可拟合非线性关系
- 对表格型（结构化）数据表现稳定

**缺点：**
- 全连接导致**参数量大**，输入维度高时容易爆参
- 无法利用图像、序列等数据的**空间/时序结构**（催生了 CNN、RNN）
- 深层训练容易**梯度消失 / 爆炸**
- 容易过拟合，需要正则化、Dropout 等辅助手段
