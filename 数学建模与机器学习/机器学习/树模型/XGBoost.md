数据集：

| ID  | $x_1$ | $x_2$ | ... | $x_n$ |
| --- | ----- | ----- | --- | ----- |
| 1   | #     | #     | #   | #     |
| ... | #     | #     | #   | #     |
| m   | #     | #     | #   | #     |
***
# ==一、开场：CART 回归树==

## （1）回归树是怎么给样本预测的？

核心映射函数：
$$
I_j=\{i\mid q(x_i)=j\}
$$
其中：
- $i$：样本的编号（样本 id） 
- $j$：叶子节点的编号，$j=1,2,3,4$ 
- $q(\boldsymbol x)$：树的映射函数，输入样本特征 $\boldsymbol x_i$，输出这个样本落到几号叶子
***
- $I_j$ 是对应叶子节点样本的集合（叶子节点 $j$ ）
- $x_i$ 是编号为 $i$ 的这个样本的特征向量：$x_i = (x_1, x_2, ... x_n)$
- $q(x)$  映射函数是决策树的 if else 决策分支，没有显式表达式（分段函数？）

def q_x: 
	if x[0] < 10: 
		if x[1] < 5: 
			return 1
		else: 
			return 2
		else: if x[1] < 8: 
			return 3
		else: 
			return 4
***
## （2）预测的函数映射是什么？

假设树是一个二叉树，最终有4个叶子节点：

- $w_1,w_2,w_3,w_4$：叶子节点值（叶子权重、预测值）
- $I_1,I_2,I_3,I_4$：每个叶子对应的样本集合

预测函数映射：
$$ T(\theta;x_i)=w_{q(x_i)} $$
其中：
- $q(x_i)$ 输出的是第 $i$ 个样本所在的叶子节点
- 那么 $w_{q(x_i)}$ 就是第 $i$ 个样本所在叶子节点的预测值

注意：
- $T$：函数名，专门表示这一棵CART回归树；
- $\theta$：**表示整棵树全部参数集合** $$ 
\theta=\big\{\, q,\; \{w_1,w_2,\dots,w_T\} \,\big\} $$
这里的权重下标 $T$ 是指节点数 4
- $\{w_j\}$：各个叶子节点的输出（单棵树的预测输出）
- $w_{q(x_i)}$：样本 $x_i$ 通过 $q$ 得到叶子编号 $j$，取出该叶子的权重 $w_j$，作为这棵树的输出。
# （3）什么是叶子节点的权重？

我们知道：
$w_{q(x_i)}$ 就是第 $i$ 个样本所在叶子节点的预测值，在 XGBoost 语境下面也被称为节点权重

因此：
单棵树 $f_t(x_i)=w_{q(x_i)}$ 等式成立，$t$ 表示 CART 树的编号（后面假定总共只有 $t$ 棵 CART 树）

***
这时候会发现，我们只知道有多少叶子节点，并不知道中间过程是怎么样的。
XGBoost就能解决这个问题
***
# ==二、XGBoost 模型推导与目标函数求解

# （1） 加法模型
$$
\hat y_i^{(t)}=\sum_{j=1}^{t} f_j(x_i)
$$
其中：
- $\hat y_i^{(t)}$：训练完 **$t$ 棵树**后，对第 $i$ 个样本的**最终预测值**
- $f_j(x_i)$：第 $j$ 棵回归树的输出
- $f_j(x_i) = T(\theta;x_i) = w_{q(x_i)}$
- $\displaystyle\sum_{j=1}^{t} f_j(x_i)$：将所有树的增量输出累加得到完整预测
***
# （2）前向分步算法

前向分步算法是**求解加法模型的迭代方法**。
为了简化优化，不一次性训练所有树，而是**逐棵训练、逐棵累加、旧树冻结**。
$$
\hat y_i^{(t)}=\hat y_i^{(t-1)} + f_t(x_i)
$$
其中：
- $\hat y_i^{(t)}$：训练完第 $t$ 棵树后的预测值
- $\hat y_i^{(t-1)}$：**前 $t-1$ 棵树的预测总和（本轮训练为常数、冻结不变）**
- $f_t(x_i)$：**当前正在训练的第 $t$ 棵新树（唯一待优化量）**

迭代完成后两者完全等价：
$$
\hat y_i^{(T)} = \sum_{j=1}^T f_j(x_i)
$$
---
# （3）目标函数
$$ \mathrm{obj}^{(t)}=\sum_{i=1}^m l\big(y_i,\hat y_i^{(t)}\big)+\sum_{j=1}^t \Omega(f_j) $$ 其中：
- $\mathrm{obj}^{(t)}$：训练完第 $t$ 棵树后的整体目标函数值
- $l(y_i,\hat y_i^{(t)})$：样本 $i$ 的损失函数，衡量标签 $y_i$ 与当前预测 $\hat y_i^{(t)}$ 的误差
- $\displaystyle\sum_{i=1}^m l\big(y_i,\hat y_i^{(t)}\big)$：全部样本的训练损失之和，$m$ 是样本总数
- $\Omega(f_j)$：第 $j$ 棵树的正则项，用来控制树的复杂度，防止过拟合 
- $\displaystyle\sum_{j=1}^t \Omega(f_j)$：前 $t$ 棵树全部正则项累加

解释：
	这里的损失函数 $l(y_i,\hat y_i)=\frac12(y_i-\hat y_i)^2$ 也就是 MSE 均方误差（可以不除2）

# （4）正则项

 上面的正则项为 $\boldsymbol{\Omega(f)=\gamma T+\frac12\lambda\sum_{j=1}^T w_j^2}$
其中：
	==超参数== $\gamma$：叶子数量惩罚系数。每多 1 个叶子，就加一份 $\gamma$；$\gamma$ 越大，越不希望树长出很多叶子。
	==超参数== $\lambda$：L2 正则系数，对叶子权重做 L2 平方惩罚；$\lambda$ 越大，会把叶子权重压得更小，避免个别叶子输出过大。

可以发现损失函数越小，模型拟合的效果越强——
但是注意到目标函数除了损失函数，还存在 L2正则化，原因就是一味地优化函数值可能导致过拟合，这里的正则函数存在的意义就是对模型的 “复杂度” 进行削减，参考奥卡姆剃刀原则
***
# （5）正则项怎么定义出来的？
XGBoost 的正则 $Omega(f)=\gamma T+\frac12\lambda\sum_{j=1}^T w_j^2$ 
是人为设计，用来刻画单棵树 f 的**模型复杂度**，没有严格的理论推导，是工程上的经验设计。
***
# （6）二阶泰勒展开

我们知道目标函数是：$$ \mathrm{obj}^{(t)}=\sum_{i=1}^m l\big(y_i,\hat y_i^{(t)}\big)+\sum_{j=1}^t \Omega(f_j) $$
也可以分解成：
$$ \mathrm{obj}^{(t)}=\sum_{i=1}^m l\big(y_i,\hat y_i^{(t)}\big)+\sum_{j=1}^{t-1} \Omega(f_j) + \Omega(f_t) $$
假设第 $1$ 棵树到第 $t-1$  棵树全都训练完毕
则 $\sum_{j=1}^{t-1} \Omega(f_j)$ 就相当于一个常量（因为前 $t-1$ 棵树各自叶子节点的个数 $T$ 都已经知道了，对应的节点的权重也都已经计算好了，而 $\boldsymbol{\Omega(f)=\gamma T+\frac12\lambda\sum_{k=1}^T w_k^2}$）

所以在模型优化过程中，常量 $\sum_{j=1}^{t-1} \Omega(f_j)$ 没有用

因此优化函数 $\mathrm{obj}^{(t)}$ 只和第 $t$ 个回归树的叶子节点个数 $T$ 以及他对应的权重 $w_1,w_2, ...w_k$ 有关系

接下来我们把目标函数写成：
$$ \mathrm{obj}^{(t)}=\sum_{i=1}^m l\big(y_i,\hat y_i^{(t)}\big) + \sum_{j=1}^{t-1} \Omega(f_j) + \gamma T+\frac12\lambda\sum_{j=1}^T w_j^2 $$
接下来是重头戏，等式化为：
$$
\mathrm{obj}^{(t)}=\sum_{j=1}^T \sum_{i∈I_j}l\big(y_i,\hat y_i^{(t)}\big) + \gamma T+\frac12\lambda\sum_{j=1}^T w_j^2 + \sum_{j=1}^{t-1} \Omega(f_j)
$$
然后合并同类项：
$$
\mathrm{obj}^{(t)}=\sum_{j=1}^T(\frac12\lambda w_j^2 + \sum_{i∈I_j}l\big(y_i,\hat y_i^{(t)}\big)) + \sum_{j=1}^{t-1} \Omega(f_j) + \gamma T
$$
同时不难发现：
$$
\hat y_i^{(t)} = \hat y_i^{(t-1)} + w_{q(x_i)}
$$
又因为：
$$
w_{q(x_i)} = w_j
$$
所以：
$$
\mathrm{obj}^{(t)}=\sum_{j=1}^T(\frac12\lambda w_j^2 + \sum_{i∈I_j}l\big(y_i,\hat y_i^{(t-1)} + w_j\big)) + \sum_{j=1}^{t-1} \Omega(f_j) + \gamma T
$$
不难看出，括号内代表的含义是指：$\boldsymbol{\frac12\lambda}$ 乘以叶子权重的平方加上该节点下所有样本的损失函数的值，我们把这个值成为“局部目标子项”：
$$\mathcal L_j=\frac12\lambda w_j^2 + \sum_{i\in I_j}l\big(y_i,\hat y_i^{(t-1)} + w_j\big)$$
连加号形式就代表把这个决策树的全部叶子节点的“局部目标子项”求和：
$$\displaystyle\sum_{j=1}^T \mathcal L_j$$
所以该决策树的目标函数的值为这个求和项加上前面所有决策树的正则总和，再加上伽马大T

接下来开始==二阶泰勒展开==：
首先考虑第 $j$ 个叶子节点：

- $\hat y_i^{(t)} = \hat y_i^{(t-1)} + w_j$ 是一个关于第 $j$ 个未知的叶子节点权重 $w_j$ 的函数，$\hat y_i^{(t-1)}$ 已知
- 设 $\hat y_i^{(t)}$ 是未知量 $x$ ，设已知量 $\hat y_i^{(t-1)}$ 是 $x_0$
- 设损失函数为  $F(x) = l(y_i,\hat y_i^{(t-1)} + w_j) = l(y_i,\hat y_i^{(t-1)} + x)$

对 $F(x)$ 在 $x = x_0$ 处做二阶泰勒展开得到：
$$
F(x)\approx F(x_0)+F'(x_0)(x-x_0)+\frac12 F''(x_0)(x-x_0)^2
$$
代入上述变量得到：
$$
F(\hat y_i^{(t)})\approx F(\hat y_i^{(t-1)})+F'(\hat y_i^{(t-1)})w_j+\frac12 F''(x_0)w_j^2
$$
也就是：
$$
F(\hat y_i^{(t)})\approx F'(\hat y_i^{(t-1)})w_j+\frac12 F''(x_0)w_j^2 + C
$$
$$
l(y_i,\hat y_i^{(t)}) = l'(y_i,\hat y_i^{(t-1)})w_j + \frac12l''(y_i,\hat y_i^{(t-1)})w_j^2 + C
$$
这里的已知量我们设为常数 $C$

为了方便，我们把一阶梯度 $l'(y_i,\hat y_i^{(t-1)})$ 表示为 $g_i$
把二阶梯度 $l''(y_i,\hat y_i^{(t-1)})$ 表示为 $h_i$

代入目标函数：
$$
\mathrm{obj}^{(t)}=\sum_{j=1}^T(\frac12\lambda w_j^2 + \sum_{i∈I_j}(g_iw_j +\frac12 h_iw_j^2) + C) + \sum_{j=1}^{t-1} \Omega(f_j) + \gamma T
$$
又发现：
$$ \sum_{i\in I_j}\big(g_i w_j+\tfrac12 h_i w_j^2\big) = w_j\sum_{i\in I_j}g_i+\frac12 w_j^2\sum_{i\in I_j}h_i $$
==我们定义叶子聚合梯度：==
$$G_j=\sum_{i\in I_j}g_i,\qquad H_j=\sum_{i\in I_j}h_i$$
代入原式子：
$$
\mathrm{obj}^{(t)}=\sum_{j=1}^T\left[\frac12\lambda w_j^2 \;+\; w_jG_j+\frac12 w_j^2H_j +C \right] +\sum_{j=1}^{t-1}\Omega(f_j)+\gamma T
$$
化简后：
$$
{ \mathrm{obj}^{(t)} = \gamma T+\sum_{j=1}^T\left(G_j w_j+\frac12\big(H_j+\lambda\big)w_j^2\right) } + C_0
$$
(把常数项合并为  $C_0$ )
对每个叶子独立求导：
$$
\dfrac{\partial \mathrm{obj}^{(t)}}{\partial w_j}=G_j+(H_j+\lambda)w_j=0
$$
得到最优叶子权重：
$$
w_j^*=-\frac{G_j}{H_j+\lambda}
$$
当所有叶子权重为最优权重时，目标函数最小！
***
# （7）最终优化目标

最终我们确定的优化目标就是：
$$
\big(w_1^*, w_2^*,\dots,w_T^*\big)=\arg\min_{\{w_j\}} \mathrm{obj}^{(t)}
$$
等式左边表示最优的一组叶子节点权重参数
$arg\min$ 表示求令右边目标函数最小的这一组参数，也就是左边的参数值

最终结果：
$$
\begin{cases} \displaystyle w_j^* = -\frac{b}{2a}=-\frac{G_j}{H_j+\lambda}\\[12pt] \displaystyle \mathrm{obj}^{(t)*} = \min\; \mathrm{obj}^{(t)}=\gamma T-\frac12\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda} \end{cases}
$$
***
# ==三、精确贪心算法求解

经过上面推导，我们给定一棵树的分裂结构 $q$ 可以直接算出最优的叶子节点权重和最优的目标函数
但是由于树的分裂结构本身是离散组合优化问题，没有解析解，不可能穷举所有树结构

因此我们选择**贪心逐层分裂**来构造树结构
贪心算法的本质是每次决策只考虑当下的最优情况，而不考虑全局实际情况

因此算法从根节点开始，每次对当前节点做最好的一次分裂，递归向下生长

***
# （1）什么是分裂增益 Gain ？

我们现在手里某一个节点，有两种选择： 
- 不分裂，直接把这个节点当成叶子；
- 把这个节点切一刀，分裂成左右两个叶子。

- 方案①不分裂：算出该节点不分裂时，这一部分给整体目标函数贡献多大的值，记为$\mathrm{obj}_\text{before}$
- 方案②分裂：分裂成左右两个叶子之后，算出他们给整体目标函数贡献多大的值，记为$\mathrm{obj}_\text{after}$
$$
\mathrm{Gain}= \mathrm{obj}_\text{before}-\mathrm{obj}_\text{after}
$$
Gain 就是解释 “做这次分裂，我们能把目标函数降低多少”
Gain 越大，这一次分裂带来的收益越高。
***
由上面我们已知叶子节点的聚合梯度（g是一阶导，h是二阶导）是：
$$G_j=\sum_{i\in I_j}g_i,\qquad H_j=\sum_{i\in I_j}h_i$$
我们知道目标函数是：
$$\mathrm{obj}^{(t)}=\gamma T-\frac12\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda}$$
可以看出每增加一个叶子节点，目标函数的值一般情况下会再变小

这里的 $T$ 会+1，但是后面的连加也会多增加一个当前叶子节点的损失收益

这里就清楚了：
- 对于方案①，不分裂，但是叶子节点会导致目标函数变化
- 对于方案②，分裂，多出来的两个叶子节点也会导致目标函数变化

显然，经过计算可以知道：
$$\begin{aligned} \mathrm{Gain} &= \mathrm{obj}_\text{before}-\mathrm{obj}_\text{after}\\[6pt] &=\boldsymbol{\frac12}\left[ \frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{G^2}{H+\lambda} \right]-\gamma \end{aligned}$$

其中：
$$G=G_L+G_R,\qquad H=H_L+H_R$$
分别代表了分类的左节点和右节点的各自样本集合的聚合梯度

假如我们分裂之后的目标函数更小（gain是正数）则可以分裂

注意：末尾的 $-\gamma$ 是用来对分裂后叶子节点变多的情况（方案②）做惩罚的（奥卡姆剃刀原则）
***
# （2）精确贪心算法寻找最佳分裂点

**一个节点最多只执行一次真实分裂，只用一个特征、一个阈值**

该节点需要遍历全部特征，寻找每个特征的最优分裂点，最后全体比较得到最优的特征分裂点

对某一个特征 $f$ ：
把节点内全部样本按照特征的取值升序排序，得到有序样本序列

该特征的候选分割点取自相邻样本之间，每一次分割会产生两种样本

我们比较这个叶子节点在属于他的样本集中做了分割以后得到的两份样本集合
再算出这两份样本集合左右两边的的聚合梯度 $G$ 和 $H$ ：

代入计算每一次分割得到的 $gain$ ，得到最优的 $gain$ ，按照最优的情况进行分裂

假如有3个特征，每个特征在内部被平均分割了3次，那么就计算9次，得到当下最优的 $gain$

***
# （3）贪心算法的约束条件

我们知道在MSE的语境下，每一个样本的二阶梯度恒等于1

因为 $l(y_i,\hat y_i)=\frac12\big(y_i-\hat y_i\big)^2$ 求二阶导数肯定是1

节点聚合二阶梯度定义：
$$H_j=\sum_{i\in I_j} h_i$$
因此 $H_j=\sum_{i\in I_j} 1 = \text{集合}I_j\text{里面样本的总个数}$

约束条件就是：
$$\text{左样本数}+\lambda \ge \mathrm{min\_child\_weight},\quad \text{右样本数}+\lambda \ge \mathrm{min\_child\_weight}$$

这里的 $min\_child\_weight$ 是一个人为给定的超参数，意为“最小子节点权重和阈值”
而 $\lambda$ 也是L2 正则化的超参数

约束条件的作用是：保证分裂完成后，左、右两个子节点的有效权重和不能过小

因为损失函数是MSE语境，因此约束间接限制分裂之后左右两边的样本数不能太少（便于理解）
实际上我们用一般损失函数（如 logistic 对数损失）也可以
***
# （4）其他终止条件（预剪枝）

除 $\mathrm{max\_gain}\le0$ 停止分裂之外，XGBoost 还有硬性终止条件

$max\_depth$：树最大深度。节点到达设定最大深度停止分裂；限制树的整体深度，抑制过拟合

# ==四、单棵CART树完整训练闭环==

> 注意：**第1棵树 t=1**：$\hat y_i^{(0)} = 0$，零棵树的基准预测
1. 接收前 $t-1$ 棵树的预测结果 $\hat y_i^{(t-1)}$；
2. 对全部样本计算损失的一阶梯度 $g_i$、二阶梯度 $h_i$；
3. 初始化根节点，根节点包含全部样本，计算根节点聚合梯度 $G=\sum g_i,\ H=\sum h_i$；
4. 对当前节点执行贪心分裂搜索，校验约束条件，选出全局最优分裂；
5. 递归生长树，触发任意终止条件时生成叶子，赋值最优叶子权重 $w_j^*=-\dfrac{G_j}{H_j+\lambda}$；
6. 得到完整第 t 棵树 $f_t(x)$；
7. 更新整体预测：$\hat y_i^{(t)}=\hat y_i^{(t-1)}+f_t(x_i)$；
8. 冻结第 t 棵树，迭代训练下一棵树

# ==五、XGBoost分类问题与回归问题的区别==

 ==1、回归任务（MSE均方误差） 损失函数：== $$ l(y_i,\hat y_i)=\frac12(y_i-\hat y_i)^2 $$ $\hat y_i$：回归输出，直接为实数值。 求一阶、二阶梯度： $$ \begin{aligned} g_i &= \frac{\partial l}{\partial \hat y_i} = \hat y_i - y_i \\ h_i &= \frac{\partial^2 l}{\partial \hat y_i^2} = 1 \end{aligned} $$ 特点：
1. 二阶梯度 $h_i\equiv1$，所以聚合 $H_j$ = 该叶子内样本数量；
2. 推理输出直接就是累加结果 $\hat y_i^{(T)}=\sum f_t(x_i)$，不需要额外激活函数；
3. $min\_child\_weight$ 等价于限制叶子最少样本数。

==2、二分类任务（Logistic对数损失）== $$ z_i = \hat y_i $$ 损失（交叉熵）： $$ l(y_i,z_i)= -y_i\log(\sigma(z_i))-(1-y_i)\log(1-\sigma(z_i)),\quad \sigma(z)=\frac{1}{1+e^{-z}} $$ 一阶、二阶梯度： $$ \begin{aligned} \hat p_i &= \sigma(z_i) \\ g_i &= \hat p_i - y_i \\ h_i &= \hat p_i(1-\hat p_i) \end{aligned} $$ 特点： 
1. $h_i$ **不再恒等于1**，和预测概率相关；$H_j=\sum h_i$ 不再等于样本数； 
2. $min\_child\_weight$：限制的是子节点二阶导之和，**不再等价样本数量**； 
3. 初始化：$z_i^{(0)}=0$，对应初始概率 $\sigma(0)=0.5$； 
4. 训练全程不用sigmoid；**推理阶段才执行sigmoid**： $$ \hat p_i=\sigma\left(\sum_{t=1}^T f_t(x_i)\right) $$
还有更多的知识点比如训练时的样本采样的超参数，以及工业场景的近似贪心就不赘述了