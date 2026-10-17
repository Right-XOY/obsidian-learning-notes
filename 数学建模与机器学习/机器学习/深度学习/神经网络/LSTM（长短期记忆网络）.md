# ==一、为什么需要 LSTM==

[[RNN（循环神经网络）]] 存在**长期依赖**问题：序列一旦过长，BPTT 中梯度连乘 $\prod \boldsymbol W_{hh}^{\mathrm T}\,\mathrm{diag}(f')$ 会指数级衰减或爆炸，导致模型学不到「很久以前」的信息。

> 一句话：LSTM 给记忆专门修了一条「高速公路」（细胞状态 $\boldsymbol C_t$），用**门控**控制信息流过，让梯度可以长距离无损传递。

经典RNN架构如图所示：

![[Pasted image 20261017105250.png|373]]

LSTM架构对比如下所示：

![[Pasted image 20261017105319.png|372]]

***

# ==二、LSTM 的核心思想==

LSTM 在 RNN 的隐藏状态 $\boldsymbol h_t$ 之外，额外引入一条贯穿所有时间步的**细胞状态（cell state）** $\boldsymbol C_t$，作为「记忆主干道」。

$\boldsymbol C_t$ 的更新只做两件事：

$$
\boldsymbol C_t = f_t \odot \boldsymbol C_{t-1} + i_t \odot \tilde{\boldsymbol C}_t
$$

- 用**遗忘门** $f_t$ 决定「丢掉多少旧记忆」
- 用**输入门** $i_t$ 决定「写入多少新记忆」

这条加法通道就是 LSTM 抗梯度消失的关键：梯度沿 $\boldsymbol C_t$ 回传时，乘上的是 $f_t$（取值 $[0,1]$，可以接近 1），而不是像 RNN 那样反复乘 $W_{hh}$。

***

# ==三、遗忘门==

![[Pasted image 20261017123353.png|306]]

$$
f_t = \sigma\big(\boldsymbol W_f[\boldsymbol h_{t-1},\boldsymbol x_t] + \boldsymbol b_f\big)
$$

提供 sigmoid 函数映射进行遗忘功能的实现，是"软门"，可以选择保留比例 [0-1]。

输出为 $f_t$，代表**对旧细胞状态 $\boldsymbol C_{t-1}$ 的保留比例**：$f_t=0$ 表示完全忘掉，$f_t=1$ 表示完全保留。

> 例：在语言模型中读到句号，遗忘门会把上一句的主语信息逐步清掉。

***

# ==四、输入门==

![[Pasted image 20261017125230.png|305]]

$$
\begin{aligned}
i_t &= \sigma\big(\boldsymbol W_i[\boldsymbol h_{t-1},\boldsymbol x_t] + \boldsymbol b_i\big) \\
\tilde{\boldsymbol C}_t &= \tanh\big(\boldsymbol W_C[\boldsymbol h_{t-1},\boldsymbol x_t] + \boldsymbol b_C\big)
\end{aligned}
$$

sigmoid 函数映射后与类似原 RNN 网络的 tanh 映射后的矩阵做权重乘积。

输出为 $i_t$ 和 $\tilde{\boldsymbol C}_t$，分别代表：

- $i_t$：**输入门控信号**，决定候选新信息有多少能写入细胞状态，取值 $[0,1]$
- $\tilde{\boldsymbol C}_t$：**候选新记忆**，由当前输入和上一隐藏状态通过 $\tanh$ 生成，取值 $(-1,1)$，就像 RNN 中算出的新信息

> 注意：候选值是用 $\tanh$ 生成的（见 [[激活函数]]），门控信号是用 $\sigma$ 生成的，两者作用不同。

***

# ==五、细胞状态更新==

![[Pasted image 20261017130148.png|305]]

$\boldsymbol C_{t-1}$ 是上一时刻的细胞状态，计算过程如下图：

![[Pasted image 20261017130422.png|306]]

$$
\boldsymbol C_t = f_t \odot \boldsymbol C_{t-1} + i_t \odot \tilde{\boldsymbol C}_t
$$

其中：
- $i_t$ 是当前时刻的**输入门**，取值为[0, 1]
- $f_t$ 是当前时刻的**遗忘门**，取值为[0, 1]
- $\odot$ 表示**按元素相乘**（Hadamard 积）

> 这一步是 LSTM 的灵魂：旧记忆「乘以遗忘比例」加上新记忆「乘以写入比例」，纯加法让梯度可以沿 $\boldsymbol C_t$ 长距离流动，不衰减。

***

# ==六、输出门==

![[Pasted image 20261017131855.png|305]]

$$
\begin{aligned}
o_t &= \sigma\big(\boldsymbol W_o[\boldsymbol h_{t-1},\boldsymbol x_t] + \boldsymbol b_o\big) \\
\boldsymbol h_t &= o_t \odot \tanh(\boldsymbol C_t)
\end{aligned}
$$

sigmoid 映射之后的值 ($o_t$) 与细胞更新、做完 tanh 映射之后的值相乘，得到当前时间步的 $\boldsymbol h_t$。

- 向上输出的 $\boldsymbol h_t$ 表示当前时间步的输出
- 向右输出的 $\boldsymbol h_t$ 表示送入下一时间步进行计算的输入

$o_t$ 决定**细胞状态中有多少要暴露给当前输出**，取值 $[0,1]$。

***

# ==七、三个门总结速查==

| 门 | 公式 | 激活 | 作用 | 输出含义 |
| --- | --- | --- | --- | --- |
| 遗忘门 $f_t$ | $\sigma(W_f[h_{t-1},x_t]+b_f)$ | sigmoid | 决定旧记忆保留多少 | 0=全忘，1=全留 |
| 输入门 $i_t$ | $\sigma(W_i[h_{t-1},x_t]+b_i)$ | sigmoid | 决定新记忆写入多少 | 0=不写，1=全写 |
| 输出门 $o_t$ | $\sigma(W_o[h_{t-1},x_t]+b_o)$ | sigmoid | 决定当前输出暴露多少 | 0=不输出，1=全输出 |
| 候选 $\tilde C_t$ | $\tanh(W_C[h_{t-1},x_t]+b_C)$ | tanh | 生成候选新记忆 | $(-1,1)$ |

***

# ==八、LSTM 为什么能抗梯度消失==

BPTT 时，梯度沿细胞状态回传：

$$
\frac{\partial \boldsymbol C_t}{\partial \boldsymbol C_{t-1}} = \mathrm{diag}(f_t)
$$

从 $t$ 时刻回传到 $k$ 时刻：

$$
\frac{\partial \boldsymbol C_t}{\partial \boldsymbol C_k} = \prod_{i=k+1}^{t} \mathrm{diag}(f_i)
$$

==关键差别：== RNN 里梯度连乘的是 $\boldsymbol W_{hh}$（一个矩阵，特征值可能远大于或小于 1），而 LSTM 连乘的是遗忘门 $f_i$（sigmoid 输出，模型可以学到让 $f_i\approx 1$ 来保持记忆）。

当遗忘门接近 1 时，梯度**几乎无损**地流过细胞状态通道，长期依赖得以学习。这条「加法记忆通道」也被称为 ==constant error carousel==（恒定误差传送带）。

> 这正是 LSTM 解决 [[RNN（循环神经网络）]] 梯度消失问题的数学本质。

***

# ==九、变体与对比==

**变体：**
- **Peephole LSTM**：让三个门也能「看到」细胞状态 $\boldsymbol C_t$，在时序预测中偶尔更准
- **双向 LSTM**：正向 + 反向两个 LSTM，拼接隐藏状态，能同时利用上下文（[[ELMo]] 就是双向 LSTM 堆叠）

**与 RNN、GRU 对比：**

| 模型       | 记忆结构          | 门数量           | 参数量 | 长依赖能力      |
| -------- | ------------- | ------------- | --- | ---------- |
| RNN      | 仅 $h_t$       | 无             | 少   | 差（梯度消失）    |
| ==LSTM== | $h_t$ + $C_t$ | 3 个（遗忘/输入/输出） | 多   | 强          |
| [[GRU]]  | 仅 $h_t$       | 2 个（重置/更新）    | 中   | 强（接近 LSTM） |

> GRU 用更少的门达到了与 LSTM 相近的效果，是「轻量化」版本。两者在大多数任务上表现接近，数据量小或追求效率时可选 GRU。

***

# ==十、LSTM 的优缺点==

**优点：**
- 有效解决长序列的梯度消失问题，能建模长期依赖
- 门控机制让记忆可控，应用广泛（机器翻译、语音识别、时序预测）

**缺点：**
- 参数量大（3 套门 + 候选向量，约 4 倍于 RNN）
- 仍需按时间步串行计算，无法并行 → 训练慢
- 长序列任务上最终被 [[注意力机制]] 主导的 Transformer 取代
