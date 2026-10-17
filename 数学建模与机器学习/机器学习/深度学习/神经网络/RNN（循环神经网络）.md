# ==一、为什么需要 RNN==

[[MLP（多层感知机）]] 处理的是「一个输入 → 一个输出」的固定维度数据，但现实中大量数据是**序列**：

- 文本：一句话里词的先后顺序决定语义
- 语音、股价、气温：当前值依赖历史值

MLP 处理序列有两大硬伤：

1. **输入维度固定**——句子长度不一，无法直接喂入
2. **没有记忆**——每个样本独立处理，无法利用「上文」信息

> 一句话：RNN 给网络加了一个「隐藏状态」当记忆，让当前输出不仅看当前输入，还看「之前记住了什么」。

***

# ==二、RNN 的结构==

RNN 的核心是**循环单元**：每个时间步 $t$ 接收当前输入 $\boldsymbol x_t$ 和上一时刻的隐藏状态 $\boldsymbol h_{t-1}$，输出新的隐藏状态 $\boldsymbol h_t$。

$$
\boldsymbol h_t = f\big(\boldsymbol W_{xh}\boldsymbol x_t + \boldsymbol W_{hh}\boldsymbol h_{t-1} + \boldsymbol b_h\big)
$$

其中：

- $\boldsymbol h_t$：==隐藏状态==（hidden state），即网络的「记忆」，维度自定义
- $\boldsymbol x_t$：第 $t$ 个时间步的输入（如第 $t$ 个词的词向量）
- $\boldsymbol W_{xh}$：输入 → 隐藏的权重矩阵
- $\boldsymbol W_{hh}$：隐藏 → 隐藏的**循环权重矩阵**，是 RNN 的灵魂
- $f$：激活函数，RNN 中通常用 $\tanh$（见 [[激活函数]]），输出范围 $(-1,1)$ 且零中心

输出层（每个时间步可有输出）：

$$
\boldsymbol y_t = g\big(\boldsymbol W_{hy}\boldsymbol h_t + \boldsymbol b_y\big)
$$

==关键特性：参数共享==

所有时间步**共享同一套** $\boldsymbol W_{xh},\boldsymbol W_{hh},\boldsymbol W_{hy}$。这带来两个好处：

- 参数量与序列长度无关，任意长的序列都能处理
- 「位置无关」的模式识别：无论「猫」出现在句子哪个位置，都用同样的方式提取特征

> 把 RNN 按时间**展开**（unfold），就是一个深度等于序列长度的超深前馈网络，只是每层权重相同——这是理解 BPTT 的关键视角。

***

# ==三、前向传播==

按时间步逐个计算：

1. 初始化 $\boldsymbol h_0$（一般为全零向量）
2. 对每个 $t=1,2,\dots,T$：
   - $\boldsymbol h_t = \tanh(\boldsymbol W_{xh}\boldsymbol x_t + \boldsymbol W_{hh}\boldsymbol h_{t-1} + \boldsymbol b_h)$
   - $\boldsymbol y_t = \mathrm{softmax}(\boldsymbol W_{hy}\boldsymbol h_t + \boldsymbol b_y)$（分类任务）
3. 收集各时刻输出，计算 [[损失函数]]（如各时刻交叉熵之和）

$$
\mathcal L = \sum_{t=1}^{T}\mathcal L_t
$$

> 以文本生成为例：输入「我 爱 中」，逐个时间步预测下一个词，第 3 步的输出应该是「国」。$\boldsymbol h_t$ 里就编码了「前面看过我爱中」这个信息。

***

# ==四、反向传播：BPTT==

BPTT（Backpropagation Through Time，随时间反向传播）：把展开后的 RNN 当作超深网络，用 [[前向与反向传播]] 的链式法则对所有时间步求梯度。

由于参数共享，$\boldsymbol W_{hh}$ 在**每个时间步都被用了一次**，所以总梯度要把所有时刻的贡献**累加**：

$$
\frac{\partial \mathcal L}{\partial \boldsymbol W_{hh}}=\sum_{t=1}^{T}\frac{\partial \mathcal L_t}{\partial \boldsymbol W_{hh}}
$$

而对单个 $\mathcal L_t$，梯度要沿时间**往回传到**更早的时刻 $k<t$，链式法则给出连乘项：

$$
\frac{\partial \mathcal L_t}{\partial \boldsymbol h_k}=\frac{\partial \mathcal L_t}{\partial \boldsymbol h_t}\prod_{i=k+1}^{t}\boldsymbol W_{hh}^{\mathrm T}\,\mathrm{diag}\big(f'(\boldsymbol z_i)\big)
$$

==梯度消失与梯度爆炸的根源就在这个连乘里：==

- $f=\tanh$ 时 $|f'|\le 1$，若 $\boldsymbol W_{hh}$ 的特征值也 $<1$，连乘 $t-k$ 次后梯度**指数级衰减到 0** → ==梯度消失==，模型学不到「很久以前」的依赖（长期依赖问题）
- 若特征值 $>1$，梯度**指数级膨胀** → ==梯度爆炸==，训练直接发散

> 本质：RNN 的「记忆通道」是一条纯靠矩阵连乘传递信息的高速公路，时间隔得越远，信息衰减得越厉害。这和深层网络梯度消失同源，只是这里「深度」是时间。

***

# ==五、解决方案==

| 方案 | 思路 | 说明 |
| --- | --- | --- |
| 梯度裁剪 | 治标 | 梯度范数超过阈值就按比例缩小，专治梯度爆炸 |
| 换激活函数 | 治标 | ReLU 导数为 1，可缓解，但易爆炸且丢失负信息 |
| 门控结构 | ==治本== | [[LSTM]]、[[GRU]] 用门控单元加「加法通道」传递记忆，梯度可以无损流动 |
| 注意力机制 | 另辟蹊径 | [[注意力机制]] 让任意两个位置直接相连，彻底绕开「按时间连乘」的瓶颈 |

> 现代 NLP 中 RNN 基本被 Transformer 取代，但 RNN 的「循环 + 门控」思想仍是理解 LSTM/GRU 的必经之路，[[ELMo]] 也是基于双向 LSTM 构建的。

***

# ==六、常见变体与应用==

**结构变体：**

- **双向 RNN（BiRNN）**：一个正向读序列、一个反向读序列，两个隐藏状态拼接——能同时利用「上下文」信息（ELMo 用的就是双向 LSTM）
- **深层 RNN**：多个 RNN 层堆叠，下层输出序列作为上层输入

**按输入输出形态分类（sequence-to-sequence）：**

| 形态 | 场景 |
| --- | --- |
| 一对一 | 普通分类（退化为 MLP） |
| 一对多 | 图像描述生成（图 → 一段文字） |
| 多对一 | 文本分类、情感分析（一句话 → 标签） |
| 多对多（同步） | 序列标注、词性标注（每步都有输出） |
| 多对多（异步） | 机器翻译（读完再译，即 Encoder-Decoder） |

***

# ==七、RNN 的优缺点==

**优点：**

- 天然处理变长序列，参数共享、参数量小
- 隐藏状态具备记忆能力，能建模时间依赖

**缺点：**

- ==长期依赖==学不好（梯度消失/爆炸）
- 必须按时间步**串行**计算，无法并行，训练慢
- 最终被 [[注意力机制]] 为主的 Transformer 在大规模任务上取代
