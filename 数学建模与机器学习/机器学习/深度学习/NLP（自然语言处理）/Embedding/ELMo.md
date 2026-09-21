（Embedding from Language Models）

不止训练一个 Q 矩阵，还加入了上下文信息（用双向 [[LSTM与GRU]]，左右 LSTM 获取上下文信息）

![[Pasted image 20260917203952.png]]

这里中间的 LSTM 层可以很多，层数越多越能提炼语义特征，层数越少越接近 [[Word2Vec]]，提取单词特征。