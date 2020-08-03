## 1.架构

![transformer_architecture](https://user-gold-cdn.xitu.io/2018/9/17/165e5814fae0765f?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

输入序列经过**word embedding**和**positional encoding**相加后，输入到encoder。

输出序列经过**word embedding**和**positional encoding**相加后，输入到decoder。

最后，decoder输出的结果，经过一个线性层，然后计算softmax。

## Encoder

encoder由6层相同的层组成，每一层分别由两部分组成：

- 第一部分是一个**multi-head self-attention mechanism**
- 第二部分是一个**position-wise feed-forward network**，是一个全连接层

两个部分，都有一个 **残差连接(residual connection)**，然后接着一个**Layer Normalization**。

如果你是一个新手，你可能会问：

- multi-head self-attention 是什么呢？
- 参差结构是什么呢？
- Layer Normalization又是什么？


