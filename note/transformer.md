论文中的transfomer用于机器翻译，本质上是一个Encoder-Decoder的结构，编码器的输出会作为解码器的输入



## 知识储备

- 自回归

  用历史数据的线形加权和+噪声扰动表示当前当前时序数据 

### RNN

t-1时刻的输出是t时刻的输入

### Attention

内积表示相似度，d_k用于归一化，softmax得到的score值相当于一个权重

### transfomer

推断时把上一次的输出embedding后作为这一次输入。当前时刻的输入是之前所有输入+上次输出。



## 2 架构



输入序列经过**word embedding**和**positional encoding**相加后，输入到encoder。

输出序列经过**word embedding**和**positional encoding**相加后，输入到decoder。

最后，decoder输出的结果，经过一个线性层，然后计算softmax。

<img src="https://n.sinaimg.cn/front/610/w1080h330/20190108/YmrV-hrkkwef7008971.jpg" alt="img" style="zoom:50%;" />

## 3 Encoder

encoder由6层相同的层组成，每一层分别由两部分组成：



- 第一部分是一个**multi-head self-attention mechanism**

  有依赖关系，无法并行

  数据经过一个self-attention结构得到加权特征向量

  $Z=Softmax(\frac{Q*K^T}{\sqrt{d_K}})*V$

- 第二部分是一个**position-wise feed-forward network**，是一个全连接层

  无依赖关系，可以并行
  
  第一层的激活函数是ReLU，第二层是一个线性激活函数
  
  $FFN(Z)=max(0,ZW1+B1)W2+B2$

两个部分，都有一个**残差连接(residual connection)**，然后接着一个**Layer Normalization**。



<img src="http://aistar.site/20190404/6.jpg" alt="img" style="zoom:50%;" />



- 输入

  word----embedding---->word vector ----自注意层--->得分矩阵z

  embedding可以加入位置编码：词嵌入+ 位置编码矩阵 ->encoder的输入

  

- multi-head self-attention 

  ![[公式]](https://www.zhihu.com/equation?tex=MultiHead%28Q%2C+K%2C+V%29+%3D+Concat%28head_1%2C+...%2C+head_h%29W%5E0)

  其中，

  ![[公式]](https://www.zhihu.com/equation?tex=head_i+%3D+Attention%28QW_i%5EQ%2C+KW_i%5EK%2C+VW_i%5EV%29)

- 残差连接

  学习目标从f x变为fx+x

  作用：在网络很深时防止梯度消失

  实现

  残差并非一定需要是x 

  





**Q**

- 多头注意力机制

  为什么能更好地知道it指代的是什么？比如it的不同注意力头部更好地集中在不同单词上，简单地求和会平均掉局部

  怎么work的？

  这里对x的编码怎么编出8*512

  线形分割。假设分成 h 份，在 ![[公式]](https://www.zhihu.com/equation?tex=d_Q) 、![[公式]](https://www.zhihu.com/equation?tex=d_K) 和 ![[公式]](https://www.zhihu.com/equation?tex=d_V)的维度上进行切分。进入到scaled dot-product attention 的 ![[公式]](https://www.zhihu.com/equation?tex=d_K) 实际上等于未进入之前的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7BD_K%7D%7Bh%7D) 。

<img src="https://n.sinaimg.cn/front/79/w1080h599/20190108/aEDY-hrkkwef7009206.jpg" alt="img" style="zoom:50%;" />



## Decoder

- 第一个部分是 multi-head self-attention mechanism
- 第二部分是 multi-head context-attention mechanism
- 第三部分是一个 position-wise feed-forward network

和 encoder 一样，上面三个部分的每一个部分，都有一个残差连接，后接一个 **Layer Normalization**。

decoder 和 encoder 不同的地方在 multi-head context-attention mechanism

<img src="http://aistar.site/20190404/7.jpg" alt="img" style="zoom:70%;" />

### 训练

- **输入** 上次的输入+embedding（标记中下一个单词）(实操中一般一次把标记全输进去加一个sequencing mask);第一次输入是一个特殊的token （bos/eos/等等）

   来之与解码器的上一个输出， ![[公式]](https://www.zhihu.com/equation?tex=K) 和 ![[公式]](https://www.zhihu.com/equation?tex=V)则来自于与编码器的输出

- **输出** 一个代表下一个预测单词的实数向量

### inference

- **输入** 上次的输入+embedding（上次的输出）
- **输出** 一个代表下一个预测单词的实数向量





## 一些细节

### Attention层

##### k q v的含义

- encoder ： self-attention

  Q、K、V 是上一层 encoder 的输出

  （第一层 encoder，是 word embedding 和 positional encoding 相加得到的输入）

- decoder ： self-attention 中

  Q、K、V 是上一层 decoder 的输出

  （第一层 decoder是 word embedding 和 positional encoding 相加得到的输入）

     但是对于 decoder，不希望它能获得将来的信息，因此还需要需要进行 sequence masking。

- 在 encoder-decoder attention 中，Q 来自于 decoder 的上一层的输出，K 和 V 是encoder 的输出，**k=v**

- Q、K、V 的维度都是一样的







### 层归一化

区别于批量归一化(对每个batch求均值),层归一化是对每一个样本取均值

### Mask

- 为保证超过样本长度的部分不处理的paddling mask
- decoder self attention 的sequential mask：上三角矩阵 ，保证读不到未来数据

### 位置编码

优点：能扩展到未知序列长度（类似变长数组）

![[公式]](https://www.zhihu.com/equation?tex=PE%28pos%2C+2i%29+%3D+sin%28pos%2F10000%5E%7B2i%2Fd_%7Bmodel%7D%7D%29)

![[公式]](https://www.zhihu.com/equation?tex=PE%28pos%2C+2i%2B1%29+%3D+cos%28pos%2F10000%5E%7B2i%2Fd_%7Bmodel%7D%7D%29)

pos 是指词语在序列中的位置。偶数位置使用正弦编码，奇数位置使用余弦编码

**note** 上述是绝对位置编码。但是使用三角函数保证了相对位置信息也被利用了





#### reference





1.动手学深度学习第10章，建议跟着视频一起看
2.Transformer中文图解：https://blog.csdn.net/longxinchen_ml/article/details/86533005
3.Transformer代码+注释：http://nlp.seas.harvard.edu/2018/04/03/attention.html?tdsourcetag=s_pcqq_aiomsg
4.https://juejin.im/post/6844903680487981069#comment

5.https://www.nowcoder.com/discuss/258321

