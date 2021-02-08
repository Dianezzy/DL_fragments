### RNN

-  zoneout vs dropout

  Zoneout是rnn 时间维度上的“dropout”，要么维持前一个时刻的hidden vector，要么按照一般的样子更新。不是指单独的cell，而是指训练时的一种trick。Dropout就是通用的一种深度学习技巧，训练时随机失活一些神经元，可以增强模型泛化抑制过拟合作用。zoneout是指随机失活一个rnncell，跳过一步。

### Tacotron&Tacotron2——基于深度学习的端到端语音合成模型

<img src="https://pic2.zhimg.com/v2-dfe2d3b63a6d822ddac36a5720fa8af6_1440w.jpg?source=172ae18b" alt="Tacotron&Tacotron2——基于深度学习的端到端语音合成模型" style="zoom:50%;" />

#### 前置知识

- Ancestor: wavenet

  - Wavenet最大的成功之处就是使用dilated causal convolution技术来增加CNN的receptive field，从而提升了模型建模long dependency的能力

    <img src="https://pic2.zhimg.com/80/v2-536d29ce9011e85c1753c1e0d21676e9_1440w.jpg" alt="img" style="zoom:50%;" />

  - 门控机制

  - 残差链接

- GRU

  <img src="https://pic3.zhimg.com/v2-5b805241ab36e126c4b06b903f148ffa_b.jpg" alt="img" style="zoom:50%;" />

- 模型和sequence-to-sequence模型非常相似，大体上由encoder和decoder组成，raw text经过pre-net, CBHG两个模块映射为hidden representation，之后decoder会生成mel-spectrogram frame

### CBHG

<img src="https://pic4.zhimg.com/80/v2-5b826f2065171228fa1e0cded2e43d33_1440w.jpg" alt="img" style="zoom:50%;" />

