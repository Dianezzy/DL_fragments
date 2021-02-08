# 深度学习理论基础



- CrossEntropyLoss：softmax + log + NLLloss

  - softmax:多分类；实数->概率0-1

  ```python
  nn.Softmax(dim=1)  #每行和为1
  ```
  - Logsoftmax :概率0-1-> 负无穷-0

  - NLLloss损失值

    为什么NLLLoss的计算方式可以用来求损失值。经过上面的计算我们知道，Softmax计算出来的值范围在[0, 1]，值的含义表示对应类别的概率，也就是说，每行（代表每张图）中最接近于1的值对应的类别，就是该图片概率最大的类别，那么经过log求值取绝对值之后，就是最接近于0的值，如果此时每行中的最小值对应的类别值与Target中的类别值相同，那么每行中的最小值求和取平均就是最小，极端的情况就是0。总结一下就是，input的预测值与Target的值越接近，NLLLoss求出来的值就越接近于0，这不正是损失值的本意所在吗，所以NLLLoss可以用来求损失值。

    

    

- dropout : 缓解过拟合，达到正则化

  https://zhuanlan.zhihu.com/p/38200980

  - 原理：在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作（隐层节点值为0），明显地减少过拟合现象。这种方式可以减少特征检测器（隐层节点）间的相互作用，检测器相互作用是指某些检测器依赖其他检测器才能发挥作用。

    使模型泛化性更强，因为它不会太依赖某些局部的特征

  









### 激活函数

- ReLU:降低神经元激活率

  通过ReLU实现**稀疏**后的模型能够更好地挖掘相关特征，拟合训练数据

  此外，相比于其它激活函数来说，ReLU有以下优势：**对于线性函数而言，ReLU的表达能力更强**，尤其体现在深度网络中；而对于非线性函数而言，ReLU由于非负区间的梯度为常数，因此**不存在梯度消失问题(Vanishing Gradient Problem)**，使得**模型的收敛速度维持在一个稳定状态**。这里稍微描述一下什么是梯度消失问题：当梯度小于1时，预测值与真实值之间的误差每传播一层会衰减一次，如果在深层模型中使用sigmoid作为激活函数，这种现象尤为明显，将导致模型收敛停滞不前。

  

- 为什么最初的神经网络用sigmoid

  任何连续多元函数都能被一组一元函数的**有限次**叠加而成

  这个数学原理被称为[Kolmogorov-Arnold representation theorem](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem)是对于Hilbert第13问题的部分回答。

  **两层**有限次叠加足够**准确**表示任何多元函数。这比多项式逼近要厉害多了，用多项式去**准确**表示一个连续多元（非多项式）函数需要无穷多项

- ReLU通常只能在隐层用的，输出层是不用的，例如，输出层二分类就是sigmoid，多分类就是softmax。



### 解决梯度消失

- 残差连接：skip connect

  **y = H(x,WH) + X**

  即将单元的输入直接与单元输出加在一起，然后再激活

  <img src="https://pic4.zhimg.com/80/v2-a75e6636ebb983125086d9f63f7b27b3_1440w.jpg" alt="img" style="zoom:50%;" />

  **神经网络的退化（秩很低，每次只有少量隐藏单元对不同的输入改变他们的值，**而大部分隐藏单元对不同的输入都是相同的反应。连乘后秩更小。虽然是一个很高维的矩阵，但是大部分维度却没有信息，表达能力没有看起来那么强大），才是难以训练深层网络根本原因所在，而不一定是梯度消失

  ![img](https://pic2.zhimg.com/80/v2-a2e2f2e3d5548367c77739d49e726a7d_1440w.jpg)

由bc可知，残差连接可以打破网络的对称性



### 解决梯度爆炸

梯度裁剪

常见的梯度裁剪有两种

- 确定一个范围，如果参数的gradient超过了，直接裁剪
- 根据若干个参数的gradient组成的的vector的L2 Norm进行裁剪

第一种方法，比较直接，对应于pytorch中的nn.utils.clip_grad_value(parameters, clip_value). 将所有的参数剪裁到 [ -clip_value, clip_value]

第二中方法也更常见，对应于pytorch中clip_grad_norm_(parameters, max_norm, norm_type=2)。

**使用位置**：在backward得到梯度之后，step()更新之前，使用梯度剪裁。从而完成计算完梯度后，进行裁剪，然后进行网络更新的过程。



### 解决梯度消失/爆炸

标准初始化

中间层正规化



### 防止过拟合





# ASR 语音识别

- Step1:特征提取

  如利用输入的waveform提取MFCC特征，然后再经过三个独立的模型再求得它们概率的乘积得到总的概率

  - 三个model
    - acoustic model (get phoneme)
    - Pronunciation model
    - 

  - MFCC
    - 对语音信号进行分帧处理:假设短时间内语音稳定
    - 用周期图(periodogram)法来进行功率谱(power spectrum)估计：
    - 对功率谱用Mel滤波器组进行滤波，计算每个滤波器里的能量：滤波器将wave用频率分成不同bin
    - 对每个滤波器的能量取log:听觉的感受不是线性的；log可以归一化掉不同信道的差别
    - 进行DCT变换：去掉不同mel滤波器的相关性
    - 保留DCT的第2-13个系数，去掉其它



#### Mel Scale

声音实际频率->人听觉听到的频率



### Highway Network 

相比于传统的神经网路随着深度增加训练很难， highway network训练很简单， 使用简单的SGD就可以， 而且即使网络很深甚至到达100层都可以很好的去optimization.



# 训练过程

1. datasets

2. dataloaders

3. define a model architecture

4. define criterion（损失函数）

5. Define optimizer（优化器）

   （BGD、SGD、MBGD、Momentum、NAG、Adagrad、Adadelta、RMSprop、Adam）









