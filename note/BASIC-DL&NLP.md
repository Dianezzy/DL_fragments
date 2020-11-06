### 深度学习理论基础



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

  



- 梯度裁剪

  常见的梯度裁剪有两种

  - 确定一个范围，如果参数的gradient超过了，直接裁剪
  - 根据若干个参数的gradient组成的的vector的L2 Norm进行裁剪

  第一种方法，比较直接，对应于pytorch中的nn.utils.clip_grad_value(parameters, clip_value). 将所有的参数剪裁到 [ -clip_value, clip_value]

  第二中方法也更常见，对应于pytorch中clip_grad_norm_(parameters, max_norm, norm_type=2)。

  **使用位置**：在backward得到梯度之后，step()更新之前，使用梯度剪裁。从而完成计算完梯度后，进行裁剪，然后进行网络更新的过程。

