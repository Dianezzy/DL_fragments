###### teacher forcing

中文 

每次输出一个word

你大  你大爷

### 1 non-autoregressive

- 自回归训练方式中是不需要显式地给出目标序列的长度的，这个长度是在解码过程中通过预测生成EOS符号隐式定义的，但在非自回归训练方式中目标序列长度需要事先确定，然后再并行地生成其中每一个单词。
- 可以认为自回归训练方式和非自回归训练方式在encoder端是完全一样的，只是在decoder端不同，自回归训练方式解码时通过此前所有时刻的结果来预测下一时刻的生成词，而非自回归训练方式在解码时通过一个隐变量来生成每一时刻的结果，使得整个解码过程独立并行。

pNA(Y|X;θ)=pL(T|x1:T′;θ)· p(yt|x1:T′;θ)



啥是hidden size

![Screen Shot 2020-08-08 at 10.30.32 AM](/Users/zhying/Desktop/Screen Shot 2020-08-08 at 10.30.32 AM.png)



![Screen Shot 2020-08-10 at 9.34.56 AM](/Users/zhying/Desktop/Screen Shot 2020-08-10 at 9.34.56 AM.png)

### 细节

- 

- 由于fertility predictor的输出是一个离散的序列，所以作者采用了类似ELBO的方法来训练整个模型。具体的公式推导参见论文原文，这里不赘述了。



### 3 FastSpeech: Fast, Robust and Controllable Text to Speech









