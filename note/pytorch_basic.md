



### 1  .Basic

在PyTorch中直接创建一个tensor的方法主要有2种：使用torch的方法（empty、rand、zeros、tensor...）、实例化torch.Tensor这个类。

```python3
#第一种
gemfield = torch.empty(7, 19)
gemfield = torch.rand(7, 19)
gemfield = torch.zeros(7, 19, dtype=torch.long)
gemfield = torch.tensor([7.0, 19])

#第二种
gemfield = torch.Tensor([1,2])
```



对用户定义tensor,即使进行了in-place运算，仍然可以改变该值
改变方式：
1.赋值运算
2.a.requires_grad_(True)
对运算结果tensor，改变该值会报runtime error

```python
import torch
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad) #False
a.requires_grad_(True)
print(a.requires_grad) #True
b = (a * a).sum()
print(b.requires_grad) #True
b.requires_grad_(False)
print(b.requires_grad) #runtime err
```



#### 不需要追踪的两种方式

1.包在代码块with torch.no_grad(): 中

2.`.detach()`

要停止张量跟踪历史记录，可以调用`.detach()`将其从计算历史记录中分离出来(推荐)，并防止跟踪将来的计算

`grad_fn`

代表tensor是由什么操纵创建的 叶节点为NULL

#### requires_grad 

对用户定义tensor,即使进行了in-place运算，仍然可以改变该值
改变方式：
1.赋值运算
2.a.requires_grad_(True)
对运算结果tensor，改变该值会报runtime error

```python
import torch
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad) #False
a.requires_grad_(True)
print(a.requires_grad) #True
b = (a * a).sum()
print(b.requires_grad) #True
b.requires_grad_(False)
print(b.requires_grad) #runtime err
```

#### tensor求导

```python
# i.e 求 d(out)/d(x)
out.backward()
print(x.grad)
```
分子为标量，backward()可以妹有入参
否则应输入等宽tensor

```python
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)
```

### 2. autograd

```python
# demo in TUTORIAL

# -*- coding: utf-8 -*-
import torch


class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu = MyReLU.apply

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
```



### 3. 

