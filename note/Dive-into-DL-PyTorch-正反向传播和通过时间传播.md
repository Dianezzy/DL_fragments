# Dive-into-DL-PyTorch-正反向传播和通过时间传播

为啥rnn中没有梯度裁剪（gradient-clipping）会报错

链式法则求导规律 前面的全不变，最后一项变成矩阵转置。注意该矩阵还是保持原位置（左乘在前，右乘在后）