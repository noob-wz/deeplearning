"""
假设我们现在有 3 个特征（比如身高、体重、年龄），我们要把它们输入给 4 个神经元。

这个 weight 的 shape 打印出来应该是什么？（比如是 [3, 4] 还是别的？）
"""



import torch
import torch.nn as nn

# 假设有 2 个样本，每个样本 3 个特征
X = torch.randn(2, 3)   

# 声明一个 3 输入、4 输出的线性层
layer = nn.Linear(in_features=3, out_features=4)
# 等价于 layer = nn.Linear(3, 4) 
# ⚠️ 注意nn.Linear()中的矩阵维度是（3,4)，不是(4,3)

# 看看 PyTorch 自动初始化的参数形状
print("W 的形状:", layer.weight.shape)  # [4, 3]
print("b 的形状:", layer.bias.shape)    # [4]，每个神经元配一个偏置


"""
如果是一个神经元，那么权重就是一个一维的向量，在进行矩阵向量乘法的时候，@会把权重 W 转化成列向量

如果是同时存在多个神经元，Pytorch的约定是：按照[输出神经元数量：输入特征数]来存储。为了让数学上跑得
通，nn.Linear 在底层计算 y = Wx + b 时，悄悄对 W 做了一次转置（Transpose，代码里叫 .T）。

也就是说，它内部真正执行的公式是：
输出 = X @ W.T + b
把 [4, 3] 转置回 [3, 4]，一切就又顺理成章了。
"""
