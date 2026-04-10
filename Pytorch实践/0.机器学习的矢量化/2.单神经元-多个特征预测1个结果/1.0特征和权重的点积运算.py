"""
如果输入的样本存在非常多的特征，如果是图像可能有几万个特征，特征和权重之间又是一一对应，手写模型：
y = w1*x1 + w2*x2 + ... + wn*xn + b
不太可能的。解决方法就是把特征和权重都打包成向量，然后进行“点积”运算，大白话就是：把向量中的对应元素相乘
再求和
"""

import torch

# x 变成了包含 3 个特征的一维向量（学习8小时，睡7小时，模拟考0.9分）
x = torch.tensor([8.0, 7.0, 0.9])

# W 也是包含 3 个权重的一维向量，分别对应上面三个特征的重要程度
W = torch.tensor([0.6, 0.3, 0.1],requires_grad=True)

b = torch.tensor(0.0,requires_grad=True)

# 使用 torch.dot() 进行点积运算
y = torch.dot(W, x) + b

import torch

x = torch.tensor([8.,7.,0.9])
W = torch.tensor([.6,.3,.1])
b = torch.tensor(0.0)

y = torch.dot(W, x) + b

loss = y