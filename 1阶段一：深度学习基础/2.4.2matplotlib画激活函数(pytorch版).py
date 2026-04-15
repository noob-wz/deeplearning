"""
只写了pytorch下的relu激活函数，主要目的是为了对比numpy版本的几个不同点：
（1）numpy版本np.maximum(0, z)不需要把0设置成 ndarray
（2）pytorch因为要给梯度画图，所以标记了叶子张量，前向传播存储了计算图，如果不考虑梯度画图，pytorch和
    numpy画图一样，不需要考虑用.detach中断计算图
"""


import torch
import matplotlib.pyplot as plt

z = torch.linspace(-5, 5, 1000, requires_grad=True)

def relu(z):
    # 错误写法：
    # return torch.maximum(0, z) # torch.maximum(input, other) 方法是逐元素取最大值，要求输入参数都必须是张量
    
    # 正确写法 1
    # return torch.maximum(torch.zeros(1), z) # 把 普通的python整数0换成 0维张量，此时经过广播机制可以逐元素对比取最大值
    # 正确写法 2
    return torch.maximum(torch.zeros_like(z), z) # 把 0 和 z 的形状对齐

a = relu(z)

# 因为 a 和 z 都是向量，但.backward不支持对向量求导，所以用.sum将向量转化成标量再求导，实际得到的就是想对每个z求导的值
a.sum().backward()

# matplotlib不能处理带计算图的tensor，通常要加上.detach()来中断计算图
plt.plot(z.detach(), a.detach(), color="b", linewidth=2, label="activation")
plt.plot(z.detach(), z.grad.detach(), color="r", linewidth=2, label="grad_value")

plt.axvline(x=0, color="r", linestyle="--", linewidth=0.5)
plt.axhline(y=1, color="r", linestyle="--", linewidth=0.5)
plt.axhline(y=0, color="r", linestyle="--", linewidth=0.5)

plt.title("relu function")
plt.xlabel("z")
plt.ylabel("activation")
plt.legend()
plt.show()