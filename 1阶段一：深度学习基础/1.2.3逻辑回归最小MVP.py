"""
这里手写逻辑回归的整个训练过程
"""

from sklearn.datasets import make_moons
import torch

X, y = make_moons(n_samples=(120, 80), noise=0.2, random_state=42)

X = torch.tensor(X, dtype=torch.float32)
y_true = torch.tensor(y, dtype=torch.float32).view(-1, 1)
# 这里需要注意的有两个地方：
# （1）y必须设置dtype=torch.float32, 如果不设置，底层在计算损失函数的时候会自动进行类型提升，但意味着开辟新空间，如果数据量非常多，开辟的新空间非常多，损耗内存。另外，如果是官方的API，比如交叉熵损失函数BCELoss()传入的参数数据类型必须严格统一，因为它们严格不支持自动类型提升
# （2）设置view(-1,1)，保证真实值和预测值绝对的形状对齐

w = torch.randn((2, 1), requires_grad=True)
b = torch.randn(1, requires_grad=True)

for epoch in range(400):
    z = X @ w + b
    y_pred = torch.sigmoid(z)
    
    loss = (-(y_true * torch.log(y_pred) + (1-y_true) * torch.log(1-y_pred))).mean()
    
    loss.backward()
    
    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad
        
    w.grad.zero_()
    b.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"第{epoch}轮的Loss为：{loss.item()}")