"""
这里手写逻辑回归的整个训练过程
"""

from sklearn.datasets import make_moons
import torch

X, y = make_moons(n_samples=(120, 80), noise=0.2, random_state=42)

X = torch.tensor(X, dtype=torch.float32)
y_true = torch.tensor(y, dtype=torch.float32).view(-1, 1)

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