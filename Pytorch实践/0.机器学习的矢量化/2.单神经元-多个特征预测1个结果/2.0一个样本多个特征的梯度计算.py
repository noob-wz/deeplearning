import torch

x = torch.tensor([8.,7.,0.9])
W = torch.tensor([.6,.3,.1], requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = torch.matmul(W, x.T) + b

loss = (y_pred - 10)**2# 假设真实值是10

loss.backward()

print(W.grad)

# W.grad 是一个一维向量，其中有三个元素，分别对应样本中的3个特征
# 梯度的shape和参数的shape永远一致