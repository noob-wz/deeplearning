"""要区分逻辑回归和二分类神经网络：

逻辑回归可以看成“最简单的二分类神经网络”，但不是所有二分类神经网络都是逻辑回归。

在这个例子中： x --> linear --> relu --> linear --> sigmoid，所以下面的模型是多层复合函数。逻辑
回归是没有隐藏层的二分类神经网络，有了隐藏层和非线性之后，就不再叫逻辑回归，而是神经网络分类器。

逻辑回归虽然它最后用了 sigmoid，但它真正决定分类边界的是：wx+b=0，所以逻辑回归是线性分类器。而多层神经
网络是非线性分类器

"""


import torch
import torch.nn as nn
from sklearn.datasets import make_moons
import torch.optim as optim

X, y = make_moons(n_samples=(150,50), noise=0.2, random_state=42)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

class Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
model = Regression()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f"第{epoch}轮的loss为：{loss.item()}") 
        
