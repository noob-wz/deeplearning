"""
用 sklearn 的 `make_moons` 生成玩具数据集并跑通前向传播
"""

import torch
import torch.nn as nn
from sklearn.datasets import make_moons

# 1. 固定随机种子
torch.manual_seed(42)

# 2. 生成 make_moons 玩具数据集
# X: (200, 2) -> 200个样本，每个样本2个特征
# y: (200,)   -> 200个标签，类别是0或1
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# 3. 把 sklearn 生成的 numpy 数组转成 PyTorch 张量，不然不能直接喂给 PyTorch 模型
X = torch.tensor(X, dtype=torch.float32)                 # shape: (200, 2)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)    # shape: (200, 1)

# 这里需要注意的有两个地方：
# （1）y必须设置dtype=torch.float32, 如果不设置，底层在计算损失函数的时候会自动进行类型提升，但意味着开辟新空间，如果数据量非常多，开辟的新空间非常多，损耗内存。另外，如果是官方的API，比如交叉熵损失函数BCELoss()传入的参数数据类型必须严格统一，因为它们严格不支持自动类型提升
# （2）设置view(-1,1)，保证真实值和预测值绝对的形状对齐

print("X.shape =", X.shape)
print("y.shape =", y.shape)

# 4. 定义一个简单的多层感知机
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=16, out_features=1)
        self.sigmoid = nn.Sigmoid()  # 二分类时把输出映射到 0~1

    def forward(self, x):
        x = self.fc1(x)      # (200, 2) -> (200, 16)
        x = self.relu(x)     # (200, 16) -> (200, 16)
        x = self.fc2(x)      # (200, 16) -> (200, 1)
        x = self.sigmoid(x)  # (200, 1) -> (200, 1)
        return x

# 5. 实例化模型
model = SimpleMLP()

# 6. 前向传播
y_pred = model(X)

print("y_pred.shape =", y_pred.shape)
print("前5个预测值：")
print(y_pred[:5])