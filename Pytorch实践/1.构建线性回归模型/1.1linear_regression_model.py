import torch
import torch.nn as nn

# ==========================================
# 1. 准备数据 (Data Preparation)
# ==========================================
# 伪造一批已知规律的数据，数据永远包含两部分：特征和标签。但是现实总有误差，所以人为地也加入一点噪音

# 生成100个随机的输入特征
x_train = torch.randn(100,1)

# 伪造已知规律的标签并人为加入噪声
y_train = 2 * x_train + 1 + torch.randn(100,1) * 0.2



# ==========================================
# 2. 定义模型 (Model Definition)
# ==========================================

class LinearRegressionModel(nn.Module):
    def __init__(self):
        # 必须调用父类的初始化
        super(LinearRegressionModel, self).__init__()
        
        # 定义一个线性层：输入特征维度为 1，输出预测值维度为 1
        # 这内部就包含了形状为(1, 1)的权重w 和形状为(1,)的偏置
        super.linear = nn.Linear(in_features=1, out_features=1)
        
    def forward(self, x):
        # 定义前向传播的过程。这里极其简单，就是让x穿过那个线性层
        return self.linear(x)