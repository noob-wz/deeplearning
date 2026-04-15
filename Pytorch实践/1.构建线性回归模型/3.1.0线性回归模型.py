import torch
import torch.nn as nn

# 1. 准备数据
# 形状: x (5, 1), y_true (5, 1)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

# 2. 实例化模型
# 进行参数定义和前向传播，并设置为叶子张量
model = nn.Linear(in_feature=1,out_features=1)

# 3. 实例化优化器
optimizer = torch.optim.SGD(model.parameters, lr=0.01)

# 4. 训练循环
for epoch in range(100):
    # 1. 前向传播
    # 直接调用模型，内部自动执行 (5, 1) @ (1, 1) + (1,) -> (5, 1)
    y_pred = model(x)
    
    # 2. 计算损失
    loss = ((y_pred - y_true)**2).mean()
    
    # 3. 反向传播
    # 计算所有叶子张量的梯度，并写入到各自的 grad 属性中。然后销毁计算图并清理内存
    loss.backward()
    
    # 4. 参数更新
    # PyTorch 在设计 Optimizer 抽象类时，直接将挂起 Autograd 引擎的逻辑封装进了底层代码。
    # 即 with torch.no_grad() 上下文管理器封装在 optimizer.step() 方法中
    optimizer.step()
        
    optimizer.zero_grad()