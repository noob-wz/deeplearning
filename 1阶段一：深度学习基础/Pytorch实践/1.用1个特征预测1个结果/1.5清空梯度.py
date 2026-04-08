"""
问题：
PyTorch 有一个经常让新手掉坑的“热心肠”设定：当你第二次运行 .backward() 时，它不会把口袋里上一次算好的梯
度扔掉，而是会把新的梯度叠加（相加）到旧的梯度上！

如果走了 10 步，口袋里的数字就是 10 步梯度的总和。这就好比你拿着昨天、前天和大前天的旧地图来决定今天怎么
走路，绝对会掉沟里。

解决方案：
每次走完一步，更新完参数后，必须把口袋里的梯度纸条撕掉（清零），干干净净地迎接下一次计算。
在 PyTorch 里，带有下划线 _ 的方法通常表示“原地修改”（直接把原来的值抹掉替换）。
        w.grad.zero_()
        b.grad.zero_()
"""

import torch

# 1. 准备数据 (特征和真实标签)
x = torch.tensor([2.0, 5.0, 1.0, 8.0, 3.0])
y = torch.tensor([70.0, 85.0, 60.0, 95.0, 72.0])

# 2. 初始化参数 (瞎猜一个开头，并贴上追踪标签)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 设定学习率
lr = 0.01

# 让机器自己训练 300 次 (这叫 300 个 Epoch)
for epoch in range(300):
    
    # 第一步：前向传播 (机器给出当前预测)
    y_pred = w * x + b
    
    # 第二步：计算损失 (看看错得多离谱，整合成一个标量)
    loss = ((y_pred - y) ** 2).mean()
    
    # 第三步：反向传播 (PyTorch自动算导数，塞进 .grad 口袋)
    loss.backward()
    
    # 第四步：更新参数 (叫停追踪，悄悄挪动 w 和 b)
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        
    # 第五步：清空梯度 (把口袋翻干净，准备下一轮)
    w.grad.zero_()
    b.grad.zero_()
    
    # 每跑 50 圈，我们看一眼它的学习进度
    if epoch % 50 == 0:
        print(f"第 {epoch} 轮: 误差(Loss)={loss.item():.2f}, 猜出的公式: y = {w.item():.2f}x + {b.item():.2f}")
        

# 到此，我们完美解决了“用 1 个特征预测 1 个结果”的任务。