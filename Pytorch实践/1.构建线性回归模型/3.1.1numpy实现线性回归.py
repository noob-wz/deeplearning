import numpy as np

# 数据：y = 2x，5 个样本
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

# 初始参数
w = 0.0
b = 0.0
lr = 0.01  # 学习率

for epoch in range(500):
    # 前向传播，计算预测值和损失
    y_pred = w * x + b
    loss = ((y_pred-y_true)**2).mean()

    # 求导，计算梯度
    dw = ((y_pred-y_true)*2*x).mean()
    db = ((y_pred-y_true)*2).mean()

    # 参数更新
    w -= lr * dw
    b -= lr * db

    if epoch % 20 == 0:
        print(f"第{epoch}轮，loss值为: {loss:.4f}")

print(f"最终的w为：{w},b为：{b}")