import numpy as np

# ============ 数据 ============
# 3个样本，4个特征
X = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [0.5, 1.5, 2.5, 3.5],
    [2.0, 0.5, 1.0, 0.8]
])  # 形状 [3, 4]

# ============ 参数 ============
# 第1层：4个输入 → 5个隐藏节点
W1 = np.random.randn(4, 5) * 0.1  # 形状 [4, 5]
b1 = np.zeros(5)                   # 形状 [5]

# 第2层：5个隐藏节点 → 3个类别
W2 = np.random.randn(5, 3) * 0.1  # 形状 [5, 3]
b2 = np.zeros(3)                   # 形状 [3]

# ============ 激活函数 ============
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))  # 减最大值防溢出
    return e / e.sum(axis=1, keepdims=True)

# ============ 前向传播 ============
# 第1层
Z1 = X @ W1 + b1   # [3,4] @ [4,5] = [3,5]
A1 = relu(Z1)       # [3,5]

# 第2层
Z2 = A1 @ W2 + b2  # [3,5] @ [5,3] = [3,3]
A2 = softmax(Z2)    # [3,3]  ← 每行是一个样本的类别概率

print("Z1 形状:", Z1.shape)  # (3, 5)
print("A1 形状:", A1.shape)  # (3, 5)
print("Z2 形状:", Z2.shape)  # (3, 3)
print("A2 形状:", A2.shape)  # (3, 3)
print()
print("预测概率：\n", A2.round(3))
print("每行之和：", A2.sum(axis=1))  # 应该全是 1.0

# ============ 损失计算 ============
labels = np.array([0, 1, 2])  # 3个样本的真实类别索引

# 用索引取出每个样本对应正确类别的概率
correct_probs = A2[np.arange(3), labels]  # [0.xx, 0.xx, 0.xx]
loss = -np.log(correct_probs).mean()

print()
print("正确类别的概率：", correct_probs.round(3))
print("交叉熵损失：", round(loss, 4))