"""
前面已经解决了单个样本多个特征的问题，通过把他们打包成向量，运用torch.dot()，代码非常简洁

但现实里通常里有非常多的样本，假设样本有10000个，那么常规的思路可能使用for循环依次对单个样本进行点积运算，
问题是这样的速度非常慢。好的解决办法是：

    把x向量变成矩阵，充分发挥GPU的并行能力，让多样本并行计算
"""

import torch
import time

# 模拟 10000 个学生，每个学生 3 个特征。我们生成一个 10000行3列 的矩阵 X
X_big = torch.randn(10000, 3) 
# W 依然是 3 个权重
W = torch.tensor([0.6, 0.3, 0.1])

# --- 笨办法：for 循环 ---
t0 = time.time()
for i in range(X_big.shape[0]):
    torch.dot(W, X_big[i])

print(f"for 循环耗时：{time.time() - t0:.5f}秒")

# --- 你的解法：矩阵乘法 (@) ---
t0 = time.time()
y_preds = X_big @ W # 等价于 y_preds = W @ X_big.T，最终y_preds都是一维向量
print(f"矩阵乘法耗时: {time.time() - t0:.5f} 秒")

