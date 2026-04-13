import torch

# 自定义生成线性回归数据集函数
def synthetic_data(w, b, num_examples):
    
    # 生成符合正态分布的数据集
    X = torch.normal(0,1,(num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)

# 在代码中通常要注意维度对齐，如果自行对齐，通常特征系数要写成一个矩阵
# true_w = torch.tensor([[2],[-3.4]])
# 这种情况下 X @ w生成的维度是(num_examples, 1)

# 不过由于这里的生成数据集的函数是针对单层单神经元的线性层，所以 w 的形状在输出特征的长度是1，形状正常来说是(输入特征数, 1)
# 一般习惯于写成一维向量
true_w = torch.tensor([2, -3.4])
# 这种情况下 X @ w生成的维度是(num_examples,)

# 偏置的维度一般是一维向量，长度等于输出特征的数量。进行加法是在各个神经元基础上加，所以是一维向量或者写成一个标量
# 因为前面的w写的是向量，考虑到维度匹配，b要写成标量或者包含一个元素的向量。如果w写的是矩阵，那么b就要写成向量或者(1,n)矩阵
true_b = torch.tensor([4.2])

features, labels = synthetic_data(true_w, true_b, 1000)


print(features, labels)