"""
设置requires_grad之后，表示需要跟踪张量的计算过程现在，我们手里已经有了开启追踪的参数 w 和 b，同时手
里有一大把同学的真实数据（特征 x）：
x = torch.tensor([2.0, 5.0, 1.0, 8.0, 3.0])

为了知道我们瞎猜的 w=1.0 和 b=0.0 到底有多离谱，我们接下来的代码应该先算什么？

（1）“计算一下预测值有多大”，这就是“前向传播”
（2）“计算预测值和真实值之间的误差有多大”，这就是“计算损失”

因为对叶子张量设置了requires_grad为True。那么后面每个样本在计算预测值和损失时，所有叶子张量参与的计算和
结果都会被保存几轮下来，存在最终的backward节点的grad_fn中
"""


import torch

# 1. 准备数据和参数
x = torch.tensor([2.0, 5.0, 1.0, 8.0, 3.0])
y = torch.tensor([70.0, 85.0, 60.0, 95.0, 72.0]) # 这是真实分数

w = torch.tensor(1.0, requires_grad=True) 
# 因为y的数量和x的数量对应可以说明x是多个样本的向量，而不是一个样本多个特征；
# w是标量，因为权重的数量和特征数量一一对应，也能说明x是多个样本的向量

b = torch.tensor(0.0, requires_grad=True)



# 2. 计算预测值（前向传播）
y_pred = w * x + b
print(y_pred)
# 输出 tensor([2., 5., 1., 8., 3.], grad_fn=<AddBackward0>) 。
# 后面多了一个小尾巴：grad_fn=<AddBackward0>，这就是 PyTorch 正在做笔记的证明！

print(y_pred[0])
# 输出 tensor(2., grad_fn=<SelectBackward0>)




# 3. 计算损失
loss = ((y_pred - y)**2).mean()
"""
⚠️ 注意：如果不写.mean()得到最终是5个数字的向量。

默认情况下，.backward() 只能在一个标量（只有一个数字的张量）上调用。

如果你强行对一个向量调求导，它会直接给你甩出一个经典的报错：
RuntimeError: grad can be implicitly created only for scalar outputs
（翻译：我只能给标量算导数，你给我一串数字，我不知道该听谁的！）

因为对一个向量求导，机器不知道该算谁对叶子张量的导数，存在冲突！

所以，这也是为什么深度学习中损失函数最后都习惯加一个 .sum() 或者 .mean()的原因
"""



# 4. 反向传播计算梯度
# 现在 loss 已经是一个带有 grad_fn 的标量了，我们只要敲下一行神迹般的代码：
loss.backward()
# 当运行这行代码之后，机器就会计算Loss对所有叶子张量的导数