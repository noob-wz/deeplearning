"""
因为每一个权重和每个样本的特征都是一一对应的，通常来说，我们假设事物之间最简单的规律就是一个线性模型。

但是，真实世界的规律即使是线性规律，但也不可能找到一个完美的线性模型，因为每个样本多少存在一点误差，那么
人类就无法手动计算“最不差”的那条线的权重的偏置是多少。

因此，就有了机器学习，让机器代替人类去寻找规律，通过一次次的权重调整来使得最终的误差最小，那么如何记录误
差达到最小，方法就是：“给参数打上追踪标签”，即requires_grad，表示需要计算梯度。目的是让机器知道该怎么去
调整权重和偏差，几何上就是Loss曲线在某组参数下的切线斜率，从而知道是该上坡还是下坡。
"""

import torch

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)

print(w.requires_grad)
# 结果: True (说明 PyTorch 已经在盯着它了)

w_normal = torch.tensor(1.0)
print(w_normal.requires_grad)
# 结果: False (普通数字，不被追踪)

# -------------------------------------------
# 错误示范：如果没有 requires_grad，之后强行让 PyTorch 去算误差对 w 的变化率（求导），会发生什么？

w_bad = torch.tensor(1.0)
# 假设我们算出了某个误差 loss，然后命令它求导 (backward)
loss = (w_bad * 2.0 - 3.0) ** 2
loss.backward() 

# 此时运行会直接报错：
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# 原因：不是所有的张量都需要计算梯度的，所以默认的requires_grad是False