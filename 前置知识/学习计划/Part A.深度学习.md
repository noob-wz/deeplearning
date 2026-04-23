# Part A: 深度学习地基（Week 1–7）

## 本阶段定位

**衔接补零期**：补零期你学会了用 Python 处理数据、用 NumPy 做数组运算、用 Matplotlib 画图、建立了"导数/梯度"的几何直觉。工具手感有了，问题变成：如何构造一个函数，让它能从数据中"学"出规律？这就是神经网络要做的事情。

**本阶段目标**：从零理解神经网络的工作原理（神经元、前向传播、梯度下降、反向传播），用 NumPy 手写实现简单网络,然后过渡到 PyTorch，最终能用 PyTorch 搭建 MLP 并训练多特征回归任务，能通过 Loss 曲线诊断训练问题。

**为什么先用 NumPy 再用 PyTorch**：PyTorch 虽然方便，但它把反向传播等细节自动化了。如果一开始就用 PyTorch，你会把神经网络当成"黑盒 API"。先用 NumPy 手写一遍前向传播和梯度下降，你会真正理解"参数更新"在做什么，后面用 PyTorch 时才不会迷糊，也能看懂之后 GNN 和 PINN 里复杂的自定义计算。

**为什么这个阶段必须稳**：Part A 是所有后续内容的地基。GNN 是神经网络的特殊结构，PINN 是训练目标的特殊定义。如果你在 Part A 对基础概念只是"似懂非懂"，后面所有内容都会变成死记硬背。不要赶进度，每个知识点都要做到"能用自己的话讲清楚"再往下走。

**完成标准（进入 Part B 之前必须达到）**：
- 能用 NumPy 手写实现 2 层 MLP 的前向传播和反向传播
- 能用 PyTorch 的 `nn.Module` 搭建 MLP，用 Adam 优化器训练一个多特征回归任务
- 能看懂 Loss 曲线，诊断出训练是否收敛、是否过拟合
- 能向别人用中文讲清楚：什么是神经元、梯度下降、反向传播

---

---

## Week 1: 神经元与前向传播

**衔接**：补零期你已经能用 NumPy 做矩阵乘法和向量运算。这周你会发现，一个"神经元"本质上就是一次向量点积加一个激活函数——你已经有工具了，现在只是给它一个名字和结构。

**本周目标**：理解单个神经元和单层网络在做什么，能用 NumPy 实现完整的前向传播（输入 → 参数 → 激活 → 输出 → Loss）。**本周不涉及训练**，只涉及"给定参数，算出输出"。训练是下周的事。

**主要资源**：
- 吴恩达 Coursera Deep Learning Specialization Course 1 Week 1（视频约 2 小时，必看）
- 3Blue1Brown "But what is a Neural Network?"（YouTube，约 20 分钟，可视化最好的入门材料）

---

### Day 1 | 神经元：把熟悉的东西重新命名

**为什么这么设计这一天**：你已经会写 `y = w * x + b`，这就是一个神经元的核心。这天的任务不是学新东西，而是把"你已经会的"放进神经网络的术语体系。这是一个温和的入口，不要因此轻视。

**理论任务**（约 30 分钟）：
- ✅ 看吴恩达 Course 1 Week 1 的前 4 个视频
- ✅ 在笔记本上画一张图：输入 x → (乘以权重 w，加上偏置 b) → 得到 z → 通过激活函数 σ → 得到输出 a
- ✅ 用自己的话写一句话："神经网络为什么要用多个神经元而不是一个？"

✅ **实践任务**（约 45 分钟）：创建文件 `week01/day01_single_neuron.py`

任务描述：实现一个单神经元的前向传播（暂时不加激活函数）

```
函数签名：forward_single(x, w, b) -> float
输入：
  x: numpy 数组，shape=(3,)，代表3个特征
  w: numpy 数组，shape=(3,)，代表3个权重
  b: float，代表偏置
计算：
  z = w · x + b（向量点积加标量）
  返回 z
约束：必须用 np.dot 实现点积
```

**验证代码**（写在文件末尾）：
```python
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, -1.0, 2.0])
b = 0.5
z = forward_single(x, w, b)
print(z)  # 应输出 5.0
# 手算：0.5*1 + (-1.0)*2 + 2.0*3 + 0.5 = 0.5 - 2 + 6 + 0.5 = 5.0
```

**验收标准**：函数对上述输入精确返回 5.0。

---

### Day 2 | 激活函数

**为什么要加激活函数**：没有激活函数，多层网络堆叠起来还是一个线性函数（两个线性函数的组合仍是线性）。激活函数引入非线性，让网络能拟合曲线而不只是直线。这对 FEM 数据非常重要——应力应变关系在塑性段就是非线性的，没有激活函数的网络根本表达不了。

**理论任务**（约 30 分钟）：
- ✅ 看吴恩达 Course 1 Week 3 的激活函数部分（约 20 分钟）
- ✅ 在笔记本上画出三条曲线的草图：sigmoid、tanh、ReLU（形状大致对即可）
- ✅ 用自己的话回答三个问题：
  1. 为什么 ReLU 最常用？
  2. 什么情况下不该用 Sigmoid？
  3. ReLU(-5) = ?  ReLU(3) = ?  Sigmoid(0) = ?（自测）

**参考答案**：
1. **为什么 ReLU 最常用？**
   > 计算快（只是 max(0, z)）；正区间梯度恒为 1 不消失；比 Sigmoid 能训练更深的网络

2. **什么情况下不该用 Sigmoid？**
   > 深层网络不要全用 Sigmoid，会梯度消失（Sigmoid 的导数最大只有 0.25，每层乘一下指数级衰减）

3. **ReLU(-5) = 0, ReLU(3) = 3, Sigmoid(0) = 0.5**

**（新增补充）ReLU 的一个坑：死神经元（dying ReLU）**

这个问题先了解，后面 Part B 讲 PINN 时会再回来：
- 如果某个神经元因为权重初始化或训练过程中的某次大梯度更新，导致它对所有训练数据的输出都是负数
- 那么 ReLU 输出永远是 0，梯度也永远是 0——这个神经元"死了"，再也不会被更新
- 实际训练中如果看到"隐藏层很大一部分神经元输出都是 0"，就是发生了这个
- 解决方法：
  - 用 He 初始化（Week 3 会讲）而不是简单的小随机数
  - 降低学习率
  - 换成 Leaky ReLU（`max(0.01*z, z)`，负区间也有小梯度）

**为什么这件事对你方向重要**（记住这条线索，Part B 会用到）：
- Part B 讲 PINN 时你会发现论文用的是 Tanh 激活（不是 ReLU）
- 原因之一：PINN 需要对输入求高阶导数，ReLU 的二阶导数处处为 0（除了原点），高阶导数完全消失
- 原因之二：ReLU 在 0 点不可导（左右导数不一致），PINN 里容易出数值问题
- 但纯数据驱动的 GNN（Part C）用 ReLU 没问题

✅ **实践任务**(约 60 分钟）：创建文件 `week01/day02_activations.py`

```
函数1：sigmoid(z) 
  输入：numpy 数组
  输出：同 shape 的 numpy 数组
  公式：1 / (1 + exp(-z))

函数2：tanh_fn(z)
  同上，公式：(exp(z) - exp(-z)) / (exp(z) + exp(-z))
  也可用 np.tanh 验证

函数3：relu(z)
  同上，用 np.maximum(0, z)

画图任务：在同一张图上画三条曲线，x 范围 -5 到 5，200 个点
  保存为 activations.png
  要求：有 legend、xlabel='z'、ylabel='activation'、title='Three Activation Functions'
```

**验证代码**：
```python
z = np.array([-2.0, 0.0, 2.0])
print(sigmoid(z))   # 应约为 [0.119, 0.500, 0.881]
print(tanh_fn(z))   # 应约为 [-0.964, 0.000, 0.964]
print(relu(z))      # 应为 [0.0, 0.0, 2.0]
```

**验收标准**：三个函数输出精确到小数点后 3 位；图能生成且四个要素齐全（三条线/legend/坐标轴标签/标题）。

---

### Day 3 | 从单神经元到"一层"

**衔接**：Day 1 学了单神经元的线性部分，Day 2 学了激活函数，今天把它们组合起来，并扩展到"一层有多个神经元"的情况。

**实践任务**（约 90 分钟）：创建文件 `week01/day03_forward_layer.py`

✅ **任务 1**：完整的单神经元前向传播（带激活）
```
函数签名：forward(x, w, b, activation_fn) -> float
  x: shape=(n_features,)
  w: shape=(n_features,)
  b: float
  activation_fn: 函数对象（sigmoid, relu 等）
返回：a = activation_fn(w·x + b)
```

✅ **任务 2**：扩展到"一层多神经元"
```
函数签名：forward_layer(x, W, b, activation_fn) -> np.ndarray
  x: shape=(n_in,)           输入特征向量
  W: shape=(n_out, n_in)     权重矩阵，每行是一个神经元的权重
  b: shape=(n_out,)          偏置向量
返回：a shape=(n_out,)
计算：z = W @ x + b; a = activation_fn(z)
```

**验证代码**：
```python
x = np.array([1.0, 2.0])
W = np.array([[0.5, 1.0], [-0.5, 2.0], [1.0, 1.0]])  # 3个神经元，每个2个输入
b = np.array([0.0, 0.5, -1.0])
a = forward_layer(x, W, b, relu)
# 手算验证：
# z[0] = 0.5*1 + 1.0*2 + 0.0 = 2.5, relu = 2.5
# z[1] = -0.5*1 + 2.0*2 + 0.5 = 4.0, relu = 4.0
# z[2] = 1.0*1 + 1.0*2 - 1.0 = 2.0, relu = 2.0
print(a)  # 应输出 [2.5, 4.0, 2.0]
```

**验收标准**：两个函数对上述测试用例精确返回。

---

### Day 4 | 损失函数（MSE）

**为什么是 MSE**：你的目标方向是回归任务——预测位移场、应变场、应力场。回归任务的标准损失函数是 MSE。平方项放大大误差的惩罚，这对物理场预测是合理的——宁可整体平均误差大一点，也不要有局部点误差巨大（后者在 FEM 后处理里会被立刻看到）。

**理论任务**（约 20 分钟）：  
- ✅ 在笔记上写出 MSE 公式：`MSE = (1/N) * Σ (y_pred_i - y_true_i)²`
- ✅ 用自己的话回答：为什么不用"误差绝对值的平均"（MAE）而用平方？
  - 参考：平方处处可导（MAE 在 0 点不可导，不方便梯度下降）；平方对大误差惩罚更重

✅ **实践任务**（约 40 分钟）：创建文件 `week01/day04_mse.py`

```
函数签名：mse_loss(y_pred, y_true) -> float
  y_pred: numpy 数组，shape=(N,)
  y_true: numpy 数组，shape=(N,)
约束：手写公式，不要直接用 np.mean 写成一行，写成 sum / N 的形式便于理解
```

**验证代码**：
```python
y_true = np.array([3.0])
y_pred = np.array([3.5])
print(mse_loss(y_pred, y_true))  # 应为 0.25（(0.5)² = 0.25）

y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 1.9, 3.2])
# 手算：((0.1)² + (0.1)² + (0.2)²) / 3 = (0.01 + 0.01 + 0.04) / 3 = 0.06/3 = 0.02
print(mse_loss(y_pred, y_true))  # 应为 0.02
```

**验收标准**：两个测试用例输出精确到小数点后 4 位（0.2500 和 0.0200）。

---

### Day 5–6（周末）| 整合练习：第一个完整的"预测流程"

**本周的闭环**：现在你有了前向传播（Day 3）和损失函数（Day 4），可以组合成一个完整的"给定参数 → 预测 → 计算损失"的流程。虽然还不能训练（下周的事），但能直观感受"参数好坏"对 Loss 的影响。

✅ **任务**：创建文件 `week01/weekend_forward_pipeline.py`

**Step 1**：生成合成数据（线性关系 + 少量噪声）
```python
np.random.seed(42)
N = 50
X = np.random.randn(N, 2)  # 50个样本，每个2个特征
y_true = 2.0*X[:,0] + 3.0*X[:,1] - 1.0 + np.random.randn(N)*0.1
# 真实关系：y = 2*x1 + 3*x2 - 1，加少量噪声
```

**Step 2**：用随机参数做前向传播
```python
np.random.seed(0)
w_random = np.random.randn(2)
b_random = 0.0
y_pred_random = X @ w_random + b_random  # 不加激活（线性回归）
loss_random = mse_loss(y_pred_random, y_true)
print(f"随机参数下 Loss: {loss_random:.4f}")
```

**Step 3**：用"接近正确"的参数
```python
w_good = np.array([2.0, 3.0])
b_good = -1.0
y_pred_good = X @ w_good + b_good
loss_good = mse_loss(y_pred_good, y_true)
print(f"正确参数下 Loss: {loss_good:.4f}")
```

**Step 4**：用你在 P3 周写的 `plot_true_vs_pred` 函数画两张图
- `random_params_tvp.png`：随机参数下的散点（应该乱）
- `good_params_tvp.png`：正确参数下的散点（应贴近对角线）

**验收标准**：
- 随机参数下 Loss 通常在几十以上（具体值取决于随机种子）
- 正确参数下 Loss < 0.02（只剩噪声方差）
- 两张图对比鲜明

---

### Week 1 完成标准（进入 Week 2 前）

- [✅] 能参考 Day 3 的文档，45 分钟内独立写出 `forward_layer(x, W, b, activation_fn)` 函数
- [✅] 能不看参考，15 分钟内独立写出 `mse_loss(y_pred, y_true)` 函数
- [✅] 能用自己的话向别人解释：什么是前向传播、为什么要加激活函数
- [✅] 理解为什么 sigmoid(0)=0.5、ReLU(-5)=0
- [✅] 知道 ReLU 有"死神经元"问题（知道即可，后面会再讲）

**如果没达到上述标准，不要进入 Week 2**。本周内容是地基，重复写一遍代码、重看一遍视频、再练一练比"硬推进"更有价值。

---

---

## Week 2: 梯度下降——让模型"学习"

**衔接**：Week 1 你能做预测，但预测质量完全取决于你填进去的参数。如果参数随机，预测就乱；如果参数"正确"，预测就准。现在的问题是：我们不知道"正确参数"是什么，怎么让模型自动调整参数逼近正确？这就是梯度下降要解决的问题。

**本周目标**：理解梯度下降算法的数学和直觉，能用 NumPy 手写实现线性回归的完整训练循环（前向 → 计算梯度 → 更新参数 → 重复）。

**主要资源**：
- 吴恩达 Course 1 Week 2（梯度下降部分，约 1.5 小时）
- 3Blue1Brown "Gradient descent, how neural networks learn"（YouTube，约 20 分钟，必看）

---

### Day 1 | 梯度下降的直觉

**衔接补零期 P3 的数学直觉**：你已经理解"导数是斜率、梯度指向上升最快方向"。今天把这个直觉用到神经网络训练上。

**理论任务**（约 1 小时）：
- ✅ 看 3Blue1Brown "Gradient descent, how neural networks learn"
- ✅ 看吴恩达 Course 1 Week 2 的"Gradient Descent"视频
- ✅ 在笔记本上画一张图：一个 U 形的 Loss 曲线，起点在山坡上，画出该点的梯度方向（指向上）和参数更新方向（反方向）
- ✅ 用自己的话回答三个问题并写下来：
  1. "梯度下降是在做什么"（一句话）
  2. 学习率太大会怎样？
  3. 学习率太小会怎样？

参考答案：
1. 沿着让 Loss 下降最快的方向（负梯度方向），一步一步调整参数
2. 太大会跨过最低点，可能震荡甚至发散
3. 太小收敛极慢，需要大量迭代

✅ **实践任务**（约 40 分钟）：创建文件 `week02/day01_1d_gradient.py`

从最简单的一维例子开始，把"梯度下降"这个抽象概念具象化。

```
任务：用梯度下降找 f(w) = (w - 3)² 的最小值
数学上已知：df/dw = 2*(w - 3)，最小值在 w=3 处 f(3)=0

代码结构：
  w = 0.0        # 起点
  lr = 0.1       # 学习率
  for step in range(20):
      f = (w - 3) ** 2
      grad = 2 * (w - 3)   # 计算梯度（已知公式）
      w = w - lr * grad    # 更新参数
      print(f"step {step}: w={w:.4f}, f(w)={f:.6f}")
```

**验收标准**：
- 第 20 步后 w > 2.9（逐渐接近 3.0）
- f(w) 单调下降，最终应约为 0.013 左右
- 能说清楚：这就是"参数从 0 慢慢调整到 3"的过程

---

### Day 2 | 线性回归的梯度推导

**为什么要自己推一次**：现在你还没用 PyTorch 的自动求导。手动把"MSE 对 w 和 b 的导数"推出来，你会真正理解"梯度从哪里来"。这个推导在第二阶段的 PINN 部分会再派上用场（那时要手动算二阶导数）。

**理论任务**（约 1.5 小时，必须在纸上写，不要只用眼看）：

假设有 N 个样本，对每个样本 i：
- ✅ 预测：`y_pred_i = w * x_i + b`（先考虑单特征、无激活，之后 Week 3 再推多层）
- ✅ 单样本 Loss：`L_i = (y_pred_i - y_true_i)²`
- ✅ 总 Loss：`L = (1/N) * Σ L_i`

用链式法则推导 `∂L/∂w` 和 `∂L/∂b`。

✅ **推导步骤**（在纸上写一遍）：
```
∂L_i/∂y_pred_i = 2*(y_pred_i - y_true_i)   （二次函数的导数）
∂y_pred_i/∂w = x_i                          （w 的系数）
∂y_pred_i/∂b = 1                            （b 的系数）

应用链式法则：
∂L_i/∂w = ∂L_i/∂y_pred_i × ∂y_pred_i/∂w = 2*(y_pred_i - y_true_i)*x_i
∂L_i/∂b = ∂L_i/∂y_pred_i × ∂y_pred_i/∂b = 2*(y_pred_i - y_true_i)

对 N 个样本求平均：
∂L/∂w = (2/N) * Σ (y_pred_i - y_true_i) * x_i
∂L/∂b = (2/N) * Σ (y_pred_i - y_true_i)
```

**验收标准**：能在纸上独立写出这两个最终公式，并解释中间的链式法则每一步。

---

### Day 3–4 | 完整训练循环

✅ **实践任务**（每天约 1.5 小时）：创建文件 `week02/day34_training_loop.py`

```python
# Step 1: 生成数据
np.random.seed(42)
N = 100
x = np.random.randn(N)
y_true = 2.5 * x + 1.0 + np.random.randn(N) * 0.1  # 真实关系：y = 2.5x + 1

# Step 2: 初始化参数
w = 0.0
b = 0.0
lr = 0.1
losses = []

# Step 3: 训练循环
for epoch in range(200):
    # 前向传播
    y_pred = w * x + b
    loss = np.mean((y_pred - y_true) ** 2)
    losses.append(loss)
    
    # 计算梯度（用昨天推出的公式）
    dw = (2 / N) * np.sum((y_pred - y_true) * x)
    db = (2 / N) * np.sum(y_pred - y_true)
    
    # 更新参数
    w = w - lr * dw
    b = b - lr * db
    
    if epoch % 20 == 0:
        print(f"epoch {epoch}: w={w:.4f}, b={b:.4f}, loss={loss:.6f}")

# Step 4: 画 Loss 曲线
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')

# Step 5: 打印最终参数
print(f"\n最终 w = {w:.4f}（期望约 2.5）")
print(f"最终 b = {b:.4f}（期望约 1.0）")
```

> **关于"epoch"这个词**（本 Part 后面会反复出现）：
> 
> 这里每一轮 `for epoch in range(200)` 代表"整个数据集过一遍"。因为本例用的是 **full batch**（一次把所有 100 个样本全部用于计算梯度），所以 1 epoch = 1 次参数更新。
> 
> Week 6 你会学到 mini-batch，那时 1 epoch 里会有多次参数更新（每个 batch 更新一次）。**先记住**："epoch = 整个数据集过一遍"。

**验收标准**：
- 最终 w ∈ [2.45, 2.55]
- 最终 b ∈ [0.95, 1.05]
- Loss 曲线单调下降，最终收敛到约 0.01 左右（噪声方差级别）
- 能看懂代码每一行在做什么，特别是梯度计算部分

---

### Day 5–6（周末）| 学习率实验

✅ **任务**：创建文件 `week02/weekend_lr_experiment.py`

用 Day 3-4 的同样数据和训练循环，测试 4 种学习率：
- lr = 0.001（太小）
- lr = 0.01（偏小）
- lr = 0.1（合适，Day 3-4 用过的）
- lr = 1.5（太大）

每种学习率独立跑 200 epoch，保存 `losses` 列表。

**可视化**：在同一张图上画 4 条 Loss 曲线，y 轴用对数刻度（`plt.yscale('log')`），有 legend 标明每条曲线对应的 lr。

**文字总结**（写在脚本末尾作为注释）：3 句话回答
1. lr=0.001 的表现如何？问题在哪？
2. lr=1.5 的表现如何？问题在哪？
3. 最合适的学习率是哪个？为什么？

**验收标准**：
- 图能生成，4 条曲线颜色不同，图例完整
- lr=1.5 的曲线应当震荡或 Loss 不降反升（如果训练直接 NaN 了，在图里说明）
- lr=0.001 的曲线下降非常慢，200 epoch 远未收敛

---

### Week 2 完成标准

- [✅] 能在纸上独立推导线性回归的 `∂L/∂w` 和 `∂L/∂b` 公式，解释每一步
- [✅] 能在 30 分钟内不看参考写出完整的 NumPy 训练循环代码
- [✅] 能用自己的话解释"梯度下降是如何让模型学习的"，完全不用公式
- [✅] 能通过 Loss 曲线判断：训练是否收敛、学习率是否合适

---

---

## Week 3: 多层网络与反向传播

**衔接**：Week 2 的线性回归只能拟合直线。但物理世界的关系常常是非线性的——应力应变在塑性段是非线性的，位移场受到几何形状的影响也是非线性的。要表达非线性关系，需要多层网络 + 激活函数。多层的参数更新需要反向传播。

**本周目标**：理解为什么单层网络有表达力局限，理解反向传播的工作机制，能用 NumPy 实现 2 层 MLP 并成功训练一个非线性回归任务（拟合 sin 函数）。

**特别说明**：本周你会第一次感到"数学上有点跟不上"。反向传播的完整矩阵形式推导对新手是反人类的，你**不需要**能独立推导出完整公式。你需要做到的是：看懂已写好的反向传播代码每一行在算什么。这是本周最重要的心态调整。

**主要资源**：
- 吴恩达 Course 1 Week 3-4（浅层 + 深层网络，约 3 小时）
- 3Blue1Brown "What is backpropagation really doing?"（YouTube，约 14 分钟，必看）
- 3Blue1Brown "Backpropagation calculus"（YouTube，约 10 分钟，可选加深）

---

### Day 1 | 从 1 层到 2 层：为什么需要多层

**理论任务**（约 1.5 小时）：
- ✅ 看吴恩达 Course 1 Week 3 的前 3 个视频
- ✅ 在纸上画出 2 层网络的结构图，**严格标注每一步的 shape**

具体例子（以下就是你要画的图内容）：
```
输入：x shape=(3,)           ← 3个特征

隐藏层（4个神经元，ReLU激活）：
  W1 shape=(4, 3)            ← 4行3列，每行是一个神经元的权重
  b1 shape=(4,)              
  z1 = W1 @ x + b1, shape=(4,)
  a1 = ReLU(z1), shape=(4,)  ← 隐藏层输出

输出层（1个神经元，无激活因为是回归）：
  W2 shape=(1, 4)
  b2 shape=(1,)
  z2 = W2 @ a1 + b2, shape=(1,)
  y = z2, shape=(1,)         ← 最终预测
```

**用自己的话回答**：
- 为什么回归任务的最后一层不加激活函数？
  - 参考：激活函数（如 ReLU、sigmoid）会限制输出范围，但物理量（应力、位移）可能是任意值，最后一层不加激活才能输出无限制的实数

✅ **实践任务**（约 1 小时）：创建文件 `week03/day01_mlp_forward.py`

实现 2 层 MLP 的前向传播，输入 batch 数据（不是单个样本）。

```
函数签名：mlp_forward(X, W1, b1, W2, b2) -> y_pred
  X: shape=(N, n_in)        N个样本，每个 n_in 个特征
  W1: shape=(n_hid, n_in)
  b1: shape=(n_hid,)
  W2: shape=(n_out, n_hid)
  b2: shape=(n_out,)
返回：y_pred shape=(N, n_out)

实现（伪代码）：
  Z1 = X @ W1.T + b1        # shape=(N, n_hid)
  A1 = np.maximum(0, Z1)    # ReLU
  Z2 = A1 @ W2.T + b2       # shape=(N, n_out)
  y_pred = Z2               # 无激活
  return y_pred
```

**验证代码**：
```python
np.random.seed(0)
X = np.random.randn(5, 3)     # 5个样本，3个特征
W1 = np.random.randn(4, 3)
b1 = np.zeros(4)
W2 = np.random.randn(1, 4)
b2 = np.zeros(1)

y_pred = mlp_forward(X, W1, b1, W2, b2)
print(y_pred.shape)  # 应输出 (5, 1)
```

**验收标准**：输出 shape 正确；代码中的每一步能说清楚维度为什么是那样。

---

### Day 2 | 反向传播的直觉（不做完整推导）

**重要心态调整**：反向传播的矩阵形式推导对新手很痛苦。本节课的目标不是让你能独立推导，而是让你做到两件事：
1. **从概念上**理解反向传播在做什么
2. **从代码上**看懂每一行反向传播代码在算哪个梯度

**理论任务**（约 1.5 小时）：
- ✅ 看 3Blue1Brown "What is backpropagation really doing?"
- ✅ 在笔记上画一张"误差信号回传"示意图
- ✅ 用自己的话回答：
  1. 反向传播要算的是什么？（Loss 对每一个参数的偏导数，即梯度）
  2. 为什么叫"反向"？（从输出层往输入层方向，一层层往回算梯度）
  3. 链式法则在其中起什么作用？（每一层的梯度 = 后一层传回的梯度 × 本层的局部梯度）

✅ **实践任务**（约 1 小时）：创建文件 `week03/day02_backprop_reading.py`

这是一个"读代码"练习。下面的反向传播代码已写好，你的任务是**给每一行写中文注释**，说明这行算的是什么。

```python
# 假设 2 层 MLP：
# Z1 = X @ W1.T + b1
# A1 = relu(Z1)
# Z2 = A1 @ W2.T + b2
# y_pred = Z2
# L = mean((y_pred - y_true)²)
#
# 正向变量都已算好：X, Z1, A1, Z2, y_pred, y_true
# 下面是反向传播：

N = X.shape[0]

# 【你来写注释：这一行在算什么梯度？】
dL_dypred = 2 * (y_pred - y_true) / N

# 【注释】
dL_dW2 = dL_dypred.T @ A1

# 【注释】
dL_db2 = dL_dypred.sum(axis=0)

# 【注释】
dL_dA1 = dL_dypred @ W2

# 【注释】
dL_dZ1 = dL_dA1 * (Z1 > 0).astype(float)

# 【注释】
dL_dW1 = dL_dZ1.T @ X

# 【注释】
dL_db1 = dL_dZ1.sum(axis=0)
```

**参考答案**（写完你自己的注释后再对比）：
- `dL_dypred`：Loss 对网络最终输出 y_pred 的梯度
- `dL_dW2`：Loss 对第二层权重矩阵 W2 的梯度
- `dL_db2`：Loss 对第二层偏置 b2 的梯度
- `dL_dA1`：Loss 对第一层激活输出 A1 的梯度（从后一层传回）
- `dL_dZ1`：Loss 对第一层线性输出 Z1 的梯度（A1 再经过 ReLU 反传）
- `dL_dW1`：Loss 对第一层权重矩阵 W1 的梯度
- `dL_db1`：Loss 对第一层偏置 b1 的梯度

**验收标准**：能给每行写出合理的中文注释，能说出"梯度是从后往前传的"。

---

### Day 2.5（新增，约 30 分钟） | 为什么权重不能全初始化为 0？He 初始化是什么？

**这一节是 Day 3-4 代码的前置铺垫**。Day 3-4 的代码里有这样一行：

```python
W1 = np.random.randn(n_hid, n_in) * np.sqrt(2.0 / n_in)
```

这个 `np.sqrt(2.0 / n_in)` 不是随便乘的。搞清楚为什么。

---

**思考实验 1：如果权重全初始化为 0 会怎样？**

假设 W1 = 全 0，前向传播时：
- Z1 = X @ W1.T + b1 = 0（所有隐藏层输出都是 0）
- A1 = ReLU(0) = 0
- 所有隐藏神经元输出完全相同——实际上网络退化成"一个神经元"

反向传播时，所有隐藏神经元梯度也完全相同——永远没办法学到不同的特征。这叫**对称性问题**（symmetry breaking）。

**结论 1**：权重必须随机初始化，让每个神经元起点不同。

**思考实验 2：如果权重全设为随机大数（比如 randn × 10）会怎样？**

前向传播时 Z1 会变得很大（可能上百），ReLU 激活后也很大。下一层的 Z2 更大——**激活值爆炸**。梯度反向传播时也会爆炸。

**思考实验 3：如果权重很小（比如 randn × 0.01）会怎样？**

Z1 很小，ReLU 过滤掉一半变成 0。下一层再缩小一次，再过滤——**激活值逐层衰减到 0**。梯度也衰减到 0，网络学不动。

**结论 2**：初始化的**量级**很关键，既不能太大也不能太小。

---

**He 初始化是什么**（核心直觉）：

数学上可以推出一个合适的量级。对于 ReLU 激活，He 等人（论文作者）推出：
```
std = sqrt(2 / n_in)     其中 n_in 是该层输入维度
```

具体为什么是 `sqrt(2/n_in)` 不需要记——直觉是：输入维度越大（n_in 大），每个权重应该越小，这样累加后量级才合适。

**对你现在的任务**：
- 用 NumPy 手写时按照公式写即可（Day 3-4 就这么做）
- 后面用 PyTorch 时 `nn.Linear` **默认就用了类似的合理初始化**（用的是 Kaiming uniform，是 He 的变种），不需要你手动做
- 这是为什么你在 Week 4-5 用 `nn.Linear` 时"不用管初始化就能训练"——PyTorch 在背后帮你做了

**回头看 Week 1 Day 2 的"死神经元"问题**：
- 当时提到"死神经元"的一个解决方法是"用 He 初始化"
- 现在你明白了：因为 He 初始化保证了权重量级合适，神经元的初始输入不会过度偏向负区间，从源头降低了"死掉"的概率

**口头自测**：
1. 为什么权重不能全初始化为 0？
2. 为什么 `sqrt(2/n_in)` 里有 `n_in`？
3. 用 PyTorch `nn.Linear` 时你需要手动初始化吗？为什么？

**验收标准**：能回答以上三个问题。

---

### Day 3–4 | 用 NumPy 实现 2 层 MLP 训练（拟合 sin 函数）

✅ **实践任务**（每天约 2 小时）：创建文件 `week03/day34_mlp_sin.py`

**Step 1**：生成非线性数据
```python
np.random.seed(42)
N = 200
x = np.linspace(-np.pi, np.pi, N).reshape(-1, 1)  # shape=(200, 1)
y_true = np.sin(x) + np.random.randn(N, 1) * 0.05  # sin 波形 + 微量噪声
```

**Step 2**：完整训练代码（提供模板，你要能看懂并跑通）

```python
# 初始化参数（用 He 初始化，见 Day 2.5）
np.random.seed(0)
n_in, n_hid, n_out = 1, 32, 1
W1 = np.random.randn(n_hid, n_in) * np.sqrt(2.0 / n_in)   # He 初始化
b1 = np.zeros(n_hid)
W2 = np.random.randn(n_out, n_hid) * np.sqrt(2.0 / n_hid)  # He 初始化
b2 = np.zeros(n_out)

lr = 0.01
losses = []

for epoch in range(2000):
    # ===== 前向传播 =====
    Z1 = x @ W1.T + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2.T + b2
    y_pred = Z2
    
    loss = np.mean((y_pred - y_true) ** 2)
    losses.append(loss)
    
    # ===== 反向传播 =====
    N = x.shape[0]
    dL_dypred = 2 * (y_pred - y_true) / N
    dL_dW2 = dL_dypred.T @ A1
    dL_db2 = dL_dypred.sum(axis=0)
    dL_dA1 = dL_dypred @ W2
    dL_dZ1 = dL_dA1 * (Z1 > 0).astype(float)
    dL_dW1 = dL_dZ1.T @ x
    dL_db1 = dL_dZ1.sum(axis=0)
    
    # ===== 参数更新 =====
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    
    if epoch % 200 == 0:
        print(f"epoch {epoch}: loss={loss:.6f}")
```

**Step 3**：可视化
- 画 Loss 曲线（`loss_curve.png`）
- 画拟合曲线（`sin_fit.png`）：
  - x 轴是 x（排序后）
  - 蓝色散点：真实值 y_true
  - 红色实线：预测值 y_pred
  - 要有 legend、title

**验收标准**：
- 最终 Loss < 0.01
- 拟合曲线能较好地跟随 sin 波形（不完美也可以，看出大致形状即可）
- 关键理解点：单层线性模型拟合不出 sin，两层带 ReLU 的 MLP 可以——**这就是"非线性"的意义**

---

### Day 5–6（周末）| 隐藏层宽度实验 + 本周巩固

✅ **Day 5 任务**：创建 `week03/day5_width_experiment.py`

用 Day 3-4 的同样代码，尝试不同隐藏层宽度：`n_hid = 8, 32, 128`

对每个宽度：
- 训练 2000 epoch
- 保存最终的 y_pred 数组

画一张 3×1 的 subplot 图（`width_comparison.png`），每个 subplot 展示一个宽度下的拟合效果（散点 + 曲线）。

**观察任务**（写在脚本末尾作为注释）：
- ✅ 宽度 8 时，拟合曲线有哪些地方偏离真实 sin？
- ✅ 宽度 32 vs 128，你觉得哪个更好？为什么（训练时间？拟合质量？）？

**Day 6 任务**：本周巩固

- ✅ 对着反向传播代码（Day 2 的），不看参考能说出每一行在算什么
- ✅ 不看参考，从零写出 MLP 的 forward 代码（Day 1 的版本）
- ✅ 把本周写的所有代码整理进项目结构：`week03/` 目录下分 `day01/`, `day02/`, ...

---

### Week 3 完成标准

- [✅] 理解 2 层 MLP 的参数 shape 和前向传播流程
- [✅] 能看懂反向传播代码每一行在算什么（**不要求独立推导**）
- [✅] 能跑通 sin 函数拟合，看到非线性表达的意义
- [✅] 能通过拟合曲线和 Loss 曲线判断训练效果
- [✅] 能解释 He 初始化为什么是 `sqrt(2/n_in)`，为什么权重不能全初始化为 0
- [✅] 能用自己的话解释：为什么要多层？为什么要有激活函数？反向传播在做什么？

---

---

## Week 4: PyTorch 入门——Tensor 与基础操作

**衔接**：Week 1-3 你已经用 NumPy 手写了 MLP 和反向传播。你应该能深切体会到：手写反向传播容易出错，而且每换一个网络结构就要重新推导梯度。PyTorch 解决的就是这个痛点——用 autograd 自动算梯度。但在用 autograd 之前，你需要先熟悉 PyTorch 的基本数据结构：Tensor。

**本周目标**：掌握 PyTorch Tensor 的创建、索引、运算，能在 NumPy ndarray 和 PyTorch Tensor 之间自由转换，能用 PyTorch 重写 Week 3 的前向传播。

**本周不做**：autograd 和 nn.Module（下周做）。这周只打 Tensor 的基础。

**主要资源**：
- PyTorch 官方 Tutorial "Learn the Basics" 的前 3 节（https://pytorch.org/tutorials/beginner/basics/intro.html）
  - Tensors
  - Datasets & DataLoaders（这节的 DataLoader 部分下周再仔细看）
  - Transforms（跳过，和本方向无关）

---

### Day 1 | Tensor 基础创建与属性

**核心概念**：PyTorch Tensor ≈ NumPy ndarray + GPU 加速 + 自动求导支持（后者下周才启用）。

✅ **实践任务**（约 1.5 小时）：创建文件 `week04/day01_tensor_basics.py`

```python
import torch
import numpy as np

# Task 1: 多种方式创建 tensor
t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])     # 从 list
t2 = torch.zeros((3, 4))                         # 全零
t3 = torch.ones((2, 2))                          # 全一
t4 = torch.arange(0, 10, 2)                      # [0,2,4,6,8]
t5 = torch.linspace(0, 1, 5)                     # [0, 0.25, 0.5, 0.75, 1.0]
t6 = torch.rand(3, 4)                            # 均匀分布 [0,1)
t7 = torch.randn(3, 4)                           # 标准正态

# Task 2: 打印每个 tensor 的 shape, dtype, device
for name, t in zip(['t1','t2','t3','t4','t5','t6','t7'], 
                   [t1,t2,t3,t4,t5,t6,t7]):
    print(f"{name}: shape={t.shape}, dtype={t.dtype}, device={t.device}")

# Task 3: NumPy <-> Tensor 转换
a_np = np.array([1.0, 2.0, 3.0])
a_torch = torch.from_numpy(a_np)       # NumPy → Tensor
print(type(a_torch))                    # <class 'torch.Tensor'>

b_torch = torch.tensor([4.0, 5.0, 6.0])
b_np = b_torch.numpy()                  # Tensor → NumPy
print(type(b_np))                       # <class 'numpy.ndarray'>
```

**验收标准**：全部无报错运行；能说清楚 `t1.shape`, `t1.dtype`, `t1.device` 分别是什么。

---

### Day 2 | Tensor 运算、reshape、索引

**实践任务**（约 1.5 小时）：创建文件 `week04/day02_tensor_ops.py`

✅ **任务 1**：基本运算（对比 NumPy 的语法基本一致）
```python
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 逐元素运算
c1 = a + b         # 应为 [[6,8],[10,12]]
c2 = a * b         # 逐元素乘法（不是矩阵乘法！）

# 矩阵乘法（和 NumPy 一样用 @）
c3 = a @ b         # 矩阵乘法

# 求和 / 均值
print(a.sum())     # 10.0
print(a.mean())    # 2.5
print(a.sum(dim=0))   # 沿着第0维求和（列求和），shape=(2,)
```

✅ **任务 2**：reshape 和维度操作
```python
x = torch.arange(12).float()    # shape=(12,)
x_2d = x.reshape(3, 4)          # shape=(3,4)
x_3d = x.reshape(2, 2, 3)       # shape=(2,2,3)
x_flat = x_2d.flatten()         # shape=(12,)

# unsqueeze / squeeze（加/减维度）
v = torch.tensor([1.0, 2.0, 3.0])  # shape=(3,)
v_row = v.unsqueeze(0)              # shape=(1, 3) 行向量
v_col = v.unsqueeze(1)              # shape=(3, 1) 列向量
v_back = v_col.squeeze()            # shape=(3,)
```

✅ **任务 3**：索引与切片（和 NumPy 基本一致）
```python
X = torch.randn(10, 5)          # 10 个样本，5 个特征
row_0 = X[0]                    # 第一个样本，shape=(5,)
col_2 = X[:, 2]                 # 所有样本的第3个特征，shape=(10,)
first_3 = X[:3]                 # 前3个样本，shape=(3, 5)
mask = X[:, 0] > 0              # 布尔掩码
X_pos = X[mask]                 # 第一列大于0的样本
```

**验收标准**：每个操作的 shape 和数值能说得清楚。特别关注 `*` 和 `@` 的区别（和 NumPy 一样）。

---

### Day 3 | Tensor 的 GPU 概念（可选，CPU 用户可略过细节）

**为什么有这节**：PyTorch 的核心优势之一是能把 Tensor 放到 GPU 上加速计算。本方向的数据量虽然不大（几千个节点），但建立 GPU 概念对后续仍有价值。

**如果你有 NVIDIA GPU**：
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

x = torch.randn(1000, 1000).to(device)    # 放到 GPU
y = torch.randn(1000, 1000).to(device)
z = x @ y                                  # 在 GPU 上计算
print(z.device)

z_cpu = z.cpu()                            # 拿回 CPU
z_np = z_cpu.numpy()                       # 转 NumPy（只能在 CPU 上做）
```

**如果你只有 CPU**：
- ✅ 只需要知道有这个概念，写代码时习惯性加 `.to(device)`
- ✅ 前期用 CPU 跑完全没问题，不影响学习

**验收标准**：了解 `device` 概念；写代码时能正确添加 `.to(device)`。

---

### Day 4–6 | 用 PyTorch 重写 Week 3 的 sin 拟合（只改数据结构）

**衔接**：你已经会用 NumPy 实现前向传播和 MSE。这几天的任务是**纯粹的语法翻译**——把 NumPy 代码改成 PyTorch Tensor 语法，**但仍然手动实现前向传播**（还不用 nn.Module），**仍然手动写反向传播**（还不用 autograd）。

**为什么这样安排**：直接跳到 autograd 和 nn.Module 会让你跳过"Tensor 的运算特性"这个基础。这几天先让你熟悉 Tensor 的运算语法，下周再引入 autograd。

✅ **实践任务**：创建文件 `week04/day456_pytorch_sin.py`

**Step 1**：把 Week 3 的 sin 数据改成 Tensor
```python
import torch
torch.manual_seed(42)
N = 200
x = torch.linspace(-torch.pi, torch.pi, N).reshape(-1, 1)
y_true = torch.sin(x) + torch.randn(N, 1) * 0.05
```

**Step 2**：把参数改成 Tensor（不加 requires_grad，下周再加）
```python
n_in, n_hid, n_out = 1, 32, 1
W1 = torch.randn(n_hid, n_in) * (2.0 / n_in) ** 0.5   # He 初始化（见 Week 3 Day 2.5）
b1 = torch.zeros(n_hid)
W2 = torch.randn(n_out, n_hid) * (2.0 / n_hid) ** 0.5  # He 初始化
b2 = torch.zeros(n_out)
```

**Step 3**：前向传播和反向传播**手动写**（用 Tensor 语法，语法和 NumPy 基本一样）

```python
for epoch in range(2000):
    # 前向
    Z1 = x @ W1.T + b1
    A1 = torch.relu(Z1)    # PyTorch 内置 ReLU
    Z2 = A1 @ W2.T + b2
    y_pred = Z2
    
    loss = ((y_pred - y_true) ** 2).mean()
    
    # 反向（手写，和 NumPy 版本逻辑完全一致）
    N = x.shape[0]
    dL_dypred = 2 * (y_pred - y_true) / N
    dL_dW2 = dL_dypred.T @ A1
    dL_db2 = dL_dypred.sum(dim=0)
    dL_dA1 = dL_dypred @ W2
    dL_dZ1 = dL_dA1 * (Z1 > 0).float()
    dL_dW1 = dL_dZ1.T @ x
    dL_db1 = dL_dZ1.sum(dim=0)
    
    # 更新
    W1 -= 0.01 * dL_dW1
    b1 -= 0.01 * dL_db1
    W2 -= 0.01 * dL_dW2
    b2 -= 0.01 * dL_db2
```

**Step 4**：画拟合曲线（注意：画图前要把 Tensor 转 NumPy 用 `.detach().numpy()`）

```python
import matplotlib.pyplot as plt
x_np = x.numpy()
y_true_np = y_true.numpy()
y_pred_np = y_pred.detach().numpy()    # 必须 .detach()（下周 Week 5 讲原理）

plt.scatter(x_np, y_true_np, s=5, label='True')
plt.plot(x_np, y_pred_np, 'r-', label='Predicted')
plt.legend(); plt.title('sin(x) fit with PyTorch Tensor')
plt.savefig('sin_pytorch.png')
```

> **关于这里的 `.detach()`**：你现在**没有**给参数加 `requires_grad=True`，理论上不加 `.detach()` 也能跑。但这是一个习惯——养成"把 Tensor 转 NumPy 前先 detach"的肌肉记忆。下周讲 autograd 时你会明白"为什么计算图里的 Tensor 必须先 detach 才能转 NumPy"。

**验收标准**：
- 代码能跑通，Loss 最终 < 0.01
- 拟合曲线和 Week 3 的 NumPy 版本质量相当
- 能说清楚：这周只是把 NumPy 语法翻译成 Tensor 语法，**核心逻辑没变**

---

### Week 4 完成标准

- [✅] 能说出 Tensor 和 ndarray 的关系（几乎一样，多了 GPU 和 autograd 支持）
- [✅] 能熟练做 Tensor 的创建、索引、reshape、矩阵乘法
- [✅] 能用 `torch.from_numpy()` 和 `.numpy()` 互相转换
- [✅] 能用 Tensor 语法写一遍 sin 拟合（手动前向 + 手动反向）

---

---

## Week 5: Autograd 与 nn.Module

**衔接**：Week 4 你把 NumPy 代码翻译成了 Tensor 语法，但**反向传播还是手写的**。这周引入 PyTorch 最核心的特性——autograd，让 PyTorch 自动帮你算梯度。同时引入 `nn.Module`，把模型定义从"一堆散乱的 W1/b1/W2/b2"组织成一个标准的类。

**本周目标**：理解 autograd 机制，能定义 `nn.Module` 子类来描述 MLP，能用 autograd 替代手写反向传播。

**主要资源**：
- PyTorch Tutorial "Automatic Differentiation with torch.autograd"（约 30 分钟阅读 + 动手）
- PyTorch Tutorial "Build the Neural Network"（讲 nn.Module）

---

### Day 1–2 | Autograd 基础

**核心概念**：
- `requires_grad=True`：告诉 PyTorch "追踪这个 tensor 参与的所有运算"
- `loss.backward()`：自动算 loss 对所有 `requires_grad=True` 的 tensor 的梯度
- 算完后，梯度存储在 `tensor.grad` 属性中

✅ **实践任务 Day 1**（约 1.5 小时）：创建 `week05/day01_autograd_intro.py`

```python
import torch

# 例子 1：最简单的 autograd
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = w * x                    # y = 6.0
y.backward()                 # 自动算 dy/dw
print(w.grad)                # 应输出 3.0（因为 dy/dw = x = 3.0）

# 例子 2：多变量
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)
y = w * x + b                # y = 7.0
y.backward()
print(w.grad)                # 应输出 3.0（dy/dw = x）
print(b.grad)                # 应输出 1.0（dy/db = 1）

# 例子 3：二次函数（验证和手推一致）
w = torch.tensor(2.0, requires_grad=True)
y = w ** 2                   # y = 4.0
y.backward()
print(w.grad)                # 应输出 4.0（dy/dw = 2w = 4）
```

**验收标准**：每个例子的梯度和手算结果一致。

✅ **实践任务 Day 2**（约 1.5 小时）：创建 `week05/day02_autograd_gotchas.py`

autograd 有几个容易踩的坑，这节专门练。

---

**坑 1**：梯度会累加，每次 backward 前要清零

```python
w = torch.tensor(2.0, requires_grad=True)
y1 = w * 3
y1.backward()
print(w.grad)                # 3.0

y2 = w * 5
y2.backward()
print(w.grad)                # 8.0！不是 5.0！因为梯度累加了

# 正确做法：
w.grad.zero_()               # 清零
y2 = w * 5
y2.backward()
print(w.grad)                # 现在是 5.0
```

---

**坑 2**：把带 requires_grad 的 Tensor 转成 NumPy 前必须 detach

**先看为什么会报错**：
```python
w = torch.tensor([1.0, 2.0], requires_grad=True)
y = w * 2       # y 继承了 requires_grad，y 也在"计算图"里
y.numpy()       # RuntimeError: Can't call numpy() on Tensor that requires grad
```

**为什么 PyTorch 要阻止你**：
- `y` 携带了"我是从 w 算出来的"这个历史信息（autograd 需要）
- 如果允许 `y.numpy()`，你就把它变成了一个 NumPy 数组，之后你做的运算 PyTorch 完全看不见
- 但 `w` 还在计算图里等着接收梯度——PyTorch 担心你无意中破坏了梯度追踪
- 所以它直接不让——**逼你主动声明"我知道我在做什么，这个 Tensor 不再需要追踪梯度"**

**detach 做了什么**：
```python
y_detached = y.detach()
# y_detached 是一个新的 Tensor：
# - 数值和 y 完全一样
# - 但它不在计算图里，requires_grad=False
# - 可以安全转成 NumPy
y_np = y_detached.numpy()
```

**一个简单的判断规则**：
- 训练时：**不要** detach（会切断梯度流，模型学不动）
- 要画图、保存结果、打印日志：**必须** detach（这些操作不需要梯度）
- 用 `torch.no_grad()` 包裹的代码块里：也不需要 detach（这个区域本身就不追踪梯度）

**回看 Week 4 Day 4-6 的代码**：你当时写了 `y_pred.detach().numpy()` 是为了画图。那时 `y_pred` 虽然没加 `requires_grad=True`（参数没启用 autograd），但下周你加了之后这行代码就是**必须的**——养成肌肉记忆。

---

**坑 3**：更新参数时要用 `torch.no_grad()` 包裹，否则 PyTorch 会把更新操作也记录到计算图里

```python
with torch.no_grad():
    w -= 0.01 * w.grad       # 正确的更新方式
    w.grad.zero_()
```

---

**坑 4**：记录标量值要用 `.item()`

```python
loss = ((y_pred - y_true) ** 2).mean()
# loss 是一个标量 Tensor（shape=()，只有一个值），不是 Python float

# 错误做法：
losses.append(loss)                    # 追加的是 Tensor，不是数值
# 问题：loss 还在计算图里（requires_grad=True），反复追加会占用大量内存
# 更严重的问题：每次 append 的 loss 实际上还是和计算图绑定的

# 正确做法：
losses.append(loss.item())             # .item() 把标量 Tensor 转成 Python float
# .item() 只能用于标量（单个值）的 Tensor
# 如果是多元素 Tensor，用 .detach().cpu().numpy() 或 .tolist()
```

**何时需要 `.item()`**：
- 要记录到 list 里做可视化
- 要打印 print
- 要和 Python 数值比较（`if loss.item() < 0.01`）

**何时不需要 `.item()`**：
- 继续用于计算 backward（保留 Tensor 形式）
- 要在 Tensor 之间做运算（仍保留在 GPU 或计算图里）

---

✅ **验证任务**：用 autograd 手动算一个二层网络的梯度，和 Week 3 手推的结果对比

```python
# 一个样本的前向（为了简化）
x = torch.tensor([1.0, 0.5])
W1 = torch.tensor([[0.5, 1.0], [-0.5, 2.0]], requires_grad=True)  # shape=(2,2)
b1 = torch.tensor([0.0, 0.0], requires_grad=True)
W2 = torch.tensor([[1.0, 1.0]], requires_grad=True)                # shape=(1,2)
b2 = torch.tensor([0.0], requires_grad=True)

z1 = W1 @ x + b1
a1 = torch.relu(z1)
y_pred = W2 @ a1 + b2
y_true = torch.tensor([2.0])
loss = ((y_pred - y_true) ** 2).mean()

loss.backward()

print(W1.grad)    # autograd 算的梯度
print(b1.grad)
print(W2.grad)
print(b2.grad)
```

这几个梯度你可以手算一遍（用 Week 3 的公式）验证，和 autograd 输出对比。

**验收标准**：autograd 输出的梯度和手算的结果在小数点后 4 位一致。

---

### Day 3 | nn.Module 基础

**为什么要用 nn.Module**：Week 4 你的参数是一堆散乱的变量 W1/b1/W2/b2，代码难管理。`nn.Module` 是 PyTorch 定义模型的标准方式，把参数封装在类里，有几个好处：
- `model.parameters()` 一次性拿到所有参数
- `.to(device)` 一次性把所有参数搬到 GPU
- 结构清晰，便于保存和加载

✅ **实践任务**（约 2 小时）：创建 `week05/day03_nn_module.py`

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)    # 自动创建 W1 和 b1
        self.fc2 = nn.Linear(hid_dim, out_dim)   # 自动创建 W2 和 b2
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# 使用
model = MLP(in_dim=1, hid_dim=32, out_dim=1)
print(model)                           # 打印模型结构

# 查看参数
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}")

# 统计参数总数
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# 应该是 1*32 + 32 + 32*1 + 1 = 97

# 前向传播
x = torch.randn(10, 1)      # 10 个样本
y_pred = model(x)            # 等价于 model.forward(x)
print(y_pred.shape)          # 应为 (10, 1)
```

**验收标准**：
- 模型能成功创建，打印结构清晰
- 参数总数正确（应为 97）
- 前向传播 shape 正确

---

**（延伸 10 分钟）nn.Linear 背后的 nn.Parameter 机制**：

你可能好奇——`nn.Linear(in_dim, out_dim)` 里说"自动创建 W 和 b"，这个"自动"是什么机制？

看源码大致是这样的：
```python
# 大致等价于：
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 关键：用 nn.Parameter 包裹，告诉 PyTorch "这是可学习参数"
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return x @ self.weight.T + self.bias
```

**关键点**：`nn.Parameter` 是一个"标记"——它告诉 `nn.Module` "这个 Tensor 是可学习参数"。
- `model.parameters()` 会自动找到所有被 `nn.Parameter` 标记的 Tensor
- 这些 Tensor 会自动 `requires_grad=True`
- optimizer 靠这个机制拿到所有要更新的参数

**什么时候你需要自己用 `nn.Parameter`**：
- **定义自定义层**（后面 Part C 的 GNN 会用到）
- **在 PINN 里定义可学习的物理参数**（Part B 会用到）：

```python
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 32)
        # 假设弹性模量 E 作为可学习参数（反演问题）
        self.E = nn.Parameter(torch.tensor(1.0))  # 必须用 nn.Parameter！
```

如果写成 `self.E = torch.tensor(1.0)`，optimizer 就看不到它，永远不会被更新。

**本周你不需要自定义层**（直接用 nn.Linear 就够了）——但提前知道这个机制，后面 Part B 和 Part C 遇到时就不会踏空。

---

### Day 4–6 | 用 nn.Module + autograd 重写 sin 拟合

**这是本周的闭环任务**：把所有零散的知识点整合成一个"标准 PyTorch 训练流程"。

✅ **实践任务**：创建 `week05/day456_mlp_sin_autograd.py`

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Step 1: 定义模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Step 2: 数据
torch.manual_seed(42)
N = 200
x = torch.linspace(-torch.pi, torch.pi, N).reshape(-1, 1)
y_true = torch.sin(x) + torch.randn(N, 1) * 0.05

# Step 3: 模型 + 手动梯度下降（不用 optimizer，为了看清每一步）
model = MLP()
lr = 0.01
losses = []

for epoch in range(2000):
    # 前向
    y_pred = model(x)
    loss = ((y_pred - y_true) ** 2).mean()
    losses.append(loss.item())        # .item() 把标量 Tensor 转 float（Day 2 坑 4）
    
    # 反向（autograd 自动算梯度）
    loss.backward()
    
    # 手动更新参数（下周引入 optimizer 替代）
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
            param.grad.zero_()           # 清零准备下次迭代
    
    if epoch % 200 == 0:
        print(f"epoch {epoch}: loss={loss.item():.6f}")

# Step 4: 可视化
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(losses); plt.yscale('log')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss')

plt.subplot(1,2,2)
x_np = x.numpy(); y_true_np = y_true.numpy()
y_pred_np = model(x).detach().numpy()   # detach 转 NumPy（Day 2 坑 2）
plt.scatter(x_np, y_true_np, s=5, label='True')
plt.plot(x_np, y_pred_np, 'r-', label='Pred')
plt.legend(); plt.title('Fit')

plt.tight_layout(); plt.savefig('week05_sin.png')
```

**体会**：对比 Week 3 的 NumPy 版本，这份代码：
- 不再手写反向传播（少了 ~10 行）
- 参数管理更清晰（`model.parameters()`）
- 本质流程没变：前向 → Loss → backward → 更新 → 清零

**验收标准**：
- 代码能跑通，Loss < 0.01
- 拟合质量和 Week 3 相当
- 能向别人讲清楚这份代码的每一个步骤在做什么

---

### Week 5 完成标准

- [✅] 能解释 `requires_grad`、`backward()`、`.grad` 的作用
- [✅] 知道 autograd 的 4 个坑：梯度累加、detach、torch.no_grad()、loss.item()
- [✅] 能解释为什么必须 detach、detach 做了什么
- [✅] 了解 `nn.Parameter` 是什么、什么时候要用（可学习物理参数会用）
- [✅] 能从零定义一个继承 `nn.Module` 的 MLP 类
- [✅] 能用 autograd + nn.Module 替代 NumPy 版本的 sin 拟合

---

---

## Week 6: 优化器 + DataLoader + 完整训练管线

**衔接**：Week 5 你已经能用 autograd 替代手写反向传播，用 nn.Module 组织模型。但训练循环里你**还在手动遍历参数更新**，**还在用整个数据集一次性训练**（full batch）。这周引入两个 PyTorch 标准组件：`Optimizer`（替代手动参数更新，支持 SGD/Adam 等）和 `DataLoader`（自动分 mini-batch）。

**本周目标**：掌握 PyTorch 的标准训练流程模板，能训练多特征回归任务（不只是单特征 sin）。

**主要资源**：
- PyTorch Tutorial "Optimizing Model Parameters"
- PyTorch 官方文档 `torch.optim` 的 Adam 和 SGD 部分

---

### Day 1 | Optimizer：SGD 和 Adam

#### 实践前先理解：SGD 和 Adam 的核心差异（约 30 分钟）

**Week 2 你手写过梯度下降**：
```
w = w - lr * grad
```
所有参数共用一个 `lr`，梯度多大就走多远。这就是 **SGD**（随机梯度下降，Stochastic Gradient Descent）的核心。

**SGD 的问题**：
- 所有参数共享同一个学习率——但真实网络里有些参数梯度大（变化快）、有些梯度小（变化慢）
- 大梯度的参数步子太大容易震荡，小梯度的参数步子太小收敛太慢
- 只能靠你手动调一个"折衷的 lr"，调起来很痛苦

**Adam 的核心思想**（不需要看论文，理解直觉即可）：
- 每个参数**独立维护**一个"有效学习率"
- 对梯度大的参数自动缩小步子；对梯度小的参数自动放大步子
- 这样即使你设的 `lr=1e-3` 不是"最优"，Adam 也能自适应调整，通常都不会崩

**一句话对比**：
- SGD：所有参数同样速度走，lr 调不好就崩
- Adam：每个参数按自己的情况走，lr 差点也没事

**为什么实践中先用 Adam**：**不是因为它精度更高**（精调后的 SGD 在某些任务上更好），而是因为它**调起来省心**。作为新手你的时间应该花在理解模型、诊断问题上，不是纠结 lr。PhyFENet 论文用 Adam 也是这个原因。

**不用深入推导 Adam 的数学**——它涉及一阶动量（梯度的移动平均）和二阶动量（梯度平方的移动平均），这些细节你到需要时再查。现在只要记住：
- **默认用 Adam，lr 先设 1e-3 或 1e-4**
- **SGD 作为对照实验**（本 Day 的对比实验会让你直观看到差别）

**口头自测**（继续前必须回答）：
1. 为什么 Adam 在实践中通常比 SGD 好调？
2. 什么时候可能会用 SGD 而不是 Adam？
   > 参考：某些任务精调后的 SGD 泛化性更好；但这不是新手该纠结的

---

#### 核心 API

- `torch.optim.SGD(model.parameters(), lr=0.01)`：最基础的随机梯度下降
- `torch.optim.Adam(model.parameters(), lr=0.001)`：自适应学习率，对大多数任务效果更好
- 用法：`optimizer.step()` 替代手动 `param -= lr * param.grad`；`optimizer.zero_grad()` 替代 `param.grad.zero_()`

✅ **实践任务**（约 1.5 小时）：创建 `week06/day01_optimizer.py`

把 Week 5 的手动参数更新改成用 optimizer：

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

torch.manual_seed(42)
N = 200
x = torch.linspace(-torch.pi, torch.pi, N).reshape(-1, 1)
y_true = torch.sin(x) + torch.randn(N, 1) * 0.05

model = MLP()
criterion = nn.MSELoss()                            # 内置 MSE
# 注意：这是先创建一个"Loss 对象"，再用它计算 loss
# 等价于之前你写的 ((y_pred - y_true) ** 2).mean()
# 
# 为什么 PyTorch 要把 Loss 设计成类？
# - 因为有些 Loss 有"可学参数"（比如带权重的分类 Loss）
# - 类的形式能统一处理带参数和不带参数的 Loss
# - 对 MSE 来说三种写法完全等价：
#   loss_a = criterion(y_pred, y_true)           # 类的方式（Week 6 以后用这个）
#   loss_b = ((y_pred - y_true) ** 2).mean()     # 直接写（Week 1-5 的方式）
#   import torch.nn.functional as F
#   loss_c = F.mse_loss(y_pred, y_true)          # 函数式
# 三种写法等价，本计划从 Week 6 起统一用类的方式。

optimizer = optim.Adam(model.parameters(), lr=0.01) # 用 Adam

losses = []
for epoch in range(2000):
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    losses.append(loss.item())
    
    optimizer.zero_grad()   # 清零梯度
    loss.backward()         # 算梯度
    optimizer.step()        # 更新参数
    
    if epoch % 200 == 0:
        print(f"epoch {epoch}: loss={loss.item():.6f}")
```

**对比实验**：同样的代码，换成 `optim.SGD(model.parameters(), lr=0.01)`，看 Loss 曲线。通常 Adam 收敛更快、更平稳。

**验收标准**：
- 用 Adam 训练 2000 epoch 后 Loss < 0.005
- 用 SGD 训练 2000 epoch 后通常 Loss 在 0.01-0.1 之间（收敛较慢）
- 能画两条 Loss 曲线对比
- 能用自己的话说清楚 Adam 相对 SGD 的优势

---

### Day 2 | DataLoader 和 TensorDataset

**为什么需要 DataLoader**：实际训练中数据量可能很大（几万个样本），一次性把所有数据丢进网络（full batch）会：
- 内存不够
- 更新方向完全由整个数据集决定，收敛慢

实际做法：把数据切成多个小 batch（比如每个 batch 32 个样本），每个 batch 更新一次参数。这叫 **mini-batch gradient descent**。

DataLoader 就是自动做这件事的工具。

✅ **实践任务**（约 1.5 小时）：创建 `week06/day02_dataloader.py`

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# 模拟多特征回归数据
N = 1000
X = torch.randn(N, 5)                   # 5 个特征
y = (X[:,0]*2 + X[:,1]*3 - X[:,2] + 0.5).reshape(-1, 1)
y = y + torch.randn(N, 1) * 0.1         # 加噪声

# 把 X 和 y 打包成 Dataset
dataset = TensorDataset(X, y)
print(f"Dataset size: {len(dataset)}")

# 用 DataLoader 自动分 batch
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 验证：遍历一遍 loader
for batch_idx, (batch_x, batch_y) in enumerate(loader):
    print(f"Batch {batch_idx}: x.shape={batch_x.shape}, y.shape={batch_y.shape}")
    if batch_idx >= 2:
        break
# 应输出前 3 个 batch 的形状，都是 (32, 5) 和 (32, 1)
# 最后一个 batch 如果不足 32 个会是剩余数量
```

**验收标准**：能看懂 DataLoader 每次返回一个 batch，batch 里样本数 = batch_size；设置 `shuffle=True` 后每个 epoch 的顺序不同。

---

#### 概念澄清：epoch / batch / iteration（约 5 分钟）

引入 DataLoader 后这三个词频繁出现，容易混：

- **epoch**（周期）：**整个数据集**完整过一遍叫 1 epoch
- **batch**（批次）：数据集被切成的小块，每块叫 1 batch
- **iteration**（迭代）：一次参数更新叫 1 iteration，通常 = 过完 1 个 batch

**举例**：
- 1000 个样本，batch_size=32
- 1 epoch 里有 `1000 / 32 ≈ 32` 个 batch
- 1 epoch = 32 个 iteration
- 训练 100 epoch = 3200 个 iteration = 3200 次参数更新

**Week 2 你之前的 full batch 情况**：
- batch_size = N（整个数据集）
- 1 epoch = 1 batch = 1 iteration
- 所以那时候你没区分这三个词

**Week 6 以后 mini-batch 情况**：
- 1 epoch > 1 batch
- 训练速度**表面看**慢了（每个 epoch 要过 32 个 batch），**但实际收敛快得多**（每个 epoch 更新了 32 次参数而不是 1 次）

---

### Day 2.5（新增，约 45 分钟）| 为什么要把数据分成训练集和验证集

**这是进入 Day 3-4 之前必须理解的概念**。Day 3-4 的代码会把数据切成"train"和"val"两部分——这个做法有非常具体的工程意义。

---

**一个核心问题**：你怎么知道你训练的模型"真的学到了规律"？

**反面例子**：假设你让学生做 100 道练习题，他把每道题的答案都背下来了。考试时如果你**原题再考一遍**，他全对——但这不代表他学会了数学，只能说明他记性好。

**神经网络也一样**：如果你用全部数据训练，然后用同样的数据评估 Loss，Loss 很低也不能说明模型泛化能力好——它可能只是"背"下了训练数据的每个点（这种现象叫**过拟合**，Week 7 会详细讲）。

**正确做法**：把数据分成两部分
- **训练集（training set）**：约 80%，用来训练模型（参数更新用）
- **验证集（validation set）**：约 20%，训练中不用来更新参数，只用来评估
  - 如果训练集 Loss 很低但验证集 Loss 很高 → 模型只是"背"了训练数据，没学到泛化规律
  - 如果两个 Loss 都下降 → 模型真的学到了规律

---

✅ **实践任务**（约 45 分钟）：创建 `week06/day25_train_val_split.py`

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# 生成数据（和 Day 2 一样）
torch.manual_seed(42)
N = 1000
X = torch.randn(N, 5)
y = (X[:,0]*2 + X[:,1]*3 - X[:,2] + 0.5).reshape(-1, 1)
y = y + torch.randn(N, 1) * 0.1

# ===== 核心操作：train/val 切分 =====
# 方法 1：按顺序切（简单直接）
n_train = int(0.8 * N)      # 80% 训练
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

print(f"训练集样本数: {len(X_train)}")    # 800
print(f"验证集样本数: {len(X_val)}")      # 200

# 方法 2：随机切（更严谨，避免数据有顺序偏差）
perm = torch.randperm(N)                   # 随机打乱索引
train_indices = perm[:n_train]
val_indices = perm[n_train:]
X_train2 = X[train_indices]
X_val2 = X[val_indices]

# 把训练集和验证集分别包装成 DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), 
                          batch_size=32, shuffle=True)   # 训练集 shuffle
val_loader = DataLoader(TensorDataset(X_val, y_val),
                        batch_size=32, shuffle=False)     # 验证集不 shuffle

# 验证两个 loader 都能用
for batch_x, batch_y in train_loader:
    print(f"Train batch: {batch_x.shape}")
    break
for batch_x, batch_y in val_loader:
    print(f"Val batch: {batch_x.shape}")
    break
```

**要回答的三个问题**（口头即可）：
1. 为什么训练集要 `shuffle=True`，验证集要 `shuffle=False`？
   > 参考：训练集打乱让每个 epoch 的 batch 组合不同，减少模型记住样本顺序；验证集只是评估，顺序不影响结果，关掉 shuffle 更快更好复现

2. 如果只用训练集评估 Loss，会有什么问题？
   > 参考：Loss 低不能证明模型学到了泛化规律，只能证明它"记住"了训练数据

3. 为什么切分比例常选 80/20 而不是 50/50？
   > 参考：训练集越多模型学得越好；验证集只需要"够代表性"即可，不需要太多。50/50 会浪费一半数据没用来学习

**验收标准**：
- 能跑通切分代码
- 能回答上面 3 个问题
- **理解一个核心观点**：训练 Loss 不是模型真正好坏的标准，验证 Loss 才是

---

### Day 3–4 | 完整训练流程（多特征回归）

**衔接**：前两天有了 Optimizer、DataLoader 和 train/val 切分概念，现在把它们整合成"完整的标准训练流程"。这个流程会是你之后所有项目的模板。

---

**关于 nn.Sequential**（10 分钟延伸）：

本节代码首次使用 `nn.Sequential`。回顾 Week 5 Day 3 你是这么写 MLP 的：
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
```

每层都要单独 `self.fcN = ...`，forward 里也要逐层调用。这种写法**控制更细**，比如要在某一层后加 dropout、或者加残差连接就方便。

`nn.Sequential` 是简化写法：
```python
self.net = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),           # 激活函数也作为一层
    nn.Linear(32, 1)
)
# forward 只需要：return self.net(x)
```

自动把所有子层"串起来"执行。

**什么时候用哪个**：
- **Sequential**：网络是"直线型"的、逐层堆叠、中间不需要特殊处理——**大多数 MLP 用这个，代码简洁**
- **分层写 self.fcN**：需要在中间层加分支、残差连接、跳接等——**后面 Part C 的 GNN 会用这种**

注意两种写法**功能完全等价**，只是代码风格不同。**不要纠结选哪个**。

---

**实践任务**：创建 `week06/day34_full_pipeline.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ===== 1. 数据 =====
torch.manual_seed(42)
N = 1000

# 模拟 FEM 风格的回归任务：
# 输入：3 个特征（可理解为：坐标 + 材料参数 + 载荷）
# 输出：1 个物理量
X = torch.randn(N, 3)
y = (2*X[:,0] + 3*X[:,1]**2 - X[:,2] + 0.5).reshape(-1, 1)
y = y + torch.randn(N, 1) * 0.2

# (划分) train / val（80% / 20%）—— 见 Day 2.5
n_train = int(0.8 * N)
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# ===== 2. 模型 =====
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(           # 用 Sequential 写法
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=3, hid_dim=64, out_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ===== 3. 训练循环（标准模板，要记住）=====
train_losses = []
val_losses = []

for epoch in range(100):
    # Train
    model.train()
    train_loss_sum = 0
    for bx, by in train_loader:
        y_pred = model(bx)
        loss = criterion(y_pred, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * bx.size(0)  # 乘 batch_size 还原
    train_loss_avg = train_loss_sum / len(train_ds)
    train_losses.append(train_loss_avg)
    
    # Validate
    model.eval()
    val_loss_sum = 0
    with torch.no_grad():                      # 验证不需要梯度
        for bx, by in val_loader:
            y_pred = model(bx)
            val_loss_sum += criterion(y_pred, by).item() * bx.size(0)
    val_loss_avg = val_loss_sum / len(val_ds)
    val_losses.append(val_loss_avg)
    
    if epoch % 10 == 0:
        print(f"epoch {epoch}: train_loss={train_loss_avg:.4f}, val_loss={val_loss_avg:.4f}")

# ===== 4. 可视化 =====
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log')
plt.legend(); plt.title('Training Progress')
plt.savefig('week06_training.png')
```

---

**这份代码中的 5 个关键点**（你需要能解释每一个）：

---

**1. `model.train()` / `model.eval()`：切换模型状态**

为什么需要两个状态？因为有些网络层在训练和推理时行为不同：

- **Dropout 层**（Week 7 Day 2 会用到）：训练时随机丢弃一些神经元防止过拟合，推理时不丢（要用完整网络做预测）
- **BatchNorm 层**（本 Part 不用但很常见）：训练时用当前 batch 的均值方差做归一化，推理时用训练期间累积的全局均值方差

所以：
- `model.train()`：告诉所有层"现在是训练模式，Dropout 开启、BatchNorm 用 batch 统计"
- `model.eval()`：告诉所有层"现在是推理模式，Dropout 关闭、BatchNorm 用全局统计"

**本周的 MLP 里没有这些层**，所以两者行为一样——但现在就养成习惯。等 Week 7 开始用 Dropout，或者后面用别的有"train/eval 差异"的层时，这习惯会避免你调试半天找不到 bug。

**面试追问**："`model.eval()` 和 `with torch.no_grad()` 有什么区别？"
> 参考：前者切换层的行为模式（Dropout/BN），后者关闭梯度追踪。两者是独立的——验证时通常两个都要用

---

**2. `optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()`**：三行一组，标准梯度更新流程

- `zero_grad()`：清空上次的梯度（否则会累加，见 Week 5 Day 2 坑 1）
- `backward()`：算梯度（autograd，见 Week 5 Day 1）
- `step()`：根据梯度更新参数（Adam 在内部做自适应学习率调整）

---

**3. `with torch.no_grad():`**：验证时不需要算梯度

为什么用：
- 节省内存（不存计算图）
- 提速（不做梯度追踪相关的记账）
- 避免无意中修改了 `.grad` 属性

---

**4. `train/val split`（见 Day 2.5）**：训练集参与梯度更新，验证集只用来评估

如果验证 Loss 开始反升而训练 Loss 还在降，说明过拟合了（Week 7 详讲）。

---

**5. 记录 loss 时乘以 `bx.size(0)` 再除以 `len(dataset)`：正确做加权平均**

核心问题：`criterion(y_pred, by)` 返回的是**当前 batch 的平均 Loss**。

假设你这个 epoch 有 4 个 batch：
- Batch 1: 32 样本, loss = 0.5
- Batch 2: 32 样本, loss = 0.3
- Batch 3: 32 样本, loss = 0.4
- Batch 4: 8 样本（最后不足一个完整 batch）, loss = 0.6

**错误做法**（直接平均）：
```python
epoch_loss = (0.5 + 0.3 + 0.4 + 0.6) / 4 = 0.45
```
这是**错的**——把"只有 8 样本的 batch"和"32 样本的 batch"等权重看待了。

**正确做法**（按样本数加权）：
```python
total_loss = 0.5*32 + 0.3*32 + 0.4*32 + 0.6*8 = 16 + 9.6 + 12.8 + 4.8 = 43.2
total_samples = 32+32+32+8 = 104
epoch_loss = 43.2 / 104 ≈ 0.4154
```
这才是真正的"所有样本的平均 Loss"。

代码里就是这个逻辑：
```python
train_loss_sum += loss.item() * bx.size(0)    # 这个 batch 所有样本的 loss 总和
# 最后：
train_loss_avg = train_loss_sum / len(train_ds)   # 除以总样本数
```

**为什么重要**：最后一个 batch 经常不满——如果不加权，评估指标会有偏差。养成习惯永远这么写。

**另一种等效做法**：`criterion(y_pred, by, reduction='sum')` 返回 batch 的 loss 总和（不是平均），这样直接 `+=` 就行，不用乘样本数。两种都可以。

---

**验收标准**：
- Train loss 和 Val loss 都在下降
- 最终 Val Loss 应在 0.04-0.1 之间（取决于随机种子）
- 能解释上面 5 个关键点

---

### Day 5–6（周末）| 模型保存与加载 + 本周整理

**Day 5**：创建 `week06/day5_save_load.py`

```python
# 保存模型
torch.save(model.state_dict(), 'model_week06.pth')

# 加载模型
model_new = MLP(in_dim=3, hid_dim=64, out_dim=1)  # 先创建同样结构
model_new.load_state_dict(torch.load('model_week06.pth'))
model_new.eval()

# 验证两个模型输出一致
x_test = torch.randn(5, 3)
with torch.no_grad():
    y1 = model(x_test)
    y2 = model_new(x_test)
assert torch.allclose(y1, y2)
print("Save/Load OK!")
```

**Day 6**：把本周所有代码整理进项目目录结构：

```
ai-cae-learning/
├── utils/
│   ├── fem_parser.py
│   ├── visualization.py
│   └── training.py        ← 新增：封装训练循环
└── week06/
    ├── day01_optimizer.py
    ├── day02_dataloader.py
    ├── day25_train_val_split.py
    ├── day34_full_pipeline.py
    └── day5_save_load.py
```

在 `utils/training.py` 中写一个通用的训练函数：
```python
def train_model(model, train_loader, val_loader, 
                criterion, optimizer, n_epochs, device='cpu'):
    """通用训练循环，返回 train_losses 和 val_losses 两个列表"""
    # 把 Day 3-4 的训练循环抽取出来
    ...
```

---

### Week 6 完成标准

- [ ] 能解释 SGD 和 Adam 的核心差异，知道为什么默认用 Adam
- [ ] 能澄清 epoch / batch / iteration 的关系
- [ ] 能解释为什么要把数据分成训练集和验证集（不是只是"按模板抄"）
- [ ] 能说清 `nn.Sequential` 和分层 `self.fcN` 两种写法什么时候用哪个
- [ ] 能从零写出"定义模型 + 定义 optimizer + DataLoader + 训练循环"的完整模板
- [ ] 能解释 `optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()` 三步的作用和顺序
- [ ] 能解释 `model.train()` / `model.eval()` 是做什么的，为什么要调
- [ ] 能解释为什么要用 `bx.size(0)` 做加权平均，错误平均会怎么样
- [ ] 能用 `state_dict` 保存和加载模型

---

---

## Week 7: 训练诊断——Loss 曲线、过拟合与归一化

**衔接**：Week 6 你能训练模型了。但在真实任务中，训练可能不收敛、可能过拟合、可能数据没归一化导致训练失败。这周教你"看信号"——从 Loss 曲线和输出判断训练是否正常，遇到问题时怎么改。

**本周目标**：能看懂各种 Loss 曲线形态并判断问题；理解过拟合 / 欠拟合的概念和应对；掌握数据归一化（Z-Score）的必要性和实现。

**主要资源**：
- 吴恩达 Course 2 Week 1（偏差/方差、正则化、归一化相关视频，约 2 小时）
- 李宏毅 2021 ML Lecture 10-12（可选，偏向中文讲解）

---

### Day 1 | 识别不健康的 Loss 曲线

**理论任务**：看吴恩达 Course 2 Week 1 的 "Basic Recipe for Machine Learning" 视频

**在笔记上画 4 种 Loss 曲线草图，并标注诊断**：

| 曲线形态 | 诊断 | 原因 | 解决方向 |
|---------|------|------|---------|
| Train loss 不降，几乎平的 | 训练未启动 | 学习率太小 / 梯度消失 | 增大 lr / 检查初始化 |
| Train loss 剧烈震荡 | 学习率太大 | lr 过大跨过最低点 | 减小 lr |
| Train loss 降得很低，Val loss 降到一半就反升 | 过拟合 | 模型太复杂或数据太少 | 加正则化 / 增加数据 |
| Train loss 和 Val loss 都停在高位 | 欠拟合 | 模型太简单或训练不够 | 加深网络 / 训练更久 |

**实践任务**（约 1 小时）：创建 `week07/day01_bad_losses.py`

人为制造四种不健康的训练，画出对应的 Loss 曲线：
- lr 设成 1e-8（不降）
- lr 设成 10（震荡）
- 数据很少（N=20），模型很大（hidden=256），训练很久（过拟合）
- 模型很小（单层 Linear），数据复杂（过拟合不上，欠拟合）

对每种情况画 train+val loss 曲线，存成 png，在图里用文字标注诊断（`plt.text()`）。

**验收标准**：4 张图能清楚展示不同问题。

---

### Day 2 | 过拟合与正则化

**理论任务**：
- 看吴恩达 Course 2 Week 1 关于"正则化"的视频
- 理解：L2 正则 = 在 Loss 里加一个 `λ * Σ w²`，让权重变小，避免模型过度依赖某些特征

**关于 Dropout**（回应 Week 6 的铺垫）：
- Dropout 是另一种正则化方法——训练时随机"丢掉"一部分神经元的输出（设为 0）
- 相当于每次训练都用"不同的子网络"，模型被迫学到冗余的特征
- 推理时不丢（这就是为什么需要 `model.train()` / `model.eval()` 区分模式）
- 在 PyTorch 里用 `nn.Dropout(p=0.5)` 即可

**实践任务**：创建 `week07/day02_regularization.py`

用 Day 1 的"过拟合"情景（N=20，hidden=256），加入 L2 正则和 Dropout：

```python
# 方法 1：L2 正则（weight_decay）
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
# weight_decay 就是 L2 正则化的 λ

# 方法 2：Dropout（在模型定义里加）
class MLP_Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),      # ← 30% 神经元被丢
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x)
```

对比：
- 无正则
- +weight_decay
- +Dropout
- +weight_decay +Dropout

的 val loss 曲线。

**验收标准**：带正则化的 val loss 应该比不带的低（或至少不那么快反升）。能观察到 Dropout 和 L2 的效果区别。

---

### Day 3 | 归一化（Z-Score）

**为什么这对你方向特别重要**：PhyFENet 论文明确指出用 Z-Score 归一化处理 FEM 数据。原因是 FEM 输入各特征的量纲差异巨大——节点坐标可能是 0-2000mm，弹性模量可能是 200000 MPa，载荷可能是 100 kN。不归一化的话，量纲大的特征会主导训练，量纲小的特征几乎学不到。

**理论任务**：
- 在纸上写 Z-Score 公式：`x_normalized = (x - mean) / std`
- 用自己的话解释：为什么归一化后各特征"平等"了？

**实践任务**（约 2 小时）：创建 `week07/day03_normalization.py`

**Step 1**：模拟"不同量纲"的数据
```python
N = 1000
coord = torch.randn(N, 1) * 1000     # 量级 1000
E = torch.randn(N, 1) * 200 + 200    # 量级 200
F = torch.randn(N, 1) * 10           # 量级 10
X = torch.cat([coord, E, F], dim=1)   # shape=(N, 3)

# 目标：y 同等依赖 3 个特征
y = (coord/1000 + E/200 + F/10).reshape(-1, 1)  # 每项量级都被压到 1 左右
y = y + torch.randn(N, 1) * 0.1
```

**Step 2**：不归一化训练
用 Week 6 的标准流程训练 MLP 100 epoch，记录 val loss。

**Step 3**：Z-Score 归一化
```python
# 注意：用训练集的 mean/std，不能用验证集！
X_mean = X_train.mean(dim=0, keepdim=True)
X_std = X_train.std(dim=0, keepdim=True)

X_train_norm = (X_train - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std     # 用训练集的 mean/std
```

y 也一样归一化。训练后，预测时要把 y 反归一化回来：
```python
y_pred_original = y_pred_norm * y_std + y_mean
```

**Step 4**：对比两个版本的 val loss 曲线

**验收标准**：归一化版本的 val loss 应明显小于不归一化版本（通常差 10-100 倍）。

**封装任务**：把归一化逻辑封装成一个类，放在 `utils/normalization.py`：
```python
class ZScoreNormalizer:
    def fit(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True) + 1e-8   # 避免除零
    def transform(self, X):
        return (X - self.mean) / self.std
    def inverse_transform(self, X):
        return X * self.std + self.mean
```

---

### Day 4 | 训练诊断综合练习

**任务**：创建 `week07/day04_debug_broken_training.py`

给你一份"坏代码"，你的任务是调试它，让它能正常训练。

```python
# ========== 这份代码有 3 个问题，找出并修复 ==========

# Bug 1: 数据没归一化
X = torch.cat([
    torch.randn(1000, 1) * 1000,   # 量级 1000
    torch.randn(1000, 1) * 0.01,   # 量级 0.01
], dim=1)
y = (X[:,0]/1000 + X[:,1]*100).reshape(-1,1) + torch.randn(1000,1)*0.1

# Bug 2: 学习率过大
optimizer = optim.Adam(model.parameters(), lr=10.0)

# Bug 3: 没清零梯度
for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    # 缺 optimizer.zero_grad()
```

**你的任务**：复制这份代码，修复 3 个 bug，让训练能正常收敛。

**验收标准**：修复后 val loss < 0.1。

---

### Day 5–6（周末）| Part A 自测与整理

**Day 5**：Part A 完整自测（限时 3 小时）

**理论题**（口头或笔记回答，每题 2-3 句）：
1. 为什么神经网络需要激活函数？ReLU 和 Tanh 各有什么特点？ReLU 的坑是什么？
2. 梯度下降的核心思想是什么（不用公式）？SGD 和 Adam 的区别？
3. 反向传播和链式法则的关系？
4. 为什么权重不能初始化为 0？He 初始化做了什么？
5. autograd 的 4 个坑是什么？
6. `model.train()` / `model.eval()` 为什么要区分？
7. 为什么要把数据分成训练集和验证集？只用训练集评估会怎样？
8. epoch / batch / iteration 的关系？
9. 为什么 FEM 数据要归一化？L2 正则化和 Dropout 的区别？

**代码题**（限时完成，不看参考）：
1. （40 分钟）从零写一个完整的 PyTorch 训练流程：模型 + 数据 + train/val split + 训练循环 + Loss 曲线可视化。具体任务：给定 3 维输入、1 维输出的随机数据，训练一个 2 层 MLP。
2. （15 分钟）写 `ZScoreNormalizer` 类（fit / transform / inverse_transform）。

**Day 6**：整理项目仓库
```
ai-cae-learning/
├── README.md              ← 写第一版 README：我在做什么 / 进度
├── utils/
│   ├── fem_parser.py
│   ├── visualization.py
│   ├── normalization.py
│   └── training.py
├── pretrain/              ← 补零期代码
└── phase1/
    ├── part_a_dl_foundations/
    │   ├── week01/
    │   ├── week02/
    │   ├── week03/
    │   ├── week04/
    │   ├── week05/
    │   ├── week06/
    │   └── week07/
    └── self_test_partA.md  ← 写自测总结
```

---

### Week 7 完成标准 + Part A 总完成标准

**Week 7**：
- [ ] 能从 Loss 曲线形态判断训练问题（4 种情况）
- [ ] 理解并能用 weight_decay 做 L2 正则化
- [ ] 理解 Dropout 是什么、为什么训练/推理行为不同
- [ ] 能从零实现 Z-Score 归一化类
- [ ] 能调试带多个 bug 的训练代码

**Part A 总完成标准**（进入 Part B 前必须全部达到）：

**理论**：
- [ ] 能用自己的话解释：神经元 / 激活函数 / 前向传播 / 反向传播 / 梯度下降 / 学习率 / 过拟合 / 正则化 / 归一化 / 训练集验证集
- [ ] 能解释 ReLU 死神经元问题、为什么 PINN 用 Tanh 不用 ReLU（为 Part B 铺路）
- [ ] 能解释 He 初始化、SGD vs Adam 的差异
- [ ] 能解释 autograd 的 4 个坑、nn.Parameter 是什么
- [ ] 理解为什么 FEM 数据训练必须归一化

**代码**（每项限时完成，不看参考）：
- [ ] 能用 NumPy 手写 2 层 MLP 的前向传播（30 分钟内）
- [ ] 能用 PyTorch + autograd + nn.Module + Adam 写完整训练流程（45 分钟内）
- [ ] 能用 DataLoader 分 batch 训练，做 train/val split
- [ ] 能判断 Loss 曲线是否健康
- [ ] 能调试训练代码（会看错误、会修 bug）

如果上述任何一项未达到，在进入 Part B 之前回头补强对应周。**不要为了赶进度带着不扎实的基础进入 Part B**——Part B 的 autograd 高阶用法和 Part C 的 GNN 会把你 Part A 的短板完全放大。

---

*下一段输出：第一阶段 Part B（Week 8–10）：自动微分深入与物理约束 Loss*