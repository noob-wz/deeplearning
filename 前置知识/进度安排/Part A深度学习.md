# 🔷 Part A · 深度学习地基（Week 1–6）

> **结束时你应达到的状态**：
> 能用 PyTorch 搭建一个 3 层 MLP，输入 6 维特征、输出 3 维预测值，
> 在合成工程数据上训练到 Val MSE < 0.01，能画出 Train/Val 双 Loss 曲线并判断是否过拟合，
> 能做一个至少 5 组的消融实验并写出每组变化带来的影响。
>
> **负荷标准**：工作日每天 1.5–2 小时，周末每天 3–4 小时，每周约 10–14 小时。

---

## 📖 能力等级说明

### 🧠 理论深度

| 等级 | 含义 | 自测方式 |
|------|------|---------|
| 🧠 L1 知道 | 知道这个东西是什么、用在哪 | 能用一句大白话向朋友解释 |
| 🧠 L2 理解 | 理解背后的原理和设计动机 | 能回答"为什么要这样做"，能举出反例 |
| 🧠 L3 推导 | 能在纸上推导核心公式 | 关上书能写出公式并解释每个符号的含义 |

### 💻 实践深度

| 等级 | 含义 | 自测方式 |
|------|------|---------|
| 💻 L1 能读懂 | 看到别人写的代码知道每一行在做什么 | 能在每行代码旁写上中文注释 |
| 💻 L2 能照写 | 打开文档/教程，对着参考能自己敲出来并跑通 | 改两个参数后能预判输出会怎么变，并验证 |
| 💻 L3 能默写 | 关掉所有网页和笔记，从空白文件写出来 | 15 分钟内写完且运行无报错 |
| 💻 L4 能魔改 | 拿到一个新需求，知道改哪里、加什么、删什么 | 遇到没见过的变体任务能独立完成 |

---
---

## Week 1 · 神经元与前向传播

> **本周目标**：理解"一个神经元接收输入、做一次计算、输出一个结果"的完整过程，
> 以及"怎么用一个数字衡量预测结果好不好"。
> 本周不涉及训练（不更新参数），不涉及 PyTorch，全部用 NumPy 手写。

---

### Day 1 ｜ 什么是神经网络

**理论目标**

- [✅] 🧠 L1 知道：神经网络是什么
  - **达标标准**：能向一个完全不懂技术的朋友解释"神经网络就是一个数学函数，你给它输入数据，它给你输出一个预测结果；它的内部有一些可以调节的参数，通过大量数据反复调节这些参数，让预测结果越来越准"
  - **自测**：用自己的话写下这段解释，不超过 3 句话

- [✅] 🧠 L1 知道：参数（权重 w 和偏置 b）的角色
  - **达标标准**：能说出"w 控制输入的重要程度，b 控制输出的偏移；训练的过程就是不断调整 w 和 b 让预测更准"
  - **自测**：如果 y = w×x + b，w=2, b=1, x=3，y 等于多少？（答案：7）

**学习动作**

- [✅] 看完吴恩达 Deep Learning Specialization → Course 1 → Week 1 的全部视频（约 2 小时）
- [✅] 看完后在纸上画出一个神经元的结构图：左边写"输入 x"，中间写"z = w×x + b"，右边写"输出 a = activation(z)"，标清箭头方向

**实践目标**

- [✅] 💻 今天不写代码，专注理解概念

---

### Day 2 ｜ 激活函数

**理论目标**

- [✅] 🧠 L2 理解：为什么需要激活函数
  - **达标标准**：能回答"如果不加激活函数会怎样？"——答案是"不管叠多少层，整个网络等价于一个简单的线性公式 y = Wx + b，无法拟合曲线或其他复杂关系"
  - **自测**：用自己的话写出这个回答

- [✅] 🧠 L2 理解：三种激活函数各自的特点和适用场景
  - **达标标准**：能填完以下表格（不看资料）

| 函数 | 公式 | 输出范围 | 一个优点 | 一个缺点 | 什么时候用 |
|------|------|---------|---------|---------|-----------|
| Sigmoid | 1/(1+e^(-z)) | (0, 1) | 输出可以当概率 | 深层网络梯度越传越小（梯度消失） | 二分类输出层 |
| Tanh | (e^z-e^(-z))/(e^z+e^(-z)) | (-1, 1) | 输出以 0 为中心，比 Sigmoid 好 | 同样有梯度消失 | 较少单独使用 |
| ReLU | max(0, z) | [0, +∞) | 计算极快，不会梯度消失 | 负输入直接归零（"死神经元"） | **隐藏层的默认选择** |

  - **自测**：遮住表格，能否在纸上重新填完？

**实践目标**

- [✅] 💻 L3 能默写：在空白 .py 文件中写出三个激活函数，不看任何参考
  - **达标标准**：写完后运行以下测试全部通过

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh_fn(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

# ---------- 自测 ----------
assert abs(sigmoid(0) - 0.5) < 1e-6,      "sigmoid(0) 应该等于 0.5"
assert abs(tanh_fn(0) - 0.0) < 1e-6,      "tanh(0) 应该等于 0.0"
assert relu(-5) == 0,                       "relu(-5) 应该等于 0"
assert relu(3) == 3,                        "relu(3) 应该等于 3"
print("全部通过 ✓")
```

- [✅] 💻 L2 能照写：用 Matplotlib 画一张图，包含三条激活函数曲线
  - **达标标准**：图中必须有——标题 "Activation Functions"、x 轴标签 "z"、y 轴标签 "output"、三条曲线各自的图例（"Sigmoid" / "Tanh" / "ReLU"）、z 的范围从 -5 到 5
  - **自测**：图上能一眼看出 ReLU 在 z<0 时是平的、Sigmoid 的输出始终在 0 到 1 之间

---

### Day 3 ｜ 损失函数

**理论目标**

- [✅] 🧠 L2 理解：损失函数的作用
  - **达标标准**：能说出"损失函数是一个公式，输入是'模型的预测值'和'真实值'，输出是一个数字表示预测有多差；这个数字越小说明预测越准，训练的目标就是让这个数字越来越小"
  - **自测**：用自己的话写下这段解释

- [✅] 🧠 L2 理解：回归任务为什么用 MSE 而不是其他公式
  - **达标标准**：能说出"MSE 用平方惩罚，偏差越大惩罚增长越快——预测偏了 1 惩罚是 1，偏了 10 惩罚是 100，这会让模型更努力纠正大偏差"
  - **自测**：如果换成绝对值（MAE），有什么不同？（答案：MAE 对大偏差和小偏差同等对待，不会特别惩罚大偏差）

- [✅] 🧠 L3 推导：MSE 公式
  - **达标标准**：关上所有资料，在纸上写出公式，并能指出每个符号的含义

```
MSE = (1/N) × Σᵢ₌₁ᴺ (ŷᵢ − yᵢ)²

N = 样本数量
ŷᵢ = 第 i 个样本的预测值
yᵢ = 第 i 个样本的真实值
```

  - **自测**：手算——如果有 2 个样本，预测值分别是 [3.5, 2.0]，真实值是 [3.0, 2.0]，MSE = ?（答案：(0.5² + 0²) / 2 = 0.125）

**实践目标**

- [✅] 💻 L3 能默写：在空白文件中写出 MSE 函数，不看参考
  - **达标标准**：以下测试全部通过

```python
import numpy as np

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# ---------- 自测 ----------
assert abs(mse_loss(np.array([3.5, 2.0]), np.array([3.0, 2.0])) - 0.125) < 1e-6
assert abs(mse_loss(np.array([1.0, 1.0]), np.array([1.0, 1.0])) - 0.0) < 1e-6
print("全部通过 ✓")
```

---

### Day 4 ｜ 前向传播

**理论目标**

- [✅] 🧠 L2 理解：前向传播是什么
  - **达标标准**：能说出"前向传播就是把输入数据从左到右过一遍网络，经过线性变换 z = w×x+b、激活函数 a = relu(z)、最后和真实值算损失 loss = MSE(a, y)，这整个过程叫一次前向传播；此时参数 w 和 b 不会变化，只是算出当前参数下的预测有多差"
  - **自测**：前向传播会改变 w 和 b 吗？（答案：不会，只有梯度下降才会改变参数）

**实践目标**

- [✅] 💻 L3 能默写：用 NumPy 对一组数据执行一次前向传播（不含训练循环），代码如下，达标标准是能从空白文件写出来并运行通过
  - **注意**：这里只做"一次"前向传播，不做训练（不更新参数）

```python
import numpy as np

# 准备数据：5 个样本，每个样本 1 个特征
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # 真实关系：y = 2x

# 初始化参数（随便给一个初始值）
w = 0.5
b = 0.1

# 前向传播三步
z = w * x + b                      # 第一步：线性变换
a = np.maximum(0, z)                # 第二步：ReLU 激活（此处全为正所以 a = z）
loss = np.mean((a - y_true) ** 2)   # 第三步：算 MSE 损失

print(f"预测值: {a}")        # 应该是 [0.6, 1.1, 1.6, 2.1, 2.6]
print(f"真实值: {y_true}")   # 应该是 [2, 4, 6, 8, 10]
print(f"MSE 损失: {loss:.4f}")  # 应该很大，因为 w=0.5 远离真实的 w=2
```

  - **自测**：预测值和真实值差距大吗？为什么？（答案：差距很大，因为 w=0.5 而真实的 w 应该是 2.0）

- [✅] 💻 L2 能照写：用 sklearn 生成更复杂的回归数据，并对这组数据执行一次前向传播
  - **达标标准**：能运行以下代码并理解每一行的作用，运行后能说出"这组数据有 100 个样本，每个样本 1 个特征，我用随机的 w 和 b 做了一次前向传播，算出来的 MSE 很大是因为参数还没训练"

```python
from sklearn.datasets import make_regression
import numpy as np

# 生成 100 个样本，1 个特征，加少量噪声
x, y_true = make_regression(n_samples=100, n_features=1, noise=5, random_state=42)
x = x.flatten()  # 变成一维数组

# 随机初始参数
w = np.random.randn()
b = np.random.randn()

# 前向传播
y_pred = w * x + b
loss = np.mean((y_pred - y_true) ** 2)
print(f"初始 MSE: {loss:.2f}（参数还没训练，所以很大）")
```

---

### Day 5–6（周末）｜ 巩固与可视化

**实践目标**

- [✅] 💻 L2 能照写：把 Day 1–4 的所有代码整合到一个 Jupyter notebook 中
  - **达标标准**：notebook 中至少有 5 个 cell，每个 cell 顶部有一行 Markdown 说明该 cell 在做什么（例如 `## 激活函数实现与可视化`）；运行 Restart & Run All 后全部 cell 无报错

- [✅] 💻 L2 能照写：画出"预测值 vs 真实值"的散点图
  - **达标标准**：使用 Day 4 中 sklearn 生成的数据，横轴是"True y（真实值）"，纵轴是"Predicted y（预测值）"，图中还画一条红色虚线 y=x 作为参考（如果预测完美，所有点应该落在这条线上），图有标题 "Prediction vs Ground Truth (Before Training)"
  - **自测**：看图——点是不是散得很开、远离红色虚线？这是对的，因为参数还没训练

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, s=10, alpha=0.7, label='Samples')
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()], 'r--', label='Ideal y=x')
plt.xlabel('True y')
plt.ylabel('Predicted y')
plt.title('Prediction vs Ground Truth (Before Training)')
plt.legend()
plt.show()
```

- [✅] 💻 L2 能照写：画出"不同 w 值对预测结果的影响"
  - **达标标准**：在同一张图上画 3 条线——w=0.5 / w=1.0 / w=2.0 时的预测值 y_pred = w×x + b（b 固定为 0），横轴是 x，纵轴是 y_pred，三条线用不同颜色和图例区分，图有标题 "Effect of Weight w on Prediction"
  - **自测**：能看出 w 越大斜率越大吗？这就是"w 控制输入特征的重要程度(线性回归和逻辑回归且特征尺度一致的前提下成立，深度学习不成立)"的直观含义

**✅ 本周产出**

- [✅] 一个 notebook 文件 `week01_forward_propagation.ipynb`
- [✅] 内容包含：三个激活函数代码 + 激活函数对比图 + MSE 函数代码 + 前向传播代码 + 预测vs真实散点图 + 不同 w 的对比图
- [✅] 所有 cell 有 Markdown 说明，Restart & Run All 无报错

---

## Week 2 · 梯度下降——让模型"学习"

> **本周目标**：理解"梯度下降是怎么一步步把 w 和 b 调到正确值的"，
> 并亲手实现一个完整的线性回归训练循环，看到 Loss 从大变小。
> 仍然用 NumPy，不涉及 PyTorch。

---

### Day 1 ｜ 梯度的直觉

**理论目标**

- [✅] 🧠 L2 理解：梯度是什么
  - **达标标准**：能说出"梯度 = Loss 对参数的导数；它是一个数字，告诉你'如果参数增大一点点，Loss 会增大多少'；梯度为正说明参数应该减小，梯度为负说明参数应该增大"
  - **自测**：如果 ∂L/∂w = 5（正数），w 应该增大还是减小？（答案：减小，因为往梯度反方向走才能让 Loss 下降）

- [✅] 🧠 L2 理解：学习率 η 的角色
  - **达标标准**：能说出"学习率控制每次调参数走多大一步——太大会在最优值附近来回震荡甚至 Loss 越来越大，太小则需要非常多步才能到达最优值"
  - **自测**：如果 Loss 曲线剧烈震荡上下跳动，最可能的原因是什么？（答案：学习率太大）

**学习动作**

- [✅] 看吴恩达 Course 1 Week 2 中关于梯度下降的视频部分（约 1 小时）

**实践目标**

- [✅] 💻 今天不写代码，专注把梯度的直觉理解透

---

### Day 2 ｜ 参数更新公式

**理论目标**

- [✅] 🧠 L3 推导：参数更新公式
  - **达标标准**：关上所有资料，在纸上写出以下公式并能解释每个符号

```
w_new = w_old − η × ∂L/∂w
b_new = b_old − η × ∂L/∂b

其中：
η（学习率）= 控制步长大小的正数，通常取 0.001 ~ 0.1 之间
∂L/∂w = Loss 对 w 的偏导数（梯度）
∂L/∂b = Loss 对 b 的偏导数（梯度）
```

  - **自测**：为什么是**减去** η × 梯度而不是加上？（答案：因为梯度指向 Loss 增大最快的方向，我们要往 Loss 减小的方向走，所以取反方向）

- [✅] 🧠 L3 推导：线性回归 y_pred = w×x + b 使用 MSE 时的梯度公式
  - **达标标准**：能在纸上推导出以下两个公式

```
∂L/∂w = (2/N) × Σ(y_pred − y_true) × x
∂L/∂b = (2/N) × Σ(y_pred − y_true)
```

  - **自测**：如果所有样本的 y_pred 恰好等于 y_true，这两个梯度等于多少？（答案：都等于 0，说明参数已经最优，不需要再调了）

**实践目标**

- [✅] 💻 今天不写完整训练循环，只验证梯度公式的正确性
  - **达标标准**：用 Day 4 的数据（x=[1,2,3,4,5], y_true=[2,4,6,8,10], w=0.5, b=0.1），手算 ∂L/∂w 和 ∂L/∂b 的值，然后用代码验证手算结果是否一致

```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
w, b = 0.5, 0.1

y_pred = w * x + b
dw = np.mean(2 * (y_pred - y_true) * x)
db = np.mean(2 * (y_pred - y_true))

print(f"∂L/∂w = {dw:.4f}")  # 用计算器验证是否和你手算一致
print(f"∂L/∂b = {db:.4f}")
```

---

### Day 3–4 ｜ 完整训练循环

**理论目标**

- [✅] 🧠 L2 理解：训练循环在做什么
  - **达标标准**：能说出"一次训练循环包含三步：(1) 前向传播——用当前参数算预测值和 Loss；(2) 计算梯度——算 Loss 对每个参数的导数；(3) 更新参数——沿梯度反方向走一小步。重复这三步几百次，参数就会慢慢逼近最优值"
  - **自测**：一次循环中，参数只更新几次？（答案：1 次）

**实践目标**

- [✅] 💻 L3 能默写 ⭐：关掉所有参考，从空白文件写出完整线性回归训练循环
  - **达标标准**：运行后 Loss 从初始值（约几十）下降到 < 0.1，最终学到的 w 接近 2.0、b 接近 0.0

```python
import numpy as np

# 数据：y = 2x，5 个样本
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

# 初始参数
w = 0.0
b = 0.0
lr = 0.01  # 学习率

# 记录每个 epoch 的 Loss 用于画图
losses = []

for epoch in range(500):
    # 第一步：前向传播
    y_pred = w * x + b
    loss = np.mean((y_pred - y_true) ** 2)
    losses.append(loss)

    # 第二步：计算梯度
    dw = np.mean(2 * (y_pred - y_true) * x)
    db = np.mean(2 * (y_pred - y_true))

    # 第三步：更新参数
    w = w - lr * dw
    b = b - lr * db

print(f"训练后: w = {w:.4f}（应接近 2.0）, b = {b:.4f}（应接近 0.0）")
print(f"最终 Loss: {losses[-1]:.6f}（应该非常小）")
```

  - **自测**：w 最终接近 2.0 了吗？如果不接近（比如只到 1.5），最可能的原因是什么？（答案：epoch 不够或学习率太小）

- [✅] 💻 L3 能默写：画 Loss 下降曲线
  - **达标标准**：横轴是 "Epoch"（0 到 500），纵轴是 "MSE Loss"，能看到曲线从高处快速下降然后趋于平缓，标题为 "Training Loss Curve"

```python
import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.show()
```

  - **自测**：曲线是不是从高处下降然后趋平？如果 Loss 一直不降或者上升，说明代码有 bug 或者学习率选错了

---

### Day 5–6（周末）｜ 学习率实验

**理论目标**

- [✅] 🧠 L2 理解：不同学习率的行为
  - **达标标准**：能描述以下四种情况各自对应什么学习率

| Loss 曲线表现 | 对应的学习率 |
|-------------|------------|
| 几乎不动，下降极慢 | 太小（如 0.001） |
| 稳步下降，最终趋平 | 合适（如 0.01） |
| 快速下降但有轻微震荡 | 偏大（如 0.1） |
| 剧烈震荡或 Loss 越来越大 | 太大（如 1.0） |

**实践目标**

- [✅] 💻 L4 能魔改：用四种学习率分别训练，把 4 条 Loss 曲线画在同一张图上
  - **达标标准**：一张图中有 4 条不同颜色的曲线，图例分别标注 "lr=0.001" / "lr=0.01" / "lr=0.1" / "lr=1.0"，横轴 "Epoch"，纵轴 "MSE Loss"，标题 "Learning Rate Comparison"
  - **自测**：4 条曲线的行为是否和上面表格描述的一致？
  - **具体做法**：把 Day 3 的训练循环封装成函数 `train(lr, epochs=500)`，返回 losses 列表，然后分别调用 4 次

```python
def train(lr, epochs=500):
    w, b = 0.0, 0.0
    losses = []
    for epoch in range(epochs):
        y_pred = w * x + b
        loss = np.mean((y_pred - y_true) ** 2)
        losses.append(loss)
        dw = np.mean(2 * (y_pred - y_true) * x)
        db = np.mean(2 * (y_pred - y_true))
        w -= lr * dw
        b -= lr * db
    return losses

for lr in [0.001, 0.01, 0.1, 1.0]:
    plt.plot(train(lr), label=f'lr={lr}')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('Learning Rate Comparison')
    plt.legend()
    plt.show()
```

- [✅] 💻 L2 能照写：在 notebook 中写一段 3–5 句话的文字分析
  - **达标标准**：文字中必须包含——(1) 哪个学习率收敛最快；(2) 哪个学习率 Loss 发散了或震荡了；(3) 你认为对这组数据最合适的学习率是哪个，为什么

**✅ 本周产出**

- [✅] 一个 notebook 文件 `week02_gradient_descent.ipynb`
- [✅] 内容包含：梯度手算验证 + 完整训练循环（Loss 从大降到接近 0）+ Loss 下降曲线 + 4 种学习率对比图 + 文字分析
- [✅] 所有 cell 有 Markdown 说明，Restart & Run All 无报错

---

## Week 3 · 多层网络与反向传播

> **本周目标**：理解"多层网络能拟合非线性关系"这件事，
> 并用 NumPy 实现一个 2 层 MLP 把 sin(x) 曲线拟合出来。
> 反向传播只需理解直觉（"误差信号逐层回传"），不需要手推完整矩阵公式。

---

### Day 1 ｜ 为什么需要多层

**理论目标**

- [✅] 🧠 L2 理解：单层网络 y = w×x + b 只能拟合直线，无法拟合 sin(x) 这样的曲线
  - **达标标准**：能说出"单层线性模型的输出永远是直线，但真实世界中很多关系是弯的（比如应力-应变曲线、位移场），所以需要多层+激活函数引入非线性"
  - **自测**：如果有一组数据是 y = x²，单层线性模型能拟合吗？（答案：不能，因为 x² 是曲线）

- [✅] 🧠 L1 知道：隐藏层的宽度和深度
  - **达标标准**：能说出"宽度 = 一层有多少个神经元，影响每层能学多少特征；深度 = 有多少层，影响能学多复杂的关系；一般来说更宽或更深 = 能力更强但也更容易过拟合"

**学习动作**

- [✅] 看吴恩达 Course 1 Week 3（浅层神经网络）
- [✅] 在纸上画一个 2 层网络：输入 1 个特征 → 隐藏层 8 个神经元 → 输出 1 个值，标出 W1(8×1)、b1(8×1)、W2(1×8)、b2(1×1) 的维度

---

### Day 2 ｜ 反向传播的直觉

**理论目标**

- [✅] 🧠 L2 理解：反向传播在做什么
  - **达标标准**：能说出"前向传播从左到右算出预测和 Loss；反向传播从右到左把 Loss 的'责任'分配给每一层的参数——越靠近输出的参数分到的责任越直接，越靠近输入的参数分到的责任经过了更多中间环节（链式法则）"
  - **自测**：反向传播算出来的东西是什么？（答案：每个参数的梯度 ∂L/∂W 和 ∂L/∂b）

- [✅] 🧠 ⚠️ 不需要 L3：不需要在纸上推导多层矩阵的链式法则完整公式。理解"信号从输出到输入逐层传递"的方向就够了。下周学 PyTorch 后 `loss.backward()` 会自动帮你完成这一步。

**学习动作**

- [✅] 看吴恩达 Course 1 Week 4（深层网络）中关于反向传播的视频
- [✅] 看李宏毅 Lecture 1 中关于 Gradient Descent 的讲解

**实践目标**

- [✅] 💻 L1 能读懂：看一段别人写的 2 层网络反向传播代码（网上搜"NumPy 2 layer neural network from scratch"），能在每一行旁边写上注释说明"这一行在算哪个梯度"
  - **达标标准**：至少能标注出 `dW2`（输出层权重梯度）和 `dW1`（隐藏层权重梯度）分别在哪一行计算

---

### Day 3–4 ｜ NumPy 实现 2 层 MLP 拟合 sin(x)

**实践目标**

- [ ] 💻 L2 能照写：用 NumPy 搭建一个 2 层 MLP，在 sin(x) 数据上训练，画出拟合曲线
  - **达标标准**：
    1. 数据：x 从 0 到 2π 取 200 个均匀点，y = sin(x) + 0.1×随机噪声
    2. 网络结构：输入层 1 个特征 → 隐藏层 32 个神经元 + ReLU → 输出层 1 个值
    3. 训练：500 个 epoch，学习率 0.01，Loss 最终下降到 < 0.05
    4. 画一张图：蓝色线是真实的 sin(x)，红色虚线是模型的预测，横轴 "x"，纵轴 "y"，标题 "2-Layer MLP Fitting sin(x)"，图例 "True" / "Predicted"
  - **自测**：红色虚线是否大致贴合蓝色线？如果完全偏离，检查学习率、epoch 数和网络宽度

```python
# 数据
x = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
y = np.sin(x) + 0.1 * np.random.randn(200, 1)

# 网络参数初始化（Xavier 初始化）
np.random.seed(42)
W1 = np.random.randn(1, 32) * np.sqrt(2.0 / 1)   # [1, 32]
b1 = np.zeros((1, 32))                              # [1, 32]
W2 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)   # [32, 1]
b2 = np.zeros((1, 1))                               # [1, 1]

lr = 0.01
for epoch in range(500):
    # 前向传播
    z1 = x @ W1 + b1           # [200, 32]
    a1 = np.maximum(0, z1)     # ReLU
    y_pred = a1 @ W2 + b2      # [200, 1]
    loss = np.mean((y_pred - y) ** 2)

    # 反向传播（套公式即可，不需要自己推导）
    dL_dy_pred = 2 * (y_pred - y) / len(x)
    dW2 = a1.T @ dL_dy_pred
    db2 = dL_dy_pred.sum(axis=0, keepdims=True)
    da1 = dL_dy_pred @ W2.T
    dz1 = da1 * (z1 > 0)       # ReLU 的梯度
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    # 更新
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# 画图
plt.plot(x, y, 'b-', alpha=0.5, label='True sin(x)')
plt.plot(x, y_pred, 'r--', label='Predicted')
plt.xlabel('x'); plt.ylabel('y')
plt.title('2-Layer MLP Fitting sin(x)')
plt.legend()
plt.show()
```

---

### Day 5–6（周末）｜ 隐藏层宽度实验

**实践目标**

- [ ] 💻 L4 能魔改：分别用隐藏层宽度 = 8 / 32 / 128 训练，画三张拟合曲线并排对比
  - **达标标准**：3 张子图（用 `plt.subplot(1,3,1)` 排列），每张图标题标注宽度（如 "Hidden=8"），能看出宽度越大拟合越精细
  - **自测**：宽度=8 时 sin(x) 的峰谷能拟合出来吗？宽度=128 呢？（预期：8 可能粗糙但大致对，128 应该很精细）

**✅ 本周产出**

- [ ] `week03_multilayer_and_backprop.ipynb`
- [ ] 包含：2 层 MLP 的 sin(x) 拟合（Loss < 0.05）+ 拟合曲线图 + 3 种宽度的对比子图

---
---

## Week 4 · PyTorch 入门——Tensor 与 Autograd

> **本周目标**：学会 PyTorch 的四个基础组件——Tensor、Autograd、nn.Module、DataLoader。
> 本周不写完整训练循环（留到 Week 5），只把每个组件单独搞清楚。

---

### Day 1 ｜ Tensor 基础

**理论目标**

- [✅] 🧠 L1 知道：PyTorch Tensor 和 NumPy ndarray 的关系
  - **达标标准**：能说出"Tensor 和 ndarray 几乎一样——都是多维数组、支持相同的运算；但 Tensor 多了两个能力：(1) 可以在 GPU 上运算加速；(2) 可以自动追踪运算过程并算梯度"

**实践目标**

- [✅] 💻 L2 能照写：完成 PyTorch 官方教程中的 "Tensors" 部分（https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html）
  - **达标标准**：能运行以下代码且理解每一行的作用

```python
import torch
import numpy as np

# 创建
a = torch.tensor([1.0, 2.0, 3.0])          # 从列表创建
b = torch.zeros(3, 4)                       # 全零 3×4 矩阵
c = torch.randn(2, 3)                       # 标准正态随机 2×3 矩阵

# 形状操作
d = a.reshape(3, 1)                          # 变成 3 行 1 列
print(f"a 的形状: {a.shape}")                # torch.Size([3])
print(f"d 的形状: {d.shape}")                # torch.Size([3, 1])

# NumPy 互转
np_arr = a.numpy()                           # Tensor → NumPy
tensor_back = torch.from_numpy(np_arr)       # NumPy → Tensor

# 运算
e = a + torch.tensor([10.0, 20.0, 30.0])    # 逐元素加
f = torch.matmul(c, b.T)                     # 矩阵乘法 [2,3] × [4,3]ᵀ → 报错？
# 注意：b 是 [3,4]，b.T 是 [4,3]，c 是 [2,3]，所以 c @ b.T 是 [2,3]×[3,4] → 错了
# 正确：c @ b 才对 → [2,3] × [3,4] = [2,4]
```

  - **自测**：`torch.randn(2, 3)` 创建的 Tensor 形状是什么？（答案：2 行 3 列）

---

### Day 2 ｜ Autograd 自动求导

**理论目标**

- [✅] 🧠 L2 理解：`requires_grad=True` 的含义
  - **达标标准**：能说出"给一个 Tensor 加上 `requires_grad=True`，就是告诉 PyTorch '请追踪这个变量参与的所有运算'，这样后面调 backward() 时才能算出梯度"

- [✅] 🧠 L2 理解：`y.backward()` 的含义
  - **达标标准**：能说出"调用 `y.backward()` 后，PyTorch 会自动用链式法则算出 y 对所有 requires_grad=True 变量的梯度，结果存在每个变量的 `.grad` 属性里"

**实践目标**

- [✅] 💻 L3 能默写：关掉所有参考，从空白文件写出以下代码并运行正确

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b    # y = 2*3 + 1 = 7

y.backward()      # 自动算梯度

print(f"y = {y.item()}")         # 7.0
print(f"∂y/∂w = {w.grad.item()}")  # 3.0（因为 ∂y/∂w = x = 3）
print(f"∂y/∂b = {b.grad.item()}")  # 1.0（因为 ∂y/∂b = 1）
```

  - **自测**：为什么 ∂y/∂w = 3.0？（答案：y = w×x + b，对 w 求导得 x，而 x=3.0）
  - **自测**：x 有 `.grad` 属性吗？（答案：没有，因为 x 没有设 `requires_grad=True`）

- [✅] 💻 L2 能照写：验证更复杂的函数 y = w² × x + b
  - **达标标准**：能预测 ∂y/∂w 应该等于 2wx（手算），然后用代码验证 `.grad` 的值和手算一致

---

### Day 3 ｜ nn.Module——定义模型

**理论目标**

- [✅] 🧠 L2 理解：nn.Module 是什么
  - **达标标准**：能说出"nn.Module 是 PyTorch 中定义模型的标准方式——你定义一个类继承 nn.Module，在 `__init__` 里声明每一层（如 nn.Linear），在 `forward` 里写数据从输入到输出的计算过程"

- [✅] 🧠 L2 理解：nn.Linear(in, out) 是什么
  - **达标标准**：能说出"nn.Linear(3, 16) 就是一个全连接层，输入 3 维，输出 16 维，内部自动创建了权重 W(16×3) 和偏置 b(16)，做的运算就是 z = Wx + b"

**实践目标**

- [✅] 💻 L3 能默写 ⭐：关掉所有参考，写出以下 MLP 定义并运行通过
  - **达标标准**：代码无报错，且 `model(test_input)` 能输出一个 (5, 1) 形状的 Tensor

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 回归任务最后一层不加激活函数

# 验证
model = MLP(in_dim=1, hid_dim=64, out_dim=1)
test_input = torch.randn(5, 1)       # 5 个样本，每个 1 维
test_output = model(test_input)
print(f"输入形状: {test_input.shape}")    # [5, 1]
print(f"输出形状: {test_output.shape}")   # [5, 1]
print(f"总参数量: {sum(p.numel() for p in model.parameters())}")
```

  - **自测**：为什么 `self.fc3` 后面没有加 `torch.relu()`？（答案：回归任务的输出可以是任意实数，加 ReLU 会把负数截断为 0，限制了输出范围）

---

### Day 4 ｜ DataLoader——分批喂数据

**理论目标**

- [✅] 🧠 L2 理解：为什么需要 DataLoader
  - **达标标准**：能说出"如果数据有 10000 个样本，一次全部送进模型会占用大量内存；DataLoader 帮你把数据切成小批次（batch），每个 batch 比如 32 个样本，模型一次只处理一个 batch，既省内存又能加速训练"

- [✅] 🧠 L1 知道：shuffle 的作用
  - **达标标准**：能说出"shuffle=True 让每个 epoch 开始时打乱数据顺序，避免模型记住数据的排列顺序"

**实践目标**

- [✅] 💻 L2 能照写：用 TensorDataset + DataLoader 包装数据，并遍历一个 batch 检查形状
  - **达标标准**：能运行以下代码，理解输出的 batch_x 和 batch_y 的形状为什么是 [32, 1]

```python
from torch.utils.data import TensorDataset, DataLoader

# 模拟 200 个样本
x_data = torch.randn(200, 1)
y_data = torch.randn(200, 1)

dataset = TensorDataset(x_data, y_data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 查看第一个 batch
for batch_x, batch_y in loader:
    print(f"batch_x 形状: {batch_x.shape}")  # [32, 1]
    print(f"batch_y 形状: {batch_y.shape}")  # [32, 1]
    break  # 只看第一个 batch
```

  - **自测**：如果 batch_size=32 且总共 200 个样本，一个 epoch 内模型会更新多少次参数？（答案：⌈200/32⌉ = 7 次，最后一个 batch 只有 200-6×32=8 个样本）

---

### Day 5–6（周末）｜ 整合练习

**实践目标**

- [✅] 💻 L2 能照写：用 PyTorch 重写 Week 3 的 sin(x) 回归——只做前向传播（不写训练循环），观察 MLP 在未训练时的输出
  - **达标标准**：
    1. 生成 sin(x) 数据并转为 Tensor
    2. 用 Day 3 的 MLP 模型做一次前向传播 `y_pred = model(x_tensor)`
    3. 画一张图：蓝色线是真实 sin(x)，红色虚线是模型未训练时的输出，标题 "MLP Output Before Training"
    4. 看图——未训练的 MLP 输出应该是一条乱七八糟的线，完全不像 sin(x)

- [✅] 💻 L2 能照写：整理 NumPy vs PyTorch 对照表
  - **达标标准**：在 notebook 中用 Markdown 表格写出以下对照，每行都有具体的 NumPy 代码和对应的 PyTorch 代码

| 操作 | NumPy 写法 | PyTorch 写法 |
|------|-----------|-------------|
| 创建数组 | `np.array([1, 2, 3])` | `torch.tensor([1.0, 2.0, 3.0])` |
| 全零矩阵 | `np.zeros((3, 4))` | `torch.zeros(3, 4)` |
| 矩阵乘法 | `np.dot(A, B)` 或 `A @ B` | `torch.matmul(A, B)` 或 `A @ B` |
| 逐元素 ReLU | `np.maximum(0, z)` | `torch.relu(z)` |
| 反向传播 | 手写 20 行梯度计算代码 | `loss.backward()` 一行 |
| 参数更新 | `w -= lr * dw` | `optimizer.step()` |

**✅ 本周产出**

- [✅] `week04_pytorch_basics.ipynb`
- [✅] 包含：Tensor 操作练习 + Autograd 验证 + MLP 定义 + DataLoader 用法 + sin(x) 未训练输出图 + 对照表

---

## Week 5 · PyTorch 训练循环与诊断

> **本周目标**：把上周学的四个组件组装成完整的训练循环，
> 在 sin(x) 数据上训练 MLP 到 Loss < 0.01，并学会通过 Loss 曲线判断训练状态。

---

### Day 1 ｜ 训练循环 5 步模板

**理论目标**

- [✅] 🧠 L2 理解：训练循环的 5 步分别在做什么
  - **达标标准**：能逐行解释以下代码中每一步的作用，具体到"这一行的输入是什么、输出是什么、为什么要这样做"

| 步骤 | 代码 | 输入 | 输出/效果 | 为什么需要这一步 |
|------|------|------|---------|---------------|
| ① 前向传播 | `pred = model(x)` | 输入数据 x | 预测值 pred | 用当前参数算预测 |
| ② 算损失 | `loss = criterion(pred, y)` | 预测值和真实值 | 一个标量 loss | 量化预测有多差 |
| ③ 清旧梯度 | `optimizer.zero_grad()` | 无 | 把所有参数的 .grad 清零 | PyTorch 默认累加梯度，不清会越加越大 |
| ④ 反向传播 | `loss.backward()` | loss | 所有参数的 .grad 被填入新梯度 | 算出每个参数该往哪调 |
| ⑤ 更新参数 | `optimizer.step()` | 参数的 .grad | 参数被更新 | 沿梯度方向走一小步 |

  - **自测**：如果忘了写 `zero_grad()` 会怎样？（答案：梯度会不断累加，导致参数更新方向错误，Loss 不降反升）

**实践目标**

- [✅] 💻 L3 能默写 ⭐⭐：关掉所有参考，从空白文件写出以下完整代码，运行后 Loss 从 ~1.0 降到 < 0.01
  - **⚠️ 这段代码会用一整年。建议写完后连续 3 天每天早上默写一遍，直到不看任何东西也能 15 分钟内写完。**

```python
import torch
import torch.nn as nn

# ---- 数据 ----
x_data = torch.linspace(0, 2 * 3.1416, 200).unsqueeze(1)  # [200, 1]
y_data = torch.sin(x_data) + 0.1 * torch.randn_like(x_data)

# ---- 模型 ----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = MLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- 训练 ----
losses = []
for epoch in range(2000):
    pred = model(x_data)              # ① 前向
    loss = criterion(pred, y_data)    # ② 损失
    optimizer.zero_grad()             # ③ 清梯度
    loss.backward()                   # ④ 反向
    optimizer.step()                  # ⑤ 更新
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# ---- 结果 ----
print(f"最终 Loss: {losses[-1]:.6f}")
```

  - **自测**：最终 Loss 是否 < 0.01？如果不是，检查 lr 和 epoch 数

- [✅] 💻 L2 能照写：画拟合曲线——蓝色线 sin(x) 真实值，红色虚线模型预测，标题 "MLP Fitting sin(x) After Training"
  - **自测**：红色虚线是否贴合蓝色线？和 Week 4 的"训练前"对比，差异应该巨大

---

### Day 2 ｜ Train/Val 划分

**理论目标**

- [✅] 🧠 L2 理解：为什么需要验证集
  - **达标标准**：能说出"训练集上 Loss 低不代表模型好，可能只是模型'背住了'训练数据的答案；验证集是模型训练时从未见过的数据，如果验证集 Loss 也低才说明模型真正学到了规律"
  - **自测**：一个模型 Train Loss=0.001 但 Val Loss=5.0，说明什么？（答案：严重过拟合——模型在训练数据上表现完美但在新数据上完全失效）

**实践目标**

- [ ] 💻 L2 能照写：用 `random_split` 把 200 个样本分成 160 个训练 + 40 个验证
  - **达标标准**：分割后两组数据无重叠，训练集 160 个样本、验证集 40 个样本

```python
from torch.utils.data import TensorDataset, random_split

dataset = TensorDataset(x_data, y_data)
train_set, val_set = random_split(dataset, [160, 40])

# 提取 tensor
x_train = torch.stack([s[0] for s in train_set])
y_train = torch.stack([s[1] for s in train_set])
x_val   = torch.stack([s[0] for s in val_set])
y_val   = torch.stack([s[1] for s in val_set])
print(f"训练集: {x_train.shape[0]} 个样本, 验证集: {x_val.shape[0]} 个样本")
```

- [ ] 💻 L4 能魔改：在训练循环中每个 epoch 末尾加上验证集评估
  - **达标标准**：同时记录 train_losses 和 val_losses 两个列表

```python
# 在训练循环的 optimizer.step() 后面加：
model.eval()                         # 切换到评估模式（关 Dropout 等）
with torch.no_grad():                # 不追踪梯度（节省内存）
    val_pred = model(x_val)
    val_loss = criterion(val_pred, y_val)
val_losses.append(val_loss.item())
model.train()                        # 切回训练模式
```

  - **自测**：`model.eval()` 做了什么？（答案：切换到评估模式，关闭 Dropout 等只在训练时使用的行为）
  - **自测**：`torch.no_grad()` 做了什么？（答案：在这个块内的运算不会被追踪梯度，节省内存和计算）

---

### Day 3 ｜ Loss 曲线诊断

**理论目标**

- [ ] 🧠 L2 理解 ⭐：能根据 Train/Val Loss 曲线诊断问题
  - **达标标准**：看到一张 Loss 曲线图后能在 10 秒内给出诊断

| 曲线表现 | 诊断 | 应该怎么做 |
|---------|------|----------|
| Train 和 Val 都在持续下降 | 正常训练中 | 继续训练，看看还能不能更低 |
| Train 在降但 Val 开始上升 | 过拟合 | 加正则化(Weight Decay) / 减小模型 / 加数据 / Early Stopping |
| Train 和 Val 都几乎不动 | 欠拟合 | 增大模型(加宽/加深) / 提高学习率 / 训更多 epoch |

  - **自测**：如果 Train Loss=0.001, Val Loss=0.5 且 Val Loss 在上升，应该怎么做？（答案：过拟合了，试试加 Weight Decay 或减小模型宽度）

**实践目标**

- [ ] 💻 L3 能默写：画 Train/Val 双 Loss 曲线
  - **达标标准**：一张图中两条线（蓝色 "Train Loss"、橙色 "Val Loss"），横轴 "Epoch"，纵轴 "Loss"，标题 "Train vs Validation Loss"，有图例

```python
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.show()
```

- [ ] 💻 L4 能魔改：制造一个过拟合场景，观察 Loss 曲线分叉
  - **具体做法**：只用 20 个训练样本 + 宽度 256 的大模型 + 训练 5000 个 epoch
  - **达标标准**：画出的双 Loss 曲线中，Train Loss 持续降到接近 0，但 Val Loss 先降后升——这就是过拟合的典型表现
  - **自测**：Val Loss 开始上升大约在第几个 epoch？这个点之后训练就应该停止了

---

### Day 4 ｜ 保存与加载模型

**理论目标**

- [ ] 🧠 L1 知道：为什么要保存模型
  - **达标标准**：能说出"训练可能要几小时甚至几天，如果不保存，关机后就要重新训练；保存后随时可以加载继续使用或在其他数据上推理"

**实践目标**

- [ ] 💻 L3 能默写：保存和加载模型权重
  - **达标标准**：保存后重新加载的模型在相同输入上的输出和保存前完全一致

```python
# 保存
torch.save(model.state_dict(), 'my_model.pth')
print("模型已保存")

# 加载（假设你重新创建了模型结构）
model_loaded = MLP()
model_loaded.load_state_dict(torch.load('my_model.pth'))
model_loaded.eval()

# 验证：相同输入应该得到相同输出
with torch.no_grad():
    out_original = model(x_val)
    out_loaded   = model_loaded(x_val)
    diff = torch.abs(out_original - out_loaded).max().item()
    print(f"最大差异: {diff}")  # 应该是 0.0 或非常小的浮点误差
```

---

### Day 5–6（周末）｜ 完整回归项目

**实践目标**

- [ ] 💻 L4 能魔改：把本周所有内容串成一个完整流程
  - **达标标准**：一个 notebook 中完成以下完整链路，每个环节之间有 Markdown 标题分隔

```
1. 生成 sin(x) 数据（200 个样本）
2. Train/Val 划分（160/40）
3. 定义 MLP 模型（1→64→64→1）
4. 训练 2000 个 epoch（记录 Train Loss 和 Val Loss）
5. 画 Train/Val 双 Loss 曲线
6. 画训练后的拟合曲线（True vs Predicted）
7. 保存模型
8. 加载模型并验证输出一致
```

  - **自测**：最终 Train Loss 和 Val Loss 都 < 0.01 吗？拟合曲线贴合吗？

**✅ 本周产出**

- [ ] `week05_training_loop.ipynb`
- [ ] 包含：完整训练循环 + Train/Val 划分 + 双 Loss 曲线 + 过拟合实验 + 拟合曲线 + 模型保存加载

---
---

## Week 6 · 调参实战——优化器、正则化与归一化

> **本周目标**：学会三种改进训练的工具（优化器选择 / 正则化 / 数据归一化），
> 并在一个多输入多输出的回归任务上做一组消融实验。

---

### Day 1 ｜ 优化器对比

**理论目标**

- [ ] 🧠 L2 理解：三种优化器的区别
  - **达标标准**：能填完以下表格（不看资料）

| 优化器 | 一句话核心思想 | 优点 | 缺点 | 什么时候用 |
|--------|-------------|------|------|-----------|
| SGD | 每步只看当前 batch 的梯度 | 简单、理论上泛化好 | 收敛慢、容易卡在局部最小 | 追求极致泛化时 |
| SGD+Momentum | 加上"惯性"——不只看当前梯度，还参考历史方向 | 收敛快、不容易被小坑困住 | 多一个超参数（momentum） | 比 SGD 好但懒得调太多参数时 |
| Adam | 对每个参数自适应调学习率——梯度大的参数步子变小，梯度小的步子变大 | 收敛最快、几乎不用调 lr | 某些情况下泛化略差于 SGD | **默认首选，大多数场景直接用** |

**实践目标**

- [ ] 💻 L3 能默写：创建 Adam 和 SGD 优化器

```python
opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
opt_sgd  = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

- [ ] 💻 L4 能魔改：在 sin(x) 数据上用 Adam 和 SGD 分别训练，把两条 Loss 曲线画在同一张图上
  - **达标标准**：一张图两条线，图例 "Adam lr=0.001" / "SGD lr=0.01 momentum=0.9"，标题 "Optimizer Comparison"
  - **具体做法**：写一个函数 `train_with_optimizer(optimizer_class, lr, **kwargs)`，返回 losses 列表
  - **自测**：Adam 是不是比 SGD 收敛更快？（通常是的）

---

### Day 2 ｜ 正则化

**理论目标**

- [ ] 🧠 L2 理解：Weight Decay（L2 正则化）
  - **达标标准**：能说出"Weight Decay 在 Loss 中额外加上 λ×Σ(w²)，这个惩罚项让权重不能长得太大——大权重意味着模型对某些特征过度依赖，限制权重大小能防止过拟合"
  - **自测**：Weight Decay 的系数 λ 设成 0 和设成 0.1 有什么区别？（答案：0 = 没有正则化；0.1 = 很强的正则化，可能导致模型欠拟合）

- [ ] 🧠 L2 理解：Dropout
  - **达标标准**：能说出"Dropout(p=0.5) 表示训练时每个神经元有 50% 概率被随机'关掉'——这迫使网络不能过度依赖某几个神经元；推理时 Dropout 关闭（`model.eval()` 自动处理），所有神经元都参与"
  - **自测**：训练时 Dropout 开，推理时 Dropout 关——如果推理时也开着会怎样？（答案：输出会有随机性，每次推理结果不一样，这是错误的）

- [ ] 🧠 L2 理解：Early Stopping
  - **达标标准**：能说出"如果 Val Loss 连续 N 个 epoch 都没有比之前最低的更低，就认为模型已经开始过拟合了，应该停止训练并用 Val Loss 最低时的模型参数"

**实践目标**

- [ ] 💻 L3 能默写：在 PyTorch 中加 Weight Decay 只需改一个参数
  - **达标标准**：能从空白写出 `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)`

- [ ] 💻 L2 能照写：在 Week 5 Day 3 的过拟合实验（20 个样本 + 256 宽度模型）上加 weight_decay=1e-3，观察 Val Loss 曲线是否不再上升
  - **达标标准**：画两张 Val Loss 曲线对比——无 Weight Decay 时 Val Loss 先降后升，加了后 Val Loss 保持平稳或持续下降
  - **自测**：加了 Weight Decay 后 Val Loss 的上升是否被抑制了？

- [ ] 💻 L2 能照写：实现 EarlyStopping 类
  - **达标标准**：以下代码运行正确

```python
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience    # 容忍多少个 epoch 没有改善
        self.counter = 0            # 当前已经没有改善的 epoch 数
        self.best = float('inf')    # 目前为止最低的 Val Loss

    def should_stop(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0       # 有改善，计数归零
        else:
            self.counter += 1      # 没改善，计数加一
        return self.counter >= self.patience  # 超过容忍限度就停

# 验证
es = EarlyStopping(patience=3)
assert es.should_stop(1.0) == False  # 第一次，best=1.0
assert es.should_stop(0.5) == False  # 改善了，best=0.5
assert es.should_stop(0.6) == False  # 没改善，counter=1
assert es.should_stop(0.7) == False  # 没改善，counter=2
assert es.should_stop(0.8) == True   # 没改善，counter=3 >= patience
print("EarlyStopping 测试通过 ✓")
```

---

### Day 3 ｜ 数据归一化

**理论目标**

- [ ] 🧠 L2 理解：为什么需要归一化
  - **达标标准**：能说出"工程数据中不同特征的数值范围差异巨大——比如坐标可能在 0~1000mm，弹性模量在 70000~200000MPa，载荷在 10~100N。如果不归一化，数值大的特征会主导梯度方向，小的特征几乎学不到。归一化后所有特征都在差不多的范围内，训练更稳定更快"
  - **自测**：如果训练数据的 x 范围是 [0, 1000]，但测试数据的 x 范围是 [500, 1500]，直接用训练时的 mean 和 std 做归一化会有问题吗？（答案：会有问题——归一化后测试数据会出现训练时没见过的值域，模型可能预测不准，这是一个"数据分布偏移"的问题）

- [ ] 🧠 L3 推导：Z-Score 公式
  - **达标标准**：关上资料在纸上写出公式，能说明每个符号的含义和"反归一化"公式

```
归一化：X̃ = (X − μ) / σ
反归一化：X = X̃ × σ + μ

μ = 训练集上每个特征的均值
σ = 训练集上每个特征的标准差
注意：μ 和 σ 只在训练集上计算，验证集/测试集用训练集的 μ 和 σ
```

  - **自测**：为什么验证集不能用自己的 μ 和 σ？（答案：因为实际部署时你不知道未来数据的分布，只能用训练时计算好的统计量）

**实践目标**

- [ ] 💻 L3 能默写 ⭐：关掉参考写出 ZScoreNormalizer 类，并通过以下测试

```python
class ZScoreNormalizer:
    def fit(self, data):
        """在训练数据上计算 μ 和 σ"""
        self.mean = data.mean(dim=0)
        self.std  = data.std(dim=0) + 1e-8  # 加小数防止除以零

    def transform(self, data):
        """归一化"""
        return (data - self.mean) / self.std

    def inverse(self, data):
        """反归一化：把归一化后的值还原回原始范围"""
        return data * self.std + self.mean

# 验证
normalizer = ZScoreNormalizer()
test_data = torch.tensor([[100.0, 0.3], [200.0, 0.25], [150.0, 0.35]])
normalizer.fit(test_data)
transformed = normalizer.transform(test_data)
recovered = normalizer.inverse(transformed)
assert torch.allclose(test_data, recovered, atol=1e-5), "反归一化后应该和原始数据一致"
print(f"归一化后的均值: {transformed.mean(dim=0)}")  # 应该接近 [0, 0]
print(f"归一化后的标准差: {transformed.std(dim=0)}")  # 应该接近 [1, 1]
print("ZScoreNormalizer 测试通过 ✓")
```

---

### Day 4–5 ｜ 多输入多输出回归 + 消融实验

**理论目标**

- [ ] 🧠 L2 理解：什么是多输入多输出回归
  - **达标标准**：能说出"真实工程数据通常有多个输入特征（比如坐标 x,y,z + 弹性模量 E + 泊松比 ν + 载荷 F = 6 个输入），预测多个输出（比如三个方向的位移 ux, uy, uz = 3 个输出）。MLP 天然支持——把 in_dim 设成 6、out_dim 设成 3 即可"

- [ ] 🧠 L2 理解：什么是消融实验
  - **达标标准**：能说出"消融实验就是每次只改一个变量（比如只改模型宽度），其他所有条件不变，看这一个变化带来了多大影响。这样你就能说清楚'宽度从 32 改到 128 让 Val MSE 降低了 40%'，而不是模糊地说'我调了一下参数效果变好了'"

**实践目标**

- [ ] 💻 L4 能魔改 ⭐：构造模拟工程数据 + 训练 + 消融实验
  - **达标标准**：按以下步骤完成，每步有明确的验证标准

**Step 1：生成数据**

```python
# 6 个输入特征 → 3 个输出
# 物理含义（模拟）：坐标(x,y,z) + 弹性模量(E) + 泊松比(nu) + 载荷(F)
#              → 位移(ux, uy, uz)
n = 1000
X = torch.rand(n, 6)  # 所有特征在 [0,1] 范围

# 用简单公式模拟物理关系（加噪声）
Y = torch.stack([
    X[:, 5] * X[:, 0] / (X[:, 3] + 0.1),   # ux ≈ F*x/E
    X[:, 5] * X[:, 1] / (X[:, 3] + 0.1),   # uy ≈ F*y/E
    X[:, 5] * X[:, 2] / (X[:, 3] + 0.1),   # uz ≈ F*z/E
], dim=1) + 0.01 * torch.randn(n, 3)
```

  - **验证**：打印 X.shape 和 Y.shape，应该是 [1000, 6] 和 [1000, 3]

**Step 2：归一化 + 划分**

```python
# 划分：800 训练 + 200 验证
X_train, X_val = X[:800], X[800:]
Y_train, Y_val = Y[:800], Y[800:]

# 归一化
x_norm = ZScoreNormalizer(); x_norm.fit(X_train)
y_norm = ZScoreNormalizer(); y_norm.fit(Y_train)

X_train_n = x_norm.transform(X_train)
X_val_n   = x_norm.transform(X_val)
Y_train_n = y_norm.transform(Y_train)
Y_val_n   = y_norm.transform(Y_val)
```

**Step 3：训练函数**

```python
def run_experiment(hid_dim, n_layers, use_normalize, optimizer_name, epochs=500):
    """跑一组实验，返回最终 Val MSE"""
    # 根据参数构建模型
    layers = [nn.Linear(6, hid_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hid_dim, hid_dim), nn.ReLU()]
    layers.append(nn.Linear(hid_dim, 3))
    model = nn.Sequential(*layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 选择数据
    xt = X_train_n if use_normalize else X_train
    yt = Y_train_n if use_normalize else Y_train
    xv = X_val_n   if use_normalize else X_val
    yv = Y_val_n   if use_normalize else Y_val

    for epoch in range(epochs):
        pred = model(xt)
        loss = criterion(pred, yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(xv)
        val_mse = criterion(val_pred, yv).item()
    return val_mse
```

**Step 4：消融实验（每次只改一个变量）**

| # | 宽度 | 层数 | 归一化 | 优化器 | Val MSE | 改了什么 |
|---|------|------|--------|--------|---------|---------|
| 1 | 32 | 2 | 无 | Adam | ? | — 基线 |
| 2 | 64 | 2 | 无 | Adam | ? | 宽度 32→64 |
| 3 | 128 | 2 | 无 | Adam | ? | 宽度 64→128 |
| 4 | 128 | 4 | 无 | Adam | ? | 层数 2→4 |
| 5 | 128 | 4 | Z-Score | Adam | ? | 加归一化 |

  - **达标标准**：5 组实验全部跑完，表格填满数字
  - **自测**：哪次变化带来的提升最大？（通常是加归一化那一步）

**Step 5：画预测 vs 真实散点图**

- [ ] 💻 L2 能照写：用最佳配置训练的模型，对验证集做预测（记得反归一化），画三张散点图分别对应 ux / uy / uz
  - **达标标准**：每张图横轴 "True"、纵轴 "Predicted"，红色虚线 y=x，标题标注是哪个输出分量（如 "ux: Predicted vs True"）
  - **自测**：点是否沿 y=x 线分布？如果是说明预测准确

---

### Day 6（周末）｜ 整理 + 写分析

**实践目标**

- [ ] 💻 L2 能照写：在 notebook 末尾用 Markdown 写一段 5–8 句话的消融实验分析
  - **达标标准**：必须包含以下内容
    1. 基线（#1）的 Val MSE 是多少
    2. 提升最大的一步是哪次变化，MSE 降低了百分之多少
    3. 有没有哪次变化反而让效果变差了（如果有）
    4. 你认为下一步还可以尝试什么

**✅ 本周产出**

- [ ] `week06_tuning.ipynb`
- [ ] 包含：优化器对比图 + Weight Decay 对比 + EarlyStopping 实现 + ZScoreNormalizer + 多特征回归 + 5 组消融实验表格 + 散点图 + 文字分析

---
---

## 📊 Part A 自测检查表（Week 1–6 结束后）

关掉所有资料，逐项检查：

**理论**
- [ ] 🧠 能在纸上写出 MSE 公式并解释每个符号的含义（L3）
- [ ] 🧠 能在纸上写出梯度下降参数更新公式并解释为什么"减去"梯度（L3）
- [ ] 🧠 能不看资料填完三种激活函数的对比表格（L2）
- [ ] 🧠 能说出 Adam 和 SGD 各自的核心思想以及"默认选谁"（L2）
- [ ] 🧠 能在纸上写出 Z-Score 公式和反归一化公式，并解释"为什么验证集用训练集的 μ 和 σ"（L3）
- [ ] 🧠 能看 Loss 曲线在 10 秒内判断出"正常 / 过拟合 / 欠拟合"并给出对策（L2）
- [ ] 🧠 能说出训练循环 5 步中每一步的输入、输出、为什么需要（L2）

**实践**
- [ ] 💻 能在 15 分钟内从空白文件默写出 MLP 定义 + 训练循环，运行无报错（L3）
- [ ] 💻 能在 5 分钟内从空白文件默写出 ZScoreNormalizer 类，通过测试（L3）
- [ ] 💻 能对一个多输入多输出的回归任务做完整的消融实验（5 组以上），并写出文字分析（L4）
- [ ] 💻 GitHub 上有 6 个 notebook（week01 到 week06），每个 Restart & Run All 无报错