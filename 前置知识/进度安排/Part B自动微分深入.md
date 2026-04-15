# 🔷 Part B · 自动微分深入（Week 7–9）

> **本部分在整个路线中的位置**：Part A 教会了你用 PyTorch 训练 MLP 做回归。
> Part B 要教你一个 Part A 没涉及的能力——**用 autograd 对网络的输出关于输入求导数**。
> 这个能力是后面 PINN（物理信息嵌入）的核心基础：如果网络输出位移 u(x)，
> 你能自动算出应变 ε = ∂u/∂x，然后检查它是否满足物理方程。
>
> **结束时你应达到的状态**：
> 能用 `torch.autograd.grad()` 对任意网络输出关于输入求一阶和二阶偏导数，
> 能把"物理方程的残差"加入 Loss 中训练网络，
> 并通过实验证明物理约束在数据稀少时能显著提升外推能力。
>
> **负荷标准**：同 Part A，每周 10–14 小时。

---
---

## Week 7 · autograd.grad()——对网络输出求导数

> **本周目标**：搞清楚 `torch.autograd.grad()` 的用法，
> 能对一个 MLP 的输出关于输入求一阶偏导数和二阶偏导数。
> 本周不涉及物理约束 Loss（留到 Week 8），只练"求导"这一个技能。

---

### Day 1 ｜ backward() 和 autograd.grad() 的区别

**理论目标**

- [ ] 🧠 L2 理解：`loss.backward()` 和 `torch.autograd.grad()` 分别做什么
  - **达标标准**：能填完以下表格（不看资料）

| | `loss.backward()` | `torch.autograd.grad(y, x, ...)` |
|---|---|---|
| **算的是什么** | Loss 对模型所有可学习参数（w, b）的梯度 | **y** 对 **x** 的导数（x 可以是任何 Tensor） |
| **结果存在哪** | 存在每个参数的 `.grad` 属性里 | 直接作为返回值返回 |
| **什么时候用** | 训练循环中更新模型参数时 | 需要算"网络输出对网络输入"的导数时（比如算 ∂u/∂x） |
| **一句话区别** | 算"参数该怎么调" | 算"输出随输入怎么变" |

  - **自测**：如果你想知道"模型预测的位移 u 随输入坐标 x 变化有多快"，应该用哪个？（答案：`autograd.grad()`，因为你要算的是输出 u 对输入 x 的导数，不是对模型参数的导数）

- [ ] 🧠 L2 理解：`grad_outputs` 参数的含义
  - **达标标准**：能说出"当 y 是一个向量（不是标量）时，PyTorch 不知道该对 y 的哪个元素求导，`grad_outputs=torch.ones_like(y)` 的意思是'对 y 的每个元素都求导，然后加起来'——数学上等价于求 Σyᵢ 对 x 的导数"
  - **自测**：如果 y 只有一个元素（标量），还需要 `grad_outputs` 吗？（答案：严格来说不需要，但加上 `torch.ones_like(y)` 不会出错，保持写法统一更好）

- [ ] 🧠 L2 理解：`create_graph=True` 参数的含义
  - **达标标准**：能说出"默认情况下 `autograd.grad()` 算完导数后会丢弃计算图（省内存）；加上 `create_graph=True` 会保留计算图，这样你可以对导数再求导（算二阶导数），或者让导数参与 Loss 计算并反向传播"
  - **自测**：如果你只需要一阶导数、且这个导数不参与后续 Loss 计算，需要 `create_graph=True` 吗？（答案：不需要，省掉能节省内存）

**实践目标**

- [ ] 💻 L3 能默写 ⭐：关掉所有参考，从空白文件写出以下代码并运行通过

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x    # y = x² + 3x，所以 dy/dx = 2x + 3

dydx = torch.autograd.grad(
    outputs=y,                        # 对谁求导：y
    inputs=x,                         # 关于谁求导：x
    grad_outputs=torch.ones_like(y),  # y 是向量，需要指定
    create_graph=True                 # 保留计算图（后面要算二阶导时需要）
)[0]                                  # 返回的是元组，取第一个元素

# ---------- 验证 ----------
# dy/dx = 2x + 3，代入 x=[1,2,3] 应该得到 [5, 7, 9]
expected = 2 * x + 3
assert torch.allclose(dydx, expected, atol=1e-5), f"期望 {expected}，得到 {dydx}"
print(f"dy/dx = {dydx.tolist()}")   # [5.0, 7.0, 9.0]
print("验证通过 ✓")
```

  - **自测**：能否解释 `[0]` 为什么需要？（答案：`autograd.grad()` 返回一个元组，即使只有一个输入也是元组，所以要取 `[0]`）

---

### Day 2 ｜ 用 autograd 验证已知函数的导数

**理论目标**

- [ ] 🧠 L2 理解：autograd 能对任何可微的 PyTorch 运算自动求精确导数（不是数值近似），所以可以用"已知导数的函数"来验证它是否正确
  - **达标标准**：能说出"sin(x) 的导数是 cos(x)，exp(x) 的导数是 exp(x)，x³ 的导数是 3x²——用 autograd 算出来的结果应该和这些精确值一致"

**实践目标**

- [ ] 💻 L2 能照写：验证三个函数的导数，每个都要用 assert 检查 autograd 结果和手算一致
  - **达标标准**：以下三段代码全部运行通过

**验证 1：y = sin(x) → dy/dx = cos(x)**

```python
x = torch.linspace(0, 6.28, 50, requires_grad=True)
y = torch.sin(x)
dydx = torch.autograd.grad(y, x, torch.ones_like(y))[0]
expected = torch.cos(x)
assert torch.allclose(dydx, expected, atol=1e-4), "sin 的导数应该是 cos"
print("sin → cos 验证通过 ✓")
```

**验证 2：y = exp(x) → dy/dx = exp(x)**

```python
x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
y = torch.exp(x)
dydx = torch.autograd.grad(y, x, torch.ones_like(y))[0]
expected = torch.exp(x)
assert torch.allclose(dydx, expected, atol=1e-4), "exp 的导数应该还是 exp"
print("exp → exp 验证通过 ✓")
```

**验证 3：y = x³ → dy/dx = 3x²**

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 3
dydx = torch.autograd.grad(y, x, torch.ones_like(y))[0]
expected = 3 * x ** 2
assert torch.allclose(dydx, expected, atol=1e-4), "x³ 的导数应该是 3x²"
print("x³ → 3x² 验证通过 ✓")
```

- [ ] 💻 L2 能照写：画一张对比图验证 sin → cos
  - **达标标准**：一张图中两条线——蓝色实线标注 "autograd dy/dx"、绿色虚线标注 "cos(x) (expected)"，横轴 "x"，纵轴 "dy/dx"，标题 "Verify: d(sin(x))/dx = cos(x)"。两条线应该完全重合。
  - **自测**：两条线重合了吗？如果没重合说明代码有 bug

---

### Day 3 ｜ 二阶导数

**理论目标**

- [ ] 🧠 L2 理解：怎么用 autograd 算二阶导数
  - **达标标准**：能说出"先对 y 关于 x 求一次导得到 dy/dx，然后对 dy/dx 再关于 x 求一次导就得到 d²y/dx²。两次调用 `autograd.grad()`，**第一次必须加 `create_graph=True`**（否则计算图被丢弃了，第二次求导会报错）"
  - **自测**：如果第一次求导时忘了 `create_graph=True`，第二次求导会怎样？（答案：报错——RuntimeError: Trying to backward through the graph a second time）

**实践目标**

- [ ] 💻 L3 能默写 ⭐：关掉参考，从空白文件写出二阶导数计算并验证

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 3    # y = x³

# 一阶导数：dy/dx = 3x²
dy = torch.autograd.grad(
    y, x, torch.ones_like(y),
    create_graph=True       # ← 必须为 True，否则无法算二阶导
)[0]

# 二阶导数：d²y/dx² = 6x
d2y = torch.autograd.grad(
    dy, x, torch.ones_like(dy),
    create_graph=True       # 如果后续还要用这个二阶导数参与 Loss，也要 True
)[0]

# ---------- 验证 ----------
assert torch.allclose(dy, 3 * x**2, atol=1e-4), f"一阶导数应该是 3x²={3*x**2}，得到 {dy}"
assert torch.allclose(d2y, 6 * x, atol=1e-4),   f"二阶导数应该是 6x={6*x}，得到 {d2y}"
print(f"dy/dx = {dy.tolist()}")     # [3.0, 12.0, 27.0]
print(f"d²y/dx² = {d2y.tolist()}")  # [6.0, 12.0, 18.0]
print("二阶导数验证通过 ✓")
```

- [ ] 💻 L2 能照写：验证 y = sin(x) 的二阶导数 = -sin(x)
  - **达标标准**：以下代码运行通过

```python
x = torch.linspace(0, 6.28, 50, requires_grad=True)
y = torch.sin(x)
dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
d2y = torch.autograd.grad(dy, x, torch.ones_like(dy))[0]
expected = -torch.sin(x)
assert torch.allclose(d2y, expected, atol=1e-3), "sin 的二阶导数应该是 -sin"
print("d²(sin)/dx² = -sin 验证通过 ✓")
```

---

### Day 4 ｜ 对 MLP 输出求偏导数

**理论目标**

- [ ] 🧠 L2 理解：autograd 不仅能对简单公式求导，也能对**神经网络的输出**关于**网络的输入**求导
  - **达标标准**：能说出"MLP 本质上是一个复杂的可微函数 u = f(x₁, x₂)，autograd 可以自动算出 ∂u/∂x₁ 和 ∂u/∂x₂——不管 MLP 有多少层、多复杂，链式法则都会自动处理"
  - **自测**：一个 MLP 有 3 层、每层 128 个神经元，autograd 能对它的输出关于输入求导吗？（答案：能，autograd 不关心网络多复杂）

- [ ] 🧠 L2 理解：多输入时偏导数怎么取
  - **达标标准**：能说出"`autograd.grad(u, inputs)` 返回的 grad 形状和 inputs 一样；如果 inputs 是 [N, 2]（N 个样本、2 维输入），那 grad 也是 [N, 2]——第 0 列是 ∂u/∂x₁，第 1 列是 ∂u/∂x₂"

**实践目标**

- [ ] 💻 L4 能魔改 ⭐：对一个 MLP 的输出关于输入求偏导数
  - **达标标准**：以下代码运行无报错，且能打印出 ∂u/∂x₁ 和 ∂u/∂x₂ 的形状都是 [10]

```python
import torch
import torch.nn as nn

# 一个简单的 MLP：输入 2 维 (x1, x2)，输出 1 维 u
class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

model = SmallMLP()

# 10 个样本，每个有 2 维输入 (x1, x2)
inputs = torch.randn(10, 2, requires_grad=True)

# 前向传播
u = model(inputs)   # u 的形状：[10, 1]

# 求 ∂u/∂inputs
grads = torch.autograd.grad(
    outputs=u,
    inputs=inputs,
    grad_outputs=torch.ones_like(u),
    create_graph=True
)[0]   # grads 形状：[10, 2]

du_dx1 = grads[:, 0]   # ∂u/∂x₁，形状 [10]
du_dx2 = grads[:, 1]   # ∂u/∂x₂，形状 [10]

print(f"inputs 形状: {inputs.shape}")    # [10, 2]
print(f"u 形状: {u.shape}")              # [10, 1]
print(f"grads 形状: {grads.shape}")      # [10, 2]
print(f"∂u/∂x₁ 形状: {du_dx1.shape}")   # [10]
print(f"∂u/∂x₂ 形状: {du_dx2.shape}")   # [10]
print(f"∂u/∂x₁ 前 3 个值: {du_dx1[:3].tolist()}")
```

  - **自测**：`du_dx1` 的值每次运行都不一样，这正常吗？（答案：正常——模型参数是随机初始化的，输入也是随机的，所以导数值每次都不同。重点是代码能跑通、形状正确。）

- [ ] 💻 L4 能魔改：在上面代码的基础上，进一步求 ∂²u/∂x₁²（对 x₁ 的二阶偏导数）
  - **达标标准**：以下代码运行无报错

```python
# ∂²u/∂x₁²：对 du_dx1 再关于 inputs 求导，取第 0 列
du_dx1_unsq = du_dx1.unsqueeze(1)   # 变成 [10, 1] 才能传给 grad_outputs
grads2 = torch.autograd.grad(
    outputs=du_dx1_unsq,
    inputs=inputs,
    grad_outputs=torch.ones_like(du_dx1_unsq),
    create_graph=True
)[0]
d2u_dx1dx1 = grads2[:, 0]   # ∂²u/∂x₁²

print(f"∂²u/∂x₁² 形状: {d2u_dx1dx1.shape}")   # [10]
print(f"∂²u/∂x₁² 前 3 个值: {d2u_dx1dx1[:3].tolist()}")
```

  - **自测**：为什么要 `du_dx1.unsqueeze(1)` 变成 [10,1]？（答案：`grad_outputs` 的形状必须和 `outputs` 一致，`du_dx1` 是 [10] 但 `torch.ones_like` 需要匹配形状。或者也可以直接用 `torch.ones(10)` 作为 grad_outputs。）

---

### Day 5–6（周末）｜ 巩固练习

**实践目标**

- [ ] 💻 L3 能默写（闭卷练习 1）：关掉所有参考，从空白写出"对 y = sin(x) 求一阶和二阶导数并用 assert 验证"的完整代码
  - **达标标准**：15 分钟内写完，运行无报错，assert 全部通过
  - **如果写不出来**：回去看 Day 2 和 Day 3 的代码，再练一遍

- [ ] 💻 L3 能默写（闭卷练习 2）：关掉所有参考，从空白写出"定义一个 2 输入 1 输出的 MLP，对输出关于输入求偏导数 ∂u/∂x₁ 和 ∂u/∂x₂"的完整代码
  - **达标标准**：15 分钟内写完，运行无报错，打印出的形状正确

- [ ] 💻 L2 能照写：把本周所有代码整理到一个 notebook 中
  - **达标标准**：notebook 标题 "Week 7: Autograd Deep Dive"，至少 5 个 cell，每个 cell 顶部有 Markdown 说明，Restart & Run All 无报错

**✅ 本周产出**

- [ ] `week07_autograd_deep.ipynb`
- [ ] 包含：backward vs grad 的区别总结 + 三个函数的导数验证（含 assert）+ sin→cos 对比图 + 二阶导数验证 + MLP 偏导数 + 闭卷练习记录

---
---

## Week 8 · 物理约束损失——用已知物理规律提升模型

> **本周目标**：学会一个全新的训练思路——
> 除了让模型输出逼近标注数据（数据损失），
> 还让模型的输出**满足你已知的物理规律**（物理约束损失）。
> 并通过实验验证：在数据稀少时，物理约束能显著提升模型的外推能力。

---

### Day 1 ｜ 核心思想

**理论目标**

- [ ] 🧠 L2 理解 ⭐⭐：物理约束损失的核心思想
  - **达标标准**：能用以下结构向别人解释清楚这个概念

```
传统做法（纯数据驱动）：
  Loss = MSE(模型预测, 标注数据)
  问题：数据少时模型在训练区域外的预测（外推）不可靠

物理约束做法：
  Loss = ω_data × MSE(模型预测, 标注数据)
       + ω_phys × MSE(物理方程残差, 0)

  含义：模型不仅要拟合数据，还要让输出满足已知的物理方程。
  即使数据只有 10 个点，物理方程在整个区域的每个点上都施加了约束，
  相当于用"先验知识"弥补数据的不足。
```

  - **自测**：如果你知道 y = sin(x) 的导数是 cos(x)，但只有 10 个 (x, sin(x)) 的标注点，你怎么把"dy/dx = cos(x)"这个知识加入训练？（答案：用 autograd 算出模型的 dy/dx，然后让它和 cos(x) 的 MSE 作为额外的 Loss 项）

- [ ] 🧠 L2 理解：ω_data 和 ω_phys 是什么
  - **达标标准**：能说出"ω_data 和 ω_phys 是权重系数，控制两项 Loss 的相对重要性。如果 ω_phys 太大，模型过度关注物理约束而忽视数据拟合；如果 ω_phys 太小，物理约束几乎不起作用，退化为纯数据驱动"

**实践目标**

- [ ] 💻 今天不写代码，把上述思想用自己的话写在笔记里（Markdown 或纸上都行），确保理解后再进入 Day 2

---

### Day 2 ｜ 实现第一个物理约束训练

**理论目标**

- [ ] 🧠 L2 理解：本次实验的具体设定
  - **达标标准**：能说出"我要训练一个 MLP 拟合 y = sin(x)，数据损失是预测值和 sin(x) 的 MSE，物理约束是模型的导数 dy/dx 和 cos(x) 的 MSE——因为我知道 sin 的导数就是 cos"

**实践目标**

- [ ] 💻 L4 能魔改 ⭐：实现带物理约束的训练
  - **达标标准**：运行后最终 Loss < 0.01，拟合曲线和 sin(x) 基本重合

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    # 全域采样点（100 个均匀分布的点）
    x = torch.linspace(0, 2 * 3.1416, 100, requires_grad=True).unsqueeze(1)
    y_true = torch.sin(x)

    # 前向传播
    y_pred = model(x)

    # 数据损失：预测值 vs sin(x)
    loss_data = nn.MSELoss()(y_pred, y_true)

    # 物理约束：模型的 dy/dx 应该等于 cos(x)
    dydx = torch.autograd.grad(
        y_pred, x, torch.ones_like(y_pred), create_graph=True
    )[0]
    loss_phys = nn.MSELoss()(dydx, torch.cos(x))

    # 总损失
    loss = loss_data + 0.1 * loss_phys

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: data={loss_data.item():.4f}, phys={loss_phys.item():.4f}")
```

  - **自测**：训练结束后 `loss_data` 和 `loss_phys` 分别是多少？两者都应该很小（< 0.01）

- [ ] 💻 L2 能照写：画拟合曲线
  - **达标标准**：蓝色线 "True sin(x)"，红色虚线 "Predicted"，标题 "MLP with Physics Constraint"

---

### Day 3 ｜ 关键实验：小样本下物理约束的价值

**理论目标**

- [ ] 🧠 L2 理解：这个实验要验证什么
  - **达标标准**：能说出"我要对比两种情况：(1) 只有 10 个标注数据点、没有物理约束；(2) 同样 10 个数据点、但加了物理约束。然后看两种方法在**数据点之间和数据点之外**（外推区域）的预测效果差多大"

**实践目标**

- [ ] 💻 L4 能魔改 ⭐⭐：实现小样本对比实验
  - **达标标准**：画出一张包含三条线的对比图，能明显看到物理约束在外推区域的优势

```python
import torch, torch.nn as nn, matplotlib.pyplot as plt

# ---- 稀疏数据：只有 10 个点 ----
torch.manual_seed(42)
x_sparse = torch.linspace(0, 3.14, 10).unsqueeze(1)   # 只在 [0, π] 内有数据
y_sparse = torch.sin(x_sparse)

# ---- 方法 A：纯数据驱动（只用 10 个点） ----
model_a = MLP()
opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)

for epoch in range(5000):
    pred = model_a(x_sparse)
    loss = nn.MSELoss()(pred, y_sparse)
    opt_a.zero_grad(); loss.backward(); opt_a.step()

# ---- 方法 B：数据 + 物理约束 ----
model_b = MLP()
opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

for epoch in range(5000):
    # 数据损失（10 个点）
    pred = model_b(x_sparse)
    loss_data = nn.MSELoss()(pred, y_sparse)

    # 物理约束（在全域 [0, 2π] 的 100 个点上施加）
    x_full = torch.linspace(0, 2 * 3.1416, 100, requires_grad=True).unsqueeze(1)
    y_full_pred = model_b(x_full)
    dydx = torch.autograd.grad(y_full_pred, x_full, torch.ones_like(y_full_pred), create_graph=True)[0]
    loss_phys = nn.MSELoss()(dydx, torch.cos(x_full))

    loss = loss_data + 0.1 * loss_phys
    opt_b.zero_grad(); loss.backward(); opt_b.step()

# ---- 画对比图 ----
x_test = torch.linspace(0, 2 * 3.1416, 200).unsqueeze(1)

model_a.eval(); model_b.eval()
with torch.no_grad():
    y_a = model_a(x_test)
    y_b = model_b(x_test)

plt.figure(figsize=(10, 5))
plt.plot(x_test, torch.sin(x_test), 'b-', lw=2, label='True sin(x)')
plt.plot(x_test, y_a, 'r--', lw=1.5, label='Data only (10 pts)')
plt.plot(x_test, y_b, 'g-.', lw=1.5, label='Data + Physics (10 pts)')
plt.scatter(x_sparse, y_sparse, c='black', zorder=5, label='Training data (10 pts)')
plt.axvline(x=3.14, color='gray', ls=':', label='Training boundary')
plt.xlabel('x'); plt.ylabel('y')
plt.title('Small Sample: Data Only vs Data + Physics Constraint')
plt.legend()
plt.show()
```

  - **自测（必须回答以下三个问题）**：
    1. 在 x ∈ [0, π]（有数据的区域），两种方法的差距大吗？（预期：差距不大，都能拟合）
    2. 在 x ∈ [π, 2π]（没有数据的外推区域），哪种方法更好？（预期：加物理约束的绿色线仍然贴近 sin(x)，纯数据的红色线偏离严重）
    3. 物理约束的价值体现在哪？（答案：在训练数据不覆盖的区域仍然给出合理预测——因为物理方程 dy/dx = cos(x) 在整个区域都成立）

- [ ] 💻 L2 能照写：在 notebook 中写一段 3–5 句话的分析
  - **达标标准**：必须包含 (1) 外推区域哪种方法更好 (2) 物理约束的核心价值是什么 (3) 什么时候物理约束最有用（答案：数据少且有已知物理规律时）

---

### Day 4 ｜ 多项损失权重实验

**理论目标**

- [ ] 🧠 L2 理解：权重 ω_phys 怎么选是一个重要的工程问题
  - **达标标准**：能说出"ω_phys 太小（如 0.001）→ 约束几乎不起作用，效果退化为纯数据驱动；ω_phys 太大（如 100）→ 模型过度关注物理约束而忽视数据拟合，可能在有数据的地方反而变差。需要实验找到平衡点。"

**实践目标**

- [ ] 💻 L4 能魔改：用 5 种不同的 ω_phys 训练，记录训练区域和外推区域的 MSE
  - **达标标准**：表格填满数字 + 画一张 5 条拟合曲线的对比图

| ω_phys | 训练区域 MSE ([0,π]) | 外推区域 MSE ([π,2π]) | 观察 |
|--------|---------------------|----------------------|------|
| 0.0 | ? | ? | 纯数据驱动（无约束） |
| 0.01 | ? | ? | 弱约束 |
| 0.1 | ? | ? | 中等约束 |
| 1.0 | ? | ? | 强约束 |
| 10.0 | ? | ? | 极强约束 |

  - **具体做法**：写一个函数 `train_with_weight(omega_phys, epochs=5000)`，返回训练好的模型，然后分别在 [0,π] 和 [π,2π] 上算 MSE
  - **自测**：
    - 哪个 ω_phys 的外推 MSE 最低？（通常是 0.1 或 1.0 附近）
    - ω_phys = 10.0 时训练区域 MSE 是不是反而变大了？（通常会——约束太强压制了数据拟合）

---

### Day 5–6（周末）｜ 整理 + 总结

**实践目标**

- [ ] 💻 L2 能照写：把 Day 1–4 整理成完整 notebook
  - **达标标准**：notebook 标题 "Week 8: Physics-Constrained Loss"，包含以下有 Markdown 标题的 section：
    1. "核心思想"——用 Markdown 写一段总结
    2. "基础实现"——Day 2 的完整代码
    3. "小样本实验"——Day 3 的对比图 + 文字分析
    4. "权重实验"——Day 4 的表格 + 对比图
  - Restart & Run All 无报错

- [ ] 💻 L2 能照写：在 notebook 最后写一段总结（5 句话以内）
  - **达标标准**：总结必须回答以下问题——"物理约束损失的核心思想是什么？在什么场景下最有价值？权重 ω_phys 选太大或太小各自的后果？"

**✅ 本周产出**

- [ ] `week08_physics_loss.ipynb`
- [ ] 包含：核心思想总结 + 物理约束训练实现 + 小样本对比图（3 条线）+ 权重实验表格 + 总结

---
---

## Week 9 · 综合练习 + Part B 收尾

> **本周目标**：把 Week 7–8 学到的 autograd 求导 + 物理约束能力练得更熟，
> 扩展到多输出和 2D 输入的场景，并完成 Part B 自测。

---

### Day 1–2 ｜ 多输出的物理约束

**理论目标**

- [ ] 🧠 L2 理解：如果网络有多个输出，可以对每个输出分别求导并施加不同的约束
  - **达标标准**：能说出"如果网络输出 u(x) 和 v(x) 两个函数，我可以约束 du/dx = v——这意味着 v 是 u 的导数。这种约束把两个输出之间的关系'告诉'了网络。"

**实践目标**

- [ ] 💻 L4 能魔改：训练一个 2 输出的 MLP，约束 du/dx = v
  - **达标标准**：训练后 u(x) 近似 sin(x)，v(x) 近似 cos(x)（因为 d(sin)/dx = cos），MSE < 0.05

```python
class TwoOutputMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.head_u = nn.Linear(64, 1)
        self.head_v = nn.Linear(64, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.head_u(h), self.head_v(h)

model = TwoOutputMLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    x = torch.linspace(0, 2*3.1416, 100, requires_grad=True).unsqueeze(1)
    u, v = model(x)

    # 数据损失：u 应该接近 sin(x)
    loss_data = nn.MSELoss()(u, torch.sin(x))

    # 物理约束：du/dx 应该等于 v
    dudx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    loss_phys = nn.MSELoss()(dudx, v)

    loss = loss_data + 0.5 * loss_phys
    opt.zero_grad(); loss.backward(); opt.step()
```

  - **自测**：训练后画两张子图——左图 u(x) vs sin(x)，右图 v(x) vs cos(x)。v(x) 有没有自动学成 cos(x)？（预期：有，因为约束 du/dx = v 迫使 v 成为 u 的导数）

---

### Day 3–4 ｜ 2D 输入的偏导数约束

**理论目标**

- [ ] 🧠 L2 理解：扩展到 2D 输入——如果网络接收 (x, y) 两个输入，你可以分别约束 ∂u/∂x 和 ∂u/∂y，甚至约束二阶偏导数 ∂²u/∂x² + ∂²u/∂y² = 0（这是拉普拉斯方程，描述稳态场）
  - **达标标准**：能说出"∂²u/∂x² + ∂²u/∂y² = 0 意味着 u(x,y) 在空间中是'平滑的'——没有局部极大或极小值出现在区域内部。如果这个方程的残差不为零，说明模型的输出违反了这个物理规律。"

**实践目标**

- [ ] 💻 L4 能魔改 ⭐：训练一个 MLP 满足拉普拉斯方程 ∂²u/∂x² + ∂²u/∂y² = 0
  - **达标标准**：训练后在内部采样点上的方程残差 MSE < 0.01，且 u(x,y) 的热力图看起来"平滑"（没有突兀的尖峰）

```python
model = nn.Sequential(
    nn.Linear(2, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 1),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10000):
    # 内部采样点：在 [0,1] × [0,1] 区域内随机取 200 个点
    xy = torch.rand(200, 2, requires_grad=True)
    u = model(xy)

    # 一阶偏导
    du = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    dudx = du[:, 0:1]   # ∂u/∂x
    dudy = du[:, 1:2]   # ∂u/∂y

    # 二阶偏导
    d2u_dx = torch.autograd.grad(dudx, xy, torch.ones_like(dudx), create_graph=True)[0]
    d2udx2 = d2u_dx[:, 0:1]   # ∂²u/∂x²

    d2u_dy = torch.autograd.grad(dudy, xy, torch.ones_like(dudy), create_graph=True)[0]
    d2udy2 = d2u_dy[:, 1:2]   # ∂²u/∂y²

    # PDE 残差：∂²u/∂x² + ∂²u/∂y² 应该等于 0
    residual = d2udx2 + d2udy2
    loss_pde = nn.MSELoss()(residual, torch.zeros_like(residual))

    # 边界条件（简化示例）：
    # u(x, 0) = x,  u(x, 1) = x,  u(0, y) = 0,  u(1, y) = 1
    x_bc_bottom = torch.stack([torch.linspace(0,1,20), torch.zeros(20)], dim=1)
    x_bc_top    = torch.stack([torch.linspace(0,1,20), torch.ones(20)], dim=1)
    x_bc_left   = torch.stack([torch.zeros(20), torch.linspace(0,1,20)], dim=1)
    x_bc_right  = torch.stack([torch.ones(20), torch.linspace(0,1,20)], dim=1)

    loss_bc = (
        nn.MSELoss()(model(x_bc_bottom), x_bc_bottom[:, 0:1]) +
        nn.MSELoss()(model(x_bc_top),    x_bc_top[:, 0:1]) +
        nn.MSELoss()(model(x_bc_left),   torch.zeros(20, 1)) +
        nn.MSELoss()(model(x_bc_right),  torch.ones(20, 1))
    )

    loss = loss_pde + 10 * loss_bc
    opt.zero_grad(); loss.backward(); opt.step()

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}: PDE={loss_pde.item():.4f}, BC={loss_bc.item():.4f}")
```

- [ ] 💻 L2 能照写：画 u(x, y) 热力图
  - **达标标准**：用 `plt.contourf` 或 `plt.imshow` 画出 u 在 [0,1]×[0,1] 区域内的值，颜色从蓝到红表示从低到高，横轴 "x"，纵轴 "y"，标题 "u(x,y) satisfying Laplace equation"

```python
# 生成网格
xx = torch.linspace(0, 1, 50)
yy = torch.linspace(0, 1, 50)
XX, YY = torch.meshgrid(xx, yy, indexing='ij')
xy_grid = torch.stack([XX.flatten(), YY.flatten()], dim=1)

model.eval()
with torch.no_grad():
    u_grid = model(xy_grid).reshape(50, 50)

plt.contourf(XX.numpy(), YY.numpy(), u_grid.numpy(), levels=20, cmap='jet')
plt.colorbar(label='u')
plt.xlabel('x'); plt.ylabel('y')
plt.title('u(x,y) satisfying Laplace equation')
plt.show()
```

  - **自测**：热力图看起来是从左（u≈0）到右（u≈1）平滑过渡的吗？（预期：是的——因为边界条件是 u(0,y)=0, u(1,y)=1，拉普拉斯方程保证了内部平滑过渡）

---

### Day 5–6（周末）｜ Part B 自测

**理论自测**（关掉所有资料，在纸上或文档中写出答案）

- [ ] 🧠 能填完 `backward()` vs `autograd.grad()` 的对比表格（L2）
  - **验证**：和 Week 7 Day 1 的表格对照，是否一致？

- [ ] 🧠 能回答"`create_graph=True` 什么时候需要？"（L2）
  - **标准答案**：(1) 要对导数再求导时（二阶导数）；(2) 导数要参与 Loss 计算并反向传播时（物理约束训练）

- [ ] 🧠 能回答"物理约束损失的核心思想是什么？"（L2）
  - **标准答案**：把已知物理方程的残差作为额外的 Loss 项，让模型输出不仅拟合数据还满足物理规律；在数据稀少时尤其有价值，因为物理方程在整个区域都施加了约束

- [ ] 🧠 能回答"权重 ω_phys 太大和太小各自有什么后果？"（L2）
  - **标准答案**：太大→过度约束，数据拟合变差；太小→约束失效，退化为纯数据驱动

**实践自测**（关掉所有参考，限时完成）

- [ ] 💻 10 分钟内从空白写出：对 y = sin(x) 求一阶和二阶导数并 assert 验证（L3）
  - **验证**：运行无报错，assert 通过

- [ ] 💻 10 分钟内从空白写出：定义一个 2 输入 1 输出 MLP，对输出求 ∂u/∂x₁ 和 ∂u/∂x₂（L3）
  - **验证**：运行无报错，打印的形状正确

- [ ] 💻 15 分钟内从空白写出：用 10 个数据点训练 MLP 拟合 sin(x)，Loss 中包含 dy/dx = cos(x) 的物理约束（L4）
  - **验证**：训练后画出拟合曲线，在外推区域能大致跟随 sin(x) 的趋势

**如果某项自测未通过**：标记为红色，回到对应的周重新练习

**✅ 本周产出**

- [ ] `week09_autograd_integration.ipynb`
- [ ] 包含：多输出约束 + 2D 拉普拉斯方程（含热力图）+ Part B 自测记录
- [ ] Part B 自测笔记（标红薄弱项）

---
---

## 📊 Part B 自测检查表（Week 7–9 结束后）

**理论**
- [ ] 🧠 能填完 `backward()` vs `autograd.grad()` 对比表（L2）
- [ ] 🧠 能解释 `create_graph=True` 的两个使用场景（L2）
- [ ] 🧠 能用 3 句话解释物理约束损失的核心思想和价值（L2）
- [ ] 🧠 能说出 ω_phys 太大/太小各自的后果（L2）

**实践**
- [ ] 💻 能在 10 分钟内默写 autograd 求一阶+二阶导数并验证（L3）
- [ ] 💻 能在 10 分钟内默写对 MLP 输出求偏导数（L3）
- [ ] 💻 能实现"数据损失 + 物理约束"组合训练（L4）
- [ ] 💻 能做权重配比实验（5 组）并分析结果（L4）
- [ ] 💻 能在 2D 输入上实现偏导数约束（L4）