# Part B: 自动微分深入 + 物理约束 Loss（Week 8–10）

## 本阶段定位

**衔接 Part A**：Part A 你用 `loss.backward()` 让 PyTorch 自动算出 **Loss 对模型参数**的梯度，用于参数更新。这是神经网络**训练**的核心机制。

**本阶段要解决的是另一个问题**：如何计算**模型输出对模型输入**的导数？

为什么需要这个能力？回到你的目标——论文第二章的 PhyFENet 框架用了 PINN 式的物理信息嵌入：

> "通过自动微分技术，能够自动计算输出对输入的精确导数。这种技术使得深度学习模型在数据域内的采样点上，可以精确地求解偏微分方程中的各个项。"

举个最具体的例子：论文第三章用 PINN 求解弹性杆问题，需要计算：
- 几何方程：`ε = du/dx`（应变 = 位移对坐标的导数）
- 本构方程：`σ = E·ε`
- 平衡方程：`dσ/dx = 0`

这里的 `du/dx`，是神经网络预测的位移 u 对输入坐标 x 的导数。这**不是 Loss 对参数的梯度**（那是 `.backward()` 做的事），而是**输出对输入的导数**。PyTorch 提供的工具叫 `torch.autograd.grad()`。

**本阶段目标**：
- 掌握 `torch.autograd.grad()` 的用法，能算任意函数对任意输入的一阶导和二阶导
- 理解"物理约束 Loss"的构造逻辑：把方程的残差作为 Loss 项
- 在简单问题上搭出"第一个物理信息嵌入"的训练流程（不是完整 PINN，是它的前身）

**完成标准（进入 Part C 前）**：
- 能用 `autograd.grad` 计算 MLP 输出对输入的一阶/二阶导数
- 能把一个简单 ODE 的残差写成 Loss 项
- 理解为什么需要把 `L_data`、`L_PDE`、`L_BC` 组合成加权总 Loss

**时间预期**：3 周，每周 10–14 小时，节奏比 Part A 略紧但仍然稳得住。如果 Week 9 的二阶导数部分你觉得吃力，可以多花一周在那里。

---

---

## Week 8: `torch.autograd.grad` 的进阶用法

**衔接**：Part A Week 5 你学的 `loss.backward()` 做的事情是：
```
loss (标量) → 对所有 requires_grad=True 的参数求梯度 → 结果写入 param.grad 属性
```

这次要学的 `torch.autograd.grad()` 做的是：
```
y (任意张量) → 对指定的输入求偏导 → 结果作为新张量返回（不写 .grad 属性）
```

两者在机制上是兄弟（都是 autograd 引擎），但使用场景完全不同。**你需要两个都掌握，不能混淆**。

**本周目标**：熟练使用 `autograd.grad` 算一阶导数，能用它对 MLP 的输出求对输入的导数。

**主要资源**：
- PyTorch 官方文档 `torch.autograd.grad()`（https://pytorch.org/docs/stable/generated/torch.autograd.grad.html）
- PyTorch Tutorial "Autograd mechanics"（可选，偏理论）

---

### Day 1 | `.backward()` vs `autograd.grad()`：两者到底有什么区别

✅ **理论任务**（约 1 小时）：在笔记上对比表格

| 对比项 | `.backward()` | `autograd.grad()` |
|-------|--------------|-------------------|
| 输入 | 一个标量（通常是 loss） | 输出张量 + 指定的输入张量 |
| 返回值 | `None`（原地修改 `.grad`） | 梯度张量组成的 tuple |
| 主要用途 | 参数更新（训练） | 算任意导数（推理时也能用） |
| 对输入张量的要求 | 输入需要 `requires_grad=True` | 同样需要 |
| 是否支持二阶导 | 不直接支持 | 支持（加 `create_graph=True`） |

✅ **实践任务**（约 1.5 小时）：创建文件 `week08/day01_backward_vs_grad.py`

**例子 1**：同一个计算，用两种方式求梯度
```python
import torch

# ===== 方式 1: .backward() =====
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = w * x + w ** 2    # y = 2*3 + 4 = 10

y.backward()           # 算 dy/dw，结果写入 w.grad
print("backward 方式，w.grad =", w.grad)    # 应为 3 + 2*2 = 7

# ===== 方式 2: autograd.grad() =====
# 注意：重新创建变量，否则梯度会累加
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = w * x + w ** 2

grads = torch.autograd.grad(y, w)   # 返回 tuple
print("autograd.grad 方式，dy/dw =", grads[0])    # 也应为 7
print("w.grad =", w.grad)           # 应为 None，autograd.grad 不写 .grad
```

**验证**：两种方式算出来的梯度值应相同（都是 7），但 `autograd.grad` 不修改 `w.grad`。

**关键观察**：
- `.backward()` 调用后，`w.grad` 被填上了 7
- `autograd.grad()` 调用后，`w.grad` 仍是 `None`（梯度以返回值形式给你）

**为什么这个差别重要**：当你要在 Loss 函数中间计算某些导数（比如 PDE 残差），你**不希望**这些导数把 `.grad` 属性污染了——那会影响后续训练时的参数更新。所以 PINN 中间的导数计算都用 `autograd.grad`，最终的 Loss 反向传播再用 `.backward()`。

**验收标准**：能跑通代码，能解释两者的区别。

---

### Day 2 | 用 `autograd.grad` 算最简单的一阶导数

**实践任务**（约 2 小时）：创建文件 `week08/day02_first_derivatives.py`

✅ **例子 1**：验证 d(x²)/dx = 2x
```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2                    # y = 9

dy_dx = torch.autograd.grad(y, x)[0]
print(f"dy/dx at x=3: {dy_dx}")   # 应为 6.0（因为 2*3=6）
```

✅ **例子 2**：对多个点同时求导
```python
x = torch.linspace(0, 2, 5, requires_grad=True)   # [0, 0.5, 1, 1.5, 2]
y = x ** 2                                          # [0, 0.25, 1, 2.25, 4]

# 这里直接 grad(y, x) 会报错，因为 y 不是标量
# 正确做法：传入 grad_outputs 参数
dy_dx = torch.autograd.grad(
    outputs=y, 
    inputs=x, 
    grad_outputs=torch.ones_like(y),   # 告诉 PyTorch：每个输出的权重是 1
    create_graph=False
)[0]
print(f"x: {x.detach()}")        # [0.0, 0.5, 1.0, 1.5, 2.0]
print(f"dy/dx: {dy_dx}")         # [0.0, 1.0, 2.0, 3.0, 4.0]（因为 2x）
```

**为什么需要 `grad_outputs`**：PyTorch 只能对标量输出求梯度。当输出是向量（shape=(5,)），你要告诉它"我对每个分量各要一个梯度"，数学上等价于把向量按权重求和变成标量。`grad_outputs=torch.ones_like(y)` 等价于"把 y 求和变成标量，再求梯度"——结果就是每个 y_i 对应的 x_i 的导数。

✅ **例子 3**：更复杂的函数 y = sin(x) * exp(x)
```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.sin(x) * torch.exp(x)

# 手算：dy/dx = cos(x)*exp(x) + sin(x)*exp(x) = exp(x)*(cos(x) + sin(x))
# 在 x=1: exp(1) * (cos(1) + sin(1)) ≈ 2.718 * (0.540 + 0.841) ≈ 3.756

dy_dx = torch.autograd.grad(y, x)[0]
print(f"autograd: {dy_dx:.4f}")    # 应约为 3.7560
```

**验证**：用 Python 手算验证：`math.exp(1) * (math.cos(1) + math.sin(1))`

**验收标准**：3 个例子的导数值都正确，能解释 `grad_outputs` 的作用。

---

### Day 3 | 对神经网络输出求导

**这是本周的核心技能**：之前你算的都是手写的 `y = x²` 这种。现在要算 `u = MLP(x)` 情况下的 `du/dx`——这是 PINN 的核心。

**实践任务**（约 2 小时）：创建文件 `week08/day03_grad_mlp_output.py`

**Step 1**：先训练一个 MLP 让它学会 u = sin(x)
```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),       # tanh 比 ReLU 更适合，后面解释
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 训练数据
x_train = torch.linspace(-torch.pi, torch.pi, 200).reshape(-1, 1)
u_true = torch.sin(x_train)

# 训练
for epoch in range(3000):
    u_pred = model(x_train)
    loss = criterion(u_pred, u_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"final loss: {loss.item():.6f}")
```

**Step 2**：用 `autograd.grad` 算 du/dx
```python
# 注意：用于求导的 x 必须 requires_grad=True
x_test = torch.linspace(-torch.pi, torch.pi, 100).reshape(-1, 1)
x_test.requires_grad = True

u_pred = model(x_test)       # u_pred shape=(100, 1)

# 求 du/dx
du_dx = torch.autograd.grad(
    outputs=u_pred,
    inputs=x_test,
    grad_outputs=torch.ones_like(u_pred),
    create_graph=False
)[0]
# du_dx shape=(100, 1)
```

**Step 3**：对比 MLP 算的 du/dx 和真实的 cos(x)
```python
import matplotlib.pyplot as plt

x_np = x_test.detach().numpy().flatten()
u_pred_np = u_pred.detach().numpy().flatten()
du_dx_np = du_dx.detach().numpy().flatten()
u_true_np = torch.sin(x_test).detach().numpy().flatten()
du_dx_true = torch.cos(x_test).detach().numpy().flatten()

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(x_np, u_true_np, 'b-', label='True sin(x)')
plt.plot(x_np, u_pred_np, 'r--', label='MLP prediction')
plt.legend(); plt.title('u = sin(x)')

plt.subplot(1,2,2)
plt.plot(x_np, du_dx_true, 'b-', label='True cos(x)')
plt.plot(x_np, du_dx_np, 'r--', label='autograd of MLP')
plt.legend(); plt.title('du/dx')

plt.tight_layout(); plt.savefig('week08_mlp_grad.png')
```

**关键观察**：即使你**从来没训练**模型让它学 cos(x)，MLP 自动学会的 sin(x) 通过 autograd 求导后，依然能给出正确的 cos(x)。这就是"神经网络可微"的体现——它本质上就是一个可微函数。

**为什么用 Tanh 而不是 ReLU**：ReLU 在 0 点不可导（尖角），它的导数是阶跃函数（不连续）。当你对 ReLU 网络求高阶导时会出问题。Tanh 处处光滑可导，适合物理信息嵌入任务。**这是 PINN 相关任务的一个工程常识**，记住它。

**验收标准**：
- MLP 能拟合 sin(x)（final loss < 0.001）
- autograd 算出的 du/dx 在整个区间上与 cos(x) 基本重合（目视即可）

---

### Day 4 | 多变量函数的偏导数

**为什么需要**：实际物理问题往往是多维的。比如 2D 弹性问题，位移 u 是 (x, y) 的函数，你需要算 ∂u/∂x 和 ∂u/∂y 两个偏导数。

**实践任务**（约 1.5 小时）：创建文件 `week08/day04_partial_derivatives.py`

**例子 1**：手动函数 f(x,y) = x² * y + y³
```python
import torch

xy = torch.tensor([2.0, 3.0], requires_grad=True)
x, y = xy[0], xy[1]
f = x**2 * y + y**3     # f = 4*3 + 27 = 39

# 同时对 x 和 y 求偏导
# 方式 1: 分别算
df_dx = torch.autograd.grad(f, x, create_graph=True, retain_graph=True)[0]
df_dy = torch.autograd.grad(f, y, create_graph=True, retain_graph=True)[0]

# 手算验证：
# ∂f/∂x = 2xy = 2*2*3 = 12
# ∂f/∂y = x² + 3y² = 4 + 27 = 31
print(f"∂f/∂x = {df_dx}")    # 应为 12.0
print(f"∂f/∂y = {df_dy}")    # 应为 31.0
```

**例子 2**：MLP 输入是 2D 坐标
```python
# 输入 (x, y) 两个坐标，MLP 输出一个位移值 u
class MLP2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.net(xy)

model = MLP2D()

# 随机输入
N = 50
x = torch.randn(N, 1, requires_grad=True)
y = torch.randn(N, 1, requires_grad=True)
u = model(x, y)

# 分别求 ∂u/∂x 和 ∂u/∂y
du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                             create_graph=False, retain_graph=True)[0]
du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                             create_graph=False)[0]

print(f"du/dx shape: {du_dx.shape}")    # (50, 1)
print(f"du/dy shape: {du_dy.shape}")    # (50, 1)
```

**关键参数解释**：
- `retain_graph=True`：算完梯度后不释放计算图，因为我们接下来还要再算一次梯度（对 y 求）。如果不加这个，第二次 `autograd.grad` 会报错。
- `create_graph=False`：本例只要一阶导，不需要保留对梯度的计算图。下周算二阶导时再开。

**验收标准**：两个偏导数的 shape 正确（都是 (50,1)）；对手动函数 f(x,y) 的验证结果正确。

---

### Day 5–6（周末）| 综合练习：用 autograd 验证函数关系

**任务**：创建文件 `week08/weekend_grad_exploration.py`

**实验**：训练一个 MLP 学习 `u = x³ - 2x`，然后用 autograd 计算其导数，与真实导数 `du/dx = 3x² - 2` 对比。

**Step 1**：训练（参考 Day 3 的模板）

**Step 2**：计算一系列测试点的 du/dx

**Step 3**：画 3 张子图：
1. 真实 u(x) vs MLP 预测 u(x)
2. 真实 du/dx vs autograd 算的 du/dx
3. 两者的**误差**（`|du/dx_pred - du/dx_true|`），用于判断 autograd 求导的精度

**观察任务**：写在脚本末尾的注释里
- 在哪些 x 值附近，du/dx 的误差最大？
- 这是因为训练数据在那些区域覆盖不足吗？

**验收标准**：图能正常生成；能说清楚 autograd 求导的精度与训练集覆盖范围的关系。

---

### Week 8 完成标准

- [ ] 能说清楚 `.backward()` 和 `torch.autograd.grad()` 的区别
- [ ] 能用 `autograd.grad` 算一阶导数（包括标量输出和向量输出情况）
- [ ] 能对 MLP 的输出求对输入的导数
- [ ] 能对多变量函数算偏导数（用 `retain_graph=True`）
- [ ] 知道 PINN 相关任务应该用 Tanh 而不是 ReLU（激活函数的可导性）

---

---

## Week 9: 高阶导数 + 物理约束 Loss 初探

**衔接**：Week 8 你能算一阶导数了。但很多物理方程涉及二阶导数——比如：
- 热传导方程：`∂²T/∂x²`
- 弹性力学平衡方程（无体力、一维）：`∂²u/∂x² = 0`
- 板弯曲方程：涉及 `∂⁴w/∂x⁴`

这周学两件事：
1. 用 `create_graph=True` 算二阶导数
2. 第一次把"方程残差"写成 Loss 项（还不是完整 PINN，但这是它的核心思想）

**本周目标**：理解二阶导数的算法机制；写出"方程残差 Loss"；用方程残差约束 MLP（不依赖数据的情况下）。

---

### Day 1 | 二阶导数：`create_graph=True` 的作用

**核心概念**：要算二阶导数 `d²y/dx²`，你需要：
1. 先算一阶导 `dy/dx`
2. 再对 `dy/dx` 求一次导

关键在 Step 2：要对 `dy/dx` 求导，PyTorch 需要**保留算 `dy/dx` 时的计算图**。默认情况下一阶求导后计算图会被释放（节省内存），所以算二阶导时必须在第一次求导时加 `create_graph=True`。

**实践任务**（约 2 小时）：创建文件 `week09/day01_second_order.py`

**例子 1**：验证 d²(x³)/dx² = 6x
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3                               # y = 8

# 一阶导：必须 create_graph=True
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx at x=2: {dy_dx}")          # 应为 12（3*2² = 12）

# 二阶导：对 dy_dx 再求一次导
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"d²y/dx² at x=2: {d2y_dx2}")      # 应为 12（6*2 = 12）
```

**例子 2**：向量情况下的二阶导
```python
x = torch.linspace(0, 2, 5, requires_grad=True)   # [0, 0.5, 1, 1.5, 2]
y = x ** 3                                         # [0, 0.125, 1, 3.375, 8]

# 一阶导
dy_dx = torch.autograd.grad(
    y, x, grad_outputs=torch.ones_like(y),
    create_graph=True
)[0]
# 应为 [0, 0.75, 3, 6.75, 12]（3x²）

# 二阶导
d2y_dx2 = torch.autograd.grad(
    dy_dx, x, grad_outputs=torch.ones_like(dy_dx),
    create_graph=False
)[0]
# 应为 [0, 3, 6, 9, 12]（6x）

print(f"x: {x.detach()}")
print(f"dy/dx: {dy_dx.detach()}")
print(f"d²y/dx²: {d2y_dx2}")
```

**关键记忆点**：`create_graph=True` 只在**中间步骤**用——即你还要继续对这个导数求高阶导。如果是最后一层导数，`create_graph=False`（或省略，默认 False）。

**验证**：两个例子的输出要和手算一致。

---

### Day 2 | MLP 输出的二阶导数

**实践任务**（约 1.5 小时）：创建文件 `week09/day02_mlp_second_order.py`

**目标**：训练 MLP 让它学 `u = sin(πx)`，然后算 `d²u/dx²`，与真实的 `-π²sin(πx)` 对比。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 模型（同 Week 8 Day 3，用 Tanh）
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# 训练数据：u = sin(πx)，x ∈ [0, 1]
torch.manual_seed(0)
x_train = torch.linspace(0, 1, 100).reshape(-1, 1)
u_train = torch.sin(torch.pi * x_train)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    u_pred = model(x_train)
    loss = ((u_pred - u_train) ** 2).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss={loss.item():.6f}")

# 测试：在 [0,1] 上均匀采样，算 u, du/dx, d²u/dx²
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
x_test.requires_grad = True
u = model(x_test)

du_dx = torch.autograd.grad(
    u, x_test, grad_outputs=torch.ones_like(u),
    create_graph=True
)[0]

d2u_dx2 = torch.autograd.grad(
    du_dx, x_test, grad_outputs=torch.ones_like(du_dx),
    create_graph=False
)[0]

# 可视化
x_np = x_test.detach().numpy().flatten()
u_np = u.detach().numpy().flatten()
du_np = du_dx.detach().numpy().flatten()
d2u_np = d2u_dx2.detach().numpy().flatten()

# 真实值
u_true = (torch.sin(torch.pi * x_test)).detach().numpy().flatten()
du_true = (torch.pi * torch.cos(torch.pi * x_test)).detach().numpy().flatten()
d2u_true = (-torch.pi**2 * torch.sin(torch.pi * x_test)).detach().numpy().flatten()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x_np, u_true, 'b-', label='True')
axes[0].plot(x_np, u_np, 'r--', label='MLP')
axes[0].set_title('u = sin(πx)'); axes[0].legend()

axes[1].plot(x_np, du_true, 'b-', label='True')
axes[1].plot(x_np, du_np, 'r--', label='autograd')
axes[1].set_title('du/dx'); axes[1].legend()

axes[2].plot(x_np, d2u_true, 'b-', label='True')
axes[2].plot(x_np, d2u_np, 'r--', label='autograd')
axes[2].set_title('d²u/dx²'); axes[2].legend()

plt.tight_layout(); plt.savefig('week09_second_order.png')
```

**关键观察**：
- 一阶导数通常拟合得较好
- 二阶导数的误差会比一阶导数大（误差累积）
- 边界附近（x=0 和 x=1）误差往往更大（训练数据覆盖边缘不足）

**验收标准**：
- 三张图能生成，u 拟合良好
- 一阶导误差在整体上较小
- 能观察到二阶导的误差比一阶大

---

### Day 3 | 物理约束 Loss 的思想：把方程残差写成 Loss

**核心概念**：假设我们想让 MLP 学的 u(x) 满足某个微分方程，比如：

```
du/dx + 2x = 0    （这就是 du/dx = -2x 的变形）
```

如果 u 完美满足这个方程，那么 `du/dx + 2x` 应该处处等于 0。

"完美"做不到，但我们可以把 `du/dx + 2x` 的**平方和**作为 Loss 的一项，让网络通过训练**尽量逼近**这个方程：

```
L_PDE = (1/N) * Σ (du/dx + 2x)²
```

这就是"物理约束 Loss"或者"PDE 残差 Loss"的核心思想。

**注意**：这一项**只约束方程，不直接约束 u 的具体值**。ODE 的解不唯一（差一个常数），所以需要再加**边界条件**来锁定具体解。这个下周做。

**理论任务**（约 30 分钟）：
- 在纸上写出以下 ODE 的残差表达式：
  1. `du/dx = sin(x)` → 残差 = `du/dx - sin(x)`
  2. `d²u/dx² - 3u = 0` → 残差 = `d²u/dx² - 3u`
- 理解：把等号右边移到左边，得到的表达式就是"残差"，残差 → 0 意味着方程被满足

---

### Day 4 | 第一个物理约束 Loss 实验

**实验设计**：不给任何标注数据（没有 u_true），只通过"u 必须满足 du/dx = -2x"这一约束训练 MLP。看能不能训出来。

**实践任务**（约 2 小时）：创建文件 `week09/day04_first_physics_loss.py`

**背景**：这个问题的解析解是 `u(x) = -x² + C`，但因为没有额外信息（边界条件），C 可以是任何常数。所以我们暂时只检查网络是否学到了**正确的形状**（即 `u(x) + x²` 应该是常数），不要求 C 的具体值。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

torch.manual_seed(0)
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练循环：只有物理约束 Loss
for epoch in range(5000):
    # 在 [0, 1] 上随机采样配点（collocation points）
    x = torch.rand(100, 1, requires_grad=True)
    u = model(x)
    
    # 算 du/dx
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    # 物理约束 Loss：du/dx + 2x → 0
    residual = du_dx + 2 * x
    loss = (residual ** 2).mean()
    
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss={loss.item():.6e}")

# 验证：画出 u(x) 和 u(x) + x² 的关系
# 如果 u(x) = -x² + C，那么 u(x) + x² = C (常数)
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
with torch.no_grad():
    u_pred = model(x_test)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(x_test.numpy(), u_pred.numpy(), 'r-', label='MLP u(x)')
plt.legend(); plt.title('u(x) from physics-only training')

plt.subplot(1,2,2)
u_plus_x2 = u_pred + x_test ** 2
plt.plot(x_test.numpy(), u_plus_x2.numpy(), 'g-')
plt.axhline(y=u_plus_x2.mean().item(), color='k', linestyle='--', label='mean')
plt.title('u(x) + x² (should be roughly constant)')
plt.legend()

plt.tight_layout(); plt.savefig('week09_physics_only.png')
```

**观察与讨论**：
- 物理约束 Loss 能把网络推到"形状正确"的解（某个 -x² + C）
- 但 C 值由初始化决定，每次训练不同
- 这说明：**只有方程约束没有数据或边界约束，解不唯一**

**验收标准**：
- 最终 loss < 1e-3
- `u(x) + x²` 应接近一个常数（波动 < 0.01），但具体常数值随机

---

### Day 5–6（周末）| 加入边界条件

**衔接**：Day 4 看到了"方程约束不足以确定唯一解"的问题。这周末加入边界条件 `u(0) = 1`，让解收敛到 `u(x) = 1 - x²`。

**任务**：创建文件 `week09/weekend_ode_with_bc.py`

**核心代码结构**：
```python
for epoch in range(5000):
    # === 物理约束部分 ===
    x_pde = torch.rand(100, 1, requires_grad=True)
    u_pde = model(x_pde)
    du_dx = torch.autograd.grad(u_pde, x_pde,
                                grad_outputs=torch.ones_like(u_pde),
                                create_graph=True)[0]
    residual = du_dx + 2 * x_pde
    L_pde = (residual ** 2).mean()
    
    # === 边界条件部分 ===
    x_bc = torch.tensor([[0.0]])        # 边界点 x=0
    u_bc_pred = model(x_bc)              # 在 x=0 处 MLP 的输出
    u_bc_true = torch.tensor([[1.0]])    # u(0) = 1
    L_bc = ((u_bc_pred - u_bc_true) ** 2).mean()
    
    # === 总 Loss（先用 1:10 权重，下周详细讨论权重）===
    loss = L_pde + 10 * L_bc
    
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**验证**：训练后，在 [0, 1] 上画 u_pred 和解析解 `u = 1 - x²`。

**验收标准**：
- 训练后 u_pred 接近 `1 - x²`
- 特别是 u(0) 应该非常接近 1.0（比如误差 < 0.01）
- 在 x=1 处 u 应接近 0

**关键体会**：物理约束 Loss + 边界条件 Loss 的组合，能**完全不依赖标注数据**求解 ODE。这就是 PINN 相对于纯数据驱动的优势——在"物理方程已知"的场景下，你不需要提前用 FEM 跑大量数据。

---

### Week 9 完成标准

- [ ] 能用 `create_graph=True` 算二阶导数
- [ ] 能写出一个 ODE 的残差表达式
- [ ] 能用"PDE 残差作为 Loss"的思路，无监督地训练 MLP 逼近 ODE 解
- [ ] 理解：只有 PDE Loss 不够，需要边界条件 Loss 锁定唯一解

---

---

## Week 10: 完整物理约束训练框架 + 权重平衡

**衔接**：Week 9 你搭出了"PDE + BC"的组合 Loss。但用的权重是拍脑袋的 `1:10`。实际 PINN 中，各项 Loss 的量级可能差几个数量级，权重配比是**工程难题**，论文第三章明确讨论了这个问题：

> "对于带孔板单向拉伸弹性变形，在小样本情况下，平衡多项损失的权重配置策略相比于其他权重配置方法更加有效。"

**本周目标**：
- 把 Week 9 的碎片代码整理成"标准 PINN 训练模板"
- 学会诊断和调整多项 Loss 的权重
- 用 PINN 求解一个更有物理意义的问题（弹性杆），为进入 Part C/D 打下基础

**本周的重要澄清**：你现在做的还不是完整的 PhyFENet 式系统——那个需要 GNN 结构（Part C 学）和 mesh-to-graph（Part D 学）。你做的是 PINN 的**基础形态**：一个 MLP + 物理 Loss。这是 PhyFENet 论文里"子网络"用的技术（原文 §2.3.5）。

---

### Day 1–2 | 标准 PINN 训练模板

**目标**：把物理约束训练流程封装成可复用的代码结构。

**实践任务**（每天约 2 小时）：创建目录 `week10/` 和文件结构：
```
week10/
├── pinn_template.py        ← 通用 PINN 训练模板
└── day12_ode_example.py    ← 用模板解 ODE
```

**Step 1**：`pinn_template.py` 框架
```python
import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    """用于物理约束训练的 MLP（用 Tanh 保证二阶可导）"""
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hid_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hid_dim, hid_dim), nn.Tanh()]
        layers.append(nn.Linear(hid_dim, out_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def compute_derivative(y, x, order=1):
    """算 y 对 x 的 order 阶导数
    
    参数：
      y: 模型输出，shape=(N, 1)
      x: 输入，shape=(N, 1)，必须 requires_grad=True
      order: 导数的阶数（1 或 2）
    返回：
      导数，shape=(N, 1)
    """
    if order == 1:
        dy_dx = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
        return dy_dx
    elif order == 2:
        dy_dx = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
        d2y_dx2 = torch.autograd.grad(
            dy_dx, x, grad_outputs=torch.ones_like(dy_dx),
            create_graph=True, retain_graph=True
        )[0]
        return d2y_dx2
    else:
        raise ValueError("只支持 1 阶或 2 阶导")


def train_pinn(model, pde_residual_fn, bc_fn, n_pde_points, 
               pde_domain, n_epochs, w_pde=1.0, w_bc=10.0, lr=1e-3):
    """通用 PINN 训练
    
    参数：
      model: PINN 实例
      pde_residual_fn: 函数，接受 (x, u, derivatives) 返回 PDE 残差
      bc_fn: 函数，接受 model 返回 BC 损失
      n_pde_points: 每个 epoch 采样的配点数
      pde_domain: (x_min, x_max) 定义域
      n_epochs: 训练总 epoch
      w_pde, w_bc: 两项权重
    返回：
      losses_dict: 记录各项 loss 的字典
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'total': [], 'pde': [], 'bc': []}
    
    for epoch in range(n_epochs):
        # PDE Loss
        x_min, x_max = pde_domain
        x_pde = torch.rand(n_pde_points, 1) * (x_max - x_min) + x_min
        x_pde.requires_grad_(True)
        u_pde = model(x_pde)
        residual = pde_residual_fn(x_pde, u_pde, model)
        L_pde = (residual ** 2).mean()
        
        # BC Loss
        L_bc = bc_fn(model)
        
        # Total
        loss = w_pde * L_pde + w_bc * L_bc
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        history['total'].append(loss.item())
        history['pde'].append(L_pde.item())
        history['bc'].append(L_bc.item())
        
        if epoch % 500 == 0:
            print(f"epoch {epoch}: total={loss.item():.4e}, "
                  f"pde={L_pde.item():.4e}, bc={L_bc.item():.4e}")
    
    return history
```

**Step 2**：`day12_ode_example.py` 用模板解 Week 9 的 ODE
```python
from pinn_template import PINN, compute_derivative, train_pinn
import torch

# 问题：du/dx = -2x，u(0) = 1
def pde_residual(x, u, model):
    du_dx = compute_derivative(u, x, order=1)
    return du_dx + 2 * x       # 残差：du/dx + 2x → 0

def bc_loss(model):
    x_bc = torch.tensor([[0.0]])
    u_bc_pred = model(x_bc)
    return ((u_bc_pred - 1.0) ** 2).mean()

torch.manual_seed(0)
model = PINN(in_dim=1, hid_dim=32, out_dim=1, n_layers=3)
history = train_pinn(
    model, pde_residual, bc_loss,
    n_pde_points=100, pde_domain=(0.0, 1.0),
    n_epochs=5000, w_pde=1.0, w_bc=10.0
)

# 验证：u_pred vs 解析解 u = 1 - x²
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
with torch.no_grad():
    u_pred = model(x_test)
u_true = 1 - x_test ** 2

error = (u_pred - u_true).abs().mean()
print(f"\n平均绝对误差: {error.item():.6f}")
```

**验收标准**：
- 模板代码能正常跑通
- 平均绝对误差 < 0.01
- 能看懂 `train_pinn` 函数的每个参数为什么需要

---

### Day 3 | 权重实验：为什么权重不好调

**实践任务**（约 2 小时）：创建文件 `week10/day03_weight_experiment.py`

**实验**：用相同模型和问题，测试 5 组 `(w_pde, w_bc)` 组合：
- (1, 0.1)：方程强，边界弱
- (1, 1)：等权重
- (1, 10)：边界强（Day 1-2 用的）
- (1, 100)：边界极强
- (0.1, 1)：方程弱

对每组：
- 训练 5000 epoch
- 记录最终的 MAE（u_pred vs u_true）
- 记录最终 `u(0)` 的值（应接近 1.0）

**观察**（写在表格中）：

| (w_pde, w_bc) | 最终 MAE | u(0) 实际值 | u(0) 偏离 |
|--------------|---------|------------|---------|
| (1, 0.1)     | ?       | ?          | ?       |
| (1, 1)       | ?       | ?          | ?       |
| (1, 10)      | ?       | ?          | ?       |
| (1, 100)     | ?       | ?          | ?       |
| (0.1, 1)     | ?       | ?          | ?       |

**用自己的话回答**：
- 边界权重太小（0.1）会导致什么现象？（u(0) 可能偏离 1.0）
- 边界权重太大（100）会导致什么现象？（可能 u(0) 非常准，但 PDE 方程满足得不够好，MAE 在其他区域大）
- 为什么说权重配比是"工程难题"？（没有通用最优，依赖具体问题）

**验收标准**：能填完表格，能从实验观察中总结出"权重影响解质量"的规律。

---

### Day 4 | 等比例权重策略

**衔接 Day 3**：手动试权重太累，PhyFENet 论文提到一种思路——让每项 Loss 的量级大致相同。

**核心想法**：
- 在训练过程中，若 L_pde 的量级是 1e-2，L_bc 的量级是 1e-5
- 说明两者的"有效贡献"是不平衡的
- 可以动态调整权重，让 `w_pde * L_pde ≈ w_bc * L_bc`

**实现思路**（伪代码）：
```
每隔 N 个 epoch：
  记录当前 L_pde 和 L_bc 的值
  如果 L_pde > L_bc：增大 w_pde（让 pde 方向更强的约束）
  反之增大 w_bc
  或者：按 1/L 的比例设置权重
```

**实践任务**（约 1.5 小时）：修改 `week10/day04_balanced_weights.py`

```python
from pinn_template import PINN, compute_derivative
import torch
import torch.optim as optim

# 用 Day 1-2 的问题，但改成动态权重
torch.manual_seed(0)
model = PINN(1, 32, 1, 3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

w_pde, w_bc = 1.0, 1.0
history = {'w_pde': [], 'w_bc': [], 'L_pde': [], 'L_bc': []}

for epoch in range(5000):
    # 计算两项 Loss
    x_pde = torch.rand(100, 1, requires_grad=True)
    u_pde = model(x_pde)
    du_dx = compute_derivative(u_pde, x_pde, order=1)
    L_pde = ((du_dx + 2 * x_pde) ** 2).mean()
    
    x_bc = torch.tensor([[0.0]])
    u_bc = model(x_bc)
    L_bc = ((u_bc - 1.0) ** 2).mean()
    
    # 每 100 epoch 更新权重
    if epoch % 100 == 0 and epoch > 0:
        with torch.no_grad():
            # 简单策略：w_i = 1 / L_i，让 w_i * L_i ≈ 1
            w_pde = 1.0 / (L_pde.item() + 1e-8)
            w_bc = 1.0 / (L_bc.item() + 1e-8)
            # 归一化让 w_pde + w_bc = 2（避免整体量级漂移）
            total = w_pde + w_bc
            w_pde = 2 * w_pde / total
            w_bc = 2 * w_bc / total
    
    loss = w_pde * L_pde + w_bc * L_bc
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    history['w_pde'].append(w_pde)
    history['w_bc'].append(w_bc)
    history['L_pde'].append(L_pde.item())
    history['L_bc'].append(L_bc.item())
    
    if epoch % 500 == 0:
        print(f"epoch {epoch}: L_pde={L_pde.item():.4e}, L_bc={L_bc.item():.4e}, "
              f"w_pde={w_pde:.2f}, w_bc={w_bc:.2f}")
```

**可视化**：画 4 条曲线在同一张图上（用 yscale='log'）：
- L_pde 随 epoch 的变化
- L_bc 随 epoch 的变化
- w_pde 随 epoch 的变化
- w_bc 随 epoch 的变化

**观察**：
- 训练初期，L_bc 可能比 L_pde 大很多（因为边界有个离散点 vs PDE 有整个域）
- 动态权重会自动增大 L_pde 所受权重的"相对重要性"

**验收标准**：
- 能画出动态权重随训练的变化
- 最终 MAE 不比固定权重版差（在合理范围内）

---

### Day 5–6（周末）| PINN 求解 1D 弹性杆（物理意义升级）

**衔接你的方向**：论文第三章 §3.1 就是用 PINN 求解带孔板弹性变形问题，这是完全一样的思路。周末用最简化版本——一维弹性杆——体验完整的"物理问题 → PINN 求解"流程。

**物理问题**：
```
一根长度 L=1 的弹性杆，左端固定，右端受轴向力 F
弹性模量 E = 1.0（单位无量纲化，真实单位不重要）
截面积 A = 1.0

三大方程：
  几何方程：ε = du/dx     （应变 = 位移对坐标的导数）
  本构方程：σ = E * ε     （胡克定律）
  平衡方程：dσ/dx = 0    （无体力，内力处处平衡）

边界条件：
  u(0) = 0               （左端固定）
  σ(L) = F/A = 1.0       （右端应力等于施加的应力）

解析解：
  u(x) = (F/EA) * x = x  （线性位移分布）
  σ(x) = F/A = 1         （应力处处均匀）
```

**实践任务**：创建文件 `week10/weekend_elastic_bar.py`

```python
from pinn_template import PINN, compute_derivative
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 参数
E = 1.0
F = 1.0
L = 1.0

torch.manual_seed(0)
model = PINN(in_dim=1, hid_dim=32, out_dim=1, n_layers=3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def get_losses():
    # 内部配点
    x = torch.rand(100, 1, requires_grad=True) * L
    u = model(x)
    
    # 用 autograd 算 du/dx
    du_dx = compute_derivative(u, x, order=1)    # 这就是 ε
    sigma = E * du_dx                             # σ = E * ε
    
    # 算 dσ/dx（需要对 sigma 再求一次导）
    dsigma_dx = torch.autograd.grad(
        sigma, x, grad_outputs=torch.ones_like(sigma),
        create_graph=True, retain_graph=True
    )[0]
    
    # PDE Loss：平衡方程 dσ/dx = 0
    L_pde = (dsigma_dx ** 2).mean()
    
    # 边界条件 1：u(0) = 0
    x_left = torch.tensor([[0.0]])
    u_left = model(x_left)
    L_bc_u = (u_left ** 2).mean()
    
    # 边界条件 2：σ(L) = F/A = 1
    x_right = torch.tensor([[L]], requires_grad=True)
    u_right = model(x_right)
    du_right = compute_derivative(u_right, x_right, order=1)
    sigma_right = E * du_right
    L_bc_sigma = ((sigma_right - F) ** 2).mean()
    
    return L_pde, L_bc_u, L_bc_sigma

for epoch in range(10000):
    L_pde, L_bc_u, L_bc_sigma = get_losses()
    # 权重：先用手动 1:10:10
    loss = L_pde + 10 * L_bc_u + 10 * L_bc_sigma
    
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"epoch {epoch}: L_pde={L_pde.item():.4e}, "
              f"L_bc_u={L_bc_u.item():.4e}, L_bc_sigma={L_bc_sigma.item():.4e}")

# 验证：画 u(x), ε(x), σ(x)
x_test = torch.linspace(0, L, 100).reshape(-1, 1)
x_test.requires_grad_(True)
u_pred = model(x_test)
eps_pred = compute_derivative(u_pred, x_test, order=1)
sigma_pred = E * eps_pred

# 解析解
u_true = x_test.detach()                   # u = x（因为 F=E=A=L=1）
eps_true = torch.ones_like(x_test).detach()
sigma_true = torch.ones_like(x_test).detach()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x_test.detach().numpy(), u_true.numpy(), 'b-', label='True')
axes[0].plot(x_test.detach().numpy(), u_pred.detach().numpy(), 'r--', label='PINN')
axes[0].set_title('u(x)'); axes[0].legend()

axes[1].plot(x_test.detach().numpy(), eps_true.numpy(), 'b-', label='True')
axes[1].plot(x_test.detach().numpy(), eps_pred.detach().numpy(), 'r--', label='PINN')
axes[1].set_title('ε(x) = du/dx'); axes[1].legend()

axes[2].plot(x_test.detach().numpy(), sigma_true.numpy(), 'b-', label='True')
axes[2].plot(x_test.detach().numpy(), sigma_pred.detach().numpy(), 'r--', label='PINN')
axes[2].set_title('σ(x) = E·ε'); axes[2].legend()

plt.tight_layout(); plt.savefig('week10_elastic_bar.png')
```

**验收标准**：
- 最终 u 的 MAE < 0.05（u_pred 接近线性分布 u = x）
- ε 应接近常数 1.0（允许小波动）
- σ 应接近常数 1.0
- 三张图的曲线应该基本重合

**关键体会**：
- 你第一次**完全不用 FEM 仿真数据**就求解了一个真实物理问题
- 整个 Loss 由"方程残差 + 边界条件"组成，没有任何标注数据
- 这就是 PhyFENet 论文里"子网络"的工作方式——只不过 PhyFENet 用图神经网络做主网络（Part C/D 会学），并融合 FEM 数据

---

### Week 10 完成标准 + Part B 总完成标准

**Week 10**：
- [ ] 能把 PINN 训练流程封装成可复用的模板
- [ ] 理解权重平衡是工程难题，没有通用最优
- [ ] 能独立实现一维弹性杆的 PINN 求解
- [ ] 能对比 PINN 解和解析解，判断求解质量

**Part B 总完成标准**（进入 Part C 前）：

理论
- [ ] 能说清楚 `.backward()` 和 `torch.autograd.grad()` 的区别和适用场景
- [ ] 能解释 `create_graph=True` 什么时候需要
- [ ] 能把一个 ODE 写成残差形式
- [ ] 能说出弹性力学三大方程（几何 / 本构 / 平衡），并解释每个方程的物理含义

代码（限时完成，不看参考）：
- [ ] 能在 10 分钟内算出一个 MLP 输出对输入的一阶和二阶导数
- [ ] 能在 20 分钟内写出一个给定 ODE 的物理约束 Loss（包含 PDE 残差 + BC）
- [ ] 能独立实现 1D 弹性杆的 PINN 求解（参考模板，30 分钟内）

如果上述任何一项未达到，Part C 的 GNN 学习会非常吃力（因为 GNN + PINN 结合是 PhyFENet 的核心）。回头补强 Week 8 或 Week 9。

---

*下一段输出：第一阶段 Part C（Week 11–16）：图神经网络*