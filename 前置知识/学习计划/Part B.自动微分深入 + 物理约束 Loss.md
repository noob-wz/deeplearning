# Part B 改造版（Week 8-10）

> 接续 Part A 的风格。Part B 包含 Week 8（autograd 进阶）、Week 9（高阶导数 + 物理约束 Loss 初探）、Week 10（完整 PINN 框架 + 1D 弹性杆）。
>
> Part A 教你"模型怎么训练"——通过 `loss.backward()` 算 Loss 对参数的梯度。Part B 解决另一个问题：**怎么算模型输出对模型输入的导数**？这是 PINN（Physics-Informed Neural Network，物理信息神经网络）的核心。
>
> 你的目标方向（PhyFENet 论文第二章）的核心思想之一就是把物理方程嵌入 Loss——这只能用本部分讲的工具实现。

---

## 📍 Part B 知识地图

```
                       【Part B 知识地图】

  ┌─────────────────────────────────────────────────────────────┐
  │                Week 8: autograd.grad 进阶                   │
  │  backward vs grad ──► 一阶导数 ──► 对MLP求导 ──► 偏导数      │
  │      (1)              (2)            (3)            (4)     │
  └─────────────┬───────────────────────────────────────────────┘
                │  Week 9: 高阶导数 + 物理 Loss 起步
                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │            Week 9: 二阶导数 + PDE 残差 Loss                 │
  │  create_graph=True ──► MLP二阶导 ──► PDE残差 ──► PDE+BC      │
  │       (5)                 (6)         (7)         (8)       │
  └─────────────┬───────────────────────────────────────────────┘
                │  Week 10: 框架化 + 真物理问题
                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │       Week 10: 完整 PINN 框架 + 1D 弹性杆                   │
  │  PINN模板 ──► 权重影响 ──► 动态权重 ──► 1D 弹性杆 PINN       │
  │   (9)         (10)         (11)         (12)                │
  └─────────────────────────────────────────────────────────────┘

  Part B 的终点产物：你能用纯物理方程（无任何 FEM 数据）训练一个 MLP
  求解 1D 弹性杆——这就是 PhyFENet 论文里"子网络"的基础形态。
  
  Part C 会让你学图神经网络（用 GNN 替代 MLP），Part D 会把它们组合成
  PhyFENet 框架的雏形。Part B 是这条链条的起点。
```

**对你方向的意义**：

PhyFENet 论文（第二章）这样描述：

> "通过自动微分技术，能够自动计算输出对输入的精确导数。这种技术使得深度学习模型在数据域内的采样点上，可以精确地求解偏微分方程中的各个项。"

具体到论文 §3.1 的弹性杆问题，需要算：
- 几何方程：`ε = du/dx`（应变 = 位移对坐标的导数）
- 本构方程：`σ = E·ε`
- 平衡方程：`dσ/dx = 0`

这里所有的 `du/dx`、`dσ/dx` 都是**神经网络输出对输入的导数**——只能用 `torch.autograd.grad` 算（不是 `.backward()`）。

**完成 Part B 后你应能做什么**：
- 给一个 1D 或简单 2D 问题，能从零写出 PINN 代码求解（无需 FEM 数据）
- 理解 PhyFENet 论文里"子网络（StrainSubNet、StressSubNet）"是怎么工作的
- 看 PINN 文献时不被"`autograd.grad`"、"`create_graph=True`" 这种细节卡住

---

## Week 8: `torch.autograd.grad` 进阶用法

### 🎯 Week 8 总览

**本周覆盖的知识点**：(1) backward vs grad 的区别、(2) 一阶导数、(3) MLP 输出对输入求导、(4) 偏导数

**本周不覆盖**：高阶导数（Week 9）、物理 Loss（Week 9）

**学完之后你应该能**：
- 🟣 **能用**：(1) 区分 `.backward()` 和 `autograd.grad()`、(2) 算任意函数的一阶导、(3) 对训练好的 MLP 算 du/dx、(4) 多变量函数偏导
- 🟡 **能讲**：为什么 PINN 中间步骤用 `autograd.grad` 而不是 `.backward()`

**本周的特点**：API 不多（核心就一个 `torch.autograd.grad`），但**坑非常多**。每个坑（`grad_outputs`、`retain_graph`、`requires_grad`）都需要单独搞清楚为什么。

---

### 进入 Week 8 之前的前置 checklist

- [✅] Part A Week 5 学的 autograd 4 个坑我都记得（梯度累加、detach、no_grad、.item()）
- [✅] 我能解释 `loss.backward()` 在做什么（自动算 Loss 对所有参数的梯度，写入 `.grad`）
- [✅] 我能用 PyTorch + nn.Module + Adam + DataLoader 写完整训练循环
- [✅] 我理解链式法则（dy/dx = dy/dg × dg/dx）
- [✅] 我接受"本周开始引入物理方程的概念，但只会用最简单的"

---

### Day 1 | `.backward()` vs `autograd.grad()`：兄弟工具，分工明确

**🎯 本日目标**：理解两个工具的本质区别和各自的应用场景。

**🟢 直觉层**（约 10 分钟）：

Part A Week 5 你用的 `loss.backward()` 做的事情是：

```
loss (一个标量) 
   ↓ .backward()
对所有 requires_grad=True 的参数求梯度
   ↓
梯度写入 param.grad 属性
   ↓ optimizer.step()
更新参数
```

这个用法很自然——训练就需要这个。

**但 PINN 里你需要做的事不一样**：你要算"网络输出 u 对输入 x 的导数 du/dx"——

- `u` **不是 Loss**（它是网络的输出，可能是个向量）
- `x` **不是参数**（它是网络的输入，是数据）
- 你**不希望**这次求导改变 `param.grad`（那会污染训练）

`loss.backward()` 干不了这个事——它专门处理"Loss 对参数"的场景。

`torch.autograd.grad()` 是"通用版"——你告诉它"算哪个对哪个的导数"，它返回值给你。

**🟡 概念层**（约 20 分钟）：

| 对比项 | `.backward()` | `autograd.grad()` |
|---|---|---|
| 输入要求 | 起点必须是**标量**（典型是 loss） | 起点和目标都可以是**任意张量** |
| 返回值 | `None`（原地修改 `.grad`） | 返回**梯度张量组成的 tuple** |
| 主要用途 | 训练时的参数更新 | 算任意"输出对输入"的导数 |
| 是否污染 `.grad` | 是（这就是它的工作机制） | 否（不修改 `.grad`） |
| 是否支持高阶导 | 不直接支持 | 支持（加 `create_graph=True`，Week 9 学） |

**关键理解**：两者**底层是同一个 autograd 引擎**——都是用计算图反向遍历做链式法则。区别只是 **API 包装方式**：

- `.backward()`：方便型，专门用于训练
- `autograd.grad()`：通用型，研究型应用必备

**🔵 数学层**（约 5 分钟）：

数学上没区别——两个 API 都做 `dy/dx`，只是结果存放方式不同。所以两者算同一个梯度时**结果完全一样**。

**🟣 代码层**（约 1 小时）：

创建文件 `week08/day01_backward_vs_grad.py`：

```python
import torch

# ========== 同一个计算，两种求梯度方式 ==========

# 方式 1：.backward()
print("===== 方式 1：.backward() =====")
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = w * x + w ** 2          # y = 2*3 + 4 = 10

y.backward()                 # 算 dy/dw，结果写入 w.grad
print(f"w.grad = {w.grad.item()}")        # 应为 7（即 x + 2w = 3 + 4 = 7）

# 方式 2：autograd.grad()
print("\n===== 方式 2：autograd.grad() =====")
w = torch.tensor(2.0, requires_grad=True)   # 重新创建，避免梯度累加
x = torch.tensor(3.0)
y = w * x + w ** 2

grads = torch.autograd.grad(y, w)            # 返回 tuple
print(f"dy/dw = {grads[0].item()}")          # 也是 7
print(f"w.grad = {w.grad}")                   # None！autograd.grad 不写 .grad

# ========== 验证两种方式的数学结果一致 ==========
print("\n===== 验证一致性 =====")
# 用同样的输入和函数，两种方式算出来的应该完全一样
# （上面已经看到都是 7）
```

**验收标准**：
- 两种方式都得到 7
- 第二种方式后 `w.grad` 是 `None`
- 你能用一句话说清楚两者的工作差异

**🔬 应用层**（约 10 分钟）：

**为什么这个差别对 PINN 重要**：

考虑 PINN 训练循环的伪代码：

```
for epoch:
    1. 在域内采样 x（输入）
    2. u = model(x)
    3. 算 du/dx（用于 PDE 残差）   ← 用什么？
    4. 算 PDE 残差 Loss
    5. 算 BC Loss
    6. total_loss = L_pde + L_bc
    7. total_loss.backward()        ← 这一步会更新参数
    8. optimizer.step()
```

第 3 步必须用 `autograd.grad`，**不能**用 `.backward()`，因为：

- 用 `.backward()` 会立即修改 `param.grad`
- 后面第 7 步又一次 `.backward()` 会再次修改 `param.grad`
- 两次修改互相干扰，模型训练崩溃

而用 `autograd.grad` 不修改 `.grad`，第 3 步算完 du/dx 后，参数的 `.grad` 仍然是干净的，第 7 步才能正常工作。

**这就是为什么所有 PINN 教程都用 `autograd.grad` 算 du/dx**——不是品味，是必须。

---

#### ⚠️ Day 1 新手最容易踩的坑

**坑 1：以为两者是"竞争关系"**
- 其实是**互补关系**——训练用 `.backward()`、中间求导用 `autograd.grad`
- 标准 PINN 训练循环里**两者都用**：`autograd.grad` 算 du/dx，最后 `loss.backward()` 算梯度更新参数

**坑 2：调用 `autograd.grad(y, x)` 而不是 `autograd.grad(y)`**
- `autograd.grad` 必须给两个参数：**对什么求导（输出）**和**关于什么（输入）**
- 漏掉第二个参数会报错

**坑 3：返回值是 tuple，要取 [0]**
- `grads = torch.autograd.grad(y, w)` 返回的是 `(dy/dw,)` 这样的 tuple
- 取出来要 `grads[0]`
- 直接 `print(grads)` 看到的是带括号的形式

**坑 4：以为重新创建变量就能避免梯度累加**
- 上面代码里我们重新创建了 w 和 x，避免了第一次 `.backward()` 留下的 `w.grad = 7` 影响
- 但这不是好习惯——实际项目里你不会这么"重新创建"
- 标准做法：**每次 backward 前清零**（`optimizer.zero_grad()` 或 `w.grad.zero_()`）

---

#### 🧠 Day 1 概念问答 quiz

**Q1**：`.backward()` 和 `torch.autograd.grad()` 的核心区别是什么？

<details><summary>答案</summary>

`.backward()` 输入要求是**标量**（典型是 loss），结果**原地写入** `param.grad`，专门用于训练。`autograd.grad()` 输入可以是**任意张量**，结果**作为返回值**给你（不修改 `.grad`），用于通用求导（包括 PINN 的 du/dx）。</details>

**Q2**：为什么 PINN 里算 du/dx **必须**用 `autograd.grad` 而不能用 `.backward()`？

<details><summary>答案</summary>

用 `.backward()` 会立即修改 `param.grad`，污染参数梯度。后面用 `loss.backward()` 算训练梯度时会和这次"污染"叠加，训练崩溃。`autograd.grad` 不修改 `.grad`，干净。</details>

**Q3**：`grads = torch.autograd.grad(y, w)`，怎么拿到具体的梯度值？

<details><summary>答案</summary>

`grads[0]`。`autograd.grad` 返回的是 tuple（即使只有一个梯度），需要索引取出。如果有多个输入：`autograd.grad(y, [w1, w2, w3])` 返回 `(grad_w1, grad_w2, grad_w3)`。</details>

**Q4**：以下两段代码哪个数学上正确？两个都正确的话，哪个更适合 PINN？

```python
# A
y.backward()
grad_w = w.grad

# B
grad_w = torch.autograd.grad(y, w)[0]
```

<details><summary>答案</summary>

**两个数学结果都对**（都是 dy/dw）。但 PINN 中**必须用 B**——A 会污染 `w.grad`，影响后续训练梯度。</details>

**Q5**：能在同一个训练循环里同时用 `.backward()` 和 `autograd.grad()` 吗？

<details><summary>答案</summary>

**能，而且必须这么做**。标准 PINN 模式：先用 `autograd.grad` 算 du/dx 等中间导数（不污染 `.grad`），最后用 `loss.backward()` 算总 Loss 对参数的梯度（这次正常修改 `.grad`），然后 `optimizer.step()` 更新。</details>

---

#### 📦 Day 1 知识卡片

| 项目 | 内容 |
|---|---|
| **核心术语** | `.backward()`、`torch.autograd.grad()` |
| **核心区别** | backward 修改 `.grad`；grad 返回 tuple |
| **使用分工** | 训练用 backward；中间求导用 grad |
| **PINN 标准模式** | 中间用 grad 算物理量，最后用 backward 算总 Loss 梯度 |
| **常见错误** | 漏掉第二个参数；忘了取 `[0]`；以为两者是竞争关系 |
| **本日产出** | `week08/day01_backward_vs_grad.py` |
| **掌握要求** | 🟣 能用 + 🟡 能讲 |

---

### Day 2 | 用 `autograd.grad` 算最简单的一阶导数

**🎯 本日目标**：熟练用 `autograd.grad` 算各种一阶导数；理解 `grad_outputs` 参数的作用。

**🟢 直觉层**（约 5 分钟）：

Day 1 的例子里 y 是个**标量**（比如 `y = 10`）。今天扩展到 **y 是向量**的情况——比如对 100 个采样点同时求导。

`autograd.grad` 默认要求 y 是标量。如果 y 是向量（比如 shape=(100,)），需要告诉它"我对每个 y_i 各要一个梯度"——这就是 `grad_outputs` 参数做的事。

**🟡 概念层**（约 20 分钟）：

**grad_outputs 的本质**:

设：$y = f(x), \quad x \in \mathbb{R}^n,; y \in \mathbb{R}^m$，其 Jacobian 为：$J_{ij} = \frac{\partial y_i}{\partial x_j}$

---

**PyTorch 实际计算的是什么？**

当你调用：

```python
torch.autograd.grad(y, x, grad_outputs=v)
```

PyTorch 返回的是：$J^T v$。也就是：$\frac{\partial}{\partial x} (v^T y)$

---

**两种等价理解（你可以任选一种思维方式）**

**① Jacobian 视角（更本质）**

* `grad_outputs = v`
* 返回：$J^T v$

👉 本质是：**Jacobian 的加权组合（VJP）**

---

**② 标量化视角（更直观）**

等价于先构造一个标量：$s = v^T y = \sum_i v_i y_i$

然后计算：$\nabla_x s$

👉 即：

```text
先把 y 变成标量 → 再求梯度
```

---

**当 `grad_outputs = torch.ones_like(y)` 时**

此时：$v = 1,1,\dots,1$，得到：$J^T \mathbf{1}= \nabla_x \sum_i y_i$

---

**在 PINN / batch 场景中的特殊性 🌟**

如果：$y_i = f(x_i)$

即每个样本独立，则：$\frac{\partial y_i}{\partial x_j} = 0 \quad (i \neq j)$

Jacobian 是对角矩阵：
$J =\begin{bmatrix}
\frac{dy_1}{dx_1} & 0 & \cdots \
0 & \frac{dy_2}{dx_2} & \
\vdots & & \ddots
\end{bmatrix}
$

因此：

$
J^T \mathbf{1}
= \begin{bmatrix}
\frac{dy_1}{dx_1}\
\frac{dy_2}{dx_2}\
\vdots
\end{bmatrix}
$

---

**🔑 关键结论**

在这种“逐点独立”的场景下：

```python
grad_outputs = torch.ones_like(y)
```

👉 **等价于对每个样本分别求导，并拼接成向量返回**

**为什么 PyTorch 这么设计**：因为通用情况下你可能想要"加权梯度"。但对 PINN 这种"每个点独立求导"的场景，永远是 `grad_outputs=torch.ones_like(y)`。

**🔵 数学层**（约 10 分钟）：

例子：`y = x²`，x 是 shape=(5,) 的向量。

```
x = [0, 0.5, 1, 1.5, 2]
y = [0, 0.25, 1, 2.25, 4]   每个 y[i] = x[i]²

dy/dx 应该是 (5,) 的向量：
  dy[0]/dx[0] = 2*0 = 0
  dy[1]/dx[1] = 2*0.5 = 1
  dy[2]/dx[2] = 2*1 = 2
  dy[3]/dx[3] = 2*1.5 = 3
  dy[4]/dx[4] = 2*2 = 4
```

注意 `dy[i]/dx[j]`（i ≠ j）= 0，因为不同位置互不影响。

**🟣 代码层**（约 1.5 小时）：

创建文件 `week08/day02_first_derivatives.py`：

```python
import torch
import math

# ========== 例 1：标量输出，最简情况 ==========
print("===== 例 1：y = x² 在 x=3 ====")
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2                    # y = 9（标量）

dy_dx = torch.autograd.grad(y, x)[0]
print(f"dy/dx at x=3: {dy_dx}")           # 应为 6.0（即 2*3）
# 手算验证：y = x²，dy/dx = 2x，x=3 时 = 6 ✓

# ========== 例 2：向量输出，需要 grad_outputs ==========
print("\n===== 例 2：y = x² 在 5 个点 ====")
x = torch.linspace(0, 2, 5, requires_grad=True)   # [0, 0.5, 1, 1.5, 2]
y = x ** 2                                          # [0, 0.25, 1, 2.25, 4]

# 错误尝试：直接 grad(y, x)
try:
    grads = torch.autograd.grad(y, x)
except Exception as e:
    print(f"报错: {e}")
# RuntimeError: grad can be implicitly created only for scalar outputs

# 正确做法：传入 grad_outputs
dy_dx = torch.autograd.grad(
    outputs=y,
    inputs=x,
    grad_outputs=torch.ones_like(y),    # 每个 y[i] 权重为 1
    create_graph=False
)[0]

print(f"x: {x.detach().tolist()}")
print(f"y = x²: {y.detach().tolist()}")
print(f"dy/dx: {dy_dx.tolist()}")
# 应为 [0.0, 1.0, 2.0, 3.0, 4.0]，即 2x

# ========== 例 3：复杂复合函数 ==========
print("\n===== 例 3：y = sin(x) * exp(x) 在 x=1 =====")
x = torch.tensor(1.0, requires_grad=True)
y = torch.sin(x) * torch.exp(x)

# 手算（链式法则 + 乘积法则）：
# dy/dx = cos(x)*exp(x) + sin(x)*exp(x) = exp(x) * (cos(x) + sin(x))
# 在 x=1: e * (cos(1) + sin(1)) ≈ 2.7183 * (0.5403 + 0.8415) ≈ 3.7560
expected = math.exp(1) * (math.cos(1) + math.sin(1))

dy_dx = torch.autograd.grad(y, x)[0]
print(f"autograd 结果: {dy_dx.item():.6f}")
print(f"手算结果: {expected:.6f}")
# 两者应几乎相同（差异在浮点精度内）
```

**验收标准**：
- 三个例子的导数值都正确
- 例 2 的输出和"`2x`"对应
- 你能用一句话解释 `grad_outputs` 在做什么

---

#### ⚠️ Day 2 新手最容易踩的坑

**坑 1：忘记 `grad_outputs` 直接对向量求导**
- `autograd.grad(y_vector, x_vector)` 报错："grad can be implicitly created only for scalar outputs"
- 必须加 `grad_outputs=torch.ones_like(y)`

**坑 2：`grad_outputs` 的 shape 和 outputs 不一致**
- `grad_outputs.shape` 必须等于 `outputs.shape`
- 不一致会报"Mismatch in shape"或得到错误结果

**坑 3：以为不同 i 之间会相互影响**
- 上面例子里 `dy[i]/dx[j]`（i≠j）= 0，因为 `y[i]` 只是 `x[i]²`，和 `x[j]` 无关
- 这种"独立性"是 PINN 多采样点求导的关键
- 如果 y 是 mixing 的（比如 `y[i] = x[0] * x[i]`），i 和 j 会相互影响——就不能这么简单理解了

**坑 4：`x` 没设 `requires_grad=True`**
- 错误：`x = torch.linspace(0, 2, 5)` 然后求导报错
- PyTorch 不追踪没开 `requires_grad` 的 tensor
- 正确：`x = torch.linspace(0, 2, 5).requires_grad_(True)` 或在 `linspace` 后加 `.requires_grad = True`

**坑 5：没有 `.detach()` 直接转 numpy**
- x 还在计算图里，直接 `x.numpy()` 报错
- 画图前必须 `x.detach().numpy()`（Part A Week 5 学过）

---

#### 🧠 Day 2 概念问答 quiz

**Q1**：`autograd.grad(y, x)` 中 y 必须是什么类型？如果 y 是向量怎么办？

<details><summary>答案</summary>

y 必须是**标量**（默认）。如果 y 是向量，需要传入 `grad_outputs` 参数（通常用 `torch.ones_like(y)`），告诉 PyTorch 怎么把向量"加权和"成标量。</details>

**Q2**：`grad_outputs=torch.ones_like(y)` 在 PINN 场景下等价于什么？

<details><summary>答案</summary>

等价于"对每个采样点分别求导，把结果拼起来返回"。数学上是对 sum(y) 求导，由独立性等于每个 y[i] 对自己 x[i] 的导数。</details>

**Q3**：`y = x²`，x.shape=(5,)，调用 `autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]` 返回的 shape 是什么？

<details><summary>答案</summary>

`(5,)`。返回值的 shape 和 inputs 一致——5 个输入，5 个梯度。</details>

**Q4**：以下代码报错的原因是什么？怎么修？
```python
x = torch.linspace(0, 1, 100)
y = x ** 2
dy_dx = torch.autograd.grad(y, x)[0]
```

<details><summary>答案</summary>

**两个错**：(1) x 没有 `requires_grad=True`，PyTorch 没追踪它；(2) y 是向量，没传 grad_outputs。修复：
```python
x = torch.linspace(0, 1, 100).requires_grad_(True)
y = x ** 2
dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]
```
</details>

**Q5**：`autograd.grad` 算梯度时，`x` 必须 `requires_grad=True` 吗？为什么？

<details><summary>答案</summary>

**是**。PyTorch 只追踪 `requires_grad=True` 的 tensor 在计算图中的依赖。如果 x 没开追踪，y 不知道 "我是从 x 算出来的"，求 dy/dx 没意义。</details>

---

#### 📦 Day 2 知识卡片

| 项目 | 内容 |
|---|---|
| **核心 API** | `torch.autograd.grad(outputs, inputs, grad_outputs, create_graph)` |
| **关键参数 grad_outputs** | y 是向量时必须给（用 `torch.ones_like(y)`） |
| **关键参数 create_graph** | 一阶导用 False；要算二阶导用 True（Week 9 学） |
| **返回值** | tuple，取 `[0]` |
| **输入要求** | x 必须 `requires_grad=True` |
| **常见错误** | 忘 grad_outputs；x 没 requires_grad；shape 不一致 |
| **本日产出** | `week08/day02_first_derivatives.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 3 | 对神经网络输出求导（PINN 的核心技能）

**🎯 本日目标**：把 Day 2 的求导从"手写函数"扩展到"训练好的 MLP"——这是 PINN 的核心能力。

**🟢 直觉层**（约 10 分钟）：

回想 Day 2 的例子：`y = x²`，求 `dy/dx`。这里 y 是手写的函数。

PINN 中，y 是**神经网络的输出**——比如 `u = MLP(x)`，u 是 MLP 给出的位移场。但**神经网络本质上也是一个函数**（就是一个非常复杂的复合函数），所以 `autograd.grad` 一样能用。

**关键观察**：网络通过反向传播自动学到 `u` 的"形状"，然后你用 autograd 对 `u` 求导，得到的就是这个形状的导数。**网络从来没有专门学过 cos(x)**，但只要它学会了 sin(x)，`autograd` 自动"导出"的 `du/dx` 就接近 cos(x)——这就是"神经网络可微"的实际意义。

**🟡 概念层**（约 15 分钟）：

PINN 中标准的 du/dx 计算模式：

```python
# 注意：x 必须 requires_grad=True
x = torch.linspace(...).reshape(-1, 1).requires_grad_(True)
u = model(x)             # u shape=(N, 1)

du_dx = torch.autograd.grad(
    outputs=u,
    inputs=x,
    grad_outputs=torch.ones_like(u),    # u 是 (N, 1) 向量
    create_graph=False                   # 一阶导用 False
)[0]
# du_dx shape=(N, 1)
```

**关于激活函数的选择**：

- **ReLU 不适合 PINN**：ReLU(z) = max(0, z) 在 0 处不可导（尖角）。它的导数是阶跃函数（不连续）。算二阶导时整个空间几乎处处为 0，物理 Loss 没有意义
- **Tanh 是 PINN 的标准选择**：处处光滑可导，所有阶导数都有意义
- **GELU、SiLU 等也可以**——只要光滑都行
- **Sigmoid** 理论上可以，但梯度消失问题严重

**这是 Part A Week 1 Day 2 提到过的**——当时讲"PINN 用 Tanh"，现在你看到了具体原因。

**🔵 数学层**（约 5 分钟）：

数学上没新内容，就是 `autograd.grad` 用在更复杂的复合函数上。

**🟣 代码层**（约 2 小时）：

创建文件 `week08/day03_grad_mlp_output.py`：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ========== Step 1: 训练 MLP 学 sin(x) ==========

torch.manual_seed(42)

class MLP(nn.Module):
    """用于 PINN 的 MLP——必须用 Tanh"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),       # ⚠️ 用 Tanh 不用 ReLU
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

print("训练 MLP 拟合 sin(x)...")
for epoch in range(3000):
    u_pred = model(x_train)
    loss = criterion(u_pred, u_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"  epoch {epoch}: loss = {loss.item():.6f}")

print(f"最终 loss: {loss.item():.6f}")

# ========== Step 2: 用 autograd 算 du/dx ==========

# 注意：用于求导的 x 必须 requires_grad=True
x_test = torch.linspace(-torch.pi, torch.pi, 100).reshape(-1, 1)
x_test.requires_grad_(True)

u_pred = model(x_test)       # u_pred shape=(100, 1)

du_dx = torch.autograd.grad(
    outputs=u_pred,
    inputs=x_test,
    grad_outputs=torch.ones_like(u_pred),
    create_graph=False
)[0]
# du_dx shape=(100, 1)

# ========== Step 3: 对比 MLP 算出的 du/dx 和真实的 cos(x) ==========

x_np = x_test.detach().numpy().flatten()
u_pred_np = u_pred.detach().numpy().flatten()
du_dx_np = du_dx.detach().numpy().flatten()
u_true_np = torch.sin(x_test).detach().numpy().flatten()
du_dx_true = torch.cos(x_test).detach().numpy().flatten()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x_np, u_true_np, 'b-', label='True sin(x)')
axes[0].plot(x_np, u_pred_np, 'r--', label='MLP prediction')
axes[0].set_title('u(x): MLP 学了 sin(x)')
axes[0].legend()

axes[1].plot(x_np, du_dx_true, 'b-', label='True cos(x)')
axes[1].plot(x_np, du_dx_np, 'r--', label='autograd of MLP')
axes[1].set_title('du/dx: 注意——MLP 没专门学 cos(x)！')
axes[1].legend()

plt.tight_layout()
plt.savefig('week08_mlp_grad.png')

print("\n关键观察：MLP 从来没有专门训练 cos(x)，但 autograd 自动算出来的 du/dx 接近 cos(x)")
print("这就是'神经网络可微'的实际意义")
```

**验收标准**：
- MLP 拟合 sin(x)，最终 loss < 0.001
- autograd 算的 du/dx 在整个区间上和 cos(x) 基本重合（目视即可）
- 你能解释为什么"MLP 没学过 cos(x)，但 autograd 算的 du/dx 还是 cos(x)"

**🔬 应用层**（约 10 分钟）：

**这就是 PINN 的"魔法"来源**：

PINN 的整个核心都建立在这个能力之上：
1. 你定义一个 MLP `u = model(x)`
2. 用 `autograd.grad` 算 `du/dx`、`d²u/dx²`
3. 把这些和 PDE 的形式比较，定义 PDE 残差作为 Loss
4. 训练让 PDE 残差变小 → 网络自动学到满足方程的 u(x)

第 2 步的能力就是今天学的。**没有这个能力，PINN 不存在**。

延伸：当你后面去看 PINN 论文（包括 PhyFENet）时，你会发现所有 PINN 实现的核心结构都包括这两步：(1) 用 MLP 输出 u；(2) 用 `autograd.grad` 求 `du/dx`。**模式一模一样**，区别只在于求几阶导、对应什么 PDE。

---

#### ⚠️ Day 3 新手最容易踩的坑

**坑 1：用 ReLU 训练 PINN 类网络**
- ReLU 不可导（0 点处）+ 二阶导处处为 0
- 简单 MLP 拟合 sin(x) 可能能跑通（一阶导还行），但物理 Loss 涉及二阶导时彻底失败
- **PINN 一定用 Tanh、GELU、SiLU 这种光滑激活**

**坑 2：训练时 x 没 `requires_grad`，求导时也没加**
- 训练时不需要 `requires_grad`（数据是固定的）
- **求导时必须给求导用的输入加 `requires_grad_(True)`**
- 这是两次不同的需求

**坑 3：以为训练好的 MLP 自动支持求导**
- 是的，**自动支持**——但前提是输入开了 `requires_grad`
- 网络本身参数有 requires_grad，但输入是用户传的，需要用户决定开不开

**坑 4：du_dx 拿出来直接 numpy 报错**
- du_dx 仍在计算图里
- 转 numpy：`du_dx.detach().numpy()`

**坑 5：训练 MLP 时也用 autograd.grad 求 dL/dw**
- 训练时**应该用 backward()**——这是它的本职工作
- 用 `autograd.grad` 也可以但会让代码繁琐
- **分工**：训练用 backward，中间求导用 autograd.grad

---

#### 🧠 Day 3 概念问答 quiz

**Q1**：MLP 训练完只学了 sin(x)，没学 cos(x)，为什么 autograd 能算出 cos(x)？

<details><summary>答案</summary>

因为 MLP 是一个**可微函数**——它学到的不仅是函数值，还内部包含了所有阶导数。autograd 沿着计算图反向遍历，自动算出 du/dx——只要 MLP 学到了正确的 u(x) = sin(x)，对它求导自然得到 cos(x)。</details>

**Q2**：PINN 为什么必须用 Tanh 而不是 ReLU？

<details><summary>答案</summary> 

(1) ReLU 在 0 处不可导（尖角）；(2) ReLU 的二阶导数处处为 0（除原点），导致 PINN 中涉及二阶导的物理 Loss 失效。Tanh 处处光滑可导，所有阶导数都有意义。</details>

**Q3**：以下代码哪一行有 bug？

```python
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)    # ← line 1
u_pred = model(x_test)                                # ← line 2
du_dx = torch.autograd.grad(u_pred, x_test, grad_outputs=torch.ones_like(u_pred))[0]  # ← line 3
```

<details><summary>答案</summary>

**line 1 缺 `requires_grad_(True)`**。x_test 没追踪，line 3 求导会报错。修复：`x_test = torch.linspace(...).reshape(-1, 1).requires_grad_(True)`。</details>

**Q4**：训练阶段（fitting sin(x)）要不要给训练数据 x 设 `requires_grad=True`？

<details><summary>答案</summary>

**不需要**。训练时 x 是数据，autograd 追踪的是从输入流向 Loss 的路径，路径上的**参数**需要 `requires_grad`，**数据**不需要。设了也不出错（Part A Week 5 提过），但浪费内存。</details>

**Q5**：能在同一个训练循环里"训练 MLP（用 backward）"和"求 du/dx（用 autograd.grad）"吗？

<details><summary>答案</summary>

**完全可以，而且是 PINN 的标准做法**。求 du/dx 用 autograd.grad（不污染参数 .grad），最后用 loss.backward() 算总 Loss 对参数的梯度并 step——这正是 Week 9 Day 4 之后会做的事。</details>

---

#### 📦 Day 3 知识卡片

| 项目 | 内容 |
|---|---|
| **核心能力** | 用 autograd 算训练好的 MLP 的输出对输入的导数 |
| **激活函数选择** | PINN 必须用 Tanh / GELU / SiLU（光滑），不能用 ReLU |
| **关键操作** | `x.requires_grad_(True)` + `autograd.grad(u, x, ...)` |
| **数据准备** | 训练数据不需要 requires_grad；求导用的输入需要 |
| **常见错误** | 忘 requires_grad；用 ReLU；忘 detach 转 numpy |
| **本日产出** | `week08/day03_grad_mlp_output.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 4 | 多变量函数的偏导数

**🎯 本日目标**：扩展到多变量情况（比如 2D 问题里 u 是 (x, y) 的函数）；学会同时算多个偏导数。

**🟢 直觉层**（约 5 分钟）：

实际物理问题往往是多维的：
- 1D 问题：u 是 x 的函数 → 算 du/dx
- 2D 问题：u 是 (x, y) 的函数 → 同时算 ∂u/∂x 和 ∂u/∂y
- 3D 问题：u 是 (x, y, z) 的函数 → 算三个偏导

PhyFENet 论文里做的是 2D 问题（带孔板），所以你需要理解多变量偏导。

**🟡 概念层**（约 20 分钟）：

**关键 API：`retain_graph=True`**

当你**需要多次对同一个计算图求导**时（比如先算 ∂u/∂x，再算 ∂u/∂y），需要保留计算图。

默认情况下，第一次 `autograd.grad` 后计算图被释放（节省内存）——再次求导会报错。`retain_graph=True` 让 PyTorch 保留计算图，允许多次求导。

**两种实现方式**：

**方式 1：分别求**（每个偏导单独算）

```python
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
f = x**2 * y + y**3

df_dx = torch.autograd.grad(f, x, retain_graph=True)[0]   # 注意 retain_graph=True
df_dy = torch.autograd.grad(f, y)[0]                       # 最后一次不需要保留
```

**方式 2：合在一起求**（一次性传入所有输入）

```python
df_dx, df_dy = torch.autograd.grad(f, [x, y])
# 输入是 list，返回也是 tuple，按 list 顺序对应
```

**两种方式的差别**：
- 方式 1 更灵活——可以分开做不同处理
- 方式 2 更简洁——一次调用得到所有偏导
- 计算量上**等价**（PyTorch 内部会智能复用）

**`retain_graph=True` 的作用**：

```python
# 不加 retain_graph：
df_dx = autograd.grad(f, x)[0]              # 算完后计算图释放
df_dy = autograd.grad(f, y)[0]              # 报错：计算图已释放
```

```python
# 加 retain_graph：
df_dx = autograd.grad(f, x, retain_graph=True)[0]   # 保留计算图
df_dy = autograd.grad(f, y)[0]                       # 仍能求
```

**🔵 数学层**（约 10 分钟）：

例子：`f(x, y) = x²y + y³`

偏导（手算）：
- ∂f/∂x = 2xy（把 y 当常数）
- ∂f/∂y = x² + 3y²（把 x 当常数）

在 (x=2, y=3)：
- ∂f/∂x = 2*2*3 = 12
- ∂f/∂y = 4 + 27 = 31

**🟣 代码层**（约 1.5 小时）：

创建文件 `week08/day04_partial_derivatives.py`：

```python
import torch
import torch.nn as nn

# ========== 例 1：手动函数的偏导数 ==========
print("===== 例 1：f(x, y) = x²y + y³ ====")
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
f = x**2 * y + y**3       # f = 4*3 + 27 = 39

# 方式 1：分别求（注意 retain_graph）
df_dx = torch.autograd.grad(f, x, retain_graph=True)[0]
df_dy = torch.autograd.grad(f, y)[0]

print(f"f(2, 3) = {f.item()}")            # 应为 39
print(f"∂f/∂x at (2,3) = {df_dx.item()}")     # 应为 12（2*2*3）
print(f"∂f/∂y at (2,3) = {df_dy.item()}")     # 应为 31（4 + 27）

# 方式 2：一次性求（更简洁）
print("\n===== 方式 2：一次性求 =====")
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
f = x**2 * y + y**3

df_dx, df_dy = torch.autograd.grad(f, [x, y])
print(f"∂f/∂x = {df_dx.item()}, ∂f/∂y = {df_dy.item()}")    # 12, 31

# ========== 例 2：MLP 输入是 2D 坐标 ==========
print("\n===== 例 2：MLP 学 u(x, y) =====")

class MLP2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, y):
        # 把 x 和 y 拼接成 (N, 2)
        xy = torch.cat([x, y], dim=1)
        return self.net(xy)

torch.manual_seed(0)
model = MLP2D()

# 准备多个 (x, y) 点
N = 50
x = torch.randn(N, 1, requires_grad=True)
y = torch.randn(N, 1, requires_grad=True)
u = model(x, y)              # u shape=(N, 1)

# 求 ∂u/∂x（注意 retain_graph）
du_dx = torch.autograd.grad(
    u, x,
    grad_outputs=torch.ones_like(u),
    retain_graph=True              # 保留计算图，因为还要求 ∂u/∂y
)[0]

# 求 ∂u/∂y（最后一次，可不加 retain_graph）
du_dy = torch.autograd.grad(
    u, y,
    grad_outputs=torch.ones_like(u)
)[0]

print(f"u shape: {u.shape}")              # (50, 1)
print(f"du/dx shape: {du_dx.shape}")      # (50, 1)
print(f"du/dy shape: {du_dy.shape}")      # (50, 1)

# 关于一次性求的方式：
# 因为 u 是向量，要给每个输入对应的 grad_outputs
# du_dx, du_dy = torch.autograd.grad(
#     u, [x, y],
#     grad_outputs=torch.ones_like(u)
# )
# 这种写法在某些版本可能行为不同，本课程统一用分别求的方式
```

**验收标准**：
- 例 1 的偏导和手算一致（12 和 31）
- 例 2 的 du/dx 和 du/dy shape 都是 (50, 1)
- 你能解释为什么第一次 `autograd.grad` 后要加 `retain_graph=True`

---

#### ⚠️ Day 4 新手最容易踩的坑

**坑 1：忘了 `retain_graph=True`，第二次求导报错**
- 错误：`RuntimeError: Trying to backward through the graph a second time...`
- 原因：第一次求导后计算图释放，第二次没东西可遍历
- 解决：第一次（除了最后一次）求导加 `retain_graph=True`

**坑 2：所有求导都加 `retain_graph=True`**
- 不必要——只有"还要再求"的那次需要
- 加多了浪费内存
- 经验法则：**最后一次求导不加，前面都加**

**坑 3：把 `retain_graph` 和 `create_graph` 混淆**
- `retain_graph=True`：保留**当前**计算图，允许多次求一阶导（本周用）
- `create_graph=True`：建立**对梯度的新计算图**，允许求高阶导（Week 9 用）
- 两者独立——一阶多次求用 retain，求高阶用 create

**坑 4：偏导写错对象**
- 想算 ∂u/∂x 写成了 ∂u/∂xy（把 xy 整体当输入）
- 在 MLP2D 里特别要注意——输入是 x 和 y 两个独立的 tensor，不是合并后的 xy
- 习惯：**始终把每个变量分开管理**

---

#### 🧠 Day 4 概念问答 quiz

**Q1**：什么时候需要 `retain_graph=True`？

<details><summary>答案</summary>

当**同一个计算图要多次求导**时。比如先求 ∂f/∂x，再求 ∂f/∂y——第一次求导后默认计算图被释放，第二次求导会报错。加 `retain_graph=True` 保留计算图。</details>

**Q2**：`retain_graph=True` 和 `create_graph=True` 有什么区别？

<details><summary>答案</summary>

`retain_graph`：保留当前的前向计算图，允许多次一阶求导。`create_graph`：把求导本身也加入新的计算图（让得到的梯度也是"可微的"），用于求高阶导（Week 9 学）。两者独立。</details>

**Q3**：以下代码哪里有问题？
```python
df_dx = autograd.grad(f, x)[0]
df_dy = autograd.grad(f, y)[0]
```

<details><summary>答案</summary>

第一次 autograd.grad 没加 `retain_graph=True`，计算图被释放；第二次报错 "Trying to backward through the graph a second time"。修复：第一次加 `retain_graph=True`。</details>

**Q4**：要算 100 个采样点处的 ∂u/∂x 和 ∂u/∂y，每个偏导的 shape 是什么？

<details><summary>答案</summary>

都是 `(100, 1)`（如果 x 和 y 都是 (100, 1)）。每个采样点对应一个偏导值。</details>

**Q5**：`torch.autograd.grad(f, [x, y])` 和分别两次求导，结果有差别吗？

<details><summary>答案</summary>

**结果完全相同**。一次性求是简洁写法，PyTorch 内部会智能复用计算图。但本课程为了清晰统一用分别求的方式（前面加 `retain_graph=True`）。</details>

---

#### 📦 Day 4 知识卡片

| 项目 | 内容 |
|---|---|
| **核心 API** | `autograd.grad(f, x, retain_graph=True)` |
| **`retain_graph` 用途** | 多次求一阶导（保留前向图） |
| **`create_graph` 用途** | 求高阶导（Week 9） |
| **多偏导写法** | 分别求（最后一次不加 retain）或合在一起（list 输入） |
| **常见错误** | 忘 retain_graph；retain 和 create 混淆 |
| **本日产出** | `week08/day04_partial_derivatives.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 5-6（周末）| 综合练习：用 autograd 验证函数关系

**🎯 本日目标**：综合运用本周学的所有 autograd 技能；观察 autograd 求导的精度受训练质量影响。

**🟣 代码层**（约 2 小时）：

创建文件 `week08/weekend_grad_exploration.py`：

**任务**：训练一个 MLP 学 `u = x³ - 2x`，用 autograd 算 `du/dx`，与真实导数 `3x² - 2` 对比。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ========== Step 1: 定义和训练 MLP ==========
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 训练数据：u = x³ - 2x，x ∈ [-2, 2]
x_train = torch.linspace(-2, 2, 200).reshape(-1, 1)
u_train = x_train ** 3 - 2 * x_train

for epoch in range(5000):
    u_pred = model(x_train)
    loss = criterion(u_pred, u_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"训练完成，最终 loss: {loss.item():.6f}")

# ========== Step 2: 计算测试点的 du/dx ==========
x_test = torch.linspace(-2.5, 2.5, 200).reshape(-1, 1)   # 测试范围比训练大
x_test.requires_grad_(True)

u_test = model(x_test)
du_dx = torch.autograd.grad(
    u_test, x_test,
    grad_outputs=torch.ones_like(u_test)
)[0]

# ========== Step 3: 真实值对比 ==========
x_np = x_test.detach().numpy().flatten()
u_pred_np = u_test.detach().numpy().flatten()
du_dx_np = du_dx.detach().numpy().flatten()

# 真实
u_true_np = (x_test.detach() ** 3 - 2 * x_test.detach()).numpy().flatten()
du_dx_true = (3 * x_test.detach() ** 2 - 2).numpy().flatten()

# 误差
abs_error = abs(du_dx_np - du_dx_true)

# ========== Step 4: 三张图 ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x_np, u_true_np, 'b-', label='True u(x)')
axes[0].plot(x_np, u_pred_np, 'r--', label='MLP u(x)')
axes[0].axvspan(-2, 2, alpha=0.2, color='green', label='Train range')
axes[0].set_title('u(x) = x³ - 2x')
axes[0].legend()

axes[1].plot(x_np, du_dx_true, 'b-', label='True du/dx')
axes[1].plot(x_np, du_dx_np, 'r--', label='autograd du/dx')
axes[1].axvspan(-2, 2, alpha=0.2, color='green')
axes[1].set_title('du/dx = 3x² - 2')
axes[1].legend()

axes[2].plot(x_np, abs_error, 'g-')
axes[2].axvspan(-2, 2, alpha=0.2, color='green')
axes[2].set_title('|du/dx 误差|')
axes[2].set_yscale('log')
axes[2].set_xlabel('x')

plt.tight_layout()
plt.savefig('weekend_grad_exploration.png')

# ========== 反思：在脚本末尾的注释里回答 ==========
"""
观察任务（写在这里）：

1. 在哪些 x 值附近，du/dx 的误差最大？
   答：

2. 这是因为训练数据在那些区域覆盖不足吗？
   答：

3. 如果你扩大训练范围到 [-3, 3]，会发生什么？
   答：

(可选实验：扩大训练范围跑一次，看误差曲线变化)
"""
```

**验收标准**：
- 三张图能生成
- 训练范围内 `du/dx` 误差应较小
- 训练范围外（如 |x| > 2）误差应明显变大——**这就是"外推"问题**

---

### ✅ Week 8 完成评估

#### 任务级 checklist

- [✅] `week08/day01_backward_vs_grad.py` 跑通，理解差异
- [✅] `week08/day02_first_derivatives.py` 三个例子的导数正确
- [✅] `week08/day03_grad_mlp_output.py` MLP 学 sin、autograd 算出 cos
- [✅] `week08/day04_partial_derivatives.py` 偏导数正确
- [✅] `week08/weekend_grad_exploration.py` 误差图分析

#### 概念级 quiz（10 题，至少 8 题对）

**Q1**：`.backward()` 和 `autograd.grad()` 的核心区别？

<details><summary>答案</summary>

backward 修改 .grad（用于训练），grad 返回 tuple（用于通用求导，不污染 .grad）。</details>

**Q2**：PINN 中**必须**用 autograd.grad 而不是 .backward 算 du/dx 的原因？

<details><summary>答案</summary>

backward 会污染 param.grad，干扰后续训练梯度。autograd.grad 不修改 .grad。</details>

**Q3**：y 是向量时调用 autograd.grad 必须给什么参数？

<details><summary>答案</summary>

`grad_outputs=torch.ones_like(y)`，告诉 PyTorch 怎么把向量"加权求和"成标量。</details>

**Q4**：PINN 必须用 Tanh 而不是 ReLU 的两个原因？

<details><summary>答案</summary>

(1) ReLU 在 0 处不可导（尖角）；(2) ReLU 二阶导处处为 0，PINN 涉及二阶导的物理 Loss 失效。</details>

**Q5**：训练好的 MLP 学了 sin(x)，没学 cos(x)，autograd 为什么能算出 cos(x)？

<details><summary>答案</summary>

MLP 是可微函数，autograd 沿计算图反向遍历自动应用链式法则。学到 u(x) = sin(x) 后，du/dx 自然是 cos(x)。</details>

**Q6**：求导用的 x 必须 `requires_grad=True` 吗？

<details><summary>答案</summary>

**是**。autograd 只追踪 requires_grad=True 的 tensor。</details>

**Q7**：`retain_graph=True` 什么时候用？

<details><summary>答案</summary>

同一个计算图要多次求一阶导时（如同时求 ∂f/∂x 和 ∂f/∂y）。最后一次求导可不加。</details>

**Q8**：`retain_graph` 和 `create_graph` 有什么区别？

<details><summary>答案</summary>

retain：保留前向图，允许多次一阶求导。create：把求导本身加入新计算图，允许求高阶导（Week 9 用）。</details>

**Q9**：训练阶段（拟合数据）需要给训练数据 x 设 `requires_grad=True` 吗？

<details><summary>答案</summary>

不需要。训练时数据是固定的，不是对它求导的对象。求导阶段（用训练好的 MLP 算 du/dx）才需要。</details>

**Q10**：PINN 的标准训练循环里，autograd.grad 和 .backward() 怎么分工？

<details><summary>答案</summary>

autograd.grad 用于循环中间求 du/dx 等（构造物理 Loss）；最后 total_loss.backward() 算总 Loss 对参数的梯度，optimizer.step() 更新。两者**协作**，不冲突。</details>

#### 🚦 自我评估

- 任务全部通过 + Quiz 8 题对 → **绿灯进入 Week 9**
- 不能解释 backward vs grad 的区别 → **黄灯**——重看 Day 1
- ReLU vs Tanh 选择不熟 → **黄灯**——重看 Day 3 应用层

#### 知识地图自查

- [✅] (1) backward vs grad 的区别 → 🟣
- [✅] (2) 一阶导数 → 🟣
- [✅] (3) 对 MLP 输出求导 → 🟣
- [✅] (4) 偏导数 → 🟣

---

## 进入 Week 9 之前

下周 Week 9 开始引入**物理约束 Loss** 的真正核心：

- **二阶导数** + `create_graph=True`
- **PDE 残差** 写成 Loss
- **第一次"无数据"训练**（仅靠物理约束）
- **加边界条件**

下周需要的前置：
- Week 8 的 autograd.grad 一阶导能熟练写
- 理解 ReLU 为什么不能用（Week 9 会反复涉及二阶导）
- **不需要新数学**

下周开始你会**第一次写 PINN**——这是你目标方向的真实出现。

Week 8 完成。

---

---

## Week 9: 高阶导数 + 物理约束 Loss 初探

### 🎯 Week 9 总览

**本周覆盖的知识点**：(5) `create_graph=True`、(6) MLP 输出的二阶导、(7) PDE 残差作为 Loss、(8) 加边界条件

**本周不覆盖**：完整 PINN 框架（Week 10）、权重平衡（Week 10）

**学完之后你应该能**：
- 🟣 **能用**：(5) 算二阶导数、(6) 对 MLP 算二阶导、(7) 把 ODE 残差写成 Loss
- 🟡 **能讲**：为什么 PDE Loss 不够、还需要 BC Loss

**本周的特点**：**Part B 的核心周**。所有 PINN 的"灵魂"都在这周——从 Day 3 开始你才真正接触到 PINN 的核心思想（用方程残差作为 Loss）。每天的内容相对简短但概念密度高。

---

### ✅ 进入 Week 9 之前的前置 checklist

- [✅] Week 8 的所有 quiz 我都能答对
- [✅] 我能用 autograd 算 MLP 的一阶导数（du/dx）
- [✅] 我能解释 ReLU 不能用于 PINN 的原因
- [✅] 我接受"本周开始密集学物理方程，但只用最简单的"
- [✅] 我能在纸上写出"如果 dy/dx = -2x，那么 y = ?"（答：y = -x² + C）

---

### Day 1 | 二阶导数：`create_graph=True` 的作用

**🎯 本日目标**：理解 `create_graph=True` 在算二阶导时的作用；能算简单函数的二阶导数。

**🟢 直觉层**（约 10 分钟）：

很多物理方程涉及二阶导数：

- 热传导方程：`∂T/∂t = α · ∂²T/∂x²`
- 弹性力学（带均匀载荷）：`E · ∂²u/∂x² + f = 0`
- 板的弯曲方程：涉及 `∂⁴w/∂x⁴`（四阶导）

要算二阶导，你需要：
1. 先算一阶导 `dy/dx`
2. 再对 `dy/dx` 求一次导

关键问题：**第二次求导时，PyTorch 需要知道 dy/dx 是怎么算出来的**——它需要"算一阶导这一步本身"也在计算图里。这就是 `create_graph=True` 做的事。

**🟡 概念层**（约 20 分钟）：

**`create_graph=True` 的本质**：

正常情况下：`autograd.grad(y, x)` 算出 dy/dx 后，**PyTorch 把"算 dy/dx 这一步"也消除掉**——dy/dx 是一个普通张量，没有计算历史。

加 `create_graph=True` 后：**dy/dx 仍然在计算图里**——它"记得自己是怎么算出来的"。所以再对 dy/dx 求导，PyTorch 能继续用链式法则算二阶导。

**标准模式**：

```python
# 一阶导：必须 create_graph=True（因为还要算二阶）
dy_dx = torch.autograd.grad(
    y, x,
    grad_outputs=torch.ones_like(y),
    create_graph=True            # ⚠️ 关键
)[0]

# 二阶导：最后一次，可不加 create_graph
d2y_dx2 = torch.autograd.grad(
    dy_dx, x,
    grad_outputs=torch.ones_like(dy_dx),
    create_graph=False           # 最后一次，可以 False
)[0]
```

**记忆方法**：**create_graph=True 用在"还要继续求导"的那次**。最后一次（不再求导了）可以 False。

**`create_graph` 和 `retain_graph` 的关系（重要）**：

- `retain_graph=True`：保留**前向**计算图（用于多次一阶求导，Day 4 的偏导）
- `create_graph=True`：让**求导本身**也进入计算图（用于求高阶导）
- 两者独立：可以单独用，也可以一起用

实际上 `create_graph=True` 会自动隐含 `retain_graph=True`（因为求导步骤要进新图，前向图自然得保留）。所以**算高阶导时 retain_graph 通常不用单独写**。

**🔵 数学层**（约 5 分钟）：

例子：`y = x³`

- `dy/dx = 3x²`
- `d²y/dx² = 6x`

在 x = 2：
- `dy/dx = 12`
- `d²y/dx² = 12`

**🟣 代码层**（约 1.5 小时）：

创建文件 `week09/day01_second_order.py`：

```python
import torch

# ========== 例 1：标量情况 ==========
print("===== 例 1：y = x³，求 d²y/dx² =====")

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3                                  # y = 8

# 一阶导：必须 create_graph=True
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx at x=2: {dy_dx.item()}")     # 应为 12（3*2² = 12）

# 二阶导
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"d²y/dx² at x=2: {d2y_dx2.item()}") # 应为 12（6*2 = 12）

# ========== 例 2：向量情况 ==========
print("\n===== 例 2：y = x³ 在 5 个点 =====")

x = torch.linspace(0, 2, 5, requires_grad=True)   # [0, 0.5, 1, 1.5, 2]
y = x ** 3                                          # [0, 0.125, 1, 3.375, 8]

# 一阶导
dy_dx = torch.autograd.grad(
    y, x,
    grad_outputs=torch.ones_like(y),
    create_graph=True
)[0]
print(f"dy/dx: {dy_dx.tolist()}")          # 应为 [0, 0.75, 3, 6.75, 12]

# 二阶导
d2y_dx2 = torch.autograd.grad(
    dy_dx, x,
    grad_outputs=torch.ones_like(dy_dx),
    create_graph=False
)[0]
print(f"d²y/dx²: {d2y_dx2.tolist()}")      # 应为 [0, 3, 6, 9, 12]

# ========== 反例：忘了 create_graph=True ==========
print("\n===== 反例：忘了 create_graph =====")
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
dy_dx = torch.autograd.grad(y, x)[0]    # 没加 create_graph

try:
    d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
except Exception as e:
    print(f"报错: {e}")
# 报错：dy_dx 不在计算图里，没法求导
```

**验收标准**：
- 例 1 和例 2 的导数都和手算一致
- 反例能复现报错（看到具体错误信息）
- 你能解释 `create_graph=True` 在做什么

---

#### ⚠️ Day 1 新手最容易踩的坑

**坑 1：忘了在一阶导上加 `create_graph=True`**
- 报错：`element 0 of tensors does not require grad and does not have a grad_fn`
- 原因：dy_dx 是普通张量，不在计算图里
- 修复：第一次求导加 `create_graph=True`

**坑 2：每次求导都加 `create_graph=True`**
- 不必要——最后一次（不再求导了）加了浪费内存
- 经验：**N 阶导，前 N-1 次加 create_graph，最后一次不加**

**坑 3：以为 `retain_graph=True` 就够了**
- `retain_graph=True` 只保留前向图，**不让求导步骤进新图**
- 第二次对 dy_dx 求导仍然报错
- 必须用 `create_graph=True`

**坑 4：算二阶导时 `grad_outputs` 用错**
- 二阶导时，`grad_outputs` 应该是 `torch.ones_like(dy_dx)`（不是 `torch.ones_like(y)`）
- shape 要和 outputs 一致——这次 outputs 是 dy_dx

**坑 5：把 ReLU MLP 当 PINN 网络用**
- ReLU(z) 是分段线性，二阶导处处为 0
- 跑代码不报错，但二阶导全是 0——物理 Loss 失效
- **必须用 Tanh 等光滑激活**

---

#### 🧠 Day 1 概念问答 quiz

**Q1**：算二阶导数为什么需要 `create_graph=True`？

<details><summary>答案</summary>

第二次求导时 PyTorch 需要"算第一次求导这一步"也在计算图里，否则 dy_dx 是普通张量没法继续求导。`create_graph=True` 让求导这一步进入新计算图。</details>

**Q2**：要算 4 阶导，应该在哪几次 `autograd.grad` 中加 `create_graph=True`？

<details><summary>答案</summary>

**前 3 次**（一、二、三阶时加），第 4 次（最后一次）不加。规则：N 阶导加 N-1 次 create_graph。</details>

**Q3**：`create_graph` 和 `retain_graph` 能单独使用吗？

<details><summary>答案</summary>

**能，且作用不同**。retain：保留前向图，多次一阶求导。create：让求导本身进新图，求高阶导。`create_graph=True` 自动隐含 retain，所以求高阶导时通常只写 create_graph。</details>

**Q4**：如果你想算 d²y/dx²，但发现 y 是用 ReLU MLP 算出来的，结果是什么？

<details><summary>答案</summary>

所有点的二阶导是 0（ReLU 是分段线性，二阶导处处为 0）。代码不报错但物理 Loss 失效。必须改用 Tanh 等光滑激活。</details>

**Q5**：算 `d²y/dx²` 时，最里面那次 grad 调用的 `grad_outputs` 应该用什么 shape？

<details><summary>答案</summary>

**要和 outputs（dy_dx）的 shape 一致**——通常是 `torch.ones_like(dy_dx)`。不是 `torch.ones_like(y)`。</details>

---

#### 📦 Day 1 知识卡片

| 项目 | 内容 |
|---|---|
| **核心 API** | `autograd.grad(..., create_graph=True)` |
| **关键规则** | N 阶导加 N-1 次 create_graph |
| **create vs retain** | retain：保留前向图（多次一阶导）；create：求导步骤进新图（求高阶导） |
| **激活函数要求** | 必须光滑可导（Tanh / GELU / SiLU）；不能用 ReLU |
| **常见错误** | 忘 create_graph；用 ReLU 网络；grad_outputs 的 shape 错 |
| **本日产出** | `week09/day01_second_order.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 2 | MLP 输出的二阶导数

**🎯 本日目标**：把 Day 1 的二阶导扩展到训练好的 MLP；观察一阶导和二阶导的精度差异。

**🟢 直觉层**（约 5 分钟）：

Day 1 的 `y = x³` 是手写函数，二阶导对得上。但**真实 PINN 里 y 是 MLP 学出来的**——它学的是个近似，所以求导也会有误差。

今天的实验：让 MLP 学 `u = sin(πx)`，然后算 `du/dx` 和 `d²u/dx²`，看精度怎么样。

**关键观察**：通常一阶导**精度还可以**，二阶导**精度会明显变差**（误差累积）。

**🟡 概念层**（约 10 分钟）：

为什么二阶导精度会变差？

直觉解释：MLP 拟合 u(x) 的误差是 ε。求一阶导后，误差变成 dε/dx——这通常和原误差量级类似。但二阶导是 d²ε/dx²——**二次微分会放大原误差的"高频部分"**（细微的振荡被求导放大）。

实践经验：
- 一阶导：精度通常和函数本身相当（甚至差不多）
- 二阶导：精度可能差 5-10 倍
- 三阶导往上：通常没法用

这就是为什么 **PINN 应用大多停在二阶导**——再高阶 autograd 算出来的"导数"没什么物理意义。

**🟣 代码层**（约 1.5 小时）：

创建文件 `week09/day02_mlp_second_order.py`：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)

class MLP(nn.Module):
    """PINN 用 MLP——必须 Tanh"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

# ========== Step 1: 训练 MLP 学 sin(πx) ==========
x_train = torch.linspace(0, 1, 100).reshape(-1, 1)
u_train = torch.sin(torch.pi * x_train)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("训练 MLP 学 sin(πx)...")
for epoch in range(5000):
    u_pred = model(x_train)
    loss = ((u_pred - u_train) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"  epoch {epoch}: loss = {loss.item():.6f}")

# ========== Step 2: 计算 u, du/dx, d²u/dx² ==========
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
x_test.requires_grad_(True)

u = model(x_test)

# 一阶导（要再求二阶，所以 create_graph=True）
du_dx = torch.autograd.grad(
    u, x_test,
    grad_outputs=torch.ones_like(u),
    create_graph=True
)[0]

# 二阶导（最后一次，可不加 create_graph）
d2u_dx2 = torch.autograd.grad(
    du_dx, x_test,
    grad_outputs=torch.ones_like(du_dx),
    create_graph=False
)[0]

# ========== Step 3: 真实值对比 ==========
x_np = x_test.detach().numpy().flatten()
u_pred_np = u.detach().numpy().flatten()
du_np = du_dx.detach().numpy().flatten()
d2u_np = d2u_dx2.detach().numpy().flatten()

# 真实
u_true = (torch.sin(torch.pi * x_test)).detach().numpy().flatten()
du_true = (torch.pi * torch.cos(torch.pi * x_test)).detach().numpy().flatten()
d2u_true = (-torch.pi**2 * torch.sin(torch.pi * x_test)).detach().numpy().flatten()

# ========== Step 4: 画三张图 ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x_np, u_true, 'b-', label='True')
axes[0].plot(x_np, u_pred_np, 'r--', label='MLP')
axes[0].set_title('u = sin(πx)')
axes[0].legend()

axes[1].plot(x_np, du_true, 'b-', label='True π·cos(πx)')
axes[1].plot(x_np, du_np, 'r--', label='autograd')
axes[1].set_title('du/dx')
axes[1].legend()

axes[2].plot(x_np, d2u_true, 'b-', label='True -π²·sin(πx)')
axes[2].plot(x_np, d2u_np, 'r--', label='autograd')
axes[2].set_title('d²u/dx²（注意误差！）')
axes[2].legend()

plt.tight_layout()
plt.savefig('week09_second_order.png')

# ========== Step 5: 量化误差 ==========
print("\n精度对比:")
print(f"u 平均绝对误差: {abs(u_pred_np - u_true).mean():.6f}")
print(f"du/dx 平均绝对误差: {abs(du_np - du_true).mean():.6f}")
print(f"d²u/dx² 平均绝对误差: {abs(d2u_np - d2u_true).mean():.6f}")
```

**验收标准**：
- 三张图能生成
- u 拟合良好
- du/dx 和真实值大致重合
- d²u/dx² 的误差比 u 和 du/dx 都大（这是正常现象）

---

#### ⚠️ Day 2 新手最容易踩的坑

**坑 1：发现 d²u/dx² 误差大就以为代码错了**
- 二阶导精度差是正常现象，不是 bug
- 通常的工程经验：一阶导误差是 1x、二阶导是 5-10x

**坑 2：训练不够导致整体不准**
- 如果 MLP 拟合 u(x) 都不好，求导自然更不好
- 标志：u 的曲线和真实有明显偏差
- 解决：训练更多 epoch（10000+），或增大网络

**坑 3：测试范围超出训练范围**
- 训练数据 x ∈ [0, 1]，测试 x ∈ [-1, 2] 时 MLP 是"外推"，效果差
- **物理问题里要让训练采样覆盖整个待求域**

**坑 4：忘了 detach 转 numpy**
- `du_dx.numpy()` 报错
- 必须 `du_dx.detach().numpy()`

---

#### 🧠 Day 2 概念问答 quiz

**Q1**：MLP 学了 sin(πx)，autograd 算 d²u/dx² 应该是多少？

<details><summary>答案</summary>`

-π² · sin(πx) ≈ -9.87 sin(πx)`。一阶导是 π·cos(πx)，二阶是 -π²·sin(πx)。</details>

**Q2**：为什么 MLP 求导的二阶导精度通常比一阶差？

<details><summary>答案</summary>

MLP 拟合的高频误差通过求导被放大。每次求导大致放大 5-10 倍误差。这是 PINN 的固有限制——理论上能算高阶导，但精度可能不够用。</details>

**Q3**：要让二阶导更精确，可以怎么做？

<details><summary>答案</summary>

(1) 训练更久；(2) 用更大的 MLP；(3) 用更光滑的激活函数（Tanh 是标准）；(4) 训练数据更密；(5) 在 PINN 训练时直接把 d²u/dx² 加入物理 Loss（让网络主动学好二阶导）。</details>

**Q4**：算 `d²u/dx²` 时，PyTorch 真的"算了两次导数"吗？

<details><summary>答案</summary>

是的——它在计算图上反向遍历两次。第一次得到 du/dx，第二次再对 du/dx 求导。这意味着**计算量大约是一次正常 forward 的 2 倍**（一阶+二阶）。三阶导是 3 倍，依此类推。</details>

**Q5**：如果你算 d²u/dx² 时把 `create_graph=True` 加在了二阶那一步而不是一阶，会怎样？

<details><summary>答案</summary>

**会报错**——一阶 du/dx 不在计算图里，无法对它求二阶导。规则是"前面的求导都加 create_graph，最后一次可不加"。</details>

---

#### 📦 Day 2 知识卡片

| 项目 | 内容 |
|---|---|
| **核心能力** | 算训练好的 MLP 的二阶导数 |
| **精度规律** | 一阶导精度好；二阶导误差大约 5-10x |
| **典型 PINN 极限** | 二阶导（再高精度不够） |
| **代码模式** | 一阶 `create_graph=True` → 二阶 `create_graph=False` |
| **常见错误** | 误把误差当 bug；测试范围超过训练范围 |
| **本日产出** | `week09/day02_mlp_second_order.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 3 | 物理约束 Loss 的核心思想

✅ **🎯 本日目标**：理解"PDE 残差作为 Loss"的核心思想；能把一个 ODE 写成残差形式。

**🟢 直觉层**（约 10 分钟）：

到 Day 2 为止，你的 MLP 都是**靠数据训练**——给一堆 (x, y) 对，模型学一个 y(x)。

**今天换一个角度思考**：如果你**没有 (x, y) 对**，但你**知道方程**——比如知道 `du/dx = -2x`——能不能让 MLP 学到这个方程的解 `u(x)`？

直觉上应该可以——只要你**告诉 MLP**："你输出的 u(x) 必须满足这个方程"。这就是 PINN 的核心思想：**用方程作为监督信号**。

**🟡 概念层**（约 25 分钟）：

**核心概念：PDE 残差（PDE Residual）**

考虑一个 ODE：
```
du/dx = -2x
```

如果 MLP 输出的 u(x) 完美满足这个方程，那么 `du/dx + 2x` 应该处处等于 0。

把 `du/dx + 2x` 叫做这个方程的**残差**——残差越接近 0，方程越被满足。

**关键步骤**（怎么把残差用作 Loss）：

1. 把方程写成"残差 = 0"形式：`du/dx + 2x = 0`
2. 对 MLP 输出求 du/dx
3. 算残差：`r = du/dx + 2x`
4. 把残差的平方平均作为 Loss：`L_PDE = mean(r²)`
5. 训练让 L_PDE 减小 → 网络逼近满足方程的 u(x)

**写残差的通用流程**：

| ODE | 残差形式 |
|---|---|
| `du/dx = sin(x)` | `r = du/dx - sin(x)` |
| `du/dx + u = 0` | `r = du/dx + u` |
| `d²u/dx² - 3u = 0` | `r = d²u/dx² - 3u` |
| `d²u/dx² + u = sin(x)` | `r = d²u/dx² + u - sin(x)` |

**通用规则**：把等号右边移到左边，得到的表达式就是残差。

**为什么要平方**：

- 残差有正有负，直接平均会抵消
- 平方让所有残差都变正（不抵消）
- 这就是 MSE 的逻辑（Part A Week 1 学过）

**关键理解**：**这个 Loss 不依赖任何标注数据**。它的"监督信号"是方程本身——只要你能写出方程，就能写出 Loss。

**🔵 数学层**（约 5 分钟）：

无新数学。把方程写成残差只是简单的代数移项。

**🟣 理论任务**（无代码，约 30 分钟）：

在纸上写出以下 ODE 的残差表达式：

| ODE | 你的残差 | 参考答案 |
|---|---|---|
| `du/dx = sin(x)` | ? | `r = du/dx - sin(x)` |
| `d²u/dx² - 3u = 0` | ? | `r = d²u/dx² - 3u` |
| `du/dx + 2u = 5` | ? | `r = du/dx + 2u - 5` |
| `d²u/dx² + 4·du/dx + 4u = 0` | ? | `r = d²u/dx² + 4·du/dx + 4u` |
| `du/dx · u = x²` | ? | `r = du/dx · u - x²` |

每一个都自己写一次，然后核对答案。

---

#### ⚠️ Day 3 新手最容易踩的坑

**坑 1：忘了把右边移到左边**
- 错误：直接把 `du/dx = sin(x)` 写成残差，没移项
- 正确：移项后 `r = du/dx - sin(x)`

**坑 2：把残差弄成"残差 = 某个非零值"**
- 错误：`r = du/dx`，希望它等于 `-2x`
- 正确：`r = du/dx + 2x`，希望它等于 0
- **残差的目标永远是 0**

**坑 3：以为残差里只能有导数项**
- 错误：以为只能写 `du/dx`、`d²u/dx²`
- 正确：残差里**可以同时有 u、du/dx、d²u/dx² 等各种项**——只要它们出现在方程里

**坑 4：把"边界条件"和"PDE 残差"混淆**
- 边界条件：在特定点的约束（比如 `u(0) = 1`）——是单点 Loss
- PDE 残差：在整个域内的方程约束——是积分 Loss
- 两者**都要**才能确定唯一解，下周末会一起做

---

#### 🧠 Day 3 概念问答 quiz

**Q1**：什么是"PDE 残差"？

<details><summary>答案</summary>

把方程写成"... = 0"形式，等号左边的表达式就是残差。残差越接近 0，方程越被满足。</details>

**Q2**：把方程 `du/dx + u = sin(x)` 写成残差形式。

<details><summary>答案</summary>

`r = du/dx + u - sin(x)`。把右边的 sin(x) 移到左边变成减号。</details>

**Q3**：PDE 残差 Loss 的形式是什么？

<details><summary>答案</summary>

`L_PDE = mean(r²)`。把残差的平方平均作为 Loss。形式上和 MSE 一样，只是被预测的"目标"是 0。</details>

**Q4**：为什么 PDE 残差 Loss 不需要标注数据？

<details><summary>答案</summary>

它的"监督信号"是方程本身——只要你能写出方程，就能算残差。和数据驱动的 MSE Loss 不一样（后者需要 (x, y_true) 对）。</details>

**Q5**：如果 MLP 完美满足方程 `du/dx = -2x`，那么在 x = [0, 0.5, 1.0] 这三个点处的残差分别是多少？

<details><summary>答案</summary>

**全是 0**。完美满足意味着 du/dx + 2x = 0 处处成立，三个点的残差都是 0。</details>

---

#### 📦 Day 3 知识卡片

| 项目 | 内容 |
|---|---|
| **核心概念** | PDE 残差、物理约束 Loss |
| **写残差通用规则** | 把方程写成"... = 0"形式 |
| **物理 Loss 形式** | `L_PDE = mean(r²)` |
| **关键性质** | 不需要标注数据，方程本身就是监督信号 |
| **常见错误** | 忘移项；目标不是 0；把 BC 和 PDE 残差混淆 |
| **本日产出** | 一张写满残差表达式的纸 |
| **掌握要求** | 🟡 能讲（能把方程写成残差） |

---

### Day 4 | 第一个物理约束 Loss 实验

**🎯 本日目标**：完整跑通"无数据 + 仅靠物理约束"的 PINN 训练；观察网络能学到什么。

**🟢 直觉层**（约 5 分钟）：

今天的实验是真正的"PINN 第一次"——**没有任何标注数据，只用方程作为约束**。

**问题**：用 PINN 求解 `du/dx = -2x`。

**已知信息**：
- 这个 ODE 的通解是 `u(x) = -x² + C`，其中 C 是任意常数
- 因为没给边界条件，C 不能确定
- 所以网络应该学到"形状是 -x² + 某个常数"

**今天的预期**：
- L_PDE 能下降到很小（方程约束被满足）
- u(x) 的形状接近抛物线（-x² 形状）
- 但具体的 C 值**每次训练都不一样**（取决于初始化）

**这正好暴露了一个问题**：**只用方程不够**——还需要边界条件锁定唯一解。这就是 Day 5-6 周末要做的。

**🟡 概念层**（约 15 分钟）：

**核心概念：配点（Collocation Points）**

PINN 训练时，**在域内随机采样若干点 x**——这些点叫"配点"。在每个配点上算 PDE 残差并求 Loss。

为什么用"随机采样"而不是固定网格？
- 固定网格：可能错过域内某些区域的精度
- 随机采样：每次 epoch 看不同的点，能更全面地约束方程

PhyFENet 论文里的"采样点策略"就是这个意思。

**关键代码模式**：

```python
for epoch in range(N):
    # 1. 随机采样 100 个配点
    x = torch.rand(100, 1, requires_grad=True) * (x_max - x_min) + x_min
    
    # 2. 网络前向
    u = model(x)
    
    # 3. 算 du/dx
    du_dx = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    # 4. 残差
    residual = du_dx + 2 * x
    L_PDE = (residual ** 2).mean()
    
    # 5. 反向 + 更新
    optimizer.zero_grad()
    L_PDE.backward()
    optimizer.step()
```

**注意**：第 5 步**还是用 `loss.backward()`**——因为现在要更新参数，正常训练流程。`autograd.grad` 只用在第 3 步算中间导数。

**🟣 代码层**（约 1.5 小时）：

创建文件 `week09/day04_first_physics_loss.py`：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

torch.manual_seed(0)
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

losses = []

# ========== 训练循环：仅有物理约束 Loss ==========
print("PINN 训练：求解 du/dx = -2x，无数据，无边界条件")
for epoch in range(5000):
    # 1. 在 [0, 1] 上随机采样配点
    x = torch.rand(100, 1, requires_grad=True)
    
    # 2. 网络前向
    u = model(x)
    
    # 3. 算 du/dx
    du_dx = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    # 4. 物理约束 Loss
    residual = du_dx + 2 * x
    L_PDE = (residual ** 2).mean()
    losses.append(L_PDE.item())
    
    # 5. 反向 + 更新
    optimizer.zero_grad()
    L_PDE.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"  epoch {epoch}: L_PDE = {L_PDE.item():.6e}")

# ========== 验证：网络学到了什么 ==========
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
with torch.no_grad():
    u_pred = model(x_test)

# u(x) 应该是 -x² + C，所以 u(x) + x² 应该是常数 C
u_plus_x2 = u_pred + x_test ** 2

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss 曲线
axes[0].plot(losses)
axes[0].set_yscale('log')
axes[0].set_title('L_PDE 训练曲线')
axes[0].set_xlabel('Epoch')

# 网络输出 u(x)
axes[1].plot(x_test.numpy(), u_pred.numpy(), 'r-')
axes[1].set_title('u(x) 网络输出')
axes[1].set_xlabel('x'); axes[1].set_ylabel('u')

# u(x) + x² 应该是常数
axes[2].plot(x_test.numpy(), u_plus_x2.numpy(), 'g-')
axes[2].axhline(y=u_plus_x2.mean().item(), color='k', linestyle='--', 
                label=f'mean = {u_plus_x2.mean().item():.4f}')
axes[2].set_title('u(x) + x² 应该是常数（C）')
axes[2].set_xlabel('x')
axes[2].legend()

plt.tight_layout()
plt.savefig('week09_physics_only.png')

# ========== 关键观察 ==========
print(f"\n最终 L_PDE: {losses[-1]:.2e}")
print(f"u(x) + x² 的范围: [{u_plus_x2.min().item():.4f}, {u_plus_x2.max().item():.4f}]")
print(f"u(x) + x² 的标准差（应接近 0）: {u_plus_x2.std().item():.4f}")
print(f"\n说明：网络学到的是 u(x) ≈ -x² + C，这个 C 是随机的（取决于初始化）")
print("要确定 C，需要边界条件——下周末做")
```

**验收标准**：
- 最终 L_PDE < 1e-3
- `u(x) + x²` 应接近一个常数（标准差 < 0.01）
- 但具体的常数值随机（你和别人跑结果不一样）

---

#### ⚠️ Day 4 新手最容易踩的坑

**坑 1：在循环里用 `loss.backward()` 但忘了 zero_grad**
- 老问题（Part A Week 5 坑 1）——梯度累加，训练崩
- 标准模式：`zero_grad → loss.backward → step`

**坑 2：第 3 步的 `autograd.grad` 想用 `loss.backward()` 替代**
- 错误！原因 Day 1 详细讲过
- 第 3 步**必须**用 `autograd.grad`

**坑 3：在循环外只采样一次 x**
- 错误：`x = torch.rand(...); for epoch ...` —— 只用一组配点
- 正确：每个 epoch 重新采样新配点
- 原因：不同采样能更全面地覆盖域

**坑 4：以为第一次跑出来的 u(x) 应该是某个特定形状**
- 错误：以为应该是 `1 - x²` 或某个具体形状
- 正确：因为没边界条件，C 是随机的——你跑两次会得到不同的 C
- 这是**正确行为**，不是 bug

---

#### 🧠 Day 4 概念问答 quiz

**Q1**：什么是"配点"（Collocation Points）？

<details><summary>答案</summary>

在域内采样的若干点，用于在这些点上算 PDE 残差。每个 epoch 通常重新采样一组配点。</details>

**Q2**：为什么 PINN 训练循环每个 epoch 重新采样配点？

<details><summary>答案</summary>

(1) 减少过拟合到特定点；(2) 更全面地覆盖整个域；(3) 类似数据增强的效果。</details>

**Q3**：标准 PINN 训练循环里，`autograd.grad` 和 `loss.backward()` 分别在哪一步？

<details><summary>答案</summary>

autograd.grad 在循环内算 du/dx 等中间导数（构造物理 Loss）；loss.backward() 在最后算总 Loss 对参数的梯度（用于参数更新）。</details>

**Q4**：今天的实验为什么不需要标注数据？

<details><summary>答案</summary>

因为 Loss 是 PDE 残差，监督信号是方程本身。只要写出方程就能算残差并 Loss，不需要 (x, u_true) 对。</details>

**Q5**：实验中网络学到的 u(x) 每次训练都不同（差一个常数），原因是什么？

<details><summary>答案</summary>

ODE `du/dx = -2x` 的通解是 u(x) = -x² + C，C 任意。仅有 PDE 约束不能确定 C——它由初始化决定。要确定 C，需要边界条件（如 u(0) = 1）。</details>

---

#### 📦 Day 4 知识卡片

| 项目 | 内容 |
|---|---|
| **核心成就** | 第一次"无数据"训练 |
| **关键概念** | 配点（每 epoch 重新采样） |
| **训练循环模式** | 采样配点 → forward → autograd 算导数 → 残差 → backward |
| **关键限制** | 只有 PDE 约束，解不唯一（差一个常数） |
| **常见错误** | 配点不重采；想用 backward 算 du/dx |
| **本日产出** | `week09/day04_first_physics_loss.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 5-6（周末）| 加入边界条件

**🎯 本日目标**：在 Day 4 的基础上加边界条件 `u(0) = 1`，让网络学到唯一解 `u(x) = 1 - x²`。

**🟢 直觉层**（约 5 分钟）：

Day 4 看到的问题：仅 PDE 约束 → 解不唯一（差一个常数）。

解决方法：**加边界条件**——告诉网络"在 x=0 处，u 必须等于 1"。这样常数 C 被锁定为 1，唯一解就是 `u(x) = 1 - x²`。

**怎么把边界条件加到 Loss 里**：

```python
# BC Loss：在 x=0 处的预测值要接近 1
x_bc = torch.tensor([[0.0]])
u_bc_pred = model(x_bc)
L_BC = (u_bc_pred - 1.0) ** 2

# 总 Loss：PDE Loss + BC Loss
L_total = L_PDE + 10 * L_BC      # 暂时用 1:10 权重
```

为什么边界条件用 10 倍权重？因为 BC Loss 只有一个点，PDE Loss 有 100 个配点——量级差很多。Week 10 会专门讨论权重问题。

**🟡 概念层**（约 15 分钟）：

**两类边界条件**（PINN 中最常见）：

| 类型 | 形式 | 例子 |
|---|---|---|
| Dirichlet | `u(x_b) = 给定值` | u(0) = 1 |
| Neumann | `du/dx(x_b) = 给定值` | du/dx(L) = F |

今天用 Dirichlet（最简单的）。

**Dirichlet BC 的 Loss 形式**：

```
L_BC = mean((u(x_b) - u_b_true)²)
```

如果只有一个边界点，就是 `(u(x_b) - u_b_true)²`。

**总 Loss 的组合**：

```
L_total = w_PDE · L_PDE + w_BC · L_BC
```

权重 w 是超参数。今天先用 1:10。Week 10 会研究怎么自动平衡。

**🟣 代码层**（约 2 小时）：

创建文件 `week09/weekend_ode_with_bc.py`：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

torch.manual_seed(0)
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

L_pde_history, L_bc_history = [], []

# ========== PINN 训练：PDE + BC ==========
print("PINN 训练：du/dx = -2x，u(0) = 1")
for epoch in range(5000):
    # === PDE Loss ===
    x_pde = torch.rand(100, 1, requires_grad=True)
    u_pde = model(x_pde)
    du_dx = torch.autograd.grad(
        u_pde, x_pde,
        grad_outputs=torch.ones_like(u_pde),
        create_graph=True
    )[0]
    residual = du_dx + 2 * x_pde
    L_PDE = (residual ** 2).mean()
    
    # === BC Loss ===
    x_bc = torch.tensor([[0.0]])           # 边界点 x=0
    u_bc_pred = model(x_bc)                 # 在 x=0 处的预测
    u_bc_true = torch.tensor([[1.0]])       # 目标值 u(0) = 1
    L_BC = ((u_bc_pred - u_bc_true) ** 2).mean()
    
    # === 总 Loss（暂时手动 1:10）===
    L_total = L_PDE + 10 * L_BC
    
    L_pde_history.append(L_PDE.item())
    L_bc_history.append(L_BC.item())
    
    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"  epoch {epoch}: L_PDE={L_PDE.item():.4e}, L_BC={L_BC.item():.4e}")

# ========== 验证：和解析解 u(x) = 1 - x² 对比 ==========
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
with torch.no_grad():
    u_pred = model(x_test)
u_true = 1 - x_test ** 2

# ========== 画图 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss 曲线
axes[0].plot(L_pde_history, label='L_PDE')
axes[0].plot(L_bc_history, label='L_BC')
axes[0].set_yscale('log')
axes[0].set_title('训练 Loss')
axes[0].legend()

# 解对比
axes[1].plot(x_test.numpy(), u_true.numpy(), 'b-', label='True u = 1-x²')
axes[1].plot(x_test.numpy(), u_pred.numpy(), 'r--', label='PINN')
axes[1].set_title('PINN vs 解析解')
axes[1].legend()
axes[1].set_xlabel('x')

plt.tight_layout()
plt.savefig('week09_pinn_with_bc.png')

# ========== 量化 ==========
print(f"\n最终 L_PDE: {L_PDE.item():.4e}")
print(f"最终 L_BC: {L_BC.item():.4e}")
print(f"u(0) 实际预测值: {model(torch.tensor([[0.0]])).item():.4f}（应接近 1.0）")
print(f"u(1) 实际预测值: {model(torch.tensor([[1.0]])).item():.4f}（应接近 0.0）")

mae = (u_pred - u_true).abs().mean()
print(f"全域平均绝对误差 (MAE): {mae.item():.6f}")
```

**验收标准**：
- 训练后 u_pred 接近 `1 - x²`
- `u(0)` 误差 < 0.01
- `u(1)` 误差 < 0.05
- MAE < 0.01

**🔬 关键体会**（约 10 分钟）：

回顾这个实验的全过程：

1. **没有任何标注数据**——你没给一组 (x, u_true) 对
2. **只用了方程 + 边界条件**——`du/dx + 2x = 0` 和 `u(0) = 1`
3. **网络自动学出了正确的 u(x) = 1 - x²**

这就是 PINN 相对于纯数据驱动的核心优势：

- **数据驱动**（如 Part A）：需要大量标注数据 (x, y_true)。在 CAE 场景里要先跑 FEM 仿真生成数据
- **PINN**：只要你能写出方程，就不需要 FEM 数据——直接物理约束训练

**当然 PINN 也有局限**：
- 复杂方程权重难调（Week 10 学）
- 收敛速度可能慢
- 仅靠 PINN 求工业级问题精度可能不够——所以 PhyFENet 论文是**结合数据 + 物理**

---

### ✅ Week 9 完成评估

#### 任务级 checklist

- [ ] `week09/day01_second_order.py` 二阶导正确
- [ ] `week09/day02_mlp_second_order.py` 二阶导和真实差异可观察
- [ ] Day 3 的残差表达式纸（5 个）写完
- [ ] `week09/day04_first_physics_loss.py` 跑通，u(x) + x² 接近常数
- [ ] `week09/weekend_ode_with_bc.py` u_pred 接近 1 - x²

#### 概念级 quiz（10 题，至少 8 题对）

**Q1**：算 d²y/dx² 时，第一次 autograd.grad 必须加什么参数？

<details><summary>答案</summary>`create_graph=True`。让一阶求导步骤进入计算图，使二阶求导可用。</details>

**Q2**：PDE 残差的写法规则是什么？

<details><summary>答案</summary>把方程写成"... = 0"形式，等号左边表达式就是残差。</details>

**Q3**：写出 ODE `d²u/dx² + u = 0` 的残差。

<details><summary>答案</summary>`r = d²u/dx² + u`。</details>

**Q4**：PDE 残差 Loss 的形式是？

<details><summary>答案</summary>`L_PDE = mean(r²)`。残差平方的平均。</details>

**Q5**：为什么 PINN 训练循环每个 epoch 重新采样配点？

<details><summary>答案</summary>更全面覆盖整个域，类似数据增强。每 epoch 看不同的点防止过拟合。</details>

**Q6**：仅有 PDE Loss 不够的原因是？

<details><summary>答案</summary>ODE 通解差一个常数（积分常数），PDE 不能锁定唯一解。需要边界条件。</details>

**Q7**：Dirichlet BC 的 Loss 形式是什么？

<details><summary>答案</summary>`L_BC = mean((u(x_b) - u_b_true)²)`。在边界点上预测值要接近给定值。</details>

**Q8**：PINN 总 Loss 怎么组合？

<details><summary>答案</summary>`L_total = w_PDE · L_PDE + w_BC · L_BC`。权重 w 通常需要手调或自动平衡（Week 10）。</details>

**Q9**：PINN 相对于纯数据驱动的优势是什么？

<details><summary>答案</summary>不需要标注数据。只要写出方程就能训练（在物理已知的情况下）。</details>

**Q10**：PINN 训练循环里，autograd.grad 和 loss.backward 分别用在哪里？

<details><summary>答案</summary>autograd.grad 用于循环内算 du/dx 等中间导数（构造物理 Loss）；loss.backward 在最后算总 Loss 对参数的梯度（用于参数更新）。</details>

#### 🚦 自我评估

- 任务全部通过 + Quiz 8 题对 → **绿灯进入 Week 10**
- 不能解释"为什么用 BC" → **黄灯**——重看 Day 4 的"u(x) + x² 应该是常数"实验
- create_graph 用法不熟 → **黄灯**——重看 Day 1

#### 知识地图自查

- [ ] (5) `create_graph=True` → 🟣
- [ ] (6) MLP 二阶导 → 🟣
- [ ] (7) PDE 残差作为 Loss → 🟣
- [ ] (8) 加 BC → 🟣

---

## 进入 Week 10 之前

下周 Week 10 是 Part B 的收尾：
- 把碎片代码整理成**标准 PINN 框架**
- 学**权重平衡**（PDE Loss 和 BC Loss 的组合权重）
- 第一次解决**真物理问题**：1D 弹性杆

下周需要的前置：
- Week 9 的 PDE Loss + BC Loss 模式熟练
- create_graph 用法熟练
- 不需要新数学

特别提醒：Week 10 会接触**弹性力学三大方程**（几何、本构、平衡）——这是你目标方向的真实物理。**虽然简化但是真物理**。

Week 9 完成。

---

---

## Week 10: 完整 PINN 框架 + 1D 弹性杆

### 🎯 Week 10 总览

**本周覆盖的知识点**：(9) PINN 框架封装、(10) 权重对结果的影响、(11) 动态权重策略、(12) 1D 弹性杆 PINN

**学完之后你应该能**：
- 🟣 **能用**：(9) 用模板写 PINN、(11) 简单的动态权重策略、(12) 求解 1D 弹性杆
- 🟡 **能讲**：(10) 权重对结果的影响、为什么权重平衡是工程难题

**本周的特点**：**Part B 的收尾**。整合前两周的所有内容，第一次接触**真物理问题**——1D 弹性杆——这是 PhyFENet 论文里 §3.1 带孔板问题的简化版。

---

### ✅ 进入 Week 10 之前的前置 checklist

- [ ] Week 9 weekend 的代码我能独立写出来
- [ ] 我能写出任意 ODE 的残差形式
- [ ] 我理解 BC Loss 和 PDE Loss 的区别
- [ ] 我接受这周开始接触弹性力学方程（公式简单，会详细讲）

---

### Day 1-2 | 标准 PINN 训练模板

**🎯 本日目标**：把碎片代码整理成可复用的 PINN 模板；用模板重新解 Week 9 的 ODE。

**🟢 直觉层**（约 5 分钟）：

Week 9 的代码每次都把整个流程写一遍——重复且容易错。今天的任务是**抽出共性，封装成模板**。下次解新方程时只需修改"残差表达式"和"边界条件"，其他代码不用动。

这是工程化思维——和 Part A Week 6 把训练循环封装成 `train_model` 函数一样。

**🟡 概念层**（约 15 分钟）：

PINN 训练的"共性"和"差异"分析：

**共性**（每次都一样的）：
- 网络架构（MLP + Tanh）
- 训练循环结构（采样 + forward + 算梯度 + Loss + backward）
- Optimizer + 学习率（Adam, 1e-3）
- Loss 历史记录

**差异**（每个问题不同）：
- PDE 残差的具体形式（涉及 du/dx、d²u/dx² 等的组合）
- 边界条件（哪个边界、什么类型、什么值）
- 域的范围（[0, 1]、[-1, 1] 还是别的）

**封装策略**：把"共性"做成函数，"差异"作为函数参数（用回调函数 / lambda）传入。

**🟣 代码层**（约 3 小时）：

创建目录 `week10/`，里面两个文件：

**文件 1**：`week10/pinn_template.py`（通用模板）

```python
"""
通用 PINN 训练模板
用法：定义自己的 pde_residual_fn 和 bc_fn，调用 train_pinn
"""
import torch
import torch.nn as nn
import torch.optim as optim


class PINN(nn.Module):
    """用于 PINN 的标准 MLP（必须用 Tanh 保证可微）"""
    def __init__(self, in_dim=1, hid_dim=32, out_dim=1, n_layers=3):
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
      order: 导数阶数（1 或 2）
    返回：
      导数 tensor，shape=(N, 1)
    """
    if order == 1:
        return torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
    elif order == 2:
        dy_dx = torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
        return torch.autograd.grad(
            dy_dx, x,
            grad_outputs=torch.ones_like(dy_dx),
            create_graph=True, retain_graph=True
        )[0]
    else:
        raise ValueError(f"暂只支持 1 阶和 2 阶导，不支持 order={order}")


def train_pinn(
    model, 
    pde_residual_fn,    # 函数：(x, model) → 残差 tensor
    bc_fn,              # 函数：(model) → BC Loss 标量
    n_pde_points,       # 每 epoch 配点数
    pde_domain,         # (x_min, x_max)
    n_epochs,
    w_pde=1.0,
    w_bc=10.0,
    lr=1e-3,
    verbose=True
):
    """通用 PINN 训练
    
    返回：history 字典
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'total': [], 'pde': [], 'bc': []}
    
    x_min, x_max = pde_domain
    
    for epoch in range(n_epochs):
        # === 采样配点 ===
        x_pde = torch.rand(n_pde_points, 1) * (x_max - x_min) + x_min
        x_pde.requires_grad_(True)
        
        # === PDE Loss ===
        residual = pde_residual_fn(x_pde, model)
        L_PDE = (residual ** 2).mean()
        
        # === BC Loss ===
        L_BC = bc_fn(model)
        
        # === 总 Loss + 反向 ===
        loss = w_pde * L_PDE + w_bc * L_BC
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history['total'].append(loss.item())
        history['pde'].append(L_PDE.item())
        history['bc'].append(L_BC.item())
        
        if verbose and epoch % 500 == 0:
            print(f"  epoch {epoch}: total={loss.item():.4e}, "
                  f"L_PDE={L_PDE.item():.4e}, L_BC={L_BC.item():.4e}")
    
    return history
```

**文件 2**：`week10/day12_ode_example.py`（用模板解 Week 9 的 ODE）

```python
"""
用 pinn_template 解 Week 9 的问题：du/dx = -2x，u(0) = 1
"""
from pinn_template import PINN, compute_derivative, train_pinn
import torch
import matplotlib.pyplot as plt

# ========== 定义具体问题 ==========

def pde_residual_fn(x, model):
    """残差：du/dx + 2x"""
    u = model(x)
    du_dx = compute_derivative(u, x, order=1)
    return du_dx + 2 * x

def bc_fn(model):
    """BC: u(0) = 1"""
    x_bc = torch.tensor([[0.0]])
    u_bc_pred = model(x_bc)
    return ((u_bc_pred - 1.0) ** 2).mean()

# ========== 训练 ==========
torch.manual_seed(0)
model = PINN(in_dim=1, hid_dim=32, out_dim=1, n_layers=3)

print("用模板解 ODE...")
history = train_pinn(
    model,
    pde_residual_fn=pde_residual_fn,
    bc_fn=bc_fn,
    n_pde_points=100,
    pde_domain=(0.0, 1.0),
    n_epochs=5000,
    w_pde=1.0,
    w_bc=10.0
)

# ========== 验证 ==========
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
with torch.no_grad():
    u_pred = model(x_test)
u_true = 1 - x_test ** 2

mae = (u_pred - u_true).abs().mean()
print(f"\n平均绝对误差 (MAE): {mae.item():.6f}")

# 画图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['pde'], label='L_PDE')
axes[0].plot(history['bc'], label='L_BC')
axes[0].plot(history['total'], label='Total')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].set_title('训练 Loss')

axes[1].plot(x_test.numpy(), u_true.numpy(), 'b-', label='True')
axes[1].plot(x_test.numpy(), u_pred.numpy(), 'r--', label='PINN')
axes[1].legend()
axes[1].set_title('解对比')

plt.tight_layout()
plt.savefig('week10_day12_pinn.png')
```

**验收标准**：
- 模板代码能正常运行
- MAE < 0.01
- 你能解释 `pde_residual_fn` 和 `bc_fn` 为什么作为参数（差异化部分）

---

#### ⚠️ Day 1-2 新手最容易踩的坑

**坑 1：`compute_derivative` 函数里 retain_graph 不加**
- 模板里我们加了 `retain_graph=True`——为了让外面的代码能多次求导
- 不加的话，调用一次 compute_derivative 后，再求别的导会报错

**坑 2：模板里 `pde_residual_fn` 写成不带 model 参数**
- 错误：`pde_residual_fn(x)`，只有 x
- 正确：`pde_residual_fn(x, model)`，需要 model 来 forward
- 否则函数里没法算 u

**坑 3：把 `bc_fn` 也写成 `(x, model)`**
- BC 通常**不需要采样**——就在固定边界点上算
- 所以参数只有 model
- 灵活性：复杂 BC 也能在 bc_fn 内部多次调用 model

**坑 4：导入路径错误**
- `from pinn_template import ...` 要求两个文件在同一目录
- 否则 Python 找不到 pinn_template
- 解决：用相对路径或加 `sys.path`

---

#### 🧠 Day 1-2 概念问答 quiz

**Q1**：PINN 训练流程的"共性"和"差异"分别有哪些？

<details><summary>答案</summary>共性：网络结构、训练循环结构、optimizer、Loss 记录。差异：PDE 残差表达式、BC、域范围。封装时把共性做模板，差异作为函数参数传入。</details>

**Q2**：为什么 `pde_residual_fn` 既要接收 x 又要接收 model？

<details><summary>答案</summary>x 是采样配点（每 epoch 不同），model 用来 forward 算 u 和导数。两者都需要才能算残差。</details>

**Q3**：`compute_derivative` 函数里 `retain_graph=True` 的目的？

<details><summary>答案</summary>让函数返回的导数仍可以被进一步求导。如果不加，调用 compute_derivative 后计算图被清空，外面没法继续求导（比如算 d²u/dx² 时需要 du/dx 上再求一次）。</details>

**Q4**：用模板解新问题时，需要修改哪些部分？

<details><summary>答案</summary>主要是 `pde_residual_fn`（写新方程残差）和 `bc_fn`（写新边界条件）。其他参数（域、配点数、权重）按需调整。模型架构和训练循环都不用改。</details>

**Q5**：模板里 `train_pinn` 返回 history 字典，里面有哪些内容？为什么记录这么多？

<details><summary>答案</summary>记录 total、pde、bc 三种 Loss。这样可以看每项 Loss 的变化趋势——比如发现 L_pde 下降但 L_bc 不下降，说明权重需要调整。</details>

---

#### 📦 Day 1-2 知识卡片

| 项目 | 内容 |
|---|---|
| **核心成果** | 可复用的 PINN 模板 |
| **架构思想** | 共性做模板，差异作为参数 |
| **关键函数** | `PINN` 类、`compute_derivative`、`train_pinn` |
| **接口设计** | pde_residual_fn(x, model)、bc_fn(model) |
| **本日产出** | `week10/pinn_template.py` + `week10/day12_ode_example.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 3 | 权重实验：为什么权重不好调

**🎯 本日目标**：通过实验观察权重对结果的影响；理解权重平衡是工程难题。

**🟢 直觉层**（约 5 分钟）：

Day 1-2 我们用了 `w_pde=1, w_bc=10`——这是凭经验拍的。今天的实验：试不同的权重组合，看结果差异。

**预期发现**：
- 权重小的那项 Loss 会很大（约束没起作用）
- 权重大的那项 Loss 会小（约束被强化）
- **没有"通用最优权重"**——依赖具体问题

**🟡 概念层**（约 10 分钟）：

权重影响的本质：

```
L_total = w_PDE · L_PDE + w_BC · L_BC
```

- 优化器只看 `L_total` 在下降——不关心两项的相对值
- 如果 `w_PDE = 0.1`，那么 PDE 项贡献小——L_total 主要由 BC 决定——BC 满足很好但 PDE 满足差
- 反过来同理

PhyFENet 论文 §3.2 明确讨论了这个问题——这是 PINN 的核心工程难题。

**🟣 代码层**（约 1.5 小时）：

创建文件 `week10/day03_weight_experiment.py`：

```python
"""
权重实验：测试不同 (w_pde, w_bc) 对结果的影响
"""
from pinn_template import PINN, compute_derivative, train_pinn
import torch
import matplotlib.pyplot as plt

# 用 Day 1-2 的 ODE
def pde_residual_fn(x, model):
    u = model(x)
    du_dx = compute_derivative(u, x, order=1)
    return du_dx + 2 * x

def bc_fn(model):
    x_bc = torch.tensor([[0.0]])
    u_bc_pred = model(x_bc)
    return ((u_bc_pred - 1.0) ** 2).mean()

# 5 组权重对比
weights = [
    (1.0, 0.1),    # 边界弱
    (1.0, 1.0),    # 等权
    (1.0, 10.0),   # 边界强（baseline）
    (1.0, 100.0),  # 边界极强
    (0.1, 1.0),    # PDE 弱（等价于 w_pde=1, w_bc=10）
]

results = []

for i, (w_pde, w_bc) in enumerate(weights):
    print(f"\n实验 {i+1}: w_pde={w_pde}, w_bc={w_bc}")
    torch.manual_seed(0)
    model = PINN()
    history = train_pinn(
        model, pde_residual_fn, bc_fn,
        n_pde_points=100, pde_domain=(0.0, 1.0),
        n_epochs=5000, w_pde=w_pde, w_bc=w_bc, verbose=False
    )
    
    # 评估
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    with torch.no_grad():
        u_pred = model(x_test)
    u_true = 1 - x_test ** 2
    mae = (u_pred - u_true).abs().mean().item()
    u_at_0 = model(torch.tensor([[0.0]])).item()
    bc_error = abs(u_at_0 - 1.0)
    
    print(f"  MAE: {mae:.4f}, u(0): {u_at_0:.4f}, BC 误差: {bc_error:.4f}")
    
    results.append({
        'w_pde': w_pde,
        'w_bc': w_bc,
        'mae': mae,
        'u_at_0': u_at_0,
        'bc_error': bc_error,
        'history': history,
        'u_pred': u_pred.numpy().flatten(),
    })

# ========== 输出表格 ==========
print("\n\n实验汇总:")
print(f"{'(w_pde, w_bc)':<20} | {'MAE':<10} | {'u(0)':<10} | {'BC 误差'}")
print("-" * 60)
for r in results:
    print(f"({r['w_pde']:5.1f}, {r['w_bc']:6.1f})       | {r['mae']:.4f}     | {r['u_at_0']:.4f}     | {r['bc_error']:.4f}")

# ========== 画图对比 ==========
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
u_true = (1 - x_test ** 2).numpy().flatten()

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for ax, r in zip(axes, results):
    ax.plot(x_test.numpy(), u_true, 'b-', label='True')
    ax.plot(x_test.numpy(), r['u_pred'], 'r--', label='PINN')
    ax.set_title(f"w_pde={r['w_pde']}, w_bc={r['w_bc']}\nMAE={r['mae']:.4f}")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('week10_weights.png')
```

**观察任务**（写在脚本末尾的注释里）：

```
权重对结果影响的总结：

1. 边界权重小 (1, 0.1) 时:
   - 现象：u(0) 偏离 1.0；MAE 较大
   - 原因：

2. 边界权重极强 (1, 100) 时:
   - 现象：u(0) 准确，但 PDE 残差不为 0
   - 原因：

3. 最佳权重 (1, 10) 是怎么找到的？
   - 答：

4. 为什么说权重平衡是工程难题？
   - 答：
```

**验收标准**：
- 5 组实验都能跑完
- 表格能填完
- 你能从实验观察出"边界权重影响 u(0) 的精度"

---

#### 🧠 Day 3 概念问答 quiz

**Q1**：边界权重 w_bc 设得太小（如 0.1），结果会怎样？

<details><summary>答案</summary>BC 约束相对弱，u(0) 可能偏离目标值（如 1.0）。整体 MAE 增大。</details>

**Q2**：边界权重 w_bc 设得太大（如 100），结果会怎样？

<details><summary>答案</summary>BC 约束极强，u(0) 非常接近 1。但 PDE 项相对被忽视，方程在域内的满足度可能下降——局部 MAE 增大。</details>

**Q3**：(w_pde=0.1, w_bc=1) 和 (w_pde=1, w_bc=10) 等价吗？

<details><summary>答案</summary>**数学上等价**——同时缩放总 Loss 不影响优化。但**实际优化**可能略有差异（因为优化器对 Loss 量级敏感）。</details>

**Q4**：为什么没有"通用最优权重"？

<details><summary>答案</summary>不同问题的两项 Loss 量级可能差很多——比如 L_PDE 在域内大量配点上累加，L_BC 只在少量边界点上。两者 baseline 量级不同。需要针对具体问题调。</details>

**Q5**：实际工程中怎么找合适权重？

<details><summary>答案</summary>(1) 凭经验初值（如 1:10）；(2) 观察两项 Loss 量级，调整到大致平衡；(3) 网格搜索（试多组取最佳）；(4) 动态权重（Day 4 学）。</details>

---

#### 📦 Day 3 知识卡片

| 项目 | 内容 |
|---|---|
| **核心结论** | 权重影响 PDE 和 BC 的相对满足度 |
| **典型规律** | w_bc 大 → BC 满足好，PDE 可能差；反之 |
| **重要认知** | 没有通用最优权重，依赖问题 |
| **本日产出** | `week10/day03_weight_experiment.py` |
| **掌握要求** | 🟡 能讲（理解工程意义） |

---

### Day 4 | 等比例权重策略（动态调整）

**🎯 本日目标**：实现一种简单的动态权重策略——让 PDE 和 BC Loss 量级自动平衡。

**🟢 直觉层**（约 5 分钟）：

手动调权重既麻烦又依赖经验。最直观的"自动化"想法：

> 如果 L_PDE 当前值是 1e-2，L_BC 是 1e-5，明显 BC 已经满足了——加大 PDE 权重让网络更关注它。

数学化：让 `w_i ∝ 1/L_i`——大的 Loss 用小权重抑制，小的 Loss 用大权重不放过。这就是**等比例权重**。

**🟡 概念层**（约 15 分钟）：

**等比例权重的核心思想**：

每隔 N 个 epoch（比如 100），根据当前两项 Loss 的相对值调整权重：

```python
w_pde = 1 / L_pde
w_bc = 1 / L_bc
# 归一化（避免总量级漂移）
total = w_pde + w_bc
w_pde = 2 * w_pde / total
w_bc = 2 * w_bc / total
```

效果：
- 如果 L_PDE 很大 → w_pde 也变大 → 优化器更关注 PDE
- 反过来同理
- 长期看两项 Loss 会被推向相同量级

**注意**：这只是一种简单策略。PhyFENet 论文用的是更精细的版本。但思想一样。

**为什么要归一化？** 直接 `w_pde = 1/L_pde` 会让 w 变得非常大（比如 L=1e-6 时 w=1e6），可能让训练不稳定。归一化让 w_pde + w_bc = 2（一个固定数），避免数量级失控。

**🟣 代码层**（约 1.5 小时）：

创建文件 `week10/day04_balanced_weights.py`：

```python
"""
等比例权重策略：自动平衡 L_PDE 和 L_BC 的相对重要性
"""
from pinn_template import PINN, compute_derivative
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 用 Week 9 的 ODE
def get_pde_residual(x, model):
    u = model(x)
    du_dx = compute_derivative(u, x, order=1)
    return du_dx + 2 * x

def get_bc_loss(model):
    x_bc = torch.tensor([[0.0]])
    u_bc_pred = model(x_bc)
    return ((u_bc_pred - 1.0) ** 2).mean()

# ========== 训练（带动态权重）==========
torch.manual_seed(0)
model = PINN(1, 32, 1, 3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 动态权重初始化
w_pde, w_bc = 1.0, 1.0

history = {'L_pde': [], 'L_bc': [], 'w_pde': [], 'w_bc': []}

print("等比例权重训练...")
for epoch in range(5000):
    # 算两项 Loss
    x_pde = torch.rand(100, 1, requires_grad=True)
    residual = get_pde_residual(x_pde, model)
    L_pde = (residual ** 2).mean()
    L_bc = get_bc_loss(model)
    
    # 每 100 epoch 更新权重（不在第 0 epoch 更新）
    if epoch % 100 == 0 and epoch > 0:
        with torch.no_grad():
            w_pde = 1.0 / (L_pde.item() + 1e-8)
            w_bc = 1.0 / (L_bc.item() + 1e-8)
            # 归一化
            total = w_pde + w_bc
            w_pde = 2 * w_pde / total
            w_bc = 2 * w_bc / total
    
    # 总 Loss
    loss = w_pde * L_pde + w_bc * L_bc
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    history['L_pde'].append(L_pde.item())
    history['L_bc'].append(L_bc.item())
    history['w_pde'].append(w_pde)
    history['w_bc'].append(w_bc)
    
    if epoch % 500 == 0:
        print(f"  epoch {epoch}: L_pde={L_pde.item():.4e}, L_bc={L_bc.item():.4e}, "
              f"w_pde={w_pde:.2f}, w_bc={w_bc:.2f}")

# ========== 评估 ==========
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
with torch.no_grad():
    u_pred = model(x_test)
u_true = 1 - x_test ** 2
mae = (u_pred - u_true).abs().mean()
print(f"\n最终 MAE: {mae.item():.6f}")

# ========== 画图 ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(history['L_pde'], label='L_PDE')
axes[0, 0].plot(history['L_bc'], label='L_BC')
axes[0, 0].set_yscale('log')
axes[0, 0].set_title('Loss 曲线')
axes[0, 0].legend()

axes[0, 1].plot(history['w_pde'], label='w_pde')
axes[0, 1].plot(history['w_bc'], label='w_bc')
axes[0, 1].set_title('权重变化')
axes[0, 1].legend()

axes[1, 0].plot(x_test.numpy(), u_true.numpy(), 'b-', label='True')
axes[1, 0].plot(x_test.numpy(), u_pred.numpy(), 'r--', label='PINN')
axes[1, 0].set_title('解')
axes[1, 0].legend()

axes[1, 1].plot(x_test.numpy(), (u_pred - u_true).numpy())
axes[1, 1].set_title('误差 u_pred - u_true')

plt.tight_layout()
plt.savefig('week10_dynamic_weights.png')
```

**观察任务**：
- 训练初期，L_PDE 和 L_BC 哪个更大？为什么？
- 权重 w_pde 和 w_bc 在训练过程中的变化趋势？
- 最终 MAE 和固定权重 (1, 10) 比，哪个更好？

**验收标准**：
- 代码跑通
- 能看到权重随训练变化
- 最终 MAE 不比固定权重 (1, 10) 差太多

---

#### 🧠 Day 4 概念问答 quiz

**Q1**：等比例权重的核心思想是？

<details><summary>答案</summary>`w_i ∝ 1/L_i`——Loss 大的项用小权重压制，Loss 小的用大权重不放过。让两项 Loss 长期向相同量级靠拢。</details>

**Q2**：为什么要归一化 `w_pde + w_bc = 2`？

<details><summary>答案</summary>避免权重数量级失控。如果 L=1e-6，直接用 1/L 会让 w=1e6，导致训练不稳定。归一化让总权重稳定，单纯调整两者的相对比例。</details>

**Q3**：等比例权重一定比固定权重好吗？

<details><summary>答案</summary>不一定。它只是个简单策略——对一些问题好用，对一些不行。PhyFENet 论文用的是更精细的版本。工程实践经常需要尝试多种策略。</details>

**Q4**：为什么不每 epoch 都更新权重？

<details><summary>答案</summary>(1) Loss 在单个 epoch 可能波动大，频繁更新让权重抖动；(2) 每 N 步更新一次让权重平稳变化，更稳定。</details>

**Q5**：动态权重策略对什么样的 PINN 问题特别有用？

<details><summary>答案</summary>多项 Loss 量级差异大的问题。比如 PINN 涉及多个 PDE + 多个 BC + 数据 Loss 时，手调几乎不可能，需要动态平衡。</details>

---

#### 📦 Day 4 知识卡片

| 项目 | 内容 |
|---|---|
| **核心策略** | w_i ∝ 1/L_i + 归一化 |
| **更新频率** | 每 N（如 100）epoch 更新一次 |
| **优缺点** | 优：自动平衡；缺：策略简单，不一定最优 |
| **本日产出** | `week10/day04_balanced_weights.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 5-6（周末）| 1D 弹性杆 PINN（物理意义升级）

**🎯 本日目标**：用 PINN 求解一个真正的物理问题——1D 弹性杆受拉伸；体验完整的"物理建模 → PINN 实现"流程。

**🟢 直觉层**（约 10 分钟）：

到这周末，你会**第一次用 PINN 求解一个真物理问题**。这是论文 §3.1 带孔板问题的最简化版——把二维问题降到一维。

**问题描述**（直观）：

```
一根长度 L=1 的弹性杆
  - 左端固定（钉死）
  - 右端施加轴向拉力 F=1
  - 弹性模量 E=1（用无量纲化的单位）
  - 截面积 A=1

杆被拉长后：
  - 各点位移 u(x) 是什么？
  - 各点应力 σ(x) 是什么？
```

直觉答案：均匀拉伸时位移是线性分布 `u(x) = F·x/(E·A) = x`，应力处处相等 `σ = F/A = 1`。

PINN 任务：让网络从"无任何数据"开始，通过物理方程学到这个解。

**🟡 概念层**（约 30 分钟）：

**弹性力学三大方程**（1D 简化版）：

**1. 几何方程**：应变 = 位移对坐标的导数

```
ε(x) = du/dx
```

物理含义：杆某点的应变 = 这点位移变化率。如果整段都拉伸 1%，每点 du/dx = 0.01。

**2. 本构方程**（胡克定律）：应力 = 弹性模量 × 应变

```
σ(x) = E · ε(x) = E · du/dx
```

物理含义：材料的力学性质。E 越大代表越难变形。

**3. 平衡方程**：内力处处平衡（无外力作用时）

```
dσ/dx = 0
```

物理含义：如果某点应力变化（dσ/dx ≠ 0），说明这点有"净力"——但杆是平衡状态，所以处处 dσ/dx = 0。

**等价代换**：用 σ = E·du/dx 代入：

```
d/dx(E · du/dx) = 0
```

如果 E 是常数（杆均匀），简化为：

```
E · d²u/dx² = 0  →  d²u/dx² = 0
```

**这个方程的解**：u(x) 必须是线性函数（二阶导为 0 的函数）。形式 u(x) = a·x + b。

**边界条件**：

- BC1: `u(0) = 0`（左端固定）
- BC2: `σ(L) = F/A`（右端施加 F 力，应力 = F/A）

代入解 u(x) = a·x + b：
- BC1: u(0) = b = 0  →  b = 0
- BC2: σ(L) = E·u'(L) = E·a = F/A  →  a = F/(E·A) = 1（本例）

所以 **u(x) = x**，**ε = du/dx = 1**，**σ = E·ε = 1**。

**PINN 实现思路**：

```
PDE Loss：在内部配点上 d²u/dx² = 0
BC1 Loss：u(0) = 0
BC2 Loss：σ(L) = F/A，即 E · du/dx(at x=L) = F/A
```

注意 **BC2 是 Neumann 类型**（约束导数值），不是 Dirichlet（约束函数值）。这比 Day 1-2 复杂——需要在边界点也算 du/dx。

**🟣 代码层**（约 3 小时）：

创建文件 `week10/weekend_elastic_bar.py`：

```python
"""
1D 弹性杆 PINN
方程：d²u/dx² = 0（平衡方程，常数 E）
BC1: u(0) = 0（左固定）
BC2: σ(L) = E·du/dx(L) = F/A（右端力）

解析解：u(x) = (F/E·A)·x = x（参数全 1）
"""
from pinn_template import PINN, compute_derivative
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# ========== 物理参数 ==========
E = 1.0    # 弹性模量
F = 1.0    # 末端力
A = 1.0    # 截面积
L = 1.0    # 杆长

torch.manual_seed(0)
model = PINN(in_dim=1, hid_dim=32, out_dim=1, n_layers=3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def get_losses():
    """算 PDE + 两个 BC 的 Loss"""
    # PDE Loss：内部配点上 d²u/dx² = 0
    x = torch.rand(100, 1, requires_grad=True) * L
    u = model(x)
    d2u_dx2 = compute_derivative(u, x, order=2)
    L_pde = (d2u_dx2 ** 2).mean()
    
    # BC1 Loss：u(0) = 0
    x_left = torch.tensor([[0.0]])
    u_left = model(x_left)
    L_bc_u = (u_left ** 2).mean()
    
    # BC2 Loss：σ(L) = E·du/dx(L) = F/A
    x_right = torch.tensor([[L]], requires_grad=True)
    u_right = model(x_right)
    du_dx_right = compute_derivative(u_right, x_right, order=1)
    sigma_right = E * du_dx_right
    L_bc_sigma = ((sigma_right - F / A) ** 2).mean()
    
    return L_pde, L_bc_u, L_bc_sigma

print("1D 弹性杆 PINN 训练...")
history = {'pde': [], 'bc_u': [], 'bc_sigma': []}

for epoch in range(10000):
    L_pde, L_bc_u, L_bc_sigma = get_losses()
    # 手动权重 1:10:10
    loss = L_pde + 10 * L_bc_u + 10 * L_bc_sigma
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    history['pde'].append(L_pde.item())
    history['bc_u'].append(L_bc_u.item())
    history['bc_sigma'].append(L_bc_sigma.item())
    
    if epoch % 1000 == 0:
        print(f"  epoch {epoch}: L_pde={L_pde.item():.4e}, "
              f"L_bc_u={L_bc_u.item():.4e}, L_bc_sigma={L_bc_sigma.item():.4e}")

# ========== 验证：u, ε, σ 三条曲线 ==========
x_test = torch.linspace(0, L, 100).reshape(-1, 1)
x_test.requires_grad_(True)
u_pred = model(x_test)
eps_pred = compute_derivative(u_pred, x_test, order=1)
sigma_pred = E * eps_pred

# 解析解
x_np = x_test.detach().numpy().flatten()
u_pred_np = u_pred.detach().numpy().flatten()
eps_pred_np = eps_pred.detach().numpy().flatten()
sigma_pred_np = sigma_pred.detach().numpy().flatten()

u_true = x_np                   # u = x
eps_true = [1.0] * 100           # ε = 1 处处
sigma_true = [1.0] * 100         # σ = 1 处处

# 误差
u_mae = abs(u_pred_np - u_true).mean()
print(f"\nu MAE: {u_mae:.6f}")

# ========== 画图 ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Loss 曲线
axes[0, 0].plot(history['pde'], label='L_pde')
axes[0, 0].plot(history['bc_u'], label='L_bc_u')
axes[0, 0].plot(history['bc_sigma'], label='L_bc_sigma')
axes[0, 0].set_yscale('log')
axes[0, 0].set_title('训练 Loss')
axes[0, 0].legend()

axes[0, 1].plot(x_np, u_true, 'b-', label='True')
axes[0, 1].plot(x_np, u_pred_np, 'r--', label='PINN')
axes[0, 1].set_title('u(x) - 位移')
axes[0, 1].legend()

axes[1, 0].plot(x_np, eps_true, 'b-', label='True ε=1')
axes[1, 0].plot(x_np, eps_pred_np, 'r--', label='PINN')
axes[1, 0].set_title('ε(x) = du/dx - 应变')
axes[1, 0].legend()

axes[1, 1].plot(x_np, sigma_true, 'b-', label='True σ=1')
axes[1, 1].plot(x_np, sigma_pred_np, 'r--', label='PINN')
axes[1, 1].set_title('σ(x) = E·ε - 应力')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('week10_elastic_bar.png')
```

**验收标准**：
- u MAE < 0.05
- u_pred 接近线性 u(x) = x
- ε 接近常数 1.0（允许小波动）
- σ 接近常数 1.0
- 四张图基本重合（除波动）

**🔬 关键体会**（约 15 分钟）：

到这里你完成了：

- **第一次用 PINN 求解真实物理问题**（一维弹性杆）
- **完全不用 FEM 数据**——所有 Loss 都来自方程和边界条件
- **学到了三大方程**（几何/本构/平衡）的实际应用

回顾论文 §3.1（PhyFENet 用 PINN 求解带孔板问题）——本质上和今天做的完全一样，只是：
- 一维变二维
- 解析解换成 FEM 数据
- 网络从 MLP 升级成 GNN（Part C 学）

**到此为止，你已经具备了 PINN 的核心能力**。Part C 接下来学 GNN，把它替换 MLP 后再加上数据，就是 PhyFENet 框架的雏形。

---

### ✅ Week 10 完成评估

#### 任务级 checklist

- [ ] `week10/pinn_template.py` 模板完成
- [ ] `week10/day12_ode_example.py` 用模板解 ODE，MAE < 0.01
- [ ] `week10/day03_weight_experiment.py` 5 组权重实验完成
- [ ] `week10/day04_balanced_weights.py` 动态权重跑通
- [ ] `week10/weekend_elastic_bar.py` 1D 弹性杆 u MAE < 0.05

#### 概念级 quiz（10 题，至少 8 题对）

**Q1**：标准 PINN 训练流程的"共性"和"差异"分别是什么？

<details><summary>答案</summary>共性：网络结构、训练循环、optimizer。差异：PDE 残差、BC、域。封装时把差异作为函数参数传入。</details>

**Q2**：`compute_derivative(y, x, order=2)` 内部需要调用几次 `autograd.grad`？

<details><summary>答案</summary>2 次。第一次算 dy/dx（必须 create_graph=True），第二次对 dy/dx 再求一次得 d²y/dx²。</details>

**Q3**：手调 PINN 权重的两个常见现象？

<details><summary>答案</summary>BC 权重小：u 在边界点偏离目标。BC 权重大：边界精确但 PDE 残差大（域内方程不满足）。</details>

**Q4**：等比例权重策略 `w_i ∝ 1/L_i` 的作用？

<details><summary>答案</summary>自动平衡 L_PDE 和 L_BC 的相对量级——大的项压制，小的项不放过。避免手动调权重的麻烦。</details>

**Q5**：弹性力学三大方程是哪三个？

<details><summary>答案</summary>(1) 几何方程：ε = du/dx（应变 = 位移导数）；(2) 本构方程：σ = E·ε（胡克定律）；(3) 平衡方程：dσ/dx = 0（无外力时内力平衡）。</details>

**Q6**：1D 弹性杆 d²u/dx² = 0 的解是什么形式？

<details><summary>答案</summary>线性函数 u(x) = a·x + b。二阶导为 0 意味着一阶导是常数，u 是线性。</details>

**Q7**：BC2 `σ(L) = F/A` 是 Dirichlet 还是 Neumann 条件？

<details><summary>答案</summary>Neumann（约束导数值）。因为 σ = E·du/dx 涉及一阶导。Dirichlet 是直接约束 u 的值。</details>

**Q8**：用 PINN 解 1D 弹性杆相对 FEM 的优势和劣势？

<details><summary>答案</summary>优势：不需要 FEM 数据；推理快（一次 forward）。劣势：精度可能不够；权重难调；复杂几何 PINN 可能收敛差。</details>

**Q9**：`σ(L) = F/A` 的 Loss 怎么写？

<details><summary>答案</summary>`L_bc_sigma = ((E * du_dx(at x=L) - F/A) ** 2).mean()`。在 x=L 处算 du/dx（用 autograd.grad），再乘 E 得 σ，与 F/A 求差平方。</details>

**Q10**：1D 弹性杆 PINN 和 PhyFENet 论文的带孔板问题有什么共性？

<details><summary>答案</summary>都用 PINN + 三大方程（几何/本构/平衡）。区别：PhyFENet 是 2D（涉及多个偏导）；用 GNN 而不是 MLP；融合了 FEM 数据。但**核心思想完全一样**。</details>

#### 🚦 自我评估

- 任务全部通过 + 弹性杆求解 + Quiz 8 题对 → **绿灯进入 Part C**
- 能跑通模板但弹性杆 MAE > 0.1 → **黄灯**——重看 Day 5-6 的边界条件实现
- 三大方程不熟 → **黄灯**——重读 Day 5-6 概念层

#### 知识地图自查

- [ ] (9) PINN 模板 → 🟣
- [ ] (10) 权重影响 → 🟡
- [ ] (11) 动态权重 → 🟣
- [ ] (12) 1D 弹性杆 → 🟣

---

## ✅ Part B 总完成标准

进入 Part C 前必须达到。

### 理论掌握

- [ ] 能说清楚 `.backward()` 和 `torch.autograd.grad()` 的区别和适用场景
- [ ] 能解释 `create_graph=True` 什么时候需要、为什么
- [ ] 能写出任意 ODE 的残差形式
- [ ] 能说出弹性力学三大方程（几何/本构/平衡）和它们的物理含义
- [ ] 能解释为什么仅 PDE Loss 不够、需要 BC

### 代码能力（限时完成，不看参考）

- [ ] **10 分钟内**：算 MLP 输出对输入的一阶和二阶导数
- [ ] **20 分钟内**：写出给定 ODE 的物理约束 Loss（PDE 残差 + BC）
- [ ] **30 分钟内**：用 PINN 模板独立解一个新的 1D ODE（不能直接复制 Week 10 的代码）
- [ ] **45 分钟内**：实现 1D 弹性杆 PINN 求解（参考解析解验证）

### 知识地图最终自查

完整 Part B 12 个知识点：

- [ ] (1) backward vs grad → 🟣
- [ ] (2) 一阶导数 → 🟣
- [ ] (3) 对 MLP 求导 → 🟣
- [ ] (4) 偏导数 → 🟣
- [ ] (5) `create_graph=True` → 🟣
- [ ] (6) MLP 二阶导 → 🟣
- [ ] (7) PDE 残差 Loss → 🟣
- [ ] (8) PDE+BC → 🟣
- [ ] (9) PINN 模板 → 🟣
- [ ] (10) 权重影响 → 🟡
- [ ] (11) 动态权重 → 🟣
- [ ] (12) 1D 弹性杆 → 🟣

至少 10 个 🟣 才算 Part B 完成。

**如果上述任何一项未达到**：回头补强。**Part C 的 GNN 学习需要 Part B 的 PINN 思维**——很多 GNN 用法是"把 PINN 中的 MLP 替换成 GNN"，没有 Part B 的基础 GNN 那部分会脱节。

---

Part B 完成。下一段进入 Part C：**图神经网络（Week 11-16）**。

Part C 的预告：
- Week 11：图数据结构 + PyG 入门
- Week 12-14：GCN、GraphSAGE、消息传递机制
- Week 15：encoder-processor-decoder 架构
- Week 16：综合实战（mesh-based 神经模拟器的最小实现）

Part C 和 Part B 的关系：**把 PINN 中的 MLP 升级成 GNN**——PhyFENet 框架的核心架构。