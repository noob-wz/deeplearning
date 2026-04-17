# Part D: PINN 与有限元数据入门（Week 17–22）

## 本阶段定位

**衔接前三部分**：
- **Part A**（Week 1-7）：MLP + 梯度下降 + PyTorch 标准训练流程
- **Part B**（Week 8-10）：`torch.autograd.grad` + 物理约束 Loss + **1D 弹性杆 PINN**（已完成）
- **Part C**（Week 11-16）：图数据结构 + GCN/GraphSAGE/带边更新的消息传递 + 编码器-处理器-解码器 + 单元聚合（已完成）

**Part C 遗留的局限**：
Week 16 的训练用的是**人工合成标签**——`ux = F*x/(E*A)` 这种一维拉伸近似公式。这里有两个不真实的地方：
1. 标签本身不严格满足"几何方程 ε=du/dx + 本构方程 σ=E·ε"的物理自洽性
2. 网格是规则矩形的，真实 FEM 是不规则三角形/四边形混合

**Part D 要补的四件事**：
1. **2D PINN**：Part B 只做了 1D 弹性杆，2D 问题会让你真正理解偏微分方程在物理问题中的样子
2. **不规则网格 → 图**：真实 FEM 网格不是规则矩形，要能处理三角形和四边形混合单元
3. **模拟 FEM 文本数据的解析**：理解 FEM 求解器（Abaqus/LS-DYNA）输出的数据结构；真实软件的使用在第二阶段做，本阶段用文本模拟
4. **GNN + 物理约束的融合（本阶段核心）**：把 Part B 的"物理约束 Loss"嵌入到 Part C 的"GNN 架构"里——这就是 PhyFENet 的核心思想

**完成标准（进入第二阶段前）**：
- 能独立实现 2D 弹性问题的 PINN 求解
- 能把任意单元类型（三角形/四边形）的网格转成 PyG 图
- 能解析 FEM 文本数据（节点/单元/材料/边界/结果）
- 能在 GNN 的 Loss 里嵌入几何方程和本构方程约束（论文 §2.3 的完整形态）
- 能实现子网络预训练 + 主网络联合训练（论文 §2.3.5）
- 在合成数据集上跑通"mesh → graph → PhyFENet with physics loss → predict"完整流程

**时间预期**：**6-9 周**，基线 6 周，实际很可能需要 7-9 周。Week 20（GNN + 物理 Loss 融合）和 Week 21（多级网络）是 Part D 的密集区——也是整个第一阶段最难的两周，因为你第一次要同时处理 GNN 的结构复杂度和 PINN 的求导复杂度。如果感觉吃力，每周可延长到 1.5-2 周。

**本阶段是第一阶段的收尾**：Week 22 结束时，你应该能在合成数据上跑通一个完整的 PhyFENet 风格系统。第二阶段（Week 23+）才引入真实 FEM 数据（FEniCS/HyperMesh/LS-DYNA）、不规则工业网格、更复杂的工程问题。

---

---

## Week 17: 从 1D PINN 到 2D PINN

**衔接**：Part B Week 10 你做了 1D 弹性杆——用 `du/dx` 算应变、`E·ε` 算应力、`dσ/dx=0` 作为平衡方程 Loss。但实际工程问题（比如论文第三章的带孔板）是 2D 的。本周把 PINN 从 1D 扩展到 2D。

**本周目标**：
- 能写 2D PINN 求解拉普拉斯方程（最简单的 2D PDE）
- 能写 2D PINN 求解 2D 弹性力学（为 Part D 后续打基础）
- 重新审视权重平衡问题（Part B Week 10 已经讲过，这周升级实验）

**本周不做**：GNN 相关内容。本周纯粹是 PINN 的 2D 扩展，让你熟悉多变量偏导数的工程写法。

---

### Day 1 | 2D 偏导数的复习与工程写法

**衔接 Part B Week 8 Day 4**：你已经会对 2D 输入求 `∂u/∂x` 和 `∂u/∂y`。本节复习并建立工程性的封装。

**实践任务**（约 1.5 小时）：创建文件 `week17/day01_2d_partial.py`

**Step 1**：工具函数封装（扩展 `utils/pinn_utils.py`）

```python
import torch

def compute_gradients_2d(u, xy, create_graph=True):
    """
    计算 u 对 2D 输入 (x, y) 的一阶偏导数
    
    参数：
      u: shape=(N, 1)，标量场的预测值
      xy: shape=(N, 2)，输入坐标，requires_grad=True
      create_graph: 若后续要算二阶导，设 True
    返回：
      grad: shape=(N, 2)，每行是 (∂u/∂x, ∂u/∂y)
    """
    grad = torch.autograd.grad(
        u, xy, grad_outputs=torch.ones_like(u),
        create_graph=create_graph, retain_graph=True
    )[0]
    return grad    # grad[:, 0] 是 du/dx, grad[:, 1] 是 du/dy


def compute_laplacian_2d(u, xy):
    """
    计算 u 对 2D 输入的拉普拉斯：∂²u/∂x² + ∂²u/∂y²
    
    返回：shape=(N, 1)
    """
    grad = compute_gradients_2d(u, xy, create_graph=True)
    # grad shape=(N, 2)
    
    # 对 du/dx 再求导得 ∂²u/∂x² 和 ∂²u/∂x∂y
    du_dx = grad[:, 0:1]    # 保持 shape=(N,1) 便于 grad
    ddu_dxx = torch.autograd.grad(
        du_dx, xy, grad_outputs=torch.ones_like(du_dx),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]
    
    # 对 du/dy 再求导得 ∂²u/∂y²
    du_dy = grad[:, 1:2]
    ddu_dyy = torch.autograd.grad(
        du_dy, xy, grad_outputs=torch.ones_like(du_dy),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]
    
    return ddu_dxx + ddu_dyy
```

**Step 2**：验证工具函数
```python
# 验证 1：f = x² + 3y²
# ∂f/∂x = 2x, ∂f/∂y = 6y, Δf = 2 + 6 = 8
xy = torch.tensor([[2.0, 1.0]], requires_grad=True)
f = xy[:, 0:1]**2 + 3 * xy[:, 1:2]**2
grad = compute_gradients_2d(f, xy, create_graph=True)
print(f"∂f/∂x: {grad[0, 0].item()}, ∂f/∂y: {grad[0, 1].item()}")  # 4.0, 6.0

lap = compute_laplacian_2d(f, xy)
print(f"Δf: {lap.item()}")    # 8.0
```

**验收标准**：`∂f/∂x = 4.0`、`∂f/∂y = 6.0`、`Δf = 8.0` 精确成立。

---

### Day 2–3 | 2D PINN 求解拉普拉斯方程

**物理问题**（最简单的 2D PDE）：
```
Δu = ∂²u/∂x² + ∂²u/∂y² = 0     在 Ω = [0,1] × [0,1] 内部
u(x, 0) = 0                       下边界
u(x, 1) = sin(πx)                 上边界
u(0, y) = 0                       左边界
u(1, y) = 0                       右边界
```

这是一个经典的"热传导稳态"或"静电势"问题，有解析解：
```
u(x, y) = sin(πx) · sinh(πy) / sinh(π)
```

**实践任务**（每天约 2 小时）：创建文件 `week17/day23_laplace_pinn.py`

**Step 1**：定义模型和残差函数
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.pinn_utils import compute_laplacian_2d

class PINN2D(nn.Module):
    def __init__(self, hid=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hid), nn.Tanh()]   # 输入 2D, Tanh 激活
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hid, hid), nn.Tanh()]
        layers.append(nn.Linear(hid, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, xy):
        return self.net(xy)

def pde_loss(model, n_points=1000):
    """在 [0,1]×[0,1] 内随机采样配点，计算 Δu 的残差"""
    xy = torch.rand(n_points, 2, requires_grad=True)
    u = model(xy)
    lap = compute_laplacian_2d(u, xy)
    return (lap ** 2).mean()

def bc_loss(model, n_points=100):
    """四条边的边界条件"""
    # 下边界 y=0: u=0
    x_bot = torch.rand(n_points, 1)
    y_bot = torch.zeros(n_points, 1)
    xy_bot = torch.cat([x_bot, y_bot], dim=1)
    loss_bot = (model(xy_bot) ** 2).mean()
    
    # 上边界 y=1: u = sin(πx)
    x_top = torch.rand(n_points, 1)
    y_top = torch.ones(n_points, 1)
    xy_top = torch.cat([x_top, y_top], dim=1)
    u_true_top = torch.sin(torch.pi * x_top)
    loss_top = ((model(xy_top) - u_true_top) ** 2).mean()
    
    # 左边界 x=0: u=0
    y_left = torch.rand(n_points, 1)
    x_left = torch.zeros(n_points, 1)
    xy_left = torch.cat([x_left, y_left], dim=1)
    loss_left = (model(xy_left) ** 2).mean()
    
    # 右边界 x=1: u=0
    y_right = torch.rand(n_points, 1)
    x_right = torch.ones(n_points, 1)
    xy_right = torch.cat([x_right, y_right], dim=1)
    loss_right = (model(xy_right) ** 2).mean()
    
    return loss_bot + loss_top + loss_left + loss_right
```

**Step 2**：训练循环
```python
torch.manual_seed(0)
model = PINN2D(hid=64, n_layers=4)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

w_pde = 1.0
w_bc = 10.0

history = {'pde': [], 'bc': [], 'total': []}

for epoch in range(20000):
    L_pde = pde_loss(model, n_points=1000)
    L_bc = bc_loss(model, n_points=100)
    loss = w_pde * L_pde + w_bc * L_bc
    
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    history['pde'].append(L_pde.item())
    history['bc'].append(L_bc.item())
    history['total'].append(loss.item())
    
    if epoch % 1000 == 0:
        print(f"epoch {epoch}: L_pde={L_pde.item():.4e}, L_bc={L_bc.item():.4e}")
```

**Step 3**：可视化对比（PINN 解 vs 解析解）
```python
# 生成网格评估 PINN
import numpy as np
nx, ny = 50, 50
x = torch.linspace(0, 1, nx)
y = torch.linspace(0, 1, ny)
X, Y = torch.meshgrid(x, y, indexing='ij')
xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

with torch.no_grad():
    u_pred = model(xy).reshape(nx, ny).numpy()

# 解析解
X_np, Y_np = X.numpy(), Y.numpy()
u_true = np.sin(np.pi * X_np) * np.sinh(np.pi * Y_np) / np.sinh(np.pi)

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, data, title in zip(
    axes, [u_true, u_pred, np.abs(u_true - u_pred)],
    ['Analytical', 'PINN', '|Error|']
):
    im = ax.imshow(data.T, origin='lower', extent=[0,1,0,1], cmap='jet')
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

plt.tight_layout(); plt.savefig('week17_laplace.png', dpi=100)
```

**验收标准**：
- 训练完最终 L_pde < 1e-3，L_bc < 1e-3
- 平均绝对误差 `|u_true - u_pred|.mean() < 0.02`
- 解析解和 PINN 解的热力图看起来**明显相似**（目视即可）

**关键体会**：2D PINN 比 1D 复杂的地方仅仅是"求导的变量多了一个"。代码结构、训练流程、权重平衡思路都**完全一样**。这也是为什么工程里 PINN 扩展到 3D 问题相对容易——概念上没有新东西。

---

### Day 4 | 2D 弹性力学 PINN（本周重点）

**衔接你的方向**：论文第三章 §3.1.3 就是 2D 带孔板的弹性问题。今天做一个**去掉孔的简化版**——矩形板单向拉伸——作为你第一次接触 2D 弹性力学 PINN。

**物理问题**：
```
矩形板 Ω = [0, L] × [0, H]，L=1, H=0.5
左端固定：u(0, y) = 0, v(0, y) = 0
右端拉伸：x=L 处施加 x 方向拉力 F=1
上下边界无力：σ_yy(x, 0) = σ_xy(x, 0) = 0, σ_yy(x, H) = σ_xy(x, H) = 0

平面应力假设下：
几何方程：
  ε_xx = ∂u/∂x
  ε_yy = ∂v/∂y
  γ_xy = ∂u/∂y + ∂v/∂x

本构方程（胡克定律，平面应力）：
  σ_xx = E/(1-ν²) · (ε_xx + ν·ε_yy)
  σ_yy = E/(1-ν²) · (ε_yy + ν·ε_xx)
  σ_xy = E/(2(1+ν)) · γ_xy

平衡方程（无体力）：
  ∂σ_xx/∂x + ∂σ_xy/∂y = 0
  ∂σ_xy/∂x + ∂σ_yy/∂y = 0
```

材料常数取简单值：E=1.0, ν=0.3

**注意**：这个问题**没有简单的解析解**（除非做单向拉伸的一维近似），所以我们的验证方式是：检查 PINN 的解在物理上是否合理（位移趋势对不对、平衡是否满足、边界条件是否满足）。这比严格求误差难一些，但**这才是工程中 PINN 真正要做的**——很多时候就没有解析解。

**实践任务**（约 3 小时）：创建文件 `week17/day04_2d_elasticity.py`

**Step 1**：定义模型（输出 2 维：ux 和 uy）
```python
class ElasticityPINN(nn.Module):
    """2D 弹性问题的 PINN，输入 (x,y)，输出 (ux, uy)"""
    def __init__(self, hid=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hid), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hid, hid), nn.Tanh()]
        layers.append(nn.Linear(hid, 2))     # 输出 2 维
        self.net = nn.Sequential(*layers)
    
    def forward(self, xy):
        return self.net(xy)      # shape=(N, 2)，第一列 ux，第二列 uy
```

**Step 2**：计算应变（工具函数）
```python
def compute_strain(u_vec, xy):
    """
    输入：
      u_vec: shape=(N, 2)，模型输出 (ux, uy)
      xy: shape=(N, 2)，坐标，requires_grad=True
    返回：
      strain: shape=(N, 3)，每行 (εxx, εyy, γxy)
    """
    ux = u_vec[:, 0:1]
    uy = u_vec[:, 1:2]
    
    # ∂ux/∂x, ∂ux/∂y
    grad_ux = torch.autograd.grad(
        ux, xy, grad_outputs=torch.ones_like(ux),
        create_graph=True, retain_graph=True
    )[0]
    # grad_uy
    grad_uy = torch.autograd.grad(
        uy, xy, grad_outputs=torch.ones_like(uy),
        create_graph=True, retain_graph=True
    )[0]
    
    eps_xx = grad_ux[:, 0:1]                            # ∂ux/∂x
    eps_yy = grad_uy[:, 1:2]                            # ∂uy/∂y
    gamma_xy = grad_ux[:, 1:2] + grad_uy[:, 0:1]        # ∂ux/∂y + ∂uy/∂x
    
    return torch.cat([eps_xx, eps_yy, gamma_xy], dim=1)
```

**Step 3**：应变 → 应力（胡克定律，平面应力）
```python
def strain_to_stress_plane_stress(strain, E=1.0, nu=0.3):
    """平面应力下的胡克定律
    strain: shape=(N, 3)  (εxx, εyy, γxy)
    返回 stress: shape=(N, 3)  (σxx, σyy, σxy)
    """
    coef = E / (1 - nu**2)
    eps_xx = strain[:, 0:1]
    eps_yy = strain[:, 1:2]
    gamma_xy = strain[:, 2:3]
    
    sigma_xx = coef * (eps_xx + nu * eps_yy)
    sigma_yy = coef * (eps_yy + nu * eps_xx)
    sigma_xy = E / (2 * (1 + nu)) * gamma_xy
    
    return torch.cat([sigma_xx, sigma_yy, sigma_xy], dim=1)
```

**Step 4**：平衡方程残差
```python
def equilibrium_residual(stress, xy):
    """
    平衡方程（无体力）：
      ∂σxx/∂x + ∂σxy/∂y = 0
      ∂σxy/∂x + ∂σyy/∂y = 0
    
    返回两个残差的拼接，shape=(N, 2)
    """
    sigma_xx = stress[:, 0:1]
    sigma_yy = stress[:, 1:2]
    sigma_xy = stress[:, 2:3]
    
    # 对 sigma_xx 求 ∂/∂x
    grad_sxx = torch.autograd.grad(
        sigma_xx, xy, grad_outputs=torch.ones_like(sigma_xx),
        create_graph=True, retain_graph=True
    )[0]
    dsxx_dx = grad_sxx[:, 0:1]
    
    # 对 sigma_yy 求 ∂/∂y
    grad_syy = torch.autograd.grad(
        sigma_yy, xy, grad_outputs=torch.ones_like(sigma_yy),
        create_graph=True, retain_graph=True
    )[0]
    dsyy_dy = grad_syy[:, 1:2]
    
    # 对 sigma_xy 求 ∂/∂x 和 ∂/∂y
    grad_sxy = torch.autograd.grad(
        sigma_xy, xy, grad_outputs=torch.ones_like(sigma_xy),
        create_graph=True, retain_graph=True
    )[0]
    dsxy_dx = grad_sxy[:, 0:1]
    dsxy_dy = grad_sxy[:, 1:2]
    
    eq1 = dsxx_dx + dsxy_dy    # 第一个平衡方程残差
    eq2 = dsxy_dx + dsyy_dy    # 第二个
    
    return torch.cat([eq1, eq2], dim=1)
```

**Step 5**：训练循环（核心）
```python
L, H, E, nu, F = 1.0, 0.5, 1.0, 0.3, 1.0

torch.manual_seed(0)
model = ElasticityPINN(hid=64, n_layers=4)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def total_loss():
    # === PDE Loss：平衡方程残差 ===
    xy_int = torch.rand(500, 2, requires_grad=True)
    xy_int[:, 0] *= L
    xy_int[:, 1] *= H
    u_int = model(xy_int)
    strain_int = compute_strain(u_int, xy_int)
    stress_int = strain_to_stress_plane_stress(strain_int, E, nu)
    res_int = equilibrium_residual(stress_int, xy_int)
    L_pde = (res_int ** 2).mean()
    
    # === BC Loss: 左端固定 u=v=0 ===
    y_left = torch.rand(100, 1) * H
    xy_left = torch.cat([torch.zeros_like(y_left), y_left], dim=1)
    u_left = model(xy_left)
    L_bc_fix = (u_left ** 2).mean()
    
    # === BC Loss: 右端 σxx=F, σxy=0 ===
    y_right = torch.rand(100, 1) * H
    xy_right = torch.cat([torch.full_like(y_right, L), y_right], dim=1)
    xy_right.requires_grad_(True)
    u_right = model(xy_right)
    strain_right = compute_strain(u_right, xy_right)
    stress_right = strain_to_stress_plane_stress(strain_right, E, nu)
    L_bc_right = ((stress_right[:, 0:1] - F) ** 2 +
                  (stress_right[:, 2:3]) ** 2).mean()
    
    # === BC Loss: 上下边界 σyy=0, σxy=0 ===
    # 下边界
    x_bot = torch.rand(100, 1) * L
    xy_bot = torch.cat([x_bot, torch.zeros_like(x_bot)], dim=1)
    xy_bot.requires_grad_(True)
    u_bot = model(xy_bot)
    strain_bot = compute_strain(u_bot, xy_bot)
    stress_bot = strain_to_stress_plane_stress(strain_bot, E, nu)
    L_bc_bot = ((stress_bot[:, 1:2]) ** 2 + (stress_bot[:, 2:3]) ** 2).mean()
    # 上边界
    x_top = torch.rand(100, 1) * L
    xy_top = torch.cat([x_top, torch.full_like(x_top, H)], dim=1)
    xy_top.requires_grad_(True)
    u_top = model(xy_top)
    strain_top = compute_strain(u_top, xy_top)
    stress_top = strain_to_stress_plane_stress(strain_top, E, nu)
    L_bc_top = ((stress_top[:, 1:2]) ** 2 + (stress_top[:, 2:3]) ** 2).mean()
    
    L_bc = L_bc_fix + L_bc_right + L_bc_bot + L_bc_top
    
    return L_pde, L_bc

for epoch in range(20000):
    L_pde, L_bc = total_loss()
    loss = L_pde + 10 * L_bc
    
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"epoch {epoch}: L_pde={L_pde.item():.4e}, L_bc={L_bc.item():.4e}")
```

**Step 6**：可视化预测结果
- 画 `ux` 的热力图：应该从左端 0 逐渐增到右端 ≈ 1（因为 F/E=1）
- 画 `uy` 的热力图：由于泊松效应，y 方向应有收缩（中线附近小，上下边界大）
- 画 `σxx` 的热力图：应该接近均匀值 1.0（因为是简单拉伸）

**验收标准**：
- 训练完 L_pde < 1e-3，L_bc < 1e-3
- `ux` 从 0 单调增长到约 1.0
- `σxx` 平均值应在 0.9-1.1 之间（接近施加的 F=1）
- **不要求**精确的误差值，因为没有解析解；只要求物理趋势合理

**关键体会**：
- 2D 弹性 PINN 的核心流程：`MLP(xy) → 位移 → 应变 → 应力 → 平衡方程残差 → Loss`
- 这个流程就是论文 §2.3.5 的子网络在做的事，只是论文用 GNN 替换了 MLP，并嵌入到主网络中
- 本周学完这个流程，Part D 后面的任务会顺很多

---

### Day 5–6（周末）| 权重平衡实验升级

**衔接 Part B Week 10 Day 3-4**：你已经试过固定权重和动态权重策略。本周在 2D 弹性问题上再做一次实验。

**任务**：创建文件 `week17/weekend_weight_balance.py`

测试 4 组权重配置在 Day 4 的 2D 弹性问题上的表现：
- 固定 `(w_pde=1, w_bc=1)`
- 固定 `(w_pde=1, w_bc=10)`（Day 4 用的）
- 固定 `(w_pde=1, w_bc=100)`
- 动态平衡（每 200 epoch 按 `w_i = 1/L_i` 更新，参考 Part B Week 10 Day 4）

对每组记录：
- 最终 L_pde 值
- 最终 L_bc 值
- `σxx` 平均值（应接近 1.0）
- `ux(L, H/2)` 的值（中线右端点的 x 位移，应接近 1.0）

填写对比表格（写在脚本末尾的注释里），回答：
1. 权重对 L_pde 和 L_bc 最终量级的影响？
2. 哪组配置下的物理一致性（σxx 平均值、ux 最大值）最好？
3. 动态平衡相对于固定权重的优势和劣势？

**验收标准**：
- 4 组实验都能跑通
- 能填写完整表格
- **不要求**哪组"必胜"——不同配置各有特点，重点是能分析

---

### Week 17 完成标准

- [ ] 能实现 2D 偏导数和拉普拉斯算子的工具函数
- [ ] 能用 PINN 求解 2D 拉普拉斯方程（带解析解验证）
- [ ] 能用 PINN 求解 2D 弹性力学问题（位移 → 应变 → 应力 → 平衡方程）
- [ ] 能对权重配置做系统实验和分析
- [ ] 理解 "MLP → 位移 → 应变 → 应力 → 平衡方程残差"这个流水线（这是 Part D 后面的核心流程）

---

---

## Week 18: FEM 数据结构与文本解析

**衔接**：Part C Week 11 你从零构造了规则网格（9 个节点 + 4 个单元），节点坐标和单元都是 Python 里的变量。但真实 FEM 求解器（Abaqus, LS-DYNA）输出的是**文本文件**（`.inp`, `.k` 等格式）。本周学习理解这种文本数据的结构，并用 Python 解析。

**本周做什么，不做什么**：
- **做**：理解 FEM 输入/输出文件的**结构逻辑**（节点段、单元段、材料段、边界条件段、结果段）
- **做**：用 Python 解析模拟文本文件，构造 `{nodes, elements, material, bc, results}` 字典
- **不做**：真正用 Abaqus/LS-DYNA 软件——这是第二阶段 Part E 的内容
- **不做**：不规则单元类型的图转换——这是 Week 19 的内容

为什么分开做：真实软件的安装和使用本身就是独立的大模块（HyperMesh + LS-DYNA 单机需要 license、跑 Demo 要花时间）。这一周先专注在"理解数据结构"上，脱离软件。

**本周目标**：
- 理解 FEM 数据的 5 大组成（节点/单元/材料/边界条件/结果）
- 能解析模拟的 FEM 文本文件（像 Abaqus `.inp` 的简化版）
- 能做基本的数据质量检查（NaN、孤立节点、范围合理性）

---

### Day 1 | FEM 数据结构总览

**理论任务**（约 1 小时）：

**FEM 求解流程的输入输出**：
```
输入（你作为 AI 工程师能接触到的）：
  1. 节点坐标 nodes: {id: [x, y, z]}
     每个节点在空间中的位置

  2. 单元构成 elements: {id: [n1, n2, n3, ...]}
     每个单元由哪些节点围成
     常见类型：三角形(3点)/四边形(4点)/四面体(4点)/六面体(8点)

  3. 单元类型标识（可选）：某些单元用 "shell"，某些用 "solid"
  
  4. 材料属性 material: {id: {E, nu, density, ...}}
     每种材料的力学参数
     单元会指向它用的材料
  
  5. 边界条件 bc:
     - 固定约束：哪些节点的哪些位移分量被锁定
     - 载荷：哪些节点或面受什么力

输出（FEM 求解后生成的）：
  6. 节点位移 displacements: {node_id: [ux, uy, uz]}
  
  7. 单元应变 strains: {elem_id: [εxx, εyy, εxy, ...]}
  
  8. 单元应力 stresses: {elem_id: [σxx, σyy, σxy, ...]}
```

**在笔记上整理出如下对照表**（这对第二阶段 Part E 直接学 Abaqus/LS-DYNA 非常有帮助）：

| FEM 概念 | 在 GNN 里对应什么 | 举例 |
|---------|-----------------|------|
| 节点坐标 | 节点特征的一部分 | (x, y, z) 作为 node_features 的前 3 列 |
| 单元构成 | 边的来源 | 单元内节点两两建边 |
| 材料属性 | 节点特征或图属性 | 均匀材料时作为全局属性，非均匀时每节点一个值 |
| 边界条件 | 节点特征 | 用 one-hot 或连续值表示 |
| 节点位移（结果） | 节点级标签 y | 预测目标 |
| 单元应变/应力（结果） | 单元级标签 y_elem | 单元聚合后的预测目标 |

**关键理解**：FEM 软件的输入输出本质上是**表格数据**（节点表、单元表、结果表），理解了这个结构，所有求解器的数据格式都相通。

---

### Day 2 | 设计一个简化 FEM 文件格式

**实践任务**（约 2 小时）：创建文件 `week18/day02_design_format.py`

**任务**：设计一个模拟 FEM 文本格式（参考 Abaqus `.inp` 的风格简化版）。这个格式不是任何真实软件的格式，但结构和真实软件类似，是后续训练数据解析能力的"练习场"。

**约定的格式**：
```
*HEADING
  Simple FEM simulation file

*NODE
   1,  0.000,  0.000,  0.000
   2,  1.000,  0.000,  0.000
   3,  1.000,  1.000,  0.000
   4,  0.000,  1.000,  0.000
   5,  0.500,  0.500,  0.000

*ELEMENT, TYPE=S4
   1,   1,   2,   3,   4
*ELEMENT, TYPE=S3
   2,   1,   2,   5
   3,   2,   3,   5

*MATERIAL, NAME=steel
*ELASTIC
   200000.0, 0.3
*DENSITY
   7.85e-9

*BOUNDARY
   1, 1, 3, 0.0     # 节点1，x/y/z方向全部固定为0

*CLOAD
   3, 1, 100.0      # 节点3，x方向施加100N载荷

*NODE_RESULT, TYPE=DISPLACEMENT
   1,  0.0000,  0.0000,  0.0000
   2,  0.0050,  0.0000,  0.0000
   3,  0.0050, -0.0015,  0.0000
   4,  0.0000, -0.0015,  0.0000
   5,  0.0025, -0.0008,  0.0000

*ELEMENT_RESULT, TYPE=STRAIN
   1,  0.005, -0.0015, 0.0
   2,  0.005, -0.0015, 0.0
   3,  0.005, -0.0015, 0.0

*ELEMENT_RESULT, TYPE=STRESS
   1,  1050.0, -21.0, 0.0
   2,  1050.0, -21.0, 0.0
   3,  1050.0, -21.0, 0.0

*END
```

**格式约定**：
- 以 `*` 开头的行是**关键字**（指示后面数据的含义）
- 以 `#` 开头的部分是注释
- 空行忽略
- 数据行用逗号分隔
- `*NODE`：每行 = node_id, x, y, z
- `*ELEMENT, TYPE=XXX`：每行 = elem_id, n1, n2, [n3, n4, ...]；S4=四节点壳，S3=三节点壳
- `*BOUNDARY`：node_id, dof_start, dof_end, value（比如 `1, 1, 3, 0.0` 表示节点 1 的 1-3 自由度全为 0）
- `*CLOAD`：node_id, dof, value（施加在单个自由度的集中力）
- `*NODE_RESULT, TYPE=DISPLACEMENT`：每行 = node_id, ux, uy, uz
- `*ELEMENT_RESULT, TYPE=STRAIN/STRESS`：每行 = elem_id, 各分量

**创建一个测试文件**：保存上述内容为 `week18/test_fem.inp`

**验收标准**：你能看明白这个文件里每个段落代表什么，能指着一行说"这是节点 3 的 x 位移是 0.0050"。

---

### Day 3–4 | 写 FEM 解析器

**实践任务**（每天 2 小时）：创建文件 `week18/day34_fem_parser.py`

**Step 1**：设计 parser 的接口

```python
def parse_fem_file(filepath):
    """
    解析一个模拟 FEM 文件
    
    返回：dict 包含以下字段
      'nodes': {node_id (int): np.array([x, y, z])}
      'elements': [{'id': eid, 'type': 'S4'|'S3', 'nodes': [n1, n2, ...]}]
      'material': {'name': 'steel', 'E': float, 'nu': float, 'density': float}
      'boundary': [{'node': nid, 'dofs': [1,2,3], 'value': 0.0}]
      'cloads': [{'node': nid, 'dof': int, 'value': float}]
      'node_results': {'displacement': {node_id: np.array([ux, uy, uz])}}
      'elem_results': {'strain': {elem_id: np.array([...])},
                       'stress': {elem_id: np.array([...])}}
    """
    # 实现见下
    pass
```

**Step 2**：实现代码（核心思路是"按关键字切块"）

```python
import numpy as np
import re

def parse_fem_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # 初始化
    data = {
        'nodes': {},
        'elements': [],
        'material': {},
        'boundary': [],
        'cloads': [],
        'node_results': {},
        'elem_results': {}
    }
    
    # 状态机：记录当前在哪个段
    current_section = None
    current_elem_type = None
    current_result_type = None
    
    for line in lines:
        line = line.strip()
        
        # 跳过注释和空行
        if not line or line.startswith('#'):
            continue
        # 去掉行尾注释
        line = line.split('#')[0].strip()
        if not line:
            continue
        
        # 关键字行
        if line.startswith('*'):
            parts = line.split(',')
            keyword = parts[0].strip().upper()
            
            if keyword == '*NODE':
                current_section = 'node'
            elif keyword == '*ELEMENT':
                current_section = 'element'
                # 提取 TYPE=XXX
                for p in parts[1:]:
                    if 'TYPE' in p.upper():
                        current_elem_type = p.split('=')[1].strip()
            elif keyword == '*MATERIAL':
                current_section = 'material'
                for p in parts[1:]:
                    if 'NAME' in p.upper():
                        data['material']['name'] = p.split('=')[1].strip()
            elif keyword == '*ELASTIC':
                current_section = 'elastic'
            elif keyword == '*DENSITY':
                current_section = 'density'
            elif keyword == '*BOUNDARY':
                current_section = 'boundary'
            elif keyword == '*CLOAD':
                current_section = 'cload'
            elif keyword == '*NODE_RESULT':
                current_section = 'node_result'
                for p in parts[1:]:
                    if 'TYPE' in p.upper():
                        current_result_type = p.split('=')[1].strip().lower()
            elif keyword == '*ELEMENT_RESULT':
                current_section = 'elem_result'
                for p in parts[1:]:
                    if 'TYPE' in p.upper():
                        current_result_type = p.split('=')[1].strip().lower()
            elif keyword == '*HEADING' or keyword == '*END':
                current_section = None
            continue
        
        # 数据行
        tokens = [t.strip() for t in line.split(',')]
        
        if current_section == 'node':
            nid = int(tokens[0])
            coords = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            data['nodes'][nid] = coords
        
        elif current_section == 'element':
            eid = int(tokens[0])
            node_ids = [int(t) for t in tokens[1:]]
            data['elements'].append({
                'id': eid, 'type': current_elem_type, 'nodes': node_ids
            })
        
        elif current_section == 'elastic':
            data['material']['E'] = float(tokens[0])
            data['material']['nu'] = float(tokens[1])
        
        elif current_section == 'density':
            data['material']['density'] = float(tokens[0])
        
        elif current_section == 'boundary':
            data['boundary'].append({
                'node': int(tokens[0]),
                'dofs': list(range(int(tokens[1]), int(tokens[2]) + 1)),
                'value': float(tokens[3])
            })
        
        elif current_section == 'cload':
            data['cloads'].append({
                'node': int(tokens[0]),
                'dof': int(tokens[1]),
                'value': float(tokens[2])
            })
        
        elif current_section == 'node_result':
            nid = int(tokens[0])
            values = np.array([float(t) for t in tokens[1:]])
            if current_result_type not in data['node_results']:
                data['node_results'][current_result_type] = {}
            data['node_results'][current_result_type][nid] = values
        
        elif current_section == 'elem_result':
            eid = int(tokens[0])
            values = np.array([float(t) for t in tokens[1:]])
            if current_result_type not in data['elem_results']:
                data['elem_results'][current_result_type] = {}
            data['elem_results'][current_result_type][eid] = values
    
    return data


# ===== 测试 =====
if __name__ == '__main__':
    data = parse_fem_file('test_fem.inp')
    
    print(f"节点数: {len(data['nodes'])}")                    # 应为 5
    print(f"单元数: {len(data['elements'])}")                # 应为 3（1个 S4 + 2个 S3）
    print(f"材料: {data['material']}")                       # 应含 E=200000, nu=0.3
    print(f"约束数: {len(data['boundary'])}")                # 应为 1
    print(f"载荷数: {len(data['cloads'])}")                  # 应为 1
    print(f"位移结果节点数: {len(data['node_results']['displacement'])}")  # 应为 5
    print(f"应力结果单元数: {len(data['elem_results']['stress'])}")       # 应为 3
    
    # 验证：节点 3 的位移
    print(f"节点 3 的位移: {data['node_results']['displacement'][3]}")
    # 应为 [0.0050, -0.0015, 0.0000]
    
    # 验证：第一个单元的类型和节点
    print(f"单元 1 类型: {data['elements'][0]['type']}")    # S4
    print(f"单元 1 节点: {data['elements'][0]['nodes']}")   # [1, 2, 3, 4]
```

**验收标准**：
- 解析结果的所有数字和 `test_fem.inp` 里的数字**精确一致**
- 节点 3 的位移精确为 [0.0050, -0.0015, 0.0000]
- 两种单元类型（S4 和 S3）都被正确解析
- 能解释代码里"状态机"的工作方式

---

### Day 5–6（周末）| 数据质量检查 + 扩展

**Day 5 任务**：数据质量检查

创建文件 `week18/day05_data_check.py`，写 `check_fem_data(data)` 函数：

```python
def check_fem_data(data):
    """
    做基础的数据质量检查，打印所有发现的问题
    返回 is_valid: bool
    """
    issues = []
    
    # 1. NaN 检查：节点坐标、位移结果
    for nid, coords in data['nodes'].items():
        if np.isnan(coords).any():
            issues.append(f"节点 {nid} 坐标含 NaN: {coords}")
    
    for nid, disp in data['node_results'].get('displacement', {}).items():
        if np.isnan(disp).any():
            issues.append(f"节点 {nid} 位移含 NaN: {disp}")
    
    # 2. 孤立节点检查：是否每个节点都被至少一个单元引用
    referenced_nodes = set()
    for elem in data['elements']:
        for n in elem['nodes']:
            referenced_nodes.add(n)
    
    all_nodes = set(data['nodes'].keys())
    isolated = all_nodes - referenced_nodes
    if isolated:
        issues.append(f"孤立节点（未被任何单元引用）: {isolated}")
    
    # 3. 单元引用的节点是否都在节点表中
    for elem in data['elements']:
        for n in elem['nodes']:
            if n not in data['nodes']:
                issues.append(f"单元 {elem['id']} 引用了不存在的节点 {n}")
    
    # 4. 结果量级检查（可选，如果知道预期范围）
    # 这里举个例子：如果位移模长 > 坐标最大值，可能异常
    if data['node_results'].get('displacement'):
        max_coord = max(np.abs(c).max() for c in data['nodes'].values())
        for nid, disp in data['node_results']['displacement'].items():
            if np.linalg.norm(disp) > max_coord:
                issues.append(f"节点 {nid} 位移量级异常大: {disp}")
    
    # 打印结果
    if issues:
        print("发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("数据质量检查通过")
        return True


# ===== 测试 =====
if __name__ == '__main__':
    data = parse_fem_file('test_fem.inp')
    check_fem_data(data)   # 应该通过
    
    # 制造一个坏数据：删掉节点 5 的记录（让它变孤立）
    # 或者修改某个节点坐标为 NaN
    # 验证 check_fem_data 能找到问题
```

**Day 6 任务**：扩展 `test_fem.inp`

手动扩展 `test_fem.inp`，加入一个更大的网格（比如 5×3 的 15 个节点、8 个四边形单元），手动填写位移结果（用简化公式 `ux = 0.01 * x`, `uy = -0.003 * x * (y - 1)`）。

验证解析器能正常处理更大的文件。

---

### Week 18 完成标准

- [ ] 能说清楚 FEM 数据的 5 大组成（节点/单元/材料/边界/结果）和各自在 GNN 里的对应
- [ ] 能设计并解析类 Abaqus 风格的 FEM 文本文件（`.inp` 格式简化版）
- [ ] 解析器支持多种单元类型（S4, S3 等）
- [ ] 能做基础数据质量检查（NaN、孤立节点、引用错误、量级异常）
- [ ] 解析的结果和测试文件里的数字精确一致

---

---

## Week 19: 不规则网格 → 图（升级版 mesh_to_pyg）

**衔接**：Part C Week 11 你只处理了规则矩形网格。Week 18 你能解析 FEM 文件了，但现实中这些文件里的单元是**三角形、四边形混合**的不规则网格。本周把"mesh → graph"转换升级到能处理这种情况。

**本周目标**：
- 能处理多种单元类型（S3 三角形、S4 四边形）混合的网格
- 能构造通用的 `mesh_to_pyg` 函数
- 能在不规则网格上做可视化（三角剖分图）
- 生成一个**物理自洽**的合成 FEM 数据集（为 Week 20-22 做准备）

**本周不做**：用 FEniCS 生成真实 FEM 数据——第二阶段做。本周用 "人造合成数据 + 简化物理公式" 继续。

---

### Day 1 | 手动生成不规则网格数据

**实践任务**（约 2 小时）：创建文件 `week19/day01_make_irregular_mesh.py`

**任务**：手动写一个函数，生成一个不规则但物理合理的"半圆形带孔板"网格。

为什么手动做：第二阶段会用 Gmsh 自动生成，但现在先手动做一个小网格，让你对不规则网格的数据结构有具体感觉。

**思路**（简化版）：
- 画一个 4×3 的基本矩形（12 节点）
- 在中心加一个节点（节点 13），用 4 个三角形连到四周
- 保持原有矩形的其他单元为四边形

```python
import numpy as np

def build_irregular_mesh():
    """
    构造一个简单的不规则网格：
    - 4x3 矩形框架 + 中心加 1 个节点 + 几个三角形单元
    
    返回：
      nodes: dict {id: np.array([x, y, z])}
      elements: list of dicts [{'id': ..., 'type': 'S3'|'S4', 'nodes': [...]}]
    """
    nodes = {}
    # 4x3 规则框架节点
    node_id = 1
    for j in range(3):    # y = 0, 1, 2
        for i in range(4):    # x = 0, 1, 2, 3
            nodes[node_id] = np.array([float(i), float(j), 0.0])
            node_id += 1
    # 此时有 12 个节点（id 1-12）
    
    # 在中心加一个节点（坐标 1.5, 1.0）
    nodes[13] = np.array([1.5, 1.0, 0.0])
    
    elements = []
    # 左下四边形：节点 1,2,6,5
    elements.append({'id': 1, 'type': 'S4', 'nodes': [1, 2, 6, 5]})
    # 右下四边形：节点 3,4,8,7
    elements.append({'id': 2, 'type': 'S4', 'nodes': [3, 4, 8, 7]})
    # 左上四边形：节点 5,6,10,9
    elements.append({'id': 3, 'type': 'S4', 'nodes': [5, 6, 10, 9]})
    # 右上四边形：节点 7,8,12,11
    elements.append({'id': 4, 'type': 'S4', 'nodes': [7, 8, 12, 11]})
    # 中间四个三角形连到中心节点 13
    elements.append({'id': 5, 'type': 'S3', 'nodes': [2, 3, 13]})
    elements.append({'id': 6, 'type': 'S3', 'nodes': [3, 7, 13]})
    elements.append({'id': 7, 'type': 'S3', 'nodes': [7, 6, 13]})
    elements.append({'id': 8, 'type': 'S3', 'nodes': [6, 2, 13]})
    
    return nodes, elements


# 可视化
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    nodes, elements = build_irregular_mesh()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 画单元
    for elem in elements:
        coords = np.array([nodes[n] for n in elem['nodes']])
        coords_closed = np.vstack([coords, coords[0:1]])   # 闭合多边形
        color = 'lightblue' if elem['type'] == 'S4' else 'lightgreen'
        ax.fill(coords_closed[:, 0], coords_closed[:, 1], 
                color=color, edgecolor='black', lw=1)
    
    # 画节点
    for nid, coord in nodes.items():
        ax.scatter(coord[0], coord[1], c='red', s=60, zorder=10)
        ax.annotate(str(nid), coord[:2], xytext=(5, 5),
                    textcoords='offset points', fontsize=9)
    
    ax.set_aspect('equal'); ax.set_title('Irregular Mesh (S4 = blue, S3 = green)')
    plt.savefig('week19_irregular_mesh.png', dpi=100)
```

**验收标准**：
- 13 个节点、8 个单元（4 个 S4 + 4 个 S3）
- 图能生成，四边形和三角形用不同颜色区分
- 中心节点 13 连到周围 4 个节点（构成 4 个三角形）

---

### Day 2–3 | 通用的 mesh_to_pyg 函数

**衔接 Part C Week 11 的 `mesh_to_edges`**：那个函数假设单元是固定的 4 节点四边形。现在要升级到支持任意单元类型。

**实践任务**（每天 2 小时）：创建文件 `utils/mesh_to_graph.py`（注意放在 `utils/` 方便复用）

```python
import numpy as np
import torch
from torch_geometric.data import Data

def build_edges_from_elements(elements, node_id_to_idx=None):
    """
    从单元连接关系构造边集合
    
    参数：
      elements: list of dicts [{'id': ..., 'type': 'S3'|'S4', 'nodes': [...]}]
      node_id_to_idx: dict，把 FEM 节点 id（1-based）映射到图索引（0-based）
                     如果 None，假设节点 id 就是 0-based 的连续索引
    返回：
      edge_set: set of tuples (src, dst)，都是 0-based 索引
               包含双向（a→b 和 b→a 都存）
    """
    edge_set = set()
    
    for elem in elements:
        node_ids = elem['nodes']
        # 转成 0-based 索引（如果提供了映射）
        if node_id_to_idx is not None:
            indices = [node_id_to_idx[n] for n in node_ids]
        else:
            indices = node_ids
        
        n = len(indices)
        # 单元内任意两点都建边（论文规则）
        for i in range(n):
            for j in range(i + 1, n):
                a, b = indices[i], indices[j]
                edge_set.add((a, b))
                edge_set.add((b, a))    # 双向
    
    return edge_set


def mesh_to_pyg(nodes_dict, elements_list,
                node_features_dict=None,
                node_labels_dict=None,
                elem_labels_dict=None):
    """
    把 mesh 数据转为 PyG Data 对象（通用版，支持混合单元类型）
    
    参数：
      nodes_dict: {node_id (int): np.array([x, y, z])}  使用 1-based id
      elements_list: [{'id': ..., 'type': ..., 'nodes': [...]}]  nodes 里的 id 也是 1-based
      node_features_dict: {node_id: np.array([...])}  可选，额外节点特征
                         如果 None，只用坐标作为特征
      node_labels_dict: {node_id: np.array([...])}  可选，节点级标签（如位移）
      elem_labels_dict: {elem_id: np.array([...])}  可选，单元级标签（如应力）
    
    返回：
      data: PyG Data 对象
      elem_list_ordered: 单元列表（按图内索引排序，方便单元聚合）
    """
    # ---- 建立 node_id 到 0-based 索引的映射 ----
    sorted_node_ids = sorted(nodes_dict.keys())
    node_id_to_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}
    N = len(sorted_node_ids)
    
    # ---- 构造节点特征 ----
    coord_features = np.stack([nodes_dict[nid] for nid in sorted_node_ids])  # (N, 3)
    
    if node_features_dict is not None:
        extra_features = np.stack([node_features_dict[nid] for nid in sorted_node_ids])
        x = np.concatenate([coord_features, extra_features], axis=1)
    else:
        x = coord_features
    x = torch.from_numpy(x.astype(np.float32))
    
    # ---- 构造边索引（用通用函数）----
    edge_set = build_edges_from_elements(elements_list, node_id_to_idx)
    src_list, dst_list = zip(*edge_set) if edge_set else ([], [])
    edge_index = torch.tensor([list(src_list), list(dst_list)], dtype=torch.long)
    
    # ---- 构造边特征（相对坐标 + 距离）----
    edge_attr_list = []
    for s, d in zip(src_list, dst_list):
        diff = x[d, :3] - x[s, :3]        # 取前 3 列作为坐标
        dist = torch.norm(diff).unsqueeze(0)
        edge_attr_list.append(torch.cat([diff, dist]))
    edge_attr = torch.stack(edge_attr_list)       # shape=(n_edges, 4) = dx,dy,dz,dist
    
    # ---- 构造节点标签 ----
    y = None
    if node_labels_dict is not None:
        y_np = np.stack([node_labels_dict[nid] for nid in sorted_node_ids])
        y = torch.from_numpy(y_np.astype(np.float32))
    
    # ---- 构造 PyG Data ----
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # ---- 单独返回 elem_list（用于后续单元聚合矩阵）----
    # 单元的节点 id 要转成图内 0-based 索引
    elem_list_ordered = []
    for elem in elements_list:
        elem_indices = [node_id_to_idx[n] for n in elem['nodes']]
        elem_list_ordered.append({
            'id': elem['id'],
            'type': elem['type'],
            'indices': elem_indices     # 图内的 0-based 索引
        })
    
    # 附加元素标签信息
    if elem_labels_dict is not None:
        sorted_elem_ids = sorted(elem_labels_dict.keys())
        y_elem = np.stack([elem_labels_dict[eid] for eid in sorted_elem_ids])
        data.y_elem = torch.from_numpy(y_elem.astype(np.float32))
    
    return data, elem_list_ordered


# ===== 测试 =====
if __name__ == '__main__':
    from week19.day01_make_irregular_mesh import build_irregular_mesh
    
    nodes, elements = build_irregular_mesh()
    data, elem_list = mesh_to_pyg(nodes, elements)
    
    print(f"节点数: {data.num_nodes}")            # 13
    print(f"边数: {data.num_edges}")              # 双向，具体数需手算验证
    print(f"节点特征 shape: {data.x.shape}")      # (13, 3) 仅坐标
    print(f"边特征 shape: {data.edge_attr.shape}")# (num_edges, 4)
    print(f"单元数: {len(elem_list)}")            # 8
    
    # 验证边数：
    # 4 个 S4 单元各贡献 6 条无向边 = 24，但相邻单元可能共享边
    # 4 个 S3 单元各贡献 3 条无向边 = 12，也可能共享
    # 精确数字请手算验证
```

**验收标准**：
- 函数能处理 S3 和 S4 混合的单元
- 节点特征包含坐标，shape=(13, 3)
- 边特征包含 (dx, dy, dz, dist)，shape=(n_edges, 4)
- 单元列表中每个单元有 `indices` 字段（0-based）
- 能处理 `nodes_dict` 的 id 不从 0 开始（比如 FEM 常用 1-based）的情况

---

### Day 4 | 升级单元聚合（支持混合单元类型）

**衔接 Part C Week 15 Day 3**：那时的聚合矩阵假设每个单元有固定数量的节点。现在要支持 S3（3 节点）和 S4（4 节点）混合。

**实践任务**（约 2 小时）：修改 `utils/gnn_models.py` 里的 `build_element_aggregation_matrix`

```python
def build_element_aggregation_matrix_generic(elem_list, num_nodes, normalize=True):
    """
    通用版本单元聚合矩阵，支持混合单元类型
    
    参数：
      elem_list: list of dicts，每个元素有 'indices' 字段（0-based 节点索引）
      num_nodes: 图中节点总数
      normalize: 是否归一化（让聚合 = 平均）
    返回：
      C: torch.Tensor shape=(num_elements, num_nodes)
    
    注意：本实现用 dense 矩阵存储，只适合小规模教学用图（节点 < 几千）。
    真实工程场景应该用稀疏矩阵 (torch.sparse_coo_tensor) 或 
    torch_scatter.scatter_mean 实现。
    """
    num_elems = len(elem_list)
    C = torch.zeros(num_elems, num_nodes)
    
    for eid, elem in enumerate(elem_list):
        indices = elem['indices']
        # 归一化：每个单元的每个节点贡献权重 = 1/单元节点数
        val = 1.0 / len(indices) if normalize else 1.0
        for nid in indices:
            C[eid, nid] = val
    
    return C


# ===== 测试 =====
if __name__ == '__main__':
    from week19.day01_make_irregular_mesh import build_irregular_mesh
    from utils.mesh_to_graph import mesh_to_pyg
    
    nodes, elements = build_irregular_mesh()
    data, elem_list = mesh_to_pyg(nodes, elements)
    
    C = build_element_aggregation_matrix_generic(elem_list, data.num_nodes)
    print(f"C shape: {C.shape}")        # (8, 13)
    
    # 验证：S4 单元的权重 = 0.25，S3 单元的权重 = 1/3
    elem_0 = elem_list[0]    # 应为 S4
    print(f"单元 0 类型: {elem_0['type']}, 非零权重: {C[0][C[0] > 0].tolist()}")
    # 应为 [0.25, 0.25, 0.25, 0.25]
    
    elem_4 = elem_list[4]    # 应为 S3
    print(f"单元 4 类型: {elem_4['type']}, 非零权重: {C[4][C[4] > 0].tolist()}")
    # 应为 [1/3, 1/3, 1/3] ≈ [0.333, 0.333, 0.333]
```

**验收标准**：
- S4 单元的 4 个节点每个权重 0.25
- S3 单元的 3 个节点每个权重约 0.333
- `C @ node_features` 能正确聚合到单元特征

---

### Day 5–6（周末）| 生成物理自洽的合成数据集

**为什么叫"物理自洽"**：Part C Week 16 的合成数据是拍脑袋公式，标签之间不严格满足物理方程。本周末做得更好一点——让位移、应变、应力在**简化模型**下数学上自洽。

**简化模型假设**：均匀单轴拉伸板，只用最简单的线弹性关系。

**实践任务**：创建文件 `week19/weekend_synth_dataset.py`

**Step 1**：构造一个 5×3 的规则网格（用 Part C Week 11 的函数，或 Week 19 Day 1 类似的方式）

**Step 2**：对每个节点和单元生成标签

```python
def generate_synthetic_labels(nodes_dict, elem_list, E, nu, F, L, H):
    """
    为一个矩形板生成"物理自洽"的合成标签
    物理假设：平面应力，左端固定，右端均匀拉伸 F
    
    简化模型（避免复杂的解析解）：
      ux(x, y) = (F / (E * H)) * x     # 左端 ux=0，右端 ux 最大
      uy(x, y) = -nu * (F / (E * H)) * y * (x / L)   # 泊松收缩
      
      几何方程导出的应变：
        εxx = du/dx = F / (E * H)  （均匀）
        εyy = dv/dy = -nu * F / (E * H) * (x / L)  （依赖 x）
        γxy = du/dy + dv/dx = -nu * F / (E * H) * (y / L)  （依赖 y）
      
      本构方程导出的应力（平面应力）：
        σxx = E/(1-nu²) * (εxx + nu * εyy)
        σyy = E/(1-nu²) * (εyy + nu * εxx)
        σxy = E/(2(1+nu)) * γxy
    
    注意：这不是严格的弹性解，但它满足几何+本构关系的自洽。
    平衡方程不严格满足（因为我们用了简化位移场），但对本周的合成数据
    够用了，目的是让 GNN 有"自洽的训练信号"。
    """
    node_labels = {}
    for nid, coord in nodes_dict.items():
        x, y, _ = coord
        ux = (F / (E * H)) * x
        uy = -nu * (F / (E * H)) * y * (x / L)
        node_labels[nid] = np.array([ux, uy])
    
    # 单元级标签：用单元中心位置估算应变和应力
    elem_labels_strain = {}
    elem_labels_stress = {}
    for elem in elem_list:
        # 单元中心坐标
        center = np.mean([nodes_dict[nid][:2] for nid in 
                         (elem['indices'] if 'indices' in elem else elem['nodes'])], axis=0)
        x, y = center
        
        eps_xx = F / (E * H)
        eps_yy = -nu * F / (E * H) * (x / L)
        gamma_xy = -nu * F / (E * H) * (y / L)
        
        coef = E / (1 - nu**2)
        sigma_xx = coef * (eps_xx + nu * eps_yy)
        sigma_yy = coef * (eps_yy + nu * eps_xx)
        sigma_xy = E / (2 * (1 + nu)) * gamma_xy
        
        eid = elem['id']
        elem_labels_strain[eid] = np.array([eps_xx, eps_yy, gamma_xy])
        elem_labels_stress[eid] = np.array([sigma_xx, sigma_yy, sigma_xy])
    
    return node_labels, elem_labels_strain, elem_labels_stress
```

**Step 3**：生成一个数据集，10 个样本，每个样本 E 和 F 不同

```python
def generate_dataset(n_samples=10, seed=0):
    np.random.seed(seed)
    # 取用 Part C Week 11 的 build_regular_mesh
    from phase1.part_c.week11 import build_regular_mesh
    
    dataset = []
    for i in range(n_samples):
        E = np.random.uniform(150, 250)
        nu = 0.3
        F = np.random.uniform(50, 150)
        L, H = 4.0, 2.0    # 5x3 网格，1-based 长度
        
        # 网格
        nodes_arr, elements_arr = build_regular_mesh(nx=5, ny=3)
        # 转成 dict 格式（1-based）
        nodes_dict = {i+1: np.array([c[0], c[1], 0.0]) for i, c in enumerate(nodes_arr)}
        elements_list = []
        for eid, nids in enumerate(elements_arr):
            elements_list.append({
                'id': eid+1, 'type': 'S4',
                'nodes': [n+1 for n in nids]
            })
        
        # 生成标签
        node_lab, strain_lab, stress_lab = generate_synthetic_labels(
            nodes_dict, elements_list, E=E, nu=nu, F=F, L=L, H=H)
        
        # 附加额外节点特征（E 和 F 信息）
        node_feat_dict = {nid: np.array([E, F]) for nid in nodes_dict.keys()}
        
        # 转成 PyG Data
        data, elem_list_ordered = mesh_to_pyg(
            nodes_dict, elements_list,
            node_features_dict=node_feat_dict,
            node_labels_dict=node_lab,
            elem_labels_dict=stress_lab    # 暂时用应力作为单元标签
        )
        # 把 strain 也附加上
        sorted_eids = sorted(strain_lab.keys())
        data.y_elem_strain = torch.from_numpy(
            np.stack([strain_lab[eid] for eid in sorted_eids]).astype(np.float32)
        )
        
        # 保存元数据（后面物理 loss 用）
        data.E = E
        data.nu = nu
        data.elem_list = elem_list_ordered
        
        dataset.append(data)
    
    return dataset


# 保存到磁盘（方便复用）
if __name__ == '__main__':
    dataset = generate_dataset(n_samples=20)
    torch.save(dataset, 'week19_dataset.pt')
    print(f"数据集已保存，共 {len(dataset)} 个样本")
    print(f"第一个样本：节点数={dataset[0].num_nodes}, 单元数={len(dataset[0].elem_list)}")
    print(f"节点标签 shape: {dataset[0].y.shape}")        # (15, 2)
    print(f"单元应力标签 shape: {dataset[0].y_elem.shape}")   # (8, 3)
```

**验收标准**：
- 数据集能生成，共 20 个样本
- 每个样本：15 节点、8 单元、节点标签 (15,2)、应力标签 (8,3)
- 能保存到磁盘并重新加载

**关于 `node_features` 的说明**：这里节点特征只包含 (x, y, z, E, F)——5 维。第二阶段会加入更多特征（边界条件 one-hot 等）。先从简单的开始。

---

### Week 19 完成标准

- [ ] 能处理混合单元类型（S3, S4）的网格
- [ ] 有通用的 `mesh_to_pyg` 函数放在 `utils/`，支持 FEM 1-based id
- [ ] 有通用的单元聚合矩阵构造函数
- [ ] 能生成物理自洽的合成数据集（geometric + constitutive 满足）
- [ ] 数据集能保存到磁盘、重新加载

---

---

## Week 20: GNN + 物理约束 Loss 融合（Part D 核心周）

**本周定位**：这是整个第一阶段最重要的一周——你第一次把 **Part B 的物理约束 Loss** 嵌入到 **Part C 的 GNN 架构**中。这就是 PhyFENet 的核心思想（论文 §2.3）。

**衔接**：
- Part B Week 10：1D PINN（MLP + 物理 Loss）
- Part C Week 15：编码器-处理器-解码器 GNN + 单元聚合
- Week 17 Day 4：2D 弹性 PINN（位移 → 应变 → 应力 → 平衡方程）
- Week 19 Weekend：物理自洽合成数据集

**本周要把这些整合起来**：
```
GNN(mesh) → 节点位移场 u
用 autograd 算 du/dx → 节点级应变 ε
（或者通过单元聚合矩阵 → 单元位移 → 单元应变）
ε 通过胡克定律 → 应力 σ
(1) 纯数据 Loss：‖u_pred - u_true‖²
(2) 几何方程 Loss：‖ε_pred - f(u_pred)‖²
(3) 本构方程 Loss：‖σ_pred - g(ε_pred)‖²
```

**本周目标**：
- 纯数据驱动的 GNN baseline 训练（作为对照组）
- 在 GNN Loss 中嵌入几何方程约束
- 在 GNN Loss 中嵌入本构方程约束
- 对比三种训练（纯数据 / 数据+几何 / 数据+几何+本构）的效果

**特别提醒**：本周的每个实验都会比前面的任务跑得慢（因为 autograd 求导 + GNN 前向），请预留充足时间，**不要赶**。

---

### Day 1 | 在 GNN 网络上算应变：图上的离散梯度

**衔接问题**：Part B 的 PINN 里，应变 `ε = du/dx` 是对"连续坐标 x"求导，用的是 autograd。但 GNN 输出的是**一组离散节点上的位移值**，不是连续函数——**怎么在图上算 `du/dx`？**

**两种思路**：

**思路 A：通过 autograd 对输入坐标求导**
- 把节点坐标设为 `requires_grad=True`
- GNN 输出位移 u(xy_i)
- 对输入坐标求导得到每个节点位置处的 `du/dx, du/dy`
- 问题：GNN 的输出对节点位置的"连续性"不一定好，求导误差大

**思路 B：在单元上用有限差分（近似离散梯度）**
- 对每个单元，用单元内节点位移的差分来近似梯度
- 例如四节点四边形单元，`∂u/∂x ≈ (u_right - u_left) / Δx`
- 问题：需要单元的几何信息，实现复杂

**论文用的思路**：论文 §2.3.5 用的是**变种思路 B** + 子网络——让一个独立的"应变网络"学习从节点位移到单元应变的映射（`strain = StrainNet(node_disp + coord)`）。这个网络只需要在单元内学"数值微分"。

**本周我们用简化版**：
- 纯 GNN baseline（只有数据 loss）先跑通
- 加入几何方程约束时，用一个轻量的 MLP 学 `strain = MLP(node_disp_of_elem)`
- 这是为 Week 21 的多级网络做铺垫

**实践任务**（约 1.5 小时）：创建文件 `week20/day01_baseline_gnn.py`

**Step 1**：基线训练——纯数据驱动

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from utils.gnn_models import PhyFENet_WithElement
# 这个类已经在 Part C Week 15 做好

# 加载 Week 19 生成的数据集
dataset = torch.load('week19_dataset.pt')
train_data = dataset[:15]
val_data = dataset[15:]

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)

# 模型：输入 node_feat=5（x,y,z,E,F），边特征 4，隐藏 64，节点输出 2（ux,uy），单元输出 3（σxx,σyy,σxy）
model = PhyFENet_WithElement(
    node_in=5, edge_in=4, hid=64,
    node_out_dim=2, elem_out_dim=3, n_mp_layers=3
)
# 参考 Part C Week 15 的实现调整参数

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_step(batch, model, criterion, use_physics=False):
    # 首先需要从 batch 构造单元聚合矩阵（每个图不同）
    # ... 略，按 Part C Week 15 思路实现
    
    node_pred, elem_pred = model(batch, elem_aggr_matrix=...)
    L_data_node = criterion(node_pred, batch.y)
    L_data_elem = criterion(elem_pred, batch.y_elem)
    
    loss = L_data_node + L_data_elem
    return loss, {'node': L_data_node.item(), 'elem': L_data_elem.item()}
```

> **实现说明**：完整代码较长，核心在 batch 中构造聚合矩阵、训练循环、评估。
> 本 Day 不细写完整代码（你已经在 Part C Week 15 有参考模板），重点是**跑通纯数据 baseline**作为对照组。

**验收标准**：
- 纯数据驱动 baseline 能收敛
- 记录验证集上 节点位移 MAE 和 单元应力 MAE

---

### Day 2–3 | 几何方程约束：让应变与位移梯度一致

**核心思想**（论文 §3.2.2 式 3.41-3.42）：
- GNN 预测了节点位移 `u_pred`
- GNN 预测了单元应变 `ε_pred`
- 数学上：应变应该 = 位移的某种空间导数（几何方程 B·u）
- 如果两者不一致，说明网络还没学到正确的物理关系
- 把 "**ε_pred 和从 u_pred 算出来的应变**" 之间的差作为 Loss 的一项

**实现方式**：引入一个小的"应变估计网络"（简化版子网络）

```python
class StrainSubNet(nn.Module):
    """
    学习从"单元节点的位移特征"估计单元应变
    输入：每个单元的节点位移 + 节点坐标（拼起来）
    输出：该单元的 (εxx, εyy, γxy)
    
    这是论文 §2.3.5 的 StrainNet 的简化版。
    """
    def __init__(self, max_nodes_per_elem=4, hid=32):
        super().__init__()
        # 输入：每个节点 (ux, uy, x, y) = 4维 * 节点数
        # 对于混合单元类型（3 或 4 节点），用 padding + mask 处理
        # 这里为了简化，先只处理 S4（4 节点）
        self.net = nn.Sequential(
            nn.Linear(4 * max_nodes_per_elem, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 3)     # εxx, εyy, γxy
        )
    
    def forward(self, node_disp, node_coord, elem_indices):
        """
        node_disp: (N, 2) 所有节点的位移
        node_coord: (N, 2) 所有节点的坐标
        elem_indices: list of lists，每个是单元的节点索引
        返回：(E, 3) 单元应变
        """
        features = []
        for indices in elem_indices:
            elem_feat = torch.cat([
                node_disp[indices].flatten(),   # ux_1, uy_1, ux_2, uy_2, ...
                node_coord[indices].flatten()
            ])
            # 处理混合单元：用零 padding 到 max_nodes_per_elem*4
            # 这里假设都是 S4，先不处理 padding
            features.append(elem_feat)
        features = torch.stack(features)
        return self.net(features)
```

**几何方程 Loss**：
```python
def geometric_loss(strain_pred_from_gnn, strain_pred_from_subnet):
    """
    论文 §3.2.2 式 3.42: L_b = ‖ε_pred - Bu‖²
    在这里：
      strain_pred_from_gnn: GNN 主网络直接预测的应变
      strain_pred_from_subnet: 子网络从位移估计的应变
    """
    return ((strain_pred_from_gnn - strain_pred_from_subnet) ** 2).mean()
```

**训练流程**：
```python
# 联合训练
for epoch in range(200):
    for batch in train_loader:
        # GNN 前向
        node_pred, elem_strain_pred = model(batch, ...)
        
        # 子网络估计应变
        strain_from_disp = strain_subnet(
            node_disp=node_pred,
            node_coord=batch.x[:, :2],   # 前两列是 x, y
            elem_indices=[elem['indices'] for elem in batch.elem_list]
        )
        
        # 数据 loss
        L_data_node = criterion(node_pred, batch.y)
        L_data_strain = criterion(elem_strain_pred, batch.y_elem_strain)
        
        # 几何方程 loss
        L_geom = geometric_loss(elem_strain_pred, strain_from_disp)
        
        # 总 loss
        loss = L_data_node + L_data_strain + 0.1 * L_geom
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**对比实验**：
- baseline：只用 `L_data_node + L_data_strain`
- +几何：加上 `L_geom`

记录验证集 MAE，对比两者。

**预期**：加入几何约束通常让预测的物理场更一致，尤其在训练数据少时效果更明显。但这个不是绝对的——如果合成数据本身就完美自洽，几何 loss 带来的提升可能很小。重点是"能实现这个机制"和"能分析结果"。

**验收标准**：
- 几何方程 loss 能正常参与训练
- 两种模型都能收敛
- 能对比两种训练的 MAE 和物理一致性（几何方程残差）
- **不要求**加几何约束后一定更好（分析型验收）

---

### Day 4–5 | 本构方程约束

**核心思想**（论文 §3.2.2 式 3.44）：
- 应变 ε → 应力 σ 的关系是本构方程（胡克定律，已知）
- 如果 GNN 预测的 ε 和 σ 不满足 `σ = D · ε`，说明没学到物理
- 把 "GNN 预测的 σ" 和 "用 ε 通过胡克定律算出的 σ" 的差作为 Loss

**实现**：Week 17 Day 4 的 `strain_to_stress_plane_stress` 直接复用

```python
from week17.day04_2d_elasticity import strain_to_stress_plane_stress

def constitutive_loss(stress_pred, strain_pred, E=1.0, nu=0.3):
    """
    论文 §3.2.2 式 3.44: L_d = ‖σ_pred - D·ε_pred‖²
    """
    stress_from_strain = strain_to_stress_plane_stress(strain_pred, E, nu)
    return ((stress_pred - stress_from_strain) ** 2).mean()
```

**完整训练流程**（在 Day 2-3 基础上再加一层）：
```python
loss = L_data_node + L_data_strain + L_data_stress + \
       w_geom * L_geom + w_const * L_const
```

**对比实验**（三种组合）：
1. baseline（只有数据 loss）
2. baseline + 几何 loss
3. baseline + 几何 loss + 本构 loss

**验收标准**：
- 三种组合都能收敛
- 记录每种组合的 节点位移 MAE、单元应变 MAE、单元应力 MAE
- 绘制对比图
- 可以分析：哪种组合的"应力场连续性"最好（可视化应力热力图看）
- **不要求**加物理约束一定更好——分析型验收

---

### Day 6 | 本周总结 + 物理一致性定义

**任务**：在 `week20/summary.md` 里写本周的总结，回答以下问题：

1. 纯数据、+几何、+本构 三种配置下，节点位移、应变、应力的 MAE 分别是多少？
2. "物理一致性"如何量化？（建议指标：几何方程残差 `‖ε_pred - B·u_pred‖` 的均值、本构方程残差 `‖σ_pred - D·ε_pred‖` 的均值）
3. 加物理约束后，哪些指标改善了？哪些没变或变差了？
4. 权重配置（`w_geom`, `w_const`）如何影响训练？

**关键体会**（写下来）：物理约束不一定会让"数据拟合 MAE"更小，但通常会让**物理一致性**更好。两者是不同维度的评估。

---

### Week 20 完成标准

- [ ] 跑通纯数据驱动的 GNN baseline（Week 19 合成数据上）
- [ ] 实现几何方程约束 loss（配合子网络估计应变）
- [ ] 实现本构方程约束 loss（用 Week 17 的 `strain_to_stress`）
- [ ] 能对比三种训练配置的性能
- [ ] 能区分"数据拟合 MAE"和"物理一致性指标"
- [ ] 理解论文 §2.3 式 2.23 的 `Loss = ω·L_PDE + ω·L_BC + ω·L_Data` 在实际实现中是什么样子

---

---

## Week 21: 多级网络（论文 §2.3.5）

**衔接 Week 20**：你已经在单个 PhyFENet 模型里嵌入了几何和本构 loss。但论文 §2.3.5 提出了一个更精细的架构——**多级网络**：
- **子网络**（StrainNet, StressNet）：独立预训练，学习单元级的几何和本构关系
- **主网络**（DisNet）：图神经网络，学习全局位移
- **联合微调**：把预训练好的子网络嵌入主网络，再一起 fine-tune

**论文原文（§2.3.5）**：
> "首先在单元尺度上基于 PINN 预训练参数量较小的子网络，学习和捕捉几何方程和物理方程。将预训练好的子网络嵌入主网络中，可以有效约束网络收敛方向和对整个结构变形的学习效果。"

**为什么要这么做**：
- 子网络参数少（几千个）+ 训练数据多（把每个 FEM 样本分解到单元级，数据量 × 单元数）→ **容易学好**
- 把学好的子网络嵌入主网络后，主网络在训练时会**被子网络的物理约束"拉住"**，不容易发散

**本周目标**：
- 独立预训练 StrainNet（学习节点位移 → 单元应变）
- 独立预训练 StressNet（学习单元应变 → 单元应力）
- 把两个子网络嵌入 PhyFENet_WithElement
- 对比"从零训练"和"子网络预训练+联合微调"两种策略

---

### Day 1–2 | 预训练 StrainNet

**数据生成**：从 Week 19 的数据集里，把所有样本的所有单元展开，得到一个**单元级数据集**。

```python
import torch

def build_strain_pretrain_dataset(gnn_dataset):
    """
    从 GNN 数据集展开成单元级数据集
    对每个样本的每个单元：
      输入 = 该单元节点的位移 + 坐标
      输出 = 该单元的真实应变
    """
    strain_inputs = []
    strain_labels = []
    for data in gnn_dataset:
        node_disp = data.y          # (N, 2)
        node_coord = data.x[:, :2]  # (N, 2)
        elem_list = data.elem_list
        
        # 每个单元做一个样本
        for i, elem in enumerate(elem_list):
            indices = elem['indices']
            # 如果是 S4（4 节点），特征 shape=(16,)
            # 如果是 S3（3 节点），shape=(12,)，需要 padding 到 16
            disp_flat = node_disp[indices].flatten()
            coord_flat = node_coord[indices].flatten()
            feat = torch.cat([disp_flat, coord_flat])
            
            # padding 到 max_nodes=4
            max_feat_dim = 4 * 4    # 4 节点 × (2 位移 + 2 坐标)
            if feat.numel() < max_feat_dim:
                feat = torch.cat([feat, torch.zeros(max_feat_dim - feat.numel())])
            
            strain_inputs.append(feat)
            strain_labels.append(data.y_elem_strain[i])
    
    X = torch.stack(strain_inputs)
    y = torch.stack(strain_labels)
    return X, y

# 预训练
X_train, y_train = build_strain_pretrain_dataset(train_data)
print(f"单元级训练样本数: {X_train.shape[0]}")    # 大约 = 训练图数 × 单元数

strain_net = StrainSubNet(max_nodes_per_elem=4, hid=32)
optimizer = optim.Adam(strain_net.parameters(), lr=1e-3)

for epoch in range(500):
    pred = strain_net(X_train)    # 这里需要改 StrainSubNet 的 forward 接受 flat 输入
    loss = ((pred - y_train) ** 2).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**验收标准**：
- 单元级数据集展开成功，样本数 = 训练图数 × 平均单元数
- StrainNet 能收敛到低 MAE（< 1e-4，因为合成数据是自洽的）
- 保存预训练好的 StrainNet 权重：`torch.save(strain_net.state_dict(), 'strain_net_pretrained.pth')`

---

### Day 3 | 预训练 StressNet

类似 StrainNet，但输入是"单元应变 + 材料参数"，输出是"单元应力"。

```python
class StressSubNet(nn.Module):
    """学习 ε → σ，输入还可以包含材料参数（E, nu）以便迁移学习"""
    def __init__(self, strain_dim=3, mat_dim=2, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(strain_dim + mat_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 3)
        )
    def forward(self, strain, material):
        inp = torch.cat([strain, material], dim=-1)
        return self.net(inp)
```

预训练同 Day 1-2。

**验收**：StressNet 能学到近乎完美的胡克定律（MAE 很小）。

---

### Day 4 | 多级网络联合训练

**核心思路**：
```
Data ─→ 主网络 DisNet (GNN) ─→ 节点位移
              │
              ├─→ 子网络 StrainNet (预训练过) ─→ 单元应变
              │                                          │
              │                                          ├─→ 子网络 StressNet (预训练过) ─→ 单元应力
              │                                          │
              ↓                                          ↓
              └─→ Loss = L_data_u + L_geom + L_const
```

**实现**（本节代码较长，给出框架）：

```python
class PhyFENet_Multilevel(nn.Module):
    """多级网络：主 GNN + 预训练的 StrainNet + StressNet"""
    def __init__(self, main_gnn, strain_net, stress_net):
        super().__init__()
        self.main = main_gnn        # 只输出位移，不直接输出应变应力
        self.strain_net = strain_net
        self.stress_net = stress_net
    
    def forward(self, batch):
        # 主网络输出位移
        disp = self.main(batch)    # (N, 2)
        
        # 子网络从位移 + 坐标算应变
        # 每个图的单元列表在 batch.elem_list 中，需要遍历
        strain = self.strain_net(disp, batch.x[:, :2], batch.elem_list_batched)
        
        # 子网络从应变 + 材料算应力
        material = batch.material_per_elem    # (total_elems, 2)
        stress = self.stress_net(strain, material)
        
        return disp, strain, stress


# 训练
model_ml = PhyFENet_Multilevel(main_gnn, strain_net, stress_net)
# 可选：冻结子网络的部分权重，只微调主网络 + 子网络顶层
optimizer = optim.Adam(model_ml.parameters(), lr=5e-4)

for epoch in range(200):
    for batch in train_loader:
        disp, strain, stress = model_ml(batch)
        
        L_data = ((disp - batch.y) ** 2).mean()
        # 由于 strain 和 stress 是子网络算的，不需要额外的"一致性 loss"
        # 只需要训练让它们逼近真实值
        L_strain_data = ((strain - batch.y_elem_strain) ** 2).mean()
        L_stress_data = ((stress - batch.y_elem_stress) ** 2).mean()
        
        loss = L_data + L_strain_data + L_stress_data
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**对比实验**：
- 策略 A：从零训练 Week 20 的 PhyFENet（作为 baseline）
- 策略 B：子网络预训练 + 联合微调（Week 21 的方案）

记录：收敛速度（到相同 loss 所需 epoch）、最终 MAE

---

### Day 5–6（周末）| 小样本场景下多级网络的优势

**衔接论文**：论文 §3.2.3 重点讨论了"数据少时物理约束的价值"。本周末做这个对比实验。

**实验设计**：
- 训练数据：分别用 3、5、10、15 个样本训练
- 策略 A：从零训练
- 策略 B：多级网络（子网络先预训练）

对每组画"训练样本数 vs 验证 MAE"曲线。

**预期**：在小样本时（3-5 个样本），策略 B 应明显优于策略 A。原因：子网络预训练时看到的单元级数据量是 "图数 × 单元数"，所以即使图样本少，子网络也能先训好。

**验收标准**：
- 能画出两条曲线
- 能分析哪个策略在什么数据量下有优势
- **不要求**某个固定的"性能提升比例"——结果取决于数据和超参

---

### Week 21 完成标准

- [ ] 能展开 GNN 数据集到单元级数据集
- [ ] 能独立预训练 StrainNet 和 StressNet
- [ ] 能构造多级网络把子网络嵌入主网络
- [ ] 能对比"从零训练"和"多级预训练+微调"的收敛速度和最终性能
- [ ] 能做小样本场景实验

---

---

## Week 22: Part D 闭环 + 第一阶段收尾

**本周定位**：Part D 和整个第一阶段的**收尾周**。把 Week 17-21 的所有内容串成一个完整 pipeline，做完整的自测。

---

### Day 1–2 | 完整 Pipeline 串联

**任务**：创建文件 `phase1/full_pipeline.py`

把以下所有阶段串成一个**单入口脚本**：
```
[1] parse_fem_file()   读模拟 FEM 文件 → dict
                       （Week 18）
          ↓
[2] mesh_to_pyg()      dict → PyG Data 对象（支持混合单元）
                       （Week 19）
          ↓
[3] ZScoreNormalizer   归一化节点特征和标签
                       （Part A Week 7）
          ↓
[4] 预训练 StrainNet 和 StressNet
                       （Week 21）
          ↓
[5] PhyFENet_Multilevel 联合训练（含几何 + 本构约束）
                       （Week 20 + Week 21）
          ↓
[6] 验证：预测 → 反归一化 → 评估 → 可视化
```

**目标**：这个脚本是你第一阶段的"最终成果"，面试或展示时直接拿它说："我从零实现了一个 PhyFENet 风格的系统。"

---

### Day 3 | 综合可视化

创建一张"展示级"的可视化图（`phase1_showcase.png`）：
- 2×3 子图布局
- 第一行：节点位移场（真实 ux / 预测 ux / 误差）
- 第二行：单元应力场（真实 σxx / 预测 σxx / 误差）
- 每张图有 colorbar、标题、坐标轴

这张图要能直接放在你未来的简历项目介绍里。

---

### Day 4 | 第一阶段整体自测

**理论题**（书面回答，每题 3-5 句）：

Part A：
1. 为什么神经网络要加激活函数？
2. 训练中出现 NaN 最常见的三个原因是什么？
3. 为什么 FEM 数据训练必须归一化？

Part B：
4. `torch.autograd.grad()` 和 `.backward()` 的区别？
5. 什么时候需要 `create_graph=True`？
6. PINN 的 Loss 由哪几部分组成？每部分约束什么？

Part C：
7. GCN 和 GraphSAGE 的核心差异？
8. 论文 §2.3.3 消息传递的四步是什么？
9. 单元聚合在 FEM-GNN 里的作用？

Part D：
10. 物理约束 Loss 的意义（尤其在小样本场景）？
11. 多级网络（子网络+主网络）的设计动机？
12. 几何方程 Loss 和本构方程 Loss 分别约束什么？

**代码题**（限时完成，不看参考）：
- （30 min）从零写 2 层 MLP + 训练循环（Part A）
- （15 min）从零写一阶+二阶 autograd 导数计算（Part B）
- （20 min）从零写规则网格 → PyG Data（Part C）
- （30 min）从零写自定义 MessagePassing 带边更新的层（Part C）
- （20 min）从零写 PINN 求解 ODE（含 PDE loss + BC loss）（Part B）

---

### Day 5 | 整理第一阶段代码仓库

```
ai-cae-learning/
├── README.md                 ← 第一阶段总结（写你做了什么）
├── utils/
│   ├── fem_parser.py         ← Week 18
│   ├── mesh_to_graph.py      ← Week 19
│   ├── pinn_utils.py         ← Week 17
│   ├── gnn_models.py         ← Part C Week 15
│   ├── sub_networks.py       ← Week 21 (StrainNet, StressNet)
│   └── normalization.py      ← Part A Week 7
├── pretrain/                 ← 补零期
├── phase1/
│   ├── part_a_dl_foundations/    (Week 1-7)
│   ├── part_b_autograd/          (Week 8-10)
│   ├── part_c_gnn/               (Week 11-16)
│   ├── part_d_pinn_fem/          (Week 17-22)
│   └── full_pipeline.py          ← 第一阶段最终 demo
├── notebooks/                ← 如果有探索性分析
└── requirements.txt
```

写 `README.md`：
- 一句话介绍：你在做什么方向
- 你实现了什么（列 Part A-D 的核心模块）
- 关键可视化（贴 Day 3 的 showcase 图）
- 下一步计划：第二阶段做什么（真实 FEM 数据）

---

### Day 6 | 写一篇技术博客（可选但强烈推荐）

**标题建议**：《从零搭建物理信息图神经网络：第一阶段总结》

**内容框架**（800-1500 字）：
1. 背景（AI + CAE 方向是什么，为什么重要）
2. 我的第一阶段做了什么（按 Part A/B/C/D 简述）
3. 最难的几个坑（挑 2-3 个你真实遇到的问题）
4. 关键洞察（物理约束的意义 / 多级网络的设计 / 等）
5. 第二阶段计划

发到：知乎、CSDN、微信公众号、或个人博客（任选）。

**为什么写博客很重要**：
- 强制你把学到的东西讲清楚
- 未来面试的"技术曝光"——面试官会搜你的名字，博客是加分项
- 帮你整理第一阶段的收获

---

### Week 22 + Part D + 第一阶段 完成标准

**Part D**：
- [ ] 能实现 2D PINN（拉普拉斯 + 弹性力学）
- [ ] 能解析模拟 FEM 文件并做数据质量检查
- [ ] 能处理混合单元类型（S3+S4）的不规则网格
- [ ] 能在 GNN Loss 中嵌入几何方程和本构方程约束
- [ ] 能实现子网络预训练 + 主网络联合训练
- [ ] 能做小样本场景的对比实验

**第一阶段总成绩**：
- [ ] 完整的代码仓库（整洁、可复用的 `utils/` + 按 Part 组织的 `phase1/`）
- [ ] 一个完整的 demo 脚本 `full_pipeline.py`（从 FEM 解析到最终预测）
- [ ] 一篇技术博客
- [ ] 理论 + 代码自测全部通过

**如果上述有任何一项未达到**：**不要**进入第二阶段。第二阶段要处理真实 FEM 数据和工业工具（HyperMesh / LS-DYNA / Abaqus），难度和工作量都会跳升。第一阶段的地基不扎实，第二阶段会崩。

---

## 第一阶段全局回顾

```
补零期    P1–P4         Python + NumPy + Matplotlib + 工程项目组织
Part A    Week 1–7      深度学习地基（MLP → PyTorch 完整训练流程）
Part B    Week 8–10     自动微分深入（autograd.grad + 1D PINN）
Part C    Week 11–16    图神经网络（GCN → GraphSAGE → 带边更新 MP → PhyFENet 架构）
Part D    Week 17–22    PINN + FEM + GNN 融合（2D PINN / FEM 解析 / 多级网络）
```

**你的第一阶段最终状态应该是**：
> "我能独立实现一个物理信息图神经网络系统，从合成 FEM 数据读入、图转换、编码器-处理器-解码器 GNN、几何+本构方程约束、多级网络训练，到结果可视化。下一步我需要处理真实 FEM 软件生成的工业数据。"

**下一阶段（第二阶段 Week 23+）会做什么**：
- 真实 FEM 软件的使用（HyperMesh 网格划分 + LS-DYNA/Abaqus 求解器）
- 工业不规则网格的处理
- 大样本数据集构建
- 迁移学习（不同材料 / 不同几何参数）
- 完整的工程项目交付（对标汽车冲压成形场景）

---

