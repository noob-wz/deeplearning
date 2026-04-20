# 第二阶段开始 · Part E: 真实 FEM 数据处理（Week 23–28）

## 第二阶段总体说明

第一阶段你完成了"零件制造"——MLP、GNN、PINN、物理约束 Loss、多级网络，都在合成数据上跑通了。但整个第一阶段有一个根本局限：**所有数据都是你自己编的**。你的 `parse_fem_file` 解析的是仿造的 Abaqus 格式；你的合成位移公式是手算近似，不严格满足物理方程；你的网格是规则矩形或手动造的 13 节点不规则图。

**第二阶段的定位**：把第一阶段的零件装到真实工程数据上，并交付一个对标汽车冲压成形场景的完整项目。

**第二阶段结构**（总约 6 个月，按模块分）：
```
Part E  Week 23-28  （约6周）   真实 FEM 数据处理
        FEniCS 入门 + Gmsh 不规则网格 + 工业数据集构建
        + HyperMesh/LS-DYNA 最小认知模块

Part F  Week 29-33  （约5周）   PhyFENet 在真实数据上的训练
        真实数据归一化策略 + 训练稳定性 + 多样本泛化 + 迁移学习

Part G  Week 34-39  （约6周）   冲压成形完整项目
        对标论文第四章：发动机盖内板式简化项目
        + 评估体系 + 可视化 + 误差分析 + 面试展示准备
```

**与第一阶段的关键差异**：
- 第一阶段的产出以 **"概念理解 + 能跑通"** 为目标
- 第二阶段的产出以 **"工程化交付 + 能展示"** 为目标
- 这意味着代码组织、数据管线、评估体系、可视化都要达到"能放在简历里"的质量

---

---

## Part E 本阶段定位

**衔接 Part D**：你已经能在自己编的合成数据上跑通 PhyFENet 风格的系统。但现在你需要回答一个面试几乎必问的问题——

> "你的方法在真实 FEM 数据上表现怎么样？"

"我的数据是自己用公式编的"是一个极差的回答。第二阶段的第一步就是让你**能独立从一个真实 FEM 求解器生成数据**，然后在这些数据上做训练和评估。

**Part E 要补的四件事**：
1. **FEniCS 入门**：开源 FEM 求解器，Python 接口，能真正解偏微分方程。这是你第一次接触"真正的有限元软件"
2. **Gmsh 生成不规则网格**：Part D 你是手写 13 节点的不规则网格，这里升级到用工具自动生成带孔板、圆弧边界等真实工程几何的网格
3. **工业级数据集构建**：能按参数扫描（不同 E、F、几何尺寸）自动批量生成 FEM 数据，保存到磁盘，做数据质量管控
4. **HyperMesh/LS-DYNA 最小认知模块**：你的目标论文第四章就是用的这两个工具。完全不接触在面试时会露馅；但不需要精通（这不是你这个岗位的核心能力），**重点是能读懂它们的文件格式和工作流程**

**Part E 完成标准（进入 Part F 前）**：
- 能独立安装 FEniCS 并跑通一个 2D 弹性问题的求解
- 能用 Gmsh 生成带孔板的不规则三角/四边形网格
- 能写一个脚本自动批量生成 50-100 个样本的 FEM 数据集
- 能把 FEniCS 输出转成 PyG `Data` 对象（对接 `utils/mesh_to_graph.py`）
- 能读懂 Abaqus `.inp` 文件和 LS-DYNA `.k` 文件的基本结构
- 能在面试时清楚讲："我用 FEniCS 做生成管线；HyperMesh/LS-DYNA 我能读懂格式但没有深度使用经验"

**时间预期**：**6-9 周，基线 6 周**。

不要低估 FEniCS 的安装难度——很多新手在这里就卡 1-2 周。FEniCS 的官方推荐是 Docker 或 conda，但 conda 环境的依赖冲突是常见问题。**本阶段预留足够的容错时间**。

**本阶段有一件事要反复提醒自己**：你不是在学成为 CAE 工程师，你是在学"能读懂 CAE 数据的 ML 工程师"。所以：
- **不要**深入研究网格质量指标的工业阈值（Jacobian、Aspect Ratio、Warpage 的具体数值）
- **不要**尝试精通 HyperMesh 的菜单操作
- **不要**把 LS-DYNA 的非线性求解原理当主线学
- **要**做到的是：能读懂数据、能生成数据、能解释数据结构

---

---

## Week 23: FEniCS 安装与第一个弹性问题

**本周定位**：FEniCS 入门。这周最现实的风险是**安装本身**——不是代码写得怎么样。如果你的第一周有大半时间在装环境，这是正常的，不要焦虑。

**本周目标**：
- 成功安装 FEniCS（dolfinx）并通过验证测试
- 用 FEniCS 求解一个最简单的 2D 弹性问题（矩形板拉伸）
- 从 FEniCS 提取节点坐标、单元连接、位移场、应力场，存成 NumPy 数组

**主要资源**：
- FEniCS 官方教程 "FEniCSx tutorial"（https://jsdokken.com/dolfinx-tutorial/），只看"Linear elasticity"这一节
- FEniCS 官方安装文档：https://docs.fenicsproject.org/dolfinx/v0.7.3/python/installation.html

---

### Day 1–2 | 环境安装

**选方案**：FEniCS 有三种安装方式，按优先级推荐：

**方案 A（推荐，最稳）：Docker**
```bash
docker pull dolfinx/dolfinx:stable
docker run -ti -v $(pwd):/root/shared -w /root/shared dolfinx/dolfinx:stable
```
优点：环境隔离，不会污染本机；能跨平台运行。
缺点：需要先学 Docker 基本操作（`docker run`, `docker exec`, `-v` volume mount）。

**方案 B：Conda**
```bash
conda create -n fenicsx-env python=3.11
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```
优点：在本机运行，调试直接。
缺点：依赖冲突较常见，可能要多次尝试。

**方案 C（降级方案）：如果 A 和 B 都失败**
使用 `meshio` 读取预先生成的 `.msh` 文件，**跳过 FEniCS 求解步骤**。这不是理想方案，但不至于让你卡在安装上。

---

**任务**（Day 1-2 的目标是通过以下验证代码）：

创建文件 `week23/day12_fenicsx_install_test.py`

```python
"""
FEniCS 安装验证。
如果这个脚本无报错运行并打印了 'All imports OK'，说明安装成功。
"""
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl

print(f"dolfinx version: {dolfinx.__version__}")
print("All imports OK")

# 最小测试：创建一个 4x4 的矩形网格
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
    [4, 4],
    cell_type=mesh.CellType.triangle
)
print(f"Mesh created: {domain.topology.index_map(0).size_local} nodes, "
      f"{domain.topology.index_map(2).size_local} cells")
```

**验收标准**：脚本运行无报错，打印出 dolfinx 版本号和网格信息（应该约 25 节点、32 单元）。

**如果装不上怎么办**：
- 记录完整报错信息
- 不要硬扛超过 3 天——如果 3 天还装不上，先用降级方案（方案 C + 预生成的 `.msh` 文件），Part E 剩余内容继续，等 Part F 开始前再回来解决 FEniCS
- 搜索："dolfinx install" + 你的操作系统 + 具体报错

---

### Day 3–4 | 求解一个 2D 弹性问题

**物理问题**：和 Part D Week 17 的选修题目一样——矩形板左端固定，右端拉伸。但这次是**真正的有限元求解**，不是 PINN 近似。

创建文件 `week23/day34_elasticity_fenicsx.py`：

```python
"""
用 FEniCS 求解 2D 平面应力线弹性问题：
- 矩形板 Ω = [0, 2] × [0, 1]
- 左端固定：u = v = 0 at x=0
- 右端拉力：施加 T_x = 50 MPa 的牵引力
- 材料：E = 210 GPa, nu = 0.3
"""
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

# ===== 1. 几何与网格 =====
L, H = 2.0, 1.0
nx, ny = 20, 10
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, H])],
    [nx, ny],
    cell_type=mesh.CellType.triangle
)

# ===== 2. 材料参数 =====
E = 210e3       # MPa
nu = 0.3
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
# 平面应力修正
lmbda_ps = 2 * mu * lmbda / (lmbda + 2 * mu)

# ===== 3. 函数空间（向量场：ux, uy）=====
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

# ===== 4. 边界条件:左端固定 =====
def left_boundary(x):
    return np.isclose(x[0], 0.0)

left_dofs = fem.locate_dofs_geometrical(V, left_boundary)
u_fixed = np.array([0.0, 0.0], dtype=default_scalar_type)
bc_left = fem.dirichletbc(u_fixed, left_dofs, V)

# ===== 5. 右端面力 =====
def right_boundary(x):
    return np.isclose(x[0], L)

facet_dim = domain.topology.dim - 1
right_facets = mesh.locate_entities_boundary(domain, facet_dim, right_boundary)
facet_tags = mesh.meshtags(
    domain, facet_dim, right_facets,
    np.full_like(right_facets, 1)
)

# ===== 6. 变分形式 =====
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda_ps * ufl.tr(epsilon(u)) * ufl.Identity(2) + 2 * mu * epsilon(u)

T = fem.Constant(domain, default_scalar_type((50.0, 0.0)))   # 牵引力
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L_form = ufl.dot(T, v) * ds(1)

# ===== 7. 求解 =====
problem = LinearProblem(a, L_form, bcs=[bc_left])
uh = problem.solve()

# ===== 8. 提取数据为 NumPy =====
node_coords = domain.geometry.x    # shape=(N, 3)
# 读位移
u_values = uh.x.array.reshape(-1, 2)   # shape=(N, 2)，每行(ux, uy)

# 读连接
connectivity = domain.topology.connectivity(2, 0)
elements = []
for i in range(domain.topology.index_map(2).size_local):
    elements.append(connectivity.links(i).tolist())
elements = np.array(elements)   # shape=(n_elem, 3) for triangles

print(f"节点数: {len(node_coords)}")
print(f"单元数: {len(elements)}")
print(f"位移范围: ux in [{u_values[:, 0].min():.4f}, {u_values[:, 0].max():.4f}]")
print(f"          uy in [{u_values[:, 1].min():.4f}, {u_values[:, 1].max():.4f}]")

# ===== 9. 保存为 .npz =====
np.savez(
    'week23_fem_result.npz',
    nodes=node_coords[:, :2],     # 只保留 x, y（z=0 for 2D）
    elements=elements,
    u=u_values,
    E=E, nu=nu, T_x=50.0
)
print("已保存到 week23_fem_result.npz")
```

**验收标准**：
- 脚本能跑通
- 位移趋势合理：ux 从左端 0 渐增到右端正值；uy 由于泊松效应在中轴线附近小、上下边界附近大
- 节点数约 (nx+1)×(ny+1) = 231
- 单元数是三角形，约为 nx×ny×2 = 400

---

### Day 5 | 可视化 FEniCS 结果

创建文件 `week23/day5_visualize.py`：

```python
"""
读取 Day 3-4 保存的 .npz 文件，用 Matplotlib 可视化位移场。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

data = np.load('week23_fem_result.npz')
nodes = data['nodes']       # (N, 2)
elements = data['elements']  # (n_elem, 3)
u = data['u']                # (N, 2)

triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, disp_col, title in zip(axes, [u[:, 0], u[:, 1]], ['ux', 'uy']):
    tcf = ax.tricontourf(triang, disp_col, levels=30, cmap='jet')
    ax.triplot(triang, 'k-', lw=0.3, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title(f'Displacement {title}')
    plt.colorbar(tcf, ax=ax)

plt.tight_layout()
plt.savefig('week23_fem_viz.png', dpi=100)
print("图已保存到 week23_fem_viz.png")
```

**验收标准**：
- 生成的图 ux 图呈从左到右的平滑梯度
- uy 图呈上下对称分布
- 两张图都能看到网格线（alpha=0.3 的三角剖分）

---

### Day 6 | Week 23 总结与反思

**写一段 200-300 字的总结**，保存为 `week23/reflection.md`，回答：
1. FEniCS 安装过程中遇到了什么坑？最后怎么解决的？（为未来自己留笔记）
2. 对比 Part D Week 17 的 PINN 2D 弹性结果和本周的 FEniCS 有限元结果，两者在**位移场的分布趋势**上是否相似？
3. 哪个更"真实"？为什么？（参考答案：FEniCS 的结果是严格有限元求解，满足所有方程；PINN 是神经网络近似，依赖训练好坏）

---

### Week 23 完成标准

- [ ] FEniCS 能正常 import（或至少通过降级方案能处理 .msh 文件）
- [ ] 能用 FEniCS 求解矩形板拉伸问题并提取节点/单元/位移数据
- [ ] 能可视化 FEniCS 的结果（ux 和 uy 的三角剖分颜色图）
- [ ] 能对比 PINN（Part D）和 FEM（本周）两种方法的输出

---

---

## Week 24: Gmsh 生成不规则工程网格

**衔接**：Week 23 用的是 FEniCS 内置的 `create_rectangle`，生成规则三角网格。但真实工程几何（带孔板、圆弧、异形）需要专门的网格划分工具——这就是 **Gmsh**。

**本周目标**：
- 用 Gmsh 生成一个带圆孔的矩形板网格
- 把 Gmsh 输出的 `.msh` 文件导入到 FEniCS 求解
- 把求解结果存成数据集用的标准格式

**关于 Gmsh 的说明**：Gmsh 既能独立运行，也能用 Python API 调用。本周主要用 Python API——这样能把网格生成纳入自动化管线。

---

### Day 1 | Gmsh 入门

**任务**：安装 Gmsh Python API（应该已经在 Week 23 装 `meshio` 时一并装了，但确认一下）。

创建文件 `week24/day01_gmsh_basics.py`：

```python
"""
用 Gmsh Python API 生成带圆孔的矩形板网格
"""
import gmsh
import numpy as np

gmsh.initialize()
gmsh.model.add("plate_with_hole")

# ===== 定义几何 =====
L, H = 2.0, 1.0       # 板的长宽
r = 0.2               # 孔半径
cx, cy = 1.0, 0.5     # 孔心坐标

# 用 OCC 内核（更适合复杂几何）
rect_tag = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
hole_tag = gmsh.model.occ.addDisk(cx, cy, 0, r, r)

# 布尔减：矩形减去圆
result, _ = gmsh.model.occ.cut([(2, rect_tag)], [(2, hole_tag)])
gmsh.model.occ.synchronize()

# ===== 网格划分参数 =====
# 全局网格尺寸
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)

# 在孔边缘局部加密（物理上应力集中处网格应该更密）
# 方法：用距离场 + 阈值场
hole_curves = gmsh.model.getBoundary(result, recursive=True, oriented=False)
hole_curve_tags = [abs(c[1]) for c in hole_curves if c[0] == 1]
# 实际中只需要孔的边界。这里简化处理——在实际应用时需要更精细地识别。

# ===== 生成 2D 网格 =====
gmsh.model.mesh.generate(2)

# ===== 保存 =====
gmsh.write("plate_hole.msh")
print("已生成网格文件: plate_hole.msh")

# 打印统计信息
n_nodes = len(gmsh.model.mesh.getNodes()[0])
print(f"节点总数: {n_nodes}")

gmsh.finalize()
```

**验收标准**：
- `plate_hole.msh` 文件生成
- 节点数约 300-500（取决于网格尺寸设置）
- 不报错

---

### Day 2 | 用 meshio 读取 .msh 文件并可视化

创建文件 `week24/day02_read_msh.py`：

```python
"""
用 meshio 读取 Gmsh 输出的 .msh 文件，提取节点和单元，可视化。
"""
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# 读取
msh = meshio.read("plate_hole.msh")

# 提取节点和单元
nodes_3d = msh.points   # shape=(N, 3)
nodes = nodes_3d[:, :2]  # 2D 问题只要 x, y

# Gmsh 2D 默认输出三角形单元
triangle_cells = msh.cells_dict.get('triangle', None)
if triangle_cells is None:
    raise ValueError("未找到三角形单元")

print(f"节点数: {len(nodes)}")
print(f"三角形单元数: {len(triangle_cells)}")
print(f"坐标范围: x in [{nodes[:, 0].min():.3f}, {nodes[:, 0].max():.3f}]")
print(f"          y in [{nodes[:, 1].min():.3f}, {nodes[:, 1].max():.3f}]")

# 可视化
triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangle_cells)
fig, ax = plt.subplots(figsize=(10, 5))
ax.triplot(triang, 'k-', lw=0.3)
ax.set_aspect('equal')
ax.set_title('Plate with Hole Mesh (Gmsh)')
plt.savefig('week24_mesh.png', dpi=120)
print("已保存 week24_mesh.png")
```

**验收标准**：
- 网格图能看到矩形外轮廓和中心的圆孔
- 圆孔附近网格密度应略高于远处（如果你设置了局部加密）
- 网格三角形形状合理（没有极端扁长的三角形）

---

### Day 3 | 在 Gmsh 网格上用 FEniCS 求解

**重要技术点**：FEniCS 0.7+ 可以直接从 `.msh` 文件读取网格。方法是用 `dolfinx.io.gmshio` 模块。

创建文件 `week24/day03_fenicsx_on_gmsh.py`：

```python
"""
用 FEniCS 在 Gmsh 生成的带孔板网格上求解弹性问题。
"""
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
import ufl

# ===== 1. 读 Gmsh 网格 =====
domain, cell_tags, facet_tags = gmshio.read_from_msh(
    "plate_hole.msh", MPI.COMM_WORLD, gdim=2
)
print(f"从 Gmsh 读入: {domain.topology.index_map(0).size_local} 节点")

# ===== 2. 材料 + 函数空间（同 Week 23）=====
E, nu = 210e3, 0.3
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
lmbda_ps = 2 * mu * lmbda / (lmbda + 2 * mu)

V = fem.functionspace(domain, ("Lagrange", 1, (2,)))

# ===== 3. 边界条件（需要自己定位）=====
# 左端固定
def left_boundary(x):
    return np.isclose(x[0], 0.0, atol=1e-4)

left_dofs = fem.locate_dofs_geometrical(V, left_boundary)
bc_left = fem.dirichletbc(
    np.array([0.0, 0.0], dtype=default_scalar_type),
    left_dofs, V
)

# 右端面力
def right_boundary(x):
    return np.isclose(x[0], 2.0, atol=1e-4)

facet_dim = domain.topology.dim - 1
right_facets = mesh.locate_entities_boundary(domain, facet_dim, right_boundary)
facet_tag_data = mesh.meshtags(
    domain, facet_dim, right_facets,
    np.full_like(right_facets, 1)
)

# ===== 4. 变分形式（同 Week 23）=====
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda_ps * ufl.tr(epsilon(u)) * ufl.Identity(2) + 2 * mu * epsilon(u)

T = fem.Constant(domain, default_scalar_type((50.0, 0.0)))
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag_data)

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L_form = ufl.dot(T, v) * ds(1)

# ===== 5. 求解 =====
problem = LinearProblem(a, L_form, bcs=[bc_left])
uh = problem.solve()

# ===== 6. 提取数据 =====
node_coords = domain.geometry.x[:, :2]
u_values = uh.x.array.reshape(-1, 2)
connectivity = domain.topology.connectivity(2, 0)
elements = np.array([connectivity.links(i).tolist() 
                    for i in range(domain.topology.index_map(2).size_local)])

np.savez(
    'week24_plate_hole.npz',
    nodes=node_coords, elements=elements, u=u_values,
    E=E, nu=nu, T_x=50.0
)
print(f"求解完成，位移最大值: {np.abs(u_values).max():.6f}")
```

**验收标准**：
- 能成功读取并求解
- 位移场 `ux` 在**孔的边缘附近**应该比板远处略大（应力集中区域变形也相对大）
- 保存的 `.npz` 文件能被 Week 23 的可视化脚本读取

---

### Day 4 | 可视化带孔板的位移和应力场

创建文件 `week24/day04_visualize_plate_hole.py`。基于 Week 23 Day 5 的可视化代码，但增加**应力场可视化**——这对下周的数据集构建很重要。

**应力场怎么算**：FEniCS 的位移是定义在节点上的（连续场），应力是从位移推导出来的（通过本构方程）。在"节点级应力"上可以用插值得到近似值。

核心代码片段：
```python
# 除了位移，再算一下 von Mises 应力
from dolfinx.fem import Expression

sigma_vm = ufl.sqrt(
    ufl.inner(sigma(uh), sigma(uh)) 
    - (ufl.tr(sigma(uh))) ** 2 / 3
)
# 插值到节点上
V_scalar = fem.functionspace(domain, ("Lagrange", 1))
sigma_vm_func = fem.Function(V_scalar)
sigma_vm_expr = Expression(sigma_vm, V_scalar.element.interpolation_points())
sigma_vm_func.interpolate(sigma_vm_expr)
sigma_vm_array = sigma_vm_func.x.array
```

然后可视化 `ux`, `uy`, `σ_vm` 三个场（三子图）。

**验收标准**：
- von Mises 应力图中，孔的边缘应该有**明显的应力集中**（颜色最亮的区域集中在孔边）
- 这是带孔板的典型现象，如果没看到，说明求解有问题

---

### Day 5–6（周末）| 把 Gmsh 几何参数化

**本周的最终目标**：写一个函数，接受 `(L, H, r)` 三个参数，返回生成的网格（准备给下周批量生成数据集用）。

创建文件 `utils/gmsh_utils.py`：

```python
"""
Gmsh 参数化几何生成工具。
"""
import gmsh
import numpy as np
import meshio

def generate_plate_with_hole(L, H, r, mesh_size_max=0.1, 
                              mesh_size_min=0.02,
                              output_file="temp_plate.msh"):
    """
    参数化生成带圆孔矩形板网格。
    
    参数：
        L, H: 板的长和宽
        r: 圆孔半径（孔心默认在板中心 L/2, H/2）
        mesh_size_max, mesh_size_min: 全局和局部网格尺寸
        output_file: 输出 .msh 文件路径
    
    返回：
        nodes: np.ndarray shape=(N, 2)
        elements: np.ndarray shape=(n_elem, 3) 三角形单元
    """
    gmsh.initialize()
    gmsh.model.add("plate_with_hole")
    
    cx, cy = L / 2, H / 2
    rect_tag = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
    hole_tag = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
    gmsh.model.occ.cut([(2, rect_tag)], [(2, hole_tag)])
    gmsh.model.occ.synchronize()
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_max)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_min)
    
    gmsh.model.mesh.generate(2)
    gmsh.write(output_file)
    gmsh.finalize()
    
    # 读取并返回
    msh = meshio.read(output_file)
    nodes = msh.points[:, :2]
    triangles = msh.cells_dict['triangle']
    
    return nodes, triangles


if __name__ == '__main__':
    # 测试：生成不同参数的网格
    for L, H, r in [(2.0, 1.0, 0.2), (2.5, 1.0, 0.15), (2.0, 1.5, 0.3)]:
        nodes, elems = generate_plate_with_hole(L, H, r)
        print(f"L={L}, H={H}, r={r}: {len(nodes)} 节点, {len(elems)} 单元")
```

**验收标准**：
- 函数能接受不同参数生成不同网格
- 调用三次测试都能生成，节点数随参数变化合理（r 大 → 节点少；mesh_size 小 → 节点多）

---

### Week 24 完成标准

- [ ] 能用 Gmsh Python API 生成带孔板网格
- [ ] 能用 FEniCS 在 Gmsh 生成的网格上求解
- [ ] 能可视化位移场和 von Mises 应力场
- [ ] 有一个参数化的网格生成工具函数 `generate_plate_with_hole` 放在 `utils/gmsh_utils.py`

---

---

## Week 25: FEniCS 结果 → PyG Data（升级版 mesh_to_pyg）

**衔接**：Part D Week 19 你写过 `mesh_to_pyg` 函数，但当时处理的是手动造的 13 节点小网格。本周要升级它——能处理 Gmsh 生成的几百节点真实网格，并**集成材料参数和边界条件信息**作为节点特征。

**本周目标**：
- 把 FEniCS 的 `.npz` 结果转成 PyG `Data` 对象
- 把材料参数、边界条件作为节点特征的一部分
- 数据质量检查升级到"工程级"（不仅是 NaN 和孤立节点，还包括物理合理性）

---

### Day 1–2 | 升级 `mesh_to_pyg` 对接 FEniCS 输出

**关键设计问题**：节点特征应该包含什么？

回顾 Part D Week 19，节点特征是 `[x, y, z, E, F]`。但真实工程中：
- 材料参数（E, nu）通常是全局的（所有节点相同）
- 边界条件是局部的（只有部分节点固定，部分节点受力）
- 几何参数（如孔半径 r）是全局的

所以合理的节点特征设计：
```
每个节点的特征（8 维）：
  [x, y, z,            # 坐标（3维）
   E, nu,              # 材料（全局，每个节点拷贝一份）
   is_fixed,           # 是否在固定边界（0 或 1）
   is_loaded,          # 是否在受力边界（0 或 1）  
   load_magnitude]     # 载荷大小（仅在受力节点非零）
```

这样的设计让 GNN 能通过节点特征"看到"边界条件信息。

创建文件 `week25/day12_fenicsx_to_pyg.py`：

```python
"""
把 FEniCS 的 .npz 输出转换成 PyG Data 对象（升级版）。
"""
import numpy as np
import torch
from torch_geometric.data import Data

def identify_boundary_nodes(nodes, L, H, tol=1e-4):
    """
    根据节点坐标判断哪些节点在固定边界（x=0），哪些在受力边界（x=L）
    
    参数：
        nodes: np.ndarray shape=(N, 2) 或 (N, 3)
        L, H: 板的长宽
    返回：
        is_fixed: bool array shape=(N,)
        is_loaded: bool array shape=(N,)
    """
    is_fixed = np.abs(nodes[:, 0] - 0.0) < tol
    is_loaded = np.abs(nodes[:, 0] - L) < tol
    return is_fixed, is_loaded


def build_node_features(nodes, E, nu, L, H, load_magnitude):
    """
    构造节点特征矩阵（8 维）。
    
    返回：
        x: np.ndarray shape=(N, 8)
    """
    N = len(nodes)
    is_fixed, is_loaded = identify_boundary_nodes(nodes, L, H)
    
    # 如果节点是 2D，补一个 z=0 列
    if nodes.shape[1] == 2:
        coords = np.column_stack([nodes, np.zeros(N)])
    else:
        coords = nodes
    
    features = np.column_stack([
        coords,                                    # 3
        np.full(N, E),                             # 1
        np.full(N, nu),                            # 1
        is_fixed.astype(np.float32),               # 1
        is_loaded.astype(np.float32),              # 1
        np.where(is_loaded, load_magnitude, 0.0),  # 1
    ])
    return features


def fenicsx_result_to_pyg(npz_path, L, H):
    """
    从 .npz 文件读取 FEniCS 结果，转成 PyG Data 对象。
    
    参数：
        npz_path: .npz 文件路径
        L, H: 几何参数（用于识别边界）
    返回：
        data: PyG Data 对象
    """
    result = np.load(npz_path)
    nodes = result['nodes']        # (N, 2)
    elements = result['elements']   # (n_elem, 3) 三角形
    u = result['u']                 # (N, 2) 位移
    E = float(result['E'])
    nu = float(result['nu'])
    T_x = float(result['T_x'])
    
    # 构造节点特征
    x_feat = build_node_features(nodes, E, nu, L, H, T_x)
    
    # 构造边（用 Part D 的 build_edges_from_elements，但适配 np.ndarray 格式）
    elements_list = []
    for i, elem_nodes in enumerate(elements):
        elements_list.append({
            'id': i + 1,
            'type': 'Tri3',  # 三角形 3 节点
            'nodes': elem_nodes.tolist()
        })
    
    # 调 Part D Week 19 的 mesh_to_pyg（从 utils 导入）
    from utils.mesh_to_graph import mesh_to_pyg
    
    nodes_dict = {i: np.concatenate([nodes[i], [0.0]]) for i in range(len(nodes))}
    node_features_dict = {i: x_feat[i, 3:] for i in range(len(nodes))}   # 特征的后 5 列
    node_labels_dict = {i: u[i] for i in range(len(nodes))}
    
    # 注意 Part D Week 19 的 mesh_to_pyg 用的是 1-based id，我们这里是 0-based
    # 要对接：可以保持 0-based 或者全部 +1
    # 简化：这里我们直接构造 PyG Data，不绕道 utils
    
    N = len(nodes)
    edge_set = set()
    for elem_nodes in elements:
        n = len(elem_nodes)
        for i in range(n):
            for j in range(i+1, n):
                a, b = int(elem_nodes[i]), int(elem_nodes[j])
                edge_set.add((a, b))
                edge_set.add((b, a))
    
    src, dst = zip(*edge_set) if edge_set else ([], [])
    edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    
    # 边特征（dx, dy, dist）
    diff = x_feat[list(dst), :2] - x_feat[list(src), :2]
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    edge_attr = np.concatenate([diff, dist], axis=1)
    
    # 单元列表（供后续单元聚合用）
    elem_list = []
    for i, elem_nodes in enumerate(elements):
        elem_list.append({
            'id': i,
            'type': 'Tri3',
            'indices': elem_nodes.tolist()
        })
    
    data = Data(
        x=torch.from_numpy(x_feat).float(),
        edge_index=edge_index,
        edge_attr=torch.from_numpy(edge_attr).float(),
        y=torch.from_numpy(u).float()
    )
    data.elem_list = elem_list
    data.E = E
    data.nu = nu
    data.T_x = T_x
    
    return data


if __name__ == '__main__':
    data = fenicsx_result_to_pyg('week24_plate_hole.npz', L=2.0, H=1.0)
    print(f"节点数: {data.num_nodes}")
    print(f"边数: {data.num_edges}")
    print(f"节点特征 shape: {data.x.shape}")    # 应为 (N, 8)
    print(f"标签 shape: {data.y.shape}")       # 应为 (N, 2)
    print(f"单元数: {len(data.elem_list)}")
```

**验收标准**：
- 能读取 Week 24 的 `.npz` 文件并转成 PyG Data
- 节点特征 shape 为 `(N, 8)`
- 节点特征中 `is_fixed` 的总数 = 左边界节点数；`is_loaded` 的总数 = 右边界节点数

---

### Day 3 | 工程级数据质量检查

创建文件 `utils/fem_quality_check.py`（扩展 Part D Week 18 的基础版本）：

```python
"""
FEM 数据质量检查（工程级）。
"""
import numpy as np
import torch

def check_fem_data_engineering(data, L=None, H=None, verbose=True):
    """
    对一个 PyG Data 对象做多层次的质量检查。
    
    返回：
        issues: list of str，发现的问题
        is_valid: bool，是否通过
    """
    issues = []
    
    # ===== 1. 基础检查（NaN、inf）=====
    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        issues.append("节点特征中含 NaN/Inf")
    if torch.isnan(data.y).any() or torch.isinf(data.y).any():
        issues.append("标签中含 NaN/Inf")
    if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
        issues.append("边特征中含 NaN/Inf")
    
    # ===== 2. 拓扑检查 =====
    # 孤立节点
    unique_nodes_in_edges = torch.unique(data.edge_index)
    if len(unique_nodes_in_edges) < data.num_nodes:
        n_isolated = data.num_nodes - len(unique_nodes_in_edges)
        issues.append(f"发现 {n_isolated} 个孤立节点")
    
    # 边的对称性（对无向图）
    edges = set([(data.edge_index[0, i].item(), data.edge_index[1, i].item()) 
                for i in range(data.edge_index.shape[1])])
    asymmetric = [(a, b) for a, b in edges if (b, a) not in edges]
    if asymmetric:
        issues.append(f"发现 {len(asymmetric)} 条非对称边（应该是双向的）")
    
    # ===== 3. 物理合理性检查 =====
    # 位移量级：对于弹性问题，位移应该 << 几何尺寸
    max_disp = torch.abs(data.y).max().item()
    if L is not None:
        if max_disp > L * 0.5:
            issues.append(f"位移量级异常大: max={max_disp:.4f}, L={L}")
    
    # 边界节点的位移应该接近 0（如果是固定边界）
    if data.x.shape[1] >= 8:
        is_fixed = data.x[:, 5] > 0.5
        if is_fixed.any():
            fixed_disp_max = torch.abs(data.y[is_fixed]).max().item()
            if fixed_disp_max > 1e-3:
                issues.append(f"固定边界节点位移不接近0: max={fixed_disp_max:.6f}")
    
    # ===== 4. 特征量级检查 =====
    # 坐标、材料、载荷的量级应该合理
    x_coords = data.x[:, :3]
    if torch.abs(x_coords).max() > 1e6:
        issues.append(f"坐标量级异常: max={torch.abs(x_coords).max():.2f}")
    
    if verbose:
        if issues:
            print("数据质量问题:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"数据质量 OK: {data.num_nodes} 节点, {data.num_edges} 边")
    
    return issues, len(issues) == 0


if __name__ == '__main__':
    from week25.day12_fenicsx_to_pyg import fenicsx_result_to_pyg
    data = fenicsx_result_to_pyg('week24_plate_hole.npz', L=2.0, H=1.0)
    issues, is_valid = check_fem_data_engineering(data, L=2.0, H=1.0)
```

**验收标准**：
- 质量检查函数能运行
- Week 24 的数据应能通过所有检查（is_valid=True）
- 能故意把某个数据改坏（如把一个节点的 y 位移改成 NaN），看函数能不能检测到

---

### Day 4–6 | 数据归一化策略升级

**关键问题**：Part A Week 7 的 `ZScoreNormalizer` 只能处理"扁平样本数据"（shape=(N, features)）。对于图数据集，每个图的节点数不同，归一化要重新设计。

**正确的归一化策略**（论文 §2.3.1 也提到）：
- 对**整个数据集**的所有节点特征计算 mean 和 std（而不是每个图单独算）
- 标签也用整个数据集的全局统计量
- 保存这些统计量以便反归一化

创建文件 `utils/graph_normalization.py`：

```python
"""
图数据集的归一化。
"""
import torch
import numpy as np

class GraphZScoreNormalizer:
    """
    对图数据集做全局 Z-Score 归一化。
    
    使用方式：
        norm = GraphZScoreNormalizer()
        norm.fit(train_dataset)           # 在训练集上计算统计量
        train_ds_norm = norm.transform(train_dataset)
        val_ds_norm = norm.transform(val_dataset)   # 用训练集的统计量
        
        y_pred_norm = model(data)
        y_pred_real = norm.inverse_transform_y(y_pred_norm)
    """
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
    
    def fit(self, dataset):
        """
        dataset: list of PyG Data 对象
        """
        all_x = torch.cat([data.x for data in dataset], dim=0)
        all_y = torch.cat([data.y for data in dataset], dim=0)
        
        self.x_mean = all_x.mean(dim=0, keepdim=True)
        self.x_std = all_x.std(dim=0, keepdim=True) + 1e-8
        self.y_mean = all_y.mean(dim=0, keepdim=True)
        self.y_std = all_y.std(dim=0, keepdim=True) + 1e-8
        
        print(f"节点特征均值: {self.x_mean.flatten().tolist()}")
        print(f"节点特征标准差: {self.x_std.flatten().tolist()}")
        print(f"标签均值: {self.y_mean.flatten().tolist()}")
        print(f"标签标准差: {self.y_std.flatten().tolist()}")
    
    def transform(self, dataset):
        """
        返回一个新的数据集（不修改原始）。
        """
        import copy
        new_dataset = []
        for data in dataset:
            new_data = copy.copy(data)
            new_data.x = (data.x - self.x_mean) / self.x_std
            new_data.y = (data.y - self.y_mean) / self.y_std
            new_dataset.append(new_data)
        return new_dataset
    
    def inverse_transform_y(self, y_normalized):
        """
        把归一化的预测转回原始尺度（通常用于推理）。
        """
        return y_normalized * self.y_std + self.y_mean
    
    def save(self, path):
        torch.save({
            'x_mean': self.x_mean, 'x_std': self.x_std,
            'y_mean': self.y_mean, 'y_std': self.y_std
        }, path)
    
    def load(self, path):
        stats = torch.load(path)
        self.x_mean, self.x_std = stats['x_mean'], stats['x_std']
        self.y_mean, self.y_std = stats['y_mean'], stats['y_std']
```

**注意事项**：
- **is_fixed 和 is_loaded 这两个 0/1 特征不应该做 Z-Score 归一化**——归一化后会变成奇怪的小数，失去"0 或 1"的语义
- 实际中可以对这两列做**跳过**（保留原值），但代码里 Z-Score 会对所有列处理
- 简单的处理：归一化后，0 变成 `(0 - mean) / std`，1 变成 `(1 - mean) / std`——数字变了但相对关系没变，GNN 仍然能区分两种情况，所以**对训练不造成实质影响**
- 更精细的做法：单独写一个 Normalizer 只处理前 5 列（坐标+材料+载荷），后 2 列保留；这个可以作为**选修**

**验收标准**：
- 归一化函数能正常 fit、transform、inverse_transform
- 对一个测试数据集，归一化后节点特征的均值接近 0、标准差接近 1（对连续列）
- 能保存和加载统计量

---

### Week 25 完成标准

- [ ] `fenicsx_result_to_pyg()` 函数能工作，把 .npz 转成 PyG Data
- [ ] 节点特征包含坐标、材料、边界条件（8 维）
- [ ] `check_fem_data_engineering` 工程级质量检查函数放在 `utils/`
- [ ] `GraphZScoreNormalizer` 图数据集归一化类放在 `utils/`
- [ ] 能对 Week 24 的单个数据做完整的"FEniCS → PyG Data → 质量检查 → 归一化"流程

---

---

## Week 26: 批量 FEM 数据集构建（工业级自动化管线）

**衔接**：Week 23-25 你能求解单个 FEM 问题并转成 PyG 格式。但训练 GNN 需要多样本——论文第三章带孔板问题用了几十个样本。本周写一个自动化管线：给定参数空间，批量生成数据集。

**本周目标**：
- 写一个参数扫描脚本，生成 50-100 个 FEM 样本
- 每个样本对应不同的 E、T（拉力）、r（孔半径）组合
- 把所有样本保存成一个数据集文件
- 能随时加载并做质量检查

---

### Day 1 | 参数扫描策略

**理论任务**（约 30 分钟）：

**参数扫描的两种方式**：
- **网格扫描（Grid Search）**：每个参数取几个离散值，做笛卡尔积。比如 E 取 3 个值、T 取 3 个值、r 取 3 个值，共 27 个样本。简单直观但样本分布僵硬。
- **拉丁超立方采样（Latin Hypercube Sampling）**：在参数空间里均匀随机采样。相同样本数下覆盖更好，工程中更常用。论文第四章数据集就是这类采样。

**本周选择**：拉丁超立方。参数：
- `E`：弹性模量，范围 [150e3, 250e3] MPa
- `T_x`：x 方向拉力，范围 [20, 80] MPa
- `r`：孔半径，范围 [0.1, 0.3]
- `L, H`：固定为 2.0, 1.0（几何变化会让网格完全不同，先固定）

用 `scipy.stats.qmc.LatinHypercube` 做采样。

创建文件 `week26/day01_lhs_sampling.py`：

```python
"""
用拉丁超立方采样生成参数组合。
"""
import numpy as np
from scipy.stats import qmc

def sample_parameters(n_samples, seed=42):
    """
    在 (E, T_x, r) 参数空间做拉丁超立方采样。
    
    返回：
        params: list of dict，每个 dict 是一组参数
    """
    # 参数范围
    bounds = {
        'E':   (150e3, 250e3),    # MPa
        'T_x': (20.0, 80.0),       # MPa
        'r':   (0.1, 0.3)          # 孔半径
    }
    
    # LHS 采样（归一化到 [0, 1]^3）
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    samples = sampler.random(n=n_samples)
    
    # 把归一化的采样映射到实际参数范围
    params = []
    for i in range(n_samples):
        p = {
            'E':   bounds['E'][0]   + samples[i, 0] * (bounds['E'][1]   - bounds['E'][0]),
            'T_x': bounds['T_x'][0] + samples[i, 1] * (bounds['T_x'][1] - bounds['T_x'][0]),
            'r':   bounds['r'][0]   + samples[i, 2] * (bounds['r'][1]   - bounds['r'][0]),
        }
        params.append(p)
    
    return params


if __name__ == '__main__':
    params = sample_parameters(n_samples=50)
    for i, p in enumerate(params[:5]):
        print(f"样本 {i}: E={p['E']:.1f}, T_x={p['T_x']:.2f}, r={p['r']:.3f}")
```

**验收标准**：
- 50 个参数组合被生成
- 参数值在给定范围内
- 前几个样本参数不是简单规则的递增（LHS 的随机性）

---

### Day 2–3 | 封装 FEniCS 求解器

**目标**：把 Week 24 Day 3 的求解脚本封装成**可反复调用的函数**。

创建文件 `utils/fenicsx_solver.py`：

```python
"""
FEniCS 求解器封装。
"""
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import Expression
import ufl
import os

def solve_plate_hole(L, H, r, E, T_x, nu=0.3, 
                      mesh_file=None, return_stress=True):
    """
    求解带孔板弹性问题，返回节点坐标、单元、位移、应力。
    
    参数：
        L, H: 板尺寸
        r: 孔半径
        E, T_x: 弹性模量和拉力
        nu: 泊松比（默认 0.3）
        mesh_file: 预先生成的网格文件，如果为 None 则调 Gmsh 生成
    
    返回：
        dict: {'nodes', 'elements', 'u', 'sigma_vm', 'E', 'T_x', 'r', ...}
    """
    # 网格：如果没有预生成则调用 Gmsh
    if mesh_file is None or not os.path.exists(mesh_file):
        from utils.gmsh_utils import generate_plate_with_hole
        mesh_file = f"/tmp/plate_hole_L{L}_H{H}_r{r:.3f}.msh"
        generate_plate_with_hole(L, H, r, output_file=mesh_file)
    
    domain, _, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=2)
    
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    lmbda_ps = 2 * mu * lmbda / (lmbda + 2 * mu)
    
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    
    # 边界条件
    def left_boundary(x):
        return np.isclose(x[0], 0.0, atol=1e-4)
    
    left_dofs = fem.locate_dofs_geometrical(V, left_boundary)
    bc_left = fem.dirichletbc(
        np.array([0.0, 0.0], dtype=default_scalar_type),
        left_dofs, V
    )
    
    def right_boundary(x):
        return np.isclose(x[0], L, atol=1e-4)
    
    facet_dim = domain.topology.dim - 1
    right_facets = mesh.locate_entities_boundary(domain, facet_dim, right_boundary)
    facet_tag_data = mesh.meshtags(
        domain, facet_dim, right_facets,
        np.full_like(right_facets, 1)
    )
    
    # 变分形式
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    def sigma(u):
        return lmbda_ps * ufl.tr(epsilon(u)) * ufl.Identity(2) + 2 * mu * epsilon(u)
    
    T = fem.Constant(domain, default_scalar_type((T_x, 0.0)))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag_data)
    
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L_form = ufl.dot(T, v) * ds(1)
    
    # 求解
    problem = LinearProblem(a, L_form, bcs=[bc_left])
    uh = problem.solve()
    
    # 提取
    node_coords = domain.geometry.x[:, :2]
    u_values = uh.x.array.reshape(-1, 2)
    connectivity = domain.topology.connectivity(2, 0)
    elements = np.array([
        connectivity.links(i).tolist() 
        for i in range(domain.topology.index_map(2).size_local)
    ])
    
    result = {
        'nodes': node_coords,
        'elements': elements,
        'u': u_values,
        'E': E, 'nu': nu, 'T_x': T_x, 'r': r, 'L': L, 'H': H
    }
    
    # 应力场（可选）
    if return_stress:
        sigma_vm_ufl = ufl.sqrt(
            ufl.inner(sigma(uh), sigma(uh)) 
            - (ufl.tr(sigma(uh))) ** 2 / 3
        )
        V_scalar = fem.functionspace(domain, ("Lagrange", 1))
        sigma_vm_func = fem.Function(V_scalar)
        sigma_vm_expr = Expression(sigma_vm_ufl, V_scalar.element.interpolation_points())
        sigma_vm_func.interpolate(sigma_vm_expr)
        result['sigma_vm'] = sigma_vm_func.x.array
    
    return result
```

**注意**：真实工程中，每次调用 FEniCS 前最好 **新建** 一个 MPI Comm 或者确保前一次的资源释放了。简化版这里没处理，对 100 个样本应该够用；如果遇到内存泄漏问题，考虑每次求解开 subprocess。

**验收标准**：单次调用 `solve_plate_hole(L=2.0, H=1.0, r=0.2, E=210e3, T_x=50.0)` 能成功返回结果字典。

---

### Day 4 | 批量生成数据集脚本

创建文件 `week26/day4_build_dataset.py`：

```python
"""
批量生成 FEM 数据集。
"""
import numpy as np
import torch
import os
import time
from utils.fenicsx_solver import solve_plate_hole
from utils.graph_normalization import GraphZScoreNormalizer
from week25.day12_fenicsx_to_pyg import fenicsx_result_to_pyg_from_dict
from week25.day3_quality_check import check_fem_data_engineering  # Day 3 的函数
from week26.day01_lhs_sampling import sample_parameters

# ===== 配置 =====
N_SAMPLES = 50   # 可以先用 50 验证，再增加
SEED = 42
OUTPUT_DIR = 'data/phase2_dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 生成参数 =====
params_list = sample_parameters(N_SAMPLES, seed=SEED)

# ===== 批量求解 =====
dataset = []
failures = []
L_geom, H_geom = 2.0, 1.0

start_time = time.time()
for i, params in enumerate(params_list):
    try:
        print(f"\n[{i+1}/{N_SAMPLES}] 求解 E={params['E']:.1f}, "
              f"T_x={params['T_x']:.2f}, r={params['r']:.3f}")
        
        result = solve_plate_hole(
            L=L_geom, H=H_geom, r=params['r'],
            E=params['E'], T_x=params['T_x']
        )
        
        # 转 PyG Data（需要在 Week 25 的函数基础上做小改造，让它接受 dict）
        data = fenicsx_result_to_pyg_from_dict(result, L=L_geom, H=H_geom)
        
        # 质量检查
        issues, is_valid = check_fem_data_engineering(
            data, L=L_geom, H=H_geom, verbose=False
        )
        if not is_valid:
            failures.append((i, params, issues))
            continue
        
        dataset.append(data)
        print(f"  节点数: {data.num_nodes}, 通过检查")
        
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        failures.append((i, params, str(e)))

elapsed = time.time() - start_time
print(f"\n{'='*50}")
print(f"完成: {len(dataset)}/{N_SAMPLES} 成功")
print(f"失败: {len(failures)}")
print(f"总耗时: {elapsed:.1f} 秒")

# ===== 保存数据集 =====
torch.save(dataset, os.path.join(OUTPUT_DIR, 'plate_hole_dataset.pt'))
print(f"数据集已保存")

# 保存失败记录（如果有）
if failures:
    with open(os.path.join(OUTPUT_DIR, 'failures.txt'), 'w') as f:
        for idx, params, reason in failures:
            f.write(f"样本 {idx}: {params}\n  失败原因: {reason}\n\n")
```

**注意事项**：
- 第一次跑建议 N_SAMPLES=5 做 smoke test，确认整个管线没问题再放大到 50
- 如果失败率高（比如 >20%），先定位哪类参数容易失败再修
- 50 个样本在普通笔记本上总耗时预估 10-30 分钟

**验收标准**：
- 生成至少 40 个有效样本（失败率 < 20%）
- `plate_hole_dataset.pt` 文件能被 torch.load 加载
- 加载后的数据集每个样本都有 `x`, `edge_index`, `edge_attr`, `y`, `elem_list` 等字段

---

### Day 5–6（周末）| 数据集探索与质量报告

创建文件 `week26/weekend_dataset_report.py`：

```python
"""
对生成的数据集做探索性分析，输出质量报告。
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

dataset = torch.load('data/phase2_dataset/plate_hole_dataset.pt')

# ===== 1. 规模统计 =====
node_counts = [d.num_nodes for d in dataset]
edge_counts = [d.num_edges for d in dataset]
elem_counts = [len(d.elem_list) for d in dataset]

print(f"样本数: {len(dataset)}")
print(f"节点数: min={min(node_counts)}, max={max(node_counts)}, "
      f"mean={np.mean(node_counts):.1f}")
print(f"单元数: min={min(elem_counts)}, max={max(elem_counts)}, "
      f"mean={np.mean(elem_counts):.1f}")

# ===== 2. 位移量级分布 =====
max_disps = [torch.abs(d.y).max().item() for d in dataset]
print(f"最大位移: min={min(max_disps):.6f}, max={max(max_disps):.6f}")

# ===== 3. 参数覆盖 =====
Es = [d.E for d in dataset]
T_xs = [d.T_x for d in dataset]

# 生成报告图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(node_counts, bins=20)
axes[0, 0].set_xlabel('Node count'); axes[0, 0].set_title('Node count distribution')

axes[0, 1].hist(max_disps, bins=30)
axes[0, 1].set_xlabel('Max displacement')
axes[0, 1].set_title('Displacement magnitude distribution')

axes[1, 0].scatter(Es, T_xs, alpha=0.6)
axes[1, 0].set_xlabel('E (MPa)'); axes[1, 0].set_ylabel('T_x (MPa)')
axes[1, 0].set_title('Parameter coverage (E vs T)')

# 随机抽一个样本可视化
sample = dataset[np.random.randint(len(dataset))]
nodes_vis = sample.x[:, :2].numpy()
u_vis = sample.y[:, 0].numpy()
axes[1, 1].scatter(nodes_vis[:, 0], nodes_vis[:, 1], c=u_vis, cmap='jet')
axes[1, 1].set_aspect('equal')
axes[1, 1].set_title(f'Sample: ux field')

plt.tight_layout()
plt.savefig('data/phase2_dataset/dataset_report.png', dpi=120)
print("质量报告已保存")

# ===== 4. 写文字总结 =====
with open('data/phase2_dataset/dataset_summary.md', 'w') as f:
    f.write(f"""# FEM 数据集质量报告

## 基本信息
- 样本数: {len(dataset)}
- 节点数范围: [{min(node_counts)}, {max(node_counts)}]
- 单元数范围: [{min(elem_counts)}, {max(elem_counts)}]
- 最大位移范围: [{min(max_disps):.6f}, {max(max_disps):.6f}]

## 参数覆盖
- E: [{min(Es):.1f}, {max(Es):.1f}] MPa
- T_x: [{min(T_xs):.2f}, {max(T_xs):.2f}] MPa

## 数据用途
- 可用于训练 PhyFENet（Part F）
- 标签量级合理，适合做 Z-Score 归一化
- 节点数不固定（每个样本不同网格），PhyFENet 的 GraphSAGE 架构可处理
""")
```

**验收标准**：
- 质量报告图包含 4 个子图
- 参数覆盖图能看出 LHS 采样的均匀性（不是对角线或聚堆）
- 位移量级分布应该是一个合理的连续分布

---

### Week 26 完成标准

- [ ] 拉丁超立方参数采样工具在 `utils/` 或 `week26/`
- [ ] FEniCS 求解器封装函数 `solve_plate_hole` 放在 `utils/fenicsx_solver.py`
- [ ] 能一次性生成 50 个样本的数据集（失败率 < 20%）
- [ ] 数据集保存为 `plate_hole_dataset.pt`
- [ ] 有一个数据集质量报告（`dataset_report.png` + `dataset_summary.md`）

---

---

## Week 27: Part E 闭环 + PhyFENet 初次对接真实数据

**本周定位**：Part E 的闭环周。你有了真实 FEniCS 数据集（Week 26），有了第一阶段搭好的 PhyFENet 架构（Part D），本周**把它们对接上跑一次训练**。这是初步验证；深入的训练调优、小样本实验、迁移学习都放到 Part F 做。

---

### Day 1–2 | 数据加载与 PhyFENet 单图训练

**本周的核心任务**（对接 Part D Week 20 的单图训练能力）：

创建文件 `week27/day12_first_training.py`：

```python
"""
PhyFENet 在 FEniCS 真实数据上的第一次训练。
先用单图训练（batch_size=1）验证。
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from utils.gnn_models import PhyFENet_Mini  # Part C Week 15
from utils.graph_normalization import GraphZScoreNormalizer
# 其他 utils

# ===== 加载数据 =====
dataset = torch.load('data/phase2_dataset/plate_hole_dataset.pt')
print(f"加载数据集: {len(dataset)} 样本")

# train/val split
torch.manual_seed(42)
import random
random.seed(42)
indices = list(range(len(dataset)))
random.shuffle(indices)
n_train = int(0.8 * len(dataset))
train_set = [dataset[i] for i in indices[:n_train]]
val_set = [dataset[i] for i in indices[n_train:]]
print(f"Train: {len(train_set)}, Val: {len(val_set)}")

# ===== 归一化 =====
normalizer = GraphZScoreNormalizer()
normalizer.fit(train_set)
train_set_norm = normalizer.transform(train_set)
val_set_norm = normalizer.transform(val_set)
normalizer.save('data/phase2_dataset/normalizer_week27.pt')

# ===== 模型 =====
# 节点特征 8 维（坐标3 + 材料2 + 边界3），输出 2 维（ux, uy）
model = PhyFENet_Mini(
    node_in=8, edge_in=3, hid=64, node_out=2, n_mp_layers=3
)
print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_loader = DataLoader(train_set_norm, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set_norm, batch_size=1)

# ===== 训练 =====
n_epochs = 200
train_losses, val_losses = [], []

for epoch in range(n_epochs):
    # Train
    model.train()
    train_loss_sum = 0
    for batch in train_loader:
        y_pred = model(batch)
        loss = criterion(y_pred, batch.y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss_sum += loss.item() * batch.num_nodes
    train_loss_avg = train_loss_sum / sum(b.num_nodes for b in train_set_norm)
    train_losses.append(train_loss_avg)
    
    # Val
    model.eval()
    val_loss_sum = 0
    with torch.no_grad():
        for batch in val_loader:
            y_pred = model(batch)
            val_loss_sum += criterion(y_pred, batch.y).item() * batch.num_nodes
    val_loss_avg = val_loss_sum / sum(b.num_nodes for b in val_set_norm)
    val_losses.append(val_loss_avg)
    
    if epoch % 10 == 0:
        print(f"epoch {epoch}: train={train_loss_avg:.6f}, val={val_loss_avg:.6f}")

# 保存训练曲线 + 模型
torch.save(model.state_dict(), 'week27_model.pt')
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.yscale('log')
plt.xlabel('Epoch'); plt.ylabel('MSE (normalized)')
plt.legend(); plt.title('First training on real FEM data')
plt.savefig('week27_training.png')
```

**关于预期**：
- 这是**第一次对接真实数据**的训练，目的是验证管线能跑通
- 不要对精度有过高期待——PhyFENet 的精细调优（物理 Loss 权重、多级网络）是 Part F 的事
- 能看到 train loss 和 val loss 都在下降就是成功

**验收标准**：
- 训练 200 epoch 无报错
- Train loss 和 val loss 都下降至少一个数量级
- 保存了模型和训练曲线

---

### Day 3 | 预测可视化

创建文件 `week27/day3_visualize_predictions.py`：

从 val_set 里抽 2-3 个样本，做可视化对比：
- FEniCS 真实位移场（ux, uy 两张图）
- PhyFENet 预测位移场（ux, uy 两张图）
- 误差场（|pred - true|）

布局建议：每个样本 2×3 子图（2 行分别是真实和预测，3 列分别是 ux/uy/误差）。

**可视化要点**：
- 用三角剖分（`matplotlib.tri.Triangulation`）
- 真实和预测用同一 colormap 和同一色阶（方便视觉对比）
- 记得**反归一化** PhyFENet 的输出（用 `normalizer.inverse_transform_y`）

**验收标准**：
- 从图上能看出 PhyFENet 预测和 FEniCS 真实结果**趋势相似**
- 能指出预测误差最大的区域（通常是孔附近的应力集中处）

---

### Day 4 | 指标化评估

创建文件 `week27/day4_metrics.py`：

```python
"""
在 val_set 上计算标准评估指标。
"""
import torch
import numpy as np

def evaluate_on_val(model, val_set_norm, normalizer):
    """
    返回：
        metrics: dict with 'mae', 'rmse', 'r2', 'per_sample_mae'
    """
    model.eval()
    all_pred = []
    all_true = []
    per_sample_mae = []
    
    with torch.no_grad():
        for data in val_set_norm:
            y_pred_norm = model(data)
            # 反归一化
            y_pred = normalizer.inverse_transform_y(y_pred_norm)
            y_true = normalizer.inverse_transform_y(data.y)
            
            all_pred.append(y_pred)
            all_true.append(y_true)
            
            sample_mae = torch.abs(y_pred - y_true).mean().item()
            per_sample_mae.append(sample_mae)
    
    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    
    mae = torch.abs(all_pred - all_true).mean().item()
    rmse = torch.sqrt(((all_pred - all_true) ** 2).mean()).item()
    
    # R²
    ss_res = ((all_true - all_pred) ** 2).sum().item()
    ss_tot = ((all_true - all_true.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'per_sample_mae_mean': np.mean(per_sample_mae),
        'per_sample_mae_std': np.std(per_sample_mae)
    }

# 使用
metrics = evaluate_on_val(model, val_set_norm, normalizer)
print(f"MAE: {metrics['mae']:.6f}")
print(f"RMSE: {metrics['rmse']:.6f}")
print(f"R²: {metrics['r2']:.4f}")
```

**验收标准**：
- R² > 0.8（初步训练应该至少达到这个水平；如果低很多，排查代码）
- 每个样本的 MAE 分布合理

**⚠️ 数据和结论的表述**：
这是第一次对接真实数据，**不要过度解读**数字。面试时这样说：
- ❌ 错误："我的 PhyFENet 在真实 FEM 数据上 MAE 是 X"
- ✅ 正确："我完成了 PhyFENet 在真实 FEniCS 生成数据上的对接，初步训练验证了管线能跑通。Part F 要做物理 Loss、多级网络、超参扫描等深入调优，得到更有说服力的结果。"

---

### Day 5–6（周末）| Part E 收尾与反思

**任务 1**：把本 Part 的所有成果整理

目录结构：
```
ai-cae-learning/
├── utils/
│   ├── gmsh_utils.py              ← Week 24
│   ├── fenicsx_solver.py          ← Week 26
│   ├── graph_normalization.py     ← Week 25
│   ├── fem_quality_check.py       ← Week 25
│   └── (原有第一阶段的模块)
├── data/
│   └── phase2_dataset/
│       ├── plate_hole_dataset.pt
│       ├── normalizer_week27.pt
│       ├── dataset_report.png
│       └── dataset_summary.md
└── phase2/
    └── part_e_real_fem/
        ├── week23/ ~ week27/
        └── README.md     ← Part E 总结
```

**任务 2**：写 Part E 总结 `phase2/part_e_real_fem/README.md`：
- 我完成了什么（FEniCS 入门、Gmsh 参数化、批量数据集、PhyFENet 初对接）
- 遇到了什么坑（重点写 FEniCS 安装）
- 关键 utils 列表
- 数据集描述（50 样本，参数范围，质量）
- 下一步（Part F: 深入训练调优）

---

### Week 27 完成标准

- [ ] PhyFENet 在 FEniCS 真实数据集上跑通完整训练
- [ ] 有评估指标（MAE, RMSE, R²）
- [ ] 有预测可视化图
- [ ] Part E 的代码和文档整理完毕

---

---

## Week 28: HyperMesh / LS-DYNA 最小认知模块

**本周定位**：Part E 的可选但强烈推荐的最后一周。你的目标论文（贺宏伟硕士论文）第四章用的是 HyperMesh + LS-DYNA，面试时被问"你熟悉这些工具吗"的概率极高。本周目标不是学会使用这些工具（那是 CAE 工程师的核心技能，不是你的核心），而是建立**读懂数据 + 理解工作流**的最小认知。

**本周时间预算**：6-7 天，偏认知而非实操。

**本周不做**：
- 不要深度学习 HyperMesh 的菜单操作
- 不要试图自己跑一个 LS-DYNA 仿真（许可证昂贵，实际工作中会有专人做）
- 不要去记所有网格质量指标的工业阈值

**本周要做**：
- 能看着一个 `.inp` (Abaqus) 或 `.k` (LS-DYNA) 文件说出它的结构
- 能用一句话解释冲压成形仿真的物理过程
- 能回答面试问题："你的论文方向是冲压成形，但你用的是 FEniCS 开源软件，对 HyperMesh/LS-DYNA 熟悉吗？"

---

### Day 1 | 冲压成形仿真的物理过程

**任务**：阅读论文第四章（贺宏伟论文），重点理解这几个概念。

**必须理解的概念**（每个写 100-200 字的笔记）：

1. **冲压成形的三个阶段**：
   - 压边（blank holder 把板料压在凹模上）
   - 拉延（凸模下行，把板料拉进凹模型腔）
   - 回弹（卸载后板料因弹性应力释放产生变形）

2. **关键物理量**：
   - **应变分布**：板料各处的拉伸/压缩程度
   - **厚度减薄率**：`(t_0 - t) / t_0`，是否超过允许值决定是否破裂
   - **成形极限图（FLD）**：主应变 vs 次应变的安全边界

3. **为什么要用 AI 加速**：一次 LS-DYNA 冲压仿真在普通工作站上要跑几小时到一天；设计迭代需要跑几十次；用 GNN 代理模型在训练好后推理是毫秒级的。

**这几个概念面试必问**。花 1-2 小时把它们弄明白。

---

### Day 2 | Abaqus .inp 文件结构解剖

**任务**：Abaqus `.inp` 是纯文本，能直接用编辑器打开。获取一个示例文件（Abaqus 示例库、网络上下载、或 Part D Week 18 你模拟的那个），对照阅读。

**关键段落**（用笔记对应起来）：

```
*HEADING                 # 标题、注释
Simple plate example

*NODE                    # 节点段
 1, 0.0, 0.0, 0.0        # 节点编号, x, y, z
 2, 1.0, 0.0, 0.0
...

*ELEMENT, TYPE=S4R       # 单元段（S4R = 4节点缩减积分壳单元）
 1, 1, 2, 3, 4           # 单元编号, 节点1-4

*MATERIAL, NAME=AL6061
*ELASTIC
 68900.0, 0.33           # 弹性模量, 泊松比
*PLASTIC
 0.0, 0.0                # 应力-塑性应变表（幂指数硬化）
 50.0, 0.01
 ...

*STEP
*STATIC
*BOUNDARY                # 约束
 1, 1, 3, 0.0            # 节点1, 1-3自由度, 值=0

*CLOAD                   # 集中载荷
 5, 1, 100.0             # 节点5, x方向, 100N

*OUTPUT, FIELD
*NODE OUTPUT
U                        # 输出位移
*ELEMENT OUTPUT  
S, E                     # 输出应力、应变

*END STEP
```

**这和 Part D Week 18 你实现的解析器很像**——只是真实文件更长、段落更多。

**实践任务**（约 1 小时）：改造 Part D Week 18 的 `parse_fem_file` 函数，让它能处理 `*MATERIAL/*ELASTIC` 段和 `*ELEMENT, TYPE=S4R` 这类带单元类型标识的行。（完整改造不必要，做到能读节点和单元就行。）

---

### Day 3 | LS-DYNA .k 文件结构

**任务**：LS-DYNA 的 `.k` 文件格式和 Abaqus 类似但用不同的关键字（以 `*` 开头，但名字不同）。对照表：

| Abaqus | LS-DYNA | 含义 |
|--------|---------|------|
| `*NODE` | `*NODE` | 节点（格式类似）|
| `*ELEMENT, TYPE=S4R` | `*ELEMENT_SHELL` | 壳单元 |
| `*MATERIAL` + `*ELASTIC` | `*MAT_ELASTIC` or `*MAT_024` etc. | 材料 |
| `*BOUNDARY` | `*BOUNDARY_SPC_NODE` | 约束 |
| `*STEP` | `*CONTROL_TERMINATION` + `*CONTROL_*` | 求解控制 |
| - | `*CONTACT_*` | 接触定义（LS-DYNA 特色）|
| - | `*INITIAL_VELOCITY` | 初始速度（显式分析常用）|

**两者的核心区别**：
- **Abaqus** 偏向隐式求解（适合静态、准静态问题）
- **LS-DYNA** 偏向显式求解（适合动态、冲击、冲压等）
- 你的目标方向（冲压成形）主要用 LS-DYNA

**任务**：网上搜一个 LS-DYNA 示例文件（或者 DYNA 官方例子的 `.k` 文件），打开看一下，**不用深入学语法**，能对着结构说出"这是节点、这是单元、这是材料"即可。

---

### Day 4 | 网格质量指标的概念理解

**任务**：建立概念认知，**不背阈值**。

**3 个主要指标**：

1. **Jacobian（雅可比）**：衡量单元几何"正规程度"。1.0 = 理想；接近 0 或负 = 单元退化（几乎不是单元了）。
2. **Aspect Ratio（纵横比）**：单元最长边 / 最短边。1.0 = 理想；太大（比如 > 10）= 单元狭长，计算结果不稳。
3. **Warpage（翘曲度）**：对壳单元/四边形的扭曲程度。0 = 平面；太大 = 单元不在一个平面内，壳单元假设失效。

**你要能做到的**：面试时被问"HyperMesh 网格质量怎么看"，能说：
> "我能理解 Jacobian、Aspect Ratio、Warpage 这些指标是什么意思——它们反映网格单元的几何退化程度。具体阈值不同项目会有不同标准，具体使用会在工作中学。"

---

### Day 5 | 工业流程的端到端认知

**画一张流程图**（手绘或用任何工具），画出汽车冲压件设计仿真的完整流程：

```
CAD 几何模型（Catia/SolidWorks/NX）
       ↓
  前处理（HyperMesh 或 ANSA）
   - 导入几何
   - 划分网格
   - 质量检查
   - 定义材料
   - 定义边界条件（压边、凸模运动等）
   - 定义接触对（凸模-板料-凹模）
       ↓
  求解器（LS-DYNA 或 AutoForm）
   - 显式动力学求解
   - 输出每个时刻的应变/应力/厚度
       ↓
  后处理（HyperView / LS-PrePost）
   - 查看应变云图
   - 查看成形极限图 FLD
   - 查看厚度分布
   - 找出破裂/起皱风险区域
       ↓
  [设计迭代：如果有问题，回到前处理调整]

【AI 切入点】
  AI 代理模型（你的研究方向）：
   - 从 FEM 历史数据训练 GNN
   - 新设计参数输入 → GNN 毫秒级预测
   - 替代反复的 FEM 求解，加速迭代
```

---

### Day 6 | 面试问题演练

**任务**：模拟几个高概率面试问题，给出自己的回答（写下来，不要只在脑子里想）。

**Q1**：你的论文方向是汽车冲压成形 AI 加速，但你用的是 FEniCS，不是工业标准的 HyperMesh + LS-DYNA，这个差距怎么解释？

**建议的回答方向**：
> "我的核心能力在 ML 部分——图神经网络架构、物理信息嵌入、多级网络训练。这些在开源 FEniCS 数据上已经完整跑通。工业软件 HyperMesh/LS-DYNA 我能读懂它们的 .inp/.k 文件格式，理解冲压成形仿真的前-求解-后处理流程，但没有自己独立操作过——这是 CAE 工程师的核心技能，我的角色是和他们协作，用他们生成的数据训练 AI 模型。入职后用几周熟悉数据格式对接就能开始工作。"

**Q2**：网格质量指标你熟悉吗？比如 Jacobian 多少算合格？

**建议的回答方向**：
> "我知道 Jacobian、Aspect Ratio、Warpage 这些是评估网格几何退化程度的指标。具体阈值我不记得——不同项目、不同单元类型有不同标准，这个是在实际项目中根据 CAE 工程师的经验来定。我能做的是用数据质量检查脚本自动筛掉明显异常的样本。"

**Q3**：为什么你选 GNN + PINN，不用 CNN 或 Transformer？

**建议的回答方向**：
> "FEM 网格是不规则拓扑——每个节点的邻居数量不一致，几何距离各异。CNN 要求规则欧氏网格，Transformer 的全局注意力对这么大的网格计算成本极高。GNN 的消息传递正好匹配网格拓扑——节点只从有限邻居收集信息，自然保留了几何局部性。PINN 的物理约束确保在数据少时也能保持物理一致性。论文第二章的 PhyFENet 把这两者结合起来。"

---

### Week 28 完成标准

- [ ] 理解冲压成形仿真的物理过程和 AI 切入点
- [ ] 能看懂 `.inp` / `.k` 文件的基本结构
- [ ] 理解网格质量指标的概念（不要求记阈值）
- [ ] 画出汽车冲压仿真的工业流程图
- [ ] 准备好面试常见问题的回答

---

---

## Part E 总完成标准

**技术能力**：
- [ ] 能独立安装 FEniCS 并解决常见问题
- [ ] 能用 FEniCS 求解 2D 弹性问题
- [ ] 能用 Gmsh 生成参数化网格
- [ ] 能写批量数据生成管线
- [ ] 有 50+ 样本的真实 FEM 数据集
- [ ] PhyFENet 在真实数据上跑通第一次训练

**工程基础**：
- [ ] `utils/` 下有至少 5 个复用工具（Gmsh/FEniCS/归一化/质量检查/求解封装）
- [ ] 数据集有质量报告和元数据
- [ ] 代码组织清晰，随时能回来做实验

**行业认知**：
- [ ] 能读懂 Abaqus `.inp` 和 LS-DYNA `.k` 文件结构
- [ ] 理解冲压成形仿真的完整工业流程
- [ ] 准备好面试常见问题的回答

**如果 Part E 收尾时以上都达到，你会是一个"懂 CAE 数据的 ML 工程师"——正是你目标岗位要的画像**。

---

