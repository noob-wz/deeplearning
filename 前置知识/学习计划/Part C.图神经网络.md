# Part C: 图神经网络（Week 11–16）

## 本阶段定位

**衔接前面**：
- Part A 让你能搭建 MLP 完成回归任务
- Part B 让你能用 autograd 算导数，把物理方程写进 Loss

但 MLP 有一个根本局限：它把**输入看成一个独立的特征向量**。对于一个节点，它不知道这个节点的"邻居"是谁。

**这个局限对 FEM 数据是致命的**。论文第二章 §2.1 明确讲到：

> "图结构数据不遵循连续性和规则性的几何性质。在图结构中，节点之间的关系并非简单的数值型距离，而是依赖于图的拓扑连接模式。"

FEM 网格就是典型的图——每个节点的位移不仅取决于它自己的材料/载荷，更**强烈依赖于它周围节点的状态**（通过刚度矩阵耦合）。MLP 表达不了这种"邻居关系"。

**Part C 要解决的是这个问题**：怎么让神经网络"看到"图结构？答案是图神经网络（GNN）。具体到论文，用的是 GraphSAGE + 边更新的机制。

**Part C 与前面的关系**：
- Part C 学的是神经网络的**新结构**（从 MLP 到 GNN）
- Part C 暂时不用 Part B 的物理约束——这里先专注于"学会图神经网络本身"
- Part D 会把 GNN（Part C）和物理约束（Part B）融合起来

**Part C 完成标准**（进入 Part D 前）：
- 理解"图数据"的结构（节点、边、邻接矩阵）
- 能用 PyTorch Geometric 构造图数据对象
- 能从零实现 GCN 和 GraphSAGE
- 能实现带边更新的消息传递（论文 §2.3.3 的核心机制）
- 能在合成力学数据上训练 GNN 预测节点量

**时间预期**：**6-9 周**，基线 6 周，实际很可能需要 7-9 周。图神经网络对新手是全新概念，Week 11 搭基础（图数据结构）、Week 12 学 GCN 都相对温和，但 **Week 13-15 是密集区**——三周连续引入新模型和新实验。如果感觉吃力，每周可延长到 1.5 周，三周总时长扩展到 4-5 周。**不要为了"赶完 6 周"硬推**——Part D 需要在 Part C 的基础上融合物理约束，Part C 不扎实，Part D 会崩。

**本阶段核心资源**：
- 斯坦福 CS224W（Jure Leskovec）：Lecture 6（GNN 介绍）、Lecture 7（GCN）、Lecture 8（GraphSAGE）。YouTube 可找到录像。
- PyTorch Geometric 官方文档 "Introduction by Example"：https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
- 论文《嵌入本构方程的图神经网络模型》§2.1、§2.3.3（反复读）

---

---

## Week 11: 图数据结构与图的基本概念

**衔接**：前面所有任务中，你的输入都是形如 `(N, features)` 的张量——N 个独立样本。这周开始，你的输入变成"图"——一个节点和它邻居的结构。

**本周目标**：
- 理解图数据的三要素：节点特征 / 边索引 / 边特征
- 能用 NumPy 写出"规则矩形网格 → 图结构"的转换
- 能用 PyTorch Geometric 的 `Data` 对象表示图
- 能可视化一个图

**本周不做**：任何神经网络训练。只打图数据结构的基础。

---

### Day 1 | 图是什么：用中学几何理解

**理论任务**（约 1 小时）：

在笔记上画一个具体的图例子：
```
    1 ———— 2
    |  \   |
    |   \  |
    4 ———— 3
```

**任务**：把这个图用以下三种方式表达出来。

**方式 1：邻接矩阵 A**（4×4 矩阵）
```
节点间是否有边：1 表示有，0 表示无。对角线为 0（不自连）

       1  2  3  4
  1 [  0  1  1  1  ]
  2 [  1  0  1  0  ]
  3 [  1  1  0  1  ]
  4 [  1  0  1  0  ]
```

**方式 2：边列表**（COO 格式）
```
edges = [(1,2), (1,3), (1,4), (2,3), (3,4)]
```

**方式 3：PyTorch Geometric 用的 `edge_index`**（所有边要双向列出）
```
edge_index = [
    [0, 1, 0, 2, 0, 3, 1, 2, 2, 3],   # source nodes
    [1, 0, 2, 0, 3, 0, 2, 1, 3, 2]    # target nodes
]
# 注意：PyG 索引从 0 开始，所以节点 1 在代码里是 index 0
```

**在笔记上回答**：
- 如果图有 N 个节点，邻接矩阵的 shape 是什么？（N×N）
- 对于**无向图**，邻接矩阵有什么性质？（对称矩阵，A[i,j] = A[j,i]）
- `edge_index` 中每一列代表什么？（一条有向边，从 source 指向 target）
- 为什么无向图的 `edge_index` 里每条边要写两次？（一次 i→j，一次 j→i）

**实践任务**（约 1 小时）：创建文件 `week11/day01_graph_representations.py`

```python
import numpy as np
import torch

# ===== 方式 1：邻接矩阵 =====
A = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0]
], dtype=np.float32)

print("邻接矩阵:\n", A)
print("是否对称:", np.array_equal(A, A.T))    # 应该 True

# 度矩阵（每个节点的度，即邻居数）
degrees = A.sum(axis=1)
print("节点度数:", degrees)    # 应为 [3, 2, 3, 2]

# ===== 方式 2：从邻接矩阵转 edge_index =====
# 找出所有 A[i,j] = 1 的位置
src, dst = np.where(A == 1)
edge_index = np.stack([src, dst], axis=0)   # shape (2, n_edges)
print("edge_index:\n", edge_index)
print("边的总数（有向）:", edge_index.shape[1])   # 应为 10（5 条无向边 × 2）

# ===== 方式 3：转成 torch tensor（PyG 要的格式）=====
edge_index_t = torch.from_numpy(edge_index).long()
print("edge_index tensor dtype:", edge_index_t.dtype)
```

**验收标准**：
- 邻接矩阵、度数、edge_index 都输出正确
- 能说清楚为什么 PyG 的 edge_index 是 (2, n_edges) 的 shape 而不是 (n_edges, 2)
  - 参考：(2, n_edges) 便于用 `edge_index[0]` 直接取所有 source、`edge_index[1]` 取所有 target

---

### Day 2 | 从规则网格到图：一个"物理上合法"的小网格

**重要说明（回应上轮讨论）**：这里我们**不用随机连接**的单元，而是用结构化的规则网格——每个单元的节点按几何相邻关系连接，邻接关系有清晰的物理含义。这避免你把 FEM 网格当成"任意图"的错误认知。

**理论任务**：

**重要前置：区分两种"邻居"概念**

FEM 网格转图的过程中，"邻居"这个词有两种不同含义，你必须清楚区分：

**邻居概念 A：几何结构邻居**
- 定义：在网格上**直接共享一条网格边**的节点
- 例子：节点 4 的结构邻居是 1, 3, 5, 7（上下左右，因为它们和 4 之间有网格线直接相连）
- 对角线上的节点（0, 2, 6, 8）**不是**结构邻居

**邻居概念 B：图邻居（论文用的）**
- 定义：按"单元内任意两节点都建边"的规则，凡是和 i 在同一个单元里的节点，都是 i 的图邻居
- 例子：节点 4 所在的 4 个单元包含节点 {0,1,2,3,4,5,6,7,8} 的全集，所以它的图邻居是**其他所有 8 个节点**
- 这包括对角线上的 0, 2, 6, 8

**为什么要用"概念 B"而不是"概念 A"**：
- 论文 §2.3.2 式 2.19 的规则就是"单元内节点两两建边"
- 单元内所有节点在物理上通过刚度矩阵耦合（即便不共享网格边），它们的特征需要共同决定该单元的变形
- 概念 B 提供了更丰富的信息传递通路，对物理场预测更有帮助

**所以 Day 2 之后的代码（`mesh_to_edges`）采用的是概念 B。**

---

现在画一个 3×3 的规则矩形网格（9 个节点，4 个四边形单元）：
```
  6 ——— 7 ——— 8
  |  e3 |  e4 |
  3 ——— 4 ——— 5
  |  e1 |  e2 |
  0 ——— 1 ——— 2
```

**问题 1**：按**概念 A（结构邻居）**，节点 4 的邻居是谁？
参考答案：1, 3, 5, 7（上下左右四个）

**问题 2**：按**概念 B（图邻居）**，节点 4 的邻居是谁？
参考答案：0, 1, 2, 3, 5, 6, 7, 8（它所在 4 个单元里的其他所有节点）

**问题 3**：单元 e1 由哪些节点组成？
参考答案：e1 由节点 0, 1, 3, 4 围成（左下四边形）。

**问题 4**：按概念 B 从单元连接关系提取边——对每个单元，每两个节点之间都构成一条图上的边。
- 单元 e1 (0,1,3,4) 贡献的边：(0,1), (0,3), (0,4), (1,3), (1,4), (3,4) —— 共 C(4,2) = 6 条边
- 注意 (0,4) 和 (1,3) 是**对角线边**（两节点不共享网格边，但在同一个单元内）
- 去重后整张图的边：每条边只保留一次（即使被多个单元共享）

**实践任务**（约 2 小时）：创建文件 `week11/day02_regular_mesh_to_graph.py`

```python
import numpy as np
import torch

def build_regular_mesh(nx, ny, dx=1.0, dy=1.0):
    """
    生成一个 nx × ny 的规则矩形网格
    
    返回：
      nodes: np.ndarray shape=(nx*ny, 2)，每行是 (x, y) 坐标
      elements: list，每个元素是 4 个节点索引（左下、右下、右上、左上）
    """
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([i * dx, j * dy])
    nodes = np.array(nodes, dtype=np.float32)
    
    # 每个单元由 4 个节点组成
    # 节点编号公式：index = j * nx + i
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i           # 左下
            n1 = j * nx + (i + 1)     # 右下
            n2 = (j + 1) * nx + (i + 1)  # 右上
            n3 = (j + 1) * nx + i     # 左上
            elements.append([n0, n1, n2, n3])
    
    return nodes, elements


def mesh_to_edges(elements):
    """
    从单元连接关系提取无向边集合
    
    返回：
      edge_set: set，每个元素是一个排好序的元组 (min_id, max_id)
    """
    edge_set = set()
    for elem in elements:
        n = len(elem)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = min(elem[i], elem[j]), max(elem[i], elem[j])
                edge_set.add((a, b))
    return edge_set


def edges_to_edge_index(edge_set):
    """
    把边集合转成 PyG 的 edge_index 格式（双向，shape=(2, 2*n_undirected_edges)）
    """
    src, dst = [], []
    for a, b in edge_set:
        src.append(a); dst.append(b)    # a → b
        src.append(b); dst.append(a)    # b → a
    edge_index = np.array([src, dst], dtype=np.int64)
    return edge_index


# ===== 测试 =====
nodes, elements = build_regular_mesh(nx=3, ny=3)
print(f"节点数: {len(nodes)}")              # 应为 9
print(f"单元数: {len(elements)}")           # 应为 4
print(f"节点坐标:\n{nodes}")

edge_set = mesh_to_edges(elements)
print(f"\n无向边数（去重后）: {len(edge_set)}")   # 应为 20

edge_index = edges_to_edge_index(edge_set)
print(f"edge_index shape: {edge_index.shape}")    # 应为 (2, 40)
```

**验收标准**：
- 9 个节点、4 个单元、16 条无向边（12 条水平/垂直边 + 4 条单元内对角线）
- edge_index shape 为 (2, 32)
- 不看参考能解释"为什么会有对角线边"（每个单元内任意两节点都建边，所以 4 个节点的单元会有 6 条边，其中包含 2 条对角线）

**关于对角线边的讨论**：论文 §2.3.2 式 2.19 也是把单元的所有节点两两建边。这不是物理上的结构线，而是"单元内信息共享"的通路。你不需要纠结几何含义，把它理解为"GNN 的信息传递通路"即可。

---

### Day 3 | 可视化图

**实践任务**（约 1.5 小时）：创建文件 `week11/day03_visualize_mesh.py`

任务 1：用 Matplotlib 把 Day 2 的网格画出来
```python
import matplotlib.pyplot as plt
from week11.day02_regular_mesh_to_graph import build_regular_mesh, mesh_to_edges

nodes, elements = build_regular_mesh(nx=4, ny=4)
edge_set = mesh_to_edges(elements)

fig, ax = plt.subplots(figsize=(6, 6))

# 画边
for a, b in edge_set:
    x1, y1 = nodes[a]
    x2, y2 = nodes[b]
    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, lw=1)

# 画节点
ax.scatter(nodes[:, 0], nodes[:, 1], c='red', s=100, zorder=10)

# 标节点编号
for i, (x, y) in enumerate(nodes):
    ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=9)

ax.set_aspect('equal')
ax.set_title('4x4 Regular Mesh as Graph')
plt.savefig('week11_mesh_graph.png', dpi=100)
plt.close()
```

任务 2：另外画一张"只有结构边"（水平+垂直）的图
- 从 edge_set 中剔除对角线边（两节点坐标差都不为零的那些）
- 画出来应该是干净的网格线

```python
# 提示代码
structural_edges = set()
for a, b in edge_set:
    dx = abs(nodes[b,0] - nodes[a,0])
    dy = abs(nodes[b,1] - nodes[a,1])
    # 如果是水平（dy=0）或垂直（dx=0），是结构边
    if dx < 0.01 or dy < 0.01:
        structural_edges.add((a, b))
```

**验收标准**：
- 两张图都能生成
- 第一张图你能看到每个单元内的 6 条边（4 条结构 + 2 条对角）
- 节点编号标注清楚

---

### Day 4 | PyTorch Geometric 的 `Data` 对象

**核心概念**：PyG 用 `Data` 对象封装一张图，包含 4 个常用字段：
- `x`：节点特征张量，shape=(N, n_features)
- `edge_index`：边索引，shape=(2, n_edges)
- `edge_attr`：边特征（可选），shape=(n_edges, n_edge_features)
- `y`：节点/图级别的标签（可选），回归任务里是浮点数

**实践任务**（约 2 小时）：创建文件 `week11/day04_pyg_data.py`

```python
import torch
import numpy as np
from torch_geometric.data import Data
from week11.day02_regular_mesh_to_graph import (
    build_regular_mesh, mesh_to_edges, edges_to_edge_index
)

# ===== Step 1: 生成一个网格 =====
nodes, elements = build_regular_mesh(nx=4, ny=4)
edge_set = mesh_to_edges(elements)
edge_index = edges_to_edge_index(edge_set)

# ===== Step 2: 构造节点特征 =====
# 这里模拟"每个节点的材料和边界条件特征"
# 节点特征 = [x, y, E, F_applied]
# 为了简化，全图 E=200，只有右侧边缘节点有载荷 F=100，其他 F=0
N = len(nodes)
x_coord = nodes[:, 0:1]                      # shape=(N, 1)
y_coord = nodes[:, 1:2]                      # shape=(N, 1)
E_feat = np.full((N, 1), 200.0)              # 弹性模量
F_feat = np.zeros((N, 1))                    # 载荷，默认 0

# 右侧边缘（x=3 的节点）施加载荷
max_x = nodes[:, 0].max()
right_mask = np.abs(nodes[:, 0] - max_x) < 0.01
F_feat[right_mask] = 100.0

x_features = np.concatenate([x_coord, y_coord, E_feat, F_feat], axis=1)
x_features = torch.from_numpy(x_features).float()    # shape=(16, 4)

# ===== Step 3: 构造边特征（节点间的相对位移和距离）=====
# 每条边的特征 = [dx, dy, distance]
src = edge_index[0]
dst = edge_index[1]

src_coords = nodes[src]      # shape=(n_edges, 2)
dst_coords = nodes[dst]
diff = dst_coords - src_coords                      # shape=(n_edges, 2)
dist = np.linalg.norm(diff, axis=1, keepdims=True)  # shape=(n_edges, 1)
edge_attr = np.concatenate([diff, dist], axis=1)    # shape=(n_edges, 3)
edge_attr = torch.from_numpy(edge_attr).float()

# ===== Step 4: 构造目标 y（这里先用随机值模拟"位移"）=====
np.random.seed(0)
y = np.random.randn(N, 2) * 0.1    # 模拟的 (ux, uy) 位移
y = torch.from_numpy(y).float()

# ===== Step 5: 封装成 Data 对象 =====
edge_index_t = torch.from_numpy(edge_index).long()
data = Data(
    x=x_features,
    edge_index=edge_index_t,
    edge_attr=edge_attr,
    y=y
)

# ===== 检查 =====
print(data)                           # 打印 Data 对象摘要
print(f"节点数: {data.num_nodes}")    # 16
print(f"边数: {data.num_edges}")      # 用 edge_index.shape[1]
print(f"节点特征维度: {data.num_node_features}")   # 4
print(f"边特征维度: {data.num_edge_features}")     # 3

# 验证：边特征的 "dist" 列应该都是正数
print(f"边距离范围: [{edge_attr[:, 2].min()}, {edge_attr[:, 2].max()}]")
```

**验收标准**：
- `Data` 对象能正常创建
- 各项 shape 和数量正确（16 个节点，4 个节点特征，3 个边特征）
- 边距离都是正数（都应该在 1.0 到 √2 之间）

---

### Day 5–6（周末）| 综合练习：构造一个"弹性杆"图数据

**衔接你的方向**：这个周末任务设计了一个简化的"弹性杆"问题，贴近论文中的实际数据形态。

**物理设定**：
- 一根 5×3 的矩形板，离散为 5×3 规则网格（15 个节点）
- 左端（x=0 的 3 个节点）固定：位移 (ux=0, uy=0)
- 右端（x=4 的 3 个节点）受均匀拉力 F=100
- 材料均匀：E=200, ν=0.3

**任务**：创建 `week11/weekend_elastic_plate.py`

**Step 1**：用 `build_regular_mesh(nx=5, ny=3)` 生成网格

**Step 2**：构造节点特征：`(x, y, is_fixed, F_applied)`
- `is_fixed`：0 或 1，标记该节点是否在左端
- `F_applied`：如果在右端，设为 100；否则为 0

**Step 3**：构造边索引和边特征（同 Day 4）

**Step 4**：构造"模拟位移标签"
- 对于真实的弹性杆拉伸，位移沿 x 线性变化，y 方向也有泊松效应
- 这里用一个简化模型：`ux(x, y) = 0.01 * x`，`uy(x, y) = -0.003 * x * (y - 1)` （粗糙近似，不要求物理完全正确）
- y 的 shape 为 (15, 2)，每行是 (ux, uy)

**Step 5**：封装成 `Data` 对象，并可视化

**可视化要求**：创建 2×1 subplot
- 上图：画出网格 + 节点特征（用颜色区分 `is_fixed` 和 `F_applied`）
- 下图：用 matplotlib 的 `quiver` 画出每个节点的位移箭头

```python
# quiver 示例
plt.quiver(nodes[:, 0], nodes[:, 1], 
           y[:, 0].numpy(), y[:, 1].numpy(),
           scale=1, angles='xy', scale_units='xy')
```

**验收标准**：
- 图能生成，左端固定节点明显（位移箭头应该为 0）
- 右端节点的位移箭头应指向 +x 方向
- 能说清楚："这就是一个最简单的 FEM-style 数据对象，后面所有 GNN 训练都在类似的数据上做"

---

### Week 11 完成标准

- [ ] 理解图数据的三种表示（邻接矩阵 / 边列表 / edge_index）
- [ ] 能从零写 "规则网格 → 图" 转换函数（`build_regular_mesh` + `mesh_to_edges`）
- [ ] 能用 PyG 的 `Data` 对象封装一个完整图（含节点特征 + 边特征 + 标签）
- [ ] 能用 Matplotlib 可视化图
- [ ] 能说清楚：为什么要把 FEM 数据用图表示而不是扁平向量

---

---

## Week 12: GCN——最简单的图卷积

> **本周定位说明**：Week 12 是 **基础对照周**，不是你方向的主力模型。
>
> 论文 §2.1.2 明确说 GCN 有局限（需要固定节点数、全图更新、所有邻居等权聚合），后续采用 GraphSAGE 并增加了边更新机制。**GraphSAGE + 带边更新的消息传递才是你真正要掌握的主角**（Week 13 和 Week 14）。
>
> 为什么仍然要用一整周学 GCN？因为 GraphSAGE 的设计动机（"为什么要把节点自身特征和邻居特征分开"）建立在你**理解 GCN 的局限**之上。跳过 GCN 直接学 GraphSAGE，你会失去那个"啊，原来 GraphSAGE 是在解决 GCN 的这个问题"的对比感，后面学起来会是死记硬背。
>
> 所以本周的目标是：**会用 GCN，建立消息传递的第一直觉**，而不是"精通 GCN 公式推导"。如果 Day 2 的 NumPy 手写你写出来就跑通了，不用花太多时间反复练；重点放在 Day 3-4 的 PyG 使用和 Day 5-6 的多图训练上。

**衔接**：Week 11 你能构造图数据了，但图只是"数据结构"，和神经网络没关系。这周开始把神经网络作用到图上——从最简单的 GCN（图卷积网络）开始。

**本周目标**：
- 理解 GCN 的核心思想：每个节点的新特征 = 邻居特征的加权平均 + 自己
- 能从零用 NumPy 实现 GCN 的一层计算（验证公式）
- 能用 PyG 的 `GCNConv` 搭建 GCN 模型
- 能在 Week 11 的弹性杆数据上训练 GCN 做简单回归

**主要资源**：
- 论文 §2.1.2（GCN 公式 2.2 和 2.3）反复读
- CS224W Lecture 7（GCN 讲解）
- PyG 官方 `GCNConv` 文档

---

### Day 1 | GCN 的核心思想

**理论任务**（约 1.5 小时）：

**最直观的理解**：一个节点 i 的新特征 = 它所有邻居 + 它自己的特征，**加权平均**，再通过一个线性变换和激活。

**数学公式**（论文 §2.1.2 式 2.2）：
```
H = σ(D^(-1/2) · Â · D^(-1/2) · X · W)
```

这个公式看起来吓人，拆成小块就不难。

**Step 1：为什么要 Â = A + I？**
- `A` 是原始邻接矩阵，但如果只用 A，节点自己的特征不会传到下一层——它只看邻居，忘了自己
- `Â = A + I` 加上单位矩阵 I，让每个节点也连自己（自环），这样节点自己的特征也被包含

**Step 2：为什么要 D^(-1/2) · Â · D^(-1/2)？**
- 不做归一化直接 `Â · X`，等于每个节点把**所有邻居的特征相加**
- 度数高的节点（有很多邻居）加完后值会变得特别大，度数低的节点值小，不均衡
- `D^(-1/2) · Â · D^(-1/2)` 相当于按度数做归一化，让每个节点的"邻居聚合"结果量级可比

**Step 3：X · W 是什么？**
- X 是所有节点的特征矩阵，shape=(N, in_features)
- W 是一层的权重矩阵，shape=(in_features, out_features)
- X · W 是把每个节点的特征通过线性变换映射到新维度

**Step 4：σ 是什么？**
- 激活函数，通常用 ReLU
- 注意：GCN 没有物理约束 loss 的需求，用 ReLU 没问题；只有 PINN 风格的网络才必须用 Tanh（因为要对输入求二阶导数）

**在笔记上用自己的话写一段**：GCN 的一层在做什么？

参考答案：
> "对每个节点，先算出它和邻居的特征加权平均（权重由归一化的邻接矩阵决定），然后把这个平均后的特征通过一个线性层映射到新的特征空间，最后过激活函数。这个过程相当于每个节点'听了一遍邻居的意见'后更新自己的表示。"

---

### Day 2 | 从零用 NumPy 实现一层 GCN

**实践任务**（约 2 小时）：创建文件 `week12/day02_gcn_numpy.py`

**目标**：手动用 NumPy 实现 GCN 公式，对照论文式 2.2 验证正确性。

```python
import numpy as np

def gcn_layer_numpy(X, A, W):
    """
    GCN 一层计算（NumPy 实现）
    
    参数：
      X: 节点特征矩阵，shape=(N, in_features)
      A: 邻接矩阵（不含自环），shape=(N, N)
      W: 权重矩阵，shape=(in_features, out_features)
    返回：
      H: 新节点特征，shape=(N, out_features)
    """
    N = A.shape[0]
    
    # Step 1: 加自环
    A_hat = A + np.eye(N)
    
    # Step 2: 算度矩阵
    D_hat = np.diag(A_hat.sum(axis=1))
    
    # Step 3: 算 D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1)))
    
    # Step 4: 归一化邻接矩阵
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    
    # Step 5: 消息聚合 + 特征变换
    H = A_norm @ X @ W
    
    # Step 6: ReLU 激活
    H = np.maximum(0, H)
    
    return H


# ===== 测试 =====
# 一个 4 节点简单图：0-1, 0-2, 1-3
A = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)

# 每个节点 3 个特征
X = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0]
], dtype=np.float32)

# 映射到 2 个输出特征
np.random.seed(0)
W = np.random.randn(3, 2).astype(np.float32)

H = gcn_layer_numpy(X, A, W)
print(f"输入 X shape: {X.shape}")
print(f"权重 W shape: {W.shape}")
print(f"输出 H shape: {H.shape}")     # 应为 (4, 2)
print(f"H:\n{H}")

# 验证 1: 节点 2（度数 1，只连 0）的新特征
# A_hat[2] = [1, 0, 1, 0]（加上自环 [0,0,1,0] + [1,0,0,0]）
# 归一化后，节点 2 从节点 0 和自己聚合
# 聚合后特征 = 系数 * (X[0] + X[2]) = ...
```

**验收标准**：
- 代码能跑通，输出 shape 正确
- 能解释每一步的目的（你需要在注释中把 Step 1-6 的目的写出来）

---

### Day 3 | 用 PyG 的 `GCNConv` 搭建真正的 GCN 模型

**实践任务**（约 2 小时）：创建文件 `week12/day03_gcn_pyg.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class SimpleGCN(nn.Module):
    """两层 GCN + 一个线性输出头"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.head(x)


# ===== 测试：用 Week 11 的弹性杆数据 =====
# 先构造一个简化版数据（或从 Week 11 Weekend 导入）
edge_index = torch.tensor([
    [0, 1, 1, 0, 1, 2, 2, 1, 2, 3, 3, 2],
    [1, 0, 0, 1, 2, 1, 1, 2, 3, 2, 2, 3]
], dtype=torch.long)
x = torch.randn(4, 5)      # 4 个节点，5 个特征
y = torch.randn(4, 2)      # 每个节点预测 2 个值（ux, uy）

data = Data(x=x, edge_index=edge_index, y=y)

model = SimpleGCN(in_channels=5, hidden_channels=16, out_channels=2)
print(model)
print(f"参数数: {sum(p.numel() for p in model.parameters())}")

# 前向测试
y_pred = model(data)
print(f"输出 shape: {y_pred.shape}")   # 应为 (4, 2)
```

**验收标准**：
- 模型能跑通前向
- 输出 shape 正确
- 能说清楚模型结构（两层 GCN + 一个线性头）

---

### Day 4 | 在弹性杆数据上训练 GCN

**实践任务**（约 2 小时）：创建文件 `week12/day04_train_gcn_bar.py`

**Step 1**：用 Week 11 Weekend 的 `weekend_elastic_plate.py` 的代码构造数据（把那个代码封装成函数）

**Step 2**：GCN 训练（单图训练，先不考虑多样本 batch）

```python
from week12.day03_gcn_pyg import SimpleGCN
from week11.weekend_elastic_plate import build_plate_data   # 假设你封装了

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)
data = build_plate_data()    # 返回 Data 对象

model = SimpleGCN(
    in_channels=data.num_node_features,     # 4
    hidden_channels=32,
    out_channels=2                           # ux, uy
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

losses = []
for epoch in range(500):
    y_pred = model(data)
    loss = criterion(y_pred, data.y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())
    if epoch % 50 == 0:
        print(f"epoch {epoch}: loss={loss.item():.6f}")

# 可视化：预测位移 vs 真实位移
y_pred = model(data).detach().numpy()
y_true = data.y.numpy()
nodes = data.x[:, :2].numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, disp, title in zip(axes, [y_true, y_pred], ['True', 'Predicted']):
    ax.quiver(nodes[:, 0], nodes[:, 1], disp[:, 0], disp[:, 1],
              angles='xy', scale_units='xy', scale=0.1)
    ax.scatter(nodes[:, 0], nodes[:, 1], c='red', s=50)
    ax.set_aspect('equal')
    ax.set_title(f'{title} Displacement')

plt.tight_layout(); plt.savefig('week12_gcn_train.png')
```

**注意事项**：
- 这里只有一张图、一个样本，GCN 会"过拟合"到这个图——这是正常的，因为我们目的是验证模型能跑通
- 实际项目会有多个样本（多个不同载荷/材料下的 FEM 样本），那时才谈得上真正的"学习"

**验收标准**：
- 代码能跑通
- Loss 能下降到 < 0.001
- 预测位移图和真实位移图在肉眼上相似

---

### Day 5–6（周末）| 巩固 + 多图数据集初步体验

**任务**：生成 3-5 张不同载荷的板数据，把它们组成一个"数据集"

**Step 1**：把 Week 11 的 `build_plate_data` 改造成能接受载荷参数 F 的函数
```python
def build_plate_data(F_applied=100.0, nx=5, ny=3, seed=None):
    # 根据 F_applied 改变右端节点的 F 特征和对应的目标位移
    ...
    return Data(x=..., edge_index=..., y=...)
```

**Step 2**：创建 5 张图，F 值分别为 50, 75, 100, 125, 150
```python
datasets = [build_plate_data(F_applied=F) for F in [50, 75, 100, 125, 150]]
```

**Step 3**：用 PyG 的 `DataLoader` 做 batch 训练
```python
from torch_geometric.loader import DataLoader

train_data = datasets[:4]
val_data = datasets[4:]

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)

# 训练循环
for epoch in range(300):
    for batch in train_loader:
        y_pred = model(batch)       # PyG 会自动把 batch 里的多个图拼接
        loss = criterion(y_pred, batch.y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**关键点**：PyG 的 DataLoader 会把多个小图**拼成一张大图**（batching 机制），`edge_index` 会自动偏移节点编号。这是 GNN 的 batch 机制和 MLP 不一样的地方——MLP 的 batch 是在第一维堆叠，GNN 的 batch 是**图的并集**。

**验收标准**：
- 多图训练能跑通
- 验证集 Loss < 训练集 Loss 的 2 倍（说明有一定泛化能力）
- 能理解 `batch` 对象里的 `batch.batch` 属性（标记每个节点属于哪张图）

---

### Week 12 完成标准

- [ ] 能解释 GCN 公式每一项的意义（加自环、归一化、特征变换、激活）
- [ ] 能用 NumPy 实现一层 GCN 计算
- [ ] 能用 PyG 的 `GCNConv` 搭建完整 GCN 模型
- [ ] 能在单图/多图数据上训练 GCN
- [ ] 理解 PyG 的多图 batch 机制

---

---

## Week 13: 消息传递范式

> **密集区提醒（Week 13-15）**：接下来三周是 Part C 的技术密集区，每周都在引入新的模型结构和实验。如果你感觉某一周的内容两周才消化得动，**这是正常的**，按实际进度走，不要硬推。进入 Week 16 之前，要求是 Week 13-15 的核心模块都能独立跑通，不要求"每周准时完成"。
>
> 本阶段典型的"吃力信号"：
> - 写 `MessagePassing` 自定义层时，`x_i / x_j / edge_attr` 三个参数的传递关系搞不清楚
> - 带边更新层写出来跑不通，不知道怎么调试
> - 训练 loss 不降或震荡，分不清是代码错还是超参问题
>
> 遇到以上任一情况，停下来先解决，**不要跳**。下周的内容都建立在本周的模块能跑通的基础上。

**衔接**：Week 12 的 GCN 是一个**特殊的消息传递形式**——所有邻居以固定权重（按度归一化）被聚合。但 FEM 问题中，我们希望更灵活地控制"邻居怎么传信息到当前节点"。

PyG 提供了一个通用抽象：**MessagePassing**。它把 GNN 的计算分成三步：
1. **message**：每条边上怎么生成消息（从 source 节点到 target 节点）
2. **aggregate**：一个节点如何聚合所有收到的消息（mean / sum / max 等）
3. **update**：节点用聚合后的消息更新自己的特征

论文 §2.3.3 的消息传递和更新模块就是用这个抽象。本周学习这个抽象，为 Week 14-15 实现论文里的"带边更新的消息传递"做准备。

**本周目标**：
- 理解消息传递的三阶段范式
- 能用 `MessagePassing` 基类写一个自定义的图卷积层
- 理解 GCN 其实是消息传递的一个特例

---

### Day 1 | 消息传递的三阶段

**理论任务**（约 1 小时）：

**用比喻理解**：把图想象成社交网络。每个节点是一个人，边是朋友关系。
1. **Message**：每个人对每个朋友说一句话（消息内容可能取决于你自己的状态、朋友的状态、和你们的关系）
2. **Aggregate**：每个人把所有收到的话"归纳"一下（比如取平均、取重要的那句）
3. **Update**：每个人根据归纳结果更新自己的想法

**对应到 GCN**：
- Message：每条边上，source 节点的特征 × 归一化系数 → 一条消息
- Aggregate：一个节点把所有入边消息求和
- Update：聚合结果通过线性层 + ReLU 输出新特征

**论文公式对照**（§2.3.3 式 2.14–2.17）：
```
聚合输入边特征和节点特征：  {e_i, {n_s, n_k}}
更新边属性：                e'_i = MLP(e_i, n_s, n_k)
聚合节点和更新后的边：      {n_k, {e_i, e_j, e_k, e_l}}
更新节点属性：              n'_k = MLP(n_k, e_i, e_j, e_k, e_l)
```

这就是一个**带边更新**的消息传递。本周先学基本形式（Week 14 学边更新）。

---

### Day 2 | PyG `MessagePassing` 基类

**实践任务**（约 2.5 小时）：创建文件 `week13/day02_message_passing_basic.py`

**目标**：从零写一个自定义的图卷积层，复现 GCN 的功能（或者说一个简化版本）。

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

class SimpleGNNLayer(MessagePassing):
    """
    一个最简单的 GNN 层：
    - message：直接传递 source 节点的特征（不加权）
    - aggregate：用 mean 聚合
    - update：过一个线性层
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')       # 指定聚合方式
        self.lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        x: 节点特征 shape=(N, in_channels)
        edge_index: shape=(2, n_edges)
        """
        # propagate 会自动调用 message / aggregate / update
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        """
        定义"从 source 到 target 的消息"
        x_j: source 节点的特征（PyG 自动从 x 和 edge_index 提取），
             shape=(n_edges, in_channels)
        
        这里最简单：直接把 source 特征作为消息
        """
        return x_j
    
    def update(self, aggr_out):
        """
        aggregate 后的结果通过这里的处理变成新节点特征
        aggr_out: shape=(N, in_channels)
        """
        return torch.relu(self.lin(aggr_out))


# ===== 测试 =====
torch.manual_seed(0)
x = torch.randn(4, 3)        # 4 个节点，3 个特征
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3],
    [1, 0, 2, 1, 3, 2]
], dtype=torch.long)

layer = SimpleGNNLayer(3, 8)
out = layer(x, edge_index)
print(f"输出 shape: {out.shape}")     # 应为 (4, 8)
print(f"输出:\n{out}")
```

**命名约定（非常重要）**：
- `x_j`：代表 source 节点（"j" 是约定俗成的 source 命名）
- `x_i`：代表 target 节点
- 如果 message 函数里同时需要 source 和 target 的特征，函数签名写成 `def message(self, x_i, x_j)`
- 如果需要边特征，传入 `edge_attr` 参数：`def message(self, x_j, edge_attr)`

**验收标准**：
- 代码能跑通
- 能解释 `aggr='mean'` 是什么意思
- 能说出 `x_j` 和 `x_i` 的含义

---

### Day 3 | 给 message 加入更复杂的逻辑

**实践任务**（约 2 小时）：创建文件 `week13/day03_complex_message.py`

**例子 1**：message 同时用到 source 和 target 节点
```python
class BothNodesLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        # 注意 in_channels * 2，因为 message 拼接了两个节点的特征
        self.lin = nn.Linear(in_channels * 2, out_channels)
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        # source 和 target 都用到：拼接
        cat = torch.cat([x_i, x_j], dim=-1)   # shape=(n_edges, 2*in_channels)
        return torch.relu(self.lin(cat))
    
    def update(self, aggr_out):
        return aggr_out
```

**例子 2**：message 用到边特征
```python
class EdgeAwareLayer(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels + edge_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # 拼接 source 特征和边特征
        cat = torch.cat([x_j, edge_attr], dim=-1)
        return torch.relu(self.lin(cat))
```

**测试两个层**：用 Week 11 的弹性板数据测试，确认能跑通。

**验收标准**：
- 两个自定义层都能前向传播，shape 正确
- 能说清楚：加入 `x_i` 让 message 知道"接收者是谁"；加入 `edge_attr` 让 message 知道"这条边有什么特点"

---

### Day 4 | GraphSAGE 的思想（论文用的主要模型）

**为什么论文用 GraphSAGE 不用 GCN**：

论文 §2.1.2 明确说：

> "GCN 适用于每次更新都需要更新全图节点特征的情况，同时更新过程中所有节点都具有相同的权重。除此之外，GCN 还需要固定的节点数量和邻接矩阵... GraphSAGE 被用于解决这些问题和局限性。"

具体差异：
| | GCN | GraphSAGE |
|---|---|---|
| 是否需要完整邻接矩阵 | 需要 | 不需要（可以只采样部分邻居） |
| 是否能处理不同节点数的新图 | 困难 | 能（归纳学习） |
| 节点自身特征的处理 | 被邻居聚合模糊 | 明确保留（拼接而不是相加） |

**GraphSAGE 的消息传递**（简化版）：
```
Message:    m_j = x_j            （source 特征，简单传过去）
Aggregate:  aggr_i = mean(m_j for j in N(i))    （邻居特征取均值）
Update:     x_i' = W · concat(x_i, aggr_i)       （把自己的特征和邻居平均拼起来做线性变换）
```

**关键点**：GraphSAGE 把"节点自己的特征"和"邻居的聚合"**分开**（拼接），而不是像 GCN 一样混在一起（通过 A + I）。这让节点身份信息更明确。

**理论任务**：用自己的话对比 GCN 和 GraphSAGE 的机制差异（写在笔记上）。

---

### Day 5–6（周末）| 用 PyG 的 `SAGEConv` 训练 GraphSAGE

**实践任务**：创建文件 `week13/weekend_graphsage.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.head = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return self.head(x)


# 用 Week 12 Weekend 的多图数据集
from week12.weekend import build_plate_data
datasets = [build_plate_data(F_applied=F) for F in [50, 75, 100, 125, 150, 175, 200]]
train_data = datasets[:5]
val_data = datasets[5:]

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)

model = GraphSAGE(in_channels=4, hidden_channels=32, out_channels=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []
for epoch in range(500):
    # Train
    model.train()
    total_loss = 0
    for batch in train_loader:
        y_pred = model(batch)
        loss = criterion(y_pred, batch.y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    train_losses.append(total_loss / len(train_data))
    
    # Val
    model.eval()
    total_val = 0
    with torch.no_grad():
        for batch in val_loader:
            y_pred = model(batch)
            total_val += criterion(y_pred, batch.y).item() * batch.num_graphs
    val_losses.append(total_val / len(val_data))

# 对比 GCN 和 GraphSAGE 的验证集 loss
```

**对比实验**：用 Week 12 的 GCN 模型和 GraphSAGE 分别跑，哪个 val loss 更低？

**验收标准**：
- 两个模型都能跑通
- GraphSAGE 的验证 loss 通常比 GCN 略低（因为节点自身特征保留得更好）
- 能通过 Loss 曲线判断训练质量

---

### Week 13 完成标准

- [ ] 理解消息传递的三阶段（message / aggregate / update）
- [ ] 能用 `MessagePassing` 基类写自定义图卷积层
- [ ] 理解 `x_i`, `x_j`, `edge_attr` 的含义和用法
- [ ] 能用 `SAGEConv` 搭建 GraphSAGE 模型
- [ ] 能说清楚 GCN 和 GraphSAGE 的核心差异

---

---

## Week 14: 带边更新的消息传递（论文核心机制）

**衔接**：Week 13 的消息传递只更新了节点特征，边特征是固定的（从 Day 1 设置后不变）。但论文 §2.3.3 的 PhyFENet 做了关键改动——**边特征也在每层被更新**。

论文原文：

> "本文针对模拟有限元和固体力学问题增加了边特征更新模块。"
> "消息传递步骤由四个阶段定义：
> （1）遍历所有边，聚合输入的边信息和边所连接的节点信息
> （2）使用聚合后的特征作为输入，通过全连接层更新边属性
> （3）遍历所有点，聚合节点属性和更新后每个点有关的边属性
> （4）使用聚合后的特征作为输入，通过全连接层更新点属性"

**为什么这个改动重要**：在 FEM 中，"边"代表两个节点之间的几何/物理关系（距离、应力传递通道）。让边特征也能"学习"，可以捕捉更丰富的局部物理关系。

**本周目标**：
- 实现论文描述的带边更新的消息传递
- 理解这个机制相对于标准 GraphSAGE 的优势

---

### Day 1–2 | 实现带边更新的消息传递

**理论任务**（约 1 小时）：在笔记上画出论文 Fig. 2.7 所示的四步消息传递示意图。

**关键点**：
- 边更新依赖于：当前边的特征 + 连接的两个节点的特征
- 节点更新依赖于：当前节点的特征 + 相邻边**更新后**的特征

**实现策略**：
- PyG 的 `MessagePassing` 基类设计上主要更新节点
- 边更新需要我们自己在外部先算好新边特征，再喂进 propagate

**实践任务**（Day 1 约 2 小时，Day 2 约 2 小时）：创建文件 `week14/day12_edge_update.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class EdgeUpdateMP(MessagePassing):
    """
    论文 §2.3.3 的带边更新消息传递
    
    每层做：
    1. 更新边特征：e' = MLP_edge(e, x_source, x_target)
    2. 节点聚合：aggr_i = mean(e'_ij for j in N(i))
    3. 更新节点特征：x' = MLP_node(x, aggr)
    """
    def __init__(self, node_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')
        # 边更新 MLP：输入 = edge_attr + x_src + x_dst
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_channels + 2 * node_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        # 节点更新 MLP：输入 = 原始节点特征 + 聚合消息
        self.node_mlp = nn.Sequential(
            nn.Linear(node_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        x: shape=(N, node_channels)
        edge_index: shape=(2, n_edges)
        edge_attr: shape=(n_edges, edge_channels)
        """
        # Step 1: 更新边特征
        src, dst = edge_index[0], edge_index[1]
        x_src = x[src]        # shape=(n_edges, node_channels)
        x_dst = x[dst]
        edge_input = torch.cat([edge_attr, x_src, x_dst], dim=-1)
        new_edge_attr = self.edge_mlp(edge_input)    # shape=(n_edges, out_channels)
        
        # Step 2 & 3: 用更新后的边作为消息，聚合并更新节点
        # 这里用 propagate，它会自动调用 message 和 update
        new_x = self.propagate(edge_index, x=x, edge_msg=new_edge_attr)
        
        return new_x, new_edge_attr
    
    def message(self, edge_msg):
        """消息就是更新后的边特征"""
        return edge_msg
    
    def update(self, aggr_out, x):
        """用原始节点特征 + 聚合消息 更新节点"""
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))


# ===== 测试 =====
torch.manual_seed(0)
N = 5
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
    [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
], dtype=torch.long)

x = torch.randn(N, 4)                      # 节点特征 4 维
edge_attr = torch.randn(edge_index.shape[1], 3)   # 边特征 3 维

layer = EdgeUpdateMP(node_channels=4, edge_channels=3, out_channels=16)
new_x, new_edge_attr = layer(x, edge_index, edge_attr)

print(f"new_x shape: {new_x.shape}")                # (5, 16)
print(f"new_edge_attr shape: {new_edge_attr.shape}")  # (10, 16)
```

**验收标准**：
- 代码跑通
- 节点特征从 4 维变到 16 维
- 边特征从 3 维变到 16 维
- 能解释每一行代码的作用（Day 2 专门做这个）

---

### Day 3 | 把多层带边更新组合成完整模型

**实践任务**（约 2 小时）：创建文件 `week14/day03_full_mp_model.py`

```python
import torch
import torch.nn as nn
from week14.day12_edge_update import EdgeUpdateMP

class MPNet(nn.Module):
    """多层消息传递网络，每层都做节点+边特征更新"""
    
    def __init__(self, in_node_dim, in_edge_dim, hid_dim, out_dim, n_layers=3):
        super().__init__()
        
        # 输入编码（第一层用 input 维度，后续用 hid 维度）
        self.layers = nn.ModuleList()
        self.layers.append(EdgeUpdateMP(in_node_dim, in_edge_dim, hid_dim))
        for _ in range(n_layers - 1):
            self.layers.append(EdgeUpdateMP(hid_dim, hid_dim, hid_dim))
        
        # 输出头（把最终节点特征映射到预测值）
        self.head = nn.Linear(hid_dim, out_dim)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
            # 这里可以加残差连接，但先用简单版本
        
        return self.head(x)


# ===== 测试：在弹性板数据上训练 =====
from week11.weekend_elastic_plate import build_plate_data

datasets = [build_plate_data(F_applied=F) for F in [50, 75, 100, 125, 150]]
# ... 训练循环同 Week 13 Weekend
```

**验收标准**：
- 模型能在弹性板数据上跑通训练
- val loss 能下降到一个合理水平

---

### Day 4 | 对比三种模型：GCN / GraphSAGE / MPNet

**实践任务**（约 2 小时）：创建文件 `week14/day04_three_models_comparison.py`

**实验目的**：对比三种 GNN 模型的训练行为和最终性能，**不是**要证明某种模型"必然更好"。三者各有特点，小合成数据上谁赢谁输受很多因素影响（初始化、学习率、隐藏维度、训练轮数等）。

**记录表格**：

| 模型 | 用不用边特征 | 最终 Train MSE | 最终 Val MSE | 训练稳定性（曲线震荡程度） |
|-----|------------|---------------|-------------|------------------------|
| GCN | 不用 | ? | ? | ? |
| GraphSAGE | 不用 | ? | ? | ? |
| MPNet (边更新) | 用 | ? | ? | ? |

**可视化**：画一张 Loss 曲线图，三条 Val Loss 曲线对比。

**分析任务**（写在脚本末尾的注释）：回答以下 3 个问题，用你看到的实际数据支撑：
1. 三种模型的最终 Val MSE 差异是否显著？如果差异不大，你觉得原因是什么（比如合成数据太简单）？
2. 哪个模型的训练曲线最稳定？哪个震荡最大？
3. MPNet 利用了边特征。在这份合成数据上，边特征是否真的提供了额外信息？（注意你用的边特征是 [dx, dy, dist]——这些信息其实也能从节点坐标算出，所以可能带来的提升有限。）

**验收标准**：
- 三个模型都能跑通，Loss 曲线都能下降
- 能完成上面 3 个分析问题的讨论
- **不要求特定的模型排名**——重点是"你能分析实验结果"，而不是"结果必须符合某个预设"

**为什么这样设计验收**：这是你第一次做模型对比实验。真实工程中，"实验结果不符合预期"是常态，**分析能力比预设结果更重要**。面试时被问"你的模型为什么比 baseline 好"，能说清楚"在哪些情况下好、哪些情况下差、为什么"——这才是面试官想听的。

---

### Day 5–6（周末）| 过平滑问题与残差连接

**核心问题**：GNN 层数太多时，会出现**过平滑**（over-smoothing）——所有节点的特征趋同，失去区分度。

**原因**：每层消息传递都把邻居特征"混合"进来，层数越多，节点和邻居越像。极端情况，所有节点特征都收敛到一样的值。

**解决方案**：残差连接（让深层网络能保留浅层信息）
```python
h = x                                    # 初始输入
for layer in layers:
    h_new = layer(h, edge_index)
    h = h + h_new                         # 残差连接
```

**实践任务**：创建文件 `week14/weekend_oversmoothing.py`

**实验 1**：观察过平滑
- 用 MPNet，分别测试 1, 2, 4, 8 层
- 每种层数训练 500 epoch，记录 Val Loss
- 额外指标：计算"节点特征相似度"——所有节点特征的平均余弦相似度（越接近 1 越"过平滑"）

**实验 2**：加残差连接
- 同样测试 2, 4, 8 层
- 观察：加残差后，深层模型的表现是否改善

**可视化**：一张 2x2 subplot
- (1,1)：Val Loss 随层数变化（两条线：有残差 vs 无残差）
- (1,2)：节点相似度随层数变化
- (2,1)：无残差的 8 层模型的训练曲线
- (2,2)：有残差的 8 层模型的训练曲线

**验收标准**：
- 能观察到过平滑现象（层数多时 Val Loss 反而变差 / 节点相似度接近 1）
- 加残差连接后，8 层模型能正常训练

**工程结论**（PhyFENet 论文的经验）：
- GNN 通常 2-4 层足够
- 深层网络必须加残差连接

---

### Week 14 完成标准

- [ ] 能从零实现论文 §2.3.3 的带边更新消息传递层
- [ ] 理解 GCN / GraphSAGE / MPNet 的差异
- [ ] 理解过平滑问题及解决方法
- [ ] 能在合成数据上训练 3-4 层的 MPNet

---

---

## Week 15: 编码器-处理器-解码器架构（论文 PhyFENet 主框架）

**衔接**：Week 12-14 你已经学完了 GNN 的核心组件（GCN / GraphSAGE / 带边更新的 MP）。但论文的 PhyFENet 框架不只是"一堆 GNN 层堆叠"，它有完整的**编码器-处理器-解码器**架构。

论文 §2.3 原文：

> "框架包括四个部分：
> 1）编码器：负责处理输入数据，编码为图
> 2）特征学习模块：负责完成消息传递和更新，学习数据特征
> 3）解码器：负责实现特征空间到输出数据的转换，输出预测结果
> 4）输出数据处理模块：主要负责配置损失函数"

**为什么要这样设计**：
- 编码器把各种"原始输入"（坐标、材料参数、边界条件）统一编码到一个潜在特征空间
- 处理器在这个统一空间中做多步消息传递
- 解码器从潜在特征空间映射回"我们关心的物理量"（位移、应变、应力）

这是 MeshGraphNet（DeepMind, 2021）推广的标准架构，也是论文沿用的。

**本周目标**：搭建完整的编码器-处理器-解码器架构；这是后续第二阶段处理真实 FEM 数据的基石。

---

### Day 1–2 | 三模块架构的设计

**理论任务**（约 1 小时）：在笔记上画架构图

```
[节点原始特征 (N, n_node_raw)]    [边原始特征 (E, n_edge_raw)]
         ↓                                  ↓
  [节点编码器 Linear]               [边编码器 Linear]
         ↓                                  ↓
  [节点潜在特征 (N, hid)]          [边潜在特征 (E, hid)]
         ↓                                  ↓
  [消息传递块 1 (带边更新)]  ←————— 连接 ——————→
         ↓                                  ↓
  [消息传递块 2]  ←——— 连接 ———→
         ↓
      ...（共 n_layers 层）
         ↓
  [节点潜在特征 (N, hid)]
         ↓
  [节点解码器 MLP]
         ↓
  [输出 y_pred (N, n_out)]
```

**关键细节**：
- 输入编码器：把"原始 3-4 维节点特征"升到"hid 维潜在空间"
- 所有消息传递层都在 hid 维空间里做（维度统一，便于残差连接）
- 输出解码器：把 hid 维特征映射到预测维度（比如 ux, uy = 2 维）

---

**实践任务 Day 1**（约 2 小时）：创建文件 `week15/day12_encoder_processor_decoder.py`

```python
import torch
import torch.nn as nn
from week14.day12_edge_update import EdgeUpdateMP

class PhyFENet_Mini(nn.Module):
    """
    简化版 PhyFENet：编码器 → 处理器 → 解码器
    
    参数：
      node_in: 输入节点特征维度（比如 4: x, y, E, F）
      edge_in: 输入边特征维度（比如 3: dx, dy, dist）
      hid: 隐藏维度
      node_out: 输出节点维度（比如 2: ux, uy）
      n_mp_layers: 消息传递层数
    """
    def __init__(self, node_in, edge_in, hid, node_out, n_mp_layers=3):
        super().__init__()
        
        # === 编码器 ===
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )
        
        # === 处理器：n_mp_layers 个消息传递块 ===
        self.processor = nn.ModuleList([
            EdgeUpdateMP(hid, hid, hid) for _ in range(n_mp_layers)
        ])
        
        # === 解码器 ===
        self.decoder = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, node_out)
        )
    
    def forward(self, data):
        # 编码
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        
        # 处理（带残差连接）
        for layer in self.processor:
            x_new, edge_attr_new = layer(x, data.edge_index, edge_attr)
            x = x + x_new                      # 残差
            edge_attr = edge_attr + edge_attr_new
        
        # 解码
        return self.decoder(x)


# ===== 测试 =====
from week11.weekend_elastic_plate import build_plate_data

torch.manual_seed(0)
data = build_plate_data(F_applied=100.0)
model = PhyFENet_Mini(node_in=4, edge_in=3, hid=32, node_out=2, n_mp_layers=3)

y_pred = model(data)
print(f"y_pred shape: {y_pred.shape}")    # 应为 (15, 2)
print(f"参数总数: {sum(p.numel() for p in model.parameters())}")
```

**验收标准**：
- 模型能跑通前向
- shape 正确

---

**实践任务 Day 2**（约 2 小时）：完整训练实验

在多图数据集上训练 PhyFENet_Mini，对比 Week 14 的 MPNet（无编码器解码器结构，直接从 in_dim 做消息传递）。

实验结论：带编解码器的架构通常收敛更快、最终 Val Loss 更低。

**验收标准**：
- 能画出两条训练曲线对比
- PhyFENet_Mini 的最终 Val Loss 应略好于 MPNet

---

### Day 3 | 单元聚合模块（论文 §2.3.4）

**衔接你方向的关键点**：论文特别提出了"单元聚合模块"——因为 FEM 里应力定义在**单元**上，位移定义在**节点**上，需要一个机制从节点特征生成单元特征。

论文原文（§2.3.4，式 2.19-2.21）：

> "对于单元 n_i^e 的属性可以表示为组成该单元的节点属性集合。在图神经网络计算过程中，可以通过构造稀疏矩阵完成节点至单元属性的转换：X_e = C · X_n"

其中 C 是稀疏矩阵，每行对应一个单元，元素 C[i, p] = 1 如果节点 p 是单元 i 的组成节点。

**实践任务**（约 2 小时）：创建文件 `week15/day03_element_aggregation.py`

```python
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import coo_matrix

def build_element_aggregation_matrix(elements, num_nodes, normalize=True):
    """
    构造"节点特征 → 单元特征"的聚合矩阵 C（教学实现版本）
    
    参数：
      elements: list of lists，每个元素是一个单元的节点索引，例如 [[0,1,2,3], [1,2,4,5]]
      num_nodes: 图中节点总数
      normalize: 是否归一化（让聚合=平均而不是求和）
    返回：
      C: torch.Tensor shape=(num_elements, num_nodes)
    
    注意：本实现用 dense 矩阵存储，只适合小规模教学用图（节点 < 几千）。
    真实工程场景（百万节点）应该用稀疏矩阵 (torch.sparse_coo_tensor) 或 
    torch_scatter.scatter_mean 实现，避免内存爆炸。
    第二阶段处理真实 FEM 数据时会替换为稀疏实现。
    """
    rows, cols, vals = [], [], []
    for eid, nids in enumerate(elements):
        for nid in nids:
            rows.append(eid)
            cols.append(nid)
            # 归一化：每个单元的每个节点贡献权重 = 1/单元节点数
            val = 1.0 / len(nids) if normalize else 1.0
            vals.append(val)
    
    # 用 PyTorch 稀疏矩阵存储
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float)
    C = torch.sparse_coo_tensor(indices, values, 
                                 size=(len(elements), num_nodes)).to_dense()
    return C


def aggregate_nodes_to_elements(x_node, C):
    """
    x_node: shape=(N, features)
    C: shape=(E, N)
    返回 x_elem: shape=(E, features)
    """
    return C @ x_node


# ===== 测试 =====
# 一个 3×3 的规则网格，4 个单元
from week11.day02_regular_mesh_to_graph import build_regular_mesh
nodes, elements = build_regular_mesh(nx=3, ny=3)
N = len(nodes)

C = build_element_aggregation_matrix(elements, num_nodes=N)
print(f"C shape: {C.shape}")        # 应为 (4, 9)

# 假设节点特征
x_node = torch.randn(N, 5)
x_elem = aggregate_nodes_to_elements(x_node, C)
print(f"x_elem shape: {x_elem.shape}")   # 应为 (4, 5)

# 验证：单元 0 的特征应该是组成它的 4 个节点的特征平均
elem_0_nodes = elements[0]    # 比如 [0, 1, 4, 3]
expected = x_node[elem_0_nodes].mean(dim=0)
actual = x_elem[0]
print(f"验证: {torch.allclose(expected, actual)}")    # 应为 True
```

**验收标准**：
- 聚合矩阵构造正确
- 单元特征 = 组成节点特征的平均
- `torch.allclose(expected, actual)` 返回 True

---

### Day 4 | 把单元聚合集成到完整模型

**实践任务**（约 2.5 小时）：创建文件 `week15/day04_full_model.py`

扩展 PhyFENet_Mini，加入单元聚合后的"单元级输出"。

```python
class PhyFENet_WithElement(nn.Module):
    """
    完整版：节点级输出 (位移) + 单元级输出 (应变/应力占位)
    """
    def __init__(self, node_in, edge_in, hid, 
                 node_out_dim, elem_out_dim):
        super().__init__()
        # ... 编码器 + 处理器（同 day12）
        
        # 节点解码器（输出位移 ux, uy）
        self.node_decoder = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, node_out_dim)
        )
        # 单元解码器（未来输出应变、应力等）
        self.elem_decoder = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, elem_out_dim)
        )
    
    def forward(self, data, elem_aggr_matrix):
        # ... 编码 + 处理（同 day12）
        
        # 节点输出
        node_out = self.node_decoder(x)
        
        # 聚合成单元特征，再通过单元解码器
        x_elem = elem_aggr_matrix @ x     # 用 Day 3 的 C 矩阵
        elem_out = self.elem_decoder(x_elem)
        
        return node_out, elem_out
```

**注意**：这个模型现在有两个输出（节点级 + 单元级），训练时需要**两项 Loss 加权**：
```python
loss = w_node * criterion(node_pred, data.y_node) + w_elem * criterion(elem_pred, data.y_elem)
```

**验收标准**：
- 模型能跑通前向
- 两个输出的 shape 正确

---

### Day 5–6（周末）| Week 15 整理

**任务 1**：把本周的所有组件封装到 `utils/gnn_models.py`
- `PhyFENet_Mini`（节点输出版）
- `PhyFENet_WithElement`（节点 + 单元输出版）
- `build_element_aggregation_matrix`

**任务 2**：写一段简短的架构说明文档 `docs/phyfenet_architecture.md`
- 三模块架构（编码器/处理器/解码器）的目的
- 为什么要加单元聚合模块
- 本周模型和论文 §2.3 的对应关系

---

### Week 15 完成标准

- [ ] 能搭建编码器-处理器-解码器架构的 GNN
- [ ] 能实现残差连接防过平滑
- [ ] 能从零写单元聚合矩阵 C
- [ ] 能把节点级和单元级输出组合在一个模型里
- [ ] 能说清楚 PhyFENet 架构（论文 §2.3）和你的实现的对应关系

---

---

## Week 16: Part C 综合实战 + 自测

> **关于本周数据的重要说明**：
>
> 本周用的数据是**结构化合成数据**，不是严格的 FEM 解。数据生成公式（`ux = F*x/(E*A)`, `uy = -ν*F*y/(E*A)`, `ε = F/(E*A)`, `σ = F/A`）是一维弹性拉伸的粗糙近似——`ux` 接近正确，但 `uy` 的泊松收缩公式是简化版本，且节点位移和单元应力之间并不严格满足几何+本构关系的物理自洽性。
>
> **为什么仍然用这个合成数据**：
> 1. 目标是验证你的 GNN 架构（编码器-处理器-解码器 + 单元聚合）能**跑通**节点级和单元级双输出的训练
> 2. 真实 FEM 数据的获取涉及 FEniCS/LS-DYNA 的使用，这是第二阶段 Part E 的内容
> 3. 这里合成数据的粗糙是可控的——能让你专注在 GNN 架构本身，不被真实数据的复杂度干扰
>
> **合理预期**：
> - 模型能收敛（Loss 下降）
> - 节点和单元输出都在合理范围内
> - 但不要过度解读模型的绝对精度——它拟合的是一个粗糙近似，不是物理真相
>
> **不要说的话**：
> - "我的模型达到了 XX% 的 FEM 精度"——这不对，标签本身不是 FEM 真解
> - "这证明了 GNN 能替代 FEM"——不能，这只证明了架构能跑
>
> 第二阶段进入真实 FEM 数据后，上述数字和结论才有可解读的物理意义。

**本周定位**：Part C 的闭环周，整合前 5 周所有内容，在一个相对完整的合成数据集上训练 PhyFENet_WithElement。

---

### Day 1–2 | 构造一个相对完整的合成数据集

**任务**：基于规则网格生成多样本数据集
- 每个样本：5×3 规则网格
- 变量：弹性模量 E（随机采样 150-300）、拉力 F（随机采样 50-150）
- 标签（模拟）：
  - 节点位移：用简化公式近似 `ux = F*x/(E*A)`，`uy = -ν * F * y / (E*A)`（粗糙）
  - 单元应变：用 `ε = F/(E*A)` 近似
  - 单元应力：`σ = F/A` 近似

（这些都是粗糙近似，不需要物理精确——目的是让你的 GNN 有东西可以学。真实 FEM 数据第二阶段才会用。）

生成 50 个样本，40 个训练，10 个验证。

---

### Day 3–4 | 训练 PhyFENet_WithElement

完整训练循环，同时监控节点 loss 和单元 loss。

**评估指标**：
- 节点位移 MAE
- 单元应变 MAE
- 单元应力 MAE

---

### Day 5 | 可视化：预测 vs 真实

生成一张 3×2 subplot：
- (0, :)：节点位移场（真实 vs 预测，quiver plot）
- (1, :)：单元应变场（真实 vs 预测，颜色图）
- (2, :)：单元应力场（真实 vs 预测，颜色图）

---

### Day 6 | Part C 自测

**理论自测**（书面回答，每题 3-5 句）：
1. 图数据有哪三种表示？各自的优缺点？
2. GCN 公式中 `D^(-1/2) · Â · D^(-1/2)` 每一项的目的？
3. GraphSAGE 相对于 GCN 的改进是什么？
4. 消息传递范式的三个阶段？
5. PhyFENet 相对于标准 GraphSAGE 的关键改动？（边更新、单元聚合、编解码器架构）
6. 什么是过平滑？怎么解决？

**代码自测**（限时完成）：
- （20 min）从零写 `build_regular_mesh` + `mesh_to_edges`
- （30 min）从零写 PyG 的 `Data` 构造（含节点特征 + 边特征）
- （40 min）从零写带边更新的 `MessagePassing` 层
- （30 min）搭建完整的编码器-处理器-解码器架构

如果上述任何一项做不到，回到对应周补强。

---

### Week 16 完成标准 + Part C 总完成标准

**Part C 总完成标准**（进入 Part D 前）：

理论
- [ ] 能说清楚图数据的结构（节点 / 边 / 邻接矩阵）
- [ ] 能对比 GCN / GraphSAGE / 带边更新 MP 三种 GNN 的机制差异
- [ ] 能画出 PhyFENet 架构图（编码器 + 带边更新消息传递 + 解码器 + 单元聚合）
- [ ] 理解过平滑问题和残差连接

代码（限时完成，不看参考）：
- [ ] 能从零写 "规则网格 → 图" 转换
- [ ] 能用 PyG 构造 Data 对象
- [ ] 能从零写自定义 MessagePassing 层（带边更新）
- [ ] 能搭建编码器-处理器-解码器架构
- [ ] 能在多图数据集上训练 GNN

如果上述任何一项未达到，Part D 的"GNN + PINN 融合"会非常困难——因为那时需要在 GNN 模型的**基础上**加物理约束 Loss。地基不稳，加层就塌。

---

*下一段输出：第一阶段 Part D（Week 17–22）：PINN 与有限元数据入门*