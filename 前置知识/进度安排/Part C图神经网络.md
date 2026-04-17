# 🔷 Part C · 图神经网络（Week 10–14）

> **本部分在整个路线中的位置**：Part A 教会了你用 MLP 做回归，Part B 教会了你用 autograd 求导和物理约束。
> 但 MLP 有一个根本局限——它把每个样本当作一个**独立的固定长度向量**处理，
> 无法处理"节点之间有连接关系"的结构化数据（比如有限元网格）。
> Part C 要教你一种专门处理"图结构数据"的网络——GNN。
>
> **结束时你应达到的状态**：
> 能解释图数据的三个要素（节点 / 边 / 特征），能用 PyTorch Geometric 搭建 GCN 和 GraphSAGE 模型，
> 能在自建的力学图数据集上训练 GNN 预测节点位移，能自定义一个支持边特征的消息传递层，
> 并通过实验证明 GNN 比 MLP 更适合有空间关系的数据。
>
> **负荷标准**：同前，每周 10–14 小时。5 周，每周只攻一个核心概念。


---
---

## Week 10 · 图数据基础概念

> **本周目标**：理解"图"是什么数据结构、怎么用矩阵表示图、怎么用 PyTorch Geometric 创建图对象。
> 本周不涉及 GNN 模型（留到 Week 12），只把"图数据"本身搞清楚。

---

### Day 1 ｜ 什么是图

**理论目标**

- [ ] 🧠 L2 理解：图（Graph）是什么
  - **达标标准**：能说出"图由**节点**（Node）和**边**（Edge）组成——节点代表实体，边代表实体之间的关系。比如工程网格中网格点是节点、相邻点之间的连接是边"
  - **自测**：一张 5 个城市之间的航线图——城市是什么？航线是什么？（答案：城市=节点，航线=边）

- [ ] 🧠 L2 理解：图和其他数据类型的根本区别
  - **达标标准**：能填完以下表格并解释为什么 MLP 和 CNN 处理不了图

| 数据类型 | 结构特点 | 例子 | 适用网络 |
|---------|---------|------|---------|
| 表格 | 每个样本是固定长度向量，样本间无关联 | Excel 一行 | MLP |
| 图像 | 固定 H×W 规则网格，像素位置固定 | 256×256 照片 | CNN |
| **图** | 节点数可变，连接关系任意 | 有限元网格 | **GNN** |

  - **自测**：有限元网格的节点数量每个模型都一样吗？（答案：不一样——不同几何形状节点数不同，这就是为什么需要 GNN）

- [ ] 🧠 L1 知道：有向图 vs 无向图
  - **达标标准**：能说出"无向图中边没有方向（A 到 B = B 到 A）；工程网格通常用**无向图**"

**实践目标**

- [ ] 💻 今天不写代码。在纸上画一个图：5 个节点（标号 0–4），5 条无向边（自己连），确保图"连通"
  - **自测**：节点 2 连了几条边？这个数叫"节点 2 的度"

---

### Day 2 ｜ 邻接矩阵与度矩阵

**理论目标**

- [ ] 🧠 L3 推导 ⭐：邻接矩阵 A
  - **达标标准**：关上资料写出定义和例子

```
邻接矩阵 A 是 N×N 矩阵（N=节点数）：
  A[i][j] = 1  如果节点 i 和 j 之间有边
  A[i][j] = 0  如果没有边
无向图中 A 是对称矩阵（A[i][j] = A[j][i]）

例子（边为 0-1, 0-2, 1-2, 2-3, 3-4）：
    0  1  2  3  4
0 [ 0  1  1  0  0 ]
1 [ 1  0  1  0  0 ]
2 [ 1  1  0  1  0 ]
3 [ 0  0  1  0  1 ]
4 [ 0  0  0  1  0 ]
```

  - **自测**：从邻接矩阵看节点 2 的度？（答案：第 2 行求和=3）

- [ ] 🧠 L3 推导：度矩阵 D
  - **达标标准**：能写出"D 是对角矩阵，D[i][i]=节点 i 的度，非对角线全 0"

- [ ] 🧠 L2 理解：自连接 Â = A + I
  - **达标标准**：能说出"对角线加 1 让每个节点也'连接到自身'——后面聚合邻居特征时自己的特征也会被纳入"

**实践目标**

- [ ] 💻 L3 能默写 ⭐：关掉参考写出以下代码并通过所有 assert

```python
import numpy as np

edges = [(0,1), (0,2), (1,2), (2,3), (3,4)]
N = 5
A = np.zeros((N, N))
for i, j in edges:
    A[i][j] = 1; A[j][i] = 1

D = np.diag(A.sum(axis=1))
A_hat = A + np.eye(N)

# ---------- 验证 ----------
assert A[0][1] == 1 and A[1][0] == 1, "0-1 有边"
assert A[0][3] == 0, "0-3 无边"
assert (A == A.T).all(), "无向图对称"
assert D[2][2] == 3, "节点 2 度=3"
assert A_hat[0][0] == 1, "自连接后对角线=1"
print("邻接矩阵 A:\n", A)
print("度矩阵对角线:", np.diag(D))
print("全部验证通过 ✓")
```

- [ ] 💻 L2 能照写：用 NetworkX 可视化
  - **达标标准**：画出 5 个带标号的节点和 5 条边，标题 "Graph with 5 nodes"

```python
import networkx as nx
import matplotlib.pyplot as plt
G = nx.from_numpy_array(A)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=14, edge_color='gray')
plt.title("Graph with 5 nodes")
plt.show()
```

---

### Day 3 ｜ 节点特征与边特征

**理论目标**

- [ ] 🧠 L2 理解：节点特征是什么
  - **达标标准**：能说出"每个节点携带一个特征向量——在工程中就是该网格点的物理属性如 [x坐标, y坐标, 弹性模量]。所有节点特征堆叠成矩阵 X∈R^{N×F}，N 是节点数、F 是特征维度"
  - **自测**：100 个节点、5 个特征 → X 形状？（答案：[100, 5]）

- [ ] 🧠 L2 理解：边特征是什么
  - **达标标准**：能说出"每条边也可以有特征，如两个节点之间的距离和方向。在工程中边特征很重要——距离近的节点影响更大"
  - **自测**：节点 A(0,0)、节点 B(3,4) 之间的边特征可以是？（答案：方向 [3,4]、距离 5.0）

**实践目标**

- [ ] 💻 L3 能默写：构造节点特征和边特征
  - **达标标准**：运行后打印形状与注释一致

```python
X = np.array([
    [0.0, 0.0, 200],  [1.0, 0.0, 200],  [0.5, 0.8, 210],
    [1.5, 0.8, 210],  [2.0, 0.0, 200],
])
print(f"节点特征 X 形状: {X.shape}")  # (5, 3)

edge_features = []
for i, j in edges:
    dx = X[j,0]-X[i,0]; dy = X[j,1]-X[i,1]
    edge_features.append([dx, dy, np.sqrt(dx**2+dy**2)])
edge_features = np.array(edge_features)
print(f"边特征形状: {edge_features.shape}")  # (5, 3)
print(f"边 0→1: dx={edge_features[0,0]:.1f}, dy={edge_features[0,1]:.1f}, dist={edge_features[0,2]:.2f}")
```

  - **自测**：边 0→2 距离？（答案：√(0.5²+0.8²)≈0.94）

---

### Day 4–5 ｜ PyTorch Geometric 入门

**理论目标**

- [ ] 🧠 L2 理解：PyG `Data` 对象包含什么
  - **达标标准**：能填完表格

| 属性 | 形状 | 含义 |
|------|------|------|
| `data.x` | [N, F] | 节点特征 |
| `data.edge_index` | [2, E] | 边索引（第 0 行源节点，第 1 行目标节点） |
| `data.edge_attr` | [E, D] | 边特征（可选） |
| `data.y` | [N, out] | 标签 |

- [ ] 🧠 L2 理解：`edge_index` 的 COO 格式
  - **达标标准**：能说出"无向图每条边写两次（正反各一次），所以 5 条无向边 → edge_index 有 10 列"

**实践目标**

- [ ] 💻 L2 能照写：安装 PyG `pip install torch-geometric`
- [ ] 💻 L3 能默写 ⭐：从空白写出 Data 对象并通过 assert

```python
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([
    [0,1, 0,2, 1,2, 2,3, 3,4,  1,0, 2,0, 2,1, 3,2, 4,3],
    [1,0, 2,0, 2,1, 3,2, 4,3,  0,1, 0,2, 1,2, 2,3, 3,4],
], dtype=torch.long)

x = torch.tensor(X, dtype=torch.float)
y = torch.randn(5, 2)
data = Data(x=x, edge_index=edge_index, y=y)

assert data.num_nodes == 5
assert data.num_edges == 10, f"5 条无向边×2=10，得到 {data.num_edges}"
print(data)  # Data(x=[5, 3], edge_index=[2, 10], y=[5, 2])
print("PyG Data 创建成功 ✓")
```

---

### Day 6（周末）｜ 可视化 + 闭卷练习

- [ ] 💻 L2 能照写：`to_networkx` 可视化 PyG 图

```python
from torch_geometric.utils import to_networkx
G = to_networkx(data, to_undirected=True)
nx.draw(G, with_labels=True, node_color='lightgreen', node_size=500)
plt.title("PyG Graph Visualization")
plt.show()
```

- [ ] 💻 L3 能默写：闭卷 15 分钟完成——定义边 → 构造 edge_index → 构造 x → 创建 Data → print

**✅ 产出**：`week10_graph_basics.ipynb`


---
---

## Week 11 · 自建力学图数据集

> **本周目标**：创建一个"一维杆件位移预测"数据集——多个样本（不同 E 和 F），每个样本是一个图。

---

### Day 1 ｜ 力学问题 → 图学习任务

**理论目标**

- [ ] 🧠 L2 理解：怎么把力学问题"翻译"成图学习
  - **达标标准**：能对照说出——网格节点=图节点、节点连接=图边、坐标/材料/载荷=节点特征、节点间距离=边特征、位移=标签 y
  - **自测**：为什么不用 MLP？（答案：节点位移受邻居约束和载荷影响，GNN 能捕捉邻居关系而 MLP 不能）

- [ ] 🧠 L2 理解：本周数据集物理设定
  - **达标标准**：能说出"一根杆件，左端固定(u=0)，右端施力 F，20 个等距节点，E 和 F 随机。解析解 u(x)=Fx/(EA)，A=1"

### Day 2–3 ｜ 编写数据集生成函数

- [ ] 💻 L4 能魔改 ⭐：`generate_bar_dataset(n_nodes=20, n_samples=100)` 返回 PyG Data 列表
  - **达标标准**：每个 Data 的 x=[20,3]、edge_index=[2,38]、y=[20,1]；y[0]=0（固定端）

```python
def generate_bar_dataset(n_nodes=20, n_samples=100):
    dataset = []
    for _ in range(n_samples):
        E = np.random.uniform(100, 300)
        F = np.random.uniform(10, 100)
        x_coords = np.linspace(0, 1.0, n_nodes)
        u_true = F * x_coords / E
        u_true[0] = 0.0
        feats = np.column_stack([x_coords, np.full(n_nodes,E), np.full(n_nodes,F)])
        src = list(range(n_nodes-1)) + list(range(1, n_nodes))
        dst = list(range(1, n_nodes)) + list(range(n_nodes-1))
        data = Data(
            x=torch.tensor(feats, dtype=torch.float),
            edge_index=torch.tensor([src,dst], dtype=torch.long),
            y=torch.tensor(u_true, dtype=torch.float).unsqueeze(1))
        dataset.append(data)
    return dataset

dataset = generate_bar_dataset()
d = dataset[0]
assert d.x.shape == (20,3) and d.edge_index.shape[1] == 38 and d.y[0].item() == 0.0
print("数据集验证通过 ✓")
```

### Day 4 ｜ 数据探索与可视化

- [ ] 💻 L2 能照写：画 3 个样本的位移分布（横轴 "Node position x"，纵轴 "Displacement u"，图例标注 E 和 F）
  - **自测**：E 大的样本位移小（材料硬），F 大的位移大（力大）——图上能看出来吗？

- [ ] 💻 L2 能照写：划分 train(80)/val(10)/test(10)

### Day 5–6（周末）｜ 给边加距离特征

- [ ] 💻 L4 能魔改：升级函数，每条边加 edge_attr=[距离]，形状 [38,1]
  - **自测**：等距节点所有边距离应相等——打印验证

**✅ 产出**：`week11_mechanics_dataset.ipynb`

---
---

## Week 12 · GCN 图卷积网络

> **本周目标**：理解 GCN 公式，用 PyG 实现，在杆件数据上训练。

---

### Day 1 ｜ GCN 核心公式

**理论目标**

- [ ] 🧠 L3 推导 ⭐：关上资料写出 `H = σ(D̂^(-½) Â D̂^(-½) X W)` 并解释每部分
  - **达标标准**：能说出直觉——"每个节点的新特征 = 自己+邻居特征的加权平均 × 权重矩阵 + 激活"
  - **自测**：1 层 GCN 能看到几跳邻居？（1 跳）2 层呢？（2 跳）

**实践目标**

- [ ] 💻 L2 能照写：NumPy 手写一层 GCN
  - **达标标准**：H 形状 [5,16]，所有值≥0（ReLU）

```python
A_hat = A + np.eye(5)
D_hat_inv_sqrt = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1)))
A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
np.random.seed(42)
W = np.random.randn(3, 16) * 0.1
H = np.maximum(0, A_norm @ X @ W)
assert H.shape == (5, 16) and H.min() >= 0
print("手写 GCN 验证通过 ✓")
```

  - **自测**：`A_norm @ X` 在做什么？（答案：每个节点特征被替换为自己+邻居的加权平均）

### Day 2–3 ｜ PyG GCN + 训练

- [ ] 💻 L3 能默写 ⭐：GCN 模型定义

```python
from torch_geometric.nn import GCNConv
class GCN(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid_ch)
        self.conv2 = GCNConv(hid_ch, hid_ch)
        self.head  = nn.Linear(hid_ch, out_ch)
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.head(x)

model = GCN(3, 64, 1)
out = model(dataset[0].x, dataset[0].edge_index)
assert out.shape == (20, 1)
print("GCN 验证通过 ✓")
```

- [ ] 💻 L3 能默写 ⭐：GNN 训练循环（含 DataLoader + val 评估），200 epoch 后 Val MSE < 0.1

```python
from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
# ... 训练循环（5 步模板不变，只是 model 接收 x 和 edge_index）
```

### Day 4 ｜ 可视化

- [ ] 💻 L2 能照写：Train/Val Loss 曲线（标题 "GCN Training Loss"）
- [ ] 💻 L2 能照写：3 个测试样本的 True vs Predicted 位移对比图（3 张子图）
  - **自测**：红色虚线贴合蓝色线吗？

### Day 5–6（周末）｜ hidden_dim 实验

- [ ] 💻 L4 能魔改：hidden_dim = 32/64/128 对比 Val MSE
  - **达标标准**：填满表格 | hidden_dim | 参数量 | Val MSE |

**✅ 产出**：`week12_gcn.ipynb`


---
---

## Week 13 · 消息传递范式与 GraphSAGE

> **本周目标**：理解 GNN 的统一抽象——"消息传递"，学习 GraphSAGE 并理解它为什么更适合工程数据。

---

### Day 1–2 ｜ 消息传递范式

**理论目标**

- [ ] 🧠 L2 理解 ⭐⭐：消息传递三步
  - **达标标准**：能按以下结构讲清楚

```
对每个节点 i，每一层 GNN：
  Step 1 - Message：每个邻居 j 生成一条消息 m_{j→i} = MSG(h_j, e_{ij})
  Step 2 - Aggregate：汇总所有邻居消息 m_i = AGG({m_{j→i}})，方式可以是 sum/mean/max
  Step 3 - Update：用汇总结果更新自身 h_i' = UPD(h_i, m_i)
```

  - **自测**：GCN 的 MSG 是什么？（邻居特征×权重）AGG 呢？（按度加权平均）

- [ ] 🧠 L2 理解：为什么重要
  - **达标标准**：能说出"几乎所有 GNN（GCN/SAGE/GAT）都是消息传递的特例，只是 MSG/AGG/UPD 设计不同"

**实践目标**

- [ ] 💻 L4 能魔改：用 PyG `MessagePassing` 基类手写层
  - **达标标准**：输出形状 [20, 16]

```python
from torch_geometric.nn import MessagePassing

class SimpleMP(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_ch, out_ch)
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_j):
        return self.lin(x_j)   # x_j = 邻居特征（PyG 自动提取）
    def update(self, aggr_out):
        return aggr_out

layer = SimpleMP(3, 16)
out = layer(dataset[0].x, dataset[0].edge_index)
assert out.shape == (20, 16)
print("SimpleMP 验证通过 ✓")
```

  - **自测**：`x_j` 是什么？（所有边的源节点特征——PyG 根据 edge_index 自动提取）
  - **自测**：`aggr='mean'` 换成 `'max'` 会怎样？（聚合方式从平均变成逐维度取最大值）

---

### Day 3–4 ｜ GraphSAGE

**理论目标**

- [ ] 🧠 L2 理解 ⭐：GraphSAGE vs GCN
  - **达标标准**：能填完表格

| | GCN | GraphSAGE |
|---|-----|-----------|
| 邻居使用 | 必须全部 | 可采样部分 |
| 节点数 | 必须固定 | 可变（归纳学习） |
| 新节点 | 不支持 | 支持 |
| 自身特征 | 和邻居混在一起平均 | 和邻居分开处理后拼接 |
| **适合工程网格？** | 不太适合 | **适合——每样本网格不同** |

  - **自测**：为什么工程仿真更适合 SAGE？（每个样本网格节点数和连接可能不同）

**实践目标**

- [ ] 💻 L3 能默写 ⭐：关掉参考写出 SAGE 模型
  - **达标标准**：输出形状 [20, 1]

```python
from torch_geometric.nn import SAGEConv

class SAGE(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hid_ch)
        self.conv2 = SAGEConv(hid_ch, hid_ch)
        self.head  = nn.Linear(hid_ch, out_ch)
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.head(x)

model_sage = SAGE(3, 64, 1)
assert model_sage(dataset[0].x, dataset[0].edge_index).shape == (20, 1)
print("SAGE 验证通过 ✓")
```

- [ ] 💻 L4 能魔改：在杆件数据上训练 SAGE（复用 W12 训练循环改模型），记录 Val MSE
  - **自测**：和 GCN 对比差距大吗？（预期：差距不大，杆件数据每样本节点数相同，SAGE 优势还没体现）

### Day 5–6（周末）｜ 对比 + 巩固

- [ ] 💻 L4 能魔改：GCN vs SAGE 数值对比表

| 模型 | hidden_dim | 参数量 | Val MSE |
|------|-----------|--------|---------|
| GCN | 64 | ? | ? |
| SAGE | 64 | ? | ? |

- [ ] 💻 L2 能照写：同一测试样本画三条线——True / GCN / SAGE
  - **达标标准**：标题 "GCN vs GraphSAGE on test sample"

- [ ] 💻 L3 能默写：闭卷 15 分钟写出 SAGE 模型
- [ ] 💻 L2 能照写：Markdown 3 句话总结——消息传递三步 + GCN vs SAGE 区别 + 为什么工程数据选 SAGE

**✅ 产出**：`week13_sage_and_mp.ipynb`

---
---

## Week 14 · 边特征处理 + 三模型对比 + 过平滑

> **本周目标**：(1) 自定义支持边特征的消息传递层 (2) MLP vs GCN vs SAGE 三模型对比 (3) 理解过平滑

---

### Day 1–2 ｜ 自定义带边特征的消息传递层

**理论目标**

- [ ] 🧠 L2 理解：为什么需要边特征
  - **达标标准**：能说出"标准 GCN/SAGE 的消息传递只用邻居节点特征，不用边特征（如距离）。工程中边特征很重要——距离近的邻居影响更大。需要自定义 MessagePassing 层，在 `message()` 中同时处理邻居特征和边特征"

**实践目标**

- [ ] 💻 L4 能魔改 ⭐：自定义 EdgeAwareMP 层
  - **达标标准**：输出形状 [20, 16]

```python
class EdgeAwareMP(MessagePassing):
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__(aggr='mean')
        self.node_mlp = nn.Linear(node_dim, out_dim)
        self.edge_mlp = nn.Linear(edge_dim, out_dim)
        self.combine  = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: 邻居节点特征 [E, node_dim]
        # edge_attr: 边特征 [E, edge_dim]
        nf = self.node_mlp(x_j)                               # [E, out_dim]
        ef = self.edge_mlp(edge_attr)                          # [E, out_dim]
        return torch.relu(self.combine(torch.cat([nf, ef], dim=-1)))  # [E, out_dim]

layer = EdgeAwareMP(node_dim=3, edge_dim=1, out_dim=16)
sample = dataset[0]  # 需要带 edge_attr 的数据
out = layer(sample.x, sample.edge_index, sample.edge_attr)
assert out.shape == (20, 16)
print("EdgeAwareMP 验证通过 ✓")
```

  - **自测**：`torch.cat([nf, ef], dim=-1)` 在做什么？（把邻居节点特征和边特征拼在一起，让 combine 层同时利用两者）

- [ ] 💻 L4 能魔改：用 EdgeAwareMP 搭完整模型训练，对比有/无边特征

| 模型 | 边特征 | Val MSE |
|------|--------|---------|
| SAGE（无边特征） | ✗ | ? |
| EdgeAwareMP（有边特征） | ✓ | ? |

  - **自测**：杆件数据上差距可能不大（等距节点），不规则网格上差距会更明显

---

### Day 3 ｜ 三模型对比：MLP vs GCN vs SAGE

**理论目标**

- [ ] 🧠 L2 理解：这个实验验证什么
  - **达标标准**：能说出"对比 MLP（忽略邻居）vs GNN（利用邻居）——预期 GNN 更好，因为位移确实受邻居约束影响"

**实践目标**

- [ ] 💻 L4 能魔改 ⭐：实现 MLP 基线

```python
class NodeMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim))
    def forward(self, x, edge_index=None):  # edge_index 不使用
        return self.net(x)
```

- [ ] 💻 L4 能魔改：三模型对比表格

| 模型 | 能否利用邻居 | Val MSE | 测试样本最大误差 |
|------|------------|---------|--------------|
| MLP | ✗ | ? | ? |
| GCN | ✓ | ? | ? |
| GraphSAGE | ✓ | ? | ? |

- [ ] 💻 L2 能照写：同一测试样本画四条线——True / MLP / GCN / SAGE
  - **达标标准**：标题 "MLP vs GCN vs SAGE"，能看出 MLP 偏差最大

- [ ] 💻 L2 能照写：3 句话分析
  - **必须回答**：(1) GNN 比 MLP 好多少（MSE 降低百分比）(2) GCN 和 SAGE 差距 (3) 为什么 GNN 更好（邻居关系）

---

### Day 4 ｜ 过平滑与残差连接

**理论目标**

- [ ] 🧠 L2 理解：过平滑是什么
  - **达标标准**：能说出"层数太多 → 每层都把邻居特征'平均'到自身 → 多层后所有节点特征变得几乎一样 → 网络无法区分不同位置的节点 → 预测下降。GNN 通常 2–4 层就够了"

- [ ] 🧠 L2 理解：残差连接怎么缓解
  - **达标标准**：能说出"h' = h + GNN(h)——即使 GNN(h) 把特征平滑掉了，原始 h 仍然保留"

**实践目标**

- [ ] 💻 L4 能魔改：层数实验

| 层数 | Val MSE |
|------|---------|
| 1 | ? |
| 2 | ? |
| 4 | ? |
| 8 | ? |

  - **自测**：8 层比 2 层差吗？（预期：是——过平滑）

- [ ] 💻 L4 能魔改：加残差连接
  - **达标标准**：以下代码运行无报错

```python
class GCNWithResidual(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.proj = nn.Linear(in_ch, hid_ch)
        self.conv1 = GCNConv(hid_ch, hid_ch)
        self.conv2 = GCNConv(hid_ch, hid_ch)
        self.head = nn.Linear(hid_ch, out_ch)
    def forward(self, x, edge_index):
        h = torch.relu(self.proj(x))
        h = h + torch.relu(self.conv1(h, edge_index))  # 残差
        h = h + torch.relu(self.conv2(h, edge_index))  # 残差
        return self.head(h)
```

  - **自测**：`h = h + torch.relu(...)` 中的 `h +` 就是残差——去掉就退化成普通 GCN

---

### Day 5–6（周末）｜ Part C 自测

**理论自测**（关掉资料）

- [ ] 🧠 能写出邻接矩阵定义并画出一个例子（L3）
- [ ] 🧠 能写出 GCN 公式 `H = σ(D̂^(-½)ÂD̂^(-½)XW)` 并解释每部分（L3）
- [ ] 🧠 能按三步描述消息传递——Message / Aggregate / Update（L2）
- [ ] 🧠 能说出 SAGE vs GCN 的至少 3 个区别以及为什么工程数据选 SAGE（L2）
- [ ] 🧠 能解释过平滑是什么 + 怎么缓解（L2）
- [ ] 🧠 能解释为什么 GNN 比 MLP 更适合有空间关系的数据（L2）

**实践自测**（关掉参考，限时）

- [ ] 💻 10 min：从空白写出 PyG Data 构造（含 edge_index 正反、assert 验证）——L3
- [ ] 💻 10 min：从空白写出 GCN 或 SAGE 模型定义——L3
- [ ] 💻 10 min：从空白写出 GNN 训练循环（含 DataLoader + val 评估）——L3
- [ ] 💻 L4 级别（EdgeAwareMP / SimpleMP）：如果写不出来标黄，后续有时间回来练

**未通过项**：标红，回到对应周重新练习

**✅ 产出**：`week14_gnn_integration.ipynb` + Part C 自测笔记

---
---

## 📊 Part C 自测检查表（Week 10–14 结束后）

**理论**
- [ ] 🧠 能解释图的三要素及和工程网格的对应（L2）
- [ ] 🧠 能写出邻接矩阵/度矩阵/自连接定义（L3）
- [ ] 🧠 能写出 GCN 公式并解释直觉含义（L3）
- [ ] 🧠 能按三步描述消息传递（L2）
- [ ] 🧠 能说出 SAGE vs GCN 区别及工程数据为什么选 SAGE（L2）
- [ ] 🧠 能解释过平滑及残差连接（L2）
- [ ] 🧠 能解释为什么 GNN 比 MLP 更适合有空间关系的数据（L2）

**实践**
- [ ] 💻 能从空白构造 PyG Data 并 assert 验证（L3）
- [ ] 💻 能从空白写出 GCN 和 SAGE 模型（L3）
- [ ] 💻 能从空白写出 GNN 训练循环（L3）
- [ ] 💻 能自定义 MessagePassing 层处理边特征（L4）
- [ ] 💻 能做 MLP vs GCN vs SAGE 三模型对比实验并分析（L4）
- [ ] 💻 能做层数实验 + 加残差连接（L4）