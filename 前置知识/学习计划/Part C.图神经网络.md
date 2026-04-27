# Part C Week 11-12 改造版（批 1）

> Part B 让你能用方程作 Loss 训练 MLP。但 MLP 有一个根本局限：它把每个输入看成独立向量，**不知道节点之间的"邻居关系"**。FEM 网格本质上是一个图——每个节点的位移强烈依赖周围节点。Part C 要解决这个问题：让神经网络"看到"图结构。
>
> Part C 完整 6 周内容会分 3 批输出：
> - Week 11-12（图数据 + GCN）
> - Week 13-14（GraphSAGE + 消息传递）
> - Week 15-16（Encoder-Processor-Decoder + 综合实战）

---

## 📍 Part C 知识地图

```
                       【Part C 知识地图】

  ┌─────────────────────────────────────────────────────────────┐
  │                Week 11: 图数据结构                          │
  │  邻接矩阵 ──► 网格转图 ──► PyG Data ──► 弹性板数据          │
  │   (1)         (2)           (3)         (4)                 │
  └─────────────┬───────────────────────────────────────────────┘
                │  Week 12: 第一个图卷积
                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                  Week 12: GCN（基础对照）                   │
  │  GCN 公式 ──► NumPy 手写 ──► PyG GCNConv ──► 多图训练       │
  │   (5)          (6)             (7)            (8)           │
  └─────────────┬───────────────────────────────────────────────┘
                │  Week 13: 消息传递范式（批 2）
                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │   Week 13-14: GraphSAGE + 消息传递（批 2）                  │
  │   消息传递抽象 ──► GraphSAGE ──► 自定义消息传递             │
  │      (9)              (10)          (11)                    │
  └─────────────┬───────────────────────────────────────────────┘
                │  Week 15-16: 完整架构 + 实战（批 3）
                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │   Week 15-16: 完整 GNN 架构 + 实战（批 3）                  │
  │   Encoder-Processor-Decoder ──► 边更新 ──► mesh-based 模拟器│
  │            (12)                    (13)         (14)        │
  └─────────────────────────────────────────────────────────────┘

  Part C 终点：你能搭出 PhyFENet 论文里的 GNN 架构（含 encoder/processor/
  decoder + 边更新），并在合成的力学数据上训练它预测节点位移。这是 Part D
  把 GNN + PINN 融合成 PhyFENet 的前置。
```

**对你方向的意义**：

PhyFENet 论文（§2.1）这样阐述：

> "图结构数据不遵循连续性和规则性的几何性质。在图结构中，节点之间的关系并非简单的数值型距离，而是依赖于图的拓扑连接模式。"

这句话的实际意思：FEM 网格里，**节点 i 的位移取决于刚度矩阵 K 怎么把它和邻居耦合**。这种"耦合"在 MLP 里没法表达——MLP 把每个节点看成独立的样本。GNN 的"消息传递"机制正好对应这种耦合。

**完成 Part C 后你应能做什么**：
- 看到 FEM 数据，能立刻把它表达成图（节点特征 + 边特征 + 标签）
- 能写出 PhyFENet 论文里 §2.3.3 的 encoder-processor-decoder 结构
- 能解释 GraphSAGE 比 GCN 好在哪里（这是论文 §2.1.2 的论证主线）

---

## Week 11: 图数据结构与图的基本概念

### 🎯 Week 11 总览

**本周覆盖的知识点**：(1) 图的三种表示（邻接矩阵/边列表/edge_index）、(2) 规则网格转图、(3) PyG Data 对象、(4) 弹性板数据构造

**本周不覆盖**：任何神经网络（Week 12 起）；变形网格 / 不规则网格（Part D 起）

**学完之后你应该能**：
- 🟣 **能用**：(1)(2)(3)(4) 全部
- 🟡 **能讲**：为什么 FEM 数据要用图表示而不是扁平向量

**本周的特点**：**纯数据周**，没有神经网络。打地基——为后面 5 周的 GNN 训练准备好图数据基础。代码不难，但**概念上需要"切换思维"**——从"样本是独立向量"转到"样本是带结构的图"。

---

### ✅ 进入 Week 11 之前的前置 checklist

- [ ] Part B 已完成，能写出 PINN 解 1D 弹性杆
- [ ] 我能用 NumPy 操作矩阵和向量（Part A 早期内容）
- [ ] 我会用 matplotlib 画基本的 scatter 和 plot
- [ ] 我已安装好 PyTorch Geometric（`import torch_geometric` 不报错）
- [ ] 我接受"本周没有 NN 训练，只是搭数据结构基础"

**关于 PyG 安装**：
- PyG 的安装比一般 PyTorch 包麻烦——需要根据你的 PyTorch 版本和 CUDA 版本选对应的安装包
- 官网安装指南：https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- 安装完后跑 `python -c "import torch_geometric; print(torch_geometric.__version__)"` 确认成功

---

### Day 1 | 图是什么：三种表示方式

**🎯 本日目标**：理解图的三种数学表示（邻接矩阵 / 边列表 / edge_index），能在它们之间互转。

**🟢 直觉层**（约 10 分钟）：

到 Part B 为止，你的"输入"都是 `(N, features)` 的张量——N 个独立样本。这种结构有个隐含假设：**样本之间没有关系**。

但很多真实数据不是这样：
- 社交网络：你的兴趣和你朋友的兴趣相关
- 分子：原子的化学性质和它连接的其他原子相关
- **FEM 网格：节点的位移和它周围节点的位移相关**

这种带"关系"的数据需要新的数学结构来表达——**图（Graph）**。

一个图由两部分组成：
- **节点（nodes / vertices）**：图里的"个体"，可以有特征
- **边（edges）**：节点之间的"关系"，也可以有特征

**🟡 概念层**（约 20 分钟）：

考虑一个具体例子：

```
    1 ———— 2
    |  \   |
    |   \  |
    4 ———— 3
```

4 个节点，5 条无向边：(1,2)、(1,3)、(1,4)、(2,3)、(3,4)

下面是**三种表示方式**——每种表示都"等价"地描述这个图，只是侧重点不同。

---

**方式 1：邻接矩阵 A**

一个 N×N 的矩阵，`A[i, j] = 1` 表示节点 i 和节点 j 有边，0 表示没有：

```
       1  2  3  4
  1 [  0  1  1  1  ]
  2 [  1  0  1  0  ]
  3 [  1  1  0  1  ]
  4 [  1  0  1  0  ]
```

**性质**：
- shape：N×N
- **无向图的邻接矩阵是对称的**（A[i,j] = A[j,i]）
- 对角线为 0（不自连，标准约定）
- 节点 i 的"度数"（邻居数）= A 第 i 行的和

**优势**：直观、便于线性代数运算（GCN 公式就用它）
**缺点**：N 很大且边稀疏时很浪费内存（FEM 大网格 N=10000，矩阵 100M 个元素，但实际有边的位置只占 0.1%）

---

**方式 2：边列表（COO 格式）**

直接列出所有边：

```
edges = [(1,2), (1,3), (1,4), (2,3), (3,4)]
```

5 条边，每条边是一个二元组。

**优势**：稀疏存储（只存有边的地方）
**缺点**：不便于矩阵运算

---

**方式 3：PyTorch Geometric 的 `edge_index`**

PyG 用的是 COO 格式的变体——shape=(2, n_edges) 的张量：

```
edge_index = [
    [0, 1, 0, 2, 0, 3, 1, 2, 2, 3],    # source nodes
    [1, 0, 2, 0, 3, 0, 2, 1, 3, 2]     # target nodes
]
```

**关键约定**：
- **PyG 的索引从 0 开始**（节点 1 在代码里是 index 0）
- **无向图的每条边写两次**：一次 i→j，一次 j→i
- **shape 是 (2, n_edges) 不是 (n_edges, 2)**——这样 `edge_index[0]` 直接拿到所有 source

为什么写两次？因为 GNN 内部用"消息传递"的方式工作（Week 13 详讲）——每条边代表"信息从 source 传到 target"的一个通路。无向图意味着信息双向流动，所以两个方向都要列出来。

**🔵 数学层**（约 5 分钟）：

无新数学。三种表示在数学上完全等价。

**🟣 代码层**（约 1 小时）：

创建文件 `week11/day01_graph_representations.py`：

```python
import numpy as np
import torch

# ========== 用例子图：4 节点 ==========
#     0 ———— 1
#     | \    |
#     |  \   |
#     3 ———— 2

# ========== 方式 1：邻接矩阵 ==========
A = np.array([
    [0, 1, 1, 1],   # 节点 0 连 1, 2, 3
    [1, 0, 1, 0],   # 节点 1 连 0, 2
    [1, 1, 0, 1],   # 节点 2 连 0, 1, 3
    [1, 0, 1, 0]    # 节点 3 连 0, 2
], dtype=np.float32)

print("邻接矩阵:")
print(A)
print(f"\n是否对称: {np.array_equal(A, A.T)}")    # 应该 True

# 度矩阵（每个节点的度，即邻居数）
degrees = A.sum(axis=1)
print(f"节点度数: {degrees}")    # 应为 [3, 2, 3, 2]

# ========== 方式 2 → 方式 3 转换 ==========
# 找出所有 A[i,j] = 1 的位置
src, dst = np.where(A == 1)
edge_index = np.stack([src, dst], axis=0)   # shape (2, n_edges)
print(f"\nedge_index:\n{edge_index}")
print(f"边的总数（含双向）: {edge_index.shape[1]}")
# 应为 10：5 条无向边 × 2 个方向

# ========== 转成 torch tensor（PyG 用的格式）==========
edge_index_t = torch.from_numpy(edge_index).long()    # 必须是 long 类型
print(f"\nedge_index tensor dtype: {edge_index_t.dtype}")
print(f"edge_index tensor shape: {edge_index_t.shape}")    # (2, 10)
```

**验收标准**：
- 代码无报错
- 邻接矩阵对称
- 节点度数正确（[3, 2, 3, 2]）
- edge_index shape=(2, 10)
- 你能口头回答：为什么 edge_index 是 (2, n_edges) 而不是 (n_edges, 2)

**🔬 应用层**（约 5 分钟）：

**这三种表示在实际使用中的分工**：

| 场景 | 推荐表示 | 原因 |
|---|---|---|
| 数学公式（如 GCN） | 邻接矩阵 | 便于矩阵乘法 |
| 中间存储 / 转换 | 边列表 | 直观、节省空间 |
| 喂给 PyG 模型 | edge_index | PyG 的标准接口 |

**FEM 网格特别**：节点上万、边稀疏（每个节点只连几个邻居）→ 用 edge_index 比邻接矩阵节省 100-1000 倍内存。

---

#### ⚠️ Day 1 新手最容易踩的坑

**坑 1：邻接矩阵和 edge_index 的索引基准不一致**
- 数学公式里节点常从 1 开始编号（节点 1, 2, 3, 4）
- 代码里**索引从 0 开始**（index 0, 1, 2, 3）
- 转换时容易差 1
- 标准做法：**代码里全用 0-indexed**

**坑 2：edge_index 写错 shape**
- 错误：`(n_edges, 2)`
- 正确：`(2, n_edges)`
- 错了的话 PyG 模型会报维度不匹配
- 记忆：**"两行宽"——每行是 source/target 的所有节点**

**坑 3：忘记无向图要双向写**
- 错误：只写 `[(0,1), (0,2), ...]`
- 正确：每条边写两次 `[(0,1), (1,0), (0,2), (2,0), ...]`
- 漏掉的话信息只能单向传，模型表现差

**坑 4：edge_index 用 float 类型**
- PyG 要求 edge_index 是 `long` 类型（整数）
- 用 float 会报错
- 标准写法：`torch.from_numpy(edge_index).long()` 或 `dtype=torch.long`

**坑 5：邻接矩阵带对角线（自环）**
- 标准约定：A[i,i] = 0（不自连）
- 但 GCN 公式里需要"加自环"——A_hat = A + I（Week 12 详讲）
- 数据准备阶段不要预先加自环，让 GCN 自己处理

---

#### 🧠 Day 1 概念问答 quiz

**Q1**：图的三种表示是什么？

<details><summary>答案</summary>(1) 邻接矩阵 A（N×N，A[i,j]=1 有边）；(2) 边列表（COO 格式，列出所有 (i,j) 对）；(3) PyG 的 edge_index（shape=(2, n_edges)）。</details>

**Q2**：无向图的邻接矩阵有什么特殊性质？

<details><summary>答案</summary>**对称矩阵**：A[i,j] = A[j,i]。因为无向边没有方向之分。</details>

**Q3**：PyG 的 edge_index 为什么 shape 是 (2, n_edges) 而不是 (n_edges, 2)？

<details><summary>答案</summary>这样 `edge_index[0]` 直接拿到所有 source 节点的 index，`edge_index[1]` 拿到所有 target——便于 GNN 内部按 source/target 分组操作。如果是 (n_edges, 2)，需要 `[:, 0]` 取列，写起来啰嗦。</details>

**Q4**：无向图的 edge_index 里每条边为什么要写两次？

<details><summary>答案</summary>GNN 用消息传递工作——每条边代表"信息从 source 流到 target"的一个通路。无向图意味着信息双向流动，所以 i→j 和 j→i 都要在 edge_index 里。</details>

**Q5**：节点 i 的"度数"怎么从邻接矩阵 A 算？怎么从 edge_index 算？

<details><summary>答案</summary>从 A：`A[i].sum()`（第 i 行求和）。从 edge_index：统计 edge_index[0] 中等于 i 的位置数（i 作为 source 出现的次数）——因为无向图每条边写了两次，这个值就是节点 i 的度数。</details>

---

#### 📦 Day 1 知识卡片

| 项目 | 内容 |
|---|---|
| **核心术语** | 节点、边、邻接矩阵、edge_index、度数 |
| **三种表示** | 邻接矩阵 (N×N)、边列表、edge_index (2, n_edges) |
| **PyG 约定** | 0-indexed；无向图边写两次；edge_index dtype=long |
| **常见错误** | shape 写反；忘双向；用 float 类型 |
| **本日产出** | `week11/day01_graph_representations.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 2 | 从规则网格到图：FEM 视角

**🎯 本日目标**：理解 FEM 网格如何转成图；写出 `build_regular_mesh` + `mesh_to_edges` 函数。

**🟢 直觉层**（约 10 分钟）：

到此你只见过抽象的图（4 节点示意）。今天接触**真实问题中的图**——FEM 网格转成的图。

FEM 的网格本质就是一个图：
- 网格的**节点**就是图的节点
- 节点之间的**力学耦合关系**就是图的边

但有个细节问题：**怎么定义"两个节点之间有边"**？这个问题对你的方向特别关键，所以专门讲。

**🟡 概念层 + 关键决策**（约 30 分钟）：

考虑一个 3×3 的规则矩形网格（9 个节点，4 个四边形单元）：

```
  6 ——— 7 ——— 8
  |  e3 |  e4 |
  3 ——— 4 ——— 5
  |  e1 |  e2 |
  0 ——— 1 ——— 2
```

节点 4 的"邻居"是谁？这里有**两种合理的定义**——你必须清楚区分：

---

**邻居概念 A：几何结构邻居**

定义：**直接共享一条网格边**的节点。

节点 4 的结构邻居：1, 3, 5, 7（上下左右）。对角线上的 0, 2, 6, 8 **不是**结构邻居——它们和 4 之间没有直接的网格线连接。

直观图：

```
              7
              |
     3 ——— 4 ——— 5
              |
              1
```

---

**邻居概念 B：单元内邻居（论文用的）**

定义：**在同一个单元里**的所有节点都互为邻居。

节点 4 在 4 个单元里（e1、e2、e3、e4）。这 4 个单元的所有节点合起来是 {0,1,2,3,4,5,6,7,8}（全集）。所以节点 4 的图邻居是**其他所有 8 个节点**——包括对角线上的 0、2、6、8。

**为什么把对角节点也算邻居（不直觉）**：
- 物理上：单元 e1 里的 4 个节点（0,1,3,4）通过单元的刚度矩阵相互耦合——节点 0 的位移会影响节点 4 的应力。**即使 0 和 4 不共享网格边，它们物理上是耦合的**
- 数学上：论文 §2.3.2 式 2.19 明确规定"单元内节点两两建边"
- 工程上：更密的连接 = 更多信息传递通路 = GNN 学到更丰富的物理特征

---

**Part C 之后所有代码用的都是概念 B**——这是论文规则。

**单元 e1 (0,1,3,4) 贡献的边**：

每个单元 4 个节点，两两建边：C(4,2) = 6 条。

```
e1 的 6 条边：(0,1)、(0,3)、(0,4)、(1,3)、(1,4)、(3,4)
其中 (0,4) 和 (1,3) 是对角线边
```

去重整张图（多个单元共享的边只保留一次）：

| 类别 | 数量 | 说明 |
|---|---|---|
| 水平结构边 | 6 | 三行各两条 |
| 垂直结构边 | 6 | 三列各两条 |
| 对角线边 | 8 | 4 个单元各 2 条对角，互不共享 |
| **总计** | **20** | （即 edge_index shape=(2, 40)） |

**🔵 数学层**（约 5 分钟）：

节点编号公式（先按 y 后按 x 顺序）：

```
对于 nx × ny 网格里坐标 (i, j) 的节点：
  index = j * nx + i

例：3×3 网格中坐标 (1, 2) 的节点（即第 2 列第 3 行）的 index = 2*3 + 1 = 7 ✓
```

**🟣 代码层**（约 2 小时）：

创建文件 `week11/day02_regular_mesh_to_graph.py`：

```python
import numpy as np
import torch

def build_regular_mesh(nx, ny, dx=1.0, dy=1.0):
    """生成 nx × ny 规则矩形网格
    
    参数：
      nx, ny: 网格在 x 和 y 方向的节点数
      dx, dy: 单元在 x 和 y 方向的尺寸
    返回：
      nodes: shape=(nx*ny, 2)，每行是 (x, y) 坐标
      elements: list[list[int]]，每个元素是 4 个节点 index（左下/右下/右上/左上）
    """
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([i * dx, j * dy])
    nodes = np.array(nodes, dtype=np.float32)
    
    # 每个单元由 4 个节点组成（按逆时针）
    # index = j * nx + i
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i             # 左下
            n1 = j * nx + (i + 1)       # 右下
            n2 = (j + 1) * nx + (i + 1) # 右上
            n3 = (j + 1) * nx + i       # 左上
            elements.append([n0, n1, n2, n3])
    
    return nodes, elements


def mesh_to_edges(elements):
    """从单元连接关系提取无向边集合（采用概念 B：单元内两两建边）
    
    返回：
      edge_set: set，元素是 (min_id, max_id) 元组
    """
    edge_set = set()
    for elem in elements:
        n = len(elem)
        # 两两建边
        for i in range(n):
            for j in range(i + 1, n):
                a, b = min(elem[i], elem[j]), max(elem[i], elem[j])
                edge_set.add((a, b))
    return edge_set


def edges_to_edge_index(edge_set):
    """把边集合转成 PyG 的 edge_index 格式（双向）"""
    src, dst = [], []
    for a, b in edge_set:
        src.append(a); dst.append(b)    # a → b
        src.append(b); dst.append(a)    # b → a
    edge_index = np.array([src, dst], dtype=np.int64)
    return edge_index


# ========== 测试 ==========
print("===== 测试 3×3 网格 =====")
nodes, elements = build_regular_mesh(nx=3, ny=3)
print(f"节点数: {len(nodes)}")              # 9
print(f"单元数: {len(elements)}")           # 4
print(f"\n节点坐标:\n{nodes}")

edge_set = mesh_to_edges(elements)
print(f"\n无向边数（去重后）: {len(edge_set)}")   # 20
print(f"  其中结构边: {12}")
print(f"  其中对角线边: {8}")

edge_index = edges_to_edge_index(edge_set)
print(f"\nedge_index shape: {edge_index.shape}")    # (2, 40)


# ========== 验证：节点 4 的图邻居 ==========
print("\n===== 节点 4 的图邻居验证 =====")
neighbors_of_4 = set()
for a, b in edge_set:
    if a == 4: neighbors_of_4.add(b)
    if b == 4: neighbors_of_4.add(a)
print(f"节点 4 的邻居: {sorted(neighbors_of_4)}")
# 应为 [0, 1, 2, 3, 5, 6, 7, 8]——除了节点 4 自己外的所有节点
```

**验收标准**：
- 9 个节点、4 个单元、20 条无向边
- edge_index shape = (2, 40)
- 节点 4 的邻居是 8 个（所有其他节点）
- 你能解释"为什么 20 条边"（12 结构 + 8 对角）

**🔬 应用层**（约 10 分钟）：

**这种"单元内两两建边"的策略对 PhyFENet 多重要**：

直接看论文 §2.3.2 式 2.19——它就是这么做的。理由：

- 单元是力学计算的"最小单位"
- 单元内的所有节点通过刚度矩阵 `K_e`（element stiffness matrix）耦合——这种耦合不分"水平/垂直/对角"
- GNN 的边就该反映这种耦合关系

**对你方向的启发**：将来你处理任何 FEM 数据，**默认用"单元内两两建边"**——除非有特殊理由（比如内存受限可能用结构边）。

---

#### ⚠️ Day 2 新手最容易踩的坑

**坑 1：`build_regular_mesh` 里节点编号公式记错**
- 错误：`index = i * nx + j`（先按 x 再按 y）
- 正确：`index = j * nx + i`（先按 y 再按 x，因为外层循环是 j）
- 错了的话坐标顺序乱掉，后续无法对应

**坑 2：单元节点顺序写反**
- 错误：随便给 4 个 index
- 正确：左下/右下/右上/左上（按逆时针）
- 这个顺序在后续算"单元面积"或"形函数"时关键
- 暂时记住"逆时针"即可

**坑 3：对角线边漏算或多算**
- 4 节点单元应该有 C(4,2) = 6 条边
- 漏算（只算 4 条结构边）→ 边数变 12 而不是 20
- 多算（误把同一条边算两次）→ 边数变 26+
- 验证方法：3×3 网格应该是 20 条无向边

**坑 4：edge_set 用 list 而不是 set**
- 用 list 不会去重——同一条边可能被多个单元添加多次
- set 自动去重
- 标准做法：用 set，最后再转 list

**坑 5：忘了边是无向的——edge 存成 (a,b) 和 (b,a)**
- 正确：一条无向边只存一次（用 (min, max) 顺序）
- 转 edge_index 时再展开成双向
- 这样 set 才能正确去重

---

#### 🧠 Day 2 概念问答 quiz

**Q1**：FEM 网格的"邻居"有两种合理定义——是哪两种？Part C 用哪个？

<details><summary>答案</summary>(1) 几何结构邻居（共享网格边）；(2) 单元内邻居（同一单元里都互邻）。Part C 用第 2 种（论文规则）。原因是单元内节点物理上通过刚度矩阵耦合。</details>

**Q2**：3×3 规则网格转图后有多少条无向边？

<details><summary>答案</summary>**20 条**——12 条结构边（6 水平 + 6 垂直）+ 8 条对角线边（4 个单元各 2 条）。</details>

**Q3**：4 节点单元用"两两建边"会产生多少条边？

<details><summary>答案</summary>C(4, 2) = 6 条。其中 4 条是结构边（单元的 4 条边），2 条是对角线。</details>

**Q4**：为什么对角线节点也要建边？

<details><summary>答案</summary>因为物理上单元内所有节点通过刚度矩阵相互耦合——即使不共享网格边，0 号节点的位移也会通过单元 e1 影响 4 号节点的应力。建对角线边让 GNN 能学到这种隐式耦合。</details>

**Q5**：3×3 网格的 edge_index shape 是多少？

<details><summary>答案</summary>(2, 40)。20 条无向边，每条双向写一次，所以 40 列。</details>

---

#### 📦 Day 2 知识卡片

| 项目 | 内容 |
|---|---|
| **核心规则** | 单元内节点两两建边（论文规则） |
| **节点编号公式** | index = j * nx + i |
| **3×3 网格** | 9 节点、4 单元、20 边、edge_index=(2,40) |
| **关键工具** | `build_regular_mesh`、`mesh_to_edges`、`edges_to_edge_index` |
| **常见错误** | 漏对角线边；用 list 不去重；编号公式反 |
| **本日产出** | `week11/day02_regular_mesh_to_graph.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 3 | 可视化图

**🎯 本日目标**：用 matplotlib 把网格图画出来；建立"图就是这种东西"的视觉直觉。

**🟢 直觉层**（约 5 分钟）：

到此你能写代码生成图数据了，但**没有"看到"过这个图**。今天的任务很简单：把图画出来。

视觉化的好处：
- 第一次看到对角线边的物理位置
- 帮你以后调试图数据时一眼看出问题（比如某条边漏了）
- 后续可以叠加节点特征（颜色）和位移（箭头）做更复杂的可视化

**🟣 代码层**（约 1.5 小时）：

创建文件 `week11/day03_visualize_mesh.py`：

```python
import numpy as np
import matplotlib.pyplot as plt

# 假设 day02 的函数已经定义好（实际项目里 import 即可）
# from week11.day02_regular_mesh_to_graph import build_regular_mesh, mesh_to_edges

# 这里直接复制函数体（如果 import 不方便）

def build_regular_mesh(nx, ny, dx=1.0, dy=1.0):
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([i * dx, j * dy])
    nodes = np.array(nodes, dtype=np.float32)
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = j * nx + (i + 1)
            n2 = (j + 1) * nx + (i + 1)
            n3 = (j + 1) * nx + i
            elements.append([n0, n1, n2, n3])
    return nodes, elements

def mesh_to_edges(elements):
    edge_set = set()
    for elem in elements:
        n = len(elem)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = min(elem[i], elem[j]), max(elem[i], elem[j])
                edge_set.add((a, b))
    return edge_set


# ========== 任务 1：画完整图（含对角线）==========
nodes, elements = build_regular_mesh(nx=4, ny=4)
edge_set = mesh_to_edges(elements)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 左图：完整图（含对角线）
ax1 = axes[0]
for a, b in edge_set:
    x1, y1 = nodes[a]
    x2, y2 = nodes[b]
    ax1.plot([x1, x2], [y1, y2], 'k-', alpha=0.4, lw=1)

ax1.scatter(nodes[:, 0], nodes[:, 1], c='red', s=80, zorder=10)

for i, (x, y) in enumerate(nodes):
    ax1.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)

ax1.set_aspect('equal')
ax1.set_title(f'4x4 网格 → 图（含对角线，共 {len(edge_set)} 条边）')
ax1.grid(True, alpha=0.3)

# ========== 任务 2：画"只有结构边"的对照图 ==========
ax2 = axes[1]
structural_edges = set()
diag_edges = set()

for a, b in edge_set:
    dx_ = abs(nodes[b, 0] - nodes[a, 0])
    dy_ = abs(nodes[b, 1] - nodes[a, 1])
    if dx_ < 0.01 or dy_ < 0.01:    # 水平或垂直
        structural_edges.add((a, b))
    else:
        diag_edges.add((a, b))

# 画结构边
for a, b in structural_edges:
    x1, y1 = nodes[a]
    x2, y2 = nodes[b]
    ax2.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, lw=1.5, 
             label='结构边' if (a,b)==list(structural_edges)[0] else None)

# 用红色虚线标出对角线（让你看清楚它们在哪）
for a, b in diag_edges:
    x1, y1 = nodes[a]
    x2, y2 = nodes[b]
    ax2.plot([x1, x2], [y1, y2], 'r--', alpha=0.5, lw=0.8,
             label='对角线' if (a,b)==list(diag_edges)[0] else None)

ax2.scatter(nodes[:, 0], nodes[:, 1], c='red', s=80, zorder=10)
for i, (x, y) in enumerate(nodes):
    ax2.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)

ax2.set_aspect('equal')
ax2.set_title(f'结构边（实线 {len(structural_edges)} 条）vs 对角线（虚线 {len(diag_edges)} 条）')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('week11_mesh_visualization.png', dpi=100)
print(f"图已保存到 week11_mesh_visualization.png")
print(f"4×4 网格统计:")
print(f"  节点: {len(nodes)}")
print(f"  单元: {len(elements)}")
print(f"  总边数: {len(edge_set)}")
print(f"  结构边: {len(structural_edges)}")
print(f"  对角线边: {len(diag_edges)}")
```

**验收标准**：
- 两张图都生成
- 左图能看到所有边（包括每个单元内的两条对角线）
- 右图清楚区分结构边（黑实线）和对角线边（红虚线）
- 节点编号清晰可读

---

#### ⚠️ Day 3 新手最容易踩的坑

**坑 1：忘了 `set_aspect('equal')`**
- 不加的话坐标轴比例会被 matplotlib 自动拉伸，方形网格画出来变长方形
- 视觉上误导

**坑 2：浮点数比较用 `==`**
- 错误：`if dx_ == 0`（因为浮点精度问题可能不严格相等）
- 正确：`if dx_ < 0.01`（容差比较）

**坑 3：节点散点画在边的"下面"**
- 默认情况下 scatter 后画的边会盖住节点
- 解决：scatter 时加 `zorder=10`（让节点在最上层）

---

#### 📦 Day 3 知识卡片

| 项目 | 内容 |
|---|---|
| **可视化要素** | 节点（scatter）、边（plot 线段）、节点编号（annotate） |
| **关键技巧** | `set_aspect('equal')`；用 `zorder` 控制层级 |
| **本日产出** | `week11/day03_visualize_mesh.py` + `mesh_visualization.png` |
| **掌握要求** | 🟣 能用 |

---

### Day 4 | PyTorch Geometric 的 `Data` 对象

**🎯 本日目标**：理解 PyG 的 `Data` 对象；用它封装一个完整的图（含节点特征 + 边特征 + 标签）。

**🟢 直觉层**（约 5 分钟）：

到此你有了：
- 节点坐标（来自 `build_regular_mesh`）
- 边索引（来自 `edges_to_edge_index`）

但实际任务里还需要更多东西：
- **节点特征**：每个节点的属性（坐标、材料参数、载荷等）
- **边特征**：每条边的属性（两节点间距离、方向等）
- **标签**：每个节点的真值（位移、应力等）

PyG 用 `Data` 对象**统一封装这些**，方便后面交给模型训练。

**🟡 概念层**（约 15 分钟）：

`Data` 对象的 4 个常用字段：

| 字段 | shape | 含义 |
|---|---|---|
| `x` | (N, n_node_features) | 节点特征 |
| `edge_index` | (2, n_edges) | 边索引 |
| `edge_attr` | (n_edges, n_edge_features) | 边特征（可选） |
| `y` | (N, n_targets) 或 (1, n_targets) | 标签 |

**节点特征 x 怎么设计**：

对 FEM 问题，节点特征通常包含：
- **几何信息**：x, y 坐标
- **材料信息**：弹性模量 E、泊松比 ν（如果各点不同）
- **边界条件**：是否被固定、是否被施加载荷
- **载荷大小**：施加的力或位移值

具体怎么定取决于问题——这是设计决策，不是死规则。

**边特征 edge_attr 怎么设计**：

最常用的边特征是"相对位置"：
- `dx = x_j - x_i`（target 相对 source 的 x 偏移）
- `dy = y_j - y_i`
- `distance = sqrt(dx² + dy²)`

**为什么要边特征？** GNN 的"消息"会用到这些——比如节点 j 影响节点 i 的程度，可能和它们之间的距离有关。

**标签 y 怎么设计**：

回归任务里 y 是浮点数：
- 节点级回归（每个节点一个值）：y.shape=(N, n_targets)
- 图级回归（整个图一个值）：y.shape=(1, n_targets)
- 本课程主要做节点级回归（预测每个节点的位移）

**🟣 代码层**（约 2 小时）：

创建文件 `week11/day04_pyg_data.py`：

```python
import numpy as np
import torch
from torch_geometric.data import Data

# 复用 day02 的函数
def build_regular_mesh(nx, ny, dx=1.0, dy=1.0):
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([i * dx, j * dy])
    nodes = np.array(nodes, dtype=np.float32)
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = j * nx + (i + 1)
            n2 = (j + 1) * nx + (i + 1)
            n3 = (j + 1) * nx + i
            elements.append([n0, n1, n2, n3])
    return nodes, elements

def mesh_to_edges(elements):
    edge_set = set()
    for elem in elements:
        n = len(elem)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = min(elem[i], elem[j]), max(elem[i], elem[j])
                edge_set.add((a, b))
    return edge_set

def edges_to_edge_index(edge_set):
    src, dst = [], []
    for a, b in edge_set:
        src.append(a); dst.append(b)
        src.append(b); dst.append(a)
    return np.array([src, dst], dtype=np.int64)


# ========== Step 1: 生成网格 ==========
nodes, elements = build_regular_mesh(nx=4, ny=4)
edge_set = mesh_to_edges(elements)
edge_index = edges_to_edge_index(edge_set)

N = len(nodes)
print(f"节点数: {N}")

# ========== Step 2: 构造节点特征 ==========
# 这里设计：节点特征 = [x, y, E, F_applied]
x_coord = nodes[:, 0:1]                      # (N, 1)
y_coord = nodes[:, 1:2]                      # (N, 1)
E_feat = np.full((N, 1), 200.0)              # 弹性模量，全 200
F_feat = np.zeros((N, 1))                    # 载荷，默认 0

# 右侧边缘（x = max）的节点施加载荷 F=100
max_x = nodes[:, 0].max()
right_mask = np.abs(nodes[:, 0] - max_x) < 0.01
F_feat[right_mask] = 100.0

# 拼接成节点特征矩阵
x_features = np.concatenate([x_coord, y_coord, E_feat, F_feat], axis=1)
x_features_t = torch.from_numpy(x_features).float()    # (N, 4)
print(f"节点特征 shape: {x_features_t.shape}")

# ========== Step 3: 构造边特征 ==========
src_idx = edge_index[0]    # source 节点索引
dst_idx = edge_index[1]    # target 节点索引

src_coords = nodes[src_idx]
dst_coords = nodes[dst_idx]
diff = dst_coords - src_coords                       # (n_edges, 2)
dist = np.linalg.norm(diff, axis=1, keepdims=True)   # (n_edges, 1)
edge_attr = np.concatenate([diff, dist], axis=1)     # (n_edges, 3)
edge_attr_t = torch.from_numpy(edge_attr).float()
print(f"边特征 shape: {edge_attr_t.shape}")

# ========== Step 4: 构造目标 y（这里用模拟值）==========
np.random.seed(0)
y = np.random.randn(N, 2) * 0.1    # 模拟的 (ux, uy) 位移
y_t = torch.from_numpy(y).float()
print(f"目标 y shape: {y_t.shape}")

# ========== Step 5: 封装成 Data 对象 ==========
edge_index_t = torch.from_numpy(edge_index).long()

data = Data(
    x=x_features_t,
    edge_index=edge_index_t,
    edge_attr=edge_attr_t,
    y=y_t
)

print(f"\n===== Data 对象 =====")
print(data)
print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"节点特征维度: {data.num_node_features}")
print(f"边特征维度: {data.num_edge_features}")

# ========== 验证 ==========
print(f"\n边距离范围: [{edge_attr[:, 2].min():.4f}, {edge_attr[:, 2].max():.4f}]")
# 应在 [1.0, sqrt(2)] 范围内（因为单元尺寸 1×1）
```

**验收标准**：
- `Data` 对象正常创建
- 各项 shape 和数量正确（16 节点、4 节点特征、3 边特征）
- 边距离都是正数（应在 1.0 到 √2 ≈ 1.414 之间）
- 你能解释每个特征的设计动机

---

#### ⚠️ Day 4 新手最容易踩的坑

**坑 1：节点特征里没归一化**
- 节点特征比如 (x=0~3, y=0~3, E=200, F=0~100)——量纲差很多
- 后面训练 GNN 时会和 Part A 一样需要归一化
- 本日先不归一化（教学清晰），下周训练前必须做
- **这是 Part A Week 7 的归一化在 GNN 任务中再次出现**

**坑 2：边特征 dtype 错**
- edge_attr 必须是 float（PyG 模型用 float32 计算）
- 用 long 或 int 会报类型错误

**坑 3：edge_attr 的行数和 edge_index 列数不一致**
- 标准约定：edge_attr.shape[0] 必须等于 edge_index.shape[1]
- 一一对应：edge_index 的第 k 列对应 edge_attr 的第 k 行
- 不一致会让 GNN 模型乱掉

**坑 4：把 y 当成单标签 (N,) 而不是 (N, n_targets)**
- 即使只预测一个值，建议保持二维 (N, 1)
- 方便和模型输出对齐

**坑 5：构造完 Data 没检查就直接用**
- 标准做法：构造完打印 `print(data)` 检查 shape
- PyG 的 Data 对象有 `validate()` 方法可以检查内部一致性

---

#### 🧠 Day 4 概念问答 quiz

**Q1**：PyG 的 `Data` 对象有哪几个常用字段？

<details><summary>答案</summary>(1) `x`：节点特征 (N, n_node_features)；(2) `edge_index`：边索引 (2, n_edges)；(3) `edge_attr`：边特征 (n_edges, n_edge_features)，可选；(4) `y`：标签。</details>

**Q2**：FEM 问题的节点特征通常包含哪些类别的信息？

<details><summary>答案</summary>(1) 几何（坐标）；(2) 材料（E、ν 等）；(3) 边界条件（是否固定/受力）；(4) 载荷大小。</details>

**Q3**：为什么需要边特征？

<details><summary>答案</summary>GNN 用消息传递机制——节点 j 影响节点 i 的方式可能取决于它们之间的关系（距离、方向等）。边特征让 GNN 能利用这些信息。</details>

**Q4**：`edge_attr` 的 shape 必须满足什么约束？

<details><summary>答案</summary>`edge_attr.shape[0]` 必须等于 `edge_index.shape[1]`——一对一对应。</details>

**Q5**：节点级回归和图级回归 y 的 shape 分别怎样？

<details><summary>答案</summary>节点级：(N, n_targets)，每个节点一组值（如每个节点预测 ux, uy）。图级：(1, n_targets)，整个图一组值（如整张图预测一个总应变能）。本课程做节点级。</details>

---

#### 📦 Day 4 知识卡片

| 项目 | 内容 |
|---|---|
| **核心 API** | `Data(x, edge_index, edge_attr, y)` |
| **典型节点特征** | 坐标 + 材料 + 边界条件 + 载荷 |
| **典型边特征** | 相对位移 (dx, dy) + 距离 |
| **关键约束** | edge_attr 行数 = edge_index 列数 |
| **常见错误** | 边特征 dtype 错；shape 不对齐 |
| **本日产出** | `week11/day04_pyg_data.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 5-6（周末）| 弹性板数据构造（贴近论文方向）

**🎯 本日目标**：综合运用本周技能，构造一个真实问题的 `Data` 对象——5×3 弹性板，固定左端、右端拉伸；模拟位移作为标签。

**🟢 物理背景**（约 5 分钟）：

这周末的任务比 Day 4 更接近你的方向：

```
一根 5×3 矩形板（15 个节点）
  - 左端（x=0 的 3 个节点）固定（位移 ux=uy=0）
  - 右端（x=4 的 3 个节点）受均匀拉力 F=100
  - 材料均匀：E=200, ν=0.3

标签（位移）：用一个简化的近似模型生成
  ux(x, y) = 0.01 * x
  uy(x, y) = -0.003 * x * (y - 1)
注：这不是真实的 FEM 解，只是个粗糙的近似——目的是让你练习构造数据。
真实 FEM 仿真需要 Part D 学。
```

**🟣 代码层**（约 2.5 小时）：

创建文件 `week11/weekend_elastic_plate.py`：

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# 引入 day02 的网格函数
def build_regular_mesh(nx, ny, dx=1.0, dy=1.0):
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([i * dx, j * dy])
    nodes = np.array(nodes, dtype=np.float32)
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = j * nx + (i + 1)
            n2 = (j + 1) * nx + (i + 1)
            n3 = (j + 1) * nx + i
            elements.append([n0, n1, n2, n3])
    return nodes, elements

def mesh_to_edges(elements):
    edge_set = set()
    for elem in elements:
        n = len(elem)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = min(elem[i], elem[j]), max(elem[i], elem[j])
                edge_set.add((a, b))
    return edge_set

def edges_to_edge_index(edge_set):
    src, dst = [], []
    for a, b in edge_set:
        src.append(a); dst.append(b)
        src.append(b); dst.append(a)
    return np.array([src, dst], dtype=np.int64)


def build_plate_data(nx=5, ny=3, F_applied=100.0):
    """构造 nx×ny 弹性板的 PyG Data 对象
    
    参数：
      nx, ny: 网格尺寸
      F_applied: 右端施加的载荷
    返回：
      Data 对象
    """
    # ----- 网格 -----
    nodes, elements = build_regular_mesh(nx=nx, ny=ny)
    edge_set = mesh_to_edges(elements)
    edge_index = edges_to_edge_index(edge_set)
    N = len(nodes)
    
    # ----- 节点特征：(x, y, is_fixed, F_applied) -----
    x_coord = nodes[:, 0:1]
    y_coord = nodes[:, 1:2]
    
    is_fixed = np.zeros((N, 1), dtype=np.float32)
    F_feat = np.zeros((N, 1), dtype=np.float32)
    
    # 左端（x=0）固定
    left_mask = np.abs(nodes[:, 0]) < 0.01
    is_fixed[left_mask] = 1.0
    
    # 右端（x=max_x）受力
    max_x = nodes[:, 0].max()
    right_mask = np.abs(nodes[:, 0] - max_x) < 0.01
    F_feat[right_mask] = F_applied
    
    x_features = np.concatenate([x_coord, y_coord, is_fixed, F_feat], axis=1)
    x_features_t = torch.from_numpy(x_features).float()
    
    # ----- 边特征：(dx, dy, distance) -----
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    diff = nodes[dst_idx] - nodes[src_idx]
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    edge_attr = np.concatenate([diff, dist], axis=1).astype(np.float32)
    edge_attr_t = torch.from_numpy(edge_attr).float()
    
    # ----- 模拟位移标签 -----
    # 简化模型：ux = 0.01 * x，uy = -0.003 * x * (y - 1)
    # （注意 y 的中间是 1，所以 (y-1) 反映"距离中线的偏离"）
    ux = 0.01 * nodes[:, 0:1]
    y_center = (ny - 1) * 0.5    # y 方向的中心
    uy = -0.003 * nodes[:, 0:1] * (nodes[:, 1:2] - y_center)
    
    # 强制左端位移为 0（边界条件）
    ux[left_mask] = 0.0
    uy[left_mask] = 0.0
    
    y_label = np.concatenate([ux, uy], axis=1).astype(np.float32)
    y_label_t = torch.from_numpy(y_label).float()
    
    # ----- 封装 -----
    edge_index_t = torch.from_numpy(edge_index).long()
    return Data(
        x=x_features_t,
        edge_index=edge_index_t,
        edge_attr=edge_attr_t,
        y=y_label_t
    ), nodes


# ========== 测试和可视化 ==========
data, nodes = build_plate_data(nx=5, ny=3, F_applied=100.0)
print("===== 弹性板 Data 对象 =====")
print(data)
print(f"节点数: {data.num_nodes}")
print(f"节点特征维度: {data.num_node_features}")
print(f"边数: {data.num_edges}")
print(f"边特征维度: {data.num_edge_features}")
print(f"目标 y shape: {data.y.shape}")

# ========== 可视化 ==========
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 上图：节点特征
ax1 = axes[0]
# 区分三种节点：固定的 / 受力的 / 普通
fixed_mask = data.x[:, 2].numpy() > 0.5
force_mask = data.x[:, 3].numpy() > 0.01
normal_mask = ~fixed_mask & ~force_mask

ax1.scatter(nodes[fixed_mask, 0], nodes[fixed_mask, 1], 
            c='red', s=200, marker='s', label='固定 (left)')
ax1.scatter(nodes[force_mask, 0], nodes[force_mask, 1],
            c='blue', s=200, marker='>', label=f'受力 F={data.x[force_mask, 3].max():.0f}')
ax1.scatter(nodes[normal_mask, 0], nodes[normal_mask, 1],
            c='gray', s=80, label='普通')

# 画边
for k in range(data.edge_index.shape[1]):
    a, b = data.edge_index[0, k].item(), data.edge_index[1, k].item()
    if a < b:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        ax1.plot([x1, x2], [y1, y2], 'k-', alpha=0.2, lw=0.5)

ax1.set_aspect('equal')
ax1.set_title('5×3 弹性板节点特征')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 下图：位移箭头
ax2 = axes[1]
ax2.quiver(nodes[:, 0], nodes[:, 1],
           data.y[:, 0].numpy(), data.y[:, 1].numpy(),
           angles='xy', scale_units='xy', scale=0.05,
           color='blue', alpha=0.8)
ax2.scatter(nodes[:, 0], nodes[:, 1], c='red', s=80, zorder=10)
ax2.set_aspect('equal')
ax2.set_title('模拟位移分布（左端固定，右端拉伸）')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('week11_elastic_plate.png', dpi=100)
print(f"\n图已保存到 week11_elastic_plate.png")
```

**验收标准**：
- 上图清楚展示三类节点（固定/受力/普通）+ 网格连接
- 下图位移箭头：左端 3 个节点箭头长度为 0；右端节点箭头指向 +x 方向；中间节点过渡均匀
- 你能口头解释这就是 PhyFENet 论文里 §3.1 带孔板问题的简化版

**🔬 关键体会**（约 5 分钟）：

到这里你拥有了一个**真实问题级**的图数据对象：

| 元素 | 内容 | 在 PhyFENet 论文里的对应 |
|---|---|---|
| 节点特征 | 坐标 + 边界条件 + 载荷 | §3.1 提到的"节点输入特征" |
| 边特征 | 节点间相对位置 | §2.3.3 提到的"边的几何特征" |
| 标签 | 位移 (ux, uy) | §3.1 的预测目标 |

**唯一的差距**：标签来自简化模型，不是真实 FEM 仿真。Part D 会学怎么用 ABAQUS / FEniCS 跑真实 FEM 生成数据。

---

### ✅ Week 11 完成评估

#### 任务级 checklist

- [ ] `week11/day01_graph_representations.py` 三种表示能互转
- [ ] `week11/day02_regular_mesh_to_graph.py` 3×3 网格 20 条边
- [ ] `week11/day03_visualize_mesh.py` 两张可视化图
- [ ] `week11/day04_pyg_data.py` Data 对象创建成功
- [ ] `week11/weekend_elastic_plate.py` 弹性板数据 + 可视化

#### 概念级 quiz（10 题，至少 8 题对）

**Q1**：图的三种表示是什么？

<details><summary>答案</summary>邻接矩阵 A、边列表、edge_index (2, n_edges)。</details>

**Q2**：无向图的邻接矩阵有什么性质？PyG 的 edge_index 怎么处理无向图？

<details><summary>答案</summary>邻接矩阵对称（A[i,j]=A[j,i]）；edge_index 把每条边写两次（i→j 和 j→i 都列出）。</details>

**Q3**：FEM 网格的"邻居"概念有两种，论文用哪种？

<details><summary>答案</summary>(1) 几何结构邻居（共享网格边）和 (2) 单元内邻居（同单元内两两建边）。论文（§2.3.2）用第 2 种。</details>

**Q4**：3×3 规则网格转图后多少条无向边？

<details><summary>答案</summary>20 条（12 结构边 + 8 对角线边）。</details>

**Q5**：PyG 的 `Data` 对象有哪些常用字段？分别是什么 shape？

<details><summary>答案</summary>x: (N, n_node_features)、edge_index: (2, n_edges)、edge_attr: (n_edges, n_edge_features)、y: (N, n_targets)。</details>

**Q6**：节点编号公式 `j * nx + i` 是按什么顺序？为什么？

<details><summary>答案</summary>先按 y 后按 x（外层 j 循环 y，内层 i 循环 x）。这样 nodes[0] 是左下角，nodes[nx-1] 是右下角，nodes[nx*ny-1] 是右上角——和直觉一致。</details>

**Q7**：FEM 节点特征通常包含哪些类别？举例。

<details><summary>答案</summary>(1) 几何（x, y 坐标）；(2) 材料（E, ν）；(3) 边界条件（is_fixed）；(4) 载荷（F_applied）。</details>

**Q8**：为什么要边特征？最常见的边特征是什么？

<details><summary>答案</summary>GNN 的消息传递可能依赖节点间的关系。最常见：相对位置 (dx, dy) + 距离。</details>

**Q9**：edge_attr 的 shape 和 edge_index 的 shape 必须满足什么约束？

<details><summary>答案</summary>edge_attr.shape[0] = edge_index.shape[1]（一对一对应）。</details>

**Q10**：为什么 FEM 数据要用图表示而不是扁平向量？

<details><summary>答案</summary>FEM 节点的位移强烈依赖周围节点（通过刚度矩阵耦合），扁平向量假设样本独立，无法表达这种邻居耦合。图表示让 GNN 能"看到"节点之间的关系。</details>

#### 🚦 自我评估

- 任务全部通过 + Quiz 8 题对 → **绿灯进入 Week 12**
- 单元内两两建边的规则不熟 → **黄灯**——重看 Day 2
- Data 对象不能独立构造 → **黄灯**——重写 Day 4

#### 知识地图自查

- [ ] (1) 图的三种表示 → 🟣
- [ ] (2) 网格转图 → 🟣
- [ ] (3) PyG Data → 🟣
- [ ] (4) 弹性板数据 → 🟣

---

## 进入 Week 12 之前

下周 Week 12 是 Part C 的**第一次 GNN 训练**：
- GCN 公式直觉
- NumPy 手写 GCN 一层
- PyG 的 GCNConv
- 在弹性板数据上训练

下周特别说明：**GCN 是基础对照周，不是论文的主力模型**。论文用 GraphSAGE（Week 13），而 GraphSAGE 的设计动机建立在"理解 GCN 的局限"之上。所以 Week 12 不要求你"精通 GCN"——会用、能解释公式、知道局限就够了。

下周需要的前置：
- Part A 的 MLP 训练流程熟练（GNN 训练流程类似）
- Week 11 的 PyG Data 对象能独立构造
- 不需要新数学

Week 11 完成。

---

---

## Week 12: GCN——最简单的图卷积

### 🎯 Week 12 总览

> **本周定位说明（重要）**：
>
> Week 12 是 **基础对照周**，不是你方向的主力模型。论文 §2.1.2 明确说 GCN 有局限（需要固定节点数、全图更新、所有邻居等权聚合），后续采用 GraphSAGE 并增加了边更新机制。**GraphSAGE + 带边更新的消息传递才是你真正要掌握的主角**（Week 13 和 Week 14）。
>
> 为什么仍然要用一整周学 GCN？因为 GraphSAGE 的设计动机（"为什么要把节点自身特征和邻居特征分开"）建立在你**理解 GCN 的局限**之上。跳过 GCN 直接学 GraphSAGE 会失去这个对比感，后面学起来会变成死记硬背。
>
> 本周目标：**会用 GCN，建立消息传递的第一直觉**。如果 Day 2 的 NumPy 手写一次就跑通了，不用花太多时间反复练；重点放在 Day 3-4 的 PyG 使用和 Day 5-6 的多图训练。

**本周覆盖的知识点**：(5) GCN 公式直觉、(6) NumPy 手写一层、(7) PyG GCNConv、(8) 多图训练

**本周不覆盖**：消息传递的通用抽象（Week 13）、GraphSAGE（Week 13）、自定义层（Week 14）

**学完之后你应该能**：
- 🟡 **能讲**：GCN 公式每一项的意义；GCN 的局限
- 🟣 **能用**：用 PyG 的 GCNConv 搭模型；在多图数据上训练

---

### ✅ 进入 Week 12 之前的前置 checklist

- [ ] Week 11 的弹性板 Data 对象能独立构造
- [ ] 我能用 NumPy 算矩阵乘法和对角矩阵
- [ ] 我能跑 PyTorch 训练循环（Part A 的内容）
- [ ] 我安装好 PyG（`from torch_geometric.nn import GCNConv` 不报错）

---

### Day 1 | GCN 的核心思想

**🎯 本日目标**：从直觉出发理解 GCN 公式每一项的意义；建立"消息传递"的第一感觉。

**🟢 直觉层**（约 15 分钟）：

GCN 的核心想法可以用一句话概括：

> **每个节点的新特征 = 它和邻居（包括自己）特征的"加权平均"，再做线性变换 + 激活**

具象化想象：你在一个微信群里，每个人都有自己的观点（特征）。GCN 的一层就是：每个人**听一遍群里所有人的观点**（加权平均），然后**更新自己的观点**（线性变换 + 激活）。重复 K 层 = 信息传递 K 跳——你的观点会受到 K 跳之外人的间接影响。

**关键 idea**：
- 邻居信息和自己信息**一起**参与新特征的计算（不是只看邻居或只看自己）
- "加权"是按节点度数归一化的（避免度数大的节点主导）

> ⚠️ 比喻警告：微信群比喻只用一次，下面进入精确数学。

**🟡 概念层**（约 30 分钟）：

GCN 的核心公式（论文 §2.1.2 式 2.2）：

```
H = σ( D^(-1/2) · Â · D^(-1/2) · X · W )
```

看着复杂，**拆 4 步就不难**。

---

**Step 1：为什么要 Â = A + I？**

如果只用原始邻接矩阵 A：
- `Â · X` 中节点 i 得到的"聚合值" = `Σ A[i,j] · X[j]`（对所有 j 求和）
- 因为 A[i, i] = 0，所以**节点自己的特征 X[i] 没被加进来**

GCN 要做"邻居 + 自己"的聚合，所以加自环：

```
Â = A + I
```

加上自环后，节点 i 的聚合值包括自己的 X[i]。

---

**Step 2：为什么要做 `D^(-1/2) · Â · D^(-1/2)`？**

不归一化时：
- `Â · X` 中节点 i 拿到 `(度数+1) 个特征向量的和`
- 度数 100 的节点：拿到 100+ 个向量的和——值大
- 度数 2 的节点：拿到 3 个向量的和——值小
- **节点之间的特征量级严重失衡**

`D^(-1/2) · Â · D^(-1/2)` 是一种"对称归一化"——大致相当于把每个邻居的贡献按 `1 / sqrt(d_i * d_j)` 归一化。效果：所有节点的聚合值量级可比。

数学严格表达：

```
归一化邻接矩阵 A_norm = D^(-1/2) · Â · D^(-1/2)
其中 D 是 Â 的度矩阵（对角矩阵）
```

具体到节点 i 和邻居 j 的关系：

```
A_norm[i, j] = 1 / sqrt(d_i * d_j)    （如果 i, j 之间有边或 i=j）
```

---

**Step 3：X · W 是什么？**

X 是所有节点的特征矩阵 (N, in_features)。W 是一层的权重矩阵 (in_features, out_features)。

`X · W` 把每个节点的特征通过线性变换映射到新维度——**和 Part A 的 MLP 单层完全一样**，只是每行（每个节点）做相同的变换。

---

**Step 4：σ 是什么？**

激活函数，通常用 ReLU。

> **注意**：GCN 没有"求二阶导"的需求，用 ReLU **没问题**。只有 PINN 风格的网络（要对输入求二阶导，Part B）才必须用 Tanh。这是 ReLU 第一次"回归"——你之前一直被告诉 Part B 用 Tanh，那是因为 PINN 需要光滑导数。GCN 没这个需求，标准 PyG GCN 默认用 ReLU。

---

**完整流程**（边讲边画图）：

```
输入: X, shape=(N, in_features)
       ↓
   X · W                    # 线性变换：(N, in_features) → (N, out_features)
       ↓
   A_norm · (X · W)         # 邻居聚合：每个节点拿到归一化的邻居特征加权和
       ↓
       σ                    # 激活
       ↓
输出: H, shape=(N, out_features)
```

数学顺序上 `A_norm · X · W` 也等价于 `A_norm · (X · W)`——矩阵乘法满足结合律。

**用自己的话总结 GCN 一层在做什么**（在笔记上写一段）：

参考答案：

> 对每个节点，把它和邻居（含自己）的特征做"按度数归一化的加权平均"，然后通过线性变换映射到新特征空间，最后过激活函数。这个过程相当于每个节点"听完邻居的意见"后更新自己的表示。

**🔵 数学层**（约 5 分钟）：

具体例子：4 节点图，邻接矩阵：

```
A = [[0, 1, 1, 0],
     [1, 0, 1, 1],
     [1, 1, 0, 0],
     [0, 1, 0, 0]]
```

度数：节点 0 度数 2、节点 1 度数 3、节点 2 度数 2、节点 3 度数 1。

加自环后 Â：

```
Â = A + I
```

度数（含自环）：3, 4, 3, 2。

`A_norm[0, 1] = 1 / sqrt(3 * 4) ≈ 0.289`（节点 0 和 1 之间）

具体计算下面代码会做。

**🟣 代码层**（无独立任务）：

今天理论为主，不写代码。Day 2 会用 NumPy 实现这个公式。

---

#### ⚠️ Day 1 新手最容易踩的坑

**坑 1：以为 GCN 的"卷积"和 CNN 的卷积是一回事**
- CNN 的卷积是滑动窗口在规则网格上的运算
- GCN 的"卷积"是邻居特征聚合——本质是一种**类比**
- 两者在数学结构上有相似点，但实现机制完全不同

**坑 2：忽略 `Â = A + I`，结果节点自己的特征丢了**
- 漏了自环：节点 i 只看邻居，不看自己
- 多层后会产生奇怪的行为
- **GCN 必须加自环**

**坑 3：忽略归一化，结果度数大的节点主导训练**
- 不归一化时不同节点聚合后量级差很多
- 训练时大量级节点的梯度大，主导更新方向
- 度数小的节点几乎学不到东西

**坑 4：以为 GCN 公式只能对应特定层数**
- GCN 公式描述的是**一层**的运算
- 多层 GCN = 多次应用这个公式
- 每层的 W 是独立的可学习参数

---

#### 🧠 Day 1 概念问答 quiz

**Q1**：GCN 的核心思想用一句话概括？

<details><summary>答案</summary>每个节点的新特征 = 它和邻居（包括自己）的特征加权平均，再线性变换 + 激活。</details>

**Q2**：为什么 `Â = A + I`（加自环）？

<details><summary>答案</summary>原始 A 的对角线为 0，节点自己的特征不会出现在聚合里——只看邻居忘了自己。加 I（单位矩阵）让对角线为 1，节点自己也参与聚合。</details>

**Q3**：为什么要 `D^(-1/2) · Â · D^(-1/2)` 归一化？

<details><summary>答案</summary>避免度数大的节点聚合值过大、度数小的节点过小。归一化让所有节点的聚合值量级可比，防止一些节点的梯度主导训练。</details>

**Q4**：GCN 的激活函数为什么可以用 ReLU？Part B 的 PINN 不是必须用 Tanh 吗？

<details><summary>答案</summary>PINN 必须用 Tanh 是因为要对网络输入求二阶导（要光滑可导）。GCN 这里没有"对输入求高阶导"的需求，ReLU 没问题。只有当你把 GCN 用在 PINN 风格的物理 Loss 训练时，才需要换成 Tanh。</details>

**Q5**：GCN 的一层公式 `H = σ(A_norm · X · W)`，矩阵乘法的顺序能不能改？

<details><summary>答案</summary>`A_norm · X · W` 和 `A_norm · (X · W)` 等价（结合律）。但运算顺序不同会影响计算量——通常 `X · W` 先做（因为 W 把特征维度变小，X · W 后矩阵更小，再乘 A_norm 更高效）。</details>

---

#### 📦 Day 1 知识卡片

| 项目 | 内容 |
|---|---|
| **核心公式** | `H = σ(D^(-1/2) · Â · D^(-1/2) · X · W)` |
| **Â** | A + I（加自环） |
| **归一化作用** | 让不同度数节点的聚合量级可比 |
| **激活函数** | 通常 ReLU（不像 PINN 必须 Tanh） |
| **关键直觉** | 邻居（含自己）特征加权平均 + 线性变换 + 激活 |
| **本日产出** | 笔记上写"GCN 一层在做什么" |
| **掌握要求** | 🟡 能讲 |

---

### Day 2 | NumPy 手写一层 GCN

**🎯 本日目标**：用 NumPy 把 Day 1 的公式逐步实现；对照公式验证每一步的正确性。

**🟢 直觉层**（约 5 分钟）：

今天的目的不是"写一个能用的 GCN"——PyG 自带 GCNConv 用就行。今天的目的是**把公式变成代码**，让你**真正理解每一步在做什么**。这种"自己写一遍"的练习对建立直觉至关重要。

**🟣 代码层**（约 1.5 小时）：

创建文件 `week12/day02_gcn_numpy.py`：

```python
import numpy as np

def gcn_layer_numpy(X, A, W):
    """GCN 一层计算（NumPy 实现）
    
    完全对照公式：H = ReLU(D^(-1/2) · Â · D^(-1/2) · X · W)
    
    参数：
      X: 节点特征矩阵，shape=(N, in_features)
      A: 邻接矩阵（不含自环），shape=(N, N)
      W: 权重矩阵，shape=(in_features, out_features)
    返回：
      H: 新节点特征，shape=(N, out_features)
    """
    N = A.shape[0]
    
    # Step 1: 加自环（Â = A + I）
    A_hat = A + np.eye(N)
    
    # Step 2: 算度矩阵 D（注意是含自环的度）
    degrees = A_hat.sum(axis=1)         # shape=(N,)
    
    # Step 3: 算 D^(-1/2)（对角矩阵的逆开根号）
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    
    # Step 4: 归一化邻接矩阵
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    
    # Step 5: 消息聚合 + 特征变换
    # 顺序：先 X·W（变换特征），再 A_norm·结果（聚合邻居）
    XW = X @ W                  # shape=(N, out_features)
    H_pre = A_norm @ XW          # shape=(N, out_features)
    
    # Step 6: ReLU 激活
    H = np.maximum(0, H_pre)
    
    return H


# ========== 测试 ==========
print("===== 4 节点图测试 =====")

# 邻接矩阵：0-1, 0-2, 1-2, 1-3
A = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)

# 节点特征：4 个节点，每个 3 个特征
X = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0]
], dtype=np.float32)

# 权重矩阵：3 → 2 维变换
np.random.seed(0)
W = np.random.randn(3, 2).astype(np.float32) * 0.5

print(f"输入 X shape: {X.shape}")
print(f"邻接 A shape: {A.shape}")
print(f"权重 W shape: {W.shape}")

H = gcn_layer_numpy(X, A, W)

print(f"\n输出 H shape: {H.shape}")     # (4, 2)
print(f"输出 H:\n{H}")

# ========== 中间步骤验证 ==========
print("\n===== 中间步骤验证 =====")

A_hat = A + np.eye(4)
print(f"Â = A + I:\n{A_hat}")

degrees = A_hat.sum(axis=1)
print(f"\n度数（含自环）: {degrees}")
# 节点 0：原邻居 1,2 + 自己 = 3 个
# 节点 1：原邻居 0,2,3 + 自己 = 4 个
# 节点 2：原邻居 0,1 + 自己 = 3 个
# 节点 3：原邻居 1 + 自己 = 2 个

D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
print(f"\nD^(-1/2):\n{D_inv_sqrt}")

A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
print(f"\n归一化后 A_norm:\n{A_norm}")

# 验证：A_norm 行和应小于等于 1（因为邻居权重被归一化）
print(f"\nA_norm 每行的和: {A_norm.sum(axis=1)}")
```

**验收标准**：
- 代码无报错
- 输出 H shape = (4, 2)
- 每个 step 的中间输出能解释（你能口头说"这步是 Â=A+I"等）
- A_norm 是对称矩阵（这是数学性质）

**🔬 应用层**（约 10 分钟）：

**手算一个具体例子加深理解**：

节点 2 的新特征第 1 个分量 `H[2, 0]`：

```
H_pre[2, 0] = A_norm[2, 0]·XW[0, 0] + A_norm[2, 1]·XW[1, 0] + A_norm[2, 2]·XW[2, 0]
            （只有 0、1、2 是节点 2 的邻居（含自己））
```

如果你打印出 A_norm 和 XW 的具体值，就能手算这个值并和 `H_pre[2, 0]` 对比。

**这一步的价值**：让你看到"GCN 的'聚合'其实就是按权重求和"——没有任何神秘的东西。

---

#### ⚠️ Day 2 新手最容易踩的坑

**坑 1：忘了 `np.eye(N)` 而不是 `np.ones((N, N))`**
- `np.eye(N)` 是单位矩阵（对角线 1，其他 0）
- `np.ones((N, N))` 是全 1 矩阵
- 加自环用 eye

**坑 2：度矩阵用的是 A 而不是 Â**
- 公式里 D 是 Â 的度矩阵（含自环）
- 用 A 算度数，节点 i 的度数会比正确值少 1
- 归一化系数错，整体结果错

**坑 3：归一化顺序写错**
- 错误：`D_inv · A_hat`（左乘 D^(-1)）
- 正确：`D^(-1/2) · A_hat · D^(-1/2)`（左右各乘一次）
- 这是 GCN 论文里"对称归一化"的标准做法
- 也有非对称版本 `D^(-1) · Â`，但 GCN 公式用对称版

**坑 4：`np.maximum(0, H_pre)` 写成 `max(0, H_pre)`**
- Python 内置 `max` 不支持 numpy 数组
- 必须用 `np.maximum`（注意是 maximum 不是 max）
- 这是 Part A Week 1 的老坑了

---

#### 🧠 Day 2 概念问答 quiz

**Q1**：GCN 公式的 6 步实现里，哪一步最容易写错？

<details><summary>答案</summary>归一化（`D^(-1/2) · Â · D^(-1/2)`）。容易：(1) 用 A 而不是 Â 算度数；(2) 顺序写错；(3) 用 D^(-1) 代替 D^(-1/2)。</details>

**Q2**：为什么 GCN 的归一化用 D^(-1/2) 两边乘而不是 D^(-1) 一次？

<details><summary>答案</summary>对称归一化（D^(-1/2) 两边）保持 A_norm 是对称矩阵，与无向图的对称性一致。非对称归一化（D^(-1)·A）在某些场景也用，但 GCN 论文用对称版。</details>

**Q3**：在 NumPy 里写 GCN 时，公式的运算顺序 `A_norm · X · W` 应该按 `A_norm · (X · W)` 还是 `(A_norm · X) · W` 实现？

<details><summary>答案</summary>**两者等价**（矩阵乘法满足结合律）。但 `X · W` 先做更高效——因为 W 把特征维度变小，先做后矩阵更小，再乘 A_norm 计算量小。</details>

**Q4**：你的代码里如果 `np.eye(N)` 写成 `np.ones((N, N))`，会有什么后果？

<details><summary>答案</summary>所有节点都被认为和所有节点相连（不只是邻居）——整张图变成完全图。聚合时每个节点都看到所有节点的特征，失去了"邻居信息"的意义。</details>

**Q5**：手写 GCN 的核心目的是什么？

<details><summary>答案</summary>建立"GCN 没有神秘"的直觉——它就是按归一化邻接矩阵做加权和。理解每步在做什么后，用 PyG 的 GCNConv 就只是"按 API 调用"。</details>

---

#### 📦 Day 2 知识卡片

| 项目 | 内容 |
|---|---|
| **核心代码模式** | 加自环 → 算度 → 归一化 → 聚合 → 激活 |
| **关键 API** | `np.eye`、`np.diag`、`np.maximum` |
| **常见错误** | 用 A 算度（应该用 Â）；归一化顺序错；忘 ReLU |
| **本日产出** | `week12/day02_gcn_numpy.py` |
| **掌握要求** | 🔵 能写（看公式能写出代码） |

---

### Day 3 | 用 PyG 的 GCNConv 搭模型

**🎯 本日目标**：从手写 GCN 转到用 PyG 的标准 API；体会"用 PyG 写 GCN 多简单"。

**🟢 直觉层**（约 5 分钟）：

Day 2 你手写了一层 GCN——大约 30 行代码。今天你会发现：用 PyG 的 GCNConv，**一行**就能完成同样的事。

```python
self.conv1 = GCNConv(in_channels=5, out_channels=16)
```

这一行内部做了：
- 加自环
- 度归一化
- 特征变换
- （可选）激活

PyG 把所有这些封装好了。**作为一个 PyG 用户，你只需要决定输入维度、输出维度、激活函数、堆叠多少层**。

**🟡 概念层**（约 10 分钟）：

`GCNConv` 的核心 API：

```python
from torch_geometric.nn import GCNConv

# 创建一层
conv = GCNConv(in_channels=in_dim, out_channels=out_dim)

# 使用
out = conv(x, edge_index)        # x: (N, in_dim), edge_index: (2, E) → out: (N, out_dim)
```

注意：
- `GCNConv` 不带激活——需要你**外面手动加**（如 `F.relu(out)`）
- `edge_index` 直接传入——PyG 内部处理归一化等

**典型 GCN 模型结构**：

```python
class SimpleGCN(nn.Module):
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
```

**结构说明**：
- 两层 GCN：让每个节点接收 2 跳之内的邻居信息
- 最后一个**线性头**（不是 GCN 层）：把 GNN 学到的隐特征映射到任务输出维度（比如位移 ux, uy → 输出 2 维）
- 头层用 nn.Linear 而不是 GCNConv 的原因：最终输出不需要再聚合邻居信息

**🟣 代码层**（约 1.5 小时）：

创建文件 `week12/day03_gcn_pyg.py`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class SimpleGCN(nn.Module):
    """两层 GCN + 线性输出头"""
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


# ========== 测试 ==========
# 构造一个简单图
edge_index = torch.tensor([
    [0, 1, 1, 0, 1, 2, 2, 1, 2, 3, 3, 2],
    [1, 0, 0, 1, 2, 1, 1, 2, 3, 2, 2, 3]
], dtype=torch.long)
x = torch.randn(4, 5)      # 4 节点，5 特征
y = torch.randn(4, 2)       # 每个节点预测 (ux, uy)

data = Data(x=x, edge_index=edge_index, y=y)

# 创建模型
model = SimpleGCN(in_channels=5, hidden_channels=16, out_channels=2)

# 打印模型结构
print(model)
print(f"\n参数总数: {sum(p.numel() for p in model.parameters())}")

# 前向测试
y_pred = model(data)
print(f"\n输入 x shape: {data.x.shape}")
print(f"输出 y_pred shape: {y_pred.shape}")    # (4, 2)
```

**验收标准**：
- 模型创建无错
- 前向输出 shape = (4, 2)
- 你能解释模型结构（两层 GCN + 一个线性头）
- 你能解释为什么最后用 Linear 而不是 GCNConv

---

#### ⚠️ Day 3 新手最容易踩的坑

**坑 1：`GCNConv` 不带激活，忘了手动加**
- 错误：`x = self.conv1(x, edge_index); x = self.conv2(x, edge_index)`（两层之间没激活）
- 等价于一层 GCN——损失非线性表达能力
- 正确：每层后加 `F.relu(x)`

**坑 2：把 `edge_index` 传错位置**
- `GCNConv` 的调用是 `conv(x, edge_index)`
- 写成 `conv(edge_index, x)` 会报错或行为奇怪
- 顺序：节点特征在前，边索引在后

**坑 3：模型 forward 里用 `model(x)` 而不是 `model(data)`**
- 我们的模型设计是接收整个 Data 对象
- 也可以改成接收 (x, edge_index)，看你怎么设计
- 重要：保持输入输出接口的一致性

**坑 4：多层 GCN 想堆 10 层**
- GCN 经典问题：**过平滑（over-smoothing）**——层多了所有节点的特征会变得相似
- 通常 GCN 用 2-3 层就够了
- 想要更深层，需要用 GAT、GraphSAGE 或加 residual connection（Week 13 起会逐步学）

---

#### 🧠 Day 3 概念问答 quiz

**Q1**：`GCNConv` 内部都做了 Day 2 公式的哪些步骤？

<details><summary>答案</summary>(1) 加自环；(2) 度归一化；(3) 邻居聚合；(4) 特征线性变换 X·W。**不包括激活**——激活要外面手动加。</details>

**Q2**：典型 GCN 模型为什么最后一层用 `nn.Linear` 而不是 `GCNConv`？

<details><summary>答案</summary>最后一层把 GNN 学到的隐特征映射到任务输出（如位移），不需要再聚合邻居信息。Linear 层只做特征变换。</details>

**Q3**：GCNConv 不自带激活，为什么要外面加？

<details><summary>答案</summary>设计灵活性。让用户决定用什么激活（ReLU / GELU / Tanh）、什么时候加（每层后还是只在某些层）、要不要加 BatchNorm 等其他操作。如果激活在 GCNConv 内部，这些灵活性就没了。</details>

**Q4**：堆 10 层 GCN 会怎样？

<details><summary>答案</summary>**过平滑（over-smoothing）**——层数多了所有节点的特征趋于相似，模型失去区分能力。通常 GCN 用 2-3 层。</details>

**Q5**：`GCNConv(in_channels, out_channels)` 内部一共有多少可学习参数？

<details><summary>答案</summary>主要是权重矩阵 `W: (in_channels, out_channels)`，参数数 = `in_channels * out_channels`。可能还有偏置（默认 True）：`out_channels` 个。</details>

---

#### 📦 Day 3 知识卡片

| 项目 | 内容 |
|---|---|
| **核心 API** | `GCNConv(in, out)` |
| **典型模型结构** | 2 层 GCN + 线性头 |
| **激活位置** | 每层 GCNConv 之后手动加 |
| **关键陷阱** | 过平滑（不要堆超过 3-4 层） |
| **本日产出** | `week12/day03_gcn_pyg.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 4 | 在弹性板数据上训练 GCN

**🎯 本日目标**：把 Week 11 的弹性板数据 + Day 3 的 GCN 模型 + Part A 的训练循环组合起来，跑出第一个 GNN 训练。

**🟢 直觉层**（约 5 分钟）：

到此你拥有了完整 GNN 训练所需的所有部件：
- 数据：弹性板 Data 对象（Week 11）
- 模型：SimpleGCN（Day 3）
- 训练循环：Part A 的标准模板

今天的任务是把它们拼起来，看 GCN 能不能在一张图上"学到"位移分布。

**注意**：这里只用 1 张图训练（一个样本）——这种情况下 GCN 一定会**完全过拟合**，目的不是真的"学习"，而是**验证模型能跑通**。真正的多样本训练在 Day 5-6 周末。

**🟣 代码层**（约 2 小时）：

创建文件 `week12/day04_train_gcn_bar.py`：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 引入前面的代码（实际项目里 import）
# from week11.weekend_elastic_plate import build_plate_data
# from week12.day03_gcn_pyg import SimpleGCN

# ========== 复用 Week 11 的弹性板数据构造（简化版）==========
from torch_geometric.data import Data

def build_plate_data(nx=5, ny=3, F_applied=100.0):
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([float(i), float(j)])
    nodes = np.array(nodes, dtype=np.float32)
    
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = j * nx + (i + 1)
            n2 = (j + 1) * nx + (i + 1)
            n3 = (j + 1) * nx + i
            elements.append([n0, n1, n2, n3])
    
    edge_set = set()
    for elem in elements:
        for i in range(4):
            for j in range(i+1, 4):
                a, b = min(elem[i], elem[j]), max(elem[i], elem[j])
                edge_set.add((a, b))
    
    src, dst = [], []
    for a, b in edge_set:
        src.append(a); dst.append(b)
        src.append(b); dst.append(a)
    edge_index = np.array([src, dst], dtype=np.int64)
    
    N = len(nodes)
    is_fixed = np.zeros((N, 1), dtype=np.float32)
    F_feat = np.zeros((N, 1), dtype=np.float32)
    is_fixed[np.abs(nodes[:, 0]) < 0.01] = 1.0
    F_feat[np.abs(nodes[:, 0] - nodes[:, 0].max()) < 0.01] = F_applied
    
    x_features = np.concatenate([nodes, is_fixed, F_feat], axis=1)
    
    # 模拟位移
    y_center = (ny - 1) * 0.5
    ux = 0.01 * (F_applied / 100) * nodes[:, 0:1]
    uy = -0.003 * (F_applied / 100) * nodes[:, 0:1] * (nodes[:, 1:2] - y_center)
    ux[np.abs(nodes[:, 0]) < 0.01] = 0
    uy[np.abs(nodes[:, 0]) < 0.01] = 0
    y_label = np.concatenate([ux, uy], axis=1).astype(np.float32)
    
    return Data(
        x=torch.from_numpy(x_features).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        y=torch.from_numpy(y_label).float()
    ), nodes


# ========== Model（同 Day 3）==========
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x)


# ========== 训练 ==========
torch.manual_seed(0)

data, nodes = build_plate_data(nx=5, ny=3, F_applied=100.0)
model = SimpleGCN(
    in_channels=data.num_node_features,    # 4: x, y, is_fixed, F
    hidden_channels=32,
    out_channels=2                         # ux, uy
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

losses = []
print("训练 GCN（单图过拟合）...")
for epoch in range(500):
    y_pred = model(data)
    loss = criterion(y_pred, data.y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if epoch % 50 == 0:
        print(f"  epoch {epoch}: loss = {loss.item():.6f}")

print(f"\n最终 loss: {loss.item():.6f}")

# ========== 可视化预测 vs 真值 ==========
y_pred = model(data).detach().numpy()
y_true = data.y.numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 左图：Loss 曲线
axes[0].plot(losses)
axes[0].set_yscale('log')
axes[0].set_title('训练 Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')

# 中图：真实位移
axes[1].quiver(nodes[:, 0], nodes[:, 1], 
               y_true[:, 0], y_true[:, 1],
               angles='xy', scale_units='xy', scale=0.05, color='blue')
axes[1].scatter(nodes[:, 0], nodes[:, 1], c='red', s=50, zorder=10)
axes[1].set_aspect('equal')
axes[1].set_title('真实位移')

# 右图：GCN 预测位移
axes[2].quiver(nodes[:, 0], nodes[:, 1],
               y_pred[:, 0], y_pred[:, 1],
               angles='xy', scale_units='xy', scale=0.05, color='red')
axes[2].scatter(nodes[:, 0], nodes[:, 1], c='red', s=50, zorder=10)
axes[2].set_aspect('equal')
axes[2].set_title('GCN 预测位移')

plt.tight_layout()
plt.savefig('week12_train_gcn.png', dpi=100)
print(f"\n图已保存到 week12_train_gcn.png")
```

**验收标准**：
- 训练能跑通
- Loss 下降到 < 1e-4（因为只有 1 张图，必然过拟合）
- 预测位移和真实位移**几乎完全相同**
- 你能解释"这个 Loss 这么低不代表模型好——它只是过拟合了 1 个样本"

**🔬 应用层**（约 5 分钟）：

**这次训练的真实意义**：

只用 1 张图训练只能验证：
- ✅ 模型能跑前向不报错
- ✅ Loss 能下降
- ✅ 模型有足够容量拟合数据
- ❌ **不能**说明模型有泛化能力（测试集还是这同一张图）

**真正的训练**应该有多个样本（多个不同载荷的板），并且 train/val 切分。这就是 Day 5-6 周末做的事。

---

#### ⚠️ Day 4 新手最容易踩的坑

**坑 1：以为 1 个样本能"学到"什么**
- 1 个样本训练后 Loss 极低——这是**过拟合**，不是学到规律
- 模型只是把这一个样本"背"下来了
- 必须有多样本 + train/val split 才能讨论泛化

**坑 2：lr 设得过大**
- 单样本训练时 lr=1e-2 可能合适
- 多样本训练通常用更小的 lr（如 1e-3）

**坑 3：把训练集 Loss 当成真实评估**
- 训练集 Loss 永远是越来越小
- 真正的评估必须看 val/test 集

---

#### 🧠 Day 4 概念问答 quiz

**Q1**：1 张图训练 GCN 最终 Loss 极低，说明什么？

<details><summary>答案</summary>说明模型有足够容量拟合这一个样本——但这是**过拟合**，不是学到规律。在新数据上模型很可能差。</details>

**Q2**：怎么验证 GCN 真的"学到"了什么？

<details><summary>答案</summary>需要多个样本 + train/val 切分。训练集 Loss 和 val 集 Loss 都降，且 val Loss 不离训练 Loss 太远，才是真正学到了。</details>

**Q3**：为什么单样本训练时可以用较大的 lr=1e-2？多样本时为什么要小？

<details><summary>答案</summary>单样本时数据少，损失曲面相对简单，大 lr 能快速收敛。多样本时损失更复杂，lr 大容易在不同样本梯度间震荡。多样本训练通常用 1e-3 或更小。</details>

---

#### 📦 Day 4 知识卡片

| 项目 | 内容 |
|---|---|
| **核心成就** | 第一次 GNN 训练 |
| **关键认知** | 单图训练只能验证"能跑"，不能验证"会泛化" |
| **真正训练** | 多样本 + train/val split（Day 5-6） |
| **本日产出** | `week12/day04_train_gcn_bar.py` |
| **掌握要求** | 🟣 能用（能跑通） |

---

### Day 5-6（周末）| 多图训练 + DataLoader

**🎯 本日目标**：构造多张图的数据集；用 PyG 的 DataLoader 做 batch 训练；建立 GNN 真实训练的工程经验。

**🟢 直觉层**（约 10 分钟）：

GNN 真实训练需要**多个样本**——每个样本是一张图。比如做 FEM 代理模型：
- 样本 1：F=50 时的板形变
- 样本 2：F=100 时的板形变
- 样本 3：F=150 时的板形变
- ...

模型从这些样本里学到"F 越大形变越大"的规律，然后能预测**新的 F 值**下的形变。

PyG 的 `DataLoader` 帮你把多张图组成 batch——内部机制和 Part A 的 DataLoader 不太一样（GNN 的 batch 是把多张小图**拼成一张大图**），但用法上类似。

**🟡 概念层**（约 20 分钟）：

**PyG DataLoader 的"图拼接"机制**：

假设有 3 张图：
- 图 1：4 个节点，6 条边
- 图 2：5 个节点，8 条边
- 图 3：3 个节点，4 条边

DataLoader 把它们拼成一张大图：
- 总节点数：4 + 5 + 3 = 12
- 总边数：6 + 8 + 4 = 18
- **节点编号自动偏移**：图 1 节点 0-3，图 2 节点 4-8（=4+0 到 4+4），图 3 节点 9-11

这样**每张小图内部的边连接关系保留不变**，但所有节点共用一个编号空间。

**关键属性**：

```python
batch.x          # (12, n_features) 所有节点特征拼接
batch.edge_index # (2, 18) 所有边
batch.y          # 标签（节点级或图级）
batch.batch      # (12,) 每个节点属于哪张图：[0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
```

`batch.batch` 是 PyG batch 对象**特有的属性**——告诉你每个节点属于哪张原始图。在做 pooling 等操作时关键。

**🟣 代码层**（约 2.5 小时）：

创建文件 `week12/day56_multi_graph.py`：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

# ========== 复用 build_plate_data ==========
def build_plate_data(nx=5, ny=3, F_applied=100.0):
    """同 Day 4，参数化 F"""
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([float(i), float(j)])
    nodes = np.array(nodes, dtype=np.float32)
    
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = j * nx + (i + 1)
            n2 = (j + 1) * nx + (i + 1)
            n3 = (j + 1) * nx + i
            elements.append([n0, n1, n2, n3])
    
    edge_set = set()
    for elem in elements:
        for i in range(4):
            for j in range(i+1, 4):
                a, b = min(elem[i], elem[j]), max(elem[i], elem[j])
                edge_set.add((a, b))
    src, dst = [], []
    for a, b in edge_set:
        src.append(a); dst.append(b)
        src.append(b); dst.append(a)
    edge_index = np.array([src, dst], dtype=np.int64)
    
    N = len(nodes)
    is_fixed = np.zeros((N, 1), dtype=np.float32)
    F_feat = np.zeros((N, 1), dtype=np.float32)
    is_fixed[np.abs(nodes[:, 0]) < 0.01] = 1.0
    F_feat[np.abs(nodes[:, 0] - nodes[:, 0].max()) < 0.01] = F_applied
    
    x_features = np.concatenate([nodes, is_fixed, F_feat], axis=1)
    
    # 模拟位移：随 F 缩放
    y_center = (ny - 1) * 0.5
    ux = 0.01 * (F_applied / 100) * nodes[:, 0:1]
    uy = -0.003 * (F_applied / 100) * nodes[:, 0:1] * (nodes[:, 1:2] - y_center)
    ux[np.abs(nodes[:, 0]) < 0.01] = 0
    uy[np.abs(nodes[:, 0]) < 0.01] = 0
    y_label = np.concatenate([ux, uy], axis=1).astype(np.float32)
    
    return Data(
        x=torch.from_numpy(x_features).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        y=torch.from_numpy(y_label).float()
    )


# ========== 模型 ==========
class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x)


# ========== Step 1: 生成多图数据集 ==========
F_train = [50, 75, 100, 125, 150, 175]
F_val = [60, 110]    # 不同的 F 测试泛化
F_test = [200]        # 在训练范围外测试外推

train_data = [build_plate_data(F_applied=F) for F in F_train]
val_data = [build_plate_data(F_applied=F) for F in F_val]

print(f"训练集: {len(train_data)} 张图")
print(f"验证集: {len(val_data)} 张图")
print(f"每张图节点数: {train_data[0].num_nodes}")

# ========== Step 2: PyG DataLoader ==========
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)

# 看看 batch 是什么样
for batch in train_loader:
    print(f"\nBatch 信息:")
    print(f"  batch.x shape: {batch.x.shape}")
    print(f"  batch.edge_index shape: {batch.edge_index.shape}")
    print(f"  batch.y shape: {batch.y.shape}")
    print(f"  batch.batch: {batch.batch}")    # 每个节点属于哪张图
    print(f"  num_graphs: {batch.num_graphs}")
    break

# ========== Step 3: 训练 ==========
torch.manual_seed(0)
model = SimpleGCN(in_channels=4, hidden_channels=64, out_channels=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_losses = []
val_losses = []

print("\n开始训练（多图 batch）...")
for epoch in range(300):
    # Train
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        y_pred = model(batch)
        loss = criterion(y_pred, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * batch.num_graphs
    avg_train = total_train_loss / len(train_data)
    train_losses.append(avg_train)
    
    # Validate
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            y_pred = model(batch)
            total_val_loss += criterion(y_pred, batch.y).item() * batch.num_graphs
    avg_val = total_val_loss / len(val_data)
    val_losses.append(avg_val)
    
    if epoch % 30 == 0:
        print(f"  epoch {epoch}: train={avg_train:.6f}, val={avg_val:.6f}")

# ========== Step 4: 可视化 ==========
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.yscale('log')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.title('多图 GCN 训练')
plt.legend()
plt.savefig('week12_multi_graph_train.png', dpi=100)
print(f"\n最终 train Loss: {train_losses[-1]:.6f}")
print(f"最终 val Loss: {val_losses[-1]:.6f}")
print(f"val/train 比: {val_losses[-1] / train_losses[-1]:.2f}")
```

**验收标准**：
- 多图训练能跑通
- 你能看懂 `batch.batch` 的含义
- val Loss 不离 train Loss 过远（通常比 1.5-3 倍内）
- 你能口头解释 PyG DataLoader 的"图拼接"机制

---

#### ⚠️ Day 5-6 新手最容易踩的坑

**坑 1：以为 PyG batch 是简单的"叠在第 0 维"**
- MLP 的 batch：把 (1, features) 叠成 (B, features)
- GNN 的 batch：把多张小图**拼成一张大图**——节点编号会偏移
- 不理解这个机制，看到 batch.x.shape 是 (12, 4) 而不是 (3, 4, 4) 会蒙

**坑 2：忘了 `batch.num_graphs`**
- 算 epoch 平均 Loss 时要除以总样本数（图数）
- `len(train_loader.dataset)` 给你总图数
- 不要用 `len(train_loader)`（那是 batch 数）

**坑 3：训练集和验证集的 F 值范围相同**
- 例：train 是 [50, 75, 100, 125, 150]、val 是 [60, 110]——val 都在 train 范围内（**内插**）
- 这容易让 val Loss 假性低
- 真正的泛化测试应该测**外推**（如 F=200，在 train 范围外）

**坑 4：归一化没做**
- 节点特征量级差异大（坐标 0-4 vs F 0-100）
- 没归一化训练效果差
- 标准做法：训练前对节点特征做 Z-Score 归一化（用 train_data 算 mean/std）
- 本日为了简化没做，但**真实工程必须做**

---

#### 🧠 Day 5-6 概念问答 quiz

**Q1**：PyG 的 DataLoader 和 PyTorch 的 DataLoader 有什么核心不同？

<details><summary>答案</summary>PyTorch 的 DataLoader 把多个样本叠在第 0 维（如 (3, 4, 4)）。PyG 的把多张图**拼成一张大图**——节点编号偏移，所有节点放在同一维度（如 (12, 4)）。</details>

**Q2**：`batch.batch` 这个属性是干什么的？

<details><summary>答案</summary>形状 (N_total,) 的 long tensor，告诉你每个节点属于哪张原始图。如 [0,0,0,1,1,1,1,2,2]——前 3 个节点属于第 0 张图。在做 pooling、损失计算等场景关键。</details>

**Q3**：算 epoch 平均 Loss 时为什么要乘 `batch.num_graphs`？

<details><summary>答案</summary>不同 batch 的图数可能不同（最后一个 batch 可能不满），不加权直接平均会让小 batch 权重过大。乘 num_graphs 累加，最后除以 `len(dataset)`，得到真正的 per-graph 平均 Loss。</details>

**Q4**：怎么测试 GCN 的真实泛化能力？

<details><summary>答案</summary>val 集 F 值要**超出 train 范围**（外推），而不是只在 train 范围内（内插）。外推性能好才说明模型学到了规律而不是简单插值。</details>

**Q5**：为什么 GCN 训练强烈推荐归一化输入特征？

<details><summary>答案</summary>节点特征量纲差异大（坐标 vs 载荷）会让训练不稳定。和 Part A Week 7 的 Z-Score 同理。FEM 数据特别需要——这是论文 §2.2 反复强调的。</details>

---

#### 📦 Day 5-6 知识卡片

| 项目 | 内容 |
|---|---|
| **核心 API** | `torch_geometric.loader.DataLoader` |
| **PyG batch 机制** | 多张小图拼成一张大图，节点编号偏移 |
| **关键属性** | `batch.x`, `batch.edge_index`, `batch.batch`, `batch.num_graphs` |
| **泛化测试** | val 用外推 F 值（不要只测内插） |
| **关键纪律** | 必须归一化（GNN 同样适用） |
| **本日产出** | `week12/day56_multi_graph.py` |
| **掌握要求** | 🟣 能用 |

---

### ✅ Week 12 完成评估

#### 任务级 checklist

- [ ] `week12/day02_gcn_numpy.py` NumPy 手写 GCN 跑通
- [ ] `week12/day03_gcn_pyg.py` PyG GCNConv 模型创建成功
- [ ] `week12/day04_train_gcn_bar.py` 单图训练 Loss < 1e-4
- [ ] `week12/day56_multi_graph.py` 多图训练，val Loss 合理

#### 概念级 quiz（10 题，至少 8 题对）

**Q1**：GCN 的核心公式 `H = σ(D^(-1/2) Â D^(-1/2) X W)` 中，每一项的意义？

<details><summary>答案</summary>Â=A+I 加自环；D^(-1/2) 度归一化；X·W 特征变换；σ 激活。整体：邻居（含自己）特征加权平均 → 线性变换 → 激活。</details>

**Q2**：为什么 GCN 必须加自环？

<details><summary>答案</summary>原始 A 对角线为 0，节点自己的特征不会被聚合。加 I 让对角线为 1，节点自己也参与。</details>

**Q3**：GCN 的归一化为什么用 `D^(-1/2)` 两边乘？

<details><summary>答案</summary>对称归一化保持矩阵对称性，符合无向图的对称特性。也可用非对称 `D^(-1)·Â`，但 GCN 论文用对称版。</details>

**Q4**：GCN 的激活为什么可以用 ReLU？什么时候必须用 Tanh？

<details><summary>答案</summary>GCN 没有"对输入求高阶导"的需求，ReLU 没问题。只有 PINN 风格的网络（要求二阶导）才必须 Tanh。</details>

**Q5**：堆 10 层 GCN 会有什么问题？

<details><summary>答案</summary>**过平滑（over-smoothing）**——所有节点特征趋于相似。GCN 通常用 2-3 层。要更深需要 residual 等技巧。</details>

**Q6**：典型 GCN 模型为什么最后用 `nn.Linear` 而不是 `GCNConv`？

<details><summary>答案</summary>最后一层把隐特征映射到任务输出（如位移），不需要再聚合邻居信息。Linear 只做特征变换。</details>

**Q7**：单样本训练 Loss 极低能说明模型好吗？

<details><summary>答案</summary>**不能**。是过拟合，不是泛化。必须多样本 + train/val split + 测外推才能讨论模型质量。</details>

**Q8**：PyG 的 DataLoader 怎么处理多张图的 batch？

<details><summary>答案</summary>把多张小图拼成一张大图，节点编号自动偏移。batch.batch 属性记录每个节点属于哪张原图。</details>

**Q9**：算 epoch 平均 Loss 时为什么要乘 `batch.num_graphs`？

<details><summary>答案</summary>不同 batch 图数可能不同（最后一批不满）。不加权平均会让小 batch 权重过大。加权后除以总图数才是 per-graph 平均。</details>

**Q10**：GCN 在 FEM 任务里的局限是什么？为什么 PhyFENet 论文用 GraphSAGE？

<details><summary>答案</summary>GCN 的局限：(1) 全图归一化操作不便于动态变化的网格；(2) 邻居等权聚合不够灵活；(3) 节点自身和邻居特征"混在一起"。GraphSAGE 把节点自身和邻居特征**分开处理**，提供更多设计自由度——这是论文 §2.1.2 的论证主线。</details>

#### 🚦 自我评估

- 任务全部通过 + 单图过拟合 + 多图训练合理 + Quiz 8 题对 → **绿灯进入 Week 13**
- 多图训练不能跑通（最常见是 PyG DataLoader 用错） → **黄灯**——重看 Day 5-6 概念层
- 不能解释 GCN 为何加自环和归一化 → **黄灯**——重看 Day 1

#### 知识地图自查

- [ ] (5) GCN 公式 → 🟡
- [ ] (6) NumPy 手写 → 🔵
- [ ] (7) PyG GCNConv → 🟣
- [ ] (8) 多图训练 → 🟣

---

# Part C Week 13-14 

> Week 12 的 GCN 是"对照周"。从本周起进入 **Part C 真正的主力内容**——你为最终目标准备的核心技能。
>
> ⚠️ **密集区提醒**：Week 13-15 是 Part C 的技术密集区，每周都引入新的模型结构和实验。如果你感觉某一周的内容两周才消化得动，**这是正常的**——按实际进度走，不要硬推。**进入 Week 16 之前，要求是 Week 13-15 的核心模块都能独立跑通**，不要求"每周准时完成"。
>
> 典型的"吃力信号"：
> - 写 `MessagePassing` 自定义层时，`x_i / x_j / edge_attr` 三个参数的传递关系搞不清
> - 带边更新层写出来跑不通，不知道怎么调试
> - 训练 loss 不降或震荡，分不清是代码错还是超参问题
>
> 遇到以上任一情况，**停下来先解决，不要跳**。

---

## Week 13: 消息传递范式 + GraphSAGE

### 🎯 Week 13 总览

**本周覆盖的知识点**：(9) 消息传递三阶段抽象、(10) GraphSAGE（论文用的主力模型）、自定义 MessagePassing 层

**本周不覆盖**：边更新（Week 14）、Encoder-Processor-Decoder（Week 15）

**学完之后你应该能**：
- 🟣 **能用**：用 `MessagePassing` 基类写自定义层；用 `SAGEConv` 训练 GraphSAGE
- 🟡 **能讲**：消息传递三阶段；GCN vs GraphSAGE 的本质差异

**本周的特点**：**这周才是 Part C 真正的开始**。Week 12 的 GCN 是"垫脚石"，让你建立"图卷积"的第一直觉。Week 13 学的消息传递是 GNN 的**通用抽象**——后面所有 GNN 模型（GraphSAGE、GAT、PhyFENet）都是它的特例。

---

### ✅ 进入 Week 13 之前的前置 checklist

- [ ] Week 12 的 GCN 多图训练能跑通
- [ ] 我能解释 GCN 公式每一项的意义
- [ ] 我能口头说出 GCN 的 2-3 个局限（提示：Week 12 Day 1 / 论文 §2.1.2）
- [ ] 我接受"消息传递抽象比 GCN 公式更通用，需要重新建立直觉"
- [ ] 我能熟练用 PyG 的 `Data` 对象和 DataLoader

---

### Day 1 | 消息传递的三阶段抽象

**🎯 本日目标**：理解消息传递（Message Passing）的通用范式；理解 GCN 是它的一个特例。

**🟢 直觉层**（约 15 分钟）：

把图想象成社交网络：每个节点是一个人，边是朋友关系。

**消息传递的一轮（一层 GNN）做的事**：

1. **Message（生成消息）**：每个人**对每个朋友说一句话**。这句话的内容可能取决于：
   - 你自己的状态
   - 朋友的状态
   - 你们之间的关系（边特征）

2. **Aggregate（聚合消息）**：每个人**把所有收到的话归纳一下**。比如取平均、取最大值、加权和。

3. **Update（更新自己）**：每个人**根据归纳结果更新自己的想法**。比如把"原来的想法"和"邻居的归纳"结合起来。

**重复 K 层 = K 轮交流** = 你的状态会受到 K 跳之外人的间接影响。

> ⚠️ 比喻警告：社交网络比喻只用一次，下面进入精确数学。

**🟡 概念层**（约 25 分钟）：

**消息传递的数学形式**（论文 §2.3.3 风格）：

对每个节点 i，每一层做：

```
1. Message: 对每条入边 (j → i)，生成消息：
   m_{j→i} = φ(x_j, x_i, e_{ji})
   其中 φ 是一个可学习函数（通常是 MLP）

2. Aggregate: 把所有入边消息聚合：
   a_i = AGG({m_{j→i} : j 是 i 的入邻居})
   其中 AGG 是 sum / mean / max 等聚合函数

3. Update: 用聚合结果更新节点：
   x_i' = ψ(x_i, a_i)
   其中 ψ 也是一个可学习函数（通常是 MLP）
```

**这是一个非常通用的抽象**——具体的 GNN 模型只是给 φ、AGG、ψ 选不同的形式：

| 模型 | Message φ | AGG | Update ψ |
|---|---|---|---|
| GCN | `c_{ji} · x_j`（c 是归一化系数） | sum | 直接是聚合结果 + 线性 + ReLU |
| GraphSAGE | `x_j` | mean | concat(x_i, agg) → MLP |
| GAT | `α_{ji} · x_j`（α 是注意力权重） | sum | MLP |
| 论文 PhyFENet | 涉及边更新（Week 14） | mean | MLP |

**关键洞察**：你现在不只学会了一个 GNN 模型，而是**学会了一个生成 GNN 模型的"框架"**。看到任何新 GNN 模型，你都能问：它的 φ 是什么？AGG 是什么？ψ 是什么？

**🔵 数学层**（约 5 分钟）：

**GCN 是消息传递的一个特例**——验证：

GCN 公式：`h_i' = σ(Σ_j c_{ij} · h_j · W)`，其中 `c_{ij} = 1/√(d_i · d_j)`（含自环）。

拆解到三阶段：
- Message：`m_{j→i} = c_{ij} · h_j · W`（线性变换 + 归一化系数）
- AGG：sum
- Update：`x_i' = σ(a_i)`（直接激活，没有进一步处理）

所以 **GCN 就是消息传递的一个具体实例**——你之前学的不白学，它现在变成了大框架的一个例子。

**🟣 代码层**（无独立任务）：

今天理论为主。Day 2 开始用 PyG 的 `MessagePassing` 基类写代码。

---

#### ⚠️ Day 1 新手最容易踩的坑

**坑 1：以为消息传递是某种"新模型"**
- 它**不是**模型——它是一个**通用框架**
- 所有具体的 GNN 模型都是它的特例
- 把它理解成"GNN 的设计语言"

**坑 2：把 message 和 aggregate 搞混**
- message：**每条边**生成一条消息（粒度是边）
- aggregate：**每个节点**把它收到的消息归纳（粒度是节点）
- 这两步在不同的"层级"工作

**坑 3：以为只有"sum"和"mean"两种 AGG**
- 常用的还有 max、attention（如 GAT）、自定义函数
- AGG 是设计选择，不是固定的

**坑 4：以为节点 i 的"邻居"自动包含 i 自己**
- 在 GCN 公式里加自环之后才包含
- 在原始消息传递里，要看怎么用 `x_i` 和聚合结果——可以单独传入也可以不传
- 通用做法：在 Update 阶段传入 `x_i`，让模型自己决定怎么用

---

#### 🧠 Day 1 概念问答 quiz

**Q1**：消息传递的三个阶段是什么？

<details><summary>答案</summary>(1) Message：每条边生成一条消息（基于源节点、目标节点、边特征）；(2) Aggregate：每个节点把所有入边消息聚合（sum/mean/max 等）；(3) Update：节点用原特征 + 聚合结果更新自己。</details>

**Q2**：GCN 是消息传递的特例吗？请说出它的 message、aggregate、update 各是什么。

<details><summary>答案</summary>**是**。Message：`c_{ij} · h_j · W`（带归一化的线性变换源特征）；AGG：sum；Update：`σ(a_i)`（直接对聚合结果激活）。</details>

**Q3**：消息传递相对于"只学 GCN"的优势是什么？

<details><summary>答案</summary>消息传递是通用框架——可以自定义 message 函数（带边特征、带注意力、带任意非线性）、自定义聚合函数（max、attention）、自定义更新函数。GCN 只是其中一个具体选择。学会消息传递 = 学会**设计**新 GNN 模型。</details>

**Q4**：消息传递的 Message 阶段输出的张量 shape 应该是什么？

<details><summary>答案</summary>`(n_edges, n_features)` —— 每条边一条消息。注意是边的数量（n_edges），不是节点数。</details>

**Q5**：Aggregate 阶段输出的张量 shape 应该是什么？

<details><summary>答案</summary>`(n_nodes, n_features)` —— 每个节点一个聚合结果。从边粒度（n_edges）"收回"到节点粒度（n_nodes）。</details>

---

#### 📦 Day 1 知识卡片

| 项目 | 内容 |
|---|---|
| **核心抽象** | Message → Aggregate → Update |
| **设计自由度** | φ（消息函数）、AGG（聚合方式）、ψ（更新函数）三处都可以自定义 |
| **粒度** | Message 在边粒度；Aggregate 和 Update 在节点粒度 |
| **关键洞察** | 学会消息传递 = 学会设计 GNN 模型 |
| **本日产出** | 笔记上写"消息传递三阶段" + GCN 是它的特例 |
| **掌握要求** | 🟡 能讲 |

---

### Day 2 | PyG `MessagePassing` 基类

**🎯 本日目标**：用 PyG 的 `MessagePassing` 基类写第一个自定义图卷积层。

**🟢 直觉层**（约 5 分钟）:

PyG 把消息传递的三阶段封装成一个基类 `MessagePassing`。你只需要：
1. 继承 `MessagePassing`
2. 在 `__init__` 里指定 AGG（aggr='mean' / 'sum' / 'max'）
3. 实现 `message` 方法（生成消息）
4. （可选）实现 `update` 方法（更新节点）
5. 在 `forward` 里调 `self.propagate()` 触发整个流程

PyG 会**自动**做"按 edge_index 收集源节点、按 target 节点聚合"等繁琐操作。

**🟡 概念层**（约 25 分钟）：

**关键命名约定（必须记住）**：

PyG 用 `_i` 和 `_j` 后缀区分目标节点和源节点：

| 符号 | 含义 |
|---|---|
| `x_j` | source 节点的特征 |
| `x_i` | target 节点的特征 |
| `edge_attr` | 边特征（不需要 _i / _j 后缀） |

**记忆方法**：j → i 这个方向（j 是源，i 是目标，因为约定信息从 j 流向 i）。

**`message` 方法的参数命名**：

```python
def message(self, x_j):                       # 只用源节点特征
    return x_j

def message(self, x_i, x_j):                  # 同时用源和目标
    return torch.cat([x_i, x_j], dim=-1)

def message(self, x_j, edge_attr):            # 用源 + 边特征
    return torch.cat([x_j, edge_attr], dim=-1)

def message(self, x_i, x_j, edge_attr):       # 三者都用（最复杂）
    return some_function(x_i, x_j, edge_attr)
```

**PyG 的"魔法"**：你写 `def message(self, x_j)`，PyG 会自动从 `propagate(edge_index, x=x)` 里的 `x` 按 `edge_index[0]`（source 索引）抽取出对应的节点特征。**你不需要手动做这件事**。

**`propagate` 的核心调用**：

```python
def forward(self, x, edge_index):
    return self.propagate(edge_index, x=x)
    #                     ^^^^^^^^^^^  ^^^
    #                     必须传        会被 PyG 拆成 x_i 和 x_j
```

**`propagate` 内部做的事**（伪代码）：

```python
# 1. Message: 收集源/目标，调用 self.message
src, dst = edge_index[0], edge_index[1]
x_j = x[src]    # 源节点特征 (n_edges, ...)
x_i = x[dst]    # 目标节点特征 (n_edges, ...)
m = self.message(x_i=x_i, x_j=x_j, ...)

# 2. Aggregate: 按 dst 聚合（用 self.aggr 指定的方法）
aggr_out = scatter(m, dst, dim=0, reduce=self.aggr)
# aggr_out shape: (n_nodes, ...)

# 3. Update
out = self.update(aggr_out, x=x, ...)

return out
```

**🔵 数学层**（约 5 分钟）：

无新数学。这节是把 Day 1 的抽象用代码实现。

**🟣 代码层**（约 2 小时）：

创建文件 `week13/day02_message_passing_basic.py`：

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

class SimpleGNNLayer(MessagePassing):
    """最简单的 GNN 层：mean-aggregate + 线性变换"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')         # 指定聚合方式
        self.lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        x: (N, in_channels) 节点特征
        edge_index: (2, n_edges)
        """
        # propagate 触发 message + aggregate + update
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        """每条边的消息 = source 节点的特征
        
        x_j: (n_edges, in_channels)，PyG 自动从 x 提取
        """
        return x_j
    
    def update(self, aggr_out):
        """聚合结果通过线性层 + ReLU 得到新节点特征
        
        aggr_out: (N, in_channels)
        """
        return torch.relu(self.lin(aggr_out))


# ========== 测试 ==========
torch.manual_seed(0)

# 4 节点图，6 条无向边（双向写 12 条）
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 1, 0, 2, 1, 3, 2],
    [1, 0, 2, 1, 3, 2, 0, 1, 1, 2, 2, 3]
], dtype=torch.long)

x = torch.randn(4, 3)        # 4 个节点，3 维特征

layer = SimpleGNNLayer(in_channels=3, out_channels=8)
out = layer(x, edge_index)

print(f"输入 x shape: {x.shape}")
print(f"输出 out shape: {out.shape}")    # 应为 (4, 8)
print(f"\n输出:\n{out}")

# ========== 验证：手动算一次，对比 ==========
print("\n===== 验证（手动算一遍）=====")

# Step 1: 找到每个节点的入邻居
src = edge_index[0].tolist()
dst = edge_index[1].tolist()

# 节点 0 的入邻居（即 dst=0 的所有 src）
neighbors_0 = [src[i] for i in range(len(dst)) if dst[i] == 0]
print(f"节点 0 的入邻居（应该是 1）: {neighbors_0}")

# Step 2: 手动 mean
manual_aggr_0 = x[neighbors_0].mean(dim=0)
print(f"节点 0 的手动聚合: {manual_aggr_0}")

# Step 3: 手动 update
manual_out_0 = torch.relu(layer.lin(manual_aggr_0))
print(f"节点 0 的手动输出: {manual_out_0}")

# 对比 PyG 的输出
print(f"PyG 的输出:        {out[0]}")
print(f"\n是否一致: {torch.allclose(manual_out_0, out[0], atol=1e-6)}")
```

**验收标准**：
- 代码无报错
- 输出 shape = (4, 8)
- 手动算的结果和 PyG 一致（用 `torch.allclose` 验证）
- 你能解释每一步在做什么

---

#### ⚠️ Day 2 新手最容易踩的坑

**坑 1：`message` 函数的参数命名错了**
- 错误：`def message(self, x_source, x_target)` 
- 正确：`def message(self, x_j, x_i)`（必须用 `_j` 和 `_i`）
- PyG 通过参数名识别"源"和"目标"——名字写错，PyG 不知道怎么传值

**坑 2：忘记 `aggr=` 参数**
- 错误：`super().__init__()`（不指定 aggr）
- 正确：`super().__init__(aggr='mean')` 或 'sum' / 'max'
- 不指定的话默认是 'add'（即 sum）——可能不是你想要的

**坑 3：`propagate` 没传 x 进去**
- 错误：`self.propagate(edge_index)`
- 正确：`self.propagate(edge_index, x=x)`
- 不传 x，message 函数里的 `x_j` 不知道从哪里来——报错

**坑 4：`message` 返回 shape 错**
- message 应该返回 `(n_edges, output_channels)` 形状
- 不是 (n_nodes, ...)
- 这是消息粒度——每条边一条消息

**坑 5：以为 `x_j` 是个 tensor 类**
- `x_j` 是 PyG 在调用时**自动构造**的——是 `x[edge_index[0]]` 的结果
- 你不能在外面直接访问 `layer.x_j`
- 它只在 `message` 函数内部存在

---

#### 🧠 Day 2 概念问答 quiz

**Q1**：在 `MessagePassing` 子类的 `message` 方法里，`x_i` 和 `x_j` 分别代表什么？怎么记住？

<details><summary>答案</summary>`x_j`：source 节点（边的起点）；`x_i`：target 节点（边的终点）。记忆：信息从 j 流向 i（j → i）。</details>

**Q2**：`aggr='mean'` 和 `aggr='sum'` 在 PINN/FEM 任务里通常哪个更好？

<details><summary>答案</summary>**通常 mean** 更好，因为不同节点度数不同，sum 会让度数大的节点聚合值过大（和 GCN 里要做归一化的原因一样）。mean 自动消除度数差异。具体哪个最好取决于任务——可以两种都试。</details>

**Q3**：以下代码哪一行有问题？
```python
class MyLayer(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index)
    
    def message(self, x_source):
        return x_source
```

<details><summary>答案</summary>**两个错**：(1) `super().__init__()` 没指定 `aggr`，默认是 'add'（可能不是你想要的）；(2) `propagate(edge_index)` 没传 x，导致 message 里 x_source 不知道是什么；(3) 命名错了——应该是 `x_j`（PyG 通过名字识别）不是 `x_source`。</details>

**Q4**：`message` 方法返回的 tensor shape 应该是？`update` 方法返回的呢？

<details><summary>答案</summary>message：`(n_edges, ...)` —— 每条边一条消息。update：`(n_nodes, ...)` —— 每个节点一个新特征。</details>

**Q5**：PyG 怎么知道每条边的"源"和"目标"是哪个节点？

<details><summary>答案</summary>从 `edge_index` 提取——`edge_index[0]` 是所有 source 索引，`edge_index[1]` 是所有 target 索引。`propagate` 内部用这两组索引去 `x` 里抽取对应特征，分别送给 `message` 函数的 `x_j` 和 `x_i` 参数。</details>

---

#### 📦 Day 2 知识卡片

| 项目 | 内容 |
|---|---|
| **核心 API** | `class XX(MessagePassing): __init__(aggr='mean')` |
| **必实现方法** | `forward(...)` 调 `self.propagate`；`message(...)` 生成消息 |
| **可选方法** | `update(...)` 默认是恒等 |
| **命名约定** | `x_j` source；`x_i` target |
| **常见错误** | 命名错（用 x_source）；忘 `aggr=`；propagate 没传 x |
| **本日产出** | `week13/day02_message_passing_basic.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 3 | 给 message 加入更复杂的逻辑

**🎯 本日目标**：扩展 message 函数——使用源节点 + 目标节点 + 边特征。这是论文消息传递的基础形式。

**🟢 直觉层**（约 5 分钟）：

Day 2 的 message 只用了 `x_j`（源特征）。但实际场景里 message 经常需要更多信息：

- **同时知道源和目标**：比如计算"我对你说的话"时，要看"我"是什么状态、"你"是什么状态
- **知道边的关系**：比如"我们距离有多远"、"我们之间是哪种类型的关系"

今天扩展 message 函数到这些场景。

**🟡 概念层 + 代码层**（合并讲，约 2.5 小时）：

创建文件 `week13/day03_complex_message.py`：

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

# ========== 例 1：同时用 x_i 和 x_j ==========

class BothNodesLayer(MessagePassing):
    """消息函数同时用源和目标节点的特征"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        # 注意：in_channels * 2，因为我们要拼接两个节点的特征
        self.lin = nn.Linear(in_channels * 2, out_channels)
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        """消息 = 源和目标特征的拼接 + 线性变换 + 激活"""
        cat = torch.cat([x_i, x_j], dim=-1)    # (n_edges, 2*in_channels)
        return torch.relu(self.lin(cat))         # (n_edges, out_channels)
    
    def update(self, aggr_out):
        """直接返回聚合结果"""
        return aggr_out


# ========== 例 2：用 edge_attr ==========

class EdgeAwareLayer(MessagePassing):
    """消息函数使用边特征"""
    
    def __init__(self, in_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels + edge_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        # 注意：把 edge_attr 也传进 propagate
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        """消息 = 源特征 + 边特征 → 线性变换 + 激活"""
        cat = torch.cat([x_j, edge_attr], dim=-1)
        return torch.relu(self.lin(cat))


# ========== 例 3：综合 (x_i, x_j, edge_attr) ==========

class FullMessageLayer(MessagePassing):
    """最完整的消息函数：源、目标、边特征都用"""
    
    def __init__(self, in_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')
        # 输入维度 = 2 个节点特征 + 1 个边特征
        self.lin = nn.Linear(in_channels * 2 + edge_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        cat = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return torch.relu(self.lin(cat))


# ========== 测试 ==========
torch.manual_seed(0)
N = 5
n_edges = 10

x = torch.randn(N, 4)        # 节点特征 4 维
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 0, 4],
    [1, 0, 2, 1, 3, 2, 4, 3, 4, 0]
], dtype=torch.long)
edge_attr = torch.randn(n_edges, 3)    # 边特征 3 维

# 测试三种层
layer1 = BothNodesLayer(in_channels=4, out_channels=16)
out1 = layer1(x, edge_index)
print(f"BothNodesLayer 输出 shape: {out1.shape}")    # (5, 16)

layer2 = EdgeAwareLayer(in_channels=4, edge_channels=3, out_channels=16)
out2 = layer2(x, edge_index, edge_attr)
print(f"EdgeAwareLayer 输出 shape: {out2.shape}")    # (5, 16)

layer3 = FullMessageLayer(in_channels=4, edge_channels=3, out_channels=16)
out3 = layer3(x, edge_index, edge_attr)
print(f"FullMessageLayer 输出 shape: {out3.shape}")  # (5, 16)
```

**验收标准**：
- 三个层都能跑通
- 三个层的输出 shape 都正确
- 你能口头解释：什么场景用哪种

**🔬 应用层**（约 5 分钟）：

**这三种 message 函数对你的方向（FEM）的意义**：

| 层类型 | 适用场景 |
|---|---|
| 只用 `x_j` | 简单基线（GraphSAGE 风格） |
| 用 `x_i, x_j` | 想区分"我对你说"和"你对我说"（GAT 风格的注意力可以基于此） |
| 用 `x_j, edge_attr` | 边特征重要的场景（如距离、相对位置） |
| 用 `x_i, x_j, edge_attr` | 论文 PhyFENet 的消息传递基础（Week 14 用） |

**论文 §2.3.3 的边更新公式 `e' = MLP(e, n_s, n_k)`**——本质就是 `FullMessageLayer` 的 message 函数（输入是源 n_s + 目标 n_k + 边 e）。

---

#### ⚠️ Day 3 新手最容易踩的坑

**坑 1：拼接后忘记调整 Linear 的输入维度**
- 错误：`self.lin = nn.Linear(in_channels, out_channels)`（拼接后维度是 2*in_channels，不是 in_channels）
- 正确：`nn.Linear(in_channels * 2, out_channels)`

**坑 2：`forward` 没把 `edge_attr` 传进 `propagate`**
- 错误：`return self.propagate(edge_index, x=x)`（漏 edge_attr）
- 正确：`return self.propagate(edge_index, x=x, edge_attr=edge_attr)`
- 不传 propagate，message 函数里的 edge_attr 是 None

**坑 3：拼接的维度搞错**
- 错误：`torch.cat([x_i, x_j], dim=0)`（在节点维度拼接，shape 翻倍）
- 正确：`torch.cat([x_i, x_j], dim=-1)`（在特征维度拼接）
- `dim=-1` 是最后一维，最稳健

---

#### 🧠 Day 3 概念问答 quiz

**Q1**：什么场景下 message 需要用 `x_i`（target 节点的特征）？

<details><summary>答案</summary>当消息需要根据"接收者是谁"来定制时。比如 GAT 用 `x_i, x_j` 计算注意力权重——"我说什么"取决于"我和你的关系强弱"。或者论文 §2.3.3 的边更新——边的更新需要边两端节点的特征。</details>

**Q2**：`message(self, x_j, edge_attr)` 函数的作用范围是什么？怎么用 `edge_attr`？

<details><summary>答案</summary>对**每条边**调用一次。`edge_attr` 是这条边的特征（shape=(n_edges, edge_channels)，自动广播匹配）。在 message 内部 cat 一下传给 MLP 即可。</details>

**Q3**：把 `dim=0` 写成 `dim=-1` 有什么区别？

<details><summary>答案</summary>`dim=0` 是第一维（边维度），拼接会让边数翻倍。`dim=-1` 是最后一维（特征维度），拼接会让特征维度翻倍。**消息拼接通常用 `dim=-1`**——拼接的是"两类特征"，不是"两批边"。</details>

**Q4**：消息函数 `message(self, x_i, x_j, edge_attr)` 返回的 shape 应该是什么？

<details><summary>答案</summary>`(n_edges, out_channels)`。每条边一条消息，消息维度由你的 MLP 决定。注意不要返回 `(n_nodes, ...)`，那是聚合后的结果。</details>

**Q5**：论文 §2.3.3 的 `e' = MLP(e, n_s, n_k)` 在 PyG 框架里对应什么？

<details><summary>答案</summary>对应 `FullMessageLayer` 的 message 函数 `message(self, x_i=n_k, x_j=n_s, edge_attr=e)` —— 输入是 source、target、边特征三者拼接，过 MLP 得到新边特征。</details>

---

#### 📦 Day 3 知识卡片

| 项目 | 内容 |
|---|---|
| **三种 message 函数** | `(x_j)` 、`(x_i, x_j)` 、`(x_j, edge_attr)` 、`(x_i, x_j, edge_attr)` |
| **拼接维度** | `dim=-1`（特征维度） |
| **必传参数** | propagate 时传 `x=x, edge_attr=edge_attr`（如果用边特征） |
| **常见错误** | Linear 维度算错；forward 没传 edge_attr；dim 用 0 |
| **本日产出** | `week13/day03_complex_message.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 4 | GraphSAGE 的思想（论文用的主力模型）

**🎯 本日目标**：理解 GraphSAGE 设计动机；建立"GCN vs GraphSAGE"的清晰对比。

**🟢 直觉层**（约 10 分钟）：

GCN 有什么问题？

设想你是一个节点，邻居有 5 个朋友。GCN 的做法是：
> "把我自己和 5 个朋友的特征**全部混在一起**做归一化平均。"

这有什么问题？**你自己的特征"被稀释了"**——你只是 6 个特征中的一个（加上自环后），无法和邻居的特征区分开。

**GraphSAGE 的改进**（论文 §2.1.2 引用）：
> "GCN 适用于每次更新都需要更新全图节点特征的情况，同时更新过程中所有节点都具有相同的权重。"

> "GraphSAGE 被用于解决这些问题和局限性。"

**核心思想**：**把"我自己的特征"和"邻居的聚合"分开处理**——明确区分。

**🟡 概念层**（约 25 分钟）：

**GraphSAGE 的消息传递**（简化版，论文用的形式）：

```
1. Message:    m_j = x_j           （直接用源节点特征作为消息——简单）
2. Aggregate:  agg_i = mean({m_j : j ∈ N(i)})    （邻居的均值聚合）
3. Update:     x_i' = ReLU(W · concat(x_i, agg_i))    
              （把"自己"和"邻居聚合"拼接，再过线性层）
```

**关键差异（GCN vs GraphSAGE）**：

| 维度 | GCN | GraphSAGE |
|---|---|---|
| 节点自身处理 | 加自环（A + I）→ 自己被混入邻居池 | **拼接**（concat）→ 自己被独立保留 |
| 是否需完整 A | 需要（公式涉及 A 矩阵）| 不需要（每个节点独立处理） |
| 是否能处理变节点数 | 困难 | **容易**（每个节点是局部操作） |
| 节点身份信息 | 被稀释 | 完整保留 |
| 公式简洁性 | 全图矩阵公式 | 每节点独立公式 |

**为什么 GraphSAGE 适合论文方向**：

PhyFENet 论文目标是**做不同网格大小的 FEM 推断**——不同样本可能节点数不同。
- GCN 需要矩阵公式 `D^(-1/2)·Â·D^(-1/2)`，**强依赖完整邻接矩阵 + 固定节点编号**
- GraphSAGE 是 per-node 的局部操作，**新图扔进来直接能用**——这就是"归纳学习"

实际工程意义：训练时用 5×3 板，推断时来了一张 7×4 板，GraphSAGE 直接能用，GCN 要重新调整邻接矩阵。

**🔵 数学层**（约 5 分钟）：

GraphSAGE 公式（一层）：

```
m_j = x_j                                    （消息 = 源特征）
agg_i = (1/|N(i)|) · Σ_{j ∈ N(i)} m_j        （邻居平均）
x_i' = ReLU( W · [x_i || agg_i] )            （拼接 + 线性 + 激活）
```

`||` 表示拼接。最终 `[x_i || agg_i]` shape=(2 * features,)。

**🟣 代码层**（约 1 小时）：

创建文件 `week13/day04_sage_implementation.py`——从零用 PyG MessagePassing 实现 GraphSAGE：

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MySAGE(MessagePassing):
    """从零实现 GraphSAGE（不用 PyG 的 SAGEConv）"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')   # 邻居均值聚合
        # Linear 输入是 2*in_channels：拼接 x_i 和 agg_i
        self.lin = nn.Linear(in_channels * 2, out_channels)
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        """消息就是 source 特征"""
        return x_j
    
    def update(self, aggr_out, x):
        """update 同时使用 aggr 结果和原节点特征 x"""
        # 把 x_i（原节点特征）和 aggr_out（邻居聚合）拼接
        cat = torch.cat([x, aggr_out], dim=-1)
        return torch.relu(self.lin(cat))


# ========== 测试 ==========
torch.manual_seed(0)
x = torch.randn(4, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 1, 0, 2, 1, 3, 2],
    [1, 0, 2, 1, 3, 2, 0, 1, 1, 2, 2, 3]
], dtype=torch.long)

layer = MySAGE(in_channels=3, out_channels=8)
out = layer(x, edge_index)

print(f"输入 shape: {x.shape}")
print(f"输出 shape: {out.shape}")    # (4, 8)
print(f"\n注意：输出维度由 Linear 决定，输入是 2*3=6，输出 8")
print(f"这就是 GraphSAGE 的核心——concat(x_i, agg_i) 然后线性变换")
```

**验收标准**：
- 代码跑通
- 输出 shape = (4, 8)
- 你能解释 update 函数里为什么 cat 的是 `x` 和 `aggr_out`

---

#### ⚠️ Day 4 新手最容易踩的坑

**坑 1：以为 GraphSAGE 一定比 GCN 好**
- 不一定。在固定节点的小图上，GCN 可能更稳定（因为有归一化）
- GraphSAGE 的优势是**归纳学习**（适合不同节点数的图）和**身份保留**
- 选哪个看具体任务

**坑 2：GraphSAGE 的 update 里 cat 错了**
- 错误：`cat([x, aggr_out], dim=0)`（拼接节点维）
- 正确：`cat([x, aggr_out], dim=-1)`（拼接特征维）

**坑 3：Linear 维度算错**
- 错误：`nn.Linear(in_channels, out_channels)`
- 正确：`nn.Linear(in_channels * 2, out_channels)`（cat 后维度翻倍）

**坑 4：把 update 的 `x` 当成可选**
- 必须显式传入：`def update(self, aggr_out, x)`
- 否则 GraphSAGE 退化成 GCN（没有节点自身保留）

---

#### 🧠 Day 4 概念问答 quiz

**Q1**：GraphSAGE 相对于 GCN 的核心改进是什么？

<details><summary>答案</summary>把"节点自身特征"和"邻居聚合"**分开处理**（拼接），而不是像 GCN 一样混在一起（通过 A + I 自环）。这样节点身份信息更明确。</details>

**Q2**：为什么 GraphSAGE 适合做"归纳学习"（处理不同节点数的新图）？

<details><summary>答案</summary>GraphSAGE 的更新是**每个节点独立计算**——只需要它的邻居信息，不需要完整邻接矩阵。所以新图扔进来直接能用。GCN 依赖整个邻接矩阵（D^(-1/2)·Â·D^(-1/2)），节点编号变了就不能直接用。</details>

**Q3**：GraphSAGE 的 update 函数为什么要 cat `x` 和 `aggr_out` 然后过 Linear？

<details><summary>答案</summary>cat 让模型同时看到"自己的状态"和"邻居的归纳"，Linear 把这两类信息融合到一个新的特征空间——这是 GraphSAGE 的核心设计。如果不 cat 直接 `aggr_out` 输出，节点自己的特征就丢了（变成纯邻居聚合）。</details>

**Q4**：GraphSAGE 公式 `x_i' = ReLU( W · [x_i || agg_i] )` 中 `[x_i || agg_i]` 的 shape 是什么？

<details><summary>答案</summary>`(2 * in_channels,)` 在每个节点上。拼接两个 in_channels 维向量，结果是 2*in_channels 维。这就是 Linear 的输入维度要 *2 的原因。</details>

**Q5**：论文为什么用 GraphSAGE 而不是 GCN？

<details><summary>答案</summary>论文 §2.1.2 给的理由：(1) GraphSAGE 适合处理变节点数的图（PhyFENet 要求处理不同 mesh）；(2) 节点身份信息保留得更好；(3) 不需要全图归一化，更灵活。</details>

---

#### 📦 Day 4 知识卡片

| 项目 | 内容 |
|---|---|
| **核心思想** | 节点自己 + 邻居聚合 = **拼接**（不是混合） |
| **公式** | `x_i' = ReLU( W · [x_i || agg_i] )` |
| **vs GCN** | 不混入自环，节点身份独立 |
| **优势** | 归纳学习；身份保留；不需完整 A |
| **本日产出** | `week13/day04_sage_implementation.py` |
| **掌握要求** | 🟡 能讲（理解 vs GCN 的差异） |

---

### Day 5-6（周末）| GraphSAGE 多图训练对比 GCN

**🎯 本日目标**：用 PyG 的 `SAGEConv` 训练 GraphSAGE，对比 Week 12 的 GCN 性能。

**🟣 代码层**（约 2.5 小时）：

创建文件 `week13/weekend_graphsage.py`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# 复用 Week 12 的弹性板数据构造（简略版）
def build_plate_data(nx=5, ny=3, F_applied=100.0):
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([float(i), float(j)])
    nodes = np.array(nodes, dtype=np.float32)
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i; n1 = j * nx + (i+1)
            n2 = (j+1) * nx + (i+1); n3 = (j+1) * nx + i
            elements.append([n0, n1, n2, n3])
    edge_set = set()
    for elem in elements:
        for i in range(4):
            for j in range(i+1, 4):
                a, b = min(elem[i], elem[j]), max(elem[i], elem[j])
                edge_set.add((a, b))
    src, dst = [], []
    for a, b in edge_set:
        src.append(a); dst.append(b)
        src.append(b); dst.append(a)
    edge_index = np.array([src, dst], dtype=np.int64)
    
    N = len(nodes)
    is_fixed = np.zeros((N, 1), dtype=np.float32)
    F_feat = np.zeros((N, 1), dtype=np.float32)
    is_fixed[np.abs(nodes[:, 0]) < 0.01] = 1.0
    F_feat[np.abs(nodes[:, 0] - nodes[:, 0].max()) < 0.01] = F_applied
    x_features = np.concatenate([nodes, is_fixed, F_feat], axis=1)
    
    y_center = (ny - 1) * 0.5
    ux = 0.01 * (F_applied / 100) * nodes[:, 0:1]
    uy = -0.003 * (F_applied / 100) * nodes[:, 0:1] * (nodes[:, 1:2] - y_center)
    ux[np.abs(nodes[:, 0]) < 0.01] = 0
    uy[np.abs(nodes[:, 0]) < 0.01] = 0
    y_label = np.concatenate([ux, uy], axis=1).astype(np.float32)
    
    return Data(
        x=torch.from_numpy(x_features).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        y=torch.from_numpy(y_label).float()
    )


# ========== GCN 模型（Week 12）==========
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x)


# ========== GraphSAGE 模型（本周）==========
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
            x = F.relu(conv(x, edge_index))
        return self.head(x)


# ========== 数据 ==========
F_train = [50, 75, 100, 125, 150, 175, 200]
F_val = [60, 110, 160]    # val 部分内插部分外推

train_data = [build_plate_data(F_applied=F) for F in F_train]
val_data = [build_plate_data(F_applied=F) for F in F_val]

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)


# ========== 通用训练函数 ==========
def train_model(model, n_epochs=300, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    
    for epoch in range(n_epochs):
        model.train()
        train_total = 0
        for batch in train_loader:
            y_pred = model(batch)
            loss = criterion(y_pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item() * batch.num_graphs
        train_losses.append(train_total / len(train_data))
        
        model.eval()
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                y_pred = model(batch)
                val_total += criterion(y_pred, batch.y).item() * batch.num_graphs
        val_losses.append(val_total / len(val_data))
    
    return train_losses, val_losses


# ========== 训练两个模型 ==========
torch.manual_seed(0)
gcn_model = GCN(in_channels=4, hidden_channels=32, out_channels=2)
print("训练 GCN...")
gcn_train, gcn_val = train_model(gcn_model)
print(f"  GCN 最终 train: {gcn_train[-1]:.6f}, val: {gcn_val[-1]:.6f}")

torch.manual_seed(0)
sage_model = GraphSAGE(in_channels=4, hidden_channels=32, out_channels=2)
print("\n训练 GraphSAGE...")
sage_train, sage_val = train_model(sage_model)
print(f"  GraphSAGE 最终 train: {sage_train[-1]:.6f}, val: {sage_val[-1]:.6f}")


# ========== 可视化对比 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(gcn_train, label='GCN', alpha=0.7)
axes[0].plot(sage_train, label='GraphSAGE', alpha=0.7)
axes[0].set_yscale('log')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Train MSE')
axes[0].set_title('Train Loss 对比')
axes[0].legend()

axes[1].plot(gcn_val, label='GCN', alpha=0.7)
axes[1].plot(sage_val, label='GraphSAGE', alpha=0.7)
axes[1].set_yscale('log')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Val MSE')
axes[1].set_title('Val Loss 对比')
axes[1].legend()

plt.tight_layout()
plt.savefig('week13_gcn_vs_sage.png', dpi=100)
print(f"\n图已保存")
```

**验收标准**：
- 两个模型都能跑通
- GraphSAGE 的 val Loss **通常**比 GCN 略低（因为节点身份保留）
- **如果 GCN 在某些 epoch 反而更好——这也是正常的**（小合成数据上结果可能反复）
- 你能从 Loss 曲线看出两个模型的训练行为差异

**🔬 应用层**（约 5 分钟）：

**结果分析的纪律**：

不要看到一次实验结果就下结论"X 模型一定比 Y 好"。**真正可靠的结论需要**：
- 多个随机种子（一个种子的偶然性大）
- 多组数据（合成数据可能让某种模型偏好）
- 多个超参组合（lr、hid_dim 不同结果可能反转）

本周这次实验的目的是**让你看到两个模型的训练曲线差异**——而不是"证明谁好"。后者要看你最终的真实数据集。

---

#### ⚠️ Day 5-6 新手最容易踩的坑

**坑 1：以为 GraphSAGE 一定赢**
- 在小合成数据上不一定。GCN 的归一化在 fixed 节点编号场景下也很稳定
- 真正的差异在**变节点数**和**大图**上才显现

**坑 2：训练曲线波动大就以为是 bug**
- 多图 batch 训练（每 epoch 几个 batch）loss 波动正常
- 看**整体趋势**——如果 100 epoch 内 loss 在下降，没问题

**坑 3：用同一个种子比较两次跑**
- 不用同一个种子可能让 GCN 第一次跑赢、GraphSAGE 第二次跑赢
- 严谨的对比要**多种子取平均**——这周不要求，但要有这个意识

---

### ✅ Week 13 完成评估

#### 任务级 checklist

- [ ] `week13/day02_message_passing_basic.py` 跑通，手算验证一致
- [ ] `week13/day03_complex_message.py` 三个 message 函数变体都能跑
- [ ] `week13/day04_sage_implementation.py` 自定义 GraphSAGE 跑通
- [ ] `week13/weekend_graphsage.py` GCN vs GraphSAGE 对比

#### 概念级 quiz（10 题，至少 8 题对）

**Q1**：消息传递的三个阶段是什么？

<details><summary>答案</summary>Message（每条边生成消息）→ Aggregate（每个节点聚合入消息）→ Update（节点用聚合结果更新自己）。</details>

**Q2**：GCN 是消息传递的特例吗？请说出它的 message、aggregate、update 各是什么。

<details><summary>答案</summary>是。Message：`c_{ij}·x_j·W`；AGG：sum；Update：σ。</details>

**Q3**：PyG 的 `MessagePassing` 子类里 `x_i` 和 `x_j` 分别代表什么？

<details><summary>答案</summary>`x_j`：source 节点特征（边起点）；`x_i`：target 节点特征（边终点）。信息从 j 流向 i。</details>

**Q4**：以下两段代码哪个对？
```python
# A
def message(self, x_source, x_target): ...
# B  
def message(self, x_j, x_i): ...
```

<details><summary>答案</summary>**B**。PyG 通过 `_j` 和 `_i` 后缀识别源/目标，参数名必须正好是这个。</details>

**Q5**：什么场景下 message 必须用 `edge_attr`？

<details><summary>答案</summary>当边的特征会影响消息内容时（比如距离、相对位置）。在 PINN/FEM 场景下边特征几乎都需要——节点间的物理耦合方式取决于它们之间的几何关系。</details>

**Q6**：GraphSAGE 相对于 GCN 的核心改进是什么？

<details><summary>答案</summary>把节点自身特征和邻居聚合**分开处理**（拼接），而不是混合（GCN 通过 A + I）。</details>

**Q7**：为什么 GraphSAGE 适合做归纳学习（处理变节点数的图）？

<details><summary>答案</summary>每个节点的更新是局部操作（只用自己 + 邻居），不依赖整个邻接矩阵的归一化。新图扔进来直接能算。</details>

**Q8**：GraphSAGE 的 update 函数里为什么 cat `x` 和 `aggr_out`？

<details><summary>答案</summary>cat 让模型同时使用"自己的状态"和"邻居的归纳"，Linear 把两类信息融合。如果不 cat，节点自己的特征丢失。</details>

**Q9**：在 PyG 的 MessagePassing 子类里，update 函数怎么拿到 `x`？

<details><summary>答案</summary>在 `propagate(edge_index, x=x, ...)` 里把 x 传进去，update 函数签名 `def update(self, aggr_out, x)` 就能拿到。PyG 自动把 propagate 的 x 透传给 update。</details>

**Q10**：论文为什么用 GraphSAGE 而不是 GCN？

<details><summary>答案</summary>论文 §2.1.2：(1) 处理变节点数；(2) 节点身份保留更好；(3) 不需要全图归一化。</details>

#### 🚦 自我评估

- 任务全部通过 + Quiz 8 题对 → **绿灯进入 Week 14**
- 不能解释 `x_j`/`x_i` → **黄灯**——重看 Day 2
- GCN vs GraphSAGE 对比说不清 → **黄灯**——重看 Day 4

#### 知识地图自查

- [ ] (9) 消息传递三阶段 → 🟡
- [ ] (10) GraphSAGE → 🟣

---

## 进入 Week 14 之前

下周 Week 14 是 Part C 的**核心机制周**：实现论文 §2.3.3 的"带边更新的消息传递"——这是 PhyFENet 区别于普通 GraphSAGE 的最大差异。

下周需要的前置：
- Week 13 的自定义 MessagePassing 能熟练写
- 理解 message 函数三种参数组合
- 不需要新数学

Week 13 完成。

---

---

## Week 14: 带边更新的消息传递（论文核心机制）

### 🎯 Week 14 总览

**本周覆盖的知识点**：(11) 带边更新的消息传递（论文 §2.3.3 核心）、过平滑 + 残差连接

**本周不覆盖**：完整 Encoder-Processor-Decoder（Week 15）、单元聚合（Week 15）

**学完之后你应该能**：
- 🟣 **能用**：从零实现论文的"带边更新 MP 层"；用残差连接训练深层 GNN
- 🟡 **能讲**：边更新相对标准 GraphSAGE 的优势；过平滑问题

**本周的特点**：**论文最核心的机制**。如果你只能记住 Part C 的一件事，那就是这周的内容——论文式 2.14-2.17 的四步消息传递。

---

### ✅ 进入 Week 14 之前的前置 checklist

- [ ] Week 13 的自定义 `MessagePassing` 能独立写
- [ ] 我能解释 GraphSAGE 的 update 函数为什么 cat `x` 和 `aggr_out`
- [ ] 我能熟练用 `propagate(edge_index, x=x, edge_attr=edge_attr)`
- [ ] 我接受"论文有自己的特殊机制（边更新），需要在 PyG 标准 API 之外做一些工作"

---

### Day 1-2 | 实现带边更新的消息传递

**🎯 本日目标**：完整实现论文 §2.3.3 描述的"四步消息传递"——其中第 1-2 步更新边特征，第 3-4 步更新节点特征。

**🟢 直觉层**（约 10 分钟）：

到此为止你学的所有 GNN（GCN、GraphSAGE）只更新**节点特征**——边特征被设置一次后不变。

**论文的关键改动**：让边特征**也在每层被更新**。

为什么这个改动重要？在 FEM 中：

- 边代表两个节点之间的"几何/物理关系"——距离、应力传递通道
- 不同载荷下，**边的"重要程度"会变化**（比如某条边在低载荷下不重要，高载荷下变成关键路径）
- 让边特征也能学习，捕捉这种动态关系

**论文原文**：
> "本文针对模拟有限元和固体力学问题增加了边特征更新模块。"

**🟡 概念层**（约 30 分钟）：

**论文 §2.3.3 的四步消息传递**（式 2.14-2.17）：

```
步骤 1：聚合输入边特征 + 边连接的两个节点特征
   inputs_e = {e_i, n_s, n_k}    （e 边特征，n_s 源节点，n_k 目标节点）

步骤 2：用 MLP 更新边属性
   e' = MLP_edge(inputs_e)       （边特征 → 新边特征）

步骤 3：聚合节点特征 + 它的所有邻边的更新后特征
   inputs_n = {n_k, e'_i, e'_j, ...}    （目标节点 + 所有连到它的边）

步骤 4：用 MLP 更新节点属性
   n' = MLP_node(inputs_n)       （节点特征 + 邻边特征 → 新节点特征）
```

**关键观察**：步骤 1-2 在**边的粒度**做，步骤 3-4 在**节点的粒度**做。

**实现策略**：
- 步骤 1-2：手动用 `edge_index` 和节点特征拼接，过 MLP（不用 `propagate`）
- 步骤 3-4：用更新后的边特征作为消息，调 `propagate`

为什么不能完全用 `propagate`？因为 `propagate` 设计上只输出节点级结果（不返回更新后的边特征）。我们需要边特征作为"独立返回值"——所以手动算边更新这一步。

**🔵 数学层**（约 10 分钟）：

具体到一个例子，**带边更新的一层做的事**：

```
输入：
  x: (N, node_dim)        节点特征
  edge_index: (2, E)
  edge_attr: (E, edge_dim) 边特征

第 1 步：边更新
  对每条边 (j → i)：
    edge_input = concat(edge_attr[k], x_src[k], x_dst[k])    shape=(node_dim*2 + edge_dim,)
    new_edge_attr[k] = MLP_edge(edge_input)                   shape=(out_dim,)
  
  得到 new_edge_attr: (E, out_dim)

第 2 步：节点更新
  对每个节点 i：
    收集所有连到 i 的边的更新后特征 new_edge_attr[k]
    aggr = mean(那些边特征)                                    shape=(out_dim,)
    node_input = concat(x[i], aggr)                            shape=(node_dim + out_dim,)
    new_x[i] = MLP_node(node_input)                            shape=(out_dim,)
  
  得到 new_x: (N, out_dim)

返回：(new_x, new_edge_attr)
```

**🟣 代码层**（约 4 小时，分两天）：

创建文件 `week14/day12_edge_update.py`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class EdgeUpdateMP(MessagePassing):
    """论文 §2.3.3 的带边更新消息传递层
    
    每层做：
    1. 更新边特征：e' = MLP_edge(e, x_src, x_dst)
    2. 节点聚合：aggr_i = mean(e'_ij for j in N(i))
    3. 更新节点特征：x' = MLP_node(x, aggr)
    """
    def __init__(self, node_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')
        
        # 边更新 MLP：输入 = edge_attr + x_src + x_dst
        # 输入维度 = edge_channels + 2 * node_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_channels + 2 * node_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # 节点更新 MLP：输入 = x + aggr
        # 输入维度 = node_channels + out_channels
        self.node_mlp = nn.Sequential(
            nn.Linear(node_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        x: (N, node_channels)
        edge_index: (2, E)
        edge_attr: (E, edge_channels)
        
        返回:
          new_x: (N, out_channels)
          new_edge_attr: (E, out_channels)
        """
        # ===== Step 1: 更新边特征 =====
        # 拿到每条边的源节点和目标节点
        src, dst = edge_index[0], edge_index[1]
        x_src = x[src]    # shape=(E, node_channels)
        x_dst = x[dst]    # shape=(E, node_channels)
        
        # 拼接边特征 + 源 + 目标
        edge_input = torch.cat([edge_attr, x_src, x_dst], dim=-1)
        # shape=(E, edge_channels + 2*node_channels)
        
        # 用 edge_mlp 更新
        new_edge_attr = self.edge_mlp(edge_input)
        # shape=(E, out_channels)
        
        # ===== Step 2 & 3: 用更新后的边作为消息聚合到节点 =====
        # 通过 propagate，把 new_edge_attr 当做消息传给每个目标节点
        new_x = self.propagate(
            edge_index, 
            x=x,                          # 给 update 用
            edge_msg=new_edge_attr        # 自定义关键字（PyG 会自动按 edge_index 处理）
        )
        
        return new_x, new_edge_attr
    
    def message(self, edge_msg):
        """消息就是更新后的边特征"""
        return edge_msg
    
    def update(self, aggr_out, x):
        """节点用原特征 + 聚合消息更新"""
        # x 是原节点特征，aggr_out 是聚合后的边特征
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(node_input)


# ========== 测试 ==========
torch.manual_seed(0)
N = 5

edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
    [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
], dtype=torch.long)

x = torch.randn(N, 4)                            # 节点 4 维特征
edge_attr = torch.randn(edge_index.shape[1], 3)  # 边 3 维特征

layer = EdgeUpdateMP(node_channels=4, edge_channels=3, out_channels=16)
new_x, new_edge_attr = layer(x, edge_index, edge_attr)

print(f"输入节点 shape: {x.shape}")              # (5, 4)
print(f"输入边 shape: {edge_attr.shape}")         # (10, 3)
print(f"输出节点 shape: {new_x.shape}")           # (5, 16)
print(f"输出边 shape: {new_edge_attr.shape}")     # (10, 16)

# ========== 验证：每条边更新逻辑正确 ==========
# 手动算第 0 条边的更新
src, dst = edge_index[0], edge_index[1]
x_src_0 = x[src[0]]
x_dst_0 = x[dst[0]]
e_0 = edge_attr[0]
manual_edge_input = torch.cat([e_0, x_src_0, x_dst_0])
manual_new_e_0 = layer.edge_mlp(manual_edge_input.unsqueeze(0)).squeeze()

print(f"\n手算第 0 条边更新: {manual_new_e_0[:3]}...")
print(f"PyG 算的:        {new_edge_attr[0][:3]}...")
print(f"是否一致: {torch.allclose(manual_new_e_0, new_edge_attr[0], atol=1e-6)}")
```

**验收标准**：
- 代码跑通无错
- 两个输出 shape 都正确：`new_x: (5, 16)`, `new_edge_attr: (10, 16)`
- 手动算的第 0 条边更新和 PyG 一致
- 你能口头解释每一步在做什么（这是 Day 2 的核心任务）

---

#### ⚠️ Day 1-2 新手最容易踩的坑

**坑 1：忘了边特征也要返回**
- 错误：只返回 `new_x`，下次循环用旧的 `edge_attr`
- 正确：返回 `(new_x, new_edge_attr)`，下层用新边特征
- 这是带边更新和无边更新的核心差异

**坑 2：`propagate` 参数名搞错**
- 自定义传给 message 的关键字（如 `edge_msg`）必须在 message 函数里**用完全相同的名字**
- 错：`propagate(..., edge_msg=...)`，但 `def message(self, msg)` —— message 收不到
- 正：参数名一致

**坑 3：edge_mlp 的输入维度算错**
- `cat([edge_attr, x_src, x_dst], dim=-1)` 后维度是 `edge_channels + 2*node_channels`
- 不是 `node_channels + edge_channels`（少了一份）
- 也不是 `2*(edge_channels + node_channels)`（多了）

**坑 4：node_mlp 的输入维度算错**
- 输入是 `cat([x, aggr_out], dim=-1)`，维度是 `node_channels + out_channels`
- 不是 `2 * out_channels`（aggr_out 是 out_channels，但 x 是 node_channels）
- 注意：x 还没被这一层变换过，它的维度是输入 `node_channels`

**坑 5：在 message 里写 `def message(self, x_j)` 而不是 `edge_msg`**
- 如果你在 propagate 里传了 `edge_msg=new_edge_attr`，message 必须用 `edge_msg`
- 用 `x_j` 会报错或行为奇怪

---

#### 🧠 Day 1-2 概念问答 quiz

**Q1**：论文 §2.3.3 的"四步消息传递"分别是什么？

<details><summary>答案</summary>(1) 聚合边特征和两端节点特征；(2) MLP 更新边特征；(3) 聚合节点和它周围的更新后边；(4) MLP 更新节点特征。前两步是边更新，后两步是节点更新。</details>

**Q2**：边更新和节点更新分别在什么粒度做？

<details><summary>答案</summary>边更新在**边粒度**（每条边独立做 MLP）；节点更新在**节点粒度**（每个节点聚合周围边消息后做 MLP）。两个粒度的转换在第 3 步（边到节点的聚合）完成。</details>

**Q3**：为什么不能完全用 PyG 的 `propagate` 实现边更新？

<details><summary>答案</summary>`propagate` 设计上只输出节点级结果（聚合后的节点特征）。但论文要求边特征也作为独立返回值供下一层用。所以边更新这一步要在 propagate 之前手动做。</details>

**Q4**：`EdgeUpdateMP` 的 forward 返回**两个**值，下一层怎么连接？

<details><summary>答案</summary>下一层接收 `(new_x, new_edge_attr)`。多层堆叠时：
```python
for layer in layers:
    x, edge_attr = layer(x, edge_index, edge_attr)
```
每一层的输出作为下一层的输入。</details>

**Q5**：边更新机制相对标准 GraphSAGE 的优势是什么？

<details><summary>答案</summary>(1) 边特征也能"学习"，捕捉动态的物理关系；(2) 在 FEM 任务里，边代表节点间的力学耦合，让它可学是合理的；(3) 增加模型表达能力。</details>

---

#### 📦 Day 1-2 知识卡片

| 项目 | 内容 |
|---|---|
| **核心机制** | 边特征 + 节点特征**双更新** |
| **核心代码** | `EdgeUpdateMP` 类，forward 返回 `(new_x, new_edge_attr)` |
| **MLP 维度算法** | edge_mlp: `edge_dim + 2*node_dim → out_dim`；node_mlp: `node_dim + out_dim → out_dim` |
| **关键模式** | 手动算边更新（edge_mlp）→ 用 propagate 做节点更新 |
| **常见错误** | 忘了返回边特征；MLP 维度算错；propagate 关键字不对应 |
| **本日产出** | `week14/day12_edge_update.py` |
| **掌握要求** | 🔵 能写（这是 Part C 最核心的代码） |

---

### Day 3 | 多层堆叠：完整 MPNet

**🎯 本日目标**：把 `EdgeUpdateMP` 堆叠成多层模型；在弹性板数据上训练。

**🟢 直觉层**（约 5 分钟）：

一层只让信息传递 1 跳。多层让信息传 K 跳。

但**和普通 GNN 一样**，太多层会有过平滑问题（Day 5-6 处理）。先做一个 3 层的稳定版本。

**🟣 代码层**（约 2 小时）：

创建文件 `week14/day03_full_mp_model.py`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# 复用 EdgeUpdateMP（实际项目里 import）
from torch_geometric.nn import MessagePassing

class EdgeUpdateMP(MessagePassing):
    def __init__(self, node_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_channels + 2 * node_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        edge_input = torch.cat([edge_attr, x[src], x[dst]], dim=-1)
        new_edge_attr = self.edge_mlp(edge_input)
        new_x = self.propagate(edge_index, x=x, edge_msg=new_edge_attr)
        return new_x, new_edge_attr
    
    def message(self, edge_msg):
        return edge_msg
    
    def update(self, aggr_out, x):
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))


# ========== MPNet 模型 ==========

class MPNet(nn.Module):
    """多层 MessagePassing 网络（带边更新）"""
    def __init__(self, in_node_dim, in_edge_dim, hid_dim, out_dim, n_layers=3):
        super().__init__()
        
        # 第一层：从输入维度到隐藏维度
        self.layers = nn.ModuleList()
        self.layers.append(EdgeUpdateMP(in_node_dim, in_edge_dim, hid_dim))
        
        # 后续层：隐藏维度到隐藏维度
        for _ in range(n_layers - 1):
            self.layers.append(EdgeUpdateMP(hid_dim, hid_dim, hid_dim))
        
        # 输出头
        self.head = nn.Linear(hid_dim, out_dim)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        
        return self.head(x)


# ========== 数据：复用 Week 11 ==========
def build_plate_data(nx=5, ny=3, F_applied=100.0):
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([float(i), float(j)])
    nodes = np.array(nodes, dtype=np.float32)
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i; n1 = j * nx + (i+1)
            n2 = (j+1) * nx + (i+1); n3 = (j+1) * nx + i
            elements.append([n0, n1, n2, n3])
    edge_set = set()
    for elem in elements:
        for i in range(4):
            for j in range(i+1, 4):
                a, b = min(elem[i], elem[j]), max(elem[i], elem[j])
                edge_set.add((a, b))
    src, dst = [], []
    for a, b in edge_set:
        src.append(a); dst.append(b)
        src.append(b); dst.append(a)
    edge_index = np.array([src, dst], dtype=np.int64)
    
    N = len(nodes)
    is_fixed = np.zeros((N, 1), dtype=np.float32)
    F_feat = np.zeros((N, 1), dtype=np.float32)
    is_fixed[np.abs(nodes[:, 0]) < 0.01] = 1.0
    F_feat[np.abs(nodes[:, 0] - nodes[:, 0].max()) < 0.01] = F_applied
    x_features = np.concatenate([nodes, is_fixed, F_feat], axis=1)
    
    # 边特征（dx, dy, dist）
    src_idx = edge_index[0]; dst_idx = edge_index[1]
    diff = nodes[dst_idx] - nodes[src_idx]
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    edge_attr = np.concatenate([diff, dist], axis=1).astype(np.float32)
    
    y_center = (ny - 1) * 0.5
    ux = 0.01 * (F_applied / 100) * nodes[:, 0:1]
    uy = -0.003 * (F_applied / 100) * nodes[:, 0:1] * (nodes[:, 1:2] - y_center)
    ux[np.abs(nodes[:, 0]) < 0.01] = 0
    uy[np.abs(nodes[:, 0]) < 0.01] = 0
    y_label = np.concatenate([ux, uy], axis=1).astype(np.float32)
    
    return Data(
        x=torch.from_numpy(x_features).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr).float(),
        y=torch.from_numpy(y_label).float()
    )


# ========== 训练 ==========
torch.manual_seed(0)

train_data = [build_plate_data(F_applied=F) for F in [50, 75, 100, 125, 150, 175, 200]]
val_data = [build_plate_data(F_applied=F) for F in [60, 110, 160]]

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)

model = MPNet(in_node_dim=4, in_edge_dim=3, hid_dim=32, out_dim=2, n_layers=3)
print(f"参数总数: {sum(p.numel() for p in model.parameters())}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []

print("\n训练 MPNet（带边更新）...")
for epoch in range(300):
    model.train()
    total = 0
    for batch in train_loader:
        y_pred = model(batch)
        loss = criterion(y_pred, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
    train_losses.append(total / len(train_data))
    
    model.eval()
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            y_pred = model(batch)
            val_total += criterion(y_pred, batch.y).item() * batch.num_graphs
    val_losses.append(val_total / len(val_data))
    
    if epoch % 30 == 0:
        print(f"  epoch {epoch}: train={train_losses[-1]:.6f}, val={val_losses[-1]:.6f}")

# ========== 可视化 ==========
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('MPNet（带边更新）训练曲线')
plt.legend()
plt.savefig('week14_mpnet_train.png', dpi=100)

print(f"\n最终 train: {train_losses[-1]:.6f}")
print(f"最终 val: {val_losses[-1]:.6f}")
```

**验收标准**：
- MPNet 能跑通训练
- val Loss 下降到合理水平（通常 < 0.001）
- 你能解释 MPNet 比 GraphSAGE 多了什么（边更新）

---

#### ⚠️ Day 3 新手最容易踩的坑

**坑 1：MPNet 的多层堆叠忘了维度衔接**
- 第一层从 `in_node_dim` 到 `hid_dim`；后续层都是 `hid_dim → hid_dim`
- 写错维度会报维度不匹配

**坑 2：忘了 edge_attr 的维度也要"升"到 hid_dim**
- 第一层处理后 edge_attr 已经是 hid_dim 了
- 后续层的 edge_channels 应该是 hid_dim，不是原始的 in_edge_dim

**坑 3：训练期间没归一化数据**
- 节点特征量纲差异大（坐标 0-4 vs F 50-200）
- 即使不归一化模型也能学，但 Loss 收敛慢
- 真实工程必须归一化

---

#### 🧠 Day 3 概念问答 quiz

**Q1**：MPNet 比 Week 13 的 GraphSAGE 多了什么？

<details><summary>答案</summary>**边特征更新**——每层不仅更新节点特征，还更新边特征，下层使用的是更新后的边特征。GraphSAGE 边特征不变（甚至不用边特征）。</details>

**Q2**：MPNet 多层堆叠时，第二层的 `edge_channels` 应该是多少？

<details><summary>答案</summary>`hid_dim`（不是原始 `in_edge_dim`）。因为第一层把 edge_attr 升维到 hid_dim 了。</details>

**Q3**：MPNet 训练 Loss 下降，说明模型有什么能力？

<details><summary>答案</summary>(1) 能从节点 + 边特征学到 ux, uy 的映射；(2) 边更新机制让模型表达能力增强；(3) 多层堆叠让信息传 K 跳。但单凭 Loss 下降**不能说明**模型 generalize 好——需要 val 集和外推测试。</details>

---

#### 📦 Day 3 知识卡片

| 项目 | 内容 |
|---|---|
| **核心模型** | MPNet（多层 EdgeUpdateMP + 输出头） |
| **维度衔接** | 第一层 in_node/in_edge → hid；后续层 hid → hid |
| **训练 vs Week 13** | 边特征更新让模型表达更强 |
| **本日产出** | `week14/day03_full_mp_model.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 4 | 三种模型对比：GCN / GraphSAGE / MPNet

**🎯 本日目标**：把 Week 12-14 学的三种 GNN 在同一份数据上对比；学会**分析实验结果**而不是预设结论。

**🟢 直觉层**（约 5 分钟）：

到此你掌握了三种 GNN：
- **GCN**（Week 12）：邻居等权聚合
- **GraphSAGE**（Week 13）：节点身份保留
- **MPNet**（Week 14）：边更新

**今天的实验目的**：在同样的数据和训练设置下对比它们。但**不是要证明谁绝对更好**——而是培养**实验分析能力**。

实际工程中"我的模型比 baseline 好"是个复杂问题——可能因为模型本身好、可能因为 lr 调得好、可能因为数据偏好某种模型。**作为工程师，要能客观分析这些因素。**

**🟣 代码层 + 分析任务**（约 2.5 小时）：

创建文件 `week14/day04_three_models.py`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, SAGEConv, MessagePassing
from torch_geometric.loader import DataLoader

# 复用前面的 EdgeUpdateMP 和 build_plate_data（这里省略，从前面 import）

# ========== 三种模型 ==========
class GCNModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.head = nn.Linear(hid_dim, out_dim)
    
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.head(x)


class SAGEModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, hid_dim)
        self.head = nn.Linear(hid_dim, out_dim)
    
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.head(x)


# MPNet 用 Day 3 的（这里 import）
# from week14.day03_full_mp_model import MPNet

# ========== 训练函数 ==========
def train_one_model(model_name, model, train_loader, val_loader, n_epochs=300):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_losses, val_losses = [], []
    
    for epoch in range(n_epochs):
        model.train()
        train_total = 0
        for batch in train_loader:
            y_pred = model(batch)
            loss = criterion(y_pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item() * batch.num_graphs
        train_losses.append(train_total / len(train_loader.dataset))
        
        model.eval()
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                y_pred = model(batch)
                val_total += criterion(y_pred, batch.y).item() * batch.num_graphs
        val_losses.append(val_total / len(val_loader.dataset))
    
    print(f"{model_name}: 最终 train={train_losses[-1]:.6f}, val={val_losses[-1]:.6f}")
    return train_losses, val_losses


# ========== 跑三个模型 ==========
torch.manual_seed(0)
train_data = [build_plate_data(F_applied=F) for F in [50, 75, 100, 125, 150, 175, 200]]
val_data = [build_plate_data(F_applied=F) for F in [60, 110, 160]]
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)

torch.manual_seed(0)
gcn = GCNModel(in_dim=4, hid_dim=32, out_dim=2)
gcn_train, gcn_val = train_one_model("GCN", gcn, train_loader, val_loader)

torch.manual_seed(0)
sage = SAGEModel(in_dim=4, hid_dim=32, out_dim=2)
sage_train, sage_val = train_one_model("GraphSAGE", sage, train_loader, val_loader)

torch.manual_seed(0)
mpnet = MPNet(in_node_dim=4, in_edge_dim=3, hid_dim=32, out_dim=2, n_layers=3)
mp_train, mp_val = train_one_model("MPNet", mpnet, train_loader, val_loader)

# ========== 可视化 ==========
plt.figure(figsize=(10, 5))
plt.plot(gcn_val, label='GCN', alpha=0.7)
plt.plot(sage_val, label='GraphSAGE', alpha=0.7)
plt.plot(mp_val, label='MPNet (with edge update)', alpha=0.7)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Val MSE')
plt.title('三种模型 Val Loss 对比')
plt.legend()
plt.savefig('week14_three_models.png', dpi=100)


# ========== 实验记录表 ==========
print("\n===== 实验记录 =====")
print(f"{'模型':<15} {'Train MSE':<15} {'Val MSE':<15} {'用边特征?'}")
print("-" * 60)
print(f"{'GCN':<15} {gcn_train[-1]:<15.6f} {gcn_val[-1]:<15.6f} {'否'}")
print(f"{'GraphSAGE':<15} {sage_train[-1]:<15.6f} {sage_val[-1]:<15.6f} {'否'}")
print(f"{'MPNet':<15} {mp_train[-1]:<15.6f} {mp_val[-1]:<15.6f} {'是'}")
```

**分析任务**（写在脚本末尾的注释里）：

```
# 实验分析（用你看到的实际数据回答）
#
# 1. 三种模型的最终 Val MSE 差异是否显著？
#    答：
# 
# 2. 哪个模型的训练曲线最稳定？哪个震荡最大？
#    答：
# 
# 3. MPNet 利用了边特征 [dx, dy, dist]。这些信息其实也能从节点坐标算出。
#    在这份合成数据上，边特征是否真的提供了额外信息？
#    答：
#
# 4. 如果你看到的结果"不符合预期"（比如 MPNet 反而最差），可能的原因有哪些？
#    答（提示）：
#    - 合成数据的标签是简化模型，可能不能体现 MPNet 的优势
#    - 学习率/隐藏维度可能不是 MPNet 的最优配置
#    - 单次跑的随机性
#    - GNN 在小数据集上 overfit 风险大
```

**验收标准**：
- 三个模型都能跑通
- 你完成了 4 个分析问题
- **你能客观接受"实验结果不一定符合预设"**——这才是工程师的素养

**🔬 应用层**（约 5 分钟）：

**为什么这种"分析能力"比"特定结果"更重要**：

面试时被问"你的模型为什么比 baseline 好"——能讲清楚"在哪些情况下好、哪些情况下差、为什么"——这才是面试官想听的。**实验结果不是用来"赢"的，而是用来"理解"的**。

---

### Day 5-6（周末）| 过平滑问题与残差连接

**🎯 本日目标**：观察过平滑现象；用残差连接缓解。

**🟢 直觉层**（约 5 分钟）：

GNN 的一个经典问题：**层数太多会过平滑**——所有节点的特征趋于相同。

**直觉**：每层消息传递让节点变得"更像它的邻居"。多层叠加，节点和它 K 跳邻居都很像，整张图的所有节点变得没有区分度。

**解决方法之一：残差连接**
```
h_new = h_old + layer(h_old)
```

让深层能保留浅层信息——这是从 ResNet 借来的思想。

**🟣 代码层**（约 2.5 小时）：

创建文件 `week14/day56_oversmoothing.py`：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader

# 复用 EdgeUpdateMP 和 build_plate_data...

class MPNet(nn.Module):
    """支持残差连接的 MPNet"""
    def __init__(self, in_node_dim, in_edge_dim, hid_dim, out_dim, n_layers, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        
        self.encoder_node = nn.Linear(in_node_dim, hid_dim)
        self.encoder_edge = nn.Linear(in_edge_dim, hid_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EdgeUpdateMP(hid_dim, hid_dim, hid_dim))
        
        self.head = nn.Linear(hid_dim, out_dim)
    
    def forward(self, data):
        x = self.encoder_node(data.x)
        edge_attr = self.encoder_edge(data.edge_attr)
        
        for layer in self.layers:
            x_new, edge_attr_new = layer(x, data.edge_index, edge_attr)
            if self.use_residual:
                x = x + x_new
                edge_attr = edge_attr + edge_attr_new
            else:
                x, edge_attr = x_new, edge_attr_new
        
        return self.head(x)


# 用一个 helper：算节点特征的"相似度"——节点特征越相似，越接近过平滑
def measure_similarity(model, data):
    """算所有节点特征的平均余弦相似度（越接近 1 越过平滑）"""
    model.eval()
    with torch.no_grad():
        x = model.encoder_node(data.x)
        edge_attr = model.encoder_edge(data.edge_attr)
        for layer in model.layers:
            x_new, edge_attr_new = layer(x, data.edge_index, edge_attr)
            if model.use_residual:
                x = x + x_new
                edge_attr = edge_attr + edge_attr_new
            else:
                x, edge_attr = x_new, edge_attr_new
        # 最终节点特征
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        sim = x_norm @ x_norm.T
        # 取上三角（不包含对角）
        N = sim.size(0)
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        return sim[mask].mean().item()


# ========== 实验：不同层数的影响 ==========
torch.manual_seed(0)
train_data = [build_plate_data(F_applied=F) for F in [50, 75, 100, 125, 150]]
val_data = [build_plate_data(F_applied=F) for F in [60, 110]]
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)

results_no_res = {}
results_with_res = {}

for n_layers in [1, 2, 4, 8]:
    print(f"\n===== {n_layers} 层 =====")
    
    # 不加残差
    torch.manual_seed(0)
    model = MPNet(4, 3, 32, 2, n_layers, use_residual=False)
    final_val = train_short(model, train_loader, val_loader, n_epochs=200)
    similarity = measure_similarity(model, train_data[0])
    results_no_res[n_layers] = (final_val, similarity)
    print(f"  无残差: val={final_val:.6f}, similarity={similarity:.4f}")
    
    # 加残差
    torch.manual_seed(0)
    model_res = MPNet(4, 3, 32, 2, n_layers, use_residual=True)
    final_val_res = train_short(model_res, train_loader, val_loader, n_epochs=200)
    sim_res = measure_similarity(model_res, train_data[0])
    results_with_res[n_layers] = (final_val_res, sim_res)
    print(f"  带残差: val={final_val_res:.6f}, similarity={sim_res:.4f}")


def train_short(model, train_loader, val_loader, n_epochs):
    """简化训练函数"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            y_pred = model(batch)
            loss = criterion(y_pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    model.eval()
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            y_pred = model(batch)
            val_total += criterion(y_pred, batch.y).item() * batch.num_graphs
    return val_total / len(val_loader.dataset)


# ========== 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

n_layers_list = sorted(results_no_res.keys())

# 左图：Val Loss
val_no_res = [results_no_res[n][0] for n in n_layers_list]
val_with_res = [results_with_res[n][0] for n in n_layers_list]
axes[0].plot(n_layers_list, val_no_res, 'o-', label='无残差')
axes[0].plot(n_layers_list, val_with_res, 's-', label='带残差')
axes[0].set_xlabel('Layers')
axes[0].set_ylabel('Val MSE')
axes[0].set_yscale('log')
axes[0].set_title('层数对 Val Loss 的影响')
axes[0].legend()

# 右图：节点相似度
sim_no_res = [results_no_res[n][1] for n in n_layers_list]
sim_with_res = [results_with_res[n][1] for n in n_layers_list]
axes[1].plot(n_layers_list, sim_no_res, 'o-', label='无残差')
axes[1].plot(n_layers_list, sim_with_res, 's-', label='带残差')
axes[1].set_xlabel('Layers')
axes[1].set_ylabel('节点相似度（越接近 1 越过平滑）')
axes[1].set_title('过平滑现象')
axes[1].legend()

plt.tight_layout()
plt.savefig('week14_oversmoothing.png', dpi=100)
```

**期望观察**：
- **不加残差**：层数越多 → 节点相似度越接近 1（过平滑）→ Val Loss 上升
- **加残差**：节点相似度增长慢；Val Loss 受层数影响小

**验收标准**：
- 实验跑通
- 能在图上观察到"过平滑随层数加剧"
- 残差连接确实让深层模型表现更好

**🔬 应用层**（约 5 分钟）：

**工程结论**（PhyFENet 论文经验）：
- **GNN 通常 2-4 层够用**——超过这个数收益递减
- **深层网络必须加残差连接**——否则过平滑
- 论文 PhyFENet 用的是 5-10 层 + 残差连接

---

#### ⚠️ Day 5-6 新手最容易踩的坑

**坑 1：把过平滑当 bug**
- 不是 bug，是 GNN 的固有现象
- 残差连接 / 跳跃连接 / 不同的聚合方式都是缓解手段

**坑 2：以为加残差就能堆任意多层**
- 残差能缓解过平滑，但不能彻底解决
- 通常 GNN 不会超过 10 层（GraphSAINT、GIN 等论文经验）

**坑 3：残差连接的位置错**
- 错：`x = layer(x); x = x + something_else`
- 正：`x_new = layer(x); x = x + x_new`
- 残差是"原始 + 新算的"，不是"原始 + 别的"

---

### ✅ Week 14 完成评估

#### 任务级 checklist

- [ ] `week14/day12_edge_update.py` `EdgeUpdateMP` 跑通，手算验证一致
- [ ] `week14/day03_full_mp_model.py` MPNet 多图训练
- [ ] `week14/day04_three_models.py` 三种模型对比 + 4 个分析问题完成
- [ ] `week14/day56_oversmoothing.py` 过平滑实验完成

#### 概念级 quiz（10 题，至少 8 题对）

**Q1**：论文 §2.3.3 的"四步消息传递"分别是什么？

<details><summary>答案</summary>(1) 聚合边特征 + 两端节点；(2) MLP 更新边特征；(3) 聚合节点 + 周围更新后边特征；(4) MLP 更新节点。前两步边粒度，后两步节点粒度。</details>

**Q2**：边更新机制相对标准 GraphSAGE 的优势？

<details><summary>答案</summary>边特征也能学习。在 FEM 中边代表节点间的物理耦合，让它可学能捕捉动态关系。</details>

**Q3**：实现带边更新时为什么不能完全用 PyG 的 `propagate`？

<details><summary>答案</summary>propagate 只输出节点级结果，不返回更新后的边特征。论文要求边特征作为下层输入，所以边更新这一步要在 propagate 之前手动做。</details>

**Q4**：`EdgeUpdateMP.forward` 的返回值是？

<details><summary>答案</summary>**两个值**：`(new_x, new_edge_attr)`。两者都要返回，下层用更新后的边特征。</details>

**Q5**：edge_mlp 的输入维度怎么算？

<details><summary>答案</summary>`edge_channels + 2 * node_channels` —— cat 边特征 + 源节点 + 目标节点。</details>

**Q6**：MPNet 多层堆叠时第二层的 `edge_channels` 是多少？

<details><summary>答案</summary>`hid_dim`（不是 in_edge_dim）。第一层已经把边升到 hid_dim 了。</details>

**Q7**：什么是过平滑？

<details><summary>答案</summary>GNN 层数太多时所有节点特征趋于相同，失去区分度。原因：每层都让节点变得更像邻居。</details>

**Q8**：怎么用代码"测量"过平滑？

<details><summary>答案</summary>算所有节点特征对的余弦相似度，取平均。值越接近 1 表示节点越相似（过平滑）。</details>

**Q9**：残差连接是怎么缓解过平滑的？

<details><summary>答案</summary>让深层能保留浅层信息。`x = x + layer(x)` —— 即使后面层把 x 平滑掉，原始 x 仍然在残差路径中保留。</details>

**Q10**：实验中 MPNet 反而比 GraphSAGE 差，可能原因？

<details><summary>答案</summary>(1) 合成数据的边特征 [dx, dy, dist] 也能从节点坐标算出，没提供新信息；(2) 学习率/隐藏维度可能不是 MPNet 的最优配置；(3) 单次跑的随机性；(4) MPNet 参数多，小数据上 overfit 风险大。</details>

#### 🚦 自我评估

- 任务全部通过 + Quiz 8 题对 → **绿灯进入 Week 15**
- `EdgeUpdateMP` 不能独立写出来 → **黄灯**——这是 Part C 的核心代码，重写直到能背
- 不能从代码追溯到论文式 2.14-2.17 → **黄灯**——重读 Day 1-2 的概念层

#### 知识地图自查

- [ ] (11) 带边更新的 MP → 🔵 能写
- [ ] 过平滑 + 残差 → 🟣

---

## 进入 Week 15 之前

下周 Week 15：完整 PhyFENet 架构（Encoder-Processor-Decoder + 单元聚合）。

下周需要的前置：
- `EdgeUpdateMP` 你能从零写出来
- 理解过平滑和残差的关系
- 不需要新数学

---

# Part C Week 15-16 改造版（批 3 / 最终批）

> 接续批 2（Week 13-14）。本文件是 **Part C 第 3 批（最后一批）**：Week 15（完整 PhyFENet 架构 + 单元聚合）和 Week 16（综合实战 + Part C 总自测）。
>
> 本批是 Part C 的收官——所有前面的零件（图数据、消息传递、边更新、过平滑处理）拼成完整的 PhyFENet 架构。学完批 3，你就具备了 PhyFENet 论文 §2.3 的核心实现能力。

---

## Week 15: 编码器-处理器-解码器架构（PhyFENet 主框架）

### 🎯 Week 15 总览

**本周覆盖的知识点**：(12) Encoder-Processor-Decoder 架构、(13) 单元聚合（论文 §2.3.4）

**本周不覆盖**：完整 PhyFENet 训练（Week 16）、PINN 集成（Part D）

**学完之后你应该能**：
- 🟣 **能用**：搭建编码器-处理器-解码器架构；实现单元聚合矩阵
- 🟡 **能讲**：为什么要这种架构；单元聚合的物理意义

**本周的特点**：**集大成**——前 4 周的零件拼成完整模型。代码不算多但**架构设计的理解很重要**。

---

### ✅ 进入 Week 15 之前的前置 checklist

- [ ] Week 14 的 `EdgeUpdateMP` 能从零独立写出来（不看参考）
- [ ] 我能解释残差连接为什么缓解过平滑
- [ ] 我能用 PyG 的 DataLoader 做多图训练
- [ ] 我接受"本周引入 FEM 单元的概念，需要补充一点 FEM 知识"

---

### Day 1-2 | 三模块架构的设计

**🎯 本日目标**：理解 PhyFENet 的"编码器-处理器-解码器"架构；从零搭建。

**🟢 直觉层**（约 10 分钟）：

到 Week 14 你的模型直接是"输入 → 几层 EdgeUpdateMP → 输出"。但论文用的不是这种"扁平"结构，而是**三段式**：

```
原始输入特征（4 维：x, y, is_fixed, F）
    ↓
[编码器] 升维到 hid_dim（比如 64 维）
    ↓
hid_dim 维"潜在特征空间"
    ↓
[处理器] 多层消息传递（都在潜在空间里做）
    ↓
hid_dim 维"加工后的潜在特征"
    ↓
[解码器] 降维到目标维度（2 维：ux, uy）
    ↓
最终预测
```

**为什么要这样设计**：

1. **潜在空间统一处理**：原始输入维度可能很小（4 维），直接做消息传递信息容量不足。先升到 hid_dim（64-128 维）让模型有足够"工作空间"
2. **维度统一便于残差**：所有 processor 层都在 hid_dim 工作，残差连接 `x + layer(x)` 维度匹配
3. **职责分离**：encoder 负责"读懂输入"，processor 负责"做计算"，decoder 负责"翻译成输出"——每个组件单独优化

这是 MeshGraphNet（DeepMind, 2021）推广的标准架构，PhyFENet 论文沿用。

**🟡 概念层**（约 20 分钟）：

**架构图**：

```
[节点原始特征 (N, n_node_raw)]      [边原始特征 (E, n_edge_raw)]
         ↓                                  ↓
   [节点编码器 MLP]                  [边编码器 MLP]
         ↓                                  ↓
   [节点潜在 (N, hid)]              [边潜在 (E, hid)]
         ↓                                  ↓
   [处理器 layer 1（带边更新）]  ←——————→
         ↓ + 残差                          ↓ + 残差
   [处理器 layer 2]              ←——————→
         ↓                                  ↓
       ...
         ↓
   [节点最终特征 (N, hid)]
         ↓
   [节点解码器 MLP]
         ↓
   [输出 y_pred (N, n_out)]
```

**关键细节**：
- **编码器**：MLP（如 2 层），把原始特征升到 hid_dim
- **处理器**：N_processor 层 EdgeUpdateMP，每层带残差
- **解码器**：MLP（如 2 层），把 hid_dim 降到任务输出维度
- **残差是必须的**——避免过平滑（Week 14 学过）

**与 Week 14 MPNet 的对比**：

| 维度 | Week 14 MPNet | Week 15 PhyFENet_Mini |
|---|---|---|
| 第一层处理 | 直接把原始维度 → hid | 编码器先升维 → hid |
| 中间层 | 直接 EdgeUpdateMP | EdgeUpdateMP + 残差 |
| 最后输出 | 一个 Linear | 解码器（多层 MLP） |
| 表达能力 | 较弱 | 较强 |

**🔵 数学层**（无新数学）：

只是架构组合，不涉及新公式。

**🟣 代码层**（约 3 小时，分两天）：

创建文件 `week15/day12_encoder_processor_decoder.py`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


# 复用 EdgeUpdateMP（Week 14）
class EdgeUpdateMP(MessagePassing):
    def __init__(self, node_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_channels + 2 * node_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        edge_input = torch.cat([edge_attr, x[src], x[dst]], dim=-1)
        new_edge_attr = self.edge_mlp(edge_input)
        new_x = self.propagate(edge_index, x=x, edge_msg=new_edge_attr)
        return new_x, new_edge_attr
    
    def message(self, edge_msg):
        return edge_msg
    
    def update(self, aggr_out, x):
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))


# ========== PhyFENet_Mini ==========
class PhyFENet_Mini(nn.Module):
    """简化版 PhyFENet：编码器 → 处理器 → 解码器
    
    参数：
      node_in: 输入节点特征维度（如 4）
      edge_in: 输入边特征维度（如 3）
      hid: 隐藏维度（如 64）
      node_out: 输出节点维度（如 2）
      n_mp_layers: 处理器层数
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
        
        # === 处理器 ===
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
        # === 编码 ===
        x = self.node_encoder(data.x)              # (N, node_in) → (N, hid)
        edge_attr = self.edge_encoder(data.edge_attr)   # (E, edge_in) → (E, hid)
        
        # === 处理（带残差连接）===
        for layer in self.processor:
            x_new, edge_attr_new = layer(x, data.edge_index, edge_attr)
            x = x + x_new                          # 残差
            edge_attr = edge_attr + edge_attr_new
        
        # === 解码 ===
        return self.decoder(x)


# ========== 测试 ==========
import numpy as np
from torch_geometric.data import Data

def build_plate_data(nx=5, ny=3, F_applied=100.0):
    """简化版数据构造（同 Week 14）"""
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([float(i), float(j)])
    nodes = np.array(nodes, dtype=np.float32)
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i; n1 = j * nx + (i+1)
            n2 = (j+1) * nx + (i+1); n3 = (j+1) * nx + i
            elements.append([n0, n1, n2, n3])
    edge_set = set()
    for elem in elements:
        for i in range(4):
            for j in range(i+1, 4):
                a, b = min(elem[i], elem[j]), max(elem[i], elem[j])
                edge_set.add((a, b))
    src, dst = [], []
    for a, b in edge_set:
        src.append(a); dst.append(b)
        src.append(b); dst.append(a)
    edge_index = np.array([src, dst], dtype=np.int64)
    
    N = len(nodes)
    is_fixed = np.zeros((N, 1), dtype=np.float32)
    F_feat = np.zeros((N, 1), dtype=np.float32)
    is_fixed[np.abs(nodes[:, 0]) < 0.01] = 1.0
    F_feat[np.abs(nodes[:, 0] - nodes[:, 0].max()) < 0.01] = F_applied
    x_features = np.concatenate([nodes, is_fixed, F_feat], axis=1)
    
    diff = nodes[edge_index[1]] - nodes[edge_index[0]]
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    edge_attr = np.concatenate([diff, dist], axis=1).astype(np.float32)
    
    y_center = (ny - 1) * 0.5
    ux = 0.01 * (F_applied / 100) * nodes[:, 0:1]
    uy = -0.003 * (F_applied / 100) * nodes[:, 0:1] * (nodes[:, 1:2] - y_center)
    ux[np.abs(nodes[:, 0]) < 0.01] = 0
    uy[np.abs(nodes[:, 0]) < 0.01] = 0
    y_label = np.concatenate([ux, uy], axis=1).astype(np.float32)
    
    return Data(
        x=torch.from_numpy(x_features).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr).float(),
        y=torch.from_numpy(y_label).float()
    ), elements


torch.manual_seed(0)
data, _ = build_plate_data(F_applied=100.0)

model = PhyFENet_Mini(node_in=4, edge_in=3, hid=64, node_out=2, n_mp_layers=3)
print(model)
print(f"\n参数总数: {sum(p.numel() for p in model.parameters())}")

y_pred = model(data)
print(f"\n输出 shape: {y_pred.shape}")    # (15, 2)
```

**验收标准**：
- 模型创建无错
- 前向输出 shape = (15, 2)（5×3 板 → 15 节点；2 维输出）
- 你能解释每一段（encoder / processor / decoder）的职责

**🔬 应用层**（Day 2，约 1 小时）：训练对比 PhyFENet_Mini vs MPNet

```python
# week15/day12 后续部分：训练对比

import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)
train_data = [build_plate_data(F_applied=F)[0] for F in [50, 75, 100, 125, 150, 175, 200]]
val_data = [build_plate_data(F_applied=F)[0] for F in [60, 110, 160]]

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)


def train(model, n_epochs=300):
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    train_l, val_l = [], []
    for ep in range(n_epochs):
        model.train()
        tt = 0
        for b in train_loader:
            yp = model(b)
            l = criterion(yp, b.y)
            opt.zero_grad(); l.backward(); opt.step()
            tt += l.item() * b.num_graphs
        train_l.append(tt / len(train_data))
        model.eval()
        vv = 0
        with torch.no_grad():
            for b in val_loader:
                vv += criterion(model(b), b.y).item() * b.num_graphs
        val_l.append(vv / len(val_data))
    return train_l, val_l


# Week 14 的 MPNet（不带 encoder/decoder/残差）
class MPNet_Bare(nn.Module):
    def __init__(self, node_in, edge_in, hid, out_dim, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(EdgeUpdateMP(node_in, edge_in, hid))
        for _ in range(n_layers - 1):
            self.layers.append(EdgeUpdateMP(hid, hid, hid))
        self.head = nn.Linear(hid, out_dim)
    
    def forward(self, data):
        x = data.x
        edge_attr = data.edge_attr
        for layer in self.layers:
            x, edge_attr = layer(x, data.edge_index, edge_attr)
        return self.head(x)


torch.manual_seed(0)
model_bare = MPNet_Bare(4, 3, 64, 2, 3)
print("\n训练 MPNet（无 encoder/decoder）...")
train_bare, val_bare = train(model_bare)

torch.manual_seed(0)
model_phyfenet = PhyFENet_Mini(4, 3, 64, 2, 3)
print("\n训练 PhyFENet_Mini（含 encoder/decoder + 残差）...")
train_phy, val_phy = train(model_phyfenet)

# 对比
plt.figure(figsize=(8, 5))
plt.plot(val_bare, label='MPNet (bare)')
plt.plot(val_phy, label='PhyFENet_Mini (encoder/decoder + residual)')
plt.yscale('log')
plt.xlabel('Epoch'); plt.ylabel('Val MSE')
plt.title('架构对比')
plt.legend()
plt.savefig('week15_arch_comparison.png', dpi=100)

print(f"\n最终 MPNet val: {val_bare[-1]:.6f}")
print(f"最终 PhyFENet_Mini val: {val_phy[-1]:.6f}")
```

**期望结果**：PhyFENet_Mini 通常更好（因为 encoder 升维 + 残差更稳）——**但不一定**。重要的是你能从曲线看出"哪个收敛更平稳、哪个最终更低"。

---

#### ⚠️ Day 1-2 新手最容易踩的坑

**坑 1：encoder 用单层 Linear**
- 单层 Linear 表达能力有限
- 标准做法：encoder 是 2 层 MLP（Linear + ReLU + Linear）

**坑 2：所有 processor 层用不同 `hid_dim`**
- 错误：第一层 64，第二层 128
- 正确：所有 processor 层都用同一个 hid_dim
- 原因：残差连接要求维度相同

**坑 3：忘了在 processor 里加残差**
- `x = layer(x, ...)` 没有残差，深层会过平滑
- 正确：`x_new = layer(x, ...); x = x + x_new`

**坑 4：decoder 直接用 Linear，不用 MLP**
- 单 Linear 把 hid_dim 直接降到 2 维，表达能力被压缩
- 标准做法：decoder 也是 2 层 MLP

**坑 5：用过深的 processor（n_mp_layers=10+）**
- 即使有残差，10 层也容易过拟合小数据
- 工程经验：3-5 层够用。论文 PhyFENet 用 5-10 层但**配合更多数据**

---

#### 🧠 Day 1-2 概念问答 quiz

**Q1**：编码器-处理器-解码器架构的三个组件分别做什么？

<details><summary>答案</summary>编码器：把原始输入升维到 hid_dim（潜在空间）；处理器：在潜在空间做多层消息传递；解码器：把潜在特征降维到任务输出。</details>

**Q2**：为什么 processor 所有层都用同一个 hid_dim？

<details><summary>答案</summary>(1) 残差连接要求维度相同；(2) 在统一的潜在空间工作便于设计；(3) 模型设计简洁。</details>

**Q3**：PhyFENet_Mini 比 Week 14 的 MPNet 多了哪些东西？

<details><summary>答案</summary>(1) 编码器（升维）；(2) 解码器（降维）；(3) 残差连接。这三者都是从 MeshGraphNet 借鉴的标准设计。</details>

**Q4**：encoder 输出维度叫什么？为什么这个维度通常比输入大？

<details><summary>答案</summary>叫"潜在维度"或"hidden dim"。比输入大是因为消息传递需要足够的"工作空间"——4 维输入信息容量太小，处理器没法做复杂运算。</details>

**Q5**：如果你想用 8 层 processor，会有什么问题？怎么解决？

<details><summary>答案</summary>过平滑——所有节点特征趋同。解决：必须加残差连接（PhyFENet_Mini 已经加了）。即使加了残差，10 层以上仍可能不稳——论文用 5-10 层是实测经验值。</details>

---

#### 📦 Day 1-2 知识卡片

| 项目 | 内容 |
|---|---|
| **核心架构** | Encoder → Processor → Decoder |
| **维度变化** | 原始 → hid_dim（统一）→ 输出维度 |
| **关键设计** | 处理器维度统一；残差连接；encoder/decoder 都用 MLP |
| **典型 n_layers** | 3-5（小数据）；5-10（大数据 + 残差） |
| **本日产出** | `week15/day12_encoder_processor_decoder.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 3 | 单元聚合模块（论文 §2.3.4）

**🎯 本日目标**：理解 FEM 中"节点量 vs 单元量"的区别；实现节点特征→单元特征的聚合。

**🟢 直觉层**（约 10 分钟）：

到此你的模型预测的是**节点上的位移**（每个节点 ux, uy）。但 FEM 中还有另一类物理量：**单元上的应力和应变**。

**为什么这两者不同**：

```
位移 u：定义在节点上（每个节点有自己的位移）
应变 ε = du/dx：通过节点位移的差算出，**定义在单元内部**
应力 σ = E·ε：也定义在单元内部
```

**节点量和单元量的关系**：

```
节点 0 ─── 单元 e1 包含节点 [0,1,3,4]
节点 1     单元 e1 的应力 = f(节点 0,1,3,4 的位移)
节点 3
节点 4
```

**你的模型现在只输出节点量，怎么得到单元量**？

**答案**：再做一次"聚合"——把单元里所有节点的特征聚合起来作为单元特征。

**🟡 概念层 + 数学层**（约 25 分钟）：

**论文 §2.3.4 的聚合矩阵 C**：

设有 N 个节点、E 个单元，每个单元由若干节点组成。定义稀疏矩阵 C（shape = (E, N)）：

```
C[i, p] = 1 / |单元 i 的节点数|     如果节点 p 是单元 i 的组成部分
C[i, p] = 0                           否则
```

意思：**第 i 行**告诉你"第 i 个单元的特征 = 它里面所有节点特征的平均（每个节点贡献 1/单元节点数）"。

数学上：

```
X_elem = C · X_node

其中：
  X_node: shape = (N, features)，节点特征
  X_elem: shape = (E, features)，单元特征
  C: shape = (E, N)，聚合矩阵
```

**具体例子**（3×3 网格 4 个单元）：

```
单元 e1 = {节点 0, 1, 3, 4}    所以 C[0, 0] = C[0, 1] = C[0, 3] = C[0, 4] = 0.25
单元 e2 = {节点 1, 2, 4, 5}    C[1, 1] = C[1, 2] = C[1, 4] = C[1, 5] = 0.25
单元 e3 = {节点 3, 4, 6, 7}    类似
单元 e4 = {节点 4, 5, 7, 8}    类似
```

C 矩阵 shape = (4, 9)。每行只有 4 个非零元素（值 = 0.25）。**这是稀疏矩阵**——大网格上必须用稀疏存储（Day 4 提）。

**🟣 代码层**（约 1.5 小时）：

创建文件 `week15/day03_element_aggregation.py`：

```python
import torch
import torch.nn as nn
import numpy as np

def build_element_aggregation_matrix(elements, num_nodes, normalize=True):
    """构造"节点 → 单元"的聚合矩阵 C
    
    参数：
      elements: list of lists，每个元素是单元的节点索引（如 [[0,1,3,4], [1,2,4,5]]）
      num_nodes: 总节点数
      normalize: True = 平均；False = 求和
    返回：
      C: torch.Tensor shape=(num_elements, num_nodes)
    
    注意：本实现用 dense 矩阵（教学用）。真实工程（百万节点）必须用稀疏矩阵。
    """
    rows, cols, vals = [], [], []
    
    for eid, nids in enumerate(elements):
        for nid in nids:
            rows.append(eid)
            cols.append(nid)
            val = 1.0 / len(nids) if normalize else 1.0
            vals.append(val)
    
    # 用稀疏 COO 格式建矩阵，再转 dense（小网格用 dense 没问题）
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float)
    C_sparse = torch.sparse_coo_tensor(
        indices, values,
        size=(len(elements), num_nodes)
    )
    return C_sparse.to_dense()


def aggregate_nodes_to_elements(x_node, C):
    """
    x_node: shape=(N, features)
    C: shape=(E, N)
    返回 x_elem: shape=(E, features)
    """
    return C @ x_node


# ========== 测试 ==========
# 一个 3×3 规则网格（4 个单元）
elements = [
    [0, 1, 4, 3],    # 单元 e1（左下）
    [1, 2, 5, 4],    # 单元 e2（右下）
    [3, 4, 7, 6],    # 单元 e3（左上）
    [4, 5, 8, 7],    # 单元 e4（右上）
]
N = 9    # 9 个节点

C = build_element_aggregation_matrix(elements, num_nodes=N, normalize=True)
print(f"C shape: {C.shape}")    # (4, 9)
print(f"\nC 矩阵:\n{C}")

# 验证：每行非零元素之和 = 1（normalize=True）
print(f"\n每行非零元素之和: {C.sum(dim=1)}")    # 应都是 1.0

# ========== 用一个具体例子验证聚合 ==========
torch.manual_seed(0)
x_node = torch.randn(N, 5)    # 9 个节点，每个 5 维特征

x_elem = aggregate_nodes_to_elements(x_node, C)
print(f"\nx_elem shape: {x_elem.shape}")    # (4, 5)

# 验证：单元 0 的特征 = 节点 [0,1,4,3] 的特征均值
expected = x_node[[0, 1, 4, 3]].mean(dim=0)
actual = x_elem[0]
print(f"\n手算单元 0 特征（节点 [0,1,4,3] 均值）:")
print(f"  expected: {expected}")
print(f"  actual:   {actual}")
print(f"  是否一致: {torch.allclose(expected, actual, atol=1e-6)}")
```

**验收标准**：
- 聚合矩阵 shape = (4, 9)
- 每行非零元素和为 1.0（归一化版本）
- 单元 0 的聚合特征 = 节点 [0,1,4,3] 特征的均值
- `torch.allclose` 返回 True

**🔬 应用层**（约 5 分钟）：

**这个机制在 PhyFENet 论文里的角色**：

PhyFENet 同时输出**节点量**（位移）和**单元量**（应力、应变）。流程：

1. GNN（编码器-处理器-解码器）输出节点级特征 → 节点解码器 → 位移
2. 同样的节点级特征 → 通过 C 聚合 → 单元级特征 → 单元解码器 → 应力/应变

**注意**：节点和单元的解码器是**两个独立的 MLP**——因为它们预测不同物理量。

---

#### ⚠️ Day 3 新手最容易踩的坑

**坑 1：用 dense 矩阵处理大网格**
- 100 万节点的网格 C 是 (E, 1M) 矩阵——内存爆炸
- 解决：用 `torch.sparse_coo_tensor` 或 `torch_scatter.scatter_mean`
- 本周代码用 dense 是教学简化，真实工程要换

**坑 2：忘记归一化**
- 不归一化 C 时，`C @ x_node` 是单元里所有节点的**和**
- 节点数多的单元会得到更大值（不可比）
- 默认应该归一化

**坑 3：单元节点顺序错**
- 写错单元节点：比如把 [0,1,4,3] 写成 [0,1,2,3]——错误对应到不在该单元内的节点
- Day 5-6 整理时会有更严格的对应

---

#### 🧠 Day 3 概念问答 quiz

**Q1**：FEM 中节点量和单元量的物理区别是什么？举例。

<details><summary>答案</summary>节点量定义在节点上（如位移 u）；单元量定义在单元内部（如应力 σ、应变 ε）。位移是"位置变化"——节点天然有；应力/应变涉及空间导数 du/dx——必须在某个空间区域（单元）内才能计算。</details>

**Q2**：聚合矩阵 C 的 shape 是什么？

<details><summary>答案</summary>`(num_elements, num_nodes)`。每行对应一个单元，列对应所有节点。C[i, p] 表示节点 p 对单元 i 的贡献权重。</details>

**Q3**：归一化的 C 矩阵每行之和应该是什么？

<details><summary>答案</summary>`1.0`。因为归一化让"单元特征 = 节点特征的均值"，所有权重之和 = 1。</details>

**Q4**：3×3 网格有 9 个节点、4 个单元，C 矩阵的形状和稀疏度是多少？

<details><summary>答案</summary>shape=(4, 9)。每行只有 4 个非零元素（每个 4 节点单元）。总非零元素 16，总元素 36，稀疏度 ~44%。大网格稀疏度更高。</details>

**Q5**：如果你的模型需要同时输出节点位移和单元应力，怎么设计？

<details><summary>答案</summary>共用 GNN 主体（encoder + processor），分两个解码器：
- 节点解码器：直接从节点最终特征 → 位移
- 单元解码器：节点最终特征 通过 C 聚合 → 单元特征 → MLP → 应力/应变

两个 Loss 加权求和：`L = w_node·MSE(节点) + w_elem·MSE(单元)`</details>

---

#### 📦 Day 3 知识卡片

| 项目 | 内容 |
|---|---|
| **核心概念** | 节点量 vs 单元量；聚合矩阵 C |
| **C 的形式** | shape=(E, N)，每行非零元素 = 单元节点 |
| **归一化** | C[i, p] = 1/单元节点数 |
| **核心运算** | `X_elem = C @ X_node` |
| **工程提醒** | 大网格用稀疏矩阵 |
| **本日产出** | `week15/day03_element_aggregation.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 4 | 把单元聚合集成到完整模型

**🎯 本日目标**：扩展 PhyFENet_Mini 加入单元级输出。

**🟢 直觉层**（约 5 分钟）：

把 Day 1-2 的节点输出 + Day 3 的单元聚合机制结合——一个模型同时输出节点位移和单元应力。

**🟣 代码层**（约 2.5 小时）：

创建文件 `week15/day04_full_model.py`：

```python
import torch
import torch.nn as nn

# ========== 完整版 PhyFENet（节点 + 单元双输出）==========
class PhyFENet_WithElement(nn.Module):
    """节点级输出（位移）+ 单元级输出（应力/应变）
    
    参数：
      node_in: 节点输入维度
      edge_in: 边输入维度
      hid: 隐藏维度
      node_out_dim: 节点输出维度（如位移 ux, uy = 2）
      elem_out_dim: 单元输出维度（如应力 σxx, σyy, σxy = 3）
      n_mp_layers: 处理器层数
    """
    def __init__(self, node_in, edge_in, hid, node_out_dim, elem_out_dim, n_mp_layers=3):
        super().__init__()
        
        # === 编码器（节点 + 边）===
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hid), nn.ReLU(),
            nn.Linear(hid, hid)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in, hid), nn.ReLU(),
            nn.Linear(hid, hid)
        )
        
        # === 处理器 ===
        self.processor = nn.ModuleList([
            EdgeUpdateMP(hid, hid, hid) for _ in range(n_mp_layers)
        ])
        
        # === 解码器：两个独立 MLP ===
        # 节点解码器（位移）
        self.node_decoder = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, node_out_dim)
        )
        # 单元解码器（应力/应变）
        self.elem_decoder = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, elem_out_dim)
        )
    
    def forward(self, data, elem_aggr_matrix):
        """
        elem_aggr_matrix: 节点 → 单元的聚合矩阵 C，shape=(E, N)
        返回：
          node_out: shape=(N, node_out_dim)
          elem_out: shape=(E, elem_out_dim)
        """
        # === 编码 ===
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        
        # === 处理（带残差）===
        for layer in self.processor:
            x_new, edge_attr_new = layer(x, data.edge_index, edge_attr)
            x = x + x_new
            edge_attr = edge_attr + edge_attr_new
        
        # === 节点解码 ===
        node_out = self.node_decoder(x)
        
        # === 单元解码（先聚合，再 MLP）===
        x_elem = elem_aggr_matrix @ x       # (N, hid) → (E, hid)
        elem_out = self.elem_decoder(x_elem)
        
        return node_out, elem_out


# ========== 测试 ==========
import numpy as np
from torch_geometric.data import Data

# 复用 EdgeUpdateMP 和 build_plate_data, build_element_aggregation_matrix（前面的代码）

torch.manual_seed(0)
data, elements = build_plate_data(F_applied=100.0)
N = data.num_nodes

# 构造聚合矩阵
C = build_element_aggregation_matrix(elements, num_nodes=N, normalize=True)
print(f"C shape: {C.shape}")

# 模型
model = PhyFENet_WithElement(
    node_in=4, edge_in=3, hid=64,
    node_out_dim=2,    # ux, uy
    elem_out_dim=3,    # σxx, σyy, σxy（占位，本周不真训练）
    n_mp_layers=3
)

# 前向
node_out, elem_out = model(data, C)
print(f"\nnode_out shape: {node_out.shape}")    # (15, 2)
print(f"elem_out shape: {elem_out.shape}")      # (8, 3) —— 5×3 网格有 8 个 quad 单元

# ========== 双 Loss 训练（伪代码）==========
# 真实场景：
#   假设 data.y_node 是位移真值，data.y_elem 是应力真值
#   loss_node = MSE(node_out, data.y_node)
#   loss_elem = MSE(elem_out, data.y_elem)
#   loss = w_node * loss_node + w_elem * loss_elem
```

**验收标准**：
- 模型创建无错
- 节点输出 shape = (15, 2)
- 单元输出 shape = (8, 3)（5×3 板有 4×2=8 个单元）
- 你能解释为什么节点和单元的 decoder 是两个独立 MLP

---

#### ⚠️ Day 4 新手最容易踩的坑

**坑 1：节点和单元解码器共用一个 MLP**
- 错：`self.shared_decoder = nn.Linear(...)`，节点和单元都过它
- 错的原因：节点输出和单元输出物理量不同，要独立学习
- 正：两个独立的 MLP

**坑 2：聚合矩阵 C 没传进 forward**
- forward 签名要 `forward(self, data, elem_aggr_matrix)`
- C 是图的属性（不是模型的属性）——每张图有自己的 C
- 多图 batch 训练时，C 怎么传？答：可以放在 Data 对象里（PyG 支持自定义字段），或者在 batch loop 里传进去

**坑 3：双 Loss 权重没考虑量级**
- 节点位移可能在 0.01 量级，单元应力在 100 量级——MSE 量级差 1e8
- 如果不调权重，模型会主要拟合应力（量级大）
- 标准做法：归一化两者到相同量级（用 train 集的 std），或动态权重（Part B Week 10 的等比例权重）

---

#### 🧠 Day 4 概念问答 quiz

**Q1**：PhyFENet 输出节点位移和单元应力，为什么用两个独立的 decoder？

<details><summary>答案</summary>(1) 节点和单元预测的物理量不同（位移 vs 应力），需要独立的"翻译规则"；(2) 两个 decoder 可以独立学习——加深 + 调整网络结构互不干扰；(3) 灵活性更高。</details>

**Q2**：双输出的 Loss 怎么组合？要注意什么？

<details><summary>答案</summary>`L = w_node · MSE(node) + w_elem · MSE(elem)`。注意两类物理量的量级差异——位移和应力可能差 100-1000 倍。建议先归一化目标值到相同量级，或动态调整权重。</details>

**Q3**：聚合矩阵 C 在多图 batch 里怎么处理？

<details><summary>答案</summary>每张图有自己的 C。两种做法：(1) 把 C 当作 Data 对象的属性（PyG 支持自定义字段），DataLoader 自动 batch；(2) 在 batch loop 里手动取每张图的 C。复杂场景要小心 PyG batch 的节点编号偏移。</details>

**Q4**：如果你的模型只输出节点级量，能否反推单元级量？

<details><summary>答案</summary>**理论上可以**——比如位移 u → 用 C 聚合可得到单元平均位移，但**不能直接得应力**（应力涉及空间导数 du/dx，不是简单聚合）。所以单元级输出通常需要单独的 head（直接学应力的映射）。</details>

---

#### 📦 Day 4 知识卡片

| 项目 | 内容 |
|---|---|
| **完整架构** | Encoder + Processor + 节点 Decoder + 单元 Decoder |
| **关键设计** | 节点和单元解码器**独立** |
| **聚合连接** | 节点 hid 特征 → C @ → 单元 hid 特征 |
| **双 Loss** | 注意量级匹配 |
| **本日产出** | `week15/day04_full_model.py` |
| **掌握要求** | 🟣 能用 |

---

### Day 5-6（周末）| Week 15 整理 + 文档化

**🎯 本日目标**：把本周的所有组件整理成可复用的工具库；写架构说明文档。

**🟣 代码层 Day 5**（约 1.5 小时）：

把核心代码封装到 `utils/gnn_models.py`：

```python
# utils/gnn_models.py 文件结构

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class EdgeUpdateMP(MessagePassing):
    """带边更新的消息传递层（论文 §2.3.3 核心）"""
    # ... Week 14 的实现


class PhyFENet_Mini(nn.Module):
    """节点级输出版（适合纯位移预测）"""
    # ... Week 15 Day 1-2 的实现


class PhyFENet_WithElement(nn.Module):
    """节点 + 单元双输出版"""
    # ... Week 15 Day 4 的实现


def build_element_aggregation_matrix(elements, num_nodes, normalize=True):
    """节点 → 单元 聚合矩阵 C"""
    # ... Week 15 Day 3 的实现


def measure_node_similarity(model, data):
    """诊断过平滑——算节点特征余弦相似度"""
    # ... Week 14 Day 5-6 的实现
```

**📝 文档化 Day 6**（约 2 小时）：

创建 `docs/phyfenet_architecture.md`，写架构说明：

```markdown
# PhyFENet 架构（Mini 版）

## 总体架构

PhyFENet 采用**编码器-处理器-解码器**架构：

```
原始输入 → [编码器] → 潜在空间 → [处理器（多层带边更新 MP）] → [解码器] → 输出
```

## 各组件作用

### 编码器
- 节点编码器：把节点原始特征（如 (x, y, is_fixed, F)）升到 hid_dim
- 边编码器：把边原始特征（如 (dx, dy, dist)）升到 hid_dim
- 实现：2 层 MLP

### 处理器
- 多层 EdgeUpdateMP（论文 §2.3.3 的带边更新消息传递）
- 每层带残差连接（防过平滑）
- 节点和边特征都被更新

### 解码器
- 节点解码器：hid_dim → 节点级输出（如位移）
- 单元解码器：hid_dim 经过 C 聚合 → 单元级输出（如应力）
- 实现：2 层 MLP

## 单元聚合机制（论文 §2.3.4）

聚合矩阵 C: shape=(num_elements, num_nodes)
每行非零元素表示该单元的节点贡献权重（归一化为 1/单元节点数）。

X_elem = C @ X_node

## 参数选择经验

- hid_dim：32-128（小数据用 32-64，大数据用 128+）
- n_mp_layers：3-5（小数据），5-10（大数据 + 残差）
- 损失权重：节点和单元的 MSE 量级差异需要平衡

## 与论文的对应关系

| 本实现 | 论文章节 |
|---|---|
| EdgeUpdateMP | §2.3.3 式 2.14-2.17 |
| 节点解码器 | §2.3.5 子网络（StrainSubNet 风格） |
| C 聚合矩阵 | §2.3.4 式 2.19 |
| 编码器-处理器-解码器 | §2.3 整体框架 |
```

**验收标准**：
- `utils/gnn_models.py` 整理完成，可以被其他脚本 import
- 文档清晰，能让别人（或未来的你）快速理解架构

---

### ✅ Week 15 完成评估

#### 任务级 checklist

- [ ] `week15/day12_encoder_processor_decoder.py` PhyFENet_Mini 跑通
- [ ] `week15/day03_element_aggregation.py` 聚合矩阵正确
- [ ] `week15/day04_full_model.py` 双输出模型跑通
- [ ] `utils/gnn_models.py` 整理完成
- [ ] `docs/phyfenet_architecture.md` 写完

#### 概念级 quiz（10 题，至少 8 题对）

**Q1**：编码器-处理器-解码器架构的三个组件分别做什么？

<details><summary>答案</summary>编码器升维到 hid_dim；处理器在 hid 空间做多层 MP；解码器降到任务输出维度。</details>

**Q2**：处理器所有层用同一个 hid_dim 的两个原因？

<details><summary>答案</summary>(1) 残差连接要求维度相同；(2) 在统一潜在空间工作便于设计。</details>

**Q3**：FEM 中节点量和单元量的物理区别？

<details><summary>答案</summary>节点量定义在节点上（位移），单元量定义在单元内（应力、应变）。后者涉及空间导数。</details>

**Q4**：聚合矩阵 C 的 shape 是什么？归一化版本每行之和是什么？

<details><summary>答案</summary>shape=(E, N)；归一化每行和=1.0。</details>

**Q5**：双输出的 PhyFENet 用一个 decoder 还是两个？为什么？

<details><summary>答案</summary>两个独立 decoder。节点和单元预测物理量不同，需要独立"翻译规则"。</details>

**Q6**：双 Loss 训练要注意什么？

<details><summary>答案</summary>节点和单元物理量量级差异大（位移 vs 应力），需要归一化或动态权重。</details>

**Q7**：3×3 网格 4 个单元的 C 矩阵 shape 是？每行有多少非零元素？

<details><summary>答案</summary>(4, 9)；每行 4 个非零元素（4 节点单元，每个 0.25）。</details>

**Q8**：处理器的层数怎么选？

<details><summary>答案</summary>3-5 层（小数据）；5-10 层（大数据 + 残差连接）。论文 PhyFENet 用 5-10 层。</details>

**Q9**：encoder 为什么用 2 层 MLP 而不是单层 Linear？

<details><summary>答案</summary>2 层 MLP 表达能力更强，能捕捉非线性映射。单层 Linear 只能做线性映射，限制了 encoder 对原始特征的"理解能力"。</details>

**Q10**：PhyFENet_WithElement 的 forward 为什么需要传入 `elem_aggr_matrix`？

<details><summary>答案</summary>聚合矩阵 C 是图的属性（每张图的 C 不同），不是模型的属性。所以要在 forward 时把 C 一起传入。多图 batch 时需要小心处理（每张图的 C 拼接或单独处理）。</details>

#### 🚦 自我评估

- 任务全部通过 + Quiz 8 题对 → **绿灯进入 Week 16**
- PhyFENet_Mini 不能独立写出 → **黄灯**——重看 Day 1-2 架构
- 单元聚合的物理意义说不清 → **黄灯**——重看 Day 3 概念层

#### 知识地图自查

- [ ] (12) Encoder-Processor-Decoder → 🟣
- [ ] (13) 单元聚合 → 🟣

---

## 进入 Week 16 之前

下周 Week 16 是 **Part C 的收官**：综合实战 + Part C 自测。

下周需要的前置：
- `EdgeUpdateMP` 能从零写
- PhyFENet_Mini 架构能看懂
- 聚合矩阵能构造

---

---

## Week 16: Part C 综合实战 + 自测

> ⚠️ **关于本周数据的重要说明**：
>
> 本周用的数据是**结构化合成数据**，**不是严格的 FEM 解**。数据生成公式（`ux = F·x/(E·A)`、`uy = -ν·F·y/(E·A)`、`ε = F/(E·A)`、`σ = F/A`）是一维弹性拉伸的粗糙近似——`ux` 接近正确，但 `uy` 的泊松收缩公式是简化版本，且节点位移和单元应力之间**并不严格满足几何 + 本构关系的物理自洽性**。
>
> **为什么仍然用这个合成数据**：
> 1. 目标是验证你的 GNN 架构（编码器-处理器-解码器 + 单元聚合）能**跑通**节点级和单元级双输出的训练
> 2. 真实 FEM 数据获取涉及 FEniCS/LS-DYNA 的使用——这是 Part E 的内容
> 3. 这里合成数据的粗糙是可控的——能让你专注在 GNN 架构本身，不被真实数据的复杂度干扰
>
> **合理预期**：
> - 模型能收敛（Loss 下降）
> - 节点和单元输出都在合理范围内
> - 但**不要过度解读模型的绝对精度**——它拟合的是粗糙近似，不是物理真相
>
> **不要说的话**：
> - "我的模型达到了 XX% 的 FEM 精度"——这不对，标签本身不是 FEM 真解
> - "这证明了 GNN 能替代 FEM"——不能，这只证明了架构能跑
>
> 第二阶段（Part E 起）进入真实 FEM 数据后，上述数字和结论才有可解读的物理意义。

### 🎯 Week 16 总览

**本周覆盖**：Part C 综合实战（多样本训练 + 双输出）、Part C 自测

**学完之后你应该能**：
- 🟣 **能用**：在多样本数据上训练带双输出的 PhyFENet
- 🟡 **能讲**：完整 PhyFENet 架构 + 它和论文 §2.3 的对应

---

### Day 1-2 | 构造相对完整的合成数据集

**🎯 本日目标**：基于规则网格生成多样本数据集（变化 E 和 F），含双标签（位移 + 应力）。

**🟢 直觉层**（约 5 分钟）：

到此你的"数据集"只是几张相同板材、不同载荷的图。本周做一个更接近真实工程的版本——同时变化弹性模量 E 和载荷 F，生成 50 个样本。

**🟣 代码层**（约 3 小时，分两天）：

创建文件 `week16/day12_dataset.py`：

```python
import torch
import numpy as np
from torch_geometric.data import Data


def build_full_plate_data(nx=5, ny=3, E=200.0, F=100.0, nu=0.3):
    """构造带双标签的数据
    
    参数：
      E: 弹性模量
      F: 载荷
      nu: 泊松比
    返回：
      Data 对象（含节点特征、边特征、节点标签、单元标签）
    """
    # ===== 网格 =====
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([float(i), float(j)])
    nodes = np.array(nodes, dtype=np.float32)
    
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i; n1 = j * nx + (i+1)
            n2 = (j+1) * nx + (i+1); n3 = (j+1) * nx + i
            elements.append([n0, n1, n2, n3])
    
    edge_set = set()
    for elem in elements:
        for i in range(4):
            for j in range(i+1, 4):
                a, b = min(elem[i], elem[j]), max(elem[i], elem[j])
                edge_set.add((a, b))
    src, dst = [], []
    for a, b in edge_set:
        src.append(a); dst.append(b)
        src.append(b); dst.append(a)
    edge_index = np.array([src, dst], dtype=np.int64)
    
    # ===== 节点特征：(x, y, is_fixed, F_applied, E) =====
    N = len(nodes)
    is_fixed = np.zeros((N, 1), dtype=np.float32)
    F_feat = np.zeros((N, 1), dtype=np.float32)
    E_feat = np.full((N, 1), E, dtype=np.float32)
    
    is_fixed[np.abs(nodes[:, 0]) < 0.01] = 1.0
    F_feat[np.abs(nodes[:, 0] - nodes[:, 0].max()) < 0.01] = F
    
    x_features = np.concatenate([nodes, is_fixed, F_feat, E_feat], axis=1)    # (N, 5)
    
    # ===== 边特征 =====
    diff = nodes[edge_index[1]] - nodes[edge_index[0]]
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    edge_attr = np.concatenate([diff, dist], axis=1).astype(np.float32)    # (n_edges, 3)
    
    # ===== 节点标签：位移 (ux, uy) =====
    # 简化的拉伸位移模型：
    # ux = (F/(E*A)) * x   （标准的轴向位移）
    # uy = -nu * (F/(E*A)) * (y - y_center)   （泊松收缩，简化版）
    A_cross = 1.0    # 截面积
    epsilon = F / (E * A_cross)    # 应变
    sigma_val = F / A_cross         # 应力
    
    y_center = (ny - 1) * 0.5
    ux = epsilon * nodes[:, 0:1]
    uy = -nu * epsilon * (nodes[:, 1:2] - y_center)
    ux[np.abs(nodes[:, 0]) < 0.01] = 0
    uy[np.abs(nodes[:, 0]) < 0.01] = 0
    y_node = np.concatenate([ux, uy], axis=1).astype(np.float32)    # (N, 2)
    
    # ===== 单元标签：应力 (σxx, σyy, σxy) =====
    # 简化的均匀应力：σxx = σ, σyy = 0, σxy = 0（简单拉伸）
    n_elem = len(elements)
    y_elem = np.zeros((n_elem, 3), dtype=np.float32)
    y_elem[:, 0] = sigma_val    # σxx
    y_elem[:, 1] = 0.0           # σyy
    y_elem[:, 2] = 0.0           # σxy
    
    # ===== 构造 Data ==========
    data = Data(
        x=torch.from_numpy(x_features).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr).float(),
        y_node=torch.from_numpy(y_node).float(),
        y_elem=torch.from_numpy(y_elem).float()
    )
    
    return data, elements


# ========== 生成多样本数据集 ==========
np.random.seed(42)
torch.manual_seed(42)

n_samples = 50
all_data = []
all_elements = []

for i in range(n_samples):
    # 随机采样 E 和 F
    E_sample = np.random.uniform(150, 300)
    F_sample = np.random.uniform(50, 150)
    data, elements = build_full_plate_data(nx=5, ny=3, E=E_sample, F=F_sample)
    all_data.append(data)
    all_elements.append(elements)

# 切分 train / val
n_train = 40
train_data = all_data[:n_train]
val_data = all_data[n_train:]
train_elements = all_elements[:n_train]
val_elements = all_elements[n_train:]

print(f"训练集: {len(train_data)} 张图")
print(f"验证集: {len(val_data)} 张图")
print(f"\n样本信息（第一个）:")
print(f"  节点特征: {train_data[0].x.shape}")
print(f"  边特征: {train_data[0].edge_attr.shape}")
print(f"  节点标签: {train_data[0].y_node.shape}")
print(f"  单元标签: {train_data[0].y_elem.shape}")
```

**验收标准**：
- 50 个样本生成成功
- 每个样本都有节点特征 (15, 5)、边特征、双标签

---

### Day 3-4 | 训练 PhyFENet_WithElement

**🎯 本日目标**：在双标签数据上训练完整的 PhyFENet。

**🟣 代码层**（约 3 小时）：

创建文件 `week16/day34_train_full_model.py`：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# 复用所有前面的代码：EdgeUpdateMP, PhyFENet_WithElement, build_element_aggregation_matrix, build_full_plate_data
# （这里 import 或直接复制定义）

# ========== 数据归一化 ==========
def normalize_dataset(train_data, val_data):
    """用训练集的统计量归一化所有数据
    
    返回归一化后的数据集和归一化器（用于后续反归一化）
    """
    # 收集训练集的所有节点特征、边特征、节点标签、单元标签
    all_x = torch.cat([d.x for d in train_data], dim=0)
    all_edge = torch.cat([d.edge_attr for d in train_data], dim=0)
    all_y_node = torch.cat([d.y_node for d in train_data], dim=0)
    all_y_elem = torch.cat([d.y_elem for d in train_data], dim=0)
    
    # 算 mean / std
    x_mean, x_std = all_x.mean(0), all_x.std(0) + 1e-8
    edge_mean, edge_std = all_edge.mean(0), all_edge.std(0) + 1e-8
    y_node_mean, y_node_std = all_y_node.mean(0), all_y_node.std(0) + 1e-8
    y_elem_mean, y_elem_std = all_y_elem.mean(0), all_y_elem.std(0) + 1e-8
    
    # 归一化
    def normalize_one(d):
        d_new = Data(
            x=(d.x - x_mean) / x_std,
            edge_index=d.edge_index,
            edge_attr=(d.edge_attr - edge_mean) / edge_std,
            y_node=(d.y_node - y_node_mean) / y_node_std,
            y_elem=(d.y_elem - y_elem_mean) / y_elem_std
        )
        return d_new
    
    train_norm = [normalize_one(d) for d in train_data]
    val_norm = [normalize_one(d) for d in val_data]
    
    stats = {
        'x_mean': x_mean, 'x_std': x_std,
        'edge_mean': edge_mean, 'edge_std': edge_std,
        'y_node_mean': y_node_mean, 'y_node_std': y_node_std,
        'y_elem_mean': y_elem_mean, 'y_elem_std': y_elem_std
    }
    
    return train_norm, val_norm, stats


# ========== 训练循环 ==========
torch.manual_seed(0)

# 生成数据（用 Day 1-2 的 build_full_plate_data）
np.random.seed(42)
n_samples = 50
all_data, all_elements = [], []
for i in range(n_samples):
    E = np.random.uniform(150, 300)
    F = np.random.uniform(50, 150)
    d, e = build_full_plate_data(E=E, F=F)
    all_data.append(d); all_elements.append(e)

train_data = all_data[:40]
val_data = all_data[40:]
train_elements = all_elements[:40]

# 归一化
train_norm, val_norm, stats = normalize_dataset(train_data, val_data)

# 聚合矩阵（5×3 网格的 elements 是固定的）
N = train_norm[0].num_nodes
C = build_element_aggregation_matrix(train_elements[0], num_nodes=N, normalize=True)

# DataLoader
train_loader = DataLoader(train_norm, batch_size=4, shuffle=True)
val_loader = DataLoader(val_norm, batch_size=4)

# 模型
model = PhyFENet_WithElement(
    node_in=5, edge_in=3, hid=64,
    node_out_dim=2, elem_out_dim=3,
    n_mp_layers=3
)
print(f"参数数: {sum(p.numel() for p in model.parameters())}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 双权重（归一化后量级接近，简单 1:1 即可）
w_node, w_elem = 1.0, 1.0

train_losses_node, train_losses_elem = [], []
val_losses_node, val_losses_elem = [], []

print("\n训练完整 PhyFENet（双输出）...")
for epoch in range(300):
    model.train()
    train_node, train_elem = 0, 0
    for batch in train_loader:
        # 注意：batch 里的数据已经归一化
        # 多图 batch 时 C 需要为每张图分别处理（这里简化：所有图共用 C，因为都是同样的 5×3 网格）
        node_pred, elem_pred = model(batch, C)
        
        # batch 的 y_node 已归一化
        loss_node = criterion(node_pred, batch.y_node)
        loss_elem = criterion(elem_pred, batch.y_elem)
        loss = w_node * loss_node + w_elem * loss_elem
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_node += loss_node.item() * batch.num_graphs
        train_elem += loss_elem.item() * batch.num_graphs
    
    train_losses_node.append(train_node / len(train_norm))
    train_losses_elem.append(train_elem / len(train_norm))
    
    # 验证
    model.eval()
    val_node, val_elem = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            node_pred, elem_pred = model(batch, C)
            val_node += criterion(node_pred, batch.y_node).item() * batch.num_graphs
            val_elem += criterion(elem_pred, batch.y_elem).item() * batch.num_graphs
    val_losses_node.append(val_node / len(val_norm))
    val_losses_elem.append(val_elem / len(val_norm))
    
    if epoch % 30 == 0:
        print(f"  epoch {epoch}: train(node={train_losses_node[-1]:.6f}, "
              f"elem={train_losses_elem[-1]:.6f}), "
              f"val(node={val_losses_node[-1]:.6f}, elem={val_losses_elem[-1]:.6f})")

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(train_losses_node, label='Train')
axes[0].plot(val_losses_node, label='Val')
axes[0].set_yscale('log')
axes[0].set_title('节点位移 Loss')
axes[0].legend()

axes[1].plot(train_losses_elem, label='Train')
axes[1].plot(val_losses_elem, label='Val')
axes[1].set_yscale('log')
axes[1].set_title('单元应力 Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('week16_full_train.png', dpi=100)

print(f"\n最终 val 节点 Loss: {val_losses_node[-1]:.6f}")
print(f"最终 val 单元 Loss: {val_losses_elem[-1]:.6f}")
```

**验收标准**：
- 训练能跑通
- 节点和单元 Loss 都在下降
- val Loss 不离 train 太远（< 5x）

---

### Day 5 | 可视化：预测 vs 真实

**🎯 本日目标**：直观可视化模型预测 vs 真实标签。

**🟣 代码层**（约 1.5 小时）：

```python
# week16/day5_visualize.py
import torch
import matplotlib.pyplot as plt

# 载入 Day 3-4 训练好的模型 + 一个 val 样本
sample_idx = 0
data = val_data[sample_idx]    # 未归一化的真实数据

# 归一化输入
data_norm = Data(
    x=(data.x - stats['x_mean']) / stats['x_std'],
    edge_index=data.edge_index,
    edge_attr=(data.edge_attr - stats['edge_mean']) / stats['edge_std'],
    y_node=data.y_node, y_elem=data.y_elem
)

# 预测（在归一化空间）
model.eval()
with torch.no_grad():
    node_pred_norm, elem_pred_norm = model(data_norm, C)

# 反归一化（拿回真实量级）
node_pred = node_pred_norm * stats['y_node_std'] + stats['y_node_mean']
elem_pred = elem_pred_norm * stats['y_elem_std'] + stats['y_elem_mean']

# 真实值
y_node_true = data.y_node
y_elem_true = data.y_elem

# 节点坐标（从输入特征前 2 维）
nodes = data.x[:, :2].numpy()

# ========== 3×2 subplot ==========
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# 第一行：节点位移（箭头）
for col, (disp, title) in enumerate([
    (y_node_true, 'True 位移'),
    (node_pred, 'Pred 位移')
]):
    ax = axes[0, col]
    ax.quiver(nodes[:, 0], nodes[:, 1],
              disp[:, 0].numpy(), disp[:, 1].numpy(),
              angles='xy', scale_units='xy', scale=0.05)
    ax.scatter(nodes[:, 0], nodes[:, 1], c='red', s=80, zorder=10)
    ax.set_aspect('equal')
    ax.set_title(title)

# 第二行：单元应力 σxx（颜色图）
# 单元中心坐标（从 elements 算）
elements_arr = np.array(train_elements[0])
elem_centers = nodes[elements_arr].mean(axis=1)    # (n_elem, 2)

for col, (sig, title) in enumerate([
    (y_elem_true[:, 0], 'True σxx'),
    (elem_pred[:, 0], 'Pred σxx')
]):
    ax = axes[1, col]
    sc = ax.scatter(elem_centers[:, 0], elem_centers[:, 1], 
                    c=sig.numpy(), s=300, marker='s', cmap='viridis')
    plt.colorbar(sc, ax=ax)
    ax.set_aspect('equal')
    ax.set_title(title)

# 第三行：节点位移 ux 单独（曲线对比）
ux_true = y_node_true[:, 0].numpy()
ux_pred = node_pred[:, 0].numpy()

for col in range(2):
    ax = axes[2, col]
    if col == 0:
        ax.scatter(nodes[:, 0], ux_true, label='True', alpha=0.7)
        ax.scatter(nodes[:, 0], ux_pred, label='Pred', alpha=0.7, marker='x')
        ax.set_xlabel('x'); ax.set_ylabel('ux')
        ax.set_title('ux 沿 x 分布')
        ax.legend()
    else:
        # 误差直方图
        err = (node_pred - y_node_true).abs().numpy()
        ax.hist(err.flatten(), bins=20)
        ax.set_xlabel('|位移误差|')
        ax.set_title('节点位移误差分布')

plt.tight_layout()
plt.savefig('week16_visualization.png', dpi=100)
```

**验收标准**：
- 3×2 图能生成
- 节点位移：True 和 Pred 箭头方向、长度大致一致
- 应力分布：True 和 Pred 颜色图相似
- ux 曲线：散点近似在同一条线上

---

### Day 6 | Part C 自测

**🎯 本日目标**：完成 Part C 的最终自测。

#### 理论自测（书面回答，每题 3-5 句）

1. 图数据有哪三种表示？各自的优缺点？

2. GCN 公式 `D^(-1/2)·Â·D^(-1/2)` 中每一项的目的？

3. GraphSAGE 相对 GCN 的改进是什么？

4. 消息传递范式的三个阶段是什么？

5. PhyFENet 相对标准 GraphSAGE 的关键改动？至少说出 3 点。

6. 什么是过平滑？怎么解决？

7. 为什么节点和单元的 decoder 要用两个独立的 MLP？

8. 双 Loss 训练要注意什么？

#### 代码自测（限时完成，不看参考）

**任务 1（20 min）**：从零写 `build_regular_mesh` + `mesh_to_edges`

**任务 2（30 min）**：从零构造 PyG 的 `Data`（含节点特征、边特征、节点标签）

**任务 3（40 min）**：从零写带边更新的 `MessagePassing` 层（`EdgeUpdateMP`）

**任务 4（30 min）**：搭建完整的编码器-处理器-解码器架构（PhyFENet_Mini）

#### 自评标准

- 4 个任务全部能在限时内完成 → **绿灯进入 Part D**
- 任务 3-4 不能独立完成 → **黄灯**——这是 Part C 的核心，必须熟练
- 任何一个任务做不到 → **红灯**——回到对应周补强

---

### ✅ Week 16 完成评估

#### 任务级 checklist

- [ ] `week16/day12_dataset.py` 50 样本数据集生成
- [ ] `week16/day34_train_full_model.py` 双输出训练成功
- [ ] `week16/day5_visualize.py` 可视化对比图
- [ ] Day 6 自测的 4 个代码任务都能限时完成

---

## ✅ Part C 总完成标准

进入 Part D 前必须达到。

### 理论掌握

- [ ] 能说清楚图数据的结构（节点 / 边 / 邻接矩阵）
- [ ] 能对比 GCN / GraphSAGE / 带边更新 MP 三种 GNN 的机制差异
- [ ] 能画出 PhyFENet 架构图（编码器 + 带边更新 MP + 解码器 + 单元聚合）
- [ ] 理解过平滑问题和残差连接

### 代码能力（限时完成，不看参考）

- [ ] 能从零写"规则网格 → 图"转换
- [ ] 能用 PyG 构造 Data 对象
- [ ] 能从零写自定义 MessagePassing 层（含边更新）
- [ ] 能搭建编码器-处理器-解码器架构
- [ ] 能在多图数据集上训练 GNN

### 知识地图最终自查

完整 Part C 14 个知识点：

- [ ] (1) 图三种表示 → 🟣
- [ ] (2) 网格转图 → 🟣
- [ ] (3) PyG Data → 🟣
- [ ] (4) 弹性板数据 → 🟣
- [ ] (5) GCN 公式 → 🟡
- [ ] (6) NumPy 手写 GCN → 🔵
- [ ] (7) PyG GCNConv → 🟣
- [ ] (8) 多图训练 → 🟣
- [ ] (9) 消息传递三阶段 → 🟡
- [ ] (10) GraphSAGE → 🟣
- [ ] (11) 带边更新 MP → 🔵
- [ ] (12) Encoder-Processor-Decoder → 🟣
- [ ] (13) 单元聚合 → 🟣
- [ ] (14) 完整 PhyFENet 训练 → 🟣

至少 12 个 🟣/🔵 才算 Part C 完成。

**关键节点必须达到 🔵 能写**：
- (6) NumPy 手写 GCN（建立直觉的关键）
- (11) 带边更新 MP（论文核心）

如果上述任何一项未达到，**Part D 的"GNN + PINN 融合"会非常困难**——因为那时需要在 GNN 基础上加物理约束 Loss。地基不稳，加层就塌。

---

## Part C 整体回顾

恭喜你完成了 Part C 的全部 6 周内容！让我们看看你走了多远：

**起点（Week 11）**：
- 你只会在张量上操作 MLP
- 不知道"图"作为数据结构存在
- 看不懂 PyG 的 Data 对象

**终点（Week 16）**：
- 能从零写 PhyFENet 架构（论文 §2.3）
- 能在多样本图数据上做端到端训练
- 能解释 GCN/GraphSAGE/MPNet 的差异
- 知道 FEM 中节点量和单元量的物理关系

**还差什么**：
- ❌ 真实 FEM 数据（合成数据 vs 真实仿真）
- ❌ PINN 的物理约束（Part B 学了，但还没和 GNN 结合）
- ❌ 大网格 / 不规则网格

**Part D 会解决前两个**：
- 用 FEniCS 跑真实 FEM 仿真生成数据
- 把 GNN（Part C）和物理约束 Loss（Part B）融合起来 → PhyFENet 框架的雏形

**Part E-G 处理后两个**——大规模训练 + 工程化部署。

到这里，你已经站在了 PhyFENet 论文方向的**真正起点上**。前面的每一步都是为了这个起点。

下一段：Part D（Week 17-22）——PINN 与有限元数据入门。