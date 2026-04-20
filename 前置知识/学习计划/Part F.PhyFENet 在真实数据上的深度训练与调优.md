# Part F: PhyFENet 在真实数据上的深度训练与调优（Week 29–33）

## 本阶段定位

**衔接 Part D 和 Part E**：
- **Part D** 你实现了 PhyFENet 的所有组件（编码器-处理器-解码器架构、带边更新消息传递、几何一致性 Loss、本构 Loss、多级网络达标线），但全部在合成数据上做的
- **Part E** 你用 FEniCS 生成了 50 个真实带孔板样本，把 PhyFENet 在这些数据上做了**第一次纯数据驱动的训练**（Week 27，单图，batch_size=1）

**Part D 和 Part E 留下的未完成项**：
- Part D batch 版本训练（标记为"进阶，可延后"）
- 物理约束 Loss 在真实数据上的效果（Part D 只在合成数据验证过机制）
- 多级网络的完整版（StressSubNet + 联合微调）
- 迁移学习（不同材料 / 不同几何参数）
- 小样本实验（论文 §3.2.3 的核心论点之一）

**Part F 要做的事**：把上面这些未完成项在真实数据上逐一落地。核心任务**不是学新概念**，而是**工程化调优**——让一个"勉强能跑"的模型变成"能放在简历里"的模型。

## ⚠️ 进入 Part F 的准入检查

**Part F 对前置数据有强依赖**。在开始 Week 29 之前，**务必检查以下字段是否齐备**。如果不齐，Part F 从第一天就会卡住。

### 数据集必备字段（每个 PyG Data 对象）

- [ ] `data.x` shape=(N, 8)：节点特征（坐标+材料+边界条件）
- [ ] `data.edge_index` shape=(2, 2E)：双向边索引
- [ ] `data.edge_attr` shape=(2E, 3)：边特征 (dx, dy, dist)
- [ ] `data.y` shape=(N, 2)：**节点位移标签** (ux, uy)
- [ ] **`data.y_elem` shape=(n_elem, 3)：单元应力标签 (σxx, σyy, σxy)** ← 关键
- [ ] **`data.y_elem_strain` shape=(n_elem, 3)：单元应变标签 (εxx, εyy, γxy)** ← 关键
- [ ] `data.elem_list`：单元列表，每项含 `indices`（节点索引）和 `type`
- [ ] `data.E`, `data.nu`, `data.T_x`, `data.r`：样本级元数据

### 如何检查

```python
dataset = torch.load('data/phase2_dataset/plate_hole_dataset.pt')
sample = dataset[0]
required = ['x', 'edge_index', 'edge_attr', 'y', 'y_elem', 
            'y_elem_strain', 'elem_list']
for field in required:
    assert hasattr(sample, field), f"缺少字段: {field}"
print("所有必备字段齐备，可以进入 Part F")
```

### 如果字段不齐（尤其是 y_elem_strain 或 y_elem 缺失）

**回到 Part E Week 26 补计算**。这不是可选项，是 Part F 的前提。

补计算流程：

```python
# 在 utils/fenicsx_solver.py 的 solve_plate_hole 函数里，
# 除了返回节点位移和 von Mises 应力，还要返回完整的应变/应力张量

# 1. 在求解后，额外计算单元级应变和应力
# 应变 ε_ij = (∂u_i/∂x_j + ∂u_j/∂x_i) / 2
# 应力 σ = D · ε

def compute_element_strain_stress(domain, uh, E, nu):
    """
    计算每个单元中心处的应变和应力张量。

    返回：
      strain: np.ndarray shape=(n_elem, 3)  (εxx, εyy, γxy)
      stress: np.ndarray shape=(n_elem, 3)  (σxx, σyy, σxy)
    """
    import ufl
    from dolfinx import fem
    from dolfinx.fem import Expression

    # 应变张量（作为函数空间上的函数）
    V_tensor = fem.functionspace(
        domain, ("DG", 0, (3,))  # 单元常量，3 分量（εxx, εyy, γxy）
    )

    strain_expr_components = [
        uh[0].dx(0),                                    # εxx = ∂ux/∂x
        uh[1].dx(1),                                    # εyy = ∂uy/∂y
        uh[0].dx(1) + uh[1].dx(0)                       # γxy = ∂ux/∂y + ∂uy/∂x
    ]

    # 用 Expression 把 UFL 表达式投影到 DG0 空间（逐单元常量）
    strain_func = fem.Function(V_tensor)
    # 具体投影代码见 dolfinx 教程 "Function projection"
    # ...（省略完整实现，详见 FEniCS 教程）

    strain_values = strain_func.x.array.reshape(-1, 3)

    # 应力 = D · 应变（平面应力）
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 - nu ** 2))
    coef = E / (1 - nu ** 2)

    stress_values = np.zeros_like(strain_values)
    stress_values[:, 0] = coef * (strain_values[:, 0] + nu * strain_values[:, 1])   # σxx
    stress_values[:, 1] = coef * (strain_values[:, 1] + nu * strain_values[:, 0])   # σyy
    stress_values[:, 2] = E / (2 * (1 + nu)) * strain_values[:, 2]                  # σxy

    return strain_values, stress_values

# 2. 在 solve_plate_hole 的返回字典里加上这两个字段
# result['y_elem_strain'] = strain_values
# result['y_elem'] = stress_values

# 3. 在 Week 25 的 fenicsx_result_to_pyg_from_dict 里，
# 把这两个字段作为 Data 对象的属性：
# data.y_elem_strain = torch.from_numpy(result['y_elem_strain']).float()
# data.y_elem = torch.from_numpy(result['y_elem']).float()

# 4. 重跑 Week 26 的批量生成脚本，重新生成数据集
```

**估计补计算时间**：1-2 天（主要是测试 DG0 投影的代码，修改后重跑 Week 26 的批量脚本）。

### 为什么这个依赖关键

- **Week 29**：`y_elem` 是 batch 训练对比的目标之一
- **Week 30**：本构 Loss 约束 `σ_pred = D·ε_pred`，需要 `y_elem_strain` 作为真实应变的参考
- **Week 31**：StrainSubNet 的监督信号就是 `y_elem_strain`，StressSubNet 的就是 `y_elem`

没有这两个字段，Part F 80% 的内容都做不了。

---

**Part F 完成标准（进入 Part G 前）**：
- 能做 batch_size > 1 的多图训练（解决 block-diagonal 聚合矩阵拼接问题）
- 能在真实数据上定量对比"纯数据驱动" vs "+物理约束" 的效果
- 能完成多级网络的完整训练（StrainSubNet + StressSubNet + 联合微调）
- 能做小样本实验（10/20/30/40 个样本对比）
- 能做迁移学习实验（E=210 GPa 训练 → E=70 GPa 微调）
- 有一套完整的评估报告（误差分析 / 失败案例分析）

**时间预期**：**6-9 周，基线 6 周**。

诚实的评估：Part F 不引入新概念，但做的事情**叠起来很多**：batch 工程细节、4 组对照实验、多级网络完整版、小样本实验、迁移学习、误差分析、推理 benchmark。每一项都不离谱，但串在一起 5 周偏紧。

**三个最容易超时的地方**：
- **Week 29** 的 `follow_batch` 对自定义字段（`y_elem`）的处理——PyG 的行为有几个陷阱，不一定一次写对
- **Week 30** 的归一化域下的物理方程处理——`σ = D·ε` 在归一化后不再严格成立，调 loss 量级和权重会反复
- **Week 31** 的完整版多级网络——StrainSubNet 在 batch 里按单元切分输入的代码比想象中繁琐

如果你感觉吃力，Week 31 可以降为"达标线"（见修正 3）。**Week 29、30、32、33 是主线，不能压缩**。

**这不是一个学新东西的阶段**——是一个"精雕细琢"的阶段。很多时间会花在调参、失败、分析、重试上。如果你 Week 30 发现物理 Loss 加了反而更差，这是正常的——分析为什么，而不是怀疑代码错。

**特别说明**：Part F 的"失败结果"和"成功结果"**同样有价值**。面试里能说"我做了 X 实验，结果发现 Y 反而变差了，原因是 Z，我后来改成 W 才解决"——这比"我的模型打败了 baseline" 更能体现技术成熟度。

---

---

## Week 29: batch 训练实现与训练稳定性

**本周定位**：把 Part D Week 20 标记为"进阶"的 batch 训练真正做出来。同时解决真实数据训练时常见的稳定性问题。

**衔接 Part D 补丁**：Part D 明确说"batch_size=2+ 的多图训练涉及聚合矩阵 block-diagonal 拼接、子网络输入在 batched tensor 上的切分——这些不是容易的事，放到 Week 22 或 Part F 处理。"这周把它做完。

**本周目标**：
- 实现 block-diagonal 单元聚合矩阵的 batch 版本
- PhyFENet_WithElement 在 batch_size=4 的数据上能正常训练
- 解决真实数据上的训练稳定性问题（梯度爆炸/消失、Loss 震荡）

---

### Day 1 | block-diagonal 聚合矩阵的正确实现

**衔接 Part D Week 20 补丁中的示例**：那里给了伪代码，本周把它写完整并测试。

创建文件 `utils/batch_element_aggregation.py`：

```python
"""
支持多图 batch 的单元聚合矩阵构造。

核心问题：PyG 会自动把多个图拼成一个大图（节点索引偏移），
但 elem_list 不会自动拼，需要手动处理。
"""
import torch
from torch_geometric.data import Batch


def build_batch_aggregation_matrix(batch):
    """
    为 batched PyG data 构造 block-diagonal 聚合矩阵。
    
    参数：
        batch: PyG Batch 对象（多个 Data 对象的拼接）
    
    返回：
        C: torch.Tensor shape=(total_elems, total_nodes)
        batch_elem: torch.LongTensor shape=(total_elems,)
                    每个单元所属的图索引（类似 batch.batch 但针对单元）
    
    block-diagonal 结构说明：
        batch 里有 2 个图，图1 有 N1 节点 E1 单元，图2 有 N2 节点 E2 单元
        则 C shape=(E1+E2, N1+N2)，形式：
            C = [C_1   0  ]
                [ 0   C_2 ]
        即图 1 的单元只从图 1 的节点聚合，图 2 单元只从图 2 节点聚合
    """
    data_list = batch.to_data_list()
    
    total_elems = sum(len(d.elem_list) for d in data_list)
    total_nodes = batch.num_nodes
    
    C = torch.zeros(total_elems, total_nodes, device=batch.x.device)
    batch_elem = torch.zeros(total_elems, dtype=torch.long, device=batch.x.device)
    
    node_offset = 0    # 节点索引偏移（PyG 已经偏移过，这里手动跟踪）
    elem_offset = 0    # 单元索引偏移
    
    for graph_idx, data in enumerate(data_list):
        n_elems = len(data.elem_list)
        for local_elem_id, elem in enumerate(data.elem_list):
            global_elem_id = elem_offset + local_elem_id
            batch_elem[global_elem_id] = graph_idx
            
            # 归一化权重：每个节点贡献 1/单元节点数
            weight = 1.0 / len(elem['indices'])
            
            for local_node_id in elem['indices']:
                global_node_id = node_offset + local_node_id
                C[global_elem_id, global_node_id] = weight
        
        node_offset += data.num_nodes
        elem_offset += n_elems
    
    return C, batch_elem


# ===== 测试 =====
if __name__ == '__main__':
    from torch_geometric.data import Data, Batch
    
    # 构造两个假数据（简化版）
    data1 = Data(
        x=torch.randn(5, 3),
        edge_index=torch.tensor([[0,1],[1,0]], dtype=torch.long),
        y=torch.randn(5, 2)
    )
    data1.elem_list = [
        {'id': 0, 'type': 'Tri3', 'indices': [0, 1, 2]},
        {'id': 1, 'type': 'Tri3', 'indices': [1, 2, 3]},
    ]
    
    data2 = Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0,1],[1,0]], dtype=torch.long),
        y=torch.randn(4, 2)
    )
    data2.elem_list = [
        {'id': 0, 'type': 'Tri3', 'indices': [0, 1, 2]},
        {'id': 1, 'type': 'Tri3', 'indices': [0, 2, 3]},
    ]
    
    batch = Batch.from_data_list([data1, data2])
    C, batch_elem = build_batch_aggregation_matrix(batch)
    
    print(f"batch.num_nodes: {batch.num_nodes}")        # 5+4=9
    print(f"C shape: {C.shape}")                         # (4, 9)
    print(f"batch_elem: {batch_elem}")                   # [0, 0, 1, 1]
    
    # 验证 block-diagonal 结构
    # 图1 的单元（行 0, 1）应该只在图1 的节点（列 0-4）有非零值
    assert (C[:2, 5:] == 0).all(), "图 1 的单元不应该聚合图 2 的节点！"
    # 图2 的单元（行 2, 3）应该只在图2 的节点（列 5-8）有非零值
    assert (C[2:, :5] == 0).all(), "图 2 的单元不应该聚合图 1 的节点！"
    print("block-diagonal 结构验证通过")
```

**验收标准**：
- 测试代码通过两个 assert
- C shape 正确（总单元数 × 总节点数）
- block-diagonal 结构严格（跨图没有非零元素）

---

### Day 2 | 改造 PhyFENet_WithElement 支持 batch

**衔接 Part D Week 20**：原来的 `PhyFENet_WithElement.forward(data, elem_aggr_matrix)` 接受单图和单个聚合矩阵。现在改造成能处理 Batch。

创建文件 `week29/day02_batched_phyfenet.py`：

```python
"""
支持 batch 的 PhyFENet_WithElement。
"""
import torch
import torch.nn as nn
from utils.gnn_models import PhyFENet_Mini   # Part C Week 15
from utils.batch_element_aggregation import build_batch_aggregation_matrix


class PhyFENet_WithElement_Batched(nn.Module):
    """
    支持 batch 训练的 PhyFENet 完整版。
    forward 直接接受 PyG Batch 对象。
    """
    def __init__(self, node_in, edge_in, hid, 
                 node_out_dim, elem_out_dim, n_mp_layers=3):
        super().__init__()
        # 编码器 + 处理器（复用 Part C Week 15 的实现）
        # 这里简化：直接调用已有的 PhyFENet_Mini 作为"主干"
        self.backbone = PhyFENet_Mini(
            node_in=node_in, edge_in=edge_in, hid=hid,
            node_out=hid,   # 主干输出 hid 维的节点特征，留给两个解码器
            n_mp_layers=n_mp_layers
        )
        # 节点解码器（输出位移）
        self.node_decoder = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, node_out_dim)
        )
        # 单元解码器（输出应力）
        self.elem_decoder = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, elem_out_dim)
        )
    
    def forward(self, batch):
        """
        batch: PyG Batch 对象（或单个 Data，兼容处理）
        
        返回：
            node_out: shape=(total_nodes, node_out_dim)
            elem_out: shape=(total_elems, elem_out_dim)
        """
        # 前向：主干输出 hid 维的节点特征
        node_feat = self.backbone(batch)   # (total_nodes, hid)
        
        # 节点输出
        node_out = self.node_decoder(node_feat)
        
        # 单元聚合
        if isinstance(batch, torch.Tensor) or not hasattr(batch, 'to_data_list'):
            # 单图情况（简化处理）
            from utils.gnn_models import build_element_aggregation_matrix
            C = build_element_aggregation_matrix(
                [e['indices'] for e in batch.elem_list],
                batch.num_nodes
            )
        else:
            # batch 情况
            C, _ = build_batch_aggregation_matrix(batch)
        
        elem_feat = C @ node_feat   # (total_elems, hid)
        elem_out = self.elem_decoder(elem_feat)
        
        return node_out, elem_out


# ===== 测试 =====
if __name__ == '__main__':
    from torch_geometric.data import Batch
    import torch
    
    # 用 Part E 的真实数据集
    dataset = torch.load('data/phase2_dataset/plate_hole_dataset.pt')
    
    # 从数据集取 4 个样本组成一个 batch
    batch = Batch.from_data_list(dataset[:4])
    
    model = PhyFENet_WithElement_Batched(
        node_in=8, edge_in=3, hid=64,
        node_out_dim=2, elem_out_dim=3
    )
    
    node_out, elem_out = model(batch)
    print(f"node_out shape: {node_out.shape}")
    print(f"elem_out shape: {elem_out.shape}")
    
    # 验证 shape
    total_nodes = sum(d.num_nodes for d in dataset[:4])
    total_elems = sum(len(d.elem_list) for d in dataset[:4])
    assert node_out.shape == (total_nodes, 2)
    assert elem_out.shape == (total_elems, 3)
    print("shape 验证通过")
```

**验收标准**：
- batch=4 的样本能跑通前向
- `node_out` 和 `elem_out` 的 shape 正确（总节点/总单元 × 输出维度）

---

### Day 3 | batch 训练能跑通 + 初步稳定性

**实践任务**：把 Week 27 Day 1-2 的单图训练改成 batch 训练。

```python
# 核心改动：DataLoader 的 batch_size 从 1 改成 4
train_loader = DataLoader(train_set_norm, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set_norm, batch_size=4)

# 训练循环（跟 Week 27 基本一致）
for epoch in range(200):
    model.train()
    for batch in train_loader:
        node_pred, elem_pred = model(batch)
        loss_node = criterion(node_pred, batch.y)
        # 单元级标签需要从 batch 里正确取出
        # PyG 默认不会为 y_elem 做拼接，需要额外处理
        loss_elem = criterion(elem_pred, batch.y_elem)   # 确认 y_elem 被正确拼接
        loss = loss_node + loss_elem
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**重要工程细节**：PyG 的 Batch 默认会对 `x`、`edge_index`、`y` 字段自动拼接和偏移，**但对 `y_elem`、`elem_list` 这类自定义字段不会**。解决方式：

方式 A：在 Data 构造时用 `follow_batch=['y_elem']`（PyG 会专门处理这个字段）。

方式 B：手动迭代 `batch.to_data_list()` 拼接（更直接但每 batch 要遍历）。

推荐用方式 A，需要在 Week 25 的 `fenicsx_result_to_pyg_from_dict` 函数里**预先计算并保存 y_elem**（单元级标签，比如 von Mises 应力值）。

```python
# 在构造 Data 时
data = Data(
    x=..., edge_index=..., y=..., 
    y_elem=sigma_vm_per_element   # (n_elems,)，单元级应力
)
# DataLoader 用 follow_batch
from torch_geometric.loader import DataLoader
loader = DataLoader(dataset, batch_size=4, follow_batch=['y_elem'])
# 这样 batch 会有 batch.y_elem（拼接好）和 batch.y_elem_batch（每个值属于哪个图）
```

**验收标准**：
- batch_size=4 训练能跑通
- Loss 持续下降（不发散）
- 训练时间比 Week 27 单图版本显著缩短（因为 GPU/CPU 并行更好）

---

### Day 4 | 梯度裁剪与学习率调度

**真实数据的训练稳定性问题**（Week 27 你可能已经体会到）：
- 偶尔某个 batch 的 loss 突然跳升
- Val loss 不再下降甚至回升
- Loss 到某个水平就"卡住"不降

**两个工程常见对策**：

**对策 1：梯度裁剪**（防止梯度爆炸）

```python
# 在 loss.backward() 后、optimizer.step() 前加一行
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**对策 2：学习率调度**（后期降低学习率精细微调）

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
)

# 每个 epoch 结束用 val_loss 更新
for epoch in range(n_epochs):
    # ... train and val ...
    scheduler.step(val_loss_avg)   # 当 val_loss 10 个 epoch 不降，lr 减半
```

**任务**：在 Day 3 的 batch 训练代码基础上加上这两项，对比加前加后的训练曲线。

**验收标准**：
- 加了梯度裁剪后，训练曲线没有突然的 loss 跳升
- 加了 lr scheduler 后，训练后期的 loss 还能继续下降（没有卡在一个水平不动）
- 画一张 2x1 的图对比"无稳定措施"vs"有稳定措施"的训练曲线

---

### Day 5 | 本周收尾：完整训练脚本

整理本周成果，把 batch 训练 + 稳定性优化 + 评估指标合并成一个**"正式训练脚本"**，放在 `phase2/part_f/week29/full_train.py`。

这个脚本要：
- 能一键运行从数据加载到最终模型保存
- 有清晰的日志（每 epoch 的 train/val loss、lr）
- 保存最佳模型（根据 val loss）
- 保存训练曲线和评估报告

这是 Part F 后续所有实验的**基础训练脚本**——后面每周会在此基础上改造（加物理 Loss、多级网络等）。

---

### Day 6 | 对比 Week 27 的单图结果

**任务**：把 Week 27 的评估指标（MAE, RMSE, R²）和本周 batch 训练的指标对比。

**预期**：batch 训练在大致相同的训练时长内，应该能达到**差不多或略好的**精度。原因：
- batch 训练的梯度估计更准（多样本平均）
- 学习率调度让后期还能继续优化

**但 batch 大小不是越大越好**：
- batch=1：梯度噪声大，但泛化可能更好
- batch=4/8：折中
- batch=过大：内存吃紧、梯度过于平均可能收敛到平坦极小点

**写一段 200 字的对比分析**记录在 `week29/comparison_note.md`。

---

### Week 29 完成标准

- [ ] `build_batch_aggregation_matrix` 函数放在 `utils/`，能正确构造 block-diagonal 聚合矩阵
- [ ] `PhyFENet_WithElement_Batched` 能在 batch_size=4 下训练
- [ ] 训练稳定性措施（梯度裁剪、lr 调度）集成到训练脚本
- [ ] `phase2/part_f/week29/full_train.py` 作为后续所有实验的基础脚本
- [ ] 对比 Week 27 的单图训练结果，有文字记录

---

---

## Week 30: 物理约束 Loss 在真实数据上的定量评估

**本周定位**：回答核心问题——**几何一致性 Loss 和本构 Loss 在真实 FEniCS 数据上到底有没有用？** Part D 只在合成数据上跑过机制验证；本周在真实数据上做**定量对比**。

**重要前置说明**：
本周的实验结果**有两种可能**：
1. 加物理约束后 MAE 明显下降 → 论文思想在真实数据上验证成功
2. 加物理约束后 MAE 持平或略差 → 物理约束在**当前设置**下没带来增量

**两种结果都有分析价值**。不要为了"实验符合预期"调参数调到自己都不相信的结果。面试里能说"我做了 X 实验，结果是 Y，分析原因是 Z"——诚实的失败比造出来的成功更有说服力。

---

### Day 1 | 设计对比实验

**实验矩阵**：

| 配置 | 数据 Loss | 几何一致性 Loss | 本构 Loss | 说明 |
|------|----------|---------------|-----------|------|
| A（baseline） | ✅ | ❌ | ❌ | 纯数据驱动 |
| B（+几何） | ✅ | ✅ | ❌ | 加几何一致性 |
| C（+本构） | ✅ | ❌ | ✅ | 加本构（不带几何一致性）|
| D（全部） | ✅ | ✅ | ✅ | 所有约束 |

**每个配置做什么**：
- 用相同的 train/val split（同一个随机种子，确保公平对比）
- 用相同的超参数（hidden=64，lr=1e-3，200 epoch）
- 4 次实验各用 3 个随机种子跑（42, 123, 2024），取平均——避免偶然性

**记录指标**：
- 训练完最终的 train/val MAE（节点位移）
- 单元应力的 MAE
- **几何一致性残差**：`|ε_pred - StrainSubNet(u_pred)|` 的均值
- **本构残差**：`|σ_pred - D·ε_pred|` 的均值
- R² 分数

---

### Day 2–3 | 在真实数据上实现物理约束 Loss

**衔接 Part D Week 20**：那里已经实现了 `geometric_consistency_loss` 和 `constitutive_loss`。但有个工程问题——Part D 用的是合成数据，Loss 直接能用；真实数据有**量级差异**（位移量级可能是 1e-3 mm，应力量级是 100 MPa），需要调整。

**量级处理的关键**：Loss 项之间的相对量级要在合理范围，不然权重几乎等于零的那项约束失效。

```python
# 简化的实现（针对归一化后的数据）
def combined_loss(
    node_pred, elem_stress_pred, elem_strain_pred,
    batch, strain_subnet, 
    E_normalized, nu,        # 归一化后的材料参数
    w_data_node=1.0, w_data_elem=1.0,
    w_geom=0.1, w_const=0.1,
    mode='all'               # 'data_only' / '+geom' / '+const' / 'all'
):
    """
    组合 Loss，可根据 mode 开关各项。
    """
    # 数据 Loss
    L_data_node = torch.nn.functional.mse_loss(node_pred, batch.y)
    L_data_elem = torch.nn.functional.mse_loss(elem_stress_pred, batch.y_elem)
    
    # 几何一致性 Loss
    L_geom = torch.tensor(0.0, device=node_pred.device)
    if mode in ('+geom', 'all'):
        # 子网络从位移估计应变
        strain_from_disp = strain_subnet.forward_batched(
            node_disp=node_pred, 
            node_coord=batch.x[:, :2],
            batch=batch
        )
        L_geom = torch.nn.functional.mse_loss(
            elem_strain_pred, strain_from_disp
        )
    
    # 本构 Loss
    L_const = torch.tensor(0.0, device=node_pred.device)
    if mode in ('+const', 'all'):
        from week17.day04_2d_elasticity import strain_to_stress_plane_stress
        # 注意：用归一化后的 E，因为 strain、stress 都是归一化域的值
        # 这里 E_normalized 需要对应处理
        stress_from_strain = strain_to_stress_plane_stress(
            elem_strain_pred, E_normalized, nu
        )
        L_const = torch.nn.functional.mse_loss(
            elem_stress_pred, stress_from_strain
        )
    
    total = w_data_node * L_data_node + w_data_elem * L_data_elem + \
            w_geom * L_geom + w_const * L_const
    
    return total, {
        'data_node': L_data_node.item(), 'data_elem': L_data_elem.item(),
        'geom': L_geom.item(), 'const': L_const.item()
    }
```

**关于归一化下的本构方程**：这是本周最容易踩坑的地方。

严格说：如果你用 Z-Score 归一化了 σ 和 ε，那么"σ = D·ε"在归一化域内**不再严格成立**（会差一个常数项和缩放因子）。两种处理方式：

**方式 1**：在反归一化后的真实物理量上计算本构 Loss（更严谨）
```python
elem_strain_real = normalizer.inverse_strain(elem_strain_pred)
elem_stress_real = normalizer.inverse_stress(elem_stress_pred)
stress_from_strain = strain_to_stress_plane_stress(
    elem_strain_real, E_real, nu
)
L_const = mse(elem_stress_real, stress_from_strain)
```

**方式 2**：把本构约束弱化为"应变和应力的比值关系要稳定"（更粗糙但更简单）

**推荐方式 1**。记得在反归一化时用 `normalizer` 的 `inverse_transform` 方法。

---

### Day 4 | 运行四组对比实验

把 Week 29 的基础训练脚本复制 4 份（或者用一个脚本加 `--config` 参数），分别跑四种配置 × 三个种子 = 12 次训练。

**每次训练**：~15-30 分钟（取决于硬件）。
**本节总计**：~3-6 小时挂机跑。

**记录**：每个配置三次的 (MAE, RMSE, R², 物理残差) 取均值和标准差。

创建表格 `week30/results_comparison.md`：

```markdown
| 配置 | MAE (位移) | 物理残差 (几何) | 物理残差 (本构) | R² |
|------|----------|----------------|----------------|----|
| A (baseline)  | 0.XXX ± 0.0XX | - | - | 0.XX |
| B (+geom)     | 0.XXX ± 0.0XX | 0.XXX ± 0.0XX | - | 0.XX |
| C (+const)    | 0.XXX ± 0.0XX | - | 0.XXX ± 0.0XX | 0.XX |
| D (all)       | 0.XXX ± 0.0XX | 0.XXX ± 0.0XX | 0.XXX ± 0.0XX | 0.XX |
```

---

### Day 5 | 结果分析

**四种典型结果模式**（你遇到哪种都正常）：

**模式 1**：D > B > C > A（最理想）
物理约束都有用。写分析："物理约束提供了额外的监督信号，在有限训练样本下约束模型向物理一致的解收敛。"

**模式 2**：B > A，但 C 或 D 略差
几何一致性起作用，本构约束可能因为归一化的复杂性导致效果不佳。写分析："几何一致性约束捕捉了位移-应变的梯度关系，对模型有帮助；本构约束在归一化后的量级可能有问题，需要进一步调权重。"

**模式 3**：所有配置结果接近
可能原因：(1) 50 个样本对这个问题太多了（物理约束的边际价值降低）；(2) 归一化本身已经让数据很容易学。写分析："在当前数据量下物理约束的增益不明显，建议做小样本实验（Week 32）看约束在数据少时是否有优势。"

**模式 4**：加约束后更差
可能原因：(1) 权重配置不对（物理 Loss 压过了数据 Loss）；(2) 实现有 bug。**先排查 bug**：
- 检查几何一致性残差是否和 StrainSubNet 的预训练质量有关
- 检查本构方程在归一化域的正确性

---

### Day 6 | 写本周实验报告

创建 `week30/experiment_report.md`：

```markdown
# Week 30 实验报告：物理约束 Loss 在真实 FEM 数据上的效果

## 实验设置
- 数据：Part E Week 26 生成的 50 个 FEniCS 带孔板样本
- Split：40 训练 / 10 验证
- 模型：PhyFENet_WithElement_Batched (hidden=64, 3 层 MP)
- 训练：batch_size=4, lr=1e-3, 200 epoch + ReduceLROnPlateau

## 结果
[粘贴表格]

## 分析
[根据四种模式之一写]

## 洞察
[1-2 条你真实的观察]

## 下一步
- 小样本实验（Week 32）：看看训练数据量小时物理约束的价值
- 多级网络（Week 31）：用预训练子网络替代联合训练
```

**面试用**：这份报告直接可以作为面试讲项目的素材。

---

### Week 30 完成标准

- [ ] 实现了支持开关各项约束的 `combined_loss` 函数
- [ ] 完成了 4 组 × 3 种子 = 12 次对比训练
- [ ] 有定量结果表格
- [ ] 有书面分析报告
- [ ] 能在面试中讲清楚"我对比了四种配置，发现 X，原因是 Y"

---

---

> **主线里程碑（Week 30 结束时）**：
> 
> 到 Week 30 结束，你已经完成了 Part F 最核心的内容：
> - batch 训练在真实数据上跑通（Week 29）
> - 物理约束 Loss 的定量对比（Week 30）
> 
> **这已经是一个可以讲的项目了**。即使 Week 31 只做到达标线，你也有足够的面试素材：
> - Week 29 的工程实现（batch 聚合矩阵）
> - Week 30 的 4 组对照实验
> - Week 32 的小样本 + 迁移学习
> - Week 33 的误差分析 + benchmark
> 
> 不要在 Week 31 上用力过猛导致后面没精力了。

---

---

## Week 31: 多级网络完整训练

> **本周分层要求说明**：
> 
> 多级网络是论文 §2.3.5 的核心思想，必须理解和部分实现。但**完整版**（两个子网络 + 联合微调 + 四策略对比）是研究级工作量。
> 
> **本周分两个层级**：
> 
> **达标线（必做）**：
> - 展开单元级数据集
> - 预训练 StrainSubNet
> - 用预训练的 StrainSubNet + 解析胡克定律（不训练 StressSubNet）替代主网络的单元解码器，做一次联合微调
> - 对比"从零训练"和"StrainSubNet 预训练 + 联合微调"两种策略
> 
> **进阶线（强烈推荐但可选）**：
> - 预训练 StressSubNet
> - 四种策略完整对比（S1/S2/S3/S4）
> - 深入调整 StrainSubNet 的 batch 处理细节
> 
> **为什么达标线用解析胡克定律代替 StressSubNet**：
> StressSubNet 学的就是胡克定律（解析公式），用解析函数更准确、实现更简单，**不降低多级网络的核心价值**——核心价值是"StrainSubNet 作为预训练子网络约束主网络"，这部分保留了。
> 
> **如果 Week 29-30 你已经很累了**：做到达标线就合格。Week 32 的小样本实验才是 Part F 最有面试说服力的部分，别在 Week 31 用力过猛耗尽精力。

---

### Day 1–2 | 展开单元级数据集 + 预训练 StrainSubNet（达标线）

任务保持原文 Week 31 Day 1-2 的内容。

**达标验收**：
- 单元级数据集展开成功，样本数 ≈ 图数 × 平均单元数
- StrainSubNet 能收敛到低 MAE（< 5e-4，因为真实 FEM 数据完全自洽）
- 验证集上的 R² > 0.95
- 权重保存到 `week31_strain_net.pt`

---

### Day 3（进阶）| 预训练 StressSubNet

> **本节是进阶内容**。如果时间紧张，可以跳过，**Day 4 的达标版多级网络不需要 StressSubNet**。
> 
> **什么情况下值得做 StressSubNet**：
> - 想做 Day 5 的四策略完整对比
> - 未来要扩展到非线性本构（比如弹塑性，真实冲压成形会用到）—— StressSubNet 是必须的，因为塑性没有解析公式
> 
> 如果做，任务保持原文 Day 3 内容。

```python
class StressSubNet(nn.Module):
    """输入应变 + 材料参数，输出应力"""
    def __init__(self, strain_dim=3, mat_dim=2, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(strain_dim + mat_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 3)
        )
    def forward(self, strain_and_mat):
        return self.net(strain_and_mat)

# 训练（同 Day 2 流程）
```

---

### Day 4 | 多级网络联合训练

**达标线版本**（单 StrainSubNet + 解析应力公式）：

```python
class PhyFENet_Multilevel_Minimal(nn.Module):
    """
    达标线版本：主 GNN + 预训练的 StrainSubNet + 解析胡克定律
    """
    def __init__(self, main_net, strain_net, E_for_constitutive, nu):
        super().__init__()
        self.main = main_net          # 只用其节点输出（位移）
        self.strain_net = strain_net  # 预训练好的，可冻结或放开微调
        # 解析胡克定律不需要 module，直接用函数
        self.E = E_for_constitutive
        self.nu = nu

    def forward(self, batch):
        # 主网络输出位移（只用 node_decoder 部分）
        node_disp = self.main.forward_node_only(batch)   # (N, 2)

        # 子网络从位移估计应变
        strain_pred = self.strain_net.forward_from_batch(
            node_disp, batch.x[:, :2], batch
        )   # (total_elems, 3)

        # 解析胡克定律算应力（反归一化后算，记得反归一化回物理量）
        from week17.day04_2d_elasticity import strain_to_stress_plane_stress
        stress_pred = strain_to_stress_plane_stress(
            strain_pred, self.E, self.nu
        )

        return node_disp, strain_pred, stress_pred
```

**达标验收**：
- 能在真实数据上跑通联合微调
- 对比"从零训练 PhyFENet"（Week 29 的 baseline）和"StrainSubNet 预训练 + 联合微调"的收敛速度
- 记录前 50 epoch 两者的 loss，做对比图
- 通常 StrainSubNet 预训练版收敛更快（子网络已经学会了局部关系）

**进阶版本**（完整多级网络 = 主网络 + StrainSubNet + StressSubNet）：
保持原文 Day 4 的 `PhyFENet_Multilevel_Full` 实现。

---

### Day 5（进阶）| 四种策略完整对比

> **本节全部是进阶内容**。Week 31 的达标线到 Day 4 为止（两种策略对比）。本节如果做不完，不影响 Week 32-33。

原文 Day 5 的四策略表格（S1/S2/S3/S4）保持不变，但标注为进阶。

---

### Day 6 | 本周总结

保持原文 Day 6 内容，但根据你实际完成到哪一步写总结（达标线版 or 进阶版）。

---

### Week 31 完成标准

**达标线（必做）**：
- [ ] 展开 GNN 数据集到单元级数据集的工具
- [ ] StrainSubNet 在真实数据上预训练完成，R² > 0.95
- [ ] `PhyFENet_Multilevel_Minimal` 类（主网络 + StrainSubNet + 解析应力）支持 batch，能跑联合微调
- [ ] 对比"从零训练"和"StrainSubNet 预训练 + 联合微调"两种策略的收敛速度，有报告

**进阶线（强烈推荐但可选）**：
- [ ] 独立预训练 StressSubNet
- [ ] `PhyFENet_Multilevel_Full` 四种策略完整对比（S1/S2/S3/S4）
- [ ] 深入调整 batch 处理细节（StrainSubNet 按单元切分的正确性验证）

---

---

## Week 32: 小样本实验与迁移学习

**本周定位**：论文 §3.2.3 的核心实验——物理约束在**数据稀缺时**价值最大。本周验证这一点，并顺便做一个迁移学习实验。

---

### Day 1–2 | 小样本对比实验

**实验设计**：
- 分别用 10、20、30、40 个训练样本
- 两种策略：纯数据 vs 数据+物理（Week 30 的 D 配置 或 Week 31 的多级网络）
- 每组 3 个种子取平均
- Val set 固定（不变），避免评估偏差

**关键指标**：Val MAE 随训练样本数的变化曲线。

**预期结果**：
- 小样本时（10 个），物理约束版应**显著优于**纯数据版
- 大样本时（40 个），差距缩小

这个曲线是**面试极强的素材**——"物理约束的价值在数据少时最明显"是论文的核心论点，你能用自己的实验图直接说明。

---

### Day 3–4 | 迁移学习实验

**实验设计**：
- 源任务：当前数据集（E=210 GPa 附近）
- 目标任务：不同材料（比如 E=70 GPa 附近，铝合金）
- 需要**重新生成** 10-20 个新 E 范围的样本（回到 Part E 的数据生成流程）

**三种策略对比**：
1. 从零训练目标任务（只用 10-20 个新样本）
2. 直接用源任务的模型预测目标任务（测泛化能力）
3. 源任务预训练 → 目标任务微调（迁移学习）

**Day 3**：生成目标任务数据集（~1-2 小时跑 FEniCS）
**Day 4**：三组对比实验

---

### Day 5 | 结果分析与可视化

两个关键图：
- 图 1：小样本实验曲线（x 轴训练样本数，y 轴 val MAE，两条线对比）
- 图 2：迁移学习柱状图（三个策略的最终 val MAE）

这两张图要做得**面试可用级别**（高 DPI、清晰图例、轴标签完整）。

---

### Day 6 | 周总结

报告模板：
- 小样本实验：x 个样本下，物理约束降低 MAE XX%
- 迁移学习：微调策略 vs 从零 MAE 降低 XX%
- 两个实验的业务意义：在真实工程中数据少、材料多变，这套方法能直接用

---

### Week 32 完成标准

- [ ] 小样本对比实验完成，有曲线图
- [ ] 迁移学习数据集生成完成（10-20 个新样本）
- [ ] 迁移学习三组对比完成
- [ ] 有面试级的两张实验图

---

---

## Week 33: Part F 收尾 + 完整评估报告

**本周定位**：把 Week 29-32 的所有实验结果整合成一个**完整的 Part F 评估报告**，为 Part G 的完整项目交付做准备。

---

### Day 1–2 | 系统化误差分析

**不是**简单地"MAE = 0.XX"——要能指出：
- 误差最大的是哪类样本？（比如孔半径大的？载荷极端的？）
- 误差在空间上的分布？（孔边缘？远处？）
- 误差和模型输出的物理合理性关系？

创建文件 `week33/day12_error_analysis.py`：

```python
"""
系统化误差分析
"""
import torch
import numpy as np

# 对每个 val 样本，计算 MAE 和样本参数
errors_per_sample = []
for data in val_set:
    y_pred = model(data)
    mae = torch.abs(y_pred - data.y).mean().item()
    errors_per_sample.append({
        'mae': mae,
        'E': data.E, 'T_x': data.T_x, 'r': data.r,
        'n_nodes': data.num_nodes
    })

# 分析 1：误差和参数的相关性
import pandas as pd
df = pd.DataFrame(errors_per_sample)
print("MAE 和参数的相关性：")
print(df.corr()['mae'].sort_values(ascending=False))

# 分析 2：误差最大的 5 个样本
df.sort_values('mae', ascending=False).head(5)
```

**写 400-600 字的分析**：哪类样本最难预测？为什么？模型的局限是什么？

---

### Day 3 | 物理一致性检查

**不只看 MAE**——PhyFENet 的卖点是"物理一致"，也要量化这个。

**定义物理一致性指标**：
- 几何残差：`|ε_pred - B·u_pred|`（用数值微分近似 B）
- 本构残差：`|σ_pred - D·ε_pred|`
- 边界条件违反：固定边界上 `|u_pred|` 的平均值

对比纯数据版和物理约束版在这些指标上的差距。

**预期**：即使 MAE 接近，物理约束版的物理一致性指标也应该更好。这是"加物理 Loss 的价值"的直接证据。

---

### Day 4 | 推理效率 benchmark

**面试高频问题**：你的 GNN 比 FEM 快多少？

**Benchmark**：
- 对同一个测试样本，分别记录 FEniCS 求解时间 vs PhyFENet 推理时间
- 重复 100 次取均值
- 报告加速比（speedup factor）

```python
import time

# FEniCS 时间
times_fem = []
for _ in range(10):
    t0 = time.time()
    result = solve_plate_hole(L=2.0, H=1.0, r=0.2, E=210e3, T_x=50.0)
    times_fem.append(time.time() - t0)
print(f"FEM: {np.mean(times_fem):.2f} ± {np.std(times_fem):.2f} 秒")

# PhyFENet 时间（加载预先生成的数据）
test_data = val_set[0]
times_ai = []
for _ in range(100):
    t0 = time.time()
    with torch.no_grad():
        y_pred = model(test_data)
    times_ai.append(time.time() - t0)
print(f"PhyFENet: {np.mean(times_ai)*1000:.2f} ± {np.std(times_ai)*1000:.2f} 毫秒")

speedup = np.mean(times_fem) / np.mean(times_ai)
print(f"加速比：{speedup:.1f} 倍")
```

**典型结果**：FEM ~3-5 秒，PhyFENet ~5-10 毫秒，加速比 ~500-1000x。

**注意**：
- 加速比的前提是"模型已训练好"。训练本身时间是固定成本
- 面试时要说清楚这个边界：加速比 = 训练好之后单次推理 vs FEM 单次求解，不包括训练时间
- 对"一次训练 → 多次推理"的参数扫描场景（设计迭代）这个加速比才有意义

---

### Day 5 | 可视化总汇

做一套"面试展示级"的可视化：

**图 1**：物理场预测对比
- 2×3 子图：FEM 真实 vs PhyFENet 预测 vs 误差（分别画 ux、uy、σ_vm）

**图 2**：训练曲线（显示多个配置对比）

**图 3**：小样本实验曲线（Week 32 已有）

**图 4**：迁移学习柱状图（Week 32 已有）

**图 5**：误差分布（直方图 + 逐样本）

**图 6**：推理时间对比柱状图（FEM vs PhyFENet）

所有图要：
- DPI ≥ 150
- 字体清晰可读
- 颜色方案一致
- 坐标轴、图例、标题齐全

这些图是 Part G 最终项目 README 和博客的直接素材。

---

### Day 6 | Part F 总报告

创建文件 `phase2/part_f/FINAL_REPORT.md`。这份报告要能**直接拿给面试官看**。

内容框架：

```markdown
# Part F: PhyFENet 在真实 FEM 数据上的训练与调优

## 数据
- 源：FEniCS 生成的 50 个 2D 带孔板弹性问题
- 参数：E ∈ [150, 250] GPa, T_x ∈ [20, 80] MPa, r ∈ [0.1, 0.3]
- Split：40 train / 10 val

## 核心结果表
| 指标 | 值 |
|------|----|
| 节点位移 MAE | X mm |
| Von Mises 应力 MAE | X MPa |
| R² | X |
| PhyFENet 推理速度 | X ms/样本 |
| FEM 求解时间 | X s/样本 |
| 加速比 | X 倍 |

## 核心发现
1. [物理约束在真实数据上是否有增益？具体数字]
2. [多级网络相对于从零训练的收敛速度优势]
3. [小样本下物理约束的价值（Week 32）]
4. [迁移学习效果]

## 工程实现亮点
1. block-diagonal 单元聚合矩阵（支持 batch 训练）
2. 归一化域下的物理约束 Loss 处理
3. 多级网络的 batch 兼容实现

## 局限和未来工作
1. 当前数据集仅线弹性问题；Part G 会扩展到冲压成形
2. 只有 2D 问题；3D 扩展是未来工作
3. 仅开源 FEniCS 数据；工业数据（LS-DYNA）的对接需要在企业环境
```

---

### Week 33 完成标准 + Part F 总完成标准

**Week 33**：
- [ ] 系统化误差分析报告
- [ ] 物理一致性指标对比
- [ ] 推理效率 benchmark
- [ ] 一套展示级可视化图
- [ ] Part F 的 FINAL_REPORT.md

**Part F 总完成标准**（进入 Part G 前）：

**技术成果**：
- [ ] batch 训练在真实数据上跑通
- [ ] 完整的物理约束 Loss 对比实验（4 组）
- [ ] 完整的多级网络训练流程
- [ ] 小样本实验 + 迁移学习实验
- [ ] 误差分析 + 物理一致性评估 + 推理效率 benchmark

**工程成果**：
- [ ] `utils/` 下的工具（batch 聚合、多级网络、combined_loss 等）
- [ ] 一套展示级可视化图
- [ ] 完整的 FINAL_REPORT.md 可直接用于面试

**准备进入 Part G 的状态**：
你现在应该有一个**基本完整的项目**——真实数据、完整的 PhyFENet 训练体系、物理约束验证、小样本+迁移学习实验、评估报告。Part G 不是"再做一个项目"——而是把这套能力**应用到冲压成形场景**（更贴近你的论文第四章），并做最后的项目包装、博客、简历对接。

---

*下一段输出：第二阶段 Part G（Week 34-39）：冲压成形完整项目*