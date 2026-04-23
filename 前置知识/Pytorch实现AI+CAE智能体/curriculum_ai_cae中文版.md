# AI for Engineering Simulation 学习课程（完整中文大纲）

> **这是一份配套文档，不是 system prompt。**
>
> 使用方式：当学生触发 **Protocol 4**（明确询问学习路径 / 下一步该学什么）时，
> 将本文件中的相关章节作为参考载入对话。
>
> 在日常对话中，system prompt 本身**不需要**加载这份完整课程；
> 过度加载会让模型在回答简单问题时也进入“stage brain”状态。
>
> 课程原则：
>
> * **Phase 1（Stage 0–6）路线中立**，所有学生都学习
> * **Stage 7** 正式做路线决策
> * **Phase 2（Stage 8–12）** 按所选路线深入
> * **Stage 13** 作品集与面试准备

---

## 课程参考说明（仅当学生明确请求学习路径时启用）

课程分为两个阶段：

### **Phase 1 — 路线中立基础阶段（Stage 0 到 Stage 6）**

无论学生最终选择哪条技术路线，所有人都先学习同样的基础内容。
这些阶段构成了 AI for Engineering Simulation 各条路线都需要的共同底座。

### **Phase 2 — 路线专项深化阶段（Stage 7 到 Stage 12）**

在 Stage 7，学生将做出有依据的路线选择。
Stage 8 之后的内容将围绕所选路线展开。

---

# PHASE 1：路线中立基础阶段

---

## Stage Pre — 基础准备（通常视为已完成）

**前置要求：** 无
**目标：** 把“会看教程”转变成“能独立写代码”。

### 主题

* Python 类与模块
* NumPy 数组与 shape 操作
* 向量化与 broadcasting
* Matplotlib 用于 loss 曲线和场场可视化
* 导数与梯度的几何直觉
* 作为 Python 工程练习的基础 FEM 文件解析
* Git 基础

### 完成信号

学生能够：

* 独立写一个函数解析简单的 FEM 风格文本文件
* 将节点坐标转成 NumPy 数组
* 生成清晰的可视化图
* 不运行代码，也能解释某个具体 shape 组合下 broadcasting 会发生什么

---

## Stage 0 — PyTorch 机制入门

**前置要求：** Stage Pre
**目标：** 理解什么是 Tensor，以及 PyTorch 如何追踪计算。

### 主题

* Tensor 与 NumPy array 的区别
* Tensor 运算与 shape 操作
* 计算图（computational graph）
* autograd 与 `.backward()`
* 仅用原始 tensor 运算手写一个单神经元，再手写一个小型 MLP
* **此阶段不使用 `nn.Module`**

### 完成信号

学生能够：

* 为一个小型 MLP 写出 forward pass 和手动权重更新
* 在运行代码前预测输出 shape
* 解释计算图何时建立、何时释放

---

## Stage 1 — 深度学习基础 + 最小训练循环

**前置要求：** Stage 0
**目标：** 从零写出一个完整训练循环，并理解每一行代码在做什么、为什么要这么写。

### 主题

* 损失函数
* 梯度下降机制
* 过拟合与欠拟合
* 正则化
* MLP 中反向传播的基本机制
* 基于 `nn.Module` 的训练循环
* 优化器：先 SGD，再 Adam
* 使用 `TensorDataset` 与 `DataLoader`

### 项目

做一个多特征回归任务，数据是一个类似工程映射的合成数据集，
例如：根据力和几何参数预测位移。

学生必须独立绘制：

* train loss
* val loss
* 预测效果图

### 完成信号

学生能够在**不看笔记**的情况下回答：

* 为什么会过拟合？
* 为什么 val loss 往往高于 train loss？
* Adam 与 SGD 的核心区别是什么？
* 保存 checkpoint 时到底保存了什么？

---

## Stage 2 — 训练纪律：优化、调参、误差分析

**前置要求：** Stage 1
**目标：** 能够看一眼 loss 曲线，就知道大概哪里出了问题。

### 主题

* 诊断 loss 曲线：爆炸、平台期、震荡、发散
* Adam
* 学习率调度（learning rate schedules）
* 回归任务中的归一化（例如 Z-score）
* 回归问题的基础误差分析：模型在哪些区域失败？
* 受控实验设计（controlled experiment design）

### 项目

升级 Stage 1 的回归器：

* 加入特征归一化
* 比较不同优化器
* 比较不同学习率调度
* 写一份控制变量明确的实验报告

### 交付物

* `baseline`
* `adam_vs_sgd`
* `normalization_ablation`
* `error_analysis.md`

### 完成信号

学生应该达到这个状态：

> “我已经训练过一个模型，而且当训练出问题时，我知道怎么诊断。”

学生应能面对一条陌生的 loss 曲线，提出三个具体假设。

---

## Stage 3 — Autograd 精通与物理损失

**前置要求：** Stage 2
**目标：** 深入理解 autograd，达到可以正确实现基于 PDE 残差的 physics-informed loss。

### 为什么这一阶段在 GNN 之前？

因为后续所有路线——PINN、带物理损失的 GNN 模拟器、带物理约束的 Neural Operator——都依赖一个共同能力：

> 计算网络输出对输入的导数。

这是共享的基础设施能力，而不是某条路线的专属技巧。

### 主题

* `torch.autograd.grad` 与 `.backward()` 的区别
* 使用 `create_graph=True` 计算高阶导数
* 多输入多输出函数的偏导数
* 把 PDE 翻译成 residual loss
* 边界条件损失
* 多项物理损失的权重策略

### 项目

实现一个 **1D 弹性杆 PINN**：

给定边界条件，预测位移场 (u(x))，使其满足平衡方程：

[
E \cdot \frac{d^2 u}{dx^2} + f = 0
]

并与解析解进行验证。

### 完成信号

学生能够：

* 不依赖参考代码，为一个新的简单 PDE（例如 1D 热传导方程）写出 physics loss
* 清楚解释为什么二阶导数场景下必须使用 `create_graph=True`

---

## Stage 4 — 图神经网络机制

**前置要求：** Stage 2（Stage 3 有帮助，但不是强制）
**目标：** 从机制层面理解 message passing，而不是把它当作一个黑盒 API。

### 为什么它在 mesh 模拟器之前？

因为基于 mesh 的神经模拟器，本质上建立在 GNN 的 message passing 机制之上。
学生必须先在“齿轮级别”理解 message passing，才能进一步理解模拟器架构。

### 主题

* 图数据表示：节点、边、特征
* PyTorch Geometric 的 `Data` 对象
* GCN 推导：归一化邻接矩阵、图拉普拉斯
* GraphSAGE：sample-and-aggregate，归纳式 vs 传导式学习
* `MessagePassing` 基类
* over-smoothing 及其成因
* GNN 中的残差连接

### 项目

从零用 `MessagePassing` 实现一个小型 GNN，
并在一个合成图回归任务上训练它，例如：根据邻域结构预测节点属性。

比较：

* GCN
* GraphSAGE

### 完成信号

学生能够：

* 在 PyG 中实现自定义的 `message` 和 `aggregate`
* 解释为什么 GraphSAGE 支持归纳学习，而 GCN 不支持
* 在运行前预测多层 GNN 的输出 shape

---

## Stage 5 — 面向机器学习的 FEM 数据处理

**前置要求：** Stage 4（也可与 Stage 3 并行学习）
**目标：** 把真实 FEM 仿真数据转换成可供任意神经模型使用的形式。

这些模型可能包括：

* GNN
* 按节点建模的 MLP
* pointwise 的 operator 模型

### 主题

* FEM 数据的构成：节点、单元、连接关系、材料、边界条件、结果场
* 常见文件格式的概念性解析：`.inp` / `.k` / `.msh`（通过 meshio 等工具）
* 处理混合单元类型：triangle、quad、tet、hex
* mesh-to-graph 转换：以节点为顶点，以单元关系构边
* 构建节点级与单元级特征
* 物理场的归一化策略
* 使用 PyG 的 `Batch` 与 block-diagonal aggregation 进行图 batching

### 引入工具

* PyTorch Geometric（更深入）
* meshio（格式转换）
* Gmsh（生成简单测试网格）

### 项目

给定一组合成 FEM 风格数据（例如：简单 2D 几何、节点位移、单元应力），
实现：

* 一个可复用的 `mesh_to_pyg` 函数
* 一个 `Dataset` 类

输出应当是：

* 可被下游模型直接使用的 batched graph sample

### 完成信号

学生能够：

* 处理混合单元类型的 mesh
* 在**不参考代码**的情况下生成正确 batched 的 PyG 数据集
* 解释如果 batching 时没有正确处理 element-to-node 索引偏移，会出什么问题

---

## Stage 6 — 第一个真正的神经模拟器：MeshGraphNets 风格基线

**前置要求：** Stage 3、4、5
**目标：** 端到端构建一个完整的 mesh-based neural simulator，
以 MeshGraphNets 架构为标准参考，建立后续路线比较的共同基线。

### 为什么选 MeshGraphNets 作为 Stage 6 的参考？

因为它：

* 是最广为引用的 mesh-based neural simulator 之一
* 有 DeepMind 官方 JAX 实现
* 也有多个社区 PyTorch 移植版本
* 它的 encoder-processor-decoder 架构已经成为 PhyFENet、Transolver 等许多后续方法的设计模板

学懂它，后续很多方法都会变成“在同一模板上的变化”。

### 主题

* encoder-processor-decoder 架构（论文线）
* edge-update message passing（论文 + 代码线）
* 使用 rollout 训练 vs 单步预测训练
* 神经模拟器评估方式：单步误差 vs 轨迹误差
* 用 FEniCS / FEniCSx 生成训练数据

### 教学风格

从这里开始，**Style 4（李沐风格）** 激活：

* 论文动机
* 代码实现
* 实验结论 / ablation

三条线并行讲解。

### 项目

在一个 **2D 弹性问题** 上实现最小版 MeshGraphNets，要求包含：

* 数据生成 pipeline
* 训练循环
* 在未见过的 mesh 上做测试
* 预测场与真实场的可视化对比

### 完成信号

学生能够：

* 解释 encoder-processor-decoder 架构中每一行的作用
* 诊断典型失败模式（例如 rollout 中误差累积）
* 说清楚 MeshGraphNets 的假设条件在哪里可能失效

---

# PHASE 2：路线专项深化阶段

---

## Stage 7 — 路线调研与知情决策（关键决策点）

**前置要求：** 完成 Stage 0 到 Stage 6
**目标：** 在投入未来 4 个月以上深入学习之前，做出**有依据、经过比较的技术路线选择**。

这不是一个普通“学习阶段”，而是一个结构化的决策过程。

### Stage 7 有两种执行方式

---

### Stage 7-Structured（默认选项，适合尚无明显偏好的学生）

这是一个 2–3 周的小型流程：

#### 第 1–3 天

阅读 1 篇关于 neural PDE solver / neural simulator 的近期综述，
覆盖至少：

* PINN
* GNN 模拟器
* Neural Operator

并写一页比较表。

#### 第 4–10 天

在同一个 toy problem 上，最小复现两种竞争方法。
例如：

* MeshGraphNets + FNO
* PINN + MeshGraphNets
* PhyFENet + FNO

问题可以选：

* 2D Poisson
* 2D elasticity

#### 第 11–12 天

收集 20–30 个当前岗位招聘信息，
统计里面出现的技术关键词。

#### 第 13–14 天

如果可能，咨询 2–3 位领域从业者（学术 + 工业都可），
最后写出正式路线决策说明。

---

### Stage 7-Direct（适合已经有明确偏好的学生）

学生直接跳到自己偏好的路线，
但仍然必须写一页理由说明：

* 你考虑过哪些路线？
* 你放弃了哪些路线？
* 为什么最终选这条？

---

### Stage 7 可能导向的路线

#### Route A — PINN 深化路线

* 自适应采样
* variational PINN
* conservative PINN
* 域分解

#### Route B — 基于 GNN 的模拟器路线

* MeshGraphNets 扩展
* PhyFENet 风格的物理嵌入
* 多尺度 GNN
* 时序 rollout 策略

#### Route C — Neural Operator 路线

* FNO 及其变体
* DeepONet
* Geo-FNO（复杂几何）
* physics-informed neural operators

#### Route D — Transformer-based 模拟器路线

* Transolver
* GNOT
* 非结构化网格上的 attention 机制

#### Route E — 混合路线

学生自己定义混合方案，例如：

* GNN + physics loss（类似 PhyFENet 的组合）

---

### 完成信号

学生能够用一段话说清楚：

* 为什么选择这条路线
* 放弃其他路线意味着什么
* 这个选择对应的岗位或研究方向是什么

---

## Stage 8 — 路线专项架构深潜

**前置要求：** Stage 7 已完成路线选择
**目标：** 在实现层面掌握所选路线的核心架构，
达到可以从零写出、复现关键结果、并识别其失败模式的程度。

### 内容

根据路线而定。

例如：

#### 如果选择 Route C（Neural Operator）

需要深入：

* FNO 的 Fourier layer 机制
* spectral convolution
* resolution invariance
* Geo-FNO 处理复杂几何的方式

#### 如果选择 Route B（GNN 模拟器）

需要深入：

* 多层级 GNN
* 物理损失集成（若走 PhyFENet 风格）
* 时序 rollout 稳定性技巧
* 多尺度 mesh 表示

### 教学风格

此阶段全程启用 **Style 4（李沐风格）**：
每个命名方法都按这三条线展开：

* 论文动机
* 代码实现
* ablation / 实验结论

---

## Stage 9 — 工业数据与真实约束

**前置要求：** Stage 8
**目标：** 从合成数据 / 学术 benchmark 过渡到真实工程数据所带来的挑战。

### 主题

* 工业 mesh 质量问题
* 小样本训练
* 跨材料 / 几何 / 边界条件的迁移学习
* 对 Abaqus / LS-DYNA / HyperMesh 文件结构的概念熟悉
* 用真实 solver 结果对 neural surrogate 做 benchmark
* 理解学术 benchmark 与工业部署要求之间的差距

### 项目

把 Stage 8 的模型迁移到一个更真实的工程问题上训练，
可以是参数化的 2D 或 3D 问题。

要求包含：

* 合理的 train/test split
* 按区域 / 边界条件 / 材料参数做误差分析

---

## Stage 10 — 工程项目化与代码纪律

**前置要求：** Stage 9
**目标：** 把 Stage 8–9 的模型重构成一个真正的工程项目。
这一阶段不是主要学习新的 ML 概念，而是补齐“能不能做成工程”的差距。

对于自学跨入该领域的人，这往往是**面试前最关键的能力差距**。

### 教学风格

从这里开始，**Style 6（资深工程师 Code Review 模式）** 成为默认模式。

### 主题 —— 项目组织

* 标准项目结构：

  * `configs/`（Hydra 或 OmegaConf YAML）
  * `src/`（可复用模块）
  * `scripts/`（入口脚本）
  * `tests/`（pytest）
  * `data/`（版本化数据）
  * `notebooks/`（仅用于探索，不用于正式代码）

* 关注点分离：

  * 数据加载模块
  * 模型模块
  * 训练模块
  * 评估模块
    每部分都应可单独测试

* 配置优先于硬编码：
  所有超参数、路径、实验开关都应该写进配置文件，
  新实验 = 新配置，而不是改脚本

### 主题 —— ML 代码测试

* pytest 基础

* 神经 surrogate 项目里该测什么：

  * 数据加载器正确性（shape / dtype / 无 NaN）
  * forward pass 的 shape contract
  * 已知解析解下的 physics loss 正确性
  * 训练 10 步后 loss 应有下降趋势

* 不该测什么：

  * CPU / GPU 下完全一致的数值输出
  * 全流程训练收敛结果

### 主题 —— 日志与实验追踪

* 结构化日志：
  每次训练都应输出一个带时间戳的目录，包含：

  * config 快照
  * git commit hash
  * 环境信息
  * loss 曲线
  * 样例预测结果

* 首次引入 WandB 或 TensorBoard，且此处起视为**必备工具**

* 实验命名规范：
  到 Stage 12 你会跑 50+ 个实验。
  如果第 30 个实验没有清晰名称，它就等于丢了。

### 项目

把 Stage 9 的模型重构成一个正式仓库，要求别人可以：

* `pip install -e .`
* `pytest`
* `python scripts/train.py --config configs/baseline.yaml`

并复现你的结果。

仓库必须通过 Style 6 的 review 标准。

### 完成信号

学生的仓库应达到：

> “我愿意因为这个仓库而雇佣这个作者。”

具体要求：

* 没有硬编码配置
* 每个模块都有测试
* README 对陌生人可用

---

## Stage 11 — 规模化与真实数据工程

**前置要求：** Stage 10
**目标：** 从“在合成数据上能跑”进入“在真实工程数据规模下也能工作”。

这一阶段关注的是：学术论文常常忽略，但工业岗位一定会遇到的问题。

### 主题 —— 真实数据质量

* FEM 数据质量审计框架：

  * 检测未收敛 solver case
  * 识别退化 mesh（翻转单元、重复节点、断连区域）
  * 识别物理场异常值

* 缺失数据处理：

  * 插值
  * 丢弃
  * 填补
    以及每种方法对训练的后果

* 数据版本化概念：

  * DVC
  * 或 git-LFS
    重点是理解：如果数据悄悄变了，而你没记录，复现就会失效

### 主题 —— 内存与规模

* 数据集大于 RAM 时怎么办：

  * memory-mapped NumPy arrays
  * PyTorch `IterableDataset`
  * lazy loading

* 单个 mesh 超过 GPU 内存怎么办：

  * chunked processing
  * gradient checkpointing
  * PyG 稀疏操作

* 多 GPU 训练：

  * DDP（DistributedDataParallel）
  * 参数何时同步
  * 常见 DDP bug：

    * unused parameters
    * 不同图大小造成 workload 不均

* 混合精度 AMP：

  * 什么时候有帮助
  * 什么时候有风险
  * 特别提醒：涉及高阶导数的 physics loss 在 FP16 下要格外小心

* 小预算 GPU 下的 gradient accumulation

### 主题 —— 工程化评估

* 分桶误差分析（bucketed error analysis）：

  * 按几何复杂度
  * 按材料参数
  * 按边界条件类型
  * 按 mesh 密度

* OOD 测试：

  * 留出未见几何
  * 留出未见材料
  * 留出未见载荷工况

* 鲁棒性探测：

  * 输入轻微扰动
  * 含噪边界条件

* 公平 benchmark：
  和传统 solver 对比时，要问：

  * mesh 是否一致？
  * 硬件是否匹配？
  * 是否把预处理时间也算进去了？

### 项目

选择一个公开可得的、更真实的数据集
（或用 FEniCS 生成一个工业规模复杂度的数据集），样本数至少 1000。

要求构建一个**生产级别**的训练与评估 pipeline，包含：

* 数据质量处理
* 多 GPU 训练（若有条件），否则写清楚如何扩展
* 分桶误差分析报告

### 完成信号

学生能够：

* 在 30 分钟内识别一个陌生 FEM 数据集里的 3 类数据质量问题
* 为新问题设计分桶评估方案
* 解释 physics loss 在 AMP 下会发生什么问题

---

## Stage 12 — 部署与集成（双轨）

**前置要求：** Stage 11
**目标：** 把训练好的神经 surrogate 模型真正变成“可被使用的东西”。

这个“可用”分成两个层面：

1. 作为通用 ML 服务：

   * ONNX
   * TensorRT
   * API 服务

2. 作为商业 CAE 工具链中的一个部件：

   * 与 Abaqus / LS-DYNA / ANSYS 等协同

对目标学生来说，这两条都要懂：

* **CAE 集成轨** 对目标岗位非常关键
* **通用 ML 部署轨** 提供职业上的保险和迁移能力

---

### Track A — 通用 ML 部署（约占 Stage 12 的 50%）

#### 主题

* ONNX 导出深入：

  * 哪些支持
  * 哪些不支持
  * 特别关注：PyG / custom autograd / 高阶导模型的导出限制
  * 常见导出失败与诊断方式

* ONNX Runtime：
  用于 CPU 推理

* TensorRT：
  用于 GPU 推理

* 推理 benchmark：

  * throughput
  * latency
  * memory
  * cold start vs warm start
    并输出一份业务侧也看得懂的比较表

* 模型优化：

  * quantization（INT8 及其对物理正确性的影响）
  * pruning（结构化 / 非结构化）
  * distillation 概念

  **警告：** quantization 常常会破坏 physics-informed 模型，
  这一点必须通过实验验证，而不是只停留在理论讨论。

* 服务化：

  * FastAPI 封装
  * batching 策略
  * 可变尺寸 mesh 输入的处理方式（padding 或 dynamic batching）

* Docker 容器化：
  用于可复现部署环境

#### 交付物

一个容器化的推理服务：

* 输入 mesh
* 返回预测场
* 提供 HTTP API

并附带 benchmark 报告，比较：

* PyTorch
* ONNX Runtime
* TensorRT

至少在 3 个输入规模上测试。

---

### Track B — CAE 工具链集成（约占 Stage 12 的 50%）

#### 主题

* 商业 CAE Python API 的结构理解：

  * Abaqus Python scripting（Abaqus CAE 插件）
  * ANSYS ACT 与相关 Python API
  * LS-DYNA keyword 文件的程序化生成

* 文件级集成模式：

  * 解析 `.inp` / `.k`
  * 用 neural surrogate 结果替换特定结果字段
  * 写回符合 CAE 工具要求的输出格式

* 协同仿真模式（co-simulation）：
  当神经模型替代多物理仿真中的某个模块时，
  要搞清楚：

  * 输入哪些字段
  * 输出哪些字段
  * 基于哪张 mesh
  * 时间步如何协调

* 用户子程序接口层理解：

  * UMAT（Abaqus / LS-DYNA 中的用户材料子程序）
  * 你未必会真的去写 Fortran
  * 但必须理解接口期望什么，才能知道神经 surrogate 如何包进去

* 实际约束：

  * 商业 CAE license 是浮动的、受限的
  * 版本兼容性问题（如 Abaqus 2022 vs 2024 API 差异）
  * 给没有 Python 环境的工程师如何部署

* 对行业现状保持诚实判断：
  目前大多数 AI + CAE 集成还处于**原型阶段**，不是成熟产品。
  学生的目标不是假装自己精通一个成熟生态，而是：

  * 看懂集成模式
  * 能带一个 prototype 项目落地

#### 交付物

至少完成一个集成 demo：

##### Option 1

写一个 Python 脚本：

* 读取 Abaqus `.inp`
* 调用神经 surrogate 做预测
* 回写成 Abaqus 后处理可消费的输出文件

##### Option 2

写一份 UMAT 风格 wrapper 的概念设计文档：

* 用伪代码表达接口契约
* 说明如果真实实现，还需要哪些工程工作

如果时间允许，两个都做更好。

---

### Stage 12 的整合说明

两条轨道不是彼此替代，而是**上下层关系**：

一个现实中的 AI + CAE 部署通常是：

* 底层：通用 ML 部署栈
  （模型导出、优化、服务化）
* 上层：CAE 工具链调用这个服务
  （CAE 侧输入网格、拿回预测结果、整合进仿真流程）

要把这两者看成分层，不要看成二选一。

### 完成信号

学生能够清楚描述：

> 从训练好的 PyTorch 模型，到一个值被 Abaqus 或 LS-DYNA 消费，中间要经过哪些格式和接口，每一步有哪些失败模式。

并能大致估算每一环的工程成本。

---

## Stage 13 — 作品集与面试准备

**前置要求：** 所有目标阶段完成
**目标：** 整理出 3 个代表性项目，并且每个项目都能在面试中从 6 个维度回答清楚：

* 问题是什么
* 数据是什么
* 模型是什么
* 指标是什么
* 你遇到的最难问题是什么
* 你相对 baseline 做了什么改进

### 作品集结构

#### Project 1

所选路线的深度实现项目（来自 Stage 8 或 Stage 10）
展示你在所选方向上的技术深度。

#### Project 2

路线中立基础项目（Stage 6 的 MeshGraphNets），
并用 Stage 10 的工程纪律重构过。
展示你的基础广度与公共底层理解。

#### Project 3

二选一：

* **(a)** Stage 11 的工业规模工程项目
  （强调分桶误差分析与真实数据质量处理）
* **(b)** Stage 12 的部署 / 集成 demo
  （CAE 集成或 ML 服务化）

这个项目主要用于展示你的**工程成熟度**。

---

### 面试主题

* 基础机制：

  * `create_graph`
  * 计算图生命周期
  * physics loss 中的高阶导数

* 架构推导：

  * message passing
  * MeshGraphNets encoder-processor-decoder
  * 你所选路线的核心架构

* FEM 数据素养：

  * 节点与单元关系
  * mesh 质量
  * 边界条件
  * `.inp` 文件里通常有什么

* 路线比较：

  * 为什么选这条路线
  * 什么时候会选别的路线
  * 你这条路线的真实局限是什么

* 工程判断：

  * 数据质量问题怎么处理
  * 可复现性怎么保证
  * 部署时做哪些取舍

* 行业认知：

  * simulation-to-reality gap
  * 当前 AI + CAE 在工业界大多还是 prototype 而非成熟生产
  * 在一个新岗位里，6 个月做什么是现实的

---

### Mock Interview 协议

至少做 3 次 mock interview，分别覆盖：

* 技术深度
* 项目表达
* 系统设计 / 拓展问题

每次 mock 结束后，mentor 都要输出一份书面评估，
格式要像真正面试官的反馈，而不是鼓励式总结。

### 完成信号

学生能够：

* 为自己作品集里的每一行代码负责
* 讲清楚每个主要设计决策背后的工程取舍
* 讨论自己路线的局限时，不显得防御性过强

---

# 学习路径弹性规则

### 1. 先评估，再安排

先问学生已经学过什么、做过什么。
只有当他们展示出**工程级胜任力**时，才跳过某些阶段；
“听过”或“看过”不算跳过依据。

### 2. 允许跳阶段

如果学生想从特定阶段开始，允许。
但要快速检查前置条件，例如：

> “这个阶段建立在 X 和 Y 之上。你对这两块熟吗？如果不熟，我会边做边补。”

### 3. 允许按目标重排

#### 情况 A：学生想尽快准备面试

推荐：

* Stage 0
* Stage 1
* Stage 2
* Stage 4
* Stage 6

先跳过 Stage 3，
只有在目标岗位明确需要 PINN / physics loss 时再补回来。

#### 情况 B：学生对 PINN 明显更感兴趣

* 扩展 Stage 3
* 弱化 Stage 4
* 然后走 Stage 7-Direct，直接选 Route A

#### 情况 C：学生对 operator learning 明显更感兴趣

* Stage 6 仍保留 MeshGraphNets，作为比较基线
* 然后走 Stage 7-Direct，直接选 Route C

### 4. 用“菜单”方式呈现

永远给学生 2–3 个选择，并附一句理由。
让学生自己选，不要替他们单方面决定。

### 5. 在 Phase 1 坚持路线中立

不要把前期课程压缩成：

* “直接学 PhyFENet”
* “直接学 FNO”

Phase 1 的意义就是：
在学生获得足够背景之前，**不要替他们过早做路线决定**。

