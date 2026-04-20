# AI 赋能冲压成形仿真 · 转行学习计划

## 第一段输出：全局结构 + 补零期（Week P1–P4）

你的目标岗位是：**用 GNN + 物理信息嵌入（PhyFENet 式框架）加速冲压成形 FEM 计算的 AI 算法工程师**。所有练习围绕这个目标选材，不做与此无关的内容。

---

## 全局结构

```
补零期   P1–P4  （约4周）
         Python 工程基础 + NumPy + Matplotlib + 数学直觉
         完成标准：能用 Python 处理数组数据、画图、模拟解析 FEM 文件

第一阶段  Week 1–22  （约22周，可弹性延长）
  Part A  W1–W7    深度学习地基（MLP / Loss / 梯度下降 / PyTorch）
  Part B  W8–W10   自动微分深入（autograd / 物理约束 Loss）
  Part C  W11–W16  图神经网络（图结构 / GCN / 消息传递 / GraphSAGE + 边更新）
  Part D  W17–W22  PINN + 有限元数据入门（ODE/PDE / 弹性力学 / mesh-to-graph）
  完成标准：能手写 PhyFENet 各模块，在合成数据上跑通

第二阶段  Week 23–44  （约22周，可弹性延长）
  Part E  W23–W27  工业 CAE 工具认知 + 真实 FEM 数据处理
  Part F  W28–W32  完整 PhyFENet 架构实现
  Part G  W33–W37  弹塑性本构嵌入 + 多级网络训练
  Part H  W38–W44  工程项目完整交付（冲压成形预测）
  完成标准：一个完整的、可展示的工程项目，含评估报告和可视化

第三阶段  Week 45–56  （约12周）
  Part I  W45–W48  作品集 + 简历打磨
  Part J  W49–W53  面试准备（行业认知 / 八股 / 手撕代码 / 模拟面试）
  Part K  W54–W56  实战面试 + Offer

总计约 56 周 ≈ 14 个月（各阶段均可弹性延长）
```

---


# 补零期（Week P1–P4）：Python 工程基础

## 为什么先做这个

在进入深度学习之前，你需要的不是 Python 语法考试，而是**工程 Python 手感**——读写文件、处理数组、画图、写可复用的函数和类。这些是你整个学习过程中每天都在用的操作。

补零期不用 LeetCode，原因是：你未来需要的是"读取一个有 10000 行节点数据的文本文件，把坐标转成 NumPy 数组，画出分布图"这类任务，而不是"反转一个链表"。所有练习的选材都在模拟你未来真实工作中会碰到的操作。

---

## Week P1：Python 基础语法与工程编程思维

**本周目标**：能写出可运行的 Python 脚本，能定义函数和类，能模拟解析简单的有限元文本文件。

**工具**：Python 3.10+，VS Code（安装 Python 扩展），所有代码写 `.py` 文件，不用 Jupyter Notebook（Notebook 之后的阶段才引入）。

**主要资源**：廖雪峰 Python3 教程（https://www.liaoxuefeng.com/wiki/1016959663602400），按顺序看"Python简介"→"安装Python"→"Python基础"→"函数"→"面向对象编程（前两节）"→"IO编程（文件读写）"。

---

### Day 1 | 安装 + 基础数据类型

**任务**：创建文件 `day01_types.py`，完成以下内容——

✅ 1. 定义以下 5 个变量并打印每个变量的值和类型：
   - `node_id = 42`（int）
   - `x_coord = 3.14159`（float）
   - `label = "node_A"`（str）
   - `coords = [1.0, 2.5, 0.0]`（list，代表 x/y/z 坐标）
   - `properties = {"E": 200.0, "nu": 0.3}`（dict，代表材料属性）

✅ 2. 对 `coords` 做切片，分别取出：
   - 第一个元素（x 坐标）
   - 最后一个元素（z 坐标）
   - 前两个元素（x 和 y）

✅ 3. 对 `properties` 字典做遍历，按"键: 值"格式打印每一对。

**验收标准**：脚本无报错运行，输出中每个变量的类型名称正确（例如 `<class 'int'>`），切片结果正确，字典遍历能打印两行。

---

### Day 2 | 控制流（条件 + 循环）

**任务**：创建文件 `day02_control.py`——

✅ 1. 写函数 `classify_strain(epsilon)`：
   - 输入：一个浮点数（代表应变值）
   - 规则：epsilon < 0.001 返回 "elastic"；0.001 ≤ epsilon < 0.01 返回 "small_plastic"；epsilon ≥ 0.01 返回 "large_plastic"
   - 调用并打印：`classify_strain(0.0005)`、`classify_strain(0.005)`、`classify_strain(0.02)` 的结果

2. 写函数 `sum_node_distances(node_list)`：
   - 输入：一个列表，每个元素是形如 `[x, y]` 的坐标列表，例如 `[[0,0],[1,0],[1,1]]`
   - 功能：用 for 循环计算相邻节点之间的欧氏距离之和（第0和第1之间，第1和第2之间，...）
   - 输出：总距离（float）
   - 验证：输入 `[[0,0],[3,0],[3,4]]`，输出应为 7.0（3 + 4）

**验收标准**：两个函数均能通过上述具体输入得到正确输出，不看参考能写出来。

---

### Day 3 | 函数（参数 + 返回值）

**任务**：创建文件 `day03_functions.py`——

1. 写函数 `read_nodes(filepath)`：
   - 功能：读取一个文本文件，文件每行格式为 `node_id  x  y  z`（用空格或制表符分隔，例如 `1  0.0  1.5  2.3`）
   - 返回：字典 `{node_id (int): [x, y, z] (list of float)}`
   - 注意：跳过以 `#` 开头的注释行，跳过空行

2. 自己创建文件 `test_nodes.txt`，内容如下：
   ```
   # This is a comment
   1  0.0  0.0  0.0
   2  1.0  0.0  0.0
   3  1.0  1.0  0.0
   4  0.0  1.0  0.0
   5  0.5  0.5  1.0
   ```

3. 调用 `read_nodes("test_nodes.txt")`，打印返回的字典，并打印"共读取 X 个节点"。

**验收标准**：返回字典包含5个键（1到5），坐标值为 float 类型；注释行和空行被正确跳过；不看参考能在20分钟内写出。

---

### Day 4 | 类（Class）

**任务**：创建文件 `day04_class.py`——

1. 定义类 `FEMNode`：
   - `__init__(self, node_id, x, y, z)`：保存节点id和初始坐标，初始化位移为 `[0.0, 0.0, 0.0]`
   - `set_displacement(self, ux, uy, uz)`：设置位移分量
   - `get_displacement_magnitude(self)`：返回位移矢量的模，即 `√(ux² + uy² + uz²)`，用 Python 内置 `math.sqrt` 实现
   - `__repr__(self)`：返回形如 `"FEMNode(id=1, coords=[0.0,0.0,0.0], disp_mag=3.0)"` 的字符串

2. 验证代码（写在文件末尾）：
   ```python
   node = FEMNode(1, 0.0, 0.0, 0.0)
   node.set_displacement(1.0, 2.0, 2.0)
   print(node.get_displacement_magnitude())  # 应输出 3.0
   print(node)  # 应输出 FEMNode(id=1, ...)
   ```

**验收标准**：`get_displacement_magnitude()` 对输入 (1.0, 2.0, 2.0) 精确返回 3.0；`__repr__` 输出格式正确。

---

### Day 5–6（周末）| 综合练习：模拟 FEM 数据解析器

**任务**：创建文件 `weekend_p1.py`——

**Step 1**：创建测试文件 `fem_simple.txt`，内容严格如下：
```
*NODE
1  0.0  0.0  0.0
2  1.0  0.0  0.0
3  1.0  1.0  0.0
4  0.0  1.0  0.0
*ELEMENT_SHELL
1  1  2  3  4
```

**Step 2**：写函数 `parse_fem_file(filepath)`：
- 解析上述格式，识别 `*NODE` 和 `*ELEMENT_SHELL` 两个关键字
- 返回字典：`{'nodes': {id: [x,y,z]}, 'elements': {id: [n1,n2,n3,n4]}}`

**Step 3**：写函数 `write_summary(data, output_filepath)`：
- 将以下内容写入文本文件：节点总数、单元总数、x 坐标的最小值和最大值、y 坐标的最小值和最大值
- 格式自定，但需要可读

**Step 4**：在 `main()` 函数中串联调用上面两个函数并测试。

**验收标准**：
- `parse_fem_file` 返回字典中，`nodes` 包含4个节点，`elements` 包含1个单元
- `write_summary` 生成的文件内容数值正确（x范围 [0.0, 1.0]，y范围 [0.0, 1.0]）
- 不看参考，能在 30 分钟内独立完成

**Week P1 完成总标准**：不看任何参考，能在 30 分钟内从零写出 `parse_fem_file()` 并通过上述自测。

---

## Week P2：NumPy 数组思维

**为什么学 NumPy（衔接上周）**：上周你用 Python 原生 list 处理坐标数据，但 FEM 网格里有几千甚至几万个节点，用 list 做计算既慢又麻烦——比如计算所有节点到原点的距离，你得写 for 循环。NumPy 的向量化操作可以在不写 for 循环的情况下一次性处理整个数组。之后你所有的神经网络数据处理都依赖 NumPy 的这种思维。

**主要资源**：NumPy 官方 "NumPy: the absolute basics for beginners"（https://numpy.org/doc/stable/user/absolute_beginners.html），全部读完，约2小时。

---

### Day 1 | 数组创建与 shape

**任务**：创建文件 `numpy_day1.py`——

用以下5种方式各创建一个数组，对每个数组打印其 `shape`、`dtype`、`ndim`：
1. `np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])` — 从嵌套 list 创建
2. `np.zeros((4, 3))` — 4个节点，每个3个坐标，全零
3. `np.ones((10,))` — 10个元素的一维数组
4. `np.arange(0, 1.0, 0.1)` — 从0到0.9，步长0.1
5. `np.linspace(0, 1, 11)` — 从0到1，均匀取11个点

然后对第1个数组（shape=(2,3)）做 reshape：
- reshape 成 (6,)，打印 shape
- reshape 成 (3,2)，打印 shape
- reshape 成 (1,6)，打印 shape

**验收标准**：所有 shape 输出正确；能说清楚"reshape 不改变数据，只改变数组的组织方式"；`ndim` 告诉你数组有几个维度（axis）。

---

### Day 2 | 索引与切片

**任务**：创建文件 `numpy_day2.py`——

创建以下数组，代表 6 个节点，每个节点有 4 个特征（x, y, z, 载荷F）：
```python
np.random.seed(42)
X = np.random.randn(6, 4)
```

完成以下操作（每步打印结果和 shape）：
1. 取出第0行（第1个节点的所有特征），shape 应为 `(4,)`
2. 取出第2列（所有节点的第3个特征，即z坐标），shape 应为 `(6,)`
3. 取出前3行（前3个节点），shape 应为 `(3, 4)`
4. 布尔索引：取出第0列（x坐标）大于0的所有行，打印结果
5. 将第4行第2列的值修改为 -999.0，打印整个数组确认修改

**验收标准**：每个操作的 shape 和结果正确；特别确认布尔索引的行数 ≤ 6。

---

### Day 3 | 向量化与广播（不用 for 循环）

**任务**：创建文件 `numpy_day3.py`——

创建节点坐标数组：
```python
np.random.seed(0)
coords = np.random.rand(100, 3) * 10  # 100个节点，坐标在[0,10]
```

完成以下操作，**全部不用 for 循环**：

1. 计算每个节点到原点(0,0,0)的欧氏距离，结果 shape 为 `(100,)`
   - 提示：先对每个坐标分量平方，再求和，再开根号
   - 验证：用 `np.linalg.norm(coords, axis=1)` 检查结果是否一致

2. 对 `coords` 做 Z-Score 标准化（对每列分别减均值除标准差）：
   - 计算每列的均值（shape=(3,)）和标准差（shape=(3,)）
   - 用广播完成标准化，结果 shape 仍为 (100,3)
   - 验证：标准化后每列均值应接近0（绝对值 < 0.001），每列标准差应接近1

3. 计算第0个节点与所有其他节点的距离，结果 shape 为 `(100,)`
   - 提示：用广播，`coords[0]` shape=(3,)，`coords` shape=(100,3)，相减会广播

**验收标准**：无 for 循环；欧氏距离与 `np.linalg.norm` 结果一致；标准化后均值 < 0.001，标准差 ∈ [0.99, 1.01]。

---

### Day 4 | 线性代数与数组拼接

**任务**：创建文件 `numpy_day4.py`——

1. 矩阵乘法练习（FEM 里大量用到）：
   - 创建 `A = np.array([[2.0, 1.0], [1.0, 3.0]])` 和 `b = np.array([5.0, 10.0])`
   - 用 `np.linalg.solve(A, b)` 求解 Ax=b，打印 x
   - 验证：用 `A @ x` 计算结果，应接近 b（误差 < 1e-10）

2. 理解 `@` 与 `*` 的区别（这个很容易混淆）：
   - 创建 `M = np.array([[1,2],[3,4]])` 和 `N = np.array([[5,6],[7,8]])`
   - 打印 `M @ N`（矩阵乘法）和 `M * N`（逐元素乘法）
   - 确认两个结果不同，写注释说明两者的区别

3. 数组拼接：
   - 创建 `node_coords = np.zeros((5, 3))` 和 `material_props = np.ones((5, 2))`
   - 用 `np.concatenate` 水平拼接，得到 shape=(5,5) 的特征矩阵
   - 用 `np.vstack` 和 `np.hstack` 各做一次拼接，打印结果 shape

**验收标准**：`np.linalg.solve` 结果满足 Ax≈b；能说清楚 `@` 是矩阵乘法、`*` 是逐元素乘法。

---

### Day 5–6（周末）| 综合练习：NumPy 处理 FEM 数据

**任务**：创建文件 `weekend_p2.py`——

**Step 1**：用 NumPy 读取 Week P1 创建的 `test_nodes.txt`，得到节点坐标数组 `coords`，shape=(5,3)。

**Step 2**：不用 for 循环，计算以下统计量并打印：
- 所有节点坐标的均值（shape=(3,)，对应 x/y/z 三个方向）
- 所有节点坐标的标准差（shape=(3,)）
- 每个节点到坐标中心（均值点）的距离（shape=(5,)）

**Step 3**：计算所有节点两两之间的距离矩阵，shape=(5,5)，不用双重 for 循环。
- 提示：用广播。`coords[:, np.newaxis, :]` shape=(5,1,3)，`coords[np.newaxis, :, :]` shape=(1,5,3)，相减后 shape=(5,5,3)
- 验证：距离矩阵的对角线（自身到自身的距离）应全为0

**Step 4**：找出距离最近的两个节点（非对角线上的最小值）：
- 将对角线设为无穷大（`np.fill_diagonal`）
- 用 `np.argmin` 找到最小值的位置

**验收标准**：距离矩阵 shape=(5,5)，对角线全为0；最近节点对的答案可以手动验证（节点3和节点4的距离应是最近的，约为 √((0.5-0)²+(0.5-1)²+(1-0)²) ≈ 1.22）。

**Week P2 完成总标准**：不看文档，20分钟内用 NumPy 写出"读取节点坐标文件→转为数组→计算统计量和距离矩阵"的完整脚本。

---

## Week P3：Matplotlib 可视化 + 数学直觉

**为什么在这里学 Matplotlib（衔接上周）**：你用 NumPy 算出了节点坐标、距离、统计量，但数字很难直接判断"这个结果对不对"。可视化是诊断数据和模型的核心工具——你后续会大量用到：Loss 曲线（判断训练是否收敛）、预测场 vs 真实场的对比云图（判断模型精度）、误差分布图（找到预测最差的区域）。

**为什么在这里建立数学直觉（同时进行）**：下周开始就要用到"导数是什么"和"梯度下降往哪走"。不需要你会做微积分题，但需要你有几何直觉。

**主要资源**：
- Matplotlib：官方 Tutorials → "Pyplot tutorial"（https://matplotlib.org/stable/tutorials/pyplot.html）
- 数学直觉：3Blue1Brown "Essence of Calculus" 系列（YouTube），只看第1、2、3集（约50分钟）

---

### Day 1 | Matplotlib 基础

**任务**：创建文件 `matplotlib_day1.py`，生成以下图，每张图保存为 `.png` 文件——

**图1** `activation_functions.png`：
- x 轴：`np.linspace(-5, 5, 200)`
- 在同一张图上画三条曲线：
  - y = sigmoid(x) = 1/(1+exp(-x))，蓝色实线，label='sigmoid'
  - y = tanh(x)，橙色实线，label='tanh'
  - y = relu(x) = max(0,x)，绿色实线，label='ReLU'
- 要求：有 xlabel='x'，ylabel='y'，title='Activation Functions'，legend，grid

**图2** `scatter_demo.png`：
- 生成数据：`x = np.random.randn(100); y = 2*x + np.random.randn(100)*0.5`
- 画散点图（`plt.scatter`），点用蓝色，alpha=0.6
- 画参考线 y=2x（红色虚线，linestyle='--'，label='y=2x'）
- 要求：xlabel='x'，ylabel='y'，title='Scatter with Reference Line'，legend

**验收标准**：两张 png 能生成，图例完整，坐标轴有标签，标题存在。用肉眼验证：sigmoid 输出范围在(0,1)之间，tanh 输出范围在(-1,1)之间，ReLU 在x<0时为0。

---

### Day 2 | 热力图与误差分布图

**任务**：创建文件 `matplotlib_day2.py`——

**图3** `heatmap_demo.png`：
- 创建 shape=(10,10) 的随机数组（代表10×10网格上的某个物理场）
- 用 `plt.imshow(data, cmap='jet', origin='lower')` 显示
- 加 `plt.colorbar(label='Value')`
- 有 title='Physical Field (Simulated)'

**图4** `histogram_demo.png`：
- 生成 500 个"预测误差"（正态分布模拟）：`errors = np.random.randn(500) * 0.05`
- 画直方图：`plt.hist(errors, bins=30, color='steelblue', edgecolor='white')`
- 加竖线标注误差=0的位置（`plt.axvline(x=0, color='red', linestyle='--')`）
- 有 xlabel='Prediction Error'，ylabel='Count'，title='Error Distribution'

**验收标准**：两张 png 能生成；热力图颜色条显示数值范围；直方图的红色竖线在x=0处。

---

### Day 3 | True vs Predicted 图（核心工具）

这个图贯穿你整个项目，单独一天专门练。

**任务**：创建文件 `plot_utils.py`，写函数 `plot_true_vs_pred`——

```
函数签名：plot_true_vs_pred(y_true, y_pred, title='True vs Predicted', save_path=None)

参数：
  y_true: 1D numpy数组，真实值
  y_pred: 1D numpy数组，预测值
  title: 字符串，图标题
  save_path: 字符串或None，如果不为None则保存图到该路径

图的要素（缺一不可）：
  - 散点：x轴为y_true，y轴为y_pred，每个点代表一个样本
  - 对角参考线：y=x，红色虚线（linestyle='--'，color='red'，label='Perfect Prediction'）
  - 图上用 plt.text() 标注 MAE 值，格式为 "MAE = 0.1234"
  - xlabel='True Values'，ylabel='Predicted Values'，title=title，legend
```

**验证代码**（写在文件末尾的 `if __name__ == '__main__':` 块中）：
```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
plot_true_vs_pred(y_true, y_pred, title='Test', save_path='test_tvp.png')
```

**验收标准**：
- 生成的图中散点在对角线附近
- 对角线为红色虚线
- 图上显示的 MAE 值约为 0.16（精确值：`(0.1+0.1+0.2+0.2+0.1)/5 = 0.14`）
- 函数在 `save_path=None` 时用 `plt.show()` 显示，不为 None 时保存到路径

---

### Day 4–6 | 数学直觉建立（微分 + 梯度）

**任务（这3天不写代码，写笔记 + 画手绘图）**——

**Day 4**：看 3Blue1Brown "Essence of Calculus" 第1集（The essence of calculus，约17分钟）
- 看完后，在笔记本上画一条曲线（任意），在曲线上标出一个点，画出该点处的切线
- 用中文写一句话解释"导数的几何意义是什么"（参考答案：导数是函数在某点处切线的斜率，描述的是函数在那个点附近以什么速率在变化）

**Day 5**：看第2集（The paradox of the derivative，约17分钟）+ 第3集（Derivative formulas through geometry，约18分钟）
- 理解 d(x²)/dx = 2x 的几何推导（不需要背住，理解"为什么是2x"的直觉即可）
- 用中文写：为什么对 `y = w*x` 求 w 的导数结果是 x？（直觉：w 增加一点点 Δw，y 增加 x·Δw，所以 y 对 w 的变化率就是 x）

**Day 6**：不看任何参考，用自己的话写一段200字以内的解释：
"梯度下降算法是如何让神经网络'学习'的？"

参考框架（用自己的语言表达，不要照抄）：
1. 我们有一个 Loss 函数（衡量预测和真实值的差距）
2. Loss 对每个参数 w 的导数告诉我们：w 增加一点，Loss 会怎么变
3. 如果导数是正数，说明 w 增加会让 Loss 增大，所以要让 w 减小
4. 按"导数的反方向"更新 w，就能让 Loss 下降
5. 重复很多次，Loss 越来越小，预测越来越准——这就是"学习"

**验收标准**：
- 手绘图上切线画法正确（切线只在那个点上"贴着"曲线，不穿入曲线）
- Day 6 的解释中没有出现"看不懂的公式"，全是用比喻和中文说清楚的逻辑

**Week P3 完成总标准**：
- 能从零写出 `plot_true_vs_pred()` 函数
- 能向别人不用任何公式解释"梯度下降在做什么"

---

## Week P4：代码工程化 + 环境准备

**为什么有这一周（衔接上三周）**：前三周的代码是分散的小脚本。但你整个一年的学习都会生产代码，如果不从一开始就建立好的组织习惯，到第10周你的代码会一团糟，找不到之前写的函数，也没法复用。这一周把之前的代码整理成工程结构，同时安装后续所需的所有库。

---

### Day 1–2 | 代码模块化与项目组织

**任务**：建立以下目录结构，把之前写的代码重新整理进去——

```
ai-cae-learning/
├── utils/
│   ├── __init__.py          ← 空文件即可
│   ├── fem_parser.py        ← 放 parse_fem_file(), read_nodes()
│   └── visualization.py     ← 放 plot_true_vs_pred()
├── data/
│   ├── test_nodes.txt
│   └── fem_simple.txt
├── pretrain/
│   ├── p1_week/             ← 放 day01 到 weekend_p1
│   ├── p2_week/
│   └── p3_week/
├── main.py                  ← 演示如何调用 utils 中的函数
└── requirements.txt         ← 记录用到的库（numpy, matplotlib等）
```

**具体操作**：
1. 在 `fem_parser.py` 中，把 `parse_fem_file()` 和 `read_nodes()` 放进去，在文件顶部加上 docstring 说明每个函数的输入输出
2. 在 `visualization.py` 中，把 `plot_true_vs_pred()` 放进去
3. 在 `main.py` 中写：
   ```python
   from utils.fem_parser import parse_fem_file
   from utils.visualization import plot_true_vs_pred
   # 调用两个函数，走通一个完整的小流程
   ```

**验收标准**：在项目根目录运行 `python main.py` 无报错，成功调用两个函数。

---

### Day 3–4 | 综合工程练习：生成→保存→读取→可视化

**任务**：创建文件 `pretrain/integration_test.py`——

**Step 1**：用 NumPy 生成模拟 FEM 数据（不依赖任何真实文件）：
- 20个节点，坐标随机分布在 [0, 10] × [0, 10] 的二维平面，z=0
- 15个四节点单元（随机连接，节点id在1到20之间）
- 每个节点有一个"真实位移"（ux, uy），用正弦函数生成：`ux = 0.1 * np.sin(x)`，`uy = 0.05 * np.cos(y)`

**Step 2**：将数据保存为文本文件（格式自定义，节点部分和单元部分分开，有关键字标识）

**Step 3**：从文件读回数据，转成 NumPy 数组

**Step 4**：生成"模拟预测位移"（加一点随机噪声）：`ux_pred = ux + np.random.randn(20) * 0.005`

**Step 5**：调用 `plot_true_vs_pred(ux, ux_pred, title='ux: True vs Predicted', save_path='ux_pred.png')`

**验收标准**：
- 全流程无报错
- 读回的数据和生成的数据一致（节点坐标误差 < 1e-6）
- 生成的图中 MAE 值约为 0.005 左右（随机噪声的量级）

---

### Day 5–6 | 安装所有库 + 验证

**任务**：在命令行中依次安装以下库，并逐一运行验证代码——

**安装命令**（CPU 版本，如果你有 NVIDIA GPU 可以去 PyTorch 官网选对应版本）：
```bash
pip install numpy matplotlib torch torchvision
pip install torch-geometric  # 按 PyG 官网步骤安装
pip install meshio scipy
```

**验证1 - NumPy 和 Matplotlib**（已在前几周验证过，快速确认）：
```python
import numpy as np
import matplotlib.pyplot as plt
print(np.__version__)
x = np.linspace(-3, 3, 100)
plt.plot(x, np.sin(x))
plt.savefig('verify_mpl.png')
print("NumPy + Matplotlib OK")
```

**验证2 - PyTorch**：
```python
import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # 应输出 tensor([2., 4., 6.])
print("PyTorch OK, version:", torch.__version__)
```

**验证3 - PyTorch Geometric**：
```python
import torch_geometric
from torch_geometric.data import Data
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[1.0], [2.0], [3.0]])
data = Data(x=x, edge_index=edge_index)
print("PyG OK, num_nodes:", data.num_nodes)  # 应输出 3
print("PyG version:", torch_geometric.__version__)
```

**验证4 - meshio**：
```python
import meshio
print("meshio OK, version:", meshio.__version__)
```

**遇到安装报错怎么办**：
- 不要跳过，把完整报错信息截图或复制保存
- PyG 的安装最容易出问题，按官网（https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html）确认 PyTorch 版本和 PyG 版本的对应关系
- 如果 GPU 版本实在装不上，先用 CPU 版本继续，GPU 的加速对你当前阶段的小数据不是必须的

**验收标准**：四个验证代码全部能无报错运行，输出与预期一致。

---

## 补零期完成总标准

进入第一阶段 Week 1 之前，需要全部达到以下标准：

**Python**
- [ ] 不看参考，30 分钟内从零写出 `parse_fem_file()` 并通过自测
- [ ] 能定义包含 `__init__`、普通方法、`__repr__` 的类
- [ ] 能组织项目目录结构，能在不同文件之间 import 函数

**NumPy**
- [ ] 不看文档，20 分钟内用 NumPy 写出"读取坐标文件→转数组→统计+可视化"的完整脚本
- [ ] 能用广播（不用 for 循环）计算成对距离矩阵
- [ ] 理解 shape / axis / broadcasting 的含义

**Matplotlib**
- [ ] 能从零写出 `plot_true_vs_pred()` 函数（包含对角线、MAE 标注）
- [ ] 能画 heatmap（imshow + colorbar）
- [ ] 能画 histogram

**数学直觉**
- [ ] 能用自己的话（不用公式）向别人解释"梯度下降是什么"
- [ ] 理解"导数是切线斜率"的几何含义

**环境**
- [ ] PyTorch、PyTorch Geometric、meshio 全部能成功 import 并通过验证代码

---

