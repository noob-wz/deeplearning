import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据 (横坐标 z)
# 生成从 -10 到 10 的 1000 个均匀分布的数字。如果这里生成的数据太少，看起来就像是折线图
# (1000,) -> 一维向量
z = np.linspace(-10, 10, 1000)

# 2. 向量化计算 (纵坐标 p)
# 利用 NumPy 的广播和底层 C 加速，瞬间完成 1000 次 Sigmoid 运算
# (1000,) -> 一维向量
p = 1 / (1 + np.exp(-z))

# 3. 渲染图像
# figsize=(8, 5) 表示图像尺寸：宽 8 高 5。单位通常可以理解成英寸。
plt.figure(figsize=(8, 5))
plt.plot(z, p, color='blue', linewidth=2, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')

# 4. 辅助工程视觉线 (标注决策边界和极限天花板/地板)
# axvline 意思是 axis vertical line，也就是“竖线”。
plt.axvline(x=0, color='green', linestyle='--', label='Decision Boundary (z=0)')
# axhline 意思是 axis horizontal line，也就是“横线”。
plt.axhline(y=0.5, color='gray', linestyle=':')
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ceiling (p=1.0)')
plt.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Floor (p=0.0)')

# 5. 排版与展示
plt.title('Sigmoid Activation Function')
plt.xlabel('Raw Logits (z)')
plt.ylabel('Probability (p)')
# 显示名称图例。凡是前面设置过 label=... 的曲线或辅助线，都会在图例中出现。
plt.legend()
# 开启网格，透明度为0.3
plt.grid(True, alpha=0.3)
plt.show()

