import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据：缩小 z 的范围，聚焦核心变化区域
z = np.linspace(-5, 5, 1000)

sigmoid_value = 1 / (1 + np.exp(-z))
tanh_value = np.tanh(z)
relu_value = np.maximum(0, z)

# 2. 设置画布大小与分辨率
plt.figure(figsize=(10, 6), dpi=100)

# 3. 绘制函数曲线（加粗线条，使用更现代的十六进制颜色）
plt.plot(z, sigmoid_value, color="#e74c3c", linewidth=2.5, label="Sigmoid")
plt.plot(z, tanh_value, color="#2ecc71", linewidth=2.5, label="Tanh")
plt.plot(z, relu_value, color="#3498db", linewidth=2.5, label="ReLU")

# 4. 建立物理锚点：添加 x=0 和 y=0 的辅助线
plt.axhline(y=0, color='black', linewidth=1.2, linestyle='-')
plt.axvline(x=0, color='black', linewidth=1.2, linestyle='-')

# 5. 强制截断 Y 轴（拯救被拍扁的 Sigmoid 和 Tanh）
plt.ylim(-1.5, 3.5)

# 6. 排版与装饰
plt.title("Activation Functions Comparison", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Input (z)", fontsize=12)
plt.ylabel("Activation Output (a)", fontsize=12)
plt.legend(fontsize=12, loc='upper left')

# 增加微弱的网格线，方便读取具体的数值边界
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()