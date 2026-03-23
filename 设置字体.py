# 1. 设置中文字体（macOS 专用：Arial Unicode MS 或 Heiti TC）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 

# 2. 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
