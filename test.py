import torch
import torch.nn as nn

# ── 拼图 1：准备数据 (X_train, y_train) ──
# 假设任务：根据“学习时长(小时)”预测“考试能不能及格(1=及格, 0=挂科)”
X_train = torch.tensor([[1.0], [2.0], [8.0], [9.0]]) # 特征：学了几个小时
y_train = torch.tensor([[0.0], [0.0], [1.0], [1.0]]) # 标签：0代表挂科，1代表及格

# ── 拼图 2：定义模型 (model) ──
# 逻辑回归 = 线性回归 (nn.Linear) + Sigmoid 掰弯机制
model = nn.Sequential(
    nn.Linear(in_features=1, out_features=1), # 先画一条直线 (wx + b)
    nn.Sigmoid()                              # 然后强行把它掰弯，限制在 0 到 1 之间
)

# ── 拼图 3：定义代价函数 (loss_function) ──
# 重点！分类问题绝不能用 MSE，这里使用的是“二元交叉熵 (Binary Cross Entropy)”
loss_function = nn.BCELoss() 

# ── 拼图 4：装载你的高级引擎并启动 ──
# (这就是你刚才贴的代码，我把学习率设为 0.1 跑得快一点)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

print("🚀 开始训练...")
for epoch in range(100):
    
    # [前向传播] 算预测值
    predictions = model(X_train)
    
    # [算代价]
    cost = loss_function(predictions, y_train)
    
    # 🔴 关键三步曲
    optimizer.zero_grad()  
    cost.backward()        
    optimizer.step()       
    
    # 每 20 圈打印一次成绩，看看计分板上的失分是不是越来越少
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100] - 计分板 (Cost): {cost.item():.4f}")

# ── 训练结束，测试一下！ ──
print("\n🔮 预测未来：")
test_student = torch.tensor([[7.0]]) # 一个新学生，复习了 7 个小时
pass_probability = model(test_student).item()
print(f"复习 7 个小时的及格概率是: {pass_probability * 100:.1f}%")