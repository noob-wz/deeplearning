import torch


X = torch.tensor([
    [8.0, 7.0, 0.9],   # 第 0 个学生的数据
    [5.0, 6.0, 0.7],   # 第 1 个学生的数据
    [3.0, 5.0, 0.5],   # 第 2 个学生的数据
    [9.0, 8.0, 1.0],   # 第 3 个学生的数据
])

y = torch.tensor([85.0, 78.0, 65.0, 92.0])

w = torch.tensor([[0.6], [0.3], [0.1]], requires_grad=True) 
b = torch.tensor([0.0],requires_grad=True)    


for epoch in range(100):
    Y_pred = X @ w + b
    loss = ((Y_pred - y)**2).mean()
    loss.backward()

    with torch.no_grad():
        w -= 0.01*w.grad
        b -= 0.01*b.grad

    w.grad.zero_()
    b.grad.zero_()
    
    # 每隔60次看一下学习进度
    if epoch % 60 == 0:
        print(f"第{epoch}次的Loss大小为：{loss.item():.5f}")
        # 加上.item()代表：我要把它当作一个普通标量拿出来记录、打印、日志统计