import torch

x = torch.tensor([[1.0], [2.0], [3.0]]) 
y = torch.tensor([[3.0], [5.0], [7.0]])

w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

for epoch in range(100):
    y_pred = x @ w + b
    
    y_pred.backward()
    
    with torch.no_grad():
        w -= 0.01* w.grad
        b -= b.grad
    
    w.grad.zero_()
    b.grad.zero_()
    
print(f"最终结果：")