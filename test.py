import numpy as np
import torch.nn

X = torch.randn(200, 2)
y = X[:,0]**2 + X[:, 1] + torch.randn(1,2)*0.01

nn.Sigmoid