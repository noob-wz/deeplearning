import torch
from torch.utils.data import Dataset, DataLoader

# 1. 编写说明书（Dataset）
class TrafficSignDataset(Dataset):
    def __init__(self, num_samples):
        # 实际工程中，这里通常是读取包含所有图片路径的 txt 或 csv 文件
        # 这里我们只记录数据集的总大小
        self.num_samples = num_samples

    def __len__(self):
        # 返回数据集的总长度。引擎依赖它来判断一个 Epoch 什么时候结束
        return self.num_samples

    def __getitem__(self, index):
        # 引擎会传入一个具体的 index (例如 42)
        # 实际工程中，这里会写：img = cv2.imread(self.image_paths[index])
        
        # 模拟：生成一张 3通道 224x224 的随机特征图
        x = torch.randn(3, 224, 224) 
        
        # 模拟：生成一个 0 到 9 之间的随机整数作为类别标签，形状为纯标量 ()
        y = torch.tensor(5) 
        
        return x, y

# 2. 实例化说明书（假设我们有 1000 张图）
dataset = TrafficSignDataset(num_samples=1000)

# 3. 实例化搬运引擎 (DataLoader)
# 将说明书交给引擎，设定每次搬运 32 张，并且打乱顺序
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. 模拟训练循环中“取数据”的动作
for epoch in range(1):
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        # Dataset 的每个元素，通常是一个样本的 (x, y)
        # DataLoader 的每个元素，通常是一批样本组成的 (x_batch, y_batch)
        
        print(f"Batch {batch_idx}:")
        print(f"  x_batch shape: {x_batch.shape}") # (32, 3, 224, 224)
        print(f"  y_batch shape: {y_batch.shape}") # (32,)
        break # 我们只打印第一个批次验证一下
    
    
