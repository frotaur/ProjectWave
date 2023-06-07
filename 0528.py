import torch

from WaveDataset import *

from torch.utils.data.dataloader import DataLoader

dataset=WaveDataset('full')
# print(dataset[0][0].shape)
for i in range(10):
    print(dataset[i][1])
# dataloader=DataLoader(dataset,batch_size=5)
# for x in dataloader:
#     print(x[0])
# y=torch.Tensor([0.7713, 0.4201],
#         [0.5405, 0.5459],
#         [0.3315, 0.1823],
#         [0.5782, 0.2363],
#         [0.8545, 0.2054],
#         [0.6424, 0.2597]])
# y=torch.randn(6,2)
# x=torch.cat([y,y],dim=1)

# import torch
# a=torch.randn(3,4) #随机生成一个shape（3，4）的tensor
# b=torch.randn(2,4) #随机生成一个shape（2，4）的tensor

# torch.cat([a,b],dim=0) 