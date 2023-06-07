import torch
import torch.nn as nn
from torch.nn import functional as F
from Trainer import Trainer

from WaveDataset import WaveDataset

from torch.utils.data.dataloader import DataLoader
from Trainer import Trainer
from WaveModel import WaveModel


mydataset = WaveDataset('full')
# M,N,n_embd,n_hidden,n_head,n_layer,dropout=0.2
model = WaveModel(mic_num=6,source_num=1,tot_timesteps=512,n_embd=20,n_hidden=120,n_head=4,n_layer=2,dropout=0.1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
m=model.to(device)
train=Trainer(model,mydataset,eval_iters=5,batch_size=1,device=device)
train.to('cuda')

print('device : ', device)

train.model.train()
train.optimization(epochs=30,eval_interval=2,lr=1e-3,loadingbar=False)