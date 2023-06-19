import torch
import torch.nn as nn
from torch.nn import functional as F
from Trainer import Trainer

from WaveDataset import WaveDataset

from torch.utils.data.dataloader import DataLoader

from Trainer import Trainer
from WaveModel import WaveModel
# block_size = 512

mydataset = WaveDataset('2sources_30plusmics_240k',max_length=512,max_mics=8)
# automatically take correct number of mic for model
max_length,mic_num,source_num = mydataset.get_config()

model = WaveModel(mic_num=mic_num,source_num=source_num,
tot_timesteps=max_length,
n_embd=180,n_hidden=560,
n_head=10,n_layer=10,dropout=0.1)
device = 'cuda:0'

m=model.to(device)
train=Trainer(model,mydataset,run_name = '240k_8mics_2sourc_thin_2',
              batch_size=100,device=device)

train.to(device)

print('device : ', device)

train.model.train()
train.optimization(epochs=200,eval_interval=2,lr=1e-4,loadingbar=False)