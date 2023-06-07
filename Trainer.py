import torch
import torch.nn as nn
from torch.nn import functional as F
from WaveModel import WaveModel
from WaveDataset import WaveDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import time


class Trainer:
    def __init__(self,WaveModel,waveDataset:WaveDataset,eval_iters=5, batch_size=16,device='cpu'):
        self.eval_iters=eval_iters
        self.model= WaveModel
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.device = device
        self.batch_size = batch_size
        self.waveDataset = waveDataset

        train_size = int(0.9*len(waveDataset)) 
        val_size = len(waveDataset)-train_size

        train_dataset,val_dataset=torch.utils.data.random_split(waveDataset,[train_size,val_size])
        self.trainLoader = DataLoader(train_dataset,batch_size=batch_size,num_workers=0)
        self.valLoader = DataLoader(val_dataset,batch_size=batch_size,num_workers=0)

    def to(self,device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model.to(device)
        self.device=device

    # # data loading
    # def get_batch(self,split,train_dataset,val_dataset):
    #     data = train_dataset if split == 'train' else val_dataset
    #     ix = torch.randint(len(data), (self.batch_size,))
    #     waveorigin = torch.stack([data[i][0] for i in ix])
    # #     print(x.shape)
    #     wavedata = torch.stack([data[i][1] for i in ix])
    # #     print(y.shape)
    #     waveorigin, wavedata = waveorigin.to(self.device), wavedata.to(self.device)
    #     return waveorigin, wavedata
    
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for key,loader in [('train',self.trainLoader), ('val',self.valLoader)]:
            losses = []
            for X in loader:
                for i in range(2):
                    X[i]=X[i].to(self.device)
                predicts, loss = self.model(X[1],X[0])
                losses.append(loss.item())
                
            out[key] = torch.tensor(losses).mean()

        self.model.train()
        return out
    
    
    def optimization(self,epochs,eval_interval,lr=1e-3,loadingbar=True):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        

        tloader=self.trainLoader

        file_path_1='./Predicted origins.txt'
        file_path_2='./True origins.txt'
        for epoch in range(epochs):
            flag=0
            # every once in a while evaluate the loss on train and val sets
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                losses = self.estimate_loss()
                print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if epoch == epochs-1:
                flag=1
            # Iterate over dataset :

            for target,data in tqdm(tloader) :
                self.optimizer.zero_grad(set_to_none=True)
                target=target.to(self.device)
                data = data.to(self.device)

                predicts, loss = self.model(data,target)
                loss.backward()

                self.optimizer.step()
                #print(f'time for step {t-time.time()}')
                #t=time.time()
                # if flag==1:
                #     print("Predicted origins:",predicts-target)
                #     with open(file_path_1, 'a') as file:
                #         predicts_np=predicts.cpu().detach().numpy()
                #         file.write(str(predicts_np) + '\n')
                #     with open(file_path_2, 'a') as file:
                #         true_np=data.cpu().detach().numpy()
                #         file.write(str(true_np) + '\n')
                    #print(f'time to write : {t-time.time()}')
                
                #t=time.time()