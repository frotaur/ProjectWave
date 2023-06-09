import torch
import torch.nn as nn
from torch.nn import functional as F
from WaveModel import WaveModel, PredictionModel
from WaveDataset import WaveDataset
from WaveValueDataset import WaveValueDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import datetime, pathlib,os


class Trainer:
    """
        Helper class, to train WaveModels.

        params :
        wave_model : Model to be trained. Should be of the WaveModel class.
        waveDataset : dataset to be trained on. Its componenets (max_length, max_mics) should
        map the parameters of the wave_model, will crash otherwise.
        batch_size : batch size
        run_name : str, name of the run, used when saving the state of the model.
        device : str, device to be trained on.
    """
    def __init__(self,wave_model : WaveModel,waveDataset:WaveDataset,batch_size=16,run_name=None,device='cpu'):
        self.model= wave_model
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.device = device
        self.batch_size = batch_size
        self.waveDataset = waveDataset

        train_size = int(0.9*len(waveDataset)) 
        val_size = len(waveDataset)-train_size

        train_dataset,val_dataset=torch.utils.data.random_split(waveDataset,[train_size,val_size])
        self.trainLoader = DataLoader(train_dataset,batch_size=batch_size,num_workers=0)
        self.valLoader = DataLoader(val_dataset,batch_size=batch_size,num_workers=0)
        self.run_name = run_name
        if self.run_name is None:
            now= datetime.datetime.now()
            self.run_name = now.strftime("%m_%d-%H-%M-%S")

    def save_weights(self):
        curfold = pathlib.Path(__file__).parent
        torch.save(self.model.state_dict(),os.path.join(curfold,self.run_name+".pt"))
        print('Save successful')

    def load_weights(self,weights_path):
        self.model.load_state_dict(torch.load(weights_path))
        print('Load successful')

    def to(self,device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model.to(device)
        self.device=device

    
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for key,loader in [('train',self.trainLoader), ('val',self.valLoader)]:
            losses = []
            for data,target in loader:
                data=data.to(self.device)
                target=target.to(self.device)
                predicts, loss = self.model(data,target)
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

            for data,target in tqdm(tloader) :
                self.optimizer.zero_grad(set_to_none=True)
                target=target.to(self.device)
                data = data.to(self.device)

                predicts, loss = self.model(data,target)
                loss.backward()

                self.optimizer.step()
                # << NOTE FOR ZILAN : I REMOVED THIS PART TEMPORARILY,
                # BUT IF YOU NEED IT YOU CAN RESTORE IT >>
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


# ====================USED FOR GPT-LIKE TRAINING, IGNORE========================
class SequenceTrainer:
    def __init__(self,PredictionModel:PredictionModel,waveDataset:WaveValueDataset,
                 eval_iters=5, batch_size=16,device='cpu'):
        self.eval_iters=eval_iters
        self.model= PredictionModel
        
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
    
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for key,loader in [('train',self.trainLoader), ('val',self.valLoader)]:
            losses = []
            for data,target in loader:
                data=data.to(self.device)
                target=target.to(self.device)

                _, loss = self.model(data,target)
                losses.append(loss.item())
                
            out[key] = torch.tensor(losses).mean()

        self.model.train()
        return out
    
    
    def optimization(self,epochs,eval_interval,lr=1e-3,loadingbar=True):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        

        tloader=self.trainLoader

        for epoch in range(epochs):
            # every once in a while evaluate the loss on train and val sets
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                losses = self.estimate_loss()
                print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Iterate over dataset :

            for data,target in tqdm(tloader) :
                self.optimizer.zero_grad()
                target=target.to(self.device)
                data = data.to(self.device)

                predicts, loss = self.model(data,target)
                loss.backward()

                self.optimizer.step()

