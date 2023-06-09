import torch
import torch.nn as nn
from torch.nn import functional as F
from Block import Block

def chamfer_dist(x, y):
    """
    Compute the Chamfer Distance between two sets of points
    
    Parameters:
    x, y: torch.Tensor with shape [batch_size, n_points, coordinates]

    Returns:
    torch.Tensor with shape [batch_size], the batched Chamfer Distance
    """
    assert x.shape==y.shape
    (B,N,D) = x.shape
    # Compute pairwise distance matrices
    # Divide by D because that's what mse loss does (averaging over all distances computed
    pairdistsquared = ((x[:,:,None,:]-y[:,None,:,:])**2).mean(dim=-1) # (B,N,N)

    # Find minimum distances
    min_distxy,_ = torch.min(pairdistsquared, dim=1,keepdim=False) # (B,N) min distance way 1
    min_distyx,_ = torch.min(pairdistsquared, dim=2, keepdim=False) # (B,N) min distance way 2

    chamfer_dist =.5*(min_distxy.mean(dim=-1) + min_distyx.mean(dim=-1)) # (B,) distance between the two cloud points
    
    return chamfer_dist.mean() # Average over batch !


class WaveModel(nn.Module):
    """
        Model that predicts mics location directly from the timeseries data.
        Not clear yet what the best way to proceed is yet, but for now uses
        the last 'token' embedding to make the prediciton of the source locations :
    """
    def __init__(self,mic_num,source_num,tot_timesteps,n_embd,n_hidden,n_head,n_layer,dropout=0.2):
        super().__init__()
        #wavedata:(6,512,3) (3,6,512)(512 3*6) (2,0,1)(3,1,2)
        
        self.position_embedding_table = nn.Embedding(tot_timesteps, n_embd)

        self.val_embed = nn.Linear(mic_num,n_embd)
        self.pos_embed = nn.Linear(mic_num*2,n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head,n_hidden,dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm（B,T,n_embd)

        self.lm_head = nn.Linear(n_embd, 2*source_num)

        self.tan=nn.Tanh()

        self.source_num=source_num
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,idx,targets):
        ## It should take as input the values (B,M,T), the 
        ## mic positions (B,M,2) and the targets (N,2) (optionally)
        ## OR, if you do it in the dataset, it will take
        ## (B,M,T,3) and (N,2) directly.

        #### CHANGES HERE TO BE DONE
        B,_,M,T = idx.shape #(B,3,M,T)
        B,N,_= targets.shape
        #tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        # x = None # (B,T,C) apply a linear layer to get from (B,M,T,3) To（B,T,C)
        #### HERE x should have shape (B,T,n_embed)
        idx=torch.permute(idx,(0,3,2,1)) # (B,T,M,3)
        vals = idx[...,0]
        pos = idx[:,0,:,1:].reshape(B,M*2) #(B,M*2)
        idx=idx.reshape(B,T,M*3)
        # idx=torch.reshape(idx,(B,T*M*3))
        pos_emb = self.position_embedding_table(torch.arange(T, device= idx.device)[None].expand(B,-1))# (B,T,C)

        idx=self.val_embed(vals)+self.pos_embed(pos)[:,None,:]+pos_emb # (B,T,n_embed)
        
        idx = self.blocks(idx) # (B,T,C)
        idx = self.ln_f(idx) # (B,T,C) 

        B,T,C=idx.shape
        #idx=idx.reshape(B,T*C)
        idx = self.lm_head(idx[:,-1]).reshape(B,self.source_num,2)#(B,n,2)
        predicts = torch.clamp(idx,min=0.,max=1.)
        
        #predicts = (self.tan(predicts)+1)/2
        # predicts=torch.permute(predicts,(0,2,1))
        ### CHANGES HERE AGAIN :
        # logits = x.reshape(n,2) # (B,T,vocab_size)
        ### We want to get an 'output' variable, which
        ### has size (B,N,2)
        # predicts = torch.permute(predicts,(0,2,1))
        ### Change this part to compute the mean squared error
        ### Using the targets
        if targets is None:
            loss = None
        else:
            # loss = F.mse_loss(predicts,targets)+F.mse_loss(predicts,idx)
            loss = chamfer_dist(predicts,targets)+F.mse_loss(predicts,idx)

        
        return predicts, loss
    

#============================================================================================
#==================FOR USE WITH PRETRAINING, DOESNT WORK GREAT IT SEEMS======================
#============================================================================================
class NoHeadModel(nn.Module):
    """
    Process the dataseries of mic recordings, but does not process the final output of the transformer
    layer. This can be attached to any head to make either source predictions, or next token prediction.
    """
    def __init__(self,mic_num,tot_timesteps,n_embd,n_hidden,n_head,n_layer,dropout=0.2):
        super().__init__()
        #wavedata:(6,512,3) (3,6,512)(512 3*6) (2,0,1)(3,1,2)

        self.position_embedding_table = nn.Embedding(tot_timesteps, n_embd)
        self.embedding_linear = nn.Linear(3*mic_num,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,n_hidden,dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm（B,T,n_embd)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,idx):
        """
            params : 
            idx : mic recordings, (B,3,M,T). Chan 0 : recorded value, chan 1,2 : position
        """
        B,_,M,T = idx.shape #(B,3,M,T)
        
        idx=torch.permute(idx,(0,3,2,1)) # (B,T,M,3)
        idx=idx.reshape(B,T,M*3)

        pos_emb = self.position_embedding_table(torch.arange(T, device= idx.device)[None].expand(B,-1))# (B,T,nembd)
        idx=self.embedding_linear(idx)+pos_emb # (B,T,nembd)
        
        idx = self.blocks(idx) # (B,T,nembd)
        idx = self.ln_f(idx) # (B,T,nembd) 

        
        return idx 


class PredictionModel(nn.Module):
    """
        Uses a NoHeadModel coupled with a linear head that acts on every timestep T,
        to produce predictions for the next token. 
        <WARNING :  to be used WITH masking in the attention heads>
    """
    def __init__(self,mic_num,tot_timesteps,n_embd,n_hidden,n_head,n_layer,dropout=0.2):
        super().__init__()
        self.transfo = NoHeadModel(mic_num,tot_timesteps,n_embd,n_hidden,n_head,n_layer,dropout)
        
        self.head_project = nn.Linear(n_embd,mic_num) # Project embeddings to (ordered) recordings of microphones
        # << TEST WITH PREDICTING MIC POSITIONS ALSO ? >>
    
    def forward(self,x,target=None):
        """
            x = (B,3,M,T)
            target = (B,3,M,T) translated by one
        """
        x=self.transfo(x) # (B,T,C)
        predict = self.head_project(x) # (B,T,M)

        if(target is not None):
            # Mse loss between prediction and target
            loss = F.mse_loss(predict,target.permute(0,3,2,1)[...,0])
        
        return predict,loss

    def get_trained_transfo(self):
        return self.transfo


class SourcePredictModel(nn.Module):
    """
        Predicts the number of sources, from the last embedding of the time-series. Works similarly 
        to WaveModel, but is designed to be used with a pretrained NoHeadModel.
    """
    def __init__(self,transf_model,n_sources,n_hidden):
        super().__init__()
        self.transfo=transf_model

        n_embd = self.transfo.position_embedding_table.weight.shape[-1]
        self.head = nn.Sequential(nn.Linear(n_embd,n_hidden),nn.LeakyReLU(),nn.Linear(n_hidden,2*n_sources))

        self.n_sources = n_sources
    

    def freeze_body(self,frozen:bool):
        """
            Freezes the weights of the body of the transformer, such that
            only the head is trained. Call BEFORE assigning an optimizer.
        """
        require = not frozen

        for par in self.transfo.parameters():
            par.requires_grad=require
        
        if(frozen):
            self.transfo.eval()
        else :
            self.transfo.train()

        
    def forward(self,x,targets=None):
        x=self.transfo(x) # (B,T,nembd)

        x = self.head(x[:,-1]) # Apply head to last token (B,2*N)
        predict = torch.clamp(x,min=0.,max=1.)

        if targets is None:
            loss = None
        else:
            # predicts = predicts.reshape(B,2*N,T)
            # B, T, C = logits.shape

            targets = targets.reshape(-1,2*self.n_sources)
            loss = F.mse_loss(x, targets)+F.mse_loss(predict,x)
        return predict,loss

#============================================================================================
#=================OTHER EMBEDDINGS OF POSITIONS==============================================
#============================================================================================
class WaveModelTesting(nn.Module):
    """
        Model that predicts mics location directly from the timeseries data.
        Not clear yet what the best way to proceed is yet, but for now uses
        the last 'token' embedding to make the prediciton of the source locations :
    """
    def __init__(self,mic_num,source_num,tot_timesteps,n_embd,n_hidden,n_head,n_layer,dropout=0.2):
        super().__init__()
        #wavedata:(6,512,3) (3,6,512)(512 3*6) (2,0,1)(3,1,2)
        eff_n_embd = n_embd
        self.position_embedding_table = nn.Embedding(tot_timesteps, n_embd)

        self.val_embed = nn.Linear(mic_num,eff_n_embd)
        self.pos_embed = nn.Linear(mic_num*2,n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd,n_head,n_hidden,dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm（B,T,n_embd)

        self.lm_head = nn.Sequential(nn.Linear(n_embd*tot_timesteps,n_embd),nn.ReLU(),nn.Linear(n_embd, 2*source_num))

        self.tan=nn.Tanh()

        self.source_num=source_num
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,idx,targets):
        ## It should take as input the values (B,M,T), the 
        ## mic positions (B,M,2) and the targets (N,2) (optionally)
        ## OR, if you do it in the dataset, it will take
        ## (B,M,T,3) and (N,2) directly.

        #### CHANGES HERE TO BE DONE
        B,_,M,T = idx.shape #(B,3,M,T)
        B,N,_= targets.shape
        #tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        # x = None # (B,T,C) apply a linear layer to get from (B,M,T,3) To（B,T,C)
        #### HERE x should have shape (B,T,n_embed)
        idx=torch.permute(idx,(0,3,2,1)) # (B,T,M,3)
        vals = idx[...,0]
        pos = idx[:,0,:,1:].reshape(B,M*2) #(B,M*2)
        idx=idx.reshape(B,T,M*3)
        # idx=torch.reshape(idx,(B,T*M*3))
        pos_emb = self.position_embedding_table(torch.arange(T, device= idx.device)[None].expand(B,-1))# (B,T,C)

        idx=self.val_embed(vals)+self.pos_embed(pos)+pos_emb # (B,T,n_embed)
        
        idx = self.blocks(idx) # (B,T,C)
        idx = self.ln_f(idx) # (B,T,C) 

        B,T,C=idx.shape
        #idx=idx.reshape(B,T*C)
        
        idx = self.lm_head(idx).reshape(B,self.source_num,2)#(B,n,2)
        predicts = torch.clamp(idx,min=0.,max=1.)
        
        #predicts = (self.tan(predicts)+1)/2
        # predicts=torch.permute(predicts,(0,2,1))
        ### CHANGES HERE AGAIN :
        # logits = x.reshape(n,2) # (B,T,vocab_size)
        ### We want to get an 'output' variable, which
        ### has size (B,N,2)
        # predicts = torch.permute(predicts,(0,2,1))
        ### Change this part to compute the mean squared error
        ### Using the targets
        if targets is None:
            loss = None
        else:
            # loss = F.mse_loss(predicts,targets)+F.mse_loss(predicts,idx)
            loss = chamfer_dist(predicts,targets)+F.mse_loss(predicts,idx)

        
        return predicts, loss
    