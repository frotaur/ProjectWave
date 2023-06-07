import torch
import torch.nn as nn
from torch.nn import functional as F
from Block import Block



class WaveModel(nn.Module):
    def __init__(self,mic_num,source_num,tot_timesteps,n_embd,n_hidden,n_head,n_layer,dropout=0.2):
        super().__init__()
        #wavedata:(6,512,3) (3,6,512)(512 3*6) (2,0,1)(3,1,2)
        
        self.position_embedding_table = nn.Embedding(tot_timesteps, n_embd)
        self.embedding_linear = nn.Linear(3*mic_num,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,n_hidden,dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm（B,T,n_embd)

        self.lm_head = nn.Linear(n_embd, 2*source_num)

        self.tan=nn.Tanh()
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
        idx=idx.reshape(B,T,M*3)
        # idx=torch.reshape(idx,(B,T*M*3))
        pos_emb = self.position_embedding_table(torch.arange(T, device= idx.device)[None].expand(B,-1))# (B,T,C)

        idx=self.embedding_linear(idx)+pos_emb # (B,T,n_embed)
        
        idx = self.blocks(idx) # (B,T,C)
        idx = self.ln_f(idx) # (B,T,C) 

        B,T,C=idx.shape
        #idx=idx.view(B,T*C)
        idx = self.lm_head(idx[:,-1])#(B,2*n)
        predicts = torch.clamp(idx,min=0.,max=1.)
        
        #predicts = (self.tan(predicts)+1)/2
        # predicts=torch.permute(predicts,(0,2,1))
        ### CHANGES HERE AGAIN :
        # logits = x.view(n,2) # (B,T,vocab_size)
        ### We want to get an 'output' variable, which
        ### has size (B,N,2)
        # predicts = torch.permute(predicts,(0,2,1))
        ### Change this part to compute the mean squared error
        ### Using the targets
        if targets is None:
            loss = None
        else:
            # predicts = predicts.view(B,2*N,T)
            # B, T, C = logits.shape
            # predicts = predicts.view(B,2*N)
            targets = targets.view(B,2*N)
            # loss = F.cross_entropy(logits, targets)
            loss = F.mse_loss(predicts, targets)+F.mse_loss(predicts,idx)
        ###
        
        return predicts, loss
    

class NoHeadModel(nn.Module):
    """
    Process the dataseries of mic recordings, but does not process the final output of the transformer
    layer.
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
        
        idx=torch.permute(idx,(0,3,2,1)) # (B,T,3,M)
        idx=idx.view(B,T,M*3)

        pos_emb = self.position_embedding_table(torch.arange(T, device= idx.device)[None].expand(B,-1))# (B,T,nembd)
        idx=self.embedding_linear(idx)+pos_emb # (B,T,nembd)
        
        idx = self.blocks(idx) # (B,T,nembd)
        idx = self.ln_f(idx) # (B,T,nembd) 

        
        return idx 

class PredictionModel(nn.Module):
    def __init__(self,mic_num,tot_timesteps,n_embd,n_hidden,n_head,n_layer,dropout=0.2):
        self.transfo = NoHeadModel(mic_num,tot_timesteps,n_embd,n_hidden,n_head,n_layer,dropout)
        
        self.head_project = nn.Linear(n_embd,mic_num) # Project embeddings to (ordered) recordings of microphones
        # << TEST WITH PREDICTING MIC POSITIONS ALSO ? >>
