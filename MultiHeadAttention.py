import torch
import torch.nn as nn
from torch.nn import functional as F
from Head import Head
import math


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_head,n_embd,dropout=0.2):
        super().__init__()
        assert n_embd%n_head==0, f'nhead={n_head} should divide nembd={n_embd} !'
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size,n_embd,dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class MultiHeadAttentionConcise(nn.Module):
    """
        Class implementing MaskedSelfAttention. Based on implementation of minGPT.
        TODO : compare to using nn.MultiheadAttention see which one is faster
        
        Parameters:
        n_embd : int
        n_head : int
        dropout : float

        Forward : Input shape should be B,L,D, where L<attn_size.
        Returns masked attention matrix of size B,L,L
    """

    def __init__(self,n_head,n_embd,dropout=0.1):
        super().__init__()
        assert n_embd%n_head==0, f"n_head should divide n_embd, but {n_embd}%{n_head}!=0"
        self.n_embd = n_embd
        self.n_head = n_head

        # Linear layer that generates the q,k,v tensors
        self.qkv_maker = nn.Linear(n_embd,3*n_embd)
        
        self.softmax = nn.Softmax(dim=3)
        self.attn_drop = nn.Dropout(dropout)

        self.out_proj = nn.Linear(n_embd,n_embd)
        self.out_drop = nn.Dropout(dropout)

        # Mask for self-attention
        self.register_buffer("mask", torch.tril(torch.ones(1000, 1000))
                                     .reshape(1, 1, 1000, 1000))
        

    
    def forward(self,x : torch.Tensor):
        B,L,D = x.shape

        q,k,v = (self.qkv_maker(x)).split(self.n_embd,dim=2) # (B,L,D)*3

        # Separate the Heads
        q=q.reshape(B,L,self.n_head,D//self.n_head).transpose(1,2) # (B,n_head,L,D')
        k=k.reshape(B,L,self.n_head,D//self.n_head).transpose(1,2) # (B,n_head,L,D')
        v=v.reshape(B,L,self.n_head,D//self.n_head).transpose(1,2) # (B,n_head,L,D')

        # Compute attn matrix
        att = (q @ k.transpose(-2,-1)) * 1./math.sqrt(q.shape[-1]) # (B,n_head,L,L)
        att = att.masked_fill_(self.mask[...,:L,:L]==0,float('-inf'))
        att = self.attn_drop(self.softmax(att))

        # Multiply by values
        att = att @ v # (B,n_head,L,D')
        att = att.transpose(1,2).reshape(B,L,D) # Reassemble heads

        att=self.out_drop(self.out_proj(att)) # Project

        return att
