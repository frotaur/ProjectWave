import torch
import torch.nn as nn
from torch.nn import functional as F
from Head import Head

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