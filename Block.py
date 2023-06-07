import torch
import torch.nn as nn
from torch.nn import functional as F
from MultiHeadAttention import MultiHeadAttention
from FeedFoward import FeedFoward
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self,n_embd, n_head,n_hidden,dropout=0.2):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(n_head,n_embd,dropout)
        self.ffwd = FeedFoward(n_embd,n_hidden,dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
