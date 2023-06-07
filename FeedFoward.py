import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, n_hidden,dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)