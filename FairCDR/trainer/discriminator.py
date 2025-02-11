import torch
import torch.nn as nn
import torch.nn.functional as F

class GenderDiscriminator(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super(GenderDiscriminator, self).__init__()
        self.embed_dim=embed_dim
        self.network = nn.Sequential(
        nn.Linear(self.embed_dim, self.embed_dim * 2, bias=True),
        nn.LeakyReLU(0.2),
        nn.Dropout(p=0.3),
        nn.Linear(self.embed_dim * 2, self.embed_dim, bias=True),
        nn.LeakyReLU(0.2),
        nn.Linear(self.embed_dim, num_classes, bias=True)
)
    def forward(self, x):
        logits = self.network(x)         
        return logits
