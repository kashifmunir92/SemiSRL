import torch
import torch.nn.functional as F
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super(Generator, self).__init__()
        self.line = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.line(x), dim=-1)
