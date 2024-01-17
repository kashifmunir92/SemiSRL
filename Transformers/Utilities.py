import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.variable as Variable
import numpy as np
import math


def clone(layer, number):
    return nn.ModuleList([layer for x in range(number)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


if __name__ == "__main__":
    tgt = torch.tensor([[1, 2, 3], [1, 2, 0]])
    print(subsequent_mask(tgt.size(-1)))
