import torch
from torch.utils import data


class Data(data.Dataset):
    def __init__(self, scr_sentence, tgt_sentence):
        self.src = scr_sentence
        self.tgt = tgt_sentence

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]
