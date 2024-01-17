import torch
import torch.nn as nn
import copy
from Encoder import Encoder
from Decoder import Decoder
from Transformer import Transformer
from Generator import Generator
from PositionEmbedding import PositionEmbedding
import MultiHeadAttention as MultiHeadAttention
import PositionwiseFeedForward as PositionwiseFeedForward
import EncoderLayer
import DecoderLayer
from Embedding import Embedding


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention.MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward.PositionwiseFeedForward(
        d_model, d_ff, dropout)
    position = PositionEmbedding(d_model, dropout)
    model = Transformer(
        Encoder(EncoderLayer.EncoderLayer(
            d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer.DecoderLayer(d_model, c(attn), c(attn),
                                          c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab), c(position)),
        nn.Sequential(Embedding(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
