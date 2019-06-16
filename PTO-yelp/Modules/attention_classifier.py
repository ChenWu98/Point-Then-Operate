import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gpu_wrapper
from Modules.subModules.attention import AttentionUnit
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class AttenClassifier(nn.Module):

    def __init__(self, emb_dim, dim_h, n_layers, dropout, bi):
        super(AttenClassifier, self).__init__()
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.dim_h = dim_h
        self.dropout = dropout
        self.n_dir = 2 if bi else 1

        self.Encoder = nn.GRU(input_size=self.emb_dim,
                              hidden_size=self.dim_h,
                              num_layers=self.n_layers,
                              dropout=self.dropout,
                              bidirectional=bi)
        self.Attention = AttentionUnit(query_dim=self.dim_h * self.n_dir,
                                       key_dim=self.dim_h * self.n_dir,
                                       atten_dim=self.dim_h)
        self.MLP = nn.Sequential(nn.Linear(self.dim_h * self.n_dir, 1),
                                 nn.Sigmoid())

    def forward(self, inp, l, null_mask):
        """

        :param inp: shape = (B, T, emb_dim)
        :param null_mask: shape = (B, T)
        :return:
        """
        B = inp.shape[0]
        T = inp.shape[1]
        inp = inp.transpose(0, 1)  # shape = (20, n_batch, emb_dim)
        packed_emb = pack(inp, l)
        outputs, h_n = self.Encoder(packed_emb)  # h_n.shape = (n_layers * n_dir, n_batch, dim_h)
        outputs = unpack(outputs, total_length=T)[0]  # shape = (20, n_batch, dim_h * n_dir)
        h_n = h_n.view(self.n_layers, self.n_dir, B, self.dim_h).transpose(1, 2).transpose(2, 3).contiguous().view(self.n_layers, B, -1)
        # shape = (n_layers, n_batch, dim_h * n_dir)
        h_n = h_n[-1, :, :]  # shape = (n_batch, dim_h * n_dir)
        context, att_weight = self.Attention(h_n,
                                             outputs.transpose(0, 1),
                                             null_mask)  # (n_batch, dim_h * n_dir), (n_batch, 20)
        cls = self.MLP(context).squeeze(1)  # shape = (n_batch, )

        return cls, att_weight
