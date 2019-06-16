import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gpu_wrapper, sample_2d
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class InsBehind(nn.Module):

    def __init__(self, dim_h, temperature, embedding, n_layers, dropout, bi, voc_size):
        super(InsBehind, self).__init__()
        self.voc_size = voc_size
        self.temperature = temperature
        self.Emb = nn.Embedding.from_pretrained(embedding, freeze=False)
        emb_dim = embedding.shape[1]
        self.Encoder = nn.GRU(input_size=emb_dim,
                              hidden_size=dim_h,
                              num_layers=n_layers,
                              dropout=dropout,
                              bidirectional=bi)
        self.MLP = nn.Linear(dim_h * (2 if bi else 1), self.voc_size)

    def forward(self, ori, pos, seq_len, sample):
        """

        :param ori: (B, T)
        :param pos: (B, )
        :param sample: bool
        :return:
        """
        if ori.shape[0] == 0:
            return ori, None, None

        B = ori.shape[0]
        T = ori.shape[1]
        inp = self.Emb(ori.transpose(0, 1))  # shape = (20, n_batch, emb_dim)
        packed_emb = pack(inp, seq_len)
        hid, h_n = self.Encoder(packed_emb)  # h_n.shape = (n_layers * n_dir, n_batch, dim_h)
        hid = unpack(hid, total_length=T)[0]  # shape = (20, n_batch, dim_h * n_dir)

        gthr_hid = []
        for b in range(B):
            gthr_hid.append(hid[pos[b], b, :])  # shape = (dim_h * n_dir, )
        gthr_hid = torch.stack(gthr_hid, dim=0)  # shape = (B, dim_h * n_dir)
        ib_lgt = self.MLP(gthr_hid)  # shape = (n_batch, voc_size)
        ib_prob = F.softmax(ib_lgt, dim=1)  # shape = (n_batch, voc_size)
        if not sample:
            # Choose max.
            sample_prob, sample_idx = torch.max(ib_prob, dim=1)  # (n_batch, ), (n_batch, )
        else:
            # Sample by multinomial distribution.
            sample_idx, sample_prob = sample_2d(probs=ib_prob, temperature=self.temperature)
        ib = torch.zeros_like(ori).copy_(ori)  # shape = (n_batch, 20)
        # i  i+1  i+2  ... ->
        # i       i+1  i+2 ...
        for b in range(B):
            ib[b, pos[b]+2:] = ib[b, pos[b]+1:-1]
        ib.scatter_(dim=1,
                    index=pos.unsqueeze(1)+1,  # shape = (n_batch, 1)
                    src=sample_idx.unsqueeze(1)  # shape = (n_batch, 1)
                    )  # shape = (n_batch, 20)

        return ib, sample_prob, ib_lgt