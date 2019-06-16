import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gpu_wrapper, sample_2d
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Gate(nn.Module):

    def __init__(self, dim_h, temperature, embedding, n_layers, dropout, bi):
        super(Gate, self).__init__()
        self.temperature = temperature
        self.Emb = nn.Embedding.from_pretrained(embedding, freeze=False)
        emb_dim = embedding.shape[1]
        self.Encoder = nn.GRU(input_size=emb_dim,
                              hidden_size=dim_h,
                              num_layers=n_layers,
                              dropout=dropout,
                              bidirectional=bi)
        self.MLP = nn.Linear(dim_h * (2 if bi else 1), 4)

    def forward(self, ori, pos, seq_len, sample):
        """

        :param ori: (B, T)
        :param pos: (B, )
        :param sample: bool
        :return: (B, ); (B, ); (B, 4)
        """

        s_idx = [ix for ix, l in sorted(enumerate(seq_len.cpu()), key=lambda x: x[1], reverse=True)]
        res_idx = [a for a, b in sorted(enumerate(s_idx), key=lambda x: x[1])]

        B = ori.shape[0]
        T = ori.shape[1]
        inp = self.Emb(ori[s_idx, :].transpose(0, 1))  # shape = (20, n_batch, emb_dim)
        packed_emb = pack(inp, seq_len[s_idx])
        hid, h_n = self.Encoder(packed_emb)  # h_n.shape = (n_layers * n_dir, n_batch, dim_h)
        hid = unpack(hid, total_length=T)[0]  # shape = (20, n_batch, dim_h * n_dir)
        hid = hid[:, res_idx, :]  # shape = (20, n_batch, dim_h * n_dir)

        gthr_hid = []
        for b in range(B):
            gthr_hid.append(hid[pos[b], b, :])  # shape = (dim_h * n_dir, )
        gthr_hid = torch.stack(gthr_hid, dim=0)  # shape = (B, dim_h * n_dir)
        oprt_lgt = self.MLP(gthr_hid)  # shape = (n_batch, 4)
        oprt_prob = F.softmax(oprt_lgt, dim=1)  # shape = (n_batch, 4)
        if not sample:
            # Choose max.
            oprt_prob, oprt_idx = torch.max(oprt_prob, dim=1)  # (n_batch, ), (n_batch, )
        else:
            # Sample by multinomial distribution.
            oprt_idx, oprt_prob = sample_2d(probs=oprt_prob, temperature=self.temperature)
        return oprt_idx, oprt_prob, oprt_lgt
