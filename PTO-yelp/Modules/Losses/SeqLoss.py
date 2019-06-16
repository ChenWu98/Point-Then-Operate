import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gpu_wrapper


class SeqLoss(nn.Module):
    def __init__(self, voc_size, pad, end, unk):
        super(SeqLoss, self).__init__()
        self.voc_size = voc_size
        self.word_weight = gpu_wrapper(torch.ones(voc_size))
        self.word_weight[pad] = 0.
        self.word_weight[end] = 1.0
        self.word_weight[unk] = 1.0

    def forward(self, logits, gts):
        """
        Summed over all positions and averaged by n_batch.
        :param logits: (B, T, V)
        :param gts: (B, T)
        :return: Scalar.
        """
        n_batch = gts.shape[0]

        return F.cross_entropy(logits.contiguous().view(-1, self.voc_size), gts.view(-1), weight=self.word_weight)
