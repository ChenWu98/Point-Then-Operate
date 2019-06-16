import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Delete(object):

    def __init__(self):
        pass

    def __call__(self, ori, pos):
        """

        :param ori: (B, T)
        :param pos: (B, )
        :return:
        """
        if ori.shape[0] == 0:
            return ori
        B = ori.shape[0]
        T = ori.shape[1]
        _del = torch.zeros_like(ori).copy_(ori)  # shape = (n_batch, 20)
        for b in range(B):
            # i   i+1   i+2  ... ->
            # i+1 i+2
            if pos[b] > -1:
                posb = pos[b]
            else:
                posb = 0
            _del[b, posb: -1] = _del[b, posb+1:]
        return _del

