import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from utils.utils import gpu_wrapper


class BackLoss(nn.Module):
    def __init__(self, reduce):
        super(BackLoss, self).__init__()
        self.reduce = reduce

    def forward(self, lgt, gt):
        if lgt is None:
            return torch.zeros_like(gt).float()
        return F.cross_entropy(lgt, gt, reduce=self.reduce)
