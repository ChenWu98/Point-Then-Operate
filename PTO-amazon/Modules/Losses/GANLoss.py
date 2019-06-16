import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gpu_wrapper

from config import Config

config = Config()


class GANLoss(nn.Module):
    def __init__(self, gan_type):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type

    def forward(self, input, target_is_real):

        if self.gan_type == 'LSGAN':
            if target_is_real:
                return torch.pow(F.sigmoid(input) - 1, 2).mean()
            else:
                return torch.pow(F.sigmoid(input), 2).mean()
        elif self.gan_type == 'vanillaGAN':
            input = input.view(-1)
            if target_is_real:
                return F.binary_cross_entropy_with_logits(input,
                                                          gpu_wrapper(Variable(torch.ones(input.shape[0]))))
            else:
                return F.binary_cross_entropy_with_logits(input,
                                                          gpu_wrapper(Variable(torch.zeros(input.shape[0]))))
        elif self.gan_type == 'WGAN_hinge':
            if target_is_real:
                return F.relu(1.0 - input).mean()
            else:
                return F.relu(input + 1.0).mean()
        else:
            raise ValueError()
