import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gpu_wrapper


class Discriminator(nn.Module):

    def __init__(self, kernels, conv_dim, D, dim_h, dropout):
        super(Discriminator, self).__init__()
        self.kernels = kernels  # List of ints.
        self.conv_dim = conv_dim  # Int.
        self.dim_h = dim_h
        self.D = D
        if self.D == 2:
            for i, kernel in enumerate(self.kernels):
                setattr(self, 'block{}'.format(i), nn.Sequential(
                    nn.Conv2d(1, conv_dim, kernel_size=(kernel, self.dim_h), padding=(0, 0), stride=1),
                    nn.LeakyReLU(0.01)
                ))
        else:
            raise NotImplementedError()
            for i, kernel in enumerate(self.kernels):
                setattr(self, 'block{}'.format(i), nn.Sequential(
                    nn.Conv1d(self.dim_h, self.conv_dim, kernel_size=kernel, padding=kernel // 2, stride=1),
                    nn.LeakyReLU(0.01)
                ))

        self.Dropout = nn.Dropout(dropout)
        self.FC = nn.Linear(len(self.kernels) * self.conv_dim, 1)

    def forward(self, x):
        """

        :param x: shape = (n_batch, max_len, dim_h)
        :return: shape = (n_batch, 1)
        """
        if self.D == 2:
            x = x.unsqueeze(1)  # shape = (n_batch, 1, max_len, dim_h)
            outputs = []
            for i, kernel in enumerate(self.kernels):
                h = getattr(self, 'block{}'.format(i))(x)  # shape = (n_batch, conv_dim, max_len, 1)
                pooled, _ = torch.max(h, dim=2)  # shape = (n_batch, conv_dim, 1)
                pooled = pooled.squeeze(2)  # shape = (n_batch, conv_dim)
                outputs.append(pooled)
            outputs = torch.cat(outputs, dim=1)  # shape = (n_batch, n * conv_dim)
            outputs = self.Dropout(outputs)  # shape = (n_batch, n * conv_dim)
            dis = self.FC(outputs)  # shape = (n_batch, 1)
            return dis
        else:
            raise NotImplementedError()
            x = x.transpose(1, 2)  # shape = (n_batch, dim_h, max_len)
            outputs = []
            for i, kernel in enumerate(self.kernels):
                h = getattr(self, 'block{}'.format(i))(x)  # shape = (n_batch, conv_dim, max_len)
                if kernel % 2 == 0:
                    h = h[:, :, :-1]  # Clip to VALID.
                pooled, _ = torch.max(h, dim=2)  # shape = (n_batch, conv_dim)
                outputs.append(pooled)
            outputs = torch.cat(outputs, dim=1)  # shape = (n_batch, n*conv_dim)
            outputs = self.Dropout(outputs)  # shape = (n_batch, n*conv_dim)
            dis = self.FC(outputs)  # shape = (n_batch, 1)
            return dis

