import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from utils.utils import gpu_wrapper


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, sample_probs, reward, mask=None):
        """

        :param sample_probs: shape = (n_batch, *)
        :param mask: shape = (n_batch, *) or None
        :param reward: shape = (n_batch, )
        :return:
        """
        if sample_probs is None:
            return gpu_wrapper(torch.zeros([1]).squeeze(0))
        sample_probs = sample_probs.contiguous().view(-1)
        sample_logprobs = torch.log(sample_probs)
        reward = reward.contiguous().view(-1)
        if mask is not None:
            mask = mask.float().contiguous().view(-1)
            output = - sample_logprobs * reward * mask
            output = torch.sum(output) / torch.sum(mask)
        else:
            output = - sample_logprobs * reward
            output = output.mean()
        return output
