import torch
import numpy as np
import torch.nn.functional as F

from config import Config
config = Config()


def gpu_wrapper(item):
    if config.gpu:
        # print(item)
        return item.cuda()
    else:
        return item


def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
            for sent in sents]


def strip_pad(sents):
    return [sent[:sent.index('<pad>')] if '<pad>' in sent else sent
            for sent in sents]


def gumbel_softmax(logits, gamma, eps=1e-20):
    """ logits.shape = (..., voc_size) """
    U = torch.zeros_like(logits).uniform_()
    G = -torch.log(-torch.log(U + eps) + eps)
    return F.softmax((logits + G) / gamma, dim=-1)


def pretty_string(flt):
    ret = '%.4f' % flt
    if flt > 0:
        ret = "+" + ret
    return ret


def sample_2d(probs, temperature):
    """probs.shape = (n_batch, n_choices)"""
    if temperature != 1:
        temp = torch.exp(torch.div(torch.log(probs + 1e-20), config.temp_att))  # shape = (n_batch, 20)
    else:
        temp = probs
    sample_idx = torch.multinomial(temp, 1)  # shape = (n_batch, 1)
    sample_probs = probs.gather(1, sample_idx)  # shape = (n_batch, 1)
    sample_idx = sample_idx.squeeze(1)  # shape = (n_batch, )
    sample_probs = sample_probs.squeeze(1)  # shape = (n_batch, )
    return sample_idx, sample_probs