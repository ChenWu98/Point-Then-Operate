import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
config = Config()


class AttentionUnit(nn.Module):

    def __init__(self, query_dim, key_dim, atten_dim):
        super(AttentionUnit, self).__init__()

        self.atten_dim = atten_dim
        self.W = nn.Linear(query_dim, atten_dim)  # Applied on query.
        self.U = nn.Linear(key_dim, atten_dim)  # Applied on keys.
        self.v = nn.Linear(atten_dim, 1)

    def forward(self, queries, keys, null_mask):
        """
        Compute the attention of each query on each key.

        :param queries: shape = (n_batch, query_dim)
        :param keys: shape = (n_batch, n_keys, key_dim); each query has n_keys keys.
        :param null_mask: shape = (n_batch, n_keys). bool. True where the key is null.
        :return: attened_keys: shape = (n_batch, key_dim)
        """

        # Basic statistics.
        n_keys = keys.shape[1]
        assert queries.shape[0] == keys.shape[0]

        t_key = self.U(keys)  # shape = (n_batch, n_keys, atten_dim)
        t_query = self.W(queries)  # shape = (n_batch, atten_dim)
        t_query = t_query.unsqueeze(1).expand(-1, n_keys, -1)  # shape = (n_batch, n_keys, atten_dim)
        alpha = self.v(torch.tanh(t_query + t_key)).squeeze(2)  # shape = (n_batch, n_keys)
        alpha.masked_fill_(null_mask, -float('inf'))
        att_weight = F.softmax(alpha, dim=1)  # shape = (n_batch, n_keys)
        attened_keys = torch.bmm(att_weight.unsqueeze(1), keys).squeeze(1)  # shape = (n_batch, key_dim)

        return attened_keys, att_weight


class AttentionUnit_v2(nn.Module):

    def __init__(self, query_dim, key_dim):
        super(AttentionUnit_v2, self).__init__()

        self.linear_in = nn.Linear(query_dim, key_dim)

    def prepare_keys(self, keys):
        return keys  # shape = (n_queries, n_keys, key_dim)

    def forward(self, queries, keys, trans_keys, null_mask):
        """
        Compute the attention of each query on each key.

        :param queries: shape = (n_batch, query_dim)
        :param keys: shape = (n_batch, n_keys, key_dim); each query has n_keys keys.
        :param trans_keys: shape = (n_batch, n_keys, atten_dim); for efficiency.
        :param null_mask: shape = (n_batch, n_keys). bool. True where the key is null.
        :return: attened_keys: shape = (n_batch, key_dim)
        """

        # Basic statistics.
        assert queries.shape[0] == keys.shape[0] == trans_keys.shape[0]

        trans_queries = self.linear_in(queries).unsqueeze(2)  # shape = (n_batch, key_dim, 1)
        alpha = torch.bmm(keys, trans_queries).squeeze(2)  # shape = (n_batch, n_keys)
        alpha.masked_fill_(null_mask, -float('inf'))

        alpha_sm = F.softmax(alpha, dim=1)  # shape = (n_batch, n_keys)
        # print(alpha_expsum)
        # print(alpha_sm)
        attened_keys = torch.bmm(alpha_sm.unsqueeze(1), keys).squeeze(1)  # shape = (n_batch, key_dim)

        return attened_keys, alpha_sm