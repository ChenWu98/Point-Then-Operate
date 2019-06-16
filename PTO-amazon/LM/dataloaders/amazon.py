from torch.utils import data
import torch
import os
from collections import defaultdict
import numpy as np
from utils.vocab import Vocabulary, build_vocab
import random
from nltk import word_tokenize
from LM.lm_config import Config

config = Config()


class Amazon(object):
    """The Amazon dataset."""

    def __init__(self, mode, noisy_for_train, sentiment, direction):
        self.mode = mode
        self.root = os.path.join('../data', 'yelp')
        self.noisy = self.mode == 'train' and noisy_for_train

        # Load data from domain 0 and domain 1.
        path = os.path.join(self.root, 'sentiment.{}.{}'.format(mode, sentiment))

        # Load vocabulary.
        print('----- Loading vocab -----')
        self.vocab = Vocabulary('../data/amazon/amazon.vocab')
        print('vocabulary size:', self.vocab.size)
        self.pad = self.vocab.word2id['<pad>']
        self.go = self.vocab.word2id['<go>']
        self.eos = self.vocab.word2id['<eos>']
        self.unk = self.vocab.word2id['<unk>']

        # Tokenize file content
        with open(path, 'r') as f:
            ids = []
            for line in f:
                words = ['<go>'] + line.split() + ['<eos>']
                if direction == 'forward':
                    pass
                elif direction == 'backward':
                    words.reverse()
                else:
                    raise ValueError()
                for word in words:
                    ids.append(self.vocab.word2id[word] if word in self.vocab.word2id else self.unk)
        self.ids = torch.LongTensor(ids)  # (very_long, )
        self.ids = batchify(self.ids, config.batch_size, config)  # shape = (???, batch_size)


def makeup(_x, n):
    x = []
    for i in range(n):
        x.append(_x[i % len(_x)])
    return x


def noise(x, unk, word_drop=0.0, k=3):
    n = len(x)
    for i in range(n):
        if random.random() < word_drop:
            x[i] = unk

    # slight shuffle such that |sigma[i]-i| <= k
    sigma = (np.arange(n) + (k+1) * np.random.rand(n)).argsort()
    return [x[sigma[i]] for i in range(n)]


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.gpu:
        data = data.cuda()
    return data