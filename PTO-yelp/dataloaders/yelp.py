from torch.utils import data
import torch
import os
from collections import defaultdict
import numpy as np
from utils.vocab import Vocabulary, build_vocab
import random
from nltk import word_tokenize


class Yelp(data.Dataset):
    """The Yelp dataset."""

    def __init__(self, mode, noisy_for_train):
        self.mode = mode
        self.root = os.path.join('../data', 'yelp')
        voc_f = os.path.join('../data/yelp', 'yelp.vocab')
        if self.mode == 'dev':
            self.max_len = 30
        else:
            self.max_len = 20
        self.noisy = self.mode == 'train' and noisy_for_train

        # Load data from domain 0 and domain 1.
        path0 = os.path.join(self.root, 'sentiment.{}.0'.format(mode))
        data0 = []
        self.remove0 = []
        with open(path0) as f:
            for i, line in enumerate(f):
                sent = line.split()
                if 4 < len(sent) < self.max_len:
                    data0.append(sent)
                else:
                    self.remove0.append(i)
        print('{}/{} removed from domain 0'.format(len(self.remove0), len(self.remove0) + len(data0)))
        path1 = os.path.join(self.root, 'sentiment.{}.1'.format(mode))
        data1 = []
        self.remove1 = []
        with open(path1) as f:
            for i, line in enumerate(f):
                sent = line.split()
                if 4 < len(sent) < self.max_len:
                    data1.append(sent)
                else:
                    self.remove1.append(i)
        print('{}/{} removed from domain 1'.format(len(self.remove1), len(self.remove1) + len(data1)))
        self.l0 = len(data0)
        self.l1 = len(data1)
        # Make up for the same length.
        if len(data0) < len(data1):
            data0 = makeup(data0, len(data1))
        if len(data1) < len(data0):
            data1 = makeup(data1, len(data0))
        assert len(data0) == len(data1)
        self.data0 = data0
        self.data1 = data1

        if self.mode == 'dev':
            self.max_len += 5

        # Load vocabulary.
        print('----- Loading vocab -----')
        self.vocab = Vocabulary(voc_f)
        print('vocabulary size:', self.vocab.size)
        self.pad = self.vocab.word2id['<pad>']
        self.go = self.vocab.word2id['<go>']
        self.eos = self.vocab.word2id['<eos>']
        self.unk = self.vocab.word2id['<unk>']

    def get_references(self):

        assert self.mode == 'test', 'Only test mode support get_references().'
        path0 = os.path.join(self.root, 'reference.0')
        path1 = os.path.join(self.root, 'reference.1')
        ref0 = []
        ori0 = []
        ref1 = []
        ori1 = []
        with open(path0) as f:
            for i, line in enumerate(f):
                if i in self.remove0:
                    continue
                ori, ref = line.split('\t')
                ori = ori.split()
                ref = word_tokenize(ref.lower())
                ori0.append(ori)
                ref0.append(ref)
        with open(path1) as f:
            for i, line in enumerate(f):
                if i in self.remove1:
                    continue
                ori, ref = line.split('\t')
                ori = ori.split()
                ref = word_tokenize(ref.lower())
                ori1.append(ori)
                ref1.append(ref)
        ori0 = [[w if w in self.vocab.word2id else self.vocab.id2word[self.unk] for w in sent] for sent in ori0]
        ref0 = [[w if w in self.vocab.word2id else self.vocab.id2word[self.unk] for w in sent] for sent in ref0]
        ori1 = [[w if w in self.vocab.word2id else self.vocab.id2word[self.unk] for w in sent] for sent in ori1]
        ref1 = [[w if w in self.vocab.word2id else self.vocab.id2word[self.unk] for w in sent] for sent in ref1]
        return ori0, ref0, ori1, ref1

    def get_val_ori(self):
        assert self.mode == 'dev'
        ori_0 = [[w if w in self.vocab.word2id else self.vocab.id2word[self.unk] for w in sent] for sent in self.data0[:self.l0]]
        ori_1 = [[w if w in self.vocab.word2id else self.vocab.id2word[self.unk] for w in sent] for sent in self.data1[:self.l1]]
        return ori_0, ori_1

    def process_sent(self, sent):
        l = len(sent)
        sent_id = [self.vocab.word2id[w] if w in self.vocab.word2id else self.unk for w in sent]
        padding = [self.pad] * (self.max_len - l)
        _sent_id = noise(sent_id, self.unk, word_drop=0.3, k=1) if self.noisy else sent_id  # TODO: changed here.
        bare = torch.LongTensor(_sent_id + padding)  # shape = (20, )
        go = torch.LongTensor([self.go] + sent_id + padding)  # shape = (21, )
        eos = torch.LongTensor(sent_id + [self.eos] + padding)  # shape = (21, )
        return bare, go, eos, torch.LongTensor([l]).squeeze()

    def __getitem__(self, index):
        sent0 = self.data0[index]
        sent1 = self.data1[index]
        bare_0, go_0, eos_0, len_0 = self.process_sent(sent0)
        bare_1, go_1, eos_1, len_1 = self.process_sent(sent1)
        return bare_0, go_0, eos_0, len_0, bare_1, go_1, eos_1, len_1

    def __len__(self):
        return len(self.data0)


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