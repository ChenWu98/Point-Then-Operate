import numpy as np
from numpy import linalg as LA
import torch
import pickle
from collections import Counter

from config import Config
import os
config = Config()


class Vocabulary(object):
    def __init__(self, vocab_file):
        with open(vocab_file, 'rb') as f:
            self.size, self.word2id, self.id2word = pickle.load(f)
        self.embedding = np.random.random_sample((self.size, config.emb_dim)) - 0.5

        if config.emb_f:
            print('Loading word vectors from', config.emb_f)
            with open(config.emb_f) as f:
                for line in f:
                    parts = line.split()
                    word = parts[0]
                    vec = np.array([float(x) for x in parts[1:]])
                    if word in self.word2id:
                        self.embedding[self.word2id[word]] = vec

        for i in range(self.size):
            self.embedding[i] /= LA.norm(self.embedding[i])

        self.embedding = torch.FloatTensor(self.embedding)


def build_vocab(data, path, min_occur=5):
    if os.path.exists(path):
        return
    id2word = ['<pad>', '<go>', '<eos>', '<unk>']
    word2id = {tok: ix for ix, tok in enumerate(id2word)}

    words = [word for sent in data for word in sent]
    cnt = Counter(words)
    for word in cnt:
        if cnt[word] >= min_occur:
            word2id[word] = len(word2id)
            id2word.append(word)
    vocab_size = len(word2id)
    with open(path, 'wb') as f:
        pickle.dump((vocab_size, word2id, id2word), f, pickle.HIGHEST_PROTOCOL)  # TODO: check if python3 fits.
