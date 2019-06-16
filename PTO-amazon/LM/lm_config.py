import os
from functools import reduce
import torch
import glob
import time
from config import Config
config = Config()


class Config(object):

    def __init__(self):

        self.dataset = config.dataset
        self.sentiment = [0, 1][1]
        self.direction = ['forward', 'backward'][0]

        self.model = ['LSTM', 'QRNN', 'GRU'][0]
        self.emsize = 400
        self.nhid = 1150
        self.nlayers = 3
        self.lr = 30
        self.clip = 0.25
        self.epochs = 500
        self.batch_size = 20
        self.bptt = 70
        self.dropout = 0.4
        self.dropouth = 0.25
        self.dropouti = 0.4
        self.dropoute = 0.1
        self.wdrop = 0.5
        self.nonmono = 5
        self.log_interval = 200
        self.seed = 141
        self.gpu = torch.cuda.is_available()
        self.save = 'LM/saved_models/{}_{}_{}.pt'.format(self.dataset, self.sentiment, self.direction)
        self.alpha = 2
        self.beta = 1
        self.wdecay = 1.2e-6
        self.resume = ''
        self.optimizer = ['sgd', 'adam'][0]
        self.when = [-1]
        self.tied = [False, True][1]
