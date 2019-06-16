import os
from functools import reduce
import torch
import glob
import math
import numpy as np


class Config(object):

    def __init__(self):

        self.dataset = 'amazon'
        self.train_mode = ['cls-only', 'pto', 'aux-cls-only'][1]

        # Model configuration.
        self.dim_h = 512
        self.n_layers = 1
        self.bidirectional = True
        self.temp_sub = 1
        self.temp_att = 0.05
        self.temp_gate = 5
        self.lambda_ins_conf = 0.1
        self.lambda_gate_conf = 0
        self.beta_att = [0.1, 0.1]
        self.lambda_lm = 7
        self.lambda_conf = 1
        self.lambda_att_conf = 0.05
        self.beta_xbar_conf = [0.6, 0.6]
        self.beta_gate_conf = [0.9, 0.9]
        self.beta_xbar_XE2 = [0.0, 0.0]
        self.beta_xbar_lm = [0.03, 0.03]
        self.beta_gate_lm = [0.03, 0.03]
        self.subtract_XE2 = [3, 3]

        # TextCNN.
        self.textCNN_conv_dim = 128
        self.textCNN_dropout = 0.5
        self.textCNN_kernels = [1, 2, 3, 4, 5, 6, 7]
        self.textCNN_lr = 5e-4

        # Training configuration.
        self.best_metric = None
        self.num_iters = 200000
        self.num_iters_decay = 0
        self.dropout = 0.5
        self.batch_size = 64
        self.emb_lr = 5e-4
        self.cls_lr = 5e-4
        self.gate_lr = 5e-4
        self.oprt_lr = 5e-4
        self.aux_cls_lr = 5e-4
        self.lm_lr = 1e-3
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.clip_norm = 30.0
        self.clip_value = float('inf')
        self.max_iter = (2, 2)
        self.s = (0.6, 2.7)
        self.cls_stop = (0.2, 0)

        # Test configuration.
        self.test_cls = 'TextCNN'

        # Data configuration.
        self.emb_f = ''
        self.emb_dim = 100

        # Miscellaneous.
        self.num_workers = 8
        self.use_tensorboard = True
        self.ROUND = 4
        self.seed = 0
        self.gpu = torch.cuda.is_available()

        # Step size.
        self.log_step = 10
        self.sample_step = 1000
        self.lr_decay_step = 1000

        # Directories.
        self.log_dir = 'outputs/logs'
        remove_all_under(self.log_dir)
        self.save_model_dir = 'outputs/saved_model'
        self.sample_dir = 'outputs/sampled_results'


def remove_all_under(directory):
    for file in glob.glob(os.path.join(directory, '*')):
        os.remove(file)
