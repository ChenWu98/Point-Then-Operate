import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
args = parser.parse_args()


###############################################################################
# Load data
###############################################################################

import os

eval_batch_size = 10
test_batch_size = 1

###############################################################################
# Training code
###############################################################################



# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000



