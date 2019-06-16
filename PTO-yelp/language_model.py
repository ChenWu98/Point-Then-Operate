import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from LM import model as MODEL
from LM.dataloaders.yelp import Yelp
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from utils.utils import gpu_wrapper

from LM.utils import batchify, get_batch, repackage_hidden
from LM.lm_config import Config

config = Config()

# Set the random seed manually for reproducibility.
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if config.gpu:
        torch.cuda.manual_seed(config.seed)


class LanguageModel(object):

    def __init__(self, dataset, direction, sentiment):
        self.dataset = dataset
        self.sentiment = sentiment
        self.direction = direction
        self.fn = 'LM/saved_models/{}_{}_{}.pt'.format(self.dataset, self.sentiment, self.direction)
        self.build()

    def build(self):
        print('----- Loading language model data -----')
        self.train_set = Yelp('train', False, config.sentiment, config.direction)
        self.test_set = Yelp('test', False, config.sentiment, config.direction)
        self.val_set = Yelp('dev', False, config.sentiment, config.direction)

        self.ntokens = self.train_set.vocab.size
        self.go = self.train_set.go
        self.eos = self.train_set.eos
        self.pad = self.train_set.pad
        self.word_weight = gpu_wrapper(torch.ones(self.ntokens))
        self.word_weight[self.pad] = 0.

        self.model = MODEL.RNNModel(config.model, self.ntokens, config.emsize, config.nhid, config.nlayers,
                                    config.dropout, config.dropouth, config.dropouti, config.dropoute, config.wdrop,
                                    config.tied)

    def train(self):
        from LM.splitcross import SplitCrossEntropyLoss
        self.criterion = None

        if not self.criterion:
            splits = []
            if self.ntokens > 500000:
                # One Billion
                # This produces fairly even matrix mults for the buckets:
                # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
                splits = [4200, 35000, 180000]
            elif self.ntokens > 75000:
                # WikiText-103
                splits = [2800, 20000, 76000]
            print('Using', splits)
            self.criterion = SplitCrossEntropyLoss(config.emsize, splits=splits, verbose=False)
        if config.gpu:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        self.params = list(self.model.parameters()) + list(self.criterion.parameters())
        total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in self.params if x.size())
        print('Model total parameters:', total_params)

        val_data = self.val_set.ids
        eval_batch_size = config.batch_size
        self.stored_loss = float('inf')
        best_val_loss = []

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            self.optimizer = None
            # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
            if config.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(self.params, lr=config.lr, weight_decay=config.wdecay)
            if config.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(self.params, lr=config.lr, weight_decay=config.wdecay)
            for epoch in range(1, config.epochs + 1):
                epoch_start_time = time.time()
                self.train_epoch(epoch)
                if 't0' in self.optimizer.param_groups[0]:
                    tmp = {}
                    for prm in self.model.parameters():
                        tmp[prm] = prm.data.clone()
                        prm.data = self.optimizer.state[prm]['ax'].clone()

                    val_loss2 = self.evaluate(val_data)
                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                          'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2),
                        val_loss2 / math.log(2)))
                    print('-' * 89)

                    if val_loss2 < self.stored_loss:
                        self.model_save(self.fn, self.model, self.criterion, self.optimizer)
                        print('Saving Averaged!')
                        self.stored_loss = val_loss2

                    for prm in self.model.parameters():
                        prm.data = tmp[prm].clone()

                else:
                    val_loss = self.evaluate(val_data, eval_batch_size)
                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                          'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                    print('-' * 89)

                    if val_loss < self.stored_loss:
                        self.model_save(self.fn, self.model, self.criterion, self.optimizer)
                        print('Saving model (new best validation)')
                        self.stored_loss = val_loss

                    if config.optimizer == 'sgd' and 't0' not in self.optimizer.param_groups[0] and (
                            len(best_val_loss) > config.nonmono and val_loss > min(best_val_loss[:-config.nonmono])):
                        print('Switching to ASGD')
                        self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=config.lr, t0=0, lambd=0.,
                                                          weight_decay=config.wdecay)

                    if epoch in config.when:
                        print('Saving model before learning rate decreased')
                        self.model_save('{}.e{}'.format(self.fn, epoch), self.model, self.criterion, self.optimizer)
                        print('Dividing learning rate by 10')
                        self.optimizer.param_groups[0]['lr'] /= 10.

                    best_val_loss.append(val_loss)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def model_load(self):
        with open(self.fn, 'rb') as f:
            self.model, self.criterion, self.optimizer = torch.load(f)

    def model_save(self, fn, model, criterion, optimizer):
        with open(fn, 'wb') as f:
            torch.save([model, criterion, optimizer], f)

    def test(self):
        # Load the best saved model.
        self.model_load()
        test_data = self.test_set.ids
        test_batch_size = config.batch_size

        # Run on test data.
        test_loss = self.evaluate(test_data, test_batch_size)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
        print('=' * 89)

    def train_epoch(self, epoch):
        # Turn on training mode which enables dropout.
        if config.model == 'QRNN':
            self.model.reset()
        total_loss = 0
        start_time = time.time()
        hidden = self.model.init_hidden(config.batch_size)
        batch, i = 0, 0
        train_data = self.train_set.ids
        while i < train_data.size(0) - 1 - 1:
            bptt = config.bptt if np.random.random() < 0.95 else config.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            # There's a very small chance that it could select a very long sequence length resulting in OOM
            # seq_len = min(seq_len, args.bptt + 10)

            lr2 = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / config.bptt
            self.model.train()
            data, targets = get_batch(train_data, i, config, seq_len=seq_len)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            self.optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs = self.model(data, hidden, return_h=True)
            raw_loss = self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets)

            loss = raw_loss
            # Activiation Regularization
            if config.alpha: loss = loss + sum(
                config.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if config.beta: loss = loss + sum(
                config.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if config.clip: torch.nn.utils.clip_grad_norm_(self.params, config.clip)
            self.optimizer.step()

            total_loss += raw_loss.data
            self.optimizer.param_groups[0]['lr'] = lr2
            if batch % config.log_interval == 0 and batch > 0:
                cur_loss = total_loss.item() / config.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    epoch, batch, len(train_data) // config.bptt, self.optimizer.param_groups[0]['lr'],
                                  elapsed * 1000 / config.log_interval, cur_loss, math.exp(cur_loss),
                                  cur_loss / math.log(2)))
                total_loss = 0
                start_time = time.time()
            ###
            batch += 1
            i += seq_len

    def inference_whole(self, data_source, seq_len):
        """

        :param data_source: shape = (batch_size, seq_len), not with <go> or <eos>.
        :param seq_len: shape = (batch_size, )
        :return: prob.shape = (batch_size, )
        """
        with torch.no_grad():
            if self.direction == 'forward':
                data_source = torch.cat([torch.zeros_like(seq_len).unsqueeze(1) + self.go, data_source], dim=1)
                data_source = data_source.t()
                batch_size = data_source.shape[1]
                max_len = data_source.shape[0] - 1
                self.model.eval()
                if config.model == 'QRNN':
                    self.model.reset()
                hidden = self.model.init_hidden(batch_size)
                output, hidden = self.model(data_source, hidden)  # output.shape = ((1+max_len) * batch_size, n_hidden)
                output = output.view(max_len + 1, batch_size, -1)  # shape = (1+max_len, batch_size, n_hidden)
                output = output[:-1, :, :]  # shape = (seq_len, batch_size, n_hidden)
                logit_all = self.model.decoder(output).view(-1, self.ntokens)  # shape = (max_len * batch_size, voc_size)

                nll = F.cross_entropy(logit_all, data_source[1:].contiguous().view(-1), weight=self.word_weight, reduction='none')
                nll = nll.view(-1, batch_size)  # shape = (max_len, batch_size)
                nll = nll.sum(0) / seq_len.float() # shape = (batch_size, )
                prob = torch.exp(-nll)  # shape = (batch_size, )
                return prob
            elif self.direction == 'backward':
                reversed_data_source = torch.zeros_like(data_source)  # shape = (batch_size, max_len)
                for B in range(reversed_data_source.shape[0]):
                    l = seq_len[B].item()
                    reversed_data_source[B, :l] = torch.flip(data_source[B, :l], dims=[0])
                reversed_data_source = torch.cat([torch.zeros_like(seq_len).unsqueeze(1) + self.eos,
                                                  reversed_data_source], dim=1)  # shape = (batch_size, max_len + 1)
                reversed_data_source = reversed_data_source.t()
                batch_size = reversed_data_source.shape[1]
                max_len = reversed_data_source.shape[0] - 1
                self.model.eval()
                if config.model == 'QRNN':
                    self.model.reset()
                hidden = self.model.init_hidden(batch_size)
                output, hidden = self.model(reversed_data_source,
                                            hidden)  # output.shape = ((1+max_len) * batch_size, n_hidden)
                output = output.view(max_len + 1, batch_size, -1)  # shape = (1+max_len, batch_size, n_hidden)
                output = output[:-1, :, :]  # shape = (seq_len, batch_size, n_hidden)
                logit_all = self.model.decoder(output).view(-1, self.ntokens)  # shape = (max_len, batch_size, voc_size)

                nll = F.cross_entropy(logit_all, reversed_data_source[1:].contiguous().view(-1), weight=self.word_weight, reduction='none')
                nll = nll.view(-1, batch_size)  # shape = (max_len, batch_size)
                nll = nll.sum(0) / seq_len.float()  # shape = (batch_size, )
                prob = torch.exp(-nll)  # shape = (batch_size, )
                return prob
            else:
                raise ValueError()

    def inference(self, data_source, index, seq_len):
        """

        :param data_source: shape = (batch_size, seq_len), not with <go> or <eos>.
        :param index: shape = (batch_size, )
        :param seq_len: shape = (batch_size, )
        :return: prob.shape = (batch_size, )
        """
        with torch.no_grad():
            if self.direction == 'forward':
                data_source = torch.cat([torch.zeros_like(index).unsqueeze(1) + self.go, data_source], dim=1)
                data_source = data_source.t()
                batch_size = data_source.shape[1]
                max_len = data_source.shape[0] - 1
                self.model.eval()
                if config.model == 'QRNN':
                    self.model.reset()
                hidden = self.model.init_hidden(batch_size)
                output, hidden = self.model(data_source, hidden)  # output.shape = ((1+max_len) * batch_size, n_hidden)
                output = output.view(max_len + 1, batch_size, -1)  # shape = (1+max_len, batch_size, n_hidden)
                output = output[:-1, :, :]  # shape = (seq_len, batch_size, n_hidden)
                logit_all = self.model.decoder(output)  # shape = (max_len, batch_size, voc_size)
                prob_all = F.softmax(logit_all, dim=2)  # shape = (max_len, batch_size, voc_size)
                prob_word = prob_all.gather(2, data_source[1:, :].unsqueeze(2)).squeeze(
                    2)  # shape = (max_len, batch_size)
                prob_word = prob_word.t()  # shape = (batch_size, max_len)
                prob = prob_word.gather(1, index.unsqueeze(1)).squeeze(1)  # shape = (batch_size, )
                return prob
            elif self.direction == 'backward':
                reversed_data_source = torch.zeros_like(data_source)  # shape = (batch_size, max_len)
                for B in range(reversed_data_source.shape[0]):
                    l = seq_len[B].item()
                    reversed_data_source[B, :l] = torch.flip(data_source[B, :l], dims=[0])
                reversed_data_source = torch.cat([torch.zeros_like(index).unsqueeze(1) + self.eos,
                                                  reversed_data_source], dim=1)  # shape = (batch_size, max_len + 1)
                reversed_data_source = reversed_data_source.t()
                batch_size = reversed_data_source.shape[1]
                max_len = reversed_data_source.shape[0] - 1
                self.model.eval()
                if config.model == 'QRNN':
                    self.model.reset()
                hidden = self.model.init_hidden(batch_size)
                output, hidden = self.model(reversed_data_source, hidden)  # output.shape = ((1+max_len) * batch_size, n_hidden)
                output = output.view(max_len + 1, batch_size, -1)  # shape = (1+max_len, batch_size, n_hidden)
                output = output[:-1, :, :]  # shape = (seq_len, batch_size, n_hidden)
                logit_all = self.model.decoder(output)  # shape = (max_len, batch_size, voc_size)
                prob_all = F.softmax(logit_all, dim=2)  # shape = (max_len, batch_size, voc_size)
                prob_word = prob_all.gather(2, reversed_data_source[1:, :].unsqueeze(2)).squeeze(
                    2)  # shape = (max_len, batch_size)
                prob_word = prob_word.t()  # shape = (batch_size, max_len)
                prob = prob_word.gather(1, (seq_len - 1 - index).unsqueeze(1)).squeeze(1)  # shape = (batch_size, )
                return prob
            else:
                raise ValueError()

    def evaluate(self, data_source, batch_size=config.batch_size):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        if config.model == 'QRNN':
            self.model.reset()
        total_loss = 0
        hidden = self.model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, config.bptt):
            data, targets = get_batch(data_source, i, config, evaluation=True)
            output, hidden = self.model(data, hidden)
            total_loss += len(data) * self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss.item() / len(data_source)


if __name__ == '__main__':
    lm = LanguageModel(config.dataset, config.direction, config.sentiment)
    lm.train()
