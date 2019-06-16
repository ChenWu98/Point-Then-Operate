import os
import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from dataloaders.yelp import Yelp
from Modules.discriminator import Discriminator
from utils.utils import gpu_wrapper
from torch.utils.data import DataLoader
from utils.vocab import Vocabulary
from config import Config

config = Config()


class ClassifierEval(object):

    def __init__(self, test_cls, dataset):

        self.vocab = Vocabulary('../data/{}/{}.vocab'.format(dataset, dataset))
        self.Emb = nn.Embedding.from_pretrained(self.vocab.embedding, freeze=False)
        self.Emb = gpu_wrapper(self.Emb)
        if test_cls == 'TextCNN':
            self.C = Discriminator(kernels=config.textCNN_kernels,
                                   conv_dim=config.textCNN_conv_dim,
                                   dim_h=100,
                                   D=2,
                                   dropout=config.textCNN_dropout)
        else:
            raise ValueError()
        self.C = gpu_wrapper(self.C)

        self.train_set, self.test_set, self.val_set = None, None, None
        self.logger, self.optim, self.best_acc = None, None, 0
        self.iter_num = 0
        self.lr = config.textCNN_lr
        self.dataset = dataset
        self.model_name = test_cls + '-' + dataset
        self.noisy = True
        self.total_iters = 200000
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.batch_size = 64
        self.num_workers = 8
        self.ROUND = 4
        self.sample_step = 4000
        self.lr_decay_step = 1000
        self.num_iters_decay = 0
        self.max_len = 20

    def class_score(self, sents, labels):
        """

        :param sents: [[str x T] x N]
        :param labels: [int x N]
        :return: float, accuracy of classification.
        """
        self.C.train(mode=False)
        self.Emb.train(mode=False)
        with torch.no_grad():
            _size = 0
            _batch = []
            preds = []
            for sent in sents:
                _size += 1
                l = len(sent)
                if l > self.max_len:
                    sent = sent[:self.max_len]
                sent_id = [self.vocab.word2id[w] for w in sent]
                padding = [self.vocab.word2id['<pad>']] * (self.max_len - l)
                bare = gpu_wrapper(torch.LongTensor(sent_id + padding))  # shape = (20, )
                _batch.append(bare)
                if _size == self.batch_size:
                    _size = 0
                    batch = torch.stack(_batch, dim=0)  # shape = (n_batch, 20)
                    emb = self.Emb(batch)  # shape = (n_batch, 20, emb_dim)
                    cls = self.C(emb).squeeze(1)  # shape = (n_batch, )
                    pred = (cls > 0.5).float()  # shape = (n_batch, )
                    preds.append(pred)
                    _batch = []
            if _size != 0:
                batch = torch.stack(_batch, dim=0)  # shape = (n_batch, 20)
                emb = self.Emb(batch)  # shape = (n_batch, 20, emb_dim)
                cls = self.C(emb).squeeze(1)  # shape = (n_batch, )
                pred = (cls > 0.5).float()  # shape = (n_batch, )
                preds.append(pred)
            preds = torch.cat(preds, dim=0)  # shape = (N, )
            # print(' '.join([str(int(_)) for _ in preds]))
            labels = gpu_wrapper(torch.tensor(np.array(labels, dtype=np.float32)))  # shape = (N, )
            # print(preds)
            # print(labels)
            assert preds.shape[0] == labels.shape[0]
            n_wrong = torch.abs(preds - labels).sum().item()
            n_all = preds.shape[0]

        self.C.train(mode=True)
        self.Emb.train(mode=True)
        return (n_all - n_wrong) / n_all

    def save_step(self):
        path = os.path.join('utils', 'best-{}.ckpt'.format(self.model_name))
        path_emb = os.path.join('utils', 'best-{}-Emb.ckpt'.format(self.model_name))
        torch.save(self.C.state_dict(), path)
        torch.save(self.Emb.state_dict(), path_emb)
        print('Saved {} and its embedding\'s checkpoints into utils/ ...'.format(self.model_name))

    def restore_model(self):
        """Restore the trained classifier."""
        print('Loading the trained best {} and its embeddings...'.format(self.model_name))
        path = os.path.join('utils', 'best-{}.ckpt'.format(self.model_name))
        path_emb = os.path.join('utils', 'best-{}-Emb.ckpt'.format(self.model_name))
        self.C.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.Emb.load_state_dict(torch.load(path_emb, map_location=lambda storage, loc: storage))

    def zero_grad(self):
        self.optim.zero_grad()

    def update_lr(self):
        return
        self.lr *= (1 - float(config.lr_decay_step / config.num_iters_decay))
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr

    def test(self):
        self.restore_model()
        self.valtest(val_or_test="test")

    def pretrain(self):
        # assert not os.path.isfile(os.path.join('utils', 'best-{}.ckpt').format(self.model_name))
        # assert not os.path.isfile(os.path.join('utils', 'best-{}-Emb.ckpt').format(self.model_name))

        print('----- Loading data -----')
        self.train_set = Yelp('train', self.noisy)
        self.test_set = Yelp('test', self.noisy)
        self.val_set = Yelp('dev', self.noisy)
        print('The train set has {} items'.format(len(self.train_set)))
        print('The test set has {} items'.format(len(self.test_set)))
        print('The val set has {} items'.format(len(self.val_set)))

        # Set trainable parameters, according to the frozen parameter list.
        self.trainable = []
        for k, v in self.C.state_dict(keep_vars=True).items():
            # k is the parameter name; v is the parameter value.
            if v.requires_grad:
                self.trainable.append(v)
                print("[C Trainable:]", k)
            else:
                print("[C Frozen:]", k)
        for k, v in self.Emb.state_dict(keep_vars=True).items():
            # k is the parameter name; v is the parameter value.
            if v.requires_grad:
                self.trainable.append(v)
                print("[Emb Trainable:]", k)
            else:
                print("[Emb Frozen:]", k)

        # Build optimizer.
        self.optim = Adam(self.trainable, self.lr, [self.beta1, self.beta2])

        # Train.
        epoch = 0
        while True:
            self.train_epoch(epoch_idx=epoch)
            epoch += 1
            if self.iter_num >= self.total_iters:
                break
        self.test()

    def train_epoch(self, epoch_idx):
        loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.C.train(mode=True)
        self.Emb.train(mode=True)

        with tqdm(loader) as pbar:
            for data in pbar:
                self.iter_num += 1
                bare_0, _, _, len_0, label_0, bare_1, _, _, len_1, label_1 = self.preprocess_data(data)
                bare_emb_0 = self.Emb(bare_0)  # shape = (n_batch, 20, emb_dim); encoder input.
                bare_emb_1 = self.Emb(bare_1)  # shape = (n_batch, 20, emb_dim); encoder input.
                cls_0 = self.C(bare_emb_0).squeeze(1)  # shape = (n_batch, )
                cls_1 = self.C(bare_emb_1).squeeze(1)  # shape = (n_batch, )
                loss0 = F.binary_cross_entropy_with_logits(cls_0, label_0)
                loss1 = F.binary_cross_entropy_with_logits(cls_1, label_1)
                loss = loss0 + loss1

                # ----- Backward and optimize -----
                self.zero_grad()
                loss.backward()
                self.optim.step()

                pbar.set_description(str(round(loss.item(), self.ROUND)))

                # Validation.
                if self.iter_num % self.sample_step == 0:
                    self.valtest('val')

                # Decay learning rates.
                if self.iter_num % self.lr_decay_step == 0 and \
                        self.iter_num > (self.total_iters - self.num_iters_decay):
                    self.update_lr()

    def valtest(self, val_or_test):

        dataset = {
            "test": self.test_set,
            "val": self.val_set
        }[val_or_test]

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        self.C.train(mode=False)
        self.Emb.train(mode=False)
        n_items = 0
        n_wrong = 0
        with tqdm(loader) as pbar, torch.no_grad():
            for data in pbar:
                bare_0, _, _, len_0, label_0, bare_1, _, _, len_1, label_1 = self.preprocess_data(data)
                bare_emb_0 = self.Emb(bare_0)  # shape = (n_batch, 20, emb_dim); encoder input.
                bare_emb_1 = self.Emb(bare_1)  # shape = (n_batch, 20, emb_dim); encoder input.
                cls_0 = self.C(bare_emb_0).squeeze(1)  # shape = (n_batch, )
                cls_1 = self.C(bare_emb_1).squeeze(1)  # shape = (n_batch, )
                pred_0 = (cls_0 > 0.5).float()  # shape = (n_batch, )
                pred_1 = (cls_1 > 0.5).float()  # shape = (n_batch, )

                n_items += bare_0.shape[0] + bare_1.shape[1]
                n_wrong += torch.abs(pred_0 - label_0).sum().item() + torch.abs(pred_1 - label_1).sum().item()
        acc = (n_items - n_wrong) / n_items
        print('\nacc =', acc)
        print()

        if val_or_test == 'val':
            if acc > self.best_acc:
                self.best_acc = acc
                self.save_step()
        if val_or_test == 'test':
            print('Testing accuracy =', acc)

        self.C.train(mode=True)
        self.Emb.train(mode=True)
        return None

    def preprocess_data(self, data):
        bare_0, go_0, eos_0, len_0, bare_1, go_1, eos_1, len_1 = data
        n_batch = bare_0.shape[0]

        bare_0 = gpu_wrapper(bare_0)  # shape = (n_batch, 20)
        go_0 = gpu_wrapper(go_0)  # shape = (n_batch, 21)
        eos_0 = gpu_wrapper(eos_0)  # shape = (n_batch, 21)
        len_0 = gpu_wrapper(len_0)  # shape = (n_batch, )
        label_0 = gpu_wrapper(torch.zeros(n_batch))  # shape = (n_batch, )

        bare_1 = gpu_wrapper(bare_1)  # shape = (n_batch, 20)
        go_1 = gpu_wrapper(go_1)  # shape = (n_batch, 21)
        eos_1 = gpu_wrapper(eos_1)  # shape = (n_batch, 21)
        len_1 = gpu_wrapper(len_1)  # shape = (n_batch, )
        label_1 = gpu_wrapper(torch.ones(n_batch))  # shape = (n_batch, )

        return bare_0, go_0, eos_0, len_0, label_0, bare_1, go_1, eos_1, len_1, label_1


if __name__ == '__main__':
    ClsEval = ClassifierEval('TextCNN', config.dataset)
    ClsEval.pretrain()
