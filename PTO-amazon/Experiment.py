import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from Modules.Losses.SeqLoss import SeqLoss
from Modules.Losses.BackLoss import BackLoss
from Modules.replace import Replace
from Modules.attention_classifier import AttenClassifier
from Modules.Losses.Reward import RewardCriterion
from tqdm import tqdm
from dataloaders.amazon import Amazon
from config import Config
from utils.utils import gpu_wrapper, strip_pad, pretty_string, sample_2d
from classifier import ClassifierEval
from Modules.insback import InsBehind
from Modules.insfront import InsFront
from language_model import LanguageModel
from Modules.gate import Gate
from Modules.delete import Delete
from utils.multi_bleu import calc_bleu_score


config = Config()
ROUND = config.ROUND
EPSILON = 1e-5
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if config.gpu:
    torch.cuda.manual_seed(config.seed)


class Experiment(object):

    def __init__(self):

        print('----- Loading data -----')
        self.train_set = Amazon('train', False)
        self.test_set = Amazon('test', False)
        self.val_set = Amazon('dev', False)
        print('The train set has {} items'.format(len(self.train_set)))
        print('The test set has {} items'.format(len(self.test_set)))
        print('The val set has {} items'.format(len(self.val_set)))

        self.vocab = self.train_set.vocab

        # Load pretrained classifier for evaluation.
        self.clseval = ClassifierEval(config.test_cls, config.dataset)
        self.clseval.restore_model()

        print('----- Loading model -----')
        embedding = self.vocab.embedding
        self.Emb = nn.Embedding.from_pretrained(embedding.clone(), freeze=False)
        self.Classifier = AttenClassifier(emb_dim=config.emb_dim,
                                          dim_h=config.dim_h,
                                          n_layers=config.n_layers,
                                          dropout=config.dropout,
                                          bi=config.bidirectional)
        self.Gate0 = Gate(dim_h=config.dim_h, n_layers=config.n_layers, dropout=config.dropout, bi=config.bidirectional,
                          temperature=config.temp_gate, embedding=embedding.clone())
        self.Gate1 = Gate(dim_h=config.dim_h, n_layers=config.n_layers, dropout=config.dropout, bi=config.bidirectional,
                          temperature=config.temp_gate, embedding=embedding.clone())
        self.InsFront0 = InsFront(dim_h=config.dim_h, embedding=embedding.clone(), n_layers=config.n_layers,
                                  dropout=config.dropout, bi=config.bidirectional, voc_size=self.vocab.size,
                                  temperature=config.temp_sub)
        self.InsFront1 = InsFront(dim_h=config.dim_h, embedding=embedding.clone(), n_layers=config.n_layers,
                                  dropout=config.dropout, bi=config.bidirectional, voc_size=self.vocab.size,
                                  temperature=config.temp_sub)
        self.InsBehind0 = InsBehind(dim_h=config.dim_h, embedding=embedding.clone(), n_layers=config.n_layers,
                                    dropout=config.dropout, bi=config.bidirectional, voc_size=self.vocab.size,
                                    temperature=config.temp_sub)
        self.InsBehind1 = InsBehind(dim_h=config.dim_h, embedding=embedding.clone(), n_layers=config.n_layers,
                                    dropout=config.dropout, bi=config.bidirectional, voc_size=self.vocab.size,
                                    temperature=config.temp_sub)
        self.Replace0 = Replace(dim_h=config.dim_h, embedding=embedding.clone(), n_layers=config.n_layers,
                                dropout=config.dropout, bi=config.bidirectional, voc_size=self.vocab.size,
                                temperature=config.temp_sub)
        self.Replace1 = Replace(dim_h=config.dim_h, embedding=embedding.clone(), n_layers=config.n_layers,
                                dropout=config.dropout, bi=config.bidirectional, voc_size=self.vocab.size,
                                temperature=config.temp_sub)
        self.Del0 = Delete()  # Not a module.
        self.Del1 = Delete()  # Not a module.
        # Language models.
        if config.train_mode == 'pto':
            self.LMf0 = LanguageModel(config.dataset, direction='forward', sentiment=0)
            self.LMf0.model_load()
            self.LMf1 = LanguageModel(config.dataset, direction='forward', sentiment=1)
            self.LMf1.model_load()
            self.LMb0 = LanguageModel(config.dataset, direction='backward', sentiment=0)
            self.LMb0.model_load()
            self.LMb1 = LanguageModel(config.dataset, direction='backward', sentiment=1)
            self.LMb1.model_load()
        # Auxiliary classifier.
        self.Aux_Emb = nn.Embedding.from_pretrained(embedding.clone(), freeze=False)
        self.Aux_Classifier = AttenClassifier(emb_dim=config.emb_dim,
                                              dim_h=config.dim_h,
                                              n_layers=config.n_layers,
                                              dropout=config.dropout,
                                              bi=config.bidirectional)

        self.modules = ['Emb', 'Classifier',
                        'Gate0', 'Gate1',
                        'InsFront0', 'InsFront1',
                        'InsBehind0', 'InsBehind1',
                        'Replace0', 'Replace1',
                        'Aux_Classifier', 'Aux_Emb']
        for module in self.modules:
            print('--- {}: '.format(module))
            print(getattr(self, module))
            setattr(self, module, gpu_wrapper(getattr(self, module)))

        self.scopes = {
            'emb': ['Emb'],
            'cls': ['Classifier'],
            'aux_cls': ['Aux_Classifier', 'Aux_Emb'],
            'gate': ['Gate0', 'Gate1'],
            'oprt': ['InsFront0', 'InsFront1', 'InsBehind0', 'InsBehind1', 'Replace0', 'Replace1'],
        }
        for scope in self.scopes.keys():
            setattr(self, scope + '_lr', getattr(config, scope + '_lr'))

        self.iter_num = -1
        self.logger = None
        if config.train_mode == 'pto':
            pass
        elif config.train_mode == 'aux-cls-only':
            self.train_set = Amazon('train', True)
            self.test_set = Amazon('test', True)
            self.val_set = Amazon('dev', True)
            self.best_acc = 0
        elif config.train_mode == 'cls-only':
            self.best_acc = 0
        self.criterionSeq, self.criterionCls, self.criterionRL, self.criterionBack = None, None, None, None

    def restore_model(self, modules):
        """Restore the trained generators and discriminator."""
        print('Loading the trained best models...')
        for module in modules:
            path = os.path.join(config.save_model_dir, 'best-{}.ckpt'.format(module))
            getattr(self, module).load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from utils.logger import Logger
        self.logger = Logger(config.log_dir)

    def log_step(self, loss):
        # Log loss.
        for loss_name, value in loss.items():
            self.logger.scalar_summary(loss_name, value, self.iter_num)
        # Log learning rate.
        for scope in self.scopes:
            self.logger.scalar_summary('{}/lr'.format(scope), getattr(self, scope + '_lr'), self.iter_num)

    def save_step(self, modules, use_iter=False):
        if use_iter:
            for module in modules:
                path = os.path.join(config.save_model_dir, '{}-{}.ckpt'.format(self.iter_num, module))
                torch.save(getattr(self, module).state_dict(), path)
        else:
            for module in modules:
                path = os.path.join(config.save_model_dir, 'best-{}.ckpt'.format(module))
                torch.save(getattr(self, module).state_dict(), path)
        print('Saved model checkpoints into {}...\n\n\n\n\n\n\n\n\n\n\n\n'.format(config.save_model_dir))

    def zero_grad(self):
        for scope in self.scopes:
            getattr(self, scope + '_optim').zero_grad()

    def step(self, scopes, clip_norm=float('inf'), clip_value=float('inf')):
        trainable = []
        for scope in scopes:
            trainable.extend(getattr(self, 'trainable_' + scope))
        # Clip on all parameters.
        if clip_norm < float('inf'):
            clip_grad_norm_(parameters=trainable, max_norm=config.clip_norm)
        if clip_value < float('inf'):
            clip_value = float(config.clip_value)
            for p in filter(lambda p: p.grad is not None, trainable):
                p.grad.data.clamp_(min=-clip_value, max=clip_value)
        # Backward.
        for scope in scopes:
            getattr(self, scope + '_optim').step()

    def update_lr(self):
        for scope in self.scopes:
            setattr(self, scope + '_lr', getattr(self, scope + '_lr') * (1 - float(config.lr_decay_step / config.num_iters_decay)))
            for param_group in getattr(self, scope + '_optim').param_groups:
                param_group['lr'] = getattr(self, scope + '_lr')

    def set_requires_grad(self, modules, requires_grad):
        if not isinstance(modules, list):
            modules = [modules]
        for module in modules:
            for param in getattr(self, module).parameters():
                param.requires_grad = requires_grad

    def set_training(self, mode):
        for module in self.modules:
            getattr(self, module).train(mode=mode)

    def restore_pretrained(self, modules):
        for module in modules:
            path = os.path.join('pretrained/pretrained-{}.ckpt'.format(module))
            getattr(self, module).load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def train(self):

        # Logging.
        if config.use_tensorboard:
            self.build_tensorboard()

        # Load pretrained.
        if config.train_mode == 'cls-only':
            pass
        elif config.train_mode == 'pto':
            self.restore_pretrained(['Classifier', 'Emb', 'Aux_Classifier', 'Aux_Emb'])
        elif config.train_mode == 'aux-cls-only':
            pass
        else:
            raise ValueError()

        # Set trainable parameters, according to the frozen parameter list.
        for scope in self.scopes.keys():
            trainable = []
            for module in self.scopes[scope]:
                for k, v in getattr(self, module).state_dict(keep_vars=True).items():
                    # k is the parameter name; v is the parameter value.
                    if v.requires_grad:
                        trainable.append(v)
                        print("[{} Trainable:]".format(module), k)
                    else:
                        print("[{} Frozen:]".format(module), k)
            setattr(self, scope + '_optim', Adam(trainable, getattr(self, scope + '_lr'), [config.beta1, config.beta2]))
            setattr(self, 'trainable_' + scope, trainable)

        # Build criterion.
        self.criterionSeq = SeqLoss(voc_size=self.train_set.vocab.size, pad=self.train_set.pad,
                                    end=self.train_set.eos, unk=self.train_set.unk)
        self.criterionCls = nn.BCELoss()
        self.criterionBack = BackLoss(reduce=False)
        self.criterionRL = RewardCriterion()

        # Train.
        epoch = 0
        while True:
            self.train_epoch(epoch_idx=epoch)
            epoch += 1
            if self.iter_num >= config.num_iters:
                break

        self.test()

    def test(self):
        config.batch_size = 500

        self.restore_pretrained(['Classifier', 'Emb', 'Replace0', 'Replace1',
                                 'InsFront0', 'InsFront1', 'InsBehind0', 'InsBehind1', 'Aux_Classifier', 'Aux_Emb'])

        self.valtest(val_or_test="test",
                     mode='multi-steps',
                     _pow_lm=(1 / config.s[0], 1 / config.s[1]),
                     cls_stop=config.cls_stop,
                     max_iter=config.max_iter)
    def train_epoch(self, epoch_idx):

        loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
        self.set_training(mode=True)

        with tqdm(loader) as pbar:
            for data in pbar:
                self.iter_num += 1
                loss = {}

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                bare_0, _, _, len_0, y_0, _, bare_1, _, _, len_1, y_1, _ = self.preprocess_data(data)
                null_mask_0 = bare_0.eq(self.train_set.pad)
                null_mask_1 = bare_1.eq(self.train_set.pad)

                # =================================================================================== #
                #                                       2. cls-only                                   #
                # =================================================================================== #
                if config.train_mode == 'cls-only':
                    # ----- Forward pass of classification -----
                    emb_bare_0 = self.Emb(bare_0)
                    emb_bare_1 = self.Emb(bare_1)
                    cls_0, att_0 = self.Classifier(emb_bare_0, len_0, null_mask_0)
                    cls_1, att_1 = self.Classifier(emb_bare_1, len_1, null_mask_1)
                    # ----- Classification loss -----
                    cls_loss_0 = self.criterionCls(cls_0, y_0)
                    cls_loss_1 = self.criterionCls(cls_1, y_1)
                    cls_loss = cls_loss_0 + cls_loss_1
                    # ----- Logging -----
                    loss['Cls/L-0'] = round(cls_loss_0.item(), ROUND)
                    loss['Cls/L-1'] = round(cls_loss_1.item(), ROUND)
                    # ----- Backward for scopes: ['emb', 'cls'] -----
                    self.zero_grad()
                    cls_loss.backward()
                    self.step(['emb', 'cls'])
                elif config.train_mode == 'aux-cls-only':
                    # ----- Forward pass of classification -----
                    emb_bare_0 = self.Aux_Emb(bare_0)
                    emb_bare_1 = self.Aux_Emb(bare_1)
                    cls_0, att_0 = self.Aux_Classifier(emb_bare_0, len_0, null_mask_0)
                    cls_1, att_1 = self.Aux_Classifier(emb_bare_1, len_1, null_mask_1)
                    # ----- Classification loss -----
                    cls_loss_0 = self.criterionCls(cls_0, y_0)
                    cls_loss_1 = self.criterionCls(cls_1, y_1)
                    cls_loss = cls_loss_0 + cls_loss_1
                    # ----- Logging -----
                    loss['Cls/L-0'] = round(cls_loss_0.item(), ROUND)
                    loss['Cls/L-1'] = round(cls_loss_1.item(), ROUND)
                    # ----- Backward for scopes: ['emb', 'cls'] -----
                    self.zero_grad()
                    cls_loss.backward()
                    self.step(['aux_cls'])

                # =================================================================================== #
                #                                        3. pto                                       #
                # =================================================================================== #
                elif config.train_mode == 'pto':

                    # ----- Forward pass of classification -----
                    emb_bare_0 = self.Emb(bare_0)
                    emb_bare_1 = self.Emb(bare_1)
                    cls_0, att_0 = self.Classifier(emb_bare_0, len_0, null_mask_0)
                    cls_1, att_1 = self.Classifier(emb_bare_1, len_1, null_mask_1)
                    # ----- Classification loss -----
                    cls_loss_0 = self.criterionCls(cls_0, y_0)
                    cls_loss_1 = self.criterionCls(cls_1, y_1)
                    cls_loss = cls_loss_0 + cls_loss_1
                    # ----- Logging -----
                    loss['Cls/L-0'] = round(cls_loss_0.item(), ROUND)
                    loss['Cls/L-1'] = round(cls_loss_1.item(), ROUND)
                    # ----- Backward for scopes: ['emb', 'cls'] -----
                    self.zero_grad()
                    cls_loss.backward()
                    self.step(['emb', 'cls'])

                    #################
                    # 0 --> 1 --> 0 #
                    #################
                    att_pg_0, Rep_xbar_pg_0, Rep_XE2_0, IF_xbar_pg_0, IB_xbar_pg_0, Del_ib_XE2_0, Del_if_XE2_0 = self.forward_pto(bare_0, len_0, null_mask_0, direction=0, loss=loss)

                    #################
                    # 1 --> 0 --> 1 #
                    #################
                    att_pg_1, Rep_xbar_pg_1, Rep_XE2_1, IF_xbar_pg_1, IB_xbar_pg_1, Del_ib_XE2_1, Del_if_XE2_1 = self.forward_pto(bare_1, len_1, null_mask_1, direction=1, loss=loss)

                    #######################
                    # Combine and logging #
                    #######################
                    tot_loss = att_pg_0 + att_pg_1 + \
                               Rep_xbar_pg_0 + Rep_XE2_0 + \
                               Rep_xbar_pg_1 + Rep_XE2_1 + \
                               IF_xbar_pg_0 + IB_xbar_pg_0 + \
                               Del_ib_XE2_0 + Del_if_XE2_0 + \
                               IF_xbar_pg_1 + IB_xbar_pg_1 + \
                               Del_ib_XE2_1 + Del_if_XE2_1

                    # ----- Backward for scopes: ['emb', 'cls', 'oprt'] -----
                    self.zero_grad()
                    tot_loss.backward()
                    self.step(['emb', 'cls', 'oprt'])

                else:
                    raise ValueError()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                verbose = False
                if verbose:
                    display = ', '.join([key + ':' + pretty_string(loss[key]) for key in loss.keys()])
                    pbar.set_description_str(display)

                # Print out training information.
                if self.iter_num % config.log_step == 0 and config.use_tensorboard:
                    self.log_step(loss)

                # Validation.
                if self.iter_num % config.sample_step == 0:
                    if config.train_mode == 'pto':
                        self.valtest(val_or_test="val",
                                     mode='multi-steps',
                                     _pow_lm=(1 / config.s[0], 1 / config.s[1]),
                                     cls_stop=config.cls_stop,
                                     max_iter=config.max_iter)
                    elif config.train_mode == 'aux-cls-only':
                        self.valtest('val', 'aux-cls')
                    elif config.train_mode == 'cls-only':
                        self.valtest('val', 'cls')
                    else:
                        raise ValueError()

                # Decay learning rates.
                if self.iter_num % config.lr_decay_step == 0 and \
                        self.iter_num > (config.num_iters - config.num_iters_decay):
                    self.update_lr()

    def forward_pto(self, bare, seq_len, null_mask, direction, loss):
        # ----- Classify -----
        cls, att = self.Classifier(self.Emb(bare), seq_len, null_mask)
        cls = cls.detach()

        # ----- Sample attention -----
        hard_att, att_prob = sample_2d(probs=att, temperature=config.temp_att)

        # ----- Fix oprt_idx -----
        B = seq_len.shape[0]
        oprt_idx = torch.zeros_like(seq_len)
        oprt_idx[:B // 8] = 0  # InsFront.
        oprt_idx[B // 8: 2 * B // 8] = 1  # InsBehind.
        oprt_idx[2 * B // 8: 7 * B // 8] = 2  # Replace.
        oprt_idx[7 * B // 8:] = 3  # Delete.

        # ----- Process each operation -----
        bare_bar = torch.zeros_like(bare)
        T_bar = torch.zeros_like(seq_len)

        # ----- InsFront -----
        IF_idx = torch.nonzero(oprt_idx == 0).view(-1)
        IF_bare_bar, IF_xbar_prob, _ = getattr(self, 'InsFront' + str(direction))(bare[IF_idx, :], hard_att[IF_idx],
                                                                                  seq_len[IF_idx], sample=True)
        bare_bar[IF_idx, :] = IF_bare_bar
        T_bar[IF_idx] = seq_len[IF_idx] + 1

        # ----- InsBehind -----
        IB_idx = torch.nonzero(oprt_idx == 1).view(-1)
        IB_bare_bar, IB_xbar_prob, _ = getattr(self, 'InsBehind' + str(direction))(bare[IB_idx, :], hard_att[IB_idx],
                                                                                   seq_len[IB_idx], sample=True)
        bare_bar[IB_idx, :] = IB_bare_bar
        T_bar[IB_idx] = seq_len[IB_idx] + 1

        # ----- Del -----
        Del_idx = torch.nonzero(oprt_idx == 3).view(-1)
        Del_bare_bar = getattr(self, 'Del' + str(direction))(bare[Del_idx, :], hard_att[Del_idx])
        bare_bar[Del_idx, :] = Del_bare_bar
        T_bar[Del_idx] = seq_len[Del_idx] - 1

        # ----- Replace -----
        Rep_idx = torch.nonzero(oprt_idx == 2).view(-1)
        Rep_bare_bar, Rep_xbar_prob, _ = getattr(self, 'Replace' + str(direction))(bare[Rep_idx, :], hard_att[Rep_idx],
                                                                                   seq_len[Rep_idx], sample=True)
        bare_bar[Rep_idx, :] = Rep_bare_bar
        T_bar[Rep_idx] = seq_len[Rep_idx]

        # ----- Update null_mask -----
        null_mask_bar = bare_bar.eq(self.train_set.pad)

        # ----- Classify -----
        # Sort and re-sort.
        s_idx = [ix for ix, l in sorted(enumerate(T_bar.cpu()), key=lambda x: x[1], reverse=True)]
        res_idx = [a for a, b in sorted(enumerate(s_idx), key=lambda x: x[1])]
        s_cls_bare_bar, _ = self.Classifier(self.Emb(bare_bar[s_idx, :]), T_bar[s_idx], null_mask_bar[s_idx, :])
        cls_bare_bar = s_cls_bare_bar[res_idx].detach()

        # ----- Reward/PG for att (w.r.t. confidence) -----
        rwd_att_CRITIC = config.beta_att[direction]
        # Great difference -> position is important.
        rwd_att = torch.abs(cls - cls_bare_bar) - rwd_att_CRITIC
        att_pg = self.criterionRL(sample_probs=att_prob, reward=rwd_att) * config.lambda_att_conf

        # ----- Reward/PG for xbar (w.r.t. confidence) -----
        rwd_xbar_conf_CRITIC = config.beta_xbar_conf[direction]
        if direction == 0:
            IF_rwd_xbar_conf = cls_bare_bar[IF_idx] - cls[IF_idx] - rwd_xbar_conf_CRITIC
            IB_rwd_xbar_conf = cls_bare_bar[IB_idx] - cls[IB_idx] - rwd_xbar_conf_CRITIC
            Rep_rwd_xbar_conf = cls_bare_bar[Rep_idx] - cls[Rep_idx] - rwd_xbar_conf_CRITIC
        else:
            IF_rwd_xbar_conf = cls[IF_idx] - cls_bare_bar[IF_idx] - rwd_xbar_conf_CRITIC
            IB_rwd_xbar_conf = cls[IB_idx] - cls_bare_bar[IB_idx] - rwd_xbar_conf_CRITIC
            Rep_rwd_xbar_conf = cls[Rep_idx] - cls_bare_bar[Rep_idx] - rwd_xbar_conf_CRITIC
        IF_xbar_conf_pg = self.criterionRL(sample_probs=IF_xbar_prob, reward=IF_rwd_xbar_conf)
        IB_xbar_conf_pg = self.criterionRL(sample_probs=IB_xbar_prob, reward=IB_rwd_xbar_conf)
        Rep_xbar_conf_pg = self.criterionRL(sample_probs=Rep_xbar_prob, reward=Rep_rwd_xbar_conf)

        # ----- Star Indices -----
        star_index = torch.zeros_like(oprt_idx)
        star_index[IF_idx] = hard_att[IF_idx]
        star_index[IB_idx] = hard_att[IB_idx] + 1
        star_index[Rep_idx] = hard_att[Rep_idx]
        n_del = Del_idx.shape[0]
        Del_if_idx = Del_idx[: n_del // 2]
        Del_ib_idx = Del_idx[n_del // 2:]
        star_index[Del_if_idx] = hard_att[Del_if_idx]
        star_index[Del_ib_idx] = hard_att[Del_ib_idx] - 1

        # ----- Reward/PG for xbar (w.r.t. language model) -----
        rwd_xbar_lm_CRITIC = config.beta_xbar_lm[direction]

        IF_word_prob_f = getattr(self, 'LMf{}'.format(1 - direction)).inference(bare_bar[IF_idx], star_index[IF_idx], T_bar[IF_idx])  # shape = (n_IF, )
        IF_word_prob_b = getattr(self, 'LMb{}'.format(1 - direction)).inference(bare_bar[IF_idx], star_index[IF_idx], T_bar[IF_idx])  # shape = (n_IF, )
        IF_word_prob = torch.sqrt(IF_word_prob_f * IF_word_prob_b)  # shape = (n_IF, )
        IF_rwd_xbar_lm = IF_word_prob - rwd_xbar_lm_CRITIC
        IF_rwd_xbar_lm = IF_rwd_xbar_lm * config.lambda_lm
        IF_xbar_lm_pg = self.criterionRL(sample_probs=IF_xbar_prob, reward=IF_rwd_xbar_lm)

        IB_word_prob_f = getattr(self, 'LMf{}'.format(1 - direction)).inference(bare_bar[IB_idx], star_index[IB_idx], T_bar[IB_idx])  # shape = (n_IB, )
        IB_word_prob_b = getattr(self, 'LMb{}'.format(1 - direction)).inference(bare_bar[IB_idx], star_index[IB_idx], T_bar[IB_idx])  # shape = (n_IB, )
        IB_word_prob = torch.sqrt(IB_word_prob_f * IB_word_prob_b)  # shape = (n_IB, )
        IB_rwd_xbar_lm = IB_word_prob - rwd_xbar_lm_CRITIC
        IB_rwd_xbar_lm = IB_rwd_xbar_lm * config.lambda_lm
        IB_xbar_lm_pg = self.criterionRL(sample_probs=IB_xbar_prob, reward=IB_rwd_xbar_lm)

        Rep_word_prob_f = getattr(self, 'LMf{}'.format(1 - direction)).inference(bare_bar[Rep_idx], star_index[Rep_idx], T_bar[Rep_idx])  # shape = (n_Rep, )
        Rep_word_prob_b = getattr(self, 'LMb{}'.format(1 - direction)).inference(bare_bar[Rep_idx], star_index[Rep_idx], T_bar[Rep_idx])  # shape = (n_Rep, )
        Rep_word_prob = torch.sqrt(Rep_word_prob_f * Rep_word_prob_b)  # shape = (n_Rep, )
        Rep_rwd_xbar_lm = Rep_word_prob - rwd_xbar_lm_CRITIC
        Rep_rwd_xbar_lm = Rep_rwd_xbar_lm * config.lambda_lm
        Rep_xbar_lm_pg = self.criterionRL(sample_probs=Rep_xbar_prob, reward=Rep_rwd_xbar_lm)

        # ------ Supervision for xbar ------
        ori = bare.gather(1, hard_att.unsqueeze(1)).squeeze(1)

        # ----- Del_if (back) -----
        _, _, Del_if_lgt = getattr(self, 'InsFront' + str(1 - direction))(bare_bar[Del_if_idx, :],
                                                                          star_index[Del_if_idx],
                                                                          T_bar[Del_if_idx], sample=True)
        Del_if_tgt = ori[Del_if_idx]
        Del_if_XE2 = self.criterionBack(Del_if_lgt, Del_if_tgt)

        # ----- Del_ib (back) -----
        _, _, Del_ib_lgt = getattr(self, 'InsBehind' + str(1 - direction))(bare_bar[Del_ib_idx, :],
                                                                           star_index[Del_ib_idx],
                                                                           T_bar[Del_ib_idx], sample=True)
        Del_ib_tgt = ori[Del_ib_idx]
        Del_ib_XE2 = self.criterionBack(Del_ib_lgt, Del_ib_tgt)

        # ----- Replace (back) -----
        _, _, Rep_lgt = getattr(self, 'Replace' + str(1 - direction))(bare_bar[Rep_idx, :],
                                                                      star_index[Rep_idx],
                                                                      T_bar[Rep_idx], sample=True)
        Rep_tgt = ori[Rep_idx]
        Rep_XE2 = self.criterionBack(Rep_lgt, Rep_tgt)

        # ----- Summarize -----
        IF_xbar_pg = IF_xbar_conf_pg * config.lambda_ins_conf + IF_xbar_lm_pg
        IB_xbar_pg = IB_xbar_conf_pg * config.lambda_ins_conf + IB_xbar_lm_pg
        Del_ib_XE2 = Del_ib_XE2.mean()
        Del_if_XE2 = Del_if_XE2.mean()

        # ----- Reward/PG for Replace's xbar (w.r.t. XE2) -----
        rwd_xbar_XE2_CRITIC = config.beta_xbar_XE2[direction]
        Rep_rwd_xbar_XE2 = config.subtract_XE2[direction] - Rep_XE2.detach() - rwd_xbar_XE2_CRITIC
        Rep_xbar_XE2_pg = self.criterionRL(sample_probs=Rep_xbar_prob, reward=Rep_rwd_xbar_XE2) * 0.05

        # ----- Combine -----
        Rep_xbar_pg = Rep_xbar_XE2_pg + Rep_xbar_conf_pg + Rep_xbar_lm_pg

        # Logging.
        loss['Att/R-{}'.format(direction)] = rwd_att.mean().item()
        loss['Rep/R-conf-{}'.format(direction)] = Rep_rwd_xbar_conf.mean().item()
        loss['Rep/R-XE2-{}'.format(direction)] = Rep_rwd_xbar_XE2.mean().item()
        loss['Rep/XE2-{}'.format(direction)] = Rep_XE2.mean().item()
        loss['Rep/R-lm-{}'.format(direction)] = Rep_rwd_xbar_lm.mean().item()
        loss['IF/R-conf-{}'.format(direction)] = IF_rwd_xbar_conf.mean().item()
        loss['IF/R-lm-{}'.format(direction)] = IF_rwd_xbar_lm.mean().item()
        loss['IB/R-conf-{}'.format(direction)] = IB_rwd_xbar_conf.mean().item()
        loss['IB/R-lm-{}'.format(direction)] = IB_rwd_xbar_lm.mean().item()
        loss['Del_f/XE2-{}'.format(direction)] = Del_if_XE2.mean().item()
        loss['Del_b/XE2-{}'.format(direction)] = Del_ib_XE2.mean().item()

        return att_pg, Rep_xbar_pg, Rep_XE2.mean(), IF_xbar_pg, IB_xbar_pg, Del_ib_XE2, Del_if_XE2

    def valtest(self, val_or_test, mode, _pow_lm=None, cls_stop=None, max_iter=None, ablation=None):
        dataset = {
                "test": self.test_set,
                "val": self.val_set
            }[val_or_test]

        loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=config.num_workers)

        self.set_training(mode=False)

        fake_sents_0, fake_sents_1, clss_0, clss_1 = [], [], [], []
        with tqdm(loader) as pbar, torch.no_grad():
            for data in pbar:
                bare_0, _, _, len_0, y_0, res_idx_0, bare_1, _, _, len_1, y_1, res_idx_1 = self.preprocess_data(data)
                null_mask_0 = bare_0.eq(self.train_set.pad)
                null_mask_1 = bare_1.eq(self.train_set.pad)

                if mode == 'multi-steps':
                    cls_0, bare_bar_0 = self.valtest_forward_multi_steps_quick(bare_0, len_0, null_mask_0, 0, _pow_lm[0], cls_stop[0], max_iter[0], ablation, res_idx_0)
                    cls_1, bare_bar_1 = self.valtest_forward_multi_steps_quick(bare_1, len_1, null_mask_1, 1, _pow_lm[1], cls_stop[1], max_iter[1], ablation, res_idx_1)

                    clss_0.append((cls_0 > 0.5)[res_idx_0])
                    clss_1.append((cls_1 > 0.5)[res_idx_1])
                    bare_bar_0 = strip_pad(
                        [[self.vocab.id2word[i.data.cpu().numpy()] for i in sent] for sent in bare_bar_0])
                    bare_bar_1 = strip_pad(
                        [[self.vocab.id2word[i.data.cpu().numpy()] for i in sent] for sent in bare_bar_1])
                    fake_sents_1.extend([bare_bar_0[i] for i in res_idx_0])
                    fake_sents_0.extend([bare_bar_1[i] for i in res_idx_1])
                elif mode == 'aux-cls':
                    cls_0, _ = self.Aux_Classifier(self.Aux_Emb(bare_0), len_0, null_mask_0)
                    cls_1, _ = self.Aux_Classifier(self.Aux_Emb(bare_1), len_1, null_mask_1)
                    clss_0.append((cls_0 > 0.5)[res_idx_0])
                    clss_1.append((cls_1 > 0.5)[res_idx_1])
                elif mode == 'cls':
                    cls_0, _ = self.Classifier(self.Emb(bare_0), len_0, null_mask_0)
                    cls_1, _ = self.Classifier(self.Emb(bare_1), len_1, null_mask_1)
                    clss_0.append((cls_0 > 0.5)[res_idx_0])
                    clss_1.append((cls_1 > 0.5)[res_idx_1])
                else:
                    raise ValueError()

        if mode == 'aux-cls':
            clss_0 = torch.cat(clss_0, dim=0).float()
            clss_1 = torch.cat(clss_1, dim=0).float()
            # ----- Attention Classifier Acc -----
            n_wrong = clss_0.sum().item() + (1 - clss_1).sum().item()
            n_all = clss_0.shape[0] + clss_1.shape[0]
            acc = (n_all - n_wrong) / n_all
            print('\nAttention classifier accuracy =\n', acc)
            if acc > self.best_acc:
                self.best_acc = acc
                self.save_step(['Aux_Classifier', 'Aux_Emb'])
            self.set_training(mode=True)
            return None
        elif mode == 'cls':
            clss_0 = torch.cat(clss_0, dim=0).float()
            clss_1 = torch.cat(clss_1, dim=0).float()
            # ----- Attention Classifier Acc -----
            n_wrong = clss_0.sum().item() + (1 - clss_1).sum().item()
            n_all = clss_0.shape[0] + clss_1.shape[0]
            acc = (n_all - n_wrong) / n_all
            print('\nAttention classifier accuracy =\n', acc)
            if acc > self.best_acc:
                self.best_acc = acc
                self.save_step(['Classifier', 'Emb'])
            self.set_training(mode=True)
            return None

        # Transfer oriented.
        fake_sents_0 = fake_sents_0[:dataset.l1]
        fake_sents_1 = fake_sents_1[:dataset.l0]
        if val_or_test == 'test':
            ori_0, ref_0, ori_1, ref_1 = dataset.get_references()
            assert len(ref_0) == len(fake_sents_1), str(len(ref_0)) + ' ' + str(len(fake_sents_1))
            assert len(ref_1) == len(fake_sents_0)
        else:
            ori_0, ori_1 = dataset.get_val_ori()
            assert len(ori_1) == len(fake_sents_0)
            assert len(ori_0) == len(fake_sents_1)

        if val_or_test == 'test':
            # ---- Moses BLEU -----
            log_dir = 'outputs/temp_results/{}'.format(config.beta_xbar_lm[0])
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            multi_BLEU = calc_bleu_score([' '.join(sent) for sent in fake_sents_1] + [' '.join(sent) for sent in fake_sents_0],
                                         [[' '.join(ref)] for ref in ref_0] + [[' '.join(ref)] for ref in ref_1],
                                         log_dir=log_dir,
                                         multi_ref=True)
            print('moses BLEU = {}'.format(round(multi_BLEU, ROUND)))

            with open(os.path.join(config.sample_dir, 'sentiment.test.0.ours'), 'w') as f0:
                for sent in fake_sents_1:
                    f0.write(' '.join(sent) + '\n')
            with open(os.path.join(config.sample_dir, 'sentiment.test.1.ours'), 'w') as f1:
                for sent in fake_sents_0:
                    f1.write(' '.join(sent) + '\n')

            # ----- Classifier Acc -----
            acc = self.clseval.class_score(fake_sents_0 + fake_sents_1, labels=[0] * len(fake_sents_0) + [1] * len(fake_sents_1))
            print('classification accuracy = {}'.format(round(acc, ROUND)))

        else:
            # ----- Classifier Acc -----
            acc = self.clseval.class_score(fake_sents_0 + fake_sents_1, labels=[0] * len(fake_sents_0) + [1] * len(fake_sents_1))
            print('\n\n\n\nclassification accuracy = {}'.format(round(acc, ROUND)))
            # ----- Validation -----
            peep_num = 50
            print('\n1')
            [print(' '.join(sent)) for sent in ori_1[:peep_num]]
            print('\n1 -> 0')
            [print(' '.join(sent)) for sent in fake_sents_0[:peep_num]]

        self.set_training(mode=True)
        return None

    def valtest_forward_multi_steps_quick(self, bare, seq_len, null_mask, direction, _pow_lm, cls_stop, max_iter, ablation, hl_res_idx=None):
        """
        Notes:
            The code is VERY messy and poorly commented.
            Variable names may NOT reflects their semantic meaning, due to the incremental changes of methodology.
        """
        _cls, att = self.Classifier(self.Emb(bare), seq_len, null_mask)

        mask = torch.zeros_like(bare).copy_(bare)
        bare_bar = torch.zeros_like(bare).copy_(bare)
        active_indices = gpu_wrapper(torch.LongTensor(list(range(bare.shape[0]))))

        for j in range(max_iter):
            if ablation == 'mask-att':
                null_mask[:, :] = 0
            s_idx = [ix for ix, l in sorted(enumerate(seq_len[active_indices].cpu()), key=lambda x: x[1], reverse=True)]
            res_idx = [a for a, b in sorted(enumerate(s_idx), key=lambda x: x[1])]
            cls_mask, _ = self.Aux_Classifier(self.Aux_Emb(mask[active_indices][s_idx, :]), seq_len[active_indices][s_idx], null_mask[active_indices][s_idx, :])  # shape = (pre_n_active, ), (pre_n_active, max_len)
            _, att_mask = self.Classifier(self.Emb(mask[active_indices][s_idx, :]), seq_len[active_indices][s_idx], null_mask[active_indices][s_idx, :])  # shape = (pre_n_active, ), (pre_n_active, max_len)
            cls_mask = cls_mask[res_idx]
            att_mask = att_mask[res_idx]
            __active_indices = torch.nonzero(torch.abs(1 - direction - cls_mask) > cls_stop).view(-1)
            if __active_indices.shape[0] == 0:
                break
            active_indices = active_indices[__active_indices]  # shape = (n_active, )
            att_mask = att_mask[__active_indices]  # shape = (n_active, max_len)
            s_idx = [ix for ix, l in sorted(enumerate(seq_len[active_indices].cpu()), key=lambda x: x[1], reverse=True)]
            res_idx = [a for a, b in sorted(enumerate(s_idx), key=lambda x: x[1])]
            i = torch.argmax(att_mask, dim=1)
            bare_bar_InsFront, _, _ = getattr(self, 'InsFront' + str(direction))(bare_bar[active_indices][s_idx], i[s_idx], seq_len[active_indices][s_idx], sample=False)
            bare_bar_InsFront = bare_bar_InsFront[res_idx]
            bare_bar_InsBehind, _, _ = getattr(self, 'InsBehind' + str(direction))(bare_bar[active_indices][s_idx], i[s_idx], seq_len[active_indices][s_idx], sample=False)
            bare_bar_InsBehind = bare_bar_InsBehind[res_idx]
            bare_bar_Replace, _, _ = getattr(self, 'Replace' + str(direction))(bare_bar[active_indices][s_idx], i[s_idx], seq_len[active_indices][s_idx], sample=False)
            bare_bar_Replace = bare_bar_Replace[res_idx]
            bare_bar_Delthis = getattr(self, 'Del' + str(direction))(bare_bar[active_indices], i)
            bare_bar_Delbefore = getattr(self, 'Del' + str(direction))(bare_bar[active_indices], i - 1)
            bare_bar_Delafter = getattr(self, 'Del' + str(direction))(bare_bar[active_indices], i + 1)
            bare_bar_NotChange = bare_bar[active_indices]

            bare_bars = [bare_bar_InsFront,
                         bare_bar_InsBehind,
                         bare_bar_Replace,
                         bare_bar_Delthis,
                         bare_bar_Delbefore,
                         bare_bar_Delafter,
                         bare_bar_NotChange]
            seq_lens = [seq_len[active_indices] + 1,
                        seq_len[active_indices] + 1,
                        seq_len[active_indices],
                        seq_len[active_indices] - 1,
                        seq_len[active_indices] - 1,
                        seq_len[active_indices] - 1,
                        seq_len[active_indices]]
            sent_probs = []
            for _bare_bar, _seq_len in zip(bare_bars, seq_lens):
                sent_prob_f = getattr(self, 'LMf{}'.format(1 - direction)).inference_whole(_bare_bar,  _seq_len)
                sent_prob_b = getattr(self, 'LMb{}'.format(1 - direction)).inference_whole(_bare_bar, _seq_len)
                cls, _ = self.Classifier(self.Emb(_bare_bar[s_idx, :]), _seq_len[s_idx], _bare_bar.eq(self.train_set.pad)[s_idx, :])
                cls = cls[res_idx]
                cls = torch.abs(direction - cls)
                sent_probs.append(torch.pow(torch.sqrt(sent_prob_f * sent_prob_b), _pow_lm) * cls)

            sent_probs = torch.stack(sent_probs, dim=1)
            try:
                for abl in range(7):
                    if abl not in ablation:
                        sent_probs[:, abl] = - float('inf')
            except Exception:
                pass
            oprt = torch.argmax(sent_probs, dim=1)
            bare_bars = torch.stack(bare_bars, dim=2)
            for __index, index in enumerate(active_indices):
                bare_bar[index, :] = bare_bars[__index, :, oprt[__index]]
                seq_len[index] = seq_lens[oprt[__index].item()][__index]

            __infront_indices = torch.nonzero(oprt == 0).view(-1)
            __insbehind_indices = torch.nonzero(oprt == 1).view(-1)
            __replace_indices = torch.nonzero(oprt == 2).view(-1)
            __notchange_indices = torch.nonzero(oprt == 6).view(-1)

            window = False
            for __index in __infront_indices:
                null_mask[active_indices[__index], i[__index]] = 1
                if window:
                    before_indices = i[__index] - 1
                    if len(before_indices.shape) > 0:
                        before_indices[torch.nonzero(before_indices < 0).view(-1)] = 0
                    null_mask[active_indices[__index], before_indices] = 1
                    null_mask[active_indices[__index], i[__index] + 1] = 1
                mask[active_indices[__index], i[__index]] = self.train_set.unk
                if window:
                    before_indices = i[__index] - 1
                    if len(before_indices.shape) > 0:
                        before_indices[torch.nonzero(before_indices < 0).view(-1)] = 0
                    mask[active_indices[__index], before_indices] = self.train_set.unk
                    mask[active_indices[__index], i[__index] + 1] = self.train_set.unk
            for __index in __insbehind_indices:
                null_mask[active_indices[__index], i[__index] + 1] = 1
                if window:
                    null_mask[active_indices[__index], i[__index]]     = 1
                    null_mask[active_indices[__index], i[__index] + 2] = 1
                mask[active_indices[__index], i[__index] + 1] = self.train_set.unk
                if window:
                    mask[active_indices[__index], i[__index]] = self.train_set.unk
                    mask[active_indices[__index], i[__index] + 2] = self.train_set.unk
            for __index in __replace_indices:
                null_mask[active_indices[__index], i[__index]] = 1
                if window:
                    before_indices = i[__index] - 1
                    if len(before_indices.shape) > 0:
                        before_indices[torch.nonzero(before_indices < 0).view(-1)] = 0
                    null_mask[active_indices[__index], before_indices] = 1
                    null_mask[active_indices[__index], i[__index] + 1] = 1

                mask[active_indices[__index], i[__index]] = self.train_set.unk
                if window:
                    before_indices = i[__index] - 1
                    if len(before_indices.shape) > 0:
                        before_indices[torch.nonzero(before_indices < 0).view(-1)] = 0
                    mask[active_indices[__index], before_indices] = self.train_set.unk
                    mask[active_indices[__index], i[__index] + 1] = self.train_set.unk
            for __index in __notchange_indices:
                null_mask[active_indices[__index], i[__index]] = 1
                if window:
                    before_indices = i[__index] - 1
                    if len(before_indices.shape) > 0:
                        before_indices[torch.nonzero(before_indices < 0).view(-1)] = 0
                    null_mask[active_indices[__index], before_indices] = 1
                    null_mask[active_indices[__index], i[__index] + 1] = 1

                mask[active_indices[__index], i[__index]] = self.train_set.unk
                if window:
                    before_indices = i[__index] - 1
                    if len(before_indices.shape) > 0:
                        before_indices[torch.nonzero(before_indices < 0).view(-1)] = 0
                    mask[active_indices[__index], before_indices] = self.train_set.unk
                    mask[active_indices[__index], i[__index] + 1] = self.train_set.unk

        return _cls, bare_bar

    def preprocess_data(self, data):
        bare_0, go_0, eos_0, len_0, bare_1, go_1, eos_1, len_1 = data
        n_batch = bare_0.shape[0]

        s_idx_0 = [ix for ix, l in sorted(enumerate(len_0), key=lambda x: x[1], reverse=True)]
        res_idx_0 = [a for a, b in sorted(enumerate(s_idx_0), key=lambda x: x[1])]
        bare_0 = gpu_wrapper(bare_0[s_idx_0, :])
        go_0 = gpu_wrapper(go_0[s_idx_0, :])
        eos_0 = gpu_wrapper(eos_0[s_idx_0, :])
        len_0 = gpu_wrapper(len_0[s_idx_0])
        y_0 = gpu_wrapper(torch.zeros(n_batch))

        s_idx_1 = [ix for ix, l in sorted(enumerate(len_1), key=lambda x: x[1], reverse=True)]
        res_idx_1 = [a for a, b in sorted(enumerate(s_idx_1), key=lambda x: x[1])]
        bare_1 = gpu_wrapper(bare_1[s_idx_1, :])
        go_1 = gpu_wrapper(go_1[s_idx_1, :])
        eos_1 = gpu_wrapper(eos_1[s_idx_1, :])
        len_1 = gpu_wrapper(len_1[s_idx_1])
        y_1 = gpu_wrapper(torch.ones(n_batch))

        return bare_0, go_0, eos_0, len_0, y_0, res_idx_0, bare_1, go_1, eos_1, len_1, y_1, res_idx_1
