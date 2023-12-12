'''
Dynamic CD
RNN + NeuralCDM
author: Fei Wang
'''
import fret
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import random
import os
from sklearn.metrics import roc_auc_score
import time


def config_ws(ws, config_dict):
    if not os.path.exists(ws):
        os.makedirs(ws)
        os.mkdir(ws + '/snapshot')
    with open(ws + '/model_config.txt', 'w') as o_f:
        json.dump(config_dict, o_f, indent=4)


def read_ws_config(ws):
    if not os.path.exists(ws):
        print("The work space has not been configured yet.")
        exit()
    with open(ws + '/model_config.txt', 'r') as i_f:
        config_dict = json.load(i_f)
    return config_dict


class NeuralCDM(nn.Module):
    def __init__(self, knowledge_dim, exer_n):
        self.knowledge_dim = knowledge_dim
        self.exer_n = exer_n
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(NeuralCDM, self).__init__()

        # prediction sub-net
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, state_emb, input_exercise, input_knowledge_point):
        # before prednet
        # k_difficulty = self.k_difficulty[input_exercise]
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_discimination = torch.sigmoid(self.e_discrimination(input_exercise))
        # prednet
        input_x = e_discimination * (state_emb - k_difficulty) * (input_knowledge_point)
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))
        output_0 = 1 - output_1
        # output_0 = torch.ones(output_1.size()).to(device) - output_1
        output = torch.cat((output_0, output_1), -1)

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_states(self, stat_idx):
        stat_emb = torch.sigmoid(self.student_emb(stat_idx))
        return stat_emb.data

    def get_knowledge_difficulty(self, exer_id):
        k_difficulty = self.k_difficulty[exer_id]
        return k_difficulty.data

    def sample_data(self, input_x):
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))
        return output_1


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = F.relu(torch.neg(w))
            w.add_(a)


class TransitionRNN(nn.Module):
    def __init__(self, input_size, stu_ho_dim=50, rnn_type='gru', batch_size=32):
        super(TransitionRNN, self).__init__()
        self.rnn_type = rnn_type
        self.input_size, self.stu_ho_dim, self.rnn_type = input_size, stu_ho_dim, rnn_type
        self.batch_size = batch_size
        self.register_buffer('rnn_hidden', None)
        self.register_buffer('initial_rnn_hidden', None)
        self.full_size_1 = stu_ho_dim
        self.full_1 = nn.Linear(self.input_size, self.full_size_1)
        self.full_2 = nn.Linear(self.stu_ho_dim, self.stu_ho_dim)
        self.inst_norm_1 = nn.InstanceNorm1d(batch_size, affine=True, track_running_stats=False)
        self.inst_norm_2 = nn.InstanceNorm1d(batch_size, affine=True, track_running_stats=False)
        self.init_hidden()
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.full_size_1, hidden_size=self.stu_ho_dim, num_layers=1)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=self.full_size_1, hidden_size=self.stu_ho_dim, num_layers=1, nonlinearity='tanh')
        else:    # lstm
            self.rnn = nn.LSTM(input_size=self.full_size_1, hidden_size=self.stu_ho_dim, num_layers=1)   #

    def init_hidden(self):
        device = next(self.parameters()).device
        if self.initial_rnn_hidden is None:
            if self.rnn_type == 'gru' or self.rnn_type == 'rnn':
                h = torch.zeros(1, self.batch_size, self.stu_ho_dim)
                # torch.nn.init.normal_(h, 0, 1)
                h = h.to(device)
                self.initial_rnn_hidden = h
            else:    # 'lstm'
                h = torch.zeros(1, self.batch_size, self.stu_ho_dim)
                c = torch.zeros(1, self.batch_size, self.stu_ho_dim)
                # torch.nn.init.normal_(h)
                # torch.nn.init.normal_(c)
                h = h.to(device)
                c = c.to(device)
                self.initial_rnn_hidden = h, c
        self.rnn_hidden = self.initial_rnn_hidden

    def forward(self, input_x):
        input_x = self.inst_norm_1(self.full_1(input_x))
        h_2toT, _ = self.rnn(input_x, self.rnn_hidden)
        h_2toT = torch.tanh(self.inst_norm_2(self.full_2(h_2toT)))
        return h_2toT


class Decoder(nn.Module):
    '''
    decode high-order student state into explicit (low-order) state
    '''
    def __init__(self, stu_ho_dim, stu_lo_dim):
        super(Decoder, self).__init__()
        self.layer = nn.Linear(stu_ho_dim, stu_lo_dim)

    def forward(self, input_ho_state):
        return torch.sigmoid(self.layer(input_ho_state))


class Model(nn.Module):
    def __init__(
            self,
            stu_ho_dim_lcl,
            rnn_type_lcl,
            attr_idx_lcl,
            max_log,
            knowledge_n,
            exer_n,
            batch_size
    ):
        super(Model, self).__init__()
        self.stu_ho_dim, self.rnn_type, self.attr_idx = stu_ho_dim_lcl, rnn_type_lcl, attr_idx_lcl
        self.stu_lo_dim = knowledge_n
        self.seq_len, self.exer_n, self.knowledge_n, self.batch_size = max_log, exer_n, knowledge_n, batch_size

        if self.attr_idx == 1:
            self.transition_rnn = TransitionRNN(exer_n * 2, self.stu_ho_dim, self.rnn_type, self.batch_size)
        else:
            self.transition_rnn = TransitionRNN(knowledge_n * 2, self.stu_ho_dim, self.rnn_type, self.batch_size)
        self.decoder = Decoder(self.stu_ho_dim, self.stu_lo_dim)
        self.neuralcdm = NeuralCDM(knowledge_n, exer_n)

    def init_hidden(self):
        self.transition_rnn.init_hidden()

    def apply_clipper(self):
        self.neuralcdm.apply_clipper()

    def train(self, mode=1):
        '''
        set the requires_grad_ before training or testing
        :param mode:
        :return:
        '''
        assert mode in [0, 1, 2]
        if mode == 0:
            super(Model, self).train(False)
        else:
            super(Model, self).train(True)
            if mode == 1:
                for param in self.parameters():
                    param.requires_grad_(True)
            else:
                for param in self.parameters():
                    param.requires_grad_(False)
                for param in self.transition_rnn.parameters():
                    param.requires_grad_(True)
                for param in self.decoder.parameters():
                    param.requires_grad_(True)

    def stage2_reinitial(self):
        for name, param in self.transition_rnn.named_parameters():
            if 'weight' in name and len(param.size()) > 1:
                nn.init.xavier_normal_(param)
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def __get_state(self, stu_id, exer_id, knowledge_relevancy, corrs, stage):
        # # construct exercises' knowledge embedding with correctness
        model_device = next(self.parameters()).device
        if self.attr_idx == 1:
            rnn_input = torch.zeros(self.batch_size, self.seq_len - 1, 2 * self.exer_n).to(model_device)
            for stu_i in range(self.batch_size):
                for seq_j in range(self.seq_len - 1):
                    corr = corrs[stu_i][seq_j]
                    rnn_input[stu_i][seq_j][corr * self.exer_n + exer_id[stu_i][seq_j]] = 1
        else:
            rnn_input = torch.zeros(self.batch_size, self.seq_len - 1, 2 * self.knowledge_n).to(model_device)
            if stage == 1 or self.attr_idx == 2:
                emb = knowledge_relevancy
            elif self.attr_idx == 3:  # stage == 2
                k_difficulty = torch.sigmoid(self.neuralcdm.k_difficulty(exer_id).detach())
                emb = knowledge_relevancy * k_difficulty
            else:  # stage == 2 and self.attr_idx == 4
                k_difficulty = torch.sigmoid(self.neuralcdm.k_difficulty(exer_id).detach())
                e_discrimination = torch.sigmoid(self.neuralcdm.e_discrimination(exer_id).detach())
                emb = knowledge_relevancy * k_difficulty * e_discrimination
            for stu_i in range(self.batch_size):
                for seq_j in range(self.seq_len - 1):
                    corr = corrs[stu_i][seq_j]
                    rnn_input[stu_i][seq_j][corr * self.knowledge_n: (corr + 1) * self.knowledge_n] = emb[stu_i][seq_j]
        # # construct input_x
        rnn_input = rnn_input.transpose(0, 1)  # seq_len - 1, batch, input_dim
        # # probability and predict high-order states
        pred_ho_states = self.transition_rnn.forward(rnn_input)
        # # predict
        pred_lo_states = self.decoder.forward(pred_ho_states.transpose(0, 1))
        return pred_lo_states

    def forward(self, stu_id, exer_id, knowledge_relevancy, corrs, stage):
        pred_lo_states = self.__get_state(stu_id, exer_id, knowledge_relevancy, corrs, stage)
        pred_out = self.neuralcdm.forward(pred_lo_states, exer_id[:, 1:], knowledge_relevancy[:, 1:, :])

        return pred_out

    def get_state(self, stu_id, exer_id, knowledge_relevancy, corrs, stage):
        with torch.no_grad():
            pred_lo_states = self.__get_state(stu_id, exer_id, knowledge_relevancy, corrs, stage)
        return pred_lo_states.detach()

    def get_kc_diff(self, exer_id):
        k_diff = torch.sigmoid(self.neuralcdm.k_difficulty(exer_id))
        return k_diff.detach()

    def get_disc(self, exer_id):
        e_disc = torch.sigmoid(self.neuralcdm.e_discrimination(exer_id))
        return e_disc.detach()


class DataLoader(object):
    def __init__(self, ws_config, cross_idx):
        super(DataLoader, self).__init__()
        self.batch_size = ws_config['batch_size']
        self.knowledge_dim = ws_config['knowledge_n']
        self.max_log = ws_config['max_log']
        self.data = []
        self.ptr = 0
        file_name = 'data/{}/train_{}.json'.format(ws_config['data'], cross_idx)
        with open(file_name, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        # ------------------------------------------------------------------------------------------------------
        # 格式，self.data = [stu_1, stu_2, ...]
        # stu = [stu_id, log_len, [log_1, log_2, ...]]  # the logs of each student have been padded to max_log
        # log = [order_id, exer_id, correct, [skill_id], skill_comb]
        # ------------------------------------------------------------------------------------------------------

    def next_batch(self):
        if self.is_end():
            return None
        next_ptr = min(self.ptr + self.batch_size, len(self.data))
        batch_len = next_ptr - self.ptr  # 真实有效的batch大小，若小于self.batch_size，则填充
        stu_ids, log_lens, exer_ids, corrs = [], [], [], []
        knowledge_relevancies = np.zeros((self.batch_size, self.max_log, self.knowledge_dim))
        for i in range(self.ptr, next_ptr):
            stu_i = self.data[i]
            stu_ids.append(stu_i[0] - 1)   # stu_id starts from 1 --> starts from 0
            log_len = stu_i[1]         # 数据预处理时有padding的操作，这里千万不要直接用len(stu[2])
            log_lens.append(log_len)
            stu_exer_ids, stu_corrs = [], []
            stu_knowledge_relevancies = np.zeros((self.max_log, self.knowledge_dim))
            for j in range(log_len):
                log_j = stu_i[2][j]
                stu_exer_ids.append(log_j[1] - 1)     # exer_id starts from 1 --> start from 0
                stu_corrs.append(log_j[2])
                for skill in log_j[3]:
                    stu_knowledge_relevancies[j][skill - 1] = 1.0   # skill starts from 1
            # padding sequence     # 多余的操作，因为数据预处理时已经对sequence进行了补齐
            stu_exer_ids += [0] * (self.max_log - log_len)
            stu_corrs += [0] * (self.max_log - log_len)
            # append in batch
            exer_ids.append(stu_exer_ids)
            knowledge_relevancies[i - self.ptr] = stu_knowledge_relevancies
            corrs.append(stu_corrs)
        # padding batch
        if batch_len < self.batch_size:
            pad_len = self.batch_size - batch_len
            stu_ids += [0] * pad_len
            log_lens += [0] * pad_len
            exer_ids += [[0] * self.max_log] * pad_len
            corrs += [[0] * self.max_log] * pad_len
        self.ptr = next_ptr

        return batch_len, np.array(stu_ids), np.array(log_lens), torch.LongTensor(exer_ids), torch.Tensor(knowledge_relevancies), torch.LongTensor(corrs)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
        random.shuffle(self.data)


class ValTestDataLoader(object):
    def __init__(self, ws_config, cross_idx, dtype='validation'):
        super(ValTestDataLoader, self).__init__()
        self.batch_size = ws_config['batch_size']
        self.knowledge_dim = ws_config['knowledge_n']
        self.max_log = ws_config['max_log']
        self.data = []
        self.ptr = 0
        if dtype == 'validation':
            file_name = 'data/{}/val_{}.json'.format(ws_config['data'], cross_idx)
        else:
            file_name = 'data/{}/test.json'.format(ws_config['data'])
        with open(file_name, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        # ------------------------------------------------------------------------------------------------------
        # 格式，self.data = [stu_1, stu_2, ...]
        # stu = [stu_id, log_len, [log_1, log_2, ...]]   # the logs of each student have been padded to max_log
        # log = [order_id, exer_id, correct, [skill_id], skill_comb]
        # ------------------------------------------------------------------------------------------------------

    def next_batch(self):
        if self.is_end():
            return None
        next_ptr = min(self.ptr + self.batch_size, len(self.data))
        batch_len = next_ptr - self.ptr  # 真实有效的batch大小，若小于self.batch_size，则填充
        stu_ids, log_lens, exer_ids, corrs = [], [], [], []
        knowledge_relevancies = np.zeros((self.batch_size, self.max_log, self.knowledge_dim))
        for i in range(self.ptr, next_ptr):
            stu_i = self.data[i]
            stu_ids.append(stu_i[0] - 1)   # stu_id starts from 1 --> starts from 0
            log_len = stu_i[1]
            log_lens.append(log_len)
            stu_exer_ids, stu_corrs = [], []
            stu_knowledge_relevancies = np.zeros((self.max_log, self.knowledge_dim))
            for j in range(log_len):
                log_j = stu_i[2][j]
                stu_exer_ids.append(log_j[1] - 1)     # exer_id starts from 1 --> start from 0
                stu_corrs.append(log_j[2])
                for skill in log_j[3]:
                    stu_knowledge_relevancies[j][skill - 1] = 1.0   # skill starts from 1
            # padding sequence
            stu_exer_ids += [0] * (self.max_log - log_len)
            stu_corrs += [0] * (self.max_log - log_len)
            # append in batch
            exer_ids.append(stu_exer_ids)
            knowledge_relevancies[i - self.ptr] = stu_knowledge_relevancies
            corrs.append(stu_corrs)
        # padding batch
        if batch_len < self.batch_size:
            pad_len = self.batch_size - batch_len
            stu_ids += [0] * pad_len
            log_lens += [0] * pad_len
            exer_ids += [[0] * self.max_log] * pad_len
            corrs += [[0] * self.max_log] * pad_len
        self.ptr = next_ptr

        return batch_len, np.array(stu_ids), np.array(log_lens), torch.LongTensor(exer_ids), torch.Tensor(knowledge_relevancies), torch.LongTensor(corrs)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


def train_stage1(ws, device='cpu', lr=0.004, n_epochs=10):
    # read the configuration of the work space
    ws_config_dict = read_ws_config(ws)
    batch_size = ws_config_dict['batch_size']
    # initialize model
    model = Model(
        stu_ho_dim_lcl=ws_config_dict['stu_ho_dim'],
        rnn_type_lcl=ws_config_dict['rnn_type'],
        attr_idx_lcl=ws_config_dict['attr_idx'],
        max_log=ws_config_dict['max_log'],
        knowledge_n=ws_config_dict['knowledge_n'],
        exer_n=ws_config_dict['exer_n'],
        batch_size=batch_size
    )

    model = model.to(device)
    global cross_idx
    data_loader = DataLoader(ws_config_dict, cross_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.NLLLoss(reduction='none')

    epoch_times = []
    for epoch in range(n_epochs):
        start = time.time()

        model.train(1)
        data_loader.reset()
        running_pred_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            batch_len, stu_ids, log_lens, exer_ids, knowledge_relevancies, corrs = data_loader.next_batch()
            exer_ids, knowledge_relevancies, corrs = exer_ids.to(device), knowledge_relevancies.to(device), corrs.to(device)
            model.init_hidden()
            optimizer.zero_grad()
            pred_out = model.forward(stu_ids, exer_ids, knowledge_relevancies, corrs, stage=1)

            batch_log_count = log_lens[:batch_len].sum() - batch_len
            pred_loss_list = loss_function(torch.log(pred_out.view(-1, 2)), corrs[:, 1:].contiguous().view(-1))
            pred_loss_list = pred_loss_list.view(batch_size, ws_config_dict['max_log'] - 1)
            pred_loss = 0.
            for stu_i in range(batch_len):
                pred_loss = pred_loss + pred_loss_list[stu_i][:log_lens[stu_i] - 1].sum()    # -1是因为预测部分只计算t=2到t=T (T即log_len)
            pred_loss = pred_loss / batch_log_count
            loss = pred_loss
            loss.backward()
            optimizer.step()
            model.apply_clipper()

            running_pred_loss += pred_loss.item()
            if batch_count % 10 == 0:
                print('[%d, %5d] pred_loss=%.3f' %
                      (epoch, batch_count, running_pred_loss / 10))
                running_pred_loss = 0.0

        end = time.time()
        epoch_time = end - start
        print('time:', epoch_time)
        epoch_times.append(epoch_time)

        val_test(ws, model, epoch, stage=1, dtype='validation')
        torch.save(model.state_dict(), f'{ws}/snapshot/stage1-{epoch}')
    print("epoch_times:", str(epoch_times))


def train_stage2(ws, stage1_epoch=5, n_epochs=10, device='cpu', lr=0.004, reinitial=False):
    '''

    :param ws:
    :param stage1_epoch: the epoch of the snapshot in stage1 that will be loaded and trained in stage2
    :param n_epochs: the number of epochs in stage 2
    :param device:
    :param lr:
    :param reinitial: whether re-initialize the parameters in rnn and decoder before training
    :return:
    '''
    # read the configuration of the work space
    ws_config_dict = read_ws_config(ws)
    assert ws_config_dict['attr_idx'] in [3, 4]
    batch_size = ws_config_dict['batch_size']
    # initialize model
    model = Model(
        stu_ho_dim_lcl=ws_config_dict['stu_ho_dim'],
        rnn_type_lcl=ws_config_dict['rnn_type'],
        attr_idx_lcl=ws_config_dict['attr_idx'],
        max_log=ws_config_dict['max_log'],
        knowledge_n=ws_config_dict['knowledge_n'],
        exer_n=ws_config_dict['exer_n'],
        batch_size=ws_config_dict['batch_size']
    )
    model.load_state_dict(torch.load(f'{ws}/snapshot/stage1-{stage1_epoch}', map_location=lambda s, loc: s))
    model = model.to(device)

    if reinitial:
        model.stage2_reinitial()
        lr = 0.004
    global cross_idx
    data_loader = DataLoader(ws_config_dict, cross_idx)
    optimizer = optim.Adam([{'params': model.transition_rnn.parameters()}, {'params': model.decoder.parameters()}], lr=lr)
    loss_function = nn.NLLLoss(reduction='none')

    epoch_times = []
    for epoch in range(n_epochs):
        start = time.time()

        model.train(2)
        data_loader.reset()
        running_pred_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            batch_len, stu_ids, log_lens, exer_ids, knowledge_relevancies, corrs = data_loader.next_batch()
            exer_ids, knowledge_relevancies, corrs = exer_ids.to(device), knowledge_relevancies.to(device), corrs.to(device)
            model.init_hidden()
            optimizer.zero_grad()
            pred_out = model.forward(stu_ids, exer_ids, knowledge_relevancies, corrs, stage=2)

            batch_log_count = log_lens[:batch_len].sum() - batch_len
            pred_loss_list = loss_function(torch.log(pred_out.view(-1, 2)), corrs[:, 1:].contiguous().view(-1))
            pred_loss_list = pred_loss_list.view(batch_size, ws_config_dict['max_log'] - 1)
            pred_loss = 0.
            for stu_i in range(batch_len):
                pred_loss = pred_loss + pred_loss_list[stu_i][:log_lens[stu_i] - 1].sum()    # -1是因为预测部分只计算t=2到t=T (T即log_len)
            pred_loss = pred_loss / batch_log_count
            loss = pred_loss
            loss.backward()
            optimizer.step()

            running_pred_loss += pred_loss.item()
            if batch_count % 10 == 0:
                print('[%d, %5d] pred_loss=%.3f' %
                      (epoch, batch_count, running_pred_loss / 10))
                running_pred_loss = 0.0
        end = time.time()
        epoch_time = end - start
        print('time:', epoch_time)
        epoch_times.append(epoch_time)

        val_test(ws, model, epoch, stage=2, dtype='validation')
        torch.save(model.state_dict(), f'{ws}/snapshot/stage2-{epoch}')
    print("epoch_times:", str(epoch_times))


def val_test(ws, model, epoch_i, stage, dtype='validation'):
    ws_config_dict = read_ws_config(ws)
    global cross_idx
    data_loader = ValTestDataLoader(ws_config_dict, cross_idx, dtype=dtype)
    device = next(model.parameters()).device
    model.train(0)

    pred_all, diag_label_all, pred_label_all = [], [], []
    while not data_loader.is_end():
        batch_len, stu_ids, log_lens, exer_ids, knowledge_relevancies, corrs = data_loader.next_batch()
        exer_ids, knowledge_relevancies, corrs = exer_ids.to(device), knowledge_relevancies.to(device), corrs.to(device)

        model.init_hidden()
        pred_out = model.forward(stu_ids, exer_ids, knowledge_relevancies, corrs, stage)
        pred_out = pred_out.to(torch.device('cpu')).detach().numpy()
        corrs = corrs.to(torch.device('cpu')).detach().numpy()
        for stu_i in range(batch_len):
            pred_all += pred_out[stu_i][:log_lens[stu_i] - 1, 1].tolist()
            pred_label_all += corrs[stu_i][1:log_lens[stu_i]].tolist()

    pred_all, pred_label_all = np.array(pred_all), np.array(pred_label_all)
    pred_auc = roc_auc_score(pred_label_all, pred_all)
    pred_rmse = np.sqrt(np.mean((pred_all - pred_label_all) ** 2))
    pred_acc = 0
    for i in range(len(pred_all)):
        if (pred_label_all[i] == 1) and (pred_all[i] > 0.5):
            pred_acc += 1
        elif (pred_label_all[i] == 0) and (pred_all[i] < 0.5):
            pred_acc += 1
    pred_acc /= len(pred_all)
    print('[{}] stage={}, epoch={}: auc={}, rmse={}, acc={}'.format(dtype, stage, epoch_i, pred_auc, pred_rmse, pred_acc))
    with open(f'{ws}/results.txt', 'a') as o_f:
        o_f.write('[{}] stage={}, epoch={}: auc={}, rmse={}, acc={}\n'.format(dtype, stage, epoch_i, pred_auc, pred_rmse, pred_acc))


def test(ws, epoch_model=None, stage=2, device='cpu'):
    # read the configuration of the work space
    ws_config_dict = read_ws_config(ws)
    # initialize model
    model = Model(
        stu_ho_dim_lcl=ws_config_dict['stu_ho_dim'],
        rnn_type_lcl=ws_config_dict['rnn_type'],
        attr_idx_lcl=ws_config_dict['attr_idx'],
        max_log=ws_config_dict['max_log'],
        knowledge_n=ws_config_dict['knowledge_n'],
        exer_n=ws_config_dict['exer_n'],
        batch_size=ws_config_dict['batch_size']
    )

    epoch_range = range(100) if epoch_model is None else range(int(epoch_model), int(epoch_model) + 1)
    for epoch in epoch_range:
        snapshot_path = f'{ws}/snapshot/stage{stage}-{epoch}'
        if not os.path.exists(snapshot_path):
            continue
        model.load_state_dict(torch.load(snapshot_path, map_location=lambda s, loc: s))
        model = model.to(device)
        val_test(ws=ws, model=model, epoch_i=epoch, stage=stage, dtype='test')


if __name__ == '__main__':
    # global configuration
    ws = 'ws/dneuralcdm/assist09'  # the path of work space
    device = 'cuda:0'  # the device on which to run the program
    cross_idx = 0  # the index of cross validation dataset
    lr = 0.004
    # work space configuration, this should be run at least once before training
    # the snapshots and results will be saved in this folder
    data_name = 'assist2009'
    # # default configuration, attr_idx in [1,2,3,4] for DIRT_1, DIRT_2, DIRT_3, DIRT_4 respectively
    ws_config_dict = {'stu_ho_dim': 50, 'rnn_type': 'gru', 'attr_idx': 1, 'data': data_name, 'batch_size': 32}
    # # data configuration, read from data_config.txt file
    with open(f'data/{data_name}/data_config.txt', encoding='utf8') as i_f:
        data_config = eval(i_f.readline())
    ws_config_dict['max_log'] = data_config['max_log']
    ws_config_dict['exer_n'] = data_config['exer_n']
    ws_config_dict['knowledge_n'] = data_config['knowledge_n']
    ws_config_dict['student_n'] = data_config['student_n']
    config_ws(ws, ws_config_dict)  # configure the work space

    # train
    train_stage1(ws, device='cuda:0', lr=lr)
    train_stage2(ws, stage1_epoch=0, n_epochs=5, device='cuda:0', lr=lr)  # only for DIRT_3 and DIRT_4
    # test
    test(ws, stage=1)  # stage=1 or 2
