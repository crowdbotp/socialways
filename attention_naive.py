import os

import torch
import numpy as np
import torch.nn as nn
from itertools import chain
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.utils.parse_utils import BIWIParser, Scale, create_dataset


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)


class AttentionRel(nn.Module):
    def __init__(self):
        super(AttentionRel, self).__init__()

    def forward(self, x, sub_batches=[]):
        bs = x.shape[0]

        oriens = torch.FloatTensor(torch.zeros(bs, 2))
        for ii in range(bs):
            oriens[ii] = x[ii, -1] - x[ii, -3]
        oriens = torch.atan2(oriens[:, 1], oriens[:, 0])

        attens = torch.zeros(bs, bs).cuda()
        for sb in sub_batches:
            x_sb = x[sb[0]:sb[1]]
            for ped in range(sb[0], sb[1]):
                others = [xx for j, xx in enumerate(x_sb) if j != (ped - sb[0])]
                POI_x = x_sb[ped-sb[0]]
                if not others:  continue
                others = torch.stack(others)
                dx = POI_x.unsqueeze(0) - others

            # attn = torch.mm(1, 1)
            # attn = attn - attn.max(1)[0]
            # exp_attn = torch.exp(attn).view(sb_len, sb_len)
            # attens[sb[0]:sb[1], sb[0]:sb[1]] = exp_attn / (exp_attn.sum(1).unsqueeze(1) + 10E-8)

        return attens


class AttentionCIDNN(nn.Module):
    def __init__(self):
        super(AttentionCIDNN, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear( 2, 32), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.fc3 = nn.Linear(64, 64)

    def forward(self, x, sub_batches=[]):
        bs = x.shape[0]
        x = x[:, -1]  # Just use the last locations

        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)

        attens = torch.zeros(bs, bs).cuda()
        for sb in sub_batches:
            sb_len = int(sb[1] - sb[0])
            h_sb = h[sb[0]:sb[1]]
            attn = torch.mm(h_sb, h_sb.transpose(0, 1))
            attn = attn - attn.max(1)[0]
            exp_attn = torch.exp(attn).view(sb_len, sb_len)
            attens[sb[0]:sb[1], sb[0]:sb[1]] = exp_attn / (exp_attn.sum(1).unsqueeze(1) + 10E-8)

        return attens


class LstmEncoder(nn.Module):
    def __init__(self, hidden_size=100, n_layers=2):
        self.hidden_size = hidden_size
        super(LstmEncoder, self).__init__()
        self.lstm = nn.LSTM(2, self.hidden_size, num_layers=n_layers, batch_first=True)
        self.lstm_h = []

    def init_lstm(self, h, c):
        self.lstm_h = (h, c)

    def forward(self, x):
        # bs = x.shape[0]
        y, self.lstm_h = self.lstm(x[:, :, 0:2], self.lstm_h)
        return y


class PredictorFC(nn.Module):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        super(PredictorFC, self).__init__()
        self.fc1 = torch.nn.Linear(self.hidden_size, 2)

    def forward(self, coded_tracks, attentions, sub_batches=[]):
        bs = coded_tracks.shape[0]
        out = torch.zeros(bs, self.hidden_size).cuda()
        for sb in sub_batches:
            out[sb[0]:sb[1]] = torch.mm(attentions[sb[0]:sb[1], sb[0]:sb[1]], coded_tracks[sb[0]:sb[1]])
        out = self.fc1(out)
        return out


class PredictorLSTM(nn.Module):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        super(PredictorLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(self.hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(torch.nn.Linear(hidden_size, 32), nn.ReLU(),
                                torch.nn.Linear(32,  2))

        self.lstm_h = []

    def init_lstm(self, h, c):
        self.lstm_h = (h, c)

    def forward(self, history_code=[], social_codes=[], noise=[]):
        bs = noise.shape[0]
        # inp = torch.cat([history_code, social_codes, noise], dim=1)
        inp = torch.FloatTensor(torch.zeros(bs, 1, hidden_size)).cuda()
        out, self.lstm_h = self.lstm(inp, self.lstm_h)
        out = self.fc(out.squeeze())
        return out


csv_file = '../data/eth/obsmat.txt'
data_file = '../data/eth/data.npz'
model_file = '../trained_models/encoder_decoder_vanilla.pt'

# parser = BIWIParser()
# parser.load(csv_file, down_sample=1)
# t_range = range(int(parser.min_t), int(parser.max_t), parser.interval)
# dataset_x, dataset_y, dataset_t = create_dataset(parser.p_data, parser.t_data, t_range)
# np.savez(data_file, dataset_x=dataset_x, dataset_y=dataset_y, dataset_t=dataset_t)
# exit(1)

data = np.load(data_file)
dataset_obsv, dataset_pred, the_batches = data['dataset_x'], data['dataset_y'], data['dataset_t']
train_size = (len(the_batches) * 4) // 5
n_next = dataset_pred.shape[1]
n_train_samples = the_batches[train_size-1][1]
n_test_samples = dataset_obsv.shape[0] - n_train_samples

# normalize
max_x = max(np.max(dataset_obsv[:, :, 0]), np.max(dataset_pred[:, :, 0]))
min_x = min(np.min(dataset_obsv[:, :, 0]), np.min(dataset_pred[:, :, 0]))
max_y = max(np.max(dataset_obsv[:, :, 1]), np.max(dataset_pred[:, :, 1]))
min_y = min(np.min(dataset_obsv[:, :, 1]), np.min(dataset_pred[:, :, 1]))
sx = (max_x - min_x)
sy = (max_y - min_y)
ss = max(sx, sy)
dataset_obsv = (dataset_obsv - np.array([min_x, min_y])) / ss
dataset_pred = (dataset_pred - np.array([min_x, min_y])) / ss
dataset_obsv = torch.FloatTensor(dataset_obsv).cuda()
dataset_pred = torch.FloatTensor(dataset_pred).cuda()


hidden_size = 64
n_lstm_layers = 1
lstm_encoder = LstmEncoder(hidden_size, n_lstm_layers).cuda()
attention = AttentionRel().cuda()
# predictor = PredictorFC(hidden_size=hidden_size).cuda()
lstm_decoder = PredictorLSTM(hidden_size=hidden_size).cuda()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(chain(attention.parameters(),
                                   lstm_encoder.parameters(),
                                   lstm_decoder.parameters()), lr=45e-4, weight_decay=2e-4)

if os.path.isfile(model_file):
    checkpoint = torch.load(model_file)
    start_epoch = checkpoint['epoch'] + 1
    min_train_ADE = checkpoint['min_error']
    attention.load_state_dict(checkpoint['attentioner_dict'])
    lstm_encoder.load_state_dict(checkpoint['lstm_encoder_dict'])
    lstm_decoder.load_state_dict(checkpoint['predictor_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    min_train_ADE = 10000
    start_epoch = 1


def predict(obsv, n_next):
    n_past = obsv.shape[1]
    bs = obsv.shape[0]
    lstm_h_c = (torch.zeros(n_lstm_layers, bs, lstm_encoder.hidden_size).cuda(),
                torch.zeros(n_lstm_layers, bs, lstm_encoder.hidden_size).cuda())
    lstm_encoder.init_lstm(lstm_h_c[0], lstm_h_c[1])
    for si in range(n_past):
        _ = lstm_encoder(obsv[:, si, :].unsqueeze(1))

    # take the hidden state of encoder and put it in decoder
    lstm_decoder.init_lstm(lstm_encoder.lstm_h[0], lstm_encoder.lstm_h[1])
    for si in range(n_next):
        # attn = attention(xs, sub_batches)
        pred_hat = lstm_decoder([], [], noise=torch.randn((bs, 16)))
        # y_hat = predictor(lstm_out.squeeze(), attn, sub_batches)
        pred_hat = pred_hat + obsv[:, -1]
        obsv = torch.cat((obsv, pred_hat.unsqueeze(1)), dim=1)

    pred_hat = obsv[:, n_past:, :]
    return pred_hat


for epoch in range(start_epoch, 15000 + 1):
    bch_size_accum = 0; sub_batches = []
    tot_train_err = 0
    for ii in range(0, train_size):
        bch = the_batches[ii]

        bch_size_accum += bch[1] - bch[0]
        sub_batches.append(bch)
        if bch_size_accum > 250 or ii == train_size-1:
            sub_batches = sub_batches - sub_batches[0][0]
            obsv = dataset_obsv[sub_batches[0][0]:sub_batches[-1][1]]
            pred = dataset_pred[sub_batches[0][0]:sub_batches[-1][1]]

            pred_hat = predict(obsv, n_next)
            batch_loss = loss_func(pred_hat, pred)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            bch_size_accum = 0
            sub_batches = []

            # calculate error
            with torch.no_grad():
                dd = torch.sqrt(torch.sum(torch.pow(ss * (pred_hat - pred), 2), dim=2))
                tot_train_err += dd.sum().item() / n_next

    train_ADE = tot_train_err / n_train_samples
    if train_ADE < min_train_ADE:
        min_train_ADE = train_ADE
    is_best_epoch = train_ADE < min_train_ADE

    # calc test error
    tot_test_err = 0
    for ii in range(train_size, len(the_batches)):
        bch = the_batches[ii]
        obsv = dataset_obsv[bch[0]:bch[1]]
        pred = dataset_pred[bch[0]:bch[1]]
        with torch.no_grad():
            pred_hat = predict(obsv, n_next)
            dd = torch.sqrt(torch.sum(torch.pow(ss * (pred_hat - pred), 2), dim=2))
            tot_test_err += dd.sum().item() / n_next

    print("epc = %4d, Train ADE = %.3f | Test ADE = %.3f" \
          % (epoch, train_ADE, tot_test_err/n_test_samples))

    if epoch % 10 == 0:
        print('saving model to file ...')
        save_checkpoint({
            'epoch': epoch,
            'min_error': min_train_ADE,
            'attentioner_dict': attention.state_dict(),
            'lstm_encoder_dict': lstm_encoder.state_dict(),
            'predictor_dict': lstm_decoder.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best_epoch, model_file)


