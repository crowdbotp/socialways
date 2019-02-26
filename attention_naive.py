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


class Attentioner(nn.Module):
    def __init__(self):
        super(Attentioner, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear( 2, 32), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.fc3 = nn.Linear(64, 64)

    def forward(self, x, sub_batches=[]):
        bs = x.shape[0]

        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)

        attens = torch.zeros(bs, bs).cuda()
        for sb in sub_batches:
            sb_len = int(sb[1] - sb[0])
            h_sb = h[sb[0]:sb[1]]
            attn = torch.mm(h_sb, h_sb.transpose(0, 1))
            attn = attn - attn.max(1)[0]
            # print(attn)
            exp_attn = torch.exp(attn).view(sb_len, sb_len)
            # print(exp_attn)
            attens[sb[0]:sb[1], sb[0]:sb[1]] = exp_attn / (exp_attn.sum(1).unsqueeze(1) + 10E-8)
            # print(attens[sb[0]:sb[1], sb[0]:sb[1]], '\n*******************')

        return attens


class LstmEncoder(nn.Module):
    def __init__(self, hidden_size=100, n_layers=2):
        self.hidden_size = hidden_size
        super(LstmEncoder, self).__init__()
        self.lstm = nn.LSTM(2, self.hidden_size, num_layers=n_layers, batch_first=True)

    def forward(self, x, h):
        bs = x.shape[0]
        y, h = self.lstm(x[:, :, 0:2], h)
        return y, h


class Predictor(nn.Module):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        super(Predictor, self).__init__()
        self.fc1 = torch.nn.Linear(self.hidden_size, 2)

    def forward(self, coded_tracks, attentions, sub_batches=[]):
        bs = coded_tracks.shape[0]
        out = torch.zeros(bs, self.hidden_size).cuda()
        for sb in sub_batches:
            out[sb[0]:sb[1]] = torch.mm(attentions[sb[0]:sb[1], sb[0]:sb[1]], coded_tracks[sb[0]:sb[1]])
        out = self.fc1(out)
        return out


csv_file = '../data/eth/obsmat.txt'
data_file = '../data/eth/data.npz'
# parser = BIWIParser()
# parser.load(csv_file, down_sample=1)
# t_range = range(int(parser.min_t), int(parser.max_t), parser.interval)
# dataset_x, dataset_y, dataset_t = create_dataset(parser.p_data, parser.t_data, t_range)
# np.savez(data_file, dataset_x=dataset_x, dataset_y=dataset_y, dataset_t=dataset_t)
# exit(1)

data = np.load(data_file)
dataset_x, dataset_y, the_batches = data['dataset_x'], data['dataset_y'], data['dataset_t']
train_size = (len(the_batches) * 4 ) // 5
n_train_samples = the_batches[train_size-1][1]

# normalize
sx = (np.max(dataset_x) - np.min(dataset_x))
sy = (np.max(dataset_y) - np.min(dataset_y))
ss = max(sx, sy)
dataset_x = (dataset_x-np.min(dataset_x))/ss
dataset_y = (dataset_y-np.min(dataset_y))/ss
dataset_x = torch.FloatTensor(dataset_x).cuda()
dataset_y = torch.FloatTensor(dataset_y).cuda()

# all_time_steps = np.union1d(np.concatenate(all_time_steps))
# for t in all_time_steps:
# #     print(t)
#
# fps = parser.actual_fps
hidden_size = 64
n_lstm_layers = 1
lstm_encoder = LstmEncoder(hidden_size, n_lstm_layers).cuda()
attentioner = Attentioner().cuda()
predictor = Predictor(hidden_size=hidden_size).cuda()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(chain(attentioner.parameters(),
                                   lstm_encoder.parameters(),
                                   predictor.parameters()), lr=2e-3, weight_decay=2e-5)

model_file = '../trained_models/attentioner.pt'
if os.path.isfile(model_file):
    checkpoint = torch.load(model_file)
    start_epoch = checkpoint['epoch'] +1
    min_err = checkpoint['min_error']
    attentioner.load_state_dict(checkpoint['attentioner_dict'])
    lstm_encoder.load_state_dict(checkpoint['lstm_encoder_dict'])
    predictor.load_state_dict(checkpoint['predictor_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    min_err = 10000
    start_epoch = 1

for epoch in range(start_epoch, 1000 + 1):
    n = 0; sub_batches = []
    epc_tot_err = 0
    for ii in range(train_size):
        bch = the_batches[ii]

        n += bch[1] - bch[0]
        sub_batches.append(bch)
        if n > 250 or ii == train_size-1:
            sub_batches = sub_batches - sub_batches[0][0]
            xs = dataset_x[sub_batches[0][0]:sub_batches[-1][1]]
            ys = dataset_y[sub_batches[0][0]:sub_batches[-1][1]]

            n_past = xs.shape[1]
            n_next = ys.shape[1]
            bs = xs.shape[0]
            lstm_h = (torch.zeros(n_lstm_layers, bs, lstm_encoder.hidden_size).cuda(),
                      torch.zeros(n_lstm_layers, bs, lstm_encoder.hidden_size).cuda())
            for si in range(n_past):
                lstm_out, lstm_h = lstm_encoder(xs[:, si, :].unsqueeze(1), lstm_h)

            for si in range(n_next):
                attn = attentioner(xs[:, -1, :], sub_batches)
                y_hat = predictor(lstm_out.squeeze(), attn, sub_batches)
                y_hat = y_hat + xs[:, -1]
                xs = torch.cat((xs, y_hat.unsqueeze(1)), dim=1)
                lstm_out, lstm_h = lstm_encoder(xs[:, si, :].unsqueeze(1), lstm_h)

            y_hat = xs[:, n_past:, :]
            batch_loss = loss_func(y_hat, ys)
            epc_tot_err += batch_loss.item() * n * n_next * 2
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            n = 0; sub_batches = []

            with torch.no_grad():
                dd = torch.sqrt(torch.sum(torch.pow(ss * (y_hat - ys), 2), dim=2))
                epc_tot_err += dd.sum().item() / n_next
                if np.math.isnan(epc_tot_err):
                    print(attn)

    err = epc_tot_err / n_train_samples
    if err < min_err:
        min_err = err
    is_best = err < min_err

    print("epc = %4d, Train ADE = %.3f" % (epoch, err))

    if epoch % 10 == 0:
        print('saving model to file ...')
        save_checkpoint({
            'epoch': epoch,
            'min_error': min_err,
            'attentioner_dict': attentioner.state_dict(),
            'lstm_encoder_dict': lstm_encoder.state_dict(),
            'predictor_dict': predictor.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best, model_file)


