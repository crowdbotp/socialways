import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# n_next = 8
# n_inp_features = 2
from scipy import ndimage


class ConstVelModel:
    def __init__(self):
        pass

    def predict(self, obsv, n_next=8):  # use config for
        #inp.ndim = 2
        avg_vel = np.array([0, 0])
        if obsv.ndim > 1 and obsv.shape[0] > 1:
            # avg_vel = (obsv[-1, :] - obsv[0, :]) / (obsv.shape[0] - 1)
            avg_vel = (obsv[-1, :2] - obsv[-2, :2]) / 1

        cur_pos = obsv[-1, :2]
        out = np.empty((0, 2))
        for i in range(0, n_next):
            out = np.vstack((out, cur_pos + avg_vel * (i+1)))

        return out


class NaivePredictor(nn.Module):
    def __init__(self, n_inp_features, out_len):
        super(NaivePredictor, self).__init__()

        self.hidden_size_lstm = 64
        self.lstm = nn.LSTM(n_inp_features, self.hidden_size_lstm, batch_first=True).cuda()
        self.fc = nn.Sequential(nn.Linear(self.hidden_size_lstm, 64), nn.Linear(64, 2 * out_len)).cuda()
        self.use_noise = False

    def forward(self, x, noise, _=0):
        batch_size = x.size(0)
        lstm_hid = (torch.zeros(1, batch_size, self.hidden_size_lstm).cuda(),
                    torch.zeros(1, batch_size, self.hidden_size_lstm).cuda())
        y, _ = self.lstm(x[:, :, 0:2], lstm_hid)
        y = y[:, -1, :]
        y = self.fc(y)
        return y.view(batch_size, -1, 2)


class SequentialPredictor(nn.Module):
    def __init__(self):
        super(SequentialPredictor, self).__init__()

        self.hidden_size_lstm = 64
        self.lstm = nn.LSTM(2, self.hidden_size_lstm, batch_first=True).cuda()
        self.fc1 = nn.Sequential(nn.Linear(self.hidden_size_lstm, 64),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(64, 64)) .cuda()
        self.use_noise = False

        self.dropout1 = nn.Dropout(0.25)
        self.fc_out = nn.Sequential(nn.Linear(self.hidden_size_lstm, 64),
                                    nn.LeakyReLU(0.1),
                                    nn.Linear(64, 2)) .cuda()

    def forward(self, x, noise, out_len):
        x = x[:, :, 0:2]
        batch_size = x.size(0)

        lstm_hid = (torch.zeros(1, batch_size, self.hidden_size_lstm).cuda(),
                    torch.zeros(1, batch_size, self.hidden_size_lstm).cuda())

        output = torch.empty((batch_size, out_len, 2)).cuda()
        for t in range(out_len):
            y, lstm_hid = self.lstm(x[:, :, 0:2], lstm_hid)
            y = y[:, -1, :]
            y = self.fc1(y)
            y = self.dropout1(y)
            y = self.fc_out(y)
            x = torch.cat([x[:, :-1, :], (y + x[:, -1, :]).unsqueeze(1)], dim=1)
            output[:, t, :] = y
        return output.view(batch_size, -1, 2)


class SequentialPredictorWithVelocity(nn.Module):
    def __init__(self):
        super(SequentialPredictorWithVelocity, self).__init__()

        self.hidden_size_lstm = 32
        self.lstm_pos = nn.LSTM(2, self.hidden_size_lstm, batch_first=True).cuda()
        self.lstm_vel = nn.LSTM(2, self.hidden_size_lstm, batch_first=True).cuda()
        self.fc1 = nn.Sequential(nn.Linear(2 * self.hidden_size_lstm + 64, 64),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(64, 64)) .cuda()
        self.use_noise = False

        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)
        self.fc_out = nn.Sequential(nn.Linear(64, 64),
                                    nn.LeakyReLU(0.1),
                                    nn.Linear(64, 64),
                                    nn.LeakyReLU(0.1),
                                    nn.Linear(64, 2)) .cuda()

    def forward(self, x, noise, out_len):
        x = x[:, :, 0:2]
        batch_size = x.size(0)

        pos_lstm_hid = (torch.zeros(1, batch_size, self.hidden_size_lstm).cuda(),
                        torch.zeros(1, batch_size, self.hidden_size_lstm).cuda())
        vel_lstm_hid = (torch.zeros(1, batch_size, self.hidden_size_lstm).cuda(),
                        torch.zeros(1, batch_size, self.hidden_size_lstm).cuda())

        output = torch.empty((batch_size, out_len, 2)).cuda()
        for t in range(out_len):
            v = (x[:, -1, :] - x[:, 0, :]) / x.size(1)
            if t == 0:
                yp, pos_lstm_hid = self.lstm_pos(x[:, :, 0:2], pos_lstm_hid)
            else:
                yp, pos_lstm_hid = self.lstm_pos(x[:, -1, 0:2].view(batch_size, 1, -1), pos_lstm_hid)
            yp = yp[:, -1, :]
            yp = self.dropout2(yp)

            yv, vel_lstm_hid = self.lstm_vel(v.unsqueeze(1), vel_lstm_hid)
            y = torch.cat([yp, yv.view(batch_size, -1)], dim=1)

            y = torch.cat([y, noise], dim=1)
            y = self.fc1(y)
            y = self.dropout1(y)
            y = self.fc_out(y) + (x[:, -1, :] - x[:, -4, :]) / 3
            x = torch.cat([x[:, :-1, :], (y + x[:, -1, :]).unsqueeze(1)], dim=1)
            output[:, t, :] = y
        return output.view(batch_size, -1, 2)


class Generator(nn.Module):
    def __init__(self, inp_len, out_len, n_inp_features, noise_len):
        super(Generator, self).__init__()
        # self.out_len = out_len

        self.use_noise = True      # For using it as a solely predictor
        self.noise_len = noise_len

        self.n_lstm_layers = 1
        self.inp_size_lstm = n_inp_features
        self.hidden_size_lstm = 48
        self.hidden_size_2 = 48
        self.hidden_size_3 = 48
        self.is_blstm = False

        self.fc_in = nn.Sequential(nn.Linear(2, self.inp_size_lstm), nn.LeakyReLU(0.1)).cuda()

        # self.embedding = nn.Linear(inp_len * 2, hidden_size_1)
        self.lstm = nn.LSTM(input_size=self.inp_size_lstm, hidden_size=self.hidden_size_lstm,
                            num_layers=self.n_lstm_layers, batch_first=True, bidirectional=self.is_blstm).cuda()

        self.fc_out = nn.Linear(self.hidden_size_2, 2).cuda()
        self.drop_out_1 = nn.Dropout(0.2)
        self.drop_out_2 = nn.Dropout(0.2)

        # Hidden Layers
        self.fc_1 = nn.Sequential(
            nn.Linear(self.hidden_size_lstm * (1 + self.is_blstm) + self.noise_len * self.use_noise, self.hidden_size_2)
            , nn.Sigmoid()).cuda()

        self.fc_2 = nn.Sequential(nn.Linear(self.hidden_size_2, self.hidden_size_3)
                                  , nn.LeakyReLU(0.1)).cuda()

    def forward(self, obsv, noise, out_len):
        # input: (B, seq_len, 2)
        # noise: (B, N)
        batch_size = obsv.size(0)
        obsv = obsv[:, :, 0:self.inp_size_lstm]

        # ===== To use Dense Layer in the input ======
        # rr = []
        # for tt in range(obsv.size(1)):
        #     rr.append(self.fc_in(obsv[:, tt, :]))
        # obsv = torch.stack(rr, 1)

        # initialize hidden state: (num_layers, minibatch_size, hidden_dim)
        lstm_hid = (torch.zeros(self.n_lstm_layers * (1 + self.is_blstm), batch_size, self.hidden_size_lstm).cuda(),
                    torch.zeros(self.n_lstm_layers * (1 + self.is_blstm), batch_size, self.hidden_size_lstm).cuda())

        pred_batch = torch.empty((batch_size, out_len, 2)).cuda()
        for tt in range(out_len):
            (lstm_out, lstm_hid) = self.lstm(obsv, lstm_hid)  # encode the input
            last_lstm_out = lstm_out[:, -1, :].view(batch_size, 1, -1)  # just keep the last output: (batch_size, 1, H)


            # combine data with noise
            if self.use_noise:
                lstm_out_and_noise = torch.cat([last_lstm_out, noise.cuda().view(batch_size, 1, -1)], dim=2)
            else:
                lstm_out_and_noise = last_lstm_out

            # lstm_out_and_noise = self.drop_out_1(lstm_out_and_noise)
            u = self.fc_1(lstm_out_and_noise)
            # u = self.drop_out_2(u)
            u = self.fc_2(u)
            next_pred_batch = self.fc_out(u).view(batch_size, 2)
            pred_batch[:, tt, :] = next_pred_batch
            next_pred_batch = next_pred_batch + obsv[:, -1, :]
            # vel_batch = (next_pred_batch - obsv[:, -1, :2]) * parser.actual_fps
            # new_obsv = torch.cat((next_pred_batch, vel_batch), dim=1).view(batch_size, 1, -1)
            # obsv = torch.cat([obsv[:, 1:, :], new_obsv], dim=1)
            obsv = torch.cat([obsv[:, 1:, :], next_pred_batch.view(-1, 1, 2)], dim=1)

        return pred_batch


class Discriminator(nn.Module):
    def __init__(self, obsv_len, pred_len, n_inp_features):
        super(Discriminator, self).__init__()
        self.out_size_lstm = 32
        self.hidden_size_fc = 32
        self.obsv_len = obsv_len
        self.pred_len = pred_len
        self.n_lstm_layers = 1
        self.is_blstm_obsv = False
        self.is_blstm_pred = False
        self.inp_size_lstm_obsv = n_inp_features
        self.inp_size_lstm_pred = 2
        self.lstm_obsv = nn.LSTM(input_size=self.inp_size_lstm_obsv, hidden_size=self.out_size_lstm,
                                 num_layers=self.n_lstm_layers, batch_first=True,
                                 bidirectional=self.is_blstm_obsv).cuda()

        self.lstm_pred = nn.LSTM(input_size=self.inp_size_lstm_pred, hidden_size=self.out_size_lstm,
                                 num_layers=self.n_lstm_layers, batch_first=True,
                                 bidirectional=self.is_blstm_pred).cuda()

        self.fc_1 = nn.Sequential(nn.Linear(self.out_size_lstm * (1 + self.is_blstm_pred) +
                                            self.out_size_lstm * (1 + self.is_blstm_obsv), self.hidden_size_fc)
                                  , nn.LeakyReLU(0.1)).cuda()
        self.classifier = nn.Linear(self.hidden_size_fc, 1).cuda()

    def forward(self, obsv, pred):
        # obsv: (B, in_seq_len, F)
        # pred: (B, out_seq_len, F)
        batch_size = obsv.size(0)
        obsv = obsv[:, :, :self.inp_size_lstm_obsv]
        pred = pred[:, :, :self.inp_size_lstm_pred]

        # initialize hidden state of obsv_lstm: (num_layers, minibatch_size, hidden_dim)
        init_state1 = (torch.zeros(self.n_lstm_layers * (1 + self.is_blstm_obsv), batch_size, self.out_size_lstm).cuda(),
                       torch.zeros(self.n_lstm_layers * (1 + self.is_blstm_obsv), batch_size, self.out_size_lstm).cuda())

        # ! lstm_out: (batch_size, seq_len, H)
        (obsv_lstm_out, _) = self.lstm_obsv(obsv, init_state1)
        obsv_lstm_out = obsv_lstm_out[:, -1, :].view(batch_size, 1, -1)  # I just need the last output: (batch_size, 1, H)

        # initialize hidden state of pred_lstm: (num_layers, minibatch_size, hidden_dim)
        init_state2 = (torch.zeros(self.n_lstm_layers * (1 + self.is_blstm_pred), batch_size, self.out_size_lstm).cuda(),
                       torch.zeros(self.n_lstm_layers * (1 + self.is_blstm_pred), batch_size, self.out_size_lstm).cuda())

        # ! lstm_out: (batch_size, seq_len, H)
        (pred_lstm_out, _) = self.lstm_pred(pred, init_state2)
        pred_lstm_out = pred_lstm_out[:, -1, :].view(batch_size, 1, -1)  # I just need the last output: (batch_size, 1, H)

        concat_lstm_outputs = torch.cat([obsv_lstm_out, pred_lstm_out], dim=2)

        u = self.fc_1(concat_lstm_outputs)

        # c: (batch_size, 1, 1)
        c = self.classifier(u)
        return c

class VanillaLSTM(nn.Module):
    def __init__(self, feature_size, pred_length, hidden_size_lstm, num_layers=1):
        super(VanillaLSTM, self).__init__()
        self.feature_size = feature_size
        self.pred_length = pred_length
        self.hidden_size = hidden_size_lstm
        self.n_layers = num_layers
        self.is_blstm = False

        hidden_size_2 = 64
        hidden_size_3 = 64

        # Initialize Layers
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size_lstm,
                            num_layers=num_layers, batch_first=True).cuda()
        self.fc_out = nn.Linear(hidden_size_3, pred_length * 2).cuda()

        # Hidden Layers
        self.fc_1 = nn.Sequential(nn.Linear(hidden_size_lstm * (1 + self.is_blstm), hidden_size_2)
                                  , nn.LeakyReLU(0.2)).cuda()

        self.fc_2 = nn.Sequential(nn.Linear(hidden_size_2, hidden_size_3),
                                  nn.LeakyReLU(0.1)).cuda()

        # it gives out just one location
        # self.one_out = nn.Linear(hidden_size_3, 2).cuda()

        #self.lstm.weight_hh_l0.data.fill_(0)
        nn.init.xavier_uniform_(self.fc_out.weight)

        self.loss_func = nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def init_state(self, minibatch_size=1):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.n_layers, minibatch_size, self.hidden_size).cuda(),
                torch.zeros(self.n_layers, minibatch_size, self.hidden_size).cuda())

    # ============ Sequence Prediction ============
    def forward(self, x):
        batch_size = x.size(0)
        self.hidden = self.init_state(batch_size)
        (y, _) = self.lstm(x[:, :, :self.feature_size], self.hidden)
        y = y[:, -1, :].view(batch_size, 1, -1)  # just keep the last output: (batch_size, 1, H)
        # y = self.relu(y)
        y = self.fc_1(y)
        # y = self.fc_2(y)
        y = self.fc_out(y.view(batch_size, -1)).view(batch_size, -1, 2)
        return y



class NavigationNet(nn.Module):
    def __init__(self, out_len):
        super(NavigationNet, self).__init__()

        n_filters_1 = 4
        n_filters_2 = 1
        kernel_size = 5
        stride_1 = 1
        stride_2 = 1
        stride_3 = 1
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=n_filters_1,  # n_filters
                kernel_size=kernel_size,  # filter size
                stride=stride_1,  # filter movement/step
                padding=1,
            ), # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, xxx, xxx)
            ).cuda()

        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(
                in_channels=n_filters_1,
                out_channels=n_filters_2,
                kernel_size=kernel_size,
                stride=stride_2,
                padding=1),  # output shape (32, yyy, yyy)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # output shape (32, yyy/4, yyy/4)
            ).cuda()

        self.conv3 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                stride=stride_3,
                padding=1),  # output shape (32, yyy, yyy)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # output shape (32, yyy/4, yyy/4)
        ).cuda()

        self.lstm_obsv_size = 64
        self.lstm_teom_size = 64
        self.lstm_obsv = nn.LSTM(2, self.lstm_obsv_size, 1, batch_first=True).cuda()
        self.lstm_teom = nn.LSTM(6*6, self.lstm_teom_size, 1, batch_first=True).cuda()
        # self.fc_mix = nn.Sequential(
        #     nn.Linear(self.lstm_obsv_size + self.lstm_teom_size, mix_size), #nn.ReLU(),
        # ).cuda()

        self.obsv_fc = nn.Linear(self.lstm_obsv_size, 64).cuda()
        self.teom_fc = nn.Linear(self.lstm_teom_size, 64).cuda()
        mix_size = 64
        # self.fc_mix = nn.ModuleList([nn.Linear(64 + 64, mix_size) for _ in range(out_len)]).cuda()
        self.fc_mix = nn.Linear(64 + 64, mix_size).cuda()

        self.out = nn.Sequential(
            nn.Linear(mix_size, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 2)
        ).cuda()

    def forward(self, obsv, teom, out_len=-1, debug=False):
        # teom = [bs, n_next, H, W, nch]
        bs = teom.shape[0]
        ts = teom.shape[1]
        # if teom.dim() == 4:
        #     teom = teom.unsqueeze(4)
        if out_len < 0 or out_len > ts:
            out_len = ts

        obsv = obsv[:, :, 0:self.lstm_obsv.input_size]
        # initialize hidden state: (num_layers, minibatch_size, hidden_dim)
        obsv_hidden_state = (torch.zeros(1, bs, self.lstm_obsv_size).cuda(),
                             torch.zeros(1, bs, self.lstm_obsv_size).cuda())


        teom_hidden_state = (torch.zeros(1, bs, self.lstm_teom_size).cuda(),
                             torch.zeros(1, bs, self.lstm_teom_size).cuda())

        y = torch.empty(bs, out_len, 2).cuda()
        c = torch.empty(bs, out_len).cuda()
        cur_vel = (obsv[:, -1, :] - obsv[:, 0, :]) / out_len
        orig_teta = torch.atan2(cur_vel[:, 1], cur_vel[:, 0])
        orig_pos = obsv[:, -1, :]
        W_2 = teom.size(2) // 2
        H_2 = teom.size(3) // 2

        for t in range(out_len):
            # teom_ch0 = teom[:, t, :, :, 0]
            # teom_ch1 = teom[:, t, :, :, 0]
            # teom_channels = torch.stack((teom_ch0, teom_ch1), 1)

            cur_pos = obsv[:, -1, :]
            dp = cur_pos - orig_pos
            rad = torch.dist(dp, torch.zeros_like(dp))
            teta = torch.atan2(dp[:, 1], dp[:, 0])
            new_teta = teta - orig_teta

            xx = torch.mul(rad, torch.cos(new_teta))
            yy = torch.mul(rad, torch.sin(new_teta))
            im_center = torch.stack((xx * teom.size(2) + W_2,
                                     yy * teom.size(3) + H_2), dim=1).to(torch.int)

            teom_cropped = torch.zeros(bs, W_2, H_2).cuda()
            # for b in range(bs):
            #     im_center_b = torch.IntTensor([max(min(im_center[b, 0], 3 * W_2 // 2), W_2 // 2),
            #                                    max(min(im_center[b, 1], 3 * H_2 // 2), H_2 // 2)])
            #     teom_cropped[b, :, :] = teom[b, t,
            #                             im_center_b[0]-W_2/2:im_center_b[0]+W_2/2:,
            #                             im_center_b[1]-H_2/2:im_center_b[1]+H_2/2]
            #
            #     if debug:
            #         imm = teom_cropped[0, 1:63, :].cpu().data.numpy()
            #         imm = ndimage.rotate(imm, 90)
            #         plt.imshow(imm)
            #         plt.show()
            #
            # conved_teom = self.conv1(teom_cropped.unsqueeze(1))
            # conved_teom = self.conv2(conved_teom)
            # conved_teom = self.conv3(conved_teom)
            # conved_teom_flat = conved_teom.view(bs, 1, -1)
            conved_teom_flat = torch.zeros(bs, 1, 6*6).cuda()

            (teom_state, teom_hidden_state) = self.lstm_teom(conved_teom_flat, teom_hidden_state)
            teom_state = teom_state.view(bs, -1)
            teom_state = self.teom_fc(teom_state)
            if t == 0:
                (obsv_state, obsv_hidden_state) = self.lstm_obsv(obsv, obsv_hidden_state)
            else:
                (obsv_state, obsv_hidden_state) = self.lstm_obsv((y[:, t-1, :].view(bs, 1, -1)).clone(), obsv_hidden_state)
            last_obsv_state = obsv_state[:, -1, :]
            last_obsv_state = self.obsv_fc(last_obsv_state)

            memory_mix = torch.cat((teom_state, last_obsv_state), 1)
            # y_mix = self.fc_mix[t](memory_mix)
            y_mix = self.fc_mix(memory_mix)
            cur_vel = (obsv[:, -1, :] - obsv[:, -3, :]) / 2
            y_fc = self.out(y_mix) + obsv[:, -1, :] + cur_vel

            # y_fc = self.out(y_mix)
            # rad = y_fc[:, 0]
            # ang = y_fc[:, 1]
            # xx = torch.mul(rad, torch.cos(ang)) + obsv[:, -1, 0] + cur_vel[:, 0]
            # yy = torch.mul(rad, torch.sin(ang)) + obsv[:, -1, 1] + cur_vel[:, 1]
            # y[:, t, :] = torch.stack((xx, yy), dim=1)

            y[:, t, :] = y_fc

            obsv = torch.cat((obsv[:, 1:, :], y[:, t, :].unsqueeze(1)), dim=1)
            c_sum = torch.sum(torch.sum(teom_cropped[:, W_2//2-5:W_2//2+5:, H_2//2-5:H_2//2+5], dim=1), dim=1) / 100.
            c[:, t] = c_sum

        return y , c # ,memory_teom  # for visualization
