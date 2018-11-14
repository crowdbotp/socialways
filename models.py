import torch
import torch.nn as nn

# n_next = 8
# n_inp_features = 2


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

        self.fc_in = nn.Sequential(nn.Linear(2, self.inp_size_lstm), nn.LeakyReLU(0.5)).cuda()

        # self.embedding = nn.Linear(inp_len * 2, hidden_size_1)
        self.lstm = nn.LSTM(input_size=self.inp_size_lstm, hidden_size=self.hidden_size_lstm,
                            num_layers=self.n_lstm_layers, batch_first=True, bidirectional=self.is_blstm).cuda()

        self.fc_out = nn.Linear(self.hidden_size_2, 2).cuda()
        self.drop_out_1 = nn.Dropout(0.1)
        self.drop_out_2 = nn.Dropout(0.05)

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
                                  , nn.LeakyReLU(0.5)).cuda()
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

