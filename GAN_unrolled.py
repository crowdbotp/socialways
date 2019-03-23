import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os.path
import scipy.optimize as op
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

# ==== Generate sim dataset =====
out_dir = "../unrolled_2"
os.makedirs(out_dir, exist_ok=True)
print('writing to ' + out_dir)
log_1NN = open(out_dir + '/log_1NN.txt', "w+")
log_1NN.write('real,  fake,  total\n')

n_samples = 512
n_past = 2
n_next = 2
noise_vec_len = 32
samples_len = n_past + n_next


def create_toy_dataset(n_samples, save=False):
    real_samples = []
    timesteps = []
    n_modes = 3
    n_conditions = 6
    for ii in range(n_samples):
        selected_way = (ii * n_conditions) // n_samples
        timestep_ii = ((ii * n_conditions) % n_samples) // n_conditions * 4
        data_angle = selected_way * (360 / n_conditions)
        data_angle = data_angle * np.pi / 180

        x0 = math.cos(data_angle) * 8
        y0 = math.sin(data_angle) * 8
        x1 = math.cos(data_angle) * 6
        y1 = math.sin(data_angle) * 6

        fixed_turn = ((ii % n_modes) - 1) * 20 * np.pi / 180
        p2_turn_rand = np.random.randn(1) * 1.5 * np.pi / 180
        p3_turn_rand = np.random.randn(1) * 2.5 * np.pi / 180

        x2 = math.cos(data_angle + fixed_turn + p2_turn_rand) * 4
        y2 = math.sin(data_angle + fixed_turn + p2_turn_rand) * 4

        x3 = math.cos(data_angle + fixed_turn + p2_turn_rand + p3_turn_rand) * 2
        y3 = math.sin(data_angle + fixed_turn + p2_turn_rand + p3_turn_rand) * 2

        real_samples.append(np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]]))
        timesteps.append(np.array([timestep_ii, timestep_ii+1, timestep_ii+2, timestep_ii+3]))

    real_samples = np.array(real_samples) / 8

    if save:
        with open(out_dir + '/toy-dataset-f2.txt', 'w+') as gt_file:
            # gt_file.write('% each row contains n points: x(1), y(1), ... y(n)\n')
            for ii, sample in enumerate(real_samples):
                sample = np.reshape(sample, (-1, 2))
                # gt_file.write("".join(map(str, sam)) + "\n")
                for tt, val in enumerate(sample):
                    gt_file.write("%.1f %.1f %.3f %.3f\n" % (timesteps[ii][tt], ii+1, val[0], val[1]))
                # gt_file.write("\n")
            gt_file.close()
    return real_samples


def compute_wasserstein(reals, fakes):
    n_reals = len(reals)
    n_fakes = len(fakes)
    D = np.ones((n_reals, n_fakes)) * 1000

    for ii in range(0, n_reals):
        for jj in range(0, n_fakes):
            sR = reals[ii]
            sF = fakes[jj]
            if sR[0, 0] - sF[0, 0] > 0.001:
                continue
            dij = math.sqrt((sR[2, 0] - sF[2, 0]) ** 2 + (sR[2, 1] - sF[2, 1]) ** 2 + \
                            (sR[3, 0] - sF[3, 0]) ** 2 + (sR[3, 1] - sF[3, 1]) ** 2)
            D[ii, jj] = dij
    row_ind, col_ind = op.linear_sum_assignment(D)
    cost = D[row_ind, col_ind].sum()
    return cost/n_reals


def compute_1nn(reals, fakes):
    smplz = []
    n_reals = len(reals)
    n_fakes = len(fakes)
    N = n_reals + n_fakes
    D = np.ones((N, N)) * 1000

    for ii in range(0, len(reals)):
        sample_and_label = [reals[ii], [1]]
        smplz.append(sample_and_label)
    for ii in range(0, len(fakes)):
        sample_and_label = [fakes[ii], [-1]]
        smplz.append(sample_and_label)

    for ii in range(0, N):
        for jj in range(ii+1, N):
            sA = smplz[ii][0]
            sB = smplz[jj][0]
            if sA[0, 0] - sB[0, 0] > 0.001:
                continue
            dij = math.sqrt((sA[2, 0] - sB[2, 0]) ** 2 + (sA[2, 1] - sB[2, 1]) ** 2 + \
                            (sA[3, 0] - sB[3, 0]) ** 2 + (sA[3, 1] - sB[3, 1]) ** 2)
            D[ii, jj], D[jj, ii] = dij, dij

    Real_pos = 0
    Real_neg = 0
    Fake_pos = 0
    Fake_neg = 0
    for ii in range(0, N):
        NN_ind = np.argmin(D[ii])
        if smplz[ii][1][0] == 1 and smplz[NN_ind][1][0] == 1:
            Real_pos += 1
        elif smplz[ii][1][0] == 1 and smplz[NN_ind][1][0] == -1:
            Real_neg += 1
        elif smplz[ii][1][0] == -1 and smplz[NN_ind][1][0] == -1:
            Fake_pos += 1
        else:  # if all_samples[ii][1][0] == -1 and all_samples[NN_ind][1][0] == 1:
            Fake_neg += 1
    return (Real_pos + Fake_pos) / N, Real_pos/n_reals, Fake_pos / n_fakes


def fc_block(in_size, out_size, activation=nn.LeakyReLU, bn=False):
    block = [nn.Linear(in_size, out_size),
             activation(0.2, inplace=True)]
    if bn: block.append(nn.BatchNorm1d(out_size, 0.8))
    return nn.Sequential(*block)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.hidden_size = 64
        self.lstm_encoder = nn.LSTM(2, self.hidden_size, 1, batch_first=True).cuda()
        self.obsv_encoder = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),nn.Sigmoid()).cuda()
        self.noise_encoder = fc_block(noise_vec_len, self.hidden_size)    .cuda()
        self.fc_out = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.Tanh(),
                                    # fc_block(hidden_size, hidden_size),
                                    nn.Linear(self.hidden_size, n_next * 2)).cuda()

    def forward(self, x, z):
        batch_size = x.size(0)

        init_state = (torch.zeros(1, batch_size, self.hidden_size).cuda(),
                      torch.zeros(1, batch_size, self.hidden_size).cuda())
        (lh, _) = self.lstm_encoder(x.cuda(), init_state)
        lh = lh[:, -1, :].view(batch_size, -1)  # I just need the last output: (batch_size, 1, H)
        xh = self.obsv_encoder(lh)

        zh = self.noise_encoder(z.view(batch_size, -1).cuda())
        hh = torch.cat([xh, zh], dim=1)

        lh = self.fc_out(hh)
        return lh.view(batch_size, n_next, 2)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        hidden_size = 32
        self.obsv_encoder = nn.Sequential(nn.Linear(n_past * 2, hidden_size),
                                          nn.Tanh(),
                                          fc_block(hidden_size, hidden_size)).cuda()
        self.pred_encoder = nn.Sequential(nn.Linear(n_next * 2, hidden_size),
                                          fc_block(hidden_size, hidden_size)).cuda()
        self.some_layers = nn.Sequential(nn.Linear(hidden_size + hidden_size, hidden_size),
                                         fc_block(hidden_size, hidden_size)).cuda()
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid()).cuda()

    def forward(self, x, y):
        batch_size = x.size(0)
        xh = self.obsv_encoder(x.view(batch_size, -1).cuda())
        yh = self.pred_encoder(y.view(batch_size, -1).cuda())

        hh = torch.cat([xh, yh], dim=1)
        hh = self.some_layers(hh)
        label = self.classifier(hh)

        return label

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

# bce_loss = nn.BCELoss()
# def adv_loss(input_, target_):
#     return bce_loss(input_, target_)


def adv_loss(input_, target_):
    neg_abs = -input_.abs()
    _loss = input_.clamp(min=0) - input_ * target_ + (1 + neg_abs.exp()).log()
    return _loss.mean()


mse_loss = nn.MSELoss()

unrolled_steps = 10
G = Generator()
D = Discriminator()
d_learning_rate = 1e-4
g_learning_rate = 1e-3
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))
last_epc = -1
last_kld = -1
last_emd = -1

real_samples = create_toy_dataset(n_samples, save=True)
exit(1)

train_data = TensorDataset(torch.FloatTensor(real_samples[:, :n_past]),
                           torch.FloatTensor(real_samples[:, n_past:]))
train_loader = DataLoader(train_data, batch_size=n_samples, shuffle=True, num_workers=1)

for epc in trange(1, 100000+1):
    d_loss_fake_sum = 0
    d_loss_real_sum = 0
    g_loss_sum = 0
    recon_loss_sum = 0

    # --- train G and D together ---
    for _, (x, y) in enumerate(train_loader):
        bs = x.size(0)  # batch size
        z = Variable(torch.FloatTensor(torch.randn(bs, noise_vec_len)), requires_grad=False).cuda()
        zeros = Variable(torch.zeros(bs, 1) + random.uniform(0, 0.2), requires_grad=False).cuda()
        ones = Variable(torch.ones(bs, 1) * random.uniform(0.8, 1.0), requires_grad=False).cuda()

        # ============== Train Discriminator ================ #
        for u in range(unrolled_steps + 1):
            D.zero_grad()
            with torch.no_grad():
                yhat = G(x, z)  # for updating discriminator

            validity = D(x, yhat)  # classify fake samples
            d_loss_fake = mse_loss(validity, zeros)

            real_value = D(x, y)  # classify real samples
            d_loss_real = mse_loss(real_value, ones)

            d_loss = d_loss_fake + d_loss_real
            d_loss.backward()  # to update D
            d_optimizer.step()

            if u == 0:
                backup = copy.deepcopy(D)

            d_loss_fake_sum += d_loss_fake.item()
            d_loss_real_sum += d_loss_real.item()

        # =============== Train Generator ================= #
        D.zero_grad()
        G.zero_grad()

        yhat = G(x, z)
        validity = D(x, yhat)  # classify a fake sample
        g_loss = mse_loss(validity, ones)
        g_loss.backward()
        g_optimizer.step()

        D.load(backup)
        del backup

        # ================================================
        # =================== T E S T ==================== #
    if (epc % 100) == 0:
        # ============== Visualize Results ============
        plt.figure(0)
        for i in range(0, n_samples):
            x = real_samples[i, :n_past]
            y = real_samples[i, n_past:]
            y = np.concatenate((x[-1].reshape(1, -1), y))

            plt.plot(x[:, 0], x[:, 1], 'r', linewidth=2.0, alpha=0.2)  # , label='obsv [%d]' %i)
            plt.plot(y[:, 0], y[:, 1], 'b', linewidth=2.0, alpha=0.2)  # , label='sample [%d]' %i)

        gen_samples = []
        for i in range(0, n_samples):
            x = real_samples[i, :n_past]
            z = torch.randn(noise_vec_len, 1)
            c = torch.rand(1, 1) * i / n_samples
            yhat1 = G(torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0),
                      z.type(torch.FloatTensor).unsqueeze(0))
            yhat1 = yhat1.cpu().data.numpy().reshape(-1, 2)
            yhat1 = np.concatenate((x, yhat1))
            plt.plot(yhat1[:, 0], yhat1[:, 1], 'r--', alpha=0.7)
            gen_samples.append(yhat1)
        gen_samples = np.array(gen_samples)

        Total_1NN, Reals_1NN, Fakes_1NN = compute_1nn(real_samples, gen_samples)
        EMD = compute_wasserstein(real_samples, gen_samples)
        if last_epc >= 0:
            plt.figure(1)
            plt.plot([last_epc, epc], [last_emd, EMD], 'b')
            plt.xlabel('Epoch')
            plt.title('Wasserstein Distance')
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(out_dir, 'EMD.svg'))

            plt.figure(2)
            gfc_reals_1nn, = plt.plot([last_epc, epc], [last_reals_1NN, Reals_1NN], 'c')
            gfc_fakse_1nn, = plt.plot([last_epc, epc], [last_fakes_1NN, Fakes_1NN], 'g')
            gfc_total_1nn, = plt.plot([last_epc, epc], [last_total_1NN, Total_1NN], 'b')
            plt.xlabel('Epoch')
            plt.title('1-NN accuracy')
            plt.legend((gfc_reals_1nn, gfc_fakse_1nn, gfc_total_1nn),
                       ('Real Samples', 'Generated Samples', 'Total'))
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(out_dir, '1NN.svg'))
        # last_kld = KLD
        last_epc = epc
        last_emd = EMD
        last_reals_1NN, last_fakes_1NN, last_total_1NN = Reals_1NN, Fakes_1NN, Total_1NN
        log_1NN.write('%.3f, %.3f, %.3f\n' % (Reals_1NN, Fakes_1NN, Total_1NN))
        log_1NN.flush()

        plt.figure(0)
        plt.title('Epoch = %d' % epc)
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.savefig(os.path.join(out_dir, 'out_%05d.png' % epc))
        plt.savefig(os.path.join(out_dir, 'last.svg'))

        with open(out_dir + '/out-%05d.csv' % epc, 'w+') as out_csv_file:
            out_csv_file.write('% each row contains n points: x(1), y(1), ... y(n)\n')
            for sample in gen_samples:
                sample = np.reshape(sample, (-1, 1))
                # gt_file.write("".join(map(str, sam)) + "\n")
                for val in sample:
                    out_csv_file.write("%.4f, " % val[0])
                out_csv_file.write("\n")
            out_csv_file.close()

        # plt.show()
        plt.clf()