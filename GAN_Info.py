import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os.path
import scipy.optimize as op

# ==== Generate sim dataset =====
n_samples = 1280 # 256
n_past = 2
n_next = 2
samples_len = n_past + n_next
n_modes = 3
n_conditions = 6
real_samples = []

for ii in range(n_samples):
    data_angle = (ii * n_conditions) // n_samples * (360 / n_conditions)
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
real_samples = np.array(real_samples) / 8

for ii in range(len(real_samples)):
    plt.plot(real_samples[ii, :n_past, 0], real_samples[ii, :n_past, 1], 'm')
    plt.plot(real_samples[ii, n_past-1:, 0], real_samples[ii, n_past-1:, 1], 'b', linewidth=4, alpha=0.07)
    plt.plot(real_samples[ii, -1:, 0], real_samples[ii, -1:, 1], 'go', ms=4, alpha=0.4)
    plt.plot(real_samples[ii, 0, 0], real_samples[ii, 0, 1], 'ro')
plt.plot(0,0, 'gx')
plt.title('Trajectory Toy Dataset')
plt.show()
exit(1)

noise_vec_len = 1
code_vec_len = 1

out_dir = "../dual_loss"
print('writing to ' + out_dir)
os.makedirs(out_dir, exist_ok=True)
log_1NN = open(out_dir + '/log_1NN.txt', "w+")
log_1NN.write('real,  fake,  total\n')

train_data = TensorDataset(torch.FloatTensor(real_samples[:, :n_past]),
                           torch.FloatTensor(real_samples[:, n_past:]))
train_loader = DataLoader(train_data, batch_size=n_samples, shuffle=True, num_workers=1)


def plot_samples():
    for i in range(0, n_samples):
        x = real_samples[i, :n_past]
        y = real_samples[i, n_past:]
        y = np.concatenate((x[-1].reshape(1, -1), y))
        plt.plot(y[:, 0], y[:, 1], 'g', label='sample [%d]' % i)
        # plt.text(y[-1, 0], y[-1, 1], 'sample %d' % i)
        plt.plot(x[:, 0], x[:, 1], 'b', label='obsv')
        # plt.text(x[0, 0], x[0, 1] + 0.02, 'observation')

    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.title('Toy Example 2')
    plt.show()


def compute_kld():
    # KL Divergence Estimation
    # (by 'y' element of last coordinate in the sample trajectory)
    n_bins = 30
    bin_bounds = np.linspace(-0.25, 1.25, n_bins + 1)
    target_hist = np.zeros(n_bins, dtype=float)
    gen_hist = np.zeros(n_bins, dtype=float)
    for i in range(0, n_samples):
        y = real_samples[i, -1, 1]
        bin_ind = int((y - bin_bounds[0]) // (bin_bounds[1] - bin_bounds[0]))
        if 0 <= bin_ind < n_bins:
            target_hist[bin_ind] += 1 / n_samples

    n_generate_samples = 1000
    xs = np.tile(real_samples[:, :n_past], (1 + n_generate_samples // n_samples, 1, 1))
    xs = xs[:n_generate_samples]
    zs = torch.randn(n_generate_samples, noise_vec_len, 1)
    cs = torch.rand(n_generate_samples, code_vec_len, 1)
    yhats = G(torch.from_numpy(xs).type(torch.FloatTensor),
              zs.type(torch.FloatTensor), cs.type(torch.FloatTensor)).cpu().data.numpy()
    for i in range(n_generate_samples):
        bin_ind = int((yhats[i, -1, 1] - bin_bounds[0]) // (bin_bounds[1] - bin_bounds[0]))
        if 0 <= bin_ind < n_bins:
            gen_hist[bin_ind] += 1 / n_generate_samples

    # ============ Save KLD Plot ============= #
    KLD = 0
    for bb in range(n_bins):
        KLD += gen_hist[bb] * np.log((gen_hist[bb] + 1e-6) / (target_hist[bb] + 1e-6))
    return KLD


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
        self.obsv_encoder = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                          nn.Sigmoid()).cuda()
        self.noise_encoder = fc_block(noise_vec_len, self.hidden_size)    .cuda()
        self.code_encoder = nn.Sequential(nn.Linear(code_vec_len, self.hidden_size),
                                          fc_block(self.hidden_size, self.hidden_size),
                                          nn.Linear(self.hidden_size, self.hidden_size),
                                          nn.Sigmoid()).cuda()
        self.fc_out = nn.Sequential(fc_block(self.hidden_size * 3, self.hidden_size),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.Sigmoid(),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.Sigmoid(),
                                    # fc_block(hidden_size, hidden_size),
                                    nn.Linear(self.hidden_size, n_next * 2)).cuda()

    def forward(self, x, z, c):
        batch_size = x.size(0)

        init_state = (torch.zeros(1, batch_size, self.hidden_size).cuda(),
                      torch.zeros(1, batch_size, self.hidden_size).cuda())
        (lh, _) = self.lstm_encoder(x.cuda(), init_state)
        lh = lh[:, -1, :].view(batch_size, -1)  # I just need the last output: (batch_size, 1, H)
        xh = self.obsv_encoder(lh)

        zh = self.noise_encoder(z.view(batch_size, -1).cuda())
        ch = self.code_encoder(c.view(batch_size, -1).cuda())
        hh = torch.cat([xh, zh, ch], dim=1)

        lh = self.fc_out(hh)
        return lh.view(batch_size, n_next, 2)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        hidden_size = 32
        self.obsv_encoder = nn.Sequential(nn.Linear(n_past * 2, hidden_size),
                                          fc_block(hidden_size, hidden_size)).cuda()
        self.pred_encoder = nn.Sequential(nn.Linear(n_next * 2, hidden_size),
                                          fc_block(hidden_size, hidden_size)).cuda()
        self.some_layers = nn.Sequential(nn.Linear(hidden_size + hidden_size, hidden_size),
                                         fc_block(hidden_size, hidden_size)).cuda()
        self.classifier = nn.Linear(hidden_size, 1).cuda()
        self.code_estimator = nn.Linear(hidden_size, code_vec_len).cuda()

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        batch_size = x.size(0)
        xh = self.obsv_encoder(x.view(batch_size, -1).cuda())
        yh = self.pred_encoder(y.view(batch_size, -1).cuda())

        hh = torch.cat([xh, yh], dim=1)
        hh = self.some_layers(hh)
        label = self.classifier(hh)
        code_h = self.code_estimator(hh)

        return label, code_h

# bce_loss = nn.BCELoss()
# def adv_loss(input_, target_):
#     return bce_loss(input_, target_)

def adv_loss(input_, target_):
    neg_abs = -input_.abs()
    _loss = input_.clamp(min=0) - input_ * target_ + (1 + neg_abs.exp()).log()
    return _loss.mean()

mse_loss = nn.MSELoss()

G = Generator()
D = Discriminator()
g_learning_rate = 1e-3
d_learning_rate = 1e-4
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate)
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate)
landa = 1000
last_epc = -1
last_kld = -1
last_emd = -1

fig = plt.gcf()

for epc in trange(1, 200000+1):
    d_loss_fake_sum = 0
    d_loss_real_sum = 0
    g_loss_sum = 0
    recon_loss_sum = 0

    # --- train G and D together ---
    for _, (x, y) in enumerate(train_loader):
        bs = x.size(0)  # batch size
        z = Variable(torch.FloatTensor(torch.randn(bs, noise_vec_len)), requires_grad=False).cuda()
        code1 = Variable(torch.FloatTensor(torch.rand(bs, code_vec_len)), requires_grad=False).cuda() # uniform
        code2 = Variable(torch.FloatTensor(torch.rand(bs, code_vec_len)), requires_grad=False).cuda()  # uniform
        zeros = Variable(torch.zeros(bs, 1) + random.uniform(0, 0.2), requires_grad=False).cuda()
        ones = Variable(torch.ones(bs, 1) * random.uniform(0.8, 1.0), requires_grad=False).cuda()

        # ============== Train Discriminator ================ #
        D.zero_grad()
        G.zero_grad()
        yhat1 = G(x, z, code1)  # for updating discriminator
        yhat1.detach()

        validity1, _ = D(x, yhat1)  # classify fake samples
        d_loss_fake = mse_loss(validity1, zeros)

        real_value, _ = D(x, y)  # classify real samples
        d_loss_real = mse_loss(real_value, ones)

        d_loss = d_loss_fake + d_loss_real
        d_loss.backward()  # to update D
        d_optimizer.step()

        d_loss_fake_sum += d_loss_fake.item()
        d_loss_real_sum += d_loss_real.item()

        # =============== Train Generator ================= #
        D.zero_grad()
        G.zero_grad()

        yhat1 = G(x, z, code1)
        validity1, _ = D(x, yhat1)  # classify a fake sample
        # yhat2 = G(x, z, code2)
        # validity2, _ = D(x, yhat2)  # classify a fake sample
        g_loss = mse_loss(validity1, ones)  # + mse_loss(validity2, ones)
        g_loss.backward()
        g_optimizer.step()

        # ================== Maximize Info ================ #
        D.zero_grad()
        G.zero_grad()

        yhat1 = G(x, z, code1)
        _, code_hat1 = D(x, yhat1)  # classify fake samples
        yhat2 = G(x, z, code2)
        _, code_hat2 = D(x, yhat2)  # classify fake samples
        info_loss = 50 * (mse_loss(code_hat1, code1) + mse_loss(code_hat2, code2))
        # info_loss += 25 * mse_loss(code2, code1) / mse_loss(yhat2, yhat1)
        info_loss.backward()  # through Q and G

        g_optimizer.step()
        d_optimizer.step()

        g_loss_sum += g_loss.item()
        recon_loss_sum += info_loss.item()

    # ================================================
    # =================== T E S T ==================== #
    if (epc % 200) == 0:
        # ============== Visualize Results ============
        plt.figure(0)
        for i in range(0, n_samples):
            x = real_samples[i, :n_past]
            y = real_samples[i, n_past:]
            y = np.concatenate((x[-1].reshape(1, -1), y))

            plt.plot(x[:, 0], x[:, 1], 'b', linewidth=3.0)  # , label='obsv [%d]' %i)
            plt.plot(y[:, 0], y[:, 1], 'g', linewidth=3.0)  # , label='sample [%d]' %i)

        K = n_samples
        gen_samples = []
        for i in range(0, K):
            x = real_samples[i, :n_past]
            z = torch.randn(noise_vec_len, 1)
            c = torch.rand(1, 1) * i/K
            yhat1 = G(torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0),
                      z.type(torch.FloatTensor).unsqueeze(0),
                      c.type(torch.FloatTensor).unsqueeze(0))
            _, code_hat1 = D(torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0), yhat1)
            yhat1 = yhat1.cpu().data.numpy().reshape(-1, 2)
            yhat1 = np.concatenate((x, yhat1))
            plt.plot(yhat1[:, 0], yhat1[:, 1], '--', alpha=0.7)
            gen_samples.append(yhat1)
        gen_samples = np.array(gen_samples)

        Total_1NN, Reals_1NN, Fakes_1NN = compute_1nn(real_samples, gen_samples)
        # KLD = compute_kld()
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
        plt.title('Epoch = %d' %epc)
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        fig.set_size_inches(10, 10)
        plt.savefig(os.path.join(out_dir, 'out_%05d.png' % epc))
        plt.savefig(os.path.join(out_dir, 'last.svg'))

        # plt.show()
        plt.clf()

