import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os.path

# ==== Generate sim dataset =====
n_past = 2
n_next = 2
samples_len = n_past + n_next
n_modes = 3
n_conditions = 3
samples = []
for ii in range(128):
    cond = (ii//n_modes) % n_conditions
    if cond == 0:
        p0 = np.array([0, 0.5])
        dir = 1
    elif cond == 1:
        p0 = np.array([0, -1.0])
        dir = 1
    else:
        p0 = np.array([5, -0.5])
        dir = -1

    p1 = dir * np.array([1, 0]) + p0
    r = np.random.randn(1) / 50
    if ii % n_modes == 0:
        p2 = dir * np.array([1, -1+r]) + p1
    elif ii % n_modes == 1:
        p2 = dir * np.array([1, 1+r]) + p1
    else: # if ii % n_modes == 2:
        p2 = dir * np.array([1, 2+r]) + p1
    p3 = p2 + np.array([1, 0]) * dir

    samples.append(np.array([p0, p1, p2, p3]))

n_samples = len(samples)
samples = np.array(samples)
samples = samples / 5
#
for ii in range(n_samples):
    plt.plot(samples[ii,:,0], samples[ii,:,1])
plt.show()
exit(1)

noise_vec_len = 1
code_vec_len = 1

out_dir = "../gan_out"
os.makedirs(out_dir, exist_ok=True)

train_data = TensorDataset(torch.FloatTensor(samples[:, :n_past]),
                           torch.FloatTensor(samples[:, n_past:]))
train_loader = DataLoader(train_data, batch_size=n_samples, shuffle=True, num_workers=1)


def plot_samples():
    for i in range(0, n_samples):
        x = samples[i, :n_past]
        y = samples[i, n_past:]
        y = np.concatenate((x[-1].reshape(1, -1), y))
        plt.plot(y[:, 0], y[:, 1], 'g', label='sample [%d]' % i)
        # plt.text(y[-1, 0], y[-1, 1], 'sample %d' % i)
        plt.plot(x[:, 0], x[:, 1], 'b', label='obsv')
        # plt.text(x[0, 0], x[0, 1] + 0.02, 'observation')

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.6, 0.6])
    plt.title('Toy Example 2')
    plt.show()




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        hidden_size = 64
        self.obsv_encoder = nn.Sequential(fc_block(n_past * 2, hidden_size),
                                          fc_block(hidden_size, hidden_size, False)).cuda()
        self.noise_encoder = fc_block(noise_vec_len, hidden_size)    .cuda()
        self.code_encoder = fc_block(code_vec_len, hidden_size).cuda()
        self.fc_out = nn.Sequential(fc_block(hidden_size * 3, hidden_size),
                                    fc_block(hidden_size, hidden_size),
                                    fc_block(hidden_size, hidden_size),
                                    # fc_block(hidden_size, hidden_size),
                                    nn.Linear(hidden_size, n_next * 2)).cuda()

    def forward(self, x, z, c):
        batch_size = x.size(0)
        xh = self.obsv_encoder(x.view(batch_size, -1).cuda())
        zh = self.noise_encoder(z.view(batch_size, -1).cuda())
        ch = self.code_encoder(c.view(batch_size, -1).cuda())
        hh = torch.cat([xh, zh, ch], dim=1)

        xh = self.fc_out(hh)
        return xh.view(batch_size, n_next, 2)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        hidden_size = 32
        self.obsv_encoder = fc_block(n_past * 2, hidden_size).cuda()
        self.pred_encoder = nn.Sequential(fc_block(n_next * 2, hidden_size),
                                          fc_block(hidden_size, hidden_size)).cuda()
        self.some_layers = nn.Sequential(fc_block(hidden_size + hidden_size, hidden_size),
                                         fc_block(hidden_size, hidden_size),
                                         fc_block(hidden_size, hidden_size),
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
g_learning_rate = 4e-4
d_learning_rate = 4e-4
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate)
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate)
landa = 1000
last_epc = -1
last_kld = -1

for epc in trange(1, 100000+1):
    d_loss_fake_sum = 0
    d_loss_real_sum = 0
    g_loss_sum = 0
    recon_loss_sum = 0

    # --- codes to train G and D together ---
    for _, (x, y) in enumerate(train_loader):
        bs = x.size(0)  # batch size
        z = Variable(torch.FloatTensor(torch.randn(bs, noise_vec_len)), requires_grad=False).cuda()
        code = Variable(torch.FloatTensor(torch.rand(bs, code_vec_len)), requires_grad=False).cuda()
        zeros = Variable(torch.zeros(bs, 1) + random.uniform(0, 0.2), requires_grad=False).cuda()
        ones = Variable(torch.ones(bs, 1) * random.uniform(0.8, 1.0), requires_grad=False).cuda()

        # ============== Train Discriminator ================ #
        D.zero_grad()
        G.zero_grad()
        yhat = G(x, z, code)  # for updating discriminator
        yhat.detach()

        fake_value, _ = D(x, yhat)  # classify fake samples
        d_loss_fake = adv_loss(fake_value, zeros)

        real_value, _ = D(x, y)  # classify real samples
        d_loss_real = adv_loss(real_value, ones)

        d_loss = d_loss_fake + d_loss_real
        d_loss.backward()  # to update D
        d_optimizer.step()

        d_loss_fake_sum += d_loss_fake.item()
        d_loss_real_sum += d_loss_real.item()

        # =============== Train Generator ================= #
        D.zero_grad()
        G.zero_grad()

        yhat = G(x, z, code)
        fake_value, _ = D(x, yhat)  # classify fake samples
        g_loss = adv_loss(fake_value, ones)
        g_loss.backward(retain_graph=True)
        g_optimizer.step()

        # ================== Maximize Info ================ #
        D.zero_grad()
        G.zero_grad()

        yhat = G(x, z, code)
        fake_value, code_hat = D(x, yhat)  # classify fake samples
        c_loss = mse_loss(code_hat, code) * 100
        c_loss.backward()  # through Q and G

        g_optimizer.step()
        d_optimizer.step()

        g_loss_sum += g_loss.item()
        recon_loss_sum += c_loss.item()

    # =================== TEST ====================
    if (epc % 100) == 0:
        # KL Divergence Estimation (by 'y' element of last coordinate in the sample trajectory)
        n_bins = 30
        bin_bounds = np.linspace(-0.25, 1.25, n_bins+1)
        target_hist = np.zeros(n_bins, dtype=float)
        gen_hist = np.zeros(n_bins, dtype=float)
        for i in range(0, n_samples):
            y = samples[i, -1, 1]
            bin_ind = int((y - bin_bounds[0]) // (bin_bounds[1] - bin_bounds[0]))
            if 0 <= bin_ind < n_bins:
                target_hist[bin_ind] += 1 / n_samples
        n_generate_samples = 1000
        xs = np.tile(samples[:, :n_past], (1 + n_generate_samples // n_samples, 1, 1))
        xs = xs[:n_generate_samples]
        zs = torch.randn(n_generate_samples, noise_vec_len, 1)
        cs = torch.rand(n_generate_samples, code_vec_len, 1)
        yhats = G(torch.from_numpy(xs).type(torch.FloatTensor),
                  zs.type(torch.FloatTensor), cs.type(torch.FloatTensor)).cpu().data.numpy()
        for i in range(n_generate_samples):
            bin_ind = int((yhats[i, -1, 1] - bin_bounds[0]) // (bin_bounds[1] - bin_bounds[0]))
            if 0 <= bin_ind < n_bins:
                gen_hist[bin_ind] += 1 / n_generate_samples

        # ============ Save KLD Plot =============#
        KLD = 0
        for bb in range(n_bins):
            KLD += gen_hist[bb] * np.log((gen_hist[bb] + 1e-6) / (target_hist[bb] + 1e-6))
        if last_epc >= 0:
            plt.figure(0)
            plt.plot([last_epc, epc], [last_kld, KLD], 'b')
            plt.xlabel('Itr')
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(out_dir, 'KLD.svg'))
        last_epc, last_kld = epc, KLD

        # ============== Visualize Results ============
        plt.figure(1)
        for i in range(0, n_samples):
            x = samples[i, :n_past]
            y = samples[i, n_past:]
            y = np.concatenate((x[-1].reshape(1, -1), y))

            plt.plot(x[:, 0], x[:, 1], 'b', linewidth=3.0)  # , label='obsv [%d]' %i)
            plt.plot(y[:, 0], y[:, 1], 'g', linewidth=3.0)  # , label='sample [%d]' %i)

        K = n_samples
        for i in range(0, K):
            x = samples[i, :n_past]
            z = torch.randn(noise_vec_len, 1)
            c = torch.rand(1, 1) * i/K
            yhat = G(torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0),
                     z.type(torch.FloatTensor).unsqueeze(0),
                     c.type(torch.FloatTensor).unsqueeze(0))
            _, code_hat = D(torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0), yhat)
            yhat = yhat.cpu().data.numpy().reshape(-1, 2)
            yhat = np.concatenate((x[-1].reshape(1, -1), yhat))
            plt.plot(yhat[:, 0], yhat[:, 1], '--', alpha=0.7)

        plt.title('Epoch = %d' %epc)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.6, 0.6])
        plt.savefig(os.path.join(out_dir, 'out_%05d.png' % epc))
        plt.savefig(os.path.join(out_dir, 'last.svg'))
        # plt.show()
        plt.clf()

