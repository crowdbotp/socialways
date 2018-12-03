import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt


# ==== Generate sim dataset =====
n_past = 2
n_next = 2
samples_len = n_past + n_next
n_modes = 3
samples = []
for ii in range(128):
    p0 = [-1, 1]
    p1 = [1, 1 + (ii % 2) * 0.0]
    if ii % n_modes == 0:
        p2 = [2, 0]
        p3 = [3, 0]
    elif ii % n_modes == 1:
        p2 = [2, 2]
        p3 = [3, 2]
    elif ii % n_modes == 2:
        p2 = [2, 3]
        p3 = [3, 3]

    samples.append(np.array([p0, p1, p2, p3]))

n_samples = len(samples)
samples = np.array(samples)
noise_vec_len = 1
code_vec_len = 1
samples = samples / 5

train_data = TensorDataset(torch.FloatTensor(samples[:, :n_past]),
                                            torch.FloatTensor(samples[:, n_past:]))
train_loader = DataLoader(train_data, batch_size=n_samples, shuffle=True, num_workers=1)


def plot_samples():
    for i in range(0, n_samples):
        x = samples[i, :n_past]
        y = samples[i, n_past:]
        y = np.concatenate((x[-1].reshape(1, -1), y))
        plt.plot(y[:, 0], y[:, 1], label='sample [%d]' % i)
        plt.text(y[-1, 0], y[-1, 1], 'sample %d' % i)

    plt.plot(x[:, 0], x[:, 1], 'y', label='obsv')
    plt.text(x[0, 0], x[0, 1] + 0.02, 'observation')
    plt.xlim([-0.2, 1])
    plt.ylim([-0.1, 0.5])
    plt.show()


def fc_block(in_size, out_size, bn=False):
    block = [nn.Linear(in_size, out_size),
             nn.LeakyReLU(0.2, inplace=True)]
    if bn: block.append(nn.BatchNorm1d(out_size, 0.8))
    return nn.Sequential(*block)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        hidden_size = 32
        self.obsv_encoder = fc_block(n_past * 2, hidden_size).cuda()
        self.noise_encoder = fc_block(noise_vec_len, hidden_size)    .cuda()
        self.code_encoder = fc_block(code_vec_len, hidden_size).cuda()
        self.fc_out = nn.Sequential(fc_block(hidden_size * 3, hidden_size),
                                    fc_block(hidden_size, n_next * 2)).cuda()

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
        self.pred_encoder = fc_block(n_next * 2, hidden_size).cuda()
        self.fc_out_label = fc_block(hidden_size + hidden_size, hidden_size).cuda()

    def forward(self, x, y):
        batch_size = x.size(0)
        xh = self.obsv_encoder(x.view(batch_size, -1).cuda())
        yh = self.pred_encoder(y.view(batch_size, -1).cuda())

        hh = torch.cat([xh, yh], dim=1)
        label = self.fc_out_label(hh)
        # code = self.fc_out_code(hh)
        # out = torch.cat([label, code], dim=1)
        return label  # out


class Qclass(nn.Module):
    def __init__(self):
        super(Qclass, self).__init__()
        hidden_size = 32
        self.obsv_encoder = nn.Sequential(nn.Linear(n_past * 2, hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size, hidden_size)).cuda()
        self.pred_encoder = nn.Sequential(nn.Linear(n_next * 2, hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size, hidden_size)).cuda()

        self.fc_out_code = nn.Linear(hidden_size + hidden_size, 1).cuda()

    def forward(self, x, y):
        batch_size = x.size(0)
        xh = self.obsv_encoder(x.view(batch_size, -1).cuda())
        yh = self.pred_encoder(y.view(batch_size, -1).cuda())

        hh = torch.cat([xh, yh], dim=1)
        code = self.fc_out_code(hh)
        return code





def bce_loss(input_, target_):
    neg_abs = -input_.abs()
    _loss = input_.clamp(min=0) - input_ * target_ + (1 + neg_abs.exp()).log()
    return _loss.mean()


G = Generator()
D = Discriminator()
Q = Qclass()
d_learning_rate = 3e-3
g_learning_rate = 3e-3
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate)
q_optimizer = torch.optim.Adam(Q.parameters(), lr=g_learning_rate)
mse_loss = nn.MSELoss()
landa = 1000

for epc in range(1, 100000+1):
    d_loss_fake_sum = 0
    d_loss_real_sum = 0
    g_loss_sum = 0
    recon_loss_sum = 0

    # for si in samples:
    # x = torch.from_numpy(si[:n_past]).type(torch.FloatTensor)
    # y = torch.from_numpy(si[n_past:]).type(torch.FloatTensor)

    # --- codes to train G and D together ---

    for _, (x, y) in enumerate(train_loader):
        bs = x.size(0)  # batch size
        z = Variable(torch.FloatTensor(torch.randn(bs, noise_vec_len)), requires_grad=False).cuda()
        code = Variable(torch.FloatTensor(torch.rand(bs, code_vec_len)), requires_grad=False).cuda()
        zeros = Variable(torch.zeros(bs, 1) + random.uniform(0, 0.2), requires_grad=False).cuda()
        ones = Variable(torch.ones(bs, 1) * random.uniform(0.8, 1.0), requires_grad=False).cuda()

        # ============== Train Discriminator ================ #
        yhat = G(x, z, code)  # for updating discriminator
        yhat.detach()
        D.zero_grad()

        fake_value = D(x, yhat)  # classify fake samples
        real_value = D(x, y)  # classify real samples

        d_loss_fake = bce_loss(fake_value, zeros)
        d_loss_real = bce_loss(real_value, ones)

        d_loss_fake.backward()  # to update D
        d_loss_real.backward()  # to update D
        d_optimizer.step()

        d_loss_fake_sum += d_loss_fake.item()
        d_loss_real_sum += d_loss_real.item()

        # =============== Train Generator ================= #
        D.zero_grad()
        G.zero_grad()
        Q.zero_grad()

        yhat = G(x, z, code)
        fake_value = D(x, yhat)  # classify fake samples
        g_loss = bce_loss(fake_value, ones)
        g_loss.backward(retain_graph=True)

        code_hat = Q(x, yhat)  # classify fake samples
        c_loss = mse_loss(code_hat, code) * 1000
        c_loss.backward()  # through Q and G

        # yhat = G(x, z, c)
        # g_gt_loss = mse_loss(yhat, y) * landa
        # g_gt_loss.backward()

        g_optimizer.step()
        q_optimizer.step()

        g_loss_sum += g_loss.item()
        recon_loss_sum += c_loss.item()
        # ********** Variety loss ************
        # y_hat_2 = generator(xs, noise_2)  # for updating generator
        # c_hat_fake_2 = discriminator(xs, y_hat_2)  # classify fake samples
        # gen_loss_variety = torch.log(y_hat_2 - y_hat_1)

        # gen_loss_fooling = discriminationLoss(c_hat_fake_2, Variable(torch.ones(batch_size, 1, 1).cuda()))
        # gen_loss_fooling = bce_loss(c_hat_fake_1, (torch.ones(batch_size, 1, 1) * random.uniform(0.7, 1.2)).cuda())

        # loss_gt = groundTruthLoss(yhat, y) / n_next
        # gen_loss_gt += loss_gt.item()
        # gen_loss = loss_gt # + (gen_loss_fooling * generator.use_noise * lambda_dc_loss)
        #
        # gen_loss.backward()
        # g_optimizer.step()
        #
        # gen_total_loss_accum += gen_loss.item()
    print('Epoch: [%d] | G Loss=%4f | D Loss Fake=%4f | D Loss Real=%4f | c Reconstruct Loss=%3f'
          % (epc, g_loss_sum/n_samples, d_loss_fake_sum/n_samples, d_loss_real_sum/n_samples, recon_loss_sum/n_samples))

    if (epc % 500) == 0:
        for i in range(0, n_modes):
            x = samples[i, :n_past]
            y = samples[i, n_past:]
            z = torch.rand(noise_vec_len, 1)
            # c = torch.rand(1, 1)
            c = torch.ones(1, 1) * i/(n_modes -1)
            yhat = G(torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0),
                     z.type(torch.FloatTensor).unsqueeze(0),
                     c.type(torch.FloatTensor).unsqueeze(0))
            code_hat = Q(torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0), yhat)
            # print(c_hat - c)
            yhat = yhat.cpu().data.numpy().reshape(-1, 2)

            y = np.concatenate((x[-1].reshape(1, -1), y))
            yhat = np.concatenate((x[-1].reshape(1, -1), yhat))
            plt.plot(x[:, 0], x[:, 1])  # , label='obsv [%d]' %i)
            plt.plot(y[:, 0], y[:, 1], label='sample [%d]' % i)
            plt.plot(yhat[:, 0], yhat[:, 1], '--', label='pred [c=%.2f]' % c.item())

        plt.title('Epoch = %d' %epc)
        # plt.suptitle('lambda = %f' %landa)
        plt.legend()
        plt.xlim([-0.2, 1])
        plt.ylim([-0.1, 0.8])
        plt.show()

