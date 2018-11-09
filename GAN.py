import glob
import os
import random
from math import atan2, cos, sin, sqrt

import imageio
import numpy as np
from matplotlib import transforms
from scipy import ndimage
from scipy.misc import imresize
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.learning_utils import adjust_learning_rate, MyConfig
from src.math_utils import ConstVelModel, cart2pol
from src.parse_utils import *
from src.kalman import MyKalman
from src.visualize import Display, FakeDisplay, to_image_frame
from src.vanilla_lstm import PredictorLSTM


data_dir = '../data/eth'
out_dir = '../teom/eth_gt/'
np.random.seed(1)
torch.manual_seed(1)
config = MyConfig(n_past=8, n_next=8)
n_past = config.n_past
n_next = config.n_next
n_inp_features = 4  # (x, y),  /vx, vy
n_out_features = 2

train_rate = 1.  # 0.8
learning_rate = 4e-3
weight_decay = 4e-3
lambda_l2_loss = 10000
lambda_dc_loss = 0
test_interval = 60  # FIXME << ===+   << ===+   << ===+
optim_betas = (0.9, 0.999)
def_batch_size = 128
n_epochs = 2000
noise_vec_len = 32


def bce_loss(input_, target_):
    neg_abs = -input_.abs()
    _loss = input_.clamp(min=0) - input_ * target_ + (1 + neg_abs.exp()).log()
    return _loss.mean()


class Generator(nn.Module):
    def __init__(self, inp_len, out_len, noise_len=noise_vec_len):
        super(Generator, self).__init__()
        self.out_len = out_len

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

        self.fc_out = nn.Linear(self.hidden_size_2, self.out_len * 2).cuda()
        self.drop_out_1 = nn.Dropout(0.1)
        self.drop_out_2 = nn.Dropout(0.05)

        # Hidden Layers
        self.fc_1 = nn.Sequential(
            nn.Linear(self.hidden_size_lstm * (1 + self.is_blstm) + self.noise_len * self.use_noise, self.hidden_size_2)
            , nn.Sigmoid()).cuda()

        self.fc_2 = nn.Sequential(nn.Linear(self.hidden_size_2, self.hidden_size_3)
                                  , nn.LeakyReLU(0.1)).cuda()

    def forward(self, obsv, noise):
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
        init_state = (torch.zeros(self.n_lstm_layers * (1 + self.is_blstm), batch_size, self.hidden_size_lstm).cuda(),
                      torch.zeros(self.n_lstm_layers * (1 + self.is_blstm), batch_size, self.hidden_size_lstm).cuda())

        (lstm_out, _) = self.lstm(obsv, init_state)  # encode the input
        last_lstm_out = lstm_out[:, -1, :].view(batch_size, 1, -1)  # just keep the last output: (batch_size, 1, H)

        # combine data with noise
        if self.use_noise:
            lstm_out_and_noise = torch.cat([last_lstm_out, noise.cuda().view(batch_size, 1, -1)], dim=2)
        else:
            lstm_out_and_noise = last_lstm_out

        lstm_out_and_noise = self.drop_out_1(lstm_out_and_noise)
        u = self.fc_1(lstm_out_and_noise)
        u = self.drop_out_2(u)
        u = self.fc_2(u)

        # decode the data to generate fake sample
        pred_batch = self.fc_out(u).view(batch_size, self.out_len, 2)
        return pred_batch


class Discriminator(nn.Module):
    def __init__(self, obsv_len, pred_len):
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


generator = Generator(n_past, n_next, noise_len=noise_vec_len)
discriminator = Discriminator(n_past, n_next)
discriminationLoss = nn.BCEWithLogitsLoss()  # Binary cross entropy
groundTruthLoss = nn.MSELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=optim_betas)
cv_model = ConstVelModel()


# parser = SeyfriedParser()
# pos_data, vel_data, time_data = parser.load('../data/sey01.sey')
parser = BIWIParser()
pos_data, vel_data, time_data = parser.load(os.path.join(data_dir, 'obsmat.txt'))
scale = parser.scale

n_ped = len(pos_data)
train_size = int(n_ped * train_rate)
test_size = n_ped - train_size
all_peds = list()
train_time_data = time_data[:train_size]
test_time_data = time_data[train_size:]


def prepare_data(smooth=True):
    print('Dont forget to smooth the trajectories?')
    if smooth:
        print('Yes! Smoothing the trajectories in train_set ...')
        for i in range(train_size):
            kf = MyKalman(1 / parser.actual_fps, n_iter=7)
            pos_data[i], vel_data[i] = kf.smooth(pos_data[i])

    # Scaling
    for i in range(len(pos_data)):
        pos_data[i] = scale.normalize(pos_data[i], shift=True)
        vel_data[i] = scale.normalize(vel_data[i], shift=False)
        _pv_i = np.hstack((pos_data[i], vel_data[i]))
        all_peds.append(_pv_i)
    train_peds = np.array(all_peds[:train_size])
    test_peds = np.array(all_peds[train_size:])

    dataset_x = []
    dataset_y = []
    for ped_i in train_peds:
        ped_i_tensor = torch.FloatTensor(ped_i)  # .cuda()
        seq_len = ped_i_tensor.size(0)
        for t in range(n_past, seq_len - n_next + 1, 1):
            _x = ped_i_tensor[t - n_past:t, :]
            _y = (ped_i_tensor[t:t + n_next, 0:2] - _x[-1, 0:2])
            dataset_x.append(_x)
            dataset_y.append(_y)
    dataset_x_tensor = torch.stack(dataset_x, 0)
    dataset_y_tensor = torch.stack(dataset_y, 0)

    train_data = torch.utils.data.TensorDataset(dataset_x_tensor, dataset_y_tensor)
    train_loader = DataLoader(train_data, batch_size=def_batch_size, shuffle=False, num_workers=4)
    return train_loader, train_peds, test_peds


def train(train_loader_):
    gen_loss_acc = 0
    dcr_loss_real = 0
    dcr_loss_fake = 0

    for i, (batch_x, batch_y) in enumerate(train_loader_):
        xs = batch_x.cuda()
        ys = batch_y.cuda()

        batch_size = xs.size(0)
        noise_1 = torch.rand(batch_size, noise_vec_len)
        noise_2 = torch.rand(batch_size, noise_vec_len)

        # =============== Train Discriminator ================= #
        discriminator.zero_grad()

        y_hat_1 = generator(xs, noise_1)  # for updating discriminator
        y_hat_1.detach()
        c_hat_fake_1 = discriminator(xs, y_hat_1)  # classify fake samples
        # disc_loss_fakes = discriminationLoss(c_hat_fake_1, Variable(torch.zeros(batch_size, 1, 1).cuda()))
        disc_loss_fakes = bce_loss(c_hat_fake_1, (torch.zeros(batch_size, 1, 1) + random.uniform(0, 0.3)).cuda())
        disc_loss_fakes.backward()

        # for _ in range(1 + 1 * (epoch < 50)):
        for _ in range(1):
            c_hat_real = discriminator(xs, ys)  # classify real samples
            # disc_loss_reals = discriminationLoss(c_hat_real, torch.ones(batch_size, 1, 1).cuda())
            disc_loss_reals = bce_loss(c_hat_real, (torch.ones(batch_size, 1, 1) * random.uniform(0.7, 1.2)).cuda())
            disc_loss_reals.backward()

        d_optimizer.step()

        dcr_loss_fake += disc_loss_fakes.item()
        dcr_loss_real += disc_loss_reals.item()

        # =============== Train Generator ================= #
        generator.zero_grad()
        discriminator.zero_grad()
        y_hat_1 = generator(xs, noise_1)  # for updating generator
        c_hat_fake_1 = discriminator(xs, y_hat_1)  # classify fake samples

        # ********** Variety loss ************
        # y_hat_2 = generator(xs, noise_2)  # for updating generator
        # c_hat_fake_2 = discriminator(xs, y_hat_2)  # classify fake samples
        # gen_loss_variety = torch.log(y_hat_2 - y_hat_1)

        # gen_loss_fooling = discriminationLoss(c_hat_fake_2, Variable(torch.ones(batch_size, 1, 1).cuda()))
        gen_loss_fooling = bce_loss(c_hat_fake_1, (torch.ones(batch_size, 1, 1) * random.uniform(0.7, 1.2)).cuda())

        gen_loss_gt = groundTruthLoss(y_hat_1, ys) / n_next
        # print('L2 loss = ', gen_loss_gt.item())
        gen_loss = (gen_loss_gt * lambda_l2_loss) + (gen_loss_fooling * generator.use_noise * lambda_dc_loss)

        gen_loss.backward()
        g_optimizer.step()

        gen_loss_acc += gen_loss.item()

    gen_loss_acc /= i
    dcr_loss_fake /= i
    dcr_loss_real /= i

    print('epoch [%3d/%d], Generator Loss: %.6f , Gen Error: %.6f || Dis Loss: Fake= %5f, Real= %.5f'
          % (0, n_epochs, gen_loss_fooling.item(), gen_loss_gt * lambda_l2_loss, dcr_loss_fake, dcr_loss_real))


# ====================== T E S T =======================
def evaluate(peds):
    running_loss = 0
    running_cntr = 0

    for ii, _ in enumerate(peds):
        ped_i = torch.FloatTensor(peds[ii]).cuda()
        for t in range(n_past, ped_i.size(0) - n_next + 1, n_past):
            x = ped_i[t - n_past:t, 0:n_inp_features].view(1, n_past, -1)

            y = (ped_i[t:t + n_next, 0:2] - x[0, -1, 0:2]).view(n_next, 2)
            y_hat = generator(x, torch.rand(1, noise_vec_len)).view(n_next, 2)

            loss = groundTruthLoss(y_hat, y)
            running_loss += loss.item()
            running_cntr += 1
    test_loss = running_loss / running_cntr
    print("Test loss = ", np.math.sqrt(test_loss) / scale.sx)


v_lstm = PredictorLSTM(4, 8, 128, num_layers=1)
v_lstm.load_state_dict(torch.load('./models/v-lstm.pt'))

GRID_SIZE = [129, 129]
Nch = 3
center = np.floor([GRID_SIZE[0] / 2, GRID_SIZE[1] / 2]).astype(int)
PIX_VALUE = 200
def build_teom(yi_hat, Y_i_hats, ped_id=0, start_t=0):
    # teom_mats = np.empty((0, GRID_SIZE, GRID_SIZE, Nch))
    teom_mats = []
    n_frames = yi_hat.shape[0]
    n_others = Y_i_hats.shape[0]
    n_samples = K
    goal = yi_hat[-1, :]
    gif_writer = imageio.get_writer(out_dir + "teom_x%d_%03d_%02d.gif" % (GRID_SIZE[0], ped_id, start_t), mode='I')
    for t in range(n_frames):
        goal_vec = goal - yi_hat[t, :]
        if t == n_frames - 1:
            goal_vec = goal - yi_hat[t-1, :]
        rot_angle = -atan2(goal_vec[1], goal_vec[0])
        # rot_angle = atan2(cur_vel[1], cur_vel[0])
        rot_matrix = [[cos(rot_angle), sin(-rot_angle)], [sin(rot_angle), cos(rot_angle)]]

        # Creating the Grids (Centered on the agent)
        cartesian_grid = np.zeros((GRID_SIZE[0], GRID_SIZE[1], Nch), dtype="uint8")
        cartesian_grid[center[0], center[1], 0] = 255

        rotated_goal = np.matmul(rot_matrix, goal_vec)
        if abs(rotated_goal[0]) >= 1: rotated_goal = rotated_goal / rotated_goal[0]
        cartesian_grid[int(round((rotated_goal[0] + 1) * center[0])),
                       int(round((rotated_goal[1] + 1) * center[1])), 2] = 255

        for j in range(n_others):
            for k in range(n_samples):
                [rot_x, rot_y] = np.matmul(rot_matrix, Y_i_hats[j, k, t, :] - yi_hat[t, :])  # Rotate
                r = sqrt(rot_x ** 2 + rot_y ** 2)
                th = atan2(rot_y, rot_x)
                # polar_loc = np.array([r, th])

                if r > 1: continue

                rot_coord = np.array([round((rot_x + 1) * center[0]), round((rot_y + 1) * center[1])]).astype(int)
                cartesian_grid[rot_coord[0], rot_coord[1], 0] = min(255, cartesian_grid[rot_coord[0], rot_coord[1], 0] + PIX_VALUE/K)  # * approach_rate[j]
                cartesian_grid[rot_coord[0], rot_coord[1], 1] = min(255, cartesian_grid[rot_coord[0], rot_coord[1], 1] + PIX_VALUE/K)  # * DCAs[j]


                # polar_coord_0 = int(round(r * (GRID_SIZE[0] - 1) / sqrt(2)))
                # polar_coord_1 = int(round(th * (GRID_SIZE[1] - 1) / (2 * np.pi) + cntr[1]))
                # aligned_polar_grid[polar_coord_0, polar_coord_1, 0] += approach_rate[j] * PIX_VALUE
                # aligned_polar_grid[polar_coord_0, polar_coord_1, 1] += DCAs[j] * PIX_VALUE

        teom_mats.append(cartesian_grid)
        cartesian_grid = ndimage.rotate(cartesian_grid, 90)
        # cartesian_grid = imresize(cartesian_grid, size=2000, interp='nearest')
        gif_writer.append_data(cartesian_grid)
    teom_mats = np.stack(teom_mats)
    return teom_mats


K = 100  # Number of samples
disp = FakeDisplay(data_dir)

# mapfile = os.path.join(data_dir, "map.png")
Hfile = os.path.join(data_dir, "H.txt")
H = np.loadtxt(Hfile)
Hinv = np.linalg.inv(H)


def build_teom_dataset(peds, time_data, samples_mode, interactive=False):
    for ii in range(0, len(peds), 1):
        ped_i_tensor = torch.FloatTensor(peds[ii]).cuda()
        for t1_ind_i in range(n_past, ped_i_tensor.size(0) - n_next + 1, n_past):
            Y_hat = np.empty((0, K, n_next, 2))

            ts = time_data[ii][t1_ind_i - n_past]
            t0 = time_data[ii][t1_ind_i - 1]
            t1 = time_data[ii][t1_ind_i]
            te = time_data[ii][t1_ind_i + n_next - 1]

            disp.grab_frame(t0)
            base = plt.gca().transData
            rot = transforms.Affine2D().rotate_deg(-90)

            # find neighbors
            for jj in range(len(peds)):
                ts_ind_j = np.array(np.where(time_data[jj] == ts))
                t1_ind_j = np.array(np.where(time_data[jj] == t1))
                te_ind_j = np.array(np.where(time_data[jj] == te))

                # check time overlap between trajectories
                # if ii == jj or ts_ind_j.size == 0 or te_ind_j.size == 0:
                if ii == jj or t1_ind_j.size == 0 or t1_ind_j[0][0] < 4:
                    continue
                t1_ind_j = t1_ind_j[0][0]
                # ts_ind_j = ts_ind_j[0][0]
                # te_ind_j = te_ind_j[0][0]

                ped_j = torch.FloatTensor(peds[jj]).cuda()
                xj = ped_j[max(t1_ind_j - n_past, 0):t1_ind_j, 0:n_inp_features].view(1, -1, n_inp_features)
                xj_np = xj.cpu().data.numpy().reshape((-1, n_inp_features))
                yj = (ped_j[t1_ind_j:t1_ind_j + n_next, 0:2] - xj[-1, -1, 0:2]).view(1, -1, 2)
                yj_np = yj.cpu().data.numpy().reshape((-1, 2))

                y_hat_j = []
                for kk in range(1, K+1):
                    if samples_mode == 'gt':
                        yj_hat_np = np.copy(yj_np)
                        for tj in range(len(yj_hat_np), n_next):
                            aaa = yj_hat_np[-1, :] + (xj_np[1, 0:2] - xj_np[0, 0:2])
                            yj_hat_np = np.vstack((yj_hat_np, aaa.reshape((1, 2))))

                        for tj in range(len(yj_hat_np)):
                            yj_hat_np[tj, :] += torch.randn(2).data.numpy() * 0.002 * (1+tj/2) + xj_np[-1, 0:2]

                    elif samples_mode == 'v-lstm':
                        yj_hat = v_lstm(xj).view(1, n_next, 2)
                        yj_hat_np = yj_hat.cpu().data.numpy().reshape((n_next, 2)) + xj_np[-1, 0:2]
                        for tj in range(len(yj_hat_np)):
                            yj_hat_np[tj, :] += torch.randn(2).data.numpy() * 0.002 * (1+tj/2) + xj_np[-1, 0:2]

                    else:  # samples_mode == 'gan':
                        yj_hat = generator(xj, torch.rand(1, noise_vec_len)).view(1, n_next, 2)
                        yj_hat_np = yj_hat.cpu().data.numpy().reshape((n_next, 2)) + xj_np[-1, 0:2]

                    # ==========================================
                    y_hat_j.append(yj_hat_np)

                    # ========== PLOT ==========
                    yj_hat_np = np.vstack((xj_np[-1, 0:2], yj_hat_np))
                    yj_hat_np_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(yj_hat_np), np.ones((yj_hat_np.shape[0],1)))))

                    if kk == 1 and len(Y_hat) == 1:
                        plt.plot(yj_hat_np_XY[:, 0], yj_hat_np_XY[:, 1], 'bo', markersize=3, transform=rot + base, label='Prediction Sample')
                    else:
                        plt.plot(yj_hat_np_XY[:, 0], yj_hat_np_XY[:, 1], 'bo', markersize=3, transform=rot + base)

                    disp.plot_path(scale.denormalize(yj_hat_np[:, 0:2]), jj)
                    if kk == K:
                        xj_np_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(xj_np[:, 0:2]), np.ones((xj_np.shape[0], 1)))))
                        plt.plot(xj_np_XY[:, 0], xj_np_XY[:, 1], 'g--', transform=rot + base)

                        yj_np_aug = np.vstack((np.zeros((1,2)), yj_np[:, 0:2])) + xj_np[-1, 0:2]
                        yj_np_aug_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(yj_np_aug), np.ones((yj_np_aug.shape[0],1)))))
                        plt.plot(yj_np_aug_XY[:, 0], yj_np_aug_XY[:, 1], 'g+', transform=rot + base)
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                y_hat_j = np.stack(y_hat_j).reshape((1, K, n_next, 2))
                Y_hat = np.vstack((Y_hat, y_hat_j))

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                disp.plot_ped(scale.denormalize(xj_np[-1, 0:2]))
                disp.plot_path(scale.denormalize(xj_np[:, 0:2]), ii, 'g.')
                disp.plot_path(scale.denormalize(yj_np[:, 0:2] + xj_np[-1, 0:2]), ii, 'g--')
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            xi = ped_i_tensor[t1_ind_i - n_past:t1_ind_i, 0:n_inp_features].view(1, n_past, -1)
            xi_np = xi.cpu().data.numpy().reshape((n_past, n_inp_features))
            yi = (ped_i_tensor[t1_ind_i:t1_ind_i + n_next, 0:2] - xi[0, -1, 0:2]).view(1, n_next, 2)
            yi_np = yi.cpu().data.numpy().reshape((n_next, 2))
            ci_real = discriminator(xi, yi)
            yi_hat = v_lstm(xi)
            # yi_hat = generator(xi, torch.rand(1, noise_vec_len)).view(1, n_next, 2)
            yi_hat = yi_hat.cpu().data.numpy().reshape((n_next, 2)) + xi_np[-1, 0:2]
            yi_hat_np = np.vstack((xi_np[-1, 0:2], yi_hat))

            # ======== Build TEOM Maps =========
            # ==================================
            teom_mats = build_teom(yi_hat, Y_hat, ii, t1_ind_i)
            out_filename = os.path.join(out_dir, '%03d_%02d.npz' % (ii, t1_ind_i))

            # preparing target vectors for training: delta-velocity w.r.t last velocity of agent
            target = []
            base_v = xi_np[-1, 2:4]  # last instant vel
            base_v = (xi_np[-1, 0:2] - xi_np[0, 0:2]) / (len(xi_np) - 1)  # observation avg vel
            # base_v = unit(base_v)  # it could be tested
            [_, base_theta] = cart2pol(base_v[0], base_v[1])
            for tf in range(n_next):
                dp_t = yi_np[tf, 0:2]  # - xi_np[-1, 0:2]
                [mag, abs_theta] = cart2pol(dp_t[0], dp_t[1])
                rel_theta = abs_theta - base_theta
                if rel_theta > np.pi:
                    rel_theta -= 2 * np.pi
                elif rel_theta < -np.pi:
                    rel_theta += 2 * np.pi
                target.append(np.array([mag, rel_theta]))

            np.savez(out_filename,
                     teom=teom_mats,
                     obsv=xi_np,
                     target=target,
                     gt=(ped_i_tensor[t1_ind_i:t1_ind_i + n_next, 0:2]).cpu().data.numpy().reshape((n_next, -1)))
            print('Saving data to %s' % out_filename)

            # =========== Const-Vel Prediction ========
            yi_cv = np.vstack((xi_np[-1, 0:2], cv_model.predict(xi_np[:, 0:2])))

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # disp.plot_path(scale.denormalize(xi_np[:, 0:2]), ii, 'g.')
            # disp.plot_path(scale.denormalize(yi_np[:, 0:2] + xi_np[-1, 0:2]), ii, 'g--')
            # disp.plot_ped(scale.denormalize(xi_np[-1, 0:2]), ii, color=(0, 100, 200))

            # ========================================
            yi_cv_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(yi_cv), np.ones((yi_cv.shape[0],1)))))
            plt.plot(yi_cv_XY[:, 0], yi_cv_XY[:, 1], 'c--', transform=rot + base, label='Const-Vel Prediction')
            # plt.plot(yi_cv[:, 0], yi_cv[:, 1], 'c--')

            yi_hat_np_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(yi_hat_np), np.ones((yi_hat_np.shape[0], 1)))))
            plt.plot(yi_hat_np_XY[:, 0], yi_hat_np_XY[:, 1], 'y--', transform=rot + base, label='Prior Prediction')
            # plt.plot(yi_hat_np[:, 0], yi_hat_np[:, 1], 'y--')

            xi_np_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(xi_np[:, 0:2]), np.ones((xi_np.shape[0],1)))))
            plt.plot(xi_np_XY[:, 0], xi_np_XY[:, 1], 'g--', transform=rot + base, label='Observation')

            xi_np_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(xi_np[:, 0:2]), np.ones((xi_np.shape[0],1)))))
            plt.plot(xi_np_XY[-1, 0], xi_np_XY[-1, 1], 'mo', markersize=7, transform=rot + base, label='Cur Position')
            # plt.plot(xi_np[-1, 0], xi_np[-1, 1], 'mo', markersize=7, label='Start Point')

            tmp_yi_np = np.vstack((np.zeros((1,2)), yi_np[:, 0:2])) + xi_np[-1, 0:2]
            yi_np_aug_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(tmp_yi_np), np.ones((tmp_yi_np.shape[0],1)))))
            plt.plot(yi_np_aug_XY[:, 0], yi_np_aug_XY[:, 1], 'r+', transform=rot + base, label='Ground Truth')
            # plt.plot(tmp_yi_np[:, 0], tmp_yi_np[1, :], 'r+')

            # plt.xlim((0, 1))
            # plt.ylim((0, 1))
            # plt.xlim((scale.min_x, scale.max_x))
            # plt.ylim((scale.min_y, scale.max_y))
            plt.xlim((0, 640))
            plt.ylim((-480, 0))

            fig = plt.gcf()
            fig.set_size_inches(16, 12)
            plt.legend()
            fig_name = '%03d_%02d.svg' % (ii, t1_ind_i)
            print('Saving figure: [%s] ...' % fig_name)
            plt.savefig(os.path.join(out_dir, fig_name))

            if interactive:
                plt.show()
            plt.clf()

            disp.add_orig_frame(0.5)
            disp.show('frame %d' % t0)


def train_loop(train_loader_, test_peds_, load=True, save=True):
    print("Train the model ...")
    if load:
        generator.load_state_dict(torch.load('./models/gan-g.pt'))
        discriminator.load_state_dict(torch.load('./models/gan-d.pt'))
    for epoch in range(1, n_epochs + 1):
        adjust_learning_rate(d_optimizer, epoch)
        adjust_learning_rate(g_optimizer, epoch)

        train(train_loader_)
        if epoch % test_interval == 0:
            if save:
                torch.save(generator.state_dict(), './models/gan-g.pt')
                torch.save(discriminator.state_dict(), './models/gan-d.pt')
            with torch.no_grad():
                evaluate(test_peds_)


def generate_samples(train_peds_, samples_mode):
    with torch.no_grad():
        # evaluate()
        print("Don't forget to correct here and test the test set!")
        K = 100
        build_teom_dataset(train_peds_, train_time_data, samples_mode.lower(), interactive=False)


def build_big_file(data_dir, output_file):
    file_list = glob.glob(os.path.join(data_dir, '*.npz'))

    obsv_list = []
    target_list = []
    teom_list = []
    #samples_x = []
    #samples_y = []

    for file in sorted(file_list):
        data_ii = np.load(file)
        gt_ii = data_ii['gt']
        obsv_ii = data_ii['obsv']
        teom_ii = data_ii['teom']
        pred_len = teom_ii.shape[0]
        grid_size = teom_ii.shape[1]
        n_ch = teom_ii.shape[3]

        yi_gt = []
        base_v = obsv_ii[-1, 2:4]  # last instant vel
        base_v = (obsv_ii[-1, 0:2] - obsv_ii[0, 0:2]) / (len(obsv_ii) - 1)  # observation avg vel
        # base_v = unit(base_v)  # it could be tested
        [_, base_theta] = cart2pol(base_v[0], base_v[1])
        for t in range(pred_len):
            dp_t = gt_ii[t, 0:2] - obsv_ii[-1, 0:2]
            [mag, abs_theta] = cart2pol(dp_t[0], dp_t[1])
            rel_theta = abs_theta - base_theta
            if rel_theta > np.pi:
                rel_theta -= 2 * np.pi
            elif rel_theta < -np.pi:
                rel_theta += 2 * np.pi
            yi_gt.append(np.array([mag, rel_theta]))

        yi_gt = np.stack(yi_gt)

        obsv_list.append(obsv_ii)
        teom_list.append(teom_ii)
        target_list.append(yi_gt)

    obsv_list = np.stack(obsv_list)
    teom_list = np.stack(teom_list)
    target_list = np.stack(target_list)

    np.savez(output_file, obsvs=obsv_list, teoms=teom_list, targets=target_list)


# ----------------- M A I N   C O D E S ----------------------
# train_loader, train_peds, test_peds = prepare_data(smooth=True)
# train_loop(train_loader, test_peds, load=True, save=True)
# generate_samples(test_peds, 'gt')
build_big_file('../teom/eth', '../teom/eth_big_file.npz')
