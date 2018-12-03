import glob
import os
import random
from math import atan2, cos, sin, sqrt, floor

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
from src.math_utils import cart2pol, eps
from src.models import ConstVelModel, NaivePredictor, Discriminator, Generator, SequentialPredictor, SequentialPredictorWithVelocity
from src.parse_utils import *
from src.kalman import MyKalman
from src.visualize import Display, FakeDisplay, to_image_frame
from src.vanilla_lstm import VanillaLSTM


data_dir = ''  # to be filled bellow
out_dir = ''  # to be filled bellow
np.random.seed(1)
torch.manual_seed(1)
config = MyConfig(n_past=8, n_next=8)
n_past = config.n_past
n_next = config.n_next
n_inp_features = 2  # (x, y),  /vx, vy
n_out_features = 2

# FIXME =======
n_epochs = 500
test_interval = 10

learning_rate = 3e-3
weight_decay = 4e-3
lambda_l2_loss = 100
lambda_dc_loss = 0
optim_betas = (0.9, 0.999)
def_batch_size = 128
noise_vec_len = 64

scale = []
Hinv = []


def bce_loss(input_, target_):
    neg_abs = -input_.abs()
    _loss = input_.clamp(min=0) - input_ * target_ + (1 + neg_abs.exp()).log()
    return _loss.mean()


# generator = Generator(n_past, n_next, n_inp_features, noise_vec_len)
# generator = NaivePredictor(n_inp_features, n_next)
generator = SequentialPredictorWithVelocity()

discriminator = Discriminator(n_past, n_next, n_inp_features)
discriminationLoss = nn.BCEWithLogitsLoss()  # Binary cross entropy
groundTruthLoss = nn.MSELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=optim_betas)
cv_model = ConstVelModel()
v_lstm = VanillaLSTM(4, 8, 128, num_layers=1)
try:    v_lstm.load_state_dict(torch.load('./models/v-lstm.pt'))
except: pass
parser = BIWIParser()
delimit = '\t'


def prepare_training_data(smooth_train_set=True, n_fold=5, test_fold=1):
    # parser = SeyfriedParser()
    # pos_data, vel_data, time_data = parser.load('../data/sey01.sey')
    pos_data, vel_data, time_data = parser.load(os.path.join(data_dir, 'obsmat.txt'))
    n_ped = len(pos_data)

    test_ids = range(int((test_fold-1)/n_fold*n_ped), int(test_fold/n_fold*n_ped))
    train_ids = [i for i in range(n_ped) if i not in test_ids]

    print('Dont forget to smooth the trajectories?')
    if smooth_train_set:
        print('Yes! Smoothing the trajectories in train_set ...')
        for i in train_ids:
            kf = MyKalman(1 / parser.actual_fps, n_iter=7)
            pos_data[i], vel_data[i] = kf.smooth(pos_data[i])

    all_peds = list()
    # Scaling
    for i in range(len(pos_data)):
        pos_data[i] = parser.scale.normalize(pos_data[i], shift=True)
        vel_data[i] = parser.scale.normalize(vel_data[i], shift=False)
        _pv_i = np.hstack((pos_data[i], vel_data[i]))
        all_peds.append(_pv_i)

    train_peds = np.concatenate((all_peds[:test_ids.start], all_peds[test_ids.stop:]))
    train_timestamps = np.concatenate((time_data[:test_ids.start], time_data[test_ids.stop:]))
    test_peds = np.array(all_peds[test_ids.start:test_ids.stop])
    test_timestamps = time_data[test_ids.start:test_ids.stop]

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
    return train_loader, train_peds, test_peds, train_timestamps, test_timestamps


def train_GAN(train_loader_):
    gen_total_loss_accum = 0
    gen_loss_gt = 0
    dcr_loss_real = 0
    dcr_loss_fake = 0

    for i, (batch_x, batch_y) in enumerate(train_loader_):
        xs = batch_x.cuda()
        ys = batch_y.cuda()

        batch_size = xs.size(0)
        noise_1 = torch.rand(batch_size, noise_vec_len).cuda()
        noise_2 = torch.rand(batch_size, noise_vec_len).cuda()

        # =============== Train Discriminator ================= #
        # discriminator.zero_grad()
        #
        # y_hat_1 = generator(xs, noise_1, n_next)  # for updating discriminator
        # y_hat_1.detach()
        # c_hat_fake_1 = discriminator(xs, y_hat_1)  # classify fake samples
        # # disc_loss_fakes = discriminationLoss(c_hat_fake_1, Variable(torch.zeros(batch_size, 1, 1).cuda()))
        # disc_loss_fakes = bce_loss(c_hat_fake_1, (torch.zeros(batch_size, 1, 1) + random.uniform(0, 0.3)).cuda())
        # disc_loss_fakes.backward()

        # for _ in range(1):
        #     c_hat_real = discriminator(xs, ys)  # classify real samples
        #     # disc_loss_reals = discriminationLoss(c_hat_real, torch.ones(batch_size, 1, 1).cuda())
        #     disc_loss_reals = bce_loss(c_hat_real, (torch.ones(batch_size, 1, 1) * random.uniform(0.7, 1.2)).cuda())
        #     disc_loss_reals.backward()

        # d_optimizer.step()

        # dcr_loss_fake += disc_loss_fakes.item()
        # dcr_loss_real += disc_loss_reals.item()

        # =============== Train Generator ================= #
        generator.zero_grad()
        discriminator.zero_grad()
        y_hat_1 = generator(xs, noise_1, n_next)  # for updating generator
        # c_hat_fake_1 = discriminator(xs, y_hat_1)  # classify fake samples

        # ********** Variety loss ************
        # y_hat_2 = generator(xs, noise_2)  # for updating generator
        # c_hat_fake_2 = discriminator(xs, y_hat_2)  # classify fake samples
        # gen_loss_variety = torch.log(y_hat_2 - y_hat_1)

        # gen_loss_fooling = discriminationLoss(c_hat_fake_2, Variable(torch.ones(batch_size, 1, 1).cuda()))
        # gen_loss_fooling = bce_loss(c_hat_fake_1, (torch.ones(batch_size, 1, 1) * random.uniform(0.7, 1.2)).cuda())

        loss_gt = lambda_l2_loss * groundTruthLoss(y_hat_1, ys) / n_next
        gen_loss_gt += loss_gt.item()
        # print('L2 loss = ', gen_loss_gt.item())
        gen_loss = loss_gt # + (gen_loss_fooling * generator.use_noise * lambda_dc_loss)

        gen_loss.backward()
        g_optimizer.step()

        gen_total_loss_accum += gen_loss.item()

    gen_total_loss_accum /= i
    gen_loss_gt /= i
    dcr_loss_fake /= i
    dcr_loss_real /= i

    # print('Gen L2 Error: %.6f || Dis Loss: Fake= %5f, Real= %.5f'
    #       % (gen_loss_gt, 0, 0))

    return gen_loss_gt


# ====================== T E S T =======================
def evaluate_GAN(peds):
    err_accum = 0
    std_accum = 0
    counter = 0

    for ii, _ in enumerate(peds):
        ped_i = torch.FloatTensor(peds[ii]).cuda()
        for t in range(n_past, ped_i.size(0) - n_next + 1, n_past):
            x = ped_i[t - n_past:t, 0:n_inp_features].view(1, n_past, -1)
            y = (ped_i[t:t + n_next, 0:2] - x[0, -1, 0:2]).view(n_next, 2)

            xK = x.repeat(K, 1, 1)
            yK = y.repeat(K, 1, 1)
            all_samples = generator(xK, torch.rand(K, noise_vec_len).cuda(), n_next)

            std_t = 0
            for tt in range(all_samples.size(1)):
                samples_tt = all_samples[:, tt, :]
                mean_pnt = torch.mean(samples_tt, dim=0)
                samples_tt = samples_tt - mean_pnt
                norms = torch.norm(samples_tt, p=2, dim=1)
                std_tt = torch.mean(norms)
                std_t += std_tt.item()

            loss = groundTruthLoss(all_samples, yK)
            err_accum += loss.item()
            std_accum += std_t / all_samples.size(1)
            counter += 1
        test_loss = err_accum / (counter + eps)
    test_std = std_accum / (counter + eps)
    print("Test loss= %4f | Test Std = %4f" %(np.math.sqrt(test_loss) / scale.sx, test_std / scale.sx))


def get_teom(yi_hat, Y_i_hats, ped_id=0, start_t=0, create_gif=False):
    # teom_mats = np.empty((0, GRID_SIZE, GRID_SIZE, Nch))
    GRID_SIZE = [129, 129]
    Nch = 3
    center = np.floor([GRID_SIZE[0] / 2, GRID_SIZE[1] / 2]).astype(int)
    PIX_VALUE = 100

    teom_mats = []
    n_frames = yi_hat.shape[0]
    n_others = Y_i_hats.shape[0]
    n_samples = K
    goal_vec = yi_hat[-1, :] - yi_hat[0, :]
    rot_angle = -atan2(goal_vec[1], goal_vec[0])
    rot_matrix = [[cos(rot_angle), sin(-rot_angle)], [sin(rot_angle), cos(rot_angle)]]

    if create_gif:
        gif_writer = imageio.get_writer(out_dir + "%03d_%02d_%dx%d.gif" % (ped_id, start_t, GRID_SIZE[0], GRID_SIZE[1]), mode='I')

    for t in range(n_frames):
        # Creating the Grids (Centered on the agent)
        cartesian_grid = np.zeros((GRID_SIZE[0], GRID_SIZE[1], Nch), dtype="uint8")
        cartesian_grid[center[0], center[1], 1] = 255

        rotated_goal = np.matmul(rot_matrix, goal_vec)
        if abs(rotated_goal[0]) >= 1:
            rotated_goal = rotated_goal / rotated_goal[0]
        cartesian_grid[int(round((rotated_goal[0] + 1) * center[0])),
                       int(round((rotated_goal[1] + 1) * center[1])), 2] = 255

        for j in range(n_others):
            for k in range(n_samples):
                [rot_x, rot_y] = np.matmul(rot_matrix, Y_i_hats[j, k, t, :] - yi_hat[0, :])  # Rotate
                r = sqrt(rot_x ** 2 + rot_y ** 2)
                th = atan2(rot_y, rot_x)
                # polar_loc = np.array([r, th])

                if r >= 1: continue

                [rr, cc] = np.array([floor((rot_x + 1) * center[0]), floor((rot_y + 1) * center[1])]).astype(int)
                cartesian_grid[rr:rr+2, cc:cc+2, 0] = \
                    np.minimum(np.ones((2, 2)) * 255, cartesian_grid[rr:rr+2, cc:cc+2, 0] + PIX_VALUE//K)


                # polar_coord_0 = int(round(r * (GRID_SIZE[0] - 1) / sqrt(2)))
                # polar_coord_1 = int(round(th * (GRID_SIZE[1] - 1) / (2 * np.pi) + cntr[1]))
                # aligned_polar_grid[polar_coord_0, polar_coord_1, 0] += approach_rate[j] * PIX_VALUE
                # aligned_polar_grid[polar_coord_0, polar_coord_1, 1] += DCAs[j] * PIX_VALUE

        teom_mats.append(cartesian_grid[:, :, 0])
        if create_gif:
            cartesian_grid = ndimage.rotate(cartesian_grid, 90)
            # cartesian_grid = imresize(cartesian_grid, size=2000, interp='nearest')
            gif_writer.append_data(cartesian_grid)
    teom_mats = np.stack(teom_mats)
    return teom_mats


K = 50  # Number of samples
disp = FakeDisplay(data_dir)

actual_len = 8
def build_teom_dataset(peds, time_data, samples_mode, interactive=False, create_maps=False, create_gifs=False):
    K = 100

    for ii in range(137, len(peds), 1):
        ped_i_tensor = torch.FloatTensor(peds[ii]).cuda()
        for t1_ind_i in range(n_past, ped_i_tensor.size(0) - n_next + 1, 2):
            out_filename = os.path.join(out_dir, '%03d_%02d.npz' % (ii, t1_ind_i))
            if os.path.isfile(out_filename):
                continue

            Y_hat = np.empty((0, K, actual_len, 2))

            # ts = time_data[ii][t1_ind_i - n_past]
            t0 = time_data[ii][t1_ind_i - 1]
            t1 = time_data[ii][t1_ind_i]
            # te = time_data[ii][t1_ind_i + n_next - 1]

            # disp.grab_frame(t0)
            base = plt.gca().transData
            rot = transforms.Affine2D().rotate_deg(-90)

            # find neighbors
            for jj in range(len(peds)):
                # ts_ind_j = np.array(np.where(time_data[jj] == ts))
                t1_ind_j = np.array(np.where(time_data[jj] == t1))
                # te_ind_j = np.array(np.where(time_data[jj] == te))

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
                        vv = (xj_np[-1, 0:2] - xj_np[0, 0:2]) / (xj_np.shape[0] - 1)
                        for tj in range(len(yj_hat_np), actual_len):
                            aaa = yj_hat_np[-1, :] + vv
                            yj_hat_np = np.vstack((yj_hat_np, aaa.reshape((1, 2))))

                        for tj in range(len(yj_hat_np)):
                            yj_hat_np[tj, :] += torch.randn(2).data.numpy() * 0.002 * (1+tj/2) + xj_np[-1, 0:2]

                    elif samples_mode == 'v-lstm':
                        yj_hat = v_lstm(xj).view(1, n_next, 2)
                        yj_hat_np = yj_hat.cpu().data.numpy().reshape((n_next, 2)) + xj_np[-1, 0:2]
                        for tj in range(len(yj_hat_np)):
                            yj_hat_np[tj, :] += torch.randn(2).data.numpy() * 0.002 * (1+tj/2) + xj_np[-1, 0:2]

                    else:  # samples_mode == 'gan':
                        yj_hat = generator(xj, torch.rand(1, noise_vec_len), actual_len).view(1, -1, 2)
                        yj_hat_np = yj_hat.cpu().data.numpy().reshape((-1, 2)) + xj_np[-1, 0:2]

                    # ==========================================
                    y_hat_j.append(yj_hat_np)

                    # ========== PLOT ==========
                    yj_hat_np = np.vstack((xj_np[-1, 0:2], yj_hat_np))
                    yj_hat_np_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(yj_hat_np), np.ones((yj_hat_np.shape[0],1)))))
                    sample_color = ([__/12 for __ in range(actual_len+1)])

                    # sample_lbl = plt.scatter(yj_hat_np_XY[:, 0], yj_hat_np_XY[:, 1], c=sample_color, alpha=0.3, transform=rot + base)

                    # disp.plot_path(scale.denormalize(yj_hat_np[:, 0:2]), jj)

                # .............................................
                if create_maps:
                    xj_np_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(xj_np[:, 0:2]), np.ones((xj_np.shape[0], 1)))))
                    plt.plot(xj_np_XY[:, 0], xj_np_XY[:, 1], 'g', transform=rot + base)

                    yj_np_aug = np.vstack((np.zeros((1,2)), yj_np[:, 0:2])) + xj_np[-1, 0:2]
                    yj_np_aug_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(yj_np_aug), np.ones((yj_np_aug.shape[0],1)))))
                    plt.plot(yj_np_aug_XY[:, 0], yj_np_aug_XY[:, 1], 'g--', markersize=10, transform=rot + base)
                # .............................................

                y_hat_j = np.stack(y_hat_j).reshape((1, K, -1, 2))
                Y_hat = np.vstack((Y_hat, y_hat_j))

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # disp.plot_ped(scale.denormalize(xj_np[-1, 0:2]))
                # disp.plot_path(scale.denormalize(xj_np[:, 0:2]), ii, 'g.')
                # disp.plot_path(scale.denormalize(yj_np[:, 0:2] + xj_np[-1, 0:2]), ii, 'g--')
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            xi = ped_i_tensor[t1_ind_i - n_past:t1_ind_i, 0:n_inp_features].view(1, n_past, -1)
            xi_np = xi.cpu().data.numpy().reshape((n_past, n_inp_features))
            yi = (ped_i_tensor[t1_ind_i:t1_ind_i + n_next, 0:2] - xi[0, -1, 0:2]).view(1, n_next, 2)
            yi_np = yi.cpu().data.numpy().reshape((n_next, 2))
            # ci_real = discriminator(xi, yi)
            # yi_hat = generator(xi, torch.rand(1, noise_vec_len)).view(1, n_next, 2)
            # yi_hat = v_lstm(xi)
            # yi_hat = yi_hat.cpu().data.numpy().reshape((n_next, 2)) + xi_np[-1, 0:2]
            yi_hat = cv_model.predict(xi_np, actual_len)
            yi_hat_np = np.vstack((xi_np[-1, 0:2], yi_hat))

            # ====================== Build TEOM Maps =========================
            teom_mats = get_teom(yi_hat, Y_hat, ii, t1_ind_i, create_gifs)
            # ================================================================

            # preparing target vectors for training:
            #              delta-velocity w.r.t last velocity of agent
            # =================================================================================
            target = []
            base_v = xi_np[-1, 2:4]  # last instant vel
            base_v = (xi_np[-1, 0:2] - xi_np[0, 0:2]) / (len(xi_np) - 1)  # observation avg vel
            # base_v = unit(base_v)  # it could be tested
            [_, base_theta] = cart2pol(base_v[0], base_v[1])
            for tt in range(len(yi_np)):
                dp_t = yi_np[tt, 0:2]  # - xi_np[-1, 0:2]
                [mag, abs_theta] = cart2pol(dp_t[0], dp_t[1])
                rel_theta = abs_theta - base_theta
                if rel_theta > np.pi:
                    rel_theta -= 2 * np.pi
                elif rel_theta < -np.pi:
                    rel_theta += 2 * np.pi
                target.append(np.array([mag, rel_theta]))

            np.savez(out_filename, teom=teom_mats, target=target,
                     obsv=ped_i_tensor[t1_ind_i - n_past:t1_ind_i, 0:4].view(1, n_past, -1),
                     gt=(ped_i_tensor[t1_ind_i:t1_ind_i + actual_len, 0:2]).cpu().data.numpy().reshape((-1, 2)))
            print('Saving data to %s' % out_filename)
            if not create_maps:
                plt.clf()
                continue

            # =========== Const-Vel Prediction ========
            yi_cv = np.vstack((xi_np[-1, 0:2], cv_model.predict(xi_np[:, 0:2])))

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # disp.plot_path(scale.denormalize(xi_np[:, 0:2]), ii, 'g.')
            # disp.plot_path(scale.denormalize(yi_np[:, 0:2] + xi_np[-1, 0:2]), ii, 'g--')
            # disp.plot_ped(scale.denormalize(xi_np[-1, 0:2]), ii, color=(0, 100, 200))

            # ========================================
            yi_cv_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(yi_cv), np.ones((yi_cv.shape[0],1)))))
            # cv_lbl = plt.plot(yi_cv_XY[:, 0], yi_cv_XY[:, 1], 'c--', transform=rot + base, label='Const-Vel Prediction')
            # plt.plot(yi_cv[:, 0], yi_cv[:, 1], 'c--')

            # yi_hat_np_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(yi_hat_np), np.ones((yi_hat_np.shape[0], 1)))))
            # plt.plot(yi_hat_np_XY[:, 0], yi_hat_np_XY[:, 1], 'y--', transform=rot + base, label='Const-Vel Prediction')
            # plt.plot(yi_hat_np[:, 0], yi_hat_np[:, 1], 'y--')

            xi_np_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(xi_np[:, 0:2]), np.ones((xi_np.shape[0],1)))))
            obsv_lbl = plt.plot(xi_np_XY[:, 0], xi_np_XY[:, 1], 'g', transform=rot + base, label='Observation')

            xi_np_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(xi_np[:, 0:2]), np.ones((xi_np.shape[0],1)))))
            curpos_lbl = plt.plot(xi_np_XY[-1, 0], xi_np_XY[-1, 1], 'mo', markersize=12, transform=rot + base, label='Cur Position')
            # plt.plot(xi_np[-1, 0], xi_np[-1, 1], 'mo', markersize=7, label='Start Point')

            tmp_yi_np = np.vstack((np.zeros((1,2)), yi_np[:, 0:2])) + xi_np[-1, 0:2]
            kff = MyKalman(1 / parser.actual_fps, n_iter=1)
            tmp_yi_np, ____ = kff.smooth(tmp_yi_np)

            yi_np_aug_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(tmp_yi_np), np.ones((tmp_yi_np.shape[0],1)))))
            gt_lbl = plt.plot(yi_np_aug_XY[:, 0], yi_np_aug_XY[:, 1], 'r+', markersize=10, transform=rot + base, label='Ground Truth')
            # plt.plot(tmp_yi_np[:, 0], tmp_yi_np[1, :], 'r+')

            # plt.xlim((0, 1))
            # plt.ylim((0, 1))
            # plt.xlim((scale.min_x, scale.max_x))
            # plt.ylim((scale.min_y, scale.max_y))
            # plt.xlim((0, 640))
            # plt.ylim((-480, 0))

            fig = plt.gcf()
            fig.set_size_inches(16, 11.9)
            # plt.legend((obsv_lbl, cv_lbl, gt_lbl, curpos_lbl, sample_lbl),
            #            ('Observation', 'Const-Vel Prediction' 'Ground Truth', 'Current Location', 'Trajectory Sample'))
            fig_name = '%03d_%02d.svg' % (ii, t1_ind_i)
            print('Saving figure: [%s] ...' % fig_name)
            plt.savefig(os.path.join(out_dir, fig_name))

            if interactive:
                plt.show()
            plt.clf()

            disp.add_orig_frame(0.5)
            disp.show('frame %d' % t0)


def train_(train_loader_, test_peds_, n_epoch, load=True, save=True):
    print("Train the model ...")
    min_train_loss = 400
    if load:
        try:
            generator.load_state_dict(torch.load('./models/G0.pt'))
            discriminator.load_state_dict(torch.load('./models/D0.pt'))
        except:
            print('Could not find model files to load...')
    for epoch in range(1, n_epochs + 1):
        adjust_learning_rate(d_optimizer, epoch)
        adjust_learning_rate(g_optimizer, epoch)

        train_loss = train_GAN(train_loader_)

        print('Training Epoch [%3d/%d] | Loss = %5f' % (epoch, n_epoch, train_loss))
        if save and train_loss < min_train_loss:
            min_train_loss = train_loss
            torch.save(generator.state_dict(), './models/G0.pt')
            torch.save(discriminator.state_dict(), './models/D0.pt')
        if epoch % test_interval == 0:
            with torch.no_grad():
                evaluate_GAN(test_peds_)


def build_big_file(data_dir):
    big_file_dir = os.path.join(data_dir, 'big_file/')
    try:
        os.makedirs(big_file_dir)
    except:
        print('Directory exist')
    output_file = os.path.join(big_file_dir, 'big_file.npz')

    file_list = glob.glob(os.path.join(data_dir, '*.npz'))

    obsv_list = []
    target_list = []
    teom_list = []
    gt_list = []
    #samples_x = []
    #samples_y = []

    for file in sorted(file_list):
        data_ii = np.load(file)
        gt_ii = data_ii['gt']
        obsv_ii = data_ii['obsv'].reshape(-1, 4)
        teom_ii = data_ii['teom']
        pred_len = teom_ii.shape[0]
        grid_size = teom_ii.shape[1]
        # n_ch = teom_ii.shape[3]

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
        gt_list.append(gt_ii)

    obsv_list = np.stack(obsv_list)
    teom_list = np.stack(teom_list)
    target_list = np.stack(target_list)
    gt_list = np.stack(gt_list)

    np.savez(output_file, obsvs=obsv_list, teoms=teom_list, targets=target_list, gts=gt_list)


# ======================================================
# ----------------- M A I N   C O D E S ----------------
test_fold = 5
# exp_names = ['eth', 'hotel', 'zara01', 'zara02', 'univ']
exp_names = ['eth', 'hotel']

for exp_name in exp_names:
    data_dir = '../data/' + exp_name
    print('************ %s *************' %exp_name)

    # ================ READ DATA =====================
    # mapfile = os.path.join(data_dir, "map.png")
    Homography_file = os.path.join(data_dir, "H.txt")
    Hinv = np.linalg.inv(np.loadtxt(Homography_file))
    train_loader_, train_peds, test_peds, train_time_data, test_time_data = \
        prepare_training_data(smooth_train_set=False, n_fold=5, test_fold=test_fold)
    scale = parser.scale


    # ===============================================
    out_dir = '../teom_new2/%s_8/train-%d/' % (exp_name, test_fold)
    try:     os.makedirs(out_dir)
    except:  print('Directory exist')
    # with torch.no_grad():
    #     build_teom_dataset(train_peds, train_time_data, samples_mode='gt', interactive=False, create_maps=False, create_gifs=False)
    #
    # try:     build_big_file(out_dir)
    # except: pass

    # =================== TRAIN ======================
    train_(train_loader_, test_peds, n_epoch=2000, load=False, save=True)

    # # ================ BUILD TEOMs ===================
    # generator.load_state_dict(torch.load('./models/G0.pt'))
    # discriminator.load_state_dict(torch.load('./models/D0.pt'))
    #
    # out_dir = '../teom_new2/%s_8/train-gt-%d/' % (exp_name, test_fold)
    # try:     os.makedirs(out_dir)
    # except:  print('Directory exist')
    # with torch.no_grad():
    #     build_teom_dataset(train_peds, train_time_data, samples_mode='gt', interactive=False, create_maps=True, create_gifs=True)
    #
    # try:     build_big_file(out_dir)
    # except: pass
    #
