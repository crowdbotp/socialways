import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from torch.utils.data import DataLoader

from src.utils.math_utils import norm
from src.utils.parse_utils import *
from src.utils.learning_utils import *
from src.utils.kalman import MyKalman
from src.models import ConstVelModel, VanillaLSTM, SequentialPredictorWithVelocity
from src.visualize import Display, FakeDisplay, to_image_frame

np.random.seed(1)
config = MyConfig(n_past=8, n_next=8)
n_past = config.n_past
n_next = config.n_next
test_interval = 50
n_inp_features = 4  # x, y, vx, vy
n_out_features = 2

train_rate = 0.8
learning_rate = 1e-3
weight_decay = 4e-3
def_batch_size = 128
n_epochs = 2000

if torch.cuda.is_available():
    print("CUDA is available!")


def update(model, y_list, y_hat_list):
    y_stack = torch.stack(y_list, 2)
    y_hat_stack = torch.stack(y_hat_list, 2)
    loss = model.loss_func(y_hat_stack, y_stack)
    model.zero_grad()
    loss.backward()
    model.optimizer.step()
    return loss.item()


# def train(model, ped_data, batch_size=def_batch_size):
def train(model, train_loader):
    running_loss = 0
    running_cntr = 0
    for i, (datas_x, datas_y) in enumerate(train_loader):
        xs = datas_x.cuda()
        ys = datas_y.cuda()
        batch_size = xs.size(0)

        ys_hat = model(xs)
        loss = model.loss_func(ys_hat, ys)
        model.zero_grad()
        loss.backward()
        model.optimizer.step()

        running_cntr += ys.shape[0]
        running_loss += loss.item()

    # y_list = []
    # y_hat_list = []
    # for ped_i in ped_data:
    #     ped_i_tensor = torch.FloatTensor(ped_i).cuda()
    #     seq_len = ped_i_tensor.size(0)
    #     for t in range(n_past, seq_len - n_next + 1):
    #         model.hidden = model.init_state()
    #
    #         x = ped_i_tensor[t-n_past:t, 0:n_inp_features]
    #
    #         # sequence-prediction
    #         # y = (ped_i_tensor[t:t + n_next, 0:2] - x[-1, 0:2]).view(n_next, 2)
    #         # y_hat = model(x).view(n_next, 2)
    #
    #         # goal-prediction
    #         y = (ped_i_tensor[t + n_next-1, 0:2] - x[-1, 0:2]).view(1, 2)
    #         y_hat = model(x).view(1, 2)
    #
    #
    #         y_list.append(y)
    #         y_hat_list.append(y_hat)
    #         if len(y_list) >= batch_size:
    #             running_loss += update(model, y_list, y_hat_list) * batch_size
    #             running_cntr += batch_size
    #             y_list = []
    #             y_hat_list = []
    #
    # if len(y_list) > 0:
    #     running_loss += update(model, y_list, y_hat_list) * len(y_list)
    #     running_cntr += len(y_list)

    return running_loss / running_cntr


cv_model = ConstVelModel()
def test_cv(ped_data):
    ade_err_accum = 0
    fde_err_accum = 0
    err_counter = 0
    for ii in range(len(ped_data)):
        ped_i_tensor = torch.FloatTensor(ped_data[ii])
        for t in range(n_past, ped_i_tensor.size(0) - n_next + 1, n_past):
            x = ped_i_tensor[t-n_past:t, 0:n_inp_features]
            x_np = x.cpu().data.numpy().reshape((n_past, n_inp_features))[:, 0:2]

            y = (ped_i_tensor[t:t+n_next, 0:2]).view(n_next, 2)
            y_np = y.cpu().data.numpy().reshape((n_next, 2))
            y_hat = cv_model.predict(x_np, n_next)

            loss = norm(y_np - y_hat, axis=1)
            ade_err_accum += np.mean(loss)
            fde_err_accum += loss[-1]

            err_counter += 1
    return ade_err_accum/err_counter, fde_err_accum/err_counter


def test(model, ped_data):
    ade_err_accum = 0
    fde_err_accum = 0
    err_counter = 0
    cv_model = ConstVelModel()
    with torch.no_grad():
        for ii in range(len(ped_data)):
            ped_i_tensor = torch.FloatTensor(ped_data[ii]).cuda()
            for t in range(n_past, ped_i_tensor.size(0) - n_next + 1, n_past):
                x = ped_i_tensor[t-n_past:t, 0:n_inp_features]

                y = (ped_i_tensor[t:t+n_next, 0:2] - x[-1, 0:2]).view(n_next, 2)
                y_hat = model(x.view(1, -1, n_inp_features)).view(n_next, 2)

                loss = model.loss_func(y_hat, y)
                ade_err_accum += np.sqrt(loss.item())
                fde_err_accum += torch.dist(y[-1, 0:2], y_hat[-1, 0:2])

                err_counter += 1

        # Display Results
        for ii in range(5, len(ped_data), 15):
            ped_i_tensor = torch.FloatTensor(ped_data[ii]).cuda()
            for t in range(n_past, ped_i_tensor.size(0) - n_next + 1, n_past):
                x = ped_i_tensor[t - n_past:t, 0:n_inp_features]
                x_np = x.cpu().data.numpy().reshape((n_past, n_inp_features))[:, 0:2]

                y = (ped_i_tensor[t:t + n_next, 0:2] - x[-1, 0:2]).view(n_next, 2)
                y_hat = model(x.view(1, -1, n_inp_features)).view(n_next, 2)
                y_np = y.cpu().data.numpy().reshape((n_next, 2))
                y_hat_np = np.vstack((np.array([0, 0]), y_hat.cpu().data.numpy().reshape((n_next, 2)))) + x_np[-1, 0:2]
                y_cv = np.vstack((x_np[-1, 0:2], cv_model.predict(x_np)))

                plt.plot(x_np[:, 0], x_np[:, 1], 'y--')
                plt.plot(y_cv[:, 0], y_cv[:, 1], 'r--')
                plt.plot(y_np[:, 0] + x_np[-1, 0], y_np[:, 1] + x_np[-1, 1], 'g--')
                plt.plot(y_hat_np[:, 0], y_hat_np[:, 1], 'b')
                plt.plot(x_np[-1, 0], x_np[-1, 1], 'mo', markersize=7, label='Start Point')

            plt.ylim((0, 1))
            plt.xlim((0, 1))
            # plt.show()

    return ade_err_accum/err_counter, fde_err_accum/err_counter

dt = 10
def generate_nice_samples():
    disp = FakeDisplay('../data/zara01')
    Homography_file = os.path.join('../data/zara01', "H.txt")
    Hinv = np.linalg.inv(np.loadtxt(Homography_file))
    t0 = time_data[0][0]
    t0 = 961-dt
    for t in range(t0, 10000, dt):
        disp.grab_frame(t)

        iis = []
        for i in range(len(pos_data)):
            # for i in [59, 58]:
            t1_ind = np.array(np.where(time_data[i] == t))
            ts_ind = np.array(np.where(time_data[i] == t-n_past*dt))
            te_ind = np.array(np.where(time_data[i] == t+(n_next-1)*dt))

            if t1_ind.size == 0 or ts_ind.size == 0 or te_ind.size == 0:
                continue

            t1_ind = t1_ind[0][0]
            ts_ind = ts_ind[0][0]
            te_ind = te_ind[0][0]

            xi = pos_data[i][ts_ind:t1_ind]
            xi_smooth = pos_data[i][ts_ind:t1_ind]
            yi = pos_data[i][t1_ind:te_ind+1]

            # teom_i = torch.FloatTensor(test_teom[0]).cuda().unsqueeze(0)
            # gt_i = test_gt[0]
            # obsv_i = test_obsv[i]
            # target_i = test_target[i]
            xi_tf = torch.FloatTensor(xi_smooth).cuda().unsqueeze(0)
            # pred, risk = cnn_model(xi_tf, teom_i, n_next, False)
            pred = model(xi_tf) + xi_tf[0, -1, :]
            pred = pred.cpu().data.numpy().reshape((n_next, 2))
            kf = MyKalman(1 / parser.actual_fps, n_iter=5)
            pred, _ = kf.smooth(pred)
            pred_cv = cv_model.predict(xi, n_next)

            # Xi = to_image_frame(Hinv, np.hstack((scale.denormalize(xi), np.ones((n_next, 1)))))
            # Xi_lbl, = plt.plot(Xi[:, 0], Xi[:, 1], 'g+')
            #
            # Yi = to_image_frame(Hinv, np.hstack((scale.denormalize(yi), np.ones((n_next, 1)))))
            # Yi_lbl, = plt.plot(Yi[:, 0], Yi[:, 1], 'r+')

            xi_lbl, = plt.plot(xi[:, 0], xi[:, 1], 'g')

            # yi_lbl, = plt.plot(yi[:, 0], yi[:, 1], 'g--')
            # yhat_lbl, = plt.plot(pred[:, 0], pred[:, 1], 'b--')
            # y_cv_lbl, = plt.plot(pred_cv[:, 0], pred_cv[:, 1], 'y--')
            plt.text(xi[-1, 0], xi[-1, 1], '%d' %i)
            iis.append(i)

            disp.plot_ped(scale.denormalize(xi[-1, 0:2]))
            disp.plot_path(scale.denormalize(xi[:, 0:2]), i, 'g--')
            disp.plot_path(scale.denormalize(yi[:, 0:2]), i, 'g.')
            disp.plot_ped()
            if i ==20:
                yi_lbl, = plt.plot(yi[:, 0], yi[:, 1], 'g--')
                yhat_lbl, = plt.plot(pred[:, 0], pred[:, 1], 'b--')
                y_cv_lbl, = plt.plot(pred_cv[:, 0], pred_cv[:, 1], 'y--')
                cur_pos_lbl = plt.plot(xi[-1, 0], xi[-1, 1], 'ro')

                # disp.plot_path(scale.denormalize(pred[:, 0:2]/20), i, 'b.')
                # disp.plot_path(scale.denormalize(pred_cv[:, 0:2]/20), i, 'y.')
            elif i in [15, 16, 18]:
                for k in range(100):
                    sample_k = generator(xi_tf, 0, n_next) + xi_tf[0, -1, :2]
                    sample_k = sample_k.cpu().data.numpy().reshape((-1, 2))
                    kf = MyKalman(1 / parser.actual_fps, n_iter=5)
                    sample_k, _ = kf.smooth(sample_k)
                    sam_lbl, = plt.plot(sample_k[:, 0], sample_k[:, 1], 'r')
                    disp.plot_path(scale.denormalize(sample_k), i, 'r--')
        # disp.show('frame')


        if len(iis) > 2:
            plt.legend((xi_lbl, yi_lbl, yhat_lbl, y_cv_lbl), ('obsv', 'gt', 'pred', 'cv'))
            plt.gcf().set_size_inches(16, 16)
            # plt.xlim([0, 1])
            # plt.ylim([0., 1.])
            print('t = %d, ids = ' %t, iis)
            plt.show()
            # return 0
        plt.clf()


if __name__ == '__main__':

    # parser = SeyfriedParser()
    # pos_data, vel_data, time_data = parser.load('../data/sey01.sey')
    parser = BIWIParser()
    pos_data, vel_data, time_data = parser.load('../data/zara01/obsmat.txt')
    scale = parser.scale

    n_ped = len(pos_data)
    train_size = int(n_ped * train_rate)
    test_size = n_ped - train_size

    print('Dont forget to smooth the trajectories?')

    # print('Yes! Smoothing the trajectories in train_set ...')
    # for i in range(train_size):
    #     kf = MyKalman(1 / parser.actual_fps, n_iter=5)
    #     pos_data[i], vel_data[i] = kf.smooth(pos_data[i])

    # Scaling
    data_set = list()
    for i in range(len(pos_data)):
        pos_data[i] = scale.normalize(pos_data[i], shift=True)
        vel_data[i] = scale.normalize(vel_data[i], shift=False)
        _pv_i = np.hstack((pos_data[i], vel_data[i]))
        data_set.append(_pv_i)
        time_data[i] = time_data[i].astype(int)
    train_peds = np.array(data_set[:train_size])
    test_peds = np.array(data_set[train_size:])

    model = VanillaLSTM(feature_size=2, pred_length=n_next, hidden_size_lstm=128, num_layers=1)
    generator = SequentialPredictorWithVelocity()
    generator.load_state_dict(torch.load('./models/G0.pt'))

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
    train_loader = DataLoader(train_data, batch_size=def_batch_size, shuffle=True, num_workers=4)
    min_test_err = 25
    train_loss = 0

    print("Train the model ...")
    model.load_state_dict(torch.load('./models/v-lstm.pt'))
    for epoch in range(1, n_epochs+1):
        lr = 0.001
        if epoch > 1000:
            lr = lr / 100000
        elif epoch > 800:
            lr = lr / 32
        elif epoch > 640:
            lr = lr / 50
        elif epoch > 320:
            lr = lr / 30
        elif epoch > 160:
            lr = lr / 15
        elif epoch > 40:
            lr = lr / 8
        # lr *= (0.6 ** (epoch // 50))

        for param_group in model.optimizer.param_groups:
            param_group["lr"] = lr

        # train_loss = train(model, train_set)
        # train_loss = train(model, train_loader)
        train_loss = math.sqrt(train_loss) / (scale.sx)
        train_loss = math.sqrt(train_loss) / (scale.sx)

        if train_loss < min_test_err:
            min_test_err = train_loss
            torch.save(model.state_dict(), './models/v-lstm.pt')

        print('******* Epoch: [%3d/%d], Loss: %.9f **********' % (epoch, n_epochs, train_loss))
        if epoch % test_interval == 0:
            test_ade, test_fde = test(model, test_peds)


            generate_nice_samples()

            # n_next = 8
            # test_ade, test_fde = test_cv(test_peds)
            # test_ade = test_ade / (scale.sx)
            # test_fde = test_fde / (scale.sx)
            # print('TEST==[%d/%d] ADE=%.6f  | FDE=%.6f' % (n_past, n_next, test_ade, test_fde))
            #
            # n_next = 12
            # test_ade, test_fde = test_cv(test_peds)
            # test_ade = test_ade / (scale.sx)
            # test_fde = test_fde / (scale.sx)
            # print('TEST==[%d/%d] ADE=%.6f  | FDE=%.6f' % (n_past, n_next, test_ade, test_fde))
            # exit(1)



