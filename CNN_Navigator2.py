import glob
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import transforms
from torch.utils.data import DataLoader, TensorDataset

from src.kalman import MyKalman
from src.learning_utils import MyConfig
from src.models import NavigationNet, ConstVelModel, VanillaLSTM
from src.parse_utils import Scale, BIWIParser
from src.math_utils import unit, cart2pol, pol2cart, norm, eps
from src.visualize import to_image_frame, Display

config = MyConfig(8,12)
n_past = config.n_past
n_next = config.n_next


train_ratio = .0
# ================== Load Data ===============
# file_name = '../teom_new/ethb_gan_12/train-5/big_file/big_file.npz'
# file_name = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/teom_new/ethc_gt_12/test-5/big_file/big_file.npz'

# =========== ETH Dataset ============
file_name = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/teom_new2/eth_12/train-5/big_file/big_file.npz'
file_name = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/teom_new2/eth_12/test-5/big_file/big_file.npz'

# =========== ZARA01  ================
# file_name = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/teom_new2/zara02_12/train-5/big_file/big_file.npz'
# file_name = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/teom_new2/zara02_12/test-5/big_file/big_file.npz'

# file_name = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/teom_new2/zara02_12/train-5/big_file/big_file.npz'
# file_name = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/teom_new2/zara01_12/test-5/big_file/big_file.npz'

data_dir = '../data/eth'
parser = BIWIParser()
pos_data, vel_data, time_data = parser.load(os.path.join(data_dir, 'obsmat.txt'))
scale = parser.scale

pos_data_smooth = []
for i in range(len(pos_data)):
    pos_data_smooth.append(np.copy(pos_data[i]))
    kf = MyKalman(1 / parser.actual_fps, n_iter=7)
    pos_data_smooth[i], _ = kf.smooth(pos_data_smooth[i])
    pos_data_smooth[i] = scale.normalize(pos_data_smooth[i], inPlace=False)
    pos_data[i] = scale.normalize(pos_data[i],inPlace=False)
print('Training on %s, Observe/Predict [%d]/[%d] frames' % (file_name, n_past, n_next))
# ............................................

# ============================================
data = np.load(file_name)
obsvs = data['obsvs']
teoms = data['teoms']
targets = data['targets']
gts = data['gts']
row_size = teoms.shape[2]
col_size = teoms.shape[3]
# teoms = teoms[:, :, row_size//4:row_size*3//4 + 1, col_size//4:col_size*3//4 + 1, 1]
print('Obsv size:%s |  TEOMs size:%s |  Targets size:%s | GT size:%s'
      % (obsvs.shape, teoms.shape, targets.shape, gts.shape))
# ............................................

# ================== SCALE ===================
teoms = (teoms/255)
y_flat = np.vstack(targets)
polar_scale = Scale()
polar_scale.min_x = min(y_flat[:, 0])
polar_scale.max_x = max(y_flat[:, 0])
polar_scale.min_y = min(y_flat[:, 1])
polar_scale.max_y = max(y_flat[:, 1])
polar_scale.calc_scale(keep_ratio=False)
polar_scale.normalize(y_flat, shift=False, inPlace=True)
targets = y_flat.reshape((obsvs.shape[0], -1, 2))
# ............................................

# ========= Divide into train & test =========
train_size = int(train_ratio * obsvs.shape[0])
train_obsv = obsvs[:train_size]
train_teom = teoms[:train_size]
train_target = targets[:train_size]
train_gt = gts[:train_size]

test_obsv = obsvs[train_size:]
test_teom = teoms[train_size:]
test_target = targets[train_size:]
test_gt = gts[train_size:]
# ============================================


LR = 0.001  # learning rate
N_EPOCH = 2000
BATCH_SIZE = 32
train = TensorDataset(torch.FloatTensor(train_obsv), torch.FloatTensor(train_teom),
                      torch.FloatTensor(train_target), torch.FloatTensor(train_gt))
train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=False)

test = TensorDataset(torch.FloatTensor(test_obsv), torch.FloatTensor(test_teom),
                     torch.FloatTensor(test_target), torch.FloatTensor(test_gt))
test_loader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=False)

cnn_model = NavigationNet(12)
vlstm_model = VanillaLSTM(2, n_next, 64)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LR)   # optimize all parameters
loss_func = nn.MSELoss()                       # the target label is not one-hotted
cv_model = ConstVelModel()

# ===================================================
Hfile = os.path.join(data_dir, "H.txt")
Hinv = np.linalg.inv(np.loadtxt(Hfile))

def visualize_one(xi, teom, target, gt_):
    obsv = torch.FloatTensor(xi).cuda().unsqueeze(0)
    teom = torch.FloatTensor(teom).cuda().unsqueeze(0)
    # target = target

    output, risk = cnn_model(obsv, teom, n_next, False)
    output = output.cpu().data.numpy().reshape((n_next, 2))
    risk = risk.cpu().data.numpy().reshape((n_next, 1))

    # output = polar_scale.denormalize(output, shift=False, inPlace=True)
    # target = polar_scale.denormalize(target, shift=False, inPlace=False)
    #
    # base_v = xi[-1, 2:4]  # last instant vel
    # base_v = (xi[-1, 0:2] - xi[0, 0:2]) / (len(xi) - 1)  # observation avg vel
    # # base_v = unit(base_v)  # it could be tested
    # [_, base_theta] = cart2pol(base_v[0], base_v[1])
    # output[:, 1] = output[:, 1] + base_theta
    # target[:, 1] = target[:, 1] + base_theta
    # y_hat = pol2cart(output[:, 0], output[:, 1]) + xi[-1, 0:2]
    # y_gt = pol2cart(target[:, 0], target[:, 1]) + xi[-1, 0:2]

    preds = []
    gt = []
    for ts in range(n_next):
        preds.append(output[ts])
        gt.append(gt_[ts])

    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(-90)

    preds = np.stack(preds)
    gt = np.stack(gt)

    preds_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(preds), np.ones((preds.shape[0], 1)))))
    plot_pred, = plt.plot(preds_XY[:, 0], preds_XY[:, 1], 'r--', transform=rot + base,)
    plot_risk = plt.scatter(preds_XY[:, 0], preds_XY[:, 1], s=2+risk*1000, alpha=0.5, transform=rot + base )
    # plt.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)

    xi_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(xi[:, 0:2]), np.ones((xi.shape[0], 1)))))
    plot_obsv, = plt.plot(xi_XY[:, 0], xi_XY[:, 1], 'g', transform=rot + base,)

    gt_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(gt), np.ones((gt.shape[0], 1)))))
    plot_gt, = plt.plot(gt_XY[:, 0], gt_XY[:, 1], 'y', transform=rot + base,)

    cur_pos_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(xi[-1, 0:2]), np.array([1]))))
    plot_pos, = plt.plot(cur_pos_XY[0], cur_pos_XY[1], 'mo', transform=rot + base,)

    plt.legend((plot_pred, plot_obsv, plot_gt, plot_pos), ('pred', 'obsv', 'gt', 'cur pos'))
    plt.gcf().set_size_inches(16, 12)

    plt.xlim((0, 640))
    plt.ylim((-480, 0))
    plt.show()


def eval(test_laoder_):
    ade_sum_12 = 0
    fde_sum_12 = 0
    ade_sum_08 = 0
    fde_sum_08 = 0
    err_counter = 0

    for _, (obsv_batch, teom_batch, target_batch, gt_batch) in enumerate(test_laoder_):
        obsv_batch = obsv_batch.cuda()
        teom_batch = teom_batch.cuda()

        batch_size = obsv_batch.size(0)
        output_batch, risk = cnn_model(obsv_batch, teom_batch)

        output_batch = output_batch.cpu().data.numpy().reshape((batch_size, n_next, 2))
        target_batch = target_batch.cpu().data.numpy().reshape((batch_size, n_next, 2))
        gt_batch = gt_batch.cpu().data.numpy().reshape(batch_size, n_next, 2)


        for ii in range(batch_size):
            xi = obsv_batch[ii].cpu().data.numpy().reshape((n_past, -1))
            teom = teom_batch[ii]
            cv_pred = cv_model.predict(xi, n_next)

            y_hats = output_batch[ii]
            y_gts = gt_batch[ii]

            # y_gts = target_batch[ii]
            # output_ii = polar_scale.denormalize(output_batch[ii], shift=False, inPlace=True)
            # target_ii = polar_scale.denormalize(target_batch[ii], shift=False, inPlace=False)
            # base_v = xi[-1, 2:4]  # last instant vel
            # base_v = (xi[-1, 0:2] - xi[0, 0:2]) / (len(xi) - 1)  # observation avg vel
            # # base_v = unit(base_v)  # it could be tested
            # [_, base_theta] = cart2pol(base_v[0], base_v[1])
            # output_ii[:, 1] = output_ii[:, 1] + base_theta
            # target_ii[:, 1] = target_ii[:, 1] + base_theta
            # y_hats = pol2cart(output_ii[:, 0], output_ii[:, 1]) + xi[-1, :2]
            # y_gts = pol2cart(target_ii[:, 0], target_ii[:, 1]) + xi[-1, :2]

            for ts in range(n_next):
                y_hat = y_hats[ts]
                y_gt = y_gts[ts]

                pred = scale.denormalize(y_hat, shift=True, inPlace=True)
                gt = scale.denormalize(y_gt, shift=True, inPlace=False)
                ade_sum_12 += norm(pred - gt) / n_next

                if ts < 8:
                    ade_sum_08 += norm(pred - gt) / 8
                if ts == 8 - 1:
                    fde_sum_08 += norm(pred - gt)

            fde_sum_12 += norm(pred - gt)
            err_counter += 1

    return ade_sum_08/err_counter, fde_sum_08/err_counter,\
           ade_sum_12/err_counter, fde_sum_12/err_counter

dt = 6
disp = Display('../data/eth')
def generate_nice_samples():
    for t in range(time_data[0][0], 10000, dt):
        t=948
        n = 0
        for i in range(len(pos_data)):
            t1_ind = np.array(np.where(time_data[i] == t))
            ts_ind = np.array(np.where(time_data[i] == t-n_past*dt))
            te_ind = np.array(np.where(time_data[i] == t+(4-1)*dt))

            if t1_ind.size == 0 or ts_ind.size == 0 or te_ind.size == 0:
                continue

            t1_ind = t1_ind[0][0]
            ts_ind = ts_ind[0][0]
            te_ind = te_ind[0][0]

            xi = pos_data[i][ts_ind:t1_ind]
            xi_smooth = pos_data_smooth[i][ts_ind:t1_ind]
            yi = pos_data[i][t1_ind:te_ind+1]

            teom_i = torch.FloatTensor(test_teom[0]).cuda().unsqueeze(0)
            gt_i = test_gt[0]
            # obsv_i = test_obsv[i]
            # target_i = test_target[i]
            xi_tf = torch.FloatTensor(xi_smooth).cuda().unsqueeze(0)
            pred, risk = cnn_model(xi_tf, teom_i, n_next, False)
            pred = pred.cpu().data.numpy().reshape((n_next, 2))
            pred_cv = cv_model.predict(xi, n_next)

            xi_lbl, = plt.plot(xi[:, 0], xi[:, 1], 'g')
            yi_lbl, = plt.plot(yi[:, 0], yi[:, 1], 'g--')
            yhat_lbl, = plt.plot(pred[:, 0], pred[:, 1], 'b--')
            y_cv_lbl, = plt.plot(pred_cv[:, 0], pred_cv[:, 1], 'y--')

            n += 1


        if n > 2:
            plt.legend((xi_lbl, yi_lbl, yhat_lbl, y_cv_lbl), ('obsv', 'gt', 'pred', 'cv'))
            plt.gcf().set_size_inches(16, 16)
            plt.xlim([0, 1])
            plt.ylim([0.1, 0.7])
            # plt.show()


        plt.clf()


for i in range(len(time_data)):
    time_data[i] = time_data[i].astype(int)


min_test_error = 1000
try:
    cnn_model.load_state_dict(torch.load('./models/navigator3.pt'))
except:
    print("Could not load the trained weights!")

generate_nice_samples()


for epoch in range(1, N_EPOCH):
    loss_accum = 0
    loss_risk_accum = 0
    loss_count = 0
    avg_loss_l2 = 0

    for _, (obsv_batch, teom_batch, target_batch, gt_batch) in enumerate(train_loader):
        obsv_batch = obsv_batch.cuda()
        teom_batch = teom_batch.cuda()
        target_batch = target_batch.cuda()
        gt_batch = gt_batch.cuda()

        batch_size = obsv_batch.size(0)
        output_batch, risk_batch = cnn_model(obsv_batch, teom_batch)


        # gt_batch = gt_batch - obsv_batch[:, -1, :]
        # loss = loss_func(output_batch, gt_batch)
        loss_gt = loss_func(output_batch, gt_batch)
        loss_risk = torch.sum(torch.sum(risk_batch, dim=1))

        total_loss = loss_gt  # + 5 * loss_risk/batch_size
        optimizer.zero_grad()  # clear gradients for this training step
        total_loss.backward()  # back-propagation, compute gradients
        optimizer.step()  # apply gradients

        loss_accum += np.sqrt(loss_gt.item()) / scale.sx * batch_size
        loss_risk_accum += loss_risk.item() * batch_size
        loss_count += batch_size

        # # ============== DEBUG ===============
        # if epoch > 0 and i == 0:
        #     for j in range(0, batch_size, 1):
        #         im = xs[j, :, :] * 255
        #         im_cnn = x_cnn[j, :, :]
        #         im_cnn.detach()
        #         im_cnn = im_cnn.cpu().data.numpy().reshape((6, 6))
        #         plt.subplot(1, 3, 1)
        #         plt.imshow(im[:, :, 0])
        #         plt.subplot(1, 3, 2)
        #         plt.imshow(im[:, :, 1])
        #         plt.subplot(1, 3, 3)
        #         plt.imshow(im_cnn)
        #         plt.show()

    avg_loss_l2 = loss_accum / (loss_count + eps)
    ade_err_08, fde_err_08, ade_err_12, fde_err_12 = eval(test_loader)
    print('Epoch [%3d/%d], Train Loss: %4f | ADE(8)= %4f | FDE(8)= %4f | ADE(12)= %4f | FDE(12)= %4f | Risk = %2f'
          % (epoch, N_EPOCH, avg_loss_l2, ade_err_08, fde_err_08, ade_err_12, fde_err_12, loss_risk_accum / (loss_count + eps)))

    if ade_err_12 < min_test_error:
        torch.save(cnn_model.state_dict(), './models/navigator3.pt')
        min_test_error = ade_err_12

    if epoch % 10 == 0 and ade_err_12 < 10.:
        for i in range(min(test_obsv.shape[0], 10)):
            rnd = np.random.randint(0, test_obsv.shape[0])
            i = rnd
            # i = 10
            obsv_i = test_obsv[i]
            target_i = test_target[i]
            teom_i = test_teom[i]
            gt_i = test_gt[i]
            # visualize_one(obsv_i, teom_i, target_i, gt_i)

        # break

