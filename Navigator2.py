import glob
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import transforms
from torch.utils.data import DataLoader, TensorDataset

from src.learning_utils import MyConfig
from src.parse_utils import Scale, BIWIParser
from src.math_utils import unit, cart2pol, pol2cart, norm
from src.visualize import to_image_frame

config = MyConfig()
n_past = config.n_past
n_next = config.n_next


class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()

        n_filters_1 = 16
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
        self.lstm_obsv = nn.LSTM(4, self.lstm_obsv_size, 1, batch_first=True).cuda()
        self.lstm_teom = nn.LSTM(6*6, self.lstm_teom_size, 1, batch_first=True).cuda()
        # self.fc_mix = nn.Sequential(
        #     nn.Linear(self.lstm_obsv_size + self.lstm_teom_size, mix_size), #nn.ReLU(),
        # ).cuda()

        self.obsv_fc = nn.Linear(self.lstm_obsv_size, 64).cuda()
        self.teom_fc = nn.Linear(self.lstm_teom_size, 64).cuda()
        mix_size = 16
        self.fc_mix = nn.ModuleList([nn.Linear(64 + 64, mix_size) for _ in range(n_next)]).cuda()

        self.out = nn.Sequential(
            nn.Linear(mix_size, 32),
            nn.ReLU(0.1),
            nn.Linear(32, 32),
            nn.ReLU(0.1),
            nn.Linear(32, 2)
        ).cuda()

    def forward(self, obsv, teom):
        # teom = [bs, n_next, H, W, nch]
        # teom = torch.unsqueeze(teom, 1)
        bs = teom.shape[0]
        ts = teom.shape[1]

        # initialize hidden state: (num_layers, minibatch_size, hidden_dim)
        obsv_state = (torch.zeros(1, bs, self.lstm_obsv_size).cuda(),
                      torch.zeros(1, bs, self.lstm_obsv_size).cuda())

        (obsv_state, _) = self.lstm_obsv(obsv, obsv_state)
        obsv_state = obsv_state[:, -1, :]
        teom_hidden_state = (torch.zeros(1, bs, self.lstm_teom_size).cuda(),
                             torch.zeros(1, bs, self.lstm_teom_size).cuda())
        obsv_state = self.obsv_fc(obsv_state)

        y = torch.zeros(bs, n_next, 2).cuda()

        for t in range(ts):
            # teom_ch0 = teom[:, t, :, :, 0]
            # teom_ch1 = teom[:, t, :, :, 0]
            # teom_channels = torch.stack((teom_ch0, teom_ch1), 1)
            teom_channels = teom[:, t, :, :].unsqueeze(1)
            conved_teom = self.conv1(teom_channels)
            conved_teom = self.conv2(conved_teom)
            conved_teom = self.conv3(conved_teom)
            conved_teom_flat = conved_teom.view(bs, 1, -1)
            (teom_state, teom_hidden_state) = self.lstm_teom(conved_teom_flat, teom_hidden_state)
            teom_state = teom_state.view(bs, -1)
            teom_state = self.teom_fc(teom_state)
            memory_mix = torch.cat((teom_state, obsv_state), 1)
            y_mix = self.fc_mix[t](memory_mix)
            y[:, t, :] = self.out(y_mix)

        return y  # ,memory_teom  # for visualization


# ================== Load Data ==================
file_name = '../teom/eth_big_file.npz'
data_dir = '../data/eth'
data = np.load(file_name)
print('Training on %s, Observe/Predict [%d]/[%d] frames' %(file_name, n_past, n_next))

obsvs = data['obsvs']
teoms = data['teoms']
targets = data['targets']

row_size = teoms.shape[2]
col_size = teoms.shape[3]
teoms = teoms[:, :, row_size//4:row_size*3//4 + 1, col_size//4:col_size*3//4 + 1, 1]

print('Obsv size:%s |  TEOMs size:%s |  Targets size:%s' % (obsvs.shape, teoms.shape, targets.shape))

# ========== SCALE ============
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
# ================================================

# ========= Divide into train & test ============
train_ratio = 0.8
train_size = int(train_ratio * obsvs.shape[0])
train_obsv = obsvs[:train_size]
train_teom = teoms[:train_size]
train_target = targets[:train_size]

test_obsv = obsvs[train_size:]
test_teom = teoms[train_size:]
test_target = targets[train_size:]
# ================================================


LR = 0.001  # learning rate
N_EPOCH = 2000
BATCH_SIZE = 32
train = TensorDataset(torch.FloatTensor(train_obsv), torch.FloatTensor(train_teom), torch.FloatTensor(train_target))
train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=False)

test = TensorDataset(torch.FloatTensor(test_obsv), torch.FloatTensor(test_teom), torch.FloatTensor(test_target))
test_loader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=False)

model = ConvLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all parameters
loss_func = nn.MSELoss()                       # the target label is not one-hotted


# ===================================================
Hfile = os.path.join(data_dir, "H.txt")
Hinv = np.linalg.inv(np.loadtxt(Hfile))

parser = BIWIParser()
pos_data, vel_data, time_data = parser.load(os.path.join(data_dir, 'obsmat.txt'))
scale = parser.scale

def visualize_one(xi, teom, target):
    obsv = torch.FloatTensor(xi).cuda().unsqueeze(0)
    teom = torch.FloatTensor(teom).cuda().unsqueeze(0)
    # target = target

    output = model(obsv, teom)
    output = output.cpu().data.numpy().reshape((n_next, 2))
    output = polar_scale.denormalize(output, shift=False, inPlace=True)

    target = polar_scale.denormalize(target, shift=False, inPlace=False)

    base_v = xi[-1, 2:4]  # last instant vel
    base_v = (xi[-1, 0:2] - xi[0, 0:2]) / (len(xi) - 1)  # observation avg vel
    # base_v = unit(base_v)  # it could be tested
    [_, base_theta] = cart2pol(base_v[0], base_v[1])
    output[:, 1] = output[:, 1] + base_theta
    target[:, 1] = target[:, 1] + base_theta
    preds = []
    gt = []
    for ts in range(n_next):
        dp_hat = pol2cart(output[ts, 0], output[ts, 1])
        dp_gt = pol2cart(target[ts, 0], target[ts, 1])

        preds.append(dp_hat + xi[-1, 0:2])
        gt.append(dp_gt + xi[-1, 0:2])

    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(-90)

    preds = np.stack(preds)
    gt = np.stack(gt)

    preds_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(preds), np.ones((preds.shape[0], 1)))))
    plot_pred, = plt.plot(preds_XY[:, 0], preds_XY[:, 1], 'r--', transform=rot + base,)

    xi_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(xi[:, 0:2]), np.ones((xi.shape[0], 1)))))
    plot_obsv, = plt.plot(xi_XY[:, 0], xi_XY[:, 1], 'g', transform=rot + base,)

    gt_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(gt), np.ones((gt.shape[0], 1)))))
    plot_gt, = plt.plot(gt_XY[:, 0], gt_XY[:, 1], 'y', transform=rot + base,)

    cur_pos_XY = to_image_frame(Hinv, np.hstack((scale.denormalize(xi[-1, 0:2]), np.array([1]))))
    plot_pos, = plt.plot(cur_pos_XY[0], cur_pos_XY[1], 'mo', transform=rot + base,)

    plt.legend((plot_pred, plot_obsv, plot_gt, plot_pos), ('pred', 'obsv', 'gt', 'cur pos'))
    plt.gcf().set_size_inches(16, 12)

    # plt.xlim((0, 640))
    # plt.ylim((-480, 0))
    # plt.show()


def eval(test_laoder_):
    ade_sum = 0
    fde_sum = 0
    err_counter = 0

    for _, (obsv_batch, teom_batch, target_batch) in enumerate(test_laoder_):
        obsv_batch = obsv_batch.cuda()
        teom_batch = teom_batch.cuda()

        batch_size = obsv_batch.size(0)
        output_batch = model(obsv_batch, teom_batch)

        output_batch = output_batch.cpu().data.numpy().reshape((batch_size, n_next, 2))
        target_batch = target_batch.cpu().data.numpy().reshape((batch_size, n_next, 2))


        for ii in range(batch_size):
            xi = obsv_batch[ii].cpu().data.numpy().reshape((n_past, -1))
            obsv = teom_batch[ii]

            output = polar_scale.denormalize(output_batch[ii], shift=False, inPlace=True)
            target = polar_scale.denormalize(target_batch[ii], shift=False, inPlace=False)

            # target = target_batch[ii]
            # output = output_batch[ii]

            base_v = xi[-1, 2:4]  # last instant vel
            base_v = (xi[-1, 0:2] - xi[0, 0:2]) / (len(xi) - 1)  # observation avg vel
            # base_v = unit(base_v)  # it could be tested
            [_, base_theta] = cart2pol(base_v[0], base_v[1])
            output[:, 1] = output[:, 1] + base_theta
            target[:, 1] = target[:, 1] + base_theta
            for ts in range(n_next):
                y_hat = pol2cart(output[ts, 0], output[ts, 1]) + xi[-1, 0:2]
                y_gt = pol2cart(target[ts, 0], target[ts, 1]) + xi[-1, 0:2]

                pred = scale.denormalize(y_hat, shift=True, inPlace=True)
                gt = scale.denormalize(y_gt, shift=True, inPlace=True)
                ade_sum += norm(pred - gt)/n_next
            fde_sum += norm(pred - gt)
            err_counter += 1

    return ade_sum/err_counter, fde_sum/err_counter



min_test_error = 1000
model.load_state_dict(torch.load(os.path.expanduser('~') + '/Dropbox/CVPR2019/models/navigator3.pt'))

for epoch in range(1, N_EPOCH):
    loss_accum = 0
    loss_count = 0

    for _, (obsv_batch, teom_batch, target_batch) in enumerate(train_loader):
        obsv_batch = obsv_batch.cuda()
        teom_batch = teom_batch.cuda()
        target_batch = target_batch.cuda()

        batch_size = obsv_batch.size(0)
        output_batch = model(obsv_batch, teom_batch)

        loss = loss_func(output_batch, target_batch)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        loss_accum += loss.item() * batch_size
        loss_count += batch_size
        pass

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

    avg_loss = loss_accum/loss_count * 100
    ade_err, fde_err = eval(test_loader)
    print('Epoch [%3d/%d], Train Loss: %5f | Test ADE= %5f | Test FDE= %5f'
          % (epoch, N_EPOCH, avg_loss, ade_err, fde_err))

    if ade_err < min_test_error:
        torch.save(model.state_dict(), os.path.expanduser('~') + '/Dropbox/CVPR2019/models/navigator3.pt')
        min_test_error = ade_err

    if epoch % 10 == 0:
        rnd = np.random.randint(0, test_obsv.shape[0])
        for i in range(test_obsv.shape[0]):
            i = rnd
            obsv_i = test_obsv[i]
            target_i = test_target[i]
            teom_i = test_teom[i]
            visualize_one(obsv_i, teom_i, target_i)
            break
