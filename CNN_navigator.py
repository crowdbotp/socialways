import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from lstm_model.parse_utils import Scale


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        n_filters_1 = 16
        n_filters_2 = 1
        kernel_size = 5
        stride_1 = 1
        stride_2 = 1
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=2,  # input height
                out_channels=n_filters_1,  # n_filters
                kernel_size=kernel_size,  # filter size
                stride=stride_1,  # filter movement/step
                padding=1,
            ), # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3),  # choose max value in 2x2 area, output shape (16, xxx, xxx)
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
        self.out = nn.Linear(n_filters_2 * 6 * 6, 2).cuda()  # fully connected layer, output 10 classes

    def forward(self, x):
        # x = torch.unsqueeze(x, 1)
        x0 = x[:, :, :, 0]
        x1 = x[:, :, :, 1]
        x = torch.stack((x0, x1), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x_flat = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x_flat)
        return output, x  # return x for visualization


# ================== MAIN CODE ==================
data = np.load('../maps_eth_48_5m/data.npz')
X, y = data['X'], data['y']
print('Input shape = %s  |  Output shape = %s' % (X.shape, y.shape))

BATCH_SIZE = 64
N_EPOCH = 100
LR = 0.001              # learning rate
train_ratio = 0.6
train_X = X[100:int(train_ratio * X.shape[0])]
train_y = y[100:int(train_ratio * X.shape[0])]

# ========== SCALE ============
train_X = (train_X/255)
y_scale = Scale()
y_scale.min_x = min(train_y[:, 0])
y_scale.min_y = min(train_y[:, 1])
y_scale.max_x = max(train_y[:, 0])
y_scale.max_y = max(train_y[:, 1])
y_scale.calc_scale(keep_ratio=False)
y_scale.normalize(train_y, shift=True, inPlace=True)


train = torch.utils.data.TensorDataset(torch.FloatTensor(train_X), torch.FloatTensor(train_y))
train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=False)
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()                       # the target label is not one-hotted

for epoch in range(1, N_EPOCH):
    loss_accum = 0
    loss_count = 0
    for i, (data_x, data_y) in enumerate(train_loader):
        xs = data_x.cuda()
        ys = data_y.cuda()

        batch_size = xs.size(0)
        ys_hat, x_cnn = cnn(xs)

        loss = loss_func(ys_hat, ys)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        loss_accum += loss.item() * batch_size
        loss_count += batch_size

        # ============== DEBUG ===============
        if epoch > 0 and i == 0:
            for j in range(0, batch_size, 1):
                im = xs[j, :, :] * 255
                im_cnn = x_cnn[j, :, :]
                im_cnn.detach()
                im_cnn = im_cnn.cpu().data.numpy().reshape((6, 6))
                plt.subplot(1, 3, 1)
                plt.imshow(im[:, :, 0])
                plt.subplot(1, 3, 2)
                plt.imshow(im[:, :, 1])
                plt.subplot(1, 3, 3)
                plt.imshow(im_cnn)
                plt.show()

    print('epoch [%3d/%d], Loss: %5f' % (epoch, N_EPOCH, loss_accum/loss_count))
