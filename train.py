import os
import time
import argparse
import matplotlib.pyplot as plt
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm, trange
from itertools import chain
from torch.autograd import Variable
from utils.parse_utils import Scale
from torch.utils.data import DataLoader
from utils.linear_models import predict_cv


# Parser arguments
parser = argparse.ArgumentParser(description='Social Ways trajectory prediction.')
parser.add_argument('--batch-size', '--b',
                    type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', '--e',
                    type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--model', '--m',
                    default='socialWays',
                    choices=['socialWays'],
                    help='pick a specific network to train'
                         '(default: "socialWays")')
parser.add_argument('--latent-dim', '--ld',
                    type=int, default=10, metavar='N',
                    help='dimension of latent space (default: 10)')
parser.add_argument('--d-learning-rate', '--d-lr',
                    type=float, default=1E-3, metavar='N',
                    help='learning rate of discriminator (default: 1E-3)')
parser.add_argument('--g-learning-rate', '--g-lr',
                    type=float, default=1E-4, metavar='N',
                    help='learning rate of generator (default: 1E-4)')
parser.add_argument('--unrolling-steps', '--unroll',
                    type=int, default=1, metavar='N',
                    help='number of steps to unroll gan (default: 1)')
parser.add_argument('--hidden-size', '--h-size',
                    type=int, default=64, metavar='N',
                    help='size of network intermediate layer (default: 64)')
parser.add_argument('--dataset', '--data',
                    default='hotel',
                    choices=['hotel'],
                    help='pick a specific dataset (default: "hotel")')
args = parser.parse_args()


# ========== set input/output files ============
dataset_name = args.dataset
model_name = args.model
input_file = '../hotel-8-12.npz'
model_file = '../trained_models/' + model_name + '-' + dataset_name + '.pt'

# FIXME: ====== training hyper-parameters ======
# Unrolled GAN
n_unrolling_steps = args.unrolling_steps
# Info GAN
use_info_loss = True
loss_info_w = 0.5
n_latent_codes = 2
# L2 GAN
use_l2_loss = False
use_variety_loss = False
loss_l2_w = 0.5  # WARNING for both l2 and variety
# Learning Rate
lr_g = args.g_learning_rate
lr_d = args.d_learning_rate
# FIXME: ====== Network Size ===================
# Batch size
batch_size = args.batch_size
# LSTM hidden size
hidden_size = args.hidden_size
n_epochs = args.epochs
num_social_features = 3
social_feature_size = args.hidden_size
noise_len = args.hidden_size // 2
n_lstm_layers = 1
use_social = False
# ==============================================

# FIXME: ======= Loda Data =====================
print(os.path.dirname(os.path.realpath(__file__)))

data = np.load(input_file)
# Data come as NxTx2 numpy nd-arrays where N is the number of trajectories,
# T is their duration.
dataset_obsv, dataset_pred, dataset_t, the_batches = \
    data['obsvs'], data['preds'], data['times'], data['batches']
# 4/5 of the batches to be used for training
train_size = max(1, (len(the_batches) * 4) // 5)
train_batches = the_batches[:train_size]
# Test batches are the remaining ones
test_batches = the_batches[train_size:]
# Size of the observed sub-paths
n_past = dataset_obsv.shape[1]
# Size of the sub-paths to predict
n_next = dataset_pred.shape[1]
# Number of training samples
n_train_samples = the_batches[train_size - 1][1]
# Number of testing samples (the remaining ones)
n_test_samples = dataset_obsv.shape[0] - n_train_samples
if n_test_samples == 0:
    n_test_samples = 1
    the_batches = np.array([the_batches[0], the_batches[0]])
print(input_file, ' # Training samples: ', n_train_samples)

# Normalize the spatial data
scale = Scale()
scale.max_x = max(np.max(dataset_obsv[:, :, 0]), np.max(dataset_pred[:, :, 0]))
scale.min_x = min(np.min(dataset_obsv[:, :, 0]), np.min(dataset_pred[:, :, 0]))
scale.max_y = max(np.max(dataset_obsv[:, :, 1]), np.max(dataset_pred[:, :, 1]))
scale.min_y = min(np.min(dataset_obsv[:, :, 1]), np.min(dataset_pred[:, :, 1]))
scale.calc_scale(keep_ratio=True)
dataset_obsv = scale.normalize(dataset_obsv)
dataset_pred = scale.normalize(dataset_pred)
ss = scale.sx
# Copy normalized observations/paths to predict into torch GPU tensors
dataset_obsv = torch.FloatTensor(dataset_obsv).cuda()
dataset_pred = torch.FloatTensor(dataset_pred).cuda()


# ================================================

# Augment tensors of positions into positions+velocity
def get_traj_4d(obsv_p, pred_p):
    obsv_v = obsv_p[:, 1:] - obsv_p[:, :-1]
    obsv_v = torch.cat([obsv_v[:, 0].unsqueeze(1), obsv_v], dim=1)
    obsv_4d = torch.cat([obsv_p, obsv_v], dim=2)
    if len(pred_p) == 0: return obsv_4d
    pred_p_1 = torch.cat([obsv_p[:, -1].unsqueeze(1), pred_p[:, :-1]], dim=1)
    pred_v = pred_p - pred_p_1
    pred_4d = torch.cat([pred_p, pred_v], dim=2)
    return obsv_4d, pred_4d


# Evaluate the error between the model prediction and the true path
def calc_error(pred_hat, pred):
    N = pred.size(0)
    T = pred.size(1)
    err_all = torch.pow((pred_hat - pred) / ss, 2).sum(dim=2).sqrt()  # N x T
    FDEs = err_all.sum(dim=0).item() / N
    ADEs = torch.cumsum(FDEs)
    for ii in range(T):
        ADEs[ii] /= (ii + 1)
    return ADEs.data.cpu().numpy(), FDEs.data().cpu().numpy()


class AttentionPooling(nn.Module):
    def __init__(self, h_dim, f_dim):
        super(AttentionPooling, self).__init__()
        self.f_dim = f_dim
        self.h_dim = h_dim
        self.W = nn.Linear(h_dim, f_dim, bias=True)

    def forward(self, f, h, sub_batches):
        Wh = self.W(h)
        S = torch.zeros_like(h)
        for sb in sub_batches:
            N = sb[1] - sb[0]
            if N == 1: continue

            for ii in range(sb[0], sb[1]):
                fi = f[ii, sb[0]:sb[1]]
                sigma_i = torch.bmm(fi.unsqueeze(1), Wh[sb[0]:sb[1]]. unsqueeze(2))
                sigma_i[ii-sb[0]] = -1000

                attentions = torch.softmax(sigma_i.squeeze(), dim=0)
                S[ii] = torch.mm(attentions.view(1, N), h[sb[0]:sb[1]])

        return S


class EmbedSocialFeatures(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EmbedSocialFeatures, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(),
                                nn.Linear(32, 64), nn.ReLU(),
                                nn.Linear(64, hidden_size))

    def forward(self, ftr_list, sub_batches):
        embedded_features = self.fc(ftr_list)
        return embedded_features


def DCA(xA_4d, xB_4d):
    dp = xA_4d[:2] - xB_4d[:2]
    dv = xA_4d[2:] - xB_4d[2:]
    ttca = torch.dot(-dp, dv) / (torch.norm(dv) ** 2 + 1E-6)
    # ttca = torch.max(ttca, 0)
    dca = torch.norm(dp + ttca * dv)
    return dca


def Bearing(xA_4d, xB_4d):
    dp = xA_4d[:2] - xB_4d[:2]
    v = xA_4d[2:]
    cos_theta = torch.dot(dp, v) / (torch.norm(dp) * torch.norm(v) + 1E-6)
    return cos_theta


def DCA_MTX(x_4d, D_4d):
    Dp = D_4d[:, :, :2]
    Dv = D_4d[:, :, 2:]
    DOT_Dp_Dv = torch.mul(Dp[:,:,0], Dv[:,:,0]) + torch.mul(Dp[:,:,1], Dv[:,:,1])
    Dv_sq = torch.mul(Dv[:,:,0], Dv[:,:,0]) + torch.mul(Dv[:,:,1], Dv[:,:,1]) + 1E-6
    TTCA = -torch.div(DOT_Dp_Dv, Dv_sq)
    DCA = torch.zeros_like(Dp)
    DCA[:, :, 0] = Dp[:, :, 0] + TTCA * Dv[:, :, 0]
    DCA[:, :, 1] = Dp[:, :, 1] + TTCA * Dv[:, :, 1]
    DCA = torch.norm(DCA, dim=2)
    return DCA


def BearingMTX(x_4d, D_4d):
    Dp = D_4d[:, :, :2]  # NxNx2
    v = x_4d[:, 2:].unsqueeze(1).repeat(1, x_4d.shape[0], 1)  # => NxNx2
    DOT_Dp_v = Dp[:, :, 0] * v[:, :, 0] + Dp[:, :, 1] * v[:, :, 1]
    COS_THETA = torch.div(DOT_Dp_v, torch.norm(Dp, dim=2) * torch.norm(v, dim=2) + 1E-6)
    return COS_THETA


def SocialFeatures(x, sub_batches):
    N = x.shape[0]  # x is NxTx4 tensor

    x_ver_repeat = x[:, -1].unsqueeze(0).repeat(N, 1, 1)
    x_hor_repeat = x[:, -1].unsqueeze(1).repeat(1, N, 1)
    Dx_mat = x_hor_repeat - x_ver_repeat

    l2_dist_MTX = Dx_mat[:, :, :2].norm(dim=2)
    bearings_MTX = BearingMTX(x[:, -1], Dx_mat)
    dcas_MTX = DCA_MTX(x[:, -1], Dx_mat)
    sFeatures_MTX = torch.stack([l2_dist_MTX, bearings_MTX, dcas_MTX], dim=2)

    return sFeatures_MTX   # directly return the Social Features Matrix


# LSTM path encoding module
class EncoderLstm(nn.Module):
    def __init__(self, hidden_size, n_layers=2):
        # Dimension of the hidden state (h)
        self.hidden_size = hidden_size
        super(EncoderLstm, self).__init__()
        # Linear embedding 4xh
        self.embed = nn.Linear(4, self.hidden_size)
        # The LSTM cell.
        # Input dimension (observations mapped through embedding) is the same as the output
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=n_layers, batch_first=True)
        self.lstm_h = []
        # init_weights(self)

    def init_lstm(self, h, c):
        # Initialization of the LSTM: hidden state and cell state
        self.lstm_h = (h, c)

    def forward(self, obsv):
        # Batch size
        bs = obsv.shape[0]
        # Linear embedding
        obsv = self.embed(obsv)
        # Reshape and applies LSTM over a whole sequence or over one single step
        y, self.lstm_h = self.lstm(obsv.view(bs, -1, self.hidden_size), self.lstm_h)
        return y


class Discriminator(nn.Module):
    def __init__(self, n_next, hidden_dim, n_latent_code):
        super(Discriminator, self).__init__()
        self.lstm_dim = hidden_dim
        self.n_next = n_next
        # LSTM Encoder for the observed part
        self.obsv_encoder_lstm = nn.LSTM(4, hidden_dim, batch_first=True)
        # FC sub-network: input is hidden_dim, output is hidden_dim//2. This ouput will be part of
        # the input of the classifier.
        self.obsv_encoder_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                                             nn.Linear(hidden_dim // 2, hidden_dim // 2))
        # FC Encoder for the predicted part: input is n_next*4 (whole predicted trajectory), output is
        # hidden_dim//2. This ouput will also be part of the input of the classifier.
        self.pred_encoder = nn.Sequential(nn.Linear(n_next * 4, hidden_dim // 2), nn.LeakyReLU(0.2),
                                          nn.Linear(hidden_dim // 2, hidden_dim // 2))
        # Classifier: input is hidden_dim (concatenated encodings of observed and predicted trajectories), output is 1
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                                        nn.Linear(hidden_dim // 2, 1))
        # Latent code inference: input is hidden_dim (concatenated encodings of observed and predicted trajectories), output is n_latent_code (distribution of latent codes)
        self.latent_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                                            nn.Linear(self.lstm_dim // 2, n_latent_code))

    def forward(self, obsv, pred):
        bs = obsv.size(0)
        lstm_h_c = (torch.zeros(1, bs, self.lstm_dim).cuda(),
                    torch.zeros(1, bs, self.lstm_dim).cuda())
        # Encoding of the observed sequence trhough an LSTM cell
        obsv_code, lstm_h_c = self.obsv_encoder_lstm(obsv, lstm_h_c)
        # Further encoding through a FC layer
        obsv_code = self.obsv_encoder_fc(obsv_code[:, -1])
        # Encoding of the predicted/next part of the sequence through a FC layer
        pred_code = self.pred_encoder(pred.view(-1, self.n_next * 4))
        both_codes = torch.cat([obsv_code, pred_code], dim=1)
        # Applies classifier to the concatenation of the encodings of both parts
        label = self.classifier(both_codes)
        # Inference on the latent code
        code_hat = self.latent_decoder(both_codes)
        return label, code_hat

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


# FC path decoding module
class DecoderFC(nn.Module):
    def __init__(self, hidden_dim):
        super(DecoderFC, self).__init__()
        # Fully connected sub-network. Input is hidden_dim, output is 2.
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                                       # torch.nn.Linear(64, 64), nn.LeakyReLU(0.2),
                                       torch.nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                                       torch.nn.Linear(hidden_dim // 2, hidden_dim // 4),
                                       torch.nn.Linear(hidden_dim // 4, 2))

    def forward(self, h, s, z):
        # For each sample in the batch, concatenate h (hidden state), s (social term) and z (noise)
        inp = torch.cat([h, s, z], dim=1)
        # Applies the fully connected layer
        out = self.fc1(inp)
        return out


# LSTM path decoding module
class DecoderLstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DecoderLstm, self).__init__()
        # Decoding LSTM
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        # Fully connected sub-network. Input is hidden_size, output is 2.
        self.fc = nn.Sequential(torch.nn.Linear(hidden_size, 64), nn.Sigmoid(),
                                torch.nn.Linear(64, 64), nn.LeakyReLU(0.2),
                                torch.nn.Linear(64, 32), nn.LeakyReLU(0.2),
                                torch.nn.Linear(32, 2))

        # init_weights(self)
        self.lstm_h = []

    def init_lstm(self, h, c):
        # Initialization of the LSTM: hidden state and cell state
        self.lstm_h = (h, c)

    def forward(self, h, s, z):
        # Batch size
        bs = z.shape[0]
        # For each sample in the batch, concatenate h (hidden state), s (social term) and z (noise)
        inp = torch.cat([h, s, z], dim=1)
        # Applies a forward step.
        out, self.lstm_h = self.lstm(inp.unsqueeze(1), self.lstm_h)
        # Applies the fully connected layer to the LSTM output
        out = self.fc(out.squeeze())
        return out


# LSTM-based path encoder
encoder = EncoderLstm(hidden_size, n_lstm_layers).cuda()
feature_embedder = EmbedSocialFeatures(num_social_features, social_feature_size).cuda()
attention = AttentionPooling(hidden_size, social_feature_size).cuda()

# Decoder
decoder = DecoderFC(hidden_size + social_feature_size + noise_len).cuda()
# decoder = DecoderLstm(social_feature_size + VEL_VEC_LEN + noise_len, traj_code_len).cuda()

# The Generator parameters and their optimizer
predictor_params = chain(attention.parameters(), feature_embedder.parameters(),
                         encoder.parameters(), decoder.parameters())
predictor_optimizer = opt.Adam(predictor_params, lr=lr_g, betas=(0.9, 0.999))

# The Discriminator parameters and their optimizer
D = Discriminator(n_next, hidden_size, n_latent_codes).cuda()
D_optimizer = opt.Adam(D.parameters(), lr=lr_d, betas=(0.9, 0.999))
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

print('hidden dim = %d | lr(G) =  %.5f | lr(D) =  %.5f' % (hidden_size, lr_g, lr_d))


def predict(obsv_p, noise, n_next, sub_batches=[]):
    # Batch size
    bs = obsv_p.shape[0]
    # Adds the velocity component to the observations.
    # This makes of obsv_4d a batch_sizexTx4 tensor
    obsv_4d = get_traj_4d(obsv_p, [])
    # Initial values for the hidden and cell states (zero)
    lstm_h_c = (torch.zeros(n_lstm_layers, bs, encoder.hidden_size).cuda(),
                torch.zeros(n_lstm_layers, bs, encoder.hidden_size).cuda())
    encoder.init_lstm(lstm_h_c[0], lstm_h_c[1])
    # Apply the encoder to the observed sequence
    # obsv_4d: batch_sizexTx4 tensor
    encoder(obsv_4d)
    if len(sub_batches) == 0:
        sub_batches = [[0, obsv_p.size(0)]]

    if use_social:
        features = SocialFeatures(obsv_4d, sub_batches)
        emb_features = feature_embedder(features, sub_batches)
        weighted_features = attention(emb_features, encoder.lstm_h[0].squeeze(), sub_batches)
    else:
        weighted_features = torch.zeros_like(encoder.lstm_h[0].squeeze())

    pred_4ds = []
    last_obsv = obsv_4d[:, -1]
    # For all the steps to predict, applies a step of the decoder
    for ii in range(n_next):
        # Takes the current output of the encoder to feed the decoder
        # Gets the ouputs as a displacement/velocity
        new_v = decoder(encoder.lstm_h[0].view(bs, -1), weighted_features.view(bs, -1), noise).view(bs, 2)
        # Deduces the predicted position
        new_p = new_v + last_obsv[:, :2]
        # The last prediction done will be new_p,new_v
        last_obsv = torch.cat([new_p, new_v], dim=1)
        # Keeps all the predictions
        pred_4ds.append(last_obsv)
        # Applies LSTM encoding to the last prediction
        # pred_4ds[-1]: batch_sizex4 tensor
        encoder(pred_4ds[-1])

    return torch.stack(pred_4ds, 1)


# ===================================================


# =============== Training Loop ==================
def train():
    tic = time.clock()
    # Evaluation metrics (ADE/FDE)
    train_ADE, train_FDE = 0, 0
    batch_size_accum = 0;
    sub_batches = []
    # For all the training batches
    for ii, batch_i in enumerate(train_batches):
        batch_size_accum += batch_i[1] - batch_i[0]
        sub_batches.append(batch_i)

        # FIXME: Just keep it for toy dataset
        # sub_batches = the_batches
        # batch_size_accum = sub_batches[-1][1]
        # ii = train_size-1

        if ii >= train_size - 1 or \
                batch_size_accum + (the_batches[ii + 1][1] - the_batches[ii + 1][0]) > batch_size:
            # Observed partial paths
            obsv = dataset_obsv[sub_batches[0][0]:sub_batches[-1][1]]
            # Future partial paths
            pred = dataset_pred[sub_batches[0][0]:sub_batches[-1][1]]
            sub_batches = sub_batches - sub_batches[0][0]
            # May have to fill with 0
            filling_len = batch_size - int(batch_size_accum)
            #obsv = torch.cat((obsv, torch.zeros(filling_len, n_past, 2).cuda()), dim=0)
            #pred = torch.cat((pred, torch.zeros(filling_len, n_next, 2).cuda()), dim=0)

            bs = batch_size_accum

            # Completes the positional vectors with velocities (to have dimension 4)
            obsv_4d, pred_4d = get_traj_4d(obsv, pred)
            zeros = Variable(torch.zeros(bs, 1) + np.random.uniform(0, 0.1), requires_grad=False).cuda()
            ones = Variable(torch.ones(bs, 1) * np.random.uniform(0.9, 1.0), requires_grad=False).cuda()
            noise = torch.FloatTensor(torch.rand(bs, noise_len)).cuda()

            # ============== Train Discriminator ================
            for u in range(n_unrolling_steps + 1):
                # Zero the gradient buffers of all parameters
                D.zero_grad()
                with torch.no_grad():
                    pred_hat_4d = predict(obsv, noise, n_next, sub_batches)

                fake_labels, code_hat = D(obsv_4d, pred_hat_4d)  # classify fake samples
                # Evaluate the MSE loss: the fake_labels should be close to zero
                d_loss_fake = mse_loss(fake_labels, zeros)
                d_loss_info = mse_loss(code_hat.squeeze(), noise[:, :n_latent_codes])
                # Evaluate the MSE loss: the real should be close to one
                real_labels, code_hat = D(obsv_4d, pred_4d)  # classify real samples
                d_loss_real = mse_loss(real_labels, ones)

                #  FIXME: which loss functinos to use for D?
                d_loss = d_loss_fake + d_loss_real
                if use_info_loss:
                    #
                    d_loss += loss_info_w * d_loss_info
                d_loss.backward()  # update D
                D_optimizer.step()

                if u == 0 and n_unrolling_steps > 0:
                    backup = copy.deepcopy(D)

            # =============== Train Generator ================= #
            # Zero the gradient buffers of all the discriminator parameters
            D.zero_grad()
            # Zero the gradient buffers of all the generator parameters
            predictor_optimizer.zero_grad()
            # Applies a forward step of prediction
            pred_hat_4d = predict(obsv, noise, n_next, sub_batches)

            # Classify the generated fake sample
            gen_labels, code_hat = D(obsv_4d, pred_hat_4d)
            # L2 loss between the predicted paths and the true ones
            g_loss_l2 = mse_loss(pred_hat_4d[:, :, :2], pred)
            # Adversarial loss (classification labels should be close to one)
            g_loss_fooling = mse_loss(gen_labels, ones)
            # Information loss
            g_loss_info = mse_loss(code_hat.squeeze(), noise[:, :n_latent_codes])

            #  FIXME: which loss functions to use for G?
            #
            g_loss = g_loss_fooling
            # If using the info loss
            if use_info_loss:
                g_loss += loss_info_w * g_loss_info
            # If using the L2 loss
            if use_l2_loss:
                g_loss += loss_l2_w * g_loss_l2
            if use_variety_loss:
                KV = 20
                all_20_losses = []
                for k in range(KV):
                    pred_hat_4d = predict(obsv, noise, n_next, sub_batches)
                    loss_l2_k = mse_loss(pred_hat_4d[k, :, :2], pred[k])
                all_20_losses.append(loss_l2_k.unsqueeze(0))
                all_20_losses = torch.cat(all_20_losses)
                variety_loss, _ = torch.min(all_20_losses, dim=0)
                g_loss += loss_l2_w * variety_loss

            g_loss.backward()
            predictor_optimizer.step()

            if n_unrolling_steps > 0:
                D.load(backup)
                del backup

            # calculate error
            with torch.no_grad():  # TODO: use the function above
                err_all = torch.pow((pred_hat_4d[:, :, :2] - pred) / ss, 2)
                err_all = err_all.sum(dim=2).sqrt()
                e = err_all.sum().item() / n_next
                train_ADE += e
                train_FDE += err_all[:, -1].sum().item()

            batch_size_accum = 0;
            sub_batches = []

    train_ADE /= n_train_samples
    train_FDE /= n_train_samples
    toc = time.clock()
    print(" Epc=%4d, Train ADE,FDE = (%.3f, %.3f) | time = %.1f" \
          % (epoch, train_ADE, train_FDE, toc - tic))


def test(n_gen_samples=20, linear=False, write_to_file=None, just_one=False):
    # =========== Test error ============
    plt.close()
    ade_avg_12, fde_avg_12 = 0, 0
    ade_min_12, fde_min_12 = 0, 0
    for ii, batch_i in enumerate(test_batches):        
        obsv = dataset_obsv[batch_i[0]:batch_i[1]]
        pred = dataset_pred[batch_i[0]:batch_i[1]]
        current_t = dataset_t[batch_i[0]]
        bs = int(batch_i[1] - batch_i[0])
        with torch.no_grad():
            all_20_errors = []
            all_20_preds = []

            linear_preds = predict_cv(obsv, n_next)
            if linear and not write_to_file:
                all_20_preds.append(linear_preds.unsqueeze(0))
                err_all = torch.pow((linear_preds[:, :, :2] - pred) / ss, 2).sum(dim=2, keepdim=True).sqrt()
                all_20_errors.append(err_all.unsqueeze(0))
            else:
                for kk in range(n_gen_samples):
                    noise = torch.FloatTensor(torch.rand(bs, noise_len)).cuda()
                    pred_hat_4d = predict(obsv, noise, n_next)
                    all_20_preds.append(pred_hat_4d.unsqueeze(0))
                    err_all = torch.pow((pred_hat_4d[:, :, :2] - pred) / ss, 2).sum(dim=2, keepdim=True).sqrt()
                    all_20_errors.append(err_all.unsqueeze(0))

            all_20_errors = torch.cat(all_20_errors)
            if write_to_file:
                file_name = os.path.join(write_to_file, str(epoch) + '-' + str(current_t) + '.npz')
                print('saving to ', file_name)
                np_obsvs = scale.denormalize(obsv[:, :, :2].data.cpu().numpy())
                np_preds_our = scale.denormalize(torch.cat(all_20_preds)[:, :, :, :2].data.cpu().numpy())
                np_preds_gtt = scale.denormalize(pred[:, :, :2].data.cpu().numpy())
                np_preds_lnr = scale.denormalize(linear_preds[:, :, :2].data.cpu().numpy())
                np.savez(file_name, timestamp=current_t,
                         obsvs=np_obsvs, preds_our=np_preds_our, preds_gtt=np_preds_gtt, preds_lnr=np_preds_lnr)

            # =============== Prediction Errors ================
            fde_min_12_i, _ = all_20_errors[:, :, -1].min(0, keepdim=True)
            ade_min_12_i, _ = all_20_errors.mean(2).min(0, keepdim=True)
            fde_min_12 += fde_min_12_i.sum().item()
            ade_min_12 += ade_min_12_i.sum().item()
            fde_avg_12 += all_20_errors[:, :, -1].mean(0, keepdim=True).sum().item()
            ade_avg_12 += all_20_errors.mean(2).mean(0, keepdim=True).sum().item()
            # ==================================================
        if just_one: break

    ade_avg_12 /= n_test_samples
    fde_avg_12 /= n_test_samples
    ade_min_12 /= n_test_samples
    fde_min_12 /= n_test_samples
    print('Avg ADE,FDE (12)= (%.3f, %.3f) | Min(20) ADE,FDE (12)= (%.3f, %.3f)' \
          % (ade_avg_12, fde_avg_12, ade_min_12, fde_min_12))


# =======================================================
# ===================== M A I N =========================
# =======================================================
if os.path.isfile(model_file):
    print('Loading model from ' + model_file)
    checkpoint = torch.load(model_file)
    start_epoch = checkpoint['epoch'] + 1

    attention.load_state_dict(checkpoint['attentioner_dict'])
    feature_embedder.load_state_dict(checkpoint['feature_embedder_dict'])
    encoder.load_state_dict(checkpoint['encoder_dict'])
    decoder.load_state_dict(checkpoint['decoder_dict'])
    predictor_optimizer.load_state_dict(checkpoint['pred_optimizer'])

    D.load_state_dict(checkpoint['D_dict'])
    D_optimizer.load_state_dict(checkpoint['D_optimizer'])
else:
    min_train_ADE = 10000
    start_epoch = 1

# FIXME: comment here to train
# wr_dir = '../preds-iccv/' + dataset_name + '/' + model_name + '/' + str(0000)
# os.makedirs(wr_dir, exist_ok=True)
# test(n_gen_samples=128, write_to_file=wr_dir)
# exit(1)

# ===================== TRAIN =========================
for epoch in trange(start_epoch, n_epochs + 1):  # FIXME : set the number of epochs
    # Main training function
    train()

    # ============== Save model on disk ===============
    if epoch % 50 == 0:  # FIXME : set the interval for running tests
        print('Saving model to file ...', model_file)
        torch.save({
            'epoch': epoch,
            'attentioner_dict': attention.state_dict(),
            'feature_embedder_dict': feature_embedder.state_dict(),
            'encoder_dict': encoder.state_dict(),
            'decoder_dict': decoder.state_dict(),
            'pred_optimizer': predictor_optimizer.state_dict(),

            'D_dict': D.state_dict(),
            'D_optimizer': D_optimizer.state_dict()
        }, model_file)

    if epoch % 5 == 0:
        wr_dir = '../medium/' + dataset_name + '/' + model_name + '/' + str(epoch)
        os.makedirs(wr_dir, exist_ok=True)
        test(128, write_to_file=wr_dir, just_one=True)
