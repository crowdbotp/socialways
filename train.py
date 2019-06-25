import os
import time
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


# ========== set input/output files ============
dataset_name = 'toy'  # FIXME: Notice to select the proper dataset
model_name = 'socialWays'
# processed_data_file = '../data/' + dataset_name +'/data.npz'
processed_data_file = '../data/eth/data_8_12.npz'
model_file = '../trained_models/' + model_name + '-' + dataset_name + '.pt'

# FIXME: ====== training hyper-parameters ======
# Unrolled GAN
n_unrolling_steps = 0
# Info GAN
use_info_loss = True
loss_info_w = 0.5
n_latent_codes = 2
# L2 GAN
use_l2_loss = False
use_variety_loss = False
loss_l2_w = 0.5  # WARNING for both l2 and variety
# Learning Rate
lr_g = 1E-4
lr_d = 1E-4
# FIXME: ====== Network Size ===================
batch_size = 128
hidden_size = 64
n_epochs = 100000
num_social_features = 3
social_feature_size = hidden_size // 2
noise_len = hidden_size // 2
n_lstm_layers = 1
# ==============================================

# FIXME: ======= Loda Data =====================
print(os.path.dirname(os.path.realpath(__file__)))

data = np.load(processed_data_file)
dataset_obsv, dataset_pred, dataset_t, the_batches = \
    data['obsvs'], data['preds'], data['times'], data['batches']
train_size = max(1, (len(the_batches) * 4) // 5)
train_batches = the_batches[:train_size]
test_batches = the_batches[train_size:]
n_past = dataset_obsv.shape[1]
n_next = dataset_pred.shape[1]
n_train_samples = the_batches[train_size-1][1]
n_test_samples = dataset_obsv.shape[0] - n_train_samples
if n_test_samples == 0:
    n_test_samples = 1
    the_batches = np.array([the_batches[0], the_batches[0]])
print(processed_data_file, ' # Training samples: ', n_train_samples)

# normalize
scale = Scale()
scale.max_x = max(np.max(dataset_obsv[:, :, 0]), np.max(dataset_pred[:, :, 0]))
scale.min_x = min(np.min(dataset_obsv[:, :, 0]), np.min(dataset_pred[:, :, 0]))
scale.max_y = max(np.max(dataset_obsv[:, :, 1]), np.max(dataset_pred[:, :, 1]))
scale.min_y = min(np.min(dataset_obsv[:, :, 1]), np.min(dataset_pred[:, :, 1]))
scale.calc_scale(keep_ratio=True)
dataset_obsv = scale.normalize(dataset_obsv)
dataset_pred = scale.normalize(dataset_pred)
ss = scale.sx

# Normalize
dataset_obsv = torch.FloatTensor(dataset_obsv).cuda()
dataset_pred = torch.FloatTensor(dataset_pred).cuda()
# ================================================


def get_traj_4d(obsv_p, pred_p):
    obsv_v = obsv_p[:, 1:] - obsv_p[:, :-1]
    obsv_v = torch.cat([obsv_v[:, 0].unsqueeze(1), obsv_v], dim=1)
    obsv_4d = torch.cat([obsv_p, obsv_v], dim=2)
    if len(pred_p) == 0: return obsv_4d
    pred_p_1 = torch.cat([obsv_p[:, -1].unsqueeze(1), pred_p[:, :-1]], dim=1)
    pred_v = pred_p - pred_p_1
    pred_4d = torch.cat([pred_p, pred_v], dim=2)
    return obsv_4d, pred_4d


def calc_error(pred_hat, pred):
    N = pred.size(0)
    T = pred.size(1)
    err_all = torch.pow((pred_hat - pred)/ss, 2).sum(dim=2).sqrt()  # N x T
    FDEs = err_all.sum(dim=0).item() / N
    ADEs = torch.cumsum(FDEs)
    for ii in range(T):
        ADEs[ii] /= (ii+1)
    return ADEs.data.cpu().numpy(), FDEs.data().cpu().numpy()


class AttentionPooling(nn.Module):
    def __init__(self):
        super(AttentionPooling, self).__init__()

    def forward(self, f, h, sub_batches=[]):
        bs = h.shape[0]

        # attens = torch.zeros(bs, bs).cuda()
        # sigma = torch.dot(f, w * h) * (N-1) / ...


        # for sb in sub_batches:
        #     x_sb = x[sb[0]:sb[1]]
        #     for ped in range(sb[0], sb[1]):
        #         others = [xx for j, xx in enumerate(x_sb) if j != (ped - sb[0])]
        #         POI_x = x_sb[ped-sb[0]]
        #         if not others: continue
        #         others = torch.stack(others)
        #         dx = POI_x.unsqueeze(0) - others
        #
        #     attn = torch.mm(1, 1)
        #     attn = attn - attn.max(1)[0]
        #     exp_attn = torch.exp(attn).view(sb_len, sb_len)
        #     attens[sb[0]:sb[1], sb[0]:sb[1]] = exp_attn / (exp_attn.sum(1).unsqueeze(1) + 10E-8)

        return attens


class EmbedSocialFeatures(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(EmbedSocialFeatures, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.fc3 = nn.Linear(64, hidden_size)

    def forward(self, ftr_list, sub_batches):
        embedded_features = []
        for ftrs in ftr_list:
            emb_i = []
            for kk in range(len(ftrs)):
                h = self.fc1(ftrs[kk].unsqueeze(0))
                h = self.fc2(h)
                h = self.fc3(h)
                emb_i.append(h)
            embedded_features.append(emb_i)

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


def SocialFeatures(x, sub_batches):
    if len(sub_batches) == 0:
        sub_batches = [[0, x.shape[0]]]
    social_features = [] # x.shape[0]*[None]
    for sb in sub_batches:
        if sb[1] - sb[0] == 1:
            social_features.append([])
            continue
        for ii in range(sb[0], sb[1]):
            xi = x[ii, -1]
            features_i = []
            for jj in range(sb[0], sb[1]):
                if ii == jj: continue
                xj = x[jj, -1]
                distance = (xi - xj).norm()
                bearing = Bearing(xi, xj)
                dca = DCA(xi, xj)
                feature_list = [distance, bearing, dca]
                features_i.append(torch.stack(feature_list))
            social_features.append(features_i)

    return social_features


class EncoderLstm(nn.Module):
    def __init__(self, hidden_size, n_layers=2):
        self.hidden_size = hidden_size
        super(EncoderLstm, self).__init__()
        self.embed = nn.Linear(4, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=n_layers, batch_first=True)
        self.lstm_h = []
        # init_weights(self)

    def init_lstm(self, h, c):
        self.lstm_h = (h, c)

    def forward(self, obsv):
        bs = obsv.shape[0]
        obsv = self.embed(obsv)
        y, self.lstm_h = self.lstm(obsv.view(bs, -1, self.hidden_size), self.lstm_h)
        return y


class Discriminator(nn.Module):
    def __init__(self, n_next, hidden_dim, n_latent_code):
        super(Discriminator, self).__init__()
        self.lstm_dim = hidden_dim
        self.n_next = n_next
        self.obsv_encoder_lstm = nn.LSTM(4, hidden_dim, batch_first=True)
        self.obsv_encoder_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.LeakyReLU(0.2),
                                             nn.Linear(hidden_dim//2, hidden_dim//2))
        self.pred_encoder = nn.Sequential(nn.Linear(n_next*4, hidden_dim//2), nn.LeakyReLU(0.2),
                                          nn.Linear(hidden_dim//2, hidden_dim//2))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.LeakyReLU(0.2),
                                        nn.Linear(hidden_dim//2, 1))

        self.latent_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.LeakyReLU(0.2),
                                        nn.Linear(self.lstm_dim//2, n_latent_code))

    def forward(self, obsv, pred, sub_batches=[]):
        bs = obsv.size(0)
        lstm_h_c = (torch.zeros(1, bs, self.lstm_dim).cuda(),
                    torch.zeros(1, bs, self.lstm_dim).cuda())
        obsv_code, lstm_h_c = self.obsv_encoder_lstm(obsv, lstm_h_c)
        obsv_code = self.obsv_encoder_fc(obsv_code[:, -1])
        pred_code = self.pred_encoder(pred.view(-1, self.n_next*4))
        both_codes = torch.cat([obsv_code, pred_code], dim=1)
        label = self.classifier(both_codes)
        code_hat = self.latent_decoder(both_codes)
        return label, code_hat

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


class DecoderFC(nn.Module):
    def __init__(self, hidden_dim):
        super(DecoderFC, self).__init__()
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                                       # torch.nn.Linear(64, 64), nn.LeakyReLU(0.2),
                                       torch.nn.Linear(hidden_dim, hidden_dim//2), nn.LeakyReLU(0.2),
                                       torch.nn.Linear(hidden_dim//2, hidden_dim//4),
                                       torch.nn.Linear(hidden_dim//4, 2))

    def forward(self, coded_tracks, social_codes, noise, sub_batches=[]):
        inp = torch.cat([coded_tracks, social_codes, noise], dim=1)
        out = self.fc1(inp)
        return out


class DecoderLstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DecoderLstm, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(torch.nn.Linear(hidden_size, 64), nn.Sigmoid(),
                                torch.nn.Linear(64, 64), nn.LeakyReLU(0.2),
                                torch.nn.Linear(64, 32), nn.LeakyReLU(0.2),
                                torch.nn.Linear(32,  2))

        # init_weights(self)
        self.lstm_h = []

    def init_lstm(self, h, c):
        self.lstm_h = (h, c)

    def forward(self, coded_tracks, social_codes, noise, sub_batches=[]):
        bs = noise.shape[0]
        # inp = torch.cat([social_codes, noise], dim=1)
        # inp = torch.FloatTensor(torch.zeros(bs, 1, hidden_size)).cuda()
        inp = torch.cat([social_codes, noise], dim=1)
        out, self.lstm_h = self.lstm(inp.unsqueeze(1), self.lstm_h)
        out = self.fc(out.squeeze())
        return out


encoder = EncoderLstm(hidden_size, n_lstm_layers).cuda()
attention = AttentionPooling().cuda()
feature_embedder = EmbedSocialFeatures(num_social_features, social_feature_size).cuda()

decoder = DecoderFC(hidden_size + social_feature_size + noise_len).cuda()
# decoder = DecoderLstm(social_feature_size + VEL_VEC_LEN + noise_len, traj_code_len).cuda()

predictor_params = chain(attention.parameters(), encoder.parameters(), decoder.parameters())
predictor_optimizer = opt.Adam(predictor_params, lr=lr_g, betas=(0.8, 0.999))

D = Discriminator(n_next, hidden_size, n_latent_codes).cuda()
D_optimizer = opt.Adam(D.parameters(), lr=lr_d, betas=(0.8, 0.999))
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

print('hidden dim = %d | lr(G) =  %.5f | lr(D) =  %.5f' % (hidden_size, lr_g, lr_d))


def predict(obsv_p, noise, n_next, sub_batches=[]):
    bs = obsv_p.shape[0]
    obsv_4d = get_traj_4d(obsv_p, [])

    # Run Obsv-Encoder
    lstm_h_c = (torch.zeros(n_lstm_layers, bs, encoder.hidden_size).cuda(),
                torch.zeros(n_lstm_layers, bs, encoder.hidden_size).cuda())
    encoder.init_lstm(lstm_h_c[0], lstm_h_c[1])
    encoder(obsv_4d)

    features = SocialFeatures(obsv_4d, sub_batches)
    emb_features = feature_embedder(features, sub_batches)
    attn = attention(emb_features, encoder.lstm_h[0])

    emb_sfeatures = torch.FloatTensor(torch.zeros(bs, feature_embedder.hidden_size)).cuda()

    pred_4ds = []
    last_obsv = obsv_4d[:, -1]
    for ii in range(n_next):
        new_v = decoder(encoder.lstm_h[0].view(bs, -1), emb_sfeatures, noise).view(bs, 2) # + last_obsv[:, 2:]
        new_p = new_v + last_obsv[:, :2]
        last_obsv = torch.cat([new_p, new_v], dim=1)
        pred_4ds.append(last_obsv)
        encoder(pred_4ds[-1])

    return torch.stack(pred_4ds, 1)
# ===================================================


# =============== Training Loop ==================
def train():
    tic = time.clock()
    train_ADE, train_FDE = 0, 0
    batch_size_accum = 0; sub_batches = []
    for ii, batch_i in enumerate(train_batches):
        batch_size_accum += batch_i[1] - batch_i[0]
        sub_batches.append(batch_i)

        # FIXME: Just keep it for toy dataset
        # sub_batches = the_batches
        # batch_size_accum = sub_batches[-1][1]
        # ii = train_size-1

        if ii >= train_size-1 or \
                batch_size_accum + (the_batches[ii+1][1] - the_batches[ii+1][0]) > batch_size:
            obsv = dataset_obsv[sub_batches[0][0]:sub_batches[-1][1]]
            pred = dataset_pred[sub_batches[0][0]:sub_batches[-1][1]]
            sub_batches = sub_batches - sub_batches[0][0]
            filling_len = batch_size - int(batch_size_accum)
            obsv = torch.cat((obsv, torch.zeros(filling_len, n_past, 2).cuda()), dim=0)
            pred = torch.cat((pred, torch.zeros(filling_len, n_next, 2).cuda()), dim=0)

            obsv_4d, pred_4d = get_traj_4d(obsv, pred)
            zeros = Variable(torch.zeros(batch_size, 1) + np.random.uniform(0, 0.1), requires_grad=False).cuda()
            ones = Variable(torch.ones(batch_size, 1) * np.random.uniform(0.9, 1.0), requires_grad=False).cuda()
            noise = torch.FloatTensor(torch.rand(batch_size, noise_len)).cuda()

            # ============== Train Discriminator ================
            for u in range(n_unrolling_steps + 1):
                D.zero_grad()
                with torch.no_grad():
                    pred_hat_4d = predict(obsv, noise, n_next, sub_batches)

                fake_labels, code_hat = D(obsv_4d, pred_hat_4d)  # classify fake samples
                d_loss_fake = mse_loss(fake_labels, zeros)
                d_loss_info = mse_loss(code_hat.squeeze(), noise[:, :n_latent_codes])

                real_labels, code_hat = D(obsv_4d, pred_4d)  # classify real samples
                d_loss_real = mse_loss(real_labels, ones)

                #  FIXME: which loss functinos to use for D?
                d_loss = d_loss_fake + d_loss_real
                if use_info_loss:
                    d_loss += loss_info_w * d_loss_info
                d_loss.backward()  # update D
                D_optimizer.step()

                if u == 0 and n_unrolling_steps > 0:
                    backup = copy.deepcopy(D)

            # =============== Train Generator ================= #
            D.zero_grad()
            predictor_optimizer.zero_grad()
            pred_hat_4d = predict(obsv, noise, n_next, sub_batches)

            gen_labels, code_hat = D(obsv_4d, pred_hat_4d)  # classify a fake sample
            g_loss_l2 = mse_loss(pred_hat_4d[:, :, :2], pred)
            g_loss_fooling = mse_loss(gen_labels, ones)
            g_loss_info = mse_loss(code_hat.squeeze(), noise[:, :n_latent_codes])

            #  FIXME: which loss functinos to use for G?
            g_loss = g_loss_fooling
            if use_info_loss:
                g_loss += loss_info_w * g_loss_info
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
                err_all = torch.pow((pred_hat_4d[:, :, :2] - pred)/ss, 2)
                err_all = err_all.sum(dim=2).sqrt()
                e = err_all.sum().item() / n_next
                train_ADE += e
                train_FDE += err_all[:, -1].sum().item()

            batch_size_accum = 0; sub_batches = []

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
        if ii > 20: break
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
                err_all = torch.pow((linear_preds[:, :, :2] - pred)/ss, 2).sum(dim=2, keepdim=True).sqrt()
                all_20_errors.append(err_all.unsqueeze(0))
            else:
                for kk in range(n_gen_samples):
                    noise = torch.FloatTensor(torch.rand(bs, noise_len)).cuda()
                    pred_hat_4d = predict(obsv, noise, n_next)
                    all_20_preds.append(pred_hat_4d.unsqueeze(0))
                    err_all = torch.pow((pred_hat_4d[:, :, :2] - pred)/ss, 2).sum(dim=2, keepdim=True).sqrt()
                    all_20_errors.append(err_all.unsqueeze(0))

            all_20_errors = torch.cat(all_20_errors)
            if write_to_file:
                file_name = os.path.join(write_to_file, str(epoch) + '-' + str(current_t) + '.npz')
                print('saving to ', file_name)
                np_obsvs = scale.denormalize(obsv[:,:,:2].data.cpu().numpy())
                np_preds_our = scale.denormalize(torch.cat(all_20_preds)[:,:,:,:2].data.cpu().numpy())
                np_preds_gtt = scale.denormalize(pred[:,:,:2].data.cpu().numpy())
                np_preds_lnr = scale.denormalize(linear_preds[:,:,:2].data.cpu().numpy())
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
    print('loading from ' + model_file)
    checkpoint = torch.load(model_file)
    start_epoch = checkpoint['epoch'] + 1
    attention.load_state_dict(checkpoint['attentioner_dict'])
    encoder.load_state_dict(checkpoint['lstm_encoder_dict'])
    decoder.load_state_dict(checkpoint['predictor_dict'])
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
    train()

    # ============== Save model on disk ===============
    if epoch % 50 == 0:  # FIXME : set the interval for running tests
        print('Saving model to file ...', model_file)
        torch.save({
            'epoch': epoch,
            'attentioner_dict': attention.state_dict(),
            'lstm_encoder_dict': encoder.state_dict(),
            'predictor_dict': decoder.state_dict(),
            'pred_optimizer': predictor_optimizer.state_dict(),

            'D_dict': D.state_dict(),
            'D_optimizer': D_optimizer.state_dict()
        }, model_file)

    if epoch % 5 == 0:
        wr_dir = '../medium/' + dataset_name + '/' + model_name + '/' + str(epoch)
        os.makedirs(wr_dir, exist_ok=True)
        test(128, write_to_file=wr_dir, just_one=True)

