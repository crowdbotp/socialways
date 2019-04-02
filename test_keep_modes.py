import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.parse_utils import BIWIParser, TrajnetParser, Scale, create_dataset

# ===== set input/output files ======
# inp_file = '../data/zara02/obsmat.txt'
# parser = BIWIParser()
# parser.load(inp_file, down_sample=1)
# p_data = parser.p_data


# FIXME : Select dataset by uncomment a pair
# trajnet_train_file = '../data/trajnet/train/stanford/bookstore_0.txt'
# trajnet_train_file = '../data/toy/toy-dataset.txt'
# trajnet_train_file = '../data/trajnet/train/mot/PETS09-S2L1.txt'
# trajnet_train_file = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/data/trajnet/train/stanford/hyang_5.txt'
trajnet_train_file = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/data/trajnet/train/stanford/hyang_6.txt'
# trajnet_train_file = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/data/trajnet/train/stanford/hyang_7.txt'
# trajnet_train_file = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/data/trajnet/train/stanford/hyang_9.txt'
# trajnet_train_file = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/data/trajnet/train/stanford/nexus_0.txt'
# trajnet_train_file = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/data/trajnet/train/stanford/nexus_7.txt'
# trajnet_train_file = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/data/trajnet/train/stanford/nexus_9.txt'


data_file = trajnet_train_file[:-3] + 'npz'
if not os.path.isfile(data_file):
    parser = TrajnetParser()
    parser.load(trajnet_train_file)
    p_data = parser.p_data

    t_range = range(int(parser.min_t), int(parser.max_t), 12)
    dataset_x, dataset_y, dataset_t, batches = create_dataset(parser.p_data, parser.t_data, t_range, n_past=8, n_next=12)
    np.savez(data_file, dataset_x=dataset_x, dataset_y=dataset_y, dataset_t=dataset_t)

dataset = np.load(data_file)
obsvs = dataset['dataset_x']
preds = dataset['dataset_y']
times = dataset['dataset_t']
samples = np.concatenate((obsvs, preds), axis=1)

selected_samples = []
selected_datafile = trajnet_train_file[:-4] + '_selected.npz'

my_obsvs = []
my_preds = []
min_obsv_len, min_pred_len = 20, 20
for ii in range(len(samples)):
    pi = samples[ii]
    # plt.plot(pi[:8, 0], pi[:8, 1], 'b', alpha=0.95)
    # plt.plot(pi[7:, 0], pi[7:, 1], 'r', alpha=0.55)
    # plt.plot(pi[-1, 0], pi[-1, 1], 'bx')

    plt.plot(pi[:, 0], pi[:, 1], 'y', alpha=0.55)
    plt.plot(pi[0, 0], pi[0, 1], 'b.')

    if -12 < pi[0, 0] < -5 and -7 < pi[0, 1] < -2:
        cur_obsv = []
        for tt in range(20):
            if pi[tt, 0] < -4:
                cur_obsv.append(pi[tt].reshape((1, 2)))
            else:
                break
        if tt < 2 or tt > 17:
            continue
        cur_obsv = np.concatenate(cur_obsv, axis=0)
        cur_pred = pi[tt:]
        min_obsv_len = min(len(cur_obsv), min_obsv_len)
        min_pred_len = min(len(cur_pred), min_pred_len)
        my_obsvs.append(cur_obsv)
        my_preds.append(cur_pred)
        plt.plot(my_obsvs[-1][:, 0], my_obsvs[-1][:, 1], 'r', alpha=0.95)
        plt.plot(my_preds[-1][:, 0], my_preds[-1][:, 1], 'g--')

        # plt.plot(pi[:, 0], pi[:, 1], 'r', alpha=0.95)
        # plt.plot(pi[0, 0], pi[0, 1], 'gx')
        # selected_samples.append(pi.reshape(1, 20, 2))

plt.show()

for ii in range(len(my_obsvs)):
    my_obsvs[ii] = my_obsvs[ii][len(my_obsvs[ii])-min_obsv_len:]
    my_preds[ii] = my_preds[ii][:min_pred_len]
    plt.plot(my_obsvs[ii][:, 0], my_obsvs[ii][:, 1], 'r', alpha=0.95)
    plt.plot(my_preds[ii][:, 0], my_preds[ii][:, 1], 'g', LineWidth=3)

my_obsvs = np.stack(my_obsvs)
my_preds = np.stack(my_preds)
N = len(my_obsvs)
# selected_samples = np.concatenate(selected_samples, axis=0)
np.savez(selected_datafile,
         obsvs=my_obsvs,
         preds=my_preds,
         times=np.zeros((N,1)),
         baches=np.array([[0, N]]))


plt.show()
