import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.utils.parse_utils import BIWIParser, TrajnetParser, Scale, create_dataset

# ===== set input/output files ======
# inp_file = '../data/zara02/obsmat.txt'
# parser = BIWIParser()
# parser.load(inp_file, down_sample=1)
# p_data = parser.p_data

# trajnet_train_file = '../data/trajnet/train/stanford/bookstore_0.txt'
# data_file = '../data/trajnet/train/stanford/bookstore_8_12.npz'


trajnet_train_file = '../data/toy/toy-dataset.txt'
data_file = '../data/toy/data.npz'
parser = TrajnetParser()
parser.load(trajnet_train_file)
p_data = parser.p_data

t_range = range(int(parser.min_t), int(parser.max_t), 1)
dataset_x, dataset_y, dataset_t = create_dataset(parser.p_data, parser.t_data, t_range, n_past=2, n_next=2)
np.savez(data_file, dataset_x=dataset_x, dataset_y=dataset_y, dataset_t=dataset_t)
exit(1)

print(len(p_data))

for ii in range(len(p_data)):
    pi = p_data[ii]
    plt.plot(pi[:, 1], pi[:, 0], 'r', alpha=0.15)
    plt.plot(pi[0, 1], pi[0, 0], 'o')
    plt.plot(pi[-1, 1], pi[-1, 0], 'x')

plt.show()
