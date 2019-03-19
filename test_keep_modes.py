import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.utils.parse_utils import BIWIParser, Scale, create_dataset


# ===== set input/output files ======
csv_file = '../data/zara02/obsmat.txt'

parser = BIWIParser()
parser.load(csv_file, down_sample=1)
p_data = parser.p_data

print(len(p_data))

for ii in range(len(p_data)):
    pi = p_data[ii]
    plt.plot(pi[:, 1], pi[:, 0], 'r')
    plt.plot(pi[0, 1], pi[0, 0], 'o')
    plt.plot(pi[-1, 1], pi[-1, 0], 'x')

plt.show()
