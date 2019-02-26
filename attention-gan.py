import os
import numpy as np
import torch
import torch.nn as nn
from .models import *

# ====== Hyper-parameters ======
lr = 10E-4
beta1 = 0.5
beta2 = 0.999
nX = 8
nY = 12
nZ = 32


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        hidden_size = 64
        self.obsv_encoder = nn.Sequential(fc_block(nX * 2, hidden_size),
                                          fc_block(hidden_size, hidden_size, False)).cuda()
        self.noise_encoder = fc_block(nZ, hidden_size).cuda()
        self.fc_out = nn.Sequential(fc_block(hidden_size * 3, hidden_size),
                                    fc_block(hidden_size, hidden_size),
                                    fc_block(hidden_size, hidden_size),
                                    # fc_block(hidden_size, hidden_size),
                                    nn.Linear(hidden_size, nY * 2)).cuda()

    def forward(self, obs, z=np.empty()):
        if np.size(z) == 0:
            z = torch.random()
        return


class Discriminator(nn.Module):
    def __init__(self):
        pass

    def forward(self, *input):
        pass




G = Generator().cuda()
D = Discriminator().cuda()
optimizer_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

# load data
out_dir = "../gan_out"
os.makedirs(out_dir, exist_ok=True)

def train():
    # train D
    # train G
    # roll back D
    pass

