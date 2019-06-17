from utils.parse_utils import SDD_Parsrer
import matplotlib.pyplot as plt
import numpy as np
import os

parser = SDD_Parsrer()
data_dir = '../data/SDD-all/gates/video2'

parser.load(os.path.join(data_dir, 'annotations.txt'))
p_data = parser.p_data

cntr = 0
selected_tracks = []
min_len = 12
for ii, p_data_i in enumerate(p_data):

    plt.plot(p_data_i[:, 0], p_data_i[:, 1], 'b')
    plt.plot(p_data_i[0, 0], p_data_i[0, 1], 'r*')

plt.show()


