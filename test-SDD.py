from src.utils.parse_utils import SDD_Parsrer
import matplotlib.pyplot as plt
import numpy as np
import os

parser = SDD_Parsrer()
data_dir = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/data/annotations/gates/video3'

parser.load(os.path.join(data_dir, 'annotations.txt'))
p_data = parser.p_data

cntr = 0
selected_tracks = []
min_len = 12
for ii, p_data_i in enumerate(p_data):
    # if len(p_data_i) < 10: continue
    # dp_i = (p_data_i[10] - p_data_i[0]) / 10
    # T = len(p_data_i)

    # for tt, pp in enumerate(p_data_i):
    #     if 90 < pp[0] < 150 and 1050 < pp[1] < 2200 and dp_i[0] > 4:
    #         p_data_i = p_data_i[tt:]
    #         selected_tracks.append(p_data_i)
    #         min_len = min(min_len, len(p_data_i))
    #         break

    plt.plot(p_data_i[:, 0], p_data_i[:, 1], 'b')
    plt.plot(p_data_i[0, 0], p_data_i[0, 1], 'r*')

#
# for ii, p_data_i in enumerate(selected_tracks):
#     selected_tracks[ii] = p_data_i[:min_len]
#
# for ii, p_data_i in enumerate(selected_tracks):
#     plt.plot(p_data_i[:min_len, 0], p_data_i[:min_len, 1], 'g--.')
#     plt.plot(p_data_i[0, 0], p_data_i[0, 1], 'm*')
#     print(len(p_data_i))

plt.show()

# selected_tracks = np.array(selected_tracks)
# N = len(selected_tracks)
# print('N = ', N)
# n_past = 3
# np.savez(os.path.join(data_dir, 'data_2_8.npz'),
#          obsvs=selected_tracks[:, :n_past],
#          preds=selected_tracks[:, n_past:],
#          times=np.zeros((N,1)),
#          baches=np.array([[0, N]]))

