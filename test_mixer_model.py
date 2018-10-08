import pickle
import numpy as np
from astropy.wcs.docstrings import mix
from keras.engine.saving import model_from_json
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from lstm_model.my_base_classes import Scale, MyConfig, ConstVelModel
from lstm_model.my_base_classes import to_supervised, load_seyfried
from tabulate import tabulate

np.random.seed(7)
data_arrays, scale = load_seyfried()
np.random.shuffle(data_arrays)
n_ped = len(data_arrays)

# Load LSTM model
model_name = "models/mixer_model_3"
json_file = open(model_name + ".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
mixer_model = model_from_json(loaded_model_json)
# load weights into new model
# mixer_model.load_weights(model_name + ".h5")
mixer_model.load_weights(model_name + "-best-weights.hdf5")
print("Loaded model from file")

# Const-Velocity Model
cv_model = ConstVelModel()


# Normalize Data between [0,1]
for i in range(len(data_arrays)):
    data_arrays[i] = scale.normalize(data_arrays[i])
data_set = np.array(data_arrays)

# Convert Time-series to supervised learning format
n_past = MyConfig().n_past
n_next = MyConfig().n_next

pos_inp_data = np.empty((0, 2 * n_past))
vel_inp_data = np.empty((0, 2 * (n_past-1)))
disp_out_data = np.empty((0, 2 * n_next))
ped_index = list()
ind = -1;
for ped in data_set:
    ind += 1
    ped_pos_seqs = np.array(to_supervised(ped, n_past, n_next, False))
    ped_vel_seqs = np.array(to_supervised(ped, n_past-1, n_next, True))
    for i in range(ped_pos_seqs.shape[0]):
        pos_inp_data = np.vstack((pos_inp_data, ped_pos_seqs[i, 0:2 * n_past]))
        vel_inp_data = np.vstack((vel_inp_data, ped_vel_seqs[i, 0:2 * (n_past - 1)]))
        disp_out_data = np.vstack((disp_out_data, ped_pos_seqs[i, 2 * n_past:2 * (n_past + n_next)]))
        ped_index.append(ind)

n_row = pos_inp_data.shape[0]
train_size = int(n_row * 0.8)
test_size = n_row - train_size

# Create Train Data
train_in_pos = pos_inp_data[0:train_size, :].reshape((train_size, n_past, 2))
train_in_vel = vel_inp_data[0:train_size, :].reshape((train_size, n_past-1, 2))
train_out = disp_out_data[0:train_size, :].reshape((train_size, n_next, 2))

# Create Test Data
test_in_pos = pos_inp_data[train_size:n_row, :].reshape(test_size, n_past, 2)
test_in_vel = vel_inp_data[train_size:n_row, :].reshape(test_size, n_past-1, 2)
test_out_gt = disp_out_data[train_size:n_row, :].reshape(test_size, n_next, 2)

mixer_test_out = mixer_model.predict([test_in_pos, test_in_vel])

tot_train_ade_err_cv = 0
tot_train_ade_err_lstm = 0
tot_train_fde_err_cv = 0
tot_train_fde_err_lstm = 0
tot_train_cnt = 0
for k in range(0, test_size, n_next):
    gt_k = scale.denormalize(test_out_gt[k, :, :])
    cv_k = scale.denormalize(cv_model.predict(test_in_pos[k, :, :]) - test_in_pos[k, n_past-1, :])
    mixer_k = scale.denormalize(mixer_test_out[k, :, :])

    cv_diff_norm = np.linalg.norm(cv_k - gt_k, 2, 1)
    tot_train_ade_err_cv += cv_diff_norm.sum(0)
    lstm_diff_norm = np.linalg.norm(mixer_k - gt_k, 2, 1)
    tot_train_ade_err_lstm += lstm_diff_norm.sum(0)

    tot_train_fde_err_cv += cv_diff_norm[n_next-1]
    tot_train_fde_err_lstm += lstm_diff_norm[n_next-1]

    tot_train_cnt += 1

    # plt.plot(gt_k[:, 0], gt_k[:, 1], 'g', label='gt')
    # plt.plot(cv_k[:, 0], cv_k[:, 1], 'b', label='cv')
    # plt.plot(lstm_k[:, 0], lstm_k[:, 1], 'r', label='lstm')
    # plt.legend()
    # plt.show()

mixer_ade = tot_train_ade_err_lstm / (tot_train_cnt * n_next)
mixer_fde = tot_train_fde_err_lstm / tot_train_cnt
const_vel_ade = tot_train_ade_err_cv / (tot_train_cnt * n_next)
const_vel_fde = tot_train_fde_err_cv / tot_train_cnt

print('***********************************')
print('Results on Test set of Seyfried1:')
print('***********************************')
print(tabulate([['Mixer Lstm', mixer_ade, mixer_fde], ['Const-Vel', const_vel_ade, const_vel_fde]],
               headers=['Method', 'ADE Error', 'FDE Error']))

first_id = ped_index[train_size]+1
for i in range(first_id, n_ped):
    indices = [j for j, x in enumerate(ped_index) if x == i]

    w, h = figaspect(4 / 1)
    fig, ax = plt.subplots(figsize=(w, h))

    for j in indices[::n_next]:
        gt_inp = test_in_pos[j-train_size, :, :]
        gt_out = test_out_gt[j-train_size, :, :]
        mixer_out = mixer_test_out[j-train_size, :, :]

        # LSTM Input
        lstm_inp = np.vstack((gt_inp[:, 0], gt_inp[:, 1]))
        lstm_inp = scale.denormalize(lstm_inp.transpose())

        if j == indices[0]:
            plt.plot(lstm_inp[0, 0], lstm_inp[0, 1], 'mo', markersize=7, label='Start Point')
            plt.plot(lstm_inp[:, 0], lstm_inp[:, 1], 'y--', label='LSTM Input (GT)')
        else:
            plt.plot(lstm_inp[:, 0], lstm_inp[:, 1], 'y--')

        orig_k = gt_inp[n_past - 1, :]
        orig_k_descaled = scale.denormalize(np.array(orig_k).transpose())
        plt.plot(orig_k_descaled[0], orig_k_descaled[1], 'mo', markersize=4)

        cv_output = cv_model.predict(lstm_inp)
        if j == indices[0]:
            plt.plot(cv_output[:, 0], cv_output[:, 1], 'r', label='Const Vel')
        else:
            plt.plot(cv_output[:, 0], cv_output[:, 1], 'r')

        mixer_out = np.vstack((mixer_out[:, 0] + orig_k[0], mixer_out[:, 1] + orig_k[1]))
        mixer_out = scale.denormalize(mixer_out.transpose())
        if j == indices[0]:
            plt.plot(mixer_out[:, 0], mixer_out[:, 1], 'b', label='LSTM Output')
        else:
            plt.plot(mixer_out[:, 0], mixer_out[:, 1], 'b')

        gt_ped_x = gt_out[:, 0] + orig_k[0]
        gt_ped_y = gt_out[:, 1] + orig_k[1]
        gt_ped = np.vstack((gt_ped_x, gt_ped_y)).transpose()
        gt_ped = scale.denormalize(gt_ped)
        if j == indices[0]:
            plt.plot(gt_ped[:, 0], gt_ped[:, 1], 'g--', label='Ground Truth Out')
        else:
            plt.plot(gt_ped[:, 0], gt_ped[:, 1], 'g--')

    plt.legend()
    plt.show()

