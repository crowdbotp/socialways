import numpy as np
from keras.engine.saving import model_from_json
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from lstm_model.utility import Scale, MyConfig, ConstVelModel, SeyfriedParser
from lstm_model.utility import to_supervised, load
from tabulate import tabulate

# Load LSTM model
model_name = "models/model1"
json_file = open(model_name + ".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
lstm_model = model_from_json(loaded_model_json)
lstm_model.load_weights(model_name + ".h5")
print("Loaded model from file")

# Const-Velocity Model
cv_model = ConstVelModel()

np.random.seed(7)
parser = SeyfriedParser()
pos_data, vel_data, time_data = parser.load('/home/jamirian/workspace/crowd_sim/tests/sey01/sey01.sey')
scale = parser.scale
n_ped = len(pos_data)

train_size = int(n_ped * 0.67)
test_size = n_ped - train_size

# with open('scale.pkl', 'wb') as scale_file:
#     pickle.dump(scale, scale_file, pickle.HIGHEST_PROTOCOL)

# Normalize Data between [0,1]
for i in range(len(pos_data)):
    pos_data[i] = scale.normalize(pos_data[i])
data_set = np.array(pos_data)

train = data_set[0:train_size]
test = data_set[train_size + 1:len(data_set)]
# test = train

n_past = MyConfig().n_past
n_next = MyConfig().n_next
n_X, n_Y = 2 * n_past, 2 * n_next
n_XIY = 2 * (n_past + n_next)


test_set = np.array([], dtype=np.float).reshape(0, n_XIY)
for ped in test:
    seq_i = np.array(to_supervised(ped, n_past, n_next))
    for i in range(seq_i.shape[0]):
        test_set = np.vstack((test_set, seq_i[i, :]))
n_test = test_set.shape[0]
test_inp = test_set[:, 0:n_X].reshape((n_test, n_past, 2))
test_out = test_set[:, n_X:n_XIY].reshape((n_test, n_next, 2))


# lstm_train_out = lstm_model.predict(train_inp)
# train_diff = lstm_train_out - train_out

lstm_test_out = lstm_model.predict(test_inp)

tot_train_ade_err_cv = 0
tot_train_ade_err_lstm = 0
tot_train_fde_err_cv = 0
tot_train_fde_err_lstm = 0
tot_train_cnt = 0
for k in range(0, n_test, n_next):
    gt_k = scale.denormalize(test_out[k, :, :])
    cv_k = scale.denormalize(cv_model.predict(test_inp[k, :, :]) - test_inp[k, n_past-1, :])
    lstm_k = scale.denormalize(lstm_test_out[k, :, :])

    cv_diff_norm = np.linalg.norm(cv_k - gt_k, 2, 1)
    tot_train_ade_err_cv += cv_diff_norm.sum(0)
    lstm_diff_norm = np.linalg.norm(lstm_k - gt_k, 2, 1)
    tot_train_ade_err_lstm += lstm_diff_norm.sum(0)

    tot_train_fde_err_cv += cv_diff_norm[n_next-1]
    tot_train_fde_err_lstm += lstm_diff_norm[n_next-1]

    tot_train_cnt += 1

    # plt.plot(gt_k[:, 0], gt_k[:, 1], 'g', label='gt')
    # plt.plot(cv_k[:, 0], cv_k[:, 1], 'b', label='cv')
    # plt.plot(lstm_k[:, 0], lstm_k[:, 1], 'r', label='lstm')
    # plt.legend()
    # plt.show()

lstm_ade = tot_train_ade_err_lstm / (tot_train_cnt * n_next)
lstm_fde = tot_train_fde_err_lstm / tot_train_cnt
cv_ade = tot_train_ade_err_cv / (tot_train_cnt * n_next)
cv_fde = tot_train_fde_err_cv / tot_train_cnt

print('***********************************')
print('Results on Test set of Seyfried1:')
print('***********************************')
print(tabulate([['Lstm', lstm_ade, lstm_fde], ['cv', cv_ade, cv_fde]],
               headers=['Method', 'ADE Error', 'FDE Error']))

# Plot Results
exit(1)

for i in range(len(test)):
    ped_i = np.array(to_supervised(test[i], n_past, n_next).values)
    num_ped_samples = ped_i.shape[0]
    gt_inp = ped_i[:, 0:n_X].reshape((num_ped_samples, n_past, 2))
    gt_out = ped_i[:, n_X:n_XIY].reshape((num_ped_samples, n_next * 2))
    print(gt_inp.shape)

    lstm_prediction = lstm_model.predict(gt_inp)
    gt_out = np.reshape(gt_out, (num_ped_samples, n_next, 2))

    w, h = figaspect(4 / 1)
    fig, ax = plt.subplots(figsize=(w, h))

    for k in range(0, num_ped_samples, n_next):
        # LSTM Input
        lstm_inp = np.vstack((gt_inp[k, :, 0], gt_inp[k, :, 1]))
        lstm_inp = scale.denormalize(lstm_inp.transpose())

        if k == 0:
            plt.plot(lstm_inp[0, 0], lstm_inp[0, 1], 'mo', markersize=7, label='Start Point')
            plt.plot(lstm_inp[:, 0], lstm_inp[:, 1], 'y--', label='LSTM Input (GT)')
        else:
            plt.plot(lstm_inp[:, 0], lstm_inp[:, 1], 'y--')

        orig_k = gt_inp[k, n_past - 1, :]
        orig_k_descaled = scale.denormalize(np.array(orig_k).transpose())
        plt.plot(orig_k_descaled[0], orig_k_descaled[1], 'mo', markersize=4)

        cv_output = cv_model.predict(lstm_inp)
        if k == 0:
            plt.plot(cv_output[:, 0], cv_output[:, 1], 'r', label='Const Vel')
        else:
            plt.plot(cv_output[:, 0], cv_output[:, 1], 'r')

        lstm_out = np.vstack((lstm_prediction[k, :, 0] + orig_k[0], lstm_prediction[k, :, 1] + orig_k[1]))
        lstm_out = scale.denormalize(lstm_out.transpose())
        if k == 0:
            plt.plot(lstm_out[:, 0], lstm_out[:, 1], 'b', label='LSTM Output')
        else:
            plt.plot(lstm_out[:, 0], lstm_out[:, 1], 'b')

        gt_ped_x = gt_out[k, :, 0] + orig_k[0]
        gt_ped_y = gt_out[k, :, 1] + orig_k[1]
        gt_ped = np.vstack((gt_ped_x, gt_ped_y)).transpose()
        gt_ped = scale.denormalize(gt_ped)
        if k == 0:
            plt.plot(gt_ped[:, 0], gt_ped[:, 1], 'g--', label='Ground Truth Out')
        else:
            plt.plot(gt_ped[:, 0], gt_ped[:, 1], 'g--')

    plt.legend()
    plt.show()

