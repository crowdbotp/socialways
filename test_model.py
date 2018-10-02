import pickle
import numpy as np
from keras.engine.saving import model_from_json
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from lstm_model.my_base_classes import Scale, MyConfig, ConstVelModel
from lstm_model.my_base_classes import to_supervised, load_seyfried

np.random.seed(7)
data_arrays, scale = load_seyfried()
n_ped = len(data_arrays)

train_size = int(n_ped * 0.67)
test_size = n_ped - train_size

# with open('scale.pkl', 'wb') as scale_file:
#     pickle.dump(scale, scale_file, pickle.HIGHEST_PROTOCOL)

# Normalize Data between [0,1]
for i in range(len(data_arrays)):
    data_arrays[i] = scale.normalize(data_arrays[i])
data_set = np.array(data_arrays)

train = data_set[0:train_size]

test = data_set[train_size + 1:len(data_set)]
# test = train

n_past = MyConfig().n_past
n_next = MyConfig().n_next
n_X, n_Y = 2 * n_past, 2 * n_next
n_XIY = 2 * (n_past + n_next)

test_set = np.array([], dtype=np.float).reshape(0, n_XIY)

n_test = len(test)

# Load model
model_name = "models/model1"
json_file = open(model_name + ".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
lstm_model = model_from_json(loaded_model_json)
# load weights into new model
lstm_model.load_weights(model_name + ".h5")
print("Loaded model from file")

cv_model = ConstVelModel()

start = 0
for i in range(len(test)):
    test_ped_i = np.array(to_supervised(test[i], n_past, n_next).values)
    num_ped_samples = test_ped_i.shape[0]
    test_Inp = test_ped_i[:, 0:n_X].reshape((num_ped_samples, n_past, 2))
    test_Out = test_ped_i[:, n_X:n_XIY].reshape((num_ped_samples, n_next * 2))
    print(test_Inp.shape)

    lstm_prediction = lstm_model.predict(test_Inp)
    lstm_prediction = np.reshape(lstm_prediction, (num_ped_samples, n_next, 2))

    test_Out = np.reshape(test_Out, (num_ped_samples, n_next, 2))

    w, h = figaspect(4 / 1)
    fig, ax = plt.subplots(figsize=(w, h))

    for k in range(0, num_ped_samples, n_next):
        # LSTM Input
        lstm_inp = np.vstack((test_Inp[k, :, 0], test_Inp[k, :, 1]))
        lstm_inp = scale.denormalize(lstm_inp.transpose())

        if k == 0:
            plt.plot(lstm_inp[0, 0], lstm_inp[0, 1], 'mo', markersize=7, label='Start Point')
            plt.plot(lstm_inp[:, 0], lstm_inp[:, 1], 'y--', label='LSTM Input (GT)')
        else:
            plt.plot(lstm_inp[:, 0], lstm_inp[:, 1], 'y--')

        orig_k = test_Inp[k, n_past - 1, :]
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

        gt_ped_x = test_Out[k, :, 0] + orig_k[0]
        gt_ped_y = test_Out[k, :, 1] + orig_k[1]
        gt_ped = np.vstack((gt_ped_x, gt_ped_y)).transpose()
        gt_ped = scale.denormalize(gt_ped)
        if k == 0:
            plt.plot(gt_ped[:, 0], gt_ped[:, 1], 'g--', label='Ground Truth Out')
        else:
            plt.plot(gt_ped[:, 0], gt_ped[:, 1], 'g--')

    plt.legend()
    plt.show()

