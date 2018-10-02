import sys
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape
import pickle
from keras.utils import plot_model
from lstm_model.my_base_classes import Scale, to_supervised, load_seyfried, MyConfig
import matplotlib.pyplot as plt

np.random.seed(7)
data_arrays, scale = load_seyfried()
n_ped = len(data_arrays)

train_size = int(n_ped * 0.67)
test_size = n_ped - train_size

# FIXME
# with open('scale.pkl', 'wb') as scale_file:
#     pickle.dump(scale, scale_file, pickle.HIGHEST_PROTOCOL)

# Normalize Data between [0,1]
for i in range(len(data_arrays)):
    data_arrays[i] = scale.normalize(data_arrays[i])
data_set = np.array(data_arrays)
train, test = data_set[0:train_size], data_set[train_size + 1:len(data_set)]

## Edge Padding
# track_lengths = np.array(track_lenght_list)
# min_len = min(track_lengths)
# max_len = max(track_lengths)
# padded_data = np.zeros((n_ped, max_len, 2))
# ped_counter = 0
# for ped in data:
#     padded_i = np.pad(ped, ((0, max_len - len(ped)), (0, 0)), 'edge')
#     padded_data[ped_counter, :, :] = padded_i
#     ped_counter += 1
#     print(padded_data.shape)


## Convert Time-series to supervised learning format
n_past = MyConfig().n_past
n_next = MyConfig().n_next
n_X, n_Y = 2 * n_past, 2 * n_next
n_XIY = 2 * (n_past + n_next)

train_set = np.array([], dtype=np.float).reshape(0, n_XIY)
for ped in train:
    train_seq_i = np.array(to_supervised(ped, n_past, n_next).values)
    for i in range(train_seq_i.shape[0]):
        train_set = np.vstack((train_set, train_seq_i[i, :]))

train_Inp = train_set[:, 0:n_X]
train_Out = train_set[:, n_X:n_XIY]

## Reshape Train Data
train_Inp = train_Inp.reshape((train_Inp.shape[0], n_past, 2))
train_Out = train_Out.reshape((train_Inp.shape[0], n_next, 2))
n_train = train_Inp.shape[0]

print(train_set.shape)
print(train_Inp.shape)
print(train_Out.shape)
print(n_train)

model_name = "models/model1"
# TODO Try non-sequential LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(n_past, 2), return_sequences=True))
# model.add(LSTM(128, input_shape=(n_past, 2)))
model.add(LSTM(50))
# model.add(Dense(32))
model.add(Dense(2 * n_next))
model.add(Reshape((n_next, 2)))
model.compile(loss='mean_squared_error', optimizer='adam')

plot_model(model, to_file=model_name+".png", show_shapes=True)
model.fit(train_Inp, train_Out, validation_split=0.33, epochs=500, batch_size=256)

# serialize model to JSON
model_json = model.to_json()
with open(model_name + ".json", "w+") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_name + ".h5")
print("Saved model to file")
