import sys
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape
import pickle
from keras.utils import plot_model

from src.learning_utils import MyConfig
from src.parse_utils import Scale, to_supervised, SeyfriedParser, BIWIParser
import matplotlib.pyplot as plt

np.random.seed(7)
# parser = SeyfriedParser()
# pos_data, vel_data, time_data = parser.load('../data/sey01.sey')

parser = BIWIParser()
pos_data, vel_data, time_data = parser.load('../data/eth.wap')

n_ped = len(pos_data)
train_size = int(n_ped * 0.67)
test_size = n_ped - train_size
scale = parser.scale

# FIXME
# with open('scale.pkl', 'wb') as scale_file:
#     pickle.dump(scale, scale_file, pickle.HIGHEST_PROTOCOL)

# Normalize Data between [0,1]
for i in range(len(pos_data)):
    pos_data[i] = scale.normalize(pos_data[i])
data_set = np.array(pos_data)
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
    train_seq_i = np.array(to_supervised(ped, n_past, n_next))
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

model_name = "models/model_test"
model = Sequential()
model.add(LSTM(64, input_shape=(n_past, 2)))
# model.add(LSTM(100, input_shape=(n_past, 2), return_sequences=True))
# model.add(LSTM(50))
# model.add(Dense(24))
model.add(Dense(2 * n_next))
model.add(Reshape((n_next, 2)))
model.compile(loss='mean_squared_error', optimizer='adam')

plot_model(model, to_file=model_name+".png", show_shapes=True)
history = model.fit(train_Inp, train_Out, validation_split=0.33, epochs=300, batch_size=256)

# save model and weights
model_json = model.to_json()
with open(model_name + ".json", "w+") as json_file:
    json_file.write(model_json)
model.save_weights(model_name + ".h5")
print("Saved model to file")


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_name + "_loss.png")
plt.show()
