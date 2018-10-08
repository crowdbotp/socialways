import sys
import numpy as np
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, Input, concatenate, Conv2D, Flatten, Dropout, SimpleRNN
from keras.utils import plot_model
from lstm_model.my_base_classes import Scale, to_supervised, load_seyfried, MyConfig
import matplotlib.pyplot as plt
import pickle

np.random.seed(7)
data_arrays, scale = load_seyfried()
np.random.shuffle(data_arrays)
n_ped = len(data_arrays)

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
for ped in data_set:
    ped_pos_seqs = np.array(to_supervised(ped, n_past, n_next, False))
    ped_vel_seqs = np.array(to_supervised(ped, n_past-1, n_next, True))
    for i in range(ped_pos_seqs.shape[0]):
        pos_inp_data = np.vstack((pos_inp_data, ped_pos_seqs[i, 0:2 * n_past]))
        vel_inp_data = np.vstack((vel_inp_data, ped_vel_seqs[i, 0:2 * (n_past - 1)]))
        disp_out_data = np.vstack((disp_out_data, ped_pos_seqs[i, 2 * n_past:2 * (n_past + n_next)]))

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
test_out = disp_out_data[train_size:n_row, :].reshape(test_size, n_next, 2)

print(train_in_pos.shape)
print(train_in_vel.shape)
print(train_out.shape)

model_name = "models/mixer_model_3"
# TODO Build Model with Functional API
pos_inputs = Input(shape=(n_past, 2), name='position_input')
vel_inputs = Input(shape=(n_past - 1, 2), name='velocity_input')

pos_lstm_out = LSTM(128, name='position_encoder', return_sequences=True)(pos_inputs)
pos_dense = Dense(128, activation='relu')(pos_lstm_out)
# pos_dense = Dense(128, activation='tanh')(pos_dense)
# pos_dense = Dropout(0.99)(pos_dense)

vel_lstm_out = LSTM(128, name='velocity_encoder', return_sequences=True)(vel_inputs)
vel_dense = Dense(128, activation='relu')(vel_lstm_out)
# vel_dense = Dense(128, activation='tanh')(vel_dense)

x = concatenate([pos_dense, vel_dense], name='mixer')
# x = Dense(8*n_next)(x)
# x = Dense(8*n_next, activation='tanh')(x)
x = Dense(8*n_next, activation='relu')(x)

mixer_out = Dense(2*n_next)(x)
reshape_out = Reshape((n_next, 2), name='reshaper')(mixer_out)

model = Model(inputs=[pos_inputs, vel_inputs], outputs=reshape_out)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
plot_model(model, to_file=model_name+".png", show_shapes=True)

filepath = model_name + "-best-weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit([train_in_pos, train_in_vel], train_out, validation_split=0.33,
                    epochs=150, batch_size=256,
                    callbacks=callbacks_list)

# serialize model to JSON
model_json = model.to_json()
with open(model_name + ".json", "w+") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
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
