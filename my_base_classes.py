import csv
import math
import numpy as np
from pandas import DataFrame, concat


class MyConfig:
    n_past = 8
    n_next = 12


class Scale(object):
    def __init__(self):
        self.min_x = +math.inf
        self.max_x = -math.inf
        self.min_y = +math.inf
        self.max_y = -math.inf

    def normalize(self, data):
        data[:, 0] = (data[:, 0] - self.min_x) / (self.max_x - self.min_x)
        data[:, 1] = (data[:, 1] - self.min_y) / (self.max_y - self.min_y)
        return data

    def denormalize(self, data):
        data_copy = data

        sx = (self.max_x - self.min_x)
        sy = (self.max_y - self.min_y)

        if np.array(data).ndim == 1:
            data_copy[0] = data[0] * sx + self.min_x
            data_copy[1] = data[1] * sy + self.min_y
        else:
            data_copy[:, 0] = data[:, 0] * sx + self.min_x
            data_copy[:, 1] = data[:, 1] * sy + self.min_y

        return data_copy


# seyfried rows are like this:
# Few Header lines for Obstacles
# id, timestamp, pos_x, pos_y, pos_z
def load_seyfried(filename='/home/jamirian/workspace/crowd_sim/tests/sey01/sey01.sey', down_sample=3):
    data_list = list()

    with open(filename, 'r') as data_file:
        csv_reader = csv.reader(data_file, delimiter=' ')
        id_list = list()
        i = 0
        for row in csv_reader:
            i += 1
            if i == 4:
                fps = row[0]

            if len(row) != 5:
                continue

            id = row[0]
            # print(row)
            ts = float(row[1])
            if ts % down_sample != 0:
                continue

            px = float(row[2])
            py = float(row[3])
            pz = float(row[4])
            if id not in id_list:
                id_list.append(id)
                data_list.append(list())
            data_list[-1].append(np.array([px, py]))

    data = list()
    track_length_list = []

    scale = Scale()
    for d in data_list:
        len_i = len(d)
        track_length_list.append(len_i)
        ped_i = np.array(d)
        scale.min_x = min(scale.min_x, min(ped_i[:, 0]))
        scale.max_x = max(scale.max_x, max(ped_i[:, 0]))
        scale.min_y = min(scale.min_y, min(ped_i[:, 1]))
        scale.max_y = max(scale.max_y, max(ped_i[:, 1]))
        data.append(ped_i)

    return data, scale


def to_supervised(data, n_in=1, n_out=1, drop_nan=True):
    """
    :type drop_nan: bool
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i-1))
        names += [('var%d(t-%d)' % (j + 1, i-1)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(1, n_out+1):
        cols.append(df.shift(-i) - df.shift(0))  # displacement
        # cols.append(df.shift(-i)) # position
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if drop_nan:
        agg.dropna(inplace=True)
    return agg


class ConstVelModel:
    def __init__(self, conf=MyConfig()):
        self.my_conf = conf

    def predict(self, inp): # use config for
        #inp.ndim = 2
        avg_vel = np.array([0, 0])
        if inp.ndim > 1 and inp.shape[0] > 1:
            for i in range(1, inp.shape[0]):
                avg_vel = avg_vel + inp[i, :]-inp[i-1, :]
            avg_vel = avg_vel / (inp.shape[0]-1)

        cur_pos = inp[-1, :]
        out = np.empty((0, 2))
        for i in range(0, self.my_conf.n_next):
            out = np.vstack((out, cur_pos + avg_vel * (i+1)))

        return out
