import csv
import numpy as np

gt_file_url = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/unrolled_2/ground-truth.csv'
gen_file_url = '/home/jamirian/workspace/crowd_sim/ped-prediction-py/unrolled_2/out-00100.csv'

real_samples = []
fake_samples = []

with open(gt_file_url, 'r') as real_file:
    csv_reader = csv.reader(real_file, delimiter=',')
    line_count = -1
    for row in csv_reader:
        line_count += 1
        if line_count == 0: continue
        vals = list(map(float, row[:-1]))
        vals = np.reshape(np.array(vals), (-1, 2))
        real_samples.append(vals)
    real_file.close()


with open(gen_file_url, 'r') as fake_file:
    csv_reader = csv.reader(fake_file, delimiter=',')
    line_count = -1
    for row in csv_reader:
        line_count += 1
        if line_count == 0: continue
        vals = list(map(float, row[:-1]))
        vals = np.reshape(np.array(vals), (-1, 2))
        real_samples.append(vals)
    fake_file.close()

