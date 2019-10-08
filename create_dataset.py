import numpy as np
from utils.parse_utils import BIWIParser, create_dataset

annot_file = 'path-to-dataset/obsmat.txt'   # FixMe: fix the input address
npz_out_file = '../data-8-12.npz'           # FixMe: fix the output filename
parser = BIWIParser()
parser.load(annot_file)

obsvs, preds, times, batches = create_dataset(parser.p_data,
                                              parser.t_data,
                                              range(parser.t_data[0][0], parser.t_data[-1][-1], parser.interval),
                                              8, 12)

np.savez(npz_out_file, obsvs=obsvs, preds=preds, times=times, batches=batches)
print('dataset was created successfully and stored in:', npz_out_file)
