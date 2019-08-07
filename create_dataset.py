import numpy as np
from utils.parse_utils import BIWIParser, create_dataset

annot_file = 'path-to-dataset/obsmat.txt'
npz_out_file = '../data-8-12.npz'
parser = BIWIParser()
parser.load(annot_file)

obsvs, preds, times, batches = create_dataset(parser.p_data,
                                              parser.t_data,
                                              range(parser.t_data[0][0], parser.t_data[-1][-1], parser.interval))

np.savez(npz_out_file, obsvs=obsvs, preds=preds, times=times, batches=batches)
