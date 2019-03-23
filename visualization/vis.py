import os
import sys
import numpy as np
import scipy.misc
# from visualize import *
from skimage import draw
# import cv2


# import seaborn as sns
# import matplotlib.pyplot as plt
#
# iris = sns.load_dataset("iris")
# grid = sns.JointGrid(iris.petal_length, iris.petal_width, space=0, size=6, ratio=50)
# grid.plot_joint(plt.scatter, color="g")
# plt.plot([0, 4], [1.5, 0], linewidth=2)

def to_image_frame(Hinv, loc):
    """
    Given H^-1 and (x, y, z) in world coordinates, returns (u, v, 1) in image
    frame coordinates.
    """
    if loc.ndim > 1:
        loc_tr = np.transpose(loc)
        loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
        return np.transpose(loc_tr/loc_tr[2])  # to pixels (from millimeters)
    else:
        loc = np.dot(Hinv, loc)  # to camera frame
        return loc / loc[2]  # to pixels (from millimeters)


def line_np(im, ll, color=[]):
    for tt in range(ll.shape[0] - 1):
        rr, cc, val = draw.line_aa(ll[tt][1], ll[tt][0], ll[tt + 1][1], ll[tt + 1][0])
        im[rr, cc] =  (val * 10).astype(np.int8)
        # scipy.misc.imsave("out.png", img)


def heat_map(im, obsvs, pred_data):
    nSmp = pred_data.shape[0]
    nPed = pred_data.shape[1]
    nPast = obsvs.shape[1]
    nNext = pred_data.shape[2]
    onesN = np.ones((nNext, 1))
    for ii in range(nPed):
        obsv_XY_i = to_image_frame(Hinv, np.hstack((obsvs[ii], onesP)))[:, :2].astype(int)
        for kk in range(nSmp):
            preds_our_XY_ik = to_image_frame(Hinv, np.hstack((pred_data[kk, ii], onesN)))[:, :2].astype(int)
            preds_our_XY_ik = np.vstack((obsv_XY_i[-1].reshape((1, -1)), preds_our_XY_ik))
            for tt in range(nNext):
                line_np(im, preds_our_XY_ik)
    # np.save('out,png', im)



def line_cv(im, ll, color):
    for tt in range(ll.shape[0] - 1):
        cv2.line(im, (ll[tt][1], ll[tt][0]), (ll[tt + 1][1], ll[tt + 1][0]), color, 2)


dataset_name = "hotel"
dir = '../preds-iccv/' + dataset_name # + '/pred-' + str(current_t) + '.npz'

Homography_file = os.path.join('../data/' + dataset_name, "H.txt")
Hinv = np.linalg.inv(np.loadtxt(Homography_file))

# cap = cv2.VideoCapture('../data/' + dataset_name + '/video.avi')
# frame_id = -1

im_size = (576, 720)
img = np.zeros(im_size, dtype=np.uint8)
for dirpath, dirnames, filenames in os.walk(dir):
    for f in filenames:
        data = np.load(os.path.join(dirpath, f))
        obsvs = data['obsvs']
        preds_gtt = data['preds_gtt']
        preds_our = data['preds_our']
        preds_lnr = data['preds_lnr']
        time_stamp = data['timestamp']

        # cap.set(POS_FRAMES, time_stamp-1)
        # ret, im = cap.read()
        im = img

        print(im.shape)

        nPed = obsvs.shape[0]
        nPast = obsvs.shape[1]
        nNext = preds_lnr.shape[1]
        onesP = np.ones((nPast, 1))
        onesN = np.ones((nNext, 1))
        nSmp = preds_our.shape[0]
        for ii in range(nPed):
            obsv_XY = to_image_frame(Hinv, np.hstack((obsvs[ii], onesP)))[:, :2].astype(int)
            pred_lnr_XY = to_image_frame(Hinv, np.hstack((preds_lnr[ii], onesN)))[:, :2].astype(int)
            pred_lnr_XY = np.vstack((obsv_XY[-1].reshape((1, -1)), pred_lnr_XY))
            pred_gtt_XY = to_image_frame(Hinv, np.hstack((preds_gtt[ii], onesN)))[:, :2].astype(int)
            pred_gtt_XY = np.vstack((obsv_XY[-1].reshape((1, -1)), pred_gtt_XY))
            for tt in range(nPast - 1):
                line_np(im, obsv_XY, (255, 0, 0))
            for tt in range(nNext - 1):
                line_np(im, pred_lnr_XY, (255,0,255))
                line_np(im, pred_gtt_XY, (255, 255, 0))

        heat_map(im, obsvs, preds_our)
            # for kk in range(nSmp):
            #     preds_our_XY_kk = to_image_frame(Hinv, np.hstack((preds_our[kk, ii], onesN)))[:, :2].astype(int)
            #     preds_our_XY_kk = np.vstack((obsv_XY[-1].reshape((1, -1)), preds_our_XY_kk))
            #     for tt in range(nNext - 1):
            #         line_np(im, preds_our_XY_kk, (0, 0, 255))

        # cv2.imshow('frame', im)
        # cv2.waitKeyEx(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     frame_id += 1
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break