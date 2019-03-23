import os
import cv2
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


def to_image_frame(Hinv, loc):
    """
    Given H^-1 and world coordinates, returns (u, v) in image coordinates.
    """
    locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
    if locHomogenous.ndim > 1:
        loc_tr = np.transpose(locHomogenous)
        loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
        locXYZ = np.transpose(loc_tr/loc_tr[2])  # to pixels (from millimeters)
        return locXYZ[:, :2].astype(int)
    else:
        locHomogenous = np.dot(Hinv, locHomogenous)  # to camera frame
        locXYZ = locHomogenous / locHomogenous[2]  # to pixels (from millimeters)
        return locXYZ[:2].astype(int)


def line_cv(im, ll, value, width):
    for tt in range(ll.shape[0] - 1):
        cv2.line(im, (ll[tt][1], ll[tt][0]), (ll[tt + 1][1], ll[tt + 1][0]), value, width)


def heat_map(im, pred_data):
    nSmp = pred_data.shape[0]
    nPed = pred_data.shape[1]

    K_im = np.zeros((nSmp, im.shape[0], im.shape[1]), np.uint8)
    for kk in range(nSmp):
        for ii in range(nPed):
            preds_our_XY_ik = to_image_frame(Hinv, pred_data[kk, ii])
            line_cv(K_im[kk], preds_our_XY_ik, value=10, width=10)
    lines_im = np.sum(K_im, axis=0).astype(np.uint8)
    lines_im = cv2.blur(lines_im, (5, 5))
    my_dpi = 96
    plt.figure(figsize=(lines_im.shape[1]/my_dpi, lines_im.shape[0]/my_dpi), dpi=my_dpi)
    cmap = sns.dark_palette("purple")
    sns.heatmap(lines_im, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    plt.margins(0, 0)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # heatmap = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # heatmap = heatmap.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.savefig('..\\..\\tmp.png')
    heatmap = cv2.imread('..\\..\\tmp.png')
    cv2.addWeighted(im, 0.4, heatmap, 1, 0, im)


dataset_name = "hotel"
dir = '..\\..\\preds-iccv\\' + dataset_name  # + '/pred-' + str(current_t) + '.npz'
Homography_file = os.path.join('..\\..\\data\\' + dataset_name, "H.txt")
Hinv = np.linalg.inv(np.loadtxt(Homography_file))
cap = cv2.VideoCapture('..\\..\\data\\' + dataset_name + '\\video.avi')

for dirpath, dirnames, filenames in os.walk(dir):
    for f in filenames:
        data = np.load(os.path.join(dirpath, f))
        obsvs = data['obsvs']
        preds_gtt = data['preds_gtt']
        preds_our = data['preds_our']
        preds_lnr = data['preds_lnr']
        time_stamp = data['timestamp']

        nPed = obsvs.shape[0]
        nPast = obsvs.shape[1]
        nNext = preds_lnr.shape[1]
        nSmp = preds_our.shape[0]
        onesP = np.ones((nPast, 1))
        onesN = np.ones((nNext + 1, 1))

        if nPed < 2:
            continue

        time_offset = -12
        cap.set(cv2.CAP_PROP_POS_FRAMES, time_stamp + time_offset)
        ret, im = cap.read()

        preds_gtt_aug = np.concatenate((obsvs[:, -1].reshape((nPed, 1, 2)), preds_gtt), axis=1)
        preds_lnr_aug = np.concatenate((obsvs[:, -1].reshape((nPed, 1, 2)), preds_lnr), axis=1)
        cur_loc_K = np.vstack([obsvs[:, -1].reshape((1, nPed, 1, 2)) for i in range(nSmp)])
        preds_our_aug = np.concatenate((cur_loc_K, preds_our), axis=2)

        heat_map(im, preds_our_aug)
        for ii in range(nPed):
            obsv_XY = to_image_frame(Hinv, obsvs[ii])
            pred_lnr_XY = to_image_frame(Hinv, preds_lnr_aug[ii])
            pred_gtt_XY = to_image_frame(Hinv, preds_gtt_aug[ii])
            line_cv(im, obsv_XY, (255, 0, 0), 2)
            line_cv(im, pred_lnr_XY, (0,100,155), 1)
            line_cv(im, pred_gtt_XY, (255, 255, 0), 1)

        cv2.imshow('Press s to save', im)
        key = cv2.waitKeyEx()
        if key == 115:
            cv2.imwrite("..\\..\\outputs-iccv\\"+dataset_name+"\\output-"+str(time_stamp)+".png", im)

