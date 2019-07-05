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


def text_cv(im, text, org, value):
    cv2.putText(im, text, org, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=value)


def draw_heatmap(im, pred_data, cmap):
    nSmp = pred_data.shape[0]
    nPed = pred_data.shape[1]

    K_im = np.zeros((nSmp, im.shape[0], im.shape[1]), np.uint8)
    for kk in range(nSmp//8):
        for ii in range(nPed):
            preds_our_XY_ik = to_image_frame(Hinv, pred_data[kk, ii])
            line_cv(K_im[kk], preds_our_XY_ik, value=1, width=10)
    lines_im = np.sum(K_im, axis=0).astype(np.uint8)
    lines_im = cv2.blur(lines_im, (15, 15))
    my_dpi = 96
    plt.figure(figsize=(lines_im.shape[1]/my_dpi, lines_im.shape[0]/my_dpi), dpi=my_dpi)

    sns.heatmap(lines_im, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    plt.margins(0, 0)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # heatmap = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # heatmap = heatmap.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.savefig('../tmp.png')
    heatmap = cv2.imread('../tmp.png')
    cv2.addWeighted(im, 1, heatmap, 1, 0, im)


def draw_gt_data(im): # SDD dataset
    # FIXME: set dataset file
    SDD_dataset = '../data/trajnet/train/stanford/nexus_9.npz'
    SDD_dataset = '../data/trajnet/train/stanford/hyang_6.npz'

    dataset = np.load(SDD_dataset)
    obsvs = dataset['dataset_x']
    preds = dataset['dataset_y']
    # times = dataset['dataset_t']
    samples = np.concatenate((obsvs, preds), axis=1)

    max_x = np.max(samples[:, :, 0])
    min_x = np.min(samples[:, :, 0])

    for ii in range(len(samples)):
        pi = samples[ii]
        # plt.plot(pi[:8, 0], pi[:8, 1], 'b', alpha=0.95)
        # plt.plot(pi[7:, 0], pi[7:, 1], 'r', alpha=0.55)
        # plt.plot(pi[-1, 0], pi[-1, 1], 'bx')

        # plt.plot(pi[:, 0], pi[:, 1], 'c', alpha=0.55)
        # plt.plot(pi[0, 0], pi[0, 1], 'b.')

        obsv_XY = to_image_frame(Hinv, samples[ii])
        line_cv(im, obsv_XY, (30, 100, 0), 3)


dataset_name = "toy"  # FIXME : dataset name

# FIXME : set prediction-folder
# preds_dir = '../preds-iccv/eth/infoGAN-i2/1000'
# preds_dir = '../preds-iccv/' + 'toy/2000'  # + str(current_t) + '.npz'
# preds_dir = '../preds-iccv/toy/just_l2/1000'
# preds_dir = '../preds-iccv/toy/just_l2_20/500'
# preds_dir = '../preds-iccv/eth'
# preds_dir = '../preds-iccv/toy/info16-u10/90000'
# preds_dir = '../preds-iccv/toy/infogan/800'
# preds_dir = '../preds-iccv/toy/info-pure-128/1100'
# preds_dir = '../preds-iccv/toy/l2-w50-128/1100'
# preds_dir = '../preds-iccv/toy/info-u5-128/1100'
# preds_dir = '../preds-iccv/hyang6/unrolled10/5200'
# preds_dir = '../preds-iccv/hyang6/info/5500'
# preds_dir = '../preds-iccv/hyang6/info/10000'
# preds_dir = '../preds-iccv/hyang6/vanilla'
# preds_dir = '../preds-iccv/gate2/vanilla/5500'
# preds_dir = '../preds-cvprw/toy'


# FIXME : For toy dataset
preds_dir    = 'medium/toy/socialWays'
out_dir_main = 'medium/figs/socialWays/'

# FIXME : For toy dataset
Hinv = np.eye(3)
im_size = (480, 480, 3)
# Hinv[0,0], Hinv[1,1] = 200, 200
# Hinv[0,2], Hinv[1,2] = im_size[0]/2, im_size[1]/2


# FIXME : for real datasets, load the video file
homography_file = os.path.join('data/' + dataset_name, 'H.txt')
if os.path.exists(homography_file):
    Hinv = np.linalg.inv(np.loadtxt(homography_file))
else:
    print('[INF] No homography file')

video_file = 'data/' + dataset_name + '/video.avi'  # FIXME: for BIWI
image_file = 'data/' + dataset_name + '/reference.jpg'   # FIXME
if os.path.exists(video_file):
    print('[INF] Using video file'+video_file)
    cap = cv2.VideoCapture()
    time_offset = -12
elif os.path.exists(image_file):
    print('[INF] Using image file '+image_file)
    cap        = None
    use_ref_im = True

    # For test
    ref_im = np.zeros((600, 600, 3), dtype=np.uint8)
    Hinv = np.zeros((3, 3))
    Hinv[0, 1], Hinv[1, 0], Hinv[2, 2] = 0.6, 0.6, 1
    Hinv[0, 2], Hinv[1, 2] = -400, 0

    ref_im = cv2.imread(image_file)
    Hinv = np.zeros((3, 3))
    Hinv[0, 1], Hinv[1, 0], Hinv[2, 2] = 1, 1, 1
    # Hinv[0, 2], Hinv[1, 2] = 550, 720

else:  # toy dataset
    print('[INF] No image nor video file')
    cap = None
    use_ref_im = False
    Hinv[0, 0], Hinv[1, 1] = 200, 200
    Hinv[0, 2], Hinv[1, 2] = 240, 240

epc_counter = 0
for dirpath, dirnames, filenames in sorted(os.walk(preds_dir)):
    for file_cntr, f in enumerate(sorted(filenames)):
        if 'stats' in f or not 'npz' in f: continue
        filename = os.path.join(dirpath, f)
        epc_str = f[:f.rfind('-')]
        if epc_str.isdigit():
            epc = int(epc_str)
        else:
            epc = epc_counter
            epc_counter += 1
        if epc%1000!=0: continue
        print('[INF] Plotting results from '+filename)

        # FIXME
        out_file = out_dir_main + '%05d' % epc + '.png'
        if os.path.exists(out_file): continue
        # Load data from file
        data = np.load(filename)
        # Observed data
        obsvs     = data['obsvs']
        # Ground truth
        preds_gtt = data['preds_gtt']
        # Prediction by the model
        preds_our = data['preds_our']
        # Prediction by linear model
        preds_lnr = data['preds_lnr']
        # Time stamp
        time_stamp = data['timestamp']

        # Number of observed trajectories
        nPed  = obsvs.shape[0]
        # Length of observed trajectories
        nPast = obsvs.shape[1]
        # Length of predicted trajectories
        nNext = preds_lnr.shape[1]
        # Number of samples generated by the generative model
        nSmp  = preds_our.shape[0]
        onesP = np.ones((nPast, 1))
        onesN = np.ones((nNext + 1, 1))

        if nPed < 2:
            continue

        # FIXME: for real datasets, read the video file
        if cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, time_stamp + time_offset))
            ret, im = cap.read()
            if not ret:
                break
        elif use_ref_im:
            # Use a reference image
            im = np.copy(ref_im)
        else:
            #
            im = np.ones(im_size, dtype=np.uint8) * 128

        preds_gtt_aug = np.concatenate((obsvs[:, -1].reshape((nPed, 1, 2)), preds_gtt), axis=1)
        preds_lnr_aug = np.concatenate((obsvs[:, -1].reshape((nPed, 1, 2)), preds_lnr), axis=1)
        # Augment the predicted data with the last observation
        cur_loc_K = np.vstack([obsvs[:, -1].reshape((1, nPed, 1, 2)) for i in range(nSmp)])
        preds_our_aug = np.concatenate((cur_loc_K, preds_our), axis=2)

        # cv2.imshow('im', im)
        # cv2.waitKeyEx()
        # Draw a heatmap for the model predictions
        cmap = sns.dark_palette("purple")
        draw_heatmap(im, preds_our_aug, cmap)
        # draw_gt_data(im) # FIXME
        text_cv(im, 'Epoch= %05d' %epc, (15, 50), (50,50,250))

        # Draw the observed parts
        for ii in range(nPed):
            # im = np.copy(ref_im)

            # if ii != 3: continue
            # cmap = sns.dark_palette((260, 10*ii, 60), input="husl")
            # im = np.ones(im_size, dtype=np.uint8) * 1

            obsv_XY = to_image_frame(Hinv, obsvs[ii])
            # obsv_XY[:,1] = obsv_XY[:,1]
            line_cv(im, obsv_XY, (255, 20, 0), 2)

            # FIXME
            # pred_lnr_XY = to_image_frame(Hinv, preds_lnr_aug[ii])
            # pred_gtt_XY = to_image_frame(Hinv, preds_gtt_aug[ii])
            # line_cv(im, pred_lnr_XY, (0, 100, 155), 1)
            # line_cv(im, pred_gtt_XY, (255, 255, 0), 1)
            # draw_heatmap(im, np.expand_dims(preds_our_aug[:, ii], 1), cmap)

            # cv2.imshow('im', im)
            # cv2.waitKeyEx()
            # exit(1)

        out_dir = out_file[:out_file.rfind('/')]
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(out_file, im)
        print('[INF] Writing image to ', out_file)

        # draw_heatmap(im, preds_our_aug, cmap)


        # out_dir = out_file[:out_file.rfind('/')]
        # os.makedirs(out_dir, exist_ok=True)
        # cv2.imwrite(out_file, im)
        # print('writing image to ', out_file)
