import os
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt


def compute_1nn(reals, fakes, obsv_len=2):
    Real_pos = 0
    Real_neg = 0
    Fake_pos = 0
    Fake_neg = 0

    n_reals = reals.shape[0]
    n_fakes = fakes.shape[0]
    n_mixed = n_reals + n_fakes
    nPed = reals.shape[1]

    for kk in range(nPed):
        mixed_sampels = []
        D = np.ones((n_mixed, n_mixed)) * 1000

        for ii in range(n_reals):
            sample_and_label = [reals[ii, kk], [1]]
            mixed_sampels.append(sample_and_label)
        for ii in range(n_fakes):
            sample_and_label = [fakes[ii, kk], [-1]]
            mixed_sampels.append(sample_and_label)

        for ii in range(0, n_mixed):
            for jj in range(ii+1, n_mixed):
                diff = mixed_sampels[ii][0][obsv_len:] - mixed_sampels[jj][0][obsv_len:]
                dij = np.mean(np.sqrt(np.sum(np.power(diff, 2), 1)))
                D[ii, jj], D[jj, ii] = dij, dij

        for ii in range(0, n_mixed):
            NN_ind = np.argmin(D[ii])
            if mixed_sampels[ii][1][0] == 1 and mixed_sampels[NN_ind][1][0] == 1:
                Real_pos += 1
            elif mixed_sampels[ii][1][0] == 1 and mixed_sampels[NN_ind][1][0] == -1:
                Real_neg += 1
            elif mixed_sampels[ii][1][0] == -1 and mixed_sampels[NN_ind][1][0] == -1:
                Fake_pos += 1
            else:  # if all_samples[ii][1][0] == -1 and all_samples[NN_ind][1][0] == 1:
                Fake_neg += 1
    return np.array([(Real_pos + Fake_pos) / (n_mixed * nPed), Real_pos/(n_reals*nPed), Fake_pos / (n_fakes*nPed)])


def compute_wasserstein(reals, fakes, obsv_len=2):
    n_reals = reals.shape[0]
    n_fakes = fakes.shape[0]
    nPed = reals.shape[1]

    cost = 0
    for kk in range(nPed):
        D = np.ones((n_reals, n_fakes)) * 1000
        for ii in range(0, n_reals):
            for jj in range(0, n_fakes):
                diff = reals[ii, kk][obsv_len:] - fakes[jj, kk][obsv_len:]
                dij = np.mean(np.sqrt(np.sum(np.power(diff, 2), 1)))
                D[ii, jj], D[jj, ii] = dij, dij

        row_ind, col_ind = op.linear_sum_assignment(D)
        cost += D[row_ind, col_ind].sum()
        # cost += np.sum(D)

    return cost/(n_reals*nPed)



def calc_and_store_stats(main_dir):
    stats_1nn = dict()
    stats_wst = dict()

    for dirpath, dirnames, filenames in sorted(os.walk(main_dir)):
        cur_dir = dirpath[dirpath.rfind('/')+1:]
        if not cur_dir.isdigit():
            continue
        epoch = int(cur_dir)

        stat_1nn_i = 0
        stat_wst_i = 0
        n_files = 0

        for ii, f in enumerate(sorted(filenames)):
            if 'npz' not in f: continue
            fake_data = np.load(os.path.join(dirpath, f))
            fake_obsvs = fake_data['obsvs']
            fake_preds = fake_data['preds_our']

            nPed = fake_obsvs.shape[0]
            if nPed < 6 :  # FIXME: number of files
                continue

            K = real_samples.shape[0]
            fake_obsvs = np.concatenate([fake_obsvs.reshape((1, nPed, 2, 2)) for _ in range(K)], axis=0)
            fake_samples = np.concatenate((fake_obsvs.reshape(-1, 2, 2), fake_preds[:K].reshape(-1, 2, 2)), axis=1)

            # fake_samples = fake_samples[:K]
            stat_1nn = compute_1nn(real_samples.reshape(K, nPed, n_past+n_next, 2), fake_samples.reshape(K, nPed, n_past+n_next, 2))
            stat_1nn_i += stat_1nn[0]
            stat_wst_i += compute_wasserstein(real_samples.reshape(K, nPed, n_past+n_next, 2), fake_samples.reshape(K, nPed, n_past+n_next, 2))
            n_files += 1

        print(main_dir, 'epoch = %d, EMD = %.5f, 1nn = %.5f'
              %(epoch, stat_wst_i/n_files, stat_1nn_i/n_files))
        stats_1nn[epoch] = stat_1nn_i / n_files
        stats_wst[epoch] = stat_wst_i / n_files

    stats_wst_list = []
    stats_1nn_list = []
    keys_wst = sorted(stats_wst.keys(), reverse=False)
    for key in keys_wst:
        stats_wst_list.append(stats_wst[key])

    keys_1nn = sorted(stats_1nn.keys(), reverse=False)
    for key in keys_1nn:
        stats_1nn_list.append(stats_1nn[key])

    np.savez(stats_file, stats_1nn=stats_1nn_list, stats_wst=stats_wst_list)


def plot_stats_1nn(K_data=-1, interval=1, ind=-1):
    stats_data = np.load(stats_file)
    stats_1nn = stats_data['stats_1nn']
    if K_data == -1: K_data = len(stats_1nn)
    epc = [50 * nItr * (i+1) for i in range(0, K_data, interval)]
    stats_1nn = [stats_1nn[i] * 100 for i in range(0, K_data, interval)]
    label, = plt.plot(epc, stats_1nn, args[ind], LineWidth=1)
    plt.fill_between(epc, stats_1nn, np.ones_like(stats_1nn) * 50, color=colors[ind], alpha=0.2)
    print(epc)
    print(stats_1nn)
    return label


def plot_stats_wst(K_data=-1, interval=1, ind=-1):
    stats_data = np.load(stats_file)
    stats_wst = stats_data['stats_wst']
    if K_data == -1: K_data = len(stats_wst)
    epc = [50 * nItr * (i+1) for i in range(len(stats_wst))]
    stats_wst = [stats_wst[i] * 1 for i in range(0, K_data, interval)]
    label, = plt.plot(epc[:K_data], stats_wst[:K_data], args[ind], LineWidth=1)
    plt.fill_between(epc[:K_data], stats_wst[:K_data], np.zeros_like(stats_wst)[:K_data] , color=colors[ind], alpha=0.2)

    return label


def plot_dataset():
    fig, ax = plt.subplots()
    for ii in range(len(real_samples)):
        # plt.plot(real_samples[ii, :2, 0], real_samples[ii, :2, 1], 'b')
        plt.plot(real_samples[ii, 1:, 0], real_samples[ii, 1:, 1], 'r')
        plt.plot(real_samples[ii, 0, 0], real_samples[ii, 0, 1], 'bo')
        plt.plot(real_samples[ii, -1, 0], real_samples[ii, -1, 1], 'g.')
        ax.arrow(real_samples[ii, 0, 0], real_samples[ii, 0, 1],
                 (real_samples[ii, 1, 0] -real_samples[ii, 0, 0]) * 0.85,
                 (real_samples[ii, 1, 1] -real_samples[ii, 0, 1]) * 0.85,  head_width=0.03, head_length=0.04, fc='k', ec='b')

    plt.xlim([-1.1, 1.1])
    plt.ylim([-1, 1])
    plt.show()
    exit(1)


dataset_file = '../data/toy/toy-768.npz'
real = np.load(dataset_file)
real_obsv, real_pred, = real['obsvs'], real['preds']
real_samples = np.concatenate((real_obsv, real_pred), axis=1)
# plot_dataset()
#FIXME

num_samples = 20  # FIXME
real_samples = real_samples.reshape((-1, 6, 4, 2))[:num_samples]

n_samples = real_obsv.shape[0]
n_past = real_obsv.shape[1]
n_next = real_pred.shape[1]

# FIXME
main_dirs = (
            '../preds-iccv/toy/VanillaGAN',
            '../preds-iccv/toy/L2-GAN',
            '../preds-iccv/toy/SGAN-V20',
            '../preds-iccv/toy/Unrolled10+L2',
            '../preds-iccv/toy/Info+Unrolled5',
            '../preds-iccv/toy/Unrolled10',
            '../preds-iccv/toy/InfoGAN',
            )


colors = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta', 'black', 'orange', 'brown']
args = ['r--x', 'g--o', 'b--*', 'c--^', 'y--v', 'm--+', 'k--*', 'r--s', 'r-->']
fig = plt.figure(figsize=(12, 2.5), dpi=100, facecolor='w', edgecolor='k')

labels = []
legends = []
nItr = 63

for i, main_dir in enumerate(main_dirs):
    stats_file = os.path.join(main_dir, 'stats' + str(num_samples) + '.npz')
    if not os.path.exists(stats_file):
        calc_and_store_stats(main_dir)

    # labels.append(plot_stats_1nn(30, 1, i))
    # plt.ylabel('1NN Accuracy %')
    # plt.ylim([73, 104])

    labels.append(plot_stats_wst(30, 1, i))
    plt.ylabel('Earth Mover\'s Distance')
    plt.ylim([-0.005, .12])

    legends.append(main_dir[main_dir.rfind('/')+1:])


# plt.xlabel('Iteration')
plt.xlim([2000, 117000])

plt.legend(labels, legends, loc='lower right')
plt.grid(color='gray', linestyle='--', linewidth=1, alpha=0.2, axis='y')
fig.patch.set_visible(False)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.show()
