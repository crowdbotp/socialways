import numpy as np
import matplotlib.pyplot as plt


def create_samples(n_samples, n_per_batch=2):
    samples = []
    time_stamps = []
    for ii in range(n_samples):
        selected_way = (ii * n_conditions) // n_samples
        data_angle = selected_way * (360 / n_conditions)
        w_i = selected_way % (n_conditions/n_per_batch)
        t0 = ii % (n_samples // n_conditions) + w_i * (n_samples // n_conditions)
        data_angle = data_angle * np.pi / 180

        x0 = np.cos(data_angle) * 8
        y0 = np.sin(data_angle) * 8
        x1 = np.cos(data_angle) * 6
        y1 = np.sin(data_angle) * 6

        fixed_turn = ((ii % n_modes) - 1) * 20 * np.pi / 180
        p2_turn_rand = np.random.randn(1) * 1.5 * np.pi / 180
        p3_turn_rand = np.random.randn(1) * 2.5 * np.pi / 180

        x2 = np.cos(data_angle + fixed_turn + p2_turn_rand) * 4
        y2 = np.sin(data_angle + fixed_turn + p2_turn_rand) * 4

        x3 = np.cos(data_angle + fixed_turn + p2_turn_rand + p3_turn_rand) * 2
        y3 = np.sin(data_angle + fixed_turn + p2_turn_rand + p3_turn_rand) * 2

        samples.append(np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]]))
        time_stamps.append(np.array([t0*4, t0*4+1, t0*4+2, t0*4+3]))

    samples = np.array(samples) / 8

    return samples, time_stamps


def write_to_file(real_samples, timesteps, filename):
    with open(filename, 'w+') as gt_file:
        # gt_file.write('% each row contains n points: x(1), y(1), ... y(n)\n')
        for ii, sample in enumerate(real_samples):
            sample = np.reshape(sample, (-1, 2))
            # gt_file.write("".join(map(str, sam)) + "\n")
            for tt, val in enumerate(sample):
                gt_file.write("%.1f %.1f %.3f %.3f\n" % (timesteps[ii][tt], ii+1, val[0], val[1]))
            # gt_file.write("\n")
        gt_file.close()
        print('writing to ' + filename)


if __name__ == '__main__':
    n_modes = 3
    n_conditions = 6
    n_samples =  768
    # samples, time_stamps = create_samples(n_samples, n_per_batch=2)  # train
    samples, time_stamps = create_samples(n_samples, n_per_batch=6)  # test
    # FIXME: set output text file
    write_to_file(samples, time_stamps, '../data/toy/toy.txt')

    t_dict = dict()
    for ii in range(n_samples):
        if time_stamps[ii][0] not in t_dict:
            t_dict[time_stamps[ii][0]] = []
        t_dict[time_stamps[ii][0]].append( ii )

    obsvs = []
    preds = []
    times = []
    batches = []
    for key, values in t_dict.items():
        batches.append([len(obsvs), len(obsvs) + len(values)])
        for value in values:
            obsvs.append(samples[value][:2])
            preds.append(samples[value][2:])
            times.append(time_stamps[value][0])

    obsvs = np.array(obsvs).astype(np.float32)
    preds = np.array(preds).astype(np.float32)
    times = np.array(times).astype(np.int32)

    # FIXME: set output data file
    np.savez('../data/toy/data.npz', obsvs=obsvs, preds=preds, times=times, batches=batches)

    # exit(1)  # TODO: uncomment this line to show debug outputs

    # display
    for bb in batches:
        ob0 = obsvs[bb[0]]
        ob1 = obsvs[bb[1]-1]
        pr0 = preds[bb[0]]
        pr1 = preds[bb[1]-1]

        plt.plot(ob0[:, 0], ob0[:, 1], 'g')
        plt.plot(ob1[:, 0], ob1[:, 1], 'g')
        plt.plot(pr0[:, 0], pr0[:, 1], 'r')
        plt.plot(pr1[:, 0], pr1[:, 1], 'r')

        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.show()





