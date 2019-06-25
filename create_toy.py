import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import argparse

# from IPython.display import HTML, Image
# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')


def create_samples(n_samples, n_conditions, n_modes, n_per_batch=2):
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


class ToyAnimation:
    def __init__(self, samples):

        # First set up the figure, the axis, and the plot element we want to animate
        self.fig = plt.figure(num=None, figsize=(16, 9), dpi=80)
        plt.subplots_adjust(left=0.23, right=0.77, bottom=0.03, top=0.99)
        ax = plt.axes(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))

        for ii in range(samples.shape[0]):
            plt.plot(samples[ii, 0, 0], samples[ii, 0, 1], 'bo', alpha=0.2, zorder=1)
            plt.plot(samples[ii, 0:2, 0], samples[ii, 0:2, 1], 'b', linewidth=2, alpha=0.2, zorder=0)
            plt.plot(samples[ii, 1:, 0], samples[ii, 1:, 1], 'r', linewidth=2, alpha=0.2, zorder=0)

        self.dt = 0.04
        self.cur_id = 0
        self.cur_progress = 0
        self.cur_loc = samples[0, 0, :]
        self.scat = ax.scatter([], [], c='green', s=72, lw=2, zorder=2)
        self.samples = samples

        self.FPS = 15
        self.DURATION = 10
        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                            frames=self.FPS * self.DURATION, interval=5, blit=False)

    def step_animation(self, dt):
        if self.cur_progress > 1:
            self.cur_id = int(np.random.randint(0, self.samples.shape[0]))
            self.cur_progress = 0

        points = self.samples[self.cur_id]
        n_sub_goals = points.shape[0] - 1
        x = self.cur_progress * n_sub_goals
        start_ind = int(min(np.floor(x), n_sub_goals-1))
        pointA = points[start_ind]
        pointB = points[start_ind+1]
        self.cur_loc = pointB * (x - start_ind) + pointA * (start_ind + 1 - x)
        self.cur_progress += dt

    def init(self):
        # initialization function: plot the background of each frame
        self.scat.set_offsets(np.zeros((self.samples.shape[0], 2), dtype=np.float32))
        return self.scat,


    # animation function.  This is called sequentially
    def animate(self, i):
        self.init()
        self.step_animation(self.dt)
        self.scat.set_offsets(self.cur_loc)
        return self.scat,


    def save(self, filename):
        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        # anim.save('../toy_animation.mp4', fps=FPS, extra_args=['-vcodec', 'libx264'])
        self.anim.save(filename, fps=self.FPS, writer='imagemagick')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('-v')
    parser.add_argument('--txt', type=str)
    parser.add_argument('--npz', type=str)
    parser.add_argument('--n_conditions', default=8, type=int)
    parser.add_argument('--n_modes', default=3, type=int)
    parser.add_argument('--n_samples', default=3*8*128, type=int)
    parser.add_argument('--anim', action="store_true")
    args = parser.parse_args()

    samples, time_stamps = create_samples(args.n_samples, args.n_conditions, args.n_modes, n_per_batch=6)

    if args.txt is not None: # FIXME: set output text file
        write_to_file(samples, time_stamps, args.txt)

    t_dict = dict()
    for ii in range(args.n_samples):
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

    if args.npz is not None:  # FIXME: set output data file
        print('writing to ' + args.npz)
        np.savez(args.npz, obsvs=obsvs, preds=preds, times=times, batches=batches)

    if args.anim:
        toy_animation = ToyAnimation(samples)
        plt.show()
        # toy_animation.save('../toy.gif')
