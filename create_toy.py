import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import argparse

# from IPython.display import HTML, Image
# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')

# Function that creates path samples
def create_samples(n_samples, n_conditions, n_modes, n_per_batch=2):
    samples = []
    time_stamps = []
    for ii in range(n_samples):
        # Determine the condition (i.e. the initial part of the path)
        # selected_way is a an integer between and n_conditions
        selected_way = (ii * n_conditions) // n_samples
        w_i = selected_way % (n_conditions/n_per_batch)
        # Initial time stamp for this sample
        t0 = ii % (n_samples // n_conditions) + w_i * (n_samples // n_conditions)
        # Approach angle
        data_angle = selected_way * (2.0*np.pi / n_conditions)

        # The first two points are located on the same radial line centered
        # on the origin.
        x0 = np.cos(data_angle) * 4
        y0 = np.sin(data_angle) * 4
        x1 = np.cos(data_angle) * 3
        y1 = np.sin(data_angle) * 3

        # Level of rotation is given by the mode index ((ii % n_modes)).
        # The modes are centered around 0
        fixed_turn = ((ii % n_modes) - n_modes//2) * 16 * np.pi / 180

        # Third point: located on a circle of radius 2
        # Normal-distributed angle deviation are introduced as p2_turn_rand
        p2_turn_rand = (np.random.rand(1) - 0.5) * 4 * np.pi / 180
        x2 = np.cos(data_angle + fixed_turn + p2_turn_rand) * 2
        y2 = np.sin(data_angle + fixed_turn + p2_turn_rand) * 2

        # Fourth point: located on a circle of radius 1
        # Normal-distributed angle deviation are introduced as p3_turn_rand
        p3_turn_rand = (np.random.rand(1) - 0.5) * 6 * np.pi / 180
        x3 = np.cos(data_angle + fixed_turn + p2_turn_rand + p3_turn_rand)
        y3 = np.sin(data_angle + fixed_turn + p2_turn_rand + p3_turn_rand)

        # Add the path
        samples.append(np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]]))
        # Add the time stamps. t0 is the starting position time stamp
        time_stamps.append(np.array([t0*4, t0*4+1, t0*4+2, t0*4+3]))

    # Scale down the paths
    samples = np.array(samples) / 4
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

# A class for the animation of the generated samples
class ToyAnimation:
    # Constructor
    def __init__(self, samples):

        # First set up the figure, the axis, and the plot element we want to animate
        self.fig = plt.figure(num=None, figsize=(16, 9), dpi=80)
        plt.subplots_adjust(left=0.23, right=0.77, bottom=0.03, top=0.99)
        ax = plt.axes(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))

        # Plot the samples
        for ii in range(samples.shape[0]):
            # Starting points in blue
            plt.plot(samples[ii, 0, 0], samples[ii, 0, 1], 'bo', alpha=0.2, zorder=1)
            # Initial part (fist two positions) in blue
            plt.plot(samples[ii, 0:2, 0], samples[ii, 0:2, 1], 'b', linewidth=2, alpha=0.2, zorder=0)
            # Second part in red
            plt.plot(samples[ii, 1:, 0], samples[ii, 1:, 1], 'r', linewidth=2, alpha=0.2, zorder=0)

        self.dt = 0.04
        self.cur_id = 0
        self.cur_progress = 0
        self.cur_loc = samples[0, 0, :]
        self.scat = ax.scatter([], [], c='green', s=72, lw=2, zorder=2)
        self.samples = samples

        self.FPS = 15
        self.DURATION = 15
        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                            frames=self.FPS * self.DURATION, interval=5, blit=False)

    # Main function of the animation: Draw the agent
    def step_animation(self, dt):
        # The animation of a single sample takes a time 1
        if self.cur_progress > 1:
            # Select a sample randomly
            self.cur_id = int(np.random.randint(0, self.samples.shape[0]))
            self.cur_progress = 0
        # Selected path
        points = self.samples[self.cur_id]
        n_sub_goals = points.shape[0] - 1
        # Determines the current path segment
        x = self.cur_progress * n_sub_goals
        start_ind = int(min(np.floor(x), n_sub_goals-1))
        pointA = points[start_ind]
        pointB = points[start_ind+1]
        # Interpolates it linearly
        self.cur_loc = pointB * (x - start_ind) + pointA * (start_ind + 1 - x)
        self.cur_progress += dt

    def init(self):
        # Initialization function: plot the background of each frame
        self.scat.set_offsets(np.zeros((self.samples.shape[0], 2), dtype=np.float32))
        return self.scat,


    # Animation function.  This is called sequentially
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
    np.random.seed(30)

    # parser.add_argument('-v')
    parser.add_argument('--txt', type=str)
    parser.add_argument('--npz', type=str)
    parser.add_argument('--n_conditions', default=6, type=int)
    parser.add_argument('--n_modes', default=3, type=int)
    parser.add_argument('--n_samples', default=3*6*12, type=int)
    parser.add_argument('--anim', action="store_true")
    args = parser.parse_args()

    # Create path samples based on the specified parameters
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
        # Separates the observed data (first two positions)
        # from the part to predict (last two positions)
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
