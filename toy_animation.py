import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, rc

from create_toy import create_samples
from IPython.display import HTML, Image
# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(num=None, figsize=(16, 9), dpi=80)
plt.subplots_adjust(left=0.23, right=0.77, bottom=0.03, top=0.99)
# plt.tight_layout()
ax = plt.axes(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))

n_modes = 3
n_conditions = 8
N = 768
samples, time_stamps = create_samples(N, n_conditions, n_modes, n_per_batch=6)  # test


for ii in range(samples.shape[0]):
    si = samples[ii]
    plt.plot(samples[ii, 0, 0], samples[ii, 0, 1], 'bo', alpha=0.2, zorder=1)
    plt.plot(samples[ii, 0:2, 0], samples[ii, 0:2, 1], 'b', linewidth=2, alpha=0.2, zorder=0)
    plt.plot(samples[ii, 1:, 0], samples[ii, 1:, 1], 'r', linewidth=2, alpha=0.2, zorder=0)


cur_id = 0
cur_progress = 0
cur_loc = samples[0, 0, :]
scat = ax.scatter([], [], c='green', s=72, lw=2, zorder=2)


def step(dt):
    global cur_id, cur_progress, cur_loc
    if cur_progress > 1:
        cur_id = int(np.random.randint(0, samples.shape[0]))
        cur_progress = 0

    points = samples[cur_id]
    n_sub_goals = points.shape[0] - 1
    x = cur_progress * n_sub_goals
    start_ind = int(min(np.floor(x), n_sub_goals-1))
    pointA = points[start_ind]
    pointB = points[start_ind+1]
    cur_loc = pointB * (x - start_ind) + pointA * (start_ind + 1 - x)
    # print(cur_progress, cur_loc)
    cur_progress += dt


# initialization function: plot the background of each frame
def init():
    scat.set_offsets(np.zeros((N, 2), dtype=np.float32))
    return scat,


# animation function.  This is called sequentially
def animate(i):
    init()
    dt = 0.04
    step(dt)
    scat.set_offsets(cur_loc)
    return scat,
    return line,


FPS = 15
DURATION = 10
# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=FPS*DURATION, interval=5, blit=False)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# anim.save('../toy_animation.mp4', fps=FPS, extra_args=['-vcodec', 'libx264'])
anim.save('../toy_animation.gif', fps=FPS, writer='imagemagick')

plt.show()