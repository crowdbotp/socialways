import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import time

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 2 *np.pi])
ax.set_ylim([-1, 1])

th = np.linspace(0, 2 * np.pi, 1000)

line, = ax.plot([],[],'b-', animated=True)
line.set_xdata(th)
# getFrame() makes a call to the CCD frame buffer and retrieves the most recent frame

# animation function
def update(data):
    line.set_ydata(data)

    return line,

def data_gen():
    t = 0
    while True:
        t +=1
        yield np.sin(th + t * np.pi/100)

# call the animator
anim = animation.FuncAnimation(fig, update, data_gen, interval=10, blit=True)
plt.show()