from math import atan2, cos, sin, sqrt, exp
from scipy import ndimage
from lstm_model.utility import Scale, to_supervised, MyConfig, SeyfriedParser
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import imageio


np.random.seed(7)
eps = np.finfo(float).eps
PIX_SCALE = 100
grid_size = [37, 37]
cnt = np.floor([grid_size[0] / 2, grid_size[1] / 2]).astype(int)

parser = SeyfriedParser()
pos_data, vel_data, time_data = parser.load('/home/jamirian/workspace/crowd_sim/tests/sey01/sey01.sey')
scale = parser.scale
print(scale.min_y, scale.max_y)


for i in range(len(pos_data)):
    pos_data[i] = scale.normalize(pos_data[i])
    vel_data[i] = scale.normalize(vel_data[i], False)
data_set = np.array(pos_data)


for i in range(0, len(pos_data)):

    # To export GIFs that depicts the sequence of perception of the agent
    gif_writer = imageio.get_writer("../outputs/RISK_%d.gif" % i, mode='I')

    ped_i_poss = pos_data[i]
    ped_i_vels = vel_data[i]
    my_pos_0 = ped_i_poss[0]
    my_goal = ped_i_poss[-1]
    print("Building the occupancy map for ped %d : \n" % i)
    for t_index in range(time_data[i].shape[0]):
        t = time_data[i][t_index]
        my_cur_loc = ped_i_poss[t_index]
        my_cur_vel = ped_i_vels[t_index]

        # Creating the Grids (Centered on the agent)
        cartesian_grid = np.zeros((grid_size[0], grid_size[1], 3), dtype="uint8")
        cartesian_grid[cnt[0], cnt[1], 1] = 250

        # The Cartesian grid which is rotated and aligned to the heading of agent
        aligned_cartesian_grid = np.zeros((grid_size[0], grid_size[1], 3), dtype="uint8")
        aligned_cartesian_grid[cnt[0], cnt[1], 1] = 250

        # Polar Grid
        aligned_polar_grid = np.zeros((grid_size[0], grid_size[1], 3), dtype="uint8")
        aligned_polar_grid[0, cnt[1], 1] = 250

        neighbor_locs = np.empty((0, 2))
        neighbor_vels = np.empty((0, 2))
        for j in range(len(pos_data)):
            if j == i:
                continue

            t_index_j = np.array(np.where(time_data[j] == t))
            if t_index_j.size:
                neighbor_locs = np.vstack((neighbor_locs, pos_data[j][t_index_j[-1]]))
                neighbor_vels = np.vstack((neighbor_vels, vel_data[j][t_index_j[-1]]))

        if not neighbor_locs.size:
            continue

        # print('#######################################')
        # print('my cur pos = ', scale.denormalize(my_cur_loc) * 100)
        # print("others pos = \n", scale.denormalize(others_locs) * 100)
        # print("others vel = \n", scale.denormalize(others_vels, False) * 100)

        relative_locs = neighbor_locs - my_cur_loc
        relative_vels = neighbor_vels - my_cur_vel
        TTCAs = np.empty((0, 1))
        DCAs = np.empty((0, 1))
        RISKs = np.empty((0, 1))
        for j in range(relative_locs.shape[0]):
            _ttca_j = - np.dot(relative_locs[j], relative_vels[j]) / (LA.norm(relative_vels[j]) + eps)
            _ttca_j = max(_ttca_j, 0)
            _dca_j = exp(-LA.norm(_ttca_j * relative_vels[j] + relative_locs[j]) / 0.40)
            _dpn = relative_locs[j] / (LA.norm(relative_locs[j]) + eps)
            _vjn = neighbor_vels[j] / (LA.norm(neighbor_vels[j] + eps))
            _risk_j = 0.5 - 0.5 * np.dot(_dpn, _vjn)

            TTCAs = np.append(TTCAs, _ttca_j)
            DCAs = np.append(DCAs, _dca_j)
            RISKs = np.append(RISKs, _risk_j)

        # print("DPs = \n", relative_locs)
        # print("DVs = \n", relative_vels)
        # print("TTCAs = \n", TTCAs)
        # print("DCAs = \n", DCAs)
        # print("RISKs = \n", RISKs)

        goal_vec = my_goal - my_cur_loc
        goal_ang = atan2(goal_vec[1], goal_vec[0])
        theta = -goal_ang
        rotation_mat = [[cos(theta), sin(-theta)], [sin(theta), cos(theta)]]

        # print("goal_vec = ", goal_vec)
        # print("goal_ang = ", goal_ang * 180 / np.pi)
        # print("rot_mat = ", rotation_mat)

        cartesian_grid[int(round(goal_vec[0] * grid_size[0] / 2 + cnt[0])),
                       int(round(goal_vec[1] * grid_size[1] / 2 + cnt[1])), 2] = 255

        rot_goal = np.matmul(rotation_mat, goal_vec)
        if abs(rot_goal[0]) >= 1:
            rot_goal = rot_goal / rot_goal[0]
        aligned_cartesian_grid[int(round(rot_goal[0] * grid_size[0] / 2 + cnt[0])),
                               int(round(rot_goal[1] * grid_size[1] / 2 + cnt[1])), 2] = 255

        goal_rad = LA.norm(rot_goal)
        goal_ang = atan2(rot_goal[1], rot_goal[0])
        aligned_polar_grid[int(round(goal_rad * grid_size[0] / sqrt(2))),
                           int(round(goal_ang * grid_size[1] / (2 * np.pi) + cnt[1])), 2] = 255

        for j in range(relative_locs.shape[0]):
            cartesian_grid[int(round(relative_locs[j][0] * grid_size[0] / 2 + cnt[0])),
                           int(round(relative_locs[j][1] * grid_size[0] / 2 + cnt[1])), 0] += RISKs[j] * PIX_SCALE

            # Rotate
            [rot_x, rot_y] = np.matmul(rotation_mat, relative_locs[j])
            r = sqrt(rot_x ** 2 + rot_y ** 2)
            th = atan2(rot_y, rot_x)
            #polar_loc = np.array([r, th])

            rot_x_coord = int(round(rot_x * grid_size[0] / 2 + cnt[0]))
            rot_y_coord = int(round(rot_y * grid_size[1] / 2 + cnt[1]))
            aligned_cartesian_grid[rot_x_coord, rot_y_coord, 0] += RISKs[j] * PIX_SCALE

            r_coord = int(round(r * grid_size[0] / sqrt(2)))
            th_coord = int(round(th * grid_size[1] / (2 * np.pi) + cnt[1]))
            aligned_polar_grid[r_coord, th_coord, 0] += RISKs[j] * PIX_SCALE

        # Rotate images to display
        cartesian_grid = ndimage.rotate(cartesian_grid, 90)
        aligned_cartesian_grid = ndimage.rotate(aligned_cartesian_grid, 90)
        aligned_polar_grid = ndimage.rotate(aligned_polar_grid, 90)

        fig = plt.gcf()
        fig.canvas.set_window_title('Seyfried - Ped %d' % i)
        fig.set_size_inches(14, 5)
        plt.subplot(1, 3, 1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Centered on Agent')
        plt.imshow(cartesian_grid)
        plt.text(-1, -5, 'Seyfried - Ped %d Frame = %d' % (i, t))
        plt.subplot(1, 3, 2)
        plt.xlabel('agent_x')
        plt.ylabel('agent_y')
        plt.title('Aligned to Agent Orien')
        plt.imshow(aligned_cartesian_grid)
        plt.subplot(1, 3, 3)
        plt.xlabel('radius')
        plt.ylabel('angle')
        plt.title('In Polar Coord')
        plt.imshow(aligned_polar_grid)
        plt.show()
        plt.savefig('../outputs/tmp.png')
        gif_writer.append_data(imageio.imread('../outputs/tmp.png'))

    exit(1)
