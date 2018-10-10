from math import atan2, cos, sin, sqrt, exp

from numpy import linalg
from scipy import ndimage

from lstm_model.my_base_classes import Scale, to_supervised, load_seyfried, MyConfig
import numpy as np
import matplotlib.pyplot as plt
import imageio


np.random.seed(7)
p_data, scale, t_data, v_data = load_seyfried('/home/jamirian/workspace/crowd_sim/tests/sey01/sey01.sey')
print(scale.min_y, scale.max_y)

for i in range(len(p_data)):
    p_data[i] = scale.normalize(p_data[i])
    v_data[i] = scale.normalize(v_data[i], False)
data_set = np.array(p_data)

grid_size = [37, 37]
center = np.floor([grid_size[0] / 2, grid_size[1] / 2]).astype(int)

for i in range(25, len(p_data)):
    gif_writer = imageio.get_writer("../outputs/RISK_%d.gif" % i, mode='I')
    fig = plt.gcf()
    fig.set_size_inches(14, 5)
    fig.canvas.set_window_title('Seyfried 01 - Ped %d' % i)

    ped_pos = p_data[i]
    ped_vel = v_data[i]
    my_pos_0 = ped_pos[0]
    my_goal = ped_pos[-1]
    # print('my_pos = ', scale.denormalize(my_pos_0))
    # print('my_goal = ', scale.denormalize(my_gol))
    print('I want to build the occu map for ped %d' % i)
    for t_ind in range(t_data[i].shape[0]):
        t = t_data[i][t_ind]
        my_cur_loc = ped_pos[t_ind]
        my_cur_vel = ped_vel[t_ind]

        # Create Images
        cart_grid = np.zeros((grid_size[0], grid_size[1], 3), dtype="uint8")
        cart_grid[center[0], center[1], 1] = 250

        rot_cart_grid = np.zeros((grid_size[0], grid_size[1], 3), dtype="uint8")
        rot_cart_grid[center[0], center[1], 1] = 250
        # polar image
        polar_grid = np.zeros((grid_size[0], grid_size[1], 3), dtype="uint8")
        polar_grid[0, center[1], 1] = 250

        others_locs = np.empty((0, 2))
        others_vels = np.empty((0, 2))
        for j in range(len(p_data)):
            if j == i:
                continue
            t_ind = np.where(t_data[j] == t)
            t_ind = np.array(t_ind)
            # if t_data[j][0] <= t and t_data[j][-1] >= t:
            if t_ind.size:
                t_ind = t_ind[-1]
                others_locs = np.vstack((others_locs, p_data[j][t_ind]))
                others_vels = np.vstack((others_vels, v_data[j][t_ind]))
                # print('ped %d is also present at frame %d' % (j, t))
        if not others_locs.size:
            continue

        # print('#######################################')
        # print('my cur pos = ', scale.denormalize(my_cur_loc) * 100)
        # print("others pos = \n", scale.denormalize(others_locs) * 100)
        # print("others vel = \n", scale.denormalize(others_vels) * 100)

        relative_locs = others_locs - my_cur_loc
        relative_vels = others_vels - my_cur_vel
        TTCAs = np.empty((0, 1))
        DCAs = np.empty((0, 1))
        RISKs = np.empty((0, 1))
        for j in range(relative_locs.shape[0]):
            ttca_j = - np.dot(relative_locs[j, :], relative_vels[j, :]) \
                     / (linalg.norm(relative_vels[j, :]) + np.finfo(float).eps)
            ttca_j = max(ttca_j, 0)
            dca_j = exp(-linalg.norm(ttca_j * relative_vels[j, :] + relative_locs[j, :]) / 0.40)
            _dpn__ = relative_locs[j, :] / (linalg.norm(relative_locs[j, :]) + np.finfo(float).eps)
            _vjn__ = others_vels[j, :] / (linalg.norm(others_vels[j, :]+ np.finfo(float).eps))
            risk_j = 0.5 - 0.5 * np.dot(_dpn__, _vjn__)

            TTCAs = np.vstack((TTCAs, ttca_j))
            DCAs = np.vstack((DCAs, dca_j))
            RISKs = np.vstack((RISKs, risk_j))
        # print("DPs = \n", relative_locs)
        # print("DVs = \n", relative_vels)
        # print("TTCAs = \n", TTCAs)
        # print("DCAs = \n", DCAs)
        # print("RISKs = \n", RISKs)

        goal_vec = my_goal - my_cur_loc
        goal_ang = atan2(goal_vec[1], goal_vec[0])
        theta = -goal_ang
        rotation_mat = [[cos(theta), sin(-theta)], [sin(theta), cos(theta)]]

        # Display Goal
        # print("goal_vec = ", goal_vec)
        # print("goal_ang = ", goal_ang * 180 / np.pi)
        # print("rot_mat = ", rotation_mat)

        cart_grid[int(round(goal_vec[0] * grid_size[0] / 2 + center[0])),
                  int(round(goal_vec[1] * grid_size[1] / 2 + center[1])), 2] = 250
        rot_goal = np.matmul(rotation_mat, goal_vec)
        if abs(rot_goal[0]) >= 1:
            rot_goal = rot_goal / rot_goal[0]
        rot_cart_grid[int(round(rot_goal[0] * grid_size[0] / 2 + center[0])),
                      int(round(rot_goal[1] * grid_size[1] / 2 + center[1])), 2] = 250

        goal_rad = linalg.norm(rot_goal)
        goal_ang = atan2(rot_goal[1], rot_goal[0])
        polar_grid[int(round(goal_rad * grid_size[0] / sqrt(2))),
                   int(round(goal_ang * grid_size[1] / (2 * np.pi) + center[1])), 2] = 250

        for j in range(relative_locs.shape[0]):
            loc_j = relative_locs[j]
            cart_grid[int(round(loc_j[0] * grid_size[0] / 2 + center[0])),
                      int(round(loc_j[1] * grid_size[0] / 2 + center[1])), 0] += RISKs[j]*100
            # Rotate
            rotated_loc_i = np.matmul(rotation_mat, loc_j)
            rot_x = rotated_loc_i[0]
            rot_y = rotated_loc_i[1]
            r = sqrt(rot_x ** 2 + rot_y ** 2)
            th = atan2(rot_y, rot_x)
            polar_loc = np.array([r, th])

            rot_x_coord = int(round(rot_x * grid_size[0] / 2 + center[0]))
            rot_y_coord = int(round(rot_y * grid_size[1] / 2 + center[1]))
            # rot_cart_grid[rot_x_coord, rot_y_coord, 0] += 80
            rot_cart_grid[rot_x_coord, rot_y_coord, 0] += RISKs[j]*100

            r_coord = int(round(r * grid_size[0] / sqrt(2)))
            th_coord = int(round(th * grid_size[1] / (2 * np.pi) + center[1]))
            # print("polar coordinate = ", r_coord, th_coord)
            # polar_grid[r_coord, th_coord, 0] += 80
            polar_grid[r_coord, th_coord, 0] += RISKs[j]*100

        # Rotate images
        cart_grid = ndimage.rotate(cart_grid, 90)
        rot_cart_grid = ndimage.rotate(rot_cart_grid, 90)
        polar_grid = ndimage.rotate(polar_grid, 90)

        plt.subplot(1, 3, 1)
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.grid(True)
        plt.title('Centered on Agent')
        plt.imshow(cart_grid)
        plt.text(-1, -5, 'Seyfried - Ped %d Frame = %d' % (i, t))
        plt.subplot(1, 3, 2)
        plt.xlabel('agent_x')
        plt.ylabel('agent_y')
        plt.title('Aligned to Agent Orien')
        plt.imshow(rot_cart_grid)
        plt.subplot(1, 3, 3)
        plt.xlabel('radius')
        plt.ylabel('angle')
        plt.title('In Polar Coord')
        plt.imshow(polar_grid)
        # plt.show()
        plt.savefig('../outputs/tmp.png')
        gif_writer.append_data(imageio.imread('../outputs/tmp.png'))

    exit(1)
