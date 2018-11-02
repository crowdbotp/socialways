import glob
import os
from math import atan2, cos, sin, sqrt, exp

import imageio
from PIL import Image
from scipy import ndimage
from lstm_model.kalman import MyKalman
from lstm_model.learning_utils import MyConfig
from lstm_model.math_utils import cart2pol, norm, unit
from lstm_model.parse_utils import Scale, SeyfriedParser, BIWIParser
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
eps = np.finfo(float).eps
conf = MyConfig()
n_next = conf.n_next

def buildMaps(input_file, out_dir, build_gifs):
    # parser = SeyfriedParser()
    # pos_data, vel_data, time_data = parser.load(input_file)
    parser = BIWIParser()
    pos_data, vel_data, time_data = parser.load(input_file)

    # TODO: apply kalman

    # FIXME: SET THE GRID SIZE HERE
    GRID_SIZE = [49, 49]
    cntr = np.floor([GRID_SIZE[0] / 2, GRID_SIZE[1] / 2]).astype(int)
    PIX_VALUE = 100
    NEIGHB_RAD = 5.0

    # scale = parser.scale
    # ==== USE SCALE for 5.0 meter ====
    scale = Scale()
    scale.min_x, scale.min_y, scale.sx, scale.sy = 0., 0., 1 / (2 * NEIGHB_RAD), 1 / (2 * NEIGHB_RAD)

    for i in range(len(pos_data)):
        pos_data[i] = scale.normalize(pos_data[i])
        vel_data[i] = scale.normalize(vel_data[i], False)


    # ========== Prepare Data For Each Pedestrian ===========
    for i in range(0, len(pos_data), 1):
        all_frames_list = list()
        ped_id = parser.all_ids[i]
        print("Build Occupancy Map for Pedestrian %03d : " % ped_id)

        # ================== export GIFs ====================
        if build_gifs:
            gif_writer = imageio.get_writer(out_dir + "risk_%03d.gif" % ped_id, mode='I')

        ped_i_poss = pos_data[i]
        ped_i_vels = vel_data[i]
        my_pos_0 = ped_i_poss[0]
        my_final_goal = ped_i_poss[-1]
        for t_index in range(time_data[i].shape[0]):
            t = time_data[i][t_index]
            my_cur_loc = ped_i_poss[t_index]
            my_cur_vel = ped_i_vels[t_index]
            cur_goal_time_index = min(t_index+n_next, len(ped_i_poss)-1)
            my_cur_goal = ped_i_poss[cur_goal_time_index]

            # Creating the Grids (Centered on the agent)
            cartesian_grid = np.zeros((GRID_SIZE[0], GRID_SIZE[1], 3), dtype="uint8")
            cartesian_grid[cntr[0], cntr[1], 2] = 128

            # The Cartesian grid which is rotated and aligned to the heading of agent
            aligned_cartesian_grid = np.zeros((GRID_SIZE[0], GRID_SIZE[1], 3), dtype="uint8")
            aligned_cartesian_grid[cntr[0], cntr[1], 2] = 128

            # Polar Grid
            aligned_polar_grid = np.zeros((GRID_SIZE[0], GRID_SIZE[1], 3), dtype="uint8")
            aligned_polar_grid[0, cntr[1], 2] = 128

            neighbor_locs = np.empty((0, 2))
            neighbor_vels = np.empty((0, 2))
            for j in range(len(pos_data)):
                t_index_j = np.array(np.where(time_data[j] == t))
                if j == i or t_index_j.size == 0:
                    continue

                dp = pos_data[j][t_index_j[-1]] - my_cur_vel
                if norm(dp) >= 1.:
                    continue
                neighbor_locs = np.vstack((neighbor_locs, pos_data[j][t_index_j[-1]]))
                neighbor_vels = np.vstack((neighbor_vels, vel_data[j][t_index_j[-1]]))

            relative_locs = np.empty((0, 2))
            if neighbor_locs.size:
                relative_locs = neighbor_locs - my_cur_loc
                relative_vels = neighbor_vels - my_cur_vel
            TTCAs = np.empty((0, 1))
            DCAs = np.empty((0, 1))
            approach_rate = np.empty((0, 1))
            for j in range(relative_locs.shape[0]):
                ttca_j = - np.dot(relative_locs[j], relative_vels[j]) / (norm(relative_vels[j]) ** 2 + eps)
                ttca_j = max(ttca_j, 0)
                dca_j = exp(-norm(ttca_j * relative_vels[j] + relative_locs[j]))
                risk_j = 0.5 - 0.5 * np.dot(unit(relative_locs[j]), unit(neighbor_vels[j]))

                TTCAs = np.append(TTCAs, ttca_j)
                DCAs = np.append(DCAs, dca_j)
                approach_rate = np.append(approach_rate, risk_j)



            # ============ 1st Approach: Use Angle to Final Goal ==================
            # goal_vec = my_final_goal - my_cur_loc
            # goal_ang = atan2(goal_vec[1], goal_vec[0])
            # rot_angle = -goal_ang

            # ============ 2nd Approach: Use Angle to Local Goal ==================
            goal_vec = my_cur_goal - my_cur_loc
            goal_ang = atan2(goal_vec[1], goal_vec[0])
            rot_angle = -goal_ang

            # ============ 3rd Approach: Use Instant Orientation ===========
            # rot_angle = atan2(my_cur_vel[1], my_cur_vel[0])

            rot_matrix = [[cos(rot_angle), sin(-rot_angle)], [sin(rot_angle), cos(rot_angle)]]

            cartesian_grid[int(round((goal_vec[0] +1) * cntr[0])),
                           int(round((goal_vec[1] +1) * cntr[1])), 2] = 255

            rot_goal = np.matmul(rot_matrix, goal_vec)
            if abs(rot_goal[0]) >= 1:
                rot_goal = rot_goal / rot_goal[0]
            aligned_cartesian_grid[int(round((rot_goal[0] + 1) * cntr[0])),
                                   int(round((rot_goal[1] + 1) * cntr[1])), 2] = 255

            goal_rad = norm(rot_goal)
            goal_ang = atan2(rot_goal[1], rot_goal[0])
            goal_coord = np.array([round(goal_rad * (GRID_SIZE[0]-1) / sqrt(2)),
                                   round(goal_ang * (GRID_SIZE[1]-1) / (2 * np.pi) + cntr[1])]).astype(int)
            aligned_polar_grid[goal_coord[0], goal_coord[1], 2] = 255

            for j in range(relative_locs.shape[0]):
                # Rotate
                [rot_x, rot_y] = np.matmul(rot_matrix, relative_locs[j])
                r = sqrt(rot_x ** 2 + rot_y ** 2)
                th = atan2(rot_y, rot_x)
                # polar_loc = np.array([r, th])

                if r > 1:
                    continue

                j_coord = np.array([round((relative_locs[j][0] +1) * cntr[0]),
                                    round((relative_locs[j][1] +1) * cntr[1])]).astype(int)
                cartesian_grid[j_coord[0], j_coord[1], 0] += approach_rate[j] * PIX_VALUE
                cartesian_grid[j_coord[0], j_coord[1], 1] += DCAs[j] * PIX_VALUE

                rot_coord = np.array([round((rot_x + 1) * cntr[0]), round((rot_y + 1) * cntr[1])]).astype(int)
                aligned_cartesian_grid[rot_coord[0], rot_coord[1], 0] += approach_rate[j] * PIX_VALUE
                aligned_cartesian_grid[rot_coord[0], rot_coord[1], 1] += DCAs[j] * PIX_VALUE

                polar_coord_0 = int(round(r * (GRID_SIZE[0] - 1) / sqrt(2)))
                polar_coord_1 = int(round(th * (GRID_SIZE[1] - 1) / (2 * np.pi) + cntr[1]))
                aligned_polar_grid[polar_coord_0, polar_coord_1, 0] += approach_rate[j] * PIX_VALUE
                aligned_polar_grid[polar_coord_0, polar_coord_1, 1] += DCAs[j] * PIX_VALUE

            # fig = plt.gcf()
            # fig.canvas.set_window_title('Seyfried - Ped %d' % i)
            # fig.set_size_inches(14, 5)
            # plt.subplot(1, 3, 1)
            # plt.xlabel('x')
            # plt.ylabel('y')
            # plt.title('Centered on Agent')
            # plt.imshow(cartesian_grid)
            # plt.text(-1, -5, 'Seyfried - Ped %d Frame = %d' % (i, t))
            # plt.subplot(1, 3, 2)
            # plt.xlabel('agent_x')
            # plt.ylabel('agent_y')
            # plt.title('Aligned to Agent Orien')
            # plt.imshow(aligned_cartesian_grid)
            # plt.subplot(1, 3, 3)
            # plt.xlabel('radius')
            # plt.ylabel('angle')
            # plt.title('In Polar Coord')
            # plt.imshow(aligned_polar_grid)
            # plt.savefig('../outputs_ETH/tmp.png')
            # plt.show()

            all_frames_list.append(aligned_cartesian_grid)

            if build_gifs:
                # Rotate images to display
                cartesian_grid = ndimage.rotate(cartesian_grid, 90)
                aligned_cartesian_grid = ndimage.rotate(aligned_cartesian_grid, 90)
                aligned_polar_grid = ndimage.rotate(aligned_polar_grid, 90)
                gif_writer.append_data(aligned_cartesian_grid)

        stacked_frames = np.stack(all_frames_list, 0)
        out_file_name = out_dir + 'risk_%03d' % ped_id
        np.save(out_file_name, stacked_frames)
        print('Stored in file: ', out_file_name)


def extractGifFrames(gif_file):
    frames = list()
    frame = Image.open(gif_file)
    nframes = 0
    while frame:
        nframes += 1
        frame.load()
        frames.append(np.asarray(frame))
        try:
            frame.seek(nframes)
        except EOFError:
            break
    return frames


def mergeData(input_file, working_dir):
    parser = BIWIParser()
    pos_data, vel_data, time_data = parser.load(input_file)
    all_ids = np.array(parser.all_ids)

    # gif_maps = working_dir + '*.gif'
    file_list = glob.glob(os.path.join(working_dir, '*.npy'))

    data_set_X = []
    data_set_y = []

    for full_path in sorted(file_list):
        _id = int(full_path.split('_')[-1].split('.')[0])
        ped_index = np.array(np.where(all_ids == _id))[0, 0]
        vels = vel_data[ped_index]
        poss = pos_data[ped_index]
        kf = MyKalman(1 / parser.actual_fps, n_iter=3)
        poss, vels = kf.smooth(poss)
        # delta_vels = vels[1:, :] - vels[:-1, :]
        polar_vels = cart2pol(vels[:, 0], vels[:, 1])
        polar_dvs = polar_vels[1:, :] - polar_vels[:-1, :]

        # all values between -pi and pi
        polar_dvs[np.where(polar_dvs[:, 1] < -np.pi), 1] += 2 * np.pi
        polar_dvs[np.where(polar_dvs[:, 1] > +np.pi), 1] -= 2 * np.pi

        # plt.plot(polar_dvs[:, 0], 'r', label='d_speed')
        # plt.plot(polar_dvs[:, 1], 'b', label='d_angle')
        # plt.legend()
        # plt.title('ped %d' % ped_index)
        # plt.show()

        mats = np.load(full_path)
        mats = mats[1:, :, :, 0:2]
        data_set_X.append(mats)
        data_set_y.append(polar_dvs)

        print('Compute Training data for Pedestrian: ', ped_index)

    X = np.concatenate(data_set_X, axis=0)
    y = np.concatenate(data_set_y, axis=0)

    out_file = working_dir + 'data.npz'
    np.savez(out_file, X=X, y=y)
    if os.path.exists(out_file):
        print('dataset stored successfully!')


# =================== MAIN CODE ====================
inp_file = '../data/eth.wap'
out_dir = '../maps_eth_48_5m/'

# buildMaps(input_file=inp_file, out_dir=out_dir, build_gifs=True)
mergeData(input_file=inp_file, working_dir=out_dir)
