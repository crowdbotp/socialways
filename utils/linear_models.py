import numpy as np
import torch

# import matplotlib.pyplot as plt
# from pykalman import KalmanFilter
# from .parse_utils import SeyfriedParser, BIWIParser


def predict_cv(obsv, n_next):
    n_past = obsv.shape[1]
    if n_past > 2:
        my_vel = (obsv[:, -1] - obsv[:, -3]) / 2.
    else:
        my_vel = (obsv[:, -1] - obsv[:, -2])

    for si in range(n_next):
        pred_hat = obsv[:, -1] + my_vel
        obsv = torch.cat((obsv, pred_hat.unsqueeze(1)), dim=1)
    pred_hat = obsv[:, n_past:, :]
    return pred_hat


# class MyKalman:
#     def __init__(self, dt, n_iter=4):
#         self.n_iter = n_iter
#         t = dt
#
#         # Const-velocity Model
#         self.A = [[1, 0, t, 0, t ** 2, 0],
#                   [0, 1, 0, t, 0, t ** 2],
#                   [0, 0, 1, 0, t, 0],
#                   [0, 0, 0, 1, 0, t],
#                   [0, 0, 0, 0, 1, 0],
#                   [0, 0, 0, 0, 0, 1]]
#
#         self.C = [[1, 0, 0, 0, 0, 0],
#                   [0, 1, 0, 0, 0, 0]]
#
#         self.Q = [[t**5/20, 0, t**4/8, 0, t**3/6, 0],
#                   [0, t**5/20, 0, t**4/8, 0, t**3/6],
#                   [t**4/8, 0, t**3/3, 0, t**2/2, 0],
#                   [0, t**4/8, 0, t**3/3, 0, t**2/2],
#                   [t**3/6, 0, t**2/2, 0, t/1, 0],
#                   [0, t**3/6, 0, t**2/2, 0, t/1]]
#         self.Q = np.array(self.Q) * 0.5
#
#         # =========== Const-velocity Model ================
#         # self.A = [[1, 0, t, 0],
#         #           [0, 1, 0, t],
#         #           [0, 0, 1, 0],
#         #           [0, 0, 0, 1]]
#         #
#         # self.C = [[1, 0, 0, 0],
#         #           [0, 1, 0, 0]]
#         #
#         # q = 0.0005
#         # self.Q = [[q, 0, 0, 0],
#         #           [0, q, 0, 0],
#         #           [0, 0, q/10, 0],
#         #           [0, 0, 0, q/10]]
#         # =================================================
#
#         r = 1
#         self.R = [[r, 0],
#                   [0, r]]
#
#         self.kf = KalmanFilter(transition_matrices=self.A, observation_matrices=self.C,
#                                transition_covariance=self.Q, observation_covariance=self.R)
#
#     def filter(self, measurement):
#         f = self.kf.em(measurement, n_iter=self.n_iter)
#         (filtered_state_means, filtered_state_covariances) = f.filter(measurement)
#         return filtered_state_means[:, 0:2], filtered_state_means[:, 2:4]
#
#     def smooth(self, measurement):
#         if measurement.shape[0] == 1:
#             return measurement, np.zeros((1, 2))
#
#         f = self.kf.em(measurement, n_iter=self.n_iter)
#         (smoothed_state_means, smoothed_state_covariances) = f.smooth(measurement)
#         return smoothed_state_means[:, 0:2], smoothed_state_means[:, 2:4]

#
# def test_kalman(seyfried=False, biwi=False):
#     if seyfried:
#         parser = SeyfriedParser()
#         pos_data, vel_data, _ = parser.load('../data/sey01.sey', down_sample=1)
#         fps = parser.actual_fps
#     elif biwi:
#         parser = BIWIParser()
#         pos_data, vel_data, _ = parser.load('../data/eth.wap', down_sample=1)
#         fps = parser.actual_fps
#     else:
#         return
#
#
#     index = 20
#     loc_measurement = pos_data[index]
#     vel_measurement = vel_data[index]
#
#
#     dt = 1 / fps
#     kf = MyKalman(dt=dt, n_iter=8)
#     filtered_pos, filtered_vel = kf.filter(loc_measurement)
#     smoothed_pos, smoothed_vel = kf.smooth(loc_measurement)
#
#
#     plt.subplot(1,2,1)
#     plt.plot(loc_measurement[0, 0], loc_measurement[0, 1], 'mo', markersize=5, label='Start Point')
#     plt.plot(loc_measurement[:, 0], loc_measurement[:, 1], 'r', label='Observation')
#     plt.plot(filtered_pos[:, 0], filtered_pos[:, 1], 'y--', label='Filter')
#     plt.plot(smoothed_pos[:, 0], smoothed_pos[:, 1], 'b--', label='Smoother')
#     plt.legend()
#
#     plt.subplot(1,2,2)
#     plt.title("Velocity")
#     plt.plot(smoothed_vel[:, 0], 'b', label='Smoothed Vx')
#     plt.plot(smoothed_vel[:, 1], 'b', label='Smoothed Vy')
#
#     plt.plot(vel_measurement[:, 0], 'g', label='Observed Vx')
#     plt.plot(vel_measurement[:, 1], 'g', label='Observed Vy')
#     plt.legend()
#
#     plt.show()


# if __name__ == '__main__':
#     test_kalman(biwi=True)
