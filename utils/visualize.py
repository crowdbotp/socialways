import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# import cv2
# POS_MSEC = cv2.CAP_PROP_POS_MSEC
# POS_FRAMES = cv2.CAP_PROP_POS_FRAMES


class FakeDisplay:
    def __init__(self, datadir):
        pass
    def grab_frame(self, frame_id):
        pass
    def plot_path(self, path, pid=-1, args=''):
        pass
    def plot_ped(self, pos=(0, 0), pid=-1, color=(0, 0, 192)):
        pass
    def show(self, title='frame'):
        pass
    def add_orig_frame(self, alpha=0.5):
        pass


class Display:
    def __init__(self, datadir):
        Hfile = os.path.join(datadir, "H.txt")
        mapfile = os.path.join(datadir, "map.png")
        obsfile = os.path.join(datadir, "obsmat.txt")
        destfile = os.path.join(datadir, "destinations.txt")


        self.cap = cv2.VideoCapture(os.path.join(datadir, 'zara01.avi'))
        self.H = np.loadtxt(Hfile)
        self.Hinv = np.linalg.inv(self.H)

        self.scale = 1
        S = np.eye(3, 3)
        S[0, 0] = self.scale
        S[1, 1] = self.scale
        self.Hinv = np.matmul(np.matmul(S, self.Hinv), np.linalg.inv(S))

        # frames, timeframes, timesteps, agents = parse_annotations(self.Hinv, obsfile)

        # self.obs_map = create_obstacle_map(mapfile)
        destinations = np.loadtxt(destfile)

        plt.ion()
        plot_prediction_metrics([], [], [])
        self.agent_num = 1
        self.sample_num = 0
        self.last_t = -1
        self.do_predictions = True
        self.draw_all_agents = False
        self.draw_all_samples = True
        self.draw_truth = True
        self.draw_past = True
        self.draw_plan = True
        self.output = []
        self.orig_frame = []

    def set_frame(self, frame):
        self.cap.set(POS_FRAMES, frame)

    def back_one_frame(self):
        frame_num = int(self.cap.get(POS_FRAMES))
        self.set_frame(frame_num - 2)

    def reset_frame(self):
        frame_num = int(self.cap.get(POS_FRAMES))
        self.set_frame(frame_num - 1)

    def next_sample(self):
        self.change_sample(lambda x: x + 1)

    def prev_sample(self):
        self.change_sample(lambda x: x - 1)

    def change_sample(self, fn):
        pass

    def do_frame(self, agent=-1, past_plan=None, with_scores=True, multi_prediction=False):
        pass

    def grab_frame(self, frame_id):
        if self.cap.isOpened():
            self.set_frame(frame_id)
            ret, self.output = self.cap.read()
            self.output = cv2.resize(self.output, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            self.orig_frame = self.output.copy()
            return ret
        return False

    def plot_ped(self, pos=(0, 0), pid=-1, color=(0, 0, 192)):
        pix_loc = to_pixels(self.Hinv, np.array([pos[0], pos[1], 1.]))
        cv2.circle(self.output, pix_loc, 5, color, 1, cv2.LINE_AA)
        if pid >= 0:
            cv2.putText(self.output, '%d' % pid, pix_loc, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0,0,200), 2)

    def plot_path(self, path, pid=-1, args=''):
        color = (255, 255, 255)
        if args.startswith('b'):
            color = (255, 0, 0)
        elif args.startswith('g'):
            color = (0, 255, 0)
        elif args.startswith('r'):
            color = (0, 0, 255)
        elif args.startswith('m'):
            color = (255, 0, 255)
        elif args.startswith('y'):
            color = (0, 255, 255)

        for i in range(len(path)):
            pos_i = path[i, 0:2]
            pix_loc = to_pixels(self.Hinv, np.array([pos_i[0], pos_i[1], 1.]))

            if '--' in args:
                if i != 0:
                    cv2.line(self.output, last_loc, pix_loc, color, 1, cv2.LINE_AA)
                last_loc = pix_loc
            elif '.' in args:
                cv2.circle(self.output, pix_loc, 3, color, -1, cv2.LINE_AA)

            else:
            #elif 'o' in args:
                cv2.circle(self.output, pix_loc, 5, color, 1, cv2.LINE_AA)



                # if pid >= 0 and i == 0:
            #     cv2.putText(self.output, '%d' % pid, pix_loc, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 200), 2)

    def add_orig_frame(self, alpha=0.5):
        self.output = cv2.addWeighted(self.orig_frame, alpha, self.output, 1-alpha, 0)

    def show(self, title='frame'):
        # plt.imshow(self.output)
        # plt.show()
        cv2.imshow(title, self.output)
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break





def plot_prediction_metrics(prediction_errors, path_errors, agents):
    plt.figure(1, (10, 10))
    plt.clf()
    if len(prediction_errors) > 0:
        plt.subplot(2, 1, 1)
        plot_prediction_error('Prediction Error', prediction_errors, agents)

        plt.subplot(2, 1, 2)
        plot_prediction_error('Path Error', path_errors, agents)

        plt.draw()


def plot_prediction_error(title, errors, agents):
    plt.title(title)
    plt.xlabel('Time (frames)');
    plt.ylabel('Error (px)')
    m = np.nanmean(errors, 1)
    lines = plt.plot(errors)
    meanline = plt.plot(m, 'k--', lw=4)
    plt.legend(lines + meanline, ['{}'.format(a) for a in agents] + ['mean'])


def plot_nav_metrics(ped_scores, IGP_scores):
    plt.clf()
    if len(ped_scores) > 0:
        plt.subplot(1, 2, 1)
        plt.title('Path Length (px)')
        plt.xlabel('IGP');
        plt.ylabel('Pedestrian')
        plt.scatter(IGP_scores[:, 0], ped_scores[:, 0])
        plot_diag()

        plt.subplot(1, 2, 2)
        plt.title('Minimum Safety (px)')
        plt.xlabel('IGP');
        plt.ylabel('Pedestrian')
        plt.scatter(IGP_scores[:, 1], ped_scores[:, 1])
        plot_diag()

        plt.draw()


def plot_diag():
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    lim = (min(0, min(xmin, ymin)), max(xmax, ymax))
    plt.plot((0, 1000), (0, 1000), 'k')
    plt.xlim(lim);
    plt.ylim(lim)


def draw_text(frame, pt, frame_txt):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    sz, baseline = cv2.getTextSize(frame_txt, font, scale, thickness)
    baseline += thickness
    lower_left = (pt[0], pt[1])
    pt = (pt[0], pt[1] - baseline)
    upper_right = (pt[0] + sz[0], pt[1] - sz[1] - 2)
    cv2.rectangle(frame, lower_left, upper_right, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(frame, frame_txt, pt, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return lower_left, upper_right


def crossline(curr, prev, length):
    diff = curr - prev
    if diff[1] == 0:
        p1 = (int(curr[1]), int(curr[0] - length / 2))
        p2 = (int(curr[1]), int(curr[0] + length / 2))
    else:
        slope = -diff[0] / diff[1]
        x = np.cos(np.arctan(slope)) * length / 2
        y = slope * x
        p1 = (int(curr[1] - y), int(curr[0] - x))
        p2 = (int(curr[1] + y), int(curr[0] + x))
    return p1, p2


def draw_path(frame, path, color):
    if path.shape[0] > 0:
        prev = path[0]
        for curr in path[1:]:
            loc1 = (int(prev[1]), int(prev[0]))  # (y, x)
            loc2 = (int(curr[1]), int(curr[0]))  # (y, x)
            p1, p2 = crossline(curr, prev, 3)
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
            cv2.line(frame, loc1, loc2, color, 1, cv2.LINE_AA)
            prev = curr


def draw_waypoints(frame, points, color):
    for loc in ((int(y), int(x)) for x, y, z in points):
        cv2.circle(frame, loc, 3, color, -1, cv2.LINE_AA)


def create_obstacle_map(map_png):
    raw_map = np.array(Image.open(map_png))
    return raw_map

# ignored_peds = [171, 216]
ignored_peds = []


def to_pixels(Hinv, loc):
    """
    Given H^-1 and (x, y, z) in world coordinates, returns (c, r) in image
    pixel indices.
    """
    loc = to_image_frame(Hinv, loc).astype(int)
    return loc[1], loc[0]


def to_image_frame(Hinv, loc):
    """
    Given H^-1 and (x, y, z) in world coordinates, returns (u, v, 1) in image
    frame coordinates.
    """
    if loc.ndim > 1:
        loc_tr = np.transpose(loc)
        loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
        return np.transpose(loc_tr/loc_tr[2])  # to pixels (from millimeters)
    else:
        loc = np.dot(Hinv, loc)  # to camera frame
        return loc / loc[2]  # to pixels (from millimeters)



def create_obstacle_map(map_png):
    raw_map = np.array(Image.open(map_png))
    return raw_map


def parse_annotations(Hinv, obsmat_txt):
    mat = np.loadtxt(obsmat_txt)
    num_frames = int(mat[-1, 0] + 1)
    num_times = np.unique(mat[:, 0]).size
    num_peds = int(np.max(mat[:, 1])) + 1
    frames = [-1] * num_frames  # maps frame -> timestep
    timeframes = [-1] * num_times  # maps timestep -> (first) frame
    timesteps = [[] for _ in range(num_times)]  # maps timestep -> ped IDs
    peds = [np.array([]).reshape(0, 4) for _ in range(num_peds)]  # maps ped ID -> (t,x,y,z) path
    frame = 0
    time = -1
    for row in mat:
        if row[0] != frame:
            frame = int(row[0])
            time += 1
            frames[frame] = time
            timeframes[time] = frame
        ped = int(row[1])
        if ped not in ignored_peds:  # TEMP HACK - can cause empty timesteps
            timesteps[time].append(ped)
        loc = np.array([row[2], row[4], 1])
        # loc = util.to_image_frame(Hinv, loc)
        loc = [time, loc[0], loc[1], loc[2]]  # loc[0], loc[1] should be img coords, loc[2] always "1"
        peds[ped] = np.vstack((peds[ped], loc))
    return frames, timeframes, timesteps, peds


def main():
    disp = Display('/home/jamirian/workspace/crowd_sim/tests/eth')
    disp.plot_ped()

    # ============== SET FRAME ==================
    time_length = 30.0
    fps = 25
    frame_seq = 749
    frame_no = (frame_seq / (time_length * fps))

    # The first argument of cap.set(), number 2 defines that parameter for setting the frame selection.
    # Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
    # The second argument defines the frame number in range 0.0-1.0
