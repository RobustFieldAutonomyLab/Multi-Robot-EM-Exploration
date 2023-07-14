import math
import numpy as np
import gtsam
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
import copy
import time

from marinenav_env.envs.utils.robot import RangeBearingMeasurement


def prob_to_logodds(p):
    return math.log(p / (1.0 - p))


def logodds_to_prob(l):
    return math.exp(l) / (1.0 + math.exp(l))


LOGODDS_FREE = prob_to_logodds(0.3)
LOGODDS_UNKNOWN = prob_to_logodds(0.5)
LOGODDS_OCCUPIED = prob_to_logodds(0.7)
MIN_LOGODDS = prob_to_logodds(0.05)
MAX_LOGODDS = prob_to_logodds(0.95)
FREE_THRESH = prob_to_logodds(0.5)
OCCUPIED_THRESH = prob_to_logodds(0.5)


def get_logodds(free: bool):
    if free:
        return LOGODDS_FREE
    else:
        return LOGODDS_OCCUPIED


class VirtualLandmark:
    def __init__(self, probability: float, x: float, y: float):
        self.sigma = 1.2  # for information matrix
        self.updated = False  # True: Covariance once visited
        self.probability = probability  # probability of being actual landmark
        self.x = x
        self.y = y
        self.information = None
        self.reset_information()

    def covariance(self):
        return np.linalg.inv(self.information)

    def update_information(self, information):
        self.information = copy.deepcopy(information)

    def update_probability(self, probability):
        self.probability = copy.deepcopy(probability)

    def reset_information(self, information=None):
        if information is None:
            init_information = np.array([[1 / (self.sigma ** 2), 0], [0, 1 / (self.sigma ** 2)]])
            self.information = init_information
        else:
            self.information = information

    def update_information_weighted(self, information):
        I_0 = self.information
        I_1 = information
        # a = |I_0| b = |I_1|
        a = np.linalg.det(I_0)
        b = np.linalg.det(I_1)
        # I_0 * x = I_1 x = I_0^{-1} * I_1
        # c = a * tr(x)
        c = a * np.trace(cho_solve(cho_factor(I_0), I_1))
        d = a + b - c
        w = 0.5 * (2 * b - c) / d
        if (w < 0 and d < 0) or (w > 1 and d < 0):
            w = float(0)
        else:
            w = float(1)
        self.information = w * I_0 + (1 - w) * I_1

    def set_updated(self, signal=None):
        if signal is None:
            self.updated = True
        else:
            self.updated = signal


class OccupancyMap:
    def __init__(self, min_x: float, min_y: float,
                 max_x: float, max_y: float,
                 cell_size: float, radius: float):
        self.minX = min_x
        self.minY = min_y
        self.maxX = max_x
        self.maxY = max_y
        self.cell_size = cell_size
        self.radius = radius
        self.num_cols = int(math.floor((self.maxX - self.minX) / self.cell_size))
        self.num_rows = int(math.floor((self.maxY - self.minY) / self.cell_size))
        self.data = np.full((self.num_rows, self.num_cols), LOGODDS_UNKNOWN)

    def update_grid(self, col: int, row: int, free: bool):
        logodds = get_logodds(free) + self.data[row, col]
        logodds = min(MAX_LOGODDS, max(logodds, MIN_LOGODDS))
        self.data[row, col] = logodds

    def update_landmark(self, point):
        col = int((point[0] - self.minX) / self.cell_size)
        row = int((point[1] - self.minY) / self.cell_size)
        self.update_grid(col, row, False)

    def update_robot(self, point):
        col = (point[0] - self.minX) / self.cell_size
        row = (point[1] - self.minY) / self.cell_size
        col_int = int(col)
        row_int = int(row)
        radius = math.ceil(self.radius / self.cell_size) + 1
        min_col = max(col_int - radius, 0)
        min_row = max(row_int - radius, 0)
        max_col = min(col_int + radius, self.num_cols)
        max_row = min(row_int + radius, self.num_rows)
        indices = np.indices((max_row - min_row, max_col - min_col)).reshape(2, -1).T
        indices[:, 0] += min_row
        indices[:, 1] += min_col
        indices_float = indices.astype(float)
        indices_float[:, :] += 0.5
        distances = cdist(indices_float, np.array([[row, col]]), 'euclidean').flatten()
        indices_within = np.where(distances < radius)[0]
        for idx in indices_within:
            self.update_grid(indices[idx][1], indices[idx][0], True)

    def reset(self):
        self.data = np.full((self.num_rows, self.num_cols), LOGODDS_UNKNOWN)

    def update(self, slam_result: gtsam.Values):
        for key in slam_result.keys():
            if key < ord('a'):  # landmark case
                self.update_landmark(slam_result.atPoint2(key))
            else:  # robot case
                pose = slam_result.atPose2(key)
                self.update_robot(np.array([pose.x(), pose.y()]))

    def to_probability(self):
        data = np.vectorize(logodds_to_prob)(self.data)
        return data

    def from_probability(self, data):
        self.data = np.vectorize(prob_to_logodds)(data)


def get_range_pose_point(pose, point):
    t_ = pose[:2]
    d = point - t_
    r = np.linalg.norm(d)
    D_r_d = d / r

    c_p = np.cos(pose[2])
    s_p = np.sin(pose[2])

    D_d_pose = np.array([[-c_p, s_p, 0.0],
                         [-s_p, -c_p, 0.0]])
    Hpose = np.matmul(D_r_d, D_d_pose)
    Hpoint = D_r_d

    return r, Hpose, Hpoint


def get_bearing_pose_point(pose, point, jacobian=True):
    if jacobian:
        theta = pose[2]
        d, D_d_pose, D_d_point = unrotate(theta, point - pose[:2])
        result, D_result_d = relative_bearing(d)
        Hpose = np.matmul(D_result_d, D_d_pose)
        Hpoint = np.matmul(D_result_d, D_d_point)
        return result, Hpose, Hpoint
    else:
        theta = pose[2]
        d, _, _ = unrotate(theta, point - pose[:2])
        return relative_bearing(d)


def unrotate(theta, p):
    c = np.cos(theta)
    s = np.sin(theta)
    q = np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1]])
    H1 = np.array([[-1.0, 0.0, q[1]], [0.0, -1.0, -q[0]]])
    H2 = np.array([[c, s], [-s, c]])
    return q, H1, H2


def from_cos_sin(c, s):
    theta = np.arccos(c)
    if s < 0:
        theta = 2 * np.pi - theta
    return theta


def relative_bearing(d, jacobian=True):
    if jacobian:
        x = d[0]
        y = d[1]
        d2 = x ** 2 + y ** 2
        n = np.sqrt(d2)
        if np.abs(n) > 1e-5:
            return from_cos_sin(x / n, y / n), np.array([-y / d2, x / d2])
        else:
            return 0, np.array([0, 0])
    else:
        x = d[0]
        y = d[1]
        d2 = x ** 2 + y ** 2
        n = np.sqrt(d2)
        if np.abs(n) > 1e-5:
            return from_cos_sin(x / n, y / n)
        else:
            return 0


class VirtualMap:
    def __init__(self, parameters):
        self.maxX = parameters["maxX"]
        self.maxY = parameters["maxY"]
        self.minX = parameters["minX"]
        self.minY = parameters["minY"]
        self.radius = parameters["radius"]
        self.cell_size = 5  # cell size for virtual map
        self.num_cols = int(math.floor((self.maxX - self.minX) / self.cell_size))
        self.num_rows = int(math.floor((self.maxY - self.minY) / self.cell_size))

        self.data = np.empty((self.num_rows, self.num_cols), dtype=object)

        self.range_bearing_model = RangeBearingMeasurement()
        # Initialize occupancy map with unknown grid
        for i in range(0, self.num_rows):
            for j in range(0, self.num_cols):
                x = j * self.cell_size + self.cell_size * .5 + self.minX
                y = i * self.cell_size + self.cell_size * .5 + self.minY
                self.data[i, j] = VirtualLandmark(0, x, y)

    def get_parameters(self):
        param = {"maxX": self.maxX, "minX": self.minX, "maxY": self.maxY, "minY": self.minY,
                 "cell_size": self.cell_size, "radius": self.radius}
        return param

    def reset_probability(self, data=None):
        if data is None:
            np.vectorize(lambda obj, prob: obj.update_probability(prob))(self.data, np.zeros(self.data.shape))
        else:
            np.vectorize(lambda obj, prob: obj.update_probability(prob))(self.data, data)

    def reset_information(self):
        np.vectorize(lambda obj: obj.reset_information())(self.data)

    def update_probability(self, slam_result: gtsam.Values):
        occmap = OccupancyMap(self.minX, self.minY,
                              self.maxX, self.maxY,
                              self.cell_size, self.radius)
        occmap.update(slam_result)
        self.reset_probability(occmap.to_probability())

    def update_probability_robot(self, pose):
        occmap = OccupancyMap(self.minX, self.minY,
                              self.maxX, self.maxY,
                              self.cell_size, self.radius)
        occmap.from_probability(self.data[:, :].probability)
        occmap.update_robot([np.array([pose.x(), pose.y()])])

        self.reset_probability(occmap.to_probability())

    def update_probability_data(self, point):
        occmap = OccupancyMap(self.minX, self.minY,
                              self.maxX, self.maxY,
                              self.cell_size, self.radius)
        occmap.from_probability(self.data[:, :].probability)
        occmap.update_landmark(point)
        self.reset_probability(occmap.to_probability())

    def update_information(self, slam_result: gtsam.Values, marginals: gtsam.Marginals):
        np.vectorize(lambda obj, prob: obj.set_updated(prob))(self.data, np.full(self.data.shape, False))
        self.reset_information()
        for key in slam_result.keys():
            if key < ord('a'):  # landmark case
                pass
            else:  # robot case
                pose = slam_result.atPose2(key)
                self.update_information_robot(np.array([pose.x(), pose.y(), pose.theta()]),
                                              marginals.marginalInformation(key))

    def find_neighbor_indices(self, point):
        col = (point[0] - self.minX) / self.cell_size
        row = (point[1] - self.minY) / self.cell_size
        col_int = int(col)
        row_int = int(row)
        radius = math.ceil(self.radius / self.cell_size) + 1
        min_col = max(col_int - radius, 0)
        min_row = max(row_int - radius, 0)
        max_col = min(col_int + radius, self.num_cols)
        max_row = min(row_int + radius, self.num_rows)
        indices = np.indices((max_row - min_row, max_col - min_col)).reshape(2, -1).T
        indices[:, 0] += min_row
        indices[:, 1] += min_col
        indices_float = indices.astype(float)
        indices_float[:, :] += 0.5
        distances = cdist(indices_float, np.array([[row, col]]), 'euclidean').flatten()
        indices_within = np.where(distances < radius)[0]
        return indices[indices_within]

    def pose_2_point_measurement(self, pose: np.ndarray, point, jacobian: bool):
        sigmas = np.array([[self.range_bearing_model.sigma_b], [self.range_bearing_model.sigma_r]])
        if not jacobian:
            bearing = get_bearing_pose_point(pose, point, False)
            range = get_range_pose_point(pose, point, False)
            return [bearing, range, sigmas]
            # bearing
            # range
            # Sigmas 2 by 1
        else:
            bearing, Hx_bearing, Hl_bearing = get_bearing_pose_point(pose, point)
            range, Hx_range, Hl_range = get_range_pose_point(pose, point)
            # bearing
            # range
            # Sigmas 2 by 1
            # Hx_bearing_range 2 by 3
            # Hl_bearing_range 2 by 2
            return [bearing, range, sigmas,
                    np.concatenate(([Hx_bearing], [Hx_range]), axis=0),
                    np.concatenate(([Hl_bearing], [Hl_range]), axis=0)]

    def predict_virtual_landmark(self, state: np.ndarray, information_matrix,
                                 virtual_landmark_position):
        bearing, range, sigmas, Hx, Hl = self.pose_2_point_measurement(state, virtual_landmark_position, True)
        R = np.diag(np.squeeze(sigmas)) ** 2
        # Hl_Hl_Hl = (Hl^T * Hl)^{-1}* Hl^T
        # cov = Hl_Hl_Hl * [Hx * Info_Mat^{-1} * Hx^T + R] * Hl_Hl_Hl^T
        Hl_Hl_Hl = np.matmul(np.linalg.pinv(np.matmul(Hl.transpose(), Hl)), Hl.transpose())
        A = np.matmul(Hx, cho_solve(cho_factor(information_matrix), Hx.transpose())) + R
        cov = np.matmul(np.matmul(Hl_Hl_Hl, A), Hl_Hl_Hl.transpose())
        # print("cov: ",cov)
        return np.linalg.pinv(cov)

    def update_information_robot(self, state: np.ndarray, information_matrix):
        indices = self.find_neighbor_indices(np.array([state[0], state[1]]))
        # time0 = time.time()
        for [i, j] in indices:
            # if self.data[i, j].probability < 0.49:
            #     continue
            info_this = self.predict_virtual_landmark(state, information_matrix,
                                                      np.array([self.data[i, j].x, self.data[i, j].y]))
            if self.data[i, j].updated:
                self.data[i, j].update_information_weighted(info_this)
            else:
                self.data[i, j].reset_information(info_this)
                self.data[i, j].set_updated()
        # time1 = time.time()
        # print(time1 - time0)

    def update(self, values: gtsam.Values, marginals: gtsam.Marginals = None):
        self.update_probability(values)
        if marginals is not None:
            self.update_information(values, marginals)

    def get_probability_matrix(self):
        probability_matrix = np.vectorize(lambda obj: obj.probability)(self.data)
        return probability_matrix

    def get_virtual_map(self):
        return self.data

    def get_sum_uncertainty(self):
        sum_uncertainty = 0.0
        for i in range(0, self.num_rows):
            for j in range(0, self.num_cols):
                sum_uncertainty += np.trace(self.data[i, j].covariance())
        return sum_uncertainty
