import math
import numpy as np
import gtsam
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
import copy
import time
import torch

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
        return np.linalg.pinv(self.information)

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
            self.updated = False
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
        radius = math.ceil(self.radius / self.cell_size)
        min_col = max(col_int - radius, 0)
        min_row = max(row_int - radius, 0)
        max_col = min(col_int + radius, self.num_cols)
        max_row = min(row_int + radius, self.num_rows)
        # ignore the part outside the region of interest
        if max_row - min_row <= 0 or max_col - min_col <= 0:
            return
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
    if r < 1e-6:
        return r, np.zeros(3), np.zeros(2)
    D_r_d = d / r

    c_p = np.cos(pose[2])
    s_p = np.sin(pose[2])

    D_d_pose = np.array([[-c_p, s_p, 0.0],
                         [-s_p, -c_p, 0.0]])
    Hpose = np.matmul(D_r_d, D_d_pose)
    Hpoint = D_r_d

    return r, Hpose, Hpoint


def get_range_pose_point_batch(pose_batch, point_batch):
    t_batch = pose_batch[:, :2]
    d_batch = point_batch - t_batch
    r_batch = torch.norm(d_batch, dim=1)

    D_r_d_batch = d_batch / r_batch.unsqueeze(-1)

    c_p_batch = torch.cos(pose_batch[:, 2])
    s_p_batch = torch.sin(pose_batch[:, 2])

    D_d_pose_batch = torch.stack([
        torch.stack([-c_p_batch, s_p_batch, torch.zeros_like(c_p_batch)]),
        torch.stack([-s_p_batch, -c_p_batch, torch.zeros_like(c_p_batch)])
    ], dim=0)
    D_d_pose_batch = D_d_pose_batch.permute(2, 0, 1)

    Hpose_batch = torch.matmul(D_r_d_batch.unsqueeze(1), D_d_pose_batch)
    Hpoint_batch = D_r_d_batch

    return nan_2_zero(r_batch), nan_2_zero(Hpose_batch.squeeze()), nan_2_zero(Hpoint_batch)


def nan_2_zero(x):
    mask_nan = torch.isnan(x)
    return torch.where(mask_nan, torch.tensor(0.0), x)


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


def get_bearing_pose_point_batch(pose_batch, point_batch):
    theta_batch = pose_batch[:, 2]
    d_batch, D_d_pose_batch, D_d_point_batch = unrotate_batch(theta_batch, point_batch - pose_batch[:, :2])
    result_batch, D_result_d_batch = relative_bearing_batch(d_batch)
    Hpose_batch = torch.matmul(D_result_d_batch.unsqueeze(1), D_d_pose_batch)
    Hpoint_batch = torch.matmul(D_result_d_batch.unsqueeze(1), D_d_point_batch)
    return result_batch, Hpose_batch.squeeze(), Hpoint_batch.squeeze()


def unrotate(theta, p):
    c = np.cos(theta)
    s = np.sin(theta)
    q = np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1]])
    H1 = np.array([[-1.0, 0.0, q[1]], [0.0, -1.0, -q[0]]])
    H2 = np.array([[c, s], [-s, c]])
    return q, H1, H2


def unrotate_batch(theta_batch, p_batch):
    c = torch.cos(theta_batch)
    s = torch.sin(theta_batch)
    q = torch.stack([c * p_batch[:, 0] + s * p_batch[:, 1],
                     -s * p_batch[:, 0] + c * p_batch[:, 1]], dim=1)
    H11 = torch.stack([-torch.ones_like(c), torch.zeros_like(c), q[:, 1]], dim=1)
    H12 = torch.stack([torch.zeros_like(c), -torch.ones_like(c), -q[:, 0]], dim=1)
    H1 = torch.stack([H11, H12], dim=1)
    H2 = torch.stack([torch.stack([c, s], dim=1), torch.stack([-s, c], dim=1)], dim=1)

    return q, H1, H2


def from_cos_sin(c, s):
    theta = np.arccos(c)
    if s < 0:
        theta = 2 * np.pi - theta
    return theta


def from_cos_sin_batch(c_batch, s_batch):
    theta_batch = torch.acos(c_batch)
    mask = s_batch < 0
    theta_batch[mask] = 2 * np.pi - theta_batch[mask]
    return theta_batch


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


def relative_bearing_batch(d_batch):
    x = d_batch[:, 0]
    y = d_batch[:, 1]
    d2 = x ** 2 + y ** 2
    n = torch.sqrt(d2)
    theta_batch = torch.zeros_like(n)
    H = torch.stack([torch.zeros_like(n), torch.zeros_like(n)], dim=1)
    mask = n > 1e-5
    theta_batch[mask] = from_cos_sin_batch(x[mask] / n[mask], y[mask] / n[mask])
    H[mask, :] = torch.stack([-y[mask] / d2[mask], x[mask] / d2[mask]], dim=1)
    return theta_batch, H


def pose_2_point_measurement(pose: np.ndarray, point, sigma_b, sigma_r, jacobian: bool):
    sigmas = np.array([[sigma_b], [sigma_r]])
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


def pose_2_point_measurement_batch(pose_batch, point_batch):
    n1, n2, n3 = point_batch.shape
    n5, n4 = pose_batch.shape
    pose_batch = pose_batch.repeat(1, n2).reshape(n5,n2,n4)

    bearing_batch, Hx_bearing_batch, Hl_bearing_batch = get_bearing_pose_point_batch(pose_batch.view(-1, n4),
                                                                                     point_batch.view(-1, n3))
    range_batch, Hx_range_batch, Hl_range_batch = get_range_pose_point_batch(pose_batch.view(-1, n4),
                                                                             point_batch.view(-1, n3))
    # bearing
    # range
    # Sigmas 2 by 1
    # Hx_bearing_range 2 by 3
    # Hl_bearing_range 2 by 2
    Hx = torch.stack([Hx_bearing_batch, Hx_range_batch], dim=1)
    Hl = torch.stack([Hl_bearing_batch, Hl_range_batch], dim=1)
    return [bearing_batch.view(n1, n2), range_batch.view(n1, n2), Hx.view(n1, n2, 2, 3), Hl.view(n1, n2, 2, 2)]


def predict_virtual_landmark(state: np.ndarray, information_matrix,
                             virtual_landmark_position, sigma_range, sigma_bearing):
    bearing, range, sigmas, Hx, Hl = pose_2_point_measurement(state,
                                                              virtual_landmark_position,
                                                              sigma_b=sigma_bearing,
                                                              sigma_r=sigma_range,
                                                              jacobian=True)
    try:
        R = np.diag(np.squeeze(sigmas)) ** 2
        # Hl_Hl_Hl = (Hl^T * Hl)^{-1}* Hl^T
        # cov = Hl_Hl_Hl * [Hx * Info_Mat^{-1} * Hx^T + R] * Hl_Hl_Hl^T
        Hl_Hl_Hl = np.matmul(np.linalg.pinv(np.matmul(Hl.transpose(), Hl)), Hl.transpose())
        A = np.matmul(Hx, cho_solve(cho_factor(information_matrix), Hx.transpose())) + R
        cov = np.matmul(np.matmul(Hl_Hl_Hl, A), Hl_Hl_Hl.transpose())
    except:
        print("Hx, Hl:", Hx, Hl)
        print("R:", R)
    return np.linalg.pinv(cov)


# process virtual landmark data in batch
def predict_virtual_landmark_batch(Hx, Hl, sigmas, information_matrix):
    # Compute R for the entire batch
    R_batch = torch.diag_embed(sigmas) ** 2

    # Compute Hl_Hl_Hl for the entire batch
    Hl_transpose = Hl.transpose(2, 3)
    Hl_Hl_Hl_batch = torch.matmul(torch.pinverse(torch.matmul(Hl_transpose, Hl)), Hl_transpose)
    L_batch = torch.linalg.cholesky(information_matrix)

    # Compute A for the entire batch
    A_batch = torch.matmul(Hx, torch.cholesky_solve(Hx.transpose(2, 3),
                                                    L_batch[:, None, :, :])) + R_batch[None, None, :, :]

    # Compute cov for the entire batch
    cov_batch = torch.matmul(torch.matmul(Hl_Hl_Hl_batch, A_batch), Hl_Hl_Hl_batch.transpose(2, 3))
    info_batch = torch.pinverse(cov_batch)
    return info_batch


class VirtualMap:
    def __init__(self, parameters):
        self.maxX = parameters["maxX"]
        self.maxY = parameters["maxY"]
        self.minX = parameters["minX"]
        self.minY = parameters["minY"]
        self.radius = parameters["radius"]
        self.cell_size = 4  # cell size for virtual map
        self.num_cols = int(math.floor((self.maxX - self.minX) / self.cell_size))
        self.num_rows = int(math.floor((self.maxY - self.minY) / self.cell_size))

        self.data = np.empty((self.num_rows, self.num_cols), dtype=object)

        self.range_bearing_model = RangeBearingMeasurement()
        self.use_torch = True

        if self.use_torch:
            self.indices_within = None
            self.generate_sensor_model()

        # Initialize occupancy map with unknown grid
        for i in range(0, self.num_rows):
            for j in range(0, self.num_cols):
                x = j * self.cell_size + self.cell_size * .5 + self.minX
                y = i * self.cell_size + self.cell_size * .5 + self.minY
                self.data[i, j] = VirtualLandmark(0, x, y)

    def generate_sensor_model(self):
        radius = math.ceil(self.radius / self.cell_size)
        grid = torch.meshgrid(torch.arange(-radius, radius, dtype=torch.int32),
                              torch.arange(-radius, radius, dtype=torch.int32),
                              indexing='ij')
        indices = torch.stack(grid, dim=-1).view(-1, 2)
        indices_float = indices.to(torch.float32)
        distances = torch.norm(indices_float, dim=-1)
        mask = distances < radius
        self.indices_within = indices[mask]

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
        time0 = time.time()
        # if len(slam_result.keys()) * self.cell_size < 100 or not self.use_torch:
        if not self.use_torch:
            for key in slam_result.keys():
                if key < ord('a'):  # landmark case
                    pass
                else:  # robot case
                    pose = slam_result.atPose2(key)
                    self.update_information_robot(np.array([pose.x(), pose.y(), pose.theta()]),
                                                  marginals.marginalInformation(key))
        else:
            poses_array = []
            information_matrix_array = []
            cnt = 0
            for key in slam_result.keys():
                if key < ord('a'):  # landmark case
                    pass
                else:
                    pose = slam_result.atPose2(key)
                    poses_array.append(np.array([pose.x(), pose.y(), pose.theta()]))
                    information_matrix_array.append(marginals.marginalInformation(key))
                    cnt += 1
            self.update_information_robot_batch(torch.tensor(np.array(poses_array)),
                                                torch.tensor(np.array(information_matrix_array)))
        time1 = time.time()
        print("time information: ", time1 - time0)

    def update_information_robot_batch(self, poses, information_matrix):
        indices, points = self.find_neighbor_indices_batch(poses[:, :2])
        _, _, Hx, Hl = pose_2_point_measurement_batch(poses, points)
        sigmas = torch.tensor(np.array([self.range_bearing_model.sigma_b, self.range_bearing_model.sigma_r]))

        info_batch = predict_virtual_landmark_batch(Hx, Hl, sigmas, information_matrix)
        info_batch = info_batch.view(-1, 2, 2)

        for n, [i, j] in enumerate(indices.view(-1, 2).numpy()):
            if i < 0 or i >= self.num_rows or j < 0 or j >= self.num_cols:
                continue
            # not yet part of the map
            if self.data[i, j].probability > 0.49:
                continue
            if self.data[i, j].updated:
                self.data[i, j].update_information_weighted(info_batch[n, :, :])
            else:
                self.data[i, j].reset_information(info_batch[n, :, :])
                self.data[i, j].set_updated()

    def find_neighbor_indices(self, point):
        col = (point[0] - self.minX) / self.cell_size
        row = (point[1] - self.minY) / self.cell_size
        col_int = int(col)
        row_int = int(row)
        radius = math.ceil(self.radius / self.cell_size)
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

    def find_neighbor_indices_batch(self, point_batch):
        col = (point_batch[:, 0] - self.minX) / self.cell_size
        row = (point_batch[:, 1] - self.minY) / self.cell_size
        center_batch = torch.stack([torch.round(row).to(dtype=torch.int32), torch.round(col).to(dtype=torch.int32)], dim=-1)
        radius = math.ceil(self.radius / self.cell_size)
        indices = self.indices_within.clone()
        indices = indices + center_batch[:, None, :]
        return indices, self.indices_batch_2_xy_batch(indices)

    def indices_batch_2_xy_batch(self, indices):
        x = indices[:, :, 1] * self.cell_size + self.cell_size * .5 + self.minX
        y = indices[:, :, 0] * self.cell_size + self.cell_size * .5 + self.minY
        return torch.stack([x, y], dim=-1)

    def update_information_robot(self, state: np.ndarray, information_matrix):
        indices = self.find_neighbor_indices(np.array([state[0], state[1]]))
        for [i, j] in indices:
            # if self.data[i, j].probability < 0.49:
            #     continue

            info_this = predict_virtual_landmark(state,
                                                 information_matrix,
                                                 np.array([self.data[i, j].x, self.data[i, j].y]),
                                                 sigma_range=self.range_bearing_model.sigma_r,
                                                 sigma_bearing=self.range_bearing_model.sigma_b)
            if self.data[i, j].updated:
                self.data[i, j].update_information_weighted(info_this)
            else:
                self.data[i, j].reset_information(info_this)
                self.data[i, j].set_updated()

    def update(self, values: gtsam.Values, marginals: gtsam.Marginals = None):
        self.update_probability(values)
        if marginals is not None:
            self.update_information(values, marginals)

    def get_probability_matrix(self):
        probability_matrix = np.vectorize(lambda obj: obj.probability)(self.data)
        return probability_matrix

    def get_virtual_map(self):
        return self.data

    def get_sum_uncertainty(self, type = "D"):
        sum_uncertainty = 0.0
        for i in range(0, self.num_rows):
            for j in range(0, self.num_cols):
                if self.data[i,j].probability > 0.49:
                    continue
                if type == "A":
                    sum_uncertainty += np.trace(self.data[i, j].covariance())
                elif type == "D":
                    if np.linalg.det(self.data[i, j].information) < 1e-4:
                        # sum_uncertainty += 1000
                        print("information:", self.data[i, j].information, np.linalg.det(self.data[i, j].information))
                        print("covariance:", self.data[i, j].covariance(), np.linalg.det(self.data[i, j].covariance()))
                    else :
                        sum_uncertainty += np.linalg.det(self.data[i, j].covariance())
        return sum_uncertainty
