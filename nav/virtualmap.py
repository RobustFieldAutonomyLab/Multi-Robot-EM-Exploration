import math
import numpy as np
import gtsam
from scipy.spatial.distance import cdist
import copy


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


class VirtualLandmark:
    def __init__(self, probability: float, x: float, y: float):
        self.sigma = 1.2  # for information matrix
        self.updated = False  # True: Covariance once visited
        self.probability = probability  # probability of being actual landmark
        self.x = x
        self.y = y
        self.init_information = np.array([[1 / (self.sigma ** 2), 0], [0, 1 / (self.sigma ** 2)]])
        self.information = copy.deepcopy(self.init_information)
    def covariance(self):
        return np.linalg.inv(self.information)


def get_logodds(free: bool):
    if free:
        return LOGODDS_FREE
    else:
        return LOGODDS_OCCUPIED


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
        local_mat = self.data[min_row:max_row, min_col:max_col]
        indices = np.argwhere(~np.isnan(local_mat))
        indices[:, 0] += min_row
        indices[:, 1] += min_col
        indices_float = indices.astype(float)
        indices_float[:, :] += 0.5
        distances = cdist(indices_float, np.array([[row, col]]), 'euclidean').flatten()
        indices_within = np.where(distances < radius)[0]
        for idx in indices_within:
            self.update_grid(indices[idx][1], indices[idx][0], False)

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


class VirtualMap:
    def __init__(self, parameters):
        self.maxX = parameters["maxX"]
        self.maxY = parameters["maxY"]
        self.minX = parameters["minX"]
        self.minY = parameters["minY"]
        self.radius = parameters["radius"]
        self.cell_size = 2  # cell size for virtual map
        self.num_cols = int(math.floor((self.maxX - self.minX) / self.cell_size))
        self.num_rows = int(math.floor((self.maxY - self.minY) / self.cell_size))

        self.data = np.empty((self.num_rows, self.num_cols), dtype=object)
        # Initialize occupancy map with unknown grid
        for i in range(0, self.num_rows):
            for j in range(0, self.num_cols):
                x = j * (self.cell_size + .5) + self.minX
                y = i * (self.cell_size + .5) + self.minY
                self.data[i, j] = VirtualLandmark(0, x, y)

    def reset_probability(self):
        self.data[:, :].probability = 0

    def reset_information(self):
        self.data[:, :].information = copy.deepcopy(self.data[0,0].init_information)

    def update_probability(self, slam_result: gtsam.Values):
        occmap = OccupancyMap(self.minX, self.minY,
                              self.maxX, self.maxY,
                              self.cell_size, self.radius)
        occmap.update(slam_result)
        self.data[:, :].probability = copy.deepcopy(occmap.to_probability())

    def update_probability_robot(self, pose):
        occmap = OccupancyMap(self.minX, self.minY,
                              self.maxX, self.maxY,
                              self.cell_size, self.radius)
        occmap.from_probability(self.data[:, :].probability)
        occmap.update_robot([np.array([pose.x(), pose.y()])])
        self.data[:, :].probability = copy.deepcopy(occmap.to_probability())

    def update_probability_data(self, point):
        occmap = OccupancyMap(self.minX, self.minY,
                              self.maxX, self.maxY,
                              self.cell_size, self.radius)
        occmap.from_probability(self.data[:, :].probability)
        occmap.update_landmark(point)
        self.data[:, :].probability = copy.deepcopy(occmap.to_probability())

    def update_information(self, slam_result: gtsam.Values):
        self.data[:, :].updated = False
        self.reset_information()
        for key in slam_result.keys():
            if key < ord('a'):  # landmark case
                pass
            else:  # robot case
                pose = slam_result.atPose2(key)
                self.update_information_robot(np.array([pose.x(), pose.y()]))


    def find_neighbor_indices(self, position):
    def update_information_robot(self, position):