import math
import numpy as np
import gtsam
from scipy.spatial.distance import cdist

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
        self.information = np.array([[1 / (self.sigma ** 2), 0], [0, 1 / (self.sigma ** 2)]])

    def covariance(self):
        return np.linalg.inv(self.information)


def get_logodds(free :bool):
    if free:
        return LOGODDS_FREE
    else:
        return LOGODDS_OCCUPIED


class OccupancyMap:
    def __init__(self, min_x:float, min_y:float,
                 max_x:float, max_y:float,
                 cell_size:float, radius: float):
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
        logodds = min( MAX_LOGODDS, max(logodds, MIN_LOGODDS))
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
        indices = np.argwhere( ~np.isnan(local_mat) )
        indices[:,0] += min_row
        indices[:,1] += min_col
        indices_float = indices.astype(float)
        indices_float[:,:] += 0.5
        distances = cdist(indices_float, np.array([[row, col]]), 'euclidean').flatten()
        indices_within = np.where(distances < radius)[0]
        for idx in indices_within:
            self.update_grid(indices[idx][1], indices[idx][0], False)

    def rest(self):
        self.data = np.full((self.num_rows, self.num_cols), LOGODDS_UNKNOWN)
    def update(self, slam_result: gtsam.Values):
        for key in slam_result.keys():
            if key < ord('a'):  # landmark case
                self.update_landmark(slam_result.atPoint2(key))
            else:  # robot case
                pose = slam_result.atPose2(key)
                self.update_robot(np.array([pose.x(), pose.y()]))


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
                self.data[i, j] = VirtualLandmark(0.5, x, y)

    def update_probability(self, slam_result: gtsam.Values):
        for key in slam_result.keys():
            if key < ord('a'):  # landmark case
                self.update_probability_landmark(slam_result.atPoint2(key))
            else:  # robot case
                self.update_probability_robot(slam_result.atPose2(key))

    def update_probability_landmark(self, point):
        col = int((point[0] - self.minX) / self.cell_size)
        rows = int((point[1] - self.minY) / self.cell_size)

    def update_probability_robot(self, pose):
        pass

    def update_probability_data(self, col: int, row: int, free: bool):
        pass
