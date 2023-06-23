import numpy as np
import gtsam
import math
from scipy.ndimage import convolve
from virtualmap import VirtualMap


class Frontier:
    def __init__(self, x=0, y=0, theta=0, relative = None, robot_id = None):
        self.pose = gtsam.Pose2(x, y, theta)
        self.connected_robot = []
        self.relatives = []
        if robot_id is not None:
            self.connected_robot.append(robot_id)
        if relative is not None:
            self.relatives.append(relative)

    def add_relative(self, relative):
        self.relatives.append(relative)

    def add_robot_connection(self, robot_id):
        self.connected_robot.append(robot_id)


def arg_nearest_distance(position, position_list):
    distances = np.sum((position_list - position) ** 2)
    index = np.argmin(distances)
    return index


class FrontierGenerator:
    def __init__(self, parameters):
        self.max_x = parameters["maxX"]
        self.max_y = parameters["maxY"]
        self.min_x = parameters["minX"]
        self.min_y = parameters['minY']
        self.cell_size = parameters["cell_size"]

        self.max_distance = 1000
        self.min_distance = 0

        self.free_threshold = 0.45
        self.obstacle_threshold = 0.55
        self.neighbor_kernel = np.array([[0, 1, 0],
                                         [1, 0, 1],
                                         [0, 1, 0]])
        self.max_visited_neighbor = 2

    def position_2_index(self, position):
        index_j = int(math.floor((position[0] - self.min_x) / self.cell_size))
        index_i = int(math.floor((position[1] - self.min_y) / self.cell_size))
        return [index_i, index_j]

    def index_2_position(self, index):
        x = index[1] * self.cell_size + self.cell_size * .5 + self.min_x
        y = index[0] * self.cell_size + self.cell_size * .5 + self.min_y
        return [x, y]

    def generate(self, probability_map, state_list, landmark_list):
        # probability_map: 2d-array, value: [0,1]
        # state_list: [gtsam.Pose2, ...]
        # landmarks_list: [[id, x, y], ...]
        data_free = probability_map < self.free_threshold
        data_free[[0, -1], :] = False # avoid the boundary being selected
        data_free[:, [0, -1]] = False
        data_occupied = probability_map > self.obstacle_threshold
        data = data_free or data_occupied

        neighbor_sum = convolve(data.astype(int), self.neighbor_kernel, mode='constant', cval=0) - data.astype(int)
        indices = np.argwhere(data & (neighbor_sum < self.max_visited_neighbor))
        indices_landmark_list = [] * len(indices)
        indices_robot_list = [] * len(indices)

        # Type 1 frontier, the nearest frontier to current position
        for i, state in enumerate(state_list):
            index = arg_nearest_distance(self.position_2_index([state.x(), state.y()]), indices)
            indices_robot_list[index].append(i)

        # Type 2 frontier, visitation of existing landmarks
        for landmark in landmark_list:
            index = arg_nearest_distance(self.position_2_index([landmark[1], landmark[2]]), indices)
            indices_landmark_list[index].append(landmark[0])

        # Type 3 frontier, potential rendervous point