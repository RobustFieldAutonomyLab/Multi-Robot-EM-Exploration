import numpy as np
import gtsam
import math
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
from virtualmap import VirtualMap


class Frontier:
    def __init__(self, x=0, y=0, theta=0, relative=None, robot_id=None):
        self.pose = gtsam.Pose2(x, y, theta)
        self.connected_robot = []
        self.connected_robot_pair = []
        self.relatives = []
        if robot_id is not None:
            self.connected_robot.append(robot_id)
        if relative is not None:
            self.relatives.append(relative)

    def add_relative(self, relative):
        self.relatives.append(relative)

    def add_robot_connection(self, robot_id):
        self.connected_robot.append(robot_id)

    def add_robot_connection_pair(self, robot_id_0, robot_id_1):
        self.connected_robot_pair.append((robot_id_0, robot_id_1))

class FrontierGenerator:
    def __init__(self, parameters):
        self.max_x = parameters["maxX"]
        self.max_y = parameters["maxY"]
        self.min_x = parameters["minX"]
        self.min_y = parameters['minY']
        self.cell_size = parameters["cell_size"]

        self.max_distance = 1000
        self.min_distance = 0

        self.max_distance_scaled = float(self.max_distance) / float(self.cell_size)
        self.min_distance_scaled = float(self.min_distance) / float(self.cell_size)

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
        data_free[[0, -1], :] = False  # avoid the boundary being selected
        data_free[:, [0, -1]] = False
        data_occupied = probability_map > self.obstacle_threshold
        data = data_free or data_occupied

        neighbor_sum = convolve(data.astype(int), self.neighbor_kernel, mode='constant', cval=0) - data.astype(int)
        indices = np.argwhere(data & (neighbor_sum < self.max_visited_neighbor))
        frontiers = {}

        indices_distances_within_list = [[] for _ in range(len(state_list))]
        for i, state in enumerate(state_list):
            state_index = self.position_2_index([state.x(), state.y()])
            distances = cdist(state_index, indices, metric='euclidean')[0]
            indices_distances_within_list[i] = np.argwhere(self.min_distance_scaled<distances<self.max_distance_scaled)

            # Type 1 frontier, the nearest frontier to current position
            index_this = np.argmin(distances)
            position_this = self.index_2_position(indices[index_this])
            frontiers[index_this] = Frontier(position_this[0], position_this[1], state.theta(), robot_id=i)

            if i == 0:
                continue
            for j in range(0, i):
                # Type 2 frontier, visitation of existing landmarks
                indices_rendezvous = np.intersect1d(indices_distances_within_list[i], indices_distances_within_list[j])
                for index_this in indices_rendezvous:
                    if index_this not in frontiers:
                        position_this = self.index_2_position(indices[index_this])
                        frontiers[index_this] = Frontier(position_this[0], position_this[1], state.theta())
                    frontiers[index_this].add_robot_connection_pair(i, j)

        for landmark in landmark_list:
            landmark_index = self.position_2_index([landmark[1], landmark[2]])
            distances = cdist(landmark_index, indices, metric='euclidean')[0]
            # Type 3 frontier, potential rendervous point, within certain range for both robots
            index_this = np.argmin(distances)
            position_this = self.index_2_position(indices[index_this])
            if index_this in frontiers:
                frontiers[index_this].add_relative(landmark[0])
            else:
                frontiers[index_this] = Frontier(position_this[0], position_this[1], 0, relative=landmark[0])
            distances = cdist(position_this, state_list, metric='euclidean')[0]
            connected_robots = np.argwhere(self.min_distance<distances<self.max_distance)
            if connected_robots != []:
                for robot_id in connected_robots:
                    if robot_id not in frontiers[index_this].connected_robot:
                        frontiers[index_this].add_robot_connection(robot_id)
            else:
                if frontiers[index_this].connected_robot == []:
                    frontiers[index_this].add_robot_connection(np.argmin(distances))

