import numpy as np
import gtsam
import math
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
from nav.virtualmap import VirtualMap
import copy
DEBUG = False


class Frontier:
    def __init__(self, position, relative=None, robot_id=None, nearest_frontier=False):
        self.position = position
        self.nearest_frontier = []
        if nearest_frontier and (robot_id is not None):
            self.nearest_frontier.append(robot_id)

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
        self.num_robot = parameters["num_robot"]

        self.nearest_frontier_flag = parameters["nearest_frontier_flag"]

        self.max_distance = 30
        self.min_distance = 20

        self.max_distance_scaled = float(self.max_distance) / float(self.cell_size)
        self.min_distance_scaled = float(self.min_distance) / float(self.cell_size)

        self.free_threshold = 0.45
        self.obstacle_threshold = 0.55
        self.neighbor_kernel = np.array([[0, 1, 0],
                                         [1, 0, 1],
                                         [0, 1, 0]])
        self.max_visited_neighbor = 2

        self.boundary_ratio = 0.1  # Avoid frontiers near boundaries since our environment actually do not have boundary
        self.boundary_value_j = int(self.boundary_ratio / self.cell_size * (self.max_x - self.min_x))
        self.boundary_value_i = int(self.boundary_ratio / self.cell_size * (self.max_y - self.min_y))
        print(self.boundary_value_i, self.boundary_value_j)

        self.frontiers = None

    def position_2_index(self, position):
        index_j = int(math.floor((position[0] - self.min_x) / self.cell_size))
        index_i = int(math.floor((position[1] - self.min_y) / self.cell_size))
        return [index_i, index_j]

    def index_2_position(self, index):
        x = index[1] * self.cell_size + self.cell_size * .5 + self.min_x
        y = index[0] * self.cell_size + self.cell_size * .5 + self.min_y
        return [x, y]

    def generate(self, probability_map, state_list, landmark_list, axis):
        # probability_map: 2d-array, value: [0,1]
        # state_list: [gtsam.Pose2, ...]
        # landmarks_list: [[id, x, y], ...]
        if len(state_list) != self.num_robot:
            raise ValueError("len(state_list) not equal to num of robots!")
        data_free = probability_map < self.free_threshold
        data_occupied = probability_map > self.obstacle_threshold
        data = data_free | data_occupied

        explored_ratio = np.mean(data)
        print("Present exploration ratio: ", explored_ratio)

        neighbor_sum = convolve(data.astype(int), self.neighbor_kernel, mode='constant', cval=0) - data.astype(int)

        data[0:self.boundary_value_i, :] = False  # avoid the boundary being selected
        data[-self.boundary_value_i:-1, :] = False
        data[:, 0:self.boundary_value_j] = False
        data[:, -self.boundary_value_j:-1] = False

        frontier_candidate_pool_data = data & (neighbor_sum < self.max_visited_neighbor)
        indices = np.argwhere(frontier_candidate_pool_data)

        if DEBUG:
            print("indices: ", indices)

        self.frontiers = {}

        indices_distances_within_list = [[] for _ in range(len(state_list))]
        for i, state in enumerate(state_list):
            state_index = self.position_2_index([state.x(), state.y()])
            distances = cdist([state_index], indices, metric='euclidean')[0]
            if DEBUG:
                print("robot ", i)
                print("distances: ", distances)
            indices_distances_within_list[i] = np.argwhere(
                np.logical_and(self.min_distance_scaled < distances, distances < self.max_distance_scaled))

            # Type 1 frontier, the nearest frontier to current position
            index_this = np.argmin(distances)

            position_this = self.index_2_position(indices[index_this])
            if DEBUG:
                print("index: ", index_this, position_this)
            self.frontiers[index_this] = Frontier(position_this, robot_id=i, nearest_frontier=True)

            if i == 0:
                continue
            for j in range(0, i):
                # Type 2 frontier, visitation of existing landmarks
                indices_rendezvous = np.intersect1d(indices_distances_within_list[i], indices_distances_within_list[j])
                for index_this in indices_rendezvous:
                    if index_this not in self.frontiers:
                        position_this = self.index_2_position(indices[index_this])
                        self.frontiers[index_this] = Frontier(position_this)
                    self.frontiers[index_this].add_robot_connection_pair(j, i)

        state_list_array = [[state.x(), state.y()] for state in state_list]

        for landmark in landmark_list:
            landmark_index = self.position_2_index([landmark[1], landmark[2]])
            distances = cdist([landmark_index], indices, metric='euclidean')[0]
            # Type 3 frontier, potential rendervous point, within certain range for both robots
            index_this = np.argmin(distances)
            position_this = self.index_2_position(indices[index_this])
            if index_this in self.frontiers:
                self.frontiers[index_this].add_relative(landmark[0])
            else:
                self.frontiers[index_this] = Frontier(position_this, relative=landmark[0])
            distances = cdist([position_this], state_list_array, metric='euclidean')[0]
            connected_robots = np.argwhere(
                np.logical_and(self.min_distance < distances, distances < self.max_distance))
            if connected_robots != []:
                for robot_id in connected_robots:
                    if robot_id not in self.frontiers[index_this].connected_robot:
                        self.frontiers[index_this].add_robot_connection(robot_id)
            else:
                if self.frontiers[index_this].connected_robot == []:
                    self.frontiers[index_this].add_robot_connection(np.argmin(distances))

        return explored_ratio

    def find_frontier_nearest_neighbor(self, robot_id):
        for value in self.frontiers.values():
            if value.connected_robot:
                if value.nearest_frontier is not None:
                    if robot_id in value.nearest_frontier:
                        return copy.deepcopy(value.position)

    def choose(self):
        goals = [[] for _ in range(self.num_robot)]
        if self.nearest_frontier_flag:
            for i in range(0, self.num_robot):
                goals[i] = self.find_frontier_nearest_neighbor(i)
        return goals

    def return_frontiers_position(self, robot_id):
        frontiers = []
        for value in self.frontiers.values():
            if robot_id in value.connected_robot:
                frontiers.append(value.position)
            for pair in value.connected_robot_pair:
                if robot_id in pair:
                    frontiers.append(value.position)
        return frontiers

    def return_frontiers_rendezvous(self):
        frontiers = []
        for value in self.frontiers.values():
            if value.connected_robot_pair != []:
                frontiers.append(value.position)
        return frontiers
