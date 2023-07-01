import numpy as np
import gtsam
import math
import copy
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist

from marinenav_env.envs.utils.robot import Odometry, RangeBearingMeasurement, RobotNeighborMeasurement
from nav.utils import get_symbol, point_to_local, world_to_local_values

DEBUG_FRONTIER = False
DEBUG_EM = False


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
        self.radius = parameters["radius"]
        self.origin = parameters["origin"]

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

        self.virtual_move_length = 0.5

        self.boundary_ratio = 0.1  # Avoid frontiers near boundaries since our environment actually do not have boundary
        self.boundary_value_j = int(self.boundary_ratio / self.cell_size * (self.max_x - self.min_x))
        self.boundary_value_i = int(self.boundary_ratio / self.cell_size * (self.max_y - self.min_y))
        if DEBUG_FRONTIER:
            print("boundary value: ", self.boundary_value_i, self.boundary_value_j)

        self.frontiers = None
        self.one_robot_per_landmark_frontier = True

    def position_2_index(self, position):
        index_j = int(math.floor((position[0] - self.min_x) / self.cell_size))
        index_i = int(math.floor((position[1] - self.min_y) / self.cell_size))
        return [index_i, index_j]

    def index_2_position(self, index):
        x = index[1] * self.cell_size + self.cell_size * .5 + self.min_x
        y = index[0] * self.cell_size + self.cell_size * .5 + self.min_y
        return [x, y]

    def generate(self, probability_map, state_list, landmark_list, axis=None):
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

        if DEBUG_FRONTIER:
            print("indices: ", indices)

        self.frontiers = {}

        indices_distances_within_list = [[] for _ in range(len(state_list))]
        for i, state in enumerate(state_list):
            state_index = self.position_2_index([state.x(), state.y()])
            distances = cdist([state_index], indices, metric='euclidean')[0]
            if DEBUG_FRONTIER:
                print("robot ", i)
                print("distances: ", distances)
            indices_distances_within_list[i] = np.argwhere(
                np.logical_and(self.min_distance_scaled < distances, distances < self.max_distance_scaled))

            # Type 1 frontier, the nearest frontier to current position
            index_this = np.argmin(distances)

            position_this = self.index_2_position(indices[index_this])
            if DEBUG_FRONTIER:
                print("index: ", index_this, position_this)
            if index_this in self.frontiers:
                self.frontiers[index_this].nearest_frontier.append(i)
                self.frontiers[index_this].connected_robot.append(i)
            else:
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
            if not self.one_robot_per_landmark_frontier:
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

    def choose(self, landmark_list_local, isam: gtsam.ISAM2, robot_state_idx_position_local, axis=None):
        goals = [[] for _ in range(self.num_robot)]
        if self.nearest_frontier_flag:
            for i in range(0, self.num_robot):
                goals[i] = self.find_frontier_nearest_neighbor(i)
            # return goals
        emt = ExpectationMaximizationTrajectory(radius=self.radius,
                                                robot_state_idx_position=robot_state_idx_position_local,
                                                landmarks=landmark_list_local,
                                                isam=isam)
        for robot_id in range(self.num_robot):
            for frontier in self.frontiers.values():
                for id_pair in frontier.connected_robot_pair:
                    if robot_id in id_pair:
                        # transform frontier position to SLAM frame
                        fp = point_to_local((frontier.position[0], frontier.position[1]), self.origin)
                        emt.do(robot_id, fp, self.origin, axis)
                if robot_id in frontier.connected_robot:
                    # transform frontier position to SLAM frame
                    fp = point_to_local((frontier.position[0], frontier.position[1]), self.origin)
                    emt.do(robot_id, fp, self.origin, axis)

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


# Everything in SLAM frame in ExpectationMaximizationTrajectory
class ExpectationMaximizationTrajectory:
    def __init__(self, radius, robot_state_idx_position, landmarks, isam: gtsam.ISAM2):
        # for odometry measurement
        self.odom_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.01, 0.01, 0.04])

        # for landmark measurement
        self.range_bearing_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.004])

        # for inter-robot measurement
        self.robot_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.004])

        # radius for generate virtual landmark and virtual inter-robot observation
        self.radius = radius

        self.landmarks = landmarks

        self.isam = isam

        self.robot_state_idx = robot_state_idx_position[0]
        self.robot_state_position = robot_state_idx_position[1]

    def generate_virtual_waypoints(self, robot_id, state_next):
        # return numpy.ndarray
        # [[x0,y0,theta0], ..., [x1,y1,theta1]]
        state_this = self.robot_state_position[robot_id]
        if isinstance(state_this, gtsam.Pose2):
            state_0 = np.array([state_this.x(), state_this.y(), state_this.theta()])
        elif isinstance(state_this, np.ndarray) or isinstance(state_next, list):
            state_0 = np.array([state_this[0], state_this[1], 0])
        else:
            raise ValueError("Only accept gtsam.Pose2 and numpy.ndarray")
        if isinstance(state_next, gtsam.Pose2):
            state_1 = np.array([state_next.x(), state_next.y(), state_next.theta()])
        elif isinstance(state_next, np.ndarray) or isinstance(state_next, list):
            state_1 = np.array([state_next[0], state_next[1], 0])
        else:
            raise ValueError("Only accept gtsam.Pose2 and numpy.ndarray")

        step = int(np.linalg.norm(state_1[0:2] - state_0[0:2]))

        waypoints = np.linspace(state_0, state_1, step)

        return waypoints

    def odometry_measurement_gtsam_format(self, measurement, robot_id=None, id0=None, id1=None):
        return gtsam.BetweenFactorPose2(get_symbol(robot_id, id0), get_symbol(robot_id, id1),
                                        gtsam.Pose2(measurement[0], measurement[1], measurement[2]),
                                        self.odom_noise_model)

    def landmark_measurement_gtsam_format(self, bearing=None, range=None, robot_id=None, id0=None, idl=None):
        return gtsam.BearingRangeFactor2D(get_symbol(robot_id, id0), idl,
                                          gtsam.Rot2(bearing), range, self.range_bearing_noise_model)

    def robot_measurement_gtsam_format(self, measurement, robot_id: tuple, id: tuple):
        return gtsam.BetweenFactorPose2(get_symbol(robot_id[0], id[0]), get_symbol(robot_id[1], id[1]),
                                        gtsam.Pose2(measurement[0], measurement[1], measurement[2]),
                                        self.robot_noise_model)

    def waypoints2landmark_observations(self, waypoints, robot_id=None):
        # waypoints: [[x, y, theta], ...]
        # landmarks: [id, x, y], ...]
        # robot_id: robot id
        # id_initial: index of the first virtual waypoint in the sequence
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        odometry_factory = Odometry(use_noise=False)
        range_bearing_factory = RangeBearingMeasurement(use_noise=False)
        id_initial = self.robot_state_idx[robot_id]
        for i, waypoint in enumerate(waypoints):
            # calculate virtual odometry between neighbor waypoints
            if i == 0:
                # the first waypoint is same as present robot position, so no need to add into graph
                odometry_factory.reset(waypoint)
                continue

            odometry_factory.add_noise(waypoint[0], waypoint[1], waypoint[2])
            odometry_this = odometry_factory.get_odom()
            # add odometry
            graph.add(self.odometry_measurement_gtsam_format(
                odometry_this, robot_id, id_initial + i - 1, id_initial + i))
            initial_estimate.insert(get_symbol(robot_id, id_initial + i),
                                    gtsam.Pose2(waypoint[0], waypoint[1], waypoint[2]))

            # calculate Euclidean distance between waypoint and landmarks
            distances = cdist([waypoint[0:2]], self.landmarks[:, 1:3], metric='euclidean')[0]
            landmark_indices = np.argwhere(distances < self.radius)
            if landmark_indices != []:
                # add landmark observation factor
                for landmark_index in landmark_indices:
                    landmark_this = self.landmarks[landmark_index]
                    # calculate range and bearing
                    rb = range_bearing_factory.add_noise_point_based(waypoint, landmark_this[2:3])
                    graph.add(self.landmark_measurement_gtsam_format(bearing=rb[1], range=rb[0],
                                                                     robot_id=robot_id,
                                                                     id0=id_initial + i,
                                                                     idl=landmark_this[0]))
        return graph, initial_estimate

    def waypoints2robot_observation(self, waypoints: tuple, robot_id: tuple):
        # waypoints: ([[x, y, theta], ...], [[x, y, theta], ...])
        # robot_id: (robot id0, robot id1)
        # id_initial: (index robot0, index robot1)
        graph = gtsam.NonlinearFactorGraph()
        id_initial = (self.robot_state_idx[robot_id[0]], self.robot_state_idx[robot_id[1]])
        robot_neighbor_factory = RobotNeighborMeasurement(use_noise=False)

        waypoints0 = waypoints[0]
        waypoints1 = waypoints[1]

        length0 = len(waypoints0)
        length1 = len(waypoints1)

        if length0 < length1:
            repeat_times = length1 - length0
            repeated_rows = np.repeat(waypoints0[-1:], repeat_times, axis=0)
            waypoints0 = np.concatenate((waypoints0, repeated_rows), axis=0)
        else:
            repeat_times = length0 - length1
            repeated_rows = np.repeat(waypoints1[-1:], repeat_times, axis=0)
            waypoints1 = np.concatenate((waypoints1, repeated_rows), axis=0)
        distances = np.linalg.norm(waypoints0[0:2] - waypoints1[0:2], axis=1)
        # find out the index of waypoints where robot could observe each other
        robot_indices = np.argwhere(distances < self.radius)
        for robot_index in robot_indices:
            # add robot observation factor
            measurement = robot_neighbor_factory.add_noise_pose_based(waypoints0[robot_index], waypoints1[robot_index])
            graph.add(self.robot_measurement_gtsam_format(
                measurement, robot_id, (id_initial[0] + robot_index, id_initial[1] + robot_index)))
        return graph

    def generate_virtual_observation_graph(self, frontier_position, robot_id):
        # the variables are either all tuples or all not tuples TODO: add safety check for this
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        if not isinstance(frontier_position, tuple):
            virtual_waypoints = self.generate_virtual_waypoints(robot_id, frontier_position)
            graph, initial_estimate = self.waypoints2landmark_observations(virtual_waypoints, robot_id)
        else:
            virtual_waypoints = [[] for _ in range(len(frontier_position))]
            for i in range(len(frontier_position)):
                virtual_waypoints[i] = self.generate_virtual_waypoints(i, frontier_position[i])
                graph_this, initial_estimate_this = self.waypoints2landmark_observations(virtual_waypoints[i],
                                                                                         robot_id[i])
                graph.add(graph_this)
                initial_estimate.insert(initial_estimate_this)
            graph_this = self.waypoints2robot_observation(tuple(virtual_waypoints), robot_id)
            graph.add(graph_this)
        return graph, initial_estimate

    def optimize_virtual_observation_graph(self, graph: gtsam.NonlinearFactorGraph, initial_estimate: gtsam.Values):
        # optimize the graph
        isam_copy = copy.deepcopy(self.isam)
        isam_copy.update(graph, initial_estimate)
        result = isam_copy.calculateEstimate()
        return result

    def do(self, frontier_position, robot_id, origin, axis):
        graph, initial_estimate = self.generate_virtual_observation_graph(frontier_position, robot_id)
        result = self.optimize_virtual_observation_graph(graph, initial_estimate)

        # draw the optimized result
        if DEBUG_EM:
            result = world_to_local_values(result, origin)
            for key in result.keys():
                if key < ord('a'):
                    axis.scatter(result.atPoint2(key)[0], result.atPoint2(key)[1], c='r', marker='^')
                else:
                    self.axis_grid.plot(result.atPose2().x(), result.atPose2().y(),
                                        marker='.', linestyle='-',
                                        alpha=0.3, linewidth=.5)