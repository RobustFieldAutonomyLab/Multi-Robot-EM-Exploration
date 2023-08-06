import numpy as np
import gtsam
import math
import copy
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist

from marinenav_env.envs.utils.robot import Odometry, RangeBearingMeasurement, RobotNeighborMeasurement
from nav.utils import get_symbol, point_to_local, world_to_local_values, point_to_world

DEBUG_FRONTIER = False
DEBUG_EM = False


class Frontier:
    def __init__(self, position, origin=None, relative=None, meet_robot_id=None, nearest_frontier=False):
        self.position = position
        # relative position in SLAM framework
        self.position_local = None
        self.selected = False
        self.nearest_frontier = nearest_frontier
        if origin is not None:
            self.position_local = point_to_local(position[0], position[1], origin)

        self.connected_robot = []
        self.relatives = []
        if meet_robot_id is not None:
            self.connected_robot.append(meet_robot_id)
        if relative is not None:
            self.relatives.append(relative)

    def add_relative(self, relative):
        self.relatives.append(relative)

    def add_robot_connection(self, robot_id):
        self.connected_robot.append(robot_id)


def compute_distance(frontier_p, robot_p):
    if not isinstance(robot_p, tuple):
        dist_cost = np.sqrt((robot_p.x() - frontier_p[0]) ** 2 + (robot_p.y() - frontier_p[1]) ** 2)
    else:
        # if a rendezvous point, calculate the mean cost for both robots
        dist_cost = np.sqrt((robot_p[0].x() - frontier_p[0]) ** 2 + (robot_p[0].y() - frontier_p[1]) ** 2)
        dist_cost += np.sqrt((robot_p[1].x() - frontier_p[0]) ** 2 + (robot_p[1].y() - frontier_p[1]) ** 2)
        dist_cost /= 2
    return dist_cost


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
        self.allocation_max_distance = 30

        self.max_distance_scaled = float(self.max_distance) / float(self.cell_size)
        self.min_distance_scaled = float(self.min_distance) / float(self.cell_size)

        self.free_threshold = 0.45
        self.obstacle_threshold = 0.55
        self.neighbor_kernel = np.array([[0, 1, 0],
                                         [1, 0, 1],
                                         [0, 1, 0]])
        self.max_visited_neighbor = 2

        self.virtual_move_length = 0.5

        self.d_weight = 100
        self.t_weight = 5000

        self.boundary_ratio = 0.1  # Avoid frontiers near boundaries since our environment actually do not have boundary
        self.boundary_value_j = int(self.boundary_ratio / self.cell_size * (self.max_x - self.min_x))
        self.boundary_value_i = int(self.boundary_ratio / self.cell_size * (self.max_y - self.min_y))
        if DEBUG_FRONTIER:
            print("boundary value: ", self.boundary_value_i, self.boundary_value_j)

        self.frontiers = None
        self.select_rendezvous = True
        self.goal_history = [[] for _ in range(self.num_robot)]

    def position_2_index(self, position):
        index_j = int(math.floor((position[0] - self.min_x) / self.cell_size))
        index_i = int(math.floor((position[1] - self.min_y) / self.cell_size))
        return [index_i, index_j]

    def index_2_position(self, index):
        x = index[1] * self.cell_size + self.cell_size * .5 + self.min_x
        y = index[0] * self.cell_size + self.cell_size * .5 + self.min_y
        return [x, y]

    def generate(self, robot_id, probability_map, state_list, landmark_list, axis=None):
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

        state = state_list[robot_id][0]
        state_index = self.position_2_index([state.x(), state.y()])
        distances = cdist([state_index], indices, metric='euclidean')[0]
        if DEBUG_FRONTIER:
            print("robot ", robot_id)
            print("distances: ", distances)
        indices_distances_within_list = np.argwhere(
            np.logical_and(self.min_distance_scaled < distances, distances < self.max_distance_scaled))

        if not self.select_rendezvous:
            for index_in_range in indices_distances_within_list:
                if index_in_range[0] not in self.frontiers:
                    position_this = self.index_2_position(indices[index_in_range[0]])
                    self.frontiers[index_in_range[0]] = Frontier(position_this,
                                                                 origin=self.origin,
                                                                 nearest_frontier=True)
        # Type 1 frontier, the nearest frontier to current position
        index_this = np.argmin(distances)
        position_this = self.index_2_position(indices[index_this])
        if DEBUG_FRONTIER:
            print("index: ", index_this, position_this)

        if index_this in self.frontiers:
            self.frontiers[index_this].nearest_frontier = True
        else:
            self.frontiers[index_this] = Frontier(position_this,
                                                  origin=self.origin,
                                                  nearest_frontier=True)

        if self.select_rendezvous:
            for j in range(0, self.num_robot):
                if j == robot_id:
                    continue
                    # TODO: finish this part
                indices_distances_within_list_other = np.argwhere(
                    np.logical_and(self.min_distance_scaled < distances, distances < self.max_distance_scaled))
                # Type 3 frontier, potential rendezvous point, within certain range for both robots
                indices_rendezvous = np.intersect1d(indices_distances_within_list, indices_distances_within_list_other)
                for index_this in indices_rendezvous:
                    if index_this not in self.frontiers:
                        position_this = self.index_2_position(indices[index_this])
                        self.frontiers[index_this] = Frontier(position_this, origin=self.origin)
                    self.frontiers[index_this].add_robot_(j)

        for landmark in landmark_list:
            landmark_index = self.position_2_index([landmark[1], landmark[2]])
            distances = cdist([landmark_index], indices, metric='euclidean')[0]
            # Type 2 frontier, visitation of existing landmarks
            index_this = np.argmin(distances)
            position_this = self.index_2_position(indices[index_this])
            if index_this in self.frontiers:
                self.frontiers[index_this].add_relative(landmark[0])
            else:
                self.frontiers[index_this] = Frontier(position_this, origin=self.origin, relative=landmark[0])
            distances = cdist([position_this], [state.x(), state.y()], metric='euclidean')[0]
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

        if DEBUG_FRONTIER:
            for frontier in self.frontiers.values():
                print("frontier: ", frontier.connected_robot)

        return explored_ratio

    def find_frontier_nearest_neighbor(self, robot_id):
        for value in self.frontiers.values():
            if value.connected_robot:
                if value.nearest_frontier is not None:
                    if robot_id in value.nearest_frontier:
                        return value.position

    def choose(self, landmark_list_local, isam, robot_state_idx_position_local, virtual_map, axis=None):
        goals = [[] for _ in range(self.num_robot)]
        if self.nearest_frontier_flag:
            for i in range(0, self.num_robot):
                goals[i] = self.find_frontier_nearest_neighbor(i)
            return goals

        emt = ExpectationMaximizationTrajectory(radius=self.radius,
                                                robot_state_idx_position=robot_state_idx_position_local,
                                                landmarks=landmark_list_local,
                                                isam=isam)
        is_goal_decided = [False for _ in range(self.num_robot)]
        if DEBUG_FRONTIER:
            axis.cla()
        for robot_id in range(self.num_robot):
            if is_goal_decided[robot_id]:
                continue
            cost_list = []
            for key, frontier in self.frontiers.items():
                if frontier.selected:
                    continue
                for id_pair in frontier.connected_robot_pair:
                    if robot_id in id_pair:
                        result, marginals = emt.do(robot_id=id_pair,
                                                   frontier_position=(frontier.position_local, frontier.position_local),
                                                   origin=self.origin,
                                                   axis=axis)
                        cost_this = self.compute_utility(virtual_map, result, marginals,
                                                         (robot_state_idx_position_local[1][id_pair[0]],
                                                          robot_state_idx_position_local[1][id_pair[1]]),
                                                         frontier.position_local, robot_id)
                        if id_pair[0] == robot_id:
                            cost_list.append((key, cost_this, id_pair[1]))
                        else:
                            cost_list.append((key, cost_this, id_pair[0]))
                if robot_id in frontier.connected_robot:
                    # return the transformed virtual SLAM result for the calculation of the information of virtual map
                    result, marginals = emt.do(robot_id=robot_id,
                                               frontier_position=frontier.position_local,
                                               origin=self.origin,
                                               axis=axis)
                    # no need to reset, since the update_information will reset the virtual map
                    cost_this = self.compute_utility(virtual_map, result, marginals,
                                                     robot_state_idx_position_local[1][robot_id],
                                                     frontier.position_local, robot_id)
                    cost_list.append((key, cost_this))
            if cost_list == []:
                # if no frontier is available, return the nearest frontier
                goals[robot_id] = self.find_frontier_nearest_neighbor(robot_id)
            else:
                min_cost = min(cost_list, key=lambda tuple_item: tuple_item[1])
                goals[robot_id] = self.frontiers[min_cost[0]].position
                is_goal_decided[robot_id] = True
                if len(min_cost) > 2:
                    goals[min_cost[2]] = self.frontiers[min_cost[0]].position
                    is_goal_decided[min_cost[2]] = True
                    # This means the robot is going to a rendezvous point
        for i in range(self.num_robot):
            self.goal_history[i].append(goals[i])
        return goals

    def compute_utility(self, virtual_map, result: gtsam.Values, marginals: gtsam.Marginals,
                        robot_p, frontier_p, robot_id):
        # robot_position could be a tuple of two robots
        # calculate the cost of the frontier for a specific robot

        virtual_map.update_information(result, marginals)
        # TODO: figure out a way to normalize the uncertainty
        u_m = virtual_map.get_sum_uncertainty()
        u_t = self.compute_utility_task_allocation(frontier_p, robot_id)
        # calculate the landmark visitation and new exploration case first
        u_d = compute_distance(frontier_p, robot_p)
        if DEBUG_EM:
            print("uncertainty & task allocation & distance cost: ", u_m, u_t, u_d)
        return u_m - u_t * self.t_weight + u_d * self.d_weight

    def compute_utility_task_allocation(self, frontier_p, robot_id):
        # local frame to global frame
        frontier_w = point_to_world(frontier_p[0], frontier_p[1], self.origin)
        u_t = 1
        for i, goal_list in enumerate(self.goal_history):
            if i == robot_id:
                continue
            for goal in goal_list:
                dist = np.sqrt((goal[0] - frontier_w[0]) ** 2 + (goal[1] - frontier_w[1]) ** 2)
                u_t -= self.compute_P_d(dist)
        return u_t

    def compute_P_d(self, dist):
        if dist < self.allocation_max_distance:
            P_d = 1 - dist / self.allocation_max_distance
        else:
            P_d = 0
        return P_d

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
    def __init__(self, radius, robot_state_idx_position, landmarks, isam: tuple):
        # for odometry measurement
        self.odom_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.01, 0.01, 0.04])

        # for landmark measurement
        self.range_bearing_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.004, 0.1])

        # for inter-robot measurement
        self.robot_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.004])

        self.slam_speed = 2

        # radius for generate virtual landmark and virtual inter-robot observation
        self.radius = radius

        self.landmarks_position = [[item[1], item[2]] for item in landmarks]
        self.landmarks_id = [item[0] for item in landmarks]

        params = gtsam.ISAM2Params()
        params.setFactorization("QR")
        self.isam = gtsam.ISAM2(params)
        self.isam.update(isam[0], isam[1])

        self.new_factor_start_index = self.isam.getVariableIndex().nFactors()

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

        step = int(np.linalg.norm(state_1[0:2] - state_0[0:2]) / self.slam_speed)

        waypoints = np.linspace(state_0, state_1, step)

        return waypoints.tolist()

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

    def waypoints2landmark_observations(self, waypoints, robot_id=None, graph=None, initial_estimate=None):
        # waypoints: [[x, y, theta], ...]
        # landmarks: [id, x, y], ...]
        # robot_id: robot id
        # id_initial: index of the first virtual waypoint in the sequence
        if graph is None:
            graph = gtsam.NonlinearFactorGraph()
        if initial_estimate is None:
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
            distances = cdist([waypoint[0:2]], self.landmarks_position, metric='euclidean')[0]
            landmark_indices = np.argwhere(distances < self.radius)
            if landmark_indices != []:
                # add landmark observation factor
                for landmark_index in landmark_indices:
                    landmark_this = self.landmarks_position[landmark_index]
                    # calculate range and bearing
                    rb = range_bearing_factory.add_noise_point_based(waypoint, landmark_this)
                    graph.add(self.landmark_measurement_gtsam_format(bearing=rb[1], range=rb[0],
                                                                     robot_id=robot_id,
                                                                     id0=id_initial + i,
                                                                     idl=landmark_this[0]))
        return graph, initial_estimate

    def waypoints2robot_observation(self, waypoints: tuple, robot_id: tuple, graph=None):
        # waypoints: ([[x, y, theta], ...], [[x, y, theta], ...])
        # robot_id: (robot id0, robot id1)
        # id_initial: (index robot0, index robot1)
        if graph is None:
            graph = gtsam.NonlinearFactorGraph()
        id_initial = (self.robot_state_idx[robot_id[0]], self.robot_state_idx[robot_id[1]])
        robot_neighbor_factory = RobotNeighborMeasurement(use_noise=False)

        waypoints0 = np.array(waypoints[0])
        waypoints1 = np.array(waypoints[1])

        length0 = len(waypoints0)
        length1 = len(waypoints1)

        if length0 < length1:
            repeat_times = length1 - length0
            repeated_rows = np.repeat(waypoints0[-1:], repeat_times, axis=0)
            waypoints0 = np.concatenate((waypoints0, repeated_rows), axis=0)
        elif length0 > length1:
            repeat_times = length0 - length1
            repeated_rows = np.repeat(waypoints1[-1:], repeat_times, axis=0)
            waypoints1 = np.concatenate((waypoints1, repeated_rows), axis=0)
        distances = np.linalg.norm(waypoints0[:, 0:2] - waypoints1[:, 0:2], axis=1)
        # find out the index of waypoints where robot could observe each other
        robot_indices = np.argwhere(distances < self.radius)
        for robot_index in robot_indices:
            # add robot observation factor
            measurement = robot_neighbor_factory.add_noise_pose_based(waypoints0[robot_index, :].flatten(),
                                                                      waypoints1[robot_index, :].flatten())
            if robot_index >= length0:
                idx0 = id_initial[0] + length0 - 1
            else:
                idx0 = id_initial[0] + robot_index

            if robot_index >= length1:
                idx1 = id_initial[1] + length1 - 1
            else:
                idx1 = id_initial[1] + robot_index
            graph.add(self.robot_measurement_gtsam_format(
                measurement, robot_id, (idx0, idx1)))
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
                virtual_waypoints[i] = self.generate_virtual_waypoints(robot_id[i], frontier_position[i])
                self.waypoints2landmark_observations(virtual_waypoints[i], robot_id[i],
                                                     graph=graph,
                                                     initial_estimate=initial_estimate)
            self.waypoints2robot_observation(tuple(virtual_waypoints), robot_id, graph=graph)
        return graph, initial_estimate

    def optimize_virtual_observation_graph(self, graph: gtsam.NonlinearFactorGraph, initial_estimate: gtsam.Values):
        # optimize the graph
        # helps nothing but remind me to delete everything after using
        isam_copy = self.isam
        isam_copy.update(graph, initial_estimate)
        result = isam_copy.calculateEstimate()
        marginals = gtsam.Marginals(isam_copy.getFactorsUnsafe(), result)

        # Delete the virtual observation factors from this iteration
        factor_index_now = isam_copy.getVariableIndex().nFactors()
        factors_to_remove = list(range(self.new_factor_start_index, factor_index_now))
        isam_copy.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), factors_to_remove)
        return result, marginals

    def do(self, frontier_position, robot_id, origin, axis):
        # return the optimized trajectories of the robots and the marginals for calculation of information  matrix
        graph, initial_estimate = self.generate_virtual_observation_graph(frontier_position=frontier_position,
                                                                          robot_id=robot_id)
        result, marginals = self.optimize_virtual_observation_graph(graph, initial_estimate)
        result = world_to_local_values(result, origin)

        # draw the optimized result
        if DEBUG_FRONTIER:
            scatters_x = []
            scatters_y = []
            for key in result.keys():
                if key < ord('a'):
                    axis.scatter(result.atPoint2(key)[0], result.atPoint2(key)[1], c='r', marker='o')
                else:
                    scatters_x.append(result.atPose2(key).x())
                    scatters_y.append(result.atPose2(key).y())
            axis.scatter(scatters_x, scatters_y, c='b', marker='.', alpha=0.1)

        return result, marginals
