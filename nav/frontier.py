import numpy as np
import gtsam
import math
import copy
from scipy.ndimage import convolve
from matplotlib.patches import Rectangle

from marinenav_env.envs.utils.robot import Odometry, RangeBearingMeasurement, RobotNeighborMeasurement
from nav.utils import get_symbol, point_to_local, local_to_world_values, point_to_world, generate_virtual_waypoints
from nav.BSP import BeliefSpacePlanning, DEBUG_BSP

DEBUG_FRONTIER = False
DEBUG_EM = True
PLOT_VIRTUAL_MAP = False


class Frontier:
    def __init__(self, position, origin=None, relative=None, rendezvous=False, nearest_frontier=False):
        self.position = position
        # relative position in SLAM framework
        self.position_local = None
        self.nearest_frontier = nearest_frontier
        if origin is not None:
            self.position_local = point_to_local(position[0], position[1], origin)

        # case rendezvous
        self.rendezvous = rendezvous
        self.waiting_punishment = None
        self.connected_robot = []

        # case landmark
        self.relatives = []

        # if meet_robot_id is not None:
        #     self.connected_robot.append(meet_robot_id)
        if relative is not None:
            self.relatives.append(relative)

    def add_relative(self, relative):
        self.relatives.append(relative)

    def add_waiting_punishment(self, robot_id, punishment):
        self.connected_robot.append(robot_id)
        self.waiting_punishment = punishment

    # def add_robot_connection(self, robot_id):
    #     self.connected_robot.append(robot_id)


def compute_distance(frontier, robot_p):
    frontier_p = frontier.position_local
    dist_cost = np.sqrt((robot_p.x() - frontier_p[0]) ** 2 + (robot_p.y() - frontier_p[1]) ** 2)
    if frontier.rendezvous:
        dist_cost += frontier.waiting_punishment
    return dist_cost


def line_parameters(start_point, end_point):
    x1, y1 = start_point
    x2, y2 = end_point
    A = y2 - y1
    B = x1 - x2
    C = (x2 * y1) - (x1 * y2)
    return A, B, C


def distance_to_line_segment(points, line_params, start_point, end_point):
    A, B, C = line_params
    distances = np.zeros(points.shape[0])
    for i, point in enumerate(points):
        x, y = point
        dot_product = (x - start_point[0]) * (end_point[0] - start_point[0]) + \
                      (y - start_point[1]) * (end_point[1] - start_point[1])
        length_squared = (end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2

        if dot_product < 0:
            distance = np.sqrt((x - start_point[0]) ** 2 + (y - start_point[1]) ** 2)
        elif dot_product > length_squared:
            distance = np.sqrt((x - end_point[0]) ** 2 + (y - end_point[1]) ** 2)
        else:
            distance = np.abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)
        distances[i] = distance

    return distances


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

        self.max_dist_robot = 30
        self.min_dist_landmark = 10
        self.allocation_max_distance = 30

        self.min_landmark_frontier_cnt = 5

        self.max_dist_robot_scaled = float(self.max_dist_robot) / float(self.cell_size)
        self.max_dist_landmark_scaled = float(self.min_dist_landmark) / float(self.cell_size)

        self.free_threshold = 0.45
        self.obstacle_threshold = 0.55
        self.neighbor_kernel = np.array([[0, 1, 0],
                                         [1, 0, 1],
                                         [0, 1, 0]])
        self.max_visited_neighbor = 2

        self.more_frontiers = False

        self.virtual_move_length = 0.5

        self.d_weight = 5
        self.t_weight = 100

        self.u_t_speed = 10
        self.num_history_goal = 5

        self.boundary_dist = 8  # Avoid frontiers near boundaries since our environment actually do not have boundary
        self.boundary_value_j = int(self.boundary_dist / self.cell_size)
        self.boundary_value_i = int(self.boundary_dist / self.cell_size)
        if DEBUG_FRONTIER:
            print("boundary value: ", self.boundary_value_i, self.boundary_value_j)

        self.frontiers = None
        self.goal_history = [[] for _ in range(self.num_robot)]

    def position_2_index(self, position):
        if not isinstance(position, np.ndarray):
            index_j = int(math.floor((position[0] - self.min_x) / self.cell_size))
            index_i = int(math.floor((position[1] - self.min_y) / self.cell_size))
            return [index_i, index_j]
        elif len(position.shape) == 1:
            index_j = int(math.floor((position[0] - self.min_x) / self.cell_size))
            index_i = int(math.floor((position[1] - self.min_y) / self.cell_size))
            return [index_i, index_j]
        else:
            indices = np.zeros_like(position)
            indices[:, 1] = np.floor((position[:, 0] - self.min_x) / self.cell_size)
            indices[:, 0] = np.floor((position[:, 1] - self.min_y) / self.cell_size)
            return indices

    def index_2_position(self, index):
        if not isinstance(index, np.ndarray):
            x = index[1] * self.cell_size + self.cell_size * .5 + self.min_x
            y = index[0] * self.cell_size + self.cell_size * .5 + self.min_y
            return [x, y]
        elif len(index.shape) == 1:
            x = index[1] * self.cell_size + self.cell_size * .5 + self.min_x
            y = index[0] * self.cell_size + self.cell_size * .5 + self.min_y
            return [x, y]
        else:
            positions = np.zeros_like(index)
            positions[:, 0] = index[:, 1] * self.cell_size + self.cell_size * .5 + self.min_x
            positions[:, 1] = index[:, 0] * self.cell_size + self.cell_size * .5 + self.min_y
            return positions

    def generate_potential_frontier_indices(self, probability_map):
        # generate frontier candidates
        data_free = probability_map < self.free_threshold
        data_occupied = probability_map > self.obstacle_threshold
        data = data_free | data_occupied

        explored_ratio = np.mean(data)
        print("Present exploration ratio: ", explored_ratio)

        neighbor_sum = convolve(data.astype(int), self.neighbor_kernel, mode='constant', cval=0) - data.astype(int)
        data[0:self.boundary_value_i, :] = False  # avoid the boundary being selected
        data[-self.boundary_value_i:, :] = False
        data[:, 0:self.boundary_value_j] = False
        data[:, -self.boundary_value_j:] = False

        frontier_candidate_pool_data = data & (neighbor_sum < self.max_visited_neighbor)
        indices = np.argwhere(frontier_candidate_pool_data)
        if DEBUG_FRONTIER:
            print("indices: ", indices)
        return explored_ratio, indices

    def generate(self, robot_id, probability_map, state_list, landmark_list, axis=None):
        # probability_map: 2d-array, value: [0,1]
        # robot_id: int, id of the robot to generate frontiers
        # state_list: [gtsam.Pose2, ...]
        # landmarks_list: [[id, x, y], ...]
        if len(state_list) != self.num_robot:
            print("state_list: ", state_list)
            print("num_robot: ", self.num_robot)
            raise ValueError("len(state_list) not equal to num of robots!")
        explored_ratio, indices = self.generate_potential_frontier_indices(probability_map)
        if PLOT_VIRTUAL_MAP:
            axis.cla()
            for index in indices:
                point_index = np.array(self.index_2_position(index))
                rectangle = Rectangle(point_index - 2, 4, 4, linewidth=.5, edgecolor='tab:blue', facecolor='tab:blue',
                                      alpha=0.3)
                axis.add_patch(rectangle)
        if indices == []:
            return explored_ratio, False

        # clear history
        self.frontiers = {}

        # state and index of the robot who needs a new goal (target robot)
        state = state_list[robot_id]
        state_index = np.array(self.position_2_index([state.x(), state.y()]))

        # distance between the target robot abd frontier candidates
        distances = np.linalg.norm(indices - state_index, axis=1)
        if DEBUG_FRONTIER:
            print("robot ", robot_id)
            print("distances: ", distances)

        # find the frontiers close enough to the target robot
        indices_distances_within_list = np.argwhere(distances < self.max_dist_robot_scaled)

        # Type 1 frontier, the nearest frontier to current position
        index_this = np.argmin(distances)
        position_this = self.index_2_position(indices[index_this])

        if index_this in self.frontiers:
            self.frontiers[index_this].nearest_frontier = True
        else:
            self.frontiers[index_this] = Frontier(position_this,
                                                  origin=self.origin,
                                                  nearest_frontier=True)
        if DEBUG_FRONTIER:
            print("index: ", index_this, position_this)
            print("distances: ", distances)

        for index_in_range in indices_distances_within_list:
            if index_in_range[0] not in self.frontiers:
                position_this = self.index_2_position(indices[index_in_range[0]])
                self.frontiers[index_in_range[0]] = Frontier(position_this, origin=self.origin)

        # Type 2 frontier, re-visitation of existing landmarks
        landmark_frontier_cnt = 0
        # frontiers near landmarks
        for landmark in landmark_list:
            landmark_index = np.array(self.position_2_index([landmark[1], landmark[2]]))
            distances = np.linalg.norm(indices - landmark_index, axis=1)
            # find the candidates close enough to the landmark
            indices_distances_within_list = np.argwhere(distances < self.max_dist_landmark_scaled)

            for index_this in indices_distances_within_list:
                if index_this[0] in self.frontiers:
                    self.frontiers[index_this[0]].add_relative(landmark[0])
                else:
                    position_this = self.index_2_position(indices[index_this[0]])
                    self.frontiers[index_this[0]] = Frontier(position_this, origin=self.origin,
                                                             relative=landmark[0])
                landmark_frontier_cnt += 1

        if (self.more_frontiers or landmark_frontier_cnt < self.min_landmark_frontier_cnt) and landmark_list != []:
            # frontiers with landmarks on the way there
            landmark_array = np.array(landmark_list)[:, 1:]
            landmark_array_index = self.position_2_index(landmark_array)
            distances = np.linalg.norm(landmark_array_index - state_index, axis=1)
            num_landmarks_close = np.count_nonzero(distances < self.max_dist_robot_scaled)
            for i, index_this in enumerate(indices):
                line_params = line_parameters(state_index, index_this)
                distances = distance_to_line_segment(landmark_array_index, line_params, state_index, index_this)
                # if there are landmarks on the way there
                if np.count_nonzero(distances < self.max_dist_landmark_scaled) > num_landmarks_close:
                    landmark_min = np.argmin(distances)
                    if i in self.frontiers:
                        self.frontiers[i].add_relative(landmark_list[landmark_min][0])
                    else:
                        position_this = self.index_2_position(index_this)
                        self.frontiers[i] = Frontier(position_this, origin=self.origin,
                                                     relative=landmark_list[landmark_min][0])
                landmark_frontier_cnt += 1

        # Type 3 frontier,meet with other robots
        if landmark_frontier_cnt < self.min_landmark_frontier_cnt:
            for robot_index in range(self.num_robot):
                if robot_index == robot_id:
                    continue  # you don't need to meet yourself
                if self.goal_history[robot_index] == []:
                    continue  # the robot has no goal
                goal_this = self.goal_history[robot_index][-1]
                # my time to the goal
                dist_this = np.sqrt((state.x() - goal_this[0]) ** 2 + (state.y() - goal_this[1]) ** 2)
                if dist_this < self.max_dist_robot_scaled:
                    continue  # the robot is too close to the goal
                self.frontiers[len(indices) + robot_index] = Frontier(goal_this,
                                                                      origin=self.origin,
                                                                      rendezvous=True)

                # neighbor's time to the goal
                dist_that = np.sqrt((state_list[robot_index].x() - goal_this[0]) ** 2 +
                                    (state_list[robot_index].y() - goal_this[1]) ** 2)
                self.frontiers[len(indices) + robot_index]. \
                    add_waiting_punishment(robot_id=robot_index,
                                           punishment=abs(dist_this - dist_that))

        if DEBUG_FRONTIER:
            for frontier in self.frontiers.values():
                print("frontier: ", frontier.position)

        return explored_ratio, True

    def find_frontier_nearest_neighbor(self):
        for value in self.frontiers.values():
            if value.nearest_frontier:
                return value.position

    def draw_frontiers(self, axis):
        if DEBUG_FRONTIER:
            axis.cla()
            scatters_x = []
            scatters_y = []
            for frontier in self.frontiers.values():
                scatters_x.append(frontier.position[0])
                scatters_y.append(frontier.position[1])
            axis.scatter(scatters_x, scatters_y, c='y', marker='.')
        if PLOT_VIRTUAL_MAP:
            for frontier in self.frontiers.values():
                if frontier.rendezvous:
                    axis.scatter(frontier.position[0], frontier.position[1],
                                 c='tab:yellow', marker='o', s=300, zorder=5)
                elif frontier.relatives == []:
                    axis.scatter(frontier.position[0], frontier.position[1],
                                 c='tab:purple', marker='o', s=300, zorder=5)
                else:
                    axis.scatter(frontier.position[0], frontier.position[1],
                                 c='tab:orange', marker='o', s=300, zorder=5)

    def choose_NF(self, robot_id):
        goal = self.find_frontier_nearest_neighbor()
        self.goal_history[robot_id].append(goal)
        return goal, None

    def choose_CE(self, robot_id, robot_state_idx_position_local):
        robot_p = robot_state_idx_position_local[1][robot_id]

        num_history_goal = min(self.num_history_goal, len(self.goal_history[robot_id]))
        goals_task_allocation = self.goal_history[robot_id][-num_history_goal:]
        goals_task_allocation.append(point_to_world(robot_p.x(), robot_p.y(), self.origin))
        cost_list = []
        for key, frontier in self.frontiers.items():
            u_t = 0
            for goal in goals_task_allocation:
                pts = generate_virtual_waypoints(goal, frontier.position, speed=self.u_t_speed)
                if pts == []:
                    pts = [frontier.position]
                u_t += self.compute_utility_task_allocation(pts, robot_id)
            # calculate the landmark visitation and new exploration case first
            u_d = compute_distance(frontier, robot_p)
            cost_list.append((key, self.t_weight * u_t + self.d_weight * u_d))
        if cost_list == []:
            # if no frontier is available, return the nearest frontier
            goal = self.find_frontier_nearest_neighbor()
        else:
            min_cost = min(cost_list, key=lambda tuple_item: tuple_item[1])
            goal = self.frontiers[min_cost[0]].position
        return goal, None

    def choose_BSP(self, robot_id, landmark_list_local, robot_state_idx_position_local, axis=None):
        if any(not sublist for sublist in self.goal_history):
            goal = self.find_frontier_nearest_neighbor()
            self.goal_history[robot_id].append(goal)
            return goal, None
        newest_goal_local = []
        for goals in self.goal_history:
            newest_goal_local.append(point_to_local(goals[-1][0], goals[-1][1], self.origin))
        robot_state_idx_position_goal_local = robot_state_idx_position_local + (newest_goal_local,)
        bsp = BeliefSpacePlanning(radius=self.radius,
                                  robot_state_idx_position_goal=robot_state_idx_position_goal_local,
                                  landmarks=landmark_list_local)
        robot_waiting = None
        cost_list = []
        for key, frontier in self.frontiers.items():
            result, marginals = bsp.do(robot_id=robot_id,
                                       frontier_position=frontier.position_local,
                                       origin=self.origin,
                                       axis = axis)
            cost_this = self.compute_utility_BSP(result, marginals,
                                                 robot_state_idx_position_local[1][robot_id],
                                                 frontier, robot_id)
            cost_list.append((key, cost_this))
        if cost_list == []:
            goal = self.find_frontier_nearest_neighbor()
        else:
            min_cost = min(cost_list, key=lambda tuple_item: tuple_item[1])
            if self.frontiers[min_cost[0]].rendezvous:
                robot_waiting = self.frontiers[min_cost[0]].connected_robot[0]
            goal = self.frontiers[min_cost[0]].position
        self.goal_history[robot_id].append(goal)
        if DEBUG_BSP:
            with open('log.txt', 'a') as file:
                print("goal: ", goal, file=file)
        return goal, robot_waiting

    def choose_EM(self, robot_id, landmark_list_local, isam, robot_state_idx_position_local, virtual_map, axis=None):
        if any(not sublist for sublist in self.goal_history):
            goal = self.find_frontier_nearest_neighbor()
            self.goal_history[robot_id].append(goal)
            return goal, None

        newest_goal_local = []
        for goals in self.goal_history:
            newest_goal_local.append(point_to_local(goals[-1][0], goals[-1][1], self.origin))
        robot_state_idx_position_goal_local = robot_state_idx_position_local + (newest_goal_local,)
        emt = ExpectationMaximizationTrajectory(radius=self.radius,
                                                robot_state_idx_position_goal=robot_state_idx_position_goal_local,
                                                landmarks=landmark_list_local,
                                                isam=isam)
        self.draw_frontiers(axis)
        robot_waiting = None
        cost_list = []
        U_m_0 = virtual_map.get_sum_uncertainty()
        for key, frontier in self.frontiers.items():
            # return the transformed virtual SLAM result for the calculation of the information of virtual map
            result, marginals = emt.do(robot_id=robot_id,
                                       frontier_position=frontier.position_local,
                                       origin=self.origin,
                                       axis=axis)
            # no need to reset, since the update_information will reset the virtual map
            cost_this = self.compute_utility_EM(virtual_map, result, marginals,
                                                robot_state_idx_position_local[1][robot_id],
                                                frontier, robot_id, U_m_0)
            cost_list.append((key, cost_this))
        if cost_list == []:
            # if no frontier is available, return the nearest frontier
            goal = self.find_frontier_nearest_neighbor()
        else:
            min_cost = min(cost_list, key=lambda tuple_item: tuple_item[1])
            if self.frontiers[min_cost[0]].rendezvous:
                robot_waiting = self.frontiers[min_cost[0]].connected_robot[0]
            goal = self.frontiers[min_cost[0]].position
        self.goal_history[robot_id].append(goal)
        if DEBUG_EM:
            with open('log.txt', 'a') as file:
                print("goal: ", goal, file=file)
        return goal, robot_waiting

    def compute_utility_BSP(self, result: gtsam.Values, marginals: gtsam.Marginals,
                            robot_p, frontier, robot_id):
        u_d = compute_distance(frontier, robot_p)
        u_m = 0
        for key in result.keys():
            if key < gtsam.symbol('a', 0):
                pass
            else:
                pose = result.atPose2(key)
                try:
                    marginal = marginals.marginalInformation(key)
                except RuntimeError:
                    marginal = np.zeros((3, 3))
                u_m += np.sqrt(marginals.marginalCovariance(key).trace())
        if DEBUG_BSP:
            with open('log.txt', 'a') as file:
                print("u_m, u_d: ", u_m, u_d, file=file)
        return u_m + self.d_weight * u_d

    def compute_utility_EM(self, virtual_map, result: gtsam.Values, marginals: gtsam.Marginals,
                           robot_p, frontier, robot_id, U_m_0):
        # robot_position could be a tuple of two robots
        # calculate the cost of the frontier for a specific robot locally
        virtual_map.reset_information()
        virtual_map.update_information(result, marginals)
        u_m = virtual_map.get_sum_uncertainty() - U_m_0
        num_history_goal = min(self.num_history_goal, len(self.goal_history[robot_id]))
        goals_task_allocation = self.goal_history[robot_id][-num_history_goal:]
        goals_task_allocation.append(point_to_world(robot_p.x(), robot_p.y(), self.origin))
        u_t = 0
        for goal in goals_task_allocation:
            pts = generate_virtual_waypoints(goal, frontier.position, speed=self.u_t_speed)
            if pts == []:
                pts = [frontier.position]
            u_t += self.compute_utility_task_allocation(pts, robot_id)
        # calculate the landmark visitation and new exploration case first
        u_d = compute_distance(frontier, robot_p)
        if DEBUG_EM:
            with open('log.txt', 'a') as file:
                print("robot id, robot position, frontier position: ", robot_id, robot_p,
                      frontier.position_local, frontier.position, file=file)
                print("uncertainty & task allocation & distance cost: ", u_m, u_t, u_d, file=file)
        return u_m + u_t * self.t_weight + u_d * self.d_weight

    def compute_utility_task_allocation(self, frontier_w_list, robot_id):
        # local frame to global frame
        u_t = 0
        for frontier_w in frontier_w_list:
            for i, goal_list in enumerate(self.goal_history):
                if i == robot_id:
                    continue
                for goal in goal_list:
                    dist = np.sqrt((goal[0] - frontier_w[0]) ** 2 + (goal[1] - frontier_w[1]) ** 2)
                    u_t += self.compute_P_d(dist)
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
    def __init__(self, radius, robot_state_idx_position_goal, landmarks, isam: tuple):
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

        self.robot_state_idx = robot_state_idx_position_goal[0]
        self.robot_state_position = robot_state_idx_position_goal[1]
        self.robot_state_goal = robot_state_idx_position_goal[2]

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

            if self.landmarks_position != []:
                # calculate Euclidean distance between waypoint and landmarks
                distances = np.linalg.norm(self.landmarks_position - np.array(waypoint[0:2]), axis=1)
                landmark_indices = np.argwhere(distances < self.radius)
            else:
                landmark_indices = []
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
        try:
            distances = np.linalg.norm(waypoints0[:, 0:2] - waypoints1[:, 0:2], axis=1)
        except IndexError:
            print("waypoint0: ", length0, waypoints0)
            print("waypoint1: ", length1, waypoints1)
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
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        virtual_waypoints = [[] for _ in range(len(self.robot_state_idx))]
        for i in range(len(self.robot_state_idx)):
            # set the target robot's virtual goal as the frontier position, other robots just reach existing goal
            if i == robot_id:
                virtual_goal = frontier_position
            else:
                virtual_goal = self.robot_state_goal[i]
            virtual_waypoints[i] = generate_virtual_waypoints(self.robot_state_position[i],
                                                              virtual_goal,
                                                              speed=self.slam_speed)
            self.waypoints2landmark_observations(virtual_waypoints[i], i,
                                                 graph=graph,
                                                 initial_estimate=initial_estimate)
        for i in range(len(self.robot_state_idx)):
            if i == robot_id or len(virtual_waypoints[i]) == 0 or len(virtual_waypoints[robot_id]) == 0:
                continue
            self.waypoints2robot_observation(tuple([virtual_waypoints[robot_id], virtual_waypoints[i]]),
                                             tuple([robot_id, i]), graph=graph)
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
        result = local_to_world_values(result, origin, robot_id)

        # draw the optimized result
        if DEBUG_FRONTIER:
            scatters_x = []
            scatters_y = []
            for key in result.keys():
                if key < gtsam.symbol('a', 0):
                    axis.scatter(result.atPoint2(key)[0], result.atPoint2(key)[1], c='r', marker='o')
                else:
                    scatters_x.append(result.atPose2(key).x())
                    scatters_y.append(result.atPose2(key).y())
            axis.scatter(scatters_x, scatters_y, c='b', marker='.', alpha=0.1)

        return result, marginals
