import numpy as np
import gtsam
import math
from scipy.ndimage import convolve
from matplotlib.patches import Rectangle

from nav.utils import point_to_local, point_to_world, generate_virtual_waypoints
from nav.BSP import BeliefSpacePlanning, DEBUG_BSP
from nav.EM import ExpectationMaximizationTrajectory, DEBUG_EM

DEBUG_FRONTIER = False
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

        self.max_dist_robot = 15
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
        self.explored_ratio = 0

        self.more_frontiers = False

        self.virtual_move_length = 0.5

        self.EMParam = {"w_d": 10, "w_t": 100, "w_t_degenerate": 1}
        self.BSPParam = {"w_d": 5}
        self.CEParam = {"w_d": 1, "w_t": 10}

        self.u_t_speed = 10
        self.num_history_goal = 5

        self.boundary_dist = parameters[
            "boundary_dist"]  # Avoid frontiers near boundaries since our environment actually do not have boundary
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

    def generate_potential_frontier_indices(self, probability_map, max_visited_neighbor):
        # generate frontier candidates
        data_free = probability_map < self.free_threshold
        data_occupied = probability_map > self.obstacle_threshold
        data = data_free | data_occupied

        explored_ratio = np.mean(data)
        self.explored_ratio = explored_ratio
        # print("Present exploration ratio: ", explored_ratio)

        neighbor_sum = convolve(data.astype(int), self.neighbor_kernel, mode='constant', cval=0) - data.astype(int)
        data[0:self.boundary_value_i, :] = False  # avoid the boundary being selected
        data[-self.boundary_value_i:, :] = False
        data[:, 0:self.boundary_value_j] = False
        data[:, -self.boundary_value_j:] = False

        frontier_candidate_pool_data = data & (neighbor_sum < max_visited_neighbor)
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
        explored_ratio, indices = self.generate_potential_frontier_indices(probability_map, self.max_visited_neighbor)
        if len(indices) == 0:
            return explored_ratio, False

        # clear history
        self.frontiers = {}

        # state and index of the robot who needs a new goal (target robot)
        state = state_list[robot_id]
        state_index = np.array(self.position_2_index([state.x(), state.y()]))

        # find the frontiers close enough to the target robot

        # Type 1 frontier, the nearest frontier to current position
        # _, indices_ = self.generate_potential_frontier_indices(probability_map, max_visited_neighbor=3)
        # distance between the target robot abd frontier candidates
        distances = np.linalg.norm(indices - state_index, axis=1)
        indices_distances_within_list = np.argwhere(distances < self.max_dist_robot_scaled)
        index_this = np.argmin(distances)
        position_this = self.index_2_position(indices[index_this])

        if index_this in self.frontiers:
            self.frontiers[index_this].nearest_frontier = True
        else:
            self.frontiers[index_this] = Frontier(position_this,
                                                  origin=self.origin,
                                                  nearest_frontier=True)
        if DEBUG_FRONTIER:
            print("robot ", robot_id)
            print("distances: ", distances)
            print("index: ", index_this, position_this)
        if self.explored_ratio < 0.2:
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
            indices_distances_within_list = np.argwhere((distances >self.max_dist_robot_scaled) &
                                                        (distances < self.max_dist_landmark_scaled))

            for index_this in indices_distances_within_list:
                if index_this[0] in self.frontiers:
                    self.frontiers[index_this[0]].add_relative(landmark[0])
                else:
                    position_this = self.index_2_position(indices[index_this[0]])
                    self.frontiers[index_this[0]] = Frontier(position_this, origin=self.origin,
                                                             relative=landmark[0])
                landmark_frontier_cnt += 1

        if (self.more_frontiers or len(self.frontiers) < self.min_landmark_frontier_cnt) and len(landmark_list) != 0:
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
                        # landmark_frontier_cnt += 1

        # if (self.more_frontiers or len(self.frontiers) < self.min_landmark_frontier_cnt) and len(landmark_list) != 0:
        #     landmark_array = np.array(landmark_list)[:, :]
        #     landmark_array= sorted(landmark_array,
        #                                   key=lambda elem: np.linalg.norm([state.x(), state.y()] - elem[1:]))
        #     distances = [np.linalg.norm([state.x(), state.y()] - elem[1:]) for elem in landmark_array]
        #     dt = int(len(landmark_array) / self.min_landmark_frontier_cnt)
        #     for i, landmark_this in enumerate(landmark_array):
        #         if len(self.frontiers) > self.min_landmark_frontier_cnt:
        #             break
        #         # if there are landmarks on the way there
        #         if i % dt == 0 and distances[i] > self.radius:
        #             # print("landmark revisit: ", landmark_this, )
        #             self.frontiers[i+len(indices)] = Frontier(landmark_this[1:] + [self.radius/2, self.radius/2],
        #                                          origin=self.origin,
        #                                          relative=int(landmark_this[0]))
        #         landmark_frontier_cnt += 1

        # Type 3 frontier,meet with other robots
        if landmark_frontier_cnt < self.min_landmark_frontier_cnt:
            for robot_index in range(self.num_robot):
                if robot_index == robot_id:
                    continue  # you don't need to meet yourself
                if len(self.goal_history[robot_index]) == 0:
                    continue  # the robot has no goal
                goal_this = self.goal_history[robot_index][-1]
                # my time to the goal
                dist_this = np.sqrt((state.x() - goal_this[0]) ** 2 + (state.y() - goal_this[1]) ** 2)
                if dist_this < self.max_dist_robot:
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
        if PLOT_VIRTUAL_MAP and axis is not None:
            axis.cla()
            for frontier in self.frontiers.values():
                if frontier.nearest_frontier:
                    axis.scatter(frontier.position[0], frontier.position[1],
                                 c='#4859af', marker='o', s=300, zorder=5)
                elif frontier.relatives == []:
                    axis.scatter(frontier.position[0], frontier.position[1],
                                 c='#4883af', marker='o', s=300, zorder=5)
                else:
                    axis.scatter(frontier.position[0], frontier.position[1],
                                 c='#865eb3', marker='o', s=300, zorder=5)

    def choose_NF(self, robot_id):
        goal = self.find_frontier_nearest_neighbor()
        self.goal_history[robot_id].append(np.array(goal))
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
                u_t += self.compute_utility_task_allocation(pts, robot_id, True)
            # calculate the landmark visitation and new exploration case first
            u_d = compute_distance(frontier, robot_p)
            cost_list.append((key, self.CEParam["w_t"] * u_t + self.CEParam["w_d"] * u_d))
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
                                       axis=axis)
            cost_this = self.compute_utility_BSP(result, marginals,
                                                 robot_state_idx_position_local[1][robot_id],
                                                 frontier)
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

    def choose_EM(self, robot_id, landmark_list_local, isam, robot_state_idx_position_local, virtual_map,
                  axis=None):
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
        for key, frontier in self.frontiers.items():
            # return the transformed virtual SLAM result for the calculation of the information of virtual map
            result, marginals = emt.do(robot_id=robot_id,
                                       frontier_position=frontier.position_local,
                                       origin=self.origin,
                                       axis=axis)

            # no need to reset, since the update_information will reset the virtual map
            cost_this = self.compute_utility_EM(virtual_map, result, marginals,
                                                robot_state_idx_position_local[1][robot_id],
                                                frontier, robot_id)
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
                            robot_p, frontier):
        u_d = compute_distance(frontier, robot_p)
        u_m = 0
        for key in result.keys():
            if key < gtsam.symbol('a', 0):
                pass
            else:
                pose = result.atPose2(key)
                try:
                    marginal = marginals.marginalInformation(key)
                    u_m += np.sqrt(marginals.marginalCovariance(key).trace())
                except RuntimeError:
                    marginal = np.zeros((3, 3))
                    u_m += 1000
        if DEBUG_BSP:
            with open('log.txt', 'a') as file:
                print("u_m, u_d: ", u_m, u_d, file=file)
        return u_m + self.BSPParam["w_d"] * u_d

    def compute_utility_EM(self, virtual_map, result: gtsam.Values, marginals: gtsam.Marginals,
                           robot_p, frontier, robot_id):
        # robot_position could be a tuple of two robots
        # calculate the cost of the frontier for a specific robot locally
        virtual_map.reset_information()
        virtual_map.update_information(result, marginals)
        u_m = virtual_map.get_sum_uncertainty()
        num_history_goal = min(self.num_history_goal, len(self.goal_history[robot_id]))
        goals_task_allocation = self.goal_history[robot_id][-num_history_goal:]
        goals_task_allocation.append(point_to_world(robot_p.x(), robot_p.y(), self.origin))
        u_t = 0
        for goal in goals_task_allocation:
            pts = generate_virtual_waypoints(np.array(goal), frontier.position, speed=self.u_t_speed)
            if pts == []:
                pts = [frontier.position]
            u_t += self.compute_utility_task_allocation(pts, robot_id)
        # calculate the landmark visitation and new exploration case first
        u_d = compute_distance(frontier, robot_p)
        w_t = max(1 - self.explored_ratio * self.EMParam["w_t_degenerate"], 0)
        if DEBUG_EM:
            with open('log.txt', 'a') as file:
                print("robot id, robot position, frontier position: ", robot_id, robot_p,
                      frontier.position_local, frontier.position, file=file)
                print("uncertainty & task allocation & distance cost & sum: ", u_m, u_t, u_d,
                      u_m + u_t * self.EMParam["w_t"] * w_t + u_d * self.EMParam["w_d"], file=file)
        return u_m + u_t * self.EMParam["w_t"] * w_t + u_d * self.EMParam["w_d"]

    def compute_utility_task_allocation(self, frontier_w_list, robot_id, ce_flag=False):
        # local frame to global frame
        u_t = 0
        for frontier_w in frontier_w_list:
            for i, goal_list in enumerate(self.goal_history):
                if i == robot_id:
                    continue
                for goal in goal_list:
                    dist = np.sqrt((goal[0] - frontier_w[0]) ** 2 + (goal[1] - frontier_w[1]) ** 2)
                    u_t += self.compute_P_d(dist, ce_flag)
        return u_t

    def compute_P_d(self, dist, ce_flag=False):
        if ce_flag:
            allocation_max_distance = self.allocation_max_distance
        else:
            allocation_max_distance = self.allocation_max_distance * \
                                      (1 - self.explored_ratio * self.EMParam["w_t_degenerate"])
        if dist < allocation_max_distance:
            P_d = 1 - dist / allocation_max_distance
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
            if len(value.connected_robot_pair) != 0:
                frontiers.append(value.position)
        return frontiers
