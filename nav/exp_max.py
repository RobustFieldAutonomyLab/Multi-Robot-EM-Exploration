import marinenav_env.envs.marinenav_env as marinenav_env
import numpy as np
import gtsam
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import copy
from APF import APF_agent
from nav.navigation import LandmarkSLAM
from nav.virtualmap import VirtualMap
from nav.frontier import FrontierGenerator, DEBUG_EM, DEBUG_FRONTIER, PLOT_VIRTUAL_MAP
from nav.utils import point_to_local, point_to_world, A_Star
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import time

DEBUG_EXP_MAX = False


def local_goal_to_world_goal(local_goal_p, local_robot_p, world_robot_p):
    if isinstance(local_robot_p, gtsam.Pose2):
        p_local = [local_robot_p.x(), local_robot_p.y(), local_robot_p.theta()]
    else:
        p_local = local_robot_p
    if isinstance(world_robot_p, gtsam.Pose2):
        p_world = [world_robot_p.x(), world_robot_p.y(), world_robot_p.theta()]
    else:
        p_world = world_robot_p
    trans_p = point_to_local(local_goal_p[0], local_goal_p[1], p_local)
    world_goal_p = point_to_world(trans_p[0], trans_p[1], p_world)
    return world_goal_p


def eigsorted(info):
    vals, vecs = np.linalg.eigh(info)
    vals = 1.0 / vals
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def plot_info_ellipse(position, info, axis, nstd=.2, **kwargs):
    vals, vecs = eigsorted(info)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=position, width=width, height=height, angle=theta, **kwargs, color='#4859af', alpha=0.3)

    axis.add_artist(ellip)
    return ellip


class ExpParams:
    def __init__(self, cell_size=4, env_width=160,
                 env_height=200, num_obs=100,
                 num_cooperative=3, boundary_dist=8,
                 start_center=np.array([30, 100]), sensor_range=10):
        self.cell_size = cell_size
        self.env_width = env_width
        self.env_height = env_height
        self.num_obs = num_obs
        self.num_cooperative = num_cooperative
        self.boundary_dist = boundary_dist
        self.dpi = 96
        self.map_path = None
        self.start_center = start_center
        self.sensor_range = sensor_range


class ExpVisualizer:

    def __init__(self,
                 seed: int = 0,
                 method: str = "NF",
                 num: int = 0,
                 folder_path: str = None,
                 params: ExpParams = ExpParams()
                 ):
        self.repeat_num = num
        params_env = {"width": params.env_width, "height": params.env_height,
                      "num_obs": params.num_obs,"num_cooperative": params.num_cooperative,
                      "sensor_range": params.sensor_range}
        self.env = marinenav_env.MarineNavEnv2(seed, params = params_env)
        self.env.reset(params.start_center)

        if params.map_path is not None:
            self.env.load_map(params.map_path)

        self.fig = None  # figure for visualization
        self.axis_graph = None  # sub figure for the map
        self.axis_grid = None  # sub figure for exploration

        self.robots_plot = []
        self.robots_last_pos = []
        self.robots_traj_plot = []

        self.dpi = params.dpi  # monitor DPI

        self.APF_agents = None
        self.a_star = None

        self.initialize_apf_agents()

        self.slam_origin = None
        self.landmark_slam = LandmarkSLAM()
        self.landmark_slam.reset_graph(len(self.env.robots))
        self.slam_frequency = 10
        self.slam_ground_truth = []
        self.exploration_terminate_ratio = 0.85

        param_virtual_map = {"maxX": self.env.width, "maxY": self.env.height, "minX": 0, "minY": 0,
                             "radius": self.env.robots[0].perception.range, "cell_size": params.cell_size}
        self.virtual_map = VirtualMap(param_virtual_map)
        self.a_star = A_Star(self.virtual_map.num_rows, self.virtual_map.num_cols, self.virtual_map.cell_size)

        self.color_list = ['#865eb3', '#48adaf', '#4883af', '#4859af', '#aeaead']

        param_frontier = self.virtual_map.get_parameters()
        param_frontier["num_robot"] = self.env.num_cooperative
        param_frontier["origin"] = [self.env.robots[0].start[0],
                                    self.env.robots[0].start[1],
                                    self.env.robots[0].init_theta]
        param_frontier["boundary_dist"] = params.boundary_dist
        self.frontier_generator = FrontierGenerator(param_frontier)

        self.slam_result = gtsam.Values()
        self.landmark_list = []

        self.max_exploration_ratio = 0.85

        self.cnt = 0

        self.method = method
        self.folder_path = folder_path

        self.history = []

    def init_visualize(self, draw_ground_truth=True):
        # Mode 1 (default): Display an episode
        self.fig = plt.figure(figsize=(32, 16))
        spec = self.fig.add_gridspec(5, 4)
        if draw_ground_truth:
            self.axis_graph = self.fig.add_subplot(spec[:, :2])
        self.axis_grid = self.fig.add_subplot(spec[:, 2:4])
        if self.axis_graph is not None:
            self.plot_graph(self.axis_graph)

    def plot_grid(self, probability=True, information=True):
        if probability:
            data = self.virtual_map.get_probability_matrix()
            custom_colors = ['#ffffff', '#4859af']
            a = to_rgba(custom_colors[0])
            b = to_rgba(custom_colors[1])
            cmap_segments = {'red': [(0.0, a[0], a[0]),
                                     (1.0, b[0], b[0])],

                             'green': [(0.0, a[1], a[1]),
                                       (1.0, b[1], b[1])],

                             'blue': [(0.0, a[2], a[2]),
                                      (1.0, b[2], b[2])]}
            custom_cmap = LinearSegmentedColormap('CustomColormap', cmap_segments)
            self.axis_grid.imshow(data, origin='lower', alpha=0.5, cmap=custom_cmap, vmin=0.0, vmax=1.0,
                                  extent=[self.virtual_map.minX, self.virtual_map.maxX,
                                          self.virtual_map.minY, self.virtual_map.maxY])
        self.axis_grid.set_xticks([])
        self.axis_grid.set_yticks([])
        self.axis_grid.set_xlim([0, self.env.width])
        self.axis_grid.set_ylim([0, self.env.height])
        if information:
            self.virtual_map.update(self.slam_result, self.landmark_slam.marginals)
            virtual_map = self.virtual_map.get_virtual_map()
            for i, map_row in enumerate(virtual_map):
                for j, virtual_landmark in enumerate(map_row):
                    plot_info_ellipse(np.array([virtual_landmark.x,
                                                virtual_landmark.y]),
                                      virtual_landmark.information, self.axis_grid,
                                      nstd=self.virtual_map.cell_size * 0.3)

    def plot_graph(self, axis):
        # plot current velocity in the mapf
        x_pos = list(np.linspace(-2.5, self.env.width + 2.5, 110))
        y_pos = list(np.linspace(-2.5, self.env.height + 2.5, 110))

        pos_x = []
        pos_y = []
        arrow_x = []
        arrow_y = []
        speeds = np.zeros((len(x_pos), len(y_pos)))
        for m, x in enumerate(x_pos):
            for n, y in enumerate(y_pos):
                v = self.env.get_velocity(x, y)
                speed = np.clip(np.linalg.norm(v), 0.1, 10)
                pos_x.append(x)
                pos_y.append(y)
                arrow_x.append(v[0])
                arrow_y.append(v[1])
                speeds[n, m] = np.log(speed)

        cmap = cm.Blues(np.linspace(0, 1, 20))
        cmap = mpl.colors.ListedColormap(cmap[10:, :-1])

        axis.contourf(x_pos, y_pos, speeds, cmap=cmap)
        axis.quiver(pos_x, pos_y, arrow_x, arrow_y, width=0.001)

        # plot the evaluation boundary
        boundary = np.array([[0.0, 0.0],
                             [self.env.width, 0.0],
                             [self.env.width, self.env.height],
                             [0.0, self.env.height],
                             [0.0, 0.0]])
        axis.plot(boundary[:, 0], boundary[:, 1], color='r', linestyle="-.", linewidth=3)

        # plot obstacles in the map
        l = True
        for obs in self.env.obstacles:
            if l:
                axis.add_patch(mpl.patches.Circle((obs.x, obs.y), radius=obs.r, color='m'))
                l = False
            else:
                axis.add_patch(mpl.patches.Circle((obs.x, obs.y), radius=obs.r, color='m'))

        axis.set_aspect('equal')
        axis.set_xlim([-2.5, self.env.width + 2.5])
        axis.set_ylim([-2.5, self.env.height + 2.5])
        axis.set_xticks([])
        axis.set_yticks([])

        # plot start and goal state of each robot
        for idx, robot in enumerate(self.env.robots):
            axis.scatter(robot.start[0], robot.start[1], marker="o", color="yellow", s=160, zorder=5)
            # axis.scatter(robot.goal[0], robot.goal[1], marker="*", color="yellow", s=500, zorder=5)
            axis.text(robot.start[0] - 1, robot.start[1] + 1, str(idx), color="yellow", fontsize=15)
            # axis.text(robot.goal[0] - 1, robot.goal[1] + 1, str(idx), color="yellow", fontsize=15)
            self.robots_last_pos.append([])
            self.robots_traj_plot.append([])

        self.plot_robots()

    def reset_one_goal(self, goal, idx):
        self.env.robots[idx].reset_goal(goal)

    def reset_goal(self, goal_list):
        self.env.reset_goal(goal_list)
        for idx, goal in enumerate(goal_list):
            try:
                self.axis_graph.scatter(goal[0], goal[1], marker=".", color="yellow", s=500, zorder=5)
                # self.axis_grid.scatter(goal[0], goal[1], marker=".", color="yellow", s=300, zorder=4, alpha=0.5)
            except:
                print("idx: ", idx)
                print("goal: ", goal)
            # self.axis_graph.text(goal[0] - 1, goal[1] + 1, str(idx), color="yellow", fontsize=15)

    def plot_robots(self):
        for robot_plot in self.robots_plot:
            robot_plot.remove()
        self.robots_plot.clear()

        for i, robot in enumerate(self.env.robots):
            d = np.matrix([[0.5 * robot.length], [0.5 * robot.width]])
            rot = np.matrix([[np.cos(robot.theta), -np.sin(robot.theta)], [np.sin(robot.theta), np.cos(robot.theta)]])
            d_r = rot * d
            xy = (robot.x - d_r[0, 0], robot.y - d_r[1, 0])

            angle_d = robot.theta / np.pi * 180
            c = 'g' if robot.cooperative else 'r'
            if self.axis_graph is not None:
                self.robots_plot.append(self.axis_graph.add_patch(mpl.patches.Rectangle(xy, robot.length,
                                                                                        robot.width, color=c,
                                                                                        angle=angle_d, zorder=7)))
                self.robots_plot.append(self.axis_graph.add_patch(mpl.patches.Circle((robot.x, robot.y),
                                                                                     robot.perception.range, color=c,
                                                                                     alpha=0.2)))
                self.robots_plot.append(
                    self.axis_graph.text(robot.x - 1, robot.y + 1, str(i), color="yellow", fontsize=15))

                if self.robots_last_pos[i] != []:
                    h = self.axis_graph.plot((self.robots_last_pos[i][0], robot.x),
                                             (self.robots_last_pos[i][1], robot.y),
                                             color='tab:orange', linestyle='--')
                    self.robots_traj_plot[i].append(h)

                self.robots_last_pos[i] = [robot.x, robot.y]

    def one_step(self, action, robot_idx=None):
        if robot_idx is not None:
            rob = self.env.robots[robot_idx]
            if not rob.reach_goal:
                current_velocity = self.env.get_velocity(rob.x, rob.y)
                rob.update_state(action, current_velocity)

        # assert len(action) == len(self.env.robots), "Number of actions not equal number of robots!"
        # for i, action in enumerate(action):
        #     rob = self.env.robots[i]
        #     if rob.reach_goal:
        #         continue
        #     current_velocity = self.env.get_velocity(rob.x, rob.y)
        #     rob.update_state(action, current_velocity)
        #
        # self.plot_robots()
        #
        # self.step += 1

    def initialize_apf_agents(self):
        self.APF_agents = []
        for robot in self.env.robots:
            self.APF_agents.append(APF_agent(robot.a, robot.w))

    def slam_one_step(self, observations, video=False, path=None):
        obs_list = self.generate_SLAM_observations(observations)
        self.landmark_slam.add_one_step(obs_list)
        self.slam_result = self.landmark_slam.get_result([self.env.robots[0].start[0],
                                                          self.env.robots[0].start[1],
                                                          self.env.robots[0].init_theta])
        # self.virtual_map.update(self.slam_result, self.landmark_slam.get_marginal())
        if video:
            self.plot_grid()
            self.visualize_SLAM()
            self.fig.savefig(path + str(self.cnt) + ".png", bbox_inches="tight")
            self.cnt += 1

    def navigate_one_step(self, max_ite, path, video=False):
        stop_signal = False

        speed = 15
        direction_list = [[0, 1], [-1, 1], [-1, 0], [-1, -1],
                          [0, -1], [1, -1], [1, 0], [1, 1]]
        plot_cnt = 0
        direction_cnt = [0] * self.env.num_cooperative
        while plot_cnt < max_ite and not stop_signal:
            plot_signal = False
            observations = self.env.get_observations()
            if self.cnt % self.slam_frequency == 0:
                self.slam_one_step(observations)
            self.cnt += 1
            actions = []
            for i, apf in enumerate(self.APF_agents):
                robot = self.env.robots[i]
                if robot.reach_goal:
                    # if reach a goal, design a new goal
                    direction_this = direction_list[(direction_cnt[i] + i) % len(direction_list)]
                    direction_cnt[i] += 1
                    new_goal = [robot.x + speed * direction_this[0],
                                robot.y + speed * direction_this[1]]
                    robot.reset_goal(new_goal)
                    plot_signal = True

                obstacles = self.landmark_slam.get_landmark_list(self.slam_origin)
                # if obstacles != []:
                #     vec, vec0, vec1 = observations[i][0]
                #     goal_new = self.a_star.a_star(vec[:2], vec[4:6], obstacles)
                #     vec[:2] = goal_new
                #     observations[i][0] = (vec, vec0, vec1)
                actions.append(apf.act(observations[i][0]))
            # moving
            for i, action in enumerate(actions):
                self.one_step(action, robot_idx=i)
            if plot_signal:
                plot_cnt += 1
                if self.plot_grid is not None:
                    self.axis_grid.cla()
                    self.virtual_map.update(self.slam_result, self.landmark_slam.get_marginal())
                    self.plot_grid()
                    self.plot_robots()
                    self.visualize_SLAM()

                    self.fig.savefig(path + str(plot_cnt) + ".png", bbox_inches="tight")
        return True

    def explore_one_step(self, max_ite, path=None, video=False):
        self.slam_origin = [self.env.robots[0].start[0],
                            self.env.robots[0].start[1],
                            self.env.robots[0].init_theta]
        stop_signal = False
        plot_cnt = 0
        while plot_cnt < max_ite and not stop_signal:
            plot_signal = False
            observations = self.env.get_observations()
            if self.cnt % self.slam_frequency == 0:
                self.slam_one_step(observations)
            self.cnt += 1
            actions = []
            for i, apf in enumerate(self.APF_agents):
                robot = self.env.robots[i]
                if robot.reach_goal:
                    if DEBUG_EM:
                        with open('log.txt', 'a') as file:
                            print("No: ", plot_cnt, file=file)
                    if robot.waiting:
                        id_that = robot.waiting_for_robot
                        robot_that = self.env.robots[id_that]
                        if robot_that.reach_goal:
                            robot.reset_waiting()
                            robot_that.reset_waiting()
                    if not robot.waiting:
                        # if reach a goal, design a new goal
                        stop_signal, new_goal = self.generate_frontier(i)
                        robot.reset_goal(new_goal)
                        # if we find a rendezvous frontier, we notify the neighbor to wait for us
                        plot_signal = True
                actions.append(apf.act(observations[i][0]))
            # moving
            for i, action in enumerate(actions):
                self.one_step(action, robot_idx=i)
            if plot_signal:
                if self.axis_grid is not None:
                    self.plot_grid()
                    self.plot_robots()
                    self.visualize_SLAM()
                    self.fig.savefig(path + str(plot_cnt) + ".png", bbox_inches="tight")
                plot_cnt += 1
        self.save()
        if stop_signal:
            return True
        else:
            return False

    # update robot state and make animation when executing action sequence
    def generate_SLAM_observations(self, observations):
        # SLAM obs_list
        # obs: obs_odom, obs_landmark, obs_robot
        # obs_odom: [dx, dy, dtheta]
        # obs_landmark: [landmark0:range, bearing, id], [landmark1], ...]
        # obs_robot: [obs0: dx, dy, dtheta, id], [obs1], ...]
        # ground truth robot: [x_r, y_r, theta_r]

        # env obs_list
        # format: {"self": [velocity,goal,pose],
        #          "static":[[obs_1.x,obs_1.y,obs_1.r, obs_1.id],...,[obs_n.x,obs_n.y,obs_n.r, obs_n.id]],
        #          "dynamic":{id_1:[[robot_1.x,robot_1.y,robot_1.vx,robot_1.vy]_(t-m),...,[]_t]...}
        slam_obs_list = np.empty((len(observations),), dtype=object)
        for i, obs in enumerate(observations):
            if obs[2]:
                continue
            self_state, static_states, dynamic_states = obs[0]
            self.env.robots[i].odometry.add_noise(self_state[4], self_state[5], self_state[6])
            slam_obs_odom = np.array(self.env.robots[i].odometry.get_odom())
            if len(static_states) != 0:
                slam_obs_landmark = np.zeros([len(static_states), 3])
                for j, static_state in enumerate(static_states):
                    rb = self.env.robots[i].landmark_observation.add_noise(static_state[0],
                                                                           static_state[1])
                    slam_obs_landmark[j, 0] = copy.deepcopy(rb[0])
                    slam_obs_landmark[j, 1] = copy.deepcopy(rb[1])
                    slam_obs_landmark[j, 2] = copy.deepcopy(static_state[3])
            else:
                slam_obs_landmark = []
            if len(dynamic_states) != 0:
                slam_obs_robot = np.zeros([len(dynamic_states), 4])
                for j, dynamic_state in enumerate(dynamic_states):
                    xyt = self.env.robots[i].robot_observation.add_noise(dynamic_state[0],
                                                                         dynamic_state[1],
                                                                         dynamic_state[4])
                    slam_obs_robot[j, 0] = copy.deepcopy(xyt[0])
                    slam_obs_robot[j, 1] = copy.deepcopy(xyt[1])
                    slam_obs_robot[j, 2] = copy.deepcopy(xyt[2])
                    slam_obs_robot[j, 3] = copy.deepcopy(dynamic_state[5])
            else:
                slam_obs_robot = []
            slam_obs_list[i] = [slam_obs_odom, slam_obs_landmark, slam_obs_robot, self_state[4:7]]
        return slam_obs_list

    def visualize_SLAM(self, start_idx=0):
        # color_list = ['tab:pink', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange', 'tab:gray', 'tab:olive']
        init_x = self.env.robots[0].start[0]
        init_y = self.env.robots[0].start[1]
        for i in range(self.env.num_cooperative):
            pose = self.landmark_slam.get_robot_trajectory(i, [init_x, init_y,
                                                               self.env.robots[0].init_theta])
            self.axis_grid.plot(pose[:, 0], pose[:, 1], color=self.color_list[i], linewidth=2, zorder=10)
            # self.axis_graph.scatter(pose[:,0],pose[:,1], marker="*", color="pink", s=500, zorder=5)

        for landmark_obs in self.landmark_list:
            self.axis_grid.plot(landmark_obs[1], landmark_obs[2], 'x', color='black', zorder=10)

    def draw_present_position(self):
        for robot in self.env.robots:
            self.axis_graph.scatter(robot.x, robot.y, marker="*", color="yellow", s=500, zorder=5)

    def generate_frontier(self, idx):
        self.virtual_map.update(self.slam_result, self.landmark_slam.get_marginal())
        probability_map = self.virtual_map.get_probability_matrix()
        latest_state = self.landmark_slam.get_latest_state(self.slam_origin)
        self.landmark_list = self.landmark_slam.get_landmark_list(self.slam_origin)
        explored_ratio, frontiers_generated = self.frontier_generator.generate(idx, probability_map,
                                                                               latest_state,
                                                                               self.landmark_list, self.axis_grid)

        if not frontiers_generated:
            self.save()
            assert "No more frontiers."

        if explored_ratio > self.exploration_terminate_ratio:
            return True, None
        time0 = time.time()
        if self.method == "EM_2":
            self.frontier_generator.EMParam["w_t"] = 0
            goal, robot_waiting = self.frontier_generator.choose_EM(idx, self.landmark_slam.get_landmark_list(),
                                                                    self.landmark_slam.get_isam(),
                                                                    self.landmark_slam.get_last_key_state_pair(),
                                                                    self.virtual_map,
                                                                    self.axis_grid)
        elif self.method == "EM_3":
            goal, robot_waiting = self.frontier_generator.choose_EM(idx, self.landmark_slam.get_landmark_list(),
                                                                    self.landmark_slam.get_isam(),
                                                                    self.landmark_slam.get_last_key_state_pair(),
                                                                    self.virtual_map,
                                                                    self.axis_grid)
        elif self.method == "NF":
            goal, robot_waiting = self.frontier_generator.choose_NF(idx)
        elif self.method == "CE":
            goal, robot_waiting = self.frontier_generator.choose_CE(idx, self.landmark_slam.get_last_key_state_pair())
        elif self.method == "BSP":
            goal, robot_waiting = self.frontier_generator.choose_BSP(idx, self.landmark_slam.get_landmark_list(),
                                                                     self.landmark_slam.get_last_key_state_pair(),
                                                                     self.axis_grid)
        time1 = time.time()
        time_this = time1 - time0
        self.record_history(explored_ratio, time_this)
        if robot_waiting is not None:
            # let them wait for each other
            self.env.robots[robot_waiting].reset_waiting(idx)
            self.env.robots[idx].reset_waiting(robot_waiting)

        slam_poses = self.landmark_slam.get_last_key_state_pair(self.slam_origin)
        if self.axis_grid is not None:
            if not DEBUG_EXP_MAX and not DEBUG_FRONTIER and not PLOT_VIRTUAL_MAP:
                self.axis_grid.cla()
            self.axis_grid.scatter(goal[0], goal[1], marker="*", color="red", s=300, zorder=6)  # , alpha=0.5)
            self.axis_grid.scatter(latest_state[idx].x(),
                                   latest_state[idx].y(),
                                   marker='*', s=300, c='black', zorder=7)
        goal = local_goal_to_world_goal(goal, slam_poses[1][idx], [self.env.robots[idx].x,
                                                                   self.env.robots[idx].y,
                                                                   self.env.robots[idx].theta])
        if explored_ratio < self.max_exploration_ratio:
            return False, goal
        else:
            return True, goal

    def save(self):
        filename = self.folder_path + "/" + self.method + "/" + str(self.env.num_obs) + "_" + str(
            self.repeat_num) + ".txt"
        np.savetxt(filename, np.array(self.history))

    def record_history(self, exploration_ratio, time_this):
        # localization error
        err_localization = 0
        err_angle = 0
        cnt = 0
        ground_truth = self.landmark_slam.get_ground_truth()
        result = self.landmark_slam.get_result(self.slam_origin)
        dist = 0
        for key in ground_truth.keys():
            pose_true = ground_truth.atPose2(key)
            pose_estimated = result.atPose2(key)
            err_localization += np.linalg.norm([pose_true.x() - pose_estimated.x(),
                                                pose_true.y() - pose_estimated.y()])
            err_angle += np.abs(pose_true.theta() - pose_estimated.theta())
            if ground_truth.exists(key + 1):
                pose_true_next = ground_truth.atPose2(key + 1)
                dist += np.linalg.norm([pose_true.x() - pose_true_next.x(),
                                        pose_true.y() - pose_true_next.y()])
            cnt += 1
        err_localization /= len(ground_truth.keys())
        err_angle /= len(ground_truth.keys())
        # landmark error
        err_landmark = 0
        landmarks_list = self.landmark_slam.get_landmark_list(self.slam_origin)
        for landmark in landmarks_list:
            landmark_id = landmark[0]
            landmark_real = np.array([self.env.obstacles[landmark_id].x, self.env.obstacles[landmark_id].y])
            landmark_estimated = landmark[1:3]
            err_landmark += np.linalg.norm(landmark_real - landmark_estimated)
        if len(landmarks_list) != 0:
            err_landmark /= len(landmarks_list)
        self.history.append([dist, err_localization, err_angle, err_landmark, exploration_ratio, time_this])
        if self.axis_grid is not None:
            print("dist, err_localization, err_angle, err_landmark, exploration_ratio, time: ",
                  dist, err_localization, err_angle, err_landmark, exploration_ratio, time_this)
        # exploration ratio

    def visualize_frontier(self):
        # color_list = ['tab:pink', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange', 'tab:gray', 'tab:olive']
        for i in range(0, self.env.num_cooperative):
            positions = self.frontier_generator.return_frontiers_position(i)
            for position in positions:
                self.axis_grid.plot(position[0], position[1], '.', color=self.color_list[i])

        meeting_position = self.frontier_generator.return_frontiers_rendezvous()
        for position in meeting_position:
            self.axis_grid.scatter(position[0], position[1], marker='o', facecolors='none', edgecolors='black')

    def draw_virtual_waypoints(self, waypoints, robot_id):
        self.axis_grid.plot(waypoints[:, 0], waypoints[:, 1],
                            marker='.', linestyle='-', color=self.color_list[robot_id],
                            alpha=0.3, linewidth=.5)
