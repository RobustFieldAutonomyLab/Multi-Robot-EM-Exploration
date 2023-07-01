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
from nav.frontier import FrontierGenerator, ExpectationMaximizationTrajectory

DEBUG_EXP_MAX = False


class ExpVisualizer:

    def __init__(self,
                 seed: int = 0,
                 dpi: int = 96,  # Monitor DPI
                 ):
        self.env = marinenav_env.MarineNavEnv2(seed)
        self.env.reset()
        self.fig = None  # figure for visualization
        self.axis_graph = None  # sub figure for the map
        self.axis_grid = None  # sub figure for exploration

        self.robots_plot = []
        self.robots_last_pos = []
        self.robots_traj_plot = []

        self.dpi = dpi  # monitor DPI

        self.APF_agents = None
        self.step = 0

        self.landmark_slam = LandmarkSLAM()
        self.landmark_slam.reset_graph(len(self.env.robots))
        self.slam_frequency = 10
        self.exploration_terminate_ratio = 0.85

        param_virtual_map = {"maxX": self.env.width, "maxY": self.env.height, "minX": 0, "minY": 0,
                             "radius": self.env.robots[0].perception.range}
        self.virtual_map = VirtualMap(param_virtual_map)

        self.color_list = ['tab:pink', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange', 'tab:gray', 'tab:olive']

        param_frontier = self.virtual_map.get_parameters()
        param_frontier["nearest_frontier_flag"] = True
        param_frontier["num_robot"] = self.env.num_cooperative
        self.frontier_generator = FrontierGenerator(param_frontier)

        self.slam_result = gtsam.Values()
        self.landmark_list = []

        self.cnt = 0

    def init_visualize(self,
                       env_configs=None  # used in Mode 2
                       ):
        # Mode 1 (default): Display an episode
        self.fig = plt.figure(figsize=(32, 16))
        spec = self.fig.add_gridspec(5, 4)
        self.axis_graph = self.fig.add_subplot(spec[:, :2])
        self.axis_grid = self.fig.add_subplot(spec[:, 2:4])

        self.plot_graph(self.axis_graph)

    def plot_grid(self, axis, probability=True, information=False):
        if not DEBUG_EXP_MAX:
            axis.cla()
        if probability:
            data = self.virtual_map.get_probability_matrix()
            axis.imshow(data, origin='lower', alpha=0.5, cmap='bone_r', vmin=0.0, vmax=1.0,
                        extent=[self.virtual_map.minX, self.virtual_map.maxX,
                                self.virtual_map.minY, self.virtual_map.maxY])
        self.axis_grid.set_xticks([])
        self.axis_grid.set_yticks([])
        if information:
            virtual_map = self.virtual_map.get_virtual_map()
            for i, map_row in enumerate(virtual_map):
                for j, virtual_landmark in enumerate(map_row):
                    self.plot_info_ellipse(np.array([virtual_landmark.x,
                                                     virtual_landmark.y]),
                                           virtual_landmark.information, self.axis_grid,
                                           nstd=self.virtual_map.cell_size * 0.04)

    def eigsorted(self, info):
        vals, vecs = np.linalg.eigh(info)
        vals = 1.0 / vals
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def plot_info_ellipse(self, position, info, axis, nstd=.2, **kwargs):
        vals, vecs = self.eigsorted(info)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=position, width=width, height=height, angle=theta, **kwargs)

        axis.add_artist(ellip)
        return ellip

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

    def reset_goal(self, goal_list):
        self.env.reset_goal(goal_list)
        for idx, goal in enumerate(goal_list):
            try:
                self.axis_graph.scatter(goal[0], goal[1], marker=".", color="yellow", s=500, zorder=5)
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
            rot = np.matrix([[np.cos(robot.theta), -np.sin(robot.theta)], \
                             [np.sin(robot.theta), np.cos(robot.theta)]])
            d_r = rot * d
            xy = (robot.x - d_r[0, 0], robot.y - d_r[1, 0])

            angle_d = robot.theta / np.pi * 180
            c = 'g' if robot.cooperative else 'r'
            self.robots_plot.append(self.axis_graph.add_patch(mpl.patches.Rectangle(xy, robot.length, \
                                                                                    robot.width, color=c, \
                                                                                    angle=angle_d, zorder=7)))
            self.robots_plot.append(self.axis_graph.add_patch(mpl.patches.Circle((robot.x, robot.y), \
                                                                                 robot.perception.range, color=c,
                                                                                 alpha=0.2)))
            self.robots_plot.append(self.axis_graph.text(robot.x - 1, robot.y + 1, str(i), color="yellow", fontsize=15))

            if self.robots_last_pos[i] != []:
                h = self.axis_graph.plot((self.robots_last_pos[i][0], robot.x),
                                         (self.robots_last_pos[i][1], robot.y),
                                         color='tab:orange', linestyle='--')
                self.robots_traj_plot[i].append(h)

            self.robots_last_pos[i] = [robot.x, robot.y]

    def one_step(self, actions, slam_signal=False, robot_idx=0):
        assert len(actions) == len(self.env.robots), "Number of actions not equal number of robots!"
        for i, action in enumerate(actions):
            rob = self.env.robots[i]
            if rob.reach_goal:
                continue
            current_velocity = self.env.get_velocity(rob.x, rob.y)
            rob.update_state(action, current_velocity)

        self.plot_robots()

        self.step += 1

    def initialize_apf_agents(self):
        self.APF_agents = []
        for robot in self.env.robots:
            self.APF_agents.append(APF_agent(robot.a, robot.w))

    def navigate_one_step(self, path, video=False):
        stop_signal = False
        odom_cnt = 0
        while not stop_signal:
            reached = 0
            if odom_cnt % self.slam_frequency == 0:
                slam_signal = True
            else:
                slam_signal = False
            # slam_signal = True
            observations = self.env.get_observations()
            if slam_signal:
                obs_list = self.generate_SLAM_observations(observations)
                self.landmark_slam.add_one_step(obs_list)
                self.slam_result = self.landmark_slam.get_result([self.env.robots[0].start[0],
                                                                  self.env.robots[0].start[1],
                                                                  self.env.robots[0].init_theta])
                # self.virtual_map.update(self.slam_result, self.landmark_slam.get_marginal())
                if video:
                    self.axis_grid.cla()
                    self.plot_grid(self.axis_grid)
                    self.visualize_SLAM()
                    self.fig.savefig(path + str(self.cnt) + ".png", bbox_inches="tight")
                    self.cnt += 1
            odom_cnt += 1

            actions = []
            for i, apf in enumerate(self.APF_agents):
                if self.env.robots[i].reach_goal:
                    actions.append(-1)
                    reached += 1
                else:
                    actions.append(apf.act(observations[i][0]))
            if reached == self.env.num_cooperative:
                stop_signal = True
            self.one_step(actions, slam_signal=slam_signal)
        return self.slam_result

    # update robot state and make animation when executing action sequence
    def generate_SLAM_observations(self, observations):
        # SLAM obs_list
        # obs: obs_odom, obs_landmark, obs_robot
        # obs_odom: [dx, dy, dtheta]
        # obs_landmark: [landmark0:range, bearing, id], [landmark1], ...]
        # obs_robot: [obs0: dx, dy, dtheta, id], [obs1], ...]

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
            slam_obs_list[i] = [slam_obs_odom, slam_obs_landmark, slam_obs_robot]
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
            self.axis_grid.plot(landmark_obs[1], landmark_obs[2], 'x', color='black')

    def draw_present_position(self):
        pass
        for robot in self.env.robots:
            self.axis_graph.scatter(robot.x, robot.y, marker="*", color="yellow", s=500, zorder=5)

    def generate_frontier(self):
        self.virtual_map.update(self.slam_result)  # , ev.landmark_slam.get_marginal())
        probability_map = self.virtual_map.get_probability_matrix()
        init_x = self.env.robots[0].start[0]
        init_y = self.env.robots[0].start[1]
        self.landmark_list = self.landmark_slam.get_landmark_list([init_x, init_y,
                                                                   self.env.robots[0].init_theta])
        explored_ratio = self.frontier_generator.generate(probability_map,
                                                          self.landmark_slam.get_latest_state([init_x, init_y,
                                                                                               self.env.robots[
                                                                                                   0].init_theta]),
                                                          self.landmark_list, self.axis_grid)
        if explored_ratio > self.exploration_terminate_ratio:
            return True, [[None] * self.env.num_cooperative]

        return False, self.frontier_generator.choose(self.landmark_slam.get_landmark_list(),
                                                     self.landmark_slam.get_isam(),
                                                     self.landmark_slam.get_last_key_state_pair(),self.axis_grid)

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

