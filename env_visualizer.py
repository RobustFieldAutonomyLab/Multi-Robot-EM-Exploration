import marinenav_env.envs.marinenav_env as marinenav_env
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import copy
import scipy.spatial
# import gym
import json
import os
from APF import APF_agent
from nav.navigation import LandmarkSLAM
from nav.virtualmap import VirtualMap
from nav.virtualmap import VirtualLandmark
from matplotlib.patches import Ellipse

class EnvVisualizer:

    def __init__(self,
                 seed: int = 0,
                 cvar_num: int = 0,  # Number of CVaR (only available in mode 5)
                 draw_envs: bool = False,  # Mode 2: plot the envrionment
                 draw_traj: bool = False,  # Mode 3: plot final trajectories given action sequences
                 video_plots: bool = False,  # Mode 4: Generate plots for a video
                 plot_dist: bool = False,  # If return distributions are needed (for IQN agent) in the video
                 plot_qvalues: bool = False,  # If Q values are needed in the video
                 dpi: int = 96,  # Monitor DPI
                 ):
        self.env = marinenav_env.MarineNavEnv2(seed)
        self.env.reset()
        self.fig = None  # figure for visualization
        self.axis_graph = None  # sub figure for the map
        self.robots_plot = []
        self.robots_last_pos = []
        self.robots_traj_plot = []
        self.LiDAR_beams_plot = []
        self.axis_title = None  # sub figure for title
        self.axis_action = None  # sub figure for action command and steer data
        self.axis_goal = None  # sub figure for relative goal measurment
        self.axis_perception = None  # sub figure for perception output
        self.axis_dvl = None  # sub figure for DVL measurement
        self.axis_dist = []  # sub figure(s) for return distribution of actions
        self.axis_qvalues = None  # subfigure for Q values of actions
        self.cvar_num = cvar_num  # number of CVaR values to plot

        self.episode_actions = []  # action sequence load from episode data
        self.episode_actions_quantiles = None
        self.episode_actions_taus = None

        self.plot_dist = plot_dist  # draw return distribution of actions
        self.plot_qvalues = plot_qvalues  # draw Q values of actions
        self.draw_envs = draw_envs  # draw only the envs
        self.draw_traj = draw_traj  # draw only final trajectories
        self.video_plots = video_plots  # draw video plots
        self.plots_save_dir = None  # video plots save directory
        self.dpi = dpi  # monitor DPI
        self.agent = None  # agent name

        self.configs = None  # evaluation configs
        self.episodes = None  # evaluation episodes to visualize

        self.APF_agents = None

        self.step = 0

        self.landmark_slam = LandmarkSLAM()
        self.landmark_slam.reset_graph(len(self.env.robots))
        self.slam_frequency = 10

        param_virtual_map = {"maxX": self.env.width, "maxY": self.env.height, "minX": 0, "minY": 0,
                             "radius": self.env.robots[0].perception.range}
        self.virtual_map = VirtualMap(param_virtual_map)
        self.slam_result = []

        self.cnt = 0

    def init_visualize(self,
                       env_configs=None  # used in Mode 2
                       ):

        # initialize subplot for the map, robot state and sensor measurments
        if self.draw_envs:
            # Mode 2: plot final trajectories given action sequences
            if env_configs is None:
                self.fig, self.axis_graph = plt.subplots(1, 1, figsize=(8, 8))
            else:
                num = len(env_configs)
                self.fig, self.axis_graphs = plt.subplots(1, num, figsize=(8 * num, 8))
        elif self.draw_traj:
            # Mode 3: plot the envrionment
            self.fig, self.axis_graph = plt.subplots(figsize=(8, 8))
        elif self.video_plots:
            # Mode 4: Generate 1080p video plots
            w = 1920
            h = 1080
            self.fig = plt.figure(figsize=(w / self.dpi, h / self.dpi), dpi=self.dpi)
            if self.plot_dist:
                assert self.cvar_num > 0, "cvar_num should be greater than 0 if plot_dist"
                spec = self.fig.add_gridspec(7, 3 + self.cvar_num)

                self.axis_title = self.fig.add_subplot(spec[0:2, :])
                self.axis_title.text(-0.9, 0.5, "Adaptive IQN performance", fontweight="bold", fontsize=45)
                self.axis_title.text(-0.9, 0, "1. Equivalent to a greedy agent when no obstcles are detected",
                                     fontsize=20)
                self.axis_title.text(-0.9, -0.5, "2. Risk sensitivity increases when approaching obstacles",
                                     fontsize=20)

                self.axis_goal = self.fig.add_subplot(spec[2, 0])
                self.axis_perception = self.fig.add_subplot(spec[3:5, 0])
                self.axis_dvl = self.fig.add_subplot(spec[5:, 0])
                self.axis_graph = self.fig.add_subplot(spec[2:, 1:3])
                for i in range(self.cvar_num):
                    self.axis_dist.append(self.fig.add_subplot(spec[2:, 3 + i]))
            elif self.plot_qvalues:
                spec = self.fig.add_gridspec(13, 4)

                self.axis_title = self.fig.add_subplot(spec[0:3, :])
                self.axis_title.text(-0.9, 0, "DQN performance", fontweight="bold", fontsize=45)
                self.axis_title.text(-0.9, -0.5,
                                     "Robust to current disturbance in robot motion, but not cautious enough when approaching obstacles",
                                     fontsize=20)

                self.axis_goal = self.fig.add_subplot(spec[3:5, 0])
                self.axis_perception = self.fig.add_subplot(spec[5:9, 0])
                self.axis_dvl = self.fig.add_subplot(spec[9:, 0])
                self.axis_graph = self.fig.add_subplot(spec[3:, 1:3])
                self.axis_qvalues = self.fig.add_subplot(spec[3:, 3])
            else:
                name = ""
                if self.agent == "APF":
                    name = "Artificial Potential Field"
                elif self.agent == "BA":
                    name = "Bug Algorithm"
                spec = self.fig.add_gridspec(13, 8)

                self.axis_title = self.fig.add_subplot(spec[0:3, :])
                self.axis_title.text(-0.9, 0, f"{name} performance", fontweight="bold", fontsize=45)
                self.axis_title.text(-0.9, -0.5, "Significantly affected by current disturbance", fontsize=20)

                self.left_margin = self.fig.add_subplot(spec[3:5, 0])
                self.left_margin.set_xticks([])
                self.left_margin.set_yticks([])
                self.left_margin.spines["left"].set_visible(False)
                self.left_margin.spines["top"].set_visible(False)
                self.left_margin.spines["right"].set_visible(False)
                self.left_margin.spines["bottom"].set_visible(False)

                self.axis_goal = self.fig.add_subplot(spec[3:5, 1:3])
                self.axis_perception = self.fig.add_subplot(spec[5:9, 1:3])
                self.axis_dvl = self.fig.add_subplot(spec[9:, 1:3])
                self.axis_graph = self.fig.add_subplot(spec[3:, 3:7])
                self.axis_action = self.fig.add_subplot(spec[5:9, 7])

            self.axis_title.set_xlim([-1.0, 1.0])
            self.axis_title.set_ylim([-1.0, 1.0])
            self.axis_title.set_xticks([])
            self.axis_title.set_yticks([])
            self.axis_title.spines["left"].set_visible(False)
            self.axis_title.spines["top"].set_visible(False)
            self.axis_title.spines["right"].set_visible(False)
            self.axis_title.spines["bottom"].set_visible(False)
        else:
            # Mode 1 (default): Display an episode
            self.fig = plt.figure(figsize=(32, 16))
            spec = self.fig.add_gridspec(5, 4)
            self.axis_graph = self.fig.add_subplot(spec[:, :2])
            self.axis_grid = self.fig.add_subplot(spec[:, 2:4])
            # self.axis_goal = self.fig.add_subplot(spec[0, 2])
            # self.axis_perception = self.fig.add_subplot(spec[1:3, 2])
            # self.axis_dvl = self.fig.add_subplot(spec[3:, 2])
            # self.axis_observation = self.fig.add_subplot(spec[:, 3])

        if self.draw_envs and env_configs is not None:
            for i, env_config in enumerate(env_configs):
                self.load_env_config(env_config)
                self.plot_graph(self.axis_graphs[i])
        else:
            self.plot_graph(self.axis_graph)

    def plot_grid(self, axis, probability = True, information = True):
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
                                           virtual_landmark.information, self.axis_grid, nstd = self.virtual_map.cell_size * 0.04)

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
        if self.draw_envs:
            x_pos = list(np.linspace(0.0, self.env.width, 100))
            y_pos = list(np.linspace(0.0, self.env.height, 100))
        else:
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

        if not self.draw_envs:
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
        if self.draw_envs:
            axis.set_xlim([0.0, self.env.width])
            axis.set_ylim([0.0, self.env.height])
        else:
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
            self.axis_graph.scatter(goal[0], goal[1], marker=".", color="yellow", s=500, zorder=5)
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

    def plot_action_and_steer_state(self, action):
        self.axis_action.clear()

        a, w = self.env.robot.actions[action]

        if self.video_plots:
            self.axis_action.text(0, 3, "Action", fontsize=15)
            self.axis_action.text(0, 2, f"a: {a:.2f}", fontsize=15)
            self.axis_action.text(0, 1, f"w: {w:.2f}", fontsize=15)
            self.axis_action.set_ylim([0, 4])
        else:
            x_pos = 0.15
            self.axis_action.text(x_pos, 6, "Steer actions", fontweight="bold", fontsize=15)
            self.axis_action.text(x_pos, 5, f"Acceleration (m/s^2): {a:.2f}", fontsize=15)
            self.axis_action.text(x_pos, 4, f"Angular velocity (rad/s): {w:.2f}", fontsize=15)

            # robot steer state
            self.axis_action.text(x_pos, 2, "Steer states", fontweight="bold", fontsize=15)
            self.axis_action.text(x_pos, 1, f"Forward speed (m/s): {self.env.robot.speed:.2f}", fontsize=15)
            self.axis_action.text(x_pos, 0, f"Orientation (rad): {self.env.robot.theta:.2f}", fontsize=15)
            self.axis_action.set_ylim([-1, 7])

        self.axis_action.set_xticks([])
        self.axis_action.set_yticks([])
        self.axis_action.spines["left"].set_visible(False)
        self.axis_action.spines["top"].set_visible(False)
        self.axis_action.spines["right"].set_visible(False)
        self.axis_action.spines["bottom"].set_visible(False)

    def plot_measurements(self, robot_idx, R_matrix=None):
        self.axis_perception.clear()
        self.axis_observation.clear()
        self.axis_dvl.clear()
        self.axis_goal.clear()
        for plot in self.LiDAR_beams_plot:
            plot[0].remove()
        self.LiDAR_beams_plot.clear()

        rob = self.env.robots[robot_idx]

        legend_size = 12
        font_size = 15

        rob.perception_output(self.env.obstacles, self.env.robots)

        # # plot Sonar beams in the world frame
        # for point in self.env.robot.sonar.reflections:
        #     x = point[0]
        #     y = point[1]
        #     if point[-1] == 0:
        #         # compute beam range end point 
        #         x = self.env.robot.x + 0.5 * (x-self.env.robot.x)
        #         y = self.env.robot.y + 0.5 * (y-self.env.robot.y)
        #     else:
        #         # mark the reflection point
        #         self.LiDAR_beams_plot.append(self.axis_graph.plot(x,y,marker='x',color='r',zorder=6))

        #     self.LiDAR_beams_plot.append(self.axis_graph.plot([self.env.robot.x,x],[self.env.robot.y,y],linestyle='--',color='r',zorder=6))

        # plot detected objects in the robot frame (rotate x-axis by 90 degree (upward) in the plot)
        c = 'g' if rob.cooperative else 'r'
        self.axis_perception.add_patch(mpl.patches.Circle((0, 0), \
                                                          rob.perception.range, color=c, \
                                                          alpha=0.2))
        self_scale = 2.0
        self.axis_perception.add_patch(
            mpl.patches.Rectangle((-0.5 * self_scale * rob.width, -0.5 * self_scale * rob.length), \
                                  self_scale * rob.width, self_scale * rob.length, color=c))

        x_pos = 0
        y_pos = 0
        relation_pos = [[0.0, 0.0]]

        for i, obs in enumerate(rob.perception.observation["static"]):
            # rotate by 90 degree 
            self.axis_perception.add_patch(mpl.patches.Circle((-obs[1], obs[0]), \
                                                              obs[2], color="m"))
            relation_pos.append([-obs[1], obs[0]])
            # include into observation info
            self.axis_observation.text(x_pos, y_pos,
                                       f"Obstacles {i} position: ({obs[0]:.2f},{obs[1]:.2f}), radius: {obs[2]:.2f}")
            y_pos += 1

        self.axis_observation.text(x_pos, y_pos, "Static obstacles", fontweight="bold", fontsize=15)
        y_pos += 2

        if rob.cooperative:
            for i, obj_history in enumerate(rob.perception.observation["dynamic"]):
                # plot the current position
                pos = obj_history[:2]
                # rotate by 90 degree
                self.axis_perception.add_patch(mpl.patches.Circle((-pos[1], pos[0]),
                                                                  rob.detect_r, color="b"))
                relation_pos.append([-pos[1], pos[0]])
                # include history into observation info
                # for ob in obj_history:
                self.axis_observation.text(x_pos, y_pos,
                                           f"position: ({obj_history[0]:.2f},{obj_history[1]:.2f}),"
                                           f" velocity: ({obj_history[2]:.2f},{obj_history[3]:.2f})")
                y_pos += 1

                # self.axis_observation.text(x_pos, y_pos, f"Object {i} history", fontweight="bold", fontsize=15)
                # y_pos += 2

            self.axis_observation.text(x_pos, y_pos, "Dynamic objects", fontweight="bold", fontsize=15)
            y_pos += 2

        if R_matrix is not None:
            # plot relation matrix
            length = len(R_matrix)
            assert len(relation_pos) == length, "The number of objects do not match size of the relation matrix"
            for i in range(length):
                for j in range(length):
                    self.axis_perception.plot([relation_pos[i][0], relation_pos[j][0]], \
                                              [relation_pos[i][1], relation_pos[j][1]],
                                              linewidth=2 * R_matrix[i][j], color='k', zorder=0)

        type = "cooperative" if rob.cooperative else "non-cooperative"
        self.axis_observation.text(x_pos, y_pos, f"Showing the observation of robot {robot_idx} ({type})",
                                   fontweight="bold", fontsize=20)

        self.axis_perception.set_xlim([-rob.perception.range - 1, rob.perception.range + 1])
        self.axis_perception.set_ylim([-rob.perception.range - 1, rob.perception.range + 1])
        self.axis_perception.set_aspect('equal')
        self.axis_perception.set_title('Perception output', fontsize=font_size)

        self.axis_perception.set_xticks([])
        self.axis_perception.set_yticks([])
        self.axis_perception.spines["left"].set_visible(False)
        self.axis_perception.spines["top"].set_visible(False)
        self.axis_perception.spines["right"].set_visible(False)
        self.axis_perception.spines["bottom"].set_visible(False)

        self.axis_observation.set_ylim([-1, y_pos + 1])
        self.axis_observation.set_xticks([])
        self.axis_observation.set_yticks([])
        self.axis_observation.spines["left"].set_visible(False)
        self.axis_observation.spines["top"].set_visible(False)
        self.axis_observation.spines["right"].set_visible(False)
        self.axis_observation.spines["bottom"].set_visible(False)

        # plot robot velocity in the robot frame (rotate x-axis by 90 degree (upward) in the plot)
        abs_velocity_r = rob.perception.observation["self"][2:]
        h1 = self.axis_dvl.arrow(0.0, 0.0, 0.0, 1.0, \
                                 color='k', \
                                 width=0.02, \
                                 head_width=0.08, \
                                 head_length=0.12, \
                                 length_includes_head=True, \
                                 label='steering direction')
        # rotate by 90 degree
        h2 = self.axis_dvl.arrow(0.0, 0.0, -abs_velocity_r[1], abs_velocity_r[0], \
                                 color='r', width=0.02, head_width=0.08, \
                                 head_length=0.12, length_includes_head=True, \
                                 label='velocity wrt seafloor')
        x_range = np.max([2, np.abs(abs_velocity_r[1])])
        y_range = np.max([2, np.abs(abs_velocity_r[0])])
        mpl.rcParams["font.size"] = 12
        self.axis_dvl.set_xlim([-x_range, x_range])
        self.axis_dvl.set_ylim([-1, y_range])
        self.axis_dvl.set_aspect('equal')
        self.axis_dvl.legend(handles=[h1, h2], loc='lower center', fontsize=legend_size)
        self.axis_dvl.set_title('Velocity Measurement', fontsize=font_size)

        self.axis_dvl.set_xticks([])
        self.axis_dvl.set_yticks([])
        self.axis_dvl.spines["left"].set_visible(False)
        self.axis_dvl.spines["top"].set_visible(False)
        self.axis_dvl.spines["right"].set_visible(False)
        self.axis_dvl.spines["bottom"].set_visible(False)

        # give goal position info in the robot frame
        goal_r = rob.perception.observation["self"][:2]
        x1 = 0.07
        x2 = x1 + 0.13
        self.axis_goal.text(x1, 0.5, "Goal Position (Relative)", fontsize=font_size)
        self.axis_goal.text(x2, 0.25, f"({goal_r[0]:.2f}, {goal_r[1]:.2f})", fontsize=font_size)

        self.axis_goal.set_xticks([])
        self.axis_goal.set_yticks([])
        self.axis_goal.spines["left"].set_visible(False)
        self.axis_goal.spines["top"].set_visible(False)
        self.axis_goal.spines["right"].set_visible(False)
        self.axis_goal.spines["bottom"].set_visible(False)

    def plot_return_dist(self, action):
        for axis in self.axis_dist:
            axis.clear()

        dist_interval = 1
        mean_bar = 0.35
        idx = 0

        xlim = [np.inf, -np.inf]
        for idx, cvar in enumerate(action["cvars"]):
            ylabelright = []

            quantiles = np.array(action["quantiles"][idx])

            q_means = np.mean(quantiles, axis=0)
            max_a = np.argmax(q_means)
            for i, a in enumerate(self.env.robot.actions):
                q_mean = q_means[i]
                # q_mean = np.mean(quantiles[:,i])

                ylabelright.append(
                    "\n".join([f"a: {a[0]:.2f}", f"w: {a[1]:.2f}"])
                )

                # ylabelright.append(f"mean: {q_mean:.2f}")

                self.axis_dist[idx].axhline(i * dist_interval, color="black", linewidth=0.5, zorder=0)
                self.axis_dist[idx].scatter(quantiles[:, i], i * np.ones(len(quantiles[:, i])) * dist_interval,
                                            color="g", marker="x", s=80, linewidth=3)
                self.axis_dist[idx].hlines(y=i * dist_interval, xmin=np.min(quantiles[:, i]),
                                           xmax=np.max(quantiles[:, i]), zorder=0)
                if i == max_a:
                    self.axis_dist[idx].vlines(q_mean, ymin=i * dist_interval - mean_bar,
                                               ymax=i * dist_interval + mean_bar, color="red", linewidth=5)
                else:
                    self.axis_dist[idx].vlines(q_mean, ymin=i * dist_interval - mean_bar,
                                               ymax=i * dist_interval + mean_bar, color="blue", linewidth=3)

            self.axis_dist[idx].tick_params(axis="x", labelsize=14)
            self.axis_dist[idx].set_ylim([-1.0, i + 1])
            self.axis_dist[idx].set_yticks([])
            if idx == len(action["cvars"]) - 1:
                self.axis_dist[idx].set_yticks(range(0, i + 1))
                self.axis_dist[idx].yaxis.tick_right()
                self.axis_dist[idx].set_yticklabels(labels=ylabelright, fontsize=12)
            if idx == 0:
                self.axis_dist[idx].set_title("adpative " + r'$\phi$' + f" = {cvar:.2f}", fontsize=15)
            else:
                self.axis_dist[idx].set_title(r'$\phi$' + f" = {cvar:.2f}", fontsize=15)
            xlim[0] = min(xlim[0], np.min(quantiles) - 5)
            xlim[1] = max(xlim[1], np.max(quantiles) + 5)

        for idx, cvar in enumerate(action["cvars"]):
            # self.axis_dist[idx].xaxis.set_ticks(np.arange(xlim[0],xlim[1]+1,(xlim[1]-xlim[0])/5))
            self.axis_dist[idx].set_xlim(xlim)

    def plot_action_qvalues(self, action):
        self.axis_qvalues.clear()

        dist_interval = 1
        mean_bar = 0.35
        ylabelright = []

        q_values = np.array(action["qvalues"])
        max_a = np.argmax(q_values)
        for i, a in enumerate(self.env.robot.actions):
            ylabelright.append(
                "\n".join([f"a: {a[0]:.2f}", f"w: {a[1]:.2f}"])
            )
            self.axis_qvalues.axhline(i * dist_interval, color="black", linewidth=1, zorder=0)
            if i == max_a:
                self.axis_qvalues.vlines(q_values[i], ymin=i * dist_interval - mean_bar,
                                         ymax=i * dist_interval + mean_bar, color="red", linewidth=8)
            else:
                self.axis_qvalues.vlines(q_values[i], ymin=i * dist_interval - mean_bar,
                                         ymax=i * dist_interval + mean_bar, color="blue", linewidth=5)

        self.axis_qvalues.set_title("Action Values", fontsize=15)
        self.axis_qvalues.tick_params(axis="x", labelsize=15)
        self.axis_qvalues.set_ylim([-1.0, i + 1])
        self.axis_qvalues.set_yticks(range(0, i + 1))
        self.axis_qvalues.yaxis.tick_right()
        self.axis_qvalues.set_yticklabels(labels=ylabelright, fontsize=14)
        self.axis_qvalues.set_xlim([np.min(q_values) - 5, np.max(q_values) + 5])

    def one_step(self, actions, slam_signal=False, robot_idx=0):
        assert len(actions) == len(self.env.robots), "Number of actions not equal number of robots!"
        for i, action in enumerate(actions):
            rob = self.env.robots[i]
            if rob.reach_goal:
                continue
            current_velocity = self.env.get_velocity(rob.x, rob.y)
            rob.update_state(action, current_velocity)

        self.plot_robots()
        # self.plot_measurements(robot_idx)
        # if not self.plot_dist and not self.plot_qvalues:
        #     self.plot_action_and_steer_state(action["action"])

        if self.step % self.env.robots[0].N == 0:
            if self.plot_dist:
                self.plot_return_dist(action)
            elif self.plot_qvalues:
                self.plot_action_qvalues(action)

        self.step += 1

    def init_animation(self):
        # plot initial robot position
        self.plot_robots()

        # plot initial measurments
        # self.plot_measurements() 

    def visualize_control(self, action_sequence, start_idx=0):
        # update robot state and make animation when executing action sequence
        actions = []

        # counter for updating distributions plot
        self.step = start_idx

        for i, a in enumerate(action_sequence):
            for _ in range(self.env.robots[0].N):
                actions.append(a)

        if self.video_plots:
            for i, action in enumerate(actions):
                self.one_step(action)
                self.fig.savefig(f"{self.plots_save_dir}/step_{self.step}.png", pad_inches=0.2, dpi=self.dpi)
        else:
            # self.animation = animation.FuncAnimation(self.fig, self.one_step,frames=actions, \
            #                                         init_func=self.init_animation,
            #                                         interval=10,repeat=False)
            for i, action in enumerate(actions):
                self.one_step(action)
            plt.show()

    def initialize_apf_agents(self):
        self.APF_agents = []
        for robot in self.env.robots:
            self.APF_agents.append(APF_agent(robot.a, robot.w))

    def navigate_one_step(self, path, video = False):
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
                    self.fig.savefig(path + str(self.cnt) + ".png",bbox_inches="tight")
                    self.cnt += 1
            odom_cnt += 1

            actions = []
            for i, apf in enumerate(self.APF_agents):
                if self.env.robots[i].reach_goal:
                    actions.append(-1)
                    reached += 1
                else:
                    actions.append(apf.act(observations[i][0]))
            if (reached == self.env.num_cooperative):
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

    def visualize_navigation(self, start_idx=0):
        # counter for updating distributions plot
        self.step = start_idx

        if self.video_plots:
            self.navigate_one_step()
            self.fig.savefig(f"{self.plots_save_dir}/step_{self.step}.png", pad_inches=0.2, dpi=self.dpi)
        else:
            self.navigate_one_step()
            # self.animation = animation.FuncAnimation(self.fig, self.one_step,frames=actions, \
            #                                         init_func=self.init_animation,
            #                                         interval=10,repeat=False)

            plt.show()

    def visualize_SLAM(self, start_idx=0):
        color_list = ['tab:pink', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange', 'tab:gray', 'tab:olive']
        init_x = self.env.robots[0].start[0]
        init_y = self.env.robots[0].start[1]
        for i in range(self.env.num_cooperative):
            pose = self.landmark_slam.get_robot_trajectory(i, [init_x, init_y,
                                                               self.env.robots[0].init_theta])
            self.axis_grid.plot(pose[:, 0], pose[:, 1], color=color_list[i], linewidth=2, zorder=10)
            # self.axis_graph.scatter(pose[:,0],pose[:,1], marker="*", color="pink", s=500, zorder=5)
        landmark_list = self.landmark_slam.get_landmark_list([init_x, init_y,
                                                              self.env.robots[0].init_theta])
        for landmark_obs in landmark_list:
            self.axis_grid.plot(landmark_obs[0], landmark_obs[1], '.', color='tab:orange')

    def load_env_config(self, episode_dict):
        episode = copy.deepcopy(episode_dict)

        # load env config
        self.env.sd = episode["env"]["seed"]
        self.env.width = episode["env"]["width"]
        self.env.height = episode["env"]["height"]
        self.env.r = episode["env"]["r"]
        self.env.v_rel_max = episode["env"]["v_rel_max"]
        self.env.p = episode["env"]["p"]
        self.env.v_range = copy.deepcopy(episode["env"]["v_range"])
        self.env.obs_r_range = copy.deepcopy(episode["env"]["obs_r_range"])
        self.env.clear_r = episode["env"]["clear_r"]
        self.env.start = np.array(episode["env"]["start"])
        self.env.goal = np.array(episode["env"]["goal"])
        self.env.goal_dis = episode["env"]["goal_dis"]
        self.env.timestep_penalty = episode["env"]["timestep_penalty"]
        # self.env.energy_penalty = np.matrix(episode["env"]["energy_penalty"])
        self.env.collision_penalty = episode["env"]["collision_penalty"]
        self.env.goal_reward = episode["env"]["goal_reward"]
        self.env.discount = episode["env"]["discount"]

        # load vortex cores
        self.env.cores.clear()
        centers = None
        for i in range(len(episode["env"]["cores"]["positions"])):
            center = episode["env"]["cores"]["positions"][i]
            clockwise = episode["env"]["cores"]["clockwise"][i]
            Gamma = episode["env"]["cores"]["Gamma"][i]
            core = marinenav_env.Core(center[0], center[1], clockwise, Gamma)
            self.env.cores.append(core)
            if centers is None:
                centers = np.array([[core.x, core.y]])
            else:
                c = np.array([[core.x, core.y]])
                centers = np.vstack((centers, c))

        if centers is not None:
            self.env.core_centers = scipy.spatial.KDTree(centers)

        # load obstacles
        self.env.obstacles.clear()
        centers = None
        for i in range(len(episode["env"]["obstacles"]["positions"])):
            center = episode["env"]["obstacles"]["positions"][i]
            r = episode["env"]["obstacles"]["r"][i]
            obs = marinenav_env.Obstacle(center[0], center[1], r)
            self.env.obstacles.append(obs)
            if centers is None:
                centers = np.array([[obs.x, obs.y]])
            else:
                c = np.array([[obs.x, obs.y]])
                centers = np.vstack((centers, c))

        if centers is not None:
            self.env.obs_centers = scipy.spatial.KDTree(centers)

        # load robot config
        self.env.robot.dt = episode["robot"]["dt"]
        self.env.robot.N = episode["robot"]["N"]
        self.env.robot.length = episode["robot"]["length"]
        self.env.robot.width = episode["robot"]["width"]
        self.env.robot.r = episode["robot"]["r"]
        self.env.robot.max_speed = episode["robot"]["max_speed"]
        self.env.robot.a = np.array(episode["robot"]["a"])
        self.env.robot.w = np.array(episode["robot"]["w"])
        self.env.robot.compute_k()
        self.env.robot.compute_actions()
        self.env.robot.init_theta = episode["robot"]["init_theta"]
        self.env.robot.init_speed = episode["robot"]["init_speed"]

        # load sonar config
        self.env.robot.sonar.range = episode["robot"]["sonar"]["range"]
        self.env.robot.sonar.angle = episode["robot"]["sonar"]["angle"]
        self.env.robot.sonar.num_beams = episode["robot"]["sonar"]["num_beams"]
        self.env.robot.sonar.compute_phi()
        self.env.robot.sonar.compute_beam_angles()

        # load action sequence
        self.episode_actions = copy.deepcopy(episode["robot"]["action_history"])

        # update env action and observation space
        # self.env.action_space = gym.spaces.Discrete(self.env.robot.compute_actions_dimension())
        obs_len = 2 + 2 + 2 * self.env.robot.sonar.num_beams
        # self.env.observation_space = gym.spaces.Box(low = -np.inf * np.ones(obs_len), \
        #                                             high = np.inf * np.ones(obs_len), \
        #                                             dtype = np.float32)

        if self.plot_dist:
            # load action cvars, quantiles and taus
            self.episode_actions_cvars = episode["robot"]["actions_cvars"]
            self.episode_actions_quantiles = episode["robot"]["actions_quantiles"]
            self.episode_actions_taus = episode["robot"]["actions_taus"]
        elif self.plot_qvalues:
            # load action values
            self.episode_actions_values = episode["robot"]["actions_values"]

    def load_env_config_from_eval_files(self, config_f, eval_f, eval_id, env_id):
        with open(config_f, "r") as f:
            episodes = json.load(f)
        episode = episodes[f"env_{env_id}"]
        eval_file = np.load(eval_f, allow_pickle=True)
        episode["robot"]["action_history"] = copy.deepcopy(eval_file["actions"][eval_id][env_id])
        self.load_env_config(episode)

    def load_env_config_from_json_file(self, filename):
        with open(filename, "r") as f:
            episode = json.load(f)
        self.load_env_config(episode)

    # def play_episode(self,start_idx=0):
    #     for plot in self.robot_traj_plot:
    #         plot[0].remove()
    #     self.robot_traj_plot.clear()

    #     current_v = self.env.get_velocity(self.env.start[0],self.env.start[1])
    #     self.env.robot.reset_state(self.env.start[0],self.env.start[1], current_velocity=current_v)

    #     self.init_visualize()

    #     self.visualize_control(self.episode_actions,start_idx)

    def load_eval_config_and_episode(self, config_file, eval_file):
        with open(config_file, "r") as f:
            self.configs = json.load(f)

        self.episodes = np.load(eval_file, allow_pickle=True)

    def play_eval_episode(self, eval_id, episode_id, robot_id):
        self.env.reset_with_eval_config(self.configs[episode_id])
        self.init_visualize()

        relations = self.episodes["relations"][eval_id][episode_id][robot_id]
        actions = self.episodes["actions"][eval_id][episode_id]

        for i in range(len(relations)):
            self.plot_robots()
            self.plot_measurements(robot_id, relations[i][0])
            action = [actions[j][i] for j in range(len(self.env.robots))]
            self.env.step(action)
            plt.show(block=False)
            plt.pause(1)

    def draw_trajectory(self,
                        only_ep_actions: bool = True,  # only draw the resulting trajectory of actions in episode data
                        all_actions: dict = None,  # otherwise, draw all trajectories from given action sequences
                        fork_state_info: dict = None  # if fork state is given, plot action distributions
                        ):
        # Used in Mode 3
        for plot in self.robot_traj_plot:
            plot[0].remove()
        self.robot_traj_plot.clear()

        self.init_visualize()

        if only_ep_actions:
            all_actions = dict(ep_agent=self.episode_actions)

        plot_fork_state = True
        trajs = []
        for actions in all_actions.values():
            traj = None
            current_v = self.env.get_velocity(self.env.start[0], self.env.start[1])
            self.env.robot.reset_state(self.env.start[0], self.env.start[1], current_velocity=current_v)
            for idx, a in enumerate(actions):

                if fork_state_info is not None and plot_fork_state:
                    if fork_state_info["id"] == idx:
                        self.plot_robot()
                        self.plot_measurements()
                        self.plot_return_dist(fork_state_info)
                        plot_fork_state = False

                for _ in range(self.env.robot.N):
                    current_velocity = self.env.get_velocity(self.env.robot.x, self.env.robot.y)
                    self.env.robot.update_state(a, current_velocity)
                    curr = np.array([[self.env.robot.x, self.env.robot.y]])
                    if traj is None:
                        traj = curr
                    else:
                        traj = np.concatenate((traj, curr))
            trajs.append(traj)

        colors = ['tab:orange', 'lime', 'r', 'b']
        styles = ['solid', 'dashed', 'dashdot', 'dashdot']

        for i, l in enumerate(all_actions.keys()):
            traj = trajs[i]
            self.axis_graph.plot(traj[:, 0], traj[:, 1], label=l, linewidth=2, zorder=4 + i, color=colors[i],
                                 linestyle=styles[i])

        mpl.rcParams["font.size"] = 15
        mpl.rcParams["legend.framealpha"] = 0.4
        self.axis_graph.legend(loc='upper left', bbox_to_anchor=(0.18, 0.95))
        self.axis_graph.set_xticks([])
        self.axis_graph.set_yticks([])

        self.fig.savefig(f"trajectory_test.png", bbox_inches="tight", dpi=self.dpi)

    def draw_video_plots(self, episode, save_dir, start_idx, agent):
        # Used in Mode 4
        self.agent = agent
        self.load_env_config(episode)
        self.plots_save_dir = save_dir
        self.play_episode(start_idx)
        return self.step

    def draw_present_position(self):
        pass
        for robot in self.env.robots:
            self.axis_graph.scatter(robot.x, robot.y, marker="*", color="yellow", s=500, zorder=5)
