import numpy as np
import copy


def theta_0_to_2pi(theta):
    while theta < 0.0:
        theta += 2 * np.pi
    while theta >= 2 * np.pi:
        theta -= 2 * np.pi
    return theta


def world_to_local(x, y, theta, origin):
    # compute transformation from world frame to robot frame
    R_wr, t_wr = pose_vector_to_matrix(origin[0], origin[1], origin[2])
    R_rw = np.transpose(R_wr)
    t_rw = -R_rw * t_wr

    R_this, t_this = pose_vector_to_matrix(x, y, theta)
    t_this = R_rw * t_this + t_rw

    return t_this[0, 0], t_this[1, 0], theta_0_to_2pi(theta - origin[2])


def pose_vector_to_matrix(x, y, theta):
    # compose matrix expression of pose vector
    R_rw = np.matrix([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    t_rw = np.matrix([[x], [y]])

    return R_rw, t_rw


class Perception:

    def __init__(self, cooperative: bool = False):
        # 2D LiDAR model with detection area as a sector
        self.observation = None
        self.range = 10.0  # range of beams (meter)
        self.angle = 2 * np.pi  # detection angle range
        # self.len_obs_history = 0  # the window size of observation history
        self.observation_format(cooperative)
        self.observed_obs = []  # indices of observed static obstacles
        self.observed_objs = []  # indiced of observed dynamic objects

    def observation_format(self, cooperative: bool = False):
        # format: {"self": [velocity,goal, odom],
        #          "static":[[obs_1.x,obs_1.y,obs_1.r, obs_1.id],...,[obs_n.x,obs_n.y,obs_n.r, obs_n.id]],
        #          "dynamic":{id_1:[[robot_1.x,robot_1.y,robot_1.vx,robot_1.vy,robot_1.theta]_(t-m),...,[]_t]...}
        if cooperative:
            self.observation = dict(self=[], static=[], dynamic=[])
        else:
            self.observation = dict(self=[], static=[])


class Odometry:
    def __init__(self, use_noise=True):
        self.use_noise = use_noise
        if use_noise:
            self.max_g_error = 2 / 180 * np.pi  # max gyro error
            self.max_a_error = 0.02  # max acceleration error
            self.max_a_p_error = 0.02  # max acceleration error percentage
            z_score = 1.96  # 95% confidence interval

            self.sigma_g = self.max_g_error / z_score
            self.sigma_a = self.max_a_error / z_score  # accel noise
            self.sigma_a_p = self.max_a_p_error / z_score  # accel noise percentage
        else:
            self.max_g_error = 0
            self.max_a_error = 0
            self.max_a_p_error = 0
            self.sigma_g = 0
            self.sigma_a = 0
            self.sigma_a_p = 0

        self.x_old = None
        self.y_old = None
        self.theta_old = None
        self.observation = [0, 0, 0]

    def reset(self, x0, y0, theta0):
        self.x_old = x0
        self.y_old = y0
        self.theta_old = theta0

    def add_noise(self, x_new, y_new, theta_new):
        if self.x_old is None:
            raise ValueError("Odometry is not initialized!")

        dx, dy, dtheta = world_to_local(x_new, y_new, theta_new,
                                        [self.x_old, self.y_old, self.theta_old])

        self.x_old = copy.deepcopy(x_new)
        self.y_old = copy.deepcopy(y_new)
        self.theta_old = copy.deepcopy(theta_new)

        if self.use_noise:
            # max acceleration 5 cm / 5% of measurement
            sigma_a_p = self.sigma_a_p * np.linalg.norm([dx, dy])
            sigma_a = np.min([self.sigma_a, sigma_a_p])

            max_a_p_error = self.max_a_p_error * np.linalg.norm([dx, dy])
            max_a_error = np.min([max_a_p_error, self.max_a_error])

            w_a = np.random.normal(0, sigma_a)
            w_g = np.random.normal(0, self.sigma_g)

            dtheta_noisy = theta_0_to_2pi(dtheta + np.clip(w_g, None, self.max_g_error))
            dx_noisy = dx + np.clip(w_a, None, max_a_error) * np.cos(dtheta_noisy)
            dy_noisy = dy + np.clip(w_a, None, max_a_error) * np.sin(dtheta_noisy)
        else:
            dx_noisy = dx
            dy_noisy = dy
            dtheta_noisy = dtheta

        self.observation = [dx_noisy, dy_noisy, dtheta_noisy]

    def get_odom(self):
        return self.observation


class RangeBearingMeasurement:
    def __init__(self, use_noise = True):
        self.use_noise = use_noise
        if use_noise:
            self.max_b_error = 0.2 / 180 * np.pi  # max bearing error
            self.max_r_error = 0.1  # max range error
            z_score = 1.96  # 95% confidence interval

            self.sigma_r = self.max_r_error / z_score
            self.sigma_b = self.max_b_error / z_score
        else:
            self.max_r_error = 0
            self.max_b_error = 0
            self.sigma_r = 0
            self.sigma_b = 0

    def add_noise(self, x_obs, y_obs):
        r = np.linalg.norm([x_obs, y_obs])
        b = np.arccos(x_obs / r)  # [0, PI]
        if y_obs < 0:
            b = 2 * np.pi - b
        if self.use_noise:
            r_noise = np.random.normal(0, self.sigma_r)
            r_noise = np.clip(r_noise, None, self.max_r_error)

            b_noise = np.random.normal(0, self.sigma_b)
            b_noise = np.clip(b_noise, None, self.max_b_error)
        else:
            r_noise = 0
            b_noise = 0

        return [r + r_noise, b + b_noise]

    def add_noise_point_based(self, robot_pose, landmark_position):
        x_r = np.reshape(np.array(landmark_position), (2, 1))

        R_wr, t_wr = pose_vector_to_matrix(robot_pose[0], robot_pose[1], robot_pose[2])

        R_rw = np.transpose(R_wr)
        t_rw = -R_rw * t_wr

        x_r = R_rw * x_r + t_rw
        x_r.resize((2,))

        return self.add_noise(x_r[0], x_r[1])


class RobotNeighborMeasurement:
    def __init__(self, use_noise = True):
        self.use_noise = use_noise
        if use_noise:
            self.max_b_error = 0.2 / 180 * np.pi  # max bearing error
            self.max_r_error = 0.05  # max range error
            z_score = 1.96  # 95% confidence interval

            self.sigma_r = self.max_r_error / z_score
            self.sigma_b = self.max_b_error / z_score
        else:
            self.max_b_error = 0
            self.max_r_error = 0
            self.sigma_r = 0
            self.sigma_b = 0

    def add_noise(self, x_obs, y_obs, theta_obs):
        if self.use_noise:
            x_noise = np.random.normal(0, self.sigma_r)
            y_noise = np.random.normal(0, self.sigma_r)

            b_noise = np.random.normal(0, self.sigma_b)

            x_noisy = x_obs + np.clip(x_noise, None, self.max_r_error)
            y_noisy = y_obs + np.clip(y_noise, None, self.max_r_error)
            theta_noisy = theta_0_to_2pi(theta_obs + np.clip(b_noise, None, self.max_b_error))
        else:
            x_noisy = x_obs
            y_noisy = y_obs
            theta_noisy = theta_obs

        return [x_noisy, y_noisy, theta_noisy]

    def add_noise_pose_based(self, pose_local, pose_observed):
        x_r = np.reshape(np.array(pose_observed[0], pose_observed[1]), (2, 1))

        R_wr, t_wr = pose_vector_to_matrix(pose_local[0], pose_local[1], pose_local[2])

        R_rw = np.transpose(R_wr)
        t_rw = -R_rw * t_wr

        x_r = R_rw * x_r + t_rw
        x_r.resize((2,))

        return self.add_noise(x_r[0], x_r[1], theta_0_to_2pi(pose_observed[2]-pose_local[2]))


class Robot:

    def __init__(self, cooperative: bool = False):

        # parameter initialization
        self.cooperative = cooperative  # if the robot is cooperative or not
        self.dt = .2  # discretized time step (second)
        self.N = 5  # number of time step per action
        self.perception = Perception(cooperative)
        self.length = 1.0
        self.width = 0.5
        self.r = 0.8  # collision range
        self.detect_r = 0.5 * np.sqrt(self.length ** 2 + self.width ** 2)  # detection range
        self.goal_dis = 2.0  # max distance to goal considered as reached
        self.max_speed = 2.0
        self.a = np.array([-0.4, 0.0, 0.4])  # linear accelerations (m/s^2)
        self.w = np.array([-np.pi / 6, 0.0, np.pi / 6])  # angular velocities (rad/s)
        self.compute_k()  # cofficient of water resistance
        self.compute_actions()  # list of actions

        self.x = None  # x coordinate
        self.y = None  # y coordinate
        self.theta = None  # steering heading angle
        self.speed = None  # steering foward speed
        self.velocity = None  # velocity wrt sea floor

        self.start = None  # start position
        self.goal = None  # goal position
        self.reach_goal = False

        self.init_theta = 0.0  # theta at initial position
        self.init_speed = 0.0  # speed at initial position

        self.action_history = []  # history of action commands in one episode
        self.trajectory = []  # trajectory in one episode

        self.odometry = Odometry()  # odometry
        # add noisy to landmark observation
        self.landmark_observation = RangeBearingMeasurement()
        self.robot_observation = RobotNeighborMeasurement()

    def compute_k(self):
        self.k = np.max(self.a) / self.max_speed

    def compute_actions(self):
        self.actions = [(acc, ang_v) for acc in self.a for ang_v in self.w]

    def compute_actions_dimension(self):
        return len(self.actions)

    def compute_dist_reward_scale(self):
        # scale the distance reward
        return 1 / (self.max_speed * self.N * self.dt)

    def compute_penalty_matrix(self):
        # scale the penalty value to [-1,0]
        scale_a = 1 / (np.max(self.a) * np.max(self.a))
        scale_w = 1 / (np.max(self.w) * np.max(self.w))
        p = -0.5 * np.matrix([[scale_a, 0.0], [0.0, scale_w]])
        return p

    def compute_action_energy_cost(self, action):
        # scale the a and w to [0,1]
        a, w = self.actions[action]
        a /= np.max(self.a)
        w /= np.max(self.w)
        return np.abs(a) + np.abs(w)

    def dist_to_goal(self):
        return np.linalg.norm(self.goal - np.array([self.x, self.y]))

    def check_reach_goal(self):
        if self.dist_to_goal() <= self.goal_dis:
            self.reach_goal = True

    def reset_goal(self, goal_this):
        self.goal = np.array(goal_this)
        self.reach_goal = False

    def reset_state(self, current_velocity=np.zeros(2)):
        # only called when resetting the environment
        self.action_history.clear()
        self.trajectory.clear()
        self.x = self.start[0]
        self.y = self.start[1]
        self.theta = self.init_theta
        self.speed = self.init_speed
        self.update_velocity(current_velocity)
        self.trajectory.append([self.x, self.y, self.theta, self.speed, self.velocity[0], self.velocity[1]])

        self.odometry.reset(self.x, self.y, self.theta)

    def get_robot_transform(self):
        # compute transformation from world frame to robot frame
        R_wr = np.matrix([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        t_wr = np.matrix([[self.x], [self.y]])
        return R_wr, t_wr

    def get_steer_velocity(self, speed=None, theta=None):
        if speed is None and theta is None:
            # Case when no parameters are provided
            return self.speed * np.array([np.cos(self.theta), np.sin(self.theta)])
        else:
            # Case when speed and theta are provided
            return speed * np.array([np.cos(theta), np.sin(theta)])

    def update_velocity(self, current_velocity=np.zeros(2), velocity=None, theta=None):
        if velocity is None and theta is None:
            steer_velocity = self.get_steer_velocity()
            self.velocity = steer_velocity + current_velocity
        else:
            steer_velocity = self.get_steer_velocity(velocity, theta)
            return steer_velocity + current_velocity

    def update_state(self, action, current_velocity):
        # update robot position in one time step
        self.update_velocity(current_velocity)
        dis = self.velocity * self.dt
        self.x += dis[0]
        self.y += dis[1]

        # update robot speed in one time step
        a, w = self.actions[action]

        # assume that water resistance force is proportion to the speed
        # self.speed += (a-self.k*self.speed) * self.dt
        # self.speed = np.clip(self.speed,0.0,self.max_speed)

        # update robot heading angle in one time step
        self.theta += w * self.dt

        # warp theta to [0,2*pi)
        while self.theta < 0.0:
            self.theta += 2 * np.pi
        while self.theta >= 2 * np.pi:
            self.theta -= 2 * np.pi
        # print(self.x, self.y, self.theta/np.pi * 180, self.velocity)
        # if add_noise:
        #     self.odometry.add_noise(self.x, self.y, self.theta)

    def check_collision(self, obj_x, obj_y, obj_r):
        d = np.sqrt((self.x - obj_x) ** 2 + (self.y - obj_y) ** 2)
        if d <= obj_r + self.r:
            return True
        else:
            return False

    def check_detection(self, obj_x, obj_y, obj_r):
        proj_pos = self.project_to_robot_frame(np.array([obj_x, obj_y]), False)

        if np.linalg.norm(proj_pos) > self.perception.range + obj_r:
            return False

        angle = np.arctan2(proj_pos[1], proj_pos[0])
        if angle < -0.5 * self.perception.angle or angle > 0.5 * self.perception.angle:
            return False

        return True

    def project_to_robot_frame(self, x, is_vector=True):
        assert isinstance(x, np.ndarray), "the input needs to be an numpy array"
        assert np.shape(x) == (2,)

        x_r = np.reshape(x, (2, 1))

        R_wr, t_wr = self.get_robot_transform()

        R_rw = np.transpose(R_wr)
        t_rw = -R_rw * t_wr

        if is_vector:
            x_r = R_rw * x_r
        else:
            x_r = R_rw * x_r + t_rw

        x_r.resize((2,))
        return np.array(x_r)

    def perception_output(self, obstacles, robots):
        # TODO: remove LiDAR reflection computations and check dynamic obstacle observation error
        if self.reach_goal:
            return None, False, True

        self.perception.observation["static"].clear()
        self.perception.observation["dynamic"].clear()

        ##### self observation (velocity and goal in self frame) #####
        # vehicle velocity wrt seafloor in self frame
        abs_velocity_r = self.project_to_robot_frame(self.velocity)

        # goal position in self frame
        goal_r = self.project_to_robot_frame(self.goal, False)
        self.perception.observation["self"] = [goal_r[0], goal_r[1],
                                               abs_velocity_r[0], abs_velocity_r[1],
                                               self.x, self.y, self.theta]

        ##### observation of other objects #####
        self.perception.observed_obs.clear()
        if self.cooperative:
            self.perception.observed_objs.clear()

        collision = False
        self.check_reach_goal()

        # static obstacles observation 
        for i, obs in enumerate(obstacles):
            if not self.check_detection(obs.x, obs.y, obs.r):
                continue

            self.perception.observed_obs.append(i)

            if not collision:
                collision = self.check_collision(obs.x, obs.y, obs.r)

            pos_r = self.project_to_robot_frame(np.array([obs.x, obs.y]), False)
            self.perception.observation["static"].append([pos_r[0], pos_r[1], obs.r, i])

        if self.cooperative:
            # dynamic objects observation
            for j, robot in enumerate(robots):
                if robot is self:
                    continue
                if robot.reach_goal:
                    # This robot is in the deactivate state, and abscent from the current map
                    continue
                if not self.check_detection(robot.x, robot.y, robot.detect_r):
                    continue

                self.perception.observed_objs.append(j)

                if not collision:
                    collision = self.check_collision(robot.x, robot.y, robot.r)

                pos_r = self.project_to_robot_frame(np.array([robot.x, robot.y]), False)
                v_r = self.project_to_robot_frame(robot.velocity)
                theta_r = theta_0_to_2pi(robot.theta - self.theta)
                new_obs = list([pos_r[0], pos_r[1], v_r[0], v_r[1], theta_r, j])
                self.perception.observation["dynamic"].append(new_obs)
        self_state = copy.deepcopy(self.perception.observation["self"])
        static_states = copy.deepcopy(self.perception.observation["static"])
        if self.cooperative:
            dynamic_states = copy.deepcopy(self.perception.observation["dynamic"])
            obs = (self_state, static_states, dynamic_states)
            return obs, collision, self.reach_goal
        else:
            return (self_state, static_states), collision, self.reach_goal
