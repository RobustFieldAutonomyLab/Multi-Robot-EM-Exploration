import numpy as np
import scipy.spatial
import marinenav_env.envs.utils.robot as robot
# import gym
import json
import copy


class Core:

    def __init__(self, x: float, y: float, clockwise: bool, Gamma: float):
        self.x = x  # x coordinate of the vortex core
        self.y = y  # y coordinate of the vortex core
        self.clockwise = clockwise  # if the vortex direction is clockwise
        self.Gamma = Gamma  # circulation strength of the vortex core


class Obstacle:

    def __init__(self, x: float, y: float, r: float):
        self.x = x  # x coordinate of the obstacle center
        self.y = y  # y coordinate of the obstacle center
        self.r = r  # radius of the obstacle


# class MarineNavEnv2(gym.Env):
class MarineNavEnv2:
    def __init__(self, seed: int = 0, schedule: dict = None):

        self.sd = seed
        self.rd = np.random.RandomState(seed)  # PRNG

        # Define action space and observation space for gym
        # self.action_space = gym.spaces.Discrete(self.robot.compute_actions_dimension())

        # parameter initialization
        self.width = 200  # x coordinate dimension of the map
        self.height = 200  # y coordinate dimension of the map
        self.r = 0.5  # radius of vortex core
        self.v_rel_max = 1.0  # max allowable speed when two currents flowing towards each other
        self.p = 0.8  # max allowable relative speed at another vortex core
        self.v_range = [5, 10]  # speed range of the vortex (at the edge of core)
        self.obs_r_range = [1, 1]  # radius range of the obstacle
        self.clear_r = 5.0  # radius of area centered at the start(goal) of each robot,
        # where no vortex cores, static obstacles, or the start(goal) of other robots exist
        self.reset_start_and_goal = True  # if the start and goal position be set randomly in reset()
        self.start = np.array([5.0, 5.0])  # robot start position
        self.goal = np.array([45.0, 45.0])  # goal position
        self.goal_dis = 2.0  # max distance to goal considered as reached
        self.timestep_penalty = -1.0
        self.collision_penalty = -50.0
        self.goal_reward = 100.0
        self.discount = 0.99
        self.num_cores = 0  # number of vortices
        self.num_obs = 60  # number of static obstacles
        self.min_start_goal_dis = 2.0
        self.num_cooperative = 3  # number of cooperative robots
        self.num_non_cooperative = 0  # number of non-cooperative robots

        self.robots = []  # list of robots
        for _ in range(self.num_cooperative):
            self.robots.append(robot.Robot(cooperative=True))
        for _ in range(self.num_non_cooperative):
            self.robots.append(robot.Robot(cooperative=False))

        self.cores = []  # list of vortex cores
        self.obstacles = []  # list of static obstacles

        self.schedule = schedule  # schedule for curriculum learning
        self.episode_timesteps = 0  # current episode timesteps
        self.total_timesteps = 0  # learning timesteps

        self.set_boundary = False  # set boundary of environment

    def get_action_space_dimension(self):
        return self.robot.compute_actions_dimension()

    def reset(self):
        # reset the environment

        if self.schedule is not None:
            steps = self.schedule["timesteps"]
            diffs = np.array(steps) - self.total_timesteps

            # find the interval the current timestep falls into
            idx = len(diffs[diffs <= 0]) - 1

            self.num_cores = self.schedule["num_cores"][idx]
            self.num_obs = self.schedule["num_obstacles"][idx]
            self.min_start_goal_dis = self.schedule["min_start_goal_dis"][idx]

            print("======== training env setup ========")
            print("num of cores: ", self.num_cores)
            print("num of obstacles: ", self.num_obs)
            print("min start goal dis: ", self.min_start_goal_dis)
            print("======== training env setup ========\n")

        self.episode_timesteps = 0

        self.cores.clear()
        self.obstacles.clear()
        self.robots.clear()

        num_cores = self.num_cores
        num_obs = self.num_obs
        robot_types = [True] * self.num_cooperative + [False] * self.num_non_cooperative
        assert len(robot_types) > 0, "Number of robots is 0!"

        ##### generate robots with randomly generated start and goal 
        num_robots = 0
        iteration = 500
        start_center = self.rd.uniform(low=5.0 * np.ones(2), high=np.array([self.width - 5.0, self.height - 5.0]))
        # goal_center = self.rd.uniform(low=5.0 * np.ones(2), high=np.array([self.width - 5.0, self.height - 5.0]))

        while True:
            start = self.rd.uniform(low=start_center - np.array([5.0, 5.0]), high=start_center + np.array([5.0, 5.0]))
            # goal = self.rd.uniform(low=goal_center - np.array([5.0, 5.0]), high=goal_center + np.array([5.0, 5.0]))
            goal = self.rd.uniform(low=start - np.array([4.0, 4.0]), high=start - np.array([2.0, 2.0]))
            iteration -= 1
            if self.check_start_and_goal(start, goal):
                rob = robot.Robot(robot_types[num_robots])
                rob.start = start
                rob.goal = goal
                self.reset_robot(rob)
                self.robots.append(rob)
                num_robots += 1
            if iteration == 0 or num_robots == len(robot_types):
                break

        ##### generate vortex with random position, spinning direction and strength
        if num_cores > 0:
            iteration = 500
            while True:
                center = self.rd.uniform(low=np.zeros(2), high=np.array([self.width, self.height]))
                direction = self.rd.binomial(1, 0.5)
                v_edge = self.rd.uniform(low=self.v_range[0], high=self.v_range[1])
                Gamma = 2 * np.pi * self.r * v_edge
                core = Core(center[0], center[1], direction, Gamma)
                iteration -= 1
                if self.check_core(core):
                    self.cores.append(core)
                    num_cores -= 1
                if iteration == 0 or num_cores == 0:
                    break

        centers = None
        for core in self.cores:
            if centers is None:
                centers = np.array([[core.x, core.y]])
            else:
                c = np.array([[core.x, core.y]])
                centers = np.vstack((centers, c))

        # KDTree storing vortex core center positions
        if centers is not None:
            self.core_centers = scipy.spatial.KDTree(centers)

        ##### generate static obstacles with random position and size
        if num_obs > 0:
            iteration = 500
            while True:
                center = self.rd.uniform(low=5.0 * np.ones(2), high=np.array([self.width - 5.0, self.height - 5.0]))
                r = self.rd.uniform(low=self.obs_r_range[0], high=self.obs_r_range[1])
                obs = Obstacle(center[0], center[1], r)
                iteration -= 1
                if self.check_obstacle(obs):
                    self.obstacles.append(obs)
                    num_obs -= 1
                if iteration == 0 or num_obs == 0:
                    break

        # TODO: check get_observations
        # return self.get_observations()

    def reset_robot(self, rob):
        # reset robot state
        rob.init_theta = self.rd.uniform(low=0.0, high=2 * np.pi)
        rob.init_speed = self.rd.uniform(low=0.0, high=rob.max_speed)
        current_v = self.get_velocity(rob.start[0], rob.start[1])
        rob.reset_state(current_velocity=current_v)

    def reset_goal(self, goal_list):
        for i, robot in enumerate(self.robots):
            robot.reset_goal(goal_list[i])

    def step(self, actions):
        # TODO: rewrite step function to update state of all robots, generate corresponding observations and rewards
        # execute action, update the environment, and return (obs, reward, done)

        rewards = [0] * len(self.robots)

        assert len(actions) == len(self.robots), "Number of actions not equal number of robots!"
        # Execute actions for all robots
        for i, action in enumerate(actions):
            rob = self.robots[i]
            # save action to history
            rob.action_history.append(action)

            dis_before = rob.dist_to_goal()

            # update robot state after executing the action    
            for _ in range(rob.N):
                current_velocity = self.get_velocity(rob.x, rob.y)
                rob.update_state(action, current_velocity)
                # save trajectory
                rob.trajectory.append([rob.x, rob.y])

            dis_after = rob.dist_to_goal()

            # constant penalty applied at every time step
            rewards[i] += self.timestep_penalty

            # reward agent for getting closer to the goal
            rewards[i] += dis_before - dis_after

        # Get observation for all robots
        observations = self.get_observations()

        dones = [True] * len(self.robots)
        infos = [{"state": "normal"}] * len(self.robots)

        # TODO: rewrite the reward and function: 
        # (1) if any collision happens, end the current episode
        # (2) if all robots reach goals, end the current episode
        # (3) when a robot reach the goal and the episode does not end, 
        #     remove it from the map  
        # if self.set_boundary and self.out_of_boundary():
        #     # No used in training 
        #     done = True
        #     info = {"state":"out of boundary"}
        # elif self.episode_timesteps >= 1000:
        #     done = True
        #     info = {"state":"too long episode"}
        # elif self.check_collision():
        #     reward += self.collision_penalty
        #     done = True
        #     info = {"state":"collision"}
        # elif self.check_reach_goal():
        #     reward += self.goal_reward
        #     done = True
        #     info = {"state":"reach goal"}
        # else:
        #     done = False
        #     info = {"state":"normal"}

        self.episode_timesteps += 1
        self.total_timesteps += 1

        return observations, rewards, dones, infos

    def out_of_boundary(self):
        # only used when boundary is set
        x_out = self.robot.x < 0.0 or self.robot.x > self.width
        y_out = self.robot.y < 0.0 or self.robot.y > self.height
        return x_out or y_out

    def get_observations(self):
        observations = []
        for robot in self.robots:
            observations.append(robot.perception_output(self.obstacles, self.robots))
        return observations

    def check_collision(self):
        if len(self.obstacles) == 0:
            return False

        for obs in self.obstacles:
            d = np.sqrt((self.robot.x - obs.x) ** 2 + (self.robot.y - obs.y) ** 2)
            if d <= obs.r + self.robot.r:
                return True
        return False

    def check_reach_goal(self):
        dis = np.array([self.robot.x, self.robot.y]) - self.goal
        if np.linalg.norm(dis) <= self.goal_dis:
            return True
        return False

    def check_start_and_goal(self, start, goal):

        # The start and goal point is far enough
        if np.linalg.norm(goal - start) < self.min_start_goal_dis:
            return False

        for robot in self.robots:

            dis_s = robot.start - start
            # Start point not too close to that of existing robots
            if np.linalg.norm(dis_s) <= self.clear_r:
                return False

            dis_g = robot.goal - goal
            # Goal point not too close to that of existing robots
            if np.linalg.norm(dis_g) <= self.clear_r:
                return False

        return True

    def check_core(self, core_j):

        # Within the range of the map
        if core_j.x - self.r < 0.0 or core_j.x + self.r > self.width:
            return False
        if core_j.y - self.r < 0.0 or core_j.y + self.r > self.width:
            return False

        for robot in self.robots:
            # Not too close to start and goal point of each robot
            core_pos = np.array([core_j.x, core_j.y])
            dis_s = core_pos - robot.start
            if np.linalg.norm(dis_s) < self.r + self.clear_r:
                return False
            dis_g = core_pos - robot.goal
            if np.linalg.norm(dis_g) < self.r + self.clear_r:
                return False

        for core_i in self.cores:
            dx = core_i.x - core_j.x
            dy = core_i.y - core_j.y
            dis = np.sqrt(dx * dx + dy * dy)

            if core_i.clockwise == core_j.clockwise:
                # i and j rotate in the same direction, their currents run towards each other at boundary
                # The currents speed at boundary need to be lower than threshold  
                boundary_i = core_i.Gamma / (2 * np.pi * self.v_rel_max)
                boundary_j = core_j.Gamma / (2 * np.pi * self.v_rel_max)
                if dis < boundary_i + boundary_j:
                    return False
            else:
                # i and j rotate in the opposite direction, their currents join at boundary
                # The relative current speed of the stronger vortex at boundary need to be lower than threshold 
                Gamma_l = max(core_i.Gamma, core_j.Gamma)
                Gamma_s = min(core_i.Gamma, core_j.Gamma)
                v_1 = Gamma_l / (2 * np.pi * (dis - 2 * self.r))
                v_2 = Gamma_s / (2 * np.pi * self.r)
                if v_1 > self.p * v_2:
                    return False

        return True

    def check_obstacle(self, obs):

        # Within the range of the map
        if obs.x - obs.r < 0.0 or obs.x + obs.r > self.width:
            return False
        if obs.y - obs.r < 0.0 or obs.y + obs.r > self.height:
            return False

        for robot in self.robots:
            # Not too close to start and goal point
            obs_pos = np.array([obs.x, obs.y])
            dis_s = obs_pos - robot.start
            if np.linalg.norm(dis_s) < obs.r + self.clear_r:
                return False
            dis_g = obs_pos - robot.goal
            if np.linalg.norm(dis_g) < obs.r + self.clear_r:
                return False

        # Not collide with vortex cores
        for core in self.cores:
            dx = core.x - obs.x
            dy = core.y - obs.y
            dis = np.sqrt(dx * dx + dy * dy)

            if dis <= self.r + obs.r:
                return False

        # Not collide with other obstacles
        for obstacle in self.obstacles:
            dx = obstacle.x - obs.x
            dy = obstacle.y - obs.y
            dis = np.sqrt(dx * dx + dy * dy)

            if dis <= obstacle.r + obs.r:
                return False

        return True

    def get_velocity(self, x: float, y: float):
        if len(self.cores) == 0:
            return np.zeros(2)

        # sort the vortices according to their distance to the query point
        d, idx = self.core_centers.query(np.array([x, y]), k=len(self.cores))
        if isinstance(idx, np.int64):
            idx = [idx]

        v_radial_set = []
        v_velocity = np.zeros((2, 1))
        for i in list(idx):
            core = self.cores[i]
            v_radial = np.matrix([[core.x - x], [core.y - y]])

            for v in v_radial_set:
                project = np.transpose(v) * v_radial
                if project[0, 0] > 0:
                    # if the core is in the outter area of a checked core (wrt the query position),
                    # assume that it has no influence the velocity of the query position
                    continue

            v_radial_set.append(v_radial)
            dis = np.linalg.norm(v_radial)
            v_radial /= dis
            if core.clockwise:
                rotation = np.matrix([[0., -1.], [1., 0]])
            else:
                rotation = np.matrix([[0., 1.], [-1., 0]])
            v_tangent = rotation * v_radial
            speed = self.compute_speed(core.Gamma, dis)
            v_velocity += v_tangent * speed

        return np.array([v_velocity[0, 0], v_velocity[1, 0]])

    def get_velocity_test(self, x: float, y: float):
        v = np.ones(2)
        return v / np.linalg.norm(v)

    def compute_speed(self, Gamma: float, d: float):
        if d <= self.r:
            return Gamma / (2 * np.pi * self.r * self.r) * d
        else:
            return Gamma / (2 * np.pi * d)

    def reset_with_eval_config(self, eval_config):
        self.episode_timesteps = 0

        # load env config
        self.sd = eval_config["env"]["seed"]
        self.width = eval_config["env"]["width"]
        self.height = eval_config["env"]["height"]
        self.r = eval_config["env"]["r"]
        self.v_rel_max = eval_config["env"]["v_rel_max"]
        self.p = eval_config["env"]["p"]
        self.v_range = copy.deepcopy(eval_config["env"]["v_range"])
        self.obs_r_range = copy.deepcopy(eval_config["env"]["obs_r_range"])
        self.clear_r = eval_config["env"]["clear_r"]
        self.start = np.array(eval_config["env"]["start"])
        self.goal = np.array(eval_config["env"]["goal"])
        self.goal_dis = eval_config["env"]["goal_dis"]
        self.timestep_penalty = eval_config["env"]["timestep_penalty"]
        self.collision_penalty = eval_config["env"]["collision_penalty"]
        self.goal_reward = eval_config["env"]["goal_reward"]
        self.discount = eval_config["env"]["discount"]

        # load vortex cores
        self.cores.clear()
        centers = None
        for i in range(len(eval_config["env"]["cores"]["positions"])):
            center = eval_config["env"]["cores"]["positions"][i]
            clockwise = eval_config["env"]["cores"]["clockwise"][i]
            Gamma = eval_config["env"]["cores"]["Gamma"][i]
            core = Core(center[0], center[1], clockwise, Gamma)
            self.cores.append(core)
            if centers is None:
                centers = np.array([[core.x, core.y]])
            else:
                c = np.array([[core.x, core.y]])
                centers = np.vstack((centers, c))

        if centers is not None:
            self.core_centers = scipy.spatial.KDTree(centers)

        # load obstacles
        self.obstacles.clear()
        for i in range(len(eval_config["env"]["obstacles"]["positions"])):
            center = eval_config["env"]["obstacles"]["positions"][i]
            r = eval_config["env"]["obstacles"]["r"][i]
            obs = Obstacle(center[0], center[1], r)
            self.obstacles.append(obs)

        # load robot config
        self.robot.dt = eval_config["robot"]["dt"]
        self.robot.N = eval_config["robot"]["N"]
        self.robot.length = eval_config["robot"]["length"]
        self.robot.width = eval_config["robot"]["width"]
        self.robot.r = eval_config["robot"]["r"]
        self.robot.max_speed = eval_config["robot"]["max_speed"]
        self.robot.a = np.array(eval_config["robot"]["a"])
        self.robot.w = np.array(eval_config["robot"]["w"])
        self.robot.compute_k()
        self.robot.compute_actions()

        # load perception config
        self.robot.perception.range = eval_config["robot"]["perception"]["range"]
        self.robot.perception.angle = eval_config["robot"]["perception"]["angle"]
        self.robot.perception.num_beams = eval_config["robot"]["perception"]["num_beams"]
        self.robot.perception.compute_phi()
        self.robot.perception.compute_beam_angles()

        # update env action and observation space
        # self.action_space = gym.spaces.Discrete(self.robot.compute_actions_dimension())
        obs_len = 2 + 2 + 2 * self.robot.perception.num_beams
        # self.observation_space = gym.spaces.Box(low = -np.inf * np.ones(obs_len), \
        #                                             high = np.inf * np.ones(obs_len), \
        #                                             dtype = np.float32)

        # reset robot state
        current_v = self.get_velocity(self.start[0], self.start[1])
        self.robot.reset_state(self.start[0], self.start[1], current_velocity=current_v)

        return self.get_observation()

    def episode_data(self):
        episode = {}

        # save environment config
        episode["env"] = {}
        episode["env"]["seed"] = self.sd
        episode["env"]["width"] = self.width
        episode["env"]["height"] = self.height
        episode["env"]["r"] = self.r
        episode["env"]["v_rel_max"] = self.v_rel_max
        episode["env"]["p"] = self.p
        episode["env"]["v_range"] = copy.deepcopy(self.v_range)
        episode["env"]["obs_r_range"] = copy.deepcopy(self.obs_r_range)
        episode["env"]["clear_r"] = self.clear_r
        episode["env"]["start"] = list(self.start)
        episode["env"]["goal"] = list(self.goal)
        episode["env"]["goal_dis"] = self.goal_dis
        episode["env"]["timestep_penalty"] = self.timestep_penalty
        # episode["env"]["energy_penalty"] = self.energy_penalty.tolist()
        # episode["env"]["angle_penalty"] = self.angle_penalty
        episode["env"]["collision_penalty"] = self.collision_penalty
        episode["env"]["goal_reward"] = self.goal_reward
        episode["env"]["discount"] = self.discount

        # save vortex cores information
        episode["env"]["cores"] = {}
        episode["env"]["cores"]["positions"] = []
        episode["env"]["cores"]["clockwise"] = []
        episode["env"]["cores"]["Gamma"] = []
        for core in self.cores:
            episode["env"]["cores"]["positions"].append([core.x, core.y])
            episode["env"]["cores"]["clockwise"].append(core.clockwise)
            episode["env"]["cores"]["Gamma"].append(core.Gamma)

        # save obstacles information
        episode["env"]["obstacles"] = {}
        episode["env"]["obstacles"]["positions"] = []
        episode["env"]["obstacles"]["r"] = []
        for obs in self.obstacles:
            episode["env"]["obstacles"]["positions"].append([obs.x, obs.y])
            episode["env"]["obstacles"]["r"].append(obs.r)

        # save robot config
        episode["robot"] = {}
        episode["robot"]["dt"] = self.robot.dt
        episode["robot"]["N"] = self.robot.N
        episode["robot"]["length"] = self.robot.length
        episode["robot"]["width"] = self.robot.width
        episode["robot"]["r"] = self.robot.r
        episode["robot"]["max_speed"] = self.robot.max_speed
        episode["robot"]["a"] = list(self.robot.a)
        episode["robot"]["w"] = list(self.robot.w)

        # save perception config
        episode["robot"]["perception"] = {}
        episode["robot"]["perception"]["range"] = self.robot.perception.range
        episode["robot"]["perception"]["angle"] = self.robot.perception.angle
        episode["robot"]["perception"]["num_beams"] = self.robot.perception.num_beams

        # save action history
        episode["robot"]["action_history"] = copy.deepcopy(self.robot.action_history)
        episode["robot"]["trajectory"] = copy.deepcopy(self.robot.trajectory)

        return episode

    def save_episode(self, filename):
        episode = self.episode_data()
        with open(filename, "w") as file:
            json.dump(episode, file)
