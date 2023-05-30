import numpy as np
import gtsam
import math


class LandmarkSLAM:
    def __init__(self, seed: int = 0):
        # Create a factor graph to hold the constraints
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.result = gtsam.Values()

        self.idx = []
        # Noise models for the prior
        self.prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.001, 0.001, 0.001])
        # Noise models for the odometry
        self.odom_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.01, 0.01, 0.008])
        # Noise models for the range bearing measurements
        self.range_bearing_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.004])

        self.parameters = gtsam.LevenbergMarquardtParams()

        self.frequency = 20
        self.frequency_count = 0
        self.odom = [0,0,0]

    def reset_graph(self, num_robot):
        self.graph.resize(0)
        self.initial.clear()
        self.idx = []
        self.idl = 0

        for i in range(0, num_robot):
            self.idx.append(0)
            if i == 0:
                self.add_prior(gtsam.symbol(chr(i + ord('a')), 0), gtsam.Pose2(0, 0, 0))
            self.add_initial_pose(gtsam.symbol(chr(i + ord('a')), 0), gtsam.Pose2(0, 0, 0))

    def add_prior(self, idx: gtsam.symbol, pose: gtsam.Pose2):
        self.graph.add(gtsam.PriorFactorPose2(idx, pose, self.prior_noise_model))

    def add_odom(self, idx0: gtsam.symbol, idx1: gtsam.symbol, pose: gtsam.Pose2):
        self.graph.add(gtsam.BetweenFactorPose2(idx0, idx1, pose, self.odom_noise_model))

    # bearing: radius, range: meter
    def add_bearing_range(self, idx: gtsam.symbol, idl: int, range: float, bearing: float):
        self.graph.add(gtsam.BearingRangeFactor2D(idx, idl,
                                                  gtsam.Rot2(bearing), range,
                                                  self.range_bearing_noise_model))

    def add_initial_pose(self, idx: gtsam.symbol, pose: gtsam.Pose2):
        self.initial.insert(idx, pose)

    def add_initial_landmark(self, idl, landmark: gtsam.Point2):
        if not self.initial.exists(idl):
            self.initial.insert(idl, landmark)

    def optimize(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, self.parameters)
        result = optimizer.optimize()
        return result

    def get_robot_value_initial(self, robot_id, idx):
        return self.initial.atPose2(gtsam.symbol(chr(robot_id + ord('a')), idx))

    def get_landmark_value_initial(self, idl):
        return self.initial.atPoint2(idl)

    def get_robot_value_optimized(self, robot_id, idx):
        if self.initial.exists():
            self.initial.atPose2(gtsam.symbol(chr(robot_id + ord('a')), idx))

    def get_symbol(self, robot_id, idx):
        return gtsam.symbol(chr(robot_id + ord('a')), idx)

    def get_robot_trajectory(self, robot_id, origin):
        pose2_list = np.zeros([self.idx[robot_id], 3])
        origin_pose = gtsam.Pose2(origin[0], origin[1], origin[2])
        for i in range(self.idx[robot_id]):
            pose = self.result.atPose2(self.get_symbol(robot_id, i))
            pose = origin_pose.compose(pose)
            pose2_list[i, 0] = pose.x()
            pose2_list[i, 1] = pose.y()
            pose2_list[i, 2] = pose.theta()
        # print("initial: ", self.initial)
        # print("graph: ", self.graph)
        # print("result: ", self.result)
        return pose2_list

    # TODO: add keyframe strategy
    def add_one_step(self, obs_list):
        # obs: obs_odom, obs_landmark, obs_robot
        # obs_odom: [dx, dy, dtheta]
        # obs_landmark: [landmark0:range, bearing, id], [landmark1], ...]
        # obs_robot: [obs0: range, bearing, id], [obs1], ...]
        if not obs_list.all():
            return

        if self.frequency_count % self.frequency != 0:
            self.frequency_count += 1
            return

        for i, obs in enumerate(obs_list):
            if not obs:
                continue
            obs_odom, obs_landmark = obs
            self.idx[i] += 1
            # add initial pose
            pre_pose = self.get_robot_value_initial(i, self.idx[i] - 1)
            initial_pose = pre_pose * gtsam.Pose2(obs_odom[0], obs_odom[1], obs_odom[2])
            self.add_initial_pose(gtsam.symbol(chr(i + ord('a')), self.idx[i]), initial_pose)
            # add odometry
            self.add_odom(gtsam.symbol(chr(i + ord('a')), self.idx[i] - 1),
                          gtsam.symbol(chr(i + ord('a')), self.idx[i]),
                          gtsam.Pose2(obs_odom[0], obs_odom[1], obs_odom[2]))
            if obs_landmark != []:
                for obs_l_this in obs_landmark:
                    r, b, idl = obs_l_this
                    idl = int(idl)
                    # add landmark initial
                    if not self.initial.exists(idl):
                        self.add_initial_landmark(idl, initial_pose.transformFrom(
                            gtsam.Point2(r * math.cos(b), r * math.sin(b))))
                    # add landmark observation
                    self.add_bearing_range(gtsam.symbol(chr(i + ord('a')), self.idx[i]), idl, r, b)

            # TODO: add observation of other robots

        self.result = self.optimize()
        # print("result: ", result)
