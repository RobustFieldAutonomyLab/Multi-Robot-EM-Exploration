import numpy as np
import gtsam
import math

class LandmarkSLAM:
    def __init__(self, seed: int = 0):
        # Create a factor graph to hold the constraints
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        self.idx = []
        # Noise models for the prior
        self.prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
        # Noise models for the odometry
        self.odom_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.1])
        # Noise models for the range bearing measurements
        self.range_bearing_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.2])

        self.parameters = gtsam.LevenbergMarquardtParams()

    def reset_graph(self, start_list):
        self.graph.clear()
        self.initial.clear()
        self.idx = []
        self.idl = 0
        for i, start in enumerate(start_list):
            self.idx.append([0])
            X = gtsam.symbol(ord(chr(i + 97)), 0)
            self.graph.add(X, gtsam.Pose2(start[0], start[1], start[2]))

    def add_prior(self, idx: gtsam.symbol, pose: gtsam.Pose2):
        self.graph.add(gtsam.PriorFactorPose2(idx, pose, self.prior_noise_model))

    def add_odom(self, idx0: gtsam.symbol, idx1: gtsam.symbol, pose: gtsam.Pose2):
        self.graph.add(gtsam.BetweenFactorPose2(idx0, idx1, pose, self.odom_noise_model))

    # bearing: radius, range: meter
    def add_bearing_range(self, idx: gtsam.symbol, idl: int, range: float, bearing: float):
        self.graph.add(gtsam.BearingRangeFactor2D(idx, idl,
                                                  gtsam.Rot2.fromRadians(bearing), range,
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

    def get_value(self, robot_id, idx):
        return self.initial.at(gtsam.symbol(ord(chr(robot_id + 97)), idx))

    #TODO: add keyframe strategy
    def add_one_step(self, obs_list):
        # obs: obs_odom, obs_landmark, obs_robot
        # obs_odom: [dx, dy, dtheta]
        # obs_landmark: [landmark0:range, bearing, id], [landmark1], ...]
        # obs_robot: [obs0: range, bearing, id], [obs1], ...]
        for i, obs in enumerate(obs_list):
            obs_odom, obs_landmark = obs
            self.idx[i] += 1
            # add initial pose
            pre_pose = self.get_value(i, self.idx[i]-1)
            initial_pose = pre_pose.compose(
                                gtsam.Pose2(obs_odom[0], obs_odom[1], obs_odom[2]))
            self.add_initial_pose(gtsam.symbol(ord(chr(i + 97)), self.idx[i]), initial_pose)
            # add odometry
            self.add_odom(gtsam.symbol(ord(chr(i + 97)), self.idx[i]-1),
                          gtsam.symbol(ord(chr(i + 97)), self.idx[i]),
                          gtsam.Pose2(obs_odom[0], obs_odom[1], obs_odom[2]))
            if obs_landmark:
                for obs_l_this in obs_landmark:
                    r, b, idl = obs_l_this
                    # add landmark initial
                    if not self.initial.exists(idl):
                        self.add_initial_landmark(idl, initial_pose.transformFrom(gtsam.Point2(r * math.cos(b), r * math.sin(b))))
                    # add landmark observation
                    self.add_bearing_range(gtsam.symbol(ord(i + 97), r, b, idl))

            # TODO: add observation of other robots

        result = self.optimize()
        print("result: ", result)
