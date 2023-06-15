import numpy as np
import gtsam
import copy


class LandmarkSLAM:
    def __init__(self):
        # Create a factor graph to hold the constraints
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.result = gtsam.Values()
        self.marginals = []
        params = gtsam.ISAM2Params()
        params.setFactorization("QR")
        self.isam = gtsam.ISAM2(params)

        self.idx = []
        # Noise models for the prior
        self.prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.001, 0.001, 0.001])
        # Noise models for the odometry
        self.odom_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.01, 0.01, 0.04])
        # Noise models for the range bearing measurements
        # self.range_bearing_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.05])
        self.range_bearing_noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Cauchy.Create(0.1), gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.004]))
        # Noise models for the robot observations
        # self.robot_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.004])
        self.robot_noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Cauchy.Create(0.1), gtsam.noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.004]))

        self.parameters = gtsam.LevenbergMarquardtParams()


        # for deugging
        # self.landmark_list = [[] for _ in range(28)]

    def reset_graph(self, num_robot):
        self.graph.resize(0)
        self.initial.clear()
        self.idx = []
        self.idl = 0

        for i in range(0, num_robot):
            self.idx.append(-1)

    def add_prior(self, idx: gtsam.symbol, pose: gtsam.Pose2):
        self.graph.add(gtsam.PriorFactorPose2(idx, pose, self.prior_noise_model))

    def add_odom(self, idx0: gtsam.symbol, idx1: gtsam.symbol, pose: gtsam.Pose2):
        self.graph.add(gtsam.BetweenFactorPose2(idx0, idx1, pose, self.odom_noise_model))

    # bearing: radius, range: meter
    def add_bearing_range(self, idx: gtsam.symbol, idl: int, range: float, bearing: float):
        self.graph.add(gtsam.BearingRangeFactor2D(idx, idl,
                                                  gtsam.Rot2(bearing), range,
                                                  self.range_bearing_noise_model))

    def add_robot_observation(self, idx0: gtsam.symbol, idx1: gtsam.symbol, pose: gtsam.Pose2):
        self.graph.add(gtsam.BetweenFactorPose2(idx0, idx1, pose, self.robot_noise_model))
        # if self.initial.exists(idx0):
        #     pose_1 = self.initial.atPose2(idx0).compose(pose)
        # if self.initial.exists(idx1):
        #     dpose = self.initial.atPose2(idx1).between(pose_1)
        #     print("Residual calculated pose: ", idx1, " from ", idx0, ":",dpose)

    def add_initial_pose(self, idx: gtsam.symbol, pose: gtsam.Pose2):
        self.initial.insert(idx, pose)

    def add_initial_landmark(self, idl, landmark: gtsam.Point2):
        if not self.initial.exists(idl):
            self.initial.insert(idl, landmark)

    def optimize(self):
        # optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, self.parameters)
        # result = optimizer.optimize()

        self.isam.update(self.graph, self.initial)
        self.graph.resize(0)
        self.initial.clear()
        self.result = self.isam.calculateEstimate()
        self.marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), self.result)

    def get_robot_value_initial(self, robot_id, idx):
            if self.initial.exists(gtsam.symbol(chr(robot_id + ord('a')), idx)):
                return self.initial.atPose2(gtsam.symbol(chr(robot_id + ord('a')), idx))

    def get_robot_value_result(self, robot_id, idx):
        if self.result.exists(gtsam.symbol(chr(robot_id + ord('a')), idx)):
            return self.result.atPose2(gtsam.symbol(chr(robot_id + ord('a')), idx))

    def get_landmark_value_initial(self, idl):
        return self.initial.atPoint2(idl)

    def get_symbol(self, robot_id, idx):
        return gtsam.symbol(chr(robot_id + ord('a')), idx)

    def get_robot_trajectory(self, robot_id, origin):
        pose2_list = np.zeros([self.idx[robot_id] + 1, 3])
        origin_pose = gtsam.Pose2(origin[0], origin[1], origin[2])
        for i in range(self.idx[robot_id] + 1):
            pose = self.result.atPose2(self.get_symbol(robot_id, i))
            pose = origin_pose.compose(pose)
            pose2_list[i, 0] = pose.x()
            pose2_list[i, 1] = pose.y()
            pose2_list[i, 2] = pose.theta()
        return pose2_list

    def get_landmark_list(self, origin):
        landmark_list = []
        # print(origin)
        origin_pose = gtsam.Pose2(origin[0], origin[1], origin[2])
        for key in self.result.keys():
            if key < ord('a'):
                landmark_position = self.result.atPoint2(key)
                if landmark_position is not None:
                    landmark_list.append(origin_pose.transformFrom(landmark_position))
        return landmark_list

    def get_result(self, origin):
        result = gtsam.Values()
        origin_pose = gtsam.Pose2(origin[0], origin[1], origin[2])
        for key in self.result.keys():
            if key < ord('a'):
                landmark_position = self.result.atPoint2(key)
                if landmark_position is not None:
                    result.insert(key, origin_pose.transformFrom(landmark_position))
            else:
                robot_pose = self.result.atPose2(key)
                if robot_pose is not None:
                    result.insert(key, origin_pose.compose(robot_pose))
        return result

    def init_SLAM(self, robot_id, obs_robot):
        if robot_id == 0:
            self.add_prior(gtsam.symbol(chr(robot_id + ord('a')), 0), gtsam.Pose2(0, 0, 0))
            self.add_initial_pose(gtsam.symbol(chr(robot_id + ord('a')), 0), gtsam.Pose2(0, 0, 0))
        elif obs_robot != []:
            for obs_r_this in obs_robot:
                idr = int(obs_r_this[3])
                if idr == 0:
                    self.add_initial_pose(gtsam.symbol(chr(robot_id + ord('a')), 0),
                                          gtsam.Pose2(0, 0, 0).compose(
                                              gtsam.Pose2(obs_r_this[0], obs_r_this[1], obs_r_this[2]).inverse()
                                          ))
        else:
            raise ValueError("Fail to initialize SLAM graph")

    # TODO: add keyframe strategy
    def add_one_step(self, obs_list):
        # obs: obs_odom, obs_landmark, obs_robot
        # obs_odom: [dx, dy, dtheta]
        # obs_landmark: [landmark0:range, bearing, id], [landmark1], ...]
        # obs_robot: [obs0: dx, dy, dtheta, id], [obs1], ...]
        if np.all(np.equal(obs_list, None)):
            return

        for i, obs in enumerate(obs_list):
            if not obs:
                continue
            self.idx[i] += 1
            obs_odom, obs_landmark, obs_robot = obs
            if self.idx[i] == 0:
                # add initial pose
                self.init_SLAM(i, obs_robot)
            else:
                # add odometry
                pre_pose = self.get_robot_value_result(i, self.idx[i] - 1)
                if not pre_pose:
                    pre_pose = self.get_robot_value_initial(i, self.idx[i] - 1)
                initial_pose = pre_pose * gtsam.Pose2(obs_odom[0], obs_odom[1], obs_odom[2])
                self.add_initial_pose(gtsam.symbol(chr(i + ord('a')), self.idx[i]), initial_pose)

                self.add_odom(gtsam.symbol(chr(i + ord('a')), self.idx[i] - 1),
                              gtsam.symbol(chr(i + ord('a')), self.idx[i]),
                              gtsam.Pose2(obs_odom[0], obs_odom[1], obs_odom[2]))

            if obs_landmark != []:
                for obs_l_this in obs_landmark:
                    r, b, idl = obs_l_this
                    idl = int(idl)
                    # add landmark initial
                    # self.landmark_list[idl].append(initial_pose.transformFrom(
                    #     gtsam.Point2(r * np.cos(b), r * np.sin(b))))
                    initial_pose = self.get_robot_value_initial(i, self.idx[i])
                    if not (self.initial.exists(idl) or self.result.exists(idl)):
                        self.add_initial_landmark(idl, initial_pose.transformFrom(
                            gtsam.Point2(r * np.cos(b), r * np.sin(b))))
                    # add landmark observation
                    self.add_bearing_range(gtsam.symbol(chr(i + ord('a')), self.idx[i]), idl, r, b)

            if obs_robot != []:
                for obs_r_this in obs_robot:
                    idr = int(obs_r_this[3])
                    # add landmark observation
                    if idr > i and obs_list[idr] is not None:
                        # This means the latest id of robot neighbor is not yet updated to this timestamp
                        self.add_robot_observation(gtsam.symbol(chr(i + ord('a')), self.idx[i]),
                                                   gtsam.symbol(chr(idr + ord('a')), self.idx[idr] + 1),
                                                   gtsam.Pose2(obs_r_this[0], obs_r_this[1], obs_r_this[2]))
                    else:
                        self.add_robot_observation(gtsam.symbol(chr(i + ord('a')), self.idx[i]),
                                                   gtsam.symbol(chr(idr + ord('a')), self.idx[idr]),
                                                   gtsam.Pose2(obs_r_this[0], obs_r_this[1], obs_r_this[2]))

        self.optimize()


    def get_marginal(self):
        return copy.deepcopy(self.marginals)
