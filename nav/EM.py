# Everything in SLAM frame in ExpectationMaximizationTrajectory
import gtsam
import numpy as np
from nav.utils import get_symbol, generate_virtual_waypoints, local_to_world_values
from marinenav_env.envs.utils.robot import Odometry, RangeBearingMeasurement, RobotNeighborMeasurement

DEBUG_EM = False


class ExpectationMaximizationTrajectory:
    def __init__(self, radius, robot_state_idx_position_goal, landmarks, isam: tuple):
        # for odometry measurement
        self.odom_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.03, 0.03, 0.004])

        # for landmark measurement
        self.range_bearing_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.004, 0.001])

        # for inter-robot measurement
        self.robot_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.01, 0.01, 0.004])

        self.slam_speed = 2

        # radius for generate virtual landmark and virtual inter-robot observation
        self.radius = radius

        self.landmarks_position = [[item[1], item[2]] for item in landmarks]
        self.landmarks_id = [item[0] for item in landmarks]

        params = gtsam.ISAM2Params()
        params.setFactorization("QR")
        self.isam = (isam[0], isam[1])
        self.params = params

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

            if len(self.landmarks_position) != 0:
                # calculate Euclidean distance between waypoint and landmarks
                distances = np.linalg.norm(self.landmarks_position - np.array(waypoint[0:2]), axis=1)
                landmark_indices = np.argwhere(distances < self.radius)
            else:
                landmark_indices = []
            if len(landmark_indices) != 0:
                # add landmark observation factor
                for landmark_index in landmark_indices:
                    landmark_this = self.landmarks_position[landmark_index[0]]
                    # calculate range and bearing
                    rb = range_bearing_factory.add_noise_point_based(waypoint, landmark_this)
                    graph.add(self.landmark_measurement_gtsam_format(bearing=rb[1], range=rb[0],
                                                                     robot_id=robot_id,
                                                                     id0=id_initial + i,
                                                                     idl=self.landmarks_id[landmark_index[0]]))
        # print(graph)
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
        isam_copy = gtsam.ISAM2(self.params)
        isam_copy.update(self.isam[0], self.isam[1])
        isam_copy.update(graph, initial_estimate)
        try:
            result = isam_copy.calculateEstimate()
            marginals = gtsam.Marginals(isam_copy.getFactorsUnsafe(), result)
        except:
            print(graph)
            print(initial_estimate)
        # with open('log.txt', 'a') as file:
        #     # print(graph, file=file)
        #     # print(initial_estimate, file=file)
        #     a = 0
        #     b = 0
        #     for key in result.keys():
        #         try:
        #             a+=marginals.marginalCovariance(key).trace()
        #             b+=np.linalg.det(marginals.marginalCovariance(key))
        #         except:
        #             print(result)
        #     print(a,b, file=file)
                # print(key, marginals.marginalCovariance(key).trace(),
                #       np.linalg.det(marginals.marginalCovariance(key)), file=file)

        return result, marginals

    def do(self, frontier_position, robot_id, origin, axis):
        # return the optimized trajectories of the robots and the marginals for calculation of information  matrixc
        graph, initial_estimate = self.generate_virtual_observation_graph(frontier_position=frontier_position,
                                                                          robot_id=robot_id)
        result, marginals = self.optimize_virtual_observation_graph(graph, initial_estimate)
        result = local_to_world_values(result, origin, robot_id)

        # draw the optimized result
        if DEBUG_EM and axis is not None:
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
