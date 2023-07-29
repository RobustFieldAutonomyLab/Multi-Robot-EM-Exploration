import gtsam
import numpy as np
import copy
from gtsam import symbol
import math


def get_symbol(robot_id, idx):
    return symbol(chr(robot_id + ord('a')), idx)


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


def point_to_local(x, y, origin):
    # compute transformation from world frame to robot frame
    R_wr, t_wr = pose_vector_to_matrix(origin[0], origin[1], origin[2])
    R_rw = np.transpose(R_wr)
    t_rw = -R_rw * t_wr

    t_this = R_rw * np.matrix([[x], [y]]) + t_rw
    return [t_this[0, 0], t_this[1, 0]]


def point_to_world(x, y, origin):
    # compute transformation from robot frame to world frame
    R_wr, t_wr = pose_vector_to_matrix(origin[0], origin[1], origin[2])

    t_this = R_wr * np.matrix([[x], [y]]) + t_wr
    return [t_this[0, 0], t_this[1, 0]]


def world_to_local_values(values: gtsam.Values, origin):
    result = gtsam.Values()
    origin_pose = gtsam.Pose2(origin[0], origin[1], origin[2])
    for key in values.keys():
        if key < ord('a'):
            landmark_position = values.atPoint2(key)
            if landmark_position is not None:
                result.insert(key, origin_pose.transformFrom(landmark_position))
        else:
            robot_pose = values.atPose2(key)
            if robot_pose is not None:
                result.insert(key, origin_pose.compose(robot_pose))
    return result


def pose_vector_to_matrix(x, y, theta):
    # compose matrix expression of pose vector
    R_rw = np.matrix([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    t_rw = np.matrix([[x], [y]])

    return R_rw, t_rw
