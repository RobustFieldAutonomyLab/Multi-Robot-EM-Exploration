import gtsam
import numpy as np
from gtsam import symbol
import networkx as nx


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


def local_to_world_values(values: gtsam.Values, origin, robot_id=None):
    result = gtsam.Values()
    origin_pose = gtsam.Pose2(origin[0], origin[1], origin[2])
    if robot_id is not None:
        key_min = chr(robot_id + ord('a'))
        key_max = chr(robot_id + ord('a') + 1)
    else:
        key_min = 0
        key_max = -1
    for key in values.keys():
        if key < gtsam.symbol('a', 0):
            landmark_position = values.atPoint2(key)
            if landmark_position is not None:
                result.insert(key, origin_pose.transformFrom(landmark_position))
        elif robot_id is not None:
            if gtsam.symbol(key_min, 0) <= key < gtsam.symbol(key_max, 0):
                robot_pose = values.atPose2(key)
                if robot_pose is not None:
                    result.insert(key, origin_pose.compose(robot_pose))
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


def from_cos_sin(c, s):
    theta = np.arccos(c)
    if s < 0:
        theta = 2 * np.pi - theta
    return theta


def generate_virtual_waypoints(state_this, state_next, speed):
    # return numpy.ndarray
    # [[x0,y0,theta0], ..., [x1,y1,theta1]]
    # state_this = self.robot_state_position[robot_id]
    if isinstance(state_this, gtsam.Pose2):
        state_0 = np.array([state_this.x(), state_this.y(), state_this.theta()])
    elif isinstance(state_this, np.ndarray) or isinstance(state_next, list):
        state_0 = np.array([state_this[0], state_this[1], 0])
    else:
        print(state_this, state_this.type)
        raise ValueError("Only accept gtsam.Pose2 and numpy.ndarray")
    if isinstance(state_next, gtsam.Pose2):
        state_1 = np.array([state_next.x(), state_next.y(), state_next.theta()])
    elif isinstance(state_next, np.ndarray) or isinstance(state_next, list):
        state_1 = np.array([state_next[0], state_next[1], 0])
    else:
        raise ValueError("Only accept gtsam.Pose2 and numpy.ndarray")

    step = int(np.linalg.norm(state_1[0:2] - state_0[0:2]) / speed)
    waypoints = np.linspace(state_0, state_1, step)

    return waypoints.tolist()


def heuristic(node, goal):
    return np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)


class A_Star:
    def __init__(self, rows, cols, cell_size):
        self.rows_scaled = int(rows / cell_size)
        self.cols_scaled = int(cols / cell_size)
        self.cell_size = cell_size

    def a_star(self, start, goal, obstacles):
        G = nx.grid_2d_graph(self.rows_scaled, self.cols_scaled)

        # Set edge weights (uniform in this case)
        for node in G.nodes():
            G.nodes[node]['weight'] = 1

        # Set some obstacles by removing nodes
        obstacles_scaled = []
        for obstacle in obstacles:
            for i in range(-1,2):
                for j in range(-1,2):
                    obstacles_scaled.append((int(obstacle[1] / self.cell_size)+i, int(obstacle[2] / self.cell_size)+j))
        G.remove_nodes_from(obstacles)

        # Find the shortest path using A* algorithm
        path = nx.astar_path(G, start, goal, heuristic=heuristic, weight='weight')
        path = np.array(path) * self.cell_size

        # Plot the graph and path
        return path[1, :]
