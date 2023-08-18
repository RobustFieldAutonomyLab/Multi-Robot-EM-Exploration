import numpy as np
from ompl import base as ob
from ompl import geometric as og
from nav.utils import point_to_world, from_cos_sin, theta_0_to_2pi
import copy


class RRT_agent:
    def __init__(self, bound_x, bound_y, obstacle_size):
        self.space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, 0)
        bounds.setLow(1, 0)
        bounds.setHigh(0, bound_x)
        bounds.setHigh(1, bound_y)
        self.space.setBounds(bounds)
        self.obstacles = None

        self.safe_distance = 1.0 + obstacle_size

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))

        planner = og.RRTstar(self.ss.getSpaceInformation())
        self.ss.setPlanner(planner)

    def is_state_valid(self, state):
        if self.obstacles is None:
            return True
        state_array = np.array([state[0], state[1]])
        distances = np.linalg.norm(self.obstacles - state_array, axis=1)
        return np.all(distances > self.safe_distance)

    def act(self, observation, obstacles, origin):
        if observation is None:
            return 0
        self_state, static_states, dynamic_states = observation
        if len(obstacles) != 0:
            print(obstacles)
            self.obstacles = obstacles
        goal = self_state[:2]
        goal = np.array(point_to_world(goal[0], goal[1], origin))
        start = np.array(self_state[4:6])

        start_state = ob.State(self.space)
        start_state[0] = start[0]
        start_state[1] = start[1]

        goal_state = ob.State(self.space)
        goal_state[0] = goal[0]
        goal_state[1] = goal[1]

        self.ss.setStartAndGoalStates(start_state, goal_state)
        solved = self.ss.solve(0.001)

        if solved:
            # project traj to agent's action
            self.ss.simplifySolution()
            path = self.ss.getSolutionPath()

            state_1 = path.getState(1)

            return [state_1[0] - start[0], state_1[1] - start[1]]

        else:
            return 0
