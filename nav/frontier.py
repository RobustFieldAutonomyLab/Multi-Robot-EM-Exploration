import numpy as np
import gtsam
from virtualmap import VirtualMap


class Frontier:
    def __init__(self, x=0, y=0, theta=0):
        self.pose = gtsam.Pose2(x, y, theta)
        self.parent = None


class FrontierGenerator:
    def __init__(self, parameters):
        self.max_x = parameters["maxX"]
        self.max_y = parameters["maxY"]
        self.min_x = parameters["minX"]
        self.min_Y = parameters['minY']

        self.max_node_ratio = 0.5
        self.max_distance = 1000
        self.min_distance = 0

    def generate_one_frontier(self):


    def generate(self):
        pass