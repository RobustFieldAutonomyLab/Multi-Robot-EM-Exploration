import numpy as np
import copy
from gtsam import symbol
import math


def get_symbol(robot_id, idx):
    return symbol(chr(robot_id + ord('a')), idx)
