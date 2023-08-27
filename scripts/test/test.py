import sys

sys.path.insert(0, "../../")
from nav.virtualmap import pose_2_point_measurement
from marinenav_env.envs.utils.robot import RangeBearingMeasurement
import time
import numpy as np
import torch
from nav.EM import ExpectationMaximizationTrajectory

em = ExpectationMaximizationTrajectory(10, )