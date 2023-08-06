import sys
import numpy as np
import gtsam

sys.path.insert(0, "../../")
import nav.exp_max
from tqdm import tqdm
from marinenav_env.envs.marinenav_env import Obstacle

if 1:
    ev = nav.exp_max.ExpVisualizer(seed=321)
    ev.init_visualize()
    ev.initialize_apf_agents()

    # Do a ten step navigate
    n = 50
    visualize = False
    goal_list = []
    speed = 5
    ev.navigate_one_step(50, "tmp/test_virtual_map")
