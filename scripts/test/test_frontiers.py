import sys

sys.path.insert(0, "../../")
import nav.exp_max
from tqdm import tqdm



if 1:
    # ev = env_visualizer.EnvVisualizer(seed=231,draw_envs=True)
    ev = nav.exp_max.ExpVisualizer(seed=123)
    ev.init_visualize()
    ev.initialize_apf_agents()

    # Do a ten step navigate
    n = 50
    visualize = False
    goal_list = []
    speed = 5
    ev.explore_one_step(50, "tmp/test_virtual_map")