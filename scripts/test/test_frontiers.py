import sys

sys.path.insert(0, "../../")
import nav.exp_max
from tqdm import tqdm



if 1:
    # ev = env_visualizer.EnvVisualizer(seed=231,draw_envs=True)
    ev = nav.exp_max.ExpVisualizer(seed=123)
    ev.init_visualize()
    # ev.initialize_apf_agents()
    ev.initialize_rvo_agents()
    with open('log.txt', 'w') as file:
        file.write('log:\n')

    # Do a ten step navigate
    n = 200
    visualize = False
    goal_list = []
    ev.explore_one_step(n, "tmp/test_virtual_map")