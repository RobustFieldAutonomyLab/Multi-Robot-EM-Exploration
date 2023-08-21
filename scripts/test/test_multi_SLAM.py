import sys

sys.path.insert(0, "../../")
import nav.exp_max
from tqdm import tqdm



if 1:
    # ev = env_visualizer.EnvVisualizer(seed=231,draw_envs=True)
    ev = nav.exp_max.ExpVisualizer(seed=123, method="BSP")# , map_path = "map_sparse_1.txt")
    # ev.init_visualize()
    with open('log.txt', 'w') as file:
        file.write('log:\n')
    with open('log_covariance.txt', 'w') as file:
        file.write('log:\n')

    # Do a ten step navigate
    n = 200
    visualize = False
    goal_list = []
    ev.explore_one_step(n, "tmp/test_virtual_map")