import sys

import gtsam

sys.path.insert(0, "../../")
import nav.exp_max
from tqdm import tqdm



if 1:


    # ev = env_visualizer.EnvVisualizer(seed=231,draw_envs=True)
    ev = nav.exp_max.ExpVisualizer(seed=123)
    ev.init_visualize()
    ev.initialize_apf_agents()
    # ev.visualize_navigation()

    # Do a ten step navigate
    n = 30
    visualize = True
    for i in tqdm(range(0, n)):
        slam_result = ev.navigate_one_step("tmp/test_virtual_map", False)
        terminate_signal, goals = ev.generate_frontier()
        if nav.exp_max.DEBUG:
            print(goals)
        if terminate_signal:
            break
        ev.reset_goal(goals)
        if visualize:
            ev.plot_grid(ev.axis_grid)
            # TODO: add exploration finish signal here
            ev.visualize_SLAM()
            ev.visualize_frontier()
            ev.fig.savefig("tmp/test_virtual_map" + str(i) + ".png", bbox_inches="tight")

    ev.draw_present_position()
    ev.fig.savefig("test_virtual_map.png", bbox_inches="tight")
