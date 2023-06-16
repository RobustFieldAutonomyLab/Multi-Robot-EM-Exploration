import sys

import gtsam

sys.path.insert(0,"../")
import env_visualizer
from tqdm import tqdm


if 1:
    # ev = env_visualizer.EnvVisualizer(seed=231,draw_envs=True)
    ev = env_visualizer.EnvVisualizer(seed = 123)
    ev.init_visualize()
    ev.initialize_apf_agents()
    # ev.visualize_navigation()

    #Do a ten step navigate
    n = 5
    for i in tqdm( range (0, n)):
        goals = []
        speed = 20
        direction_list = [[0,1], [-1,1], [-1,0], [-1,-1],
                          [0,-1], [1, -1], [1,0], [1,1]]

        for j, robot in enumerate(ev.env.robots):
            direction_this = direction_list[(j*2 + i) % len(direction_list)]
            goals.append([robot.x + speed * direction_this[0],
                          robot.y + speed * direction_this[1]])
        ev.axis_grid.cla()
        ev.reset_goal(goals)
        ev.navigate_one_step("tmp/test_virtual_map", False)
        if True:
            ev.plot_grid(ev.axis_grid)
            ev.visualize_SLAM()
            ev.fig.savefig("tmp/test_virtual_map" + str(i) + ".png",bbox_inches="tight")


    ev.draw_present_position()
    ev.fig.savefig("test_virtual_map.png",bbox_inches="tight")
