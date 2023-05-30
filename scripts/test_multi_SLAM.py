import sys
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
    n = 10
    for i in tqdm( range (0, n)):
        goals = []
        speed = 20
        direction_list = [[0,1], [-1,1], [-1,0], [-1,-1],
                       [0,-1], [1, -1], [1,0], [1,1]]

        for j, robot in enumerate(ev.env.robots):
            direction_this = direction_list[(j*2 + i) % len(direction_list)]
            goals.append([robot.x + speed * direction_this[0],
                          robot.y + speed * direction_this[1]])
        ev.reset_goal(goals)
        ev.navigate_one_step()
    ev.draw_present_position()
    ev.visualize_SLAM()

    ev.fig.savefig("test_multi_robot.png",bbox_inches="tight")