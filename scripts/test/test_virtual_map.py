import sys
import numpy as np
import gtsam

sys.path.insert(0, "../../")
import env_visualizer
from tqdm import tqdm
from marinenav_env.envs.marinenav_env import Obstacle

if 1:
    # ev = env_visualizer.EnvVisualizer(seed=231,draw_envs=True)
    ev = env_visualizer.EnvVisualizer(seed=321)
    ev.env.num_obs = 3
    ev.env.reset()
    ev.env.obstacles.clear()
    ev.env.obstacles.append(Obstacle(75, 70, 1))
    ev.env.obstacles.append(Obstacle(75, 72, 1))
    ev.env.obstacles.append(Obstacle(73, 73, 1))
    ev.init_visualize()
    ev.initialize_apf_agents()
    # ev.visualize_navigation()

    # Do a ten step navigate
    n = 60
    visualize = True
    goal_list = []
    for i in tqdm(range(0, n)):
        goals = []
        speed = 5
        direction_list = [[0, 1], [-1, 1], [-1, 0], [-1, -1],
                          [0, -1], [1, -1], [1, 0], [1, 1]]
        for j, robot in enumerate(ev.env.robots):
            if i < 44:
                if i <10:
                    tmp = int(i / 4)
                elif i < 35:
                    tmp = int((i-10) / 3)
                else:
                    tmp = int((i-24) / 4)

                # direction_this = direction_list[(j*2 + i) % len(direction_list)]
                # goals.append([robot.x + speed * direction_this[0],
                #               robot.y + speed * direction_this[1]])
                direction_this = direction_list[tmp % len(direction_list)]
                if j == 0:
                    goals.append([robot.x + speed * direction_this[0],
                                  robot.y + speed * direction_this[1]])
                elif j == 2:
                    goals.append([robot.x + speed * direction_this[0],
                                  robot.y - speed * direction_this[1]])
                else:
                    goals.append([robot.x - speed * direction_this[0],
                                  robot.y + speed * direction_this[1]])
            else:
                if j != 1:
                    goals.append([ev.env.robots[0].x - speed,
                                  ev.env.robots[0].y])
                else:
                    goals.append([robot.x + speed,
                                  robot.y])
        goal_list.append(goals[0])

        ev.axis_grid.cla()
        ev.reset_goal(goals)
        slam_result = ev.navigate_one_step("tmp/test_virtual_map", False)
        ev.virtual_map.update(slam_result, ev.landmark_slam.get_marginal())
        if visualize:
            ev.plot_grid(ev.axis_grid)
            ev.visualize_SLAM()
            ev.fig.savefig("tmp/test_virtual_map" + str(i) + ".png", bbox_inches="tight")

    ev.draw_present_position()
    ev.fig.savefig("test_virtual_map.png", bbox_inches="tight")
