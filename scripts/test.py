import sys
sys.path.insert(0,"../")
import env_visualizer
# ev = env_visualizer.EnvVisualizer(seed=231,draw_envs=True)
ev = env_visualizer.EnvVisualizer(seed=231)
ev.init_visualize()
test_command = [-2]*len(ev.env.robots)
test_commands = [test_command]
ev.visualize_control(test_commands)
ev.fig.savefig("test_multi_robot.png",bbox_inches="tight")
