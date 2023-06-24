import sys
sys.path.insert(0,"../../")
import env_visualizer

# ev = env_visualizer.EnvVisualizer(seed=231,draw_envs=True)
ev = env_visualizer.EnvVisualizer(seed=231)
ev.init_visualize()

ev.visualize_navigation()
ev.fig.savefig("test_multi_robot.png",bbox_inches="tight")