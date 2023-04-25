import sys
sys.path.insert(0,"../")
import env_visualizer

dir = "../training_data/training_2023-04-24-20-17-46/seed_0/"
eval_configs = "eval_configs.json"
evaluations = "evaluations.npz"

# ev = env_visualizer.EnvVisualizer(seed=231,draw_envs=True)
ev = env_visualizer.EnvVisualizer(seed=231)

ev.load_eval_config_and_episode(dir+eval_configs,dir+evaluations)
ev.play_eval_episode(99,29,1)