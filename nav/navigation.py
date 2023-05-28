import env_visualizer


class LandmarkSLAM:
    def __init__(self, seed:int=0):
        self.env_v = env_visualizer.EnvVisualizer(seed=seed)
