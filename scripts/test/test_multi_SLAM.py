import sys
import os
import random

sys.path.insert(0, "../../")
import nav.exp_max
from tqdm import tqdm
import numpy as np

n = 1
max_ite = 300
cnt = 0
pbar = tqdm(total=n)

folder_name = "statistic_file"

# Specify the path where you want to create the folder
path = os.path.join(os.getcwd(), folder_name)
if not os.path.exists(path) or not os.path.isdir(path):
    os.mkdir(path)
path_ = os.path.join(path, "NF")
if not os.path.exists(path_) or not os.path.isdir(path_):
    os.mkdir(path_)

with open("log.txt", 'w') as f:
    f.write("")
params = nav.exp_max.ExpParams(env_width=80,
                               env_height=50,
                               num_obs=10,
                               num_cooperative=2,
                               boundary_dist=6,
                               cell_size=2,
                               start_center=np.array([20, 25]),
                               sensor_range=5)
for i in range(n):
    ev = nav.exp_max.ExpVisualizer(seed=799, method="EM_2", num=i, folder_path=path,
                                   params=params)  # , map_path = "map.txt")

    ev.init_visualize(False)
    success = False
    # try:
    success = ev.explore_one_step(max_ite, "tmp/test_virtual_map")

    # except:
    #     i = i - 1
    # if success:
    #     cnt += 1
    pbar.update(1)  # Manually update the progress bar

pbar.close()
