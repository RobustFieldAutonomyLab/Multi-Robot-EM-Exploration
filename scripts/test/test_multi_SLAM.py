import sys
import os
import random
sys.path.insert(0, "../../")
import nav.exp_max
from tqdm import tqdm

n = 30
max_ite = 200
cnt = 0
pbar = tqdm(total=n)

folder_name = "statistic_file"

# Specify the path where you want to create the folder
path = os.path.join(os.getcwd(), folder_name)
if not os.path.exists(path) or not os.path.isdir(path):
    os.mkdir(path)
path = os.path.join(path, "BSP")
if not os.path.exists(path) or not os.path.isdir(path):
    os.mkdir(path)

for i in range(n):
    ri = random.randint(1, 100)

    ev = nav.exp_max.ExpVisualizer(seed=123, method="BSP", num=i)  # , map_path = "map.txt")
    success = False
    try:
        success = ev.explore_one_step(max_ite)
    except:
        i = i - 1
    if success:
        cnt += 1
    pbar.update(1)  # Manually update the progress bar

pbar.close()
