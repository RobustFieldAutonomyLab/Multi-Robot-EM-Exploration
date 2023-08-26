import sys
import os
from tqdm import tqdm
import numpy as np
import multiprocessing
sys.path.insert(0, "../../")
import nav.exp_max


def run(i, seed, method, folder_path):
    params = nav.exp_max.ExpParams()
    ev = nav.exp_max.ExpVisualizer(seed=seed, method=method, params = params, folder_path=folder_path)  # , map_path = "map.txt")
    try:
        success = ev.explore_one_step(max_ite)
    except:
        print("Error in " + str(i) + method)
        ev.save()
        return False
    if not success:
        return False
    else:
        return True


if __name__ == "__main__":
    n = 30
    max_ite = 500
    cnt = 0

    folder_name = "statistic_file/test2"
    small_folder_names = ["BSP",  "CE", "NF", "EM_2", "EM_3"] #
    rd = np.random.RandomState(321)

    # Specify the path where you want to create the folder
    path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(path) or not os.path.isdir(path):
        os.mkdir(path)
    for name in small_folder_names:
        path_ = os.path.join(path, name)
        if not os.path.exists(path_) or not os.path.isdir(path_):
            os.mkdir(path_)

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    results = []

    for i in range(n):
        seed = rd.randint(0, 1000)
        print(i, seed)
        for j, name in enumerate(small_folder_names):
            result = pool.apply_async(run, (i, seed, name, folder_name,))
            results.append(result)
    pool.close()
    pool.join()

    final_results = [result.get() for result in results]
