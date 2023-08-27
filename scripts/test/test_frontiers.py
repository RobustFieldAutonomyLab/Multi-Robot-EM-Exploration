import sys
import os
import numpy as np
import multiprocessing

sys.path.insert(0, "../../")
import nav.exp_max


def run(i, seed, method, folder_path, params=nav.exp_max.ExpParams()):
    # params = nav.exp_max.ExpParams(env_width=100,
    #                                env_height=100,
    #                                num_obs=30,
    #                                num_cooperative=2,
    #                                boundary_dist=6,
    #                                cell_size=2,
    #                                start_center=np.array([20, 50]),
    #                                sensor_range=7.5,
    #                                )

    # params = nav.exp_max.ExpParams(num_obs=20, map_path = "map.txt", env_width=200, env_height = 200)
    ev = nav.exp_max.ExpVisualizer(num=i, seed=seed, method=method, params=params,
                                   folder_path=folder_path)
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
    n = 50
    max_ite = 500
    cnt = 0

    small_folder_names = ["BSP", "CE", "NF", "EM_2", "EM_3"]  #
    rd = np.random.RandomState(321)

    # Specify the path where you want to create the folder

    # run(0, 233, "EM_2", folder_name + "/EM_2")
    if 1:
        folder_name = "statistic_file/test3"
        path = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(path) or not os.path.isdir(path):
            os.mkdir(path)
        for name in small_folder_names:
            path_ = os.path.join(path, name)
            if not os.path.exists(path_) or not os.path.isdir(path_):
                os.mkdir(path_)
        params = nav.exp_max.ExpParams(env_width=100,
                                       env_height=100,
                                       num_obs=30,
                                       num_cooperative=3,
                                       boundary_dist=4,
                                       cell_size=2,
                                       start_center=np.array([20, 50]),
                                       sensor_range=7.5)
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)
        results = []

        for i in range(n):
            seed = rd.randint(0, 1000)
            print(i, seed)
            for j, name in enumerate(small_folder_names):
                result = pool.apply_async(run, (i, seed, name, folder_name, params,))
                results.append(result)
                # run(i, seed, name, folder_name, params)
        pool.close()
        pool.join()

        final_results = [result.get() for result in results]
    if 0:
        folder_name = "statistic_file/test3"
        path = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(path) or not os.path.isdir(path):
            os.mkdir(path)
        for name in small_folder_names:
            path_ = os.path.join(path, name)
            if not os.path.exists(path_) or not os.path.isdir(path_):
                os.mkdir(path_)
        params = nav.exp_max.ExpParams(env_width=100,
                                       env_height=100,
                                       num_obs=30,
                                       num_cooperative=3,
                                       boundary_dist=6,
                                       cell_size=2,
                                       start_center=np.array([20, 50]),
                                       sensor_range=7.5)
        # num_processes = multiprocessing.cpu_count()
        # pool = multiprocessing.Pool(processes=num_processes)
        # results = []

        for i in range(n):
            seed = rd.randint(0, 1000)
            print(i, seed)
            for j, name in enumerate(small_folder_names):
                run(i, seed, name, folder_name, params)
                # result = pool.apply_async(run, (i, seed, name, folder_name, params,))
                # results.append(result)
        # pool.close()
        # pool.join()

        # final_results = [result.get() for result in results]
    if 0:
        for i in range(n):
            seed = rd.randint(0, 1000)
            print(i, seed)
            for j, name in enumerate(small_folder_names):
                result = run(i, seed, name, folder_name)