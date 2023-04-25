import os
import argparse
import itertools
from multiprocessing import Pool
import json
from datetime import datetime
import numpy as np
from marinenav_env.envs.marinenav_env import MarineNavEnv2
from policy.agent import Agent
from policy.trainer import Trainer

parser = argparse.ArgumentParser(description="Train IQN model")

parser.add_argument(
    "-C",
    "--config-file",
    dest="config_file",
    type=open,
    required=True,
    help="configuration file for training parameters",
)
parser.add_argument(
    "-P",
    "--num-procs",
    dest="num_procs",
    type=int,
    default=1,
    help="number of subprocess workers to use for trial parallelization",
)
parser.add_argument(
    "-D",
    "--device",
    dest="device",
    type=str,
    default="cpu",
    help="device to run all subprocesses, could only specify 1 device in each run"
)

def product(*args, repeat=1):
    # This function is a modified version of 
    # https://docs.python.org/3/library/itertools.html#itertools.product
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def trial_params(params):
    if isinstance(params,(str,int,float)):
        return [params]
    elif isinstance(params,list):
        return params
    elif isinstance(params, dict):
        keys, vals = zip(*params.items())
        mix_vals = []
        for val in vals:
            val = trial_params(val)
            mix_vals.append(val)
        return [dict(zip(keys, mix_val)) for mix_val in itertools.product(*mix_vals)]
    else:
        raise TypeError("Parameter type is incorrect.")

def params_dashboard(params):
    print("\n====== Training Setup ======\n")
    print("seed: ",params["seed"])
    print("total_timesteps: ",params["total_timesteps"])
    print("eval_freq: ",params["eval_freq"])
    print("\n")

def run_trial(device,params):
    exp_dir = os.path.join(params["save_dir"],
                           "training_"+params["training_time"],
                           "seed_"+str(params["seed"]))
    os.makedirs(exp_dir)

    param_file = os.path.join(exp_dir,"trial_config.json")
    with open(param_file, 'w+') as outfile:
        json.dump(params, outfile)

    # schedule of curriculum training
    training_schedule = dict(timesteps=[0,1000000,2000000],
                             num_cooperative=[0,0,0],
                             num_non_cooperative=[1,1,1],
                             num_cores=[4,6,8],
                             num_obstacles=[6,8,10],
                             min_start_goal_dis=[30.0,35.0,40.0],
                             )
    
    eval_schedule = dict(num_episodes=[10,10,10],
                         num_cooperative=[0,0,0],
                         num_non_cooperative=[1,1,1],
                         num_cores=[4,6,8],
                         num_obstacles=[6,8,10],
                         min_start_goal_dis=[30.0,35.0,40.0],
                         )
    
    train_env = MarineNavEnv2(seed=params["seed"],schedule=training_schedule)

    eval_env = MarineNavEnv2(seed=253)

    non_cooperative_agent = Agent(cooperative=False,device=device)
    # cooperative_agent = Agent(cooperative=False,device=device)

    trainer = Trainer(train_env=train_env,
                      eval_env=eval_env,
                      eval_schedule=eval_schedule,
                      cooperative_agent=None,
                      non_cooperative_agent=non_cooperative_agent)
    
    trainer.save_eval_config(exp_dir)

    trainer.learn(total_timesteps=params["total_timesteps"],
                  eval_freq=params["eval_freq"],
                  eval_log_path=exp_dir)

if __name__ == "__main__":
    args = parser.parse_args()
    params = json.load(args.config_file)
    params_dashboard(params)
    trial_param_list = trial_params(params)

    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")

    if args.num_procs == 1:
        for param in trial_param_list:
            param["training_time"]=timestamp
            run_trial(args.device,param)
    else:
        with Pool(processes=args.num_procs) as pool:
            for param in trial_param_list:
                param["training_time"]=timestamp
                pool.apply_async(run_trial,(args.device,param))
            
            pool.close()
            pool.join()    

