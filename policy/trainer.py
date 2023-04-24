import json
import numpy as np
import os
import copy
import time

class Trainer():
    def __init__(self,
                 train_env,
                 eval_env,
                 eval_schedule,
                 non_cooperative_agent=None,
                 cooperative_agent=None,
                 UPDATE_EVERY=1,
                 learning_starts=1000,
                 target_update_interval=10000,
                 exploration_fraction=0.1,
                 initial_eps=1.0,
                 final_eps=0.05
                 ):
        
        self.train_env = train_env
        self.eval_env = eval_env
        self.cooperative_agent = cooperative_agent
        self.noncooperative_agent = non_cooperative_agent
        self.eval_config = []
        self.create_eval_configs(eval_schedule)

        self.UPDATE_EVERY = UPDATE_EVERY
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.initial_eps = initial_eps
        self.final_eps = final_eps

        # Current time step
        self.current_timestep = 0

        # Learning time step (start counting after learning_starts time step)
        self.learning_timestep = 0

        # Evaluation data
        self.eval_timesteps = []
        self.eval_actions = []
        self.eval_rewards = []
        self.eval_successes = []
        self.eval_times = []
        self.eval_energies = []
        self.eval_relations = []
        self.eval_obs = []
        self.eval_objs = []

    def create_eval_configs(self,eval_schedule):
        self.eval_config.clear()

        count = 0
        for i,num_episode in enumerate(eval_schedule["num_episodes"]):
            for _ in range(num_episode): 
                self.eval_env.num_cooperative = eval_schedule["num_cooperative"][i]
                self.eval_env.num_non_cooperative = eval_schedule["num_non_cooperative"][i]
                self.eval_env.num_cores = eval_schedule["num_cores"][i]
                self.eval_env.num_obs = eval_schedule["num_obstacles"][i]
                self.eval_env.min_start_goal_dis = eval_schedule["min_start_goal_dis"][i]

                self.eval_env.reset()

                # save eval config
                self.eval_config.append(self.eval_env.episode_data())
                count += 1

    def save_eval_config(self,directory):
        file = os.path.join(directory,"eval_configs.json")
        with open(file, "w+") as f:
            json.dump(self.eval_config, f)

    def learn(self,
              total_timesteps,
              eval_freq,
              eval_log_path,
              verbose=True):
        
        states,_,_ = self.train_env.reset()

        # # Sample CVaR value from (0.0,1.0)
        # cvar = 1 - np.random.uniform(0.0, 1.0)

        # current episode 
        ep_rewards = np.zeros(len(self.train_env.robots))
        ep_length = 0
        ep_num = 0
        
        while self.current_timestep <= total_timesteps:
            start_1 = time.time()

            eps = self.linear_eps(total_timesteps)
            
            # gather actions for robots from agents 
            start_2 = time.time()
            actions = []
            for i,rob in enumerate(self.train_env.robots):
                if rob.reach_goal:
                    actions.append(None)
                    continue

                if rob.cooperative:
                    action,_,_,_ = self.cooperative_agent.act(states[i],eps)
                else:
                    action,_,_,_ = self.noncooperative_agent.act(states[i],eps)
                actions.append(action)
            end_2 = time.time()
            elapsed_time_2 = end_2 - start_2
            if self.current_timestep % 100 == 0:
                print("Elapsed time 2: {:.6f} seconds".format(elapsed_time_2))

            # execute actions in the training environment
            next_states, rewards, dones, infos, end_episode = self.train_env.step(actions)

            # save experience in replay memory
            for i,rob in enumerate(self.train_env.robots):
                if rob.reach_goal:
                    continue

                if rob.cooperative:
                    ep_rewards[i] += self.cooperative_agent.GAMMA ** ep_length * rewards[i]
                    if self.cooperative_agent.training:
                        self.cooperative_agent.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))
                else:
                    ep_rewards[i] += self.noncooperative_agent.GAMMA ** ep_length * rewards[i]
                    if self.noncooperative_agent.training:
                        self.noncooperative_agent.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))
                
            states = next_states
            ep_length += 1
            
            # Learn, update and evaluate models after learning_starts time step 
            if self.current_timestep >= self.learning_starts:
                start_3 = time.time()

                for agent in [self.cooperative_agent,self.noncooperative_agent]:
                    if agent is None:
                        continue

                    if not agent.training:
                        continue

                    # Learn every UPDATE_EVERY time steps.
                    if self.learning_timestep % self.UPDATE_EVERY == 0:
                        # If enough samples are available in memory, get random subset and learn
                        if agent.memory.size() > agent.BATCH_SIZE:
                            agent.train()

                    # Update the target model every target_update_interval time steps
                    if self.learning_timestep % self.target_update_interval == 0:
                        agent.soft_update()

                end_3 = time.time()
                elapsed_time_3 = end_3 - start_3
                if self.current_timestep % 100 == 0:
                    print("Elapsed time 3: {:.6f} seconds".format(elapsed_time_3))

                # Evaluate learning agents every eval_freq time steps
                if self.learning_timestep % eval_freq == 0: 
                    # self.evaluation()

                    for agent in [self.cooperative_agent,self.noncooperative_agent]:
                        if agent is None:
                            continue
                        if not agent.training:
                            continue
                        # save the latest models
                        agent.save_latest_model(eval_log_path)

                self.learning_timestep += 1

            if end_episode:
                ep_num += 1
                
                if verbose:
                    # print abstract info of learning process
                    print("======== Episode Info ========")
                    print("current ep_length: ",ep_length)
                    print("current ep_num: ",ep_num)
                    print("current exploration rate: ",eps)
                    print("current timesteps: ",self.current_timestep)
                    print("total timesteps: ",total_timesteps)
                    print("======== Episode Info ========\n")
                    print("======== Robots Info ========")
                    for i,rob in enumerate(self.train_env.robots):
                        info = infos[i]["state"]
                        print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info}")
                    print("======== Robots Info ========\n") 
                
                ep_rewards = np.zeros(len(self.train_env.robots))
                ep_length = 0

                states,_,_ = self.train_env.reset()
                # cvar = 1 - np.random.uniform(0.0, 1.0)

            self.current_timestep += 1

            end_1 = time.time()
            elapsed_time_1 = end_1 - start_1
            if self.current_timestep % 100 == 0:
                print("Elapsed time 1: {:.6f} seconds".format(elapsed_time_1))

    def linear_eps(self,total_timesteps):
        
        progress = self.current_timestep / total_timesteps
        if progress < self.exploration_fraction:
            r = progress / self.exploration_fraction
            return self.initial_eps + r * (self.final_eps - self.initial_eps)
        else:
            return self.final_eps

    def evaluation(self):
        """Evaluate performance of the agent
        Params
        ======
            eval_env (gym compatible env): evaluation environment
            eval_config: eval envs config file
        """
        actions_data = []
        rewards_data = []
        successes_data = []
        times_data = []
        energies_data = []
        relations_data = []
        obs_data = []
        objs_data = []
        
        for idx, config in enumerate(self.eval_config):
            print(f"Evaluating episode {idx}")
            state,_,_ = self.eval_env.reset_with_eval_config(config)
            obs = [[copy.deepcopy(rob.perception.observed_obs)] for rob in self.eval_env.robots]
            objs = [[copy.deepcopy(rob.perception.observed_objs)] for rob in self.eval_env.robots]
            
            rob_num = len(self.eval_env.robots)

            actions = [[] for _ in range(rob_num)]
            relations = [[] for _ in range(rob_num)]
            rewards = [0.0]*rob_num
            times = [0.0]*rob_num
            energies = [0.0]*rob_num
            end_episode = False
            length = 0
            
            while not end_episode:
                # gather actions for robots from agents 
                action = []
                for i,rob in enumerate(self.eval_env.robots):
                    if rob.reach_goal:
                        action.append(None)
                        continue

                    if rob.cooperative:
                        a,quantiles,taus,R_matrix = self.cooperative_agent.act(state[i])
                    else:
                        a,quantiles,taus,R_matrix = self.noncooperative_agent.act(state[i])
                    action.append(a)
                    actions[i].append(a)
                    relations[i].append(R_matrix.tolist())

                # execute actions in the training environment
                state, reward, done, info, end_episode = self.train_env.step(action)
                
                for i,rob in enumerate(self.eval_env.robots):
                    if rob.reach_goal:
                        continue
                    
                    if rob.cooperative:
                        rewards[i] += self.cooperative_agent.GAMMA ** length * reward[i]
                    else:
                        rewards[i] += self.noncooperative_agent.GAMMA ** length * reward[i]
                    times[i] += rob.dt * rob.N
                    energies[i] += rob.compute_action_energy_cost(action[i])
                    obs[i].append(copy.deepcopy(rob.perception.observed_obs))
                    objs[i].append(copy.deepcopy(rob.perception.observed_objs))

                length += 1


            success = True if self.eval_env.check_all_reach_goal() else False

            actions_data.append(actions)
            rewards_data.append(rewards)
            successes_data.append(success)
            times_data.append(times)
            energies_data.append(energies)
            relations_data.append(relations)
            obs_data.append(obs)
            objs_data.append(objs)
        
        avg_r = np.mean(rewards_data)
        success_rate = np.sum(successes_data)/len(successes_data)
        # idx = np.where(np.array(successes_data) == 1)[0]
        # avg_t = np.mean(np.array(time_data)[idx])
        # avg_e = np.mean(np.array(energy_data)[idx])

        print(f"++++++++ Evaluation Info ++++++++")
        print(f"Avg cumulative reward: {avg_r:.2f}")
        print(f"Success rate: {success_rate:.2f}")
        # print(f"Avg time: {avg_t:.2f}")
        # print(f"Avg energy: {avg_e:.2f}")
        print(f"++++++++ Evaluation Info ++++++++\n")

        self.eval_timesteps.append(self.current_timestep)
        self.eval_actions.append(actions_data)
        self.eval_rewards.append(rewards_data)
        self.eval_successes.append(successes_data)
        self.eval_times.append(times_data)
        self.eval_energies.append(energies_data)
        self.eval_relations.append(relations_data)
        self.eval_obs.append(obs_data)
        self.eval_objs.append(objs_data)

    def save_evaluation(self,eval_log_path):
        filename = "evaluations.npz"
        
        np.savez(
            os.path.join(eval_log_path,filename),
            timesteps=self.eval_timesteps,
            actions=self.eval_actions,
            rewards=self.eval_rewards,
            successes=self.eval_successes,
            times=self.eval_times,
            energies=self.eval_energies,
            relations=self.eval_relations,
            obs=self.eval_obs,
            objs=self.eval_objs
        )