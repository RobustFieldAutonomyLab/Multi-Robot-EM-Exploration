import json
import numpy as np
import os

class Trainer():
    def __init__(self,
                 train_env,
                 eval_env,
                 eval_schedule,
                 non_cooperative_agent,
                 cooperative_agent=None,
                 UPDATE_EVERY=4,
                 learning_starts=10000,
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

    def save_eval_config(self,filename):
        with open(filename, "w+") as f:
            json.dump(self.eval_config, f)

    def learn(self,
              total_timesteps,
              eval_freq,
              eval_log_path,
              verbose=True):
        
        states = self.train_env.reset()

        # # Sample CVaR value from (0.0,1.0)
        # cvar = 1 - np.random.uniform(0.0, 1.0)

        # current episode 
        ep_rewards = np.zeros(len(self.train_env.robots))
        ep_length = 0
        ep_num = 0
        num_initial_robots = len(self.train_env.robots)
        num_reach_goal = 0
        
        while self.current_timestep <= total_timesteps:
            eps = self.linear_eps(total_timesteps)
            
            # gather actions for robots from agents 
            actions = []
            for i,rob in enumerate(self.train_env.robots):
                if rob.cooperative:
                    action = self.cooperative_agent.act(states[i],eps)
                else:
                    action = self.noncooperative_agent.act(states[i],eps)
                actions.append(action)

            # execute actions in the training environment
            next_states, rewards, dones, infos, end_episode, robot_types = self.train_env.step(actions)

            # save experience in replay memory
            # TODO: consider setting robot to None when it reaches the goal and the episode does not end
            for i,cooperative in enumerate(robot_types):
                if cooperative:
                    ep_rewards[i] += self.cooperative_agent.GAMMA ** ep_length * rewards[i]
                    self.cooperative_agent.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))
                else:
                    ep_rewards[i] += self.noncooperative_agent.GAMMA ** ep_length * rewards[i]
                    self.noncooperative_agent.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))
                
                if infos[i]["state"] == "reach goal":
                    num_reach_goal += 1

            states = next_states
            ep_length += 1
            
            # Learn, update and evaluate models after learning_starts time step 
            if self.current_timestep >= self.learning_starts:
                for agent in [self.cooperative_agent,self.noncooperative_agent]:
                    if agent is None:
                        continue

                    # Learn every UPDATE_EVERY time steps.
                    if self.learning_timestep % self.UPDATE_EVERY == 0:
                        # If enough samples are available in memory, get random subset and learn
                        if agent.memory.size() > agent.BATCH_SIZE:
                            agent.train()

                    # Update the target model every target_update_interval time steps
                    if self.learning_timestep % self.target_update_interval == 0:
                        agent.soft_update(self.policy_local, self.policy_target)

                # Evaluate learning agents every eval_freq time steps
                if self.learning_timestep % eval_freq == 0: 
                    self.evaluation()

                    for agent in [self.cooperative_agent,self.noncooperative_agent]:
                        if agent is None:
                            continue
                        # save the latest models
                        agent.save_latest_model(eval_log_path)

                self.learning_timestep += 1

            if end_episode:
                ep_num += 1
                
                if verbose:
                    # print abstract info of learning process
                    print("======== training info ========")
                    print("current ep_length: ",ep_length)
                    print("robots_num: ",num_initial_robots)
                    print("success_robots_num: ",num_reach_goal)
                    print("episodes_num: ",ep_num)
                    print("exploration_rate: ",eps)
                    print("current_timesteps: ",self.current_timestep)
                    print("total_timesteps: ",total_timesteps)
                    print("======== training info ========\n") 
                
                ep_reward = 0.0
                ep_length = 0

                states = self.train_env.reset()
                num_initial_robots = len(self.train_env.robots)
                num_reach_goal = 0
                # cvar = 1 - np.random.uniform(0.0, 1.0)

            self.current_timestep += 1

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
        action_data = []
        reward_data = []
        success_data = []
        time_data = []
        energy_data = []
        
        for idx, config in enumerate(eval_config.values()):
            print(f"Evaluating episode {idx}")
            observation = eval_env.reset_with_eval_config(config)
            actions = []
            cumulative_reward = 0.0
            length = 0
            energy = 0.0
            done = False
            
            while not done and length < 1000:
                if greedy:
                    action,quantiles,taus,R_matrix = self.act(observation,eps=0.0)
                else:
                    action,quantiles,taus,R_matrix,cvar = self.act_adaptive(observation,eps=0.0)    
                observation, reward, done, info = eval_env.step(action)
                cumulative_reward += eval_env.discount ** length * reward
                length += 1
                energy += eval_env.robot.compute_action_energy_cost(int(action))
                actions.append(int(action))

            success = True if info["state"] == "reach goal" else False
            time = eval_env.robot.dt * eval_env.robot.N * length

            action_data.append(actions)
            reward_data.append(cumulative_reward)
            success_data.append(success)
            time_data.append(time)
            energy_data.append(energy)
        
        avg_r = np.mean(reward_data)
        success_rate = np.sum(success_data)/len(success_data)
        idx = np.where(np.array(success_data) == 1)[0]
        avg_t = np.mean(np.array(time_data)[idx])
        avg_e = np.mean(np.array(energy_data)[idx])

        policy = "greedy" if greedy else "adaptive"
        print(f"++++++++ Evaluation info ({policy} IQN) ++++++++")
        print(f"Avg cumulative reward: {avg_r:.2f}")
        print(f"Success rate: {success_rate:.2f}")
        print(f"Avg time: {avg_t:.2f}")
        print(f"Avg energy: {avg_e:.2f}")
        print(f"++++++++ Evaluation info ({policy} IQN) ++++++++\n")

        self.eval_timesteps.append(self.current_timestep)
        self.eval_actions.append(action_data)
        self.eval_rewards.append(reward_data)
        self.eval_successes.append(success_data)
        self.eval_times.append(time_data)
        self.eval_energies.append(energy_data)

    def save_evaluation(self,eval_log_path):
        filename = "evaluations.npz"
        
        np.savez(
            os.path.join(eval_log_path,filename),
            timesteps=self.eval_timesteps,
            actions=self.eval_actions,
            rewards=self.eval_rewards,
            successes=self.eval_successes,
            times=self.eval_times,
            energies=self.eval_energies
        )