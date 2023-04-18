import torch
import torch.optim as optim
from policy.policy import PolicyNetwork
from policy.replay_buffer import ReplayBuffer

class Agent():
    def __init__(self, 
                 self_dimension,
                 static_dimension,
                 feature_dimension,
                 gcn_hidden_dimension,
                 feature_2_dimension,
                 iqn_hidden_dimension,
                 action_size,
                 dynamic_dimension=None,
                 cooperative=False, 
                 BATCH_SIZE=32, 
                 BUFFER_SIZE=1_000_000,
                 LR=1e-4, 
                 TAU=1.0, 
                 GAMMA=0.99, 
                 UPDATE_EVERY=4,
                 learning_starts=10000,
                 target_update_interval=10000,
                 exploration_fraction=0.1,
                 initial_eps=1.0,
                 final_eps=0.05, 
                 device="cpu", 
                 seed=0):
        
        self.device = device
        self.LR = LR
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.initial_eps = initial_eps
        self.final_eps = final_eps

        self.policy_local = PolicyNetwork(self_dimension,static_dimension,feature_dimension,
                                          gcn_hidden_dimension,feature_2_dimension,iqn_hidden_dimension,
                                          action_size,dynamic_dimension,cooperative,seed).to(device)
        self.policy_target = PolicyNetwork(self_dimension,static_dimension,feature_dimension,
                                          gcn_hidden_dimension,feature_2_dimension,iqn_hidden_dimension,
                                          action_size,dynamic_dimension,cooperative,seed).to(device)
        
        self.optimizer = optim.Adam(self.policy_local.parameters(), lr=self.LR)

        self.memory = ReplayBuffer(BUFFER_SIZE)

        # current time step
        self.current_timestep = 0

        # learning time step (start counting after learning_starts time step)
        self.learning_timestep = 0

        # evaluation data
        self.eval_timesteps = dict(greedy=[],adaptive=[])
        self.eval_actions = dict(greedy=[],adaptive=[])
        self.eval_rewards = dict(greedy=[],adaptive=[])
        self.eval_successes = dict(greedy=[],adaptive=[])
        self.eval_times = dict(greedy=[],adaptive=[])
        self.eval_energies = dict(greedy=[],adaptive=[])
