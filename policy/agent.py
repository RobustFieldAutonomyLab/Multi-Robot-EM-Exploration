import torch
import torch.optim as optim
from policy.models import PolicyNetwork
from policy.replay_buffer import ReplayBuffer
import numpy as np
import random

class Agent():
    def __init__(self, 
                 self_dimension=4,
                 static_dimension=3,
                 feature_dimension=36,
                 gcn_hidden_dimension=216,
                 feature_2_dimension=216,
                 iqn_hidden_dimension=64,
                 action_size=9,
                 dynamic_dimension=4,
                 cooperative=True, 
                 BATCH_SIZE=1, 
                 BUFFER_SIZE=1_000_000,
                 LR=1e-4, 
                 TAU=1.0, 
                 GAMMA=0.99,  
                 device="cpu", 
                 seed_1=100,
                 seed_2=101,
                 training=True):
        
        self.cooperative = cooperative
        self.device = device
        self.LR = LR
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.training = training
        self.action_size = action_size

        if training:
            self.policy_local = PolicyNetwork(self_dimension,static_dimension,feature_dimension,
                                            gcn_hidden_dimension,feature_2_dimension,iqn_hidden_dimension,
                                            action_size,dynamic_dimension,cooperative,device,seed_1).to(device)
            self.policy_target = PolicyNetwork(self_dimension,static_dimension,feature_dimension,
                                            gcn_hidden_dimension,feature_2_dimension,iqn_hidden_dimension,
                                            action_size,dynamic_dimension,cooperative,device,seed_2).to(device)
        
            self.optimizer = optim.Adam(self.policy_local.parameters(), lr=self.LR)

            self.memory = ReplayBuffer(BUFFER_SIZE,BATCH_SIZE,cooperative)

    def act(self, state, eps=0.0, cvar=1.0):
        """Returns action index and quantiles 
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
        """
        state_t = self.state_to_tensor(state) 
        self.policy_local.eval()
        with torch.no_grad():
            quantiles, taus, R_matrix = self.policy_local(state_t, self.policy_local.iqn.K, cvar)
            action_values = quantiles.mean(dim=0)
        self.policy_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        
        return action, quantiles.cpu().data.numpy(), taus.cpu().data.numpy(), R_matrix.cpu().data.numpy()

    def act_adaptive(self, state, eps=0.0):
        """adptively tune the CVaR value, compute action index and quantiles
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
        """
        cvar = self.adjust_cvar(state)
        return self.act(state, eps, cvar), cvar

    def adjust_cvar(self,state):
        # scale CVaR value according to the closest distance to obstacles
        sonar_points = state[4:]
        
        closest_d = np.inf
        for i in range(0,len(sonar_points),2):
            x = sonar_points[i]
            y = sonar_points[i+1]

            if np.abs(x) < 1e-3 and np.abs(y) < 1e-3:
                continue

            closest_d = min(closest_d, np.linalg.norm(sonar_points[i:i+2]))
        
        cvar = 1.0
        if closest_d < 10.0:
            cvar = closest_d / 10.0

        return cvar
    
    def state_to_tensor(self,states):
        if self.cooperative:
            self_state_batch,static_batch,static_nums,dynamic_batch,dynamic_nums,dynamic_indices = states
        else:
            self_state_batch,static_batch,static_nums = states

        self_state_batch = torch.tensor(self_state_batch).float().to(self.device)
        empty = (len(static_batch) == 0)
        static_batch = None if empty else torch.tensor(static_batch).float().to(self.device)
        static_nums = None if empty else torch.tensor(static_nums).to(self.device)
        
        if self.cooperative:
            empty = (len(dynamic_batch) == 0)
            dynamic_batch = None if empty else torch.tensor(dynamic_batch).float().to(self.device)
            dynamic_nums = None if empty else torch.tensor(dynamic_nums).to(self.device)
            dynamic_indices = None if empty else torch.tensor(dynamic_indices).to(self.device)
            return (self_state_batch,static_batch,static_nums,dynamic_batch,dynamic_nums,dynamic_indices)
        else:
            return (self_state_batch,static_batch,static_nums)
    
    def train(self):
        """Update value parameters using given batch of experience tuples
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = self.memory.sample()
        states = self.state_to_tensor(states)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = self.state_to_tensor(next_states)
        dones = torch.tensor(dones).float().to(self.device)

        self.optimizer.zero_grad()
        # Get max predicted Q values (for next states) from target model
        Q_targets_next, _ = self.policy_target(next_states)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1) # (batch_size, 1, N)
        
        # Compute Q targets for current states 
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA ** self.n_step * Q_targets_next * (1. - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected, taus = self.policy_local(states)
        Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, 8, 8), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
        
        loss = quantil_l.sum(dim=1).mean(dim=1) # keepdim=True if per weights get multiple
        loss = loss.mean()

        # minimize the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 0.5)
        self.optimizer.step()

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.policy_target.parameters(), self.policy_local.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

    def save_latest_model(self,directory):
        self.policy_local.save(directory)

    def load_model(self,path,agent_type,device="cpu"):
        self.policy_local = PolicyNetwork.load(path,agent_type,device)


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (8, 8), "huber loss has wrong shape"
    return loss