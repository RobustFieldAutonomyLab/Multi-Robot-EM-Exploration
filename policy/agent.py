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
                 BATCH_SIZE=32, 
                 BUFFER_SIZE=1_000_000,
                 LR=1e-4, 
                 TAU=1.0, 
                 GAMMA=0.99,  
                 device="cpu", 
                 seed_1=0,
                 seed_2=1):
        
        self.cooperative = cooperative
        self.device = device
        self.LR = LR
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE

        self.policy_local = PolicyNetwork(self_dimension,static_dimension,feature_dimension,
                                          gcn_hidden_dimension,feature_2_dimension,iqn_hidden_dimension,
                                          action_size,dynamic_dimension,cooperative,seed_1).to(device)
        self.policy_target = PolicyNetwork(self_dimension,static_dimension,feature_dimension,
                                          gcn_hidden_dimension,feature_2_dimension,iqn_hidden_dimension,
                                          action_size,dynamic_dimension,cooperative,seed_2).to(device)
        
        self.optimizer = optim.Adam(self.policy_local.parameters(), lr=self.LR)

        self.memory = ReplayBuffer(BUFFER_SIZE,BATCH_SIZE)

    def act(self, state, eps=0.0, cvar=1.0):
        """Returns action index and quantiles 
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.policy_local.eval()
        with torch.no_grad():
            quantiles, taus, R_matrix = self.policy_local(state, self.policy_local.K, cvar)
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
    
    def train(self):
        """Update value parameters using given batch of experience tuples
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        batch = self.memory.sample()
        for sample in batch:
            state, action, reward, next_state, done = sample
            
            self.optimizer.zero_grad()
        
            # Get max predicted quantile values (for next states) from target model
            Q_target_next, _, _ = self.policy_target(next_state)
            action_next = Q_target_next.mean(0).argmax()
            Q_target_next = Q_target_next[:,action_next].detach().unsqueeze(0) # (1, N) 
            Q_target = reward + (self.GAMMA * Q_target_next * (1. - done))
            
            # Get expected quantile values (not expected values) from local model
            Q_expected, taus, _ = self.policy_local(state)
            Q_expected = Q_expected[:,action].unsqueeze(0)

            # Quantile Huber loss
            td_error = Q_target - Q_expected
            assert td_error.shape == (8, 8), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
            
            loss = quantil_l.sum(dim=1).mean(dim=1) # keepdim=True if per weights get multiple
            loss = loss.mean()

            # minimize the loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 0.5)
            self.optimizer.step()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

    def save_latest_model(self,directory):
        self.policy_local.save(directory)


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (8, 8), "huber loss has wrong shape"
    return loss