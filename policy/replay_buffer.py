import random
from collections import deque
import torch

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, cooperative, seed=249):
        """
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.cooperative = cooperative
        self.seed = random.seed(seed)
    
    def add(self,item):
        """Add a new experience to memory."""
        self.memory.append(item)

    def state_batch(self,states):   
        self_state_batch = []
        static_batch = []
        static_nums = [0]
        if self.cooperative:
            dynamic_batch = []
            dynamic_nums = [0]
            dynamic_indices = []
            count = 0
        
        for state in states:
            self_state_batch.append(state[0])

            last_num = 0 if len(static_nums) == 0 else static_nums[-1]
            static_nums.append(last_num + len(state[1]))
            for feature in state[1]:
                static_batch.append(feature)

            if self.cooperative:
                last_num = 0 if len(dynamic_nums) == 0 else dynamic_nums[-1]
                dynamic_nums.append(last_num + len(state[2]))
                for i,feature in enumerate(state[2]):
                    dynamic_batch.append(feature)
                    dynamic_indices.append([count,state[3][i][-1]])
                    count += 1

        if self.cooperative:
            return (self_state_batch,static_batch,static_nums,dynamic_batch,dynamic_nums,dynamic_indices)
        else:
            return (self_state_batch,static_batch,static_nums) 
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        samples = random.sample(self.memory, k=self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for sample in samples:
            state, action, reward, next_state, done = sample 
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = self.state_batch(states)
        next_states = self.state_batch(next_states)

        return states, actions, rewards, next_states, dones

    def size(self):
        """Return the current size of internal memory."""
        return len(self.memory)