from torch.utils.data import Dataset

class ReplayBuffer(Dataset):
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, index):
        return self.memory[index]
    
    def push(self,element):
        if len(self.memory) < self.capacity:
            self.memory.append(element)
        else:
            self.memory[self.position] = element
        self.position = (self.position+1) % self.capacity