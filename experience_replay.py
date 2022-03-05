
import numpy as np
import collections

import torch
from buffers import get_exp_buffer, get_priorotized_exp_buffer

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

class PrioritizedExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.indices = []
        
    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, tournement_size=4, epsilon=0.2):
        available_idx = set(range(len(self.buffer)))
        self.indices = []
        greedy_size = int(batch_size * (1-epsilon))
        for _ in range(greedy_size):
            tournement_idx = np.random.choice(list(available_idx), tournement_size, replace=False)
            _, _, _, _, _, td_error = zip(*[self.buffer[idx] for idx in tournement_idx])
            td_error = [self.buffer[idx][-1].item() for idx in tournement_idx]

            self.indices.append(tournement_idx[np.argmax(td_error)])
            available_idx = available_idx - set([self.indices[-1]])
        
        if epsilon > 0:
            indices_eps = np.random.choice(list(available_idx), batch_size - greedy_size, replace=False)
            self.indices += list(indices_eps)

        states, actions, rewards, dones, next_states, _ = zip(*[self.buffer[idx] for idx in self.indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)
    
    def update_td_error(self, td_error):
        for i,index in enumerate(self.indices):
            state, action, reward, is_done, new_state, _ = self.buffer[index]
            exp = get_priorotized_exp_buffer()(state, action, reward, is_done, new_state, td_error[i])
            self.buffer[index] = exp
        
