
import numpy as np
import collections

import torch

from buffers import get_exp_buffer, get_priorotized_exp_buffer
# Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
# Priorotized_Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state', 'td_error'])


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):

        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = get_exp_buffer()(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward
    
    def play_priorotized_step(self, net, target_net, gamma, epsilon=0.0, device="cpu"):
        done_reward = None

        state_a = np.array([self.state], copy=False)
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net(state_v)

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        next_state_v = torch.tensor(new_state).to(device)
        next_state_values = target_net(next_state_v.unsqueeze(0)).max(1)[0]

        td_error = torch.abs(reward + gamma * next_state_values - q_vals_v.max(1)[0])

        exp = get_priorotized_exp_buffer()(self.state, action, reward, is_done, new_state, td_error)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward