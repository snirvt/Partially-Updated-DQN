from copy import deepcopy
import gym
import gym.spaces

# <!-- https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c -->
# https://github.com/jorditorresBCN/Deep-Reinforcement-Learning-Explained/blob/master/DRL_15_16_17_DQN_Pong.ipynb

DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import collections

from gym_wrappers import make_env

import torch
import torch.nn as nn        # Pytorch neural network package
import torch.optim as optim  # Pytorch optimization package

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

import numpy as np
from dqn import DQN, DQN_2_layers

test_env = make_env(DEFAULT_ENV_NAME)

import time
import numpy as np
import collections


MEAN_REWARD_BOUND = 19.0           

from hook_handler import rand_hook, weighted_rand_hook, greedy_hook, epsilon_greedy_hook

gamma = 0.99                   
batch_size = 64                
replay_size = 10000            
learning_rate = 1e-4           
# sync_target_frames = 1000      
replay_start_size = 10000      

eps_start=1.0
eps_decay=.999985
eps_min=0.02

transfer_learning = True



from experience_replay import ExperienceReplay
from agent import Agent
import datetime
env = make_env(DEFAULT_ENV_NAME)

# name = 'conv_history_dict'
# name = 'test_history_dict'
# name = 'control_batch64_fc128-128_sync1000'
# name = 'batch64_fc128-128_independent_grad_rand_hook_p01_min1'
name = 'batch64_fc128-128_dependent_grad_rand_hook_p01_min1'

# name = 'debug'

try:
    res_dict = np.load(name+'.npy', allow_pickle=True).item()
except:
    res_dict = {}
    np.save(name+'.npy',res_dict, allow_pickle=True)

for i in range(30):
    print(">>>Training starts at ",datetime.datetime.now())
    
    net = DQN_2_layers(env.observation_space.shape, env.action_space.n).to(device)
    
    net.fc[0].weight.register_hook(lambda grad: rand_hook(grad, p=0.1))
    net.fc[2].weight.register_hook(lambda grad: rand_hook(grad, p=0.1))
    net.fc[4].weight.register_hook(lambda grad: rand_hook(grad, p=0.1))


    if transfer_learning:
        pretrained_dqn = DQN(env.observation_space.shape, env.action_space.n)
        pretrained_dqn.load_state_dict(torch.load('PongNoFrameskip-v4-conv.dat',map_location=device))    
        net.transfer_learning(deepcopy(pretrained_dqn.conv))
    
    buffer = ExperienceReplay(replay_size)
    agent = Agent(env, buffer)

    epsilon = eps_start

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_rewards = []
    frame_idx = 0  

    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(epsilon*eps_decay, eps_min)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)

            mean_reward = np.mean(total_rewards[-10:])

            print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (
                frame_idx, len(total_rewards), mean_reward, epsilon))

            if best_mean_reward is None or best_mean_reward < mean_reward:
                # torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-conv.dat")
                np.save(name+'.npy',res_dict, allow_pickle=True)
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f" % (best_mean_reward))

            if len(total_rewards) > 200:
                res_dict[len(res_dict.keys())] = deepcopy(total_rewards)
                print("Played all the episodes")
                break

        if len(buffer) < replay_start_size:
            continue

        batch = buffer.sample(batch_size)
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1).type(torch.int64)).squeeze(-1)

        next_state_values = net(next_states_v).max(1)[0]

        next_state_values[done_mask] = 0.0

        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * gamma + rewards_v

        loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

        # net.freeze()
        optimizer.zero_grad()
        loss_t.backward()
        # optimizer.param_groups[0]['params'][-2] = rand_hook(optimizer.param_groups[0]['params'][-2])
        # optimizer.param_groups[0]['params'][-4] = rand_hook(optimizer.param_groups[0]['params'][-4])
        # optimizer.param_groups[0]['params'][-6] = rand_hook(optimizer.param_groups[0]['params'][-6])
        optimizer.step()
        
    # writer.close()
    # torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-conv.dat")
    np.save(name+'.npy',res_dict, allow_pickle=True)
    print(">>>Training ends at ",datetime.datetime.now())
