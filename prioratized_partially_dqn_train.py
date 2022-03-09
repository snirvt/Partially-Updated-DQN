from copy import deepcopy
import gym
import gym.spaces
import sys

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
print(device)

import numpy as np
from dqn import DQN, DQN_2_layers
import time
import numpy as np
import collections

from hook_handler import rand_hook, weighted_rand_hook, greedy_hook, epsilon_greedy_hook

gamma = 0.99                   
batch_size = 64                
replay_size = 10000            
learning_rate =  0.0001         
replay_start_size = replay_size      

eps_start=1.0
eps_decay = 0.98
eps_min=0.02

update_rate = 1

print('gamma: {}'.format(gamma))
print('batch_size: {}'.format(batch_size))
print('replay_size: {}'.format(replay_size))
print('learning_rate: {}'.format(learning_rate))
print('replay_start_size: {}'.format(replay_start_size))
print('eps_start: {}'.format(eps_start))
print('eps_decay: {}'.format(eps_decay))
print('eps_min: {}'.format(eps_min))
print('update_rate: {}'.format(update_rate))

from experience_replay import ExperienceReplay, PrioritizedExperienceReplay
from agent import Agent
import datetime
env = make_env(DEFAULT_ENV_NAME)
name = 'small_dqn_2frames_lr0.0001_first_layer_only_rand_hook_grad_1_05_priorazited'
transfer_learning = False
print(name)

try:
    res_dict = np.load('results/'+name+'.npy', allow_pickle=True).item()
except:
    res_dict = {}
    np.save('results/'+name+'.npy',res_dict, allow_pickle=True)

for i in range(30):
    print(">>>Training starts at ",datetime.datetime.now())
    
    net = DQN_2_layers(env.observation_space.shape, env.action_space.n).to(device)
    
    # net.conv[0].weight.register_hook(lambda grad: weighted_rand_hook(grad, p_non_zero = 1, p_all=1))
    # net.conv[2].weight.register_hook(lambda grad: weighted_rand_hook(grad, p_non_zero = 1, p_all=1))
    # net.conv[4].weight.register_hook(lambda grad: weighted_rand_hook(grad, p_non_zero = 1, p_all=1))

    # net.fc[0].weight.register_hook(lambda grad: greedy_hook(grad, p_non_zero = 1, p_all=0.8))
    net.fc[2].weight.register_hook(lambda grad: rand_hook(grad, p_non_zero = 1, p_all=0.5))
    # print('net.fc[0].weight.register_hook(lambda grad: greedy_hook(grad, p_non_zero = 1, p_all=0.8))')
    print('net.fc[2].weight.register_hook(lambda grad: rand_hook(grad, p_non_zero = 1, p_all=0.5))')

    # net.fc[2].weight.register_hook(lambda grad: greedy_hook(grad, p_non_zero = 0.25, p_all=0.5))
    # net.fc[2].weight.register_hook(lambda grad: epsilon_greedy_hook(grad, p_non_zero = 1, p_all=0.25, epsilon = 0.9))
    # print('net.fc[2].weight.register_hook(lambda grad: epsilon_greedy_hook(grad, p_non_zero = 1, p_all=0.25, epsilon = 0.9))')
    # net.fc[4].weight.register_hook(lambda grad: rand_hook(grad, p_non_zero = 1/2, p_all=1/4))

    if transfer_learning:
        pretrained_dqn = DQN(env.observation_space.shape, env.action_space.n)
        pretrained_dqn.load_state_dict(torch.load('PongNoFrameskip-v4-conv.dat',map_location=device))    
        net.transfer_learning(deepcopy(pretrained_dqn.conv))
    
    buffer = PrioritizedExperienceReplay(replay_size)
    agent = Agent(env, buffer)

    epsilon = eps_start

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_rewards = []
    frame_idx = 0  

    best_mean_reward = None

    while True:
        frame_idx += 1
        reward = agent.play_priorotized_step(net, net, gamma, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            epsilon = max(epsilon*eps_decay, eps_min)

            # mean_reward = np.mean(total_rewards[-10:])
            mean_reward = reward

            print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (
                frame_idx, len(total_rewards), mean_reward, epsilon))

            if best_mean_reward is None or best_mean_reward <= mean_reward:
                # torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-conv.dat")
                np.save('results/'+name+'.npy',res_dict, allow_pickle=True)
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f" % (best_mean_reward))

            if len(total_rewards) > 500:
                res_dict[len(res_dict.keys())] = deepcopy(total_rewards)
                np.save(name+'.npy',res_dict, allow_pickle=True)
                print("Played all the episodes")
                break

        if len(buffer) < replay_start_size:
            continue

        if frame_idx % update_rate == 0:
            # optimizer.zero_grad() 
            ''' Auto Encoder '''
            # batch = buffer.sample(batch_size)
            # states, actions, rewards, dones, next_states = batch
            # optimizer.zero_grad()
            # auto_encoder_data = states
            # auto_encoder_data_tensor = torch.tensor(auto_encoder_data).to(device)
            # ae_pred = net.forward_ae(auto_encoder_data_tensor)
            # loss_ae = nn.MSELoss()(auto_encoder_data_tensor[:,1:,:,:].reshape(-1,84*84), ae_pred.reshape(-1,84*84))
            # loss_ae.backward()
            # optimizer.step()


            batch = buffer.sample(batch_size)
            states, actions, rewards, dones, next_states = batch
            ''' DQN '''
            optimizer.zero_grad()
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
            TD_error_abs = torch.abs(state_action_values - expected_state_action_values)
            buffer.update_td_error(TD_error_abs)            
            loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

            # optimizer.zero_grad()
            loss_t.backward()

            # optimizer.param_groups[0]['params'][-8] = weighted_rand_hook(optimizer.param_groups[0]['params'][-8], p_non_zero=1/2, p_all=1/10)
            # optimizer.param_groups[0]['params'][-10] = weighted_rand_hook(optimizer.param_groups[0]['params'][-10], p_non_zero=1/2, p_all=1/10)
            # optimizer.param_groups[0]['params'][-12] = weighted_rand_hook(optimizer.param_groups[0]['params'][-12], p_non_zero=1/2, p_all=1/10)

            # optimizer.param_groups[0]['params'][-2] = weighted_rand_hook(optimizer.param_groups[0]['params'][-2], p_non_zero=1/2, p_all=1/5)
            # optimizer.param_groups[0]['params'][-4] = weighted_rand_hook(optimizer.param_groups[0]['params'][-4], p_non_zero=1/2, p_all=1/5)
            # optimizer.param_groups[0]['params'][-6] = weighted_rand_hook(optimizer.param_groups[0]['params'][-6], p_non_zero=1/2, p_all=1/5)
            optimizer.step()
        
    np.save('results/'+name+'.npy',res_dict, allow_pickle=True)
    print(">>>Training ends at ",datetime.datetime.now())
