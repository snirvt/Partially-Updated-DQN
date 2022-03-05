from copy import deepcopy
import gym
import gym.spaces
import sys
# <!-- https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c -->
# https://github.com/jorditorresBCN/Deep-Reinforcement-Learning-Explained/blob/master/DRL_15_16_17_DQN_Pong.ipynb

DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 
# DEFAULT_ENV_NAME = 'Breakout-v4'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import collections

from gym_wrappers import make_env

import torch
import torch.nn as nn        # Pytorch neural network package
import torch.optim as optim  # Pytorch optimization package

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")
print(device)

import numpy as np
from dqn import DQN, DQN_2_layers

import time
import numpy as np
import collections
from experience_replay import ExperienceReplay, PrioritizedExperienceReplay
from agent import Agent
import datetime



test_env = make_env(DEFAULT_ENV_NAME)

MEAN_REWARD_BOUND = 19.0           

gamma = 0.99                   
batch_size = 32                
replay_size = 3000            
learning_rate =  0.00025     
sync_target_frames = 500      
replay_start_size = replay_size      

# eps_start=1.0
eps_start=2

# eps_decay=.999985
eps_decay=0.98

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

# Main training loop Starts

env = make_env(DEFAULT_ENV_NAME)

# name = 'conv_history_dict'
# name = 'test_history_dict'
# name = 'control_batch64_fc128-128_sync1000'
# name = 'control_batch64_fc128-128_sync1'
# name = 'control_batch64_fc128-128_sync500'
# name = 'control_batch64_fc128-128_sync250'
# name = 'no_transfer_control_batch128_fc128-128_sync1_replay_size5000'
# name = 'control_small_dqn_2frames_lr0.0001_sync1000'
# name = 'control_small_dqn_2frames_lr0.0001_sync500'
# name = 'control_small_dqn_2frames_lr0.0001_sync250'
# name = 'control_small_dqn_2frames_lr0.0001_sync125'
# name = 'control_small_dqn_2frames_lr0.0001_sync62'
# name = 'control_small_dqn_2frames_lr0.0001_sync1'

# name = 'control_breakout_v4_skip2'
name = 'debug'
print(name)
# transfer_learning = True
transfer_learning = False

try:
    res_dict = np.load('results/'+name+'.npy', allow_pickle=True).item()
except:
    res_dict = {}
    np.save('results/'+name+'.npy',res_dict, allow_pickle=True)

for i in range(30):
    print(">>>Training starts at ",datetime.datetime.now())
    
    net = DQN_2_layers(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQN_2_layers(env.observation_space.shape, env.action_space.n).to(device)
    
    if transfer_learning:
        pretrained_dqn = DQN(env.observation_space.shape, env.action_space.n)
        pretrained_dqn.load_state_dict(torch.load('PongNoFrameskip-v4-conv.dat',map_location=device))    
        net.transfer_learning(deepcopy(pretrained_dqn.conv))
        target_net.transfer_learning(deepcopy(pretrained_dqn.conv))
    
    buffer = PrioritizedExperienceReplay(replay_size)
    agent = Agent(env, buffer)

    epsilon = eps_start

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_rewards = []
    frame_idx = 0  

    best_mean_reward = None

    while True:
        frame_idx += 1

        reward = agent.play_priorotized_step(net, target_net, gamma, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            epsilon = max(epsilon*eps_decay, eps_min)


            # mean_reward = np.mean(total_rewards[-10:])
            mean_reward = reward

            print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (
                frame_idx, len(total_rewards), mean_reward, epsilon))

            if best_mean_reward is None or best_mean_reward < mean_reward:
                # torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-conv.dat")
                # np.save(name+'.npy',res_dict, allow_pickle=True)
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f" % (best_mean_reward))

            if len(total_rewards) > 1000:
                res_dict[len(res_dict.keys())] = deepcopy(total_rewards)
                np.save('results/'+name+'.npy',res_dict, allow_pickle=True)
                print("Played all the episodes")
                break

        if len(buffer) < replay_start_size:
            continue
        if frame_idx % update_rate == 0:

            batch = buffer.sample(batch_size)
            states, actions, rewards, dones, next_states = batch

            states_v = torch.tensor(states).to(device)
            next_states_v = torch.tensor(next_states).to(device)
            actions_v = torch.tensor(actions).to(device)
            rewards_v = torch.tensor(rewards).to(device)
            done_mask = torch.ByteTensor(dones).to(device)

            optimizer.zero_grad()

            state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1).type(torch.int64)).squeeze(-1)

            next_state_values = target_net(next_states_v).max(1)[0]

            next_state_values[done_mask] = 0.0

            next_state_values = next_state_values.detach()

            expected_state_action_values = next_state_values * gamma + rewards_v
            TD_error_abs = torch.abs(state_action_values - expected_state_action_values)
            buffer.update_td_error(TD_error_abs)

            loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)
            
            # optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

        if frame_idx % sync_target_frames == 0:
            target_net.load_state_dict(net.state_dict())
        
    np.save('results/'+name+'.npy',res_dict, allow_pickle=True)
    print(">>>Training ends at ",datetime.datetime.now())


