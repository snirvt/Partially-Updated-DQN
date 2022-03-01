import numpy as np
import matplotlib.pyplot as plt
import gym
import torch

from gym_wrappers import make_env
from agent import Agent
from experience_replay import ExperienceReplay
from dqn import DQN, DQN_2_layers


DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


render_img = True
# print_vals = True

# model = np.load("bestQ.npy", allow_pickle=True)





env = make_env(DEFAULT_ENV_NAME)


model = DQN(env.observation_space.shape, env.action_space.n)
# model.load_state_dict(torch.load('PongNoFrameskip-v4-conv.dat',map_location=device))    


buffer = ExperienceReplay(capacity=0)
agent = Agent(env, buffer)

cnt = 0
# observation = env.reset()
if render_img:
    _ = env.render(mode="human")

total_reward = 0
print_list = []
for num_step in range(200):
    cnt+=1
    # f_s = get_features(observation)
    # action = get_mean_action(f_s, bestQ)
    agent.play_step(model)
    # observation, reward, done, info = env.step(action)
    # reward += 20
    if render_img:
        _ = env.render(mode="human")
    # if print_vals:
        # print_list.append([cnt, *action[0], observation, reward[0]])
    # total_reward += reward
    # if done:
    #     observation = env.reset()
    #     break
env.close()

# print('Games over\nNumber of steps: {}\nTotal reward: {}\n'.format(num_step+1, total_reward))



