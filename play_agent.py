import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import time

from gym_wrappers import make_env
from agent import Agent
from experience_replay import ExperienceReplay
from dqn import DQN, DQN_2_layers


DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 
# DEFAULT_ENV_NAME = 'Breakout-v0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

render_img = True

env = make_env(DEFAULT_ENV_NAME)
env = gym.wrappers.Monitor(env, 'video', force=True)

model = DQN_2_layers(env.observation_space.shape, env.action_space.n)
# model = DQN_2_layers(env.observation_space.shape, 6)
model.load_state_dict(torch.load("models/trained_dqn.dat",map_location=device))    


buffer = ExperienceReplay(capacity=0)
agent = Agent(env, buffer)

if render_img:
    _ = env.render(mode="human")

print_list = []
for num_step in range(2000):
    reward = agent.play_step(model, epsilon=0)
    if render_img:
        _ = env.render(mode="human")
        # time.sleep(0.02)
    if reward is not None:
        break
env.close()


