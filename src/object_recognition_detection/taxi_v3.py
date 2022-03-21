import numpy as np
import gym
import random
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')
action_size = env.action_space.n
state_size = env.observation_space.n

episodes = 10000
learning_rate = 0.1
dicout_rate =0.95
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_epsilon = 0.01

rewards_all_episodes = []


q_table = np.zeros((state_size , action_size))

for episode in range (episodes):
    done = False 
    state = env.reset()
    episode_reward = 0
    while not done :
        if (random.uniform(0, 1) > epsilon):
            
            action = np.argmax(q_table[state,:]) 
        else:
            action = env.action_space.sample()
            
            
        new_state , reward , done , info = env.step(action)
        #env.render()
        episode_reward += reward
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + dicout_rate * np.max(q_table[new_state, :]))
        state = new_state

    rewards_all_episodes.append(episode_reward)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_epsilon*episode)

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

plt.plot(rewards_all_episodes)
env.close()