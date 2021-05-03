import gym
import snake_envs

import numpy as np
import matplotlib.pyplot as plt
import random

# Idk why I used gym as it is useless
env = gym.make('CustomSnake-v0', board_size=[21,21])

action_size = env.action_space.n
state_x, state_y = env.board_size

# TO BE DEFINED
LOOKUP = 1
Q = np.zeros((21, 21, action_size))

# HYPERPARAMETERS
train_episodes = 1000000
max_steps = 1000

alpha = 1
gamma = 1

epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.0001

# TRAINING PHASE
training_rewards = []

""" DOES NOT WORK PROPERLY. JUST A TEST """

for episode in range(train_episodes):
    snake_pos = env.reset()
    cumulative_training_rewards = 0
    
    # state = env.board.around_snake(LOOKUP)

    for step in range(max_steps):
        if epsilon < 0.012: env.render(frame_speed=.001)
        state = np.subtract(env.board.snake.head, env.board.food)
        # print(Q[state[0], state[1]])

        if random.uniform(0, 1) > epsilon: 
            action = np.argmax(Q[state[0],state[1],:])
        else:
            action = env.action_space.sample()

        new_pos, reward, done, info = env.step(action)
        new_state = np.subtract(env.board.snake.head, env.board.food)

        # print(new_state)
        # print(new_state, reward, done, info)

        # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        try:
            pass
            Q[state[0], state[1], action] = Q[state[0], state[1], action] + alpha * (reward + gamma * np.max(Q[new_state[0], new_state[1]]) - Q[state[0], state[1], action]) 
        except:
            print("ERROR_Q_TABLE", action)

        cumulative_training_rewards += reward 

        state = new_state
        snake_pos = new_pos
        
        if done == True:
            print ("Cumulative reward for episode {}: {}".format(episode, cumulative_training_rewards))
            break
        
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    # append the episode cumulative reward to the list
    training_rewards.append(cumulative_training_rewards)

# print([[e[0] + b[0], e[1] + b[1]] for e in a])