from custom_snake.snake_env import SnakeEnv

import numpy as np
import matplotlib.pyplot as plt
import random, json

def load_Q(file):
    q = {}
    with open(file, "r") as f:
        q = json.load(f)
        print(file, "loaded!")
    return q

def save_Q(file, Q):
    with open(file, "w") as f:
        json.dump(Q, f)

env = SnakeEnv([10,10])

action_size = env.action_space.n
state_x, state_y = env.board_size

# TO BE DEFINED
VERSION = 4
LOOKUP = 1
Q = load_Q("q-table-v{}.json".format(VERSION))

# HYPERPARAMETERS
train_episodes = 1000
max_steps = 1000

alpha = 0.7
gamma = 0.618

epsilon = 0.0
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 1/train_episodes

# TRAINING PHASE
training_rewards = []

""" DOES NOT WORK PROPERLY. JUST A TEST """

def get_finite_state(state):
    str_state = list(map(str, state.flatten()))
    return ''.join(str_state)

for episode in range(train_episodes):
    snake_pos = env.reset()
    cumulative_training_rewards = 0
    
    state = get_finite_state(env.board.around_snake(LOOKUP))

    for step in range(max_steps):

        exp_exp_tradeoff = random.uniform(0, 1)
        
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q[state])        
        else:
            action = env.action_space.sample()
        
        env.render(frame_speed=.001)
        _, reward, done, info = env.step(action)
        new_state = get_finite_state(env.board.around_snake(LOOKUP))

        # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action]) 
        cumulative_training_rewards += reward  
        state = new_state
        
        if done == True:
            # print ("Cumulative reward for episode {}: {}".format(episode, cumulative_training_rewards))
            break
        
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    training_rewards.append(cumulative_training_rewards)

print ("Training score over time: " + str(sum(training_rewards)/train_episodes))

# save_Q("q-table-v{}.json".format(VERSION+1), Q)