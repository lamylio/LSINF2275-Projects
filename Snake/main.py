import gym
import snake_envs

env = gym.make('CustomSnake-v0', board_size=[20,20])

while(1): # DEBUG
    env.render()