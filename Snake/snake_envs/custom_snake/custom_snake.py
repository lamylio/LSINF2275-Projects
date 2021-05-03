import gym, numpy as np, matplotlib.pyplot as plt

from snake_envs.custom_snake.utils.action import Action
from snake_envs.custom_snake.utils.board import Board

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_size=[20,20], size_factor=1):
        self.action_space = Action(4)
        self.observation_space = int((board_size[0] * board_size[1]) / (size_factor*size_factor))

        self.board_size = board_size
        self.size_factor = size_factor
        self.board = Board(board_size, size_factor)

        self.viewer = None
        self.fig = None
        # print('Init successful!', self.action_space.n, self.observation_space)

    def step(self, action):
        action = (action % 4)
        return self.board.step(action)

    def reset(self):
        self.board.reset()
        return self.board.snake.head

    def render(self, mode='human', close=False, frame_speed=.1):
        if close and self.viewer is not None:
            plt.close(self.fig)
            self.viewer.clear()
            self.fig = None
            self.viewer = None
            return

        if self.viewer is None:
            plt.suptitle('Snake board')
            plt.gcf().canvas.set_window_title('Snake played by a Q-Value Reinforcement Learning Algorithm')
            self.viewer= plt.subplot(111)
            self.viewer.set_title("Snake")
            self.viewer.xaxis.set_visible(False)
            self.viewer.yaxis.set_visible(False)
            plt.ioff()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.board.display)
            plt.pause(frame_speed)