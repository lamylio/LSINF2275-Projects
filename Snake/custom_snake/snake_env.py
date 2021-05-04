import numpy as np, matplotlib.pyplot as plt

from custom_snake.utils.action import Action
from custom_snake.utils.board import Board

class SnakeEnv():
    def __init__(self, board_size=[20,20]):
        self.action_space = Action(4)
        self.observation_space = int(board_size[0] * board_size[1])

        self.board_size = board_size
        self.board = Board(board_size)

        self.viewer = None
        self.fig = None
        self.episode = 0
        # print('Init successful!', self.action_space.n, self.observation_space)

    def step(self, action):
        action = (action % 4)
        return self.board.step(action)

    def reset(self):
        self.episode+=1
        self.board.reset()
        return self.board.snake.head

    def render(self, close=False, frame_speed=.1):
        if close and self.viewer is not None:
            self.viewer.clear()
            self.viewer = None
            return

        if self.viewer is None:
            plt.gcf().canvas.set_window_title('Snake')
            self.viewer= plt.subplot(111)
            self.viewer.xaxis.set_visible(False)
            self.viewer.yaxis.set_visible(False)
            plt.ion()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.board.display)
            self.viewer.title.set_text('Snake board\nEpisode {}'.format(self.episode))
            plt.pause(frame_speed)