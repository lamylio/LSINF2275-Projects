import matplotlib.pyplot as plt

from custom_snake.utils.board import Board

""" 
@SnakeEnv
- board_size=(20,20) : vector of length 2 indicating the [x,y] dimension of the board

@description
Working environment and entry point with the game @Board component.

@properties
- action_space : instance of @Action representing the set of possible actions. 
- observation_space : the #board_size surface
- board_size : vector of length 2 indicating the [height, width] of the #board
- viewer : @Subplot from matplotlib to draw the #board#display
- board : instance of @Board

@methods
- reset()
- step(action: int)
- render(frame_speed:int = .1)

"""
class SnakeEnv():
    def __init__(self, board_size=[20,20], snake_start_length=3):
        assert snake_start_length > 0, "Snake must be at list of size 1"
        self.action_space = [0,1,2,3]
        self.observation_space = int(board_size[0] * board_size[1]) # Observation space is the board grid

        self.board_size = board_size
        self.board = Board(board_size, snake_start_length) # Instanciate the Board class

        self.viewer = None # No render by default
        self.episode = 0

    def reset(self):
        self.episode+=1
        self.board.reset() # reset the board
        return self.board.snake.head

    """ 
    Call the underlying board.step(action) method and returns the results 
    In the form of a tuple (next_state, reward, done, info)
    """
    def step(self, action: int):
        action = (action % 4)
        return self.board.step(action)

    """ Render the current board state via a matplotlib figure """
    def render(self, frame_speed: float = .1):
        if self.viewer is None:
            plt.gcf().canvas.set_window_title('Snake')
            self.viewer= plt.subplot(111)
            self.viewer.xaxis.set_visible(False)
            self.viewer.yaxis.set_visible(False)
        else:
            self.viewer.clear()
            self.viewer.imshow(self.board.display)
            self.viewer.title.set_text('Snake board\nEpisode {}'.format(self.episode))
            plt.pause(frame_speed)