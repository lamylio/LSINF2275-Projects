import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from numpy import reshape

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
        self.viewer_info = None
        self.viewer_lookup = None
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
    def render(self, epsilon=0, frame_speed: float = .1 ):

        if self.viewer is None:
            plt.gcf().canvas.set_window_title('Snake')

            spec = gridspec.GridSpec(1,2, width_ratios= [2,1])
            spec_sub = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=spec[1])

            self.viewer = plt.subplot(spec[0])
            self.viewer.set_title('Snake board')
            self.viewer.xaxis.set_visible(False)
            self.viewer.yaxis.set_visible(False)

            self.viewer_info = plt.subplot(spec_sub[0])
            self.viewer_lookup = plt.subplot(spec_sub[1])
            self.viewer_apple = plt.subplot(spec_sub[2])

            self.viewer_info.xaxis.set_visible(False)
            self.viewer_info.yaxis.set_visible(False)

        else:
            self.viewer.clear()
            self.viewer_info.clear()
            self.viewer_lookup.clear()
            self.viewer_apple.clear()
            
        self.viewer_info.set_title('infos')
        self.viewer_lookup.set_title('around snake head')
        self.viewer_apple.set_title('apple position')

        # Snake board
        self.viewer.imshow(self.board.display)
        
        # display length
        self.viewer_info.text(0.1, 0.9, 'length: ' + str(len(self.board.snake.body) + 1))
        # display epsilon
        self.viewer_info.text(0.1, 0.85, 'epsilon: ' + str(round(epsilon, 4)))
        # display episodes
        self.viewer_info.text(0.1, 0.8, 'episode: ' + str(self.episode))

        # display around snake
        state = self.board.get_finite_state()
        around = reshape(state[:-2], (3,3)).astype(int)
        apple_rel = state[-2:]

        print(apple_rel)

        self.viewer_lookup.imshow(around)

        plt.pause(frame_speed)