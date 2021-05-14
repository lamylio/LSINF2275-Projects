import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from numpy import reshape, zeros, arange

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
    def render(self, epsilon=0, lookup=1, frame_speed: float = .1 ):
        lookup_size = 2*lookup+1
        if self.viewer is None:
            
            self.figure = plt.figure(num="Snake test", figsize=(8,6))

            spec = gridspec.GridSpec(1,2, width_ratios= [2,1])
            spec_sub = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=spec[1])

            self.viewer = self.figure.add_subplot(spec[0])
            self.viewer_info = self.figure.add_subplot(spec_sub[0])
            self.viewer_lookup = self.figure.add_subplot(spec_sub[1])
            self.viewer_food = self.figure.add_subplot(spec_sub[2])

            # Sub-titles
            self.viewer.set_title('Snake board')
            self.viewer_lookup.set_title('Snake vision')
            self.viewer_food.set_title('Food position')

            # Better display of snake board
            self.viewer.grid()
            self.viewer.set_xticks(arange(0,self.board_size[0],1))
            self.viewer.set_yticks(arange(0,self.board_size[1],1))
            self.viewer.set_xticklabels([])
            self.viewer.set_yticklabels([])

            # Remove the axes of the informations viewer
            self.viewer_info.xaxis.set_visible(False)
            self.viewer_info.yaxis.set_visible(False)

            # Better display of snake vision
            self.viewer_lookup.grid()
            self.viewer_lookup.set_xticks(arange(0,lookup_size,1))
            self.viewer_lookup.set_yticks(arange(0,lookup_size,1))
            self.viewer_lookup.set_xticklabels([])
            self.viewer_lookup.set_yticklabels([])

            # Better display of food position
            self.viewer_food.grid()
            self.viewer_food.set_xticks(arange(0,3,1))
            self.viewer_food.set_yticks(arange(0,3,1))
            self.viewer_food.set_xticklabels([])
            self.viewer_food.set_yticklabels([])

        else:
            self.viewer.clear()
            self.viewer_info.clear()
            self.viewer_lookup.clear()
            self.viewer_food.clear()
        
        # Snake board
        self.viewer.imshow(self.board.display, extent=[-0.01,self.board_size[0],0,self.board_size[1]])
        
        # display length
        self.viewer_info.text(0.1, 0.7, 'Score: ' + str(len(self.board.snake.body) + 1))
        # display epsilon
        self.viewer_info.text(0.1, 0.5, 'Epsilon: ' + str(round(epsilon, 4)))
        # display episodes
        self.viewer_info.text(0.1, 0.2, 'Episode: ' + str(self.episode))

        
        state = self.board.get_finite_state(lookup)
        # display around snake
        colors_map = list(self.board.BOARD_COLORS.values())
        around = reshape([colors_map[self.board.BOARD_VALUES.get("WALL")] if e == self.board.BOARD_VALUES.get("SNAKE_BODY") 
        else colors_map[e] for e in state[:-2]], (lookup_size,lookup_size,3)).astype("uint8")
        self.viewer_lookup.imshow(around, extent =[0, lookup_size, 0, lookup_size])
        
        # Apple
        ax, ay = state[-2:]
        food = zeros((3,3,3), dtype="uint8")
        food[:,:,:] = self.board.BOARD_COLORS.get("EMPTY_SPACE")

        food[1,1] = self.board.BOARD_COLORS.get("SNAKE_HEAD")
        food[1-ax, 1-ay] = self.board.BOARD_COLORS.get("FOOD")

        self.viewer_food.imshow(food, extent=[0, 3, 0,3])
        plt.pause(frame_speed)