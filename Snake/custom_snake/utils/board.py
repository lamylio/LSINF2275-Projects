from numpy import array, zeros, where, reshape
from numpy.random import randint, choice

from custom_snake.utils.snake import Snake
""" 
@Board
- board_size=(20,20) : vector of length 2 indicating the [x,y] dimension of the board

@description
Corresponds to the game board and manages the logic allowing the game to function. 
Accesses and controls the @Snake component. Assume the matrixes start at [0,0], top left corner.

@constants
- BOARD_VALUES : dictionary representing each game element by an id. (for #board)
- BOARD_COLORS : dictionary representing each game element by an RGB array. (for #display)

@properties 
- board_size : [height, width] of the #board (and #display)
- x : height of the #board_size
- y : width of the #board_size
- board : matrix representing the game with #BOARD_VALUES
- display : matrix representing the game with #BOARD_COLORS
- snake : @Snake instance
- food : coordinates [x,y] of the food

@methods
- reset() : None : reset the game, instanciate a new @Snake and generate new food.
- step(action) : tuple : move the @Snake, compute the reward and returns to @SnakeEnv
- update_snake() : None : update the #board and #display with current @Snake#head coordinates.
- update_food(): None : random choose an empty space and update #board and #display.
- step_results() : int : compute the score/reward of the new @Snake#head coordinates.
- update_board_and_display(x, y, game_element) : None : update #board and #display at [x,y] indexes with matched value of #game_element 
- get_color(coordinates) : #BOARD_COLORS : returns the #display value at index [coordinates]
- get_type(coordinates) : #BOARD_VALUES :  returns the #board value at index [coordinates]
- off_board(coordinates) : bool : returns true if the @Snake#head is outside of the #board
- around_snake(radius) : np.array[(2*radius+1, 2*radius+1)] : returns an array with #BOARD_VALUES around the @Snake#head
- win() : bool : is the game won ? 

"""
class Board():

    BOARD_VALUES = {
        "EMPTY_SPACE": 0,
        "SNAKE_BODY":  1,
        "SNAKE_HEAD": 2,
        "FOOD": 3,
        "WALL": 4
    }

    BOARD_COLORS = {
        "EMPTY_SPACE": array([255, 255, 255], dtype="uint8"),
        "SNAKE_BODY": array([102, 204, 0], dtype="uint8"),
        "SNAKE_HEAD": array([103, 153, 0], dtype="uint8"),
        "FOOD": array([204, 51, 0], dtype="uint8")
    }

    def __init__(self, board_size):
        self.board_size = board_size
        self.x, self.y = board_size
        self.snake = None
        self.food = None
        self.board = None
        self.display = zeros((self.x, self.y, 3), dtype="uint8")

    def reset(self):
        self.board = zeros(self.board_size)
        self.display[:,:,:] = self.BOARD_COLORS.get("EMPTY_SPACE")
        self.snake = Snake((randint(self.x-3), randint(self.y-1)))
        self.update_snake()
        self.update_food()

    def step(self, action):
        self.snake.step(action)
        rewards = self.step_results()
        if rewards > 0: 
            self.update_food()
            self.snake.body.append(self.snake.old)
        next_state = self.snake.head
        done = rewards < 0 or self.win()
        info = {"snake_length": len(self.snake.body)+1}
        
        self.update_snake()
        return next_state, rewards, done, info

    def update_snake(self):
        if(self.off_board(self.snake.head)): return
        
        head_x, head_y = self.snake.head
        self.update_board_and_display(head_x, head_y, "SNAKE_HEAD")

        if self.snake.old is not None:
            old_x, old_y = self.snake.old
            self.update_board_and_display(old_x, old_y, "EMPTY_SPACE")

        for part in self.snake.body: 
            self.update_board_and_display(part[0], part[1], "SNAKE_BODY")


    def update_food(self):
        possible_x, possible_y = where(self.board[:,:] == self.BOARD_VALUES.get("EMPTY_SPACE"))
        random_index = randint(len(possible_x))
        choice_x = possible_x[random_index]
        choice_y = possible_y[random_index]
        self.food = (choice_x, choice_y)
        self.update_board_and_display(choice_x, choice_y, "FOOD")


    def step_results(self):
        # Snake is dead by walls
        if(self.off_board(self.snake.head)): 
            return -1

        # Snake is dead by eating his body
        if(self.get_type(self.snake.head) == self.BOARD_VALUES.get("SNAKE_BODY")):
            return -2

        # Food eaten
        if(self.get_type(self.snake.head) == self.BOARD_VALUES.get("FOOD")): 
            return 10

        # None
        return 0

    def win(self):
        return len(self.snake.body) >= self.x + self.y - 1

    def update_board_and_display(self, x, y, game_element):
        self.board[x, y] = self.BOARD_VALUES.get(game_element)
        self.display[x, y] = self.BOARD_COLORS.get(game_element)

    def get_color(self, coordinates):
        x, y = coordinates
        return self.display[x, y]

    def get_type(self, coordinates):
        x, y = coordinates
        return self.board[x, y]

    def off_board(self, coordinates):
        return coordinates[0]<0 or coordinates[0]>=self.x or coordinates[1]<0 or coordinates[1]>=self.y

    def around_snake(self, radius):
        arounds = []
        center_x, center_y = self.snake.head

        low_x, low_y = center_x-radius, center_y-radius
        high_x, high_y = center_x+radius, center_y+radius

        for x in range(low_x, high_x+1):
            for y in range(low_y, high_y+1):
                if(self.off_board((x, y))): arounds.append(self.BOARD_VALUES.get("WALL"))
                else: arounds.append(self.board[x, y])

        return reshape(arounds, (2*radius+1, 2*radius+1)).astype(int)