
from numpy.random import randint
import numpy as np

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
- snake_start_length : int

@methods
- reset() : None : reset the game, instanciate a new @Snake and generate new food.
- step(action) : tuple : move the @Snake, compute the reward and returns to @SnakeEnv
- update_snake() : None : update the #board and #display with current @Snake#head coordinates.
- update_food(): None : random choose an empty space and update #board and #display.
- step_results() : tuple(int,string) : compute the score/reward of the new @Snake#head coordinates. Also gives a reason.
- update_board_and_display(x, y, game_element) : None : update #board and #display at [x,y] indexes with matched value of #game_element 
- get_color(coordinates) : #BOARD_COLORS : returns the #display value at index [coordinates]
- get_type(coordinates) : #BOARD_VALUES :  returns the #board value at index [coordinates]
- off_board(coordinates) : bool : returns true if the @Snake#head is outside of the #board
- game_won() : bool : is the game won ? 
- get_around(coordinates, radius) : np.array[(2*radius+1, 2*radius+1)] : returns an array with #BOARD_VALUES around the coordinates
- get_finite_state(radius) : call #get_around(self.snake.head, radius) and returns it flattened plus food relative position  
"""
class Board():

    BOARD_VALUES = {
        "EMPTY_SPACE": 0,
        "SNAKE_BODY":  1,
        "SNAKE_HEAD": 2,
        "FOOD": 3,
    }

    BOARD_COLORS = {
        "EMPTY_SPACE": [255, 255, 255],
        "SNAKE_BODY": [102, 204, 0],
        "SNAKE_HEAD": [103, 153, 0],
        "FOOD": [204, 51, 0],
        "WALL": [123,78,63]
    }

    def __init__(self, board_size, snake_start_length):
        self.board_size = board_size
        self.x, self.y = board_size
        self.snake = None
        self.food = None
        self.board = None
        self.display = np.zeros((self.x, self.y, 3), dtype="uint8")
        self.snake_start_length = snake_start_length

    def reset(self):
        self.board = np.zeros(self.board_size)
        self.display[:,:,:] = self.BOARD_COLORS.get("EMPTY_SPACE")
        self.snake = Snake((randint(self.x-self.snake_start_length), randint(self.y-1)), self.snake_start_length)
        self.update_snake()
        self.update_food()

    def step(self, action):
        self.snake.step(action)

        reward, reason = self.step_results()
        if reason == "FOOD": 
            self.update_food()
            self.snake.body.append(self.snake.old)

        next_state = self.snake.head
        # Reset the game if dead. Except if stuck, punish 10 times then reset.
        done = reason in ["WALL", "BODY"] or self.game_won() or self.snake.blocked > 10
        info = {"reward_type": reason, "snake_length": len(self.snake.body)+1}
        
        self.update_snake()
        return next_state, reward, done, info

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
        possible_x, possible_y = np.where(self.board[:,:] == self.BOARD_VALUES.get("EMPTY_SPACE"))
        random_index = randint(len(possible_x))
        choice_x = possible_x[random_index]
        choice_y = possible_y[random_index]
        self.food = (choice_x, choice_y)
        self.update_board_and_display(choice_x, choice_y, "FOOD")


    def step_results(self):
        type_at_head = self.get_type(self.snake.head)
        # Snake is dead by wall
        if(self.off_board(self.snake.head)): 
            return -5, "WALL"

        # Has not moved (reversed on himself)
        if not self.snake.has_moved: return -5, "STUCK"

        # Snake is dead by eating his body
        if(type_at_head == self.BOARD_VALUES.get("SNAKE_BODY")):
            return -6, "BODY"

        # Food eaten
        if(type_at_head == self.BOARD_VALUES.get("FOOD")): 
            return 10, "FOOD"

        distance = abs(np.linalg.norm(np.subtract(self.snake.head, self.food)))
        distance_previous = abs(np.linalg.norm(np.subtract(self.snake.previous, self.food)))
        
        # Snake is now closer to the food
        if distance < distance_previous: return 1, "CLOSER"

        return -1.5, "NONE"

    def game_won(self):
        return len(self.snake.body) >= self.x * self.y - 1

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

    def get_around(self, coordinates, radius):
        arounds = []
        center_x, center_y = coordinates

        low_x, low_y = center_x-radius, center_y-radius
        high_x, high_y = center_x+radius, center_y+radius

        for x in range(low_x, high_x+1):
            for y in range(low_y, high_y+1):
                if(self.off_board((x, y))): arounds.append(self.BOARD_VALUES.get("SNAKE_BODY"))
                else: arounds.append(self.board[x, y])

        return np.array(arounds).astype(int)

    def get_finite_state(self, radius=1):
        around = self.get_around(self.snake.head, radius)

        sx, sy = self.snake.head
        fx, fy = self.food

        rx, ry = 0, 0
        dx, dy = sx-fx, sy-fy
        
        if dx < 0 : rx = -1
        elif dx > 0 : rx = 1
        
        if dy < 0 : ry = -1
        elif dy > 0: ry = 1

        return [*around, rx, ry]