from numpy import array, zeros, where, reshape, linalg
from numpy.random import randint, choice
from custom_snake.utils.snake import Snake

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
        self.display = zeros((self.x, self.y, 3), dtype="uint8")

    def reset(self):
        self.board = zeros(self.board_size)
        self.display[:,:,:] = self.BOARD_COLORS.get("EMPTY_SPACE")
        self.snake = Snake((randint(self.x-3), randint(self.y-1)))
        self.update_snake()
        self.update_food()

    def step(self, action):
        moved = self.snake.step(action)
        reward, reason = self.step_results()
        if reason == "FOOD": 
            self.update_food()
            self.snake.body.append(self.snake.old)
        next_state = self.snake.head
        done = reason in ["WALL", "BODY"] or self.win()
        info = {"snake_length": len(self.snake.body)+1}
        
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
        possible_x, possible_y = where(self.board[:,:] == self.BOARD_VALUES.get("EMPTY_SPACE"))
        random_index = randint(len(possible_x))

        choice_x = possible_x[random_index]
        choice_y = possible_y[random_index]

        self.food = (choice_x, choice_y)
        self.update_board_and_display(choice_x, choice_y, "FOOD")


    def step_results(self):
        # Snake is dead by walls
        if(self.off_board(self.snake.head)): 
            return -10, "WALL"

        # Has not moved
        if not self.snake.has_moved: return -5, "STUCK"

        # Snake is dead by eating his body
        if(self.get_type(self.snake.head) == self.BOARD_VALUES.get("SNAKE_BODY")):
            return -20, "BODY"

        # Food eaten
        if(self.get_type(self.snake.head) == self.BOARD_VALUES.get("FOOD")): 
            return 10, "FOOD"


        # Reward for being close to food
        return round(self.x - abs(linalg.norm(self.food-self.snake.head)))/10, "DISTANCE"

    def win(self):
        return len(self.snake.body) >= self.x + self.y - 1

    # ----------------

    def update_board_and_display(self, x, y, dict_get):
        self.board[x, y] = self.BOARD_VALUES.get(dict_get)
        self.display[x, y] = self.BOARD_COLORS.get(dict_get)

    def get_color(self, coordinates):
        x, y = coordinates
        return self.display[x, y]

    def get_type(self, coordinates):
        x, y = coordinates
        return self.board[x, y]

    def off_board(self, coordinates):
        return coordinates[0]<0 or coordinates[0]>=self.x or coordinates[1]<0 or coordinates[1]>=self.y

    def around_snake(self, radius):
        arounds = list()
        center_x, center_y = self.snake.head

        low_x, low_y = center_x-radius, center_y-radius
        high_x, high_y = center_x+radius, center_y+radius

        for x in range(low_x, high_x+1):
            for y in range(low_y, high_y+1):
                if(self.off_board((x, y))): arounds.append(self.BOARD_VALUES.get("WALL"))
                else: arounds.append(self.get_type((x,y)))

        return reshape(arounds, (2*radius+1, 2*radius+1)).astype(int)