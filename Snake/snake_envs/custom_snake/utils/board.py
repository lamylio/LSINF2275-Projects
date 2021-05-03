import numpy as np
class Board():

    BOARD_VALUES = {
        "EMPTY_SPACE": 0,
        "SNAKE_BODY":  1,
        "SNAKE_HEAD": 2,
        "FOOD": 3
    }

    BOARD_COLORS = {
        "EMPTY_SPACE": np.array([255, 255, 255], dtype=np.uint8),
        "SNAKE_BODY": np.array([102, 204, 0], dtype=np.uint8),
        "SNAKE_HEAD": np.array([103, 153, 0], dtype=np.uint8),
        "FOOD": np.array([204, 51, 0], dtype=np.uint8)
    }

    def __init__(self, board_size, size_factor):
        self.board_size = board_size
        self.size_factor = size_factor

        self.board = np.zeros(board_size)
        self.x, self.y = board_size
        self.display_board = np.zeros((self.x, self.y, 3), dtype=np.uint8)
        self.display_board[:,:,:] = self.BOARD_COLORS.get("EMPTY_SPACE")
        self.display_board[2,2] = self.BOARD_COLORS.get("SNAKE_HEAD")
        self.display_board[2,3] = self.BOARD_COLORS.get("SNAKE_BODY")

    def get_color(self, coordinates):
        x, y = coordinates
        return self.display_board[x, y,:]


    def off_board(self, coordinates):
        return \
            coordinates[0]<0 or coordinates[0]>=self.x or \
            coordinates[1]<0 or coordinates[1]>=self.y
