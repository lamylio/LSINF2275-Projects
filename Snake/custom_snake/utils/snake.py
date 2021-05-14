from numpy import add

""" 
@Snake
- head_coordinates=(0,0) : [x,y] coordinates of the head at spawn
- length=3 : the length of the snake at spawn

@description
Class that stores and manages the coordinates of each part (head and body) of the snake component. 
Has no logic and only executes the instructions, except for preventing the snake to turn back on itself. 

@properties
- head : int[x,y] : coordinates of the snake's head
- body : list[int[x,y]] : list of coordinates for each body part of the snake
- old : int[x,y] : the tail of the snake to be removed from the @Board#display after #step()
- previous : int[x,y] : coordinates of the first body part
- has_moved : bool : has the snake moved after #step() ? 

@constants
- MOVEMENTS : dict : map the [x,y] coordinates change for each action

@methods
- step(action) : int[x,y] : move the #head then each piece of #body by #MOVEMENTS[action]. 
- is_equal(coordinates_1, coordinates_2) : bool : compares two [x,y] coordinates

"""
class Snake():

    MOVEMENTS = {
        0: (-1,0), # UP
        1: (0,1),  # RIGHT
        2: (1,0),  # DOWN
        3: (0,-1), # LEFT
    }

    def __init__(self, head_coordinates=(0,0), snake_start_length=3):
        self.head = head_coordinates
        self.body = list()
        for i in range(1, snake_start_length):
            self.body.append([self.head[0]+i, self.head[1]])

        self.old = None
        self.previous = self.body[0]
        self.has_moved = False
        self.blocked = 0

    def step(self, action):
        m = self.MOVEMENTS.get(action, (0,0))
        self.old = self.body[-1]
        
        previous = self.head
        next_head = add(self.head, m)
        if self.is_equal(next_head, self.previous): # cancel if eat himself instant
            self.has_moved = False 
            self.blocked += 1
            return False
        else: 
            self.blocked = 0
            self.previous, self.head = previous, next_head
        
        for i,part in enumerate(self.body):
            self.body[i], previous = previous, part

        self.has_moved = True
        return True

    def is_equal(self, coordinates_1, coordinates_2):
        for i in range(len(coordinates_1)):
            if coordinates_1[i] != coordinates_2[i]: return False
        return True