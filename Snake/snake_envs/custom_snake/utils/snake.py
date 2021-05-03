import numpy as np

class Snake():

    UP, RIGHT, DOWN, LEFT = range(4)

    def __init__(self, head_coordinates=(0,0), length=3):
        self.head = head_coordinates
        self.body = list()
        for i in range(1, length):
            self.body.append((self.head[0]+i, self.head[1]))

        self.old = None
        # print("Snake init", self.head, self.body)

    def step(self, action):
        m = self.movement(action)
        self.old = self.body[-1]
        
        previous = self.head
        self.head = np.add(self.head, m)
        for i,part in enumerate(self.body):
            self.body[i], previous = previous, part
            
        # print("M:", m, "Old:", self.old, "Head:", self.head)
        return self.head

    def movement(self, action):
        if action == self.UP: return (-1,0)
        elif action == self.RIGHT: return (0,1)
        elif action == self.DOWN: return (1, 0)
        elif action == self.LEFT: return (0,-1)
        else: return (0,0)