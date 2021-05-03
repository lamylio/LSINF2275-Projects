from numpy import arange
from numpy.random import randint

class Action():
    def __init__(self, nb_actions):
        self.n = nb_actions

    def sample(self):
        return randint(self.n)

    def toString(self, action):
        if action == 0:
            return "UP"
        elif action == 1:
            return "RIGHT"
        elif action == 2:
            return "DOWN"
        elif action == 4:
            return "LEFT"
        else:
            return "UKN"
        