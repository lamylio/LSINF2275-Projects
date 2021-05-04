from numpy.random import randint

""" 
@Action
- nb_actions : int : number of possible actions to define

@description
Helper class to randomly pick an action from the actions set.
Kind of not necessary, but follows the package "gym" practices.

@properties
- n : the number of possible actions

@methods
- sample() : returns numpy.random.randint(#n)
- toString(action) : returns a String correspondance of the action 
"""
class Action():
    def __init__(self, nb_actions: int = 4):
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
        