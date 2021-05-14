from random import choices
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

import numpy as np
from tensorflow.python.keras.layers.core import Dropout

class SnakeNet():

    def __init__(self, input_radius = 3, dense_dims=(48,32), batch_size=16):
        self.radius = input_radius
        self.input_dim = (input_radius*2+1, input_radius*2+1, )
        self.dense_dims = dense_dims
        self.output_dims = 4

        self.batch_size = batch_size
        self.build()

    def build(self):
        # I changed things, that doesn't work anymore
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=self.input_dim))
        self.model.add(Flatten())
        for i in range(1, len(self.dense_dims)): self.model.add(Dense(self.dense_dims[i], activation="relu"))
        self.model.add(Dense(self.output_dims, name="predictions")) # softmax ?
        self.model.compile(Adam(0.01), loss='mse')
        return self.model

    def environment_to_input(self, ENV):
        sx, sy = ENV.board.snake.head
        fx, fy = ENV.board.food
        around = ENV.board.get_around((sx, sy), self.radius)
        # state = np.append(around, int(sx-fx > 0))
        # state = np.append(state, int(sy-fy > 0))
        return np.reshape(around, self.input_dim).astype(int)