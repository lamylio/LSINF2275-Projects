import random
from custom_snake.snake_env import SnakeEnv
from tqdm import tqdm

import numpy as np
import sqlite3

from utils_sql import *

# ====================================================================

PARAMS = {
    'LOOKUP': 1,

    'BOARD_SIZE': [10,10],
    'SNAKE_START_LENGTH': 3,

    'RENDER_SPEED': 0.05,
    'MAX_STEPS': 1000
}

RESULTS = {
    'MAX_SCORE': 3,
}

TABLE = "LOOKUP_{}".format(PARAMS.get('LOOKUP'))
DB_PATH = "./resources/sql/q-values.db"

# ====================================================================
if __name__=='__main__':
    
    try:
        DB = sqlite3.connect(DB_PATH)
        CUR = DB.cursor()
        print("TABLE", TABLE, "LOADED AND CONTAINS", get_table_length(CUR, TABLE), "ROWS!")

        ENV = SnakeEnv(PARAMS.get("BOARD_SIZE", [20,20]), PARAMS.get("SNAKE_START_LENGTH", 3))

        BAR = tqdm()
        has_won = False

        while(has_won is False):
            snake_pos = ENV.reset()
            state = ENV.board.get_finite_state(ENV.board.get_around(snake_pos,PARAMS.get("LOOKUP")), snake_pos, ENV.board.food)

            for _ in range(PARAMS.get("MAX_STEPS", 1000)):

                q_values = get_values_from_state(CUR, TABLE, state)
                up, right, down, left = map(round, q_values)
                if any(q_values): action = np.argmax(q_values)
                else: 
                    action = random.choice(ENV.action_space)
                    print("\nstate not found in Q - random action :", action)

                BAR.set_description_str(f"↑ {up} | → {right} | ↓ {down} | ← {left} || action : {action}")
                ENV.render(frame_speed=PARAMS.get('RENDER_SPEED', 0.5))
                snake_pos, reward, done, info = ENV.step(action)

                state = ENV.board.get_finite_state(ENV.board.get_around(snake_pos,PARAMS.get("LOOKUP")), snake_pos, ENV.board.food) # update state

                if done == True:
                    RESULTS.update({"MAX_SCORE": max(RESULTS.get("MAX_SCORE",3), info.get('snake_length'))})

                    if ENV.board.game_won(): has_won = True
                    BAR.update()
                    break

        print(RESULTS)

    finally:
        CUR.close()
        DB.close()

