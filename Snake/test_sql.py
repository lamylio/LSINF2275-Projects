import random
from custom_snake.snake_env import SnakeEnv
from tqdm import tqdm

import numpy as np
import sqlite3, json

from utils_sql import *

# ====================================================================

PARAMS = {
    'LOOKUP': 1,

    'BOARD_SIZE': [10,10],
    'SNAKE_START_LENGTH': 3,

    'RENDER_SPEED': 0.001,
    'MAX_STEPS': 1000,
    'EPISODES': 1000,

}

RESULTS = {
    'MAX_SCORE': 3,
    'SCORES': [],
    'SCORE_MEANS': [],
}

TABLE = "LOOKUP_{}".format(PARAMS.get('LOOKUP'))
DB_PATH = "./resources/sql/q-values-alpha.db"

# ====================================================================
if __name__=='__main__':
    
    try:
        # Connect to the database
        DB = sqlite3.connect(DB_PATH)
        CUR = DB.cursor()

        # Ensure the table exists, if not, create it
        create_table_if_not_exists(CUR, TABLE)
        print("TABLE", TABLE, "LOADED AND CONTAINS", get_table_length(CUR, TABLE), "ROWS!")

        # Define the environment and the progress bar
        ENV = SnakeEnv(PARAMS.get("BOARD_SIZE", [20,20]), PARAMS.get("SNAKE_START_LENGTH", 3))
        BAR = tqdm(range(1, PARAMS.get("EPISODES", 10000)+1))

        # Main loop
        for episode in BAR:
            snake_pos = ENV.reset()
            finite_state = ENV.board.get_finite_state(PARAMS.get("LOOKUP"))
            state = ''.join(str(finite_state))

            for step in range(PARAMS.get("MAX_STEPS", 5000)):

                *q_values, _ = get_values_from_state(CUR, TABLE, state)                
                up, right, down, left = map(round, q_values)
                if any(q_values): action = np.argmax(q_values)
                else: 
                    action = random.choice(ENV.action_space)
                    print("\nstate not found in Q - random action :", action)

                BAR.set_description_str(f"↑ {up} | → {right} | ↓ {down} | ← {left} || action : {action}")
                if PARAMS.get('RENDER_SPEED', 0.1) > 0: ENV.render({"state":finite_state, "lookup":PARAMS["LOOKUP"]}, frame_speed=PARAMS.get('RENDER_SPEED', 0.1))
                snake_pos, reward, done, info = ENV.step(action)

                finite_state = ENV.board.get_finite_state(PARAMS.get("LOOKUP"))
                state = ''.join(str(finite_state))

                if done == True:
                    score = info.get('snake_length')-3
                    RESULTS["MAX_SCORE"] = max(RESULTS["MAX_SCORE"], info.get('snake_length'))
                    if ENV.board.game_won(): 
                        has_won = True
                        print("WON !!!")

                    RESULTS["SCORES"].append(score)
                    RESULTS["SCORE_MEANS"].append(np.mean(RESULTS["SCORES"][max(0, episode-100):]))
                    break

        with open("./resources/json/test_results.json", "w") as f:
            json.dump(RESULTS, f)

    finally:
        CUR.close()
        DB.close()