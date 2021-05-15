from custom_snake.snake_env import SnakeEnv
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import random
import sqlite3

from utils_sql import *

# ====================================================================

PARAMS = {
    'LOOKUP': 2,

    'BOARD_SIZE': [10,10],
    'SNAKE_START_LENGTH': 3,
    'RENDER_SPEED': 0.1,

    'EPISODES': 50000,
    'MAX_STEPS': 5000,
    
    'ALPHA': 0.8,
    'GAMMA': 0.9,
    'EPSILON': 1.05,
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
        BAR = tqdm(range(1, PARAMS.get("EPISODES", 10000)+1))

        for episode in BAR:

            snake_pos = ENV.reset()
            finite_state = ENV.board.get_finite_state(PARAMS.get("LOOKUP"))
            state = ''.join(str(finite_state))

            for step in range(PARAMS.get("MAX_STEPS", 1000)):

                q_values = get_values_from_state(CUR, TABLE, state)

                if random.uniform(0, 1) > PARAMS.get("EPSILON", 1) and any(q_values):
                    action = np.argmax(q_values)
                else: action = random.choice(ENV.action_space)  

                snake_pos, reward, done, info = ENV.step(action)
                finite_state = ENV.board.get_finite_state(PARAMS.get("LOOKUP"))
                new_state = ''.join(str(finite_state))
                if PARAMS["RENDER_SPEED"] > 0 and episode >= PARAMS["EPISODES"]-3: ENV.render({"state":finite_state, "lookup":PARAMS["LOOKUP"]}, frame_speed=PARAMS.get('RENDER_SPEED', 0.1))



                if not any(q_values): insert_new_state(CUR, TABLE, state)
                # Bellman | Q(s,a):= Q(s,a) + alpha * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                bellman_right = reward + PARAMS.get("GAMMA") * np.max(get_values_from_state(CUR, TABLE, new_state)) - q_values[action]
                bellman_left = q_values[action] + PARAMS.get("ALPHA") * bellman_right
                update_value_from_state_and_action(CUR, TABLE, state, action, bellman_left) # push to db

                info.update({'max_score': RESULTS["MAX_SCORE"], 'epsilon': round(PARAMS["EPSILON"], 4), 'steps': step})
                up, right, down, left = map(round, q_values)
                BAR.set_postfix_str(f"↑ {up} | → {right} | ↓ {down} | ← {left}")
                BAR.set_description_str(str(info))

                state = new_state
                if done:
                    PARAMS["EPSILON"] = max(PARAMS["EPSILON"] - 1/(0.99*PARAMS["EPISODES"]), 0)
                    RESULTS.update({"MAX_SCORE": max(RESULTS.get("MAX_SCORE",3), info.get('snake_length'))})
                    break


            save_changes(DB)                
        save_changes(DB)
        print(RESULTS)
        print("\nTABLE", TABLE, "NOW CONTAINS", get_table_length(CUR, TABLE), "ROWS!")
    finally:
        CUR.close()
        DB.close()



