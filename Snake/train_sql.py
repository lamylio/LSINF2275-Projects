from custom_snake.snake_env import SnakeEnv
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import random
import sqlite3

from utils_sql import *

# ====================================================================

PARAMS = {
    'LOOKUP': 1,

    'BOARD_SIZE': [8,8],
    'SNAKE_START_LENGTH': 3,
    'RENDER_SPEED': 0,

    'EPISODES': 10000,
    'MAX_STEPS': 1000,
    
    'ALPHA': 1,
    'GAMMA': 1,
    'EPSILON': 1,
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
            state = ''.join(str(ENV.board.get_finite_state(PARAMS.get("LOOKUP"))))

            for step in range(PARAMS.get("MAX_STEPS", 1000)):

                q_values = get_values_from_state(CUR, TABLE, state)

                if random.uniform(0, 1) > PARAMS.get("EPSILON", 1) and any(q_values):
                    action = np.argmax(q_values)
                else: action = random.choice(ENV.action_space)  

                snake_pos, reward, done, info = ENV.step(action)
                new_state = ''.join(str(ENV.board.get_finite_state(PARAMS.get("LOOKUP"))))
                if PARAMS["RENDER_SPEED"] > 0 or episode >= PARAMS["EPISODES"]-10: ENV.render(frame_speed=PARAMS.get('RENDER_SPEED', 0.5)+0.01)


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



