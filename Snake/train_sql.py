from custom_snake.snake_env import SnakeEnv
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import random
import sqlite3
import json
import math

from utils_sql import *

# ====================================================================

def plot(scores, mean_scores, epsilons) :
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score', color="tab:blue")
    plt.tick_params(axis="y", color="tab:blue")
    plt.plot(scores, marker=".", color="tab:blue")
    plt.plot(mean_scores, color="tab:orange")

    ax2 = plt.twinx()
    ax2.plot(epsilons, color="tab:red")
    ax2.set_ylabel("Epsilon", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    plt.show(block=False)
    plt.pause(.0001)

# ====================================================================

PARAMS = {
    'LOOKUP': 2,

    'BOARD_SIZE': [10,10],
    'SNAKE_START_LENGTH': 3,

    'SAVE_EVERY_STEPS': 30,
    'PLOT_TRAINING': False,

    'EPISODES': 30000,
    'MAX_STEPS': 1000,
    
    'MIN_ALPHA': 0.05,
    'MIN_EPSILON': 0.05,
    'EPSILON_METHOD': "SMOOTH", # LINEAR, SMOOTH, SPECIAL

    'EPSILON': 1,
    'GAMMA': 0.9,
}

RESULTS = {
    'MAX_SCORE': 0,
    'SCORES': [],
    'SCORE_MEANS': [],
    'EPSILONS': []
}

TABLE = "LOOKUP_{}".format(PARAMS.get('LOOKUP'))
DB_PATH = "./resources/sql/q-values-alpha-smooth-30.db"

# ====================================================================
def update_epsilon(epsilon, episode):
    method = PARAMS.get("EPSILON_METHOD", "LINEAR")
    min_eps = PARAMS.get("MIN_EPSILON", 0.05)
    max_eps = PARAMS.get("EPISODES", 10000)
    rate = 1/(.99*max_eps)

    if method == "SMOOTH": return np.exp(-5.5*rate*episode)
    elif method == "SPECIAL": 
        r = .1*max_eps
        return (1-(1-math.sin((episode+1.5*r)/r))**.5)**2
    else: return max(min_eps, epsilon - rate)

# ====================================================================
if __name__=='__main__':
    
    try:
        DB = sqlite3.connect(DB_PATH)
        CUR = DB.cursor()
        create_table_if_not_exists(CUR, TABLE)

        print("TABLE", TABLE, "WAS LOADED AND CONTAINS", get_table_length(CUR, TABLE), "ROWS!")

        ENV = SnakeEnv(PARAMS.get("BOARD_SIZE"), PARAMS.get("SNAKE_START_LENGTH"))
        BAR = tqdm(range(1, PARAMS.get("EPISODES", 10000)+1))

        epsilon = PARAMS.get("EPSILON", 1)

        for episode in BAR:

            snake_pos = ENV.reset()
            finite_state = ENV.board.get_finite_state(PARAMS.get("LOOKUP", 1))
            state = ''.join(str(finite_state))

            for step in range(PARAMS.get("MAX_STEPS", 1000)):
                
                # Retrieve the qvalues and the alpha for the state
                *q_values, alpha = get_values_from_state(CUR, TABLE, state)
                if not any(q_values): insert_new_state(CUR, TABLE, state)

                # If U(0,1) > epsilon then use argmax, else random
                if random.uniform(0, 1) > epsilon and any(q_values):
                    action = np.argmax(q_values)
                else: action = random.choice(ENV.action_space)  

                # Call step
                snake_pos, reward, done, info = ENV.step(action)

                # Get the new state for Q(s')
                finite_state = ENV.board.get_finite_state(PARAMS.get("LOOKUP",1))
                new_state = ''.join(str(finite_state))

                # Bellman | Q(s,a):= Q(s,a) + alpha * [R(s,a) + gamma * max Q(s') - Q(s,a)]
                bellman_right = reward + PARAMS.get("GAMMA", 1) * np.max(get_values_from_state(CUR, TABLE, new_state)[:4]) - q_values[action]
                bellman_left = q_values[action] + alpha * bellman_right

                new_alpha = round(max(PARAMS.get("MIN_ALPHA", .05), alpha-PARAMS.get("ALPHA_DECREASE", .05)), 2)
                update_value_from_state(CUR, TABLE, state, action, bellman_left) # q-value change
                update_value_from_state(CUR, TABLE, state, "alpha", new_alpha) # alpha change

                # Progress bar informations display
                info.update({'max_score': RESULTS["MAX_SCORE"], 'epsilon': round(epsilon, 4), 'steps': step, 'state_alpha': alpha})
                BAR.set_description_str(str(info))
                
                up, right, down, left = map(round, q_values)
                BAR.set_postfix_str(f"↑ {up} | → {right} | ↓ {down} | ← {left}")

                # Update the state
                state = new_state

                # Update epsilon and max score
                if done:
                    epsilon = update_epsilon(epsilon, episode)
                    score = info.get('snake_length')-3
                    RESULTS["MAX_SCORE"] = max(RESULTS["MAX_SCORE"], info.get('snake_length'))

                    # Update the results, for plot
                    RESULTS["SCORES"].append(score)
                    RESULTS["SCORE_MEANS"].append(np.mean(RESULTS["SCORES"][max(0, episode-100):]))
                    RESULTS["EPSILONS"].append(epsilon)
                    if PARAMS.get("PLOT_TRAINING", False): plot(RESULTS["SCORES"], RESULTS["SCORE_MEANS"], RESULTS["EPSILONS"])
                    break

            # Save every x steps
            if step % PARAMS["SAVE_EVERY_STEPS"] == 0: save_changes(DB)                
        
        # Save each episode anyways
        save_changes(DB)
        print("\nTABLE", TABLE, "NOW CONTAINS", get_table_length(CUR, TABLE), "ROWS!")
        with open("./resources/json/train_results.json", "w") as f:
            json.dump(RESULTS, f)

    finally:
        CUR.close()
        DB.close()



