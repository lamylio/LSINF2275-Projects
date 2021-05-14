from collections import deque
from custom_snake.snake_env import SnakeEnv
from model import SnakeNet

import numpy as np
from numpy import append, argmax

from random import uniform, choice, sample
from tqdm import tqdm

from collections import deque

# ====================================================================

def get_finite_state(around, snake_pos, food_pos):
    sx, sy = snake_pos
    fx, fy = food_pos
    state = append(around, int(sx-fx > 0))
    state = append(state, int(sy-fy > 0))
    return state

# ====================================================================

PARAMS = {
    'LOOKUP': 5,
    'DENSE_DIMS': [64],

    'BOARD_SIZE': [16,16],
    'SNAKE_START_LENGTH': 3,

    'EPISODES': 100_000,
    'SAVE_EVERY': 1000,
    'MAX_STEPS': 1000,
    'BATCH_SIZE': 16,
    'EPOCHS': 10,

    'ALPHA': 0.8,
    'GAMMA': 0.67,
    'EPSILON': 1,
    'EPSILON_MIN': 0.01,
    'DECAY_RATE': 1e-6,

    'RENDER_SPEED': 0.01
}

RESULTS = {
    'MAX_SCORE': 3,
    'LOSS': 0
}

# ====================================================================

ENV = SnakeEnv(PARAMS["BOARD_SIZE"], PARAMS["SNAKE_START_LENGTH"])
AGENT = SnakeNet(PARAMS["LOOKUP"], PARAMS["DENSE_DIMS"], PARAMS["BATCH_SIZE"])
AGENT.model.summary()

BAR = tqdm(range(1, PARAMS.get("EPISODES", 10000)+1))
MEMORY = deque()

# ====================================================================

for episode in BAR:

    snake_pos = ENV.reset()
    state = AGENT.environment_to_input(ENV)
    for step in range(PARAMS["MAX_STEPS"]):

        if uniform(0,1) > PARAMS["EPSILON"]:
            action = argmax(AGENT.model.predict_on_batch(state))
        else:
            action = choice(ENV.action_space)

        snake_pos, reward, done, info = ENV.step(action)
        new_state = AGENT.environment_to_input(ENV)
        if PARAMS["RENDER_SPEED"] > 0: ENV.render(PARAMS["RENDER_SPEED"])

        experience = state, action, reward, new_state
        MEMORY.append(experience)

        if step >= PARAMS["BATCH_SIZE"]:
            X, Y = [], []
            for s, a, r, ns in sample(list(MEMORY), PARAMS["BATCH_SIZE"]):
                print(s)
                qv = AGENT.model.predict(s).flatten()
                print(qv)
                if r <= -5: qv[a] = -5
                else: qv[a] = r + PARAMS["GAMMA"] * max(AGENT.model.predict_on_batch(ns))

                X.append(s)
                Y.append(qv)
            RESULTS["LOSS"] += AGENT.model.train_on_batch(np.array(X), np.array(Y))
        
        if done:
            PARAMS["EPSILON"] -= 1/(100+PARAMS["EPISODES"]) 
            max_results = max(RESULTS.get("MAX_SCORE",3), info.get('snake_length'))
            RESULTS.update({"MAX_SCORE": max_results})
            info.update({'max_score': max_results, 'epsilon': round(PARAMS["EPSILON"], 4), 'loss': RESULTS["LOSS"], 'steps': step})
            BAR.set_description_str(str(info))
            break;

        state = new_state

    if (episode) % (PARAMS.get("SAVE_EVERY", 1000)) == 0: 
        AGENT.model.save_weights("weights-{}-{}-{}--{}-{}.h5".format(PARAMS['BOARD_SIZE'], PARAMS['LOOKUP'], PARAMS['DENSE_DIMS'], PARAMS['ALPHA'], PARAMS['GAMMA']))
        print(f"\nSAVED CHANGES AT EPISODE {episode} !")
