from gym.envs.registration import register

register(id='CustomSnake-v0',
    entry_point='snake_envs.custom_snake:SnakeEnv',
    kwargs={"board_size": [20,20]}
)