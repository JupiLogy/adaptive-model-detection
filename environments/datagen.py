import numpy as np

def random_traverse(timesteps, env):
    action_space = env[0].shape[0]
    state_space = env[0].shape[1]
    observation_space = env[1].shape[1]
    data = np.zeros((timesteps, 2), dtype=np.uint8)
    state = np.random.choice(range(state_space), p=env[2])
    for t in range(timesteps):
        action = np.random.choice(range(action_space))
        state = np.random.choice(range(state_space), p=env[0][action, state, :])
        observation = np.random.choice(range(observation_space), p=env[1][state, :])
        data[t, 0] = action
        data[t, 1] = observation
    return data
