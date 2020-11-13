import numpy as np

# 2-Markov Model with certainty of current state
ENV1 = [
    np.array([[[0, 1, 0], [0, 0, 1], [1, 0, 0]]]),
    np.array([[1, 0], [0, 1], [0, 1]]),
    np.array([1, 0, 0])
]

# inf-Markov Model with certainty of current state
ENV2 = [
    np.array([[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]]]) * 0.5,
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]),
    np.array([1, 1, 1, 1]) * 0.25
]

# 2-Markov Model with uncertainty of current state
ENV3 = [
    np.array([[[0, 0.5, 0.5], [1, 0, 0], [0, 1, 0]]]),
    np.array([[1, 0], [0, 1], [0, 1]]),
    np.array([1, 0, 0])
]

# inf-Markov Model with uncertainty of current state
ENV4 = [
    np.array([[[0.9, 0.1], [0.1, 0.9]]]),
    np.array([[0.9, 0.1], [0.1, 0.9]]),
    np.array([0.5, 0.5])
]

ENVX = [
    np.array([[[0.9, 0.1], [0.1, 0.9]]]),
    np.array([[1, 0], [0, 1]]),
    np.array([0.5, 0.5])
]

ENVY = [
    np.array([[[0.1, 0.9], [0.9, 0.1]]]),
    np.array([[1, 0], [0, 1]]),
    np.array([0.5, 0.5])
]
