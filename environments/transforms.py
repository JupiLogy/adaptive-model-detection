"""
This program is intended to generate the optimal Predictive State Representation
of a POMDP, given the POMDP as input.
"""
import numpy as np
import numpy.linalg as alg


def pomdp_to_psr(MODEL, Q_in=None):
    """
        Returns:
        Q = Core Tests
        U = Outcome vector (prob of each test from each underlying state)
        M = Set of mao(t) vectors
    """
    if Q_in is not None:
        Q = search([[]], Q_in, MODEL)
    else:
        Q = search([[]], [[[]]], MODEL)
    U = np.transpose([test_to_prob(Q[i], MODEL) for i in range(len(Q))])
    Up = np.linalg.pinv(U)
    Up[np.abs(Up) < 1e-10] = 0

    M = np.zeros((MODEL[0].shape[0], MODEL[1].shape[1])).tolist()
    for a in range(MODEL[0].shape[0]):
        for o in range(MODEL[1].shape[1]):
            T = MODEL[0][a]
            O = np.diag((MODEL[1][:, o]))
            M[a][o] = get_m(Up, U, T, O)

    return [Q, U, M]


def linearly_independent(v, s):
    """
    Let v be a vector (represented as list), s be a list of vectors (also as lists).

    This function will return False if v is linearly dependent upon s,
    and True if v is linearly independent from s.

    >>> linearly_independent([2, 1], [[1, 1], [3, 2]])
    False

    >>> linearly_independent([2, 1], [[1, 1], [3, 3]])
    True
    """
    s = np.array(s).transpose()
    if (
        alg.norm((s @ alg.lstsq(s, np.array(v), rcond=None)[0]) - np.array(v))
        < 10 ** -8
    ):
        return False
    else:
        return True


def test_to_prob(test, MODEL):
    """
    Input test as a list of pairs of action observations.
    Output is the probability of the test conditions given the current 'state'.
    """
    state_space = MODEL[0].shape[1]
    u = np.ones(state_space)
    for state in range(state_space):
        new_states = np.zeros(state_space)
        new_states[state] = 1
        for pair in test:
            if pair:
                new_states = new_states @ MODEL[0][pair[0]]
                new_states *= MODEL[1][:, pair[1]]
        u[state] = sum(new_states)
    return u


def search(test, set, MODEL):
    action_space = MODEL[0].shape[0]
    observation_space = MODEL[1].shape[1]
    for a in range(action_space):
        for o in range(observation_space):
            if test == [[]] and linearly_independent(
                test_to_prob([[a, o]], MODEL),
                [test_to_prob(existing_test, MODEL) for existing_test in set],
            ):
                set = search([[a, o]], set + [[[a, o]]], MODEL)
            elif linearly_independent(
                test_to_prob([[a, o]] + test, MODEL),
                [test_to_prob(existing_test, MODEL) for existing_test in set],
            ):
                set = search([[a, o]] + test, set + [[[a, o]] + test], MODEL)
                # print(set)
    return set


def get_m(Up, U, T, O):
    M = np.transpose(Up @ T @ O @ U)
    M[np.abs(M) < 1e-10] = 0
    return M
