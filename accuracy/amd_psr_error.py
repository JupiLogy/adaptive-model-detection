from environments.transforms import pomdp_to_psr
from environments.datagen import random_traverse
import torch
import numpy as np

def amd_psr_error(env, model):
    o = env[1].shape[1]
    datastream = random_traverse(10000, env)

    # Get real parameters based on collected tests
    Qr, Ur, Mr = pomdp_to_psr(
        env
    )
    Sr = np.array([np.transpose(np.mean(Ur, axis=0))])

    # Get agent's parameters
    Sa, Ma, Aa, Xa = model.get_psr_params()
    Qa = []
    for ind in [np_in_list(t, model.norm_t)[1] for t in model.core_t]:
        Qa.append(model.norm_t[ind])
    Sa = np.array([normalise(Sa)])

    error = 0
    for datapoint in datastream:
        squaresum = 0

        # Get probabilities for next observation, calc error
        for obs in range(o):
            next_r = Sr[-1] @ Mr[datapoint[0]][obs][0]
            next_a = (
                Sa[-1]
                @ Ma[datapoint[0]][obs][0]
            )

            squaresum += (next_r - next_a) ** 2
        error += squaresum / o

        a, obs = datapoint
        ao_r = Mr[a][obs][0]
        new_r = np.zeros(Sr.shape[1])
        for ind in range(len(Qr)):
            aot_r = Mr[a][obs][ind]
            new_r[ind] = (aot_r @ Sr[-1] ) / (ao_r @ Sr[-1])

        ao_a = Ma[a][obs][0]
        new_a = np.zeros(Sa.shape[1])
        for ind, t in enumerate(Qa):
            aot_a = Ma[a][obs][ind]
            new_a[ind] = (aot_a @ Sa[-1]) / (ao_a @ Sa[-1])

        Sr = np.append(Sr, np.nan_to_num([new_r]), axis=0)

        new_a = normalise(new_a)
        Sa = np.append(Sa, np.nan_to_num([new_a]), axis=0)

    error /= 10000
    return error


def np_in_list(arr, lis):
    a = [np.array_equal(arr, x) for x in lis]
    if True in a and len(lis) > 0:
        return True, a.index(True)
    else:
        return False, 0


def normalise(arr):
    if arr.min() < 0:
        arr -= arr.min()
    bigg = max(arr)
    arr = arr/bigg
    return arr
