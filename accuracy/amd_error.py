import torch
import numpy as np

def amd_error(Sr, Mr, Qr, y_pred, a, o, dtype=torch.float):
    ao_r = Mr[int(a)][int(o)][0]
    new_r = np.zeros(Sr.shape[1])
    for ind in range(len(Qr)):
        aot_r = Mr[int(a)][int(o)][ind]
        if ao_r @ Sr[-1] < 1e-5:
            # Need to reset somehow if environment is switched
            new_r[ind] = 1e-2
        else:
            new_r[ind] = (aot_r @ Sr[-1]) / (ao_r @ Sr[-1])
    Sr = np.append(Sr, np.nan_to_num([new_r]), axis=0)
    err = 0
    next_r = torch.tensor([0, 0], dtype=dtype)
    for obs in range(2):
        next_r[obs] = Sr[-1] @ Mr[0][obs][0]
        err += (next_r[obs] - y_pred[obs]) ** 2
    return err/2, Sr
