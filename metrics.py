import numpy as np
from statsmodels.tsa.stattools import acf
def ESS(data):
    M = len(data)
    return M/(2*acf(data, nlags=40).sum() - 1)

def ESJD(data):
    N = data.shape[0]
    ESJD = (1/(N - 1)) * np.sum(np.linalg.norm(np.diff(data, axis=0),
                                               axis=1) ** 2)
    return ESJD

def min_samples_needed(data, true_value, fn, eps=1e-4):
    diff = 9999
    count = 1
    N = data.shape[0]
    while diff > eps:
        diff = np.linalg.norm(fn(data[0:count, :], axis=0) - true_value) ** 2
        count += 1
        if count == N:
            return count
    return count