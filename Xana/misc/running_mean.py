import numpy as np


def running_mean(x):
    """
    Calculate running average
    Initialize M1 = x1 and S1 = 0.

    For subsequent x‘s, use the recurrence formulas

    Mk = Mk-1+ (xk – Mk-1)/k
    Sk = Sk-1+ (xk – Mk-1)*(xk – Mk).

    For 2 ≤ k ≤ n, the kth estimate of the variance is s2 = Sk/(k – 1).
    """
    out = np.zeros((x.size, 3))
    for i in range(x.size):
        if i == 0:
            out[i, :] = (x[i], 0, 0)
        else:
            Mi = out[i - 1, 0] + (x[i] - out[i - 1, 0]) / (i + 1)
            Si = out[i - 1, 2] + (x[i] - out[i - 1, 0]) * (x[i] - out[i, 0])
            out[i, :] = (Mi, np.sqrt(np.abs(Si) / i ** 2), Si)
    return out[:, :2]


def runavr(x):
    out = np.zeros((x.size - 1, 3))
    for i in range(x.size - 1):
        if i == 0:
            out[i, :] = (x[i], 0, 0)
        else:
            Mi = out[i - 1, 0] + (x[i] - out[i - 1, 0]) / (i + 1)
            Si = out[i - 1, 2] + (x[i] - out[i - 1, 0]) * (x[i] - out[i, 0])
            out[i, :] = (Mi, np.sqrt(Si / i ** 2), Si)
    return out
