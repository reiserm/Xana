import numpy as np


def betaratio(kv, kb, prob, err=None, perframe=True):
    """Calculate speckle contrast from photon probability ratios."""
    prob = np.ma.array(prob, mask=np.zeros_like(prob, dtype=np.bool))
    kb = np.ma.array(kb, mask=kb == 0)
    beta = np.ma.array(np.zeros((kv.size - 1, kb.size)))
    dbeta = beta.copy()

    for i, ki in enumerate(kv[:-1]):
        c = i + 2 < kv.size
        prob[i : i + 1 + c][prob[i + 1 : i + 2 + c] == 0] = np.ma.masked
        p1 = prob[i]
        p2 = prob[i + 1]
        a = p1 / p2
        divd = a * kb - (ki + 1.0)
        divs = 1.0 + (1.0 - a) * ki
        beta[i] = divd / kb / divs
        prob.mask = np.ma.nomask

    if err is not None:
        dbeta = 1.0 / kb * np.sqrt(2 * (1 + np.abs(beta)) / (err[0] * err[1]))
    else:
        dbeta = np.ones((kb.size, 1), dtype=np.float32)

    return beta, dbeta
