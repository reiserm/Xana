import numpy as np


def map2ct(mat):
    ct = np.zeros_like(mat) * np.nan
    for i in range(1, mat.shape[1]):
        ct[:-i, i] = mat[i:, i]
    return ct


def map2g2(mat):
    mat = np.rot90(mat, 1)
    n2 = mat.shape[1]
    for i in range(-n2, n2):
        if i == -n2:
            ct = np.ones((int(n2 / 2), 2 * n2)) * np.nan
        tmp = np.diag(mat, i)
        tmp = tmp[int(tmp.size / 2) :]
        ct[: tmp.size, i + n2] = tmp
    return ct
