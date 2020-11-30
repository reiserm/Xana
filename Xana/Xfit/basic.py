import numpy as np


def fitline(x, y, dy):
    A = np.vstack((np.ones_like(x), x)).T
    C = np.diag(dy * dy)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
    return (m_ls, np.sqrt(cov[1, 1])), (b_ls, np.sqrt(cov[0, 0]))


def fitline0(x, y, dy):
    m, dm = np.ma.average(y / x, weights=(dy / x) ** (-2), returned=1)
    return m, 1 / np.sqrt(dm)


def fitconstant(y, dy):
    b, db = np.ma.average(y, weights=dy ** (-2), returned=1)
    return b, 1 / np.sqrt(db)
