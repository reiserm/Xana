import numpy as np


def imgkicker(et, nim, st=0.0, lt=np.inf, rt=0.001, att=0):
    t = np.arange(1, nim + 1) * (rt + et) * np.exp(-att)
    ind = (t > st) & (t < lt)
    return ind
