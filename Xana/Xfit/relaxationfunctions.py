import numpy as np
from numpy import exp, sqrt, log10


def g2(x, **p):
    t = p["t"]
    b = p["b"]
    g = p["g"]
    a = p["a"]
    return a + b * exp(-2.0 * (x * 1.0 / t) ** (1.0 * g))

def logarithmic(x, **p):
    t = p["t"]
    b = p["b"]
    g = p["g"]
    a = p["a"]
    return a - b * np.log((x/t)**g)

def algebraic(x, **p):
    t = p["t"]
    b = p["b"]
    g = p["g"]
    a = p["a"]
    return a + b * (x/t)**(-g)

