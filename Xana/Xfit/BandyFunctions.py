import numpy as np
from scipy.special import gamma, gammaincc


def V2(x, **p):
    t = p["t"]
    g = p["g"]
    return (
        2 ** ((g - 2) / g)
        * t
        / (g * x ** 2)
        * (
            2 ** (1 / g)
            * x
            * (gamma(1 / g) - gamma(1 / g) * gammaincc(1 / g, 2 * (x / t) ** g))
            + t * (gamma(2 / g) * gammaincc(2 / g, 2 * (x / t) ** g) - gamma(2 / g))
        )
    )


def V2a(x, **p):
    a = p["a"]
    b = p["b"]
    return a + b * V2(x, **p)


"""
def M(x, **p):
    return 1/bandy_V2(x, **p)

def bandy_Ma(x, **p):
    return 1/bandy_V2a(x, **p)

def bandy_logMa(x, **p):
    return log10(1/bandy_V2a(x, **p))

def bandy_logV2(x, *p):
    return log10(bandy_V2(x, *p))

def bandy_logM(x, **p):
    return log10(bandy_M(x, **p))

def bandy_Corr(x, **p):
    c = p['c']
    return bandy_M(x, **p) - 1/c

def bandy_logCorr(x, **p):
    c = p['c']
    return bandy_M(x, **p) - 1/c
"""
