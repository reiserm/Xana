from numpy import exp, sqrt, log10


def g2(x, **p):
    t = p['t']
    b = p['b']
    g = p['g']
    a = p['a']
    return a + b*exp(-2.*(x*1./t)**(1.*g))
