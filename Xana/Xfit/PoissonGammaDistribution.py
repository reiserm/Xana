#! /usr/bin/env python
import numpy as np
from numpy import log
from scipy.special import gamma


def PoissonGamma(x, M, p, ind_var="kb"):
    """Return Poisson-Gamma distribution as a function of k or kb. The other one is passed
    as a parameter. Depending on the argument ind_var.
    """
    if ind_var == "kb":
        kb = x
        k = p
    elif ind_var == "k":
        k = x
        kb = p
    return (
        gamma(1.0 * k + M)
        / (gamma(1.0 * M) * gamma(1.0 * k + 1.0))
        * 1.0
        * (1.0 * kb / (M * 1.0 + kb)) ** k
        * (1.0 * M / (kb + M)) ** M
    )


# def stirling(x):
#     return np.sqrt(2*np.pi/x)*(x/np.e)**x

# def PoissonGamma(x, **p):
#     k = p['k']
#     M = p['M']
#     return gamma(1.*k+M)/(gamma(1.*M)*gamma(1.*k+1.))*1.*(1.*x/(M*1.+x))**k*(1.*M/(x+M))**M

# def PoissonGamma_approx(x, **p):
#     k = p['k']
#     M = p['M']
#     return 1. -.5*log(2*np.pi*(k+M)/(M*(1+k))) + (k+M)*log((k+M)/(x+M)) + k*log(x) - (k+1)*log(1+k)

# def PoissonGamma_indk(x, **p):
#     kb = p['kb']
#     M = p['M']
#     return gamma(1.*x+M)/(gamma(1.*M)*gamma(1.*x+1.))*1.*(1.*kb/(M*1.+kb))**x*(1.*M/(kb+M))**M

# def PoissonGamma_indk_approx(x, **p):
#     kb = p['kb']
#     M = p['M']
#     return 1. -.5*log(2*np.pi*(x+M)/(M*(1+x))) + (x+M)*log((x+M)/(kb+M)) + x*log(kb) - (x+1)*log(1+x)
