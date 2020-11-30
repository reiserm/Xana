#! /usr/bin/env python
import numpy as np
import lmfit
from Xfit.PoissonGammaDistribution import PoissonGamma as poisgam


def rescale(y, rng=(0, 1)):
    return ((rng[1] - rng[0]) * (y - min(y))) / (max(y) - min(y)) + rng[0]


def correct_probability(prob, kappa, method="shift p2"):
    """Correct probability"""
    if method == "shift p2":
        p = prob[2:]
        c = p[2] * kappa
        p[2] -= c
        p[1] += 2 * c
        p[0] -= c
    elif method == "kb fraction":
        kb = prob[1] * kappa
        p = prob[2:]
        p[2] -= c
        p[2][p[2] <= 0] = 0
        p[1] += 2 * c
        p[0] -= c
    return prob


def detector_correction(
    prob,
    prob_ref,
    npix,
    err=None,
    kv=None,
    init={},
    fix=None,
    method="Nelder-mead",
    mspacing=200,
    correction_method="shift p2",
):
    """Fit the Poisson-Gamma distribution using the likelihood ratio approach."""
    if kv is None:
        kv = np.arange(prob.shape[0] - 2)

    # make initial guess for parameters
    for vn in ["kappa"]:
        if vn not in init.keys():
            if vn == "kappa":
                init[vn] = (0, 0, 1)

    # initialize fit parameters
    pars = lmfit.Parameters()
    pars.add(
        "kappa", value=init["kappa"][0], min=init["kappa"][1], max=init["kappa"][2]
    )

    if fix is not None:
        for vn in fix.keys():
            pars[vn].set(value=fix[vn], vary=0)

    if err is not None:
        err = np.abs(err)
        wgt = err.copy()
        wgt[wgt > 0] = 1.0 / wgt[wgt > 0] ** 2
    else:
        wgt = None

    M = np.logspace(0, 2, mspacing)

    def chi2(prob):
        """Calculate likelihood ratio"""
        kb = prob[1]
        prob = prob[kv + 2]
        chi2 = np.zeros((mspacing))
        for j, m in enumerate(M):
            for i in range(kv.size):
                probi = prob[i]
                ind = np.where(probi)
                pg = poisgam(kb[ind], m, kv[i], ind_var="kb")
                chi2[j] += np.sum(probi[ind] * np.log(pg / probi[ind]))
            chi2[j] *= -2 * npix
        return rescale(chi2, (0, 1))

    chi2_ref = chi2(prob_ref)

    def residual(pars, prob, eps=None):
        """Residual function to minimize"""
        prob = prob.copy()
        v = pars.valuesdict()

        prob = correct_probability(prob, v["kappa"], correction_method)
        return np.sum(np.abs(chi2_ref - chi2(prob)))

    out = lmfit.minimize(
        residual, pars, args=(prob,), kws={"eps": wgt}, method=method, nan_policy="omit"
    )

    pars_arr = np.zeros((1, 2))
    for i, vn in enumerate(["kappa"]):
        pars_arr[i, 0] = out.params[vn].value
        # pars_arr[i,1] = pars_arr[i,0]**2*out.params[vn].stderr
    gof = np.array([out.chisqr, out.redchi, out.bic, out.aic])

    return pars_arr, gof, out, lmfit.fit_report(out)
