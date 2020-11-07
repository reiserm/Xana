#! /usr/bin/env python
import numpy as np
import lmfit
from .PoissonGammaDistribution import PoissonGamma as poisgam


def fit_pg_likelihood(
    prob, npix, err=None, kv=None, init={}, fix=None, method="Nelder-mead"
):
    """Fit the Poisson-Gamma distribution using the likelihood ratio approach."""
    if kv is None:
        kv = np.arange(prob.shape[0] - 2)

    kb = prob[1]
    prob = prob[kv + 2]

    # make initial guess for parameters
    for vn in ["M"]:
        if vn not in init.keys():
            if vn == "M":
                init[vn] = (4, 1, None)

    # initialize fit parameters
    pars = lmfit.Parameters()
    pars.add("M", value=init["M"][0], min=init["M"][1], max=init["M"][2])

    if fix is not None:
        for vn in fix.keys():
            pars[vn].set(value=fix[vn], vary=0)

    if err is not None:
        err = np.abs(err)
        wgt = err.copy()
        wgt[wgt > 0] = 1.0 / wgt[wgt > 0] ** 2
    else:
        wgt = None

    # independent variable is <k>
    def likelihood_ratio(pars, prob, eps=None):
        """2D Residual function to minimize"""
        v = pars.valuesdict()

        chi2 = 0
        for i in range(kv.size):
            probi = prob[i]
            ind = np.where(probi)[0]
            pg = poisgam(kb[ind], v["M"], kv[i], ind_var="kb")
            chi2 += np.sum(probi[ind] * np.log(pg / probi[ind]))
        chi2 *= -2 * npix
        return chi2

    out = lmfit.minimize(
        likelihood_ratio,
        pars,
        args=(prob,),
        kws={"eps": wgt},
        method=method,
        nan_policy="omit",
    )

    pars_arr = np.zeros((1, 2))
    for i, vn in enumerate(["M"]):
        pars_arr[i, 0] = 1.0 / out.params[vn].value
        # pars_arr[i,1] = pars_arr[i,0]**2*out.params[vn].stderr
    gof = np.array([out.chisqr, out.redchi, out.bic, out.aic])

    return pars_arr, gof, out, lmfit.fit_report(out)
