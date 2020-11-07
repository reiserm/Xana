#! /usr/bin/env python
import numpy as np
import lmfit
from .PoissonGammaDistribution import PoissonGamma as poisgam
from ..Xplot.PlotPoissonGamma import plot_poissongamma


def fit_pg(
    prob,
    err=None,
    ind_var="kb",
    krange=None,
    init={},
    fix=None,
    logscale=False,
    qv=None,
    modes=1,
    pbthres=0,
    vary_par=False,
    method="leastsq",
    doplot=False,
):
    """Fit the Poisson-Gamma distribution
    by minimizing the residuals"""

    if krange is None:
        krange = np.arange(prob.shape[0] - 2)

    kv = prob[krange + 2, 0]
    kb = prob[1]
    prob = prob[krange + 2]

    if ind_var == "kb":
        x = kb
        p = kv
        pn = "k"
    elif ind_var == "k":
        x = kv
        p = kb
        pn = "kb"

    # make initial guess for parameters
    for vn in ["M", "kb"]:
        if vn not in init.keys():
            if vn == "M":
                init[vn] = (1, 0, None)
            elif vn == "kb":
                init[vn] = (kb.mean(), 0, None)

    # initialize fit parameters
    pars = lmfit.Parameters()
    for i in range(p.size):
        pars.add(
            "M{}".format(i), value=init["M"][0], min=init["M"][1], max=init["M"][2]
        )
        if i > 0:
            pars["M{}".format(i)].expr = "M0"
        pars.add(pn + "{}".format(i), value=p[i], vary=False)

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
    def residuals(pars, x, data=None, eps=None):
        """2D Residual function to minimize"""
        v = pars.valuesdict()

        residuals = np.zeros((p.size, x.size))

        for i, pi in enumerate(p):
            if logscale:
                model = np.log10(poisgam(x, v["M{}".format(i)], pi, ind_var))
            elif not logscale:
                model = poisgam(x, v["M{}".format(i)], pi, ind_var)
            if eps is not None:
                residuals[i] = np.abs(data[i] - model) * np.abs(eps[i])
            else:
                residuals[i] = np.abs(data[i] - model)
        return residuals.flatten()

    out = lmfit.minimize(
        residuals,
        pars,
        args=(x,),
        kws={"data": prob, "eps": wgt},
        method=method,
        nan_policy="omit",
    )

    pars_arr = np.zeros((1, 2))
    for i, vn in enumerate(["M0"]):
        pars_arr[i, 0] = out.params[vn].value
        pars_arr[i, 1] = 1.0 * out.params[vn].stderr
    gof = np.array([out.chisqr, out.redchi, out.bic, out.aic])

    if doplot:
        plot_poissongamma(x, prob, p, pars_arr, ind_var=ind_var)

    return pars_arr, gof, out, lmfit.fit_report(out)
