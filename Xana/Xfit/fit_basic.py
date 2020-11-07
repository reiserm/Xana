import numpy as np
import lmfit
from lmfit.model import Model
import corner


def get_ml_solution(res, fix):
    # find the maximum likelihood solution
    highest_prob = np.argmax(res.lnprob)
    hp_loc = np.unravel_index(highest_prob, res.lnprob.shape)
    mle_soln = res.chain[hp_loc]
    fit_report = "\nMaximum likelihood Estimation"
    fit_report += "\n-----------------------------"

    i = 0
    for par in res.params:
        if par in fix:
            fit_report += f"{par} fixed to {fix[par]}"
        else:
            # res.params[par].value = mle_soln[i]
            quantiles = np.percentile(res.flatchain[par], [2.28, 15.9, 50, 84.2, 97.7])
            res.params[par].stderr = 0.5 * (quantiles[3] - quantiles[1])
            fit_report += "\n {} = {} +/- {}".format(
                par, res.params[par].value, res.params[par].stderr
            )
            i += 1

    return res, fit_report


def residual(pars, x, func, data=None, eps=None):
    """2D Residual function to minimize"""
    model = func.eval(pars, x=x)

    if eps is not None:
        resid = (data - model) * eps
    else:
        resid = data - model
    return resid


def lnlike(pars, x, func, data=None, eps=None):
    v = pars.valuesdict()
    model = func.eval(pars, x=x)
    if eps is not None:
        inv_sigma2 = 1.0 / (1 / eps ** 2 + model ** 2 * np.exp(2 * np.log(v["f"])))
    else:
        inv_sigma2 = 1.0 / (model ** 2 * np.exp(2 * np.log(v["f"])))

    return -0.5 * (
        np.sum(
            residual(pars, x, func, data) ** 2 * inv_sigma2
            - np.log(inv_sigma2 / (2 * np.pi))
        )
    )


def init_pars(model, init, x, y):
    pars = model.make_params()
    # make initial guess for parameters
    for vn in pars:
        if vn not in init:
            if model.name == "Model(linear)":
                if vn == "m":
                    init[vn] = (np.nanmean(np.diff(y) / np.diff(x)), None, None)
                elif vn == "b":
                    init[vn] = (y[0], None, None)
            if model.name == "Model(power)":
                if vn == "a":
                    init[vn] = (y[0], None, None)
                elif vn == "n":
                    init[vn] = (1, None, None)
                elif vn == "b":
                    init[vn] = (y[0], None, None)
            if model.name == "Model(quadratic)":
                if vn == "a":
                    init[vn] = (np.nanmean(np.diff(y) / (np.diff(x) ** 2)), None, None)
                elif vn == "b":
                    init[vn] = (np.nanmean(y), None, None)
                elif vn == "c":
                    init[vn] = (y[0], None, None)
            if model.name == "Model(exponential)":
                if vn == "a":
                    init[vn] = (y[0], None, None)
                elif vn == "t":
                    init[vn] = (x[len(x) // 2], 0, None)
                elif vn == "b":
                    init[vn] = (0, None, None)
                elif vn == "g":
                    init[vn] = (1, None, None)
        p = init[vn]
        pars[vn].set(p[0], min=p[1], max=p[2])
    return pars


# Defined Models: straight line, power law, quadratic, cubic, exponential
def linear(x, m, b):
    return m * x + b


def power(x, a, n, b):
    return a * x ** n + b


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def exponential(x, a, t, b, g):
    return a * np.exp(-((x / t) ** g)) + b


# Main Fit Function
def fit_basic(
    x,
    y,
    dy=None,
    model="line",
    init={},
    fix=None,
    method="leastsq",
    emcee=False,
    plot_corner=False,
    **kwargs,
):

    if model in "linear":
        func = linear
    elif model in "power":
        func = power
    elif model in "quadratic":
        func = quadratic
    elif model in "exponential":
        func = exponential
    else:
        raise ValueError("Model {} not defined.".format(model))

    model = Model(func, nan_policy="omit")

    pars = init_pars(model, init, x, y)

    if fix is not None:
        for vn in fix:
            pars[vn].set(value=fix[vn], vary=0)

    if dy is not None:
        dy = np.abs(dy)
        wgt = np.array([1.0 / dy[i] if dy[i] > 0 else 0 for i in range(len(y))])
        is_weighted = True
    else:
        wgt = None
        is_weighted = False

    if emcee:
        mi = lmfit.minimize(
            residual,
            pars,
            args=(x, model),
            kws={"data": y, "eps": wgt},
            method="nelder",
            nan_policy="omit",
        )
        # mi.params.add('f', value=1, min=0.001, max=2)
        mini = lmfit.Minimizer(
            residual, mi.params, fcn_args=(x, model), fcn_kws={"data": y, "eps": wgt}
        )
        out = mini.emcee(
            burn=300, steps=1000, thin=20, params=mi.params, is_weighted=is_weighted
        )
        out, fit_report = get_ml_solution(out, fix)
        print(list(out.params.valuesdict().values()))
        if plot_corner:
            corner.corner(
                out.flatchain,
                labels=out.var_names,
                truths=list(out.params.valuesdict().values()),
            )

    else:
        out = lmfit.minimize(
            residual,
            pars,
            args=(x, model),
            method=method,
            kws={"data": y, "eps": wgt},
            nan_policy="omit",
            **kwargs,
        )
        fit_report = lmfit.fit_report(out)

    pars_arr = np.zeros((len(pars), 2))
    for i, vn in enumerate(pars):
        pars_arr[i, 0] = out.params[vn].value
        pars_arr[i, 1] = out.params[vn].stderr if out.params[vn].stderr else 0

    if not emcee:
        gof = np.array([out.chisqr, out.redchi, out.bic, out.aic])
    else:
        gof = 0
    return pars_arr, gof, out, fit_report, model
