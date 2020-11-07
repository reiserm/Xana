import numpy as np
from numpy import exp, sqrt, pi
import lmfit


def gaussian(x, **p):
    gauss = 0
    for i in range(p["nm"]):
        mst = str(i)
        gauss += (p["a" + mst] / (sqrt(2 * pi) * p["sig" + mst])) * exp(
            -((x - p["cen" + mst]) ** 2) / (2 * p["sig" + mst] ** 2)
        )
    gauss += +p["bg"]
    return gauss


def fitgaussian(
    x,
    y,
    err=None,
    mode="standard",
    nmodes=1,
    start=None,
    lb=None,
    ub=None,
    doplot=0,
    h_plot=None,
    ax=None,
    output="pars",
    xl=None,
    ylim=None,
    color=None,
):
    """ Fit data with Gaussian peak."""

    # initialize parameters
    pars = lmfit.Parameters()
    for i in range(nmodes):
        mst = str(i)
        if i > 0:
            start = np.array(start) * 1.1
        pars.add("cen" + mst, value=start[0], min=lb[0], max=ub[0], vary=1)
        pars.add("sig" + mst, value=start[1], min=lb[1], max=ub[1], vary=1)
        pars.add("a" + mst, value=start[2], min=lb[2], max=ub[2], vary=1)
    pars.add("bg", value=1.0, vary=0)
    pars.add("nm", value=nmodes, vary=0)

    if "bg" in mode:
        if len(start) == 3:
            pars["bg"].set(vary=1)
        elif len(start) == 4:
            pars["bg"].set(value=start[3], min=lb[3], max=ub[3], vary=1)
    if err is not None:
        wgt = 1.0 / err ** 2
    else:
        wgt = np.ones_like(y)

    if "logx" in mode:
        wgt = 1 / np.log10(x)

    mod = lmfit.Model(gaussian)
    out = mod.fit(y, pars, x=x, weights=wgt)

    # plot results
    if doplot:
        if xl is None:
            if ax is None:
                ax = plt.gca()
            xl = ax.get_xlim()
            if xl[0] == 0:
                xl = (np.min(x) * 0.9, np.max(x) * 1.1)
        xf = np.logspace(np.log10(xl[0]), np.log10(xl[1]), 100)
        v = out.best_values
        gf = gaussian(xf, **v)
        if "legend" in doplot:
            labstr = (
                r"$\mu: {0[0]:.2g},\, \sigma: {0[1]:.2g},\,"
                + r" \mathrm{{a}}: {0[2]:.2g},\, \mathrm{{bg}}: {0[3]:.2g}$"
            )
            labstr = labstr.format([v[name] for name in out.var_names])
        else:
            labstr = ""

        pl = []
        pl.append(ax.plot(xf, gf, "-", label=labstr, linewidth=1))
        if "data" in doplot:
            pl.append(ax.plot(x, y, "o", markersize=2.5))

        if color is None:
            if h_plot is not None:
                color = h_plot.get_color()
            elif "data" in doplot:
                color = pl[0][0].get_color()
            else:
                color = "gray"

        for p in pl:
            p[0].set_color(color)

        ax.legend()
        if "report" in doplot:
            print(out.fit_report())
    if output == "pars":
        pars = np.zeros((3 * nmodes + 1, 2))
        for i, vn in enumerate(out.var_names):
            pars[i, 0] = out.best_values[vn]
            param = list(out.params.values())
            pars[i, 1] = 1.0 * param[i].stderr
        return pars
    elif output == "fit":
        return out
