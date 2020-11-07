import numpy as np
import lmfit
from .g2function import g2
import matplotlib
from matplotlib import pyplot as plt
from ..Xplot.niceplot import niceplot


def fitg2(
    t,
    cf,
    err=None,
    mode="semilogx",
    modes=1,
    init={},
    fix={},
    dofit=False,
    doplot=False,
    marker="o",
    qv=None,
    h_plot=None,
    ax=None,
    xl=None,
    yl=None,
    color=None,
    alpha=1.0,
    markersize=3.0,
    cf_id=None,
    data_label=None,
    errthr=1e-4,
):

    # define residual function
    def res(pars, x, data=None, eps=None):
        """2D Residual function to minimize"""
        v = pars.valuesdict()
        for i in range(modes):
            if i == 0:
                model = g2(
                    x,
                    t=v["t{}".format(i)],
                    b=v["b{}".format(i)],
                    g=v["g{}".format(i)],
                    a=v["a"],
                )
            else:
                model += g2(
                    x,
                    t=v["t{}".format(i)],
                    b=v["b{}".format(i)],
                    g=v["g{}".format(i)],
                    a=0,
                )
        if eps is not None:
            resid = np.abs(data - model) * np.abs(eps)
        else:
            resid = np.abs(data - model)
        return resid

    # make initial guess for parameters
    for i in range(modes):
        for s in "tgb":
            vn = s + "{}".format(i)
            if vn not in init.keys():
                if s == "t":
                    t0 = np.logspace(np.log10(t.min()), np.log10(t.max()), modes + 2)[
                        i + 1
                    ]
                    init[vn] = (t0, -np.inf, np.inf)
                elif s == "g":
                    init[vn] = (1, 0.2, 1.8)
                elif s == "b":
                    init[vn] = (0.1, 0, 1)
    if "a" not in init.keys():
        init["a"] = (np.mean(cf[-10:]), 0, 2)
    if "beta" not in init.keys():
        init["beta"] = (0.2, 0, 1)

    # initialize parameters
    pars = lmfit.Parameters()
    for i in range(modes):
        for s in "tgb":
            vn = s + "{}".format(i)
            pars.add(vn, value=init[vn][0], min=init[vn][1], max=init[vn][2], vary=1)
    pars.add("a", value=init["a"][0], min=init["a"][1], max=init["a"][2], vary=1)
    pars.add(
        "beta", value=init["beta"][0], min=init["beta"][1], max=init["beta"][2], vary=1
    )

    if modes > 1:
        beta_constraint = "a+beta-1-" + "-".join(
            ["b{}".format(x) for x in range(1, modes)]
        )
        pars["b0"].set(expr=beta_constraint)
    else:
        pars["b0"].set(expr="beta")

    if err is not None:
        inderr = np.isfinite(err) & (err > 0)
    else:
        inderr = np.ones_like(cf, bool)

    ind = np.where(np.isfinite(cf) & inderr)[0]
    ind = ind[:-2]
    t = t[ind]
    cf = cf[ind]
    if err is not None:
        err = err[ind]

    for vn in fix.keys():
        if vn in pars.keys():
            pars[vn].set(value=fix[vn], min=-np.inf, max=np.inf, vary=0)

    if err is not None:
        wgt = err.copy()
        wgt = 1.0 / wgt
    else:
        wgt = None

    if "semilogx" in mode:
        if err is not None:
            wgt = np.log10(wgt)
        elif err == "t":
            wgt = 1.0 / np.log10(t)
        else:
            wgt = None

    if "y" in mode:
        if err is not None:
            wgt = cf - 1

    if dofit:
        out = lmfit.minimize(
            res, pars, args=(t,), kws={"data": cf, "eps": wgt}, nan_policy="omit"
        )

    # do all the plotting
    pl = []
    if doplot:
        if xl is None:
            if ax is None:
                ax = plt.gca()
            xl = ax.get_xlim()
            if xl[0] == 0:
                xl = (np.min(t) * 0.5, np.max(t) * 1.5)

        xf = np.logspace(np.log10(xl[0]), np.log10(xl[1]), 50)
        if dofit:
            for i in range(modes):
                if i == 0:
                    g2f = g2(
                        xf,
                        t=out.params["t{}".format(i)].value,
                        b=out.params["b{}".format(i)].value,
                        g=out.params["g{}".format(i)].value,
                        a=out.params["a"].value,
                    )
                else:
                    g2f += g2(
                        xf,
                        t=out.params["t{}".format(i)].value,
                        b=out.params["b{}".format(i)].value,
                        g=out.params["g{}".format(i)].value,
                        a=0,
                    )

        if "legf" in doplot and dofit:
            pard = {
                "t": r"$t: {:.2e}\mathrm{{s}},\,$",
                "g": r"$\gamma: {:.2g},\,$",
                "b": r"$\mathrm{{b}}: {:.3g},\,$",
                "a": r"$\mathrm{{a}}: {:.3g},\,$",
            }
            labstr_fit = ""
            for i in range(modes):
                for vn in "tgba":
                    if vn == "a" and i > 0:
                        continue
                    elif vn == "a" and i == 0:
                        vnn = "a"
                    else:
                        vnn = vn + str(i)
                    if vnn in out.var_names:
                        labstr_fit += pard[vn].format(out.params[vnn].value)
                    elif fix is not None and vnn in fix.keys():
                        labstr_fit += "fix " + pard[vn].format(fix[vnn])
                    else:
                        labstr_fit += pard[vn].format(0)
        else:
            labstr_fit = None

        if "legd" in doplot:
            if qv is not None:
                labstr_data = r"$\mathsf{{q}} = {:.3f}\,\mathsf{{nm}}^{{-1}}$".format(
                    qv
                )
            elif data_label is not None:
                labstr_data = data_label
            else:
                labstr_data = "data"
        elif "legid" in doplot:
            if cf_id is not None:
                labstr_data = r"id: {}".format(cf_id)
            else:
                labstr_data = None
        else:
            labstr_data = None

        if "fit" in doplot and dofit:
            if "g1" in doplot:
                g2f = np.sqrt((g2f - out.params["a"].value) / out.params["b0"].value)
            pl.append(ax.plot(xf, g2f, "-", label=labstr_fit, linewidth=1))

        if "data" in doplot:
            if "g1" in doplot:
                cf = np.sqrt((cf - out.params["a"].value) / out.params["b0"].value)
            if err is not None:
                pl.append(
                    ax.errorbar(
                        t,
                        cf,
                        yerr=err,
                        linestyle="",
                        marker=marker,
                        label=labstr_data,
                        alpha=alpha,
                        markersize=markersize,
                    )
                )
            else:
                pl.append(
                    ax.plot(
                        t,
                        cf,
                        marker,
                        label=labstr_data,
                        alpha=alpha,
                        markersize=markersize,
                    )
                )

        if color is None:
            if h_plot is not None:
                color = h_plot.get_color()
            elif "data" in doplot:
                color = pl[0][0].get_color()
            else:
                color = "gray"

        for p in pl:
            if type(p) == list:
                p[0].set_color(color)
            elif type(p) == matplotlib.container.ErrorbarContainer:
                p[0].set_color(color)
                p[2][0].set_color(color)

        if "g1" in doplot:
            ax.set_xscale("linear")
            ax.set_yscale("log")
        else:
            ax.set_xscale("log")
            ax.set_yscale("linear")

        if "leg" in doplot:
            ax.legend()

        if "report" in doplot and dofit:
            print(lmfit.fit_report(out))

        if "redchi" in doplot and dofit:
            print(out.redchi)

        ax.set_xlabel(r"delay time $\tau$ [s]")
        ax.set_ylabel(r"$g_2(\tau)$")

        niceplot(ax, autoscale=0)
        ax.set_xlim(*xl)
        if yl is not None:
            ax.set_ylim(*yl)

    pars_arr = np.zeros((3 * modes + 2, 2))
    if dofit:
        for i, vn in enumerate(pars.keys()):
            pars_arr[i, 0] = out.params[vn].value
            try:
                pars_arr[i, 1] = 1.0 * out.params[vn].stderr
            except TypeError:
                pars_arr[i, 1] = 1
        gof = np.array([out.chisqr, out.redchi, out.bic, out.aic])
        return pars_arr, gof, out, lmfit.fit_report(out), pl
    else:
        return 0, 0, 0, 0, pl
