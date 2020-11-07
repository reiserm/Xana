import numpy as np
import lmfit
from fit.BandyFunctions import *
from matplotlib import pyplot as plt
from misc.niceplot import niceplot


def fitbandy(
    t,
    cf,
    err=None,
    mode="standard",
    modes=1,
    init={},
    fix=None,
    doplot=False,
    marker="o",
    qv=None,
    h_plot=None,
    ax=None,
    output="pars",
    xl=None,
    ylim=None,
    color=None,
):

    # make initial guess for parameters
    for i in range(modes):
        for s in "tgb":
            vn = s + "{}".format(i)
            if vn not in init.keys():
                if s == "t":
                    t0 = np.percentile(t, 100 / (modes + 1) * (i + 1))
                    init[vn] = (t0, t0 / 100, t0 * 100)
                elif s == "g":
                    init[vn] = (1, 0.2, 1.8)
                elif s == "b":
                    init[vn] = (0.1, 0, 1)
    if "a" not in init.keys():
        init["a"] = (0.0, 0.0, 0.2)

    # initialize parameters
    pars = lmfit.Parameters()
    for i in range(modes):
        for s in "tgb":
            vn = s + "{}".format(i)
            pars.add(vn, value=init[vn][0], min=init[vn][1], max=init[vn][2], vary=1)
    pars.add("a", value=init["a"][0], min=init["a"][1], max=init["a"][2], vary=0)

    if "off" in mode:
        pars["a"].set(vary=1)

    if fix is not None:
        for vn in fix.keys():
            pars[vn].set(value=fix[vn], vary=0)

    if err is not None:
        wgt = 1.0 / err
    else:
        wgt = 1.0 / t

    if "semilogx" in mode:
        if err is not None:
            wgt = 1.0 / np.log10(err)
        else:
            wgt = 1.0 / np.log10(t)

    # define residual function
    def res(pars, x, data=None, eps=None):
        """2D Residual function to minimize"""
        v = pars.valuesdict()
        for i in range(modes):
            if i == 0:
                model = V2a(
                    x,
                    t=v["t{}".format(i)],
                    b=v["b{}".format(i)],
                    g=v["g{}".format(i)],
                    a=v["a"],
                )
            else:
                model += V2a(
                    x,
                    t=v["t{}".format(i)],
                    b=v["b{}".format(i)],
                    g=v["g{}".format(i)],
                    a=0,
                )
        if eps is not None:
            resid = np.abs(data - model) / np.abs(eps)
        else:
            resid = np.abs(data - model)
        return resid

    out = lmfit.minimize(
        res, pars, args=(t,), kws={"data": cf, "eps": err}, nan_policy="omit"
    )

    # do all the plotting
    if doplot:
        if xl is None:
            if ax is None:
                ax = plt.gca()
            xl = ax.get_xlim()
            if xl[0] == 0:
                xl = (np.min(t) * 0.9, np.max(t) * 1.1)
        xf = np.logspace(np.log10(xl[0]), np.log10(xl[1]), 50)

        for i in range(modes):
            if i == 0:
                g2f = V2a(
                    xf,
                    t=out.params["t{}".format(i)].value,
                    b=out.params["b{}".format(i)].value,
                    g=out.params["g{}".format(i)].value,
                    a=out.params["a"].value,
                )
            else:
                g2f += V2a(
                    xf,
                    t=out.params["t{}".format(i)].value,
                    b=out.params["b{}".format(i)].value,
                    g=out.params["g{}".format(i)].value,
                    a=0,
                )

        if "legf" in doplot:
            pard = {
                "t": r"$t: {:.2g}\mathrm{{s}},\,$",
                "g": r"$\gamma: {:.2g},\,$",
                "b": r"$\mathrm{{b}}: {:.2g},\,$",
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
            labstr_fit = ""

        if "legd" in doplot:
            if qv is not None:
                labstr_data = r"$\mathsf{{q}} = {:.3f}\,\mathsf{{nm}}^{{-1}}$".format(
                    qv * 10
                )
            else:
                labstr_data = "data"
        else:
            labstr_data = ""

        pl = []
        if "fit" in doplot:
            if "log" in doplot:
                g2f = np.log(np.sqrt(out.best_values["b"] / (g2f - 1)))
                xf = xf / out.best_values["t"]
            pl.append(ax.plot(xf, g2f, "-", label=labstr_fit, linewidth=1))

        if "data" in doplot:
            if "log" in doplot:
                t = t / out.best_values["t"]
                cf = np.sqrt((cf - 1) / out.best_values["b"])
                pl.append(ax.plot(t, cf, marker, markersize=2.5))
            pl.append(ax.plot(t, cf, marker, label=labstr_data, markersize=2.5))

        if color is None:
            if h_plot is not None:
                color = h_plot.get_color()
            elif "data" in doplot:
                color = pl[0][0].get_color()
            else:
                color = "gray"

        if "log" in doplot:
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        else:
            ax.set_xscale("log")
            ax.set_yscale("linear")

        for p in pl:
            p[0].set_color(color)

        if "leg" in doplot:
            ax.legend()

        if "report" in doplot:
            print(out.fit_report())

        ax.set_xlabel(r"delay time $\tau$ [s]")
        ax.set_ylabel(r"$g_2(\tau)$")

        niceplot(ax)
    if output == "pars":
        pars_arr = np.zeros((3 * modes + 1, 2))
        for i, vn in enumerate(pars.keys()):
            pars_arr[i, 0] = out.params[vn].value
            pars_arr[i, 1] = 1.0 * out.params[vn].stderr
        return pars_arr
    elif output == "fit":
        return out
