import numpy as np
import lmfit
import re
from fit.BandyFunctions import *


def fitbandy(
    et,
    M,
    err=None,
    mode="M",
    scaling="logxlogy",
    start=None,
    lb=None,
    ub=None,
    fix=None,
    doplot=0,
    h_plot=None,
    ax=None,
    output="pars",
    xl=None,
    ylim=None,
    cond=None,
    color=None,
    linestyle="-",
):
    pars = lmfit.Parameters()
    pars.add("t", value=start[0], min=lb[0], max=ub[0], vary=1)
    pars.add("g", value=start[1], min=lb[1], max=ub[1], vary=1)
    pars.add("b", value=start[2], min=lb[2], max=ub[2], vary=1)
    pars.add("a", value=0, vary=0)

    if "off" in mode:
        if len(start) == 3:
            pars["a"].set(vary=1)
        elif len(start) == 4:
            pars["a"].set(value=start[3], min=lb[3], max=ub[3], vary=1)

    if fix is not None:
        for vn in fix.keys():
            pars[vn].set(value=fix[vn], vary=0)

    if err is not None:
        if "logy" in scaling:
            wgt = (M * np.log(10) / err) ** 2
        else:
            wgt = 1.0 / err ** 2
    else:
        err = np.ones_like(M)
        wgt = np.ones_like(M)

    Mp = M
    etp = et
    errp = err
    if cond is not None:
        et = et[cond]
        M = M[cond]
        err = err[cond]
        wgt = wgt[cond]

    if "logy" in scaling:
        M = np.log10(M)
    if "logx" in scaling:
        wgt *= et ** 10

    if "M" in mode:
        func = bandy_logMa
    mod = lmfit.Model(func)
    out = mod.fit(M, pars, x=et, weights=wgt)

    if doplot:
        if xl is None:
            if ax is None:
                ax = plt.gca()
            xl = ax.get_xlim()
            if xl[0] == 0:
                xl = (np.min(et) * 0.9, np.max(et) * 1.1)
        xf = np.logspace(np.log10(xl[0]), np.log10(xl[1]), 50)
        bandyf = 1 / bandy_V2a(
            xf,
            g=out.best_values["g"],
            t=out.best_values["t"],
            b=out.best_values["b"],
            a=out.best_values["a"],
        )

        if "V2" in doplot:
            bandyf = 1 / bandyf
            Mp = 1 / Mp
            errp = Mp ** 2 * errp
        if "g2" in doplot:
            Mp += 1.0
            bandyf += 1.0

        if "legend" in doplot:
            pard = {
                "t": r"$t_0: {:.2g}\mathrm{{s}},\,$",
                "g": r"$\gamma: {:.2g},\,$",
                "b": r"$\mathrm{{b}}: {:.2g},\,$",
                "a": r"$\mathrm{{a}}: {:.2g},\,$",
            }
            labstr = ""
            for vn in "tgba":
                if vn in out.var_names:
                    labstr += pard[vn].format(out.best_values[vn])
                elif fix is not None and vn in fix.keys():
                    labstr += "fix " + pard[vn].format(fix[vn])
                else:
                    labstr += pard[vn].format(0)
        else:
            labstr = ""

        pl = []
        pl.append(
            ax.plot(xf, bandyf, linestyle, label=labstr, color=color, linewidth=1)
        )
        if "data" in doplot:
            pl.append(
                ax.errorbar(
                    etp, Mp, yerr=errp, marker="d", markersize=3.5, linestyle=""
                )
            )

        if color is None:
            if h_plot is not None:
                color = h_plot[0].get_color()
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
        pars = np.zeros((4, 2))
        for i, vn in enumerate(out.var_names):
            pars[i, 0] = out.best_values[vn]
            param = list(out.params.values())
            pars[i, 1] = param[i].stderr
        return pars
    elif output == "fit":
        return out


def fitBandy2(et, M, dM=None, mode="logM", start=None, lb=None, ub=None):
    def residual(pars, x, data=None, eps=None):
        # unpack parameters:
        #  extract .value attribute for each parameter
        parvals = pars.valuesdict()
        period = parvals["period"]
        shift = parvals["shift"]
        decay = parvals["decay"]

        if abs(shift) > pi / 2:
            shift = shift - sign(shift) * pi

        if abs(period) < 1.0e-10:
            period = sign(period) * 1.0e-10

        model = parvals["amp"] * sin(shift + x / period) * exp(-x * x * decay * decay)

        if data is None:
            return model
        if eps is None:
            return model - data
        return (model - data) / eps

    pars = lmfit.Parameters()
    pars.add("t", value=start[0], min=lb[0], max=ub[0], vary=1)
    pars.add("g", value=start[1], min=lb[1], max=ub[1], vary=1)
    if dM is not None:
        wgt = 1.0 / dM ** 2
    else:
        wgt = np.ones_like(M)
    if mode == "M":
        mod = lmfit.Model(bandy_M)
    elif mode == "logM":
        mod = lmfit.Model(bandy_logM)
    elif mode == "logMa" or mode == "logMa_off":
        mod = lmfit.Model(bandy_logMa)
        if "off" in mode:
            pars.add("a", value=0, min=0, vary=1)
        else:
            pars.add("a", value=0, vary=0)
        pars.add("b", value=start[2], min=lb[2], max=ub[2], vary=1)
    elif mode == "logV2a":
        mod = lmfit.Model(bandy_V2a)
        wgt = np.log10(wgt)
        pars.add("a", value=0, vary=0)
        pars.add("b", value=start[2], min=lb[2], max=ub[2], vary=1)
    if "log" in mode:
        M = np.log10(M)
        if dM is not None:
            wgt = 1.0 / np.log10(dM) ** 2
    return mod.fit(M, pars, x=et, weights=wgt)
