import numpy as np
from matplotlib import pyplot as plt
from ..Xfit.basic import fitline, fitline0, fitconstant
from ..Xfit.MCMC_straight_line import mcmc_sl
from ..Xfit.fit_basic import fit_basic
from ..Xplot.niceplot import niceplot
from matplotlib.offsetbox import AnchoredText
from matplotlib import ticker


def plot_parameters(
    pars,
    parameter,
    R=250e-9,
    T=22,
    fit=None,
    modes=1,
    ax=None,
    marker="o",
    textbox=False,
    alpha=1,
    log="",
    label=None,
    ci=0,
    corner_axes=0,
    mfc=None,
    format_ticks=True,
    cmap=None,
    init={},
    fix=None,
    viscosity=False,
    fit_report=False,
    emcee=False,
    exc=None,
    excfit=None,
    excbad=True,
    weighted=True,
    xl=None,
    xlim=[None, None],
    ylim=[None, None],
    **kwargs
):
    def getD(eta, err=0):
        dD = 0
        D = kb * (T + 273.15) / (6 * np.pi * R * eta)
        if err:
            dD = D * err / eta
        return D, dD

    def geteta(D, err=0):
        deta = 0
        eta = kb * (T + 273.15) / (6 * np.pi * R * D * 1e-18)
        if err:
            deta = eta * err / D
        return eta, deta

    def blc(q, L, k, lc):
        def A(q):
            return 4 * np.pi / lc * q / k * np.sqrt(1 - q ** 2 / (4 * k ** 2))

        return 2 * (A(q) * L - 1 + np.exp(-A(q) * L)) / (A(q) * L) ** 2

    def line(x, p):
        return p[0] * x + p[1]

    def power(x, p):
        return p[0] * x ** p[1] + p[2]

    if type(modes) == int:
        modes = np.arange(modes - 1, modes)
    else:
        modes = np.array(modes)
        modes -= 1

    if parameter in [0, "G", "dispersion", "tau"]:
        name = "t"
    elif parameter in [1, "kww"]:
        name = "g"
    elif parameter in [2, "f0", "ergodicity"]:
        name = "b"

    if "ax" is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))

    kb = 1.381e-23
    m_unit = {
        "G": "nm{} s-1".format(alpha),
        "kww": "nm{}".format(alpha),
        "f0": "nm{}".format(alpha),
        "tau": "nm{} s".format(alpha),
    }
    b_unit = {"G": "s-1", "kww": "", "f0": "", "tau": "s"}
    y_label = {
        "G": r"$\Gamma (s^{-1})$",
        "kww": "kww",
        "f0": "ergodicity",
        "tau": r"$\tau\,(s)$",
    }

    if fit == "" or fit is None:
        dofit = False
    else:
        dofit = True

    qv = np.array(pars["q"])
    qv = qv ** alpha

    # values to be excluded
    iip = np.arange(qv.size)
    iif = iip.copy()
    if exc is not None:
        iip = np.delete(iip, exc)
    if excfit is not None:
        iif = np.delete(iif, np.hstack((excfit)))

    if xl is None:
        x = np.linspace(np.min(qv[iif]), np.max(qv[iif]), 100)
    else:
        x = np.linspace(xl[0], xl[1], 100)

    textstr = ""
    markers = ["^", "v"] if (len(modes) < 3) else ["o"]
    for ii, i in enumerate(modes):
        if label is None:
            labstr = "mode {}: {}".format(i + 1, parameter)
        else:
            labstr = label
        textstr += labstr

        # -------plot decay rates--------
        try:
            y = np.asarray(pars["{}{}".format(name, i)], dtype=np.float32)
            dy = np.asarray(pars["d{}{}".format(name, i)], dtype=np.float32)
        except KeyError:
            return np.zeros(5)

        y = np.ma.masked_where(~np.isfinite(y), y)
        dy = np.ma.masked_array(dy, mask=y.mask)

        if parameter == "G":
            y = 1 / y
            dy = y ** 2 * dy
        else:
            pass

        nf = np.where(dy.filled(0) <= 0)[0]
        bad_points = nf.size
        if bad_points:
            print("Found {} points with zero error\n".format(bad_points))
            if excbad:
                iff = np.array([p for p in iif if p not in nf])
                iip = np.array([p for p in iip if p not in nf])
                print("Excluded bad points.")
                if len(iff) == 0 or len(iip) == 0:
                    return np.zeros(5)

        color = cmap(ci)
        marker = markers[i]
        ax.errorbar(
            qv[iip], y[iip], dy[iip], fmt=marker, label=labstr, color=color, mfc=mfc
        )

        if dofit:
            if fit == "mcmc_line":
                m, b, f_m, m_ls, b_ls = mcmc_sl(
                    qv[iif], y[iif], dy[iif], doplot=corner_axes
                )
                # ax[0].plot(x2,m_ls*x2+b_ls)
                m, b = [(x[0], np.mean(x[1:])) for x in (m, b)]
            else:
                res = fit_basic(
                    qv[iif],
                    y[iif],
                    dy[iif],
                    fit,
                    init=dict(init),
                    fix=fix,
                    emcee=emcee,
                    **kwargs
                )
                fitpar = res[0].astype(np.float32)
                yf = res[4].eval(res[2].params, x=x)

            ax.plot(x, yf, color=color, label=None)

        if parameter in ["G", "tau"]:
            if viscosity:
                power = 1 if (parameter == "G") else -1
                textstr += (
                    "\neta = {0[0]:.4g} +/- {0[1]:.2g} [cP]".format(
                        np.array(geteta(*fitpar[0])) * 1e3
                    )
                    ** power
                )
        elif parameter == "f0" and dofit and "t" in res[2].params.keys():
            msd = 1 / (2 * res[2].params["t"].value)
            dmsd = 2 * msd ** 2 * res[2].params["t"].stderr
            r_loc = np.sqrt(6 * (msd))
            dr_loc = 6 / 2 / r_loc * dmsd
            textstr += "localization length: {:.2f} +/- {:.2f} nm\n".format(
                r_loc, dr_loc
            )

        if fit_report and dofit:
            print("\n" + textstr)
            print("-" * 16)
            print(res[3])

    # if format_ticks:
    #     x_labels = ax.get_xticks()
    #     try:
    #         @ticker.FuncFormatter
    #         def major_formatter(x, pos):
    #             return "{:.2f}".format(x)
    #         ax.ticklabel_format(axis='x', useMathText=True,
    #                             style='sci', scilimits=(0, 0))
    #     except:
    #         pass

    # set style
    if alpha == 1:
        x_lab = r"$\mathrm{q} (nm^{-1})$"
    else:
        x_lab = r"$\mathrm{{q}}^{0} (nm^{{-{0}}})$".format(alpha)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_label[parameter])

    if "x" in log:
        ax.set_xscale("log")
    if "y" in log:
        ax.set_yscale("log")

    if textbox:
        at = AnchoredText(
            textstr,
            loc=2,
        )
        ax.add_artist(at)

    ax.legend(loc="best")

    # ax.get_yaxis().get_major_formatter().set_useOffset(False)
    # niceplot(ax,)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if dofit:
        return res
    else:
        return np.zeros(5)
