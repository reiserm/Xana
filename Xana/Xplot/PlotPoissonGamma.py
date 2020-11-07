#! /usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from ..Xfit.PoissonGammaDistribution import PoissonGamma as pg


def plot_poissongamma(
    x,
    y,
    p,
    M,
    cmap="tab20",
    ind_var="kb",
    confint=None,
    q=None,
    ax=None,
    xlim=None,
    log="xy",
    **kwargs
):
    if ax is None:
        ax = plt.gca()

    for i, pi in enumerate(p):
        print(pi, M)
        # plot data
        ax.errorbar(
            1 / x,
            y[i],
            yerr=None,
            marker="o",
            linestyle="",
            label=ind_var + r"{0:d}".format(int(pi)),
        )

        # kb vector to plot fit
        xf = np.logspace(np.log10(1 / x.max()), np.log10(1 / x.min()), 100)

        # plot confident intervals
        if confint is not None:
            for Mc in confint:
                pbf = pg(xf, Mc, pi, ind_var)
                ax.plot(1 / xf, pbf, "--", color="gray")

        pbf = pg(xf, M[0][0], pi, ind_var)
        print(pbf)
        ax.plot(1 / xf, pbf, "-")
        ax.annotate(
            r"$\beta={0:.2g}\pm{1:.2g}$".format(
                1 / M[0][0], 1 / M[0][0] ** 2 * M[0][1]
            ),
            xy=(0.8, 0.1),
            xycoords="axes fraction",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", ec=(0.57, 0.57, 0.57), fc="w"),
        )

    if "x" in log:
        ax.set_xscale("log")
    if "y" in log:
        ax.set_yscale("log")
    ax.set_xlabel(r"1/$\langle \mathrm{{k}}\rangle$")
    ax.set_ylabel("photon probability")
    #    ax.set_title(r'$\mathsf{{q}} = {:.2f}\times 10^{{-3}}\,\mathsf{{\AA}}^{{-1}}$'.format(q*1e3))
    ax.legend(loc="best")
