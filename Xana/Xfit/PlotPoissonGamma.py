#! /usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import fit.fitPoissonGamma as fpg


def plotpoissongamma(
    kb,
    p,
    kv,
    M,
    dkb=0,
    dp=0,
    dM=0,
    cmap="tab20",
    ind_var="kb",
    confint=None,
    q=0,
    ax=None,
    xlim=None,
    axscale=None,
    **kwargs
):
    if ax is None:
        ax = plt.gca()

    if ind_var == "kb":
        # axis scale
        if axscale is None:
            xscale = "log"
            yscale = "log"
        else:
            xscale = axscale[0]
            yscale = axscale[1]

        # define colormap
        cmap = plt.get_cmap(name=cmap)
        ci = np.arange(kv.size)

        for i, k in enumerate(kv):
            kbi = kb.copy()
            pi = p[:, i]
            dpi = dp[:, i]
            c1 = cmap(ci[i])
            c2 = cmap(ci[i])
            ind = pi > 0
            kbi = kbi[ind]
            dkbi = dkb.copy()[ind]
            pi = pi[ind]
            dpi = dpi[ind]
            # plot data
            ax.errorbar(
                1 / kbi,
                pi,
                yerr=dpi,
                xerr=1 / kbi ** 2 * dkbi,
                marker="o",
                linestyle="",
                color=c2,
                markeredgewidth=0.1,
                markeredgecolor="gray",
                label=r"$k={0:d}$".format(int(k)),
            )
            yl = ax.get_ylim()

            # define x-axis limits
            if xlim is None:
                kbs = kbi[(kbi > 0) & (kbi < np.inf)]
                if kbs.size > 0:
                    xmin = np.min(1 / kbs) * 0.9
                    xmax = np.max(1 / kbs) * 1.1
                else:
                    xmin, xmax = ax.get_xlim()
            else:
                xmin, xmax = xlim

            # kb vector to plot fit
            kbf = np.logspace(np.log10(1 / xmax), np.log10(1 / xmin), 50)

            # plot confident intervals
            if confint is not None:
                for Mc in confint:
                    pbf_lb = fpg.PoissonGamma(kbf, k=k, M=Mc)
                    ax.plot(1 / kbf, pbf_lb, "--", color="gray")

            if not np.isnan(M):
                # plot fit
                # calculate estimated function and plot fit
                pbf = fpg.PoissonGamma(kbf, k=k, M=M)
                ax.plot(1 / kbf, pbf, "-", color=c1)
                ax.annotate(
                    r"$\beta={0:.2g}\pm{1:.2g}$".format(1 / M, 1 / M ** 2 * dM),
                    xy=(0.8, 0.1),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round", ec=(0.57, 0.57, 0.57), fc="w"),
                )

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_ylim(yl)
        ax.set_xlabel(r"1/$\langle \mathrm{{k}}\rangle$")
        ax.set_ylabel("photon probability")
        ax.set_title(
            r"$\mathsf{{q}} = {:.2f}\times 10^{{-3}}\,\mathsf{{\AA}}^{{-1}}$".format(
                q * 1e3
            )
        )
        ax.legend(loc="best")

    if ind_var == "k" or ind_var == "ks2":

        # define colormap
        cmap = plt.get_cmap(name=cmap)
        nplots = p.shape[0]
        ci = np.arange(nplots)

        for i in range(nplots):
            kbi = kb[i]
            pi = p[i, :]
            dpi = dp[i, :]
            c1 = "gray"
            c2 = cmap(ci[i])

            # define x-axis limits
            if xlim is None:
                xmin = np.min(kv)
                xmax = np.max(kv)
            else:
                xmin, xmax = xlim

            # kb vector to plot fit
            kvf = np.linspace(xmin, xmax, 100)

            # plot confident intervals
            if confint is not None:
                for Mc in confint:
                    pbf_lb = fpg.PoissonGamma_indk(kvf, kb=kbi, M=Mc)
                    ax.plot(1 / kvf, pbf_lb, "--", color="gray")

            if not np.isnan(M):
                # plot fit
                # calculate estimated function and plot fit
                if (kvf.max() + M) > 160:
                    pbf = np.exp(fpg.PoissonGamma_indk_approx(kvf, kb=kbi, M=M))
                else:
                    pbf = fpg.PoissonGamma_indk(kvf, kb=kbi, M=M)
                ax.plot(kvf, pbf, "-", color=c1)
                ax.annotate(
                    r"$\beta={0:.2g}\pm{1:.2g}$".format(1 / M, 1 / M ** 2 * dM),
                    xy=(0.8, 0.1),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round", ec=(0.57, 0.57, 0.57), fc="w"),
                )

            # plot data
            ax.errorbar(
                kv,
                pi,
                yerr=dpi,
                marker="o",
                linestyle="",
                color=c2,
                markeredgewidth=0.1,
                markeredgecolor="gray",
                label=r"$\langle k\rangle={0:.3g}$".format(kbi),
            )

        if axscale is None:
            xscale = "linear"
            yscale = "log"
        else:
            xscale = axscale[0]
            yscale = axscale[1]

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)

        ax.set_xlabel(r"k")
        ax.set_ylabel("photon probability")
        ax.set_title(
            r"$\mathsf{{q}} = {:.2f}\times 10^{{-3}}\,\mathsf{{\AA}}^{{-1}}$".format(
                q * 1e3
            )
        )
        ax.legend(loc="best")
