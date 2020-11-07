import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd

from ..Decorators import Decorators
from ..Xplot.niceplot import niceplot


@Decorators.init_figure((2, 2), (9, 8))
def staticcontrast(
    obj,
    nq=None,
    ax=None,
    data="original",
    cmap="magma",
    change_marker=False,
    color_mode=1,
    t_max=np.inf,
    trace=True,
    markersize=3,
    **kwargs
):

    if data == "original":
        corrFunc = list(obj.corrFunc)
    elif data == "rescaled":
        corrFunc = list(obj.corrFuncRescaled)
    else:
        raise ValueError("No usable correlation data defined.")

    if nq is None:
        pass
    elif type(nq) == int:
        obj.nq = np.arange(nq)
    else:
        obj.nq = nq

    if color_mode == 0:
        color_multiplier, color_repeater = obj.nq.size, len(corrFunc)
    elif color_mode == 1:
        color_multiplier, color_repeater = obj.nq.size * len(corrFunc), 1

    obj.update_colors(cmap, color_multiplier, color_repeater)
    obj.update_markers(len(corrFunc), change_marker)

    staticContrast = [[]] * len(corrFunc)

    tr = np.zeros((len(obj.db_id), len(obj.nq), 2))
    for i, ii in enumerate(obj.db_id):
        tmp = obj.Xana.get_item(ii)["trace"][:, obj.nq]
        tr[i] = np.vstack((np.mean(tmp, 0), np.std(tmp, 0))).T
    if trace:
        axtr = ax[1].twinx()
        niceplot(axtr)
        axtr.grid()
        axtr.set_ylabel("average intensity")
        axtr.set_yscale("log")

    ci = 0
    rates = np.zeros((len(obj.corrFunc), obj.nq.size, 2, 2))
    for j, (cfi, dcfi) in enumerate(corrFunc):
        ti = cfi[1:, 0]
        if t_max > 0:
            t_ind = np.where(ti <= t_max)[0]
        elif t_max < 0:
            t_ind = np.arange(1, cfi.shape[0] + t_max)
        tf = np.logspace(np.log10(ti[0]), np.log10(ti[-1]), 100)
        for i, qi in enumerate(obj.nq):
            if i == 0:
                cf_id = "id {}".format(obj.db_id[j])
            else:
                cf_id = None
            rates[j, i, 0, 1] = cfi[0, qi + 1]
            rates[j, i, 1, :] = np.ma.average(
                cfi[t_ind + 1, qi + 1],
                weights=1 / dcfi[t_ind + 1, qi + 1] ** 2,
                returned=1,
            )
            rates[j, i, 1, 1] = np.sqrt(1 / rates[j, i, 1, 1])
            ax[0].errorbar(
                ti,
                cfi[1:, qi + 1],
                dcfi[1:, qi + 1],
                label="{:.2e}nm-1".format(obj.Xana.setup["qv"][qi]),
                color=obj.colors[ci],
                marker=obj.markers[j % len(obj.markers)],
                markersize=2,
            )
            ax[0].plot(
                tf,
                np.ones_like(tf) * rates[j, i, 1, 0],
                color=obj.colors[ci],
            )

            ax[1].errorbar(
                rates[j, i, 0, 1],
                rates[j, i, 1, 0],
                rates[j, i, 1, 1],
                color=obj.colors[ci],
                label=cf_id,
                marker=obj.markers[j % len(obj.markers)],
                markersize=markersize,
            )
            if trace:
                axtr.errorbar(
                    rates[j, i, 0, 1],
                    tr[j, qi, 0],
                    tr[j, qi, 1],
                    fmt="s",
                    color=obj.colors[ci],
                )
            ax[2].errorbar(
                tr[j, i, 0],
                rates[j, i, 1, 0],
                xerr=tr[j, i, 1],
                yerr=rates[j, i, 1, 1],
                color=obj.colors[ci],
                marker=obj.markers[j % len(obj.markers)],
                markersize=markersize,
            )

            ci += 1
        staticContrast[j] = rates

    for axi in ax:
        niceplot(axi)
    ax[0].set_xscale("log")
    ax[1].legend(loc="best")
    ax[0].set_xlabel(r"delay time $\tau$ [s]")
    ax[0].set_ylabel(r"$g_2(\tau)$")
    ax[1].set_ylabel(r"contrast")
    ax[1].set_xlabel(r"q[nm-1]")

    ax[2].set_xscale("log")
    ax[2].set_xlabel("average intensity")
    ax[2].set_ylabel("contrast")

    plt.tight_layout()

    return staticContrast
