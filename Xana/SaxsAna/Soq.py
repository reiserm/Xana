import numpy as np
from matplotlib import pyplot as plt
from ..Decorators import Decorators
from ..Analist import AnaList
from .pysaxs3 import get_soq
import copy
from ..Xplot.niceplot import niceplot


class Soq(AnaList):
    """
    Saxs class: Display and analyze Small Angle X-Ray scattering singals.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return "Saxs class for displaying saxs signal."

    def __repr__(self):
        return self.__str__()

    def get_sec(self, img):

        if np.shape(img) != np.shape(self.Xana.setup.mask):
            qsec = self.Xana.setup["qsec"][0]
            mask = self.Xana.mask.copy()
            dim = np.shape(img)
            mask = mask[qsec[0] : dim[0] + qsec[0], qsec[1] : dim[1] + qsec[1]]

            setup = copy.deepcopy(self.Xana.setup)
            setup["ctr"] = (setup["ctr"][0] - qsec[1], setup["ctr"][1] - qsec[0])
            return mask, setup
        else:
            return self.Xana.mask, self.Xana.setup

    @Decorators.init_figure()
    @Decorators.input2list
    def plot_soq(
        self,
        series_id,
        shade=False,
        cmap="tab10",
        cmap_shade="inferno",
        color_mode=0,
        ax=None,
        change_marker=0,
        markersize=3,
        legend="id",
        qexp=None,
        Iscaling=None,
        A=1.0,
        norm=False,
        normto=None,
        bg=None,
        logax="xy",
        Ae=1.0,
        show_legend=True,
        color="b",
        normto_exposure=False,
        **kwargs,
    ):

        if color_mode < 2:
            if color_mode == 0:
                color_multiplier, color_repeater = len(series_id), 1
            elif color_mode == 1:
                self.colors = [color] * len(series_id)
            self.update_colors(cmap, color_multiplier, color_repeater)
        elif color_mode == 2:
            self.colors = [color] * len(series_id)

        # niceplot(ax)

        ncorrections = sum([bool(normto), bool(bg)])

        if normto is not None:
            del series_id[series_id.index(normto)]
            series_id.insert(0, normto)

        Ib = 0
        if bg is not None:
            del series_id[series_id.index(bg)]
            series_id.insert(bg, 0)

        output = []
        for i, sid in enumerate(series_id):
            saxsd = self.Xana.get_item(sid)

            if "soq" not in saxsd:
                Isaxs = saxsd["Isaxs"]

                if Isaxs.ndim == 3:
                    Isaxs = self.Xana.arrange_tiles(Isaxs)

                #                mask, setup = self.get_sec(Isaxs)
                q, I, e = get_soq(Isaxs, self.Xana.setup)
            else:
                tmp = saxsd["soq"]
                q = tmp[:, 0]
                I = tmp[:, 1]
                e = tmp[:, 2]

            legstr = None
            if legend == "id":
                legstr = "id {}".format(int(sid))
            elif legend == "t_exposure":
                legstr = r"$t_{{e}} = {:.2e}s$".format(
                    self.Xana.db.loc[sid, "t_exposure"]
                )
            elif legend == "sample":
                legstr = self.Xana.db.loc[sid, "sample"]

            ylabel = r"$I(q)$"
            if Iscaling != None:
                ylabel += r" $\cdot q^{{{}}}$".format(Iscaling)
                I *= q ** Iscaling
                e *= q ** Iscaling

            xlabel = r"$q$ ($\mathrm{nm}^{-1}$)"
            if qexp != None:
                xlabel = r"$q^{{{}}}$ ($\mathrm{{nm}}^{{-{}}}$)".format(qexp, qexp)
                q *= q ** (qexp - 1)

            if bg is not None:
                if i == 0:
                    Ib = I
                    ncorrections = 0
                    continue
                else:
                    I -= Ib

            if normto_exposure:
                I /= self.Xana.db.loc[sid, "t_exposure"]

            if norm:
                e /= I.mean()
                I /= I.mean()

            if normto is not None:
                if i == 0:
                    normto = (q, I, e)
                    continue
                else:
                    I /= normto[1]
                    e /= normto[1]

            ax.errorbar(
                q,
                I * A,
                e * A * Ae,
                fmt="-",
                marker=self.markers[i % len(self.markers)],
                color=self.colors[i],
                label=legstr,
                markersize=markersize,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if "x" in logax:
                ax.set_xscale("log")
            if "y" in logax:
                ax.set_yscale("log")

            output.append((q, I * A, e * A))

        if show_legend:
            ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
            plt.tight_layout(rect=[0, 0, 0.98, 1])
        if shade:
            ax = plt.gca()
            shadeqrois(
                ax, self.Xana.setup["qv"], self.Xana.setup["dqv"], cmap=cmap_shade
            )
