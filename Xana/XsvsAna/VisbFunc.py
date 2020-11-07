import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import itertools
import copy
import pandas as pd

from ..Decorators import Decorators
from ..Analist import AnaList
from .PlotContrast import plot_quicklook
from ..Xplot.niceplot import niceplot
from .PhotonStats import prob2beta, average_beta, beta_from_likelihood, prob2betasigma


class VisbFunc(AnaList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratesl = None
        self.fit_result = None
        self.pars = None
        self.v2 = None
        self.prob = None
        self.t_exposure = None
        self.contrast = None
        self.v2plotl = None
        self.nq = np.arange(len(self.Xana.setup.qroi))
        self.db_id = None

    def __str__(self):
        return "Corr Func class for g2 displaying."

    def __repr__(self):
        return self.__str__()

    @Decorators.input2list
    def get_prob(self, db_id, merge=False, verbose=True):
        """Load photon probabilities from database"""
        self.db_id = db_id
        self.prob = [[] for _ in range(2)]
        t_exposure = np.empty(len(db_id))
        l = []
        for i, sid in enumerate(db_id):
            try:
                d = self.Xana.get_item(sid)
                l.append(d["prob"])
                t_exposure[i] = d["prob"][0, 0, 0]
            except KeyError:
                print("Could not load item {}".format(sid))

        ind = np.argsort(t_exposure)
        self.t_exposure = t_exposure[ind]
        self.prob[0] = [l[i] for i in ind]

        if verbose:
            print("Loaded probabilities of {} series.".format(len(db_id)))

        self.prob[1] = copy.deepcopy(self.prob[0])

        if merge:
            self.merge_problist()

    def merge_problist(self):
        """merge lists of photon probabilities"""
        self.t_exposure, cnt = np.unique(self.t_exposure, return_counts=True)
        cnt = np.cumsum(cnt)
        cnt = np.append(0, cnt)

        for ii, probl in enumerate(self.prob):
            prob = []
            for i, j in zip(cnt[:-1], cnt[1:]):
                prob.append(np.concatenate(probl[i:j], -1))
            self.prob[ii] = prob

    def calculate_contrast(self, method="formula", sigma=3):
        """Calculate speckle contrast in different ways."""
        self.contrast = [[] for _ in range(2)]
        for ii, probl in enumerate(self.prob[:1]):
            for prob in probl:
                if method == "formula":
                    gproi = self.Xana.setup.gproi
                    self.contrast[ii].append(prob2beta(prob, gproi))
                    # self.contrast[ii+1].append(prob2betasigma(prob, gproi, sigma))

    def calculate_v2(self, ratio=0, **kwargs):
        """Calculate Bandy function in different ways."""
        methods = ["av", "md", "mx", "pg", "ml"]
        self.v2 = {}
        v2_av = average_beta(self.t_exposure, self.Xana.setup.qv, self.contrast, ratio)
        for i, m in enumerate(methods[:3]):
            self.v2[m] = v2_av[i]

        self.v2["ml"] = beta_from_likelihood(
            self.t_exposure,
            self.Xana.setup.qv,
            self.prob,
            self.Xana.setup.gproi,
            **kwargs,
        )

        return None

    def quicklook(
        self,
        plot="default",
        idx=None,
        nq=None,
        ax=None,
        color_mode=0,
        change_marker=0,
        cmap="magma",
        *args,
        **kwargs,
    ):
        """Plot overview of photon probabilities and speckle contrast"""
        if plot == "default":
            plot = ["bvi", "brvi", "kbvi", "bvkb", "pbb", "pbk"]
        npl = len(plot)

        if nq is None:
            nq = self.nq

        if idx is None:
            idx = range(len(self.contrast[0]))
        nlines = len(idx)

        if ax is None:
            n = int(np.ceil(npl / 2))
            fig, ax = plt.subplots(
                n, int(npl > 1) + 1, figsize=(5 + 4 * (npl > 1), 3.5 * n)
            )

        if type(ax) == np.ndarray:
            ax = ax.flatten()
        else:
            ax = [
                ax,
            ]

        ax_idx = np.arange(npl, dtype=np.int16)
        c_idx = np.arange(nlines)

        if color_mode == 0:
            color_multiplier, color_repeater = nq.size, nlines
        elif color_mode == 1:
            color_multiplier, color_repeater = nq.size * nlines, 1
        self.update_colors(cmap, color_multiplier, color_repeater)
        self.update_markers(nlines, change_marker)

        for i, p in enumerate(plot):
            for j, jj in enumerate(idx):
                plot_quicklook(
                    self, p, jj, nq, ax[ax_idx[i]], c0=c_idx[j], *args, **kwargs
                )
        plt.tight_layout()

    @Decorators.init_figure()
    def plot_g2(
        self,
        nq=None,
        err=True,
        ax=None,
        nmodes=1,
        data="original",
        cmap="magma",
        change_marker=False,
        color_mode=0,
        dofit=True,
        **kwargs,
    ):

        if data == "original" or self.corrFuncRescaled is None:
            corrFunc = list(self.corrFunc)
        elif data == "rescaled":
            corrFunc = list(self.corrFuncRescaled)
        else:
            raise ValueError("No usable correlation data defined.")

        if nq is None:
            pass
        elif type(nq) == int:
            self.nq = np.arange(nq)
        else:
            self.nq = nq

        if color_mode == 0:
            color_multiplier, color_repeater = self.nq.size, len(corrFunc)
        elif color_mode == 1:
            color_multiplier, color_repeater = self.nq.size * len(corrFunc), 1

        self.update_colors(cmap, color_multiplier, color_repeater)
        self.update_markers(len(corrFunc), change_marker)

        self.g2plotl = [[]] * len(corrFunc)
        self.pars = [[]] * len(corrFunc)
        self.fit_result = [[[] for i in range(self.nq.size)]] * len(
            corrFunc
        )  # possible bug: will cause
        # wrong references
        ci = 0
        for j, (cfi, dcfi) in enumerate(corrFunc):
            rates = np.zeros((self.nq.size, 3 * nmodes + 3, 2))
            ti = cfi[1:, 0]
            for i, qi in enumerate(self.nq):
                if i == 0:
                    cf_id = self.db_id[j]
                else:
                    cf_id = None
                res = fitg2(
                    ti,
                    cfi[1:, qi + 1],
                    err=dcfi[1:, qi + 1],
                    qv=self.Xana.setup.qv[qi],
                    ax=ax,
                    color=self.colors[ci],
                    dofit=True,
                    marker=self.markers[j % len(self.markers)],
                    cf_id=cf_id,
                    modes=nmodes,
                    **kwargs,
                )
                self.fit_result[j][i] = res[2:4]
                self.g2plotl[j].append(list(itertools.chain.from_iterable(res[4])))
                if dofit:
                    if i == 0:
                        db_tmp = self.init_pars(list(res[2].params.keys()))
                    entry = [cfi[0, qi + 1], *res[0].flatten(), *res[1]]
                    db_tmp.loc[i] = entry
                else:
                    db_tmp = 0
                ci += 1
            self.pars[j] = db_tmp
        plt.tight_layout()
        plt.show()

    @staticmethod
    def init_pars(names):
        # names = [names[i//2] if (i+1)%2 else '+/-' for i in range(len(names)*2)]
        names = [
            names[i // 2] if (i + 1) % 2 else "d" + names[i // 2]
            for i in range(len(names) * 2)
        ]
        names.insert(0, "q")
        names.extend(["chisqr", "redchi", "bic", "aic"])
        return pd.DataFrame(columns=names)

    def reset_rescaled(self):
        """Reset rescaled probabilities by copying the original ones."""
        self.prob[1] = copy.deepcopy(self.prob[0])

    @Decorators.init_figure()
    @Decorators.input2list
    def plot_trace(self, db_id, log="", ax=None):
        axtop = ax.twiny()

        ci = 0
        for sid in db_id:
            corfd = self.Xana.get_item(sid)
            trace = corfd["trace"]
            framen = np.arange(trace.shape[0])
            time = corfd["twotime_xy"]
            for i, iq in enumerate(self.nq):
                ax.plot(time, trace[:, iq], color=self.colors[ci])
                axtop.plot(framen, trace[:, iq], color=self.colors[ci])
                ci += 1

        ax.set_ylabel("photons per pixel")
        ax.set_xlabel("time in [s]")
        # niceplot(ax, autoscale=0)
        axtop.set_xlabel("frame number")
        # niceplot(axtop, autoscale=False, grid=False)
        if "logx" in log:
            ax.set_xscale("log")
        if "logy" in log:
            ax.set_yscale("log")

        # ax.set_title('trace', fontweight='bold', fontsize=14, y=1.14)

    def get_static_contrast(self, *args, **kwargs):
        self.staticContrast = staticcontrast(self, *args, **kwargs)

    def g2_totxt(self, savname):
        return 0
