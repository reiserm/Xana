import numpy as np
import lmfit
from .g2function import g2 as g2func
from copy import copy
import sys
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import re
from scipy.special import gamma


class G2:
    def __init__(self, cf, nq, dcf=None):

        self.t = np.ma.masked_invalid(cf[1:, 0])
        self.qv = cf[0, 1:].copy()
        self.nq = np.asarray(nq)

        self.cf = np.ma.masked_invalid(cf[1:, 1:]).T
        ind_sort = np.argsort(self.t)
        self.t = self.t[ind_sort].astype(np.float64)
        self.cf = self.cf[:, ind_sort].astype(np.float64)

        if isinstance(dcf, np.ndarray):
            self.dcf = np.ma.masked_invalid(dcf[1:, 1:]).T
            self.dcf = self.dcf[:, ind_sort]
        else:
            self.dcf = None

        # properties
        self.nmodes = 1
        self.fitglobal = []
        self.fitqdep = None
        self.fix = None

        # attributes defined in private methods
        self._lmpars = None
        self._weights = None
        self._varnames = None

        self._fit_data = None
        self._fit_weights = None

        # fit results
        self.pars = None
        self.fit_result = []

        self._minimizer = None

        # plot handles
        self.plot_handles = None

    @property
    def nmodes(self):
        return self.__nmodes

    @nmodes.setter
    def nmodes(self, nmodes):
        if nmodes <= 0:
            print("Number of modes has been set to 1.")
            self.__nmodes = 1
        else:
            self.__nmodes = nmodes

    @property
    def fitglobal(self):
        return self.__fitglobal

    @fitglobal.setter
    def fitglobal(self, fitglobal):
        dofitglobal = bool(fitglobal)
        if dofitglobal:
            self.ndat = len(self.nq)
        else:
            self.ndat = 1
            fitglobal = []
        self.__fitglobal = fitglobal

    @staticmethod
    def _reduce_func(arr):
        return -0.5 * np.nansum(arr * arr)

    @staticmethod
    def _iter_cb(params, iter, resid, *args, **kws):
        if (iter) % 128 == 0:
            print("\rOptimizing...(%d)" % iter, end="", flush=1)
        else:
            pass

    def fit(
        self,
        mode="sig",
        nmodes=1,
        fitglobal=[],
        init={},
        fix={},
        lmfit_pars={},
        fitqdep={},
    ):
        """
        Function that computes the fits using lmfit's minimizer
        """

        self.fitglobal = fitglobal
        self.nmodes = nmodes
        self.fitqdep = fitqdep
        self.fix = fix

        # make the parameters
        if not lmfit_pars.get("is_weighted", True):
            init["__lnsigma"] = 1

        self._init_parameters(init, fix)

        # setting the weights of the data
        self._get_weights(mode)

        self._minimizer = lmfit.Minimizer(
            self._residuals,
            params=self._lmpars,
            reduce_fcn=self._reduce_func,
            iter_cb=self._iter_cb,
            nan_policy="omit",
        )

        if bool(self.fitglobal):
            # set data to fit
            self._fit_data = self.cf[self.nq]
            self._fit_weights = self._weights[self.nq]

            # do the fit
            out = self._minimizer.minimize(**lmfit_pars)

            self._write_to_pars(out)
            self.fit_result.append((out, lmfit.fit_report(out)))

        else:
            for line, qi in enumerate(self.nq):
                # set data to fit
                self._fit_data = self.cf[qi]
                self._fit_weights = self._weights[qi]

                # do the fit
                out = self._minimizer.minimize(**lmfit_pars)

                self._write_to_pars(out, line=line)
                self.fit_result.append((out, lmfit.fit_report(out)))

        self.pars["q"] = self.qv[self.nq]
        self.pars = self.pars.apply(pd.to_numeric)

        return self.pars, self.fit_result

    def plot(
        self,
        doplot=False,
        marker="o",
        ax=None,
        xl=None,
        yl=None,
        colors=None,
        alpha=1.0,
        linestyle="",
        markersize=6,
        markeredgecolor="w",
        data_label=None,
        confint=False,
        pars=None,
        **kwargs,
    ):

        sucfit = True if (self.pars is not None) else False

        if pars is not None:
            if len(pars):
                self.pars = pars
                self.nmodes = (
                    max([int(x[1]) if x.startswith("t") else 0 for x in pars.columns])
                    + 1
                )
                sucfit = True

        if xl is None:
            if ax is None:
                ax = plt.gca()
            xl = ax.get_xlim()
            if xl[0] == 0:
                xl = (np.min(self.t) * 0.5, np.max(self.t) * 1.5)

        xf = np.logspace(np.log10(xl[0]), np.log10(xl[1]), 50)

        for ii, iq in enumerate(self.nq):
            pl = []
            if sucfit:
                ipars = np.abs(self.pars["q"] - self.qv[iq]).idxmin()
                v = self.pars.iloc[ipars]
                g2f = 0
                for i in range(self.nmodes):
                    ve = f"{i}"
                    g2f += g2func(
                        xf,
                        t=v["t" + ve],
                        b=v["b" + ve],
                        g=v["g" + ve],
                        a=v["a"] * 1.0 * (not bool(i)),
                    )
                if confint:
                    g2fci = np.zeros((2, len(xf)))
                    for i in range(self.nmodes):
                        ve = f"{i}"
                        for j in range(2):
                            g2fci[j] += g2func(
                                xf,
                                t=v["t" + ve]
                                * (1 + (-1) ** j * 0.01 * confint.get("t" + ve, 0)),
                                b=v["b" + ve]
                                * (1 + (-1) ** j * 0.01 * confint.get("b" + ve, 0)),
                                g=v["g" + ve]
                                * (1 + (-1) ** j * 0.01 * confint.get("g" + ve, 0)),
                                a=v["a"] * 1.0 * (not bool(i)),
                            )

                labstr_fit = None
                if "legf" in doplot and sucfit:
                    pard = {
                        "t": r"$t: {:.2e}\mathrm{{s}},\,$",
                        "g": r"$\gamma: {:.2g},\,$",
                        "b": r"$\mathrm{{b}}: {:.3g},\,$",
                        "a": r"$\mathrm{{a}}: {:.3g},\,$",
                    }
                    labstr_fit = ""
                    for i in range(self.nmodes):
                        for vn in "tgba":
                            if vn == "a" and i > 0:
                                continue
                            elif vn == "a" and i == 0:
                                vnn = "a"
                            else:
                                vnn = vn + str(i)
                            if vnn in self.pars.columns:
                                labstr_fit += pard[vn].format(self.pars.loc[ii, vnn])
                            elif fix is not None and vnn in fix.keys():
                                labstr_fit += "fix " + pard[vn].format(fix[vnn])
                            else:
                                labstr_fit += pard[vn].format(0)

            labstr_data = None
            if "legd" in doplot:
                if data_label is not None:
                    labstr_data = data_label
                else:
                    labstr_data = (
                        r"$\mathsf{{q}} = {:.3f}\,\mathsf{{nm}}^{{-1}}$".format(
                            self.qv[iq]
                        )
                    )

            if "legq" in doplot:
                labstr_data = r"$\mathsf{{q}} = {:.3f}\,\mathsf{{nm}}^{{-1}}$".format(
                    self.qv[iq]
                )

            if "fit" in doplot and sucfit:
                if "g1" in doplot:
                    g2f = np.sqrt(
                        (g2f - out.params["a"].value) / out.params["b0"].value
                    )
                pl.append(ax.plot(xf, g2f, "-", label=labstr_fit, linewidth=1))
                if confint:
                    for j in range(2):
                        pl.append(ax.plot(xf, g2fci[j], ":", linewidth=1))

            if "data" in doplot:
                if "g1" in doplot:
                    cf = np.sqrt((cf - out.params["a"].value) / out.params["b0"].value)
                if self.dcf is not None:
                    pl.append(
                        ax.errorbar(
                            self.t.filled(np.nan),
                            self.cf[iq].filled(np.nan),
                            yerr=self.dcf[iq].filled(np.nan),
                            linestyle=linestyle,
                            marker=marker,
                            label=labstr_data,
                            alpha=alpha,
                            markersize=markersize,
                            mec=markeredgecolor,
                            markeredgewidth=1,
                        )
                    )
                else:
                    pl.append(
                        ax.plot(
                            self.t,
                            self.cf[iq],
                            marker,
                            label=labstr_data,
                            linestyle=linestyle,
                            alpha=alpha,
                            markersize=markersize,
                            mec=markeredgecolor,
                            markeredgewidth=1,
                        )
                    )

            if colors is None:
                if "data" in doplot:
                    color = pl[-1].get_color()
                else:
                    color = "gray"
            else:
                color = colors[ii]

            for p in pl:
                if isinstance(p, list):
                    p[0].set_color(color)
                elif isinstance(p, matplotlib.container.ErrorbarContainer):
                    for child in p.get_children():
                        child.set_color(color)

            self.plot_handles = pl

            if "g1" in doplot:
                ax.set_xscale("linear")
                ax.set_yscale("log")
            else:
                ax.set_xscale("log")
                ax.set_yscale("linear")

            if "leg" in doplot:
                ax.legend()

            if "report" in doplot and sucfit:
                print(lmfit.fit_report(out))

            ax.set_xlabel(r"$\tau\,(\mathrm{s})$")
            ax.set_ylabel(r"$g_2(\tau)$")

            ax.set_xlim(*xl)
            if yl is not None:
                ax.set_ylim(*yl)

        return None

    def _calc_model(
        self,
        v,
    ):
        """Calculate multi mode g2 funktion"""
        # v = pars.valuesdict()
        vn = self._varnames
        model = np.zeros((self.ndat, self.t.size), dtype=np.float64)

        for j in range(self.ndat):
            for i in range(self.nmodes):
                a = v[vn["a"][j][i]] if (i == 0) else 0
                model[j] += g2func(
                    self.t,
                    t=v[vn["t"][j][i]],
                    b=v[vn["b"][j][i]],
                    g=v[vn["g"][j][i]],
                    a=a,
                )
        return model

    def _residuals(
        self,
        pars,
    ):
        """2D Residual function to minimize"""
        v = pars.valuesdict()
        model = self._calc_model(v)

        resid = (self._fit_data - model) * self._fit_weights

        return np.squeeze(resid.flatten())

    def _init_parameters(self, init, fix):
        """Initialize lmfit parameters dictionary"""

        self._initial_guess(init)

        # initialize parameters
        pars = lmfit.Parameters()
        for vn, vinit in init.items():
            if vn in self.fitqdep:
                pars.add(vn)
            else:
                pars.add(vn, value=vinit[0], min=vinit[1], max=vinit[2], vary=1)
        pars.add("a", value=init["a"][0], min=init["a"][1], max=init["a"][2], vary=1)
        pars.add(
            "beta",
            value=init["beta"][0],
            min=init["beta"][1],
            max=init["beta"][2],
            vary=1,
        )

        self._init_pars_dataframe(init)
        self._make_varnames()

        # setting contrast constraint
        beta_constraint = self._get_beta_constraint(ndat=-1)
        pars["b0"].set(expr=beta_constraint)

        # setting parameters fixed
        for vn in fix.keys():
            if vn in pars.keys():
                pars[vn].set(value=fix[vn], min=-np.inf, max=np.inf, vary=0)

        # modifying pars for global fitting if not globalfit then ndat=1
        lpars = list(pars.items())[::-1]
        for j in range(self.ndat - 1):
            for pk, pv in sorted(lpars):
                if (pk not in self.fitglobal) or (pk in self.fitqdep):
                    pt = copy(pv)
                    pt.name += f"_{j}"
                    if pk == "b0":
                        pb0 = pt
                        beta_constraint = self._get_beta_constraint(ndat=j)
                        pb0.set(expr=beta_constraint)
                        continue
                    # print(pars)
                    # print(pt)
                    pars.add(pt)

            # add the constraint for b0
            pars.add(pb0)

        re_par = re.compile("[tgb]\d{1}(?=_)")
        for p in self.fitqdep:
            for new_par, par_kw in self.fitqdep[p]["pars"].items():
                pars.add(new_par, **par_kw)
            for j in range(self.ndat):
                vn = p + f"_{j-1}" * bool(j)
                q = self.qv[self.nq[j]]
                expr = copy(self.fitqdep[p]["expr"])
                cpars = re_par.findall(expr)
                for c in cpars:
                    expr = expr.replace(f"{c}_", self._varnames[c[0]][j][int(c[1])])
                expr = expr.replace("q", f"{q:.5f}")
                pars[vn].set(expr=expr)

        if sum([x.startswith("t") for x in self.fitglobal]) == 0:
            for j in range(self.ndat):
                for i in range(1, self.nmodes):
                    vc = f"t{i}" + f"_{j-1}" * bool(j)
                    vs = f"t{i-1}" + f"_{j-1}" * bool(j)
                    pars.add(vs + "_dtmp", value=pars[vc].value - pars[vs].value, min=0)
                    pars[vc].set(expr=f"{vs} + {vs}" + "_dtmp")

        # print(pars)
        pars._asteval.symtable["gamma"] = gamma
        self._lmpars = pars

    def _get_weights(self, mode):
        """
        mask data points to exclude them from the fit based on error bars and nan values
        and define weights for the fit
        """
        dcf = self.dcf
        cf = self.cf.copy()
        if dcf is not None:
            excerr = dcf.filled(0) <= 0
            wgt = dcf.copy()
            wgt = np.ma.masked_where(excerr, wgt)
            if mode == "semilogx":
                wgt = np.log10(wgt)
            elif mode == "semilogx2":
                wgt = 1 / np.log10(wgt / cf)
            elif mode == "semilogx3":
                wgt = 1 / (cf * np.log10(wgt / cf))
            elif (mode == "equal") or (mode == "none") or (mode == None):
                wgt = np.ones_like(wgt)
            elif mode == "logt":
                wgt = np.ones_like(cf) * np.log10(self.t)
            elif mode == "t":
                wgt = 1 / self.t
            elif mode == "sig**2":
                wgt = 1 / wgt ** 2
            elif mode == "sig":
                wgt = 1 / wgt
            elif mode == "logsig":
                wgt = 1 / np.log10(wgt)
            elif mode == "logsig+1":
                wgt = 1 / np.log10(wgt + 1)
            elif mode == "data":
                wgt = 1 / self.cf.copy()
            elif mode == "test":
                wgt = np.log10(self.t[0] / self.t + 1) / np.log10(wgt + 1)
            else:
                raise ValueError(f"Error mode {mode} not understood.")
        else:
            wgt = 1 / cf.copy()

        excdat = ~np.isfinite(self.cf) | wgt.mask
        self.cf = np.ma.masked_where(excdat, self.cf)

        self._weights = wgt

    def _init_pars_dataframe(self, init):
        """
        initialize pandas DataFrame for returning results
        """
        names = list(init.keys())
        cols = [
            names[i // 2] if (i + 1) % 2 else "d" + names[i // 2]
            for i in range(len(names) * 2)
        ]
        cols.insert(0, "q")
        cols.extend(["chisqr", "redchi", "bic", "aic"])
        self.pars = pd.DataFrame(columns=cols)

    def _initial_guess(self, init):
        """
        make initial guess for parameters if not provided by init
        """
        for i in range(self.nmodes):
            for s in "tgb":
                vn = s + "{}".format(i)
                if vn not in init.keys():
                    if s == "t":
                        t0 = np.logspace(
                            np.log10(self.t.min()),
                            np.log10(self.t.max()),
                            self.nmodes + 2,
                        )[i + 1]
                        init[vn] = (t0, -np.inf, np.inf)
                    elif s == "g":
                        init[vn] = (1, 0.2, 1.8)
                    elif s == "b":
                        init[vn] = (0.1, 0, 1)

        if "a" not in init:
            init["a"] = (1, 0, 2)
        if "beta" not in init:
            init["beta"] = (0.2, 0, 1)
        if bool(init.get("__lnsigma", False)):
            if not isinstance(init["__lnsigma"], tuple):
                init["__lnsigma"] = (np.log(0.1), None, None)

    def _get_beta_constraint(self, ndat=-1):
        dc = f"_{ndat}" * bool(ndat + 1)  # counter for fit_global
        dcb = "" if ("beta" in self.fitglobal) or ("beta" in self.fix) else dc
        dca = "" if ("a" in self.fitglobal) or ("a" in self.fix) else dc
        if self.nmodes > 1:
            beta_constraint = (
                "a"
                + dca
                + " + beta"
                + dcb
                + " - 1 - "
                + "-".join([f"b{x}" + dc for x in range(1, self.nmodes)])
            )
        else:
            beta_constraint = "beta" + dcb
        return beta_constraint

    def _make_varnames(self):
        """Make dict of varnames for easy handling in _calc_model"""
        self._varnames = {
            "t": [],
            "g": [],
            "b": [],
            "a": [],
            "beta": [],
        }

        if "__lnsigma" in self.pars.columns:
            self._varnames["__lnsigma"] = []

        for k in self._varnames.keys():
            cond = k in ["a", "beta", "__lnsigma"]
            for j in range(self.ndat):
                self._varnames[k].append([])
                jj = j - 1
                for i in range(self.nmodes):
                    ve = k + f"{i}" if not cond else k
                    if (ve not in self.fitglobal) or (ve in self.fitqdep):
                        ve += f"_{jj}" * bool(j)
                    self._varnames[k][j].append(ve)
                    if cond:
                        break

    def _write_to_pars(self, out, line=0):
        """Save fit results in self.pars variable"""
        for k in self._varnames.keys():
            cond = k in ["a", "beta", "__lnsigma"]
            gof = np.hstack([out.chisqr, out.redchi, out.bic, out.aic])
            for j in range(self.ndat):
                for i in range(self.nmodes):
                    df_name = k + f"{i}" if not cond else k
                    if (df_name not in self.fitglobal) or (df_name in self.fitqdep):
                        param_name = df_name + f"_{j-1}" * bool(j)
                    else:
                        param_name = df_name
                    value = out.params[param_name].value
                    try:
                        stderr = 1.0 * out.params[param_name].stderr
                    except TypeError:
                        stderr = np.nan
                    self.pars.loc[j + line, df_name] = value
                    self.pars.loc[j + line, "d" + df_name] = stderr

                self.pars.iloc[j + line, -4:] = gof
