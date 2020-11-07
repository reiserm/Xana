import numpy as np
from matplotlib import pyplot as plt
import pickle
from mpldatacursor import datacursor
import FitPoissonGamma
import MBetaFormulas
import FitBetaDist
import niceplot


class PhotonProbs(object):
    """docstring"""

    def __init__(self, run, darks, datdir="./", savdir="./"):
        """Constructor"""
        self.NValues = []
        self.Run = run
        self.Darks = darks
        self.Prob = []
        self.Beta = {
            "Fit": np.empty((0, 4), dtype=np.float32),
            "Formula": np.empty((0, 4), dtype=np.float32),
        }
        self.Beta_dat = np.empty((0, 4), dtype=np.float32)
        self.Filename = "tmp.pkl"
        self.dt = 1.0
        self.jfunc = dict()

    def add_probs(self, n, file):
        """Add photon probability matrix to PhotonProbs object."""
        probs = np.loadtxt(file)
        if n in self.NValues:
            np.concatenate((self.Prob[self.NValues == n], probs))
        else:
            self.Prob.append(probs)
            self.NValues.append(n)

    def calc_beta(self, n, k, method, disteval="Fit"):
        idx = self.NValues.index(n)
        kb = self.Prob[idx][:, 0]
        nkb = kb.size
        sp = np.ones((nkb, 1), dtype=np.uint8)
        if method == "Fit":
            pb = self.Prob[idx][:, k + 1]
            fit_res = FitPoissonGamma.FitPG(kb, pb, k)
            beta = fit_res[0][0] ** (-1)
            err = fit_res[1][0] * fit_res[0][0] ** (-2)
            pars = [n, k, beta, err]
        elif method == "Formula":
            if k == 12:
                pb = self.Prob[idx][:, 2:4]
                M, dM = MBetaFormulas.formula_M(kb, pb[:, 0], pb[:, 1])
                tmp = np.concatenate(
                    [sp * n, sp * k, 1.0 / M, 1.0 / M ** 2 * dM], axis=1
                )
            elif k == 0:
                pb = self.Prob[idx][:, 1]
                beta0, dbeta0 = MBetaFormulas.formula_beta0(kb, pb)
                tmp = np.concatenate([sp * n, sp * k, beta0, dbeta0], axis=1)
            elif k == 1:
                pb = self.Prob[idx][:, 1]
                beta1, dbeta1 = MBetaFormulas.formula_beta1(kb, pb)
                tmp = np.concatenate([sp * n, sp * k, beta1, dbeta1], axis=1)
            self.Beta_dat = np.vstack((self.Beta_dat, tmp))
            beta = tmp[:, 2]
            if disteval == "Fit":
                pars_fit = FitBetaDist.fitbetanorm(beta)
                pars = [n, k, pars_fit[0], pars_fit[1]]
            elif disteval == "mean":
                pars = [n, k, np.mean(beta), np.std(beta)]
        self.Beta[method] = np.vstack((self.Beta[method], pars))

    def plot_poissongamma(self, n, kmax=2):
        idx = self.NValues.index(n)
        kb = self.Prob[idx][:, 0]
        pb = self.Prob[idx][:, 1 : kmax + 2]
        for ik in np.arange(0, kmax + 1):
            plt.loglog(kb, pb[:, ik], "o")
            idx = (self.Beta["Fit"][:, 0] == n) & (self.Beta["Fit"][:, 1] == ik)
            plt.loglog(
                kb,
                FitPoissonGamma.PoissonGamma(kb, ik, self.Beta["Fit"][idx, 2] ** (-1)),
            )
        # niceplot.niceplot('plot_poissongamma')
        plt.show()

    def plot_jfunction(self, method, k, errf=1.0):
        self.make_jfunctions()
        t = self.jfunc[method + str(k)][:, 0]
        beta = self.jfunc[method + str(k)][:, 1]
        err = self.jfunc[method + str(k)][:, 2]
        pl = plt.errorbar(
            t, beta, err * errf, marker="o", label="J {}, k={}".format(method, k)
        )
        plt.legend(loc="best")
        # niceplot.niceplot('plot_jfunction')
        plt.show()

    def make_jfunctions(self):
        for mth in self.Beta:
            ki = np.unique(self.Beta[mth][:, 1]).astype(np.uint8)
            for k in ki:
                idx = np.where(self.Beta[mth][:, 1] == k)[0]
                bdx = self.Beta[mth][idx, :]
                bdx[:, 0] = (bdx[:, 0] - 1) * self.dt
                bdx = bdx[:, [0, 2, 3]]
                ind = np.argsort(bdx[:, 0])
                self.jfunc[mth + str(k)] = bdx[ind, :]

    def set_beta(self, method, n, k, beta, err):
        idx = np.where((self.Beta[method][:, 0] == n) & (self.Beta[method][:, 1] == k))
        pars = [n, k, beta, err]
        if idx[0].size != 0:
            self.Beta[method][idx] = pars
        else:
            self.Beta[method] = np.vstack((self.Beta[method], pars))

    def load(self, update=True):
        f = open(self.Filename, "rb")
        tmp_dict = pickle.load(f)
        f.close()
        if update:
            self.__dict__.update(tmp_dict)
        elif not update:
            self.__dict__ = tmp_dict

    def save(self):
        f = open(self.Filename, "wb")
        pickle.dump(self.__dict__, f, 2)
        f.close()
