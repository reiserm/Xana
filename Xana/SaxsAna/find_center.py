import numpy as np
from matplotlib import pyplot as plt
from lmfit.models import GaussianModel as gauss_mod
from lmfit.models import VoigtModel as voigt_mod
from lmfit.models import LinearModel as lin_mod


def find_center(img, mask=None, doplot=False, fit_report=False):

    mod = voigt_mod() + lin_mod()
    if mask is None:
        mask = np.ones_like(img).astype("bool")

    ctr = np.zeros(2)

    if doplot:
        fig, ax = plt.subplots(2, 1)

    for i in range(2):
        dat = np.ma.masked_array(img, mask=~(mask).astype("bool"))
        dat = dat.mean(i)
        par = mod.make_params()
        x = np.arange(len(dat))
        par["center"].set(np.mean(dat * x) / dat.mean())
        par["slope"].set(0, vary=False)
        par["intercept"].set(dat.mean())
        res = mod.fit(dat, params=par, x=x)

        if doplot:
            res.plot_fit(ax=ax.flat[i])
            if i:
                ax.flat[i].set_title("")
        if fit_report:
            print(res.fit_report())

        ctr[i] = res.params["center"].value
    return ctr.round().astype("int16")
