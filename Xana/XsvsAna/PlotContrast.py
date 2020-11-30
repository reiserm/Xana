import numpy as np
from matplotlib import pyplot as plt
from ..Xfit.MCMC_straight_line import mcmc_sl
from ..Xfit.fit_basic import fit_basic
from ..Xfit.FitPoissonGamma import fit_pg
from ..Xplot.niceplot import niceplot
from matplotlib.offsetbox import AnchoredText
from matplotlib import ticker
from ..misc.running_mean import running_mean
from ..misc.resample import resample
from matplotlib.colors import LogNorm
from matplotlib import ticker


def rebin_array(arr, maxlen=500):
    """rebin array and reduce its length to maxlen for generating plots"""
    l = arr.shape
    if l[0] > maxlen:
        arr = arr[: -(l[0] % maxlen)]
        arr = arr.reshape(maxlen, -1, *l[1:])
        arr = arr.mean(1)
    return arr


def plot_contrast_vs_images(
    x, y, ax, label=None, marker="o", color="b", ind_avr=False, alpha=0.5
):
    """"""
    ax.plot(
        x, y, ms=3, label=label, linestyle="", marker=marker, color=color, alpha=alpha
    )
    ax.set_xlabel("image number")
    ax.set_ylabel(r"$\beta$")
    if ind_avr:
        arrow(x.max(), y.mean(), ax, color)
    return True


def plot_contrast_histogram(x, ax, label=None, color="b", ind_avr=False):
    """"""
    ax.hist(
        x,
        bins=100,
        density=True,
        histtype="step",
        label=label,
        color=color,
    )
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$P(\beta)$")
    if ind_avr:
        xx = np.ones(2) * x.mean()
        h = np.histogram(x, bins=100, density=True)
        de = np.diff(h[1][:2]) / 2
        e = h[1][:-1] + de
        ymax = h[0][np.argmin(np.abs(xx[0] - h[1] + 0.5 * np.mean(np.diff(h[1]))))]
        yy = np.array([0, ymax])
        ax.plot(xx, yy, "--", color=color)
    return True


def plot_contrast_running_average(
    x, y, dy, ax, label=None, marker="o", color="b", ind_avr=False, alpha=0.5
):
    """"""
    ax.errorbar(x, y, dy, ms=3, label=label, fmt=marker, color=color, alpha=alpha)
    ax.set_xlabel("image number")
    ax.set_ylabel(r"$\beta$ (run. avr.)")
    return True


def plot_kbar_vs_images(
    x, y, ax, label=None, marker="o", color="b", npix=False, ind_avr=False, alpha=0.5
):
    """"""
    if npix:
        y = y * npix
        ylab = r"$k_{tot}$"
    else:
        ylab = r"$\langle k\rangle$"
    ax.plot(
        x,
        y,
        label=label,
        linestyle="",
        markersize=3,
        marker=marker,
        color=color,
        alpha=alpha,
    )
    ax.set_xlabel("image number")
    ax.set_ylabel(ylab)
    ax.set_yscale("log")
    if ind_avr:
        arrow(x.max(), y.mean(), ax, color)
    return True


def plot_poisson_gamma(x, y, dy, ax, label=None, marker="o", color="b", ind_avr=False):
    """"""
    ax.errorbar(x, y, yerr=dy, label=label, ms=3, ls="", marker=marker, color=color)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$P(k)$")
    ax.set_yscale("log")
    return True


def plot_contrast_kbar_correlation(
    x, y, ax, label=None, cmap="inferno", npix=False, ind_avr=False, alpha=0.5
):
    """"""
    if npix:
        x = x * npix
        xlab = r"$k_{tot}$"
    else:
        xlab = r"$\langle k\rangle$"
    ax.hist2d(
        x, y, bins=50, density=True, norm=LogNorm(), cmap=cmap, label=label, alpha=alpha
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"$\beta$")
    return True


def plot_contrast_vs_kbar(
    x, y, ax, label=None, marker="o", color="b", npix=False, ind_avr=False, alpha=0.5
):
    """"""
    if npix:
        x = x * npix
        xlab = r"$k_{tot}$"
    else:
        xlab = r"$\langle k\rangle$"
    ax.plot(
        x,
        y,
        linestyle="",
        marker=marker,
        markersize=2,
        color=color,
        label=label,
        alpha=alpha,
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"$\beta$")
    return True


def plot_prob_vs_kbar(
    x,
    y,
    ax,
    probk,
    label=None,
    marker="o",
    color="b",
    npix=False,
    ind_avr=False,
    alpha=0.5,
):
    """"""
    if npix:
        x = x * npix
        y = y * npix
        xlab = r"$k_{tot}$"
        ylab = r"n(%d)" % probk
    else:
        ylab = r"P(%d)" % probk
        xlab = r"$\langle k\rangle$"
    ax.plot(
        x,
        y,
        linestyle="",
        marker=marker,
        markersize=2,
        color=color,
        label=label,
        alpha=alpha,
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return True


def arrow(x, y, ax, color):
    c1 = (x, y)
    c2 = (x * 1.1, y)
    arprp = dict(
        arrowstyle="-|>",
        connectionstyle="arc3",
        color=color,
    )
    ax.annotate(
        "", xy=c1, xycoords="data", xytext=c2, textcoords="data", arrowprops=arprp
    )


def plot_quicklook(
    obj,
    plot,
    idx,
    nq,
    ax=None,
    c0=0,
    ratio=0,
    maxlen=500,
    format_ticks=False,
    ind_avr=True,
    lfs=10,
    total_counts=False,
    return_respfnc=False,
    probk=2,
    fit_pg=True,
    alpha=0.5,
    **kwargs
):
    """Plot quick overview of speckle contrast, mean photon counts and Probability Distribution"""
    contrast = obj.contrast[0][idx]
    prob = obj.prob[0][idx][1:, 1:]

    for i, j in enumerate(nq):
        ci = i + c0 * len(nq)  # index for colors and markers
        color = obj.colors[ci]
        marker = obj.markers[idx % len(obj.markers)]
        if total_counts:
            npix = obj.Xana.setup["gproi"][j]
        else:
            npix = False

        if i == 0:
            labstr = "t_exp: {:.2e}s".format(obj.t_exposure[idx])
        else:
            labstr = None

        b = contrast[j, ratio + 1]
        xv_b = np.where(~b.mask)[0]
        x_b = rebin_array(xv_b, maxlen)
        br = rebin_array(b[xv_b], maxlen)

        bt = running_mean(b[xv_b])
        bt = rebin_array(bt, maxlen)

        kb = contrast[j, 0]
        xv_kb = np.where(~kb.mask)[0]
        kb = rebin_array(kb[xv_b], maxlen)
        p = prob[j, probk + 1][xv_b]

        pp = np.mean(prob[j, 1:], -1)
        ind = np.where(pp)[0]
        dpp = np.std(prob[j, 1:], -1)
        k = np.arange(pp.size)

        if plot == "bvi":
            labmean = "%.2f" % br.mean()
            plot_contrast_vs_images(x_b, br, ax, labmean, marker, color, ind_avr, alpha)
        elif plot == "pbb":
            plot_contrast_histogram(b[~b.mask], ax, labstr, color, ind_avr)
        elif plot == "brvi":
            plot_contrast_running_average(
                x_b, bt[:, 0], bt[:, 1], ax, labstr, marker, color, alpha
            )
        elif plot == "kbvi":
            labmean = "%.3g" % kb.mean()
            plot_kbar_vs_images(
                x_b, kb, ax, labmean, marker, color, npix, ind_avr, alpha
            )
        elif plot == "bvkb":
            plot_contrast_vs_kbar(kb, br, ax, labstr, marker, color, npix, alpha)
        elif plot == "pbk":
            plot_poisson_gamma(k[ind], pp[ind], dpp[ind], ax, labstr, marker, color)
            # if fit_pg:
            #     pb_pg = np.hstack((np.zeros(k[ind].size)))
            #     fitpg()
        elif plot == "pkvkb":
            plot_prob_vs_kbar(kb, p, ax, probk, labstr, marker, color, npix, alpha)

    if ind_avr:
        ax.legend(loc="best")
    # niceplot(ax, autoscale=True, grid=False, lfs=lfs)


# def plot_poisson_gamma(obj, idx, nq, dofit=True):
#     """Show the photon statistics by plotting the Poisson-Gamma distribution
#     """
#         if data == 'original' or self.corrFuncRescaled is None:
#             corrFunc = list(self.corrFunc)
#         elif data == 'rescaled':
#             corrFunc = list(self.corrFuncRescaled)
#         else:
#             raise ValueError('No usable correlation data defined.')

#         if nq is None:
#             pass
#         elif type(nq) == int:
#             self.nq = np.arange(nq)
#         else:
#             self.nq = nq

#         if color_mode == 0:
#             color_multiplier, color_repeater = self.nq.size, len(corrFunc)
#         elif color_mode == 1:
#             color_multiplier, color_repeater = self.nq.size*len(corrFunc), 1

#         self.update_colors(cmap, color_multiplier, color_repeater)
#         self.update_markers( len(corrFunc), change_marker)

#         self.g2plotl = [[]] * len(corrFunc)
#         self.pars = [[]] * len(corrFunc)
#         self.fit_result = [[[] for i in range(self.nq.size)]] * len(corrFunc) # possible bug: will cause
#                                                                               # wrong references
#         ci = 0
#         for j, (cfi, dcfi) in enumerate(corrFunc):
#             rates = np.zeros((self.nq.size, 3*nmodes+3, 2))
#             ti = cfi[1:,0]
#             for i,qi in enumerate(self.nq):
#                 if i == 0:
#                     cf_id = self.db_id[j]
#                 else:
#                     cf_id = None
#                 res = fitg2(ti, cfi[1:,qi+1], err=dcfi[1:,qi+1], qv=self.Xana.setup['qv'][qi],
#                             ax=ax, color=self.colors[ci], dofit=True,
#                             marker=self.markers[j%len(self.markers)], cf_id=cf_id,
#                             modes=nmodes, **kwargs)
#                 self.fit_result[j][i] = res[2:4]
#                 self.g2plotl[j].append(list(itertools.chain.from_iterable(res[4])))
#                 if dofit:
#                     if i == 0:
#                         db_tmp = self.init_pars(list(res[2].params.keys()))
#                     entry = [cfi[0,qi+1], *res[0].flatten(), *res[1]]
#                     db_tmp.loc[i] = entry
#                 else:
#                     db_tmp = 0
#                 ci += 1
#             self.pars[j] = db_tmp
#         plt.tight_layout()


#     for j,ei in zip(tind,extf[tind]):
#         nn = np.ceil(len(roi_list)/2)
#         f, ax = plt.subplots(int(nn), 2, figsize=(9,nn*4))
#         f.suptitle('exposure time = {:.2g}s'.format(ei))
#         ax = ax.ravel()

#         for l,roii in enumerate(roi_list):
#             kv, kb, dkb, pba, dpba = selectdata(xsvs, ei, t_int=ext, dxsvs=xsvs_err, roi_idx=roii, **psel)

#             out = fpg.fitpg_iterative(kb, pba, kv, pberr=1/pba, **pfit)
#             M[roii,0,:] = qv[roii]
#             M[roii,j+1,:3] = (out[0],out[1],np.mean(out[2]))

#             plotpoissongamma(out[2], pba, kv, out[0], dkb, dpba*0, out[1],
#                              confint=(1/np.mean(cnorm[roii+2,np.where(cnorm[0,1:]==ei)]),),
#                              q=qv[roii], ax=ax[l], cmap='Set1',**pfit)

#         if doplot:
#             plt.tight_layout()
