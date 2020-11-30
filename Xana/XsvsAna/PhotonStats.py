import numpy as np
from .betaratio import betaratio
from ..Xfit.FitPoissonGammaLikelihood import fit_pg_likelihood


def prob2beta(prob, gproi):
    """Use betaratios to calculate the speckle contrast of a given data set of
    photon probabilities.
    """
    nq, nbins, ltimes = np.shape(prob[1:, 1:])
    beta = np.ma.array(np.empty((nq, nbins - 1, ltimes), dtype=np.float32))
    k = prob[0, 2:, 0]

    for j in range(nq):
        kb = prob[j + 1, 1]
        beta[j, 1:] = betaratio(k, kb, prob[j + 1, 2:], (ltimes, gproi[j]))[0]
        beta[j, 0] = kb

    return beta


def prob2betasigma(prob, gproi, sigma=3):
    """Use betaratios to calculate the speckle contrast of a given data set of
    photon probabilities using only kbar values in a sigma-sigma intercal.
    """
    nq, nbins, ltimes = np.shape(prob[1:, 1:])
    beta = np.ma.array(np.empty((nq, nbins - 1, 1), dtype=np.float32))
    k = prob[0, 2:, 0]
    for j in range(nq):
        kb = prob[j + 1, 1]
        mean_kb = np.mean(kb[kb > 0])
        std_kb = np.std(kb[kb > 0])
        ind = np.where(np.abs(kb - mean_kb) < std_kb * sigma)[0]
        prob_sel = prob[j + 1, 2:]
        prob_sel = np.mean(prob_sel[:, ind], -1)
        beta[j, 1:] = betaratio(k, mean_kb, prob_sel, (ltimes, gproi[j]))[0]
        beta[j, 0] = mean_kb

    return beta


def average_beta(t, q, contrast, ratio=0):
    """calculate the speckle contrast of a series of speckle patterns in three ways:
    1) calculating the arithmetik mean (v2_av)
    2) calculating the median (v2_md)
    3) calculating the value with the highest counts (v2_mx)
    """
    v2_av = np.zeros((t.size + 1, q.size + 1, 2), dtype=np.float32)
    v2_av[1:, 0, 0] = t
    v2_av[0, 1:, 0] = q
    v2_md = v2_av.copy()
    v2_mx = v2_av.copy()

    for it in range(t.size):
        for iq in range(q.size):
            beta = contrast[0][it][iq, ratio + 1]
            v2_av[it + 1, iq + 1] = np.ma.mean(beta), np.ma.std(beta)
            v2_md[it + 1, iq + 1] = np.ma.median(beta), np.ma.std(beta)
            h = np.histogram(beta[~beta.mask], bins=200, range=(-2, 8), normed=True)
            e = h[1][:-1] + np.diff(h[1][:2]) / 2
            v2_mx[it + 1, iq + 1] = e[np.argmax(h[0])], 0.05

    return v2_av, v2_md, v2_mx


def beta_from_likelihood(t, q, prob, npix, **kwargs):
    """Use likelihood ratio minimization to estimate the contrast."""
    v2_ml = np.zeros((t.size + 1, q.size + 1, 2), dtype=np.float32)
    v2_ml[1:, 0, 0] = t
    v2_ml[0, 1:, 0] = q
    for it in range(t.size):
        for iq in range(q.size):
            probi = prob[0][it][iq + 1]
            v2_ml[it + 1, iq + 1] = fit_pg_likelihood(probi, npix[iq], **kwargs)[0]

    return v2_ml
