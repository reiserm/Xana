import numpy as np


def mat2evt(roi):
    """Function to convert matrix of npix x ntimes
    into a vector of events
    """
    ntimes, npix = roi.shape
    s = np.sum(roi, 1).astype(np.int32)
    t = np.zeros(s.sum(), np.int32)
    css = np.append(0, np.cumsum(s))
    for i in range(css.size - 1):
        t[css[i] : css[i + 1]] = i
    pix = t.copy()

    eventpix = np.tile(np.arange(npix), ntimes).astype(np.int32)
    roi = roi.ravel()
    ind = np.where(roi)
    roi = roi[ind]
    csroi = np.append(0, np.cumsum(roi)).astype(np.int32)
    eventpix = eventpix[ind]
    for i in range(len(csroi) - 1):
        pix[csroi[i] : csroi[i + 1]] = eventpix[i]

    return pix, t, s


def bin_multitau(d, par=16, variance=None):
    """Function to bin the correlation function. Returns correlation
    functions similar to the multi tau approach.
    """
    nt = []
    nd = []
    er = []

    if variance is None:
        variance = np.ones(d.shape[0], dtype=d.dtype)

    # variance = np.ma.masked_array(variance)
    variance[variance == 0] = np.nan

    for i in range(par):
        nt.append(d[i, 0])
        nd.append(d[i, 1])
        er.append(np.sqrt(variance[i]))

    t = d[par:, 0]
    val = d[par:, 1]
    variance = variance[par:]

    while len(t) >= par:
        invsvar = 1 / (1 / variance[:-1] + 1 / variance[1:])
        nval = (val[:-1] / variance[:-1] + val[1:] / variance[1:]) * invsvar
        tt = (t[:-1] + t[1:]) / 2
        for i in range(0, par, 2):
            nt.append(tt[i])
            nd.append(nval[i])
            er.append(np.sqrt(invsvar[i]))
        t = tt[par:-1]
        val = nval[par:-1]
        variance = invsvar[par:-1]

    # for i in range(len(t)):
    #     nt.append(t[i])
    #     nd.append(val[i])
    #     er.append(variance[i])

    x = np.array([nt, nd, er]).T
    return x


def ttc_to_g2(ttc, time=None):
    """Calculate g2 function from TTC

    Args:
        cor (np.ndarray): sqaure correlation matrix (TTC)
        time (np.ndarray, optional): 1D vector of lag times. Defaults to None.
            If None, np.arange will be used for evenly spaced time bins.
    """

    ntimes = ttc.shape[0]
    if time is None:
        time = np.arange(ntimes) + 1

    # initialize output array
    g2 = np.ones((ntimes, 3))
    g2[:, 0] = time
    for i in range(1, ntimes):
        dia = np.diag(ttc, k=i)
        ind = np.where(np.isfinite(dia))
        if len(dia[ind]):
            g2[i - 1, 1] = np.mean(dia[ind])
            g2[i - 1, 2] = np.std(dia[ind])
    g2[:, 2] *= np.sqrt(1.0 / (ntimes))
    return g2
