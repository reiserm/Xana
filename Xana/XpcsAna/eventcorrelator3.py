import numpy as np
from collections.abc import Iterable
from .xpcsmethods import bin_multitau, mat2evt, ttc_to_g2

from .cpy_ecorr import fecorrt3m


def eventcorrelator(
    data, qroi, qv=None, dt=1.0, method="matrix", twotime_par=None, **kwargs
):
    """Calculate correlation function of event."""
    lqv = len(qroi)
    rlqv = range(lqv)

    # check if TTC should be stored
    if twotime_par is not None:
        if isinstance(twotime_par, int):
            twotime_par = [twotime_par]
        elif isinstance(twotime_par, Iterable):
            twotime_par = list(twotime_par)
        else:
            raise ValueError("Unsupported twotime_par type: " f"{type(twotime_par)}")

    if qv is None:
        qv = np.arange(lqv)

    print("\nRunning eventcorrelator", flush=1)
    for roii in rlqv:
        if method == "matrix":
            roi = data[:, qroi[roii][0], qroi[roii][1]]
            ntimes, npix = np.shape(roi)
            pix, t, s = mat2evt(roi)
        elif method == "events":
            npix = qroi[roii][0].size
            pix, t, s = data[roii]
            pix = pix.astype("int32")
            t = t.astype("int32")
            s = s.astype("int32")
            ntimes = s.size

        # initialize variables
        if roii == 0:
            ttcf = {}
            cfmt = []
            trace = []
            cc = np.zeros((ntimes + 1, lqv + 1), np.float32)
            tt = np.arange(1, ntimes + 1) * dt
            cc[0, 1:] = qv
            cc[1:, 0] = tt
            chi4 = cc.copy()

        indpi = np.lexsort((t, pix))
        t = t[indpi]
        pix = pix[indpi]

        lpi = len(pix)
        cor = np.zeros((ntimes, ntimes), "int32")
        cor = np.asfortranarray(cor)

        # the eventcorrelator
        cor = fecorrt3m(pix, t, cor, lpi, ntimes)
        lens = len(s)
        s = s.astype(dtype=np.float32)
        cor = np.array(cor, dtype=np.float32)
        s.shape = (lens, 1)
        trace.append(s)
        norm = np.dot(s, np.flipud(s.T)) / ntimes
        cor = cor / norm * npix / ntimes

        # getting the diagonal entries
        tmp = np.mean(np.diag(cor, k=1))
        ntimes = cor.shape[0]
        for i in range(ntimes):
            cor[i, i] = tmp

        if twotime_par is not None:
            if roii in twotime_par:
                ttcf[roii] = cor.copy()

        g2 = ttc_to_g2(cor, time=None)

        cc[1:, roii + 1] = g2[:, 1]
        chi4[1:, roii + 1] = g2[:, 2] ** 2

        # rebin the correlation function
        cfmt.append(
            bin_multitau(
                np.vstack((tt, cc[1:, roii + 1])).T, variance=chi4[1:, roii + 1]
            )
        )

    ntimebins = cfmt[0].shape[0]
    corf = np.empty((ntimebins + 1, len(qv) + 1))
    corf[1:, 0] = cfmt[0][:, 0]
    corf[0, 1:] = qv
    dcorf = corf.copy()
    for i in rlqv:
        corf[1:, i + 1] = cfmt[i][:, 1]
        dcorf[1:, i + 1] = cfmt[i][:, 2]

    trace = np.squeeze(np.array(trace)).T

    corfd = {
        "corf": corf,
        "dcorf": dcorf,
        "corf_full": cc,
        "dcorf_full": chi4,
        "trace": trace,
        "qv": qv,
        "qroi": qroi,
        "twotime_corf": ttcf,
        "twotime_xy": tt,
        "twotime_par": twotime_par,
    }
    return corfd
