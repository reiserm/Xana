import os
import numpy as np
import multiprocessing as mp
from readDet.openFile_mp import getEDFimagedata
import time


def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (
        new_shape[0],
        arr.shape[0] // new_shape[0],
        new_shape[1],
        arr.shape[1] // new_shape[1],
    )
    return arr.reshape(shape).mean(-1).mean(1)


def make_spacer(nf):
    chn = 16
    chn2 = int(chn / 2)
    srch = int(np.ceil(np.log2(nf / chn))) + 1
    rcr = int(chn + chn2 * (srch - 1))

    lag = np.zeros(rcr, dtype=np.float32)  # initialize lag time vector

    for ir in range(srch):
        if ir == 0:
            lag[:chn] = np.arange(1, chn + 1)
        else:
            sl = slice(chn2 * (ir + 1), chn2 * (ir + 2))
            lag[sl] = 2 ** ir * np.arange(1 + chn2, chn + 1)

    rcrc = rcr - np.where(lag[sl] > nf)[0].size - 1
    return lag[:rcrc].astype(np.uint16)


def xsvs_roi(imgr):
    global nb, nb2
    kb = np.round(np.mean(imgr))
    ne = np.max([0, np.floor(np.log2(np.max([kb, 1]) / nb2))])
    if (kb <= 0) or (ne == 0):
        kbr = np.arange(nb + 1) - 0.5
        bc = kbr[:-1] + 0.5
    elif (kb > 0) and (ne > 0):
        kbr = np.arange(0, (2 ** ne + 1) * (nb + 1), 2 ** ne + 1) - 0.5
        bc = kbr[:-1] + 2 ** (ne - 1) + 0.5

    nt, gp = np.shape(imgr)
    sr = np.zeros((nt + 1, nb + 1))
    dsr = sr.copy()
    sr[0, 1:] = bc
    dsr[0, 1:] = bc

    for j in range(nt):
        imgi = imgr[j, :]
        tmp = np.append(np.sum(imgi), np.histogram(imgi, kbr)[0])
        sr[j + 1, :] = tmp / gp
        dsr[j + 1, :] = np.sqrt(tmp) / gp

    return sr, dsr


def calc_xsvs_pproc(img, qroi, pool):
    global nb
    last = np.shape(img)[0]
    roi = np.zeros((len(qroi), last + 1, nb + 1), np.float32)
    droi = np.zeros_like(roi)

    workin = []
    for qi in qroi:
        workin.append(img[:, qi[0], qi[1]])
    del img

    for i, res in enumerate(pool.map(xsvs_roi, workin)):
        roi[i] = res[0]
        droi[i] = res[1]
    return roi, droi


def sum_data(data, spacer, nt, method):
    # Sum images to calculate Pairs or Joerg function
    sumdata = np.zeros((nt, *data.shape[:-1]), dtype=np.float32)
    for image_num in range(nt):
        imi = image_num
        if method == 2:
            imf = image_num + spacer
            S = np.sum(data[:, :, imi:imf], axis=-1)
        elif method == 1:
            imf = image_num + spacer - 1
            if imi == imf:
                S = data[:, :, imi]
            else:
                S = data[:, :, imi] + data[:, :, imf]
        sumdata[:, :, image_num] = S
    return sumdata


def sum_events(events, spacer, nt, method):
    """Sum events to calculate Pairs or Joerg function"""
    nq = len(events[0])
    sumevents = [[] for i in range(nt)]
    for image_num in range(nt):
        imi = image_num
        if method == 2:
            imf = image_num + spacer
            S = [np.hstack(events[imi:imf][iq]) for iq in range(nq)]
        elif method == 1:
            imf = image_num + spacer - 1
            if imi == imf:
                S = events[imi]
            else:
                S = [np.append(events[imi][iq], events[imf][iq]) for iq in range(nq)]
        sumevents[image_num] = S
    return sumevents


def evt2prob(events, qroi, nbins):
    ltimes = len(events)
    nq = len(events[0])
    prob = np.empty((nq, ltimes + 1, nbins + 1), np.float32)
    kvec = np.append(0, np.arange(nbins))
    for it in range(ltimes):
        for iq in range(nq):
            counts = np.unique(events[it][iq], return_counts=1)[1]
            tmp = np.histogram(counts, nbins - 1, range=(1, nbins - 1))[0]
            npix = qroi[iq][0].size
            prob[iq, 0, :] = kvec
            prob[iq, it + 1, :] = (
                np.hstack((counts.sum(), npix - counts.size, tmp)) / npix
            )
    return prob, prob * 0.0


# --MAIN FUNCTION---
def pyxsvs(
    datin,
    qroi,
    last=None,
    method="xsvs",
    filen=0,
    data_format="edfid10",
    data_form="events",
    max_spacer=1,
    nspacer=1,
    dt=(1.0, 1.0),
    nbins=15,
    nproc=4,
):

    time_i = time.time()

    provided_methods = {"xsvs": 0, "pairs": 1, "joerg": 2}
    method = provided_methods[method]

    readmethods = {
        "edfid10": getEDFimagedata,
    }
    getDataFunc = readmethods[data_format]

    if nproc:
        pool = mp.Pool(processes=nproc)

    # initialize output array
    xsvs = []
    dxsvs = []

    if method == 0:
        lags = tv = None
        nfiles = len(datin)
        for n, dati in enumerate(datin):
            if data_form == "events":
                roi, droi = evt2prob(dati, qroi, nbins)
            else:
                img = getDataFunc(dati, output=0)
                roi, droi = calc_xsvs_pproc(img, qroi, pool)
            xsvs.append(roi)
            dxsvs.append(droi)

    elif method:
        datin = datin[filen]
        if data_form == "events":
            levt = len(datin)
            lags = make_spacer(levt)
            print("Using lags: {}".format(lags))
            for inds, spacer in enumerate(lags):
                nt = levt - spacer + 1
                roi = np.zeros((len(qroi), nt + 1, nbins + 1), np.float32)
                droi = np.zeros_like(roi)

                Sevt = sum_events(datin, spacer, nt, method)
                roi, droi = evt2prob(Sevt, qroi, nbins)

                xsvs.append(roi)
                dxsvs.append(droi)
        else:
            img = getDataFunc(datin, last=(last,), nprocs=4, dtype=np.int32)
            print("Data loaded.", flush=1)
            for inds, spacer in enumerate(lags):
                nt = img.shape[0] - spacer + 1
                roi = np.zeros((len(qroi), nt + 1, nbins + 1), np.float32)
                droi = np.zeros_like(roi)

                Sdata = sum_data(img, spacer, nt, method)
                roi, droi = calc_xsvs_pproc(Sdata, qroi, pool)
                xsvs.append(roi)
                dxsvs.append(droi)

        tv = lags * dt[0] + (lags - 1) * dt[1]

    if nproc:
        pool.close()
        pool.join()

    time_f = time.time() - time_i
    print("Elapsed time: {}min".format(round(time_f / 60, 2)))
    return {"xsvs": xsvs, "dxsvs": dxsvs, "time": tv, "spacer": lags}
