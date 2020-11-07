from time import time
import numpy as np
import pickle as pkl
from multiprocessing import Process, Queue
from .mp_prob import mp_prob
from ..misc.progressbar import progress
import sys


def pyxsvs(
    data,
    qroi,
    nbins=15,
    t_e=1.0,
    qv=None,
    method="full",
    nprocs=1,
    verbose=1,
    qsec=(0, 0),
):
    """Calculate photon proababilities."""
    time0 = time()
    lqv = len(qroi)
    rlqv = range(lqv)

    if qv is None:
        qv = np.arange(lqv)

    if isinstance(data, np.ndarray):
        nf, dim2, dim1 = np.shape(data)

        def get_chunk():
            return data

    elif isinstance(data, dict):
        nf = data["nimages"]

        def get_chunk():
            return data["dataQ"].get()

    if verbose:
        print("Number of images is:", nf)
        print("Loading data in chunks.")
        print("Number of ROIs: ", lqv)

    lind = []
    total_pixels = 0
    for iq in rlqv:
        npixel = len(qroi[iq][0])
        lind.append(npixel)
        total_pixels += npixel

    nprocs = min(nprocs, lqv)  # cannot use more processes than q-values
    tmp_pix = 0
    if nprocs >= lqv:
        q_sec = np.arange(lqv + 1)
    else:
        q_sec = [
            lqv,
        ]
        for iq in rlqv[::-1]:
            tmp_pix += lind[iq]
            if tmp_pix >= np.floor(total_pixels / nprocs):
                q_sec.append(iq)
                total_pixels -= tmp_pix
                nprocs -= 1
                tmp_pix = 0
            if iq == 0 or nprocs == 0:
                q_sec.append(0)
    q_sec = np.unique(q_sec)
    del tmp_pix
    nprocs = len(q_sec) - 1
    print("Using {} processes.".format(nprocs))

    trace = np.empty((nf, lqv))

    # ----multiprocessing----
    # create lists of queues and processes
    qur = []
    qure = []
    pcorr = []
    for i in range(nprocs):
        qur.append(Queue(16))
        qure.append(Queue(1))
    for i in range(nprocs):
        q_beg = q_sec[i]
        q_end = q_sec[i + 1]
        pcorr.append(
            Process(
                target=mp_prob,
                args=(
                    method,
                    nbins,
                    nf,
                    lind[q_beg:q_end],
                    q_end - q_beg,
                    qur[i],
                    qure[i],
                ),
            )
        )

    # start processes
    for i in range(nprocs):
        pcorr[i].start()
    # -----------------------

    # start processing data
    tcalc_cum = 0
    t0 = 0
    last_chunk = -1
    while t0 < nf:
        progress(t0, nf)

        c_idx, chunk = get_chunk()
        chunk_size = chunk.shape[0]
        idx = slice(t0, t0 + chunk_size)

        chunk_diff = c_idx - last_chunk
        if chunk_diff != 1:
            raise IOError(
                "Chunks have been read in wrong order: chunk index difference is % and not 1."
                % chunk_diff
            )
        last_chunk = c_idx

        for jj, (i, j) in enumerate(zip(q_sec[:-1], q_sec[1:])):
            tmp_put = []
            for qi in range(i, j):
                q0 = qroi[qi][0] - qsec[0]
                q1 = qroi[qi][1] - qsec[1]
                roi = chunk[:, q0, q1]
                trace[idx, qi] = roi.mean(-1)
                tmp_put.append(roi)
            qur[jj].put(tmp_put)

        t0 += chunk_size

    progress(1, 1)

    # read data from output queue and shut down processes
    from_proc = []
    for i in range(nprocs):
        from_proc.append(qure[i].get())
        pcorr[i].join()
        qure[i].close()
        qure[i].join_thread()

    # concatenate data from different processes
    p = from_proc[0][0]
    tcalc_cum = from_proc[0][1]
    for i in range(1, nprocs):
        p = np.concatenate((p, from_proc[i][0]), axis=0)
        tcalc_cum = max(tcalc_cum, from_proc[i][1])

    # initialize correlation array 'cc'
    prob = np.zeros((lqv + 1, nbins + 2, nf), dtype=np.float32)
    prob[1:, 1:] = p
    prob[0, 0, 0] = t_e
    prob[1:, 0, 0] = qv
    prob[0, 1:, 0] = np.append(0, np.arange(nbins))

    if verbose:
        print("\rFinished calculating correlation functions.")

    del p, from_proc

    if verbose:
        print("Elapsed time: {:.2f} min".format((time() - time0) / 60.0))
        print(
            "Elapsed time for calulating probabilities: {:.2f} min".format(
                tcalc_cum / 60.0
            )
        )

    probd = {
        "prob": prob,
        "t_exposure": t_e,
        "trace": trace,
        "qv": qv,
        "qroi": qroi,
    }

    return probd
