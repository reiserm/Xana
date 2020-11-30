import numpy as np
from time import time


def mp_prob(method, nbins, nf, lind, nq, quc, quce):
    def xsvs_full(chunk):
        for qi in xnq:
            roi = chunk[qi]
            for i, line in enumerate(roi):
                tmp = np.append(
                    np.sum(line), np.histogram(line, bins=np.arange(nbins + 1) - 0.5)[0]
                )
                prob[qi, :, t0 + i] = tmp / lind[qi]

    tcalc = time()
    xnq = range(nq)
    prob = np.zeros((nq, nbins + 1, nf), dtype=np.float32)

    t0 = 0
    while t0 < nf:
        chunk = quc.get()
        chunk_size = chunk[0].shape[0]
        xsvs_full(chunk)
        t0 += chunk_size

    # END OF MAIN LOOP put results to output queue
    quc.close()
    quc.join_thread()
    tcalc = time() - tcalc
    quce.put([prob, tcalc])
