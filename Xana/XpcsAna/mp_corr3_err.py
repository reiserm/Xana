import numpy as np
from time import time


def mp_corr(
    nf,
    chn,
    nreg,
    lags,
    npix,
    nq,
    queue_in=None,
    queue_out=None,
    data=None,
    use_mp=True,
    verbose=False,
):
    """Multi-tau correlator."""

    def verbose0(reg):
        # some output for testing purposes
        print("\nn: ", n)
        print("reg: ", reg)
        # print("sst {}\nsr:    {}\nsl:    {}".format(sst, sr, sl))
        print("g2c: ", g2c)
        # print("datreg: ", datreg[reg])
        # print("corf: ", corf)

    def verbose1(reg):
        # some output for testing purposes
        print("\nn: ", n)
        print("reg: ", reg)
        print("sst {}\nsr:    {}\nsl:    {}".format(sst, sr, sl))
        print("srr {}\nsll {}".format(srr[reg], sll[reg]))
        print("g2c: ", g2c)
        # print("corf: ", corf)
        # print("datreg: ", datreg[reg])

    def running_mean(m, s, x, k):
        mt = 1.0 * m
        m += (x - m) / k
        s += (x - mt) * (x - m)

    def correlator(matrx):

        for qi in xnq:
            k = 0
            # update mean
            sst[qi][k] += matrx[qi].mean()

            # update register
            datreg[qi][k, n % chn] = matrx[qi].copy()
            if qi == 0:
                regc[k] += 1

            for l in range(chn - 1):
                if n >= l + 1:
                    corf[qi][l] += (
                        datreg[qi][k, n % chn] * datreg[qi][k, (n - l - 1) % chn]
                    )
                    srr[qi][l] += datreg[qi][k, n % chn]
                    sll[qi][l] += datreg[qi][k, (n - l - 1) % chn]
                    if qi == 0:
                        g2c[l] += 1

            # print(np.max(srr[0]), np.max(sll[0]))
            k += 1
            correlator2(k, qi)

    def correlator2(k, qi):
        if ((n + 1) % 2 ** k) == 0:
            datreg[qi][k, regc[k] % chn] = (
                datreg[qi][k - 1, (regc[k - 1] - 2) % chn]
                + datreg[qi][k - 1, (regc[k - 1] - 1) % chn]
            ) / 2.0
            if qi == 0:
                regc[k] += 1

        for c in range(chn2):
            l = (k + 1) * chn2 + c - 1
            nn = chn2 * 2 ** k + (c + 1) * 2 ** k
            # if qi == 0:
            #     print("(n,k,c,l,nn)=", n,k,c,l,nn)
            if (((n + 1) % 2 ** k) == (nn % 2 ** k)) and ((n + 1) >= nn):
                # if qi == 0:
                #     print('Compute g2')
                #     print("regc mod chn", regc[k]%chn)
                #     print("regc", regc[k])
                #     print(l, (g2c[l]+chn2+c)%chn, g2c[l]%chn)
                corf[qi][l] += (
                    datreg[qi][k, (g2c[l] + chn2 + c) % chn]
                    * datreg[qi][k, g2c[l] % chn]
                )
                srr[qi][l] += datreg[qi][k, (g2c[l] + chn2 + c) % chn]
                sll[qi][l] += datreg[qi][k, g2c[l] % chn]
                if qi == 0:
                    g2c[l] += 1

        if (k + 1) < nreg:  # and not (n + 1) % 2**(k + 1):
            k += 1
            correlator2(k, qi)

    # initialization of variables
    tcalc = time()
    chn2 = int(chn / 2)
    xnq = range(nq)

    # initialize counters
    nlag = len(lags)
    g2c = np.zeros(nlag, dtype=np.int32)
    regc = np.zeros(nreg, dtype=np.int32)
    sst = np.zeros((nq, nreg), dtype=np.float32)

    # initialize registers
    datreg = []
    sll = []
    srr = []
    corf = []
    for iq in xnq:
        datreg.append(np.zeros((nreg, chn, npix[iq]), dtype=np.float32))
        sll.append(np.zeros((nlag, npix[iq]), dtype=np.float32))
        srr.append(np.zeros((nlag, npix[iq]), dtype=np.float32))
        corf.append(np.zeros((nlag, npix[iq]), dtype=np.float32))

    # END of declaring and initializing variables####
    n = 0
    while n < nf:
        if use_mp:
            chunk = queue_in.get()
        else:
            chunk = data
        chunk_size = chunk[0].shape[0]
        ni = 0
        while ni < chunk_size:
            matr = [qchunk[ni] for qchunk in chunk]
            correlator(matr)
            ni += 1
            n += 1

    dcorf = []
    for qi in xnq:
        corf[qi] /= g2c[:, None]
        srr[qi] /= g2c[:, None]
        sll[qi] /= g2c[:, None]
        # if qi == 0:
        #     print(srr[qi], sll[qi])
        srr[qi] = np.ma.masked_invalid(srr[qi])
        sll[qi] = np.ma.masked_invalid(sll[qi])
        srr[qi] = np.mean(srr[qi], axis=1)  # / npix[qi]
        sll[qi] = np.mean(sll[qi], axis=1)  # / npix[qi]
        # print(corf[qi].shape, npix[qi], srr)
        corf[qi] = corf[qi] / (srr[qi][:,None] * sll[qi][:,None])
        dcorf.append(1 / np.sqrt(npix[qi]) * np.std(corf[qi], axis=1))
        corf[qi] = np.mean(corf[qi], axis=1)

    # print('corf: ', corf)
    # print('\nsr: ', sr)
    # print('\nsl: ', sl)
    # print('\ng2c: ',g2c)

    tcalc = time() - tcalc
    if use_mp:
        # END OF MAIN LOOP put results to output queue
        queue_in.close()
        queue_in.join_thread()
        queue_out.put([corf, dcorf, tcalc])
    else:
        return corf, dcorf, tcalc
