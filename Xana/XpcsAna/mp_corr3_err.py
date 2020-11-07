import numpy as np
from time import time


def mp_corr(nf, chn, srch, rcr, lind, nq, quc=None, quce=None, data=None, use_mp=True):
    """Multi-tau correlator."""

    def verbose0(reg):
        return
        # some output for testing purposes
        print("\nn: ", n)
        print("reg: ", reg)
        print("sst {}\nsr:    {}\nsl:    {}".format(sst, sr, sl))
        print("nk: ", nk)
        print("datreg: ", datreg[reg])
        print("corf: ", corf)

    def verbose1(reg):
        return
        # some output for testing purposes
        print("\nn: ", n)
        print("reg: ", reg)
        print("sst {}\nsr:    {}\nsl:    {}".format(sst, sr, sl))
        print("srr {}\nsll {}".format(srr[reg], sll[reg]))
        print("nk: ", nk)
        print("corf: ", corf)
        print("datreg: ", datreg[reg])

    def running_mean(m, s, x, k):
        mt = 1.0 * m
        m += (x - m) / k
        s += (x - mt) * (x - m)

    def correlator(reg, matrx):

        if n < chn:
            if n:
                nk[:n] += 1
            for cindex in xnq:
                sr[cindex, :n] = np.roll(sr[cindex, :n], 1)
                sr[cindex, 0] = sr[cindex, 1] + sst[reg][cindex]
                sst[reg][cindex] = matrx[cindex].mean()
                sl[cindex, :n] += sst[reg][cindex]

            for cindex in xnq:
                x = np.dot(matrx[cindex], datreg[reg][cindex][:, :n]) / lind[cindex]
                running_mean(corf[cindex, :n], dcorf[cindex, :n], x, nk[:n])
                datreg[reg][cindex] = np.concatenate(
                    (matrx[cindex][:, None], datreg[reg][cindex][:, : chn - 1]), axis=1
                )
            verbose0(reg)
            if n % 2:
                for cindex in xnq:
                    matrx[cindex] = (
                        datreg[reg][cindex][:, 0] + datreg[reg][cindex][:, 1]
                    ) / 2.0

                correlator2(1, matrx)
        else:
            nk[:chn] += 1
            for cindex in xnq:
                sr[cindex, :chn] = np.roll(sr[cindex, :chn], 1)
                sr[cindex, 0] = sr[cindex, 1] + sst[reg][cindex]
                sst[reg][cindex] = matrx[cindex].mean()
                sl[cindex, :chn] += sst[reg][cindex]

                x = np.dot(matrx[cindex], datreg[reg][cindex]) / lind[cindex]
                running_mean(corf[cindex, :chn], dcorf[cindex, :chn], x, nk[:chn])

                datreg[reg][cindex] = np.concatenate(
                    (matrx[cindex][:, None], datreg[reg][cindex][:, : chn - 1]), axis=1
                )

            verbose0(reg)
            if n % 2:
                for cindex in xnq:
                    matrx[cindex] = (
                        datreg[reg][cindex][:, 0] + datreg[reg][cindex][:, 1]
                    ) / 2.0

                correlator2(1, matrx)

    def correlator2(reg, matrx):
        condition = (reg + 1) < srch and (n + 1) % 2 ** (
            reg + 1
        ) == 0  # if condition is true, move to
        kreg = int((n + 1) / 2 ** reg)  # the next register
        # print('\n\ncondition: ', condition)
        # print('kreg: ', kreg)

        if 1 < kreg <= chn:
            for cindex in xnq:
                srr[reg][cindex, : kreg - 1] = np.roll(srr[reg][cindex, : kreg - 1], 1)
                srr[reg][cindex, 0] = srr[reg][cindex, 1] + sst[reg][cindex]
                sst[reg][cindex] = matrx[cindex].mean()
                sll[reg][cindex, : kreg - 1] += sst[reg][cindex]

        elif kreg > chn:
            for cindex in xnq:
                srr[reg][cindex, :chn] = np.roll(srr[reg][cindex, :chn], 1)
                srr[reg][cindex, 0] = srr[reg][cindex, 1] + sst[reg][cindex]
                sst[reg][cindex] = matrx[cindex].mean()
                sll[reg][cindex, :chn] += sst[reg][cindex]

        if (chn2 + 1) < kreg <= chn:
            inb = chn2 * (reg + 1)
            ine = chn2 * reg + kreg - 1
            sl[:, inb:ine] = sll[reg][:, chn2 : kreg - 1]
            sr[:, inb:ine] = srr[reg][:, chn2 : kreg - 1]

            nk[inb:ine] += 1
            for cindex in xnq:
                x = (
                    np.dot(matrx[cindex], datreg[reg][cindex][:, chn2 : kreg - 1])
                    / lind[cindex]
                )
                # print('x: ', x)
                running_mean(
                    corf[cindex, inb:ine], dcorf[cindex, inb:ine], x, nk[inb:ine]
                )
                datreg[reg][cindex] = np.concatenate(
                    (matrx[cindex][:, None], datreg[reg][cindex][:, : chn - 1]), axis=1
                )
            verbose1(reg)

            if condition:
                for cindex in xnq:
                    matrx[cindex] = (
                        datreg[reg][cindex][:, 0] + datreg[reg][cindex][:, 1]
                    ) / 2.0
                reg += 1
                correlator2(reg, matrx)

        elif kreg > chn:
            inb = chn2 * (reg + 1)
            ine = chn2 * (reg + 2)
            sl[:, inb:ine] = sll[reg][:, chn2:chn]
            sr[:, inb:ine] = srr[reg][:, chn2:chn]

            nk[inb:ine] += 1
            for cindex in xnq:
                x = (
                    np.dot(matrx[cindex], datreg[reg][cindex][:, chn2:chn])
                    / lind[cindex]
                )
                # print('x: ', x)
                running_mean(
                    corf[cindex, inb:ine], dcorf[cindex, inb:ine], x, nk[inb:ine]
                )
                datreg[reg][cindex] = np.concatenate(
                    (matrx[cindex][:, None], datreg[reg][cindex][:, : chn - 1]), axis=1
                )

            verbose1(reg)
            if condition:
                for cindex in xnq:
                    matrx[cindex] = (
                        datreg[reg][cindex][:, 0] + datreg[reg][cindex][:, 1]
                    ) / 2.0
                reg += 1
                correlator2(reg, matrx)

        else:
            for cindex in xnq:
                sst[reg][cindex] = matrx[cindex].mean()
                datreg[reg][cindex] = np.concatenate(
                    (matrx[cindex][:, None], datreg[reg][cindex][:, : chn - 1]), axis=1
                )

            verbose1(reg)
            if condition:
                for cindex in xnq:
                    matrx[cindex] = (
                        datreg[reg][cindex][:, 0] + datreg[reg][cindex][:, 1]
                    ) / 2.0
                reg += 1
                correlator2(reg, matrx)

    # finished initializing part
    tcalc = time()
    chn2 = int(chn / 2)
    datregt = []
    datreg = []
    xnq = range(nq)

    for ir in range(srch):
        for iq in xnq:
            datregt.append(np.zeros((lind[iq], chn), dtype=np.float32))
        datreg.append(datregt)
        datregt = []
    del datregt

    corf = np.zeros((nq, rcr), dtype=np.float32)
    dcorf = corf.copy()
    sl = corf.copy()
    sr = corf.copy()
    nk = np.zeros(rcr, dtype=np.float32)

    sll = []
    srr = []
    sst = []
    for ir in range(srch):
        sll.append(np.zeros((nq, chn), dtype=np.float32))
        srr.append(np.zeros((nq, chn), dtype=np.float32))
        sst.append(np.zeros(nq))

    # END of declaring and initializing variables####
    n = 0
    while n < nf:
        if use_mp:
            chunk = quc.get()
        else:
            chunk = data
        chunk_size = chunk[0].shape[0]
        ni = 0
        while ni < chunk_size:
            matr = [qchunk[ni] for qchunk in chunk]
            correlator(0, matr)
            ni += 1
            n += 1

    # print('corf: ', corf)
    # print('\nsr: ', sr)
    # print('\nsl: ', sl)
    # print('\nnk: ',nk)

    tcalc = time() - tcalc
    if use_mp:
        # END OF MAIN LOOP put results to output queue
        quc.close()
        quc.join_thread()
        quce.put([corf.T, dcorf.T, nk, sr.T, sl.T, tcalc])
    else:
        return corf.T, dcorf.T, nk, sr.T, sl.T, tcalc
