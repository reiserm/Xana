#! /usr/bin/env python
import numpy as np
from misc.imgkicker import imgkicker
from scipy.stats import norm


def find_kb(sel, dsel, kbr):
    if (len(kbr) == 1) or (kbr[0] == kbr[1]):
        ind = np.append(0, np.argmin(np.abs(sel[1:, 0] - kbr[0])) + 1)
    else:
        ind = np.append(
            0, np.where((sel[1:, 0] > kbr[0]) & (sel[1:, 0] < kbr[1]))[0] + 1
        )
    if ind.size == 1:
        raise ValueError("No data found in kb range.")
    sel = sel[ind, :]
    dsel = dsel[ind, :]
    return sel, dsel


def findk(kr, k_range):
    iin = np.array([np.where(kr == ki)[0][0] if ki in kr else -1 for ki in k_range])
    iout = np.append(0, np.where(iin >= 0)[0] + 1)
    iin = np.append(0, np.delete(iin, np.where(iin < 0)) + 1)
    if iin.size == iout.size == 1:
        raise ValueError(
            "Could not find any k values defined in k_range. Mean counts: {:.2g}".format(
                np.mean(kr)
            )
        )
    return iin, iout


def get_all(sel, d1, ind, xsvs, dxsvs, roi_idx, k_range, indk, kbr):
    # init output array
    sel = np.vstack((sel, np.zeros((np.sum(d1), sel.size))))
    dsel = sel.copy()
    for ii, i in enumerate(ind):
        kr = xsvs[i][roi_idx, 0, 1:]
        iin, iout = findk(kr, k_range)
        dind = np.arange(1 + np.sum(d1[:ii]), 1 + np.sum(d1[: ii + 1]))
        sel[dind[:, None], iout] = xsvs[i][roi_idx, indk[ii][:, None], iin]
        dsel[dind[:, None], iout] = dxsvs[i][roi_idx, indk[ii][:, None], iin]
    sel, dsel = find_kb(sel, dsel, kbr)
    return sel, dsel


def selectdata(
    xsvs,
    t,
    k_range=None,
    t_int=None,
    dxsvs=None,
    method="all",
    roi_idx=0,
    t_short=0.0,
    t_long=np.inf,
    t_readout=0.0,
    att=0,
    bw=1.0,
    series=None,
    kicker="imgkicker",
    kbr=(0, np.inf),
    pbthres=None,
    verbose=False,
):
    """method to select photon probablity data"""

    # define k-range
    if k_range is None:
        k_range = np.arange(20)

    # first column of output array contains k values
    sel = np.append(0, k_range)

    # define integration time vector if not passed
    if t_int is None:
        t_int = np.arange(len(xsvs))

    if dxsvs is None:
        dxsvs = np.ones_like(xsvs).astype(np.uint8)

    ind = np.where(t_int == t)[0]
    if ind.size == 0:
        raise ValueError("Exposure time {:.2g} was not found.".format(t))
    if verbose:
        print("Found {} series: {}.".format(len(ind), ind))
    if series is not None:
        ind = ind[series]

    # use imagekicker to select only a certain time region from a data set
    # the image indices of each series are saved in the list indk
    # d1 contains the number of 1s (or selected images) in indk
    d1 = []
    indk = []
    for ii, i in enumerate(ind):
        if type(kicker) == str:
            if kicker == "imgkicker":
                tmp = imgkicker(t, xsvs[i].shape[1], t_short, t_long, t_readout, att)
                tmp[:2] = 0
                indk.append(np.where(tmp)[0])
        else:
            indk.append(kicker + 1)
        d1.append(indk[ii].size)
    d1 = np.array(d1)

    ###---MAIN PART--- select data based on four different methods:
    ###--------------- all -- bin_all -- bin_series -- bin
    # get probability data for from all images
    if method == "all":
        sel, dsel = get_all(sel, d1, ind, xsvs, dxsvs, roi_idx, k_range, indk, kbr)

    # average over entire series by fitting with a Gaussian distribution
    elif method == "bin_series":
        # init output array
        sel = np.vstack((sel, np.zeros((ind.size, sel.size))))
        dsel = sel.copy()
        for ii, i in enumerate(ind):
            tmp = np.zeros((d1[ii], sel.shape[-1]))
            tmperr = tmp.copy()
            kr = xsvs[i][roi_idx, 0, 1:]
            iin, iout = findk(kr, k_range)
            tmp[:, iout] = xsvs[i][roi_idx, indk[ii][:, None], iin]
            tmperr[:, iout] = dxsvs[i][roi_idx, indk[ii][:, None], iin]
            tmp, tmperr = find_kb(tmp, tmperr, kbr)
            for r in range(np.shape(tmp)[1]):
                col = tmp[1:, r]
                gz = np.isfinite(col)  # np.where(col>0)[0]
                mu, std = (np.nanmean(col[gz]), np.nanstd(col[gz]))
                sel[ii + 1, r] = mu
                dsel[ii + 1, r] = std

    # average over every single data point
    elif method == "bin_all":
        sel, dsel = get_all(sel, d1, ind, xsvs, dxsvs, roi_idx, k_range, indk, kbr)
        dsel[1, :] = np.nanstd(dsel[1:, :], axis=0)
        sel[1, :] = np.nanmean(sel[1:, :], axis=0)
        dsel = dsel[:2, :]
        sel = sel[:2, :]

    # more advanced binning
    elif method == "bin":
        selt, dselt = get_all(sel, d1, ind, xsvs, dxsvs, roi_idx, k_range, indk, kbr)
        kv = selt[0, :]

        # define bins
        if bw < 1:
            if bw < 0:
                bw *= -1
                logscale = 1
            else:
                logscale = 0
            kb = sel[1:, 0]
            kbrange = np.max(kb[kb < np.inf]) - np.min(kb[kb > 0])
            bw *= kbrange
            nb = np.ceil(kbrange / bw).astype(np.int32)

            if nb >= kb.size:
                raise ValueError("More bins than data points")

            bins = np.linspace(np.min(kb[kb > 0]), np.max(kb[kb < np.inf]), nb + 1)
            if logscale:
                bins = np.logspace(np.log10(bins[-1]), np.log10(bins[0]), nb + 1)
                bins = bins[::-1]

            tmp = sel.copy()
            sel = np.zeros((nb, kv.size))
            dsel = sel.copy()
            # calculated mean and standard deviation of histograms
            for ii in range(bins.size - 1):
                idx = (kb >= bins[ii]) & (kb < bins[ii + 1])
                ttmp = tmp[idx, :]
                for s in range(kv.size):
                    mu, std = (np.nanmean(ttmp[:, s]), np.nanstd(ttmp[:, s]))
                    if np.isnan(mu):
                        print(ttmp[:, s])

                    sel[ii, s] = mu
                    dsel[ii, s] = std

        elif bw >= 1:

            for ii in range(len(ind)):
                if ii == 0:
                    sels = selt[1 : d1[ii], :]
                    dsels = dselt[1 : d1[ii], :]
                else:
                    sels = selt[1 + d1[ii - 1] : 1 + d1[ii], :]
                    dsels = dselt[1 + d1[ii - 1] : 1 + d1[ii], :]
                r = int(np.shape(sels)[0] % bw)
                if r > 0:
                    sels = sels[:-r]
                    dsels = dsels[:-r]
                nkb = int(np.shape(sels)[0] / bw)
                nk = sels.shape[1]
                sels = np.mean(np.reshape(sels, (bw, nkb, nk), "F"), axis=0)
                dsels = np.mean(np.reshape(dsels, (bw, nkb, nk), "F"), axis=0)
                if ii == 0:
                    sel = np.vstack((selt[0, :], sels))
                    dsel = np.vstack((dselt[0, :], dsels))
                else:
                    sel = np.vstack((sel, sels))
                    dsel = np.vstack((dsel, dsels))

    kv = sel[0, 1:]
    kb = sel[1:, 0]
    dkb = dsel[1:, 0]
    pba = sel[1:, 1:]
    dpba = dsel[1:, 1:]

    if pbthres is not None:
        # print('Length of pba is {}'.format(len(pba)))
        # print(pba)
        for i in range(len(pba)):
            if i == 0:
                ind = (pba[i] > pbthres).astype(np.bool)
            else:
                ind = ind & (pba[i] > pbthres).astype(np.bool)

        if all(~ind):
            raise ValueError("All pba below threshold.")
        ind = np.where(ind)[0]
        kv = kv[ind]
        pba = pba[:, ind]
        dpba = dpba[:, ind]

    # return selected data
    return kv, kb, dkb, pba, dpba
