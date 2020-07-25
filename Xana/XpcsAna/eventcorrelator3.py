from time import time
import warnings
import numpy as np
import sys
from scipy.ndimage import gaussian_filter
from ..misc.progressbar import progress
from .xpcsmethods import cftomt, mat2evt

from .cpy_ecorr import fecorrt3m

import pdb
#---MAIN FUNCTION---
def eventcorrelator(data, qroi, qv=None, dt=1., method='matrix',
                    twotime_par=-1, **kwargs):
    '''
    Event correlator
    '''
    time0 = time()
    lqv = len(qroi)
    rlqv = range(lqv)
    twotime_par = np.array(twotime_par)

    if qv is None:
        qv = np.arange(lqv)

    for roii in rlqv:
        print('\nAnalyzing ROI: {}'.format(roii), flush=1)
        if method == 'matrix':
            roi = data[:,qroi[roii][0],qroi[roii][1]]
            ntimes, npix = np.shape(roi)
            pix, t, s = mat2evt(roi)
        elif method == 'events':
            npix = qroi[roii][0].size
            pix, t, s = data[roii]
            ntimes = s.size
            # # old code
            # ntimes = len(data)
            # npix = qroi[roii][0].size
            # pix, t, s = [], [], []
            # for i, dati in enumerate(data):
            #     lpix = len(dati[roii])
            #     pix.append(dati[roii])
            #     t.append(np.zeros(lpix)+i)
            #     s.append(lpix)
            # pix = np.concatenate(pix)
            # t = np.concatenate(t)
            # s = np.asarray(s)

        if roii == 0:
            ttcf = {}
            cfmt = []
            trace = []
            cc = np.zeros((ntimes,lqv+1), np.float32)
            tt = np.arange(1,ntimes+1)*dt
            cc[0,1:] = qv
            z = cc.copy()

        indpi = np.argsort(pix)
        t = t[indpi]
        pix = pix[indpi]

        lpi = len(pix)
        cor = np.zeros((ntimes, ntimes), 'int32')
        print('starting fortran routine', flush=1)
        cor = np.asfortranarray(cor)
        cor = fecorrt3m(pix, t, cor, lpi, ntimes)
        lens = len(s)
        s = s.astype(dtype=np.float32)
        cor = np.array(cor, dtype=np.float32)
        s.shape = (lens, 1)
        norm = np.dot(s, np.flipud(s.T)) / ntimes
        cor = cor / norm * npix / ntimes
        tmp = np.mean(np.diag(cor, k=1))
        for i in range(ntimes - 1):
            cor[i,i] = tmp

        x = np.ones((ntimes-1,3))
        x[:,0] = np.arange(1,ntimes)
        for i in range(1,ntimes-1):
            dia = np.diag(cor,k=i)
            ind = np.where(np.isfinite(dia))
            if len(dia[ind]):
                x[i-1,1] = np.mean(dia[ind])
                x[i-1,2] = np.std(dia[ind])
        x[:,2] *= np.sqrt(1.0/(ntimes-1))
        x[:,0] *= dt
        cc[1:,roii+1] = x[:,1]
        z[1:,roii+1] = x[:,2]**2
        if roii == 0:
            cc[1:,0] = x[:,0]
            z[1:,0] = x[:,0]
        del x
        cfmt.append(cftomt(cc[1:,[0,roii+1]],err2=z[1:,roii+1]))
        trace.append(s)

        if roii in twotime_par:
            ttcf[roii] = cor.copy()
        del cor

    shp = cfmt[0].shape[0]
    corf = np.empty((shp+1,len(qv)+1))
    corf[1:,0] = cfmt[0][:,0]
    corf[0,1:] = qv
    dcorf = corf.copy()
    for i in rlqv:
        corf[1:,i+1] = cfmt[i][:,1]
        dcorf[1:,i+1] = cfmt[i][:,2]

    trace = np.squeeze(np.array(trace)).T

    corfd = {'corf':corf,
             'dcorf':dcorf,
             'corf_full':cc,
             'dcorf_full':z,
             'trace':trace,
             'qv':qv,
             'qroi':qroi,
             'twotime_corf':ttcf,
             'twotime_xy':tt
    }
    return corfd
