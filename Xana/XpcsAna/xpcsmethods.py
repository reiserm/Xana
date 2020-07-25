import numpy as np
import pdb

def mat2evt(roi):
    '''Function to convert matrix of npix x ntimes
       into a vector of events
    '''
    ntimes, npix = roi.shape
    s = np.sum(roi,1).astype(np.int32)
    t = np.zeros(s.sum(),np.int32)
    s0 = 0
    css = np.append(0,np.cumsum(s))
    for i in range(css.size-1):
        t[css[i]:css[i+1]] = i
    pix = t.copy()

    eventpix = np.tile(np.arange(npix),ntimes).astype(np.int32)
    roi = roi.ravel() 
    ind = np.where(roi)
    roi = roi[ind]
    csroi = np.append(0,np.cumsum(roi)).astype(np.int32)
    eventpix = eventpix[ind]
    for i in range(len(csroi)-1):
        pix[csroi[i]:csroi[i+1]] = eventpix[i]

    return pix, t, s

def cftomt(d, par=16, err2=None):
    '''Function to bin the correlation function. Returns correlation
       functions similar to the multi tau approach.
    '''
    nt = []
    nd = []
    er = []

    if err2 is None:
        err2 = np.ones(d.shape[0], dtype=d.dtype)

    # err2 = np.ma.masked_array(err2)
    err2[err2==0] = np.nan

    for i in range(par):
        nt.append(d[i,0])
        nd.append(d[i,1])
        er.append(np.sqrt(err2[i]))

    t = d[par:,0]
    val = d[par:,1]
    err2 = err2[par:]

    while len(t) >= par:
        invsvar = 1/(1/err2[:-1] + 1/err2[1:])
        nval = (val[:-1]/err2[:-1] + val[1:]/err2[1:]) * invsvar
        tt = (t[:-1] + t[1:])/2
        for i in range(0,par,2):
            nt.append(tt[i])
            nd.append(nval[i])
            er.append(np.sqrt(invsvar[i]))
        t = tt[par:-1]
        val = nval[par:-1]
        err2 = invsvar[par:-1]

    # for i in range(len(t)):
    #     nt.append(t[i])
    #     nd.append(val[i])
    #     er.append(err2[i])

    x = np.array([nt,nd,er]).T
    return x

