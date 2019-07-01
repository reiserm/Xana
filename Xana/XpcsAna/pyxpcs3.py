from time import time
import numpy as np
import pickle as pkl
from multiprocessing import Process, Queue
from .mp_corr3_err import mp_corr
from scipy.optimize import leastsq
from ..misc.progressbar import progress
import sys
from matplotlib import pyplot as plt


def rebin( a, newshape ):
    """Rebin an array to a new shape.
    """
    assert len(a.shape) == len(newshape)
    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]

def errfunc(pa,xdata,ydata):
    """Fit function for fitting the variance of the two-time correlation function.
    """
    return (pa[0]+pa[1]*xdata-ydata)/np.sqrt(ydata*1e-8)

def trc(matr):
    """Calculate the two-time correlation function.
    """
    meanmatr = np.mean(matr, axis=1)
    meanmatr[meanmatr<=0] = 1.
    tmp, lenmatr = np.shape(matr)
    meanmatr.shape = 1, tmp
    trcm = np.dot(matr, matr.T) / lenmatr / np.dot(meanmatr.T, meanmatr)
    return trcm

def vartrc(ttc):
    """Calculate the variance of the two-time correlation function.
    """
    pc0 = [1.0,0.1]
    n,tmp = np.shape(ttc)
    vtmp = []
    for it in range(1,n-1):
        #ydata=diag(ttc,it)
        #xdata=arange(1,len(ydata)+1)
        #p1,success=leastsq(errfuncc,pc0,args=(xdata,ydata))
        #vtmp.append(var(ydata/(p1[0]+p1[1]*xdata)))              
        vtmp.append(np.var(np.diag(ttc,it)))         
    return vtmp

def recurf(ll):
    """Helper function used for calculating the two-time correlation function.
    """
    global l, y, v
    y[ll+1].append( (y[ll][0]+y[ll][1]) * 0.5)
    y[ll]=[]
    v[ll+1].append(vartrc(y[ll+1][-1]))
    if l[ll+1] == 1:
        recurf(ll+1)
    else:
        l[ll+1] += 1
    l[ll]=0  

def avr(saxs, ctr=-1, mask=None):
    """Old version of normalization function
    """
    dim1, dim2 = np.shape(saxs)

    if mask is None:
        mask = np.ones((dim1,dim2))
    saxs = saxs*mask
    
    if ctr == -1:
        return np.ones((dim1,dim2))*np.mean(saxs)
    cx,cy = ctr
    [X,Y] = np.mgrid[1-cy:dim1+1-cy,1-cx:dim2+1-cx]
    q = np.round(np.sqrt(X**2+Y**2)).astype(np.int64)

    q = q.ravel()
    mask = mask.flatten()
    saxs = saxs.flatten()

    qm = list(range(int(q.max()+1)))
    qr = list(range(len(qm)))
    for i in qm:
        qr[i] = []
    for i in range(len(q)):
        if (mask[i]):
            qr[q[i]].append(i)
    while [] in qr:
        qr.remove([])
    for i in qr:
        saxs[i] = np.mean(saxs[i])
    return saxs.reshape(dim1,dim2)

def avr_better( saxs, ctr, mask ):
    """Return an average saxs image for normalization of images.
    """
    cx, cy = ctr
    dim1, dim2 = np.shape(saxs)
    [X,Y] = np.mgrid[1-cy : dim1+1-cy, 1-cx : dim2+1-cx]
    q = np.float32(np.sqrt(X**2 + Y**2))
    n = np.int16(q + 0.5)
    q[mask==0] = 0
    n[mask==0] = 0
    max_n = n.max() + 1
    mean_saxs = np.zeros(max_n+1, np.float32)
    new_saxs = np.zeros_like(saxs, np.float32)
    radi = np.zeros((max_n,2), np.float32)
    
    for i in range(max_n):
        ind = np.where((n==i) & (mask==1))
        if ind[0].size:
            mean_saxs[i] = np.mean(saxs[ind])
            
    for i in range(dim1):
        for j in range(dim2):
            if q[i,j] > 0:
                par = int(q[i,j])
                f1 = q[i,j] - par
                if mean_saxs[par+1]>0 and mean_saxs[par]>0 :
                    new_saxs[i,j] = mean_saxs[par+1]*f1 + mean_saxs[par]*(1-f1) 
                if mean_saxs[par+1]>0 and mean_saxs[par]==0 :
                    new_saxs[i,j] = mean_saxs[par+1]
                if mean_saxs[par+1]==0 and mean_saxs[par]>0 :
                    new_saxs[i,j] = mean_saxs[par]
    return new_saxs

def calculate_twotime_correlation_function(ttdata, tt_max_images=5000):
    """Calculate two-time correlation function:
    The input ttdata has to be a numpy array of shape (nimages, npixels).
    """
    global l, v, y

    print("Start calculating TRC and Chi4...")

    nf, lind = ttdata.shape
    if nf > tt_max_images: # two-time correlation function is calculated for max tt_max_images
        ttchunk = nf // tt_max_images
        nfnew = ttchunk * tt_max_images
        print('Reducing two-time correlation data from {} to {} images by rebinning.'.format(nf,tt_max_images))
        ttdata = np.mean(ttdata[:nfnew].reshape(ttchunk,-1,lind, order='F'), 0)

    lind2 = lind // 16 
    l = np.zeros(5)
    y=[]
    v=[]
    for i in range(5):
        y.append([])
        v.append([])

    ib = 0
    for i in range(16):
        ie = ib + lind2
        y[0].append(trc(ttdata[:,ib:ie]))
        v[0].append(vartrc(y[0][-1]))
        if l[0]==1:
            recurf(0)
        else:
            l[0] += 1
        ib += lind2

    vm = []
    for i in range(4,-1,-1):
        vm.append(np.mean(v[i],0))
    vm = np.array(vm)

    del ttdata
    del v

    ttcf = y[4][-1] # this is the two-time correlation function

    dia1 = np.mean(np.diag(ttcf,1))
    t = np.arange(np.shape(ttcf)[0])
    ttcf[t,t] = dia1

    N = np.array([1, 2, 4, 8, 16]) / float(lind)
    z = vm.T/N
    #p0=[0,1]
    #it=range(len(ttcf[1:,0]))
    #p1=zeros((len(ttcf[1:,0]),len(p0)+1))
    #p1[:,0]=(asfarray(it)+1.0)*dt 
    #xdata=ttcf[0,:]
    #for i in it:
    #    ydata=ttcf[i+1,:]
    #    p1[i,1:], success = leastsq(errfunc, p0, args=(xdata,ydata))

    return ttcf, z



#####################
#---MAIN FUNCTION---#
#####################
def pyxpcs( data, qroi, dt=1., qv=None, saxs=None, mask=1., ctr=(0,0), twotime_par=-1,
            qsec=(0,0), norm='symmetric_whole', nprocs=1, verbose=True, chn=16,
            tt_max_images=5000):
    """Calculate g2 correlation functions with a given dataset or chunks of a data set.
    """
    time0 = time()
    chn2 = int(chn/2)
    lqv = len(qroi)
    rlqv = range(lqv)

    if qv is None:
        qv = np.arange(lqv)

    if isinstance(data, np.ndarray):
        nf, *dim = np.shape(data)
        def get_chunk():
            return (0, data)
    elif isinstance(data, dict):
        nf = data['nimages']
        dim = data['dim']
        def get_chunk():
            return data['dataQ'].get()

    if verbose:
        print('Number of images is:', nf)
        print('shape of image section is:', dim)

    if isinstance(mask, np.ndarray):
        mask = mask[qsec[0]:qsec[0]+dim[0],qsec[1]:qsec[1]+dim[1]]
        if saxs is not None and saxs.shape!=mask.shape:
            saxs = saxs[qsec[0]:qsec[0]+dim[0],qsec[1]:qsec[1]+dim[1]]

   # normalize with average saxs image
    if saxs is not None:
        print('Start computing SAXS for normalization.')
        dim2, dim1 = np.shape(saxs)
        saxs_img = saxs.copy()
        ctr = (ctr[0]-qsec[1],ctr[1]-qsec[0])
        saxs_img = saxs_img * mask
        saxs_img = avr_better(saxs_img, ctr, mask)
        saxs_imgc = np.ones((dim2,dim1))
        saxs_img[saxs_img==0] = 1.
        for i in rlqv:
            q0 = qroi[i][0] - qsec[0]
            q1 = qroi[i][1] - qsec[1]
            saxs_imgc[q0,q1] = np.mean(saxs_img[q0,q1])/saxs_img[q0,q1]

        # saxs_imgc[np.where(np.isinf(saxs_imgc))] = 1.0
        print('Done')

        if verbose:
            print('Shape of saxs_img:',  np.shape(saxs_img))
            print('Sum of saxs_img:', np.sum(saxs_img))
    if verbose:
        print('Number of ROIs: ', lqv)

    lind = []
    total_pixels = 0
    for iq in rlqv:
        npixel = len(qroi[iq][0])
        lind.append(npixel)
        total_pixels += npixel

    nprocs = min(nprocs,lqv) # cannot use more processes than q-values
    tmp_pix = 0
    if nprocs >= lqv:
        q_sec = np.arange(lqv+1)
    else:
        q_sec = [lqv,]
        for iq in rlqv[::-1]:
            tmp_pix += lind[iq]
            if tmp_pix >= np.floor(total_pixels/nprocs):
                q_sec.append(iq)
                total_pixels -= tmp_pix
                nprocs -= 1
                tmp_pix =  0
            if iq == 0 or nprocs == 0:
                q_sec.append(0)
    q_sec = np.unique(q_sec)
    del tmp_pix
    nprocs = len(q_sec) - 1
    print('Using {} processes.'.format(nprocs))

    #----twotime----
    # writing data for two time correlatino function calculation to tttdata
    if twotime_par != -1:
        ttdata = np.empty((nf,lind[twotime_par]), dtype=np.float32)
    #-------------

    srch = int(np.ceil(np.log2(nf/chn))) + 1
    rcr =  int(chn + chn2*(srch-1)) 

    print('Number of registers is {} with {} total correlation points.'.format(srch, rcr))

    lag = np.zeros(rcr, dtype=np.float32)    # initialize lag time vector

    for ir in range(srch):
        if ir == 0:
            lag[:chn] = np.arange(1,chn+1)
        else:
            sl = slice(chn2*(ir+1),chn2*(ir+2))
            lag[sl] = 2**ir*np.arange(1+chn2,chn+1)

    rcrc = rcr - np.where(lag[sl]>nf)[0].size - 1
    lag *= dt # scale lag-vector with time step
    
    trace = np.empty((nf,lqv))
    tt_vec = np.linspace(0,nf,tt_max_images)*dt

    #----multiprocessing----
    # create lists of queues and processes
    qur = []  
    qure = []  
    pcorr = []
    for i in range(nprocs):
        qur.append(Queue(16))
        qure.append(Queue(1))
    for i in range(nprocs):
        q_beg = q_sec[i]
        q_end = q_sec[i+1]
        pcorr.append(Process(target=mp_corr, args=(i, nf-1, chn, srch, rcr,
                lind[q_beg:q_end], q_end-q_beg, qur[i], qure[i])))
        
    # start processes
    for i in range(nprocs):
        pcorr[i].start()
    #-----------------------

    # start processing data
    tcalc_cum = 0
    t0 = 0
    last_chunk = -1
    lin_mask = np.where(mask)
    while t0 < nf - 1:
        progress(t0,nf)

        c_idx, chunk = get_chunk()
        chunk_size = chunk.shape[0]
        idx = slice(t0,t0+chunk_size)
        # matr[matr<0] = 0

        chunk_diff = c_idx - last_chunk
        if chunk_diff != 1:
            raise IOError('Chunks have been read in wrong order: chunk index difference is % and not 1.' %
                             chunk_diff)
        last_chunk = c_idx
        
        if saxs is not None:
            chunk = chunk * saxs_imgc # normalize with mean saxs image

        # save data for two time correlation
        if twotime_par != -1:
            ttdata[idx,:] = chunk[:,qroi[twotime_par][0]-qsec[0],qroi[twotime_par][1]-qsec[1]]

        for jj,(i,j) in enumerate(zip(q_sec[:-1], q_sec[1:])):
            tmp_put = []
            for qi in range(i,j):
                q0 = qroi[qi][0] - qsec[0]
                q1 = qroi[qi][1] - qsec[1]
                trace[idx,qi] = chunk[:,q0,q1].mean(-1)
                if norm == 'symmetric':
                    normfactor = trace[idx,qi].copy()
                    normfactor[normfactor==0] = 1.
                    tmp_put.append(chunk[:,q0,q1]/normfactor[:,None])
                elif norm == 'symmetric_whole':
                    tmp_put.append(chunk[:,q0,q1]/np.mean(chunk[:,lin_mask[0],lin_mask[1]]))
                elif norm == 'none':
                    tmp_put.append(chunk[:,q0,q1])
                elif norm == 'corrcoef':
                    tmp_mat = chunk[:,q0,q1]/trace[idx,qi,None]
                    tmp_put.append((tmp_mat-tmp_mat.mean(-1)[:,None])
                                   /np.sqrt(np.var(tmp_mat,-1))[:,None])
            qur[jj].put(tmp_put)

        t0 += chunk_size

    progress(1,1)
        
    # read data from output queue
    from_proc = []
    for i in range(nprocs):
        from_proc.append(qure[i].get())
        pcorr[i].join()
        qure[i].close()
        qure[i].join_thread()
        
    # get correlation functions and normalization from processes
    corf = from_proc[0][0]
    dcorf = from_proc[0][1]
    nk = from_proc[0][2]
    sr = from_proc[0][3]
    sl = from_proc[0][4]
    tcalc_cum = from_proc[0][5]
    for i in range(1,nprocs):
        corf = np.concatenate((corf,from_proc[i][0]), axis=1)
        dcorf = np.concatenate((dcorf,from_proc[i][1]), axis=1)
        sr = np.concatenate((sr,from_proc[i][3]), axis=1)
        sl = np.concatenate((sl,from_proc[i][4]), axis=1)
        tcalc_cum = max(tcalc_cum,from_proc[i][5])

    if norm in ['symmetric', 'sym_!trace', 'symmetric_whole', 'none']:
        tmp = nk[:rcrc,None]**2/(sr[:rcrc]*sl[:rcrc])
        corf = corf[:rcrc] * tmp
        dcorf = np.abs(dcorf[:rcrc] * tmp**2 / (nk[:rcrc,None]**2))
    elif norm == 'corrcoef':
        corf = corf[:rcrc]
        dcorf = np.abs(dcorf[:rcrc])

    # initialize correlation array 'cc'
    cc = np.zeros((rcrc+1,lqv+1), dtype=np.float32)
    dcc = cc.copy()
    cc[1:,0] = lag[:rcrc]
    cc[1:,1:] = corf
    dcc[1:,0] = lag[:rcrc]
    dcc[1:,1:] = np.sqrt(dcorf)
    cc[0,1:] = qv
    dcc[0,1:] = qv
    
    if verbose:
        print("\rFinished calculating correlation functions.") 

    del corf, dcorf, rcr, srch, lag, from_proc


    #----twotime and chi4----
    tcalc_cumtrc0 = time()
    if twotime_par != -1:
        ttcf, chi4 = calculate_twotime_correlation_function(ttdata, tt_max_images)
    else:
        ttcf = chi4 = 0

    tcalc_cumtrc = time()-tcalc_cumtrc0
    if verbose:
        print('Elapsed time: {:.2f} min'.format((time()-time0)/60.))
        print('Elapsed time for correlate: {:.2f} min'.format(tcalc_cum/60.))
        print('Elapsed time for TRC and Chi4: {:.2f} min'.format(tcalc_cumtrc/60.))

    corfd = {'corf':cc,
             'dcorf':dcc,
             'trace':trace,
             'qv':qv,
             'qroi':qroi,
             'Isaxs':saxs,
             'mask':mask,
             'twotime_corf':{twotime_par:ttcf},
             'twotime_par':twotime_par,
             'twotime_xy':tt_vec,
             'chi4':chi4
    }

    return corfd
