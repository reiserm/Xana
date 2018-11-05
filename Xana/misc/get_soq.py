import numpy as np

def get_soq(saxs, mask, cx, cy, setup=None):
    dim1, dim2 = np.shape(saxs)
    [X,Y] = np.mgrid[1-cy:dim1+1-cy,1-cx:dim2+1-cx]
    q = np.int32(np.sqrt(X**2+Y**2))
    q = q.ravel()
    saxs = saxs.ravel()
    mask = mask.ravel()
    qm = list(range(int(q.max()+1)))
    qr = list(range(len(qm)))
    soq = np.zeros(len(qr))
    qv = np.arange(len(soq)).astype(np.float32)
    if setup is not None:
        lambdaw = setup['lambdaw']
        pix_size = setup['pix_size']
        distance = setup['distance']
        wf = 4*np.pi/lambdaw
        qmat = wf*np.sin(np.arctan(np.sqrt(X**2+Y**2)*pix_size/distance)/2)
        qmat = qmat.ravel()
    else:
        qmat = q
    for i in qm:
        qr[i] = []
    for i in range(len(q)):
        qr[round(q[i])].append(i)
    while [] in qr:
        qr.remove([])
    for j,i in enumerate(qr[1:]):
        soq[j+1] = np.sum(saxs[i])/np.sum(mask[i])
        qv[j+1] = np.mean(qmat[i])
        saxs[i] = soq[j+1]
    return soq, qv, np.reshape(saxs,(dim1,dim2))