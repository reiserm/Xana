import numpy as np

def resample( x, y, dy=None, npoints=100, log=True, method=0):
    """Resample data.
    """
    if dy is not None:
        mask = dy<=0
        y = np.ma.masked_array(y, mask=mask)
        dy = np.ma.masked_array(dy, mask=mask)
    
    if log:
        newx = np.logspace(np.log10(x.min()), np.log10(x.max()), npoints)
    else:
        newx = np.linspace(x.min(), x.max(), npoints)

    ind = np.digitize(x, newx)
    un_ind = np.unique(ind)
    newx = np.zeros(un_ind.size)
    newy = np.zeros((newx.size, y.shape[-1]))
    newdy = newy.copy()
    for j, i in enumerate(un_ind):
        ii = np.where((ind==i))[0]
        if method:
            m = np.mean(y[ii],0)
            tmp = np.abs(m[None,:]-y[ii])
            dy[ii][tmp>0] = tmp[tmp>0]
        newx[j] = np.average(x[ii],)
        newy[j], newdy[j] = np.ma.average(y[ii], weights=1/dy[ii]**2, axis=0, returned=1)
    newdy[newdy>0] = np.sqrt(1/newdy[newdy>0])
    return newx, newy, newdy


