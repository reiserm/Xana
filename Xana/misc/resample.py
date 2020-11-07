import numpy as np


def resample(x, y, dy=None, npoints=100, log=True, method=0):
    """Resample data."""
    x = np.ma.masked_where(np.isnan(x), x)
    y = np.ma.masked_where(np.isnan(y), y)

    if y.ndim == 1:
        y = np.expand_dims(y, 1)
        if dy is not None:
            dy = np.expand_dims(dy, 1)

    if dy is not None:
        mask = (dy <= 0) | np.isnan(y)
        y = np.ma.masked_array(y, mask=mask)
        dy = np.ma.masked_array(dy, mask=mask)
        returned = True
    else:
        returned = False
        dy = np.ones_like(y)

    if log:
        newx = np.logspace(np.log10(x.min()), np.log10(x.max()), npoints + 1)
    else:
        newx = np.linspace(x.min(), x.max(), npoints + 1)

    ind = np.digitize(x.filled(0), newx)
    un_ind = np.unique(ind)
    newx = np.zeros(un_ind.size)
    newy = np.zeros((newx.size, y.shape[-1]))
    newdy = newy.copy()
    for j, i in enumerate(un_ind):
        ii = np.where((ind == i))[0]
        if y[ii].filled(0).sum() == 0:
            newx[j] = np.nan
            newy[j] = np.nan
            continue
        if method:
            m = np.nanmean(y[ii], 0)
            tmp = (m[None, :] - y[ii]) ** 2 / m[None, :] ** 2
            dy[ii][tmp > 0] = tmp[tmp > 0]
        newx[j] = np.ma.average(
            x[ii],
        )
        newy[j], newdy[j] = np.ma.average(
            y[ii], weights=1 / dy[ii] ** 2, axis=0, returned=1
        )
    # newx = newx[1:]
    # newy = newy[1:]
    # newdy = newdy[1:]
    newdy[newdy > 0] = np.sqrt(1 / newdy[newdy > 0])
    if returned:
        return newx, newy, newdy
    else:
        return newx, newy
