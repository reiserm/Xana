import numpy as np
from .integrate import get_soq
import copy


def get_sec(img, mask, setup):

    if np.shape(img) != np.shape(mask):
        qsec = setup['qsec'][0]
        mask = mask.copy()
        dim = np.shape(img)
        mask = mask[qsec[0]:dim[0]+qsec[0], qsec[1]:dim[1]+qsec[1]]

        setup = copy.deepcopy(setup)
        setup['ctr'] = (setup['ctr'][0]-qsec[1], setup['ctr'][1]-qsec[0])
        return mask, setup
    else:
        return mask, setup


def pysaxs(data, load=False, calc_soq=True, **kwargs):

    if load:
        Isaxs = obj.get_item(sid)['Isaxs']
    elif isinstance(data, dict):
        sid = data['sid']
        mask = data['mask']
        setup = data['setup']
        Isaxs, Vsaxs = data['get_series'](sid, method='average', **kwargs)
        saxsd = {'Isaxs':Isaxs, 'Vsaxs':Vsaxs}
    else:
        raise ValueError('Could not handle input type during SAXS analysis.')

    if calc_soq and setup is not None and Isaxs.ndim==2:

        mask, setup = get_sec(Isaxs, mask, setup)
        tmp = get_soq(Isaxs, mask, setup, Vsaxs)
        soq = np.hstack(tmp)
        soq = soq.reshape(-1, 3, order='F')
        saxsd['soq'] = soq

    return saxsd
