import numpy as np
from SaxsAna.integrate import get_soq
from misc.xsave import save_result
import copy


def get_sec(obj, img):

    if np.shape(img) != np.shape(obj.xdata.mask):
        qsec = obj.setup['qsec'][0]
        mask = obj.xdata.mask.copy()
        dim = np.shape(img)
        mask = mask[qsec[0]:dim[0]+qsec[0], qsec[1]:dim[1]+qsec[1]]

        setup = copy.deepcopy(obj.setup)
        setup['ctr'] = (setup['ctr'][0]-qsec[1], setup['ctr'][1]-qsec[0])
        return mask, setup
    else:
        return obj.xdata.mask, obj.setup


def saxs(obj, series_id, load=False, output='2d', filename="", handle_existing='next',
         return_saxs=False, nprocs=8, verbose=True, dark=None, calc_soq=True, **kwargs):

    if dark is not None and type(dark)==int:
        dark = obj.get_item(dark)['Isaxs']

    for sid in series_id:
        print('\n#### Starting SAXS Analysis ####\nSeries: {} in folder {}\n'.format(sid,
                                                                        obj.xdata.datdir))
        if load:
            saxsd = obj.get_item(sid)
            Isaxs = saxsd['Isaxs']
        else:
            Isaxs, Vsaxs = obj.xdata.get_series(sid, output=output, method='average',
                                                verbose=verbose, nprocs=nprocs,
                                                dark=dark, **kwargs)
            saxsd = {'Isaxs':Isaxs, 'Vsaxs':Vsaxs}

        if calc_soq and obj.setup is not None and Isaxs.ndim==2:

            mask, setup = get_sec(obj, Isaxs)
            tmp = get_soq(Isaxs, mask, setup, Vsaxs)
            soq = np.hstack(tmp)
            soq = soq.reshape(-1, 3, order='F')
            saxsd['soq'] = soq

        f = obj.xdata.datdir.split('/')[-2] + '_s' + str(obj.xdata.meta.loc[sid, 'series']) + filename
        savfile = save_result(saxsd, 'saxs', obj.savdir, f, handle_existing=handle_existing)
        obj.add_db_entry(sid, savfile)
        if return_saxs:
            return saxsd, savfile
