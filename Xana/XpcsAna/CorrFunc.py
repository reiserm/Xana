import numpy as np
from matplotlib import pyplot as plt
import copy
import pandas as pd
import collections

from ..Decorators import Decorators
from ..Analist import AnaList
from ..XParam.parameters import plot_parameters
from ..Xfit.fitg2global import G2
from ..Xplot.niceplot import niceplot
from ..misc.resample import resample as resample_func
from .StaticContrast import staticcontrast


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
        updating only top-level keys, dict_merge recurses down into dicts nested
        to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
        ``dct``.
        :param dct: dict onto which the merge is executed
        :param merge_dct: dct merged into dct
        :return: None
    """
    for k, v in merge_dct.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.Mapping):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


class CorrFunc(AnaList):

    def __init__(self, Xana, **kwargs):
        super().__init__(Xana, **kwargs)
        self.fit_result = [None]
        self.pars = [None]
        self.corrFunc = [None]
        self.corrFuncRescaled = [None]
        self.staticContrast = [None]
        self.corrFuncChi2 = [None]
        self.twotime = [None]
        self.g2plotl = [None]
        self.nq = np.arange(len(self.Xana.setup.qroi))
        self.db_id = []
        self.ncfs = 0
        self.fit_config = []

    def __str__(self):
        return 'Corr Func class for g2 displaying.'

    def __getstate__(self):
        d = dict(vars(self))
        # d['twotime'] = None
        d['g2plotl'] = None
        return d

    def __setstate__(self, d):
        # deal with old type of db_id and fit_config properties
        if isinstance(d['db_id'], np.ndarray):
            d['db_id'] = [d['db_id'],]
        if isinstance(d['fit_config'], dict):
            d['fit_config'] = [d['fit_config'],]
        self.__dict__.update(d)

    def __add__(self, cf2):
        cf3 = CorrFunc(self.Xana)
        extend_items = [
            'corrFunc',
            'corrFuncRescaled',
            'db_id',
            'pars',
            'fit_result',
            'fit_config',

        ]

        for item in extend_items:
            value = copy.deepcopy(getattr(self, item))
            value.extend(getattr(cf2, item))
            setattr(cf3, item, value)

        return cf3

    @Decorators.input2list
    def get_g2(self, db_id, merge='append', **kwargs):
        self.corrFunc = []
        if merge == 'merge':
            self.merge_g2(db_id, **kwargs)
            self.db_id.append(db_id)
        elif merge == 'append':
            for sid in db_id:
                self.db_id.append([sid])
                try:
                    d = self.Xana.get_item(sid)
                    self.corrFunc.append((np.ma.masked_invalid(d['corf']),
                                          np.ma.masked_invalid(d['dcorf'])))
                except KeyError:
                    print('Could not load item {}'.format(sid))
            print('Loaded {} correlation functions.'.format(len(db_id)))
        self.corrFuncRescaled = copy.deepcopy(self.corrFunc)

    @Decorators.init_figure()
    def plot_g2(self, nq=None, ax=None, data='rescaled', cmap='jet', change_marker=False,
                color_mode=0, color='b', dofit=False, index = None, exclude=None,
                **kwargs):

        fit_keys = ['nmodes', 'mode', 'fix', 'init', 'fitglobal', 'lmfit_pars', 'fitqdep']
        fit_kwargs = {k:i for k,i in kwargs.items() if k in fit_keys}

        if data == 'original' or self.corrFuncRescaled is None:
            corrFunc = list(self.corrFunc)
        elif data == 'rescaled':
            corrFunc = list(self.corrFuncRescaled)
        else:
            raise ValueError('No usable correlation data defined.')

        ind_cfs = np.arange(len(corrFunc))
        ind_pars = ind_cfs.copy()
        if index is not None:
            if isinstance(index, (int, np.integer)):
                s = slice(index, index+1)
            elif isinstance(index, (list, tuple)):
                s = slice(index[0], index[1])
            else:
                raise ValueError(f'Index of type {type(index)} not supported. User int or list.')
            corrFunc = corrFunc[s]
            ind_pars = ind_pars[s]
            ind_cfs = np.arange(len(corrFunc))
        elif index is None and exclude is not None:
            ind_cfs = np.delete(ind_cfs, exclude)
            ind_pars = np.delete(ind_pars, exclude)

        ncf = ind_cfs.size

        if nq is None:
            pass
        elif type(nq) == int:
            self.nq = np.arange(nq)
        else:
            self.nq = nq

        if color_mode < 2:
            if color_mode == 0:
                color_multiplier, color_repeater = self.nq.size, ncf
            elif color_mode == 1:
                color_multiplier, color_repeater = self.nq.size * \
                    ncf, 1
            self.update_colors(cmap, color_multiplier, color_repeater)
        elif color_mode == 2:
            self.colors = [color]*len(self.nq)*ncf
        self.update_markers(ncf, change_marker)

        if self.fit_config is None:
            self.fit_config = [dict(fit_kwargs),] * ncf
        elif isinstance(self.fit_config, list):
            if len(self.fit_config) == 0:
                self.fit_config = [dict(fit_kwargs),] * ncf
            for d in self.fit_config:
                dict_merge(d, fit_kwargs)

        if dofit:
            self.fit_result = [[]] * ncf

        if dofit or self.pars is None or (len(self.pars)!=ncf) :
            self.pars = [[]] * ncf

        if len(self.pars) > len(self.db_id):
            tmp = [np.where(np.diff(x)<0)[0] for x in self.db_id]
            tmp = [np.split(x, y+1) for x,y in zip(self.db_id,tmp)]
            self.db_id = [item for sublist in tmp for item in sublist]

        ci = 0
        for j, (ipar,icf) in enumerate(zip(ind_pars,ind_cfs)):
            cfi, dcfi = corrFunc[icf]
            g2 = G2(cfi, self.nq, dcfi)
            if dofit:
                res = g2.fit(**self.fit_config[j])
                self.pars[j] = res[0]
                self.fit_result[j] = res[1]
            if 'doplot' in kwargs:
                g2.plot(marker=self.markers[j % len(self.markers)],
                        ax=ax,
                        colors=self.colors[ci*len(self.nq):(ci+1)*len(self.nq)],
                        data_label='id {}'.format(self.db_id[j][0]), pars=self.pars[ipar],
                        **kwargs)
                ci += 1

    @staticmethod
    def init_pars(names, entry0):
        names = [names[i//2] if (i+1) % 2 else 'd'+names[i//2]
                 for i in range(len(names)*2)]
        names.insert(0, entry0)
        names.extend(['chisqr', 'redchi', 'bic', 'aic'])
        return pd.DataFrame(columns=names)

    def reset_rescaled(self):
        self.corrFuncRescaled = copy.deepcopy(self.corrFunc)

    def rescale(self, index=None, normby='average', norm_baseline=True,
                   norm_contrast=False, nq=None, baseline=1., contrast=1.,
                   interval=(1, -1), weighted=False):

        def rescale(y, mn, mx, rng=(0, 1)):
            p = (rng[1]-rng[0])/(mx-mn)
            return p * (y - mn) + rng[0], p

        def normFunc(corrFunc, pars):
            norm_b = np.nanmin(corrFunc[0][1:, nq+1], axis=0)
            norm_c = np.nanmax(corrFunc[0][1:, nq+1], axis=0)
            if normby == 'fit':
                for iq in range(nq.size):
                    norm_b[iq] = pars.loc[iq, 'a']
                    if norm_contrast:
                        norm_c[iq] = pars.loc[iq, 'beta'] + pars.loc[iq, 'a']
            elif normby == 'average':
                for iq in range(nq.size):
                    if weighted:
                        weights = 1/corrFunc[1][interval[1]:, nq[iq]+1]**2
                    else:
                        weights = None
                    norm_b[iq] = np.ma.average(
                        corrFunc[0][interval[1]:, nq[iq]+1], weights=weights)
                    if norm_contrast:
                        if weighted:
                            weights = 1 / \
                                corrFunc[1][1:max([interval[0],1])+1, nq[iq]+1]**2
                        else:
                            weights = None
                        norm_c[iq] = np.ma.average(
                            corrFunc[0][1:max([interval[0],1])+1, nq[iq]+1], weights=weights)

            if norm_contrast is False:
                initial_contrast = norm_c - norm_b
            else:
                initial_contrast = contrast
            corrFunc[0][1:, nq+1], p = rescale(corrFunc[0][1:, nq+1], norm_b, norm_c,
                                               (baseline, initial_contrast + baseline))
            corrFunc[1][1:, nq+1] *= p

        if self.pars is None:
            self.pars = [None] * len(self.corrFunc)
        if nq is None:
            nq = self.nq
        if index is None:
            index = slice(len(self.corrFuncRescaled))
        else:
            index = slice(index, index+1, 1)

        for corrFunc, pars in zip(self.corrFuncRescaled[index], self.pars[index]):
            normFunc(corrFunc, pars)

    def rescale_user(self, ind=None, offset=None, factor=None,  nq=None):

        if nq is None:
            nq = self.nq

        if offset is None:
            offset = np.zeros(len(self.corrFuncRescaled))

        if factor is None:
            factor = np.ones(len(self.corrFuncRescaled))

        for corrFunc, o, f in zip(self.corrFuncRescaled, offset, factor):
            corrFunc[0][1:, nq+1] *= f
            corrFunc[0][1:, nq+1] -= f - 1
            corrFunc[1][1:, nq+1] *= f
            corrFunc[0][1:, nq+1] += o

    def merge_g2(self, in_list, limit=0., chi2sig=3, cutoff=0):
        self.corrFuncChi2 = []

        t_exp = np.zeros(len(in_list))
        nframes = t_exp.copy()
        for ii, i in enumerate(in_list):
            t_exp[ii] = self.Xana.db.loc[i]['t_exposure']
            nframes[ii] = self.Xana.db.loc[i]['nframes']

        ind = np.argsort(t_exp)[::-1]
        t_exp = t_exp[ind]
        nframes = nframes[ind]
        in_list = np.array(in_list)[ind]

        uq_et, uq_inv, uq_cnt,  = np.unique(
            t_exp, return_inverse=True, return_counts=True)

        counter = np.zeros(uq_et.size, dtype=np.int32)
        for i, cnti in enumerate(uq_cnt):
            indall = np.where(uq_inv == i)[0]
            for j, ind in enumerate(indall):
                counter[i] += nframes[ind]
                item = in_list[ind]
                try:
                    d = self.Xana.get_item(item)
                    if j == 0:
                        qv = d['corf'][0,1:]
                        t = d['corf'][1:, 0]
                        cf = np.zeros((cnti, t.size, qv.size))
                        dcf = np.zeros_like(cf)

                    cft = d['corf'][1:-(cutoff+1), 1:qv.size+1]

                    if cft.shape[0] > t.size:
                        t = d['corf'][1:, 0]
                        pad = np.zeros((cnti, t.size-cf.shape[1], qv.size))
                        cf = np.concatenate((cf, pad), axis=1)
                        dcf = np.concatenate((dcf, pad), axis=1)

                    cf[j, :cft.shape[0], :qv.size] = cft
                    dcf[j, :cft.shape[0], :qv.size] = d['dcorf'][1:-(cutoff+1), 1:qv.size+1]
                except ValueError as v:
                    print('Tried loading database entry: ', item)
                    raise ValueError(v)

            cf = np.ma.masked_array(cf, mask=np.isnan(cf))
            cf = np.ma.masked_less_equal(cf.filled(-1), limit, copy=False)
            dcf = np.ma.masked_array(dcf, mask=np.isnan(dcf))
            dcf = np.ma.masked_where(dcf.filled(-1)<=0, dcf, copy=False)
            cf = np.ma.masked_where(dcf.mask, cf, copy=False)
            dcf = np.ma.masked_where(cf.mask, dcf, copy=False)

            cfm, dcfm = np.ma.average(cf, weights=1/dcf**2, returned=1, axis=0)

            chi2arr = np.ma.sum((cf - cfm)**2 / cfm**2, 1)
            chi2arr /= cf.shape[1] - 1
            chi2arr = chi2arr[:,0]  # np.max(chi2arr, -1)

            chi2cond = chi2arr > (chi2arr.mean() + chi2sig * chi2arr.std())
            chi2ret = (in_list[indall[chi2cond]], chi2arr.compressed())
            self.corrFuncChi2.append(chi2ret)

            cfm = np.ma.hstack((t[:, None], cfm))
            dcfm = np.ma.sqrt(1/dcfm)
            dcfm = np.ma.hstack((t[:, None], dcfm))

            self.corrFunc.append((np.ma.vstack((np.append(0, qv), cfm)),
                                  np.ma.vstack((np.append(0, qv), dcfm))))

        tmp = "Merged g2 functions: "
        print('{:<22}{} (exposure times)'.format(tmp, np.round(uq_et, 6)))
        print('{:<22}{} (number of correlation functions)'.format('', uq_cnt))
        print('{:<22}{} (total number of images)'.format('', counter))

    def merge_g2list(self, resample=False, cutoff=-1, **kwargs):
        self.db_id = [np.hstack(self.db_id), ]
        for ii, cf_master in enumerate([self.corrFunc, self.corrFuncRescaled]):
            if cf_master is not None:
                for i, cf in enumerate(cf_master):
                    if i == 0:
                        cf_tmp = cf[0][:cutoff]
                        dcf_tmp = cf[1][:cutoff]
                    else:
                        cf_tmp = np.vstack((cf_tmp, cf[0][1:cutoff]))
                        dcf_tmp = np.vstack((dcf_tmp, cf[1][1:cutoff]))

                if resample:
                    new_t, new_cf, new_dcf = resample_func(cf_tmp[1:, 0], cf_tmp[1:, 1:],
                                                           dcf_tmp[1:, 1:], resample,
                                                           **kwargs)
                    cf_tmp = np.hstack((new_t[:, None], new_cf))
                    cf_tmp = np.vstack((cf_master[0][0][0, :], cf_tmp))
                    dcf_tmp = np.hstack((new_t[:, None], new_dcf))
                    dcf_tmp = np.vstack((cf_master[0][0][0, :], dcf_tmp))

                indsort = np.argsort(cf_tmp[1:,0])
                cf_tmp[1:] = cf_tmp[indsort+1]
                dcf_tmp[1:] = dcf_tmp[indsort+1]
                if ii == 0:
                    self.corrFunc = [(cf_tmp, dcf_tmp), ]
                elif ii == 1:
                    self.corrFuncRescaled = [(cf_tmp, dcf_tmp), ]

    def plot_parameters(self, plot, cmap='jet', ax=None, change_axes=True,
                        cindoff=0, extpar_name='extpar', extpar_vec=None,
                        color_discrete=True, exclude=None, include=None, **kwargs):
        """Plot Fit parameter (decay rates, kww exponent, etc.)
        """
        npl = len(plot)
        npars = len(self.pars)

        ind_pars = np.arange(npars)
        if exclude is not None:
            ind_pars = np.delete(ind_pars, exclude)
        elif include is not None:
            ind_pars = np.array(include)
        npars = ind_pars.size

        if ax is None and change_axes:
            n = int(np.ceil(npl/2))
            fig, ax = plt.subplots(
                n, int(npl > 1)+1, figsize=(5+4*(npl > 1), 4*n))
        elif ax is None and not change_axes:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))

        if isinstance(ax, np.ndarray):
            ax = ax.flatten()
        else:
            ax = [ax,]

        modes = kwargs.get('modes', 1)
        if isinstance(modes, (list, tuple)):
            mm = max([len(x) if isinstance(x, (list,tuple)) else 1 for x in modes])
        else:
            mm = 1

        if change_axes:
            ax_idx = np.arange(len(ax))
            c_idx = np.repeat(np.arange(npars), len(ax_idx))
        else:
            ax_idx = np.zeros(npl, dtype=np.int16)
            c_idx = np.arange(npars)

        if color_discrete:
            cmap = plt.get_cmap(cmap, npars+1)
        else:
            cmap = plt.get_cmap(cmap)

        self.pars2 = [[]] * npl
        self.fit_result2 = [[[] for i in range(npars)]] * npl

        if extpar_vec is None:
            extpar_vec = np.zeros(npars)
        else:
            assert len(extpar_vec) == npars

        for i, p in enumerate(plot):
            pars_initialized = False
            db_tmp = 0
            for ipar, ind_par in enumerate(ind_pars):
                pars = self.pars[ind_par]
                kwargsl = {key: value[i] if isinstance(value, (tuple, list)) else value \
                           for (key, value) in kwargs.items()}
                res = plot_parameters(pars, p, ax=ax[ax_idx[i]], ci=c_idx[ipar], cmap=cmap,
                                      **kwargsl)
                self.fit_result2[i][ipar] = res[2:4]
                if not isinstance(res, np.ndarray):
                    if not pars_initialized:
                        db_tmp = self.init_pars(list(res[2].params.keys()), extpar_name)
                        pars_initialized = True
                    entry = [extpar_vec[ipar], *res[0].flatten(), *res[1]]
                    db_tmp.loc[ipar] = entry
            self.pars2[i] = db_tmp

        plt.tight_layout()

    @Decorators.input2list
    def get_twotime(self, db_id, twotime_par=None):
        """Receive two-time correlation functions from database
        """
        self.twotime = 0.
        i = 0
        for sid in db_id:
            d = self.Xana.get_item(sid)
            if twotime_par is None:
                twotime_par = d['twotime_par']
            try:
                self.twotime += d['twotime_corf'][twotime_par]
                i += 1
            except ValueError as e:
                print('Could not average %d error message was\n\t%s' % (int(sid), e))
        self.twotime /= i

    @Decorators.init_figure()
    @Decorators.input2list
    def plot_twotime(self, db_id, clim=(None, None), ax=None, interpolation='gaussian', \
                     twotime_par=None):
        """Plot two-time correlation functions read from database
        """
        self.get_twotime(db_id, twotime_par)

        vmin, vmax = clim
        corfd = self.Xana.get_item(db_id[0])
        ax.set_title(
            r'q = {:.2g}$\mathrm{{nm}}^{{-1}}$'.format(corfd['qv'][corfd['twotime_par']]))
        tt = corfd['twotime_xy']
        im = ax.imshow(self.twotime, cmap=plt.get_cmap('jet'), origin='lower',
                       interpolation=interpolation, extent=[
                           tt[0], tt[-1], tt[0], tt[-1]],
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel(r'$t_1$ [s]',)
        ax.set_ylabel(r'$t_2$ [s]',)
        cl = plt.colorbar(im, ax=ax)
        cl.ax.set_ylabel('correlation', fontsize=12)
        niceplot(ax, autoscale=False, grid=False)

    @Decorators.init_figure()
    @Decorators.input2list
    def plot_trace(self, db_id, log='', ax=None, cmap='Set1'):
        axtop = ax.twiny()

        ci = 0
        cmap = plt.get_cmap(cmap)
        for sid in db_id:
            corfd = self.Xana.get_item(sid)
            trace = corfd['trace']
            framen = np.arange(trace.shape[0])
            time0 = corfd['twotime_xy'][0]
            time1 = corfd['twotime_xy'][-1]
            time = np.linspace(time0, time1, framen.size)
            for i, iq in enumerate(self.nq):
                ax.plot(
                    time, trace[:, iq], 'o-', color=cmap(ci), markersize=2, label=str(i))
                axtop.plot(
                    framen, trace[:, iq], 'o-', color=cmap(ci), markersize=2, label=str(i))
                ci += 1

        ax.set_ylabel('photons per pixel')
        ax.set_xlabel('time in [s]')
        niceplot(ax, autoscale=0)
        axtop.set_xlabel('frame number')
        niceplot(axtop, autoscale=False, grid=False)
        if 'x' in log:
            ax.set_xscale('log')
        if 'y' in log:
            ax.set_yscale('log')

        plt.legend()

        # ax.set_title('trace', fontweight='bold', fontsize=14, y=1.14)

    def get_static_contrast(self, *args, **kwargs):
        self.staticContrast = staticcontrast(self, *args, **kwargs)

    def g2_totxt(self, savname):
        return 0
