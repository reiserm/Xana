import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import itertools
import copy
import pandas as pd

from Decorators import Decorators
from Analist import AnaList
from XParam.parameters import plot_parameters
from Xfit.fitg2 import fitg2
from Xplot.niceplot import niceplot
from misc.resample import resample as resample_func
from XpcsAna.StaticContrast import staticcontrast


class CorrFunc(AnaList):

    def __init__(self, Xana, **kwargs):
        super().__init__(Xana, **kwargs)
        self.fit_result = None
        self.pars = None
        self.corrFunc = None
        self.corrFuncRescaled = None
        self.staticContrast = None
        self.corrFuncChi2 = None
        self.twotime = None
        self.g2plotl = None
        self.nq = np.arange(len(self.Xana.setup['qroi']))
        self.db_id = None

    def __str__(self):
        return 'Corr Func class for g2 displaying.'

    def __repr__(self):
        return 

    def __add__(self, cf2):
        cf3 = CorrFunc(self.Xana)
        cf3.fit_result = copy.deepcopy(self.fit_result)
        cf3.fit_result.extend(cf2.fit_result)
        cf3.pars = copy.deepcopy(self.pars)
        cf3.pars.extend(cf2.pars)
        cf3.corrFunc = copy.deepcopy(self.corrFunc)
        cf3.corrFunc.extend(cf2.corrFunc)
        cf3.corrFuncRescaled = copy.deepcopy(self.corrFuncRescaled)
        cf3.corrFuncRescaled.extend(cf2.corrFuncRescaled)
        cf3.db_id = np.append(self.db_id, cf2.db_id)
        return cf3

    @Decorators.input2list
    def get_g2(self, db_id, merge='append', **kwargs):
        self.db_id = db_id
        self.corrFunc = []
        if merge == 'merge':
            self.merge_g2(db_id, **kwargs)
        elif merge == 'append':
            for sid in db_id:
                try:
                    d = self.Xana.get_item(sid)
                    self.corrFunc.append((d['corf'], d['dcorf']))
                except KeyError:
                    print('Could not load item {}'.format(sid))
            print('Loaded {} correlation functions.'.format(len(db_id)))
        self.corrFuncRescaled = copy.deepcopy(self.corrFunc)

    @Decorators.init_figure()
    def plot_g2(self, nq=None, err=True, ax=None, nmodes=1, data='rescaled', cmap='jet',
                change_marker=False, color_mode=0, color='b', dofit=True, **kwargs):
        if data == 'original' or self.corrFuncRescaled is None:
            corrFunc = list(self.corrFunc)
        elif data == 'rescaled':
            corrFunc = list(self.corrFuncRescaled)
        else:
            raise ValueError('No usable correlation data defined.')

        if nq is None:
            pass
        elif type(nq) == int:
            self.nq = np.arange(nq)
        else:
            self.nq = nq

        if color_mode < 2:
            if color_mode == 0:
                color_multiplier, color_repeater = self.nq.size, len(corrFunc)
            elif color_mode == 1:
                color_multiplier, color_repeater = self.nq.size * \
                    len(corrFunc), 1
            self.update_colors(cmap, color_multiplier, color_repeater)
        elif color_mode == 2:
            self.colors = [color]*len(self.nq)*len(corrFunc)

        self.update_markers(len(corrFunc), change_marker)

        self.g2plotl = [[]] * len(corrFunc)
        self.pars = [[]] * len(corrFunc)
        self.fit_result = [[[] for i in range(self.nq.size)]] * len(corrFunc)

        ci = 0
        for j, (cfi, dcfi) in enumerate(corrFunc):
            rates = np.zeros((self.nq.size, 3*nmodes+3, 2))
            ti = cfi[1:, 0]
            for i, qi in enumerate(self.nq):
                if i == 0:
                    cf_id = self.db_id[j]
                else:
                    cf_id = None
                res = fitg2(ti, cfi[1:, qi+1], err=dcfi[1:, qi+1],
                            qv=cfi[0,1+qi],
                            ax=ax, color=self.colors[ci],
                            marker=self.markers[j % len(self.markers)], cf_id=cf_id,
                            modes=nmodes, dofit=dofit, **kwargs)
                self.fit_result[j][i] = res[2:4]
                self.g2plotl[j].append(
                    list(itertools.chain.from_iterable(res[4])))
                if dofit:
                    if i == 0:
                        db_tmp = self.init_pars(list(res[2].params.keys()))
                    entry = [cfi[0, qi+1], *res[0].flatten(), *res[1]]
                    db_tmp.loc[i] = entry
                else:
                    db_tmp = 0
                ci += 1
            self.pars[j] = db_tmp

    @staticmethod
    def init_pars(names):
        names = [names[i//2] if (i+1) % 2 else 'd'+names[i//2]
                 for i in range(len(names)*2)]
        names.insert(0, 'q')
        names.extend(['chisqr', 'redchi', 'bic', 'aic'])
        return pd.DataFrame(columns=names)

    def reset_rescaled(self):
        self.corrFuncRescaled = copy.deepcopy(self.corrFunc)

    def rescale(self, index=None, normby='average', norm_baseline=True,
                   norm_contrast=False, nq=None, baseline=1., contrast=None,
                   interval=(1, -1), weighted=True):

        def rescale(y, mn, mx, rng=(0, 1)):
            p = (rng[1]-rng[0])/(mx-mn)
            return p * (y - mn) + rng[0], p

        def normFunc(corrFunc, pars):
            norm_b = np.min(corrFunc[0][1:, nq+1], axis=0)
            norm_c = np.max(corrFunc[0][1:, nq+1], axis=0)
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
                            corrFunc[0][1:max([interval[0],1])+1, nq[iq]+1],
                            weights=weights)

            if contrast is None:
                initial_contrast = norm_c - norm_b
            else:
                initial_contrast = contrast
            corrFunc[0][1:, nq+1], p = rescale(
                corrFunc[0][1:, nq+1], norm_b, norm_c,
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

    def rescale_user(self, offset=0., nq=None):

        if nq is None:
            nq = self.nq

        for corrFunc, o in zip(self.corrFuncRescaled, offset):
            corrFunc[0][1:, nq+1] += o

    def merge_g2(self, in_list, limit=0.005, chi2sig=3):
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
            t_exp, return_inverse=1, return_counts=1)

        counter = np.zeros(uq_et.size, dtype=np.int32)
        for i, cnti in enumerate(uq_cnt):
            indall = np.where(uq_inv == i)[0]
            for j, ind in enumerate(indall):
                counter[i] += nframes[ind]
                item = in_list[ind]
                try:
                    d = self.Xana.get_item(item)
                    if j == 0:
                        max_t_item = in_list[np.argmax(nframes[indall])]
                        max_t = self.Xana.get_item(max_t_item)['corf'].shape[0] - 1
                        qv = d['corf'][0,1:]
                        t = d['corf'][1:, 0]
                        cf = np.zeros((cnti, t.size, qv.size))
                        dcf = np.zeros_like(cf)
                    cft = d['corf'][1:, 1:qv.size+1]
                    cf[j, :cft.shape[0], :qv.size] = cft
                    dcf[j, :cft.shape[0], :qv.size] = d['dcorf'][1:, 1:qv.size+1]
                except ValueError as v:
                    print('Tried loading database entry: ', item)
                    raise ValueError(v)

            cf = np.ma.masked_array(cf, mask=((cf < limit) | np.isnan(cf) | (dcf <= 0)))
            dcf = np.ma.masked_array(dcf, mask=cf.mask)
            
            cfm, dcfm = np.ma.average(cf, weights=1/dcf**2, returned=1, axis=0)

            chi2arr = np.ma.sum((cf - cfm)**2 / cfm**2, 1)
            chi2arr /= cf.shape[1] - 1
            chi2arr = np.max(chi2arr,-1)

            chi2cond = chi2arr > (chi2arr.mean() + chi2sig * chi2arr.std())
            chi2ret = (in_list[indall[chi2cond]], chi2arr.compressed())
            self.corrFuncChi2.append(chi2ret)

            cfm = np.hstack((t[:, None], cfm.filled(np.nan)))
            dcfm = np.ma.sqrt(1/dcfm)
            dcfm = np.hstack((t[:, None], dcfm.filled(np.nan)))
            
            self.corrFunc.append((np.vstack((np.append(0, qv), cfm)),
                                  np.vstack((np.append(0, qv), dcfm))))

        tmp = "Merged g2 functions: "
        print('{:<22}{} (exposure times)'.format(tmp, np.round(uq_et, 6)))
        print('{:<22}{} (number of correlation functions)'.format('', uq_cnt))
        print('{:<22}{} (total number of images)'.format('', counter))

    def merge_g2list(self, resample=False, cutoff=-1, **kwargs):
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
                    new_t, new_cf, new_dcf = resample_func(cf_tmp[1:, 0], cf_tmp[1:, 1:], dcf_tmp[1:, 1:],
                                                           resample, **kwargs)
                    cf_tmp = np.hstack((new_t[:, None], new_cf))
                    cf_tmp = np.vstack((cf_master[0][0][0, :], cf_tmp))
                    dcf_tmp = np.hstack((new_t[:, None], new_dcf))
                    dcf_tmp = np.vstack((cf_master[0][0][0, :], dcf_tmp))

                if ii == 0:
                    self.corrFunc = [(cf_tmp, dcf_tmp), ]
                elif ii == 1:
                    self.corrFuncRescaled = [(cf_tmp, dcf_tmp), ]

    def plot_parameters(self, plot, ax=None, change_axes=True,
                        cindoff=0, **kwargs):
        """Plot Fit parameter (decay rates, kww exponent, etc.)
        """
        npl = len(plot)
        npars = len(self.pars)

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

        mm = max([len(x) if isinstance(x, (list,tuple)) else 1    
                  for x in kwargs.get('modes',[1])])
            
        if change_axes:
            ax_idx = np.arange(len(ax))
            c_idx = np.repeat(np.arange(mm*npars), npl)
        else:
            ax_idx = np.zeros(npl, dtype=np.int16)
            c_idx = np.arange(npl*npars)

        for ipar, pars in enumerate(self.pars):
            for i, p in enumerate(plot):
                kwargsl = {key: value[i] if type(value) == tuple else value \
                           for (key, value) in kwargs.items()}
                plot_parameters(pars, p, ax=ax[ax_idx[i]], ci=c_idx[ipar:ipar+mm],
                                 **kwargsl)
        plt.tight_layout()
        return ax

    @Decorators.input2list
    def get_twotime(self, db_id, twotime_par=None):
        """Receive two-time correlation functions from database
        """
        self.twotime = 0.
        for i, sid in enumerate(db_id):
            d = self.Xana.get_item(sid)
            if twotime_par is None:
                twotime_par = d['twotime_par']
            self.twotime += d['twotime_corf'][twotime_par]
        self.twotime /= i + 1

    @Decorators.init_figure()
    @Decorators.input2list
    def plot_twotime(self, db_id, clim=(None, None), ax=None, interpolation='gaussian'):
        """Plot two-time correlation functions read from database
        """
        self.get_twotime(db_id)

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
    def plot_trace(self, db_id, log='', ax=None):
        axtop = ax.twiny()

        ci = 0
        for sid in db_id:
            corfd = self.Xana.get_item(sid)
            trace = corfd['trace']
            framen = np.arange(trace.shape[0])
            time0 = corfd['twotime_xy'][0]
            time1 = corfd['twotime_xy'][-1]
            time = np.linspace(time0, time1, framen.size)
            for i, iq in enumerate(self.nq):
                ax.plot(
                    time, trace[:, iq], 'o', color=self.colors[ci], markersize=2, label=str(i))
                axtop.plot(
                    framen, trace[:, iq], 'o', color=self.colors[ci], markersize=2, label=str(i))
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
