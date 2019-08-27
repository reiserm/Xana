import numpy as np
import lmfit
from copy import copy
import sys
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import re

class MSD:

    def __init__(self, msd):

        self.t = np.ma.masked_invalid(msd[0])
        ind_sort = np.argsort(self.t)
        self.t = self.t[ind_sort]

        self.msd = np.ma.masked_invalid(msd[1,ind_sort])
        self.dmsd = np.ma.masked_invalid(msd[2,ind_sort])

        # properties
        self.fix = None

        # attributes defined in private methods
        self._lmpars = None
        self._weights = None
        self._varnames = None

        self._fit_data = None
        self._fit_weights = None
        self._fit_logarithmic = False

        # fit results
        self.pars = None
        self.fit_result = []

        self._minimizer = None

    @staticmethod
    def _reduce_func(arr):
        return -0.5 * np.nansum(arr * arr)

    @staticmethod
    def _iter_cb(params, iter, resid, *args, **kws):
        if (iter) % 4 == 0:
            print('\rOptimizing...(%d)' % iter, end='', flush=1)
        else:
            pass


    def fit(self, mode='sig', init={}, fix={}, lmfit_pars={}, fitqdep={}):
        '''
        Function that computes the fits using lmfit's minimizer
        '''

        self.fix = fix

        # make the parameters
        if not lmfit_pars.get('is_weighted', True):
            init['__lnsigma'] = 1

        self._init_parameters(init, fix)

        # setting the weights of the data
        self._fit_data = self.msd
        self._get_weights(mode)
        self._fit_weights = self._weights


        self._minimizer = lmfit.Minimizer(self._residuals, params=self._lmpars,
                                          reduce_fcn=self._reduce_func,
                                          iter_cb=self._iter_cb,
                                          nan_policy='omit')

        # do the fit
        out = self._minimizer.minimize(**lmfit_pars)

        self._write_to_pars(out,)
        self.fit_result.append((out, lmfit.fit_report(out)))

        # convert to numeric data types
        self.pars = self.pars.apply(pd.to_numeric)

        self._fit_logarithmic = False

        return self.pars, self.fit_result


    def plot(self, doplot=False, marker='o', ax=None, xl=None, yl=None, colors=None, alpha=1.,
             markersize=3., data_label=None, confint=False, pars=None, **kwargs):

        sumidit = True if (self.pars is not None) else False

        if pars is not None :
            if len(pars):
                self.pars = pars
                sumidit = True

        if xl is None:
            if ax is None:
                ax = plt.gca()
            xl = ax.get_xlim()
            if xl[0] == 0:
                xl = (np.min(self.t)*0.5,np.max(self.t)*1.5)

        xf = np.logspace(np.log10(xl[0]),np.log10(xl[1]), 50)

        for ii, iq in enumerate([0]):
            pl = []
            if sumidit:
                msdf = self._calc_model(self.fit_result[ii][0].params, x=xf)

                labstr_fit = None
                if 'legf' in doplot and sumidit:
                    pard = {'t':r'$t: {:.2e}\mathrm{{s}},\,$',
                            'g':r'$\gamma: {:.2g},\,$',
                            'b':r'$\mathrm{{b}}: {:.3g},\,$',
                            'a':r'$\mathrm{{a}}: {:.3g},\,$'}
                    labstr_fit = ''
                    for i in range(self.nmodes):
                        for vn in 'tgba':
                            if vn == 'a' and i > 0:
                                continue
                            elif vn == 'a' and i == 0:
                                vnn = 'a'
                            else:
                                vnn = vn + str(i)
                            if vnn in self.pars.columns:
                                labstr_fit += pard[vn].format(self.pars.loc[ii, vnn])
                            elif fix is not None and vnn in fix.keys():
                                labstr_fit += 'fix '+ pard[vn].format(fix[vnn])
                            else:
                                labstr_fit += pard[vn].format(0)

            labstr_data = None
            if 'legd' in doplot:
                if data_label is not None:
                    labstr_data = data_label
                else:
                    labstr_data = r'$\mathsf{{q}} = {:.3f}\,\mathsf{{nm}}^{{-1}}$'.format(self.qv[iq])

            if 'legq' in doplot:
                labstr_data = r'$\mathsf{{q}} = {:.3f}\,\mathsf{{nm}}^{{-1}}$'.format(self.qv[iq])

            if 'fit' in doplot and sumidit:
                pl.append(ax.plot(xf, msdf, '-', label=labstr_fit, linewidth=1))

            if 'data' in doplot:
                pl.append(ax.errorbar(self.t.filled(np.nan), self.msd.filled(np.nan), yerr=self.dmsd.filled(np.nan),
                                          linestyle='', marker=marker, label=labstr_data,
                                          alpha=alpha, markersize=markersize)[0])

            if colors is None:
                if 'data' in doplot:
                    color = pl[-1].get_color()
                else:
                    color = 'gray'
            else:
                color = colors[ii]

            for p in pl:
                if isinstance(p, list):
                    p[0].set_color(color)
                elif isinstance(p, matplotlib.container.ErrorbarContainer):
                    for child in p.get_children():
                        child.set_color(color)

            ax.set_xscale('log')
            ax.set_yscale('log')

            if 'leg' in doplot:
                ax.legend()

            if 'report' in doplot and sumidit:
                print(lmfit.fit_report(out))

            ax.set_xlabel(r'$\tau\,(\mathrm{s})$')
            ax.set_ylabel(r'$g_2(\tau)$')

            ax.set_xlim(*xl)
            if yl is not None:
                ax.set_ylim(*yl)

        return None

    def _calc_model(self, v, x=None):
        """Calculate multi mode g2 funktion
        """
        t = self.t if x is None else x
        model = 6 * v['d']**2 * (1 - np.exp(-(v['D0']/v['d']**2 * t)**v['a'])) ** (1/v['a']) * (1 + v['Dm']/v['d']**2 * t)
        if self._fit_logarithmic:
            return np.log10(model)
        else:
            return model

    def _residuals(self, pars, *args, **kwargs):
        """2D Residual function to minimize
        """
        v = pars.valuesdict()
        model = self._calc_model(v)

        resid = (self._fit_data - model) * np.abs(self._fit_weights)
        return resid


    def _init_parameters(self, init, fix):
        '''Initialize lmfit parameters dictionary
        '''

        self._make_varnames()
        self._initial_guess(init)

        # initialize parameters
        pars = lmfit.Parameters()
        for vn, vinit in init.items():
            pars.add(vn, value=vinit[0], min=vinit[1], max=vinit[2], vary=1)


        self._init_pars_dataframe(init)

        # setting parameters fixed
        for vn in fix.keys():
            if vn in pars.keys():
                pars[vn].set(value=fix[vn], min=-np.inf, max=np.inf, vary=0)

        # print(pars)
        self._lmpars = pars

    def _get_weights(self, mode):
        '''
        mask data points to exclude them from the fit based on error bars and nan values
        and define weights for the fit
        '''
        dmsd = self.dmsd
        msd = self.msd.copy()
        if dmsd is not None and mode is not 'none':
            excerr = (dmsd.filled(0)<=0)
            wgt = dmsd.copy()
            wgt = np.ma.masked_where(excerr, wgt)
            if mode == 'semilogx':
                wgt = np.log10(wgt)
            elif mode == 'semilogx2':
                wgt = 1/np.log10(wgt/msd)
            elif mode == 'semilogx3':
                wgt = 1/(msd * np.log10(wgt/msd))
            elif (mode == 'equal') or (mode == 'none') or (mode == None):
                wgt = np.ones_like(wgt)
            elif mode == 'logt':
                wgt = 1/np.log10(self.t)
            elif mode == 't':
                wgt = 1/self.t
            elif mode == 'sig**2':
                wgt = 1/wgt**2
            elif mode == 'sig':
                wgt = 1/wgt
            elif mode == 'logsig':
                wgt = 1/np.log10(wgt)
            elif mode == 'data':
                wgt = 1/self.msd.copy()
            elif mode == 'test':
                self._fit_data = np.log10(self._fit_data)
                wgt = 1/np.log10(wgt)
                self._fit_logarithmic = True
            else:
                raise ValueError(f'Error mode {mode} not understood.')
        else:
            wgt = np.ma.masked_array(np.ones_like(self.msd))

        excdat = ~np.isfinite(self.msd) | wgt.mask
        self.msd = np.ma.masked_where(excdat, self.msd)

        self._weights = wgt

    def _init_pars_dataframe(self, init):
        """
        initialize pandas DataFrame for returning results
        """
        names = list(init.keys())
        cols = [names[i // 2] if (i + 1) % 2 else 'd' + names[i // 2]
                for i in range(len(names) * 2)]
        cols.extend(['chisqr', 'redchi', 'bic', 'aic'])
        self.pars = pd.DataFrame(columns=cols)

    def _initial_guess(self, init):
        """
        make initial guess for parameters if not provided by init
        """
        for vn in self._varnames:
            if vn not in init.keys():
                if vn == 'd':
                    init[vn] = (50, 0, None)
                elif vn == 'Dm':
                    init[vn] = (1e8, 0, None)
                elif vn == 'D0':
                    init[vn] = (1e8, 0, None)
                elif vn == 'a':
                    init[vn] = (1, 0, None)

    def _make_varnames(self):
        """Make dict of varnames for easy handling in _calc_model
        """
        self._varnames = {'d': [],
                          'D0': [],
                          'Dm': [],
                          'a': []
                          }

    def _write_to_pars(self, out):
        """ Save fit results in self.pars variable
        """
        for k in self._varnames.keys():
            cond = k in ['__lnsigma']
            gof = np.hstack([out.chisqr, out.redchi, out.bic, out.aic])
            param_name = df_name = k
            value = out.params[param_name].value
            try:
                stderr = 1. * out.params[param_name].stderr
            except TypeError:
                stderr = np.nan
            self.pars.loc[0, df_name] = value
            self.pars.loc[0, 'd'+df_name] = stderr
            self.pars.iloc[0, -4:] = gof




