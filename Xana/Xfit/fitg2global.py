import numpy as np
import lmfit
from .g2function import g2 as g2func
from copy import copy
import re
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from ..Xplot.niceplot import niceplot

class G2:

    def __init__(self, cf, nq, dcf=None):

        self.t = np.ma.masked_invalid(cf[1:,0])
        self.qv = cf[0,1:].copy()
        self.nq = np.asarray(nq)

        self.cf = np.ma.masked_invalid(cf[1:,1:]).T
        ind_sort = np.argsort(self.cf[0])
        self.t = self.t[ind_sort]
        self.cf = self.cf[:,ind_sort]

        if isinstance(dcf, np.ndarray):
            self.dcf = np.ma.masked_invalid(dcf[1:,1:]).T
            self.dcf = self.dcf[:,ind_sort]
        else:
            self.dcf = None

        # properties
        self.nmodes = 1
        self.fitglobal = []

        # attributes defined in private methods
        self._lmpars = None
        self._weights = None
        self._sum_residuals = False

        # fit results
        self.pars = None
        self.fit_result = []

    @property
    def nmodes(self):
        return self.__nmodes

    @nmodes.setter
    def nmodes(self, nmodes):
        if nmodes <= 0:
            print('Number of modes has been set to 1.')
            self.__nmodes = 1
        else:
            self.__nmodes = nmodes

    @property
    def fitglobal(self):
        return self.__fitglobal

    @fitglobal.setter
    def fitglobal(self, fitglobal):
        dofitglobal = bool(fitglobal)
        if dofitglobal:
            self.ndat = len(self.nq)
        else:
            self.ndat = 1
            fitglobal = []
        self.__fitglobal = fitglobal

    def fit(self, mode='semilogx', nmodes=1, fitglobal=[], init={}, fix={}, lmfit_pars={},):
        '''
        Function that computes the fits using lmfit's minimizer
        '''        

        self.fitglobal = fitglobal
        self.nmodes = nmodes

        # make the parameters
        if lmfit_pars.get('method', 0) == 'emcee':
            self._sum_residuals = True
            if not bool(init.get('__lnsigma',0)):
                init['__lnsigma'] = 1
        self._init_parameters(init, fix)

        # setting the weights of the data
        self._get_weights(mode)

        if bool(self.fitglobal):
            out = lmfit.minimize(self._residuals, params=self._lmpars,
                                 args=(self.t,), kws={'data':self.cf[self.nq],
                                                      'eps':self._weights[self.nq]},
                                 nan_policy='omit', **lmfit_pars)

            self.pars['q'] = self.qv[self.nq]
            for i, vn in enumerate(out.params.keys()):
                if '_' not in vn:
                    idx = 0
                    nm = vn
                else:
                    nm, idx = vn.split('_')
                    idx = int(idx) + 1
                self.pars.loc[idx, nm] = out.params[vn].value
                try:
                    err = 1.*out.params[vn].stderr
                except TypeError:
                    err = np.nan
                self.pars.loc[idx, 'd'+nm] = err

            gof = np.array([out.chisqr, out.redchi, out.bic, out.aic])                      
            self.pars.iloc[:,-4:] = gof
            self.pars = self.pars.apply(pd.to_numeric)
            self.fit_result.append((out, lmfit.fit_report(out)))

        else:
            
            for qi in self.nq:
                out = lmfit.minimize(self._residuals, params=self._lmpars,
                                     args=(self.t,), kws={'data':self.cf[qi:qi+1],
                                                          'eps':self._weights[qi:qi+1]},
                                     nan_policy='omit', **lmfit_pars)

                pars_arr = np.zeros((len(out.params), 2))
                for i, vn in enumerate(out.params.keys()):
                    pars_arr[i,0] = out.params[vn].value
                    try:
                        pars_arr[i,1] = 1.*out.params[vn].stderr
                    except TypeError:
                        pars_arr[i,1] = 1
                gof = np.array([out.chisqr, out.redchi, out.bic, out.aic])
                pars = np.hstack((self.qv[qi], pars_arr.flatten(), gof))
                # print(pars.shape)
                # print(self.pars.shape)
                self.pars.loc[self.pars.shape[0]] = pars
                self.fit_result.append((out, lmfit.fit_report(out)))

        return self.pars, self.fit_result

    def plot(self, doplot=False, marker='o', ax=None, xl=None, yl=None, colors=None, alpha=1.,
             markersize=3., data_label=None, confint=False, **kwargs):

        sucfit = True if self.pars is not None else False

        if xl is None:
            if ax is None:
                ax = plt.gca()
            xl = ax.get_xlim()
            if xl[0] == 0:
                xl = (np.min(self.t)*0.5,np.max(self.t)*1.5)
            
        xf = np.logspace(np.log10(xl[0]),np.log10(xl[1]), 50)

        for ii, iq in enumerate(self.nq):
            pl = []
            if sucfit:
                v = self.pars.iloc[ii]
                g2f = 0
                for i in range(self.nmodes):
                    ve = f'{i}'
                    g2f += g2func(xf, t=v['t'+ve], b=v['b'+ve], g=v['g'+ve],
                                a=v['a']*1.*(not bool(i)))
                if confint:
                    g2fci = np.zeros((2,len(xf)))
                    for i in range(self.nmodes):
                        ve = f'{i}'
                        for j in range(2):
                            g2fci[j] += g2func(xf,
                                               t=v['t'+ve]*(1 + (-1)**j * .01 * confint.get('t'+ve, 0)),
                                               b=v['b'+ve]*(1 + (-1)**j * .01 * confint.get('b'+ve, 0)),
                                               g=v['g'+ve]*(1 + (-1)**j * .01 * confint.get('g'+ve, 0)),
                                               a=v['a']*1.*(not bool(i)))

                labstr_fit = None
                if 'legf' in doplot and sucfit:
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

            if 'fit' in doplot and sucfit:
                if 'g1' in doplot:
                    g2f = np.sqrt((g2f-out.params['a'].value)/out.params['b0'].value)
                pl.append(ax.plot(xf, g2f, '-', label=labstr_fit, linewidth=1))
                if confint:
                    for j in range(2):
                        pl.append(ax.plot(xf, g2fci[j], ':', linewidth=1))

            if 'data' in doplot:
                if 'g1' in doplot:
                    cf = np.sqrt((cf-out.params['a'].value)/out.params['b0'].value)
                if self.dcf is not None:
                    pl.append(ax.errorbar(self.t.filled(np.nan), self.cf[iq].filled(np.nan), yerr=self.dcf[iq].filled(np.nan),
                                          linestyle='', marker=marker, label=labstr_data,
                                          alpha=alpha, markersize=markersize))
                else:
                    pl.append(ax.plot(self.t, self.cf[iq], marker, label=labstr_data,
                                      alpha=alpha, markersize=markersize))

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

            if 'g1' in doplot:
                ax.set_xscale('linear')
                ax.set_yscale('log')
            else:
                ax.set_xscale('log')
                ax.set_yscale('linear')

            if 'leg' in doplot:
                ax.legend()

            if 'report' in doplot and sucfit:
                print(lmfit.fit_report(out))

            ax.set_xlabel(r'delay time $\tau$ [s]')
            ax.set_ylabel(r'$g_2(\tau)$')

            niceplot(ax, autoscale=0)
            ax.set_xlim(*xl)
            if yl is not None:
                ax.set_ylim(*yl)

        return None
    
    def _residuals(self, pars, x, data=None, eps=0):
        """2D Residual function to minimize
        """
        v = pars.valuesdict()
        resid = np.zeros((self.ndat,self.t.size))
        model = np.zeros(x.size, dtype=np.float32)
        for j in range(self.ndat):
            jj = j - 1
            for i in range(self.nmodes):
                ve = f'{i}' + f'_{jj}'*bool(j)
                model += g2func(x, t=v['t'+ve], b=v['b'+ve], g=v['g'+ve],
                            a=v['a'+f'_{jj}'*bool(j)]*1.*(not bool(i)))
            if np.sum(eps) > 0:
                resid[j] = (data[j] - model) * np.abs(eps[j])
            else:
                resid[j] = (data[j] - model)
            model *= 0

        if not self._sum_residuals:
            return np.squeeze(resid.flatten())
        else:
            return np.sum(resid.flatten())

    def _init_parameters(self, init, fix):
        '''Initialize lmfit parameters dictionary
        '''

        self._initial_guess(init)

        # initialize parameters
        pars = lmfit.Parameters()
        for vn, vinit in init.items():
            pars.add(vn, value=vinit[0], min=vinit[1], max=vinit[2], vary=1)
        pars.add('a', value=init['a'][0], min=init['a'][1], max=init['a'][2], vary=1)
        pars.add('beta', value=init['beta'][0], min=init['beta'][1], max=init['beta'][2], vary=1)

        self._init_pars_dataframe(init)
        
        # setting contrast constraint
        if self.nmodes > 1:
            beta_constraint = 'a + beta - 1 -' + '-'.join([f'b{x}' for x in range(1,self.nmodes)])
            pars['b0'].set(expr=beta_constraint)
        else:
            pars['b0'].set(expr='beta')

        # setting parameters fixed
        for vn in fix.keys():
            if vn in pars.keys():
                pars[vn].set(value=fix[vn], min=-np.inf, max=np.inf, vary=0)

        # modifying pars for global fitting if not globalfit then ndat=1
        pv = list(pars.values())
        for j in range(self.ndat-1):
            for p in pv:
                pt = copy(p)
                pt.name += '_' + str(j)
                pars.add(pt)

        re_beta = re.compile('beta_\d{1}')
        re_b = re.compile('b\d{1}')
        for p in list(pars.keys()):
            if '_' in p and p.startswith('b') and not p.startswith('be'):
                pt = p.split('_')[-1]
                texp = pars[p.split('_')[0]].expr
                if bool(texp) and self.nmodes > 1:
                    texp = re_beta.sub('beta_%s' % pt, texp)
                    tmp  = re_b.findall(texp)[0]
                    texp = re_b.sub(tmp+'_%s' % pt, texp)
                    pars[p].set(expr=texp.replace('beta', 'beta_%d' % int(pt)))
                
        for gpar in self.fitglobal:
            for vn in pars.keys():
                if gpar in vn and len(vn.replace(gpar, '')):
                    pars[vn].set(expr=pars[gpar].name)

        self._lmpars = pars

    def _get_weights(self, mode):
        '''
        mask data points to exclude them from the fit based on error bars and nan values
        and define weights for the fit
        '''
        dcf = self.dcf
        if dcf is not None:
            excerr = (dcf.filled(0)<=0)
            wgt = dcf.copy()
            wgt = np.ma.masked_where(excerr, wgt)
            if mode == 'semilogx':
                wgt = np.log10(wgt)
            elif mode == 'equal':
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
                wgt = 1/self.cf.copy()
            elif mode == 'none' or mode == None:
                wgt *= 0
                wgt = wgt.astype('uint8')
            else:
                raise ValueError(f'Error mode {mode} not understood.')
        else:
            excerr = np.zeros_like(self.cf, bool)
            wgt = None

        excdat = ~np.isfinite(self.cf) | wgt.mask
        self.cf = np.ma.masked_where(excdat, self.cf)

        self._weights = wgt

    def _init_pars_dataframe(self, init):
        """
        initialize pandas DataFrame for returning results
        """
        names = list(init.keys())
        cols = [names[i // 2] if (i + 1) % 2 else 'd' + names[i // 2]
                for i in range(len(names) * 2)]
        cols.insert(0, 'q')
        cols.extend(['chisqr', 'redchi', 'bic', 'aic'])
        self.pars = pd.DataFrame(columns=cols)

    def _initial_guess(self, init):
        """
        make initial guess for parameters if not provided by init
        """
        for i in range(self.nmodes):
            for s in 'tgb':
                vn = s+'{}'.format(i)
                if vn not in init.keys():
                    if s == 't':
                        t0 = np.logspace(np.log10(self.t.min()), np.log10(self.t.max()), self.nmodes+2)[i+1]
                        init[vn] = (t0, -np.inf, np.inf)
                    elif s == 'g':
                        init[vn] = (1, .2, 1.8)
                    elif s == 'b':
                        init[vn] = (.1, 0, 1)

        if 'a' not in init:
            init['a'] = (1, 0, 2)
        if 'beta' not in init:
            init['beta'] = (0.2, 0, 1)
        if bool(init.get('__lnsigma', 0)):
            if not isinstance(init['__lnsigma'], tuple):
                init['__lnsigma'] = (np.log(0.1), None, None)

    def _parameter_expr(self,):
        pass

