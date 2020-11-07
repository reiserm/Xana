import numpy as np
import emcee
import corner
import scipy.optimize as op


def mcmc_sl(x, y, dy, doplot=0):
    A = np.vstack((np.ones_like(x), x)).T
    C = np.diag(dy * dy)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

    def lnlike(theta, x, y, yerr):
        m, b, lnf = theta
        model = m * x + b
        inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
        return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    nll = lambda *args: -lnlike(*args)
    result = op.minimize(nll, [m_ls, b_ls, np.log(0.5)], args=(x, y, dy))
    m_ml, b_ml, lnf_ml = result["x"]

    def lnprior(theta):
        m, b, lnf = theta
        if 0.0 < m and -50.0 < b < 50.0 and -15.0 < lnf < 2.0:
            return 0.0
        return -np.inf

    def lnprob(theta, x, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, x, y, yerr)

    ndim, nwalkers = 3, 100
    pos = [result["x"] + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, dy))
    sampler.run_mcmc(pos, 500)
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    if doplot:
        fig = corner.corner(
            samples, labels=["$m$", "$b$", "$\ln\,f$"], truths=[m_ml, b_ml, lnf_ml]
        )

        xl = np.linspace(x.min(), x.max(), 100)
        for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
            doplot.plot(xl, m * xl + b, color="k", alpha=0.1)
        # plt.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
        # plt.errorbar(x, y, yerr=yerr, fmt=".k")

    samples[:, 2] = np.exp(samples[:, 2])
    m_mcmc, b_mcmc, f_mcmc = map(
        lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
        zip(*np.percentile(samples, [16, 50, 84], axis=0)),
    )
    return m_mcmc, b_mcmc, f_mcmc, m_ls, b_ls
