kb = 1.381e-23
R = 11e-9
T = 25
e_rheo = 10


def getD(eta, err=0):
    dD = 0
    D = kb * (T + 273.15) / (6 * np.pi * R * eta)
    if err:
        dD = D * err / eta
    return D, dD


def geteta(D, err=0):
    deta = 0
    eta = kb * (T + 273.15) / (6 * np.pi * R * D * 1e-18)
    if err:
        deta = eta * err / D
    return eta, deta


fig, ax = plt.subplots(2, 2, figsize=(9, 8))
ax = ax.ravel()
cmap = plt.get_cmap("Set1")

qp = np.arange(nq)

qvn = qv[qp] * 10
qvn2 = qvn ** 2
x = np.linspace(np.min(qvn), np.max(qvn), 100)
x2 = x * x

# -------plot decay rates--------
for i in range(2):
    y = 1 / rates[qp, 1 + 3 * i, 0]
    dy = y ** 2 * rates[qp, 1 + 3 * i, 1]
    ax[0].errorbar(
        qvn2[qp],
        y,
        dy,
        fmt="o",
        color=cmap(i),
        alpha=0.6,
        label=r"$\Gamma_{}$".format(i),
    )

    nfl = [np.arange(9)]
    if i == 0:
        nfl.append(np.arange(11, 16))
        qp = np.arange(9)

    if i == 1:
        continue

    for nf in nfl:
        nf = np.delete(nf, np.where(rates[nf, 1, 1] == 0))
        D_mc, b_mc = emceefit_sl(qvn2[nf], y[nf], dy[nf])[:2]
        D_mc, b_mc = map(lambda x: (x[0], np.mean(x[1:])), [D_mc, b_mc])
        popt, pcov = np.polyfit(qvn2[nf], y[nf], 1, w=1 / dy[nf], cov=1)
        perr = np.sqrt(np.diag(pcov))
        D_exp, b_exp = (popt[0], perr[0]), (popt[1], perr[1])

        y_dif = y[nf] / qvn2[nf]
        dy_dif = dy[nf] / qvn2[nf]
        D_dif = (
            np.sum(y_dif / dy_dif ** 2) / np.sum(1 / dy_dif ** 2),
            np.sqrt(1 / np.sum(1 / dy_dif ** 2)),
        )

        e_dif = geteta(*D_dif)
        e_exp = geteta(*D_exp)
        e_mle = geteta(*D_mc)
        D_rheo = getD(e_rheo)

        ax[0].plot(x2, np.polyval(popt, x2), color=cmap(i))
        # ax[0].plot(x2, np.polyval([D_dif[0],0],x2), color='gray')
        print("\nResults for {} decay:".format(i + 1))
        print("-" * 20)
        print("D_rheo = {:.2f} nm2s-1".format(D_rheo[0] * 1e18))
        print(r"D_exp = {:.2f} +/- {:.2f} nm2s-1".format(*D_exp))
        print(r"b_exp = {:.2f} +/- {:.2f} s-1".format(*b_exp))
        print(r"D_mle = {:.4f} +/- {:.4f} nm2s-1".format(*D_mc))
        print(r"b_mle = {:.4f} +/- {:.4f} s-1".format(*b_mc))
        print(r"D_dif = {:.2f} +/- {:.2f} nm2s-1".format(*D_dif))
        print("\neta_rheo = {:.2f} Pas".format(e_rheo))
        print(r"eta_exp = {:.2f} +/- {:.2f} Pas".format(*e_exp))
        print(r"eta_mle = {:.4f} +/- {:.4f} Pas".format(*e_mle))
        print(r"eta_dif = {:.2f} +/- {:.2f} Pas".format(*e_dif))

# -------plot KWW exponent--------
qp = np.arange(nq)
g = rates[qp, 2, 0]
dg = rates[qp, 2, 1]
ax[2].errorbar(
    qvn, g, dg, fmt="o", color=cmap(0), alpha=0.6, label=r"$\gamma_{}$".format(0)
)
qp = np.arange(9)
g = rates[qp, 5, 0]
dg = rates[qp, 5, 1]
ax[2].errorbar(
    qvn[qp], g, dg, fmt="s", color=cmap(1), alpha=0.6, label=r"$\gamma_{}$".format(1)
)
# -------plot nonergodicity parameter--------
def blc(q, L, k, lc):
    def A(q):
        return 4 * np.pi / lc * q / k * np.sqrt(1 - q ** 2 / (4 * k ** 2))

    return 2 * (A(q) * L - 1 + np.exp(-A(q) * L)) / (A(q) * L) ** 2


b = np.sum(rates[:, 3::3, 0], 1)
db = np.sum(rates[:, 3::3, 1], 1)

ax[1].errorbar(qvn2, b, db, fmt="o", color=cmap(0), alpha=0.6, label=r"$f_0(q)$")

nf = np.arange(8)
nf = np.delete(nf, np.where(rates[nf, 1, 1] == 0))
y = np.log(b[nf])
dy = db[nf] / b[nf]

popt1, cov1 = np.polyfit(qvn2[nf], y, 1, w=1 / dy, cov=1)
cov1 = np.sqrt(np.diag(cov1))[0]
ax[1].plot(x2, exp(np.polyval(popt1, x2)), "-", color=cmap(0))
# label=r'${:.3g}\, \exp{{(-q^2\cdot{:.3g}nm^{{2}})}}$'.format(np.exp(popt1[1]),-1*popt1[0]))

# ----
b = rates[qp, 6, 0]
db = rates[qp, 6, 1]
ax[1].errorbar(qvn2[qp], b, db, fmt="s", color=cmap(1), alpha=0.6, label=r"$f_1(q)$")

# ----

y_th = exp(popt[1]) * blc(x, 1e-5, 2 * np.pi / (12.4 / 21 / 10), 1e-6)

y = np.mean(cnorm[2 : nq + 2, 1:6], 1)
dy = np.std(cnorm[2 : nq + 2, 1:6], 1)
ax[1].errorbar(qvn2, y, dy, fmt="o", color="k", alpha=0.6, label=r"$\beta_0(q)$")

popt2, cov2 = np.polyfit(qvn2, np.log(y), 1, w=y / dy, cov=1)
cov2 = np.sqrt(np.diag(cov2))[0]
ax[1].plot(x2, exp(np.polyval(popt2, x2)), "-", color="k")
# ax[1].legend(loc=3)

x_labels = ax[1].get_xticks()
ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0g"))
c = popt2[0] - popt1[0]
r_loc = np.sqrt(6 * (c) / 2)
dr_loc = 3 / 2 / r_loc * np.sqrt(cov1 ** 2 + cov2 ** 2)
print("\n\nlocalization length: {:.2f} +/- {:.2f} nm".format(r_loc, dr_loc))


# set style
ax[0].set_xlabel(r"$\mathrm{q}^2$ [$\mathrm{nm}^{-2}]$")
ax[0].set_ylabel(r"$\Gamma [s^{-1}]$")
ax[2].set_xlabel(r"$\mathrm{q}$ [$\mathrm{nm}^{-1}]$")
ax[2].set_ylabel(r"KWW exponent")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"$\mathrm{q}^2$ [$\mathrm{nm}^{-2}]$")
ax[1].set_ylabel(r"contrast")
for axi in ax:
    axi.legend(loc="best")
    # niceplot(axi)
plt.tight_layout()
plt.savefig("um2018/cfpars1.eps")
