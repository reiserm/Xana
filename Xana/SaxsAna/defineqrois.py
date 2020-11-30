import numpy as np
from ..Xplot.plotqrois import plotqrois


def check_dimension(setup, img):
    s = np.shape(img)
    if s != setup.detector.dim:
        setup.dim = s
        print(f"Detector dimensiion has been changed to {s}.")
    else:
        return None


def getqroi(saxs, setup, qr, phir=None, mirror=False):

    q = setup.ai.array_from_unit(unit="q_nm^-1")
    radius = setup.ai.array_from_unit(unit="r_m") / setup.detector.pixel1
    phi = setup.ai.chiArray()

    qv = qr[:, 0]
    dqv = qr[:, 1]

    if phir is None:
        phiv = [0.0]
        dphi = [360.0]
    else:
        phiv = phir[:, 0]
        dphi = phir[:, 1]

    phiv = phiv * np.pi / 180
    dphi = dphi * np.pi / 180

    ind = []
    r = []
    ph = []

    for i in range(len(qv)):
        for j in range(len(phiv)):
            tmp_q = (
                (q >= (qv[i] - dqv[i] / 2)) & (q <= (qv[i] + dqv[i] / 2)) & setup.mask
            )
            phit = phi.copy()
            phit = (phit - phiv[j]) % (2 * np.pi)
            if mirror:
                tmp_phi = (phit <= dphi[j]) | (
                    ((phit - np.pi) % (2 * np.pi)) <= dphi[j]
                )
            else:
                tmp_phi = phit <= (dphi[j])
            tmp = np.where(tmp_q & tmp_phi)
            del phit
            if len(tmp[0]):
                ind.append(tmp)
                r_min = radius[tmp].min()
                r_max = radius[tmp].max()
                r.append((r_max, r_max - r_min))
    return ind, r


def flatten_init(inp):
    def convert(s):
        if np.issubdtype(type(s[0]), np.number):
            return (np.array([s[0]]), s[1])
        else:
            return s

    def get_stack(p):
        p = convert(p)
        return np.hstack((p[0], np.ones(len(p[0])) * p[1])).reshape(-1, 2, order="F")

    i = 0
    for p in inp:
        stack = get_stack(p)
        if i == 0:
            rois = stack.copy()
        else:
            rois = np.vstack((rois, stack))
        i += 1
    return rois


def defineqrois(
    setup,
    Isaxs,
    qv_init=None,
    phiv_init=[(0, 360)],
    plot=False,
    d=250,
    mirror=False,
    **kwargs,
):

    # check_dimension(setup, Isaxs)

    if qv_init is None:
        try:
            qv_init = setup.qv_init
            if phiv_init is None and "phiv_init" in vars(setup):
                phiv_init = setup.phiv_init
        except KeyError:
            print('Setup does not contain "qv_init" to initialize Q ROIs.')
    else:
        qv_init = flatten_init(qv_init)
        phiv_init = flatten_init(phiv_init)
        setup.qv_init = qv_init
        setup.phiv_init = phiv_init

    phiv_init[:, 0] -= phiv_init[:, 1] / 2

    qroi, radii = getqroi(Isaxs, setup, qv_init, phir=phiv_init, mirror=mirror)

    setup.dqv = qv_init[:, 1]
    setup.phiv = phiv_init
    setup.qv = np.repeat(qv_init[:, 0], setup.phiv.shape[0])
    setup.radii = radii

    setup.gproi = np.array([len(x[0]) for x in qroi], dtype=int)
    setup.qroi = qroi
    setup.qv = setup.qv[: len(setup.qroi)]

    xmin = min([x[0].min() for x in setup.qroi])
    ymin = min([x[1].min() for x in setup.qroi])
    xmax = max([x[0].max() for x in setup.qroi])
    ymax = max([x[1].max() for x in setup.qroi])

    qsec = ((xmin, ymin), (xmax, ymax))
    qsec_dim = (xmax - xmin + 1, ymax - ymin + 1)
    setup.qsec = qsec
    setup.qsec_dim = qsec_dim
    setup.qsec_mask = setup.mask[
        qsec[0][0] : qsec_dim[0] + qsec[0][0], qsec[0][1] : qsec_dim[1] + qsec[0][1]
    ]
    setup.qsec_center = (setup.center[0] - qsec[0][1], setup.center[1] - qsec[0][0])
    setup.qsec_ai = setup._update_ai(setup.qsec_center)

    print("Added the following Q-values [nm-1]:\n{}".format(setup.qv))

    if plot:
        plotqrois(Isaxs, setup, method=plot, d=d, shade=True, mirror=mirror, **kwargs)
