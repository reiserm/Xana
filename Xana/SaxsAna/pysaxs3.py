import numpy as np
import copy


def get_soq(Isaxs, setup, Vsaxs=None, nbins=1000):

    sI = np.shape(Isaxs)
    if sI == setup.detector.dim:
        ai = setup.ai
        mask = setup.mask
    elif sI == setup.qsec_dim:
        ai = setup.qsec_ai
        mask = setup.qsec_mask
    else:
        raise ValueError(
            f"Average image of shape {sI} does not match defined Azimuthal Integrators."
        )

    if mask is None:
        mask = np.ones_like(sI)
        print("No mask defined.")

    if Vsaxs is None:
        q, ii, e = ai.integrate1d(
            Isaxs,
            nbins,
            mask=~(mask.astype(bool)),
            unit="q_nm^-1",
            error_model="poisson",
        )
    else:
        q, ii, e = ai.integrate1d(
            Isaxs, nbins, mask=~(mask.astype(bool)), unit="q_nm^-1", variance=Vsaxs
        )
    return q, ii, e


def pysaxs(data, load=False, calc_soq=True, **kwargs):

    if load:
        Isaxs = obj.get_item(sid)["Isaxs"]
    elif isinstance(data, dict):
        sid = data["sid"]
        setup = data["setup"]
        Isaxs, Vsaxs = data["get_series"](sid, method="average", **kwargs)
        saxsd = {"Isaxs": Isaxs, "Vsaxs": Vsaxs}
    else:
        raise ValueError("Could not handle input type during SAXS analysis.")

    tmp = get_soq(Isaxs, setup, Vsaxs)
    soq = np.hstack(tmp)
    soq = soq.reshape(-1, 3, order="F")
    saxsd["soq"] = soq

    return saxsd
