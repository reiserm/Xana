import numpy as np
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator


def get_soq(Isaxs, mask, setup, Vsaxs=None, nbins=1000):
    ai = AzimuthalIntegrator(
        dist=setup["distance"],
        pixel1=setup["pix_size"][0] * 1e-6,
        pixel2=setup["pix_size"][1] * 1e-6,
    )
    ai.setFit2D(setup["distance"] * 1000, setup["ctr"][0], setup["ctr"][1])
    ai.wavelength = setup["lambda"] * 1e-10
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
