import h5py
import numpy as np


def nxsinfo(filename):
    f = h5py.File(filename)
    ct = f["/entry/instrument/detector/count_time/"].value / 1000.0
    nf = f["/entry/instrument/detector/data/"].shape
    nf = nf[0]
    f.close()
    return float(ct), nf
