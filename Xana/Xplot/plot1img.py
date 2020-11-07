import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from ipywidgets import interact, fixed

from readDet.openFile import getNXimagedata
from readDet.processData import getFilenames
from misc.nxsinfo import nxsinfo

import h5py


def getimg(run, imgn, fnames, dim, ffmask=0, qroi=0, rgn=0):
    img = getNXimagedata(
        fnames[run], first=(imgn, 0, 0), last=(imgn + 1, dim[1], dim[0])
    )
    img = np.squeeze(img)
    if rgn:
        img = img[rgn[0] : rgn[1], rgn[2] : rgn[3]]
        if type(ffmask) == np.ndarray:
            ffmask = ffmask[rgn[0] : rgn[1], rgn[2] : rgn[3]]
    if type(ffmask) == np.ndarray:
        img = img * ffmask
        gp = np.sum(ffmask > 0)
    else:
        gp = dim[0] * dim[1]
    if qroi:
        for i in range(len(qroi)):
            img[qroi[i]] = np.max(img)
    return img, gp


def browse_images(
    nf=0,
    run=0,
    fnames=0,
    dim=0,
    exp=0,
    gp=0,
    im=0,
    txt=0,
    fig=0,
    ffmask=0,
    qroi=0,
    rgn=0,
):
    def view_image(i=0, **kargs):
        img = getimg(run, i, fnames, dim, ffmask, qroi, rgn)[0]
        im.set_data(img)
        textstr = get_text(img, run, i, exp, gp)
        txt.set_text(textstr)
        fig.canvas.draw_idle()

    interact(
        view_image,
        i=(0, nf - 1),
        run=fixed(run),
        fnames=fixed(fnames),
        dim=fixed(dim),
        exp=fixed(exp),
        gp=fixed(gp),
        im=fixed(im),
        txt=fixed(txt),
        fig=fixed(fig),
        ffmask=fixed(ffmask),
        qroi=fixed(qroi),
        rgn=fixed(rgn),
    )


def get_text(img, run, imgn, exp, gp):
    img_sum = np.sum(img)
    img_mean = img_sum / gp
    mx = np.max(img)
    if type(exp) == np.ndarray:
        exp = exp[run]
        ill = imgn * (exp + 0.001)
    textstr = (
        r"$\mathrm{{I}}_{{img}} = {0:.3g}$"
        + "\n"
        + r"$\langle \mathrm{{I}}\rangle _{{pix}} = {1:.3g}$"
        + "\n"
        + r"$\mathrm{{I}}_{{max}} = {6:.3g}$"
        + "\n"
        + r"$\mathrm{{t}}_{{exp}} = {2:.3g}s$"
        + "\n"
        + r"$\mathrm{{t}}_{{ill}} = {3:.3g}s$"
        + "\n"
        + r"$\mathrm{{run}}={4:d}$"
        + "\n"
        + r"$\mathrm{{imgn}}={5:d}$"
    ).format(img_sum, img_mean, exp, ill, run, imgn, mx)
    return textstr


def plot1img(run, datdir, imgn, ffmask=0, rgn=0, exp=0, qroi=0, cb=0, norm=None):
    if type(datdir) == str:
        fnames = getFilenames(datdir, "nxs")
    else:
        fnames = datdir
    ct, nf = nxsinfo(fnames[run])
    if type(ffmask) == np.ndarray:
        m, n = np.shape(ffmask)
    else:
        m, n = np.shape(getNXimagedata(fnames[run], last=(1,)))[1:]
    img, gp = getimg(run, imgn, fnames, (m, n), ffmask, qroi, rgn)
    out = img.copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    ax.tick_params("both", length=0, width=0, which="major")
    if cb:
        im = ax.imshow(
            img, vmin=cb[0], vmax=cb[1], cmap=plt.get_cmap("magma"), norm=norm
        )
    else:
        im = ax.imshow(img, cmap=plt.get_cmap("magma"), norm=norm)
    plt.colorbar(im)
    textstr = get_text(img, run, imgn, exp, gp)
    props = dict(boxstyle="round", facecolor="white")
    txt = ax.text(
        -0.4,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=props,
    )
    propdic = {
        "nf": nf,
        "run": run,
        "fnames": fnames,
        "dim": (m, n),
        "exp": exp,
        "gp": gp,
        "im": im,
        "txt": txt,
        "fig": fig,
        "ffmask": ffmask,
        "qroi": qroi,
        "rgn": rgn,
    }
    plt.show(block=0)
    return out, propdic
