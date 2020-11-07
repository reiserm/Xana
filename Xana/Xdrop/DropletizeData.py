import numpy as np
import numpy.ma as ma
from Xdrop.dropletizem import dropimgood_sel
import time
import pickle
import re
from matplotlib import pyplot as plt


def dropletizedata(data, pars, mask=None, dark=None, savdir="./", savname=None):
    # print(dropimgood_sel.__doc__)
    dim = data.shape
    mshape = dim[-2:]
    data = data.reshape(-1, *mshape)
    datdrop = np.zeros(data.shape, dtype=np.uint16)

    if dark is None:
        dark = np.zeros(mshape)

    if mask is None:
        mask = np.ones(mshape, dtype=np.bool)

    pix = []
    for imgn in range(data.shape[0]):
        img = data[imgn].astype(int)
        imd = dropimgood_sel(
            img,
            dark,
            ~mask,
            pars["background"],
            pars["lower_threshold"],
            pars["upper_threshold"],
            pars["number_photons"],
            pars["adusPphoton"],
            pars["pixelPdroplet"],
        )

        datdrop[imgn] = imd[-1]
    #        pix.append(imd[:-1])
    if savname is not None:
        np.save(savdir + savname + "_dropletized.npy", datdrop)
        f = open(savdir + savname + "_pix.pkl", "wb")
        pickle.dump({"pix": pix}, f)
        f.close()

    datdrop = np.squeeze(datdrop.reshape(dim))
    return datdrop


def testDropletizing(data, pars, dark=None, mask=None):
    im = data.copy()
    if "roi" in pars.keys():
        roi = pars["roi"]
        xl = roi[0]
        yl = roi[1]
    else:
        xl = [0, dark.shape[1]]
        yl = [0, dark.shape[0]]
    im = im[yl[0] : yl[1], xl[0] : xl[1]]
    if dark is not None:
        dark = dark[yl[0] : yl[1], xl[0] : xl[1]]
        im = im - 1.0 * dark
    if mask is not None:
        mask = mask[yl[0] : yl[1], xl[0] : xl[1]]
        im *= mask
    nx = im.shape[1]
    ny = im.shape[0]
    pars["nx"] = nx
    pars["ny"] = ny
    imd1 = dropletizedata(im, pars, mask=mask, dark=dark)

    tstr = ["raw image", "parameters 1", "parameters 2"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 6))
    ax = ax.ravel()
    im[im < pars["background"]] = 0
    for i, imp in enumerate(
        [
            im,
            imd1,
        ]
    ):
        pl = ax[i].imshow(imp, interpolation="nearest", cmap="Blues")
        ind_x = np.arange(xl[1] - xl[0])
        ind_y = np.arange(yl[1] - yl[0])
        x, y = np.meshgrid(ind_x, ind_y)
        for xi, yi in zip(x.flatten(), y.flatten()):
            if imp[yi, xi] > 0:
                c = "{0}".format(int(imp[yi, xi]))
                ax[i].text(
                    xi,
                    yi,
                    c,
                    va="center",
                    ha="center",
                    fontsize=8,
                    color="w",
                    fontweight="bold",
                )

        ax[i].set_xticks(ind_x - 0.5)
        ax[i].set_yticks(ind_y - 0.5)
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_xlim(ind_x[0] - 0.5, ind_x[-1] + 0.5)
        ax[i].set_ylim(ind_y[0] - 0.5, ind_y[-1] + 0.5)

    I = im.sum()
    ph = imd1.sum()
    ax[0].set_title(
        r"$I={:.00f}\,adus\, ({:.00f}ph$)".format(I, I / pars["adusPphoton"])
    )
    ax[1].set_title(r"$I={:.00f}\,ph$".format(ph))
    plt.tight_layout(pad=2)
    """
    number = 0
    if saveimages:
        while os.path.isfile(savdir + savname):
            number = int(re.findall(r'\d+', savname)[0]) + 1
            savname = 'pic_{0:02}.pdf'.format(number)
        plt.savefig(savdir + savname)#, dpi=300)
    """
    plt.show()
