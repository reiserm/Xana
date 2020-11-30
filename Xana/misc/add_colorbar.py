import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def add_colorbar(
    ax,
    vec,
    label=None,
    cmap="magma",
    discrete=False,
    tick_step=1,
    qscale=0,
    location="right",
    show_offset=False,
    **kwargs
):

    ncolors = len(vec)
    tick_indices = np.arange(0, len(vec), tick_step)
    vec = np.array(vec)[tick_indices]
    vec *= 10 ** qscale
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)

    change_ticks = False
    if discrete:
        norm = mpl.colors.NoNorm()
        change_ticks = True
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap, ncolors)
        elif isinstance(cmap, (list, np.ndarray)):
            cmap = mpl.colors.ListedColormap(cmap)
    else:
        cmap = plt.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vec.min(), vmax=vec.max())

    if location == "right":
        orientation = "vertical"
    elif location == "top":
        orientation = "horizontal"

    cax = mpl.colorbar.make_axes(ax, location=location, **kwargs)[0]
    cb = mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap, orientation=orientation)

    # set up color bar ticks
    if location == "top":
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

    cb.set_label(label)
    if change_ticks:
        cb.set_ticks(tick_indices)
        cb.set_ticklabels(list(map(lambda x: "$%.{}f$".format(3 - qscale) % x, vec)))
        if qscale and show_offset:
            cb.ax.text(
                1.0,
                1.04,
                r"$\times 10^{{-{}}}$".format(qscale),
                transform=cb.ax.transAxes,
            )
    cb.ax.invert_yaxis()
    cb.ax.set_in_layout(True)
    cax.grid(False)
    cb.ax.grid(False)
