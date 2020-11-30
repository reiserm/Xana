from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import matplotlib.path as path
from matplotlib.colors import ListedColormap

from time import sleep
import ipywidgets as widgets


def masker(data, mask1=[]):

    if not isinstance(mask1, np.ndarray):
        mask1 = np.ones_like(data)

    xs = []
    ys = []
    mask = []

    def invert_mask(*args):
        ind = mask1 > 0
        mask1[ind] = 0
        mask1[~ind] = 1
        mask.set_data(mask1)

    def reset_mask(*args):
        mask1[:] = 1
        mask.set_data(mask1)

    def save_mask(*args):
        savname = savwid.value
        savname = savname.replace(".npy", "") + ".npy"
        np.save(savname, mask1.astype(bool))

    def cancel_masking(*args):
        del xs[:]
        del ys[:]
        update_plot()

    def mask_roi(*args):
        lx, ly = np.shape(mask1)
        x, y = np.meshgrid(np.arange(lx), np.arange(ly))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        xy = np.vstack((np.array(ys), np.array(xs))).T
        new_submask = ~path.Path(xy).contains_points(points).reshape(ly, lx).T
        np.multiply(mask1, new_submask, out=mask1)
        del xs[:]
        del ys[:]
        update_plot()

    def mask_outside(*args):
        lx, ly = np.shape(mask1)
        x, y = np.meshgrid(np.arange(lx), np.arange(ly))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        xy = np.vstack((np.array(ys), np.array(xs))).T
        new_submask = path.Path(xy).contains_points(points).reshape(ly, lx).T
        np.multiply(mask1, new_submask, out=mask1)
        del xs[:]
        del ys[:]
        update_plot()

    def unmask_roi(*args):
        lx, ly = np.shape(mask1)
        x, y = np.meshgrid(np.arange(lx), np.arange(ly))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        xy = np.vstack((np.array(ys), np.array(xs))).T
        new_submask = path.Path(xy).contains_points(points).reshape(ly, lx).T
        mask1[new_submask] = True
        del xs[:]
        del ys[:]
        update_plot()

    def above_masking(*args):
        thres = int(mvwid.value)
        mask1[data >= thres] = False
        update_plot()

    def below_masking(*args):
        thres = int(mvwid.value)
        mask1[data <= thres] = False
        update_plot()

    def onkey(event, line):
        if event.key == "c":
            cancel_masking()
        elif event.key == "m":
            mask_roi()
        elif event.key == "o":
            mask_outside()
        elif event.key == "u":
            unmask_roi()
        elif event.key == "i":
            invert_mask()
        elif event.key == "a":
            above_masking()
        elif event.key == "b":
            below_masking()

    def onclick(event, line):
        xs.append(event.xdata)
        ys.append(event.ydata)

        line.set_data(xs, ys)
        line.figure.canvas.draw_idle()

    def update_plot(*args):
        line.set_data(xs, ys)
        mask.set_data(mask1.astype(int))

    mvwid = widgets.Text(
        value="0", description="mask value:", placeholder="type herer", disabled=False
    )
    above_button = widgets.Button(
        description="mask >= (a)",
        disabled=False,
        tooltip="all values set to default",
    )
    above_button.on_click(above_masking)
    below_button = widgets.Button(
        description="mask <= (b)",
        disabled=False,
        tooltip="all values set to default",
    )
    below_button.on_click(below_masking)
    row_thres = widgets.HBox([mvwid, above_button, below_button])

    cancel_button = widgets.Button(
        description="cancel (c)",
        disabled=False,
        tooltip="cancel roi",
    )
    cancel_button.on_click(cancel_masking)
    reset_button = widgets.Button(
        description="reset",
        disabled=False,
        tooltip="all values set to default",
    )
    reset_button.on_click(reset_mask)

    row3 = widgets.HBox([cancel_button, reset_button])

    mask_button = widgets.Button(
        description="mask (m)",
        disabled=False,
        tooltip="mask roi",
        icon="",
    )
    mask_button.on_click(mask_roi)
    unmask_button = widgets.Button(
        description="unmask (u)",
        disabled=False,
        tooltip="unmask roi",
    )
    unmask_button.on_click(unmask_roi)
    outside_button = widgets.Button(
        description="mask outside (o)",
        disabled=False,
        tooltip="mask everything outside the roi",
    )
    outside_button.on_click(mask_outside)
    invert_button = widgets.Button(
        description="invert (i)",
        disabled=False,
        tooltip="invert the complete mask",
        icon="",
    )
    invert_button.on_click(invert_mask)

    row2 = widgets.HBox([mask_button, unmask_button, outside_button, invert_button])

    savwid = widgets.Text(
        value="example_mask.npy",
        placeholder="type here",
        description="Filename:",
        disabled=False,
    )
    cdir_button = widgets.Button(
        description="change directory",
        disabled=False,
        tooltip="change saving directory",
    )
    save_button = widgets.Button(
        description="save mask",
        disabled=False,
        tooltip="save mask to npy file",
    )
    save_button.on_click(save_mask)
    row_save = widgets.HBox([savwid, cdir_button, save_button])

    out = widgets.Output()
    display(out)
    with out:

        display(widgets.VBox([row2, row3, row_thres, row_save]))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_position([0.06, 0.05, 0.94, 0.94])

        cmap = mpl.colors.ListedColormap(["orange", "white"])

        # Get the colormap colors
        my_cmap = cmap(np.arange(cmap.N))

        # Set alpha
        my_cmap[:, -1] = 0.7

        # Create new colormap
        cmap = ListedColormap(my_cmap)
        cmap.set_over("orange", alpha=0)

        bounds = [0, 0.5, 1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(
            data,
            interpolation="nearest",
            norm=mpl.colors.LogNorm(),
            aspect="auto",
        )
        mask = plt.imshow(
            mask1.astype(int),
            interpolation="nearest",
            norm=norm,
            cmap=cmap,
            aspect="auto",
        )
        plt.axis("equal")
        (line,) = ax.plot([], [], ".-k")
        cid_m = fig.canvas.mpl_connect(
            "button_press_event", lambda event: onclick(event, line)
        )
        cid_k = fig.canvas.mpl_connect(
            "key_press_event", lambda event: onkey(event, line)
        )
        plt.show()
