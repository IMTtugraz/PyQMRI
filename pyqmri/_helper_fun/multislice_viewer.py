#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple Volumetric image viewer with scrolling option."""
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# 3D viewer  ##########s#######################################################
###############################################################################


def imshow(volume, vmin=None, vmax=None):
    """Volumetric image viewer for python.

    Shows up to 4D volumes. volume is assumed to be real valued.

    Parameters
    ----------
      volume : numpy.array
        The (real values) image data to display
      vmin : float, None
        Minimum value of the display window. If None, uses minimum value in
        middle slice of the volume.
      vmax : float, None
        Maximum value of the display window. If None, uses maximum value in
        middle slice of the volume.
    """
    plot_rgb = 0
    if volume.shape[-1] == 3:
        plot_rgb = 1
    if volume.ndim - plot_rgb == 2:
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = None
        ax.imshow(volume, vmin=vmin, vmax=vmax)
    elif volume.ndim - plot_rgb == 3:
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index], vmin=vmin, vmax=vmax)
    elif volume.ndim - plot_rgb == 4:
        fig, ax = plt.subplots(int(np.ceil(np.sqrt(volume.shape[0]))), int(
            np.ceil(np.sqrt(volume.shape[0]))))
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        ax = ax.flatten()
        ni = int(np.ceil(np.sqrt(volume.shape[0])))
        nj = int(np.ceil(np.sqrt(volume.shape[0])))
        for j in range(nj):
            for i in range(ni):
                ax[i + ni * j].axis('off')
                if i + ni * j >= volume.shape[0]:
                    ax[i + ni * j].volume = np.zeros_like(volume[0])
                else:
                    ax[i + ni * j].volume = volume[i + (j * ni)]
                    ax[i + ni * j].index = volume[i + (j * ni)].shape[0] // 2
                    ax[i + ni * j].imshow(volume[i + (j * ni),
                                                 ax[i + ni * j].index],
                                          vmin=vmin, vmax=vmax)
    else:
        raise NameError('Unsupported Dimensions')
    fig.canvas.mpl_connect('scroll_event', _process_scroll)


def _process_scroll(event):
    fig = event.canvas.figure
    ax = fig.axes
    for i, axes in enumerate(ax):
        if axes.index is not None:
            volume = axes.volume
            if (int((axes.index - event.step) >= volume.shape[0]) or
                    int((axes.index - event.step) < 0)):
                pass
            else:
                ax[i].index = int((axes.index - event.step) % volume.shape[0])
                ax[i].images[0].set_array(volume[ax[i].index])
                fig.canvas.draw()
