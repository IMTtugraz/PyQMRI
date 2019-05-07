#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:55:39 2017

@author: omaier
"""
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# 3D viewer  ##########s#######################################################
###############################################################################


def imshow(volume, vmin=None, vmax=None):
    """
    Volumetric image viewer for python. Shows up to 4D volumes.
    volume is assumed to be real valued.
    """
    if volume.ndim == 2:
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = None
        ax.imshow(volume, vmin=vmin, vmax=vmax)
    elif volume.ndim == 3:
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index], vmin=vmin, vmax=vmax)
    elif volume.ndim == 4:
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
    fig.canvas.mpl_connect('scroll_event', process_scroll)


def process_scroll(event):
    fig = event.canvas.figure
    ax = fig.axes
    for i in range(len(ax)):
        if ax[i].index is not None:
            volume = ax[i].volume
            if (int((ax[i].index - event.step) >= volume.shape[0]) or
                    int((ax[i].index - event.step) < 0)):
                pass
            else:
                ax[i].index = int((ax[i].index - event.step) % volume.shape[0])
                ax[i].images[0].set_array(volume[ax[i].index])
                fig.canvas.draw()
