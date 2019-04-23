#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:45:18 2018

@author: omaier
Copyright 2019 Oliver Maier

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import h5py
from tkinter import filedialog
from tkinter import Tk


def run():
    ##########################################################################
    # Select input file ######################################################
    ##########################################################################

    root = Tk()
    root.withdraw()
    root.update()
    file = filedialog.askopenfilename()
    root.destroy()

    file = h5py.File(file)

    res_name = []
    for name in file.items():
        res_name.append(name[0])

    data = file[res_name[2]][()]
    file.close()
    data = data[-3, :, :]

    [z, y, x] = data[0].shape
    M0 = np.abs(data[0])
    mask = np.ones_like(M0)
    mask[M0 < 0.25] = 0

    M0 = (np.abs(data[0])) * mask
    T1 = (np.abs(data[1])) * mask
    M0_min = M0.min()
    M0_max = M0.max()
    T1_min = T1.min()
    T1_max = T1.max()

    def update_img(num):
        if num >= x:
            num = x - num - 1
            for i in range(2):
                ax[1 +
                   2 *
                   i].images[0].set_array(ax[1 +
                                             2 *
                                             i].volume[int(np.floor(num /
                                                                    x *
                                                                    z))])
                ax[2 + 2 * i].images[0].set_array(ax[2 + 2 * i].volume[num])
                ax[7 + 2 * i].images[0].set_array(ax[7 + 2 * i].volume[num])
        else:
            for i in range(2):
                ax[1 + 2 *
                    i].images[0].set_array(
                   ax[1 + 2 * i].volume[int(num / x * z)])
                ax[2 + 2 * i].images[0].set_array(ax[2 + 2 * i].volume[num])
                ax[7 + 2 * i].images[0].set_array(ax[7 + 2 * i].volume[num])

    # Attaching 3D axis to the figure
    plt.ion()
    ax = []
    figure = plt.figure(figsize=(12, 6))
    figure.subplots_adjust(hspace=0, wspace=0)
    gs = gridspec.GridSpec(2, 6, width_ratios=[
                           x / (20 * z), x / z, 1, x / z, 1, x / (20 * z)],
                           height_ratios=[x / z, 1])
    figure.tight_layout()
    figure.patch.set_facecolor(plt.cm.viridis.colors[0])
    for grid in gs:
        ax.append(plt.subplot(grid))
        ax[-1].axis('off')

    ax[1].volume = M0
    ax[2].volume = np.flip(np.transpose(M0, (2, 1, 0)), -1)
    ax[7].volume = np.transpose(M0, (1, 0, 2))
    M0_plot = ax[1].imshow((M0[int(z / 2), ...]))
    M0_plot_cor = ax[7].imshow((M0[:, int(M0.shape[1] / 2), ...]))
    M0_plot_sag = ax[2].imshow(
        np.flip((M0[:, :, int(M0.shape[-1] / 2 + 3)]).T, 1))
    ax[1].set_title('Proton Density in a.u.', color='white')
    ax[1].set_anchor('SE')
    ax[2].set_anchor('SW')
    ax[7].set_anchor('NW')
    cax = plt.subplot(gs[:, 0])
    cbar = figure.colorbar(M0_plot, cax=cax)
    cbar.ax.tick_params(labelsize=12, colors='white')
    cax.yaxis.set_ticks_position('left')
    for spine in cbar.ax.spines:
        cbar.ax.spines[spine].set_color('white')
    M0_plot.set_clim([M0_min, M0_max])
    M0_plot_cor.set_clim([M0_min, M0_max])
    M0_plot_sag.set_clim([M0_min, M0_max])

    ax[3].volume = T1
    ax[4].volume = np.flip(np.transpose(T1, (2, 1, 0)), -1)
    ax[9].volume = np.transpose(T1, (1, 0, 2))
    T1_plot = ax[3].imshow((T1[int(z / 2), ...]))
    T1_plot_cor = ax[9].imshow((T1[:, int(T1.shape[1] / 2), ...]))
    T1_plot_sag = ax[4].imshow(
        np.flip((T1[:, :, int(T1.shape[-1] / 2 + 3)]).T, 1))
    ax[3].set_title('T1 in  ms', color='white')
    ax[3].set_anchor('SE')
    ax[4].set_anchor('SW')
    ax[9].set_anchor('NW')
    cax = plt.subplot(gs[:, 5])
    cbar = figure.colorbar(T1_plot, cax=cax)
    cbar.ax.tick_params(labelsize=12, colors='white')
    for spine in cbar.ax.spines:
        cbar.ax.spines[spine].set_color('white')
    plt.draw()
    plt.pause(1e-10)
    T1_plot.set_clim([T1_min, T1_max])
    T1_plot_sag.set_clim([T1_min, T1_max])
    T1_plot_cor.set_clim([T1_min, T1_max])
    ax = np.array(ax)

    line_ani = animation.FuncAnimation(
        figure, update_img, x, interval=100, blit=False)
    print("Press any key to continue....")
    plt.waitforbuttonpress()
    line_ani.save(
        "3D_reco.gif",
        writer="imagemagick",
        dpi=70,
        fps=20,
        savefig_kwargs={
            'facecolor': plt.cm.viridis.colors[0]})


if __name__ == '__main__':
    run()
