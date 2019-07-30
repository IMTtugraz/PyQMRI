#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:42:42 2017

@author: omaier
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyqmri.models.template import BaseModel, constraints, DTYPE
plt.ion()
unknowns_TGV = 2
unknowns_H1 = 0


class Model(BaseModel):
    def __init__(self, par, images):
        super().__init__(par)
        self.constraints = []
        self.T1b = par["file"]["T1b"][()]
        self.T1 = par["file"]["T1"][()]
        self.lambd = par["file"]["lambd"][()]
        self.M0 = par["file"]["M0"][()]
        self.tau = par["file"]["tau"][()]
        self.t = par['t']
        self.alpha = par["file"]["alpha"][()]

        self.NScan = par["NScan"]
        self.NSlice = par["NSlice"]
        self.dimY = par["dimY"]
        self.dimX = par["dimX"]
        self.unknowns = unknowns_TGV+unknowns_H1
        self.images = images
        self.dscale = par["dscale"]

        for j in range(unknowns_TGV + unknowns_H1):
            self.uk_scale.append(1)
        self.guess = self._set_init_scales(images)
#        self.f = par["file"]["f"][()]*self.dscale
#        self.del_t = par["file"]["del_t"][()]
#        self.guess = np.array((self.f, self.del_t), dtype=DTYPE)
#        self.images = np.transpose(self.images, [0, 2, 3, 1])
        self.constraints.append(
            constraints(0,
                        200 * self.dscale,
                        False))
        self.constraints.append(
            constraints(0,
                        4/60,
                        True))

    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_forward_3D(self, x):
        f = x[0, ...] * self.uk_scale[0]
        del_t = x[1, ...] * self.uk_scale[1]

        S = np.zeros((self.NScan, self.NSlice, self.dimY, self.dimX),
                     dtype=DTYPE)
        for j in range((self.t).size):
            ind_low = self.t[j] >= del_t
            ind_high = self.t[j] < (del_t+self.tau[j])
            ind = ind_low & ind_high
            if np.any(ind):
                S[j, ind] = 2*self.alpha[ind]*self.M0[ind]/self.lambd[ind] *\
                            f[ind]/(1/self.T1[ind]+f[ind]/self.lambd[ind]) * \
                            np.exp(-(del_t[ind])/self.T1b[ind]) * \
                            (1-np.exp(-(self.t[j]-del_t[ind]) *
                             (1/self.T1[ind]+f[ind]/self.lambd[ind])))
            ind = self.t[j] >= del_t + self.tau[j]
            if np.any(ind):
                S[j, ind] = 2*self.alpha[ind]*self.M0[ind]/self.lambd[ind] *\
                            f[ind]/(1/self.T1[ind]+f[ind]/self.lambd[ind]) * \
                            np.exp(-(del_t[ind])/self.T1b[ind]) * \
                            np.exp(-(self.t[j]-del_t[ind]-self.tau[j, ind]) *
                                   (1/self.T1[ind]+f[ind]/self.lambd[ind])) * \
                            (1-np.exp(-self.tau[j, ind] *
                                      (1/self.T1[ind]+f[ind]/self.lambd[ind])))
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        f_sc = self.uk_scale[0]
        del_t = x[1, ...]
        del_t_sc = self.uk_scale[1]
        grad = np.zeros((self.unknowns, self.NScan,
                         self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        t = self.t
        for j in range((self.t).size):
            ind_low = self.t[j] >= x[1, ...]*self.uk_scale[1]
            ind_high = self.t[j] < (x[1, ...]*self.uk_scale[1]+self.tau[j])
            ind = ind_low & ind_high
            M0 = self.M0[ind]
            T1 = self.T1[ind]
            T1b = self.T1b[ind]
            lambd = self.lambd[ind]
            tau = self.tau[j, ind]
            alpha = self.alpha[ind]
            f = x[0, ind]
            del_t = x[1, ind]
            if np.any(ind):
                grad[0, j, ind] = (-2*M0*f*f_sc**2*(del_t*del_t_sc - t[j]) *
                                   np.exp((del_t*del_t_sc - t[j]) *
                                          (f*f_sc/lambd + 1/T1)) *
                                   np.exp(-del_t*del_t_sc/T1b) /
                                   (lambd**2*(f*f_sc/lambd + 1/T1)) -
                                   2*M0*f*f_sc**2 *
                                   (-np.exp((del_t*del_t_sc - t[j]) *
                                            (f*f_sc/lambd + 1/T1)) + 1) *
                                   np.exp(-del_t*del_t_sc/T1b) /
                                   (lambd**2*(f*f_sc/lambd + 1/T1)**2) +
                                   2*M0*f_sc*(-np.exp(
                                       (del_t*del_t_sc - t[j]) *
                                       (f*f_sc/lambd + 1/T1)) + 1) *
                                   np.exp(-del_t*del_t_sc/T1b) /
                                   (lambd*(f*f_sc/lambd + 1/T1)))*alpha
                grad[1, j, ind] = (-2*M0*del_t_sc*f*f_sc *
                                   np.exp((del_t*del_t_sc - t[j]) *
                                          (f*f_sc/lambd + 1/T1)) *
                                   np.exp(-del_t*del_t_sc/T1b)/lambd -
                                   2*M0*del_t_sc*f*f_sc*(
                                       - np.exp((del_t*del_t_sc - t[j]) *
                                                (f*f_sc/lambd + 1/T1)) + 1) *
                                   np.exp(-del_t*del_t_sc/T1b) /
                                   (T1b*lambd*(f*f_sc/lambd + 1/T1)))*alpha
            ind = self.t[j] >= x[1, ...]*self.uk_scale[1] + self.tau[j]
            M0 = self.M0[ind]
            T1 = self.T1[ind]
            T1b = self.T1b[ind]
            lambd = self.lambd[ind]
            tau = self.tau[j, ind]
            alpha = self.alpha[ind]
            f = x[0, ind]
            del_t = x[1, ind]
            if np.any(ind):
                grad[0, j, ind] = (2*M0*f*f_sc**2*tau *
                                   np.exp(-tau*(f*f_sc/lambd + 1/T1)) *
                                   np.exp((f*f_sc/lambd + 1/T1) *
                                          (del_t*del_t_sc - t[j] + tau)) *
                                   np.exp(-del_t*del_t_sc/T1b) /
                                   (lambd**2*(f*f_sc/lambd + 1/T1)) +
                                   2*M0*f*f_sc**2 *
                                   (1 - np.exp(-tau*(f*f_sc/lambd + 1/T1))) *
                                   (del_t*del_t_sc - t[j] + tau) *
                                   np.exp((f*f_sc/lambd + 1/T1) *
                                          (del_t*del_t_sc - t[j] + tau)) *
                                   np.exp(-del_t*del_t_sc/T1b) /
                                   (lambd**2*(f*f_sc/lambd + 1/T1)) -
                                   2*M0*f*f_sc**2 *
                                   (1 - np.exp(-tau*(f*f_sc/lambd + 1/T1))) *
                                   np.exp((f*f_sc/lambd + 1/T1) *
                                          (del_t*del_t_sc - t[j] + tau)) *
                                   np.exp(-del_t*del_t_sc/T1b) /
                                   (lambd**2*(f*f_sc/lambd + 1/T1)**2) +
                                   2*M0*f_sc *
                                   (1 - np.exp(-tau*(f*f_sc/lambd + 1/T1))) *
                                   np.exp((f*f_sc/lambd + 1/T1) *
                                          (del_t*del_t_sc - t[j] + tau)) *
                                   np.exp(-del_t*del_t_sc/T1b) /
                                   (lambd*(f*f_sc/lambd + 1/T1)))*alpha
                grad[1, j, ind] = (2*M0*del_t_sc*f*f_sc *
                                   (1 - np.exp(-tau*(f*f_sc/lambd + 1/T1))) *
                                   np.exp((f*f_sc/lambd + 1/T1) *
                                          (del_t*del_t_sc - t[j] + tau)) *
                                   np.exp(-del_t*del_t_sc/T1b)/lambd -
                                   2*M0*del_t_sc*f*f_sc *
                                   (1 - np.exp(-tau*(f*f_sc/lambd + 1/T1))) *
                                   np.exp((f*f_sc/lambd + 1/T1) *
                                          (del_t*del_t_sc - t[j] + tau)) *
                                   np.exp(-del_t*del_t_sc/T1b) /
                                   (T1b*lambd*(f*f_sc/lambd + 1/T1)))*alpha
        grad[~np.isfinite(grad)] = 1e-20
        grad = np.array(grad, dtype=DTYPE)
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        images = self._execute_forward_3D(x)
        f = np.abs(x[0, ...] * self.uk_scale[0]/self.dscale)
        del_t = np.abs(x[1, ...] * self.uk_scale[1])
        f_min = f.min()
        f_max = f.max()
        del_t_min = del_t.min()
        del_t_max = del_t.max()
        ind = 30  # int(images.shape[-1]/2)
        if dim_2D:
            if not self.figure:
                plt.ion()
                self.figure, self.ax = plt.subplots(1, 2, figsize=(12, 5))
                self.f_plot = self.ax[0].imshow((f))
                self.ax[0].set_title('Proton Density in a.u.')
                self.ax[0].axis('off')
                self.figure.colorbar(self.f_plot, ax=self.ax[0])
                self.del_t_plot = self.ax[1].imshow((del_t))
                self.ax[1].set_title('del_t in  ms')
                self.ax[1].axis('off')
                self.figure.colorbar(self.del_t_plot, ax=self.ax[1])
                self.figure.tight_layout()
                plt.draw()
                plt.pause(1e-10)
            else:
                self.f_plot.set_data((f))
                self.f_plot.set_clim([f_min, f_max])
                self.del_t_plot.set_data((del_t))
                self.del_t_plot.set_clim([del_t_min, del_t_max])
                plt.draw()
                plt.pause(1e-10)
        else:
            [z, y, x] = f.shape
            self.ax = []
            if not self.figure:
                plt.ion()
                self.figure = plt.figure(figsize=(12, 6))
                self.figure.subplots_adjust(hspace=0, wspace=0)
                self.gs = gridspec.GridSpec(
                    3, 6,
                    width_ratios=[
                        x / (20 * z), x / z, 1, x / z, 1, x / (20 * z)],
                    height_ratios=[x / z, 1, 1])
                self.figure.tight_layout()
                self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
                for grid in self.gs:
                    self.ax.append(plt.subplot(grid))
                    self.ax[-1].axis('off')

                self.f_plot = self.ax[1].imshow(
                    (f[int(self.NSlice / 2), ...]))
                self.f_plot_cor = self.ax[7].imshow(
                    (f[:, int(f.shape[1] / 2), ...]))
                self.f_plot_sag = self.ax[2].imshow(
                    np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
                self.ax[1].set_title('CBF', color='white')
                self.ax[1].set_anchor('SE')
                self.ax[2].set_anchor('SW')
                self.ax[7].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 0])
                cbar = self.figure.colorbar(self.f_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)

                self.del_t_plot = self.ax[3].imshow(
                    (del_t[int(self.NSlice / 2), ...]))
                self.del_t_plot_cor = self.ax[9].imshow(
                    (del_t[:, int(del_t.shape[1] / 2), ...]))
                self.del_t_plot_sag = self.ax[4].imshow(
                    np.flip((del_t[:, :, int(del_t.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('ATT', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[9].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 5])
                cbar = self.figure.colorbar(self.del_t_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)
                self.plot_ax = plt.subplot(self.gs[-1, :])

                self.time_course_ref = self.plot_ax.plot(
                    self.t, np.abs(
                        self.images[:, int(self.NSlice/2), ind, ind]), 'g')[0]
                self.time_course = self.plot_ax.plot(
                    self.t, np.abs(
                        images[:, int(self.NSlice/2), ind, ind]), 'r')[0]
                self.plot_ax.set_ylim(
                    np.abs(self.images[:,
                                       int(self.NSlice/2),
                                       ind,
                                       ind]).min() - np.abs(
                            self.images[:,
                                        int(self.NSlice/2),
                                        ind,
                                        ind]).min() * 0.01,
                    np.abs(self.images[:,
                                       int(self.NSlice/2),
                                       ind,
                                       ind]).max() + np.abs(
                           self.images[:,
                                       int(self.NSlice/2),
                                       ind,
                                       ind]).max() * 0.01)
                for spine in self.plot_ax.spines:
                    self.plot_ax.spines[spine].set_color('white')
                plt.draw()
                plt.show()
                plt.pause(1e-4)
            else:
                self.f_plot.set_data((f[int(self.NSlice / 2), ...]))
                self.f_plot_cor.set_data((f[:, int(f.shape[1] / 2), ...]))
                self.f_plot_sag.set_data(
                    np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
                self.f_plot.set_clim([f_min, f_max])
                self.f_plot_cor.set_clim([f_min, f_max])
                self.f_plot_sag.set_clim([f_min, f_max])
                self.del_t_plot.set_data((del_t[int(self.NSlice / 2), ...]))
                self.del_t_plot_cor.set_data(
                    (del_t[:, int(del_t.shape[1] / 2), ...]))
                self.del_t_plot_sag.set_data(
                    np.flip((del_t[:, :, int(del_t.shape[-1] / 2)]).T, 1))
                self.del_t_plot.set_clim([del_t_min, del_t_max])
                self.del_t_plot_sag.set_clim([del_t_min, del_t_max])
                self.del_t_plot_cor.set_clim([del_t_min, del_t_max])

                self.time_course.set_ydata(
                    np.abs(images[:, int(self.NSlice/2), ind, ind]))
                self.plot_ax.set_ylim(
                    np.abs(self.images[:,
                                       int(self.NSlice/2),
                                       ind,
                                       ind]).min() - np.abs(
                            self.images[:,
                                        int(self.NSlice/2),
                                        ind,
                                        ind]).min() * 0.01,
                    np.abs(self.images[:,
                                       int(self.NSlice/2),
                                       ind,
                                       ind]).max() + np.abs(
                           self.images[:,
                                       int(self.NSlice/2),
                                       ind,
                                       ind]).max() * 0.01)

                plt.draw()
                plt.pause(1e-10)

    def _set_init_scales(self, images):
        test_f = 70 * self.dscale * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_del_t = 1.5/60 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        x = np.array([test_f,
                      test_del_t], dtype=DTYPE)
        return x
