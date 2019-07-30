#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:42:42 2017

@author: omaier
"""

from Models.Model import BaseModel, constraints, DTYPE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


unknowns_TGV = 2
unknowns_H1 = 0


class Model(BaseModel):
    def __init__(self, par, images):
        super().__init__(par)
        self.TR = par['TR']
        self.fa_bl = par['fa_bl']
        self.fa_corr = par['fa_corr']

        phi_corr = np.zeros_like(images, dtype=DTYPE)

        for i in range(np.size(self.fa_bl)):
            phi_corr[i, :, :, :] = self.fa_bl[i] * self.fa_corr

        self.sin_phi_bl = np.sin(phi_corr)
        self.cos_phi_bl = np.cos(phi_corr)

        self.uk_scale.append(1 / np.median(np.abs(images)))
        for j in range(unknowns_TGV + unknowns_H1 - 1):
            self.uk_scale.append(1)

        self._set_init_scales()

        result = np.array([1 / self.uk_scale[0] * np.ones((self.NSlice,
                                                           self.dimY,
                                                           self.dimX),
                                                          dtype=DTYPE),
                           1 / self.uk_scale[1] * np.exp(-self.TR / (1600 * np.ones((self.NSlice,
                                                                                     self.dimY,
                                                                                     self.dimX),
                                                                                    dtype=DTYPE))),
                           1 / self.uk_scale[2] * np.exp(-self.TR / (80 * np.ones((self.NSlice,
                                                                                   self.dimY,
                                                                                   self.dimX),
                                                                                  dtype=DTYPE)))],
                          dtype=DTYPE)
        self.guess = result
        self.constraints.append(
            constraints(-10 / self.uk_scale[0], 10 / self.uk_scale[0], False))
        self.constraints.append(constraints(
            np.exp(-self.TR / (10)) / self.uk_scale[1], np.exp(-self.TR / (5500)) / self.uk_scale[1], True))
        self.constraints.append(constraints(
            np.exp(-self.TR / (0)) / self.uk_scale[2], np.exp(-self.TR / (2500)) / self.uk_scale[2], True))

    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_forward_3D(self, x):
        M0 = x[0, ...]
        E1 = x[1, ...]
        E2 = x[2, ...]
        M0_sc = self.uk_scale[0]
        T1_sc = self.uk_scale[1]
        T2_sc = self.uk_scale[2]
        S = M0 * M0_sc * (-E1 * T1_sc + 1) * self.sin_phi_bl / (-E1 * E2 * \
                          T1_sc * T2_sc - (E1 * T1_sc - E2 * T2_sc) * self.cos_phi_bl + 1)
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        M0 = x[0, ...]
        E1 = x[1, :, :]
        E2 = x[2, ...]
        M0_sc = self.uk_scale[0]
        T1_sc = self.uk_scale[1]
        T2_sc = self.uk_scale[2]

        grad_M0 = (M0_sc * (-E1 * T1_sc + 1) * self.sin_phi_bl / (-E1 * E2 * \
                   T1_sc * T2_sc - (E1 * T1_sc - E2 * T2_sc) * self.cos_phi_bl + 1))
        grad_T1 = (-M0 * M0_sc * T1_sc * self.sin_phi_bl / (-E1 * E2 * T1_sc * T2_sc - (E1 * T1_sc - E2 * T2_sc) * self.cos_phi_bl + 1) + M0 * M0_sc * (-E1 * T1_sc + 1)
                   * (E2 * T1_sc * T2_sc + T1_sc * self.cos_phi_bl) * self.sin_phi_bl / (-E1 * E2 * T1_sc * T2_sc - (E1 * T1_sc - E2 * T2_sc) * self.cos_phi_bl + 1)**2)
        grad_T2 = M0 * M0_sc * (-E1 * T1_sc + 1) * (E1 * T1_sc * T2_sc - T2_sc * self.cos_phi_bl) * \
            self.sin_phi_bl / (-E1 * E2 * T1_sc * T2_sc - (E1 * T1_sc - E2 * T2_sc) * self.cos_phi_bl + 1)**2

        grad = np.array([grad_M0, grad_T1, grad_T2], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
#    print('Grad Scaling E1', np.linalg.norm(np.abs(grad[0]))/np.linalg.norm(np.abs(grad[1])))
#    print('Grad Scaling E2', np.linalg.norm(np.abs(grad[0]))/np.linalg.norm(np.abs(grad[2])))
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...] * self.uk_scale[0])
        T1 = np.abs(-self.TR / np.log(x[1, ...] * self.uk_scale[1]))
        T2 = np.abs(-self.TR / np.log(x[2, ...] * self.uk_scale[2]))
        M0_min = M0.min()
        M0_max = M0.max()
        T1_min = T1.min()
        T1_max = T1.max()
        T2_min = T2.min()
        T2_max = T2.max()
        if dim_2D:
            if not self.figure:
                plt.ion()
                self.figure, self.ax = plt.subplots(1, 2, figsize=(12, 5))
                self.M0_plot = self.ax[0].imshow((M0))
                self.ax[0].set_title('Proton Density in a.u.')
                self.ax[0].axis('off')
                self.figure.colorbar(self.M0_plot, ax=self.ax[0])
                self.T1_plot = self.ax[1].imshow((T1))
                self.ax[1].set_title('T1 in  ms')
                self.ax[1].axis('off')
                self.figure.colorbar(self.T1_plot, ax=self.ax[1])
                self.figure.tight_layout()
                plt.draw()
                plt.pause(1e-10)
            else:
                self.M0_plot.set_data((M0))
                self.M0_plot.set_clim([M0_min, M0_max])
                self.T1_plot.set_data((T1))
                self.T1_plot.set_clim([T1_min, T1_max])
                plt.draw()
                plt.pause(1e-10)
        else:
            [z, y, x] = M0.shape
            self.ax = []
            if not self.figure:
                plt.ion()
                self.figure = plt.figure(figsize=(12, 6))
                self.figure.subplots_adjust(hspace=0, wspace=0)
                self.gs = gridspec.GridSpec(2,
                                            10,
                                            width_ratios=[x / (20 * z),
                                                          x / z,
                                                          1,
                                                          x / z,
                                                          1,
                                                          x / (20 * z),
                                                          4 * x / (20 * z),
                                                          x / z,
                                                          1,
                                                          x / (20 * z)],
                                            height_ratios=[x / z,
                                                           1])
                self.figure.tight_layout()
                self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
                for grid in self.gs:
                    self.ax.append(plt.subplot(grid))
                    self.ax[-1].axis('off')

                self.M0_plot = self.ax[1].imshow(
                    (M0[int(self.NSlice / 2), ...]))
                self.M0_plot_cor = self.ax[11].imshow(
                    (M0[:, int(M0.shape[1] / 2), ...]))
                self.M0_plot_sag = self.ax[2].imshow(
                    np.flip((M0[:, :, int(M0.shape[-1] / 2)]).T, 1))
                self.ax[1].set_title('Proton Density in a.u.', color='white')
                self.ax[1].set_anchor('SE')
                self.ax[2].set_anchor('SW')
                self.ax[11].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 0])
                cbar = self.figure.colorbar(self.M0_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                self.T1_plot = self.ax[3].imshow(
                    (T1[int(self.NSlice / 2), ...]))
                self.T1_plot_cor = self.ax[13].imshow(
                    (T1[:, int(T1.shape[1] / 2), ...]))
                self.T1_plot_sag = self.ax[4].imshow(
                    np.flip((T1[:, :, int(T1.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('T1 in  ms', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[13].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 5])
                cbar = self.figure.colorbar(self.T1_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.T2_plot = self.ax[7].imshow(
                    (T2[int(self.NSlice / 2), ...]))
                self.T2_plot_cor = self.ax[17].imshow(
                    (T2[:, int(T2.shape[1] / 2), ...]))
                self.T2_plot_sag = self.ax[8].imshow(
                    np.flip((T2[:, :, int(T2.shape[-1] / 2)]).T, 1))
                self.ax[7].set_title('T2 in  ms', color='white')
                self.ax[7].set_anchor('SE')
                self.ax[8].set_anchor('SW')
                self.ax[17].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 9])
                cbar = self.figure.colorbar(self.T2_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)
            else:
                self.M0_plot.set_data((M0[int(self.NSlice / 2), ...]))
                self.M0_plot_cor.set_data((M0[:, int(M0.shape[1] / 2), ...]))
                self.M0_plot_sag.set_data(
                    np.flip((M0[:, :, int(M0.shape[-1] / 2)]).T, 1))
                self.M0_plot.set_clim([M0_min, M0_max])
                self.M0_plot_cor.set_clim([M0_min, M0_max])
                self.M0_plot_sag.set_clim([M0_min, M0_max])
                self.T1_plot.set_data((T1[int(self.NSlice / 2), ...]))
                self.T1_plot_cor.set_data((T1[:, int(T1.shape[1] / 2), ...]))
                self.T1_plot_sag.set_data(
                    np.flip((T1[:, :, int(T1.shape[-1] / 2)]).T, 1))
                self.T1_plot.set_clim([T1_min, T1_max])
                self.T1_plot_sag.set_clim([T1_min, T1_max])
                self.T1_plot_cor.set_clim([T1_min, T1_max])
                self.T2_plot.set_data((T2[int(self.NSlice / 2), ...]))
                self.T2_plot_cor.set_data((T2[:, int(T2.shape[1] / 2), ...]))
                self.T2_plot_sag.set_data(
                    np.flip((T2[:, :, int(T2.shape[-1] / 2)]).T, 1))
                self.T2_plot.set_clim([T2_min, T2_max])
                self.T2_plot_sag.set_clim([T2_min, T2_max])
                self.T2_plot_cor.set_clim([T2_min, T2_max])
                plt.draw()
                plt.pause(1e-10)

    def _set_init_scales(self):
        test_T1 = np.reshape(
            np.linspace(
                10,
                5500,
                self.dimX *
                self.dimY *
                self.NSlice),
            (self.NSlice,
             self.dimX,
             self.dimY))
        test_T2 = np.reshape(
            np.linspace(
                10,
                150,
                self.dimX *
                self.dimY *
                self.NSlice),
            (self.NSlice,
             self.dimX,
             self.dimY))
        test_M0 = np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_T1 = 1 / self.uk_scale[1] * np.exp(-self.TR / (test_T1))
        test_T2 = 1 / self.uk_scale[2] * np.exp(-self.TR / (test_T2))

        G_x = self._execute_forward_3D(
            np.array([test_M0 / self.uk_scale[0], test_T1, test_T2], dtype=DTYPE))
        self.uk_scale[0] *= 1 / np.median(np.abs(G_x))

        DG_x = self._execute_gradient_3D(
            np.array(
                [
                    test_M0 /
                    self.uk_scale[0] *
                    np.ones(
                        (self.NSlice,
                         self.dimY,
                         self.dimX),
                        dtype=DTYPE),
                    test_T1,
                    test_T2],
                dtype=DTYPE))

        for j in range(unknowns_TGV + unknowns_H1 - 1):
            self.uk_scale[j + 1] *= np.linalg.norm(
                np.abs(DG_x[0, ...])) / np.linalg.norm(np.abs(DG_x[j + 1, ...]))

        print('T1 scale: ', self.uk_scale[1], '/ T2_scale: ', self.uk_scale[2],
              '/ M0_scale: ', self.uk_scale[0])
        return (test_M0, test_T1, test_T2)
