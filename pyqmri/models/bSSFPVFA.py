#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyqmri.models.template import BaseModel, constraints, DTYPE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

unknowns_TGV = 4
unknowns_H1 = 0


class Model(BaseModel):
    def __init__(self, par, images):
        super().__init__(par)
        if len(par['TR']) > 1:
            self.TR = par['TR'][0]
        else:
            self.TR = par['TR']

        NScan_VFA = int(par["NScan_VFA"])
        self.fa_fl = par['flip_angle(s)'][:NScan_VFA] * np.pi / 180
        self.fa_bl = par['flip_angle(s)'][NScan_VFA:] * np.pi / 180
        self.fa_corr = par['fa_corr']

        phi_corr = np.zeros_like(images, dtype=DTYPE)
        for i in range(np.size(self.fa_fl)):
            phi_corr[i, :, :, :] = self.fa_fl[i] * self.fa_corr

        for i in range(
            np.size(
                self.fa_fl),
            np.size(
                self.fa_bl) +
            np.size(
                self.fa_fl)):
            phi_corr[i, :, :, :] = self.fa_bl[
                i - np.size(self.fa_fl)] * self.fa_corr

        self.sin_phi_fl = np.sin(phi_corr[:NScan_VFA])
        self.cos_phi_fl = np.cos(phi_corr[:NScan_VFA])

        self.sin_phi_bl = np.sin(phi_corr[NScan_VFA:])
        self.cos_phi_bl = np.cos(phi_corr[NScan_VFA:])

        self.uk_scale = []
        for i in range(unknowns_TGV + unknowns_H1):
            self.uk_scale.append(1)
#        self.uk_scale[0] = 1 / np.median(np.abs(images[:NScan_VFA]))

        self.guess = self._set_init_scales()
        self.constraints.append(
            constraints(
                0 / self.uk_scale[0],
                1000 / self.uk_scale[0],
                False))
        self.constraints.append(constraints(
            np.exp(-self.TR / (1)) / self.uk_scale[1],
            np.exp(-self.TR / (5500)) / self.uk_scale[1],
            True))
        self.constraints.append(constraints(
            np.exp(-self.TR / (1)) / self.uk_scale[2],
            np.exp(-self.TR / (2500)) / self.uk_scale[2],
            True))
        self.constraints.append(
            constraints(
                0 / self.uk_scale[3],
                1 / self.uk_scale[3],
                True))

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
        M02 = x[3, ...]
        M0_sc = self.uk_scale[0]
        T1_sc = self.uk_scale[1]
        T2_sc = self.uk_scale[2]
        M02_sc = self.uk_scale[3]
        S1 = M0 * M0_sc * (-E1 * T1_sc + 1) * self.sin_phi_fl / \
            (-E1 * T1_sc * self.cos_phi_fl + 1)
        S2 = M0 * M0_sc / (M02 * M02_sc) * (-E1 * T1_sc + 1) * self.sin_phi_bl / (-E1 * E2 * \
                             T1_sc * T2_sc - (E1 * T1_sc - E2 * T2_sc) * self.cos_phi_bl + 1)
        S = (np.concatenate((S1, S2)))
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        M0 = x[0, ...]
        E1 = x[1, :, :]  # self.E1#
        E2 = x[2, ...]  # self.E2#
        M02 = x[3, ...]
        M0_sc = self.uk_scale[0]
        T1_sc = self.uk_scale[1]
        T2_sc = self.uk_scale[2]
        M02_sc = self.uk_scale[3]

        grad_M0_fl = (M0_sc * (-E1 * T1_sc + 1) *
                      self.sin_phi_fl / (-E1 * T1_sc * self.cos_phi_fl + 1))
        grad_T1_fl = (M0 * self.uk_scale[0] * self.uk_scale[1] * (-E1 * T1_sc + 1) * self.sin_phi_fl * self.cos_phi_fl / (-E1 * T1_sc * \
                      self.cos_phi_fl + 1)**2 - M0 * self.uk_scale[0] * self.uk_scale[1] * self.sin_phi_fl / (-E1 * T1_sc * self.cos_phi_fl + 1))
        grad_T2_fl = np.zeros_like(grad_T1_fl)
        grad_M02_fl = np.zeros_like(
            M0 * M0_sc * (-E1 * T1_sc + 1) * self.sin_phi_fl / (-E1 * T1_sc * self.cos_phi_fl + 1))

        grad_M0_bl = (M0_sc / (M02 * M02_sc) * (-E1 * T1_sc + 1) * self.sin_phi_bl / \
                                   (-E1 * E2 * T1_sc * T2_sc - (E1 * T1_sc - E2 * T2_sc) * self.cos_phi_bl + 1))
        grad_T1_bl = (-M0 * M0_sc / (M02 * M02_sc)  * T1_sc * self.sin_phi_bl / (-E1 * E2 * T1_sc * T2_sc - (E1 * T1_sc - E2 * T2_sc) * self.cos_phi_bl + 1) + M0 * M0_sc / (M02 * M02_sc)  * (-E1 * T1_sc + 1)
                      * (E2 * T1_sc * T2_sc + T1_sc * self.cos_phi_bl) * self.sin_phi_bl / (-E1 * E2 * T1_sc * T2_sc - (E1 * T1_sc - E2 * T2_sc) * self.cos_phi_bl + 1)**2)
        grad_T2_bl = (M0 * M0_sc / (M02 * M02_sc)  * (-E1 * T1_sc + 1) * (E1 * T1_sc * T2_sc - T2_sc * self.cos_phi_bl) *
                      self.sin_phi_bl / (-E1 * E2 * T1_sc * T2_sc - (E1 * T1_sc - E2 * T2_sc) * self.cos_phi_bl + 1)**2)

        grad_M02_bl = - M02_sc * (M0 * M0_sc / (M02 * M02_sc)**2 * (-E1 * T1_sc + 1) * self.sin_phi_bl / (-E1 * E2 * \
                             T1_sc * T2_sc - (E1 * T1_sc - E2 * T2_sc) * self.cos_phi_bl + 1))

        grad_M0 = np.concatenate((grad_M0_fl, grad_M0_bl))
        grad_T1 = np.concatenate((grad_T1_fl, grad_T1_bl))
        grad_T2 = np.concatenate((grad_T2_fl, grad_T2_bl))
        grad_M02 = np.concatenate((grad_M02_fl, np.mean(grad_M02_bl)*np.ones_like(grad_M02_bl)))
        grad = np.array([grad_M0, grad_T1, grad_T2, grad_M02], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...] * self.uk_scale[0])
        T1 = np.abs(-self.TR / np.log(x[1, ...] * self.uk_scale[1]))
        T2 = np.abs(-self.TR / np.log(x[2, ...] * self.uk_scale[2]))
        k = np.abs(x[3, ...]) * self.uk_scale[3]
        M0_min = M0.min()
        M0_max = M0.max()
        T1_min = T1.min()
        T1_max = T1.max()
        T2_min = T2.min()
        T2_max = T2.max()
        k_min = k.min()
        k_max = k.max()
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
                                            14,
                                            width_ratios=[x / (20 * z),
                                                          x / z,
                                                          1,
                                                          x / z,
                                                          1,
                                                          x / (20 * z),
                                                          4 * x / (20 * z),
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
                self.M0_plot_cor = self.ax[15].imshow(
                    (M0[:, int(M0.shape[1] / 2), ...]))
                self.M0_plot_sag = self.ax[2].imshow(
                    np.flip((M0[:, :, int(M0.shape[-1] / 2)]).T, 1))
                self.ax[1].set_title('Proton Density in a.u.', color='white')
                self.ax[1].set_anchor('SE')
                self.ax[2].set_anchor('SW')
                self.ax[15].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 0])
                cbar = self.figure.colorbar(self.M0_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                self.T1_plot = self.ax[3].imshow(
                    (T1[int(self.NSlice / 2), ...]))
                self.T1_plot_cor = self.ax[17].imshow(
                    (T1[:, int(T1.shape[1] / 2), ...]))
                self.T1_plot_sag = self.ax[4].imshow(
                    np.flip((T1[:, :, int(T1.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('T1 in  ms', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[17].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 5])
                cbar = self.figure.colorbar(self.T1_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.T2_plot = self.ax[7].imshow(
                    (T2[int(self.NSlice / 2), ...]))
                self.T2_plot_cor = self.ax[21].imshow(
                    (T2[:, int(T2.shape[1] / 2), ...]))
                self.T2_plot_sag = self.ax[8].imshow(
                    np.flip((T2[:, :, int(T2.shape[-1] / 2)]).T, 1))
                self.ax[7].set_title('T2 in  ms', color='white')
                self.ax[7].set_anchor('SE')
                self.ax[8].set_anchor('SW')
                self.ax[21].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 9])
                cbar = self.figure.colorbar(self.T2_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.k = self.ax[11].imshow((k[int(self.NSlice / 2), ...]))
                self.k_cor = self.ax[25].imshow(
                    (k[:, int(T2.shape[1] / 2), ...]))
                self.k_sag = self.ax[12].imshow(
                    np.flip((k[:, :, int(T2.shape[-1] / 2)]).T, 1))
                self.ax[11].set_title('k in  a.u.', color='white')
                self.ax[11].set_anchor('SE')
                self.ax[12].set_anchor('SW')
                self.ax[25].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 13])
                cbar = self.figure.colorbar(self.k, cax=cax)
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

                self.k.set_data((k[int(self.NSlice / 2), ...]))
                self.k_cor.set_data((k[:, int(k.shape[1] / 2), ...]))
                self.k_sag.set_data(
                    np.flip((k[:, :, int(k.shape[-1] / 2)]).T, 1))
                self.k.set_clim([k_min, k_max])
                self.k_sag.set_clim([k_min, k_max])
                self.k_cor.set_clim([k_min, k_max])

                plt.draw()
                plt.pause(1e-10)

    def _set_init_scales(self):

        test_T1 = 800 * np.ones((self.NSlice, self.dimY,
                                 self.dimX), dtype=DTYPE)
        test_T2 = 200 * np.ones((self.NSlice, self.dimY,
                                 self.dimX), dtype=DTYPE)
        test_M0 = np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_T1 = 1 / self.uk_scale[1] * np.exp(-self.TR / (
            test_T1 * np.ones((self.NSlice, self.dimY, self.dimX),
                              dtype=DTYPE)))
        test_T2 = 1 / self.uk_scale[2] * np.exp(-self.TR / (
            test_T2 * np.ones((self.NSlice, self.dimY, self.dimX),
                              dtype=DTYPE)))
        test_M02 = 1 / \
            self.uk_scale[3] * np.ones((self.NSlice, self.dimY, self.dimX),
                                       dtype=DTYPE)

        x = np.array([test_M0, test_T1, test_T2, test_M02], dtype=DTYPE)

        return x
