#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyqmri.models.template import BaseModel, constraints, DTYPE
plt.ion()



class Model(BaseModel):
    def __init__(self, par):
        super().__init__(par)
        self.constraints = []
        self.TR = par["TR"]
        self.fa = par["flip_angle(s)"]
        try:
            self.fa_corr = par["fa_corr"]
        except KeyError:
            self.fa_corr = 1
            print("No flipangle correction found!")

        phi_corr = np.zeros(
          (self.NScan, self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        for i in range(np.size(par["flip_angle(s)"])):
            phi_corr[i, :, :, :] = par["flip_angle(s)"][i] *\
                np.pi / 180 * self.fa_corr

        par["unknowns_TGV"] = 2
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]

        self.sin_phi = np.sin(phi_corr)
        self.cos_phi = np.cos(phi_corr)

        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        self.constraints.append(
            constraints(0 / self.uk_scale[0],
                        10 / self.uk_scale[0],
                        False))
        self.constraints.append(
            constraints(np.exp(-self.TR / (50)),
                        np.exp(-self.TR / (5500)),
                        True))

    def rescale(self, x):
        tmp_x = np.copy(x)
        tmp_x[0] *= self.uk_scale[0]
        tmp_x[1] = -self.TR / np.log(tmp_x[1] * self.uk_scale[1])
        return tmp_x

    def _execute_forward_2D(self, x, islice):
        print('uk_scale[1]: ', self.uk_scale[1])
        E1 = x[1, ...] * self.uk_scale[1]
        S = x[0, ...] * self.uk_scale[0] * (-E1 + 1) * \
            self.sin_phi[:, islice, ...]\
            / (-E1 * self.cos_phi[:, islice, ...] + 1)
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_2D(self, x, islice):
        E1 = x[1, ...] * self.uk_scale[1]
        M0 = x[0, ...]
        E1[~np.isfinite(E1)] = 0
        grad_M0 = self.uk_scale[0] * (-E1 + 1) * self.sin_phi[:, islice, ...]\
            / (-E1 * self.cos_phi[:, islice, ...] + 1)
        grad_T1 = M0 * self.uk_scale[0] * self.uk_scale[1] * (-E1 + 1) *\
            self.sin_phi[:, islice, ...] * self.cos_phi[:, islice, ...] /\
            (-E1 * self.cos_phi[:, islice, ...] + 1)**2 -\
            M0 * self.uk_scale[0] * self.uk_scale[1] *\
            self.sin_phi[:, islice, ...] /\
            (-E1 * self.cos_phi[:, islice, ...] + 1)
        grad = np.array([grad_M0, grad_T1], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def _execute_forward_3D(self, x):
        E1 = x[1, ...] * self.uk_scale[1]
        S = x[0, ...] * self.uk_scale[0] * (-E1 + 1) * self.sin_phi /\
            (-E1 * self.cos_phi + 1)
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        E1 = x[1, ...] * self.uk_scale[1]
        M0 = x[0, ...]
        E1[~np.isfinite(E1)] = 0
        grad_M0 = self.uk_scale[0] * (-E1 + 1) * self.sin_phi /\
            (-E1 * self.cos_phi + 1)
        grad_T1 = M0 * self.uk_scale[0] * self.uk_scale[1] * (-E1 + 1) *\
            self.sin_phi * self.cos_phi / (-E1 * self.cos_phi + 1)**2 -\
            M0 * self.uk_scale[0] * self.uk_scale[1] * self.sin_phi /\
            (-E1 * self.cos_phi + 1)
        grad = np.array([grad_M0, grad_T1], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...] * self.uk_scale[0])
        T1 = np.abs(-self.TR / np.log(x[1, ...] * self.uk_scale[1]))
        M0_min = M0.min()
        M0_max = M0.max()
        T1_min = T1.min()
        T1_max = T1.max()

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
                self.gs = gridspec.GridSpec(
                    2, 6,
                    width_ratios=[
                        x / (20 * z), x / z, 1, x / z, 1, x / (20 * z)],
                    height_ratios=[x / z, 1])
                self.figure.tight_layout()
                self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
                for grid in self.gs:
                    self.ax.append(plt.subplot(grid))
                    self.ax[-1].axis('off')

                self.M0_plot = self.ax[1].imshow(
                    (M0[int(self.NSlice / 2), ...]))
                self.M0_plot_cor = self.ax[7].imshow(
                    (M0[:, int(M0.shape[1] / 2), ...]))
                self.M0_plot_sag = self.ax[2].imshow(
                    np.flip((M0[:, :, int(M0.shape[-1] / 2)]).T, 1))
                self.ax[1].set_title('Proton Density in a.u.', color='white')
                self.ax[1].set_anchor('SE')
                self.ax[2].set_anchor('SW')
                self.ax[7].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 0])
                cbar = self.figure.colorbar(self.M0_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)

                self.T1_plot = self.ax[3].imshow(
                    (T1[int(self.NSlice / 2), ...]))
                self.T1_plot_cor = self.ax[9].imshow(
                    (T1[:, int(T1.shape[1] / 2), ...]))
                self.T1_plot_sag = self.ax[4].imshow(
                    np.flip((T1[:, :, int(T1.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('T1 in  ms', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[9].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 5])
                cbar = self.figure.colorbar(self.T1_plot, cax=cax)
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
                plt.draw()
                plt.pause(1e-10)

    def computeInitialGuess(self, *args):
        test_T1 = 1500 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_M0 = np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_T1 = np.exp(-self.TR / (test_T1))
        x = np.array([test_M0 / self.uk_scale[0],
                      test_T1 / self.uk_scale[1]], dtype=DTYPE)
        self.guess = x
