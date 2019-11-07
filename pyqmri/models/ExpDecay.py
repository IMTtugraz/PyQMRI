#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyqmri.models.template import BaseModel, constraints, DTYPE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


unknowns_TGV = 2
unknowns_H1 = 0


class Model(BaseModel):
    def __init__(self, par, images):
        super().__init__(par)
        self.TE = np.ones((self.NScan, 1, 1, 1))
        try:
            for i in range(self.NScan):
                self.TE[i, ...] = par["TE"][i] * np.ones((1, 1, 1))
        except KeyError:
            raise KeyError("No TE found!")

        self.uk_scale = []
        for j in range(unknowns_TGV + unknowns_H1):
            self.uk_scale.append(1)

        self.guess = self._set_init_scales()

        self.constraints.append(
            constraints(
                1e-4 / self.uk_scale[0],
                10 / self.uk_scale[0],
                False))
        self.constraints.append(
            constraints(
                ((1 / 150) / self.uk_scale[1]),
                ((1 / 5) / self.uk_scale[1]),
                True))

    def rescale(self, x):
        tmp_x = np.copy(x)
        tmp_x[0] *= self.uk_scale[0]
        tmp_x[1] = 1/(tmp_x[1]*self.uk_scale[1])
        return tmp_x

    def _execute_forward_2D(self, x, islice):
        R2 = x[1, ...] * self.uk_scale[1]
        S = x[0, ...] * self.uk_scale[0] * np.exp(-self.TE * (R2))
        S[~np.isfinite(S)] = 1e-200
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_2D(self, x, islice):
        M0 = x[0, ...]
        R2 = x[1, ...]
        grad_M0 = self.uk_scale[0] * np.exp(-self.TE * (R2 * self.uk_scale[1]))
        grad_T2 = -M0 * self.uk_scale[0] * self.TE * \
            self.uk_scale[1] * np.exp(-self.TE * (R2 * self.uk_scale[1]))
        grad = np.array([grad_M0, grad_T2], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def _execute_forward_3D(self, x):
        R2 = x[1, ...] * self.uk_scale[1]
        S = x[0, ...] * self.uk_scale[0] * np.exp(-self.TE * (R2))
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        M0 = x[0, ...]
        R2 = x[1, ...]
        grad_M0 = np.exp(-self.TE * (R2 * self.uk_scale[1])) * self.uk_scale[0]
        grad_T2 = -M0 * self.TE * \
            self.uk_scale[1] * np.exp(-self.TE * (R2 * self.uk_scale[1])) * \
            self.uk_scale[0]
        grad = np.array([grad_M0, grad_T2], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        print(
            'Grad Scaling',
            np.linalg.norm(
                np.abs(grad_M0)) /
            np.linalg.norm(
                np.abs(grad_T2)))
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...]) * self.uk_scale[0]
        T2 = 1 / (np.abs(x[1, ...]) * self.uk_scale[1])
        M0_min = M0.min()
        M0_max = M0.max()
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
                self.T2_plot = self.ax[1].imshow((T2))
                self.ax[1].set_title('T2 in  ms')
                self.ax[1].axis('off')
                self.figure.colorbar(self.T2_plot, ax=self.ax[1])
                self.figure.tight_layout()
                plt.draw()
                plt.pause(1e-10)
            else:
                self.M0_plot.set_data((M0))
                self.M0_plot.set_clim([M0_min, M0_max])
                self.T2_plot.set_data((T2))
                self.T2_plot.set_clim([T2_min, T2_max])
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
                    2, 6, width_ratios=[
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

                self.T2_plot = self.ax[3].imshow(
                    (T2[int(self.NSlice / 2), ...]))
                self.T2_plot_cor = self.ax[9].imshow(
                    (T2[:, int(T2.shape[1] / 2), ...]))
                self.T2_plot_sag = self.ax[4].imshow(
                    np.flip((T2[:, :, int(T2.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('T2 in  ms', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[9].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 5])
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
        test_M0 = 1 * np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_T2 = 1 / self.uk_scale[1] * 1/50 * \
            np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        x = np.array([test_M0, test_T2], dtype=DTYPE)
        return x
