#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyqmri.models.template import BaseModel, constraints, DTYPE
plt.ion()

unknowns_TGV = 4
unknowns_H1 = 0


class Model(BaseModel):
    def __init__(self, par, images):
        super().__init__(par)
        self.constraints = []
        self.images = images

        self.TE = np.ones((self.NScan, 1, 1, 1))
        try:
            for i in range(self.NScan):
                self.TE[i, ...] = par["TE"][i] * np.ones((1, 1, 1))
        except KeyError:
            raise KeyError("No TE found!")

        for j in range(unknowns_TGV + unknowns_H1):
            self.uk_scale.append(1)

        self.guess = self._set_init_scales(images)

        self.constraints.append(
            constraints(-100 / self.uk_scale[0],
                        100 / self.uk_scale[0],
                        False))
        self.constraints.append(
                constraints(0 / self.uk_scale[1],
                            1 / self.uk_scale[1],
                            True))
        self.constraints.append(
                constraints(((1 / 150) / self.uk_scale[2]),
                            ((1 / 1e-4) / self.uk_scale[2]),
                            True))
        self.constraints.append(
                constraints(((1 / 1500) / self.uk_scale[3]),
                            ((1 / 150) / self.uk_scale[3]),
                            True))

    def rescale(self, x):
        M0 = x[0, ...] * self.uk_scale[0]
        f = x[1, ...] * self.uk_scale[1]
        T21 = 1 / (x[2, ...] * self.uk_scale[2])
        T22 = 1 / (x[3, ...] * self.uk_scale[3])
        return np.array((M0, f, T21, T22))

    def execute_forward_2D(self, x, islice):
        pass

    def execute_gradient_2D(self, x, islice):
        pass

    def execute_forward_3D(self, x):
        M0 = x[0, ...] * self.uk_scale[0]
        f = x[1, ...] * self.uk_scale[1]
        T21 = x[2, ...] * self.uk_scale[2]
        T22 = x[3, ...] * self.uk_scale[3]
        S = M0 * (f * np.exp(-self.TE * (T21)) +
                  (1-f) * np.exp(-self.TE * (T22)))
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def execute_gradient_3D(self, x):
        M0 = x[0, ...]
        f = x[1, ...]
        T21 = x[2, ...]
        T22 = x[3, ...]
        grad_M0 = self.uk_scale[0] * (
            f * self.uk_scale[1] * np.exp(-self.TE * (
                T21 * self.uk_scale[2])) +
            (1 - f * self.uk_scale[1]) *
            np.exp(-self.TE * (T22 * self.uk_scale[3])))

        grad_f = M0 * self.uk_scale[0] * (
            self.uk_scale[1] * np.exp(-self.TE * (T21 * self.uk_scale[2])) -
            self.uk_scale[1] * np.exp(-self.TE * (T22 * self.uk_scale[3])))

        grad_T21 = -self.uk_scale[0] * M0 * f * self.uk_scale[1] * \
            self.TE * \
            self.uk_scale[2] * np.exp(-self.TE * (T21 * self.uk_scale[2]))

        grad_T22 = -self.uk_scale[0] * M0 * (1-f * self.uk_scale[1]) * \
            self.TE * \
            self.uk_scale[3] * np.exp(-self.TE * (T22 * self.uk_scale[4]))
        grad = np.array([grad_M0, grad_f, grad_T21, grad_T22], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def _set_init_scales(self, images):
        test_T21 = 1/150 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_T22 = 1/500 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_M0 = np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_f = 0.5*np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)

        x = np.array([test_M0 / self.uk_scale[0],
                      test_f / self.uk_scale[1],
                      test_T21 / self.uk_scale[2],
                      test_T22 / self.uk_scale[3]], dtype=DTYPE)
        return x

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...]) * self.uk_scale[0]
        M0_min = M0.min()
        M0_max = M0.max()

        f = np.abs(x[1, ...]) * self.uk_scale[1]

        T21 = 1 / (np.abs(x[2, ...]) * self.uk_scale[2])
        f_min = f.min()
        f_max = f.max()
        T21_min = T21.min()
        T21_max = T21.max()

        T22 = 1 / (np.abs(x[3, ...]) * self.uk_scale[3])
        T22_min = T22.min()
        T22_max = T22.max()

        [z, y, x] = f.shape
        self.ax = []
        if not self.figure:
            plt.ion()
            self.figure = plt.figure(figsize=(12, 6))
            self.figure.subplots_adjust(hspace=0, wspace=0)
            self.gs = gridspec.GridSpec(2,
                                        17,
                                        width_ratios=[x / z,
                                                      1,
                                                      x / (20 * z),
                                                      x / z * 0.25,
                                                      x / (20 * z),
                                                      x / z,
                                                      1,
                                                      x / z,
                                                      1,
                                                      x / (20 * z),
                                                      x / z * 0.25,
                                                      x / (20 * z),
                                                      x / z,
                                                      1,
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

            self.M0_plot = self.ax[0].imshow(
                (M0[int(self.NSlice / 2), ...]))
            self.M0_plot_cor = self.ax[17].imshow(
                (M0[:, int(f.shape[1] / 2), ...]))
            self.M0_plot_sag = self.ax[1].imshow(
                np.flip((M0[:, :, int(f.shape[-1] / 2)]).T, 1))
            self.ax[0].set_title('Proton Density in a.u.', color='white')
            self.ax[0].set_anchor('SE')
            self.ax[1].set_anchor('SW')
            self.ax[17].set_anchor('NW')
            cax = plt.subplot(self.gs[:, 2])
            cbar = self.figure.colorbar(self.M0_plot, cax=cax)
            cbar.ax.tick_params(labelsize=12, colors='white')
#           cax.yaxis.set_ticks_position('left')
            for spine in cbar.ax.spines:
                cbar.ax.spines[spine].set_color('white')

            self.f_plot = self.ax[5].imshow(
                (f[int(self.NSlice / 2), ...]))
            self.f_plot_cor = self.ax[22].imshow(
                (f[:, int(f.shape[1] / 2), ...]))
            self.f_plot_sag = self.ax[6].imshow(
                np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
            self.ax[5].set_title('Proton Density in a.u.', color='white')
            self.ax[5].set_anchor('SE')
            self.ax[6].set_anchor('SW')
            self.ax[22].set_anchor('NW')
            cax = plt.subplot(self.gs[:, 4])
            cbar = self.figure.colorbar(self.f_plot, cax=cax)
            cbar.ax.tick_params(labelsize=12, colors='white')
            cax.yaxis.set_ticks_position('left')
            for spine in cbar.ax.spines:
                cbar.ax.spines[spine].set_color('white')

            self.T21_plot = self.ax[7].imshow(
                (T21[int(self.NSlice / 2), ...]))
            self.T21_plot_cor = self.ax[24].imshow(
                (T21[:, int(T21.shape[1] / 2), ...]))
            self.T21_plot_sag = self.ax[8].imshow(
                np.flip((T21[:, :, int(T21.shape[-1] / 2)]).T, 1))
            self.ax[7].set_title('T21 in  ms', color='white')
            self.ax[7].set_anchor('SE')
            self.ax[8].set_anchor('SW')
            self.ax[24].set_anchor('NW')
            cax = plt.subplot(self.gs[:, 9])
            cbar = self.figure.colorbar(self.T21_plot, cax=cax)
            cbar.ax.tick_params(labelsize=12, colors='white')
            for spine in cbar.ax.spines:
                cbar.ax.spines[spine].set_color('white')

            self.T22_plot = self.ax[14].imshow(
                (T22[int(self.NSlice / 2), ...]))
            self.T22_plot_cor = self.ax[31].imshow(
                (T22[:, int(T22.shape[1] / 2), ...]))
            self.T22_plot_sag = self.ax[15].imshow(
                np.flip((T22[:, :, int(T22.shape[-1] / 2)]).T, 1))
            self.ax[14].set_title('T22 in  ms', color='white')
            self.ax[14].set_anchor('SE')
            self.ax[15].set_anchor('SW')
            self.ax[31].set_anchor('NW')
            cax = plt.subplot(self.gs[:, 16])
            cbar = self.figure.colorbar(self.T22_plot, cax=cax)
            cbar.ax.tick_params(labelsize=12, colors='white')
            for spine in cbar.ax.spines:
                cbar.ax.spines[spine].set_color('white')

            plt.draw()
            plt.pause(1e-10)
        else:
            self.M0_plot.set_data((M0[int(self.NSlice / 2), ...]))
            self.M0_plot_cor.set_data((M0[:, int(f.shape[1] / 2), ...]))
            self.M0_plot_sag.set_data(
                np.flip((M0[:, :, int(f.shape[-1] / 2)]).T, 1))
            self.M0_plot.set_clim([M0_min, M0_max])
            self.M0_plot_cor.set_clim([M0_min, M0_max])
            self.M0_plot_sag.set_clim([M0_min, M0_max])

            self.f_plot.set_data((f[int(self.NSlice / 2), ...]))
            self.f_plot_cor.set_data(
                (f[:, int(f.shape[1] / 2), ...]))
            self.f_plot_sag.set_data(
                np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
            self.f_plot.set_clim([f_min, f_max])
            self.f_plot_cor.set_clim([f_min, f_max])
            self.f_plot_sag.set_clim([f_min, f_max])
            self.T21_plot.set_data((T21[int(self.NSlice / 2), ...]))
            self.T21_plot_cor.set_data(
                (T21[:, int(T21.shape[1] / 2), ...]))
            self.T21_plot_sag.set_data(
                np.flip((T21[:, :, int(T21.shape[-1] / 2)]).T, 1))
            self.T21_plot.set_clim([T21_min, T21_max])
            self.T21_plot_sag.set_clim([T21_min, T21_max])
            self.T21_plot_cor.set_clim([T21_min, T21_max])

            self.T22_plot.set_data((T22[int(self.NSlice / 2), ...]))
            self.T22_plot_cor.set_data(
                (T22[:, int(T22.shape[1] / 2), ...]))
            self.T22_plot_sag.set_data(
                np.flip((T22[:, :, int(T22.shape[-1] / 2)]).T, 1))
            self.T22_plot.set_clim([T22_min, T22_max])
            self.T22_plot_sag.set_clim([T22_min, T22_max])
            self.T22_plot_cor.set_clim([T22_min, T22_max])
            plt.draw()
            plt.pause(1e-10)
