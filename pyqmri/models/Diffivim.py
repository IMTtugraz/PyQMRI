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

        self.b = np.ones((self.NScan, 1, 1, 1))

        for i in range(self.NScan):
            self.b[i, ...] = par["b_value"][i] * np.ones((1, 1, 1))

        if np.max(self.b) > 100:
            self.b /= 1000

        self.uk_scale = []
        for j in range(unknowns_TGV + unknowns_H1):
            self.uk_scale.append(1)

        self.unknowns = par["unknowns_TGV"] + par["unknowns_H1"]
        try:
            self.b0 = np.flip(
                np.transpose(par["file"]["b0"][()], (0, 2, 1)), 0)
        except KeyError:
            if par["imagespace"] is True:
                self.b0 = images[0]
            else:
                self.b0 = images[0]*par["dscale"]
        self.phase = np.exp(1j*(np.angle(images)-np.angle(images[0])))
        self.guess = self._set_init_scales()

        self.constraints.append(
            constraints(
                0 /
                self.uk_scale[0],
                100 /
                self.uk_scale[0],
                False))
        self.constraints.append(
            constraints(
                0 / self.uk_scale[1],
                1 / self.uk_scale[1],
                True))
        self.constraints.append(
            constraints(
                (0 / self.uk_scale[2]),
                (3 / self.uk_scale[2]),
                True))
        self.constraints.append(
            constraints(
                (0 / self.uk_scale[3]),
                (0.5 / self.uk_scale[3]),
                True))

    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_forward_3D(self, x):
        M0 = x[0, ...] * self.uk_scale[0]
        f = x[1, ...] * self.uk_scale[1]
        ADC1 = (x[2, ...]) * self.uk_scale[2]
        ADC2 = (x[3, ...]) * self.uk_scale[3]

        tmp = np.exp(-self.b * (ADC1 + ADC2))
        tmp2 = np.exp(-ADC1 * self.b)

        S = M0 * (f * tmp + (-f + 1) * tmp2)

        S *= self.phase
        S[~np.isfinite(S)] = 0
        S = S.astype(DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        M0 = x[0, ...]
        f = x[1, ...]
        ADC1 = (x[2, ...])
        ADC2 = (x[3, ...])
        M0_sc = self.uk_scale[0]
        f_sc = self.uk_scale[1]
        ADC1_sc = self.uk_scale[2]
        ADC2_sc = self.uk_scale[3]
        grad_M0 = M0_sc * (f * f_sc *
                           np.exp(-self.b * (ADC1 * ADC1_sc + ADC2 * ADC2_sc))
                           + (-f * f_sc + 1) *
                           np.exp(-ADC1 * ADC1_sc * self.b)) * self.phase

        grad_f = M0 * M0_sc * (-f_sc * np.exp(-ADC1 * ADC1_sc * self.b) +
                               f_sc * np.exp(-self.b *
                                             (ADC1 * ADC1_sc +
                                              ADC2 * ADC2_sc))) * self.phase

        grad_ADC1 = M0 * M0_sc * self.phase * (-ADC1_sc * self.b * f * f_sc *
                                               np.exp(
                                                 -self.b * (ADC1 * ADC1_sc +
                                                            ADC2 * ADC2_sc)) -
                                               ADC1_sc * self.b * (
                                                   -f * f_sc + 1) *
                                               np.exp(
                                                   -ADC1 * ADC1_sc * self.b))

        grad_ADC2 = -ADC2_sc * M0 * M0_sc * self.b * f * f_sc * \
            np.exp(-self.b * (ADC1 * ADC1_sc + ADC2 * ADC2_sc)) * self.phase

        grad = np.array([grad_M0, grad_f, grad_ADC1, grad_ADC2], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...]) * self.uk_scale[0]
        M0_min = M0.min()
        M0_max = M0.max()

        f = np.abs(x[1, ...]) * self.uk_scale[1]
        ADC1 = (np.abs(x[2, ...]) * self.uk_scale[2])
        f_min = f.min()
        f_max = f.max()
        ADC1_min = ADC1.min()
        ADC1_max = ADC1.max()

        ADC2 = (np.abs(x[3, ...]) * self.uk_scale[3])
        ADC2_min = ADC2.min()
        ADC2_max = ADC2.max()

        if dim_2D:
            if not self.figure:
                plt.ion()
                self.figure, self.ax = plt.subplots(1, 2, figsize=(12, 5))
                self.f_plot = self.ax[0].imshow((f))
                self.ax[0].set_title('Proton Density in a.u.')
                self.ax[0].axis('off')
                self.figure.colorbar(self.f_plot, ax=self.ax[0])
                self.ADC_plot = self.ax[1].imshow((ADC1))
                self.ax[1].set_title('ADC1 in  ms')
                self.ax[1].axis('off')
                self.figure.colorbar(self.ADC1_plot, ax=self.ax[1])
                self.figure.tight_layout()
                plt.draw()
                plt.pause(1e-10)
            else:
                self.f_plot.set_data((f))
                self.f_plot.set_clim([f_min, f_max])
                self.ADC1_plot.set_data((ADC1))
                self.ADC1_plot.set_clim([ADC1_min, ADC1_max])
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
                                            13,
                                            width_ratios=[x / (20 * z),
                                                          x / z,
                                                          1,
                                                          x / z,
                                                          1,
                                                          x / (20 * z),
                                                          x / z,
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

                self.M0_plot = self.ax[1].imshow(
                    (M0[int(self.NSlice / 2), ...]))
                self.M0_plot_cor = self.ax[14].imshow(
                    (M0[:, int(M0.shape[1] / 2), ...]))
                self.M0_plot_sag = self.ax[2].imshow(
                    np.flip((M0[:, :, int(M0.shape[-1] / 2)]).T, 1))
                self.ax[1].set_title('Proton Density in a.u.', color='white')
                self.ax[1].set_anchor('SE')
                self.ax[2].set_anchor('SW')
                self.ax[14].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 0])
                cbar = self.figure.colorbar(self.M0_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.f_plot = self.ax[8].imshow((f[int(self.NSlice / 2), ...]))
                self.f_plot_cor = self.ax[21].imshow(
                    (f[:, int(f.shape[1] / 2), ...]))
                self.f_plot_sag = self.ax[9].imshow(
                    np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
                self.ax[8].set_title('f', color='white')
                self.ax[8].set_anchor('SE')
                self.ax[9].set_anchor('SW')
                self.ax[21].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 7])
                cbar = self.figure.colorbar(self.f_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC1_plot = self.ax[3].imshow(
                    (ADC1[int(self.NSlice / 2), ...]))
                self.ADC1_plot_cor = self.ax[16].imshow(
                    (ADC1[:, int(ADC1.shape[1] / 2), ...]))
                self.ADC1_plot_sag = self.ax[4].imshow(
                    np.flip((ADC1[:, :, int(ADC1.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('ADC1', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[16].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 5])
                cbar = self.figure.colorbar(self.ADC1_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC2_plot = self.ax[10].imshow(
                    (ADC2[int(self.NSlice / 2), ...]))
                self.ADC2_plot_cor = self.ax[23].imshow(
                    (ADC2[:, int(ADC2.shape[1] / 2), ...]))
                self.ADC2_plot_sag = self.ax[11].imshow(
                    np.flip((ADC2[:, :, int(ADC2.shape[-1] / 2)]).T, 1))
                self.ax[10].set_title('ADC2', color='white')
                self.ax[10].set_anchor('SE')
                self.ax[11].set_anchor('SW')
                self.ax[23].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 12])
                cbar = self.figure.colorbar(self.ADC2_plot, cax=cax)
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
                self.ADC1_plot.set_data((ADC1[int(self.NSlice / 2), ...]))
                self.ADC1_plot_cor.set_data(
                    (ADC1[:, int(ADC1.shape[1] / 2), ...]))
                self.ADC1_plot_sag.set_data(
                    np.flip((ADC1[:, :, int(ADC1.shape[-1] / 2)]).T, 1))
                self.ADC1_plot.set_clim([ADC1_min, ADC1_max])
                self.ADC1_plot_sag.set_clim([ADC1_min, ADC1_max])
                self.ADC1_plot_cor.set_clim([ADC1_min, ADC1_max])

                self.f_plot.set_data((f[int(self.NSlice / 2), ...]))
                self.f_plot_cor.set_data((f[:, int(f.shape[1] / 2), ...]))
                self.f_plot_sag.set_data(
                    np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
                self.f_plot.set_clim([f_min, f_max])
                self.f_plot_cor.set_clim([f_min, f_max])
                self.f_plot_sag.set_clim([f_min, f_max])
                self.ADC2_plot.set_data((ADC2[int(self.NSlice / 2), ...]))
                self.ADC2_plot_cor.set_data(
                    (ADC2[:, int(ADC2.shape[1] / 2), ...]))
                self.ADC2_plot_sag.set_data(
                    np.flip((ADC2[:, :, int(ADC2.shape[-1] / 2)]).T, 1))
                self.ADC2_plot.set_clim([ADC2_min, ADC2_max])
                self.ADC2_plot_sag.set_clim([ADC2_min, ADC2_max])
                self.ADC2_plot_cor.set_clim([ADC2_min, ADC2_max])
                plt.draw()
                plt.pause(1e-10)

    def _set_init_scales(self):
        test_M0 = self.b0
        f = 0.3 * np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        ADC_1 = 1 * np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        ADC_2 = 0.1 * np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)

        x = np.array(
                [test_M0,
                 f,
                 ADC_1,
                 ADC_2],
                dtype=DTYPE)
        return x
