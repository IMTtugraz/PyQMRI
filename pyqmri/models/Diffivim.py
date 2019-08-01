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
                self.b0 = images[0]
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
                (5 / self.uk_scale[2]),
                True))
        self.constraints.append(
            constraints(
                (0 / self.uk_scale[3]),
                (50 / self.uk_scale[3]),
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
        ADCs = (x[2, ...]) * self.uk_scale[2]
        ADCf = (x[3, ...]) * self.uk_scale[3]

        S = M0 * (f * np.exp(-self.b * ADCf) +
                  (-f + 1) * np.exp(-self.b * ADCs))

        S *= self.phase
        S[~np.isfinite(S)] = 0
        S = S.astype(DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        M0 = x[0, ...]
        f = x[1, ...]
        ADCs = (x[2, ...])
        ADCf = (x[3, ...])
        M0_sc = self.uk_scale[0]
        f_sc = self.uk_scale[1]
        ADCs_sc = self.uk_scale[2]
        ADCf_sc = self.uk_scale[3]

        grad_M0 = M0_sc * (f * f_sc *
                           np.exp(-self.b * ADCf * ADCf_sc)
                           + (-f * f_sc + 1) *
                           np.exp(-self.b * ADCs * ADCs_sc)) * self.phase

        grad_f = M0 * M0_sc * (f_sc * np.exp(-self.b * ADCf * ADCf_sc)
                               - f_sc * np.exp(-ADCs * ADCs_sc * self.b)
                               ) * self.phase

        grad_ADCs = M0 * M0_sc * self.phase * (- ADCs_sc * self.b *
                                               (-f * f_sc + 1) *
                                               np.exp(
                                                   -ADCs * ADCs_sc * self.b))

        grad_ADCf = -ADCf_sc * self.b * M0 * M0_sc * f * f_sc * \
            np.exp(-self.b * ADCf * ADCf_sc) * self.phase

        grad = np.array([grad_M0, grad_f, grad_ADCs, grad_ADCf], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...]) * self.uk_scale[0]
        M0_min = M0.min()
        M0_max = M0.max()

        f = np.abs(x[1, ...]) * self.uk_scale[1]
        ADCs = (np.abs(x[2, ...]) * self.uk_scale[2])
        f_min = f.min()
        f_max = f.max()
        ADCs_min = ADCs.min()
        ADCs_max = ADCs.max()

        ADCf = (np.abs(x[3, ...]) * self.uk_scale[3])
        ADCf_min = ADCf.min()
        ADCf_max = ADCf.max()

        if dim_2D:
            if not self.figure:
                plt.ion()
                self.figure, self.ax = plt.subplots(1, 2, figsize=(12, 5))
                self.f_plot = self.ax[0].imshow((f))
                self.ax[0].set_title('Proton Density in a.u.')
                self.ax[0].axis('off')
                self.figure.colorbar(self.f_plot, ax=self.ax[0])
                self.ADC_plot = self.ax[1].imshow((ADCs))
                self.ax[1].set_title('ADCs in  ms')
                self.ax[1].axis('off')
                self.figure.colorbar(self.ADCs_plot, ax=self.ax[1])
                self.figure.tight_layout()
                plt.draw()
                plt.pause(1e-10)
            else:
                self.f_plot.set_data((f))
                self.f_plot.set_clim([f_min, f_max])
                self.ADCs_plot.set_data((ADCs))
                self.ADCs_plot.set_clim([ADCs_min, ADCs_max])
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

                self.ADCs_plot = self.ax[3].imshow(
                    (ADCs[int(self.NSlice / 2), ...]))
                self.ADCs_plot_cor = self.ax[16].imshow(
                    (ADCs[:, int(ADCs.shape[1] / 2), ...]))
                self.ADCs_plot_sag = self.ax[4].imshow(
                    np.flip((ADCs[:, :, int(ADCs.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('ADCs', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[16].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 5])
                cbar = self.figure.colorbar(self.ADCs_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCf_plot = self.ax[10].imshow(
                    (ADCf[int(self.NSlice / 2), ...]))
                self.ADCf_plot_cor = self.ax[23].imshow(
                    (ADCf[:, int(ADCf.shape[1] / 2), ...]))
                self.ADCf_plot_sag = self.ax[11].imshow(
                    np.flip((ADCf[:, :, int(ADCf.shape[-1] / 2)]).T, 1))
                self.ax[10].set_title('ADCf', color='white')
                self.ax[10].set_anchor('SE')
                self.ax[11].set_anchor('SW')
                self.ax[23].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 12])
                cbar = self.figure.colorbar(self.ADCf_plot, cax=cax)
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
                self.ADCs_plot.set_data((ADCs[int(self.NSlice / 2), ...]))
                self.ADCs_plot_cor.set_data(
                    (ADCs[:, int(ADCs.shape[1] / 2), ...]))
                self.ADCs_plot_sag.set_data(
                    np.flip((ADCs[:, :, int(ADCs.shape[-1] / 2)]).T, 1))
                self.ADCs_plot.set_clim([ADCs_min, ADCs_max])
                self.ADCs_plot_sag.set_clim([ADCs_min, ADCs_max])
                self.ADCs_plot_cor.set_clim([ADCs_min, ADCs_max])

                self.f_plot.set_data((f[int(self.NSlice / 2), ...]))
                self.f_plot_cor.set_data((f[:, int(f.shape[1] / 2), ...]))
                self.f_plot_sag.set_data(
                    np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
                self.f_plot.set_clim([f_min, f_max])
                self.f_plot_cor.set_clim([f_min, f_max])
                self.f_plot_sag.set_clim([f_min, f_max])
                self.ADCf_plot.set_data((ADCf[int(self.NSlice / 2), ...]))
                self.ADCf_plot_cor.set_data(
                    (ADCf[:, int(ADCf.shape[1] / 2), ...]))
                self.ADCf_plot_sag.set_data(
                    np.flip((ADCf[:, :, int(ADCf.shape[-1] / 2)]).T, 1))
                self.ADCf_plot.set_clim([ADCf_min, ADCf_max])
                self.ADCf_plot_sag.set_clim([ADCf_min, ADCf_max])
                self.ADCf_plot_cor.set_clim([ADCf_min, ADCf_max])
                plt.draw()
                plt.pause(1e-10)

    def _set_init_scales(self):
        test_M0 = self.b0
        f = 0.3 * np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        ADC_1 = 1 * np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        ADC_2 = 5 * np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)

        x = np.array(
                [test_M0,
                 f,
                 ADC_1,
                 ADC_2],
                dtype=DTYPE)
        return x
