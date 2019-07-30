#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyqmri.models.template import BaseModel, constraints, DTYPE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

unknowns_TGV = 3
unknowns_H1 = 0


class Model(BaseModel):
    def __init__(self, par, images):
        super().__init__(par)
        self.b = np.ones((self.NScan, 1, 1, 1))
        try:
            self.NScan = par["T2PREP"].size
            for i in range(self.NScan):
                self.b[i, ...] = par["T2PREP"][i] * np.ones((1, 1, 1))
        except BaseException:
            self.NScan = par["b_value"].size
            for i in range(self.NScan):
                self.b[i, ...] = par["b_value"][i] * np.ones((1, 1, 1))

        if np.max(self.b) > 100:
            self.b /= 1000

        for i in range(unknowns_TGV + unknowns_H1):
            self.uk_scale.append(1)
        self.dscale = par["dscale"]
        try:
            self.b0 = np.flip(
                np.transpose(par["file"]["b0"][()], (0, 2, 1)), 0)
        except KeyError:
            self.b0 = images[0]*par["dscale"]

        self.guess = self._set_init_scales(images)

        self.constraints.append(
            constraints(
                -np.inf / self.uk_scale[0],
                np.log(500) / self.uk_scale[0],
                True))
        self.constraints.append(
            constraints(
                (0 / self.uk_scale[1]),
                (3 / self.uk_scale[1]),
                True))
        self.constraints.append(
            constraints(
                (0 / self.uk_scale[2]),
                (2 / self.uk_scale[2]),
                True))

    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_forward_3D(self, x):
        ADC = x[1, ...] * self.uk_scale[1]
        kurt = x[2, ...] * self.uk_scale[2]

        S = x[0, ...] * self.uk_scale[0] +\
            (1 / 6 * ADC**2 * self.b**2 * kurt - ADC * self.b)

        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_3D(self, x):

        ADC = x[1, ...]
        kurt = x[2, ...]

        ADC_sc = self.uk_scale[1]
        kurt_sc = self.uk_scale[2]

        ADC = ADC*ADC_sc

        grad_ADC = (2 / 6 * ADC * ADC_sc *
                    self.b**2 * kurt * kurt_sc - self.b * ADC_sc)

        grad_kurt = 1 / 6 * ADC**2 * self.b**2 * kurt_sc
        grad_M0 = self.uk_scale[0] * np.ones_like(grad_ADC)

        grad = np.array([grad_M0, grad_ADC, grad_kurt],
                        dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.exp(np.abs(x[0, ...]) * self.uk_scale[0])
        ADC = (np.abs(x[1, ...]) * self.uk_scale[1])
        kurt = (np.abs(x[2, ...]) * self.uk_scale[2])
        M0_min = M0.min()
        M0_max = M0.max()
        ADC_min = ADC.min()
        ADC_max = ADC.max()
        kurt_min = kurt.min()
        kurt_max = kurt.max()
        if dim_2D:
            if not self.figure:
                plt.ion()
                self.figure, self.ax = plt.subplots(1, 2, figsize=(12, 5))
                self.M0_plot = self.ax[0].imshow((M0))
                self.ax[0].set_title('Proton Density in a.u.')
                self.ax[0].axis('off')
                self.figure.colorbar(self.M0_plot, ax=self.ax[0])
                self.ADC_plot = self.ax[1].imshow((ADC))
                self.ax[1].set_title('ADC in  ms')
                self.ax[1].axis('off')
                self.figure.colorbar(self.ADC_plot, ax=self.ax[1])
                self.figure.tight_layout()
                plt.draw()
                plt.pause(1e-10)
            else:
                self.M0_plot.set_data((M0))
                self.M0_plot.set_clim([M0_min, M0_max])
                self.ADC_plot.set_data((ADC))
                self.ADC_plot.set_clim([ADC_min, ADC_max])
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
                                                          x / (2 * z),
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

                self.ADC_plot = self.ax[3].imshow(
                    (ADC[int(self.NSlice / 2), ...]))
                self.ADC_plot_cor = self.ax[13].imshow(
                    (ADC[:, int(ADC.shape[1] / 2), ...]))
                self.ADC_plot_sag = self.ax[4].imshow(
                    np.flip((ADC[:, :, int(ADC.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('ADC', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[13].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 5])
                cbar = self.figure.colorbar(self.ADC_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                plt.draw()

                self.kurt_plot = self.ax[7].imshow(
                    (kurt[int(self.NSlice / 2), ...]))
                self.kurt_plot_cor = self.ax[17].imshow(
                    (kurt[:, int(ADC.shape[1] / 2), ...]))
                self.kurt_plot_sag = self.ax[8].imshow(
                    np.flip((kurt[:, :, int(ADC.shape[-1] / 2)]).T, 1))
                self.ax[7].set_title('Kurtosis', color='white')
                self.ax[7].set_anchor('SE')
                self.ax[8].set_anchor('SW')
                self.ax[17].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 9])
                cbar = self.figure.colorbar(self.kurt_plot, cax=cax)
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
                self.ADC_plot.set_data((ADC[int(self.NSlice / 2), ...]))
                self.ADC_plot_cor.set_data(
                    (ADC[:, int(ADC.shape[1] / 2), ...]))
                self.ADC_plot_sag.set_data(
                    np.flip((ADC[:, :, int(ADC.shape[-1] / 2)]).T, 1))
                self.ADC_plot.set_clim([ADC_min, ADC_max])
                self.ADC_plot_sag.set_clim([ADC_min, ADC_max])
                self.ADC_plot_cor.set_clim([ADC_min, ADC_max])

                self.kurt_plot.set_data((kurt[int(self.NSlice / 2), ...]))
                self.kurt_plot_cor.set_data(
                    (kurt[:, int(kurt.shape[1] / 2), ...]))
                self.kurt_plot_sag.set_data(
                    np.flip((kurt[:, :, int(kurt.shape[-1] / 2)]).T, 1))
                self.kurt_plot.set_clim([kurt_min, kurt_max])
                self.kurt_plot_sag.set_clim([kurt_min, kurt_max])
                self.kurt_plot_cor.set_clim([kurt_min, kurt_max])
                plt.draw()
                plt.pause(1e-10)

    def _set_init_scales(self, images):

        test_M0 = self.b0
        ADC = 1 * np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        kurt = 1 * np.ones((self.NSlice, self.dimY,
                            self.dimX), dtype=DTYPE)

        x = np.array(
                [
                    test_M0,
                    ADC,
                    kurt],
                dtype=DTYPE)
        return x