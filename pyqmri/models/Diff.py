#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyqmri.models.template import BaseModel, constraints, DTYPE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


class Model(BaseModel):
    def __init__(self, par):
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
        self.uk_scale = []
        par["unknowns_TGV"] = 2
        par["unknowns_H1"] = 0 
        par["unknowns"] = par["unknowns_TGV"] + par["unknowns_H1"]
        for i in range(par["unknowns_TGV"] + par["unknowns_H1"]):
            self.uk_scale.append(1)
        try:
            self.b0 = np.flip(
                np.transpose(
                    par["file"]["b0"][()], (0, 2, 1)), 0)
        except KeyError:
            print("No b0 image provided")
            self.b0 =  None

        self.constraints.append(
            constraints(
                0 / self.uk_scale[0],
                100 / self.uk_scale[0],
                False))
        self.constraints.append(
            constraints(
                (0 / self.uk_scale[1]),
                (5 / self.uk_scale[1]),
                True))
#        for j in range(phase_maps):
#            self.constraints.append(constraints(
#                (-2*np.pi / self.uk_scale[-phase_maps + j]),
#                (2*np.pi / self.uk_scale[-phase_maps + j]), True))

    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_forward_3D(self, x):
        ADC = x[1, ...] * self.uk_scale[1]
        S = x[0, ...] * self.uk_scale[0] * np.exp(-self.b * ADC)

        S *= self.phase

        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        M0 = x[0, ...]
        ADC = x[1, ...]
        grad_M0 = np.exp(-self.b * (ADC * self.uk_scale[1])) * self.uk_scale[0]

        grad_M0 *= self.phase
        grad_ADC = -grad_M0 * M0 * self.b * self.uk_scale[1]

        grad = np.array([grad_M0, grad_ADC],
                        dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...]) * self.uk_scale[0]
        ADC = (np.abs(x[1, ...]) * self.uk_scale[1])
        M0_min = M0.min()
        M0_max = M0.max()
        ADC_min = ADC.min()
        ADC_max = ADC.max()

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
            self.ax_phase = []
            if not self.figure:
                plt.ion()
                self.figure = plt.figure(figsize=(12, 6))
                self.figure.subplots_adjust(hspace=0, wspace=0)
                self.gs = gridspec.GridSpec(2,
                                            6,
                                            width_ratios=[
                                              x / (20 * z), x / z, 1,
                                              x / z, 1, x / (20 * z)],
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

                self.ADC_plot = self.ax[3].imshow(
                    (ADC[int(self.NSlice / 2), ...]))
                self.ADC_plot_cor = self.ax[9].imshow(
                    (ADC[:, int(ADC.shape[1] / 2), ...]))
                self.ADC_plot_sag = self.ax[4].imshow(
                    np.flip((ADC[:, :, int(ADC.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('ADC in  ms', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[9].set_anchor('NW')
                cax = plt.subplot(self.gs[:, 5])
                cbar = self.figure.colorbar(self.ADC_plot, cax=cax)
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

                plt.draw()
                plt.pause(1e-10)


    def computeInitialGuess(self, *args):
        self.phase = np.exp(1j*(np.angle(args[0])-np.angle(args[0][0])))
        if self.b0 is not None:
            test_M0 = self.b0
        else:
            test_M0 = args[0][0]
        ADC = np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)

        x = np.array((test_M0, ADC))
        self.guess = x
