#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyqmri.models.template import BaseModel, constraints, DTYPE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
unknowns_TGV = 7
unknowns_H1 = 0


class Model(BaseModel):
    def __init__(self, par, images):
        super().__init__(par)
        self.images = images
        self.NSlice = par['NSlice']

        self.figure_phase = None

        self.b = np.ones((self.NScan, 1, 1, 1))
        self.dir = par["DWI_dir"].T
        for i in range(self.NScan):
            self.b[i, ...] = par["b_value"][i] * np.ones((1, 1, 1))

        if np.max(self.b) > 100:
            self.b /= 1000

        self.dir = self.dir[:, None, None, None, :]

        self.uk_scale = []
        for j in range(unknowns_TGV + unknowns_H1):
            self.uk_scale.append(1)

        self.unknowns = par["unknowns_TGV"] + par["unknowns_H1"]
        try:
            self.b0 = np.flip(
                np.transpose(par["file"]["b0"][()], (0, 2, 1)), 0)
        except KeyError:
            self.b0 = images[0]
        self.phase = np.exp(1j*(np.angle(images)-np.angle(images[0])))
        self.guess = self._set_init_scales(images)

        self.constraints.append(
            constraints(
                0 / self.uk_scale[0],
                10 / self.uk_scale[0],
                False))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[1]),
                (10e0 / self.uk_scale[1]),
                True))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[2]),
                (10e0 / self.uk_scale[2]),
                True))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[3]),
                (10e0 / self.uk_scale[3]),
                True))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[4]),
                (10e0 / self.uk_scale[4]),
                True))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[5]),
                (10e0 / self.uk_scale[5]),
                True))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[6]),
                (10e0 / self.uk_scale[6]),
                True))

    def rescale(self, x):
        M0 = x[0, ...] * self.uk_scale[0]
        ADC_x = (np.real(x[1, ...]**2) * self.uk_scale[1]**2)
        ADC_xy = (np.real(x[2, ...] * self.uk_scale[2] *
                          x[1, ...] * self.uk_scale[1]))
        ADC_y = (np.real(x[2, ...]**2 * self.uk_scale[2]**2 +
                         x[3, ...]**2 * self.uk_scale[3]**2))
        ADC_xz = (np.real(x[4, ...] * self.uk_scale[4] *
                          x[1, ...] * self.uk_scale[1]))
        ADC_z = (np.real(x[4, ...]**2 * self.uk_scale[4]**2 +
                         x[5, ...]**2 * self.uk_scale[5]**2 +
                         x[6, ...]**2 * self.uk_scale[6]**2))
        ADC_yz = (np.real(x[2, ...] * self.uk_scale[2] *
                          x[4, ...] * self.uk_scale[4] +
                          x[6, ...] * self.uk_scale[6] *
                          x[3, ...] * self.uk_scale[3]))

        return np.array((M0, ADC_x, ADC_xy, ADC_y, ADC_xz, ADC_z, ADC_yz))

    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_forward_3D(self, x):
        ADC = x[1, ...]**2 * self.uk_scale[1]**2 * self.dir[..., 0]**2 + \
              (x[2, ...]**2 * self.uk_scale[2]**2 +
               x[3, ...]**2 * self.uk_scale[3]**2) * self.dir[..., 1]**2 + \
              (x[4, ...]**2 * self.uk_scale[4]**2 +
               x[5, ...]**2 * self.uk_scale[5]**2 +
               x[6, ...]**2 * self.uk_scale[6]**2) * self.dir[..., 2]**2 +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[1, ...] * self.uk_scale[1]) * \
              self.dir[..., 0] * self.dir[..., 1] + \
              2 * (x[4, ...] * self.uk_scale[4] *
                   x[1, ...] * self.uk_scale[1]) *\
              self.dir[..., 0] * self.dir[..., 2] +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[4, ...] * self.uk_scale[4] +
                   x[6, ...] * self.uk_scale[6] *
                   x[3, ...] * self.uk_scale[3]) * \
              self.dir[..., 1] * self.dir[..., 2]

        S = (x[0, ...] * self.uk_scale[0] *
             np.exp(- ADC * self.b)).astype(DTYPE)

        S *= self.phase
        S[~np.isfinite(S)] = 0
        return S

    def _execute_gradient_3D(self, x):
        ADC = x[1, ...]**2 * self.uk_scale[1]**2 * self.dir[..., 0]**2 + \
              (x[2, ...]**2 * self.uk_scale[2]**2 +
               x[3, ...]**2 * self.uk_scale[3]**2) * self.dir[..., 1]**2 + \
              (x[4, ...]**2 * self.uk_scale[4]**2 +
               x[5, ...]**2 * self.uk_scale[5]**2 +
               x[6, ...]**2 * self.uk_scale[6]**2) * self.dir[..., 2]**2 +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[1, ...] * self.uk_scale[1]) * \
              self.dir[..., 0] * self.dir[..., 1] + \
              2 * (x[4, ...] * self.uk_scale[4] *
                   x[1, ...] * self.uk_scale[1]) *\
              self.dir[..., 0] * self.dir[..., 2] +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[4, ...] * self.uk_scale[4] +
                   x[6, ...] * self.uk_scale[6] *
                   x[3, ...] * self.uk_scale[3]) * \
              self.dir[..., 1] * self.dir[..., 2]

        grad_M0 = self.uk_scale[0] * np.exp(- ADC * self.b)
        del ADC

        grad_M0 *= self.phase
        grad_ADC_x = -x[0, ...] * self.b * grad_M0 * \
            (2 * x[1, ...] * self.uk_scale[1]**2 * self.dir[..., 0]**2 +
             2 * self.uk_scale[1] * x[2, ...] * self.uk_scale[2] *
             self.dir[..., 0] * self.dir[..., 1] +
             2 * self.uk_scale[1] * x[4, ...] * self.uk_scale[4] *
             self.dir[..., 0] * self.dir[..., 2])
        grad_ADC_xy = -x[0, ...] * self.b * grad_M0 * \
            (2 * x[1, ...] * self.uk_scale[1] * self.uk_scale[2] *
             self.dir[..., 0] * self.dir[..., 1] +
             2 * x[2, ...] * self.uk_scale[2]**2 *
             self.dir[..., 1]**2 +
             2 * self.uk_scale[2] * x[4, ...] * self.uk_scale[4] *
             self.dir[..., 1] * self.dir[..., 2])

        grad_ADC_y = -x[0, ...] * self.b * grad_M0 *\
            (2 * x[3, ...] * self.uk_scale[3]**2 * self.dir[..., 1]**2 +
             2 * self.uk_scale[3] * x[6, ...] * self.uk_scale[6] *
             self.dir[..., 1] * self.dir[..., 2])
        grad_ADC_xz = -x[0, ...] * self.b * grad_M0 *\
            (2 * x[1, ...] * self.uk_scale[1] * self.uk_scale[4] *
             self.dir[..., 0] * self.dir[..., 2] +
             2 * x[2, ...] * self.uk_scale[2] * self.uk_scale[4] *
             self.dir[..., 1] * self.dir[..., 2] +
             2 * x[4, ...] * self.uk_scale[4]**2 * self.dir[..., 2]**2)

        grad_ADC_z = -2 * x[5, ...] * self.uk_scale[5]**2 *\
            x[0, ...]*self.b*self.dir[..., 2]**2*grad_M0

        grad_ADC_yz = - x[0, ...] * self.b * grad_M0 *\
            (2 * x[3, ...] * self.uk_scale[3] * self.uk_scale[6] *
             self.dir[..., 1] * self.dir[..., 2] +
             2 * x[6, ...] * self.uk_scale[6]**2 * self.dir[..., 2]**2)

        grad = np.array(
            [grad_M0,
             grad_ADC_x,
             grad_ADC_xy,
             grad_ADC_y,
             grad_ADC_xz,
             grad_ADC_z,
             grad_ADC_yz], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 0
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...]) * self.uk_scale[0]
        ADC_x = (np.real(x[1, ...]**2 * self.uk_scale[1]**2))
        ADC_xy = (np.real(x[2, ...] * self.uk_scale[2] *
                          x[1, ...] * self.uk_scale[1]))
        M0_min = M0.min()
        M0_max = M0.max()
        ADC_x_min = ADC_x.min()
        ADC_x_max = ADC_x.max()
        ADC_xy_min = ADC_xy.min()
        ADC_xy_max = ADC_xy.max()

        ADC_y = (np.real(x[2, ...]**2 * self.uk_scale[2]**2 +
                         x[3, ...]**2 * self.uk_scale[3]**2))
        ADC_xz = (np.real(x[4, ...] * self.uk_scale[4] *
                          x[1, ...] * self.uk_scale[1]))
        ADC_y_min = ADC_y.min()
        ADC_y_max = ADC_y.max()
        ADC_xz_min = ADC_xz.min()
        ADC_xz_max = ADC_xz.max()

        ADC_z = (np.real(x[4, ...]**2 * self.uk_scale[4]**2 +
                         x[5, ...]**2 * self.uk_scale[5]**2 +
                         x[6, ...]**2 * self.uk_scale[6]**2))
        ADC_yz = (np.real(x[2, ...] * self.uk_scale[2] *
                          x[4, ...] * self.uk_scale[4] +
                          x[6, ...] * self.uk_scale[6] *
                          x[3, ...] * self.uk_scale[3]))
        ADC_z_min = ADC_z.min()
        ADC_z_max = ADC_z.max()
        ADC_yz_min = ADC_yz.min()
        ADC_yz_max = ADC_yz.max()

        if dim_2D:
            if not self.figure:
                plt.ion()
                self.figure, self.ax = plt.subplots(1, 2, figsize=(12, 5))
                self.M0_plot = self.ax[0].imshow((M0))
                self.ax[0].set_title('Proton Density in a.u.')
                self.ax[0].axis('off')
                self.figure.colorbar(self.M0_plot, ax=self.ax[0])
                self.ADC_x_plot = self.ax[1].imshow((ADC_x))
                self.ax[1].set_title('ADC_x in  ms')
                self.ax[1].axis('off')
                self.figure.colorbar(self.ADC_x_plot, ax=self.ax[1])
                self.figure.tight_layout()
                plt.draw()
                plt.pause(1e-10)
            else:
                self.M0_plot.set_data((M0))
                self.M0_plot.set_clim([M0_min, M0_max])
                self.ADC_x_plot.set_data((ADC_x))
                self.ADC_x_plot.set_clim([ADC_x_min, ADC_x_max])
                plt.draw()
                plt.pause(1e-10)
        else:
            [z, y, x] = M0.shape
            self.ax = []
            self.ax_phase = []
            self.ax_kurt = []
            if not self.figure:
                plt.ion()
                self.figure = plt.figure(figsize=(12, 6))
                self.figure.subplots_adjust(hspace=0, wspace=0)
                self.gs = gridspec.GridSpec(8,
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
                                                           1,
                                                           x / z,
                                                           1,
                                                           x / z,
                                                           1,
                                                           x / z,
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
                self.ax[11].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 0])
                cbar = self.figure.colorbar(self.M0_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_x_plot = self.ax[3].imshow(
                    (ADC_x[int(self.NSlice / 2), ...]))
                self.ADC_x_plot_cor = self.ax[13].imshow(
                    (ADC_x[:, int(ADC_x.shape[1] / 2), ...]))
                self.ADC_x_plot_sag = self.ax[4].imshow(
                    np.flip((ADC_x[:, :, int(ADC_x.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('ADC_x', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[13].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 5])
                cbar = self.figure.colorbar(self.ADC_x_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_xy_plot = self.ax[7].imshow(
                    (ADC_xy[int(self.NSlice / 2), ...]))
                self.ADC_xy_plot_cor = self.ax[17].imshow(
                    (ADC_xy[:, int(ADC_x.shape[1] / 2), ...]))
                self.ADC_xy_plot_sag = self.ax[8].imshow(
                    np.flip((ADC_xy[:, :, int(ADC_x.shape[-1] / 2)]).T, 1))
                self.ax[7].set_title('ADC_xy', color='white')
                self.ax[7].set_anchor('SE')
                self.ax[8].set_anchor('SW')
                self.ax[17].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 9])
                cbar = self.figure.colorbar(self.ADC_xy_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_y_plot = self.ax[23].imshow(
                    (ADC_y[int(self.NSlice / 2), ...]))
                self.ADC_y_plot_cor = self.ax[33].imshow(
                    (ADC_y[:, int(ADC_y.shape[1] / 2), ...]))
                self.ADC_y_plot_sag = self.ax[24].imshow(
                    np.flip((ADC_y[:, :, int(ADC_y.shape[-1] / 2)]).T, 1))
                self.ax[23].set_title('ADC_y', color='white')
                self.ax[23].set_anchor('SE')
                self.ax[24].set_anchor('SW')
                self.ax[33].set_anchor('NE')
                cax = plt.subplot(self.gs[2:4, 5])
                cbar = self.figure.colorbar(self.ADC_y_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_xz_plot = self.ax[27].imshow(
                    (ADC_xz[int(self.NSlice / 2), ...]))
                self.ADC_xz_plot_cor = self.ax[37].imshow(
                    (ADC_xz[:, int(ADC_y.shape[1] / 2), ...]))
                self.ADC_xz_plot_sag = self.ax[28].imshow(
                    np.flip((ADC_xz[:, :, int(ADC_y.shape[-1] / 2)]).T, 1))
                self.ax[27].set_title('ADC_xz', color='white')
                self.ax[27].set_anchor('SE')
                self.ax[28].set_anchor('SW')
                self.ax[37].set_anchor('NE')
                cax = plt.subplot(self.gs[2:4, 9])
                cbar = self.figure.colorbar(self.ADC_xz_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_z_plot = self.ax[43].imshow(
                    (ADC_z[int(self.NSlice / 2), ...]))
                self.ADC_z_plot_cor = self.ax[53].imshow(
                    (ADC_z[:, int(ADC_z.shape[1] / 2), ...]))
                self.ADC_z_plot_sag = self.ax[44].imshow(
                    np.flip((ADC_z[:, :, int(ADC_z.shape[-1] / 2)]).T, 1))
                self.ax[43].set_title('ADC_z', color='white')
                self.ax[43].set_anchor('SE')
                self.ax[44].set_anchor('SW')
                self.ax[53].set_anchor('NE')
                cax = plt.subplot(self.gs[4:6, 5])
                cbar = self.figure.colorbar(self.ADC_z_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_yz_plot = self.ax[47].imshow(
                    (ADC_yz[int(self.NSlice / 2), ...]))
                self.ADC_yz_plot_cor = self.ax[57].imshow(
                    (ADC_yz[:, int(ADC_z.shape[1] / 2), ...]))
                self.ADC_yz_plot_sag = self.ax[48].imshow(
                    np.flip((ADC_yz[:, :, int(ADC_z.shape[-1] / 2)]).T, 1))
                self.ax[47].set_title('ADC_yz', color='white')
                self.ax[47].set_anchor('SE')
                self.ax[48].set_anchor('SW')
                self.ax[57].set_anchor('NE')
                cax = plt.subplot(self.gs[4:6, 9])
                cbar = self.figure.colorbar(self.ADC_yz_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                plt.draw()
                plt.pause(1e-10)
                self.figure.canvas.draw_idle()

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

                self.ADC_x_plot.set_data((ADC_x[int(self.NSlice / 2), ...]))
                self.ADC_x_plot_cor.set_data(
                    (ADC_x[:, int(ADC_x.shape[1] / 2), ...]))
                self.ADC_x_plot_sag.set_data(
                    np.flip((ADC_x[:, :, int(ADC_x.shape[-1] / 2)]).T, 1))
                self.ADC_x_plot.set_clim([ADC_x_min, ADC_x_max])
                self.ADC_x_plot_sag.set_clim([ADC_x_min, ADC_x_max])
                self.ADC_x_plot_cor.set_clim([ADC_x_min, ADC_x_max])

                self.ADC_xy_plot.set_data((ADC_xy[int(self.NSlice / 2), ...]))
                self.ADC_xy_plot_cor.set_data(
                    (ADC_xy[:, int(ADC_xy.shape[1] / 2), ...]))
                self.ADC_xy_plot_sag.set_data(
                    np.flip((ADC_xy[:, :, int(ADC_xy.shape[-1] / 2)]).T, 1))
                self.ADC_xy_plot.set_clim([ADC_xy_min, ADC_xy_max])
                self.ADC_xy_plot_sag.set_clim([ADC_xy_min, ADC_xy_max])
                self.ADC_xy_plot_cor.set_clim([ADC_xy_min, ADC_xy_max])

                self.ADC_y_plot.set_data((ADC_y[int(self.NSlice / 2), ...]))
                self.ADC_y_plot_cor.set_data(
                    (ADC_y[:, int(ADC_y.shape[1] / 2), ...]))
                self.ADC_y_plot_sag.set_data(
                    np.flip((ADC_y[:, :, int(ADC_y.shape[-1] / 2)]).T, 1))
                self.ADC_y_plot.set_clim([ADC_y_min, ADC_y_max])
                self.ADC_y_plot_sag.set_clim([ADC_y_min, ADC_y_max])
                self.ADC_y_plot_cor.set_clim([ADC_y_min, ADC_y_max])

                self.ADC_xz_plot.set_data((ADC_xz[int(self.NSlice / 2), ...]))
                self.ADC_xz_plot_cor.set_data(
                    (ADC_xz[:, int(ADC_xz.shape[1] / 2), ...]))
                self.ADC_xz_plot_sag.set_data(
                    np.flip((ADC_xz[:, :, int(ADC_xz.shape[-1] / 2)]).T, 1))
                self.ADC_xz_plot.set_clim([ADC_xz_min, ADC_xz_max])
                self.ADC_xz_plot_sag.set_clim([ADC_xz_min, ADC_xz_max])
                self.ADC_xz_plot_cor.set_clim([ADC_xz_min, ADC_xz_max])

                self.ADC_z_plot.set_data((ADC_z[int(self.NSlice / 2), ...]))
                self.ADC_z_plot_cor.set_data(
                    (ADC_z[:, int(ADC_z.shape[1] / 2), ...]))
                self.ADC_z_plot_sag.set_data(
                    np.flip((ADC_z[:, :, int(ADC_z.shape[-1] / 2)]).T, 1))
                self.ADC_z_plot.set_clim([ADC_z_min, ADC_z_max])
                self.ADC_z_plot_sag.set_clim([ADC_z_min, ADC_z_max])
                self.ADC_z_plot_cor.set_clim([ADC_z_min, ADC_z_max])

                self.ADC_yz_plot.set_data((ADC_yz[int(self.NSlice / 2), ...]))
                self.ADC_yz_plot_cor.set_data(
                    (ADC_yz[:, int(ADC_yz.shape[1] / 2), ...]))
                self.ADC_yz_plot_sag.set_data(
                    np.flip((ADC_yz[:, :, int(ADC_yz.shape[-1] / 2)]).T, 1))
                self.ADC_yz_plot.set_clim([ADC_yz_min, ADC_yz_max])
                self.ADC_yz_plot_sag.set_clim([ADC_yz_min, ADC_yz_max])
                self.ADC_yz_plot_cor.set_clim([ADC_yz_min, ADC_yz_max])

                self.figure.canvas.draw_idle()

                plt.draw()
                plt.pause(1e-10)

    def _set_init_scales(self, images):
        test_M0 = self.b0
        ADC = 1 * np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)

        x = np.array(
                [
                    test_M0 / self.uk_scale[0],
                    ADC,
                    0 * ADC,
                    ADC,
                    0 * ADC,
                    ADC,
                    0 * ADC],
                dtype=DTYPE)
        return x
