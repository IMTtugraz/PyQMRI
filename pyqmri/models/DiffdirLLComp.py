#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyqmri.models.template import BaseModel, constraints, DTYPE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
unknowns_TGV = 14
unknowns_H1 = 0


class Model(BaseModel):
    def __init__(self, par, images):
        super().__init__(par)
        self.images = images
        self.NSlice = par['NSlice']

        self.figuref = None

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
            print("No b0 image provided")
            self.b0 =  None

        self.constraints.append(
            constraints(
                0 / self.uk_scale[0],
                10 / self.uk_scale[0],
                False))

        self.constraints.append(
            constraints(
                (-1e0 / self.uk_scale[1]),
                (1e0 / self.uk_scale[1]),
                True))
        self.constraints.append(
            constraints(
                (-1e0 / self.uk_scale[2]),
                (1e0 / self.uk_scale[2]),
                True))
        self.constraints.append(
            constraints(
                (-1e0 / self.uk_scale[3]),
                (1e0 / self.uk_scale[3]),
                True))
        self.constraints.append(
            constraints(
                (-1e0 / self.uk_scale[4]),
                (1e0 / self.uk_scale[4]),
                True))
        self.constraints.append(
            constraints(
                (-1e0 / self.uk_scale[5]),
                (1e0 / self.uk_scale[5]),
                True))
        self.constraints.append(
            constraints(
                (-1e0 / self.uk_scale[6]),
                (1e0 / self.uk_scale[6]),
                True))

        self.constraints.append(
            constraints(
                0 / self.uk_scale[0],
                1 / self.uk_scale[0],
                False))

        self.constraints.append(
            constraints(
                (-5e0 / self.uk_scale[1]),
                (5e0 / self.uk_scale[1]),
                True))
        self.constraints.append(
            constraints(
                (-5e0 / self.uk_scale[2]),
                (5e0 / self.uk_scale[2]),
                True))
        self.constraints.append(
            constraints(
                (-5e0 / self.uk_scale[3]),
                (5e0 / self.uk_scale[3]),
                True))
        self.constraints.append(
            constraints(
                (-5e0 / self.uk_scale[4]),
                (5e0 / self.uk_scale[4]),
                True))
        self.constraints.append(
            constraints(
                (-5e0 / self.uk_scale[5]),
                (5e0 / self.uk_scale[5]),
                True))
        self.constraints.append(
            constraints(
                (-5e0 / self.uk_scale[6]),
                (5e0 / self.uk_scale[6]),
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

        f = x[7, ...] * self.uk_scale[7]

        fADC_x = (np.real(x[8, ...]**2) * self.uk_scale[8]**2)
        fADC_xy = (np.real(x[9, ...] * self.uk_scale[9] *
                           x[8, ...] * self.uk_scale[8]))
        fADC_y = (np.real(x[9, ...]**2 * self.uk_scale[9]**2 +
                          x[10, ...]**2 * self.uk_scale[10]**2))
        fADC_xz = (np.real(x[11, ...] * self.uk_scale[11] *
                           x[8, ...] * self.uk_scale[8]))
        fADC_z = (np.real(x[11, ...]**2 * self.uk_scale[11]**2 +
                          x[12, ...]**2 * self.uk_scale[12]**2 +
                          x[13, ...]**2 * self.uk_scale[13]**2))
        fADC_yz = (np.real(x[9, ...] * self.uk_scale[9] *
                           x[11, ...] * self.uk_scale[11] +
                           x[13, ...] * self.uk_scale[13] *
                           x[10, ...] * self.uk_scale[10]))

        return np.array((M0, ADC_x, ADC_xy, ADC_y, ADC_xz, ADC_z, ADC_yz, f,
                         fADC_x, fADC_xy, fADC_y, fADC_xz, fADC_z, fADC_yz))

    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_forward_3D(self, x):
        ADCs = x[1, ...]**2 * self.uk_scale[1]**2 * self.dir[..., 0]**2 + \
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

        ADCf = x[8, ...]**2 * self.uk_scale[8]**2 * self.dir[..., 0]**2 + \
              (x[9, ...]**2 * self.uk_scale[9]**2 +
               x[10, ...]**2 * self.uk_scale[10]**2) * self.dir[..., 1]**2 + \
              (x[11, ...]**2 * self.uk_scale[11]**2 +
               x[12, ...]**2 * self.uk_scale[12]**2 +
               x[13, ...]**2 * self.uk_scale[13]**2) * self.dir[..., 2]**2 +\
              2 * (x[9, ...] * self.uk_scale[9] *
                   x[8, ...] * self.uk_scale[8]) * \
              self.dir[..., 0] * self.dir[..., 1] + \
              2 * (x[11, ...] * self.uk_scale[11] *
                   x[8, ...] * self.uk_scale[8]) *\
              self.dir[..., 0] * self.dir[..., 2] +\
              2 * (x[9, ...] * self.uk_scale[9] *
                   x[11, ...] * self.uk_scale[11] +
                   x[13, ...] * self.uk_scale[13] *
                   x[10, ...] * self.uk_scale[10]) * \
              self.dir[..., 1] * self.dir[..., 2]

        S = x[0] * self.uk_scale[0] * (
              x[7]*self.uk_scale[7] * np.exp(-self.b * ADCf) +
              (-x[7]*self.uk_scale[7] + 1) * np.exp(-self.b * ADCs))

        S *= self.phase
        S[~np.isfinite(S)] = 0
        return S.astype(DTYPE)

    def _execute_gradient_3D(self, x):
        ADCs = x[1, ...]**2 * self.uk_scale[1]**2 * self.dir[..., 0]**2 + \
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

        ADCf = x[8, ...]**2 * self.uk_scale[8]**2 * self.dir[..., 0]**2 + \
              (x[9, ...]**2 * self.uk_scale[9]**2 +
               x[10, ...]**2 * self.uk_scale[10]**2) * self.dir[..., 1]**2 + \
              (x[11, ...]**2 * self.uk_scale[11]**2 +
               x[12, ...]**2 * self.uk_scale[12]**2 +
               x[13, ...]**2 * self.uk_scale[13]**2) * self.dir[..., 2]**2 +\
              2 * (x[9, ...] * self.uk_scale[9] *
                   x[8, ...] * self.uk_scale[8]) * \
              self.dir[..., 0] * self.dir[..., 1] + \
              2 * (x[11, ...] * self.uk_scale[11] *
                   x[8, ...] * self.uk_scale[8]) *\
              self.dir[..., 0] * self.dir[..., 2] +\
              2 * (x[9, ...] * self.uk_scale[9] *
                   x[11, ...] * self.uk_scale[11] +
                   x[13, ...] * self.uk_scale[13] *
                   x[10, ...] * self.uk_scale[10]) * \
              self.dir[..., 1] * self.dir[..., 2]

        expADCf = np.exp(-self.b * ADCf)
        expADCs = np.exp(-self.b * ADCs)

        grad_M0 = self.uk_scale[0] * (
            x[7]*self.uk_scale[7] * expADCf +
            (-x[7]*self.uk_scale[7] + 1) * expADCs)



        grad_ADCs_x = -x[0] * self.uk_scale[0]*self.b*(1 - x[7] * self.uk_scale[7])*(2*x[1]*self.uk_scale[1]**2*self.dir[..., 0]**2 + 2*self.uk_scale[1]*x[2]*self.uk_scale[2]*self.dir[..., 0]*self.dir[..., 1] + 2*self.uk_scale[1]*x[4]*self.uk_scale[4]*self.dir[..., 0]*self.dir[..., 2])*expADCs
        grad_ADCs_xy = -x[0] * self.uk_scale[0]*self.b*(1 - x[7] * self.uk_scale[7])*(2*x[1]*self.uk_scale[1]*self.uk_scale[2]*self.dir[..., 0]*self.dir[..., 1] + 2*x[2]*self.uk_scale[2]**2*self.dir[..., 1]**2 + 2*self.uk_scale[2]*x[4]*self.uk_scale[4]*self.dir[..., 1]*self.dir[..., 2])*expADCs

        grad_ADCs_y = -x[0] * self.uk_scale[0]*self.b*(1 - x[7] * self.uk_scale[7])*(2*x[3]*self.uk_scale[3]**2*self.dir[..., 1]**2 + 2*self.uk_scale[3]*x[6]*self.uk_scale[6]*self.dir[..., 1]*self.dir[..., 2])*expADCs

        grad_ADCs_xz = -x[0] * self.uk_scale[0]*self.b*(1 - x[7] * self.uk_scale[7])*(2*x[1]*self.uk_scale[1]*self.uk_scale[4]*self.dir[..., 0]*self.dir[..., 2] + 2*x[2]*self.uk_scale[2]*self.uk_scale[4]*self.dir[..., 1]*self.dir[..., 2] + 2*x[4]*self.uk_scale[4]**2*self.dir[..., 2]**2)*expADCs

        grad_ADCs_z = -2*x[5]*self.uk_scale[5]**2*x[0] * self.uk_scale[0]*self.b*self.dir[..., 2]**2*(1 - x[7] * self.uk_scale[7])*expADCs

        grad_ADCs_yz = -x[0] * self.uk_scale[0]*self.b*(1 - x[7] * self.uk_scale[7])*(2*x[3]*self.uk_scale[3]*self.uk_scale[6]*self.dir[..., 1]*self.dir[..., 2] + 2*x[6]*self.uk_scale[6]**2*self.dir[..., 2]**2)*expADCs

        grad_f = x[0]*self.uk_scale[0]*self.uk_scale[7]*(
            -expADCs + expADCf)

        grad_ADCf_x = -x[0] * self.uk_scale[0]*self.b*x[7] * self.uk_scale[7]*(2*x[8]*self.uk_scale[8]**2*self.dir[..., 0]**2 + 2*self.uk_scale[8]*x[9]*self.uk_scale[9]*self.dir[..., 0]*self.dir[..., 1] + 2*self.uk_scale[8]*x[11]*self.uk_scale[11]*self.dir[..., 0]*self.dir[..., 2])*expADCf
        grad_ADCf_xy = -x[0] * self.uk_scale[0]*self.b*x[7] * self.uk_scale[7]*(2*x[8]*self.uk_scale[8]*self.uk_scale[9]*self.dir[..., 0]*self.dir[..., 1] + 2*x[9]*self.uk_scale[9]**2*self.dir[..., 1]**2 + 2*self.uk_scale[9]*x[11]*self.uk_scale[11]*self.dir[..., 1]*self.dir[..., 2])*expADCf

        grad_ADCf_y = -x[0] * self.uk_scale[0]*self.b*x[7] * self.uk_scale[7]*(2*x[10]*self.uk_scale[10]**2*self.dir[..., 1]**2 + 2*self.uk_scale[10]*x[13]*self.uk_scale[13]*self.dir[..., 1]*self.dir[..., 2])*expADCf
        grad_ADCf_xz = -x[0] * self.uk_scale[0]*self.b*x[7] * self.uk_scale[7]*(2*x[8]*self.uk_scale[8]*self.uk_scale[11]*self.dir[..., 0]*self.dir[..., 2] + 2*x[9]*self.uk_scale[9]*self.uk_scale[11]*self.dir[..., 1]*self.dir[..., 2] + 2*x[11]*self.uk_scale[11]**2*self.dir[..., 2]**2)*expADCf

        grad_ADCf_z = -2*x[12]*self.uk_scale[12]**2*x[0] * self.uk_scale[0]*self.b*self.dir[..., 2]**2*x[7] * self.uk_scale[7]*expADCf
        grad_ADCf_yz = -x[0] * self.uk_scale[0]*self.b*x[7] * self.uk_scale[7]*(2*x[10]*self.uk_scale[10]*self.uk_scale[13]*self.dir[..., 1]*self.dir[..., 2] + 2*x[13]*self.uk_scale[13]**2*self.dir[..., 2]**2)*expADCf

        grad = np.array(
            [grad_M0,
             grad_ADCs_x,
             grad_ADCs_xy,
             grad_ADCs_y,
             grad_ADCs_xz,
             grad_ADCs_z,
             grad_ADCs_yz,
             grad_f,
             grad_ADCf_x,
             grad_ADCf_xy,
             grad_ADCf_y,
             grad_ADCf_xz,
             grad_ADCf_z,
             grad_ADCf_yz], dtype=DTYPE)*self.phase
        grad[~np.isfinite(grad)] = 0
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...]) * self.uk_scale[0]
        ADCs_x = (np.real(x[1, ...]**2 * self.uk_scale[1]**2))
        ADCs_xy = (np.real(x[2, ...] * self.uk_scale[2] *
                           x[1, ...] * self.uk_scale[1]))
        M0_min = M0.min()
        M0_max = M0.max()
        ADCs_x_min = ADCs_x.min()
        ADCs_x_max = ADCs_x.max()
        ADCs_xy_min = ADCs_xy.min()
        ADCs_xy_max = ADCs_xy.max()

        ADCs_y = (np.real(x[2, ...]**2 * self.uk_scale[2]**2 +
                          x[3, ...]**2 * self.uk_scale[3]**2))
        ADCs_xz = (np.real(x[4, ...] * self.uk_scale[4] *
                           x[1, ...] * self.uk_scale[1]))
        ADCs_y_min = ADCs_y.min()
        ADCs_y_max = ADCs_y.max()
        ADCs_xz_min = ADCs_xz.min()
        ADCs_xz_max = ADCs_xz.max()

        ADCs_z = (np.real(x[4, ...]**2 * self.uk_scale[4]**2 +
                          x[5, ...]**2 * self.uk_scale[5]**2 +
                          x[6, ...]**2 * self.uk_scale[6]**2))
        ADCs_yz = (np.real(x[2, ...] * self.uk_scale[2] *
                           x[4, ...] * self.uk_scale[4] +
                           x[6, ...] * self.uk_scale[6] *
                           x[3, ...] * self.uk_scale[3]))
        ADCs_z_min = ADCs_z.min()
        ADCs_z_max = ADCs_z.max()
        ADCs_yz_min = ADCs_yz.min()
        ADCs_yz_max = ADCs_yz.max()

        f = np.abs(x[7, ...]) * self.uk_scale[7]
        ADCf_x = (np.real(x[8, ...]**2 * self.uk_scale[8]**2))
        ADCf_xy = (np.real(x[9, ...] * self.uk_scale[9] *
                           x[8, ...] * self.uk_scale[8]))
        f_min = f.min()
        f_max = f.max()
        ADCf_x_min = ADCf_x.min()
        ADCf_x_max = ADCf_x.max()
        ADCf_xy_min = ADCf_xy.min()
        ADCf_xy_max = ADCf_xy.max()

        ADCf_y = (np.real(x[9, ...]**2 * self.uk_scale[9]**2 +
                          x[10, ...]**2 * self.uk_scale[10]**2))
        ADCf_xz = (np.real(x[11, ...] * self.uk_scale[11] *
                           x[8, ...] * self.uk_scale[8]))
        ADCf_y_min = ADCf_y.min()
        ADCf_y_max = ADCf_y.max()
        ADCf_xz_min = ADCf_xz.min()
        ADCf_xz_max = ADCf_xz.max()

        ADCf_z = (np.real(x[11, ...]**2 * self.uk_scale[11]**2 +
                          x[12, ...]**2 * self.uk_scale[12]**2 +
                          x[13, ...]**2 * self.uk_scale[13]**2))
        ADCf_yz = (np.real(x[9, ...] * self.uk_scale[9] *
                           x[11, ...] * self.uk_scale[11] +
                           x[13, ...] * self.uk_scale[13] *
                           x[10, ...] * self.uk_scale[10]))
        ADCf_z_min = ADCf_z.min()
        ADCf_z_max = ADCf_z.max()
        ADCf_yz_min = ADCf_yz.min()
        ADCf_yz_max = ADCf_yz.max()

        if dim_2D:
            pass
        else:
            [z, y, x] = M0.shape
            self.ax = []
            self.axf = []
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

                self.ADCs_x_plot = self.ax[3].imshow(
                    (ADCs_x[int(self.NSlice / 2), ...]))
                self.ADCs_x_plot_cor = self.ax[13].imshow(
                    (ADCs_x[:, int(ADCs_x.shape[1] / 2), ...]))
                self.ADCs_x_plot_sag = self.ax[4].imshow(
                    np.flip((ADCs_x[:, :, int(ADCs_x.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('ADCs_x', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[13].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 5])
                cbar = self.figure.colorbar(self.ADCs_x_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCs_xy_plot = self.ax[7].imshow(
                    (ADCs_xy[int(self.NSlice / 2), ...]))
                self.ADCs_xy_plot_cor = self.ax[17].imshow(
                    (ADCs_xy[:, int(ADCs_x.shape[1] / 2), ...]))
                self.ADCs_xy_plot_sag = self.ax[8].imshow(
                    np.flip((ADCs_xy[:, :, int(ADCs_x.shape[-1] / 2)]).T, 1))
                self.ax[7].set_title('ADCs_xy', color='white')
                self.ax[7].set_anchor('SE')
                self.ax[8].set_anchor('SW')
                self.ax[17].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 9])
                cbar = self.figure.colorbar(self.ADCs_xy_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCs_y_plot = self.ax[23].imshow(
                    (ADCs_y[int(self.NSlice / 2), ...]))
                self.ADCs_y_plot_cor = self.ax[33].imshow(
                    (ADCs_y[:, int(ADCs_y.shape[1] / 2), ...]))
                self.ADCs_y_plot_sag = self.ax[24].imshow(
                    np.flip((ADCs_y[:, :, int(ADCs_y.shape[-1] / 2)]).T, 1))
                self.ax[23].set_title('ADCs_y', color='white')
                self.ax[23].set_anchor('SE')
                self.ax[24].set_anchor('SW')
                self.ax[33].set_anchor('NE')
                cax = plt.subplot(self.gs[2:4, 5])
                cbar = self.figure.colorbar(self.ADCs_y_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCs_xz_plot = self.ax[27].imshow(
                    (ADCs_xz[int(self.NSlice / 2), ...]))
                self.ADCs_xz_plot_cor = self.ax[37].imshow(
                    (ADCs_xz[:, int(ADCs_y.shape[1] / 2), ...]))
                self.ADCs_xz_plot_sag = self.ax[28].imshow(
                    np.flip((ADCs_xz[:, :, int(ADCs_y.shape[-1] / 2)]).T, 1))
                self.ax[27].set_title('ADCs_xz', color='white')
                self.ax[27].set_anchor('SE')
                self.ax[28].set_anchor('SW')
                self.ax[37].set_anchor('NE')
                cax = plt.subplot(self.gs[2:4, 9])
                cbar = self.figure.colorbar(self.ADCs_xz_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCs_z_plot = self.ax[43].imshow(
                    (ADCs_z[int(self.NSlice / 2), ...]))
                self.ADCs_z_plot_cor = self.ax[53].imshow(
                    (ADCs_z[:, int(ADCs_z.shape[1] / 2), ...]))
                self.ADCs_z_plot_sag = self.ax[44].imshow(
                    np.flip((ADCs_z[:, :, int(ADCs_z.shape[-1] / 2)]).T, 1))
                self.ax[43].set_title('ADCs_z', color='white')
                self.ax[43].set_anchor('SE')
                self.ax[44].set_anchor('SW')
                self.ax[53].set_anchor('NE')
                cax = plt.subplot(self.gs[4:6, 5])
                cbar = self.figure.colorbar(self.ADCs_z_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCs_yz_plot = self.ax[47].imshow(
                    (ADCs_yz[int(self.NSlice / 2), ...]))
                self.ADCs_yz_plot_cor = self.ax[57].imshow(
                    (ADCs_yz[:, int(ADCs_z.shape[1] / 2), ...]))
                self.ADCs_yz_plot_sag = self.ax[48].imshow(
                    np.flip((ADCs_yz[:, :, int(ADCs_z.shape[-1] / 2)]).T, 1))
                self.ax[47].set_title('ADCs_yz', color='white')
                self.ax[47].set_anchor('SE')
                self.ax[48].set_anchor('SW')
                self.ax[57].set_anchor('NE')
                cax = plt.subplot(self.gs[4:6, 9])
                cbar = self.figure.colorbar(self.ADCs_yz_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                plt.draw()
                plt.pause(1e-10)
                self.figure.canvas.draw_idle()

            if not self.figuref:
                plt.ion()
                self.figuref = plt.figure(figsize=(12, 6))
                self.figuref.subplots_adjust(hspace=0, wspace=0)
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
                self.figuref.tight_layout()
                self.figuref.patch.set_facecolor(plt.cm.viridis.colors[0])
                for grid in self.gs:
                    self.axf.append(plt.subplot(grid))
                    self.axf[-1].axis('off')

                self.f_plot = self.axf[1].imshow(
                    (f[int(self.NSlice / 2), ...]))
                self.f_plot_cor = self.axf[11].imshow(
                    (f[:, int(f.shape[1] / 2), ...]))
                self.f_plot_sag = self.axf[2].imshow(
                    np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
                self.axf[1].set_title('f in a.u.', color='white')
                self.axf[1].set_anchor('SE')
                self.axf[2].set_anchor('SW')
                self.ax[11].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 0])
                cbar = self.figure.colorbar(self.f_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCf_x_plot = self.axf[3].imshow(
                    (ADCf_x[int(self.NSlice / 2), ...]))
                self.ADCf_x_plot_cor = self.axf[13].imshow(
                    (ADCf_x[:, int(ADCf_x.shape[1] / 2), ...]))
                self.ADCf_x_plot_sag = self.axf[4].imshow(
                    np.flip((ADCf_x[:, :, int(ADCf_x.shape[-1] / 2)]).T, 1))
                self.axf[3].set_title('ADCf_x', color='white')
                self.axf[3].set_anchor('SE')
                self.axf[4].set_anchor('SW')
                self.axf[13].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 5])
                cbar = self.figure.colorbar(self.ADCf_x_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCf_xy_plot = self.axf[7].imshow(
                    (ADCf_xy[int(self.NSlice / 2), ...]))
                self.ADCf_xy_plot_cor = self.axf[17].imshow(
                    (ADCf_xy[:, int(ADCf_x.shape[1] / 2), ...]))
                self.ADCf_xy_plot_sag = self.axf[8].imshow(
                    np.flip((ADCf_xy[:, :, int(ADCf_x.shape[-1] / 2)]).T, 1))
                self.axf[7].set_title('ADCf_xy', color='white')
                self.axf[7].set_anchor('SE')
                self.axf[8].set_anchor('SW')
                self.axf[17].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 9])
                cbar = self.figure.colorbar(self.ADCf_xy_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCf_y_plot = self.axf[23].imshow(
                    (ADCf_y[int(self.NSlice / 2), ...]))
                self.ADCf_y_plot_cor = self.axf[33].imshow(
                    (ADCf_y[:, int(ADCf_y.shape[1] / 2), ...]))
                self.ADCf_y_plot_sag = self.axf[24].imshow(
                    np.flip((ADCf_y[:, :, int(ADCf_y.shape[-1] / 2)]).T, 1))
                self.axf[23].set_title('ADCf_y', color='white')
                self.axf[23].set_anchor('SE')
                self.axf[24].set_anchor('SW')
                self.axf[33].set_anchor('NE')
                cax = plt.subplot(self.gs[2:4, 5])
                cbar = self.figure.colorbar(self.ADCf_y_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCf_xz_plot = self.axf[27].imshow(
                    (ADCf_xz[int(self.NSlice / 2), ...]))
                self.ADCf_xz_plot_cor = self.axf[37].imshow(
                    (ADCf_xz[:, int(ADCf_y.shape[1] / 2), ...]))
                self.ADCf_xz_plot_sag = self.axf[28].imshow(
                    np.flip((ADCf_xz[:, :, int(ADCf_y.shape[-1] / 2)]).T, 1))
                self.axf[27].set_title('ADCf_xz', color='white')
                self.axf[27].set_anchor('SE')
                self.axf[28].set_anchor('SW')
                self.axf[37].set_anchor('NE')
                cax = plt.subplot(self.gs[2:4, 9])
                cbar = self.figure.colorbar(self.ADCf_xz_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCf_z_plot = self.axf[43].imshow(
                    (ADCf_z[int(self.NSlice / 2), ...]))
                self.ADCf_z_plot_cor = self.axf[53].imshow(
                    (ADCf_z[:, int(ADCf_z.shape[1] / 2), ...]))
                self.ADCf_z_plot_sag = self.axf[44].imshow(
                    np.flip((ADCf_z[:, :, int(ADCf_z.shape[-1] / 2)]).T, 1))
                self.axf[43].set_title('ADCf_z', color='white')
                self.axf[43].set_anchor('SE')
                self.axf[44].set_anchor('SW')
                self.axf[53].set_anchor('NE')
                cax = plt.subplot(self.gs[4:6, 5])
                cbar = self.figure.colorbar(self.ADCf_z_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADCf_yz_plot = self.axf[47].imshow(
                    (ADCf_yz[int(self.NSlice / 2), ...]))
                self.ADCf_yz_plot_cor = self.axf[57].imshow(
                    (ADCf_yz[:, int(ADCf_z.shape[1] / 2), ...]))
                self.ADCf_yz_plot_sag = self.axf[48].imshow(
                    np.flip((ADCf_yz[:, :, int(ADCf_z.shape[-1] / 2)]).T, 1))
                self.axf[47].set_title('ADCf_yz', color='white')
                self.axf[47].set_anchor('SE')
                self.axf[48].set_anchor('SW')
                self.axf[57].set_anchor('NE')
                cax = plt.subplot(self.gs[4:6, 9])
                cbar = self.figure.colorbar(self.ADCf_yz_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                plt.draw()
                plt.pause(1e-10)
                self.figure.canvas.draw_idle()

            else:
                self.M0_plot.set_data((M0[int(self.NSlice / 2), ...]))
                self.M0_plot_cor.set_data((M0[:, int(M0.shape[1] / 2), ...]))
                self.M0_plot_sag.set_data(
                    np.flip((M0[:, :, int(M0.shape[-1] / 2)]).T, 1))
                self.M0_plot.set_clim([M0_min, M0_max])
                self.M0_plot_cor.set_clim([M0_min, M0_max])
                self.M0_plot_sag.set_clim([M0_min, M0_max])

                self.ADCs_x_plot.set_data((ADCs_x[int(self.NSlice / 2), ...]))
                self.ADCs_x_plot_cor.set_data(
                    (ADCs_x[:, int(ADCs_x.shape[1] / 2), ...]))
                self.ADCs_x_plot_sag.set_data(
                    np.flip((ADCs_x[:, :, int(ADCs_x.shape[-1] / 2)]).T, 1))
                self.ADCs_x_plot.set_clim([ADCs_x_min, ADCs_x_max])
                self.ADCs_x_plot_sag.set_clim([ADCs_x_min, ADCs_x_max])
                self.ADCs_x_plot_cor.set_clim([ADCs_x_min, ADCs_x_max])

                self.ADCs_xy_plot.set_data(
                    (ADCs_xy[int(self.NSlice / 2), ...]))
                self.ADCs_xy_plot_cor.set_data(
                    (ADCs_xy[:, int(ADCs_xy.shape[1] / 2), ...]))
                self.ADCs_xy_plot_sag.set_data(
                    np.flip((ADCs_xy[:, :, int(ADCs_xy.shape[-1] / 2)]).T, 1))
                self.ADCs_xy_plot.set_clim([ADCs_xy_min, ADCs_xy_max])
                self.ADCs_xy_plot_sag.set_clim([ADCs_xy_min, ADCs_xy_max])
                self.ADCs_xy_plot_cor.set_clim([ADCs_xy_min, ADCs_xy_max])

                self.ADCs_y_plot.set_data((ADCs_y[int(self.NSlice / 2), ...]))
                self.ADCs_y_plot_cor.set_data(
                    (ADCs_y[:, int(ADCs_y.shape[1] / 2), ...]))
                self.ADCs_y_plot_sag.set_data(
                    np.flip((ADCs_y[:, :, int(ADCs_y.shape[-1] / 2)]).T, 1))
                self.ADCs_y_plot.set_clim([ADCs_y_min, ADCs_y_max])
                self.ADCs_y_plot_sag.set_clim([ADCs_y_min, ADCs_y_max])
                self.ADCs_y_plot_cor.set_clim([ADCs_y_min, ADCs_y_max])

                self.ADCs_xz_plot.set_data(
                    (ADCs_xz[int(self.NSlice / 2), ...]))
                self.ADCs_xz_plot_cor.set_data(
                    (ADCs_xz[:, int(ADCs_xz.shape[1] / 2), ...]))
                self.ADCs_xz_plot_sag.set_data(
                    np.flip((ADCs_xz[:, :, int(ADCs_xz.shape[-1] / 2)]).T, 1))
                self.ADCs_xz_plot.set_clim([ADCs_xz_min, ADCs_xz_max])
                self.ADCs_xz_plot_sag.set_clim([ADCs_xz_min, ADCs_xz_max])
                self.ADCs_xz_plot_cor.set_clim([ADCs_xz_min, ADCs_xz_max])

                self.ADCs_z_plot.set_data((ADCs_z[int(self.NSlice / 2), ...]))
                self.ADCs_z_plot_cor.set_data(
                    (ADCs_z[:, int(ADCs_z.shape[1] / 2), ...]))
                self.ADCs_z_plot_sag.set_data(
                    np.flip((ADCs_z[:, :, int(ADCs_z.shape[-1] / 2)]).T, 1))
                self.ADCs_z_plot.set_clim([ADCs_z_min, ADCs_z_max])
                self.ADCs_z_plot_sag.set_clim([ADCs_z_min, ADCs_z_max])
                self.ADCs_z_plot_cor.set_clim([ADCs_z_min, ADCs_z_max])

                self.ADCs_yz_plot.set_data(
                    (ADCs_yz[int(self.NSlice / 2), ...]))
                self.ADCs_yz_plot_cor.set_data(
                    (ADCs_yz[:, int(ADCs_yz.shape[1] / 2), ...]))
                self.ADCs_yz_plot_sag.set_data(
                    np.flip((ADCs_yz[:, :, int(ADCs_yz.shape[-1] / 2)]).T, 1))
                self.ADCs_yz_plot.set_clim([ADCs_yz_min, ADCs_yz_max])
                self.ADCs_yz_plot_sag.set_clim([ADCs_yz_min, ADCs_yz_max])
                self.ADCs_yz_plot_cor.set_clim([ADCs_yz_min, ADCs_yz_max])

                self.f_plot.set_data(
                    (f[int(self.NSlice / 2), ...]))
                self.f_plot_cor.set_data((f[:, int(f.shape[1] / 2), ...]))
                self.f_plot_sag.set_data(
                    np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
                self.f_plot.set_clim([f_min, f_max])
                self.f_plot_cor.set_clim([f_min, f_max])
                self.f_plot_sag.set_clim([f_min, f_max])

                self.ADCf_x_plot.set_data((ADCf_x[int(self.NSlice / 2), ...]))
                self.ADCf_x_plot_cor.set_data(
                    (ADCf_x[:, int(ADCf_x.shape[1] / 2), ...]))
                self.ADCf_x_plot_sag.set_data(
                    np.flip((ADCf_x[:, :, int(ADCf_x.shape[-1] / 2)]).T, 1))
                self.ADCf_x_plot.set_clim([ADCf_x_min, ADCf_x_max])
                self.ADCf_x_plot_sag.set_clim([ADCf_x_min, ADCf_x_max])
                self.ADCf_x_plot_cor.set_clim([ADCf_x_min, ADCf_x_max])

                self.ADCf_xy_plot.set_data(
                    (ADCf_xy[int(self.NSlice / 2), ...]))
                self.ADCf_xy_plot_cor.set_data(
                    (ADCf_xy[:, int(ADCf_xy.shape[1] / 2), ...]))
                self.ADCf_xy_plot_sag.set_data(
                    np.flip((ADCf_xy[:, :, int(ADCf_xy.shape[-1] / 2)]).T, 1))
                self.ADCf_xy_plot.set_clim([ADCf_xy_min, ADCf_xy_max])
                self.ADCf_xy_plot_sag.set_clim([ADCf_xy_min, ADCf_xy_max])
                self.ADCf_xy_plot_cor.set_clim([ADCf_xy_min, ADCf_xy_max])

                self.ADCf_y_plot.set_data((ADCf_y[int(self.NSlice / 2), ...]))
                self.ADCf_y_plot_cor.set_data(
                    (ADCf_y[:, int(ADCf_y.shape[1] / 2), ...]))
                self.ADCf_y_plot_sag.set_data(
                    np.flip((ADCf_y[:, :, int(ADCf_y.shape[-1] / 2)]).T, 1))
                self.ADCf_y_plot.set_clim([ADCf_y_min, ADCf_y_max])
                self.ADCf_y_plot_sag.set_clim([ADCf_y_min, ADCf_y_max])
                self.ADCf_y_plot_cor.set_clim([ADCf_y_min, ADCf_y_max])

                self.ADCf_xz_plot.set_data(
                    (ADCf_xz[int(self.NSlice / 2), ...]))
                self.ADCf_xz_plot_cor.set_data(
                    (ADCf_xz[:, int(ADCf_xz.shape[1] / 2), ...]))
                self.ADCf_xz_plot_sag.set_data(
                    np.flip((ADCf_xz[:, :, int(ADCf_xz.shape[-1] / 2)]).T, 1))
                self.ADCf_xz_plot.set_clim([ADCf_xz_min, ADCf_xz_max])
                self.ADCf_xz_plot_sag.set_clim([ADCf_xz_min, ADCf_xz_max])
                self.ADCf_xz_plot_cor.set_clim([ADCf_xz_min, ADCf_xz_max])

                self.ADCf_z_plot.set_data((ADCf_z[int(self.NSlice / 2), ...]))
                self.ADCf_z_plot_cor.set_data(
                    (ADCf_z[:, int(ADCf_z.shape[1] / 2), ...]))
                self.ADCf_z_plot_sag.set_data(
                    np.flip((ADCf_z[:, :, int(ADCf_z.shape[-1] / 2)]).T, 1))
                self.ADCf_z_plot.set_clim([ADCf_z_min, ADCf_z_max])
                self.ADCf_z_plot_sag.set_clim([ADCf_z_min, ADCf_z_max])
                self.ADCf_z_plot_cor.set_clim([ADCf_z_min, ADCf_z_max])

                self.ADCf_yz_plot.set_data(
                    (ADCf_yz[int(self.NSlice / 2), ...]))
                self.ADCf_yz_plot_cor.set_data(
                    (ADCf_yz[:, int(ADCf_yz.shape[1] / 2), ...]))
                self.ADCf_yz_plot_sag.set_data(
                    np.flip((ADCf_yz[:, :, int(ADCf_yz.shape[-1] / 2)]).T, 1))
                self.ADCf_yz_plot.set_clim([ADCf_yz_min, ADCf_yz_max])
                self.ADCf_yz_plot_sag.set_clim([ADCf_yz_min, ADCf_yz_max])
                self.ADCf_yz_plot_cor.set_clim([ADCf_yz_min, ADCf_yz_max])

                self.figure.canvas.draw_idle()

                plt.draw()
                plt.pause(1e-10)

    def computeInitialGuess(self, images):
        self.phase = np.exp(1j*(np.angle(images)-np.angle(images[0])))
        self.guess = self._set_init_scales(images)
        if self.b0 is not None:
            test_M0 = self.b0
        else:
            test_M0 = images[0]
        ADC = 1 * np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)

        x = np.array(
                [
                    test_M0 / self.uk_scale[0],
                    ADC/10,
                    0 * ADC,
                    ADC/10,
                    0 * ADC,
                    ADC/10,
                    0 * ADC,
                    0.6 * np.ones_like(test_M0),
                    ADC,
                    0 * ADC,
                    ADC,
                    0 * ADC,
                    ADC,
                    0 * ADC],
                dtype=DTYPE)
        return x
