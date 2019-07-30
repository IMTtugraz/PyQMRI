#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyqmri.models.template import BaseModel, constraints, DTYPE
import numexpr as ne
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

unknowns_TGV = 2
unknowns_H1 = 0


class Model(BaseModel):
    def __init__(self, par, images):
        super().__init__(par)

        self.TR = par['time_per_slice'] - \
            (par['tau'] * par['Nproj_measured'] + par['gradient_delay'])
        self.fa = par["flip_angle(s)"] * np.pi / 180

        self.tau = par["tau"]
        self.td = par["gradient_delay"]
        self.NLL = par["NScan"]
        self.Nproj = par["Nproj"]
        self.Nproj_measured = par["Nproj_measured"]

        phi_corr = np.zeros_like(par["fa_corr"], dtype=DTYPE)
        phi_corr = np.real(
            self.fa) * np.real(par["fa_corr"]) + 1j *\
            np.imag(self.fa) * np.imag(par["fa_corr"])

        self.sin_phi = np.sin(phi_corr)
        self.cos_phi = np.cos(phi_corr)

        for j in range(unknowns_TGV + unknowns_H1):
            self.uk_scale.append(1)

        self.guess = self._set_init_scales()

        self.constraints.append(constraints(0, 300, False))
        self.constraints.append(constraints(
            np.exp(-self.scale / 10) / self.uk_scale[1],
            np.exp(-self.scale / 5500) / self.uk_scale[1], True))

    def rescale(self, x):
        tmp_x = np.copy(x)
        tmp_x[0] *= self.uk_scale[0]
        tmp_x[1] = -self.scale / np.log(tmp_x[1] * self.uk_scale[1])
        return tmp_x

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...] * self.uk_scale[0])
        T1 = np.abs(-self.scale / np.log(x[1, ...] * self.uk_scale[1]))
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
                self.gs = gridspec.GridSpec(2, 6, width_ratios=[
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

    def _execute_forward_2D(self, x, islice):
        S = np.zeros((self.NLL, self.Nproj, self.dimY, self.dimX), dtype=DTYPE)
        M0_sc = self.uk_scale[0]
        TR = self.TR
        tau = self.tau
        td = self.td
        sin_phi = self.sin_phi[islice, ...]
        cos_phi = self.cos_phi[islice, ...]
        N = self.Nproj_measured
        scale = self.scale
        Efit = x[1, ...] * self.uk_scale[1]
        Etau = Efit**(tau / scale)
        Etr = Efit**(TR / scale)
        Etd = Efit**(td / scale)
        M0 = x[0, ...]
        M0_sc = self.uk_scale[0]
        F = (1 - Etau) / (1 - Etau * cos_phi)
        Q = (-Etr * Etd * F * (-(Etau * cos_phi)**(N - 1) + 1) *
             cos_phi + Etr *
             Etd - 2 * Etd + 1) / (Etr * Etd * (Etau * cos_phi)**(N - 1) *
                                   cos_phi + 1)
        Q_F = Q - F

        for i in range(self.NLL):
            for j in range(self.Nproj):
                n = i * self.Nproj + j + 1
                S[i, j, ...] = M0 * M0_sc * \
                    ((Etau * cos_phi)**(n - 1) * Q_F + F) * sin_phi

        return np.array(np.mean(S, axis=1, dtype=DTYPE), dtype=DTYPE)

    def _execute_gradient_2D(self, x, islice):
        grad = np.zeros(
            (2,
             self.NLL,
             self.Nproj,
             self.dimY,
             self.dimX),
            dtype=DTYPE)
        M0_sc = self.uk_scale[0]
        TR = self.TR
        tau = self.tau
        td = self.td
        sin_phi = self.sin_phi[islice, ...]
        cos_phi = self.cos_phi[islice, ...]
        N = self.Nproj_measured
        scale = self.scale
        Efit = x[1, ...] * self.uk_scale[1]
        Etau = Efit**(tau / scale)
        Etr = Efit**(TR / scale)
        Etd = Efit**(td / scale)

        M0 = x[0, ...]
        M0_sc = self.uk_scale[0]

        F = (1 - Etau) / (1 - Etau * cos_phi)
        Q = (-Etr * Etd * (-Etau + 1) * (-(Etau * cos_phi)**(N - 1) + 1) *
             cos_phi / (-Etau * cos_phi + 1) + Etr * Etd - 2 * Etd + 1) / \
            (Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi + 1)
        Q_F = Q - F
        tmp1 = ((-TR * Etr * Etd * (-Etau + 1) *
                 (-(Etau * cos_phi)**(N - 1) + 1)
                 * cos_phi / (x[1, ...] * scale * (-Etau * cos_phi + 1)) +
                 TR * Etr * Etd / (x[1, ...] * scale) - tau * Etr * Etau *
                 Etd * (-Etau + 1) *
                 (-(Etau * cos_phi)**(N - 1) + 1) * cos_phi**2 / (x[1, ...] *
                 scale * (-Etau * cos_phi + 1)**2) +
                 tau * Etr * Etau * Etd * (-(Etau * cos_phi)**(N - 1) + 1) *
                 cos_phi / (x[1, ...] * scale * (-Etau * cos_phi + 1)) +
                 tau * Etr * Etd * (Etau * cos_phi)**(N - 1) *
                 (N - 1) * (-Etau + 1) * cos_phi / (x[1, ...] * scale *
                 (-Etau * cos_phi + 1)) -
                 td * Etr * Etd * (-Etau + 1) *
                 (-(Etau * cos_phi)**(N - 1) + 1) * cos_phi / (x[1, ...] *
                 scale * (-Etau * cos_phi + 1)) +
                 td * Etr * Etd / (x[1, ...] * scale) - 2 *
                 td * Etd / (x[1, ...] * scale)) /
                (Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi + 1) +
                (-TR * Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi /
                 (x[1, ...] * scale) - tau * Etr * Etd *
                 (Etau * cos_phi)**(N - 1) * (N - 1) *
                 cos_phi / (x[1, ...] * scale) - td * Etr * Etd *
                 (Etau * cos_phi)**(N - 1) * cos_phi / (x[1, ...] *
                 scale)) * (-Etr * Etd * (-Etau + 1) *
                (-(Etau * cos_phi)**(N - 1) + 1) * cos_phi /
                 (-Etau * cos_phi + 1) + Etr * Etd - 2 * Etd + 1) /
                (Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi + 1)**2 -
                tau * Etau * (-Etau + 1) * cos_phi / (x[1, ...] * scale *
                (-Etau * cos_phi + 1)**2)
                + tau * Etau / (x[1, ...] * scale * (-Etau * cos_phi + 1)))

        tmp2 = tau * Etau * (-Etau + 1) * cos_phi / (x[1, ...] * scale * (
            -Etau * cos_phi + 1)**2) - tau * Etau / (x[1, ...] * scale * (
                -Etau * cos_phi + 1))

        tmp3 = (-(-Etau + 1) / (-Etau * cos_phi + 1) + (-Etr * Etd *
                (-Etau + 1) * (-(Etau * cos_phi)**(N - 1) + 1) * cos_phi /
                (-Etau * cos_phi + 1) + Etr * Etd - 2 * Etd + 1) / (
                    Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi +
                    1)) / (x[1, ...] * scale)
        for i in range(self.NLL):
            for j in range(self.Nproj):
                n = i * self.Nproj + j + 1

                grad[0, i, j, ...] = M0_sc * \
                    ((Etau * cos_phi)**(n - 1) * Q_F + F) * sin_phi

                grad[1, i, j, ...] = M0 * M0_sc * ((Etau * cos_phi)**(
                    n - 1) * tmp1 + tmp2 + tau * (Etau * cos_phi)**(n - 1) *
                                     (n - 1) * tmp3) * sin_phi

        return np.array(
            np.mean(
                grad,
                axis=2,
                dtype=DTYPE),
            dtype=DTYPE)

    def _execute_forward_3D(self, x):
        S = np.zeros(
            (self.NLL,
             self.Nproj,
             self.NSlice,
             self.dimY,
             self.dimX),
            dtype=DTYPE)

        TR = self.TR
        tau = self.tau
        td = self.td
        sin_phi = self.sin_phi
        cos_phi = self.cos_phi
        N = self.Nproj_measured
        scale = self.scale
        Efit = x[1, ...] * self.uk_scale[1]
        Etau = Efit**(tau / scale)
        Etr = Efit**(TR / scale)
        Etd = Efit**(td / scale)
        M0 = x[0, ...]
        M0_sc = self.uk_scale[0]

        F = (1 - Etau) / (1 - Etau * cos_phi)
        Q = (-Etr * Etd * F * (-(Etau * cos_phi)**(N - 1) + 1) *
             cos_phi + Etr *
             Etd - 2 * Etd + 1) / (Etr * Etd * (Etau * cos_phi)**(N - 1) *
                                   cos_phi + 1)
        Q_F = Q - F

        def numexpeval_S(M0, M0_sc, sin_phi, cos_phi, n, Q_F, F, Etau):
            return ne.evaluate(
                "M0*M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi")
        for i in range(self.NLL):
            for j in range(self.Nproj):
                n = i * self.Nproj + j + 1
                S[i, j, ...] = numexpeval_S(
                    M0, M0_sc, sin_phi, cos_phi, n, Q_F, F, Etau)

        return np.array(np.mean(S, axis=1, dtype=DTYPE), dtype=DTYPE)

    def _execute_gradient_3D(self, x):
        grad = np.zeros(
            (2,
             self.NLL,
             self.Nproj,
             self.NSlice,
             self.dimY,
             self.dimX),
            dtype=DTYPE)
        M0_sc = self.uk_scale[0]
        TR = self.TR
        tau = self.tau
        td = self.td
        sin_phi = self.sin_phi
        cos_phi = self.cos_phi
        N = self.Nproj_measured
        scale = self.scale
        Efit = x[1, ...] * self.uk_scale[1]
        Etau = Efit**(tau / scale)
        Etr = Efit**(TR / scale)
        Etd = Efit**(td / scale)
        M0 = x[0, ...]
        M0_sc = self.uk_scale[0]

        F = (1 - Etau) / (1 - Etau * cos_phi)
        Q = (-Etr * Etd * (-Etau + 1) * (-(Etau * cos_phi)**(N - 1) + 1) *
             cos_phi / (-Etau * cos_phi + 1) + Etr * Etd - 2 * Etd + 1) / \
            (Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi + 1)
        Q_F = Q - F
        tmp1 = ((-TR * Etr * Etd * (-Etau + 1) *
                 (-(Etau * cos_phi)**(N - 1) + 1)
                 * cos_phi / (x[1, ...] * scale * (-Etau * cos_phi + 1)) +
                 TR * Etr * Etd / (x[1, ...] * scale) - tau * Etr * Etau *
                 Etd * (-Etau + 1) *
                 (-(Etau * cos_phi)**(N - 1) + 1) * cos_phi**2 / (x[1, ...] *
                 scale * (-Etau * cos_phi + 1)**2) +
                 tau * Etr * Etau * Etd * (-(Etau * cos_phi)**(N - 1) + 1) *
                 cos_phi / (x[1, ...] * scale * (-Etau * cos_phi + 1)) +
                 tau * Etr * Etd * (Etau * cos_phi)**(N - 1) *
                 (N - 1) * (-Etau + 1) * cos_phi / (x[1, ...] * scale *
                 (-Etau * cos_phi + 1)) -
                 td * Etr * Etd * (-Etau + 1) *
                 (-(Etau * cos_phi)**(N - 1) + 1) * cos_phi / (x[1, ...] *
                 scale * (-Etau * cos_phi + 1)) +
                 td * Etr * Etd / (x[1, ...] * scale) - 2 *
                 td * Etd / (x[1, ...] * scale)) /
                (Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi + 1) +
                (-TR * Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi /
                 (x[1, ...] * scale) - tau * Etr * Etd *
                 (Etau * cos_phi)**(N - 1) * (N - 1) *
                 cos_phi / (x[1, ...] * scale) - td * Etr * Etd *
                 (Etau * cos_phi)**(N - 1) * cos_phi / (x[1, ...] *
                 scale)) * (-Etr * Etd * (-Etau + 1) *
                (-(Etau * cos_phi)**(N - 1) + 1) * cos_phi /
                 (-Etau * cos_phi + 1) + Etr * Etd - 2 * Etd + 1) /
                (Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi + 1)**2 -
                tau * Etau * (-Etau + 1) * cos_phi / (x[1, ...] * scale *
                (-Etau * cos_phi + 1)**2)
                + tau * Etau / (x[1, ...] * scale * (-Etau * cos_phi + 1)))

        tmp2 = tau * Etau * (-Etau + 1) * cos_phi / (x[1, ...] * scale * (
            -Etau * cos_phi + 1)**2) - tau * Etau / (x[1, ...] * scale * (
                -Etau * cos_phi + 1))

        tmp3 = (-(-Etau + 1) / (-Etau * cos_phi + 1) + (-Etr * Etd *
                (-Etau + 1) * (-(Etau * cos_phi)**(N - 1) + 1) * cos_phi /
                (-Etau * cos_phi + 1) + Etr * Etd - 2 * Etd + 1) / (
                    Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi +
                    1)) / (x[1, ...] * scale)

        def numexpeval_M0(M0_sc, sin_phi, cos_phi, n, Q_F, F, Etau):
            return ne.evaluate(
                "M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi")

        def numexpeval_T1(
                M0,
                M0_sc,
                Etau,
                cos_phi,
                sin_phi,
                n,
                tmp1,
                tmp2,
                tmp3,
                tau):
            return ne.evaluate(
                "M0*M0_sc*((Etau*cos_phi)**(n - 1)*tmp1 + "
                "tmp2 + tau*(Etau*cos_phi)**(n - 1)*(n - 1)*tmp3)*sin_phi")

        for i in range(self.NLL):
            for j in range(self.Nproj):
                n = i * self.Nproj + j + 1

                grad[0, i, j, ...] = numexpeval_M0(
                    M0_sc, sin_phi, cos_phi, n, Q_F, F, Etau)
                grad[1, i, j, ...] = numexpeval_T1(
                    M0, M0_sc, Etau, cos_phi, sin_phi, n,
                    tmp1, tmp2, tmp3, tau)

        return np.array(np.mean(grad, axis=2, dtype=DTYPE), dtype=DTYPE)

    def _set_init_scales(self):
        self.scale = 100  # 100

        test_T1 = 800 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_M0 = np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_T1 = np.exp(-self.scale / (test_T1))

        x = np.array([test_M0 / self.uk_scale[0],
                      test_T1 / self.uk_scale[1]], dtype=DTYPE)
        return x
