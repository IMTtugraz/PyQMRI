#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the inversion recovers Look-Locker quantification model."""
import numexpr as ne
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pyqmri.models.template import BaseModel, constraints, DTYPE
plt.ion()


class Model(BaseModel):
    """Inversion recovery Look-Locker model for MRI parameter quantification.

    This class holds a IRLL model for T1 quantification from
    complex MRI data. It realizes a forward application of the analytical
    signal expression and the partial derivatives with respesct to
    each parameter of interest, as required by the abstract methods in
    the BaseModel. The fitting target is the exponential term itself
    which is easier to fit than the corresponding timing constant.

    The rescale function applies a transformation and returns the expected
    T1 values in ms.

    The implemented signal model follows the work from Henderson et al. (1999)

    The model should only be used for radially acquired data!

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the sequence related parametrs,
        e.g. TR, TE, TI, to fully describe the acquisitio process

    Attributes
    ----------
      TR : float
        Repetition time of the IRLL sequence.
      fa : float
        Flip angle of the gradient echo sequence.
      tau : float
        Repetition time for one gradient echo read out.
      td : float
        Delay prio to first read-out point after inversion.
      Nproj : int
        Number of projections per bin
      Nproj_measrued : int
        Total number of projections measured
      uk_scale : list of float
        Scaling factors for each unknown to balance the partial derivatives.
      guess : numpy.array, None
        The initial guess. Needs to be set using "computeInitialGuess"
        prior to fitting.
      scale : float
        A scaling factor to balance the different exponential terms.

    """

    def __init__(self, par):
        super().__init__(par)

        self.TR = par['time_per_slice'] - \
            (par['tau'] * par['Nproj_measured'] + par['gradient_delay'])
        self.fa = par["flip_angle(s)"] * np.pi / 180

        self.tau = par["tau"]
        self.td = par["gradient_delay"]
        self.Nproj = par["Nproj"]
        self.Nproj_measured = par["Nproj_measured"]

        par["unknowns_TGV"] = 2
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]

        phi_corr = np.zeros_like(par["fa_corr"], dtype=DTYPE)
        phi_corr = np.real(
            self.fa) * np.real(par["fa_corr"]) + 1j *\
            np.imag(self.fa) * np.imag(par["fa_corr"])

        self._sin_phi = np.sin(phi_corr)
        self._cos_phi = np.cos(phi_corr)

        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        self.scale = 100

        self.constraints.append(constraints(0, 300, False))
        self.constraints.append(constraints(
            np.exp(-self.scale / 10) / self.uk_scale[1],
            np.exp(-self.scale / 5500) / self.uk_scale[1], True))

        self.guess = None

        self._ax = None

        self._M0_plot = None
        self._M0_plot_cor = None
        self._M0_sag = None

        self._T1_plot = None
        self._T1_plot_cor = None
        self._T1_sag = None

    def rescale(self, x):
        """Rescale the unknowns with the scaling factors.

        Rescales each unknown with the corresponding scaling factor and
        applies a transformation for the time constants of the exponentials,
        yielding a resulting T1 in milliseconds.

        Parameters
        ----------
          x : numpy.array
            The array of unknowns to be rescaled

        Returns
        -------
          numpy.array:
            The rescaled unknowns
        """
        tmp_x = np.copy(x)
        tmp_x[0] *= self.uk_scale[0]
        tmp_x[1] = -self.scale / np.log(tmp_x[1] * self.uk_scale[1])
        return tmp_x

    def plot_unknowns(self, x, dim_2D=False):
        """Plot the unkowns in an interactive figure.

        This function can be used to plot intermediate results during the
        optimization process.

        Parameters
        ----------
          x : numpy.array
            The array of unknowns to be displayed
          dim_2D : bool, false
            Currently unused.
        """
        M0 = np.abs(x[0, ...] * self.uk_scale[0])
        T1 = np.abs(-self.scale / np.log(x[1, ...] * self.uk_scale[1]))
        M0_min = M0.min()
        M0_max = M0.max()
        T1_min = T1.min()
        T1_max = T1.max()
        if dim_2D:
            if not self.figure:
                plt.ion()
                self.figure, self._ax = plt.subplots(1, 2, figsize=(12, 5))
                self._M0_plot = self._ax[0].imshow((M0))
                self._ax[0].set_title('Proton Density in a.u.')
                self._ax[0].axis('off')
                self.figure.colorbar(self._M0_plot, ax=self._ax[0])
                self._T1_plot = self._ax[1].imshow((T1))
                self._ax[1].set_title('T1 in  ms')
                self._ax[1].axis('off')
                self.figure.colorbar(self._T1_plot, ax=self._ax[1])
                self.figure.tight_layout()
                plt.draw()
                plt.pause(1e-10)
            else:
                self._M0_plot.set_data((M0))
                self._M0_plot.set_clim([M0_min, M0_max])
                self._T1_plot.set_data((T1))
                self._T1_plot.set_clim([T1_min, T1_max])
                plt.draw()
                plt.pause(1e-10)
        else:
            [z, y, x] = M0.shape
            self._ax = []
            if not self.figure:
                plt.ion()
                self.figure = plt.figure(figsize=(12, 6))
                self.figure.subplots_adjust(hspace=0, wspace=0)
                gs = gridspec.GridSpec(
                    2, 6, width_ratios=[
                        x / (20 * z), x / z, 1,
                        x / z, 1, x / (20 * z)],
                    height_ratios=[x / z, 1]
                    )
                self.figure.tight_layout()
                self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
                for grid in gs:
                    self._ax.append(plt.subplot(grid))
                    self._ax[-1].axis('off')

                self._M0_plot = self._ax[1].imshow(
                    (M0[int(self.NSlice / 2), ...]))
                self._M0_plot_cor = self._ax[7].imshow(
                    (M0[:, int(M0.shape[1] / 2), ...]))
                self._M0_plot_sag = self._ax[2].imshow(
                    np.flip((M0[:, :, int(M0.shape[-1] / 2)]).T, 1))
                self._ax[1].set_title('Proton Density in a.u.', color='white')
                self._ax[1].set_anchor('SE')
                self._ax[2].set_anchor('SW')
                self._ax[7].set_anchor('NW')
                cax = plt.subplot(gs[:, 0])
                cbar = self.figure.colorbar(self._M0_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)

                self._T1_plot = self._ax[3].imshow(
                    (T1[int(self.NSlice / 2), ...]))
                self._T1_plot_cor = self._ax[9].imshow(
                    (T1[:, int(T1.shape[1] / 2), ...]))
                self._T1_plot_sag = self._ax[4].imshow(
                    np.flip((T1[:, :, int(T1.shape[-1] / 2)]).T, 1))
                self._ax[3].set_title('T1 in  ms', color='white')
                self._ax[3].set_anchor('SE')
                self._ax[4].set_anchor('SW')
                self._ax[9].set_anchor('NW')
                cax = plt.subplot(gs[:, 5])
                cbar = self.figure.colorbar(self._T1_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)
            else:
                self._M0_plot.set_data((M0[int(self.NSlice / 2), ...]))
                self._M0_plot_cor.set_data((M0[:, int(M0.shape[1] / 2), ...]))
                self._M0_plot_sag.set_data(
                    np.flip((M0[:, :, int(M0.shape[-1] / 2)]).T, 1))
                self._M0_plot.set_clim([M0_min, M0_max])
                self._M0_plot_cor.set_clim([M0_min, M0_max])
                self._M0_plot_sag.set_clim([M0_min, M0_max])
                self._T1_plot.set_data((T1[int(self.NSlice / 2), ...]))
                self._T1_plot_cor.set_data((T1[:, int(T1.shape[1] / 2), ...]))
                self._T1_plot_sag.set_data(
                    np.flip((T1[:, :, int(T1.shape[-1] / 2)]).T, 1))
                self._T1_plot.set_clim([T1_min, T1_max])
                self._T1_plot_sag.set_clim([T1_min, T1_max])
                self._T1_plot_cor.set_clim([T1_min, T1_max])
                plt.draw()
                plt.pause(1e-10)

    def _execute_forward_3D(self, x):
        S = np.zeros(
            (self.NScan,
             self.Nproj,
             self.NSlice,
             self.dimY,
             self.dimX),
            dtype=DTYPE)

        TR = self.TR
        tau = self.tau
        td = self.td
        sin_phi = self._sin_phi
        cos_phi = self._cos_phi
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
        for i in range(self.NScan):
            for j in range(self.Nproj):
                n = i * self.Nproj + j + 1
                S[i, j, ...] = numexpeval_S(
                    M0, M0_sc, sin_phi, cos_phi, n, Q_F, F, Etau)

        return np.array(np.mean(S, axis=1, dtype=DTYPE), dtype=DTYPE)

    def _execute_gradient_3D(self, x):
        grad = np.zeros(
            (2,
             self.NScan,
             self.Nproj,
             self.NSlice,
             self.dimY,
             self.dimX),
            dtype=DTYPE)
        M0_sc = self.uk_scale[0]
        TR = self.TR
        tau = self.tau
        td = self.td
        sin_phi = self._sin_phi
        cos_phi = self._cos_phi
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
        tmp1 = (
            (
                - TR * Etr * Etd * (-Etau + 1)
                * (-(Etau * cos_phi)**(N - 1) + 1)
                * cos_phi / (x[1, ...] * scale * (-Etau * cos_phi + 1))
                + TR * Etr * Etd / (x[1, ...] * scale) - tau * Etr * Etau
                * Etd * (-Etau + 1)
                * (-(Etau * cos_phi)**(N - 1) + 1) * cos_phi**2
                / (x[1, ...] * scale * (-Etau * cos_phi + 1)**2)
                + tau * Etr * Etau * Etd * (-(Etau * cos_phi)**(N - 1) + 1)
                * cos_phi / (x[1, ...] * scale * (-Etau * cos_phi + 1))
                + tau * Etr * Etd * (Etau * cos_phi)**(N - 1)
                * (N - 1) * (-Etau + 1) * cos_phi
                / (x[1, ...] * scale * (-Etau * cos_phi + 1))
                - td * Etr * Etd * (-Etau + 1)
                * (-(Etau * cos_phi)**(N - 1) + 1) * cos_phi
                / (x[1, ...] * scale * (-Etau * cos_phi + 1))
                + td * Etr * Etd / (x[1, ...] * scale) - 2 *
                td * Etd / (x[1, ...] * scale)
            ) / (Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi + 1)
            + (
                -TR * Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi
                / (x[1, ...] * scale) - tau * Etr * Etd
                * (Etau * cos_phi)**(N - 1) * (N - 1)
                * cos_phi / (x[1, ...] * scale) - td * Etr * Etd
                * (Etau * cos_phi)**(N - 1) * cos_phi
                / (x[1, ...] * scale)
            ) * (
                -Etr * Etd * (-Etau + 1)
                * (-(Etau * cos_phi)**(N - 1) + 1) * cos_phi
                / (-Etau * cos_phi + 1) + Etr * Etd - 2 * Etd + 1
            ) / (
                Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi + 1
            )**2 -
            tau * Etau * (-Etau + 1) * cos_phi
            / (x[1, ...] * scale * (-Etau * cos_phi + 1)**2)
            + tau * Etau / (x[1, ...] * scale * (-Etau * cos_phi + 1))
            )

        tmp2 = (
            tau * Etau * (-Etau + 1) * cos_phi
            / (x[1, ...] * scale * (-Etau * cos_phi + 1)**2)
            - tau * Etau / (x[1, ...] * scale * (-Etau * cos_phi + 1))
            )

        tmp3 = (
            -(-Etau + 1) / (-Etau * cos_phi + 1)
            + (
                - Etr * Etd * (-Etau + 1) * (-(Etau * cos_phi)**(N - 1) + 1)
                * cos_phi / (-Etau * cos_phi + 1) + Etr * Etd - 2 * Etd + 1
                ) / (
                    Etr * Etd * (Etau * cos_phi)**(N - 1) * cos_phi +
                    1
                    )
            ) / (x[1, ...] * scale)

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

        for i in range(self.NScan):
            for j in range(self.Nproj):
                n = i * self.Nproj + j + 1

                grad[0, i, j, ...] = numexpeval_M0(
                    M0_sc, sin_phi, cos_phi, n, Q_F, F, Etau)
                grad[1, i, j, ...] = numexpeval_T1(
                    M0, M0_sc, Etau, cos_phi, sin_phi, n,
                    tmp1, tmp2, tmp3, tau)

        return np.array(np.mean(grad, axis=2, dtype=DTYPE), dtype=DTYPE)

    def computeInitialGuess(self, *args):
        """Initialize unknown array for the fitting.

        This function provides an initial guess for the fitting.

        Parameters
        ----------
          args : list of objects
            Serves as universal interface. No objects need to be passed
            here.
        """
        test_T1 = 800 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_M0 = 1e-3*np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_T1 = np.exp(-self.scale / (test_T1))

        self.guess = np.array(
            [test_M0 / self.uk_scale[0],
             test_T1 / self.uk_scale[1]], dtype=DTYPE)
