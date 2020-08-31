#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the bi-exponential model for fitting."""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pyqmri.models.template import BaseModel, constraints, DTYPE
plt.ion()


class Model(BaseModel):
    """Bi-exponential model for MRI parameter quantification.

    This class holds a bi-exponential model for fitting complex MRI data.
    It realizes a forward application of the analytical signal expression
    and the partial derivatives with respesct to each parameter of interest,
    as required by the abstract methods in the BaseModel.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the sequence related parametrs,
        e.g. TR, TE, TI, to fully describe the acquisitio process

    Attributes
    ----------
      TE : float
        Echo time (or any other timing valid in a bi-exponential fit).
      uk_scale : list of float
        Scaling factors for each unknown to balance the partial derivatives.
      guess : numpy.array
        The initial guess. Needs to be set using "computeInitialGuess"
        prior to fitting.
    """

    def __init__(self, par):
        super().__init__(par)
        self.TE = np.ones((self.NScan, 1, 1, 1))

        for i in range(self.NScan):
            self.TE[i, ...] = par["TE"][i] * np.ones((1, 1, 1))

        par["unknowns_TGV"] = 5
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]

        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        self.constraints.append(
            constraints(
                -100 / self.uk_scale[0],
                100 / self.uk_scale[0], False))
        self.constraints.append(
            constraints(
                0 / self.uk_scale[1],
                1 / self.uk_scale[1],
                True))
        self.constraints.append(
            constraints(
                ((1 / 150) / self.uk_scale[2]),
                ((1 / 1e-4) / self.uk_scale[2]),
                True))
        self.constraints.append(
            constraints(
                0 / self.uk_scale[3],
                1 / self.uk_scale[3],
                True))
        self.constraints.append(
            constraints(
                ((1 / 1500) / self.uk_scale[4]),
                ((1 / 150) / self.uk_scale[4]),
                True))

    def rescale(self, x):
        """Rescale the unknowns with the scaling factors.

        Rescales each unknown with the corresponding scaling factor and
        applies a 1/x transformation for the time constants of the
        exponentials, yielding a result in milliseconds.

        Parameters
        ----------
          x : numpy.array
            The array of unknowns to be rescaled

        Returns
        -------
          numpy.array:
            The rescaled unknowns
        """
        M0 = x[0, ...] * self.uk_scale[0]
        M01 = x[1, ...] * self.uk_scale[1]
        T21 = 1 / (x[2, ...] * self.uk_scale[2])
        M02 = x[3, ...] * self.uk_scale[3]
        T22 = 1 / (x[4, ...] * self.uk_scale[4])
        return np.array((M0, M01, T21, M02, T22))

    def _execute_forward_3D(self, x):
        M0 = x[0, ...] * self.uk_scale[0]
        M01 = x[1, ...] * self.uk_scale[1]
        T21 = x[2, ...] * self.uk_scale[2]
        M02 = x[3, ...] * self.uk_scale[3]
        T22 = x[4, ...] * self.uk_scale[4]
        S = M0 * (M01 * np.exp(-self.TE * (T21)) +
                  M02 * np.exp(-self.TE * (T22)))
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        M0 = x[0, ...]
        M01 = x[1, ...]
        T21 = x[2, ...]
        M02 = x[3, ...]
        T22 = x[4, ...]
        grad_M0 = self.uk_scale[0] * (
            M01 * self.uk_scale[1] * np.exp(-self.TE * (
                T21 * self.uk_scale[2])) +
            M02 * self.uk_scale[3] *
            np.exp(-self.TE * (T22 * self.uk_scale[4])))
        grad_M01 = self.uk_scale[0] * M0 * self.uk_scale[1] * \
            np.exp(-self.TE * (T21 * self.uk_scale[2]))
        grad_T21 = -self.uk_scale[0] * M0 * M01 * self.uk_scale[1] * \
            self.TE * \
            self.uk_scale[2] * np.exp(-self.TE * (T21 * self.uk_scale[2]))
        grad_M02 = self.uk_scale[0] * M0 * self.uk_scale[3] * \
            np.exp(-self.TE * (T22 * self.uk_scale[4]))
        grad_T22 = -self.uk_scale[0] * M0 * M02 * self.uk_scale[3] * \
            self.TE * \
            self.uk_scale[4] * np.exp(-self.TE * (T22 * self.uk_scale[4]))
        grad = np.array([grad_M0, grad_M01, grad_T21,
                         grad_M02, grad_T22], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

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
        M0 = np.abs(x[0, ...]) * self.uk_scale[0]
        M0_min = M0.min()
        M0_max = M0.max()

        M01 = np.abs(x[1, ...]) * self.uk_scale[1]
        T21 = 1 / (np.abs(x[2, ...]) * self.uk_scale[2])
        M01_min = M01.min()
        M01_max = M01.max()
        T21_min = T21.min()
        T21_max = T21.max()

        M02 = np.abs(x[3, ...]) * self.uk_scale[3]
        T22 = 1 / (np.abs(x[4, ...]) * self.uk_scale[4])
        M02_min = M02.min()
        M02_max = M02.max()
        T22_min = T22.min()
        T22_max = T22.max()

        [z, y, x] = M01.shape
        self._ax = []
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
                self._ax.append(plt.subplot(grid))
                self._ax[-1].axis('off')

            self._M0_plot = self._ax[0].imshow(
                (M0[int(self.NSlice / 2), ...]))
            self._M0_plot_cor = self._ax[17].imshow(
                (M0[:, int(M01.shape[1] / 2), ...]))
            self._M0_plot_sag = self._ax[1].imshow(
                np.flip((M0[:, :, int(M01.shape[-1] / 2)]).T, 1))
            self._ax[0].set_title('Proton Density in a.u.', color='white')
            self._ax[0].set_anchor('SE')
            self._ax[1].set_anchor('SW')
            self._ax[17].set_anchor('NW')
            cax = plt.subplot(self.gs[:, 2])
            cbar = self.figure.colorbar(self._M0_plot, cax=cax)
            cbar.ax.tick_params(labelsize=12, colors='white')
#           cax.yaxis.set_ticks_position('left')
            for spine in cbar.ax.spines:
                cbar.ax.spines[spine].set_color('white')

            self._M01_plot = self._ax[5].imshow(
                (M01[int(self.NSlice / 2), ...]))
            self._M01_plot_cor = self._ax[22].imshow(
                (M01[:, int(M01.shape[1] / 2), ...]))
            self._M01_plot_sag = self._ax[6].imshow(
                np.flip((M01[:, :, int(M01.shape[-1] / 2)]).T, 1))
            self._ax[5].set_title('Proton Density in a.u.', color='white')
            self._ax[5].set_anchor('SE')
            self._ax[6].set_anchor('SW')
            self._ax[22].set_anchor('NW')
            cax = plt.subplot(self.gs[:, 4])
            cbar = self.figure.colorbar(self._M01_plot, cax=cax)
            cbar.ax.tick_params(labelsize=12, colors='white')
            cax.yaxis.set_ticks_position('left')
            for spine in cbar.ax.spines:
                cbar.ax.spines[spine].set_color('white')

            self._M02_plot = self._ax[12].imshow(
                (M02[int(self.NSlice / 2), ...]))
            self._M02_plot_cor = self._ax[29].imshow(
                (M02[:, int(M02.shape[1] / 2), ...]))
            self._M02_plot_sag = self._ax[13].imshow(
                np.flip((M02[:, :, int(M02.shape[-1] / 2)]).T, 1))
            self._ax[12].set_title('Proton Density in a.u.', color='white')
            self._ax[12].set_anchor('SE')
            self._ax[13].set_anchor('SW')
            self._ax[29].set_anchor('NW')
            cax = plt.subplot(self.gs[:, 11])
            cbar = self.figure.colorbar(self._M02_plot, cax=cax)
            cbar.ax.tick_params(labelsize=12, colors='white')
            cax.yaxis.set_ticks_position('left')
            for spine in cbar.ax.spines:
                cbar.ax.spines[spine].set_color('white')

            self._T21_plot = self._ax[7].imshow(
                (T21[int(self.NSlice / 2), ...]))
            self._T21_plot_cor = self._ax[24].imshow(
                (T21[:, int(T21.shape[1] / 2), ...]))
            self._T21_plot_sag = self._ax[8].imshow(
                np.flip((T21[:, :, int(T21.shape[-1] / 2)]).T, 1))
            self._ax[7].set_title('T21 in  ms', color='white')
            self._ax[7].set_anchor('SE')
            self._ax[8].set_anchor('SW')
            self._ax[24].set_anchor('NW')
            cax = plt.subplot(self.gs[:, 9])
            cbar = self.figure.colorbar(self._T21_plot, cax=cax)
            cbar.ax.tick_params(labelsize=12, colors='white')
            for spine in cbar.ax.spines:
                cbar.ax.spines[spine].set_color('white')

            self._T22_plot = self._ax[14].imshow(
                (T22[int(self.NSlice / 2), ...]))
            self._T22_plot_cor = self._ax[31].imshow(
                (T22[:, int(T22.shape[1] / 2), ...]))
            self._T22_plot_sag = self._ax[15].imshow(
                np.flip((T22[:, :, int(T22.shape[-1] / 2)]).T, 1))
            self._ax[14].set_title('T22 in  ms', color='white')
            self._ax[14].set_anchor('SE')
            self._ax[15].set_anchor('SW')
            self._ax[31].set_anchor('NW')
            cax = plt.subplot(self.gs[:, 16])
            cbar = self.figure.colorbar(self._T22_plot, cax=cax)
            cbar.ax.tick_params(labelsize=12, colors='white')
            for spine in cbar.ax.spines:
                cbar.ax.spines[spine].set_color('white')

            plt.draw()
            plt.pause(1e-10)
        else:
            self._M0_plot.set_data((M0[int(self.NSlice / 2), ...]))
            self._M0_plot_cor.set_data((M0[:, int(M01.shape[1] / 2), ...]))
            self._M0_plot_sag.set_data(
                np.flip((M0[:, :, int(M01.shape[-1] / 2)]).T, 1))
            self._M0_plot.set_clim([M0_min, M0_max])
            self._M0_plot_cor.set_clim([M0_min, M0_max])
            self._M0_plot_sag.set_clim([M0_min, M0_max])

            self._M01_plot.set_data((M01[int(self.NSlice / 2), ...]))
            self._M01_plot_cor.set_data(
                (M01[:, int(M01.shape[1] / 2), ...]))
            self._M01_plot_sag.set_data(
                np.flip((M01[:, :, int(M01.shape[-1] / 2)]).T, 1))
            self._M01_plot.set_clim([M01_min, M01_max])
            self._M01_plot_cor.set_clim([M01_min, M01_max])
            self._M01_plot_sag.set_clim([M01_min, M01_max])
            self._T21_plot.set_data((T21[int(self.NSlice / 2), ...]))
            self._T21_plot_cor.set_data(
                (T21[:, int(T21.shape[1] / 2), ...]))
            self._T21_plot_sag.set_data(
                np.flip((T21[:, :, int(T21.shape[-1] / 2)]).T, 1))
            self._T21_plot.set_clim([T21_min, T21_max])
            self._T21_plot_sag.set_clim([T21_min, T21_max])
            self._T21_plot_cor.set_clim([T21_min, T21_max])

            self._M02_plot.set_data((M02[int(self.NSlice / 2), ...]))
            self._M02_plot_cor.set_data(
                (M02[:, int(M02.shape[1] / 2), ...]))
            self._M02_plot_sag.set_data(
                np.flip((M02[:, :, int(M02.shape[-1] / 2)]).T, 1))
            self._M02_plot.set_clim([M02_min, M02_max])
            self._M02_plot_cor.set_clim([M02_min, M02_max])
            self._M02_plot_sag.set_clim([M02_min, M02_max])
            self._T22_plot.set_data((T22[int(self.NSlice / 2), ...]))
            self._T22_plot_cor.set_data(
                (T22[:, int(T22.shape[1] / 2), ...]))
            self._T22_plot_sag.set_data(
                np.flip((T22[:, :, int(T22.shape[-1] / 2)]).T, 1))
            self._T22_plot.set_clim([T22_min, T22_max])
            self._T22_plot_sag.set_clim([T22_min, T22_max])
            self._T22_plot_cor.set_clim([T22_min, T22_max])
            plt.draw()
            plt.pause(1e-10)

    def computeInitialGuess(self, *args):
        """Initialize unknown array for the fitting.

        This function provides an initial guess for the fitting.

        Parameters
        ----------
          args : list of objects
            Serves as universal interface. No objects need to be passed
            here.
        """
        test_M0 = 0.1 * np.ones(
            (self.NSlice, self.dimY, self.dimX),
            dtype=DTYPE)
        test_M01 = 0.5 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_T11 = 1/10 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_T12 = 1/150 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)

        self.guess = np.array([
            test_M0,
            test_M01,
            test_T11,
            test_M01,
            test_T12], dtype=DTYPE)
