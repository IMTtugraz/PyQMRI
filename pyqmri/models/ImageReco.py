#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the simple image model for image reconstruction."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyqmri.models.template import BaseModel, constraints, DTYPE
plt.ion()


class Model(BaseModel):
    """Image reconstruction model for MRI.

    A simple linear image model to perform image reconstruction with
    joint regularization on all Scans.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the sequence related parametrs,
        e.g. TR, TE, TI, to fully describe the acquisitio process
    Attributes
    ----------
      guess : numpy.array, None
        Initial guess for the images. Set after object creation using
        "computeInitialGuess"
    """

    def __init__(self, par):
        super().__init__(par)

        par["unknowns_TGV"] = self.NScan
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]

        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        for j in range(par["unknowns"]):
            self.constraints.append(
                constraints(-100 / self.uk_scale[j],
                            100 / self.uk_scale[j],
                            False))
        self._image_plot = []
        self.guess = None

    def rescale(self, x):
        """Rescale the unknowns with the scaling factors.

        Rescales each unknown with the corresponding scaling.

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
        for j in range(self.NScan):
            tmp_x[j] *= self.uk_scale[j]
        return tmp_x

    def _execute_forward_3D(self, x):
        S = np.zeros_like(x)
        for j in range(self.NScan):
            S[j, ...] = x[j, ...] * self.uk_scale[j]
        S[~np.isfinite(S)] = 1e-20
        return S

    def _execute_gradient_3D(self, x):
        grad_M0 = np.zeros(((self.NScan, )+x.shape), dtype=DTYPE)
        for j in range(self.NScan):
            grad_M0[j, ...] = self.uk_scale[j]*np.ones_like(x)
        grad_M0[~np.isfinite(grad_M0)] = 1e-20
        return grad_M0

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
        M0 = np.zeros_like(x)
        M0_min = []
        M0_max = []
        M0 = np.abs(M0)
        for j in range(x.shape[0]):
            M0[j, ...] = np.abs(x[j, ...] * self.uk_scale[j])
            M0_min.append(M0[j].min())
            M0_max.append(M0[j].max())
        if dim_2D:
            raise NotImplementedError("2D Not Implemented")

        [z, y, x] = M0.shape[1:]
        if not self.figure:
            plt.ion()
            ax = []
            plot_dim = int(np.ceil(np.sqrt(M0.shape[0])))
            self.figure = plt.figure(figsize=(12, 6))
            self.figure.subplots_adjust(hspace=0.3, wspace=0)
            wd_ratio = np.tile([1, 1 / 20, 1 / (5)], plot_dim)
            gs = gridspec.GridSpec(
                plot_dim, 3 * plot_dim,
                width_ratios=wd_ratio, hspace=0.3, wspace=0)
            self.figure.tight_layout()
            self.figure.patch.set_facecolor('black')
            for grid in gs:
                ax.append(plt.subplot(grid))
                ax[-1].axis('off')
            for j in range(M0.shape[0]):
                self._image_plot.append(
                    ax[3 * j].imshow(
                        (M0[j, int(z/2)]),
                        vmin=M0_min[j],
                        vmax=M0_max[j], cmap='gray'))
                ax[3 *
                   j].set_title('Image: ' +
                                str(j), color='white')
                ax[3 * j + 1].axis('on')
                cbar = self.figure.colorbar(
                    self._image_plot[j], cax=ax[3 * j + 1])
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)
        else:
            for j in range(M0.shape[0]):
                self._image_plot[j].set_data((M0[j, int(z/2)]))
                self._image_plot[j].set_clim([M0_min[j],
                                             M0_max[j]])

            self.figure.canvas.draw_idle()
            plt.draw()
            plt.pause(1e-10)

    def computeInitialGuess(self, *args):
        """Initialize unknown array for the fitting.

        This function provides an initial guess for the fitting.

        Parameters
        ----------
          args : list of objects
            Assumes the images series at position 0 and uses it as initial
            guess.
        """
        self.guess = ((args[0]).astype(DTYPE))
