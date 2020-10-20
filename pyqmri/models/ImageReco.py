#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the simple image model for image reconstruction."""
import numpy as np
from pyqmri.models.template import BaseModel, constraints


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
        uk_names = []
        for j in range(self.NScan):
            tmp_x[j] *= self.uk_scale[j]
            uk_names.append("Image_"+str(j))

        const = []
        for constrained in self.constraints:
            const.append(constrained.real)

        return {"data": tmp_x,
                "unknown_name": uk_names,
                "real_valued": const}

    def _execute_forward_3D(self, x):
        S = np.zeros_like(x)
        for j in range(self.NScan):
            S[j, ...] = x[j, ...] * self.uk_scale[j]
        S[~np.isfinite(S)] = 1e-20
        return S

    def _execute_gradient_3D(self, x):
        grad_M0 = np.zeros(((self.NScan, )+x.shape), dtype=self._DTYPE)
        for j in range(self.NScan):
            grad_M0[j, ...] = self.uk_scale[j]*np.ones_like(x)
        grad_M0[~np.isfinite(grad_M0)] = 1e-20
        return grad_M0

    def computeInitialGuess(self, *args):
        """Initialize unknown array for the fitting.

        This function provides an initial guess for the fitting.

        Parameters
        ----------
          args : list of objects
            Assumes the images series at position 0 and uses it as initial
            guess.
        """
        self.guess = ((args[0]).astype(self._DTYPE))
