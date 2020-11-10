#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the variable flip angle model for T1 fitting."""
import numpy as np
from pyqmri.models.template import BaseModel, constraints


class Model(BaseModel):
    """Variable flip angle model for MRI parameter quantification.

    This class holds a variable flip angle model for T1 quantification from
    complex MRI data. It realizes a forward application of the analytical
    signal expression and the partial derivatives with respesct to
    each parameter of interest, as required by the abstract methods in
    the BaseModel. The fitting target is the exponential term itself
    which is easier to fit than the corresponding timing constant.

    The rescale function applies a transformation and returns the expected
    T1 values in ms.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the sequence related parametrs,
        e.g. TR, TE, TI, to fully describe the acquisitio process

    Attributes
    ----------
      TR : float
        Repetition time of the gradient echo sequence.
      fa : numpy.array
        A vector containing all flip angles, one per scan.
      uk_scale : list of float
        Scaling factors for each unknown to balance the partial derivatives.
      guess : numpy.array, None
        The initial guess. Needs to be set using "computeInitialGuess"
        prior to fitting.
    """

    def __init__(self, par):
        super().__init__(par)
        self.TR = par["TR"]
        self.fa = par["flip_angle(s)"]

        try:
            self.fa_corr = par["fa_corr"]
        except KeyError:
            self.fa_corr = 1
            print("No flipangle correction found!")

        phi_corr = np.zeros(
            (self.NScan, self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        for i in range(np.size(par["flip_angle(s)"])):
            phi_corr[i, :, :, :] = par["flip_angle(s)"][i] *\
                np.pi / 180 * self.fa_corr

        par["unknowns_TGV"] = 2
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]

        self._sin_phi = np.sin(phi_corr)
        self._cos_phi = np.cos(phi_corr)

        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        self.constraints.append(
            constraints(0 / self.uk_scale[0],
                        1e5 / self.uk_scale[0],
                        False))
        self.constraints.append(
            constraints(np.exp(-self.TR / (50)),
                        np.exp(-self.TR / (5500)),
                        True))
        self.guess = None

        self._ax = None

        self._M0_plot = None
        self._M0_plot_cor = None
        self._M0_plot_sag = None

        self._T1_plot = None
        self._T1_plot_cor = None
        self._T1_plot_sag = None

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
        tmp_x[1] = -self.TR / np.log(tmp_x[1] * self.uk_scale[1])
        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return {"data": tmp_x,
                "unknown_name": ["M0", "T1"],
                "real_valued": const}

    def _execute_forward_3D(self, x):
        E1 = x[1, ...] * self.uk_scale[1]
        S = x[0, ...] * self.uk_scale[0] * (-E1 + 1) * self._sin_phi /\
            (-E1 * self._cos_phi + 1)
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=self._DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        E1 = x[1, ...] * self.uk_scale[1]
        M0 = x[0, ...]
        E1[~np.isfinite(E1)] = 0
        grad_M0 = self.uk_scale[0] * (-E1 + 1) * self._sin_phi /\
            (-E1 * self._cos_phi + 1)
        grad_T1 = M0 * self.uk_scale[0] * self.uk_scale[1] * (-E1 + 1) *\
            self._sin_phi * self._cos_phi / (-E1 * self._cos_phi + 1)**2 -\
            M0 * self.uk_scale[0] * self.uk_scale[1] * self._sin_phi /\
            (-E1 * self._cos_phi + 1)
        grad = np.array([grad_M0, grad_T1], dtype=self._DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def computeInitialGuess(self, *args):
        """Initialize unknown array for the fitting.

        This function provides an initial guess for the fitting.

        Parameters
        ----------
          args : list of objects
            Serves as universal interface. No objects need to be passed
            here.
        """
        test_T1 = 1500 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        test_M0 = 0.1*np.ones(
            (self.NSlice, self.dimY, self.dimX),
            dtype=self._DTYPE)
        test_T1 = np.exp(-self.TR / (test_T1))
        x = np.array([test_M0 / self.uk_scale[0],
                      test_T1 / self.uk_scale[1]], dtype=self._DTYPE)
        self.guess = x
