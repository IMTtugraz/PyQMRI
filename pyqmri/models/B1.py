#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the variable flip angle model for T1 fitting."""
import numpy as np
from pyqmri.models.template import BaseModel, constraints


class Model(BaseModel):
    """B1 model for MRI parameter quantification.

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


        par["unknowns_TGV"] = 2
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]


        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        self.constraints.append(
            constraints(0,
                        1e5,
                        False))
        self.constraints.append(
            constraints(0,
                        1e5,
                        False))

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
        for j in range(x.shape[0]):
            tmp_x[j] *= self.uk_scale[j]
        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        tmp2 = np.copy(x)
        tmp2[0] = np.abs(tmp_x[1])/np.abs(tmp_x[0])
        tmp2[1] = np.angle(tmp_x[1])-np.angle(tmp_x[0])
        
        
        return {"data": tmp2,
                "unknown_name": ["P1", "P2"],
                "real_valued": const}

    def _execute_forward_3D(self, x):
        
        S1 = x[0]*self.uk_scale[0]
        S2 = x[1]*self.uk_scale[1]
        S3 = (x[0]*self.uk_scale[0]-x[1]*self.uk_scale[1])
        
        S = np.array([S1,S2,S3], dtype=self._DTYPE)
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=self._DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        
        grad_r1_p1 = self.uk_scale[0] * np.ones_like(x[0])
        grad_r1_p2 = np.zeros_like(grad_r1_p1)
        grad_r1_p3 = self.uk_scale[0] * np.ones_like(x[0])
        
        grad_r1 = np.array([grad_r1_p1, grad_r1_p2, grad_r1_p3])

        grad_r2_p1 = np.zeros_like(grad_r1_p1)
        grad_r2_p2 = self.uk_scale[1] * np.ones_like(x[0])
        grad_r2_p3 = -self.uk_scale[1] * np.ones_like(x[0])
        
        grad_r2 = np.array([grad_r2_p1, grad_r2_p2, grad_r2_p3])
        
        grad = np.array([grad_r1, grad_r2], dtype=self._DTYPE)
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

        x = np.array([(args[0][0]),
                      (args[0][1])], dtype=self._DTYPE)
        self.guess = x
