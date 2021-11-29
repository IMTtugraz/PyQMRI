#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the bi-exponential model for fitting."""
import numpy as np
from pyqmri.models.template import BaseModel, constraints
from skimage.restoration import unwrap_phase


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
      guess : numpy.array, None
        The initial guess. Needs to be set using "computeInitialGuess"
        prior to fitting.
    """

    def __init__(self, par):
        super().__init__(par)
        self.TE = np.ones((self.NScan, 1, 1, 1))
        

        for i in range(self.NScan):
            self.TE[i, ...] = par["TEs"][i] * np.ones((1, 1, 1))

        par["unknowns_TGV"] = 3
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]

        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        self.constraints.append(
            constraints(
                0 / self.uk_scale[0],
                1000 / self.uk_scale[0], False))
        self.constraints.append(
            constraints(
                0.001 / self.uk_scale[1],
                150 / self.uk_scale[1],
                True))
       	self.constraints.append(
	    constraints(
		-300 / self.uk_scale[2],
		300 / self.uk_scale[2], True))
        
        self.guess = None



    def rescale(self, x):
        """Rescale the unknowns with the scaling factors.

        Rescales each unknown with the corresponding scaling factor and
        (applies a 1/x transformation for the time constants of the
        exponentials), yielding a result in milliseconds.

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
        tmp_x[0] = x[0, ...] * self.uk_scale[0]
        tmp_x[1] = x[1, ...] * self.uk_scale[1]
       	tmp_x[2] = x[2, ...] * self.uk_scale[2]

        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return {"data": tmp_x,
                "unknown_name": ["M0", "R2s", "B0"],
                "real_valued": const}

    def _execute_forward_3D(self, x):
        M0 = x[0, ...] * self.uk_scale[0]
        R2s = x[1, ...] * self.uk_scale[1]
       	B0 = x[2, ...] * self.uk_scale[2]

        S = M0 * np.exp(-self.TE * (R2s - 1j*B0))
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=self._DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        M0 = x[0, ...]
        R2s = x[1, ...]
       	B0 = x[2, ...]
        
        grad_M0 = self.uk_scale[0] * np.exp(-self.TE * (
                R2s * self.uk_scale[1] - 1j*B0*self.uk_scale[2]))

        grad_R2s = -self.uk_scale[0] * M0 * \
            self.TE * self.uk_scale[1] * np.exp(-self.TE * (R2s * self.uk_scale[1] - 1j*B0*self.uk_scale[2]))
	
       	grad_B0 = self.uk_scale[0] * M0 * \
	    self.TE * self.uk_scale[2] * 1j * np.exp(-self.TE * (R2s * self.uk_scale[1] - 1j*B0*self.uk_scale[2]))

        grad = np.array([grad_M0, grad_R2s, grad_B0], dtype=self._DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def computeInitialGuess(self, **kwargs):
        """Initialize unknown array for the fitting.

        This function provides an initial guess for the fitting.

        Parameters
        ----------
          args : list of objects
            Serves as universal interface. No objects need to be passed
            here.
        """
        self.dscale = kwargs["dscale"]
        
        phase = np.angle(kwargs["images"])
        for j in range(phase.shape[0]):
            phase[j,:] = unwrap_phase(phase[j].squeeze())
                       
        mat_te = 1/(self.TE.squeeze()@self.TE.squeeze())*self.TE.squeeze()
        
        del_b0 = np.zeros(phase.shape[1:])
        for i in range(phase.shape[-3]):
            for j in range(phase.shape[-2]):
                for k in range(phase.shape[-1]):
                    del_b0[i,j,k] = mat_te@phase[:,i,j,k]
                    
        # import ipdb
        # import pyqmri
        # import matplotlib.pyplot as plt
        # ipdb.set_trace()
        
        
        test_M0 = 1e-5 * np.ones(
            (self.NSlice, self.dimY, self.dimX),
            dtype=self._DTYPE)
        test_R2s = 20 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
       	test_B0 = del_b0

        self.guess = np.array([
            test_M0,
            test_R2s,
       	    test_B0], dtype=self._DTYPE)
