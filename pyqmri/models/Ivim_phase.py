#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the intravoxel incoherent motion  (IVIM) model for fitting."""
from pyqmri.models.template import BaseModel, constraints
import numpy as np


class Model(BaseModel):
    """Intravoxel incoherent motion (IVIM) model for MRI parameter quantification.
    
    This class holds an IVIM model for fitting complex MRI data.
    It realizes a forward application of the analytical signal expression
    and the partial derivatives with respect to each parameter of interest,
    as requierd by the abstract methods in the BaseModel.
    
    The fitting is based on the two-compartment bi-exponential IVIM approach, 
    assuming a pure tissue diffusion and a pseudo-diffusion compartment
    related to perfusion.
    
    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the sequence related parameters, 
        e.g. b-values, to fully describe the acquisition process
        
    Attributes
    ----------
      b : float
        b values for each diffusino direction.
      dir : numpy.array
        The diffusion direction vectors. Assumed to have length 1.
      uk_scale : list of float
        Scaling factors for each unknown to balance the partial derivatives.
      guess : numpy.array
        The initial guess. Needs to be set using "computeInitialGuess"
        prior to fitting.
      phase : numpy.array
        The phase of each diffusion direction relative to the b0 image.
        Estimated during the initial guess using the image series of all
        directions/bvalue pairs.
      b0 : numpy.array
        The b0 image if present in the data file. None else.
    """  
           
    def __init__(self, par):
        super().__init__(par)
        self.NSlice = par['NSlice']

        self.figure_phase = None

        self.b = np.ones((self.NScan, 1, 1, 1))
        self.dir = par["DWI_dir"].T
        for i in range(self.NScan):
            self.b[i, ...] = par["b_value"][i] * np.ones((1, 1, 1))

        if np.max(self.b) > 100:
            self.b /= 1000

        self.dir = self.dir[:, None, None, None, :]
        par["unknowns_TGV"] = 4 + len(self.b)
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"] + par["unknowns_H1"]
        self.unknowns = par["unknowns_TGV"] + par["unknowns_H1"]
        self.uk_scale = []
        for j in range(self.unknowns):
            self.uk_scale.append(1)

        try:
            self.b0 = np.flip(
                np.transpose(par["file"]["b0"][()], (0, 2, 1)), 0)
        except KeyError:
            print("No b0 image provided")
            self.b0 = None

        self.constraints.append(
            constraints(
                0 / self.uk_scale[0],
                10 / self.uk_scale[0],
                False))
        self.constraints.append(
            constraints(
                (0 / self.uk_scale[1]),
                (5 / self.uk_scale[1]),
                True))
        self.constraints.append(
            constraints(
                (0.01 / self.uk_scale[2]),
                (1 / self.uk_scale[2]),
                True))
        self.constraints.append(
            constraints(
                (5 / self.uk_scale[3]),
                (300 / self.uk_scale[3]),
                True))
        
        for j in range(len(self.b)):
            self.constraints.append(
                constraints(
                    (-np.pi / self.uk_scale[3]),
                    (np.pi / self.uk_scale[3]),
                    True))
        
        self.ivim_scale = 1
        
        
    def rescale(self, x):
        tmp_x = np.copy(x)
        for j in range(tmp_x.shape[0]):
            tmp_x[j] = x[j] * self.uk_scale[j]

        
        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return{"data":tmp_x, 
               "unknown_name": ["M0", "ADC", "f", "ADC_ivim"] + len(self.b)*["phase"],
               "real_valued": const}

    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_forward_3D(self, x):        
        ADC = x[1, ...] * self.uk_scale[1]

        S = (x[0, ...] * self.uk_scale[0] * (
                x[2, ...] * self.uk_scale[2]
                * np.exp(-(x[3, ...] * self.uk_scale[3]) * self.b * self.ivim_scale)
                + (1-x[2, ...] * self.uk_scale[2])
                * np.exp(- ADC * self.b)
             )).astype(self._DTYPE)

        S *= np.exp(1j*x[4:]* self.uk_scale[4:][:,None,None,None])
        S[~np.isfinite(S)] = 0
        return S

    def _execute_gradient_3D(self, x):
        ADC = x[1, ...] * self.uk_scale[1]

        grad_M0 = self.uk_scale[0] * (
            x[2, ...] * self.uk_scale[2]
            * np.exp(- (x[3, ...] * self.uk_scale[3]) * self.b * self.ivim_scale)
            + (1-x[2, ...] * self.uk_scale[2])
            * np.exp(- ADC * self.b))

        grad_ADC = x[0, ...] * self.uk_scale[0] * (
            - self.b * self.uk_scale[1] * np.exp(- ADC * self.b) + (
                x[2, ...] * self.b * self.uk_scale[2] * self.uk_scale[1] * np.exp(- ADC * self.b)))
        

        grad_f = (x[0, ...] * self.uk_scale[0] * self.uk_scale[2] * (
            np.exp(-(x[3, ...] * self.uk_scale[3]) * self.b * self.ivim_scale)
            - np.exp(- ADC * self.b)))

        grad_ADC_ivim = (
            -x[0, ...] * self.b*self.uk_scale[0]*self.ivim_scale * self.uk_scale[3] * (
                x[2, ...] * self.uk_scale[2] *
                np.exp(- (x[3, ...] * self.uk_scale[3]) * self.b * self.ivim_scale))
            )

        grad = np.array(
            [grad_M0,
             grad_ADC,
             grad_f,
             grad_ADC_ivim], dtype=self._DTYPE)
        grad[~np.isfinite(grad)] = 0

        phase_grad = np.zeros(((len(self.b),)+grad_M0.shape), dtype=self._DTYPE)
        for j in range(len(self.b)):
            phase_grad[j,j] = x[0]*grad_M0[j]*1j*self.uk_scale[4+j]
        
        grad = np.concatenate((grad, phase_grad), axis=0)
        grad *= np.exp(1j*x[4:]* self.uk_scale[4:][:,None,None,None])
        return grad


    def computeInitialGuess(self, *args):
        """Initialize unknown array for the fiting.
        This function provides an initial guess for the fiting. args[0] is
        assumed to contain the image series which is used for phase 
        correction.
        
        Parameters
        ----------
          args : list of objects
            Assumes the image series at position 0 and optionally computes
            a phase based on the difference between each image series minus
            the first image in the series (Scan i minus Scan 0). This 
            phase correction is needed as each diffusion weighting has a 
            different phase.
        """
        
        phase = np.zeros_like(np.angle(args[0]))
        if self.b0 is not None:
            test_M0 = self.b0
        else:
            test_M0 = 1e-5*np.ones_like(args[0][0])
            

        if np.allclose(args[2],-1):  
            # default setting
            ADC = 1 * np.ones(args[0].shape[-3:], dtype=self._DTYPE)
            f = 0.2 * np.ones(args[0].shape[-3:], dtype=self._DTYPE)
            ADC_ivim = 20 * np.ones(args[0].shape[-3:], dtype=self._DTYPE)
        else:
            assert len(args[2]) == self.unknowns-1

            ADC  = args[2][0] *np.ones(args[0].shape[-3:], dtype=self._DTYPE)
            f = args[2][1] * np.ones(args[0].shape[-3:], dtype=self._DTYPE)
            ADC_ivim = args[2][2] * np.ones(args[0].shape[-3:], dtype=self._DTYPE)
            

        x =np.concatenate((np.array(
                [
                    test_M0 / self.uk_scale[0],
                    ADC,
                    f,
                    ADC_ivim],
                dtype=self._DTYPE),
                phase), axis=0)
        x_scale = np.max(np.abs(x).reshape(x.shape[0], -1), axis=-1)
        x_scale[x_scale==0] = 1
        self.uk_scale = x_scale
        self.guess = x/x_scale[:,None,None,None]
        for uk in range(self.unknowns):
            self.constraints[uk].update(x_scale[uk])
