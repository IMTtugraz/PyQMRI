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

        self.outdir = par["outdir"]

        self.b = np.ones((self.NScan, 1, 1, 1))
        self.dir = par["DWI_dir"].T
        for i in range(self.NScan):
            self.b[i, ...] = par["b_value"][i] * np.ones((1, 1, 1))

        if np.max(self.b) > 100:
            self.b /= 1000

        self.dir = self.dir[:, None, None, None, :]
        par["unknowns_TGV"] = 4
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"] + par["unknowns_H1"]
        self.unknowns = par["unknowns"]
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
                1e5 / self.uk_scale[0],
                False))
        self.constraints.append(
            constraints(
                (0 / self.uk_scale[1]),
                (5 / self.uk_scale[1]),
                True))
        self.constraints.append(
            constraints(
                (0 / self.uk_scale[2]), 
                (1 / self.uk_scale[2]),
                True))
        self.constraints.append(
            constraints(
                ( 5+1e-5 / self.uk_scale[3]),  
                (300 / self.uk_scale[3]),
                True))
        
        self.guess = None
        self.phase = None

    def rescale(self, x):
        """Rescale the unknowns with the scaling factors. 
        
        Rescales each unknown with the corresponding scaling factor.
        
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
        tmp_x[3] = x[3, ...] * self.uk_scale[3]
        
        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return{"data":tmp_x, 
               "unknown_name": ["M0", "ADC", "f", "ADC_ivim"],
               "real_valued": const}

    def _execute_forward_3D(self, x):        
        ADC = x[1, ...] * self.uk_scale[1]

        S = (x[0, ...] * self.uk_scale[0] * (
                x[2, ...] * self.uk_scale[2]
                * np.exp(-(x[3, ...] * self.uk_scale[3]) * self.b)
                + (1-x[2, ...] * self.uk_scale[2])
                * np.exp(- ADC * self.b)
             )).astype(self._DTYPE)

        S *= self.phase
        S[~np.isfinite(S)] = 0
        return S*self.dscale

    def _execute_gradient_3D(self, x):
        ADC = x[1, ...] * self.uk_scale[1]

        grad_M0 = self.uk_scale[0] * (
            x[2, ...] * self.uk_scale[2]
            * np.exp(- (x[3, ...] * self.uk_scale[3]) * self.b)
            + (1-x[2, ...] * self.uk_scale[2])
            * np.exp(- ADC * self.b))

        grad_ADC = x[0, ...] * self.uk_scale[0] * (
            - self.b * self.uk_scale[1] * np.exp(- ADC * self.b) + (
                x[2, ...] * self.b * self.uk_scale[2] * self.uk_scale[1] * np.exp(- ADC * self.b)))
        

        grad_f = (x[0, ...] * self.uk_scale[0] * self.uk_scale[2] * (
            np.exp(-(x[3, ...] * self.uk_scale[3]) * self.b)
            - np.exp(- ADC * self.b)))

        grad_ADC_ivim = (
            -x[0, ...] * self.b*self.uk_scale[0] * self.uk_scale[3] * (
                x[2, ...] * self.uk_scale[2] *
                np.exp(- (x[3, ...] * self.uk_scale[3]) * self.b))
            )

        grad = np.array(
            [grad_M0,
             grad_ADC,
             grad_f,
             grad_ADC_ivim], dtype=self._DTYPE)
        grad[~np.isfinite(grad)] = 0
        grad *= self.phase
        return grad*self.dscale


    def computeInitialGuess(self, **kwargs):
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
        
        self.phase = np.exp(1j*(np.angle(kwargs['images'])-np.angle(kwargs['images'][0])))
        self.dscale = kwargs["dscale"]
        if self.b0 is not None:
            test_M0 = self.b0
        else:
            test_M0 = kwargs['images'][0]/self.dscale
            

        if np.allclose(kwargs['initial_guess'],-1):  
            #default setting
            ADC = 1 * np.ones(kwargs['images'].shape[-3:], dtype=self._DTYPE)
            f = 0.2 * np.ones(kwargs['images'].shape[-3:], dtype=self._DTYPE)
            ADC_ivim = 50 * np.ones(kwargs['images'].shape[-3:], dtype=self._DTYPE)
        else:
            assert len(kwargs['initial_guess']) == self.unknowns-1

            ADC  = kwargs['initial_guess'][0] *np.ones(
                kwargs['images'].shape[-3:], dtype=self._DTYPE)
            f = kwargs['initial_guess'][1] * np.ones(
                kwargs['images'].shape[-3:], dtype=self._DTYPE)
            ADC_ivim = kwargs['initial_guess'][2] * np.ones(
                kwargs['images'].shape[-3:], dtype=self._DTYPE)
        self.weights = kwargs["weights"]
        with open(self.outdir+"initial_guess.txt", 'w') as file:
            file.write('ADC '+np.array2string(np.absolute(np.unique(ADC)))+' \n')
            file.write('f '+np.array2string(np.absolute(np.unique(f)))+' \n')
            file.write('Ds '+np.array2string(np.absolute(np.unique(ADC_ivim)))+'\n')    
            file.write("Weights:" + np.array2string(self.weights))

        x = np.array(
                [
                    test_M0 / self.uk_scale[0],
                    ADC,
                    f,
                    ADC_ivim],
                dtype=self._DTYPE)
        self.guess = x
