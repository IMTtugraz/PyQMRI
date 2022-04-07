#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyqmri.models.template import BaseModel, constraints
import numpy as np


class Model(BaseModel):
    """Combined diffusion tensor and IVIM model for MRI parameter 
    quantification.

    This class holds a DTI-IVIM model for fitting complex MRI data.
    It realizes a forward application of the analytical signal expression
    and the partial derivatives with respesct to each parameter of interest,
    as required by the abstract methods in the BaseModel.

    The fitting of the diffusion tensor is based on the Cholesky decomposition
    of the DTI tensor to achiev an implicit positive definite constrained on 
    each DTI tensor component.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the sequence related parametrs,
        e.g. TR, TE, TI, to fully describe the acquisitio process

    Attributes
    ----------
      b : float
        b values for each diffusion direction.
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
        par["unknowns_TGV"] = 9
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
                1e10 / self.uk_scale[0],
                False))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[1]),
                (10e0 / self.uk_scale[1]),
                True))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[2]),
                (10e0 / self.uk_scale[2]),
                True))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[3]),
                (10e0 / self.uk_scale[3]),
                True))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[4]),
                (10e0 / self.uk_scale[4]),
                True))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[5]),
                (10e0 / self.uk_scale[5]),
                True))
        self.constraints.append(
            constraints(
                (-10e0 / self.uk_scale[6]),
                (10e0 / self.uk_scale[6]),
                True))
        self.constraints.append(
            constraints(
                #(0 / self.uk_scale[7]),
                (1e-4 / self.uk_scale[7]),     
                (0.9999 / self.uk_scale[7]),
                True))
        self.constraints.append(
            constraints(
                (5.001 / self.uk_scale[8]),
                (300 / self.uk_scale[8]),
                True))
        
        # par["weights"] = 1*np.array([1]*len(self.constraints),dtype=par["DTYPE_real"])
        # par["weights"][0] *= 1e1
        # par["weights"][1:-2] *= 0.5
        # par["weights"][-2:] *= 0.1
        self.guess = None
        self.phase = None

    def rescale(self, x):   
        """Rescale the unknowns with the scaling factors.

        Rescales each unknown with the corresponding scaling factor. As the
        DTI tensor is fitted using the Cholesky decompotion, each entry
        of the original tensor is recovered by combining the appropriate
        Cholesky factors after rescaling.

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
        tmp_x[1] = (np.real(x[1, ...]**2) * self.uk_scale[1]**2)
        tmp_x[2] = (np.real(x[2, ...] * self.uk_scale[2] *
                          x[1, ...] * self.uk_scale[1]))
        tmp_x[3] = (np.real(x[2, ...]**2 * self.uk_scale[2]**2 +
                         x[3, ...]**2 * self.uk_scale[3]**2))
        tmp_x[4] = (np.real(x[4, ...] * self.uk_scale[4] *
                         x[1, ...] * self.uk_scale[1]))
        tmp_x[5] = (np.real(x[4, ...]**2 * self.uk_scale[4]**2 +
                         x[5, ...]**2 * self.uk_scale[5]**2 +
                         x[6, ...]**2 * self.uk_scale[6]**2))
        tmp_x[6] = (np.real(x[2, ...] * self.uk_scale[2] *
                          x[4, ...] * self.uk_scale[4] +
                          x[6, ...] * self.uk_scale[6] *
                          x[3, ...] * self.uk_scale[3]))
        tmp_x[7] = x[7, ...] * self.uk_scale[7]
        tmp_x[8] = x[8, ...] * self.uk_scale[8]
        
        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return{"data": tmp_x, 
               "unknown_name": ["M0", "ADC_x", "ADC_xy", "ADC_y", "ADC_xz", 
                                "ADC_z", "ADC_yz", "f", "ADC_ivim"],
               "real_valued": const}
        
    def _execute_forward_3D(self, x):
        ADC = x[1, ...]**2 * self.uk_scale[1]**2 * self.dir[..., 0]**2 + \
              (x[2, ...]**2 * self.uk_scale[2]**2 +
               x[3, ...]**2 * self.uk_scale[3]**2) * self.dir[..., 1]**2 + \
              (x[4, ...]**2 * self.uk_scale[4]**2 +
               x[5, ...]**2 * self.uk_scale[5]**2 +
               x[6, ...]**2 * self.uk_scale[6]**2) * self.dir[..., 2]**2 +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[1, ...] * self.uk_scale[1]) * \
              self.dir[..., 0] * self.dir[..., 1] + \
              2 * (x[4, ...] * self.uk_scale[4] *
                   x[1, ...] * self.uk_scale[1]) *\
              self.dir[..., 0] * self.dir[..., 2] +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[4, ...] * self.uk_scale[4] +
                   x[6, ...] * self.uk_scale[6] *
                   x[3, ...] * self.uk_scale[3]) * \
              self.dir[..., 1] * self.dir[..., 2]

        S = (x[0, ...] * self.uk_scale[0] * (
                x[7, ...] * self.uk_scale[7]
                * np.exp(-(x[8, ...] * self.uk_scale[8]) * self.b)
                + (1-x[7, ...] * self.uk_scale[7])
                * np.exp(- ADC * self.b)
             )).astype(self._DTYPE)

        S *= self.phase
        S[~np.isfinite(S)] = 0
        return S*self.dscale

    def _execute_gradient_3D(self, x):
        ADC = x[1, ...]**2 * self.uk_scale[1]**2 * self.dir[..., 0]**2 + \
              (x[2, ...]**2 * self.uk_scale[2]**2 +
               x[3, ...]**2 * self.uk_scale[3]**2) * self.dir[..., 1]**2 + \
              (x[4, ...]**2 * self.uk_scale[4]**2 +
               x[5, ...]**2 * self.uk_scale[5]**2 +
               x[6, ...]**2 * self.uk_scale[6]**2) * self.dir[..., 2]**2 +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[1, ...] * self.uk_scale[1]) * \
              self.dir[..., 0] * self.dir[..., 1] + \
              2 * (x[4, ...] * self.uk_scale[4] *
                   x[1, ...] * self.uk_scale[1]) *\
              self.dir[..., 0] * self.dir[..., 2] +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[4, ...] * self.uk_scale[4] +
                   x[6, ...] * self.uk_scale[6] *
                   x[3, ...] * self.uk_scale[3]) * \
              self.dir[..., 1] * self.dir[..., 2]

        grad_M0 = self.uk_scale[0] * (
            x[7, ...] * self.uk_scale[7]
            * np.exp(- (x[8, ...] * self.uk_scale[8]) * self.b)
            + (1-x[7, ...] * self.uk_scale[7])
            * np.exp(- ADC * self.b))

        grad_ADC_x = -x[0, ...] * self.b * grad_M0 * \
            (2 * x[1, ...] * self.uk_scale[1]**2 * self.dir[..., 0]**2 +
             2 * self.uk_scale[1] * x[2, ...] * self.uk_scale[2] *
             self.dir[..., 0] * self.dir[..., 1] +
             2 * self.uk_scale[1] * x[4, ...] * self.uk_scale[4] *
             self.dir[..., 0] * self.dir[..., 2])
        grad_ADC_xy = -x[0, ...] * self.b * grad_M0 * \
            (2 * x[1, ...] * self.uk_scale[1] * self.uk_scale[2] *
             self.dir[..., 0] * self.dir[..., 1] +
             2 * x[2, ...] * self.uk_scale[2]**2 *
             self.dir[..., 1]**2 +
             2 * self.uk_scale[2] * x[4, ...] * self.uk_scale[4] *
             self.dir[..., 1] * self.dir[..., 2])

        grad_ADC_y = -x[0, ...] * self.b * grad_M0 *\
            (2 * x[3, ...] * self.uk_scale[3]**2 * self.dir[..., 1]**2 +
             2 * self.uk_scale[3] * x[6, ...] * self.uk_scale[6] *
             self.dir[..., 1] * self.dir[..., 2])
        grad_ADC_xz = -x[0, ...] * self.b * grad_M0 *\
            (2 * x[1, ...] * self.uk_scale[1] * self.uk_scale[4] *
             self.dir[..., 0] * self.dir[..., 2] +
             2 * x[2, ...] * self.uk_scale[2] * self.uk_scale[4] *
             self.dir[..., 1] * self.dir[..., 2] +
             2 * x[4, ...] * self.uk_scale[4]**2 * self.dir[..., 2]**2)

        grad_ADC_z = -2 * x[5, ...] * self.uk_scale[5]**2 *\
            x[0, ...]*self.b*self.dir[..., 2]**2*grad_M0

        grad_ADC_yz = - x[0, ...] * self.b * grad_M0 *\
            (2 * x[3, ...] * self.uk_scale[3] * self.uk_scale[6] *
             self.dir[..., 1] * self.dir[..., 2] +
             2 * x[6, ...] * self.uk_scale[6]**2 * self.dir[..., 2]**2)

        grad_f = (x[0, ...] * self.uk_scale[0] * self.uk_scale[7] * (
            np.exp(-(x[8, ...] * self.uk_scale[8]) * self.b)
            - np.exp(- ADC * self.b)))

        grad_ADC_ivim = (
            -x[0, ...] * self.b*self.uk_scale[0] * self.uk_scale[8] * (
                x[7, ...] * self.uk_scale[7] *
                np.exp(- (x[8, ...] * self.uk_scale[8]) * self.b))
            )

        grad = np.array(
            [grad_M0,
             grad_ADC_x,
             grad_ADC_xy,
             grad_ADC_y,
             grad_ADC_xz,
             grad_ADC_z,
             grad_ADC_yz,
             grad_f,
             grad_ADC_ivim], dtype=self._DTYPE)
        grad[~np.isfinite(grad)] = 0
        grad *= self.phase
        return grad*self.dscale

    def computeInitialGuess(self, **kwargs):
        self.phase = np.exp(1j*(np.angle(kwargs['images'])-np.angle(kwargs['images'][0])))
        self.dscale = kwargs["dscale"]
        if self.b0 is not None:
            test_M0 = self.b0
        else:
            test_M0 = kwargs['images'][0]/self.dscale
        
        if np.allclose(kwargs['initial_guess'],-1):
            #default setting
            ADC = 1 * np.ones(kwargs['images'].shape[-3:], dtype=self._DTYPE)
            f = 0.5 * np.ones(kwargs['images'].shape[-3:], dtype=self._DTYPE)
            ADC_ivim = 50 * np.ones(kwargs['images'].shape[-3:], dtype=self._DTYPE)
        else:
            assert len(kwargs['initial_guess']) == 3
            #ADC into Cholesky coefficient 
            ADC = np.sqrt(kwargs['initial_guess'][0]) * np.ones(
                kwargs['images'].shape[-3:], dtype=self._DTYPE)
            f = kwargs['initial_guess'][-2] * np.ones(
                kwargs['images'].shape[-3:], dtype=self._DTYPE)
            ADC_ivim = kwargs['initial_guess'][-1] * np.ones(
                kwargs['images'].shape[-3:], dtype=self._DTYPE)
            
        self.weights = kwargs["weights"]
       
        with open(self.outdir+"initial_guess.txt", 'w') as file:
            file.write('ADC '+np.array2string(np.absolute(np.square(np.unique(ADC))))+ ' \n')
            file.write('f '+np.array2string(np.absolute(np.unique(f)))+' \n')
            file.write('Ds '+np.array2string(np.absolute(np.unique(ADC_ivim)))+'\n')    
            file.write("Weights:" + np.array2string(self.weights))

        x = np.array(
                [
                    test_M0 / self.uk_scale[0],
                    ADC,
                    0 * ADC,
                    ADC,
                    0 * ADC,
                    ADC,
                    0 * ADC,
                    f,
                    ADC_ivim],
                dtype=self._DTYPE)
        self.guess = x
