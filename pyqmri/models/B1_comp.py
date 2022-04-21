#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the variable flip angle model for T1 fitting."""
import numpy as np
from pyqmri.models.template import BaseModel, constraints
from skimage.restoration import unwrap_phase


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


        par["unknowns_TGV"] = self.NScan+6
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]
        
        self.TE = par["TE"]
        self.bss_sign = -par["bss_sign"]
        self.even_odd_sign = par["even_odd_sign"]
        self.pos_offset = par["pos_offset"]
        self.neg_offset = par["neg_offset"]
        self.same_sign = par["same_sign"]
        self.kbs = par["kbs"]
        
        self.unknowns = par["unknowns"]

        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        for j in range(self.NScan):
            self.constraints.append(
                constraints(0,
                            1e5,
                            True))
                
        self.constraints.append(
            constraints(-np.pi,
                        np.pi,
                        True))
        self.constraints.append(
            constraints(-10*np.pi,
                        10*np.pi,
                        True))
        self.constraints.append(
            constraints(-10*np.pi,
                        10*np.pi,
                        True))
        self.constraints.append(
            constraints(-10*np.pi,
                        10*np.pi,
                        True))
        self.constraints.append(
            constraints(-10*np.pi,
                        10*np.pi,
                        True))
        self.constraints.append(
            constraints(-10*np.pi,
                        10*np.pi,
                        True))
        
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
            
        # tmp_x[:self.NScan] = np.angle(tmp_x[:self.NScan])
        test = (tmp_x[self.NScan]).real        
        tmp_x[self.NScan] = np.sqrt(2*test/self.kbs*1e12)
        tmp_x[self.NScan][~np.isfinite(tmp_x[self.NScan])] = 0
       
        tmp_names = []
        for j in range(self.NScan):
            tmp_names.append("Image_"+str(j))
        
        return {"data": tmp_x,
                "unknown_name":   tmp_names+["BSS",
                                  "DeltaB0", "Even/Odd",
                                  "PosOffset", "NegOffset",
                                  "SameSign"],
                "real_valued": const}

        

    def _execute_forward_3D(self, x):
        
        
        S = (x[:self.NScan]*self.uk_scale[:self.NScan, None, None, None]
            # * np.exp(-self.TE*x[self.NScan]*self.uk_scale[self.NScan])
            * np.exp(1j*(2*np.pi+self.bss_sign*x[self.NScan]*self.uk_scale[self.NScan] 
                          + self.TE*x[self.NScan+1]*self.uk_scale[self.NScan+1] 
                          + self.even_odd_sign*x[self.NScan+2]*self.uk_scale[self.NScan+2] 
                          + self.pos_offset*x[self.NScan+3]*self.uk_scale[self.NScan+3] 
                          + self.neg_offset*x[self.NScan+4]*self.uk_scale[self.NScan+4] 
                          + self.same_sign*x[self.NScan+5]*self.uk_scale[self.NScan+5])))

        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=self._DTYPE)
        return S*self.dscale

    def _execute_gradient_3D(self, x):
               
                
        grad_tmp = (self.uk_scale[:self.NScan, None, None, None]
            # * np.exp(-self.TE*x[self.NScan]*self.uk_scale[self.NScan])
            * np.exp(1j*(2*np.pi+self.bss_sign*x[self.NScan]*self.uk_scale[self.NScan] 
                          + self.TE*x[self.NScan+1]*self.uk_scale[self.NScan+1] 
                          + self.even_odd_sign*x[self.NScan+2]*self.uk_scale[self.NScan+2] 
                          + self.pos_offset*x[self.NScan+3]*self.uk_scale[self.NScan+3] 
                          + self.neg_offset*x[self.NScan+4]*self.uk_scale[self.NScan+4] 
                          + self.same_sign*x[self.NScan+5]*self.uk_scale[self.NScan+5])))
        
        grad_M0 = np.zeros((self.NScan,)+grad_tmp.shape, dtype=self._DTYPE)
        
        for j in range(self.NScan):
            grad_M0[j,j] = grad_tmp[j]
        
        
        # grad_R2s = - grad_tmp*x[:self.NScan] * self.TE*self.uk_scale[self.NScan]
        
        grad_BSS = grad_tmp*x[:self.NScan] * 1j*self.bss_sign*self.uk_scale[self.NScan]
        
        grad_delB0 = grad_tmp*x[:self.NScan] * 1j*self.TE*self.uk_scale[self.NScan+1]
        
        grad_even_odd = grad_tmp*x[:self.NScan] * 1j*self.even_odd_sign*self.uk_scale[self.NScan+2]
        
        grad_offset_pos = grad_tmp*x[:self.NScan] * 1j*self.pos_offset*self.uk_scale[self.NScan+3]
        
        grad_offset_neg = grad_tmp*x[:self.NScan] * 1j*self.neg_offset*self.uk_scale[self.NScan+4]
        
        grad_same_sign = grad_tmp*x[:self.NScan] * 1j*self.same_sign*self.uk_scale[self.NScan+5]
        
                
        grad = np.concatenate((grad_M0,
                               np.array([grad_BSS, grad_delB0, 
                                        grad_even_odd, grad_offset_pos, 
                                        grad_offset_neg,
                                        grad_same_sign], dtype=self._DTYPE)
                               )
                              )
        grad[~np.isfinite(grad)] = 1e-20
        return grad*self.dscale

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
        M0_init = np.abs(kwargs["images"])/self.dscale
        R2_init = 1/50 * np.ones(
            (M0_init.shape[-3:]), dtype=self._DTYPE)
        
        phase = np.angle(kwargs["images"])
        for j in range(phase.shape[0]):
            phase[j,:] = unwrap_phase(phase[j].squeeze())
            
        bss_est = (-phase[2]+phase[int(self.NScan/2+2)])/2
        
        while np.sum(bss_est[bss_est<0]):
            bss_est[bss_est<0] += 2*np.pi
                       
        mat_te = 1/(self.TE.squeeze()@self.TE.squeeze())*self.TE.squeeze()
        
        del_b0 = np.zeros(phase.shape[1:])
        for i in range(phase.shape[-3]):
            for j in range(phase.shape[-2]):
                for k in range(phase.shape[-1]):
                    del_b0[i,j,k] = mat_te@phase[:,i,j,k]
                    
                    
        phase_pos = phase[0] -del_b0*self.TE[0]
        phase_neg = phase[int(self.NScan/2)] -del_b0*self.TE[int(self.NScan/2)]
                    
        # M0_init = M0_init*(
        #     np.conj(np.exp(1j*del_b0*self.TE))*
        #     np.conj(np.exp(1j*self.bss_sign*bss_est))*
        #     np.conj(np.exp(1j*self.pos_offset*phase_pos))*
        #     np.conj(np.exp(1j*self.neg_offset*phase_neg))
        #     )
        
        self.guess = np.concatenate((M0_init.astype(self._DTYPE),
                            np.array([
                            bss_est,
                            del_b0,
                            0*R2_init,
                            phase_pos,
                            phase_neg,
                            0*R2_init], dtype=self._DTYPE)
                            )
                           )
