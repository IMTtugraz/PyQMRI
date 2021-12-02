#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyqmri.models.template import BaseModel, constraints
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


class Model(BaseModel):
    def __init__(self, par):
        super().__init__(par)
        self.b = np.ones((self.NScan, 1, 1, 1))
        try:
            # self.NScan = par["T2PREP"].size
            for i in range(self.NScan):
                self.b[i, ...] = par["T2PREP"][i] * np.ones((1, 1, 1))
        except BaseException:
            # self.NScan = par["b_value"].size
            for i in range(self.NScan):
                self.b[i, ...] = par["b_value"][i] * np.ones((1, 1, 1))
        if np.max(self.b) > 100:
            self.b /= 1000
        self.uk_scale = []
        par["unknowns_TGV"] = 2
        par["unknowns_H1"] = 0 
        par["unknowns"] = par["unknowns_TGV"] + par["unknowns_H1"]
        self.unknowns = par["unknowns"]
        
        for i in range(par["unknowns_TGV"] + par["unknowns_H1"]):
            self.uk_scale.append(1)
        try:
            self.b0 = np.flip(
                np.transpose(
                    par["file"]["b0"][()], (0, 2, 1)), 0)
        except KeyError:
            print("No b0 image provided")
            self.b0 =  None

        self.constraints.append(
            constraints(
                0 / self.uk_scale[0],
                1e10 / self.uk_scale[0],
                False))
        self.constraints.append(
            constraints(
                (0 / self.uk_scale[1]),
                (5 / self.uk_scale[1]),
                True))
#        for j in range(phase_maps):
#            self.constraints.append(constraints(
#                (-2*np.pi / self.uk_scale[-phase_maps + j]),
#                (2*np.pi / self.uk_scale[-phase_maps + j]), True))

    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_forward_3D(self, x):
        ADC = x[1, ...] * self.uk_scale[1]
        S = x[0, ...] * self.uk_scale[0] * np.exp(-self.b * ADC)

        S *= self.phase

        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=self._DTYPE)
        return S*self.dscale

    def _execute_gradient_3D(self, x):
        M0 = x[0, ...]
        ADC = x[1, ...]
        grad_M0 = np.exp(-self.b * (ADC * self.uk_scale[1])) * self.uk_scale[0]

        grad_M0 *= self.phase
        grad_ADC = -grad_M0 * M0 * self.b * self.uk_scale[1]

        grad = np.array([grad_M0, grad_ADC],
                        dtype=self._DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
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
            ADC = 1 * np.ones(kwargs['images'].shape[-3:], dtype = self._DTYPE)
        else:
            #custom initial guess
            ADC = kwargs['initial_guess'][0] * np.ones(kwargs['images'].shape[-3:], dtype = self._DTYPE)

        x = np.array((test_M0, ADC))
        self.guess = x
