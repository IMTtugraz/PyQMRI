# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:19:12 2021

@author: ssrauh
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyqmri.models.template import BaseModel, constraints
import numpy as np


class Model(BaseModel):
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
        par["unknowns_TGV"] = 8
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
            
            
        #fix D*
        self.Dstar = 100
        
        
        self.constraints.append(
            constraints(
                0 / self.uk_scale[0],
                10 / self.uk_scale[0],
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
                (0 / self.uk_scale[7]),
                (1 / self.uk_scale[7]),
                True))
        

    def rescale(self, x):
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
        
        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return{"data": tmp_x, 
               "unknown_name": ["M0", "ADC_x", "ADC_xy", "ADC_y", "ADC_xz", 
                                "ADC_z", "ADC_yz", "f"],
               "real_valued": const}
        
    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

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
                * np.exp(-self.Dstar * self.b)
                + (1-x[7, ...] * self.uk_scale[7])
                * np.exp(- ADC * self.b)
             )).astype(self._DTYPE)

        S *= self.phase
        S[~np.isfinite(S)] = 0
        return S

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
            * np.exp(- self.Dstar * self.b)
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
            np.exp(-self.Dstar * self.b)
            - np.exp(- ADC * self.b)))


        grad = np.array(
            [grad_M0,
             grad_ADC_x,
             grad_ADC_xy,
             grad_ADC_y,
             grad_ADC_xz,
             grad_ADC_z,
             grad_ADC_yz,
             grad_f], dtype=self._DTYPE)
        grad[~np.isfinite(grad)] = 0
        grad *= self.phase
        return grad


    def computeInitialGuess(self, *args):
        self.phase = np.exp(1j*(np.angle(args[0])-np.angle(args[0][0])))
        if self.b0 is not None:
            test_M0 = self.b0
        else:
            test_M0 = args[0][0]
        
        if np.allclose(args[2],-1):
            # default setting
            ADC = 1 * np.ones(args[0].shape[-3:], dtype=self._DTYPE)
            f = 0.1 * np.ones(args[0].shape[-3:], dtype=self._DTYPE)
        else:
            assert len(args[2]) == self.unknowns-1
            
            ADC = args[2][0] * np.ones(args[0].shape[-3:], dtype=self._DTYPE)
            f = args[2][-1] * np.ones(args[0].shape[-3:], dtype=self._DTYPE)


        x = np.array(
                [
                    test_M0 / self.uk_scale[0],
                    ADC,
                    0 * ADC,
                    ADC,
                    0 * ADC,
                    ADC,
                    0 * ADC,
                    f],
                dtype=self._DTYPE)
        self.guess = x
