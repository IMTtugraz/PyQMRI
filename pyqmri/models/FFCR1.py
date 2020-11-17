
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyqmri.models.template import BaseModel, constraints
plt.ion()



class Model(BaseModel):
    def __init__(self, par):
        super().__init__(par)
        self.constraints = []
        self.t = par["t"]
        self.b = par["b"]

        if len(self.t.shape)<2:
            self.b = self.b[None,:]
            self.t = self.t[None,:]

        self.numT1Scale = len(self.b)-1

        par["unknowns_TGV"] = 2 + len(self.b)
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]

        self.unknowns = par["unknowns"]

        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        self.constraints.append(
            constraints(-10,
                        10,
                        False))
        self.constraints.append(
            constraints(0,
                        np.inf,
                        True))
        self.constraints.append(
            constraints(1/2000,
                        1/10,
                        True))
        for j in range(self.numT1Scale):
            self.constraints.append(
                constraints(0,
                            100,
                            True))

    def rescale(self, x):
        tmp_x = np.copy(x)
        tmp_x[0] *= self.uk_scale[0]
        tmp_x[1] *= self.uk_scale[1]
        tmp_x[2] = 1 / (tmp_x[2] * self.uk_scale[2])
        ukname = ["C", "alpha", "T1_1"]
        for j in range(self.numT1Scale):
            tmp_x[3+j] *= self.uk_scale[3+j]
            ukname.append("T1_"+str(2+j))
        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return {"data": tmp_x,
                "unknown_name": ukname,
                "real_valued": const}

    def _execute_forward_2D(self, x, islice):
        pass

    def _execute_gradient_2D(self, x, islice):
        pass

    def _execute_forward_3D(self, x):
        S = np.zeros(
            (self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=self._DTYPE)
        t = self.t[0][:, None, None, None]
        S[:len(t)] = (
            -x[0] * self.uk_scale[0]
            * np.exp(-t * x[2] * self.uk_scale[2])
            + (1 - np.exp(-t * x[2] * self.uk_scale[2]))
            * x[1] * self.uk_scale[1]*self.b[0])
        for j in range(self.numT1Scale):
            offset = len(self.t[j+1])
            t = self.t[j+1][:, None, None, None]
            S[offset*(j+1):offset*(j+2)] = (
                -x[0] * self.uk_scale[0]
                * np.exp(-t
                         * x[2] * self.uk_scale[2]
                         * x[3+j] * self.uk_scale[3+j])
                + (1 - np.exp(-t
                              * x[2] * self.uk_scale[2]
                              * x[3+j] * self.uk_scale[3+j]))
                * x[1] * self.uk_scale[1]*self.b[1+j]
                )
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=self._DTYPE)
        return S

    def _execute_gradient_3D(self, x):

        gradM0 = self._gradM0(x)
        gradXi = self._gradXi(x)
        gradR1 = self._gradR1(x)
        gradCx = np.zeros(
            (self.numT1Scale,
             self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=self._DTYPE)
        for j in range(self.numT1Scale):
            self._gradCx(gradCx, x, j)
        gradCx[~np.isfinite(gradCx)] = 1e-20

        grad = np.concatenate(
            (np.array([gradM0, gradXi, gradR1], dtype=self._DTYPE),
             gradCx), axis=0)
        return grad

    def _gradM0(self, x):
        grad = np.zeros(
            (self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=self._DTYPE)
        t = self.t[0][:, None, None, None]
        grad[:len(t)] = (
            - self.uk_scale[0]
            * np.exp(-t * x[2] * self.uk_scale[2])
            )
        for j in range(self.numT1Scale):
            offset = len(self.t[j+1])
            t = self.t[j+1][:, None, None, None]
            grad[offset*(j+1):offset*(j+2)] = (
                -self.uk_scale[0]
                * np.exp(-t
                         * x[2] * self.uk_scale[2]
                         * x[3+j] * self.uk_scale[3+j])
                )
        grad[~np.isfinite(grad)] = 1e-20

        return grad

    def _gradXi(self, x):
        grad = np.zeros(
            (self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=self._DTYPE)
        t = self.t[0][:, None, None, None]
        grad[:len(t)] = (
            (1 - np.exp(-t * x[2] * self.uk_scale[2]))
            * self.uk_scale[1]*self.b[0])
        for j in range(self.numT1Scale):
            offset = len(self.t[j+1])
            t = self.t[j+1][:, None, None, None]
            grad[offset*(j+1):offset*(j+2)] = (
                (1 - np.exp(-t
                            * x[2] * self.uk_scale[2]
                            * x[3+j] * self.uk_scale[3+j]))
                * self.uk_scale[1]*self.b[1+j]
                )
        grad[~np.isfinite(grad)] = 1e-20

        return grad

    def _gradR1(self, x):
        grad = np.zeros(
            (self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=self._DTYPE)
        t = self.t[0][:, None, None, None]
        grad[:len(t)] = (
            x[0]*self.uk_scale[0]
            * self.uk_scale[2]*t
            * np.exp(-t * x[2] * self.uk_scale[2])
            + self.uk_scale[2] * self.b[0] * t
            * x[1] * self.uk_scale[1]
            * np.exp(- x[2] * self.uk_scale[2] * t)
            )
        for j in range(self.numT1Scale):
            offset = len(self.t[j+1])
            t = self.t[j+1][:, None, None, None]
            grad[offset*(j+1):offset*(j+2)] = (
                x[3+j] * self.uk_scale[3+j] * x[0] * self.uk_scale[0]
                * self.uk_scale[2]*t
                * np.exp(-t
                         * x[2] * self.uk_scale[2]
                         * x[3+j] * self.uk_scale[3+j])
                + x[3+j] * self.uk_scale[3+j] * self.uk_scale[2]
                * self.b[1+j] * t
                * x[1] * self.uk_scale[1]
                * np.exp(-t
                         * x[2] * self.uk_scale[2]
                         * x[3+j] * self.uk_scale[3+j])
                )
        grad[~np.isfinite(grad)] = 1e-20

        return grad

    def _gradCx(self, grad, x, ind):
        offset = len(self.t[ind+1])
        t = self.t[ind+1][:, None, None, None]
        grad[ind, (ind+1)*offset:(ind+2)*offset] = (
            x[0]*self.uk_scale[0] * x[2] * self.uk_scale[2]
            * self.uk_scale[3+ind]*t
            * np.exp(-t
                     * x[2] * self.uk_scale[2]
                     * x[3+ind] * self.uk_scale[3+ind])
            + self.uk_scale[3+ind] * self.b[1+ind] * t
            * x[2] * self.uk_scale[2]
            * x[1] * self.uk_scale[1]
            * np.exp(- t
                     * x[2] * self.uk_scale[2]
                     * x[3+ind] * self.uk_scale[3+ind])
            )

    def computeInitialGuess(self, *args):
        self.dscale = args[1]
        test_M0 = 1e-3*np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        self.constraints[0].update(1/args[1])
        test_Xi = 1*np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        # self.constraints[1].update(1/args[1])
        test_R1 = 1/500 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        test_Cx = []
        self.b *= args[1]
        for j in range(self.numT1Scale):
            test_Cx.append(1 *
                np.ones(
                    (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE))
        self.guess = np.array(
            [test_M0, test_Xi, test_R1] + test_Cx, dtype=self._DTYPE)