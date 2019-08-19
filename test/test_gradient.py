#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:26:41 2019

@author: omaier
"""

import pyqmri
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from pyqmri._helper_fun import CLProgram as Program
from pkg_resources import resource_filename
import pyopencl.array as clarray
import numpy as np

DTYPE = np.complex128
DTYPE_real = np.float64


class tmpArgs():
    pass


def setupPar(par):
    par["NScan"] = 10
    par["NC"] = 15
    par["NSlice"] = 20
    par["dimX"] = 128
    par["dimY"] = 128
    par["Nproj"] = 21
    par["N"] = 256
    par["unknowns_TGV"] = 2
    par["unknowns_H1"] = 0
    par["unknowns"] = 2
    par["dz"] = 1
    par["weights"] = [1, 1]


class GradientTest(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = False
        parser.devices = [0]
        parser.use_GPU = True

        par = {}
        pyqmri.pyqmri._setupOCL(parser, par)
        setupPar(par)
        if DTYPE == np.complex128:
            file = open(
                    resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels_double.c'))
        else:
            file = open(
                    resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels.c'))
        prg = Program(
            par["ctx"][0],
            file.read())
        file.close()

        self.grad = pyqmri.operator.OperatorFiniteGradient(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)
        self.grad._ratio[0] = 1
        self.grad._ratio[1] = 1
        self.gradin = np.random.randn(par["unknowns"], par["NSlice"],
                                      par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["unknowns"], par["NSlice"],
                                 par["dimY"], par["dimX"])
        self.divin = np.random.randn(par["unknowns"], par["NSlice"],
                                     par["dimY"], par["dimX"], 4) +\
            1j * np.random.randn(par["unknowns"], par["NSlice"],
                                 par["dimY"], par["dimX"], 4)
        self.gradin = self.gradin.astype(DTYPE)
        self.divin = self.divin.astype(DTYPE)
        self.dz = par["dz"]
        self.queue = par["queue"][0]

    def test_grad_outofplace(self):
        gradx = np.zeros_like(self.gradin)
        grady = np.zeros_like(self.gradin)
        gradz = np.zeros_like(self.gradin)

        gradx[..., :-1] = np.diff(self.gradin, axis=-1)
        grady[..., :-1, :] = np.diff(self.gradin, axis=-2)
        gradz[:, :-1, ...] = np.diff(self.gradin, axis=-3)/self.dz

        grad = np.stack((gradx,
                         grady,
                         gradz), axis=-1)

        inp = clarray.to_device(self.queue, self.gradin)
        outp = self.grad.fwdoop(inp)
        outp = outp.get()

        np.testing.assert_allclose(outp[..., :-1], grad)

    def test_grad_inplace(self):
        gradx = np.zeros_like(self.gradin)
        grady = np.zeros_like(self.gradin)
        gradz = np.zeros_like(self.gradin)

        gradx[..., :-1] = np.diff(self.gradin, axis=-1)
        grady[..., :-1, :] = np.diff(self.gradin, axis=-2)
        gradz[:, :-1, ...] = np.diff(self.gradin, axis=-3)/self.dz

        grad = np.stack((gradx,
                         grady,
                         gradz), axis=-1)

        inp = clarray.to_device(self.queue, self.gradin)
        outp = clarray.to_device(self.queue, self.divin)
        self.grad.fwd(outp, inp)
        outp = outp.get()

        np.testing.assert_allclose(outp[..., :-1], grad)

    def test_adj_outofplace(self):
        inpgrad = clarray.to_device(self.queue, self.gradin)
        inpdiv = clarray.to_device(self.queue, self.divin)

        outgrad = self.grad.fwdoop(inpgrad)
        outdiv = self.grad.adjoop(inpdiv)

        outgrad = outgrad.get()
        outdiv = outdiv.get()

        a = np.vdot(outgrad[..., :-1].flatten(),
                    self.divin[..., :-1].flatten())/self.gradin.size
        b = np.vdot(self.gradin.flatten(), -outdiv.flatten())/self.gradin.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)

    def test_adj_inplace(self):
        inpgrad = clarray.to_device(self.queue, self.gradin)
        inpdiv = clarray.to_device(self.queue, self.divin)

        outgrad = clarray.zeros_like(inpdiv)
        outdiv = clarray.zeros_like(inpgrad)

        self.grad.fwd(outgrad, inpgrad)
        self.grad.adj(outdiv, inpdiv)

        outgrad = outgrad.get()
        outdiv = outdiv.get()

        a = np.vdot(outgrad[..., :-1].flatten(),
                    self.divin[..., :-1].flatten())/self.gradin.size
        b = np.vdot(self.gradin.flatten(), -outdiv.flatten())/self.gradin.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)


class GradientStreamedTest(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = True
        parser.devices = [0]
        parser.use_GPU = True

        par = {}
        pyqmri.pyqmri._setupOCL(parser, par)
        setupPar(par)
        if DTYPE == np.complex128:
            file = open(
                    resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels_streamed_double.c'))
        else:
            file = open(
                    resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels_streamed.c'))

        prg = []
        for j in range(1):
            prg.append(
                Program(
                    par["ctx"][0],
                    file.read()))
        file.close()

        par["par_slices"] = 4

        self.grad = pyqmri.operator.OperatorFiniteGradientStreamed(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)

        for j in range(len(self.grad._ratio)):
            self.grad._ratio[j][0] = 1
            self.grad._ratio[j][1] = 1

        self.gradin = np.random.randn(par["NSlice"], par["unknowns"],
                                      par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NSlice"], par["unknowns"],
                                 par["dimY"], par["dimX"])
        self.divin = np.random.randn(par["NSlice"], par["unknowns"],
                                     par["dimY"], par["dimX"], 4) +\
            1j * np.random.randn(par["NSlice"], par["unknowns"],
                                 par["dimY"], par["dimX"], 4)
        self.gradin = self.gradin.astype(DTYPE)
        self.divin = self.divin.astype(DTYPE)
        self.dz = par["dz"]

    def test_grad_outofplace(self):
        gradx = np.zeros_like(self.gradin)
        grady = np.zeros_like(self.gradin)
        gradz = np.zeros_like(self.gradin)

        gradx[..., :-1] = np.diff(self.gradin, axis=-1)
        grady[..., :-1, :] = np.diff(self.gradin, axis=-2)
        gradz[:-1, ...] = np.diff(self.gradin, axis=0)/self.dz

        grad = np.stack((gradx,
                         grady,
                         gradz), axis=-1)

        outp = self.grad.fwdoop([[self.gradin]])

        np.testing.assert_allclose(outp[..., :-1], grad)

    def test_grad_inplace(self):
        gradx = np.zeros_like(self.gradin)
        grady = np.zeros_like(self.gradin)
        gradz = np.zeros_like(self.gradin)

        gradx[..., :-1] = np.diff(self.gradin, axis=-1)
        grady[..., :-1, :] = np.diff(self.gradin, axis=-2)
        gradz[:-1, ...] = np.diff(self.gradin, axis=0)/self.dz

        grad = np.stack((gradx,
                         grady,
                         gradz), axis=-1)

        outp = np.zeros_like(self.divin)

        self.grad.fwd([outp], [[self.gradin]])

        np.testing.assert_allclose(outp[..., :-1], grad)

    def test_adj_outofplace(self):

        outgrad = self.grad.fwdoop([[self.gradin]])
        outdiv = self.grad.adjoop([[self.divin]])

        a = np.vdot(outgrad[..., :-1].flatten(),
                    self.divin[..., :-1].flatten())/self.gradin.size
        b = np.vdot(self.gradin.flatten(), -outdiv.flatten())/self.gradin.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)

    def test_adj_inplace(self):

        outgrad = np.zeros_like(self.divin)
        outdiv = np.zeros_like(self.gradin)

        self.grad.fwd([outgrad], [[self.gradin]])
        self.grad.adj([outdiv], [[self.divin]])

        a = np.vdot(outgrad[..., :-1].flatten(),
                    self.divin[..., :-1].flatten())/self.gradin.size
        b = np.vdot(self.gradin.flatten(), -outdiv.flatten())/self.gradin.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=1)


if __name__ == '__main__':
    unittest.main()
