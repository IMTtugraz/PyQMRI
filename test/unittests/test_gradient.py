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

DTYPE = np.complex64
DTYPE_real = np.float32
RTOL=1e-4
ATOL=1e-7

class tmpArgs():
    pass


def setupPar(par):
    par["NScan"] = 10
    par["NC"] = 15
    par["NSlice"] = 10
    par["dimX"] = 128
    par["dimY"] = 128
    par["Nproj"] = 21
    par["N"] = 256
    par["unknowns_TGV"] = 2
    par["unknowns_H1"] = 0
    par["unknowns"] = 2
    par["dz"] = 1
    par["weights"] = np.array([1, 0.1])


class GradientTest(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = False
        parser.devices = -1
        parser.use_GPU = True

        par = {}
        pyqmri.pyqmri._setupOCL(parser, par)
        setupPar(par)
        if DTYPE == np.complex128:
            file = resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels_double.c')
        else:
            file = resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels.c')

        prg = []
        for j in range(len(par["ctx"])):
          with open(file) as myfile:
            prg.append(Program(
                par["ctx"][j],
                myfile.read()))
        prg = prg[0]
        
        self.weights = par["weights"]

        self.grad = pyqmri.operator.OperatorFiniteGradient(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)

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
        
        # tmp_in = self.gradin
        # tmp_in = tmp_in.reshape(self.gradin.shape[0], -1).T
        # tmp_in = np.require(tmp_in, requirements='C')
        # grad_rot = np.zeros_like(tmp_in)
        # for j in range(tmp_in.shape[0]):
        #     grad_rot[j] = self.UTE[j]@tmp_in[j]

        # grad_rot = np.require(np.reshape((grad_rot.T),self.gradin.shape), requirements='C')

        gradx[..., :-1] = np.diff(self.gradin, axis=-1)
        grady[..., :-1, :] = np.diff(self.gradin, axis=-2)
        gradz[:, :-1, ...] = np.diff(self.gradin, axis=-3)*self.dz

        grad = np.stack((gradx,
                         grady,
                         gradz), axis=-1)
        
        grad *= self.weights[:, None, None, None, None]

        inp = clarray.to_device(self.queue, self.gradin)
        outp = self.grad.fwdoop(inp)
        outp = outp.get()

        np.testing.assert_allclose(outp[..., :-1], grad, rtol=RTOL, atol=ATOL)

    def test_grad_inplace(self):
        gradx = np.zeros_like(self.gradin)
        grady = np.zeros_like(self.gradin)
        gradz = np.zeros_like(self.gradin)

        gradx[..., :-1] = np.diff(self.gradin, axis=-1)
        grady[..., :-1, :] = np.diff(self.gradin, axis=-2)
        gradz[:, :-1, ...] = np.diff(self.gradin, axis=-3)*self.dz

        grad = np.stack((gradx,
                         grady,
                         gradz), axis=-1)
        
        grad *= self.weights[:, None, None, None, None]

        inp = clarray.to_device(self.queue, self.gradin)
        outp = clarray.to_device(self.queue, self.divin)
        outp.add_event(self.grad.fwd(outp, inp))
        outp = outp.get()

        np.testing.assert_allclose(outp[..., :-1], grad, rtol=RTOL, atol=ATOL)

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

        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    def test_adj_inplace(self):
        inpgrad = clarray.to_device(self.queue, self.gradin)
        inpdiv = clarray.to_device(self.queue, self.divin)

        outgrad = clarray.zeros_like(inpdiv)
        outdiv = clarray.zeros_like(inpgrad)

        outgrad.add_event(self.grad.fwd(outgrad, inpgrad))
        outgrad.add_event(self.grad.adj(outdiv, inpdiv))

        outgrad = outgrad.get()
        outdiv = outdiv.get()

        a = np.vdot(outgrad[..., :-1].flatten(),
                    self.divin[..., :-1].flatten())/self.gradin.size
        b = np.vdot(self.gradin.flatten(), -outdiv.flatten())/self.gradin.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
        
class GradientTestPrecond(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = False
        parser.devices = -1
        parser.use_GPU = True

        par = {}
        pyqmri.pyqmri._setupOCL(parser, par)
        setupPar(par)
        if DTYPE == np.complex128:
            file = resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels_double.c')
        else:
            file = resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels.c')

        prg = []
        for j in range(len(par["ctx"])):
          with open(file) as myfile:
            prg.append(Program(
                par["ctx"][j],
                myfile.read()))
        prg = prg[0]
        

        self.grad = pyqmri.operator.OperatorFiniteGradient(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)
        
        self.grad.precond = True
        self.weights = par["weights"]
        
        
        self.UTE = (np.random.randn(par["unknowns"],par["unknowns"], par["NSlice"]*par["dimY"]*par["dimX"])
                    +1j*np.random.randn(par["unknowns"],par["unknowns"], par["NSlice"]*par["dimY"]*par["dimX"]))
        self.UTE = self.UTE.astype(DTYPE)
        
                    
        
        self.grad.updatePrecondMat(np.require(self.UTE
                                    ,requirements='C'))
        
        self.UTE = np.require(self.UTE.transpose(2,0,1), requirements='C')

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
        
        tmp_in = self.gradin.copy()
        tmp_in = tmp_in.reshape(self.gradin.shape[0], -1)
        grad_rot = np.zeros_like(tmp_in)
        for j in range(tmp_in.shape[1]):
            grad_rot[..., j] = self.UTE[j]@tmp_in[..., j]

        grad_rot = np.require(np.reshape(grad_rot, self.gradin.shape), requirements='C')

        gradx[..., :-1] = np.diff(grad_rot, axis=-1)
        grady[..., :-1, :] = np.diff(grad_rot, axis=-2)
        gradz[:, :-1, ...] = np.diff(grad_rot, axis=-3)*self.dz

        grad = np.stack((gradx,
                         grady,
                         gradz), axis=-1)
        
        grad *= self.weights[:, None, None, None, None]

        inp = clarray.to_device(self.queue, self.gradin)
        outp = self.grad.fwdoop(inp)
        outp = outp.get()

        np.testing.assert_allclose(outp[..., :-1], grad, rtol=RTOL, atol=ATOL)

    def test_grad_inplace(self):
        gradx = np.zeros_like(self.gradin)
        grady = np.zeros_like(self.gradin)
        gradz = np.zeros_like(self.gradin)
        
        
        tmp_in = self.gradin.copy()
        tmp_in = tmp_in.reshape(self.gradin.shape[0], -1)
        grad_rot = np.zeros_like(tmp_in)
        for j in range(tmp_in.shape[1]):
            grad_rot[..., j] = self.UTE[j]@tmp_in[..., j]

        grad_rot = np.require(np.reshape(grad_rot, self.gradin.shape), requirements='C')

        gradx[..., :-1] = np.diff(grad_rot, axis=-1)
        grady[..., :-1, :] = np.diff(grad_rot, axis=-2)
        gradz[:, :-1, ...] = np.diff(grad_rot, axis=-3)*self.dz

        grad = np.stack((gradx,
                         grady,
                         gradz), axis=-1)
        
        grad *= self.weights[:, None, None, None, None]
    

        inp = clarray.to_device(self.queue, self.gradin)
        outp = clarray.to_device(self.queue, self.divin)
        outp.add_event(self.grad.fwd(outp, inp))
        outp = outp.get()

        np.testing.assert_allclose(outp[..., :-1], grad, rtol=RTOL, atol=ATOL)

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

        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    def test_adj_inplace(self):
        inpgrad = clarray.to_device(self.queue, self.gradin)
        inpdiv = clarray.to_device(self.queue, self.divin)

        outgrad = clarray.zeros_like(inpdiv)
        outdiv = clarray.zeros_like(inpgrad)

        outgrad.add_event(self.grad.fwd(outgrad, inpgrad))
        outgrad.add_event(self.grad.adj(outdiv, inpdiv))

        outgrad = outgrad.get()
        outdiv = outdiv.get()

        a = np.vdot(outgrad[..., :-1].flatten(),
                    self.divin[..., :-1].flatten())/self.gradin.size
        b = np.vdot(self.gradin.flatten(), -outdiv.flatten())/self.gradin.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


class GradientStreamedTest(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = True
        parser.devices = -1
        parser.use_GPU = True

        par = {}
        pyqmri.pyqmri._setupOCL(parser, par)
        setupPar(par)
        if DTYPE == np.complex128:
            file = resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels_double_streamed.c')
        else:
            file = resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels_streamed.c')

        prg = []
        for j in range(len(par["ctx"])):
          with open(file) as myfile:
            prg.append(Program(
                par["ctx"][j],
                myfile.read()))

        par["par_slices"] = 1

        self.grad = pyqmri.operator.OperatorFiniteGradientStreamed(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)

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
        gradz[:-1, ...] = np.diff(self.gradin, axis=0)*self.dz

        grad = np.stack((gradx,
                         grady,
                         gradz), axis=-1)
        
        # grad *= self.weights[None, :, None, None, None]

        outp = self.grad.fwdoop([[self.gradin]])

        np.testing.assert_allclose(outp[..., :-1], grad, rtol=RTOL, atol=ATOL)

    def test_grad_inplace(self):
        gradx = np.zeros_like(self.gradin)
        grady = np.zeros_like(self.gradin)
        gradz = np.zeros_like(self.gradin)

        gradx[..., :-1] = np.diff(self.gradin, axis=-1)
        grady[..., :-1, :] = np.diff(self.gradin, axis=-2)
        gradz[:-1, ...] = np.diff(self.gradin, axis=0)*self.dz

        grad = np.stack((gradx,
                         grady,
                         gradz), axis=-1)
        
        # grad *= self.weights[None, :, None, None, None]

        outp = np.zeros_like(self.divin)

        self.grad.fwd([outp], [[self.gradin]])

        np.testing.assert_allclose(outp[..., :-1], grad, rtol=RTOL, atol=ATOL)

    def test_adj_outofplace(self):

        outgrad = self.grad.fwdoop([[self.gradin]])
        outdiv = self.grad.adjoop([[self.divin]])

        a = np.vdot(outgrad[..., :-1].flatten(),
                    self.divin[..., :-1].flatten())/self.gradin.size
        b = np.vdot(self.gradin.flatten(), -outdiv.flatten())/self.gradin.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    def test_adj_inplace(self):

        outgrad = np.zeros_like(self.divin)
        outdiv = np.zeros_like(self.gradin)

        self.grad.fwd([outgrad], [[self.gradin]])
        self.grad.adj([outdiv], [[self.divin]])

        a = np.vdot(outgrad[..., :-1].flatten(),
                    self.divin[..., :-1].flatten())/self.gradin.size
        b = np.vdot(self.gradin.flatten(), -outdiv.flatten())/self.gradin.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

class GradientTestICTV(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = False
        parser.devices = -1
        parser.use_GPU = True

        par = {}
        pyqmri.pyqmri._setupOCL(parser, par)
        setupPar(par)
        if DTYPE == np.complex128:
            file = resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels_double.c')
        else:
            file = resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels.c')

        prg = []
        for j in range(len(par["ctx"])):
          with open(file) as myfile:
            prg.append(Program(
                par["ctx"][j],
                myfile.read()))
        prg = prg[0]
        
        self.weights = par["weights"]
        
        dt = np.random.randn(par["NScan"]-1)*10

        self.grad = pyqmri.operator.OperatorFiniteSpaceTimeGradient(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real,
            dt=dt,
            tsweight=0.5)

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


    def test_adj_outofplace(self):
        inpgrad = clarray.to_device(self.queue, self.gradin)
        inpdiv = clarray.to_device(self.queue, self.divin)

        outgrad = self.grad.fwdoop(inpgrad)
        outdiv = self.grad.adjoop(inpdiv)

        outgrad = outgrad.get()
        outdiv = outdiv.get()

        a = np.vdot(outgrad.flatten(),
                    self.divin.flatten())/self.gradin.size
        b = np.vdot(self.gradin.flatten(), -outdiv.flatten())/self.gradin.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    def test_adj_inplace(self):
        inpgrad = clarray.to_device(self.queue, self.gradin)
        inpdiv = clarray.to_device(self.queue, self.divin)

        outgrad = clarray.zeros_like(inpdiv)
        outdiv = clarray.zeros_like(inpgrad)

        outgrad.add_event(self.grad.fwd(outgrad, inpgrad))
        outgrad.add_event(self.grad.adj(outdiv, inpdiv))

        outgrad = outgrad.get()
        outdiv = outdiv.get()

        a = np.vdot(outgrad.flatten(),
                    self.divin.flatten())/self.gradin.size
        b = np.vdot(self.gradin.flatten(), -outdiv.flatten())/self.gradin.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


if __name__ == '__main__':
    unittest.main()
