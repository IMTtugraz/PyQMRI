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
from pyqmri._helper_fun import _goldcomp as goldcomp
from pkg_resources import resource_filename
import pyopencl.array as clarray
import pyopencl as cl
import numpy as np
import h5py


DTYPE = np.complex128
DTYPE_real = np.float64


def setupPar(par):
    par["NScan"] = 10
    par["NC"] = 8
    par["NSlice"] = 12
    par["dimX"] = 128
    par["dimY"] = 128
    par["Nproj"] = 34
    par["N"] = 256
    par["unknowns_TGV"] = 2
    par["unknowns_H1"] = 0
    par["unknowns"] = 2
    par["dz"] = 1
    par["weights"] = [1, 1]
    file = h5py.File('./test/smalltest.h5')

    par["traj"] = file['real_traj'][()].astype(DTYPE) + \
        1j*file['imag_traj'][()].astype(DTYPE)

    par["dcf"] = np.sqrt(np.array(goldcomp.cmp(
                     par["traj"]), dtype=DTYPE_real)).astype(DTYPE_real)
    par["dcf"] = np.require(np.abs(par["dcf"]),
                            DTYPE_real, requirements='C')


class tmpArgs():
    pass


class OperatorKspaceRadial(unittest.TestCase):
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

        self.op = pyqmri.operator.OperatorKspace(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)

        self.opinfwd = np.random.randn(par["unknowns"], par["NSlice"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["unknowns"], par["NSlice"],
                                 par["dimY"], par["dimX"])
        self.opinadj = np.random.randn(par["NScan"], par["NC"], par["NSlice"],
                                       par["Nproj"], par["N"]) +\
            1j * np.random.randn(par["NScan"], par["NC"], par["NSlice"],
                                 par["Nproj"], par["N"])
        self.model_gradient = np.random.randn(par["unknowns"], par["NScan"],
                                              par["NSlice"],
                                              par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["unknowns"], par["NScan"],
                                 par["NSlice"],
                                 par["dimY"], par["dimX"])

        self.C = np.random.randn(par["NC"], par["NSlice"],
                                 par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["NC"], par["NSlice"],
                                 par["dimY"], par["dimX"])

        self.model_gradient = self.model_gradient.astype(DTYPE)
        self.C = self.C.astype(DTYPE)
        self.opinfwd = self.opinfwd.astype(DTYPE)
        self.opinadj = self.opinadj.astype(DTYPE)
        self.queue = par["queue"][0]
        self.grad_buf = cl.Buffer(par["ctx"][0],
                                  cl.mem_flags.READ_ONLY |
                                  cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=self.model_gradient.data)
        self.coil_buf = cl.Buffer(par["ctx"][0],
                                  cl.mem_flags.READ_ONLY |
                                  cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=self.C.data)

    def test_adj_outofplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = self.op.fwdoop([inpfwd, self.coil_buf, self.grad_buf])
        outadj = self.op.adjoop([inpadj, self.coil_buf, self.grad_buf])

        outfwd = outfwd.get()
        outadj = outadj.get()

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)

    def test_adj_inplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = clarray.zeros_like(inpadj)
        outadj = clarray.zeros_like(inpfwd)

        self.op.fwd(outfwd, [inpfwd, self.coil_buf, self.grad_buf])
        self.op.adj(outadj, [inpadj, self.coil_buf, self.grad_buf])

        outfwd = outfwd.get()
        outadj = outadj.get()

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)


class OperatorKspaceCartesian(unittest.TestCase):
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

        par["mask"] = np.ones((par["dimY"], par["dimX"]),
                              dtype=DTYPE_real)

        self.op = pyqmri.operator.OperatorKspace(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real, trafo=False)

        self.opinfwd = np.random.randn(par["unknowns"], par["NSlice"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["unknowns"], par["NSlice"],
                                 par["dimY"], par["dimX"])
        self.opinadj = np.random.randn(par["NScan"], par["NC"], par["NSlice"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NScan"], par["NC"], par["NSlice"],
                                 par["dimY"], par["dimX"])

        self.model_gradient = np.random.randn(par["unknowns"], par["NScan"],
                                              par["NSlice"],
                                              par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["unknowns"], par["NScan"],
                                 par["NSlice"],
                                 par["dimY"], par["dimX"])
        self.C = np.random.randn(par["NC"], par["NSlice"],
                                 par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["NC"], par["NSlice"],
                                 par["dimY"], par["dimX"])

        self.model_gradient = self.model_gradient.astype(DTYPE)
        self.C = self.C.astype(DTYPE)
        self.opinfwd = self.opinfwd.astype(DTYPE)
        self.opinadj = self.opinadj.astype(DTYPE)
        self.queue = par["queue"][0]
        self.grad_buf = cl.Buffer(par["ctx"][0],
                                  cl.mem_flags.READ_ONLY |
                                  cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=self.model_gradient.data)
        self.coil_buf = cl.Buffer(par["ctx"][0],
                                  cl.mem_flags.READ_ONLY |
                                  cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=self.C.data)

    def test_adj_outofplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = self.op.fwdoop([inpfwd, self.coil_buf, self.grad_buf])
        outadj = self.op.adjoop([inpadj, self.coil_buf, self.grad_buf])

        outfwd = outfwd.get()
        outadj = outadj.get()

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)

    def test_adj_inplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = clarray.zeros_like(inpadj)
        outadj = clarray.zeros_like(inpfwd)

        self.op.fwd(outfwd, [inpfwd, self.coil_buf, self.grad_buf])
        self.op.adj(outadj, [inpadj, self.coil_buf, self.grad_buf])

        outfwd = outfwd.get()
        outadj = outadj.get()

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)


class OperatorImageSpace(unittest.TestCase):
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
                        'pyqmri', 'kernels/OpenCL_Kernels_double.c'))
        else:
            file = open(
                    resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels.c'))
        prg = Program(
            par["ctx"][0],
            file.read())
        file.close()

        self.op = pyqmri.operator.OperatorImagespace(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)
        self.opinfwd = np.random.randn(par["unknowns"], par["NSlice"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["unknowns"], par["NSlice"],
                                 par["dimY"], par["dimX"])
        self.opinadj = np.random.randn(par["NScan"], 1, par["NSlice"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NScan"], 1, par["NSlice"],
                                 par["dimY"], par["dimX"])
        self.model_gradient = np.random.randn(par["unknowns"], par["NScan"],
                                              par["NSlice"],
                                              par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["unknowns"], par["NScan"],
                                 par["NSlice"],
                                 par["dimY"], par["dimX"])

        self.model_gradient = self.model_gradient.astype(DTYPE)
        self.opinfwd = self.opinfwd.astype(DTYPE)
        self.opinadj = self.opinadj.astype(DTYPE)
        self.queue = par["queue"][0]
        self.grad_buf = cl.Buffer(par["ctx"][0],
                                  cl.mem_flags.READ_ONLY |
                                  cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=self.model_gradient.data)

    def test_adj_outofplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = self.op.fwdoop([inpfwd, [], self.grad_buf])
        outadj = self.op.adjoop([inpadj, [], self.grad_buf])

        outfwd = outfwd.get()
        outadj = outadj.get()

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)

    def test_adj_inplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = clarray.zeros_like(inpadj)
        outadj = clarray.zeros_like(inpfwd)

        self.op.fwd(outfwd, [inpfwd, [], self.grad_buf])
        self.op.adj(outadj, [inpadj, [], self.grad_buf])

        outfwd = outfwd.get()
        outadj = outadj.get()

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)


class OperatorImageSpaceStreamed(unittest.TestCase):
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
                    par["ctx"][j],
                    file.read()))
        file.close()

        par["par_slices"] = 1

        self.op = pyqmri.operator.OperatorImagespaceStreamed(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)
        self.opinfwd = np.random.randn(par["NSlice"], par["unknowns"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NSlice"], par["unknowns"],
                                 par["dimY"], par["dimX"])
        self.opinadj = np.random.randn(par["NSlice"], par["NScan"], 1,
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NSlice"], par["NScan"], 1,
                                 par["dimY"], par["dimX"])
        self.model_gradient = np.random.randn(par["NSlice"], par["unknowns"],
                                              par["NScan"],
                                              par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["NSlice"], par["unknowns"],
                                 par["NScan"],
                                 par["dimY"], par["dimX"])

        self.model_gradient = self.model_gradient.astype(DTYPE)
        self.opinfwd = self.opinfwd.astype(DTYPE)
        self.opinadj = self.opinadj.astype(DTYPE)

    def test_adj_outofplace(self):

        outfwd = self.op.fwdoop([[self.opinfwd, [], self.model_gradient]])
        outadj = self.op.adjoop([[self.opinadj, [], self.model_gradient]])

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)

    def test_adj_inplace(self):

        outfwd = np.zeros_like(self.opinadj)
        outadj = np.zeros_like(self.opinfwd)

        self.op.fwd([outfwd], [[self.opinfwd, [], self.model_gradient]])
        self.op.adj([outadj], [[self.opinadj, [], self.model_gradient]])

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)


class OperatorCartesianKSpaceStreamed(unittest.TestCase):
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
                    par["ctx"][j],
                    file.read()))
        file.close()

        par["par_slices"] = 1
        par["mask"] = np.ones((par["dimY"], par["dimX"]),
                              dtype=DTYPE_real)

        self.op = pyqmri.operator.OperatorKspaceStreamed(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real,
            trafo=False)

        self.opinfwd = np.random.randn(par["NSlice"], par["unknowns"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NSlice"], par["unknowns"],
                                 par["dimY"], par["dimX"])
        self.opinadj = np.random.randn(par["NSlice"], par["NScan"], par["NC"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NSlice"], par["NScan"], par["NC"],
                                 par["dimY"], par["dimX"])
        self.model_gradient = np.random.randn(par["NSlice"], par["unknowns"],
                                              par["NScan"],
                                              par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["NSlice"], par["unknowns"],
                                 par["NScan"],
                                 par["dimY"], par["dimX"])
        self.C = np.random.randn(par["NSlice"], par["NC"],
                                 par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["NSlice"], par["NC"],
                                 par["dimY"], par["dimX"])

        self.model_gradient = self.model_gradient.astype(DTYPE)
        self.opinfwd = self.opinfwd.astype(DTYPE)
        self.opinadj = self.opinadj.astype(DTYPE)
        self.C = self.C.astype(DTYPE)

    def test_adj_outofplace(self):

        outfwd = self.op.fwdoop([[self.opinfwd, self.C, self.model_gradient]])
        outadj = self.op.adjoop([[self.opinadj, self.C, self.model_gradient]])

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)

    def test_adj_inplace(self):

        outfwd = np.zeros_like(self.opinadj)
        outadj = np.zeros_like(self.opinfwd)

        self.op.fwd([outfwd], [[self.opinfwd, self.C, self.model_gradient]])
        self.op.adj([outadj], [[self.opinadj, self.C, self.model_gradient]])

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)


class OperatorRadialKSpaceStreamed(unittest.TestCase):
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
                    par["ctx"][j],
                    file.read()))
        file.close()

        par["par_slices"] = 1
        par["mask"] = np.ones((par["dimY"], par["dimX"]),
                              dtype=DTYPE_real)

        self.op = pyqmri.operator.OperatorKspaceStreamed(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)

        self.opinfwd = np.random.randn(par["NSlice"], par["unknowns"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NSlice"], par["unknowns"],
                                 par["dimY"], par["dimX"])
        self.opinadj = np.random.randn(par["NSlice"], par["NScan"], par["NC"],
                                       par["Nproj"], par["N"]) +\
            1j * np.random.randn(par["NSlice"], par["NScan"], par["NC"],
                                 par["Nproj"], par["N"])
        self.model_gradient = np.random.randn(par["NSlice"], par["unknowns"],
                                              par["NScan"],
                                              par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["NSlice"], par["unknowns"],
                                 par["NScan"],
                                 par["dimY"], par["dimX"])
        self.C = np.random.randn(par["NSlice"], par["NC"],
                                 par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["NSlice"], par["NC"],
                                 par["dimY"], par["dimX"])

        self.model_gradient = self.model_gradient.astype(DTYPE)
        self.opinfwd = self.opinfwd.astype(DTYPE)
        self.opinadj = self.opinadj.astype(DTYPE)
        self.C = self.C.astype(DTYPE)

    def test_adj_outofplace(self):

        outfwd = self.op.fwdoop([[self.opinfwd, self.C, self.model_gradient]])
        outadj = self.op.adjoop([[self.opinadj, self.C, self.model_gradient]])

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)

    def test_adj_inplace(self):

        outfwd = np.zeros_like(self.opinadj)
        outadj = np.zeros_like(self.opinfwd)

        self.op.fwd([outfwd], [[self.opinfwd, self.C, self.model_gradient]])
        self.op.adj([outadj], [[self.opinadj, self.C, self.model_gradient]])

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=14)


if __name__ == '__main__':
    unittest.main()
