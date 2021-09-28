#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:26:41 2019

@author: omaier
"""

import pyqmri
import os
from os.path import join as pjoin
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from pyqmri._helper_fun import CLProgram as Program
from pyqmri._helper_fun import _goldcomp as goldcomp
from pkg_resources import resource_filename
import pyopencl.array as clarray
import numpy as np
import h5py


DTYPE = np.complex64
DTYPE_real = np.float32
RTOL=1e-2
ATOL=1e-4
data_dir = os.path.realpath(pjoin(os.path.dirname(__file__), '..'))

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
    par["weights"] = np.array([1, 1])
    par["overlap"] = 1
    file = h5py.File(pjoin(data_dir, 'smalltest.h5'), 'r')

    par["traj"] = np.stack((
                file['imag_traj'][()].astype(DTYPE_real),
                file['real_traj'][()].astype(DTYPE_real)),
                axis=-1)
    
    par["traj"] = np.require(par["traj"][..., ::2, :], requirements='C')
    
    # Check if traj is scaled
    max_traj_val = np.max(np.abs(par["traj"]))
    if  np.allclose(max_traj_val, 0.5, rtol=1e-1):
       par["traj"] *= par["dimX"]
    elif np.allclose(max_traj_val, 1, rtol=1e-1):
       par["traj"] *= par["dimX"]/2
    
    overgrid_factor_a = (1/np.linalg.norm(
        par["traj"][..., -2, :]-par["traj"][..., -1, :], axis=-1))
    overgrid_factor_b = (1/np.linalg.norm(
        par["traj"][..., 0, :]-par["traj"][..., 1, :], axis=-1))
    par["ogf"] = np.min((overgrid_factor_a,
                         overgrid_factor_b))
    # print("Estimated OGF: ", par["ogf"])
    par["traj"] *= par["ogf"]

    par["dcf"] = np.sqrt(np.array(goldcomp.cmp(
                     par["traj"]), dtype=DTYPE_real)).astype(DTYPE_real)
    par["dcf"] = np.require(np.abs(par["dcf"]),
                            DTYPE_real, requirements='C')
    par["fft_dim"] = [-2, -1]
    par["is3D"] = False


class tmpArgs():
    pass


class OperatorKspaceRadial(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = False
        parser.devices = -1
        parser.use_GPU = False

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
        self.grad_buf = clarray.to_device(self.queue, self.model_gradient)
        self.coil_buf = clarray.to_device(self.queue, self.C)
        
        # parser = tmpArgs()
        # parser.streamed = False
        # parser.devices = -1
        # parser.use_GPU = True

        # par = {}
        # pyqmri.pyqmri._setupOCL(parser, par)
        # setupPar(par)
        
        # prg = []
        # for j in range(len(par["ctx"])):
        #     with open(file) as myfile:
        #         prg.append(Program(
        #             par["ctx"][j],
        #             myfile.read()))
        # prg = prg[0]

        # self.op_GPU = pyqmri.operator.OperatorKspace(
        #     par, prg,
        #     DTYPE=DTYPE,
        #     DTYPE_real=DTYPE_real)
        
        # self.queue_GPU = par["queue"][0]
        # self.grad_buf_GPU = clarray.to_device(self.queue_GPU, self.model_gradient)
        # self.coil_buf_GPU = clarray.to_device(self.queue_GPU, self.C)
        

    def test_adj_outofplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = self.op.fwdoop([inpfwd, self.coil_buf, self.grad_buf])
        outadj = self.op.adjoop([inpadj, self.coil_buf, self.grad_buf])

        outfwd = outfwd.map_to_host(wait_for=outfwd.events)
        outadj = outadj.map_to_host(wait_for=outadj.events)

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    def test_adj_inplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = clarray.zeros_like(inpadj)
        outadj = clarray.zeros_like(inpfwd)

        outfwd.add_event(self.op.fwd(outfwd, [inpfwd, self.coil_buf, self.grad_buf]))
        outadj.add_event(self.op.adj(outadj, [inpadj, self.coil_buf, self.grad_buf]))

        outfwd = outfwd.map_to_host(wait_for=outfwd.events)
        outadj = outadj.map_to_host(wait_for=outadj.events)

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
    
    # def test_CPU_vs_GPU_fwd(self):
    #     inpfwd_CPU = clarray.to_device(self.queue, self.opinfwd)
    #     outfwd_CPU = clarray.zeros(self.queue, self.opinadj.shape, dtype=DTYPE)
    #     outfwd_CPU.add_event(self.op.fwd(outfwd_CPU, [inpfwd_CPU, self.coil_buf, self.grad_buf]))
    #     outfwd_CPU = outfwd_CPU.map_to_host(wait_for=outfwd_CPU.events)
        
    #     inpfwd_GPU = clarray.to_device(self.queue_GPU, self.opinfwd)
    #     outfwd_GPU = clarray.zeros(self.queue_GPU, self.opinadj.shape, dtype=DTYPE)
    #     outfwd_GPU.add_event(self.op_GPU.fwd(outfwd_GPU, [inpfwd_GPU, self.coil_buf_GPU, self.grad_buf_GPU]))
    #     outfwd_GPU = outfwd_GPU.map_to_host(wait_for=outfwd_GPU.events)
        
    #     np.testing.assert_allclose(outfwd_CPU, outfwd_GPU, rtol=RTOL, atol=ATOL)

        
    # def test_CPU_vs_GPU_adj(self):
    #     inpadj_CPU = clarray.to_device(self.queue, self.opinadj)
    #     outadj_CPU = clarray.zeros(self.queue, self.opinfwd.shape, dtype=DTYPE)
    #     outadj_CPU.add_event(self.op.adj(outadj_CPU, [inpadj_CPU, self.coil_buf, self.grad_buf]))
    #     outadj_CPU = outadj_CPU.map_to_host(wait_for=outadj_CPU.events)
        
    #     inpadj_GPU = clarray.to_device(self.queue_GPU, self.opinadj)
    #     outadj_GPU = clarray.zeros(self.queue_GPU, self.opinfwd.shape, dtype=DTYPE)
    #     outadj_GPU.add_event(self.op_GPU.adj(outadj_GPU, [inpadj_GPU, self.coil_buf_GPU, self.grad_buf_GPU]))
    #     outadj_GPU = outadj_GPU.map_to_host(wait_for=outadj_GPU.events)     
        
    #     np.testing.assert_allclose(outadj_CPU, outadj_GPU, rtol=RTOL, atol=ATOL)

class OperatorKspaceCartesian(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = False
        parser.devices = -1
        parser.use_GPU = False

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
        self.grad_buf = clarray.to_device(self.queue, self.model_gradient)
        self.coil_buf = clarray.to_device(self.queue, self.C)
        
        # parser = tmpArgs()
        # parser.streamed = False
        # parser.devices = -1
        # parser.use_GPU = True

        # par = {}
        # pyqmri.pyqmri._setupOCL(parser, par)
        # setupPar(par)
        
        # prg = []
        # for j in range(len(par["ctx"])):
        #     with open(file) as myfile:
        #         prg.append(Program(
        #             par["ctx"][j],
        #             myfile.read()))
        # prg = prg[0]
        # par["mask"] = np.ones((par["dimY"], par["dimX"]),
        #                       dtype=DTYPE_real)
        
        # self.op_GPU = pyqmri.operator.OperatorKspace(
        #     par, prg,
        #     DTYPE=DTYPE,
        #     DTYPE_real=DTYPE_real, trafo=False)
        
        # self.queue_GPU = par["queue"][0]
        # self.grad_buf_GPU = clarray.to_device(self.queue_GPU, self.model_gradient)
        # self.coil_buf_GPU = clarray.to_device(self.queue_GPU, self.C)

    def test_adj_outofplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = self.op.fwdoop([inpfwd, self.coil_buf, self.grad_buf])
        outadj = self.op.adjoop([inpadj, self.coil_buf, self.grad_buf])

        outfwd = outfwd.map_to_host(wait_for=outfwd.events)
        outadj = outadj.map_to_host(wait_for=outadj.events)

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    def test_adj_inplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = clarray.zeros_like(inpadj)
        outadj = clarray.zeros_like(inpfwd)


        outfwd.add_event(self.op.fwd(outfwd, [inpfwd, self.coil_buf, self.grad_buf]))
        outadj.add_event(self.op.adj(outadj, [inpadj, self.coil_buf, self.grad_buf]))
    
        outfwd = outfwd.map_to_host(wait_for=outfwd.events)
        outadj = outadj.map_to_host(wait_for=outadj.events)
        

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    # def test_CPU_vs_GPU_fwd(self):
    #     inpfwd_CPU = clarray.to_device(self.queue, self.opinfwd)
    #     outfwd_CPU = clarray.zeros(self.queue, self.opinadj.shape, dtype=DTYPE)
    #     outfwd_CPU.add_event(self.op.fwd(outfwd_CPU, [inpfwd_CPU, self.coil_buf, self.grad_buf]))
    #     outfwd_CPU = outfwd_CPU.map_to_host(wait_for=outfwd_CPU.events)
        
    #     inpfwd_GPU = clarray.to_device(self.queue_GPU, self.opinfwd)
    #     outfwd_GPU = clarray.zeros(self.queue_GPU, self.opinadj.shape, dtype=DTYPE)
    #     outfwd_GPU.add_event(self.op_GPU.fwd(outfwd_GPU, [inpfwd_GPU, self.coil_buf_GPU, self.grad_buf_GPU]))
    #     outfwd_GPU = outfwd_GPU.map_to_host(wait_for=outfwd_GPU.events)
        
    #     np.testing.assert_allclose(outfwd_CPU, outfwd_GPU, rtol=RTOL, atol=ATOL)

        
    # def test_CPU_vs_GPU_adj(self):
    #     inpadj_CPU = clarray.to_device(self.queue, self.opinadj)
    #     outadj_CPU = clarray.zeros(self.queue, self.opinfwd.shape, dtype=DTYPE)
    #     outadj_CPU.add_event(self.op.adj(outadj_CPU, [inpadj_CPU, self.coil_buf, self.grad_buf]))
    #     outadj_CPU = outadj_CPU.map_to_host(wait_for=outadj_CPU.events)
        
    #     inpadj_GPU = clarray.to_device(self.queue_GPU, self.opinadj)
    #     outadj_GPU = clarray.zeros(self.queue_GPU, self.opinfwd.shape, dtype=DTYPE)
    #     outadj_GPU.add_event(self.op_GPU.adj(outadj_GPU, [inpadj_GPU, self.coil_buf_GPU, self.grad_buf_GPU]))
    #     outadj_GPU = outadj_GPU.map_to_host(wait_for=outadj_GPU.events)     
        
    #     np.testing.assert_allclose(outadj_CPU, outadj_GPU, rtol=RTOL, atol=ATOL)
        
class OperatorKspaceSMSCartesian(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = False
        parser.devices = -1
        parser.use_GPU = False

        par = {}
        par["packs"] = 6
        par["MB"] = 2
        par["shift"] = np.array([0, 64]).astype(DTYPE_real)
        par["numofpacks"] = 1
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

        par["mask"] = np.ones((par["dimY"], par["dimX"]),
                              dtype=DTYPE_real)

        self.op = pyqmri.operator.OperatorKspaceSMS(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)

        self.opinfwd = np.random.randn(par["unknowns"], par["NSlice"],
                                        par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["unknowns"], par["NSlice"],
                                  par["dimY"], par["dimX"])
        self.opinadj = np.random.randn(par["NScan"], par["NC"], par["packs"],
                                        par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NScan"], par["NC"], par["packs"],
                                  par["dimY"], par["dimX"])

        self.model_gradient = np.random.randn(par["NSlice"], par["unknowns"],
                                              par["NScan"],
                                              par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["NSlice"], par["unknowns"],
                                  par["NScan"],
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
        self.grad_buf = clarray.to_device(self.queue, self.model_gradient)
        self.coil_buf = clarray.to_device(self.queue, self.C)
        
        # parser = tmpArgs()
        # parser.streamed = False
        # parser.devices = -1
        # parser.use_GPU = True

        # par = {}
        # par["packs"] = 6
        # par["MB"] = 2
        # par["shift"] = np.array([0, 64]).astype(DTYPE_real)
        # par["numofpacks"] = 1
        # pyqmri.pyqmri._setupOCL(parser, par)
        # setupPar(par)
        # par["mask"] = np.ones((par["dimY"], par["dimX"]),
        #                       dtype=DTYPE_real)
        
        # prg = []
        # for j in range(len(par["ctx"])):
        #     with open(file) as myfile:
        #         prg.append(Program(
        #             par["ctx"][j],
        #             myfile.read()))
        # prg = prg[0]

        # self.op_GPU = pyqmri.operator.OperatorKspaceSMS(
        #     par, prg,
        #     DTYPE=DTYPE,
        #     DTYPE_real=DTYPE_real)
        
        # self.queue_GPU = par["queue"][0]
        # self.grad_buf_GPU = clarray.to_device(self.queue_GPU, self.model_gradient)
        # self.coil_buf_GPU = clarray.to_device(self.queue_GPU, self.C)

    def test_adj_outofplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = self.op.fwdoop([inpfwd, self.coil_buf, self.grad_buf])
        outadj = self.op.adjoop([inpadj, self.coil_buf, self.grad_buf])

        outfwd = outfwd.map_to_host(wait_for=outfwd.events)
        outadj = outadj.map_to_host(wait_for=outadj.events)

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    def test_adj_inplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = clarray.zeros_like(inpadj)
        outadj = clarray.zeros_like(inpfwd)


        outfwd.add_event(self.op.fwd(outfwd, [inpfwd, self.coil_buf, self.grad_buf]))
        outadj.add_event(self.op.adj(outadj, [inpadj, self.coil_buf, self.grad_buf]))
        
        outfwd = outfwd.map_to_host(wait_for=outfwd.events)
        outadj = outadj.map_to_host(wait_for=outadj.events)

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
        
    # def test_CPU_vs_GPU_fwd(self):
    #     inpfwd_CPU = clarray.to_device(self.queue, self.opinfwd)
    #     outfwd_CPU = clarray.zeros(self.queue, self.opinadj.shape, dtype=DTYPE)
    #     outfwd_CPU.add_event(self.op.fwd(outfwd_CPU, [inpfwd_CPU, self.coil_buf, self.grad_buf]))
    #     outfwd_CPU = outfwd_CPU.map_to_host(wait_for=outfwd_CPU.events)
        
    #     inpfwd_GPU = clarray.to_device(self.queue_GPU, self.opinfwd)
    #     outfwd_GPU = clarray.zeros(self.queue_GPU, self.opinadj.shape, dtype=DTYPE)
    #     outfwd_GPU.add_event(self.op_GPU.fwd(outfwd_GPU, [inpfwd_GPU, self.coil_buf_GPU, self.grad_buf_GPU]))
    #     outfwd_GPU = outfwd_GPU.map_to_host(wait_for=outfwd_GPU.events)
        
    #     np.testing.assert_allclose(outfwd_CPU, outfwd_GPU, rtol=RTOL, atol=ATOL)

        
    # def test_CPU_vs_GPU_adj(self):
    #     inpadj_CPU = clarray.to_device(self.queue, self.opinadj)
    #     outadj_CPU = clarray.zeros(self.queue, self.opinfwd.shape, dtype=DTYPE)
    #     outadj_CPU.add_event(self.op.adj(outadj_CPU, [inpadj_CPU, self.coil_buf, self.grad_buf]))
    #     outadj_CPU = outadj_CPU.map_to_host(wait_for=outadj_CPU.events)
        
    #     inpadj_GPU = clarray.to_device(self.queue_GPU, self.opinadj)
    #     outadj_GPU = clarray.zeros(self.queue_GPU, self.opinfwd.shape, dtype=DTYPE)
    #     outadj_GPU.add_event(self.op_GPU.adj(outadj_GPU, [inpadj_GPU, self.coil_buf_GPU, self.grad_buf_GPU]))
    #     outadj_GPU = outadj_GPU.map_to_host(wait_for=outadj_GPU.events)     
        
    #     np.testing.assert_allclose(outadj_CPU, outadj_GPU, rtol=RTOL, atol=ATOL)
        


class OperatorImageSpace(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = False
        parser.devices = -1
        parser.use_GPU = False

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
        self.grad_buf = clarray.to_device(self.queue, self.model_gradient)
        
        # parser = tmpArgs()
        # parser.streamed = False
        # parser.devices = -1
        # parser.use_GPU = True

        # par = {}
        # pyqmri.pyqmri._setupOCL(parser, par)
        # setupPar(par)
        
        # prg = []
        # for j in range(len(par["ctx"])):
        #     with open(file) as myfile:
        #         prg.append(Program(
        #             par["ctx"][j],
        #             myfile.read()))
        # prg = prg[0]

        # self.op_GPU = pyqmri.operator.OperatorImagespace(
        #     par, prg,
        #     DTYPE=DTYPE,
        #     DTYPE_real=DTYPE_real)
        
        # self.queue_GPU = par["queue"][0]
        # self.grad_buf_GPU = clarray.to_device(self.queue_GPU, self.model_gradient)

    def test_adj_outofplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = self.op.fwdoop([inpfwd, [], self.grad_buf])
        outadj = self.op.adjoop([inpadj, [], self.grad_buf])

        outfwd = outfwd.map_to_host(wait_for=outfwd.events)
        outadj = outadj.map_to_host(wait_for=outadj.events)

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    def test_adj_inplace(self):
        inpfwd = clarray.to_device(self.queue, self.opinfwd)
        inpadj = clarray.to_device(self.queue, self.opinadj)

        outfwd = clarray.zeros_like(inpadj)
        outadj = clarray.zeros_like(inpfwd)

        outfwd.add_event(
            self.op.fwd(outfwd, [inpfwd, [], self.grad_buf]))
        outadj.add_event(
            self.op.adj(outadj, [inpadj, [], self.grad_buf]))

        outfwd = outfwd.map_to_host(wait_for=outfwd.events)
        outadj = outadj.map_to_host(wait_for=outadj.events)

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
        
    # def test_CPU_vs_GPU_fwd(self):
    #     inpfwd_CPU = clarray.to_device(self.queue, self.opinfwd)
    #     outfwd_CPU = clarray.zeros(self.queue, self.opinadj.shape, dtype=DTYPE)
    #     outfwd_CPU.add_event(self.op.fwd(outfwd_CPU, [inpfwd_CPU, [], self.grad_buf]))
    #     outfwd_CPU = outfwd_CPU.map_to_host(wait_for=outfwd_CPU.events)
        
    #     inpfwd_GPU = clarray.to_device(self.queue_GPU, self.opinfwd)
    #     outfwd_GPU = clarray.zeros(self.queue_GPU, self.opinadj.shape, dtype=DTYPE)
    #     outfwd_GPU.add_event(self.op_GPU.fwd(outfwd_GPU, [inpfwd_GPU, [], self.grad_buf_GPU]))
    #     outfwd_GPU = outfwd_GPU.map_to_host(wait_for=outfwd_GPU.events)
        
    #     np.testing.assert_allclose(outfwd_CPU, outfwd_GPU, rtol=RTOL, atol=ATOL)

        
    # def test_CPU_vs_GPU_adj(self):
    #     inpadj_CPU = clarray.to_device(self.queue, self.opinadj)
    #     outadj_CPU = clarray.zeros(self.queue, self.opinfwd.shape, dtype=DTYPE)
    #     outadj_CPU.add_event(self.op.adj(outadj_CPU, [inpadj_CPU, [], self.grad_buf]))
    #     outadj_CPU = outadj_CPU.map_to_host(wait_for=outadj_CPU.events)
        
    #     inpadj_GPU = clarray.to_device(self.queue_GPU, self.opinadj)
    #     outadj_GPU = clarray.zeros(self.queue_GPU, self.opinfwd.shape, dtype=DTYPE)
    #     outadj_GPU.add_event(self.op_GPU.adj(outadj_GPU, [inpadj_GPU, [], self.grad_buf_GPU]))
    #     outadj_GPU = outadj_GPU.map_to_host(wait_for=outadj_GPU.events)     
        
    #     np.testing.assert_allclose(outadj_CPU, outadj_GPU, rtol=RTOL, atol=ATOL)
        
