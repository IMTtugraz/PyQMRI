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


DTYPE = np.complex128
DTYPE_real = np.float64
RTOL=1e-12
ATOL=1e-16
data_dir = os.path.realpath(pjoin(os.path.dirname(__file__), '..'))

def setupPar(par):
    par["NScan"] = 10
    par["NC"] = 10
    par["NSlice"] = 32
    par["dimX"] = 32
    par["dimY"] = 32
    par["Nproj"] = 200
    par["N"] = 64
    par["unknowns_TGV"] = 2
    par["unknowns_H1"] = 0
    par["unknowns"] = 2
    par["dz"] = 1
    par["weights"] = np.array([1, 1])
    par["overlap"] = 1
    file = h5py.File(pjoin(data_dir, 'VFA_phantom_3D_radial.h5'), 'r')

    par["traj"] = file['traj'][()].astype(DTYPE_real)
    
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
    par["fft_dim"] = (-3, -2, -1)
    par["is3D"] = True


class tmpArgs():
    pass


class OperatorKspaceRadial(unittest.TestCase):
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

        self.op = pyqmri.operator.OperatorKspace(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)

        self.opinfwd = np.random.randn(par["unknowns"], par["NSlice"],
                                        par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["unknowns"], par["NSlice"],
                                  par["dimY"], par["dimX"])
        self.opinadj = np.random.randn(par["NScan"], par["NC"], 1,
                                        par["Nproj"], par["N"]) +\
            1j * np.random.randn(par["NScan"], par["NC"], 1,
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
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

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
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
