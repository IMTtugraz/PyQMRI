#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from pkg_resources import resource_filename
import pyopencl as cl
import pyopencl.array as clarray
import pyqmri.operator as operator
from pyqmri._helper_fun._utils import CLProgram as Program
DTYPE = np.complex64
DTYPE_real = np.float32


class CGSolver:
    def __init__(self, par, NScan=1, trafo=1, SMS=0):
        self.C = par["C"]
        self.traj = par["traj"]
        self.NSlice = par["NSlice"]
        NScan_save = par["NScan"]
        par["NScan"] = NScan
        self.NScan = NScan
        self.dimX = par["dimX"]
        self.dimY = par["dimY"]
        self.NC = par["NC"]
        self.ctx = par["ctx"][0]
        self.queue = par["queue"][0]
        self.N = par["N"]
        self.Nproj = par["Nproj"]
        self.prg = Program(
            self.ctx,
            open(resource_filename('mbpq', 'kernels/OpenCL_Kernels.c')).read())
        self.coil_buf = cl.Buffer(self.ctx,
                                  cl.mem_flags.READ_ONLY |
                                  cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=self.C.data)
        if SMS:
            self.op = operator.OperatorKspaceSMS(par, self.prg, trafo)
        else:
            self.op = operator.OperatorKspace(par, self.prg, trafo)
        self.FT = self.op.NUFFT.FFT
        self.FTH = self.op.NUFFT.FFTH
        self.tmp_result = clarray.empty(
                          self.queue,
                          (self.NScan, self.NC,
                           self.NSlice, self.dimY, self.dimX),
                          DTYPE, "C")
        self.tmp_sino = clarray.empty(
                          self.queue,
                          (self.NScan, self.NC,
                           self.NSlice, self.Nproj, self.N),
                          DTYPE, "C")
        par["NScan"] = NScan_save

    def __del__(self):
        del self.coil_buf
        del self.tmp_result
        del self.queue
        del self.ctx
        del self.tmp_sino
        del self.FT
        del self.FTH

    def run(self, data, iters=30, lambd=1e-1, tol=1e-8, guess=None):
        if guess is not None:
            x = clarray.to_device(self.queue, guess)
        else:
            x = clarray.empty(self.queue,
                              (self.NScan, 1,
                               self.NSlice, self.dimY, self.dimX),
                              DTYPE, "C")
        b = clarray.empty(self.queue,
                          (self.NScan, 1,
                           self.NSlice, self.dimY, self.dimX),
                          DTYPE, "C")
        Ax = clarray.empty(self.queue,
                           (self.NScan, 1,
                            self.NSlice, self.dimY, self.dimX),
                           DTYPE, "C")

        data = clarray.to_device(self.queue, data)
        self.operator_rhs(b, data)
        res = b
        p = res
        delta = np.linalg.norm(res.get())**2/np.linalg.norm(b.get())**2

#        print("Initial Residuum: ", delta)

        for i in range(iters):
            self.operator_lhs(Ax, p)
            Ax = Ax + lambd*p
            alpha = (clarray.vdot(res, res) /
                     (clarray.vdot(p, Ax))).real.get()
            x = (x + alpha*p)
            res_new = res - alpha*Ax
            delta = np.linalg.norm(res_new.get())**2 /\
                np.linalg.norm(b.get())**2
            if delta < tol:
                print(
                    "Converged after %i iterations to %1.3e." % (i, delta))
                del Ax, \
                    b, res, p, data, res_new
                return np.squeeze(x.get())
#            print("Res after iteration %i: %1.3e." % (i, delta))
            beta = (clarray.vdot(res_new, res_new) /
                    clarray.vdot(res, res)).real.get()
            p = res_new+beta*p
            (res, res_new) = (res_new, res)
        del Ax, b, res, p, data, res_new
        return np.squeeze(x.get())

    def eval_fwd_kspace_cg(self, y, x, wait_for=[]):
        return self.prg.operator_fwd_cg(self.queue,
                                        (self.NSlice, self.dimY, self.dimX),
                                        None,
                                        y.data, x.data, self.coil_buf,
                                        np.int32(self.NC),
                                        np.int32(self.NScan),
                                        wait_for=wait_for)

    def operator_lhs(self, out, x):
        self.tmp_result.add_event(self.eval_fwd_kspace_cg(
            self.tmp_result, x, wait_for=self.tmp_result.events+x.events))
        self.tmp_sino.add_event(self.FT(
            self.tmp_sino, self.tmp_result))
        return self.operator_rhs(out, self.tmp_sino)

    def operator_rhs(self, out, x, wait_for=[]):
        self.tmp_result.add_event(self.FTH(
            self.tmp_result, x, wait_for=wait_for+x.events))
        return self.prg.operator_ad_cg(self.queue,
                                       (self.NSlice, self.dimY, self.dimX),
                                       None,
                                       out.data, self.tmp_result.data,
                                       self.coil_buf, np.int32(self.NC),
                                       np.int32(self.NScan),
                                       wait_for=(self.tmp_result.events +
                                                 out.events+wait_for))
