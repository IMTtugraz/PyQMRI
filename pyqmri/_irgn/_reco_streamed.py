#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2019 Oliver Maier

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division

import numpy as np
import time
import sys
from pkg_resources import resource_filename
import pyopencl as cl
import pyopencl.array as clarray

import pyqmri._transforms._pyopencl_nufft_slicefirst as NUFFT

DTYPE = np.complex64
DTYPE_real = np.float32


class MyAllocator:
    def __init__(self, context,
                 flags=cl.mem_flags.READ_WRITE):
        self.context = context
        self.flags = flags

    def __call__(self, size):
        return cl.Buffer(self.context, self.flags, size)


class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build("-cl-mad-enable -cl-fast-relaxed-math")
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel


class ModelReco:
    def __init__(self, par, trafo=1, imagespace=False):
        par["overlap"] = 1
        self.overlap = par["overlap"]
        self.par_slices = par["par_slices"]
        self.par = par
        self.C = np.require(
            np.transpose(par["C"], [1, 0, 2, 3]), requirements='C')
        self.unknowns_TGV = par["unknowns_TGV"]
        self.unknowns_H1 = par["unknowns_H1"]
        self.unknowns = par["unknowns"]
        self.NSlice = par["NSlice"]
        self.NScan = par["NScan"]
        self.dimX = par["dimX"]
        self.dimY = par["dimY"]
        self.scale = 1
        self.NC = par["NC"]
        self.N = par["N"]
        self.Nproj = par["Nproj"]
        self.dz = 1
        self.fval_old = 0
        self.fval = 0
        self.fval_init = 0
        self.SNR_est = par["SNR_est"]
        self.ctx = par["ctx"]
        self.queue = par["queue"]
        self.gn_res = []
        self.num_dev = len(par["num_dev"])
        if np.mod(self.NSlice/(self.par_slices*self.num_dev), 2):
            raise ValueError(
                "Number of Slices devided by parallel "
                "computed slices and devices needs to be an even number")
        self.NUFFT = []
        self.prg = []
        self.alloc = []
        self.ratio = []
        self.tmp_img = []
        if imagespace:
            self.operator_forward = self.operator_forward_imagespace
            self.operator_adjoint = self.operator_adjoint_imagespace
            for j in range(self.num_dev):
                self.alloc.append(MyAllocator(self.ctx[j]))
                self.ratio.append(
                    clarray.to_device(
                        self.queue[4*j],
                        (np.ones(self.unknowns)).astype(dtype=DTYPE_real)))
        else:
            self.operator_forward = self.operator_forward_kspace
            self.operator_adjoint = self.operator_adjoint_kspace
            for j in range(self.num_dev):
                self.alloc.append(MyAllocator(self.ctx[j]))
                self.ratio.append(
                    clarray.to_device(
                        self.queue[4*j],
                        (np.ones(self.unknowns)).astype(dtype=DTYPE_real)))
                for i in range(2):
                    self.tmp_img.append(
                        clarray.empty(
                            self.queue[4*j+i],
                            (self.par_slices+self.overlap, self.NScan,
                             self.NC, self.dimY, self.dimX),
                            DTYPE, "C"))
                    self.NUFFT.append(
                        NUFFT.PyOpenCLNUFFT(self.ctx[j],
                                            self.queue[4*j+i], par,
                                            radial=trafo))
        for j in range(self.num_dev):
            self.prg.append(
                Program(
                    self.ctx[j],
                    open(
                        resource_filename(
                            'pyqmri',
                            'kernels/OpenCL_Kernels_streamed.c')).read()))

    def operator_forward_kspace(self, out, x, idx=0, idxq=0, wait_for=[]):
        self.tmp_img[2*idx+idxq].add_event(self.prg[idx].operator_fwd(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            self.tmp_img[2*idx+idxq].data, x.data,
            self.coil_buf_part[idx+idxq*self.num_dev].data,
            self.grad_buf_part[idx+idxq*self.num_dev].data,
            np.int32(self.NC),
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=self.tmp_img[2*idx+idxq].events+x.events+wait_for))
        return self.NUFFT[2*idx+idxq].fwd_NUFFT(
            out, self.tmp_img[2*idx+idxq],
            wait_for=out.events+wait_for+self.tmp_img[2*idx+idxq].events)

    def operator_adjoint_kspace(self, out, x, z,
                                idx=0, idxq=0, last=0, wait_for=[]):
        self.tmp_img[2*idx+idxq].add_event(
            self.NUFFT[2*idx+idxq].adj_NUFFT(
                self.tmp_img[2*idx+idxq], x,
                wait_for=wait_for+x.events+self.tmp_img[2*idx+idxq].events))
        return self.prg[idx].update_Kyk1(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            out.data, self.tmp_img[2*idx+idxq].data,
            self.coil_buf_part[idx+idxq*self.num_dev].data,
            self.grad_buf_part[idx+idxq*self.num_dev].data,
            z.data, np.int32(self.NC), np.int32(self.NScan),
            self.ratio[idx].data, np.int32(self.unknowns),
            np.int32(last), np.float32(self.dz),
            wait_for=(
                self.tmp_img[2*idx+idxq].events+out.events+z.events+wait_for))

    def operator_forward_imagespace(self, out, x, idx=0, idxq=0, wait_for=[]):
        return (self.prg[idx].operator_fwd_imagespace(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            out.data, x.data,
            self.grad_buf_part[idx+idxq*self.num_dev].data,
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=out.events+x.events+wait_for))

    def operator_adjoint_imagespace(self, out, x, z,
                                    idx=0, idxq=0, last=0, wait_for=[]):
        return self.prg[idx].update_Kyk1_imagespace(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            out.data, x.data,
            self.grad_buf_part[idx+idxq*self.num_dev].data,
            z.data,
            np.int32(self.NScan),
            self.ratio[idx].data, np.int32(self.unknowns),
            np.int32(last), np.float32(self.dz),
            wait_for=(x.events+out.events+z.events+wait_for))

    def f_grad(self, grad, u, idx=0, idxq=0, wait_for=[]):
        return self.prg[idx].gradient(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX),
            None, grad.data, u.data,
            np.int32(self.unknowns),
            self.ratio[idx].data, np.float32(self.dz),
            wait_for=grad.events + u.events + wait_for)

    def bdiv(self, div, u, idx=0, idxq=0, last=0, wait_for=[]):
        return self.prg[idx].divergence(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            div.data, u.data, np.int32(self.unknowns),
            self.ratio[idx].data, np.int32(last), np.float32(self.dz),
            wait_for=div.events + u.events + wait_for)

    def sym_grad(self, sym, w,  idx=0, idxq=0, wait_for=[]):
        return self.prg[idx].sym_grad(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            sym.data, w.data, np.int32(self.unknowns), np.float32(self.dz),
            wait_for=sym.events + w.events + wait_for)

    def sym_bdiv(self, div, u, idx=0, idxq=0, first=0, wait_for=[]):
        return self.prg[idx].sym_divergence(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            div.data, u.data,
            np.int32(self.unknowns), np.int32(first), np.float32(self.dz),
            wait_for=div.events + u.events + wait_for)

    def update_Kyk2(self, div, u, z, idx=0, idxq=0, first=0, wait_for=[]):
        return self.prg[idx].update_Kyk2(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            div.data, u.data, z.data,
            np.int32(self.unknowns), np.int32(first), np.float32(self.dz),
            wait_for=div.events + u.events + z.events+wait_for)

    def update_primal(self, x_new, x, Kyk, xk,
                      tau, delta, idx=0, idxq=0, wait_for=[]):
        return self.prg[idx].update_primal(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            x_new.data, x.data, Kyk.data, xk.data, np.float32(tau),
            np.float32(tau/delta), np.float32(1/(1+tau/delta)),
            self.min_const[idx].data, self.max_const[idx].data,
            self.real_const[idx].data, np.int32(self.unknowns),
            wait_for=x_new.events+x.events+Kyk.events+xk.events+wait_for)

    def update_z1(self, z_new, z, gx, gx_, vx, vx_,
                  sigma, theta, alpha, omega, idx=0, idxq=0, wait_for=[]):
        return self.prg[idx].update_z1(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            z_new.data, z.data, gx.data, gx_.data, vx.data, vx_.data,
            np.float32(sigma), np.float32(theta),
            np.float32(1/alpha), np.int32(self.unknowns_TGV),
            np.int32(self.unknowns_H1), np.float32(1 / (1 + sigma / omega)),
            wait_for=(z_new.events+z.events+gx.events +
                      gx_.events+vx.events+vx_.events+wait_for))

    def update_z1_tv(self, z_new, z, gx, gx_,
                     sigma, theta, alpha, omega, idx=0, idxq=0, wait_for=[]):
        return self.prg[idx].update_z1_tv(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            z_new.data, z.data, gx.data, gx_.data, np.float32(sigma),
            np.float32(theta),
            np.float32(1/alpha), np.int32(self.unknowns_TGV),
            np.int32(self.unknowns_H1), np.float32(1 / (1 + sigma / omega)),
            wait_for=z_new.events+z.events+gx.events+gx_.events+wait_for)

    def update_z2(self, z_new, z, gx, gx_,
                  sigma, theta, beta,  idx=0, idxq=0, wait_for=[]):
        return self.prg[idx].update_z2(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            z_new.data, z.data, gx.data, gx_.data, np.float32(sigma),
            np.float32(theta),
            np.float32(1/beta),  np.int32(self.unknowns),
            wait_for=z_new.events+z.events+gx.events+gx_.events+wait_for)

    def update_r(self, r_new, r, A, A_, res,
                 sigma, theta, lambd,  idx=0, idxq=0, wait_for=[]):
        return self.prg[idx].update_r(
            self.queue[4*idx+idxq], (r.size,), None, r_new.data, r.data,
            A.data, A_.data, res.data, np.float32(sigma), np.float32(theta),
            np.float32(1/(1+sigma/lambd)),
            wait_for=r_new.events+r.events+A.events+A_.events+wait_for)

    def update_v(self, v_new, v, Kyk2, tau,  idx=0, idxq=0, wait_for=[]):
        return self.prg[idx].update_v(
            self.queue[4*idx+idxq], (v[..., 0].size,), None,
            v_new.data, v.data, Kyk2.data, np.float32(tau),
            wait_for=v_new.events+v.events+Kyk2.events+wait_for)

    def update_primal_explicit(self, x_new, x, Kyk, xk, ATd,
                               tau, delta, lambd, idx=0, idxq=0, wait_for=[]):
        return self.prg[idx].update_primal_explicit(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            x_new.data, x.data, Kyk.data, xk.data, ATd.data, np.float32(tau),
            np.float32(1/delta), np.float32(lambd), self.min_const[idx].data,
            self.max_const[idx].data,
            self.real_const[idx].data, np.int32(self.unknowns),
            wait_for=(x_new.events+x.events+Kyk.events +
                      xk.events+ATd.events+wait_for))

    def eval_const(self):
        num_const = (len(self.model.constraints))
        min_const = np.zeros((num_const), dtype=np.float32)
        max_const = np.zeros((num_const), dtype=np.float32)
        real_const = np.zeros((num_const), dtype=np.int32)
        for j in range(num_const):
            min_const[j] = np.float32(self.model.constraints[j].min)
            max_const[j] = np.float32(self.model.constraints[j].max)
            real_const[j] = np.int32(self.model.constraints[j].real)

        self.min_const = []
        self.max_const = []
        self.real_const = []
        for j in range(self.num_dev):
            self.min_const.append(
                clarray.to_device(self.queue[4*j], min_const))
            self.max_const.append(
                clarray.to_device(self.queue[4*j], max_const))
            self.real_const.append(
                clarray.to_device(self.queue[4*j], real_const))

###############################################################################
# Scale before gradient #######################################################
###############################################################################
    def set_scale(self, x):
        x = clarray.to_device(self.queue[0], x)
        grad = clarray.to_device(self.queue[0], np.zeros_like(self.z1))
        grad.add_event(
            self.f_grad(
                grad,
                x,
                wait_for=grad.events +
                x.events))
        x = np.transpose(x.get(), [1, 0, 2, 3])
        grad = np.transpose(grad.get(), [1, 0, 2, 3, 4])
        scale = np.reshape(
            x, (self.unknowns, self.NSlice * self.dimY * self.dimX))
        grad = np.reshape(
            grad, (self.unknowns, self.NSlice * self.dimY * self.dimX * 4))
        print("Diff between x: ", np.linalg.norm(scale, axis=-1))
        print("Diff between grad x: ", np.linalg.norm(grad, axis=-1))
        scale = np.linalg.norm(grad, axis=-1)
        scale = 1/scale
        scale[~np.isfinite(scale)] = 1
        sum_scale = np.linalg.norm(
            scale[:self.unknowns_TGV])/(1000/np.sqrt(self.NSlice))
        for i in range(self.num_dev):
            for j in range(x.shape[0])[:self.unknowns_TGV]:
                self.ratio[i][j] = scale[j] / sum_scale
        sum_scale = np.linalg.norm(
            scale[self.unknowns_TGV:])/(1000)
        for i in range(self.num_dev):
            for j in range(x.shape[0])[self.unknowns_TGV:]:
                self.ratio[i][j] = scale[j] / sum_scale
        print("Ratio: ", self.ratio[0])

###############################################################################
# Start a 3D Reconstruction, set TV to True to perform TV instead of TGV#######
# Precompute Model and Gradient values for xk #################################
# Call inner optimization #####################################################
# input: bool to switch between TV (1) and TGV (0) regularization #############
# output: optimal value of x ##################################################
###############################################################################
    def execute_3D(self, TV=0):
        self.FT = self.FT_streamed
        iters = self.irgn_par["start_iters"]

        self.r = np.zeros_like(self.data, dtype=DTYPE)
        self.r = np.require(np.transpose(self.r, [2, 0, 1, 3, 4]),
                            requirements='C')
        self.z1 = np.zeros(
            ([self.NSlice, self.unknowns, self.dimY, self.dimX, 4]),
            dtype=DTYPE)

        self.result = np.zeros(
            (self.irgn_par["max_gn_it"]+1, self.unknowns,
             self.NSlice, self.dimY, self.dimX),
            dtype=DTYPE)
        self.result[0, :, :, :, :] = np.copy(self.model.guess)

        result = np.copy(self.model.guess)

        if TV == 1:
            self.tau = np.float32(1/np.sqrt(8))
            self.beta_line = 400
            self.theta_line = np.float32(1.0)
        elif TV == 0:
            L = np.float32(0.5*(18.0 + np.sqrt(33)))
            self.tau = np.float32(1/np.sqrt(L))
            self.beta_line = 400
            self.theta_line = np.float32(1.0)
            self.v = np.zeros(
                ([self.NSlice, self.unknowns, self.dimY, self.dimX, 4]),
                dtype=DTYPE)
            self.z2 = np.zeros(
                ([self.NSlice, self.unknowns, self.dimY, self.dimX, 8]),
                dtype=DTYPE)
        else:
            print("Not implemented")
            return
        for i in range(self.irgn_par["max_gn_it"]):
            start = time.time()
            self.grad_x = np.nan_to_num(self.model.execute_gradient(result))
            scale = np.reshape(
                self.grad_x,
                (self.unknowns,
                 self.NScan * self.NSlice * self.dimY * self.dimX))
            scale = np.linalg.norm(scale, axis=-1)
            print("Initial norm of the model Gradient: \n", scale)
            scale = 1e3 / np.sqrt(self.unknowns) / scale
            print("Scalefactor of the model Gradient: \n", scale)
            if not np.mod(i, 1):
                for uk in range(self.unknowns):
                    self.model.constraints[uk].update(scale[uk])
                    result[uk, ...] *= self.model.uk_scale[uk]
                    self.grad_x[uk] /= self.model.uk_scale[uk]
                    self.model.uk_scale[uk] *= scale[uk]
                    result[uk, ...] /= self.model.uk_scale[uk]
                    self.grad_x[uk] *= self.model.uk_scale[uk]
            scale = np.reshape(
                self.grad_x,
                (self.unknowns,
                 self.NScan * self.NSlice * self.dimY * self.dimX))
            scale = np.linalg.norm(scale, axis=-1)
            print("Scale of the model Gradient: \n", scale)

            self.set_scale(np.require(
                np.transpose(result, [1, 0, 2, 3]), requirements='C'))

            self.step_val = np.nan_to_num(self.model.execute_forward(result))
            self.step_val = np.require(
                np.transpose(self.step_val, [1, 0, 2, 3]), requirements='C')
            self.grad_x = np.require(
                np.transpose(self.grad_x, [2, 0, 1, 3, 4]), requirements='C')

            self.irgn_par["delta_max"] = self.delta_max / \
                                         (1e3) * np.linalg.norm(result)
            self.irgn_par["delta"] = np.minimum(
                self.delta /
                (1e3)*np.linalg.norm(result)*self.irgn_par["delta_inc"]**i,
                self.irgn_par["delta_max"])

            result = self.irgn_solve_3D(result, iters, self.data, i, TV)
            self.result[i + 1, ...] = self.model.rescale(result)

            iters = np.fmin(iters * 2, self.irgn_par["max_iters"])
            self.irgn_par["gamma"] = np.maximum(
                self.irgn_par["gamma"] * self.irgn_par["gamma_dec"],
                self.irgn_par["gamma_min"])
            self.irgn_par["omega"] = np.maximum(
                self.irgn_par["omega"] * self.irgn_par["omega_dec"],
                self.irgn_par["omega_min"])

            end = time.time() - start
            self.gn_res.append(self.fval)
            print("-" * 75)
            print("GN-Iter: %d  Elapsed time: %f seconds" % (i, end))
            print("-" * 75)
            if np.abs(self.fval_old - self.fval) / self.fval_init < \
               self.irgn_par["tol"]:
                print("Terminated at GN-iteration %d because "
                      "the energy decrease was less than %.3e" %
                      (i, np.abs(self.fval_old - self.fval) / self.fval_init))
                self.calc_residual_ksapce(np.require(np.transpose(result, [1, 0, 2, 3]), requirements='C'), np.require(
            np.transpose(self.data, [2, 0, 1, 3, 4]), requirements='C'), i+1, TV)
                break
            self.fval_old = self.fval
        self.calc_residual_ksapce(np.require(np.transpose(result, [1, 0, 2, 3]), requirements='C'), np.require(
            np.transpose(self.data, [2, 0, 1, 3, 4]), requirements='C'), i+1, TV)

###############################################################################
### Precompute constant terms of the GN linearization step ####################
### input: linearization point x ##############################################
########## numeber of innner iterations iters #################################
########## Data ###############################################################
########## bool to switch between TV (1) and TGV (0) regularization ###########
### output: optimal value of x for the inner GN step ##########################
###############################################################################
###############################################################################
    def irgn_solve_3D(self, x, iters, data, GN_it, TV=0):
        x = np.require(np.transpose(x, [1, 0, 2, 3]), requirements='C')
        data = np.require(
            np.transpose(data, [2, 0, 1, 3, 4]), requirements='C')
        b = np.zeros(data.shape, dtype=DTYPE)
        DGk = np.zeros_like(data.astype(DTYPE))
        self.FT(b, self.step_val[:, :, None, ...]*self.C[:, None, ...])
        self.operator_forward_streamed(DGk, x)
        res = data - b + DGk

        self.calc_residual_ksapce(x, data, GN_it, TV)

        if TV == 1:
            x = self.tv_solve_3D(x, res, iters)
        elif TV == 0:
            x = self.tgv_solve_3D(x, res, iters)
        x = np.require(np.transpose(x, [1, 0, 2, 3]), requirements='C')
        return x

    def calc_residual_ksapce(self, x, data, GN_it, TV=0):
        b = np.zeros(data.shape, dtype=DTYPE)
        if TV == 1:
            x = clarray.to_device(self.queue[0], x)
            grad = clarray.to_device(self.queue[0], np.zeros_like(self.z1))
            grad.add_event(self.f_grad(grad, x, wait_for=grad.events+x.events))
            x = np.require(
                np.transpose(x.get(), [1, 0, 2, 3]), requirements='C')
            self.FT(b,
                    self.step_val[:, :, None, ...]*self.C[:, None, ...])
            grad = grad.get()
            self.fval = (
                self.irgn_par["lambd"]/2*np.linalg.norm(data - b)**2 +
                self.irgn_par["gamma"]*np.sum(np.abs(
                    grad[:, :self.unknowns_TGV])) +
                self.irgn_par["omega"] / 2 *
                np.linalg.norm(grad[:, self.unknowns_TGV:])**2)
            del grad, b
        elif TV == 0:
            x = clarray.to_device(self.queue[0], x)
            v = clarray.to_device(self.queue[0], self.v)
            grad = clarray.to_device(self.queue[0], np.zeros_like(self.z1))
            sym_grad = clarray.to_device(self.queue[0], np.zeros_like(self.z2))
            grad.add_event(
                self.f_grad(grad, x, wait_for=grad.events+x.events))
            sym_grad.add_event(
                self.sym_grad(sym_grad, v, wait_for=sym_grad.events+v.events))
            x = np.require(
                np.transpose(x.get(), [1, 0, 2, 3]), requirements='C')
            self.FT(b,
                    self.step_val[:, :, None, ...]*self.C[:, None, ...])
            grad = grad.get()
            self.fval = (
                self.irgn_par["lambd"]/2*np.linalg.norm(data - b)**2 +
                self.irgn_par["gamma"]*np.sum(np.abs(
                    grad[:, :self.unknowns_TGV]-self.v)) +
                self.irgn_par["gamma"]*(2)*np.sum(np.abs(sym_grad.get())) +
                self.irgn_par["omega"] / 2 *
                np.linalg.norm(grad[:, self.unknowns_TGV:])**2)
            del grad, sym_grad, v, b

        if GN_it == 0:
            self.fval_init = self.fval
        print("-" * 75)
        print("Function value at GN-Step %i: %f" %
              (GN_it, 1e3*self.fval / self.fval_init))
        print("-" * 75)

###############################################################################
# Start a 3D Reconstruction, set TV to True to perform TV instead of TGV#######
# Precompute Model and Gradient values for xk #################################
# Call inner optimization #####################################################
# input: bool to switch between TV (1) and TGV (0) regularization #############
# output: optimal value of x ##################################################
###############################################################################
    def execute_3D_imagespace(self, TV=0):
        iters = self.irgn_par["start_iters"]
        self.NC = 1
        self.N = self.dimX
        self.Nproj = self.dimY

        self.r = np.zeros_like(self.data, dtype=DTYPE)
        self.r = np.require(np.transpose(self.r, [1, 0, 2, 3]),
                            requirements='C')
        self.z1 = np.zeros(
            ([self.NSlice, self.unknowns, self.dimY, self.dimX, 4]),
            dtype=DTYPE)

        self.result = np.zeros(
            (self.irgn_par["max_gn_it"]+1, self.unknowns,
             self.NSlice, self.dimY, self.dimX),
            dtype=DTYPE)
        self.result[0, ...] = np.copy(self.model.guess)

        result = np.copy(self.model.guess)

        if TV == 1:
            self.tau = np.float32(1/np.sqrt(8))
            self.beta_line = 400
            self.theta_line = np.float32(1.0)
        elif TV == 0:
            L = np.float32(0.5*(18.0 + np.sqrt(33)))
            self.tau = np.float32(1/np.sqrt(L))
            self.beta_line = 400
            self.theta_line = np.float32(1.0)
            self.v = np.zeros(
                ([self.NSlice, self.unknowns, self.dimY, self.dimX, 4]),
                dtype=DTYPE)
            self.z2 = np.zeros(
                ([self.NSlice, self.unknowns, self.dimY, self.dimX, 8]),
                dtype=DTYPE)
        else:
            print("Not implemented")
            return

        for i in range(self.irgn_par["max_gn_it"]):
            start = time.time()
            self.grad_x = np.nan_to_num(self.model.execute_gradient(result))
            scale = np.reshape(
                self.grad_x,
                (self.unknowns,
                 self.NScan * self.NSlice * self.dimY * self.dimX))
            scale = np.linalg.norm(scale, axis=-1)
            print("Initial norm of the model Gradient: \n", scale)
            scale = 1e3 / np.sqrt(self.unknowns) / scale
            print("Scalefactor of the model Gradient: \n", scale)
            if not np.mod(i, 1):
                for uk in range(self.unknowns):
                    self.model.constraints[uk].update(scale[uk])
                    result[uk, ...] *= self.model.uk_scale[uk]
                    self.grad_x[uk] /= self.model.uk_scale[uk]
                    self.model.uk_scale[uk] *= scale[uk]
                    result[uk, ...] /= self.model.uk_scale[uk]
                    self.grad_x[uk] *= self.model.uk_scale[uk]
            scale = np.reshape(
                self.grad_x,
                (self.unknowns,
                 self.NScan * self.NSlice * self.dimY * self.dimX))
            scale = np.linalg.norm(scale, axis=-1)
            print("Scale of the model Gradient: \n", scale)

            self.set_scale(np.require(
                np.transpose(result, [1, 0, 2, 3]), requirements='C'))

            self.step_val = np.nan_to_num(self.model.execute_forward(result))
            self.step_val = np.require(
                np.transpose(self.step_val, [1, 0, 2, 3]), requirements='C')
            self.grad_x = np.require(
                np.transpose(self.grad_x, [2, 0, 1, 3, 4]), requirements='C')

            self.irgn_par["delta_max"] = self.delta_max / \
                                         (1e3) * np.linalg.norm(result)
            self.irgn_par["delta"] = np.minimum(
                self.delta /
                (1e3)*np.linalg.norm(result)*self.irgn_par["delta_inc"]**i,
                self.irgn_par["delta_max"])

            result = self.irgn_solve_3D_imagespace(result, iters,
                                                   self.data, i, TV)
            self.result[i + 1, ...] = self.model.rescale(result)

            iters = np.fmin(iters * 2, self.irgn_par["max_iters"])
            self.irgn_par["gamma"] = np.maximum(
                self.irgn_par["gamma"] * self.irgn_par["gamma_dec"],
                self.irgn_par["gamma_min"])
            self.irgn_par["omega"] = np.maximum(
                self.irgn_par["omega"] * self.irgn_par["omega_dec"],
                self.irgn_par["omega_min"])

            end = time.time() - start
            self.gn_res.append(self.fval)
            print("-" * 75)
            print("GN-Iter: %d  Elapsed time: %f seconds" % (i, end))
            print("-" * 75)
            if np.abs(self.fval_old - self.fval) / self.fval_init < \
               self.irgn_par["tol"]:
                print("Terminated at GN-iteration %d because "
                      "the energy decrease was less than %.3e" %
                      (i, np.abs(self.fval_old - self.fval) / self.fval_init))
                self.calc_residual_imagespace(
                    np.require(
                        np.transpose(result, [1, 0, 2, 3]), requirements='C'),
                    np.require(
                        np.transpose(self.data, [1, 0, 2, 3]),
                        requirements='C'), i+1, TV)
                break
            self.fval_old = self.fval
        self.calc_residual_imagespace(
            np.require(
                np.transpose(result, [1, 0, 2, 3]), requirements='C'),
            np.require(
                np.transpose(self.data, [1, 0, 2, 3]),
                requirements='C'), i+1, TV)

###############################################################################
### Precompute constant terms of the GN linearization step ####################
### input: linearization point x ##############################################
########## numeber of innner iterations iters #################################
########## Data ###############################################################
########## bool to switch between TV (1) and TGV (0) regularization ###########
### output: optimal value of x for the inner GN step ##########################
###############################################################################
###############################################################################
    def irgn_solve_3D_imagespace(self, x, iters, data, GN_it, TV=0):

        x = np.require(np.transpose(x, [1, 0, 2, 3]), requirements='C')
        data = np.require(
            np.transpose(data, [1, 0, 2, 3]), requirements='C')
        DGk = np.zeros_like(data.astype(DTYPE))
        self.operator_forward_streamed(DGk, x)

        res = data - self.step_val + DGk

        self.calc_residual_imagespace(x, data, GN_it, TV)

        if TV == 1:
            x = self.tv_solve_3D(x, res, iters)
        elif TV == 0:
            x = self.tgv_solve_3D(x, res, iters)
        x = np.require(np.transpose(x, [1, 0, 2, 3]), requirements='C')
        return x

    def calc_residual_imagespace(self, x, data, GN_it, TV=0):
        if TV == 1:
            x = clarray.to_device(self.queue[0], x)
            grad = clarray.to_device(self.queue[0], np.zeros_like(self.z1))
            grad.add_event(self.f_grad(grad, x, wait_for=grad.events+x.events))
            grad = grad.get()
            self.fval = (
                self.irgn_par["lambd"]/2 *
                np.linalg.norm(data - self.step_val)**2 +
                self.irgn_par["gamma"]*np.sum(
                    np.abs(grad[:, :self.unknowns_TGV])) +
                self.irgn_par["omega"] / 2 *
                np.linalg.norm(grad[:, self.unknowns_TGV:])**2)
            del grad
        elif TV == 0:
            x = clarray.to_device(self.queue[0], x)
            v = clarray.to_device(self.queue[0], self.v)
            grad = clarray.to_device(self.queue[0], np.zeros_like(self.z1))
            sym_grad = clarray.to_device(self.queue[0], np.zeros_like(self.z2))
            grad.add_event(
                self.f_grad(grad, x, wait_for=grad.events+x.events))
            sym_grad.add_event(
                self.sym_grad(sym_grad, v, wait_for=sym_grad.events+v.events))
            grad = grad.get()
            self.fval = (
                self.irgn_par["lambd"]/2 *
                np.linalg.norm(data - self.step_val)**2 +
                self.irgn_par["gamma"]*np.sum(
                    np.abs(grad[:, :self.unknowns_TGV]-self.v)) +
                self.irgn_par["gamma"]*(2)*np.sum(np.abs(sym_grad.get())) +
                self.irgn_par["omega"] / 2 *
                np.linalg.norm(grad[:, self.unknowns_TGV:])**2)
            del grad, sym_grad, v

        if GN_it == 0:
            self.fval_init = self.fval
        print("-" * 75)
        print("Function value at GN-Step %i: %f" %
              (GN_it, 1e3*self.fval / self.fval_init))
        print("-" * 75)

    def tgv_solve_3D(self, x, res, iters):
        alpha = self.irgn_par["gamma"]
        beta = self.irgn_par["gamma"] * 2

        tau = self.tau
        tau_new = np.float32(0)

        xk = x.copy()
        x_new = np.zeros_like(x)

        r = np.zeros_like(self.r)
        r_new = np.zeros_like(r)
        z1 = np.zeros_like(self.z1)
        z1_new = np.zeros_like(z1)
        z2 = np.zeros_like(self.z2)
        z2_new = np.zeros_like(z2)
        v = np.zeros_like(self.v)
        v_new = np.zeros_like(v)
        res = (res).astype(DTYPE)

        delta = self.irgn_par["delta"]
        omega = self.irgn_par["omega"]
        mu = 1/delta

        theta_line = self.theta_line
        beta_line = self.beta_line
        beta_new = np.float32(0)
        mu_line = np.float32(0.5)
        delta_line = np.float32(1)
        ynorm = np.float32(0.0)
        lhs = np.float32(0.0)
        primal = np.float32(0.0)
        primal_new = np.float32(0)
        dual = np.float32(0.0)
        gap_init = np.float32(0.0)
        gap_old = np.float32(0.0)
        gap = np.float32(0.0)
        self.eval_const()

        Kyk1 = np.zeros_like(x)
        Kyk1_new = np.zeros_like(x)
        Kyk2 = np.zeros_like(z1)
        Kyk2_new = np.zeros_like(z1)
        gradx = np.zeros_like(z1)
        gradx_xold = np.zeros_like(z1)
        symgrad_v = np.zeros_like(z2)
        symgrad_v_vold = np.zeros_like(z2)
        Axold = np.zeros_like(res)
        Ax = np.zeros_like(res)

    # Allocate temporary Arrays
        (Axold_part, Kyk1_part, Kyk2_part, xk_part, v_part, res_part,
         z1_new_part, z2_new_part, r_new_part, Kyk1_new_part, Kyk2_new_part,
         x_new_part, Ax_part, v_new_part, gradx_part, gradx_xold_part,
         symgrad_v_part, symgrad_v_vold_part, x_part,
         r_part, z1_part, z2_part) = self.preallocate_space(TGV=True)

    # Warmup
        self.stream_initial(
            (x, r, z1, z2, Axold, Kyk1, Kyk2, gradx_xold, symgrad_v_vold, v),
            (x_part, r_part, z1_part, z2_part, Axold_part, Kyk1_part,
             Kyk2_part, gradx_xold_part, symgrad_v_vold_part, v_part),
            TGV=True)
    # Start Iterations
        for myit in range(iters):
            self.stream_primal_update(
                (x_new, xk, x, v_new, v, Kyk1, gradx, Ax, Kyk2, symgrad_v),
                (x_new_part, xk_part, x_part, v_new_part, v_part, Kyk1_part,
                 gradx_part, Ax_part, Kyk2_part, symgrad_v_part),
                tau, delta, TGV=True)

            beta_new = beta_line*(1+mu*tau)
            tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
            beta_line = beta_new

            while True:
                theta_line = tau_new/tau

                ynorm = 0
                lhs = 0

                (ynorm, lhs) = self.stream_dual_update(
                    (z1, gradx, gradx_xold, v_new, v, r, Ax, Axold, res, Kyk1,
                     z1_new, r_new, Kyk1_new, z2, symgrad_v, symgrad_v_vold,
                     Kyk2, z2_new, Kyk2_new),
                    (z1_part, z1_new_part, z2_part, z2_new_part, gradx_part,
                     gradx_xold_part, v_new_part, v_part, r_part, r_new_part,
                     Ax_part, Axold_part, res_part, Kyk1_part,
                     Kyk1_new_part, Kyk2_part, Kyk2_new_part, symgrad_v_part,
                     symgrad_v_vold_part),
                    (ynorm, lhs, beta_line, tau_new, omega,
                     alpha, beta, theta_line),
                    TGV=True)

                if np.sqrt(beta_line)*tau_new*(abs(lhs)**(1/2)) <= \
                   (abs(ynorm)**(1/2))*delta_line:
                    break
                else:
                    tau_new = tau_new*mu_line
            (Kyk1, Kyk1_new, Kyk2, Kyk2_new, Axold, Ax, z1, z1_new,
             z2, z2_new, r, r_new, gradx_xold, gradx, symgrad_v_vold,
             symgrad_v, tau) = (
             Kyk1_new, Kyk1, Kyk2_new, Kyk2, Ax, Axold, z1_new, z1,
             z2_new, z2, r_new, r, gradx, gradx_xold, symgrad_v,
             symgrad_v_vold, tau_new)

            if not np.mod(myit, 10):
                if self.irgn_par["display_iterations"]:
                    self.model.plot_unknowns(
                        np.transpose(x_new, [1, 0, 2, 3]))
                if self.unknowns_H1 > 0:
                    primal_new = (
                        self.irgn_par["lambd"]/2 *
                        np.vdot(Axold-res, Axold-res) +
                        alpha*np.sum(abs((gradx[:, :self.unknowns_TGV]-v))) +
                        beta*np.sum(abs(symgrad_v)) +
                        1/(2*delta)*np.vdot(x_new-xk, x_new-xk) +
                        self.irgn_par["omega"] / 2 *
                        np.vdot(gradx[self.unknowns_TGV:],
                                gradx[self.unknowns_TGV:])).real

                    dual = (
                        - delta/2*np.vdot(-Kyk1.flatten(), -Kyk1.flatten())
                        - np.vdot(xk.flatten(), (-Kyk1).flatten())
                        + np.sum(Kyk2)
                        - 1/(2*self.irgn_par["lambd"])
                        * np.vdot(r.flatten(), r.flatten())
                        - np.vdot(res.flatten(), r.flatten())
                        - 1 / (2 * self.irgn_par["omega"])
                        * np.vdot(z1[self.unknowns_TGV:],
                                  z1[self.unknowns_TGV:])).real
                else:
                    primal_new = (
                        self.irgn_par["lambd"]/2 *
                        np.vdot(Axold-res, Axold-res) +
                        alpha*np.sum(abs((gradx[:, :self.unknowns_TGV]-v))) +
                        beta*np.sum(abs(symgrad_v)) +
                        1/(2*delta)*np.vdot(x_new-xk, x_new-xk)).real

                    dual = (
                        - delta/2*np.vdot(-Kyk1.flatten(), -Kyk1.flatten())
                        - np.vdot(xk.flatten(), (-Kyk1).flatten())
                        + np.sum(Kyk2)
                        - 1/(2*self.irgn_par["lambd"])
                        * np.vdot(r.flatten(), r.flatten())
                        - np.vdot(res.flatten(), r.flatten())).real

                gap = np.abs(primal_new - dual)
                if myit == 0:
                    gap_init = gap
                if np.abs((primal-primal_new) / self.fval_init) <\
                   self.irgn_par["tol"]:
                    print("Terminated at iteration %d because the energy "
                          "decrease in the primal problem was less than %.3e" %
                          (myit, np.abs(primal-primal_new) / self.fval_init))
                    self.v = v_new
                    self.r = r
                    self.z1 = z1
                    self.z2 = z2
                    return x_new
                if (gap > gap_old*self.irgn_par["stag"]) and myit > 1:
                    self.v = v_new
                    self.r = r
                    self.z1 = z1
                    self.z2 = z2
                    print("Terminated at iteration %d "
                          "because the method stagnated" % (myit))
                    return x_new
                if np.abs((gap-gap_old)/gap_init) < self.irgn_par["tol"]:
                    self.v = v_new
                    self.r = r
                    self.z1 = z1
                    self.z2 = z2
                    print("Terminated at iteration %d because the relative "
                          "energy decrease of the PD gap was less than %.3e" %
                          (myit, np.abs((gap-gap_old) / gap_init)))
                    return x_new
                primal = primal_new
                gap_old = gap
                sys.stdout.write(
                    "Iteration: %04d ---- Primal: "
                    "%2.2e, Dual: %2.2e, Gap: %2.2e \r"
                    % (myit, 1000*primal/self.fval_init,
                       1000*dual/self.fval_init,
                       1000*gap/self.fval_init))
                sys.stdout.flush()
            (x, x_new) = (x_new, x)
            (v, v_new) = (v_new, v)

        self.v = v
        self.r = r
        self.z1 = z1
        self.z2 = z2
        return x

    def tv_solve_3D(self, x, res, iters):
        alpha = self.irgn_par["gamma"]
        tau = self.tau
        tau_new = np.float32(0)

        xk = x.copy()
        x_new = np.zeros_like(x)

        r = np.zeros_like(self.r)
        r_new = np.zeros_like(r)
        z1 = np.zeros_like(self.z1)
        z1_new = np.zeros_like(z1)
        res = (res).astype(DTYPE)

        delta = self.irgn_par["delta"]
        omega = self.irgn_par["omega"]
        mu = 1/delta
        theta_line = np.float32(1.0)
        beta_line = np.float32(400)
        beta_new = np.float32(0)
        mu_line = np.float32(0.5)
        delta_line = np.float32(1)
        ynorm = np.float32(0.0)
        lhs = np.float32(0.0)
        primal = np.float32(0.0)
        primal_new = np.float32(0)
        dual = np.float32(0.0)
        gap_init = np.float32(0.0)
        gap_old = np.float32(0.0)

        self.eval_const()

        Kyk1 = np.zeros_like(x)
        Kyk1_new = np.zeros_like(x)
        gradx = np.zeros_like(z1)
        gradx_xold = np.zeros_like(z1)
        Axold = np.zeros_like(res)
        Ax = np.zeros_like(res)

    # Allocate temporary Arrays
        (Axold_part, Kyk1_part, xk_part, res_part, z1_new_part,
         r_new_part, Kyk1_new_part,
         x_new_part, Ax_part, gradx_part, gradx_xold_part,
         x_part, r_part, z1_part) = self.preallocate_space(TGV=False)

    # Warmup
        self.stream_initial(
            (x, r, z1, Axold, Kyk1, gradx_xold),
            (x_part, r_part, z1_part, Axold_part, Kyk1_part, gradx_xold_part),
            TGV=False)

        for myit in range(iters):
            self.stream_primal_update(
                (x_new, xk, x, Kyk1, gradx, Ax),
                (x_new_part, xk_part, x_part, Kyk1_part, gradx_part, Ax_part),
                tau, delta, TGV=False)

            beta_new = beta_line*(1+mu*tau)
            tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
            beta_line = beta_new

            while True:
                theta_line = tau_new/tau

                ynorm = 0
                lhs = 0

                (ynorm, lhs) = self.stream_dual_update(
                    (z1, gradx, gradx_xold, r, Ax, Axold, res, Kyk1,
                     z1_new, r_new, Kyk1_new),
                    (z1_part, z1_new_part, gradx_part,
                     gradx_xold_part, r_part, r_new_part, Ax_part, Axold_part,
                     res_part, Kyk1_part, Kyk1_new_part),
                    (ynorm, lhs, beta_line, tau_new, omega, alpha, theta_line),
                    TGV=False)

                if np.sqrt(beta_line)*tau_new*(abs(lhs)**(1/2)) <= \
                   (abs(ynorm)**(1/2))*delta_line:
                    break
                else:
                    tau_new = tau_new*mu_line
            (Kyk1, Kyk1_new,  Axold, Ax, z1, z1_new, r, r_new, gradx_xold,
             gradx, tau) = (
             Kyk1_new, Kyk1,  Ax, Axold, z1_new, z1, r_new, r, gradx,
             gradx_xold, tau_new)

            if not np.mod(myit, 10):
                if self.irgn_par["display_iterations"]:
                    self.model.plot_unknowns(np.transpose(x_new, [1, 0, 2, 3]))
                if self.unknowns_H1 > 0:
                    primal_new = (
                        self.irgn_par["lambd"]/2 *
                        np.vdot(Axold-res, Axold-res) +
                        alpha*np.sum(abs((gradx[:, :self.unknowns_TGV]))) +
                        1/(2*delta)*np.vdot(x_new-xk, x_new-xk) +
                        self.irgn_par["omega"] / 2 *
                        np.vdot(gradx[self.unknowns_TGV:],
                                gradx[self.unknowns_TGV:])).real

                    dual = (
                        -delta/2*np.vdot(-Kyk1, -Kyk1) - np.vdot(xk, (-Kyk1))
                        - 1/(2*self.irgn_par["lambd"])*np.vdot(r, r)
                        - np.vdot(res, r)
                        - 1 / (2 * self.irgn_par["omega"])
                        * np.vdot(z1[self.unknowns_TGV:],
                                  z1[self.unknowns_TGV:])).real
                else:
                    primal_new = (
                        self.irgn_par["lambd"]/2 *
                        np.vdot(Axold-res, Axold-res) +
                        alpha*np.sum(abs((gradx[:, :self.unknowns_TGV]))) +
                        1/(2*delta)*np.vdot(x_new-xk, x_new-xk)).real

                    dual = (
                        -delta/2*np.vdot(-Kyk1, -Kyk1) - np.vdot(xk, (-Kyk1))
                        - 1/(2*self.irgn_par["lambd"])*np.vdot(r, r)
                        - np.vdot(res, r)).real

                gap = np.abs(primal_new - dual)
                if myit == 0:
                    gap_init = gap
                if np.abs(primal-primal_new) / self.fval_init < \
                   self.irgn_par["tol"]:
                    print("Terminated at iteration %d because the energy "
                          "decrease in the primal problem was less than %.3e" %
                          (myit, np.abs(primal-primal_new) / self.fval_init))
                    self.r = r
                    self.z1 = z1
                    return x_new
                if (gap > gap_old*self.irgn_par["stag"]) and myit > 1:
                    self.r = r
                    self.z1 = z1
                    print("Terminated at iteration %d because "
                          "the method stagnated" % (myit))
                    return x_new
                if np.abs((gap-gap_old)/gap_init) < self.irgn_par["tol"]:
                    self.r = r
                    self.z1 = z1
                    print("Terminated at iteration %d because the relative "
                          "energy decrease of the PD gap was less than %.3e" %
                          (myit, np.abs((gap-gap_old) / gap_init)))
                    return x_new
                primal = primal_new
                gap_old = gap
                sys.stdout.write(
                    "Iteration: %04d ---- "
                    "Primal: %2.2e, Dual: %2.2e, Gap: %2.2e \r"
                    % (myit, 1000 * primal / self.fval_init,
                       1000 * dual / self.fval_init,
                       1000 * gap / self.fval_init))
                sys.stdout.flush()
            (x, x_new) = (x_new, x)

        self.r = r
        self.z1 = z1
        return x

    def execute(self, TV=0, imagespace=0, reco_2D=0):
        if reco_2D:
            print("2D currently not implemented, "
                  "3D can be used with a single slice.")
            return
        else:
            self.irgn_par["lambd"] *= self.SNR_est
            self.delta = self.irgn_par["delta"]
            self.delta_max = self.irgn_par["delta_max"]

            if imagespace:
                self.execute_3D_imagespace(TV)
            else:
                self.execute_3D(TV)

    def FT_streamed(self, outp, inp):
        cl_out = []
        j = 0
        for i in range(self.num_dev):
            cl_out.append(
                clarray.empty(
                    self.queue[4*i],
                    (self.par_slices+self.overlap, self.NScan,
                     self.NC, self.Nproj, self.N),
                    dtype=DTYPE))
            cl_out.append(
                clarray.empty(
                    self.queue[4*i+1],
                    (self.par_slices+self.overlap, self.NScan,
                     self.NC, self.Nproj, self.N),
                    dtype=DTYPE))
        cl_data = []
        for i in range(self.num_dev):
            idx_start = i*self.par_slices
            idx_stop = (i+1)*self.par_slices+self.overlap
            cl_data.append(clarray.to_device(
                self.queue[4*i], inp[idx_start:idx_stop, ...]))
            cl_out[2*i].add_event(
                self.NUFFT[2*i].fwd_NUFFT(cl_out[2*i], cl_data[i]))
        for i in range(self.num_dev):
            idx_start = (i+1+self.num_dev-1)*self.par_slices
            idx_stop = (i+2+self.num_dev-1)*self.par_slices
            if idx_stop == self.NSlice:
                idx_start -= self.overlap
            else:
                idx_stop += self.overlap
            cl_data.append(clarray.to_device(
                self.queue[4*i+1], inp[idx_start:idx_stop, ...]))
            cl_out[2*i+1].add_event(
                self.NUFFT[2*i+1].fwd_NUFFT(
                    cl_out[2*i+1], cl_data[self.num_dev+i]))

        for j in range(2*self.num_dev,
                       int(self.NSlice/(2*self.par_slices*self.num_dev) +
                           (2*self.num_dev-1))):
            for i in range(self.num_dev):
                self.queue[4*(self.num_dev-1)+3].finish()
                if i > 1:
                    self.queue[4*(i-1)+2].finish()
                idx_start = i*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
                idx_stop = (i+1)*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev)) *\
                    self.par_slices+self.overlap
                cl_out[2*i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2],
                        outp[idx_start:idx_stop, ...],
                        cl_out[2*i].data,
                        wait_for=cl_out[2*i].events, is_blocking=False))
            for i in range(self.num_dev):
                idx_start = i*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
                idx_stop = ((i+1)*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices) +\
                    self.overlap
                cl_data[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], cl_data[i].data,
                        inp[idx_start:idx_stop, ...],
                        wait_for=cl_data[i].events, is_blocking=False))
            for i in range(self.num_dev):
                cl_out[2*i].add_event(
                    self.NUFFT[2*i].fwd_NUFFT(cl_out[2*i], cl_data[i]))
            for i in range(self.num_dev):
                self.queue[4*(self.num_dev-1)+2].finish()
                if i > 1:
                    self.queue[4*(i-1)+3].finish()
                idx_start = i*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *\
                    self.par_slices
                idx_stop = (i+1)*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *\
                    self.par_slices+self.overlap
                cl_out[2*i+1].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], outp[idx_start:idx_stop, ...],
                        cl_out[2*i+1].data, wait_for=cl_out[2*i+1].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                idx_start = i*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev) *\
                    self.par_slices
                idx_stop = (i+1)*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev) *\
                    self.par_slices
                if idx_stop == self.NSlice:
                    idx_start -= self.overlap
                else:
                    idx_stop += self.overlap
                cl_data[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], cl_data[i+self.num_dev].data,
                        inp[idx_start:idx_stop, ...],
                        wait_for=cl_data[i+self.num_dev].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                cl_out[2*i+1].add_event(
                    self.NUFFT[2*i+1].fwd_NUFFT(
                        cl_out[2*i+1], cl_data[i+self.num_dev]))
        if j < 2*self.num_dev:
            j = 2*self.num_dev
        else:
            j += 1
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+3].finish()
            if i > 1:
                self.queue[4*(i-1)+2].finish()
            idx_start = i*self.par_slices+(
                2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(
                2*self.num_dev*(j-2*self.num_dev))*self.par_slices+self.overlap
            cl_out[2*i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+2], outp[idx_start:idx_stop, ...],
                    cl_out[2*i].data, wait_for=cl_out[2*i].events,
                    is_blocking=False))
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+2].finish()
            if i > 1:
                self.queue[4*(i-1)+3].finish()
            idx_start = i*self.par_slices+(
                2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(
                2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            if idx_stop == self.NSlice:
                idx_start -= self.overlap
            else:
                idx_stop += self.overlap
            cl_out[2*i+1].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+3], outp[idx_start:idx_stop, ...],
                    cl_out[2*i+1].data, wait_for=cl_out[2*i+1].events,
                    is_blocking=False))
        for i in range(self.num_dev):
            self.queue[4*i+2].finish()
            self.queue[4*i+3].finish()
        del cl_out

    def operator_forward_streamed(self, outp, inp):
        cl_out = []
        self.coil_buf_part = []
        self.grad_buf_part = []
        j = 0

        for i in range(self.num_dev):
            cl_out.append(
                clarray.empty(
                    self.queue[4*i],
                    (self.par_slices+self.overlap, self.NScan,
                     self.NC, self.Nproj, self.N), dtype=DTYPE))
            cl_out.append(
                clarray.empty(
                    self.queue[4*i+1],
                    (self.par_slices+self.overlap, self.NScan, self.NC,
                     self.Nproj, self.N), dtype=DTYPE))

        cl_data = []
        for i in range(self.num_dev):
            idx_start = i*self.par_slices
            idx_stop = (i+1)*self.par_slices+self.overlap
            cl_data.append(
                clarray.to_device(
                    self.queue[4*i], inp[idx_start:idx_stop, ...]))
            self.coil_buf_part.append(
                clarray.to_device(
                    self.queue[4*i], self.C[idx_start:idx_stop, ...],
                    allocator=self.alloc[i]))
            self.grad_buf_part.append(
                clarray.to_device(
                    self.queue[4*i],
                    self.grad_x[idx_start:idx_stop, ...],
                    allocator=self.alloc[i]))
        for i in range(self.num_dev):
            cl_out[2*i].add_event(
                self.operator_forward(cl_out[2*i], cl_data[i], i, 0))
        for i in range(self.num_dev):
            idx_start = (i+1+self.num_dev-1)*self.par_slices
            idx_stop = (i+2+self.num_dev-1)*self.par_slices
            if idx_stop == self.NSlice:
                idx_start -= self.overlap
            else:
                idx_stop += self.overlap

            cl_data.append(
                clarray.to_device(
                    self.queue[4*i+1], inp[idx_start:idx_stop, ...]))
            self.coil_buf_part.append(
                clarray.to_device(
                    self.queue[4*i+1], self.C[idx_start:idx_stop, ...],
                    allocator=self.alloc[i]))
            self.grad_buf_part.append(
                clarray.to_device(
                    self.queue[4*i+1],
                    self.grad_x[idx_start:idx_stop, ...],
                    allocator=self.alloc[i]))
        for i in range(self.num_dev):
            cl_out[2*i+1].add_event(
                self.operator_forward(
                    cl_out[2*i+1], cl_data[self.num_dev+i], i, 1))
        for j in range(2*self.num_dev,
                       int(self.NSlice/(2*self.par_slices*self.num_dev) +
                           (2*self.num_dev-1))):
            for i in range(self.num_dev):
                self.queue[4*(self.num_dev-1)+3].finish()
                if i > 1:
                    self.queue[4*(i-1)+2].finish()
                idx_start = i*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
                idx_stop = (i+1)*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev))*self.par_slices
                cl_out[2*i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2],
                        outp[idx_start:idx_stop, ...],
                        cl_out[2*i].data,
                        wait_for=cl_out[2*i].events, is_blocking=False))
            for i in range(self.num_dev):
                idx_start = i*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
                idx_stop = ((i+1)*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices) +\
                    self.overlap
                cl_data[i] = clarray.to_device(
                    self.queue[4*i], inp[idx_start:idx_stop, ...])
                self.coil_buf_part[i] = (clarray.to_device(
                    self.queue[4*i], self.C[idx_start:idx_stop, ...]))
                self.grad_buf_part[i] = (clarray.to_device(
                    self.queue[4*i], self.grad_x[idx_start:idx_stop, ...]))
            for i in range(self.num_dev):
                cl_out[2*i].add_event(
                    self.operator_forward(cl_out[2*i], cl_data[i], i, 0))
            for i in range(self.num_dev):
                self.queue[4*(self.num_dev-1)+2].finish()
                if i > 1:
                    self.queue[4*(i-1)+3].finish()
                idx_start = i*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *\
                    self.par_slices
                idx_stop = (i+1)*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *\
                    self.par_slices
                cl_out[2*i+1].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], outp[idx_start:idx_stop, ...],
                        cl_out[2*i+1].data, wait_for=cl_out[2*i+1].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                idx_start = i*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev) *\
                    self.par_slices
                idx_stop = (i+1)*self.par_slices+(
                    2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev) *\
                    self.par_slices
                if idx_stop == self.NSlice:
                    idx_start -= self.overlap
                else:
                    idx_stop += self.overlap
                cl_data[i+self.num_dev] = clarray.to_device(
                    self.queue[4*i+1], inp[idx_start:idx_stop, ...])
                self.coil_buf_part[self.num_dev+i] = (
                    clarray.to_device(self.queue[4*i+1],
                                      self.C[idx_start:idx_stop, ...]))
                self.grad_buf_part[self.num_dev+i] = (
                    clarray.to_device(self.queue[4*i+1],
                                      self.grad_x[idx_start:idx_stop, ...]))
            for i in range(self.num_dev):
                cl_out[2*i+1].add_event(
                    self.operator_forward(
                        cl_out[2*i+1], cl_data[self.num_dev+i], i, 1))
        if j < 2*self.num_dev:
            j = 2*self.num_dev
        else:
            j += 1
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+3].finish()
            if i > 1:
                self.queue[4*(i-1)+2].finish()
            idx_start = i*self.par_slices+(
                2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(
                2*self.num_dev*(j-2*self.num_dev))*self.par_slices
            cl_out[2*i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+2], outp[idx_start:idx_stop, ...],
                    cl_out[2*i].data, wait_for=cl_out[2*i].events,
                    is_blocking=False))
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+2].finish()
            if i > 1:
                self.queue[4*(i-1)+3].finish()
            idx_start = i*self.par_slices+(
                2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(
                2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            if idx_stop == self.NSlice:
                idx_start -= self.overlap
            else:
                idx_stop += self.overlap
            cl_out[2*i+1].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+3], outp[idx_start:idx_stop, ...],
                    cl_out[2*i+1].data, wait_for=cl_out[2*i+1].events,
                    is_blocking=False))
        for i in range(self.num_dev):
            self.queue[4*i+2].finish()
            self.queue[4*i+3].finish()

    def preallocate_space(self, TGV=True):
        Axold_part = []
        Kyk1_part = []
        xk_part = []
        res_part = []
        z1_new_part = []
        r_new_part = []
        Kyk1_new_part = []
        x_new_part = []
        Ax_part = []
        gradx_part = []
        gradx_xold_part = []
        x_part = []
        r_part = []
        z1_part = []
        if TGV:
            v_part = []
            v_new_part = []
            symgrad_v_part = []
            symgrad_v_vold_part = []
            Kyk2_part = []
            Kyk2_new_part = []
            z2_part = []
            z2_new_part = []
        for i in range(2*self.num_dev):
            Axold_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.NScan,
                     self.NC, self.Nproj, self.N),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            Kyk1_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.unknowns,
                     self.dimY, self.dimX),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            xk_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.unknowns,
                     self.dimY, self.dimX),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            res_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.NScan,
                     self.NC, self.Nproj, self.N),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            z1_new_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.unknowns,
                     self.dimY, self.dimX, 4),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            z1_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.unknowns,
                     self.dimY, self.dimX, 4),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            r_new_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.NScan,
                     self.NC, self.Nproj, self.N),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            r_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.NScan,
                     self.NC, self.Nproj, self.N),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            Kyk1_new_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.unknowns,
                     self.dimY, self.dimX),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            x_new_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.unknowns,
                     self.dimY, self.dimX),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            x_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.unknowns,
                     self.dimY, self.dimX),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            Ax_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.NScan,
                     self.NC, self.Nproj, self.N),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            gradx_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.unknowns,
                     self.dimY, self.dimX, 4),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            gradx_xold_part.append(
                clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.unknowns,
                     self.dimY, self.dimX, 4),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
            if TGV:
                v_part.append(clarray.empty(
                    self.queue[4*int(np.mod(i, self.num_dev))],
                    (self.par_slices+self.overlap, self.unknowns,
                     self.dimY, self.dimX, 4),
                    dtype=DTYPE,
                    allocator=self.alloc[int(np.mod(i, self.num_dev))]))
                v_new_part.append(
                    clarray.empty(
                        self.queue[4*int(np.mod(i, self.num_dev))],
                        (self.par_slices+self.overlap, self.unknowns,
                         self.dimY, self.dimX, 4),
                        dtype=DTYPE,
                        allocator=self.alloc[int(np.mod(i, self.num_dev))]))
                z2_part.append(
                    clarray.empty(
                        self.queue[4*int(np.mod(i, self.num_dev))],
                        (self.par_slices+self.overlap, self.unknowns,
                         self.dimY, self.dimX, 8),
                        dtype=DTYPE,
                        allocator=self.alloc[int(np.mod(i, self.num_dev))]))
                z2_new_part.append(
                    clarray.empty(
                        self.queue[4*int(np.mod(i, self.num_dev))],
                        (self.par_slices+self.overlap, self.unknowns,
                         self.dimY, self.dimX, 8),
                        dtype=DTYPE,
                        allocator=self.alloc[int(np.mod(i, self.num_dev))]))
                symgrad_v_part.append(
                    clarray.empty(
                        self.queue[4*int(np.mod(i, self.num_dev))],
                        (self.par_slices+self.overlap, self.unknowns,
                         self.dimY, self.dimX, 8),
                        dtype=DTYPE,
                        allocator=self.alloc[int(np.mod(i, self.num_dev))]))
                symgrad_v_vold_part.append(
                    clarray.empty(
                        self.queue[4*int(np.mod(i, self.num_dev))],
                        (self.par_slices+self.overlap, self.unknowns,
                         self.dimY, self.dimX, 8),
                        dtype=DTYPE,
                        allocator=self.alloc[int(np.mod(i, self.num_dev))]))
                Kyk2_part.append(
                    clarray.empty(
                        self.queue[4*int(np.mod(i, self.num_dev))],
                        (self.par_slices+self.overlap, self.unknowns,
                         self.dimY, self.dimX, 4),
                        dtype=DTYPE,
                        allocator=self.alloc[int(np.mod(i, self.num_dev))]))
                Kyk2_new_part.append(
                    clarray.empty(
                        self.queue[4*int(np.mod(i, self.num_dev))],
                        (self.par_slices+self.overlap, self.unknowns,
                         self.dimY, self.dimX, 4),
                        dtype=DTYPE,
                        allocator=self.alloc[int(np.mod(i, self.num_dev))]))

        if TGV:
            return (Axold_part, Kyk1_part, Kyk2_part, xk_part, v_part,
                    res_part, z1_new_part, z2_new_part, r_new_part,
                    Kyk1_new_part, Kyk2_new_part, x_new_part, Ax_part,
                    v_new_part, gradx_part, gradx_xold_part, symgrad_v_part,
                    symgrad_v_vold_part, x_part, r_part, z1_part, z2_part)
        else:
            return (Axold_part, Kyk1_part, xk_part, res_part, z1_new_part,
                    r_new_part, Kyk1_new_part, x_new_part, Ax_part,
                    gradx_part, gradx_xold_part, x_part, r_part, z1_part)

    def stream_initial(self, arrays, parts, TGV=True):
        if TGV:
            (x, r, z1, z2, Axold, Kyk1, Kyk2,
             gradx_xold, symgrad_v_vold, v) = arrays
            (x_part, r_part, z1_part, z2_part, Axold_part, Kyk1_part,
             Kyk2_part, gradx_xold_part, symgrad_v_vold_part, v_part) = parts
        else:
            (x, r, z1, Axold, Kyk1, gradx_xold) = arrays
            (x_part, r_part, z1_part, Axold_part, Kyk1_part,
             gradx_xold_part) = parts
        j = 0
        last = 0
        for i in range(self.num_dev):
            idx_start = (self.NSlice)-((i+1)*self.par_slices)-self.overlap
            idx_stop = (self.NSlice)-(i*self.par_slices)
            x_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], x_part[i].data,
                    x[idx_start:idx_stop, ...],
                    wait_for=x_part[i].events, is_blocking=False))
            r_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], r_part[i].data,
                    r[idx_start:idx_stop, ...],
                    wait_for=r_part[i].events, is_blocking=False))
            z1_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], z1_part[i].data,
                    z1[idx_start:idx_stop, ...],
                    wait_for=z1_part[i].events, is_blocking=False))
            self.coil_buf_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], self.coil_buf_part[i].data,
                    self.C[idx_start:idx_stop, ...],
                    wait_for=self.coil_buf_part[i].events, is_blocking=False))
            self.grad_buf_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], self.grad_buf_part[i].data,
                    self.grad_x[idx_start:idx_stop, ...],
                    wait_for=self.grad_buf_part[i].events, is_blocking=False))
            if TGV:
                v_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], v_part[i].data,
                        v[idx_start:idx_stop, ...],
                        wait_for=v_part[i].events, is_blocking=False))

        for i in range(self.num_dev):
            if i == 0:
                last = 1
            else:
                last = 0
            if TGV:
                symgrad_v_vold_part[i].add_event(
                    self.sym_grad(
                        symgrad_v_vold_part[i], v_part[i], i, 0))
            Axold_part[i].add_event(
                self.operator_forward(
                    Axold_part[i], x_part[i], i, 0))
            Kyk1_part[i].add_event(
                self.operator_adjoint(
                    Kyk1_part[i], r_part[i], z1_part[i], i, 0, last))
        last = 0
        for i in range(self.num_dev):
            idx_start = (self.NSlice)-((i+2+self.num_dev-1)*self.par_slices)
            idx_stop = (self.NSlice)-((i+1+self.num_dev-1)*self.par_slices)
            if idx_start == 0:
                idx_stop += self.overlap
            else:
                idx_start -= self.overlap
            x_part[self.num_dev+i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], x_part[self.num_dev+i].data,
                    x[idx_start:idx_stop, ...],
                    wait_for=x_part[self.num_dev+i].events, is_blocking=False))
            r_part[self.num_dev+i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], r_part[self.num_dev+i].data,
                    r[idx_start:idx_stop, ...],
                    wait_for=r_part[self.num_dev+i].events, is_blocking=False))
            z1_part[self.num_dev+i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], z1_part[self.num_dev+i].data,
                    z1[idx_start:idx_stop, ...],
                    wait_for=z1_part[self.num_dev+i].events,
                    is_blocking=False))
            self.coil_buf_part[self.num_dev+i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], self.coil_buf_part[self.num_dev+i].data,
                    self.C[idx_start:idx_stop, ...],
                    wait_for=self.coil_buf_part[self.num_dev+i].events,
                    is_blocking=False))
            self.grad_buf_part[self.num_dev+i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], self.grad_buf_part[self.num_dev+i].data,
                    self.grad_x[idx_start:idx_stop, ...],
                    wait_for=self.grad_buf_part[self.num_dev+i].events,
                    is_blocking=False))
            if TGV:
                v_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], v_part[i+self.num_dev].data,
                        v[idx_start:idx_stop, ...],
                        wait_for=v_part[i+self.num_dev].events,
                        is_blocking=False))
        for i in range(self.num_dev):
            if TGV:
                symgrad_v_vold_part[i+self.num_dev].add_event(
                    self.sym_grad(
                        symgrad_v_vold_part[i+self.num_dev],
                        v_part[self.num_dev+i], i, 1))
            Axold_part[i+self.num_dev].add_event(
                self.operator_forward(
                    Axold_part[i+self.num_dev],
                    x_part[self.num_dev+i], i, 1))
            Kyk1_part[i+self.num_dev].add_event(
                self.operator_adjoint(
                    Kyk1_part[i+self.num_dev],
                    r_part[self.num_dev+i], z1_part[self.num_dev+i],
                    i, 1, last))
    # Stream
        for j in range(2*self.num_dev,
                       int(self.NSlice /
                           (2*self.par_slices*self.num_dev) +
                           (2*self.num_dev-1))):
            for i in range(self.num_dev):
                # Get Data
                self.queue[4*(self.num_dev-1)+3].finish()
                if i > 1:
                    self.queue[4*(i-1)+2].finish()
                idx_start = (
                    self.NSlice -
                    ((i+1)*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev))*self.par_slices) -
                    self.overlap)
                idx_stop = (
                    self.NSlice -
                    (i*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev)*self.par_slices)))
                Axold_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], Axold[idx_start:idx_stop, ...],
                        Axold_part[i].data,
                        wait_for=Axold_part[i].events, is_blocking=False))
                Kyk1_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], Kyk1[idx_start:idx_stop, ...],
                        Kyk1_part[i].data,
                        wait_for=Kyk1_part[i].events, is_blocking=False))
                if TGV:
                    symgrad_v_vold_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+2],
                            symgrad_v_vold[idx_start:idx_stop, ...],
                            symgrad_v_vold_part[i].data,
                            wait_for=symgrad_v_vold_part[i].events,
                            is_blocking=False))
                # Put Data
            for i in range(self.num_dev):
                idx_start = (
                    self.NSlice -
                    ((i+1)*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices) -
                    self.overlap)
                idx_stop = (
                    self.NSlice -
                    (i*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)))
                x_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], x_part[i].data,
                        x[idx_start:idx_stop, ...],
                        wait_for=x_part[i].events, is_blocking=False))
                r_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], r_part[i].data,
                        r[idx_start:idx_stop, ...],
                        wait_for=r_part[i].events, is_blocking=False))
                z1_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], z1_part[i].data,
                        z1[idx_start:idx_stop, ...],
                        wait_for=z1_part[i].events, is_blocking=False))
                self.coil_buf_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], self.coil_buf_part[i].data,
                        self.C[idx_start:idx_stop, ...],
                        wait_for=self.coil_buf_part[i].events,
                        is_blocking=False))
                self.grad_buf_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], self.grad_buf_part[i].data,
                        self.grad_x[idx_start:idx_stop, ...],
                        wait_for=self.grad_buf_part[i].events,
                        is_blocking=False))
                if TGV:
                    v_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i], v_part[i].data,
                            v[idx_start:idx_stop, ...],
                            wait_for=v_part[i].events, is_blocking=False))
            for i in range(self.num_dev):
                if TGV:
                    symgrad_v_vold_part[i].add_event(
                        self.sym_grad(
                            symgrad_v_vold_part[i], v_part[i], i, 0))
                Axold_part[i].add_event(
                    self.operator_forward(
                        Axold_part[i], x_part[i], i, 0))
                Kyk1_part[i].add_event(
                    self.operator_adjoint(
                        Kyk1_part[i], r_part[i], z1_part[i], i, 0, last))
            for i in range(self.num_dev):
                self.queue[4*(self.num_dev-1)+2].finish()
                if i > 1:
                    self.queue[4*(i-1)+3].finish()
                idx_start = (
                    self.NSlice -
                    ((i+1)*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                     self.par_slices)-self.overlap)
                idx_stop = (
                    self.NSlice -
                    (i*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                     self.par_slices))
                Axold_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], Axold[idx_start:idx_stop, ...],
                        Axold_part[i+self.num_dev].data,
                        wait_for=Axold_part[i+self.num_dev].events,
                        is_blocking=False))
                Kyk1_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], Kyk1[idx_start:idx_stop, ...],
                        Kyk1_part[i+self.num_dev].data,
                        wait_for=Kyk1_part[i+self.num_dev].events,
                        is_blocking=False))
                if TGV:
                    symgrad_v_vold_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+3],
                            symgrad_v_vold[idx_start:idx_stop, ...],
                            symgrad_v_vold_part[i+self.num_dev].data,
                            wait_for=symgrad_v_vold_part[i +
                                                         self.num_dev].events,
                            is_blocking=False))
            for i in range(self.num_dev):
                idx_start = (
                    self.NSlice -
                    ((i+1)*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev) *
                     self.par_slices))
                idx_stop = (
                    self.NSlice -
                    (i*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev) *
                     self.par_slices))
                if idx_start == 0:
                    idx_stop += self.overlap
                else:
                    idx_start -= self.overlap
                x_part[self.num_dev+i].add_event(
                      cl.enqueue_copy(
                          self.queue[4*i+1], x_part[self.num_dev+i].data,
                          x[idx_start:idx_stop, ...],
                          wait_for=x_part[self.num_dev+i].events,
                          is_blocking=False))
                r_part[self.num_dev+i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], r_part[self.num_dev+i].data,
                        r[idx_start:idx_stop, ...],
                        wait_for=r_part[self.num_dev+i].events,
                        is_blocking=False))
                z1_part[self.num_dev+i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], z1_part[self.num_dev+i].data,
                        z1[idx_start:idx_stop, ...],
                        wait_for=z1_part[self.num_dev+i].events,
                        is_blocking=False))
                self.coil_buf_part[self.num_dev+i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], self.coil_buf_part[
                            self.num_dev+i].data,
                        self.C[idx_start:idx_stop, ...],
                        wait_for=self.coil_buf_part[self.num_dev+i].events,
                        is_blocking=False))
                self.grad_buf_part[self.num_dev+i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1],
                        self.grad_buf_part[self.num_dev+i].data,
                        self.grad_x[idx_start:idx_stop, ...],
                        wait_for=self.grad_buf_part[self.num_dev+i].events,
                        is_blocking=False))
                if TGV:
                    v_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+1], v_part[i+self.num_dev].data,
                            v[idx_start:idx_stop, ...],
                            wait_for=v_part[i+self.num_dev].events,
                            is_blocking=False))
            for i in range(self.num_dev):
                self.queue[4*i+3].finish()
                if TGV:
                    symgrad_v_vold_part[i+self.num_dev].add_event(
                        self.sym_grad(
                            symgrad_v_vold_part[i+self.num_dev],
                            v_part[self.num_dev+i], i, 1))
                Axold_part[i+self.num_dev].add_event(
                    self.operator_forward(
                        Axold_part[i+self.num_dev], x_part[self.num_dev+i],
                        i, 1))
                Kyk1_part[i+self.num_dev].add_event(
                    self.operator_adjoint(
                        Kyk1_part[i+self.num_dev], r_part[self.num_dev+i],
                        z1_part[self.num_dev+i], i, 1, last))
    # Collect last block
        if j < 2*self.num_dev:
            j = 2*self.num_dev
        else:
            j += 1
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+3].finish()
            if i > 1:
                self.queue[4*(i-1)+2].finish()
            idx_start = (
                self.NSlice -
                ((i+1)*self.par_slices +
                 (2*self.num_dev*(j-2*self.num_dev)) *
                 self.par_slices)-self.overlap)
            idx_stop = (
                self.NSlice -
                (i*self.par_slices +
                 (2*self.num_dev*(j-2*self.num_dev)*self.par_slices)))
            Axold_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+2], Axold[idx_start:idx_stop, ...],
                    Axold_part[i].data,
                    wait_for=Axold_part[i].events, is_blocking=False))
            Kyk1_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+2], Kyk1[idx_start:idx_stop, ...],
                    Kyk1_part[i].data,
                    wait_for=Kyk1_part[i].events, is_blocking=False))
            if TGV:
                symgrad_v_vold_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2],
                        symgrad_v_vold[idx_start:idx_stop, ...],
                        symgrad_v_vold_part[i].data,
                        wait_for=symgrad_v_vold_part[i].events,
                        is_blocking=False))
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+2].finish()
            if i > 1:
                self.queue[4*(i-1)+3].finish()
            idx_start = (
                self.NSlice -
                ((i+1)*self.par_slices +
                 (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                 self.par_slices))
            idx_stop = (
                self.NSlice -
                (i*self.par_slices +
                 (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                 self.par_slices))
            if idx_start == 0:
                idx_stop += self.overlap
            else:
                idx_start -= self.overlap
            Axold_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+3], Axold[idx_start:idx_stop, ...],
                    Axold_part[i+self.num_dev].data,
                    wait_for=Axold_part[i+self.num_dev].events,
                    is_blocking=False))
            Kyk1_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+3], Kyk1[idx_start:idx_stop, ...],
                    Kyk1_part[i+self.num_dev].data,
                    wait_for=Kyk1_part[i+self.num_dev].events,
                    is_blocking=False))
            if TGV:
                symgrad_v_vold_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3],
                        symgrad_v_vold[idx_start:idx_stop, ...],
                        symgrad_v_vold_part[i+self.num_dev].data,
                        wait_for=symgrad_v_vold_part[i+self.num_dev].events,
                        is_blocking=False))
    # Warmup
        j = 0
        first = 0
        for i in range(self.num_dev):
            idx_start = i*self.par_slices
            idx_stop = (i+1)*self.par_slices+self.overlap
            x_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], x_part[i].data,
                    x[idx_start:idx_stop, ...],
                    wait_for=x_part[i].events, is_blocking=False))
            if TGV:
                z1_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], z1_part[i].data,
                        z1[idx_start:idx_stop, ...],
                        wait_for=z1_part[i].events, is_blocking=False))
                z2_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], z2_part[i].data,
                        z2[idx_start:idx_stop, ...],
                        wait_for=z2_part[i].events, is_blocking=False))
        for i in range(self.num_dev):
            if i == 0:
                first = 1
            else:
                first = 0
            if TGV:
                Kyk2_part[i].add_event(
                    self.update_Kyk2(
                        Kyk2_part[i], z2_part[i], z1_part[i], i, 0, first))
            gradx_xold_part[i].add_event(
                self.f_grad(
                    gradx_xold_part[i], x_part[i], i, 0))
        first = 0
        for i in range(self.num_dev):
            idx_start = (i+1+self.num_dev-1)*self.par_slices
            idx_stop = (i+2+self.num_dev-1)*self.par_slices
            if idx_stop == self.NSlice:
                idx_start -= self.overlap
            else:
                idx_stop += self.overlap
            x_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], x_part[i+self.num_dev].data,
                    x[idx_start:idx_stop, ...],
                    wait_for=x_part[i+self.num_dev].events, is_blocking=False))
            if TGV:
                z1_part[self.num_dev+i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], z1_part[self.num_dev+i].data,
                        z1[idx_start:idx_stop, ...],
                        wait_for=z1_part[self.num_dev+i].events,
                        is_blocking=False))
                z2_part[self.num_dev+i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], z2_part[self.num_dev+i].data,
                        z2[idx_start:idx_stop, ...],
                        wait_for=z2_part[self.num_dev+i].events,
                        is_blocking=False))
        for i in range(self.num_dev):
            if TGV:
                Kyk2_part[i+self.num_dev].add_event(
                    self.update_Kyk2(
                        Kyk2_part[i+self.num_dev], z2_part[self.num_dev+i],
                        z1_part[self.num_dev+i], i, 1, first))
            gradx_xold_part[i+self.num_dev].add_event(
                self.f_grad(
                    gradx_xold_part[i+self.num_dev], x_part[self.num_dev+i],
                    i, 1))
    # Stream
        for j in range(2*self.num_dev,
                       int(self.NSlice/(2*self.par_slices*self.num_dev) +
                           (2*self.num_dev-1))):
            for i in range(self.num_dev):
                # Get Data
                self.queue[4*(self.num_dev-1)+3].finish()
                if i > 1:
                    self.queue[4*(i-1)+2].finish()
                idx_start = (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev) *
                              self.par_slices))
                idx_stop = ((i+1)*self.par_slices +
                            (2*self.num_dev*(j-2*self.num_dev)) *
                            self.par_slices+self.overlap)
                if TGV:
                    Kyk2_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+2], Kyk2[idx_start:idx_stop, ...],
                            Kyk2_part[i].data,
                            wait_for=Kyk2_part[i].events,
                            is_blocking=False))
                gradx_xold_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], gradx_xold[idx_start:idx_stop, ...],
                        gradx_xold_part[i].data,
                        wait_for=gradx_xold_part[i].events, is_blocking=False))
            for i in range(self.num_dev):
                # Put Data
                idx_start = (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev+1) *
                              self.par_slices))
                idx_stop = ((i+1)*self.par_slices +
                            (2*self.num_dev*(j-2*self.num_dev+1)) *
                            self.par_slices+self.overlap)
                x_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], x_part[i].data,
                        x[idx_start:idx_stop, ...],
                        wait_for=x_part[i].events, is_blocking=False))
                if TGV:
                    z1_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i], z1_part[i].data,
                            z1[idx_start:idx_stop, ...],
                            wait_for=z1_part[i].events, is_blocking=False))
                    z2_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i], z2_part[i].data,
                            z2[idx_start:idx_stop, ...],
                            wait_for=z2_part[i].events, is_blocking=False))
            for i in range(self.num_dev):
                if TGV:
                    Kyk2_part[i].add_event(
                        self.update_Kyk2(
                            Kyk2_part[i], z2_part[i], z1_part[i], i, 0, first))
                gradx_xold_part[i].add_event(
                    self.f_grad(
                        gradx_xold_part[i], x_part[i], i, 0))
            for i in range(self.num_dev):
                self.queue[4*(self.num_dev-1)+2].finish()
                if i > 1:
                    self.queue[4*(i-1)+3].finish()
                idx_start = (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                             self.par_slices)
                idx_stop = ((i+1)*self.par_slices +
                            (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                            self.par_slices+self.overlap)
                if TGV:
                    Kyk2_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+3], Kyk2[idx_start:idx_stop, ...],
                            Kyk2_part[i+self.num_dev].data,
                            wait_for=Kyk2_part[i+self.num_dev].events,
                            is_blocking=False))
                gradx_xold_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], gradx_xold[idx_start:idx_stop, ...],
                        gradx_xold_part[i+self.num_dev].data,
                        wait_for=gradx_xold_part[i+self.num_dev].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                idx_start = (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev+1) +
                              self.num_dev)*self.par_slices)
                idx_stop = ((i+1)*self.par_slices +
                            (2*self.num_dev*(j-2*self.num_dev+1) +
                             self.num_dev)*self.par_slices)
                if idx_stop == self.NSlice:
                    idx_start -= self.overlap
                else:
                    idx_stop += self.overlap
                x_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], x_part[i+self.num_dev].data,
                        x[idx_start:idx_stop, ...],
                        wait_for=x_part[i+self.num_dev].events,
                        is_blocking=False))
                if TGV:
                    z1_part[self.num_dev+i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+1], z1_part[self.num_dev+i].data,
                            z1[idx_start:idx_stop, ...],
                            wait_for=z1_part[self.num_dev+i].events,
                            is_blocking=False))
                    z2_part[self.num_dev+i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+1], z2_part[self.num_dev+i].data,
                            z2[idx_start:idx_stop, ...],
                            wait_for=z2_part[self.num_dev+i].events,
                            is_blocking=False))
            for i in range(self.num_dev):
                if TGV:
                    Kyk2_part[i+self.num_dev].add_event(
                        self.update_Kyk2(
                            Kyk2_part[i+self.num_dev], z2_part[i],
                            z1_part[self.num_dev+i], i, 1, first))
                gradx_xold_part[i+self.num_dev].add_event(
                    self.f_grad(
                        gradx_xold_part[i+self.num_dev],
                        x_part[self.num_dev+i], i, 1))
    # Collect last block
        if j < 2*self.num_dev:
            j = 2*self.num_dev
        else:
            j += 1
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+3].finish()
            if i > 1:
                self.queue[4*(i-1)+2].finish()
            idx_start = (i*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev)*self.par_slices))
            idx_stop = ((i+1)*self.par_slices +
                        (2*self.num_dev*(j-2*self.num_dev)) *
                        self.par_slices+self.overlap)
            if TGV:
                Kyk2_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], Kyk2[idx_start:idx_stop, ...],
                        Kyk2_part[i].data,
                        wait_for=Kyk2_part[i].events,
                        is_blocking=False))
            gradx_xold_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+2], gradx_xold[idx_start:idx_stop, ...],
                    gradx_xold_part[i].data,
                    wait_for=gradx_xold_part[i].events, is_blocking=False))
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+2].finish()
            if i > 1:
                self.queue[4*(i-1)+3].finish()
            idx_start = (i*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev) +
                          self.num_dev)*self.par_slices)
            idx_stop = ((i+1)*self.par_slices +
                        (2*self.num_dev*(j-2*self.num_dev) +
                         self.num_dev)*self.par_slices)
            if idx_stop == self.NSlice:
                idx_start -= self.overlap
            else:
                idx_stop += self.overlap
            if TGV:
                Kyk2_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], Kyk2[idx_start:idx_stop, ...],
                        Kyk2_part[i+self.num_dev].data,
                        wait_for=Kyk2_part[i+self.num_dev].events,
                        is_blocking=False))
            gradx_xold_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+3], gradx_xold[idx_start:idx_stop, ...],
                    gradx_xold_part[i+self.num_dev].data,
                    wait_for=gradx_xold_part[i+self.num_dev].events,
                    is_blocking=False))
        for i in range(self.num_dev):
            self.queue[4*i+2].finish()
            self.queue[4*i+3].finish()

    def stream_primal_update(self, arrays, parts, tau, delta, TGV=True):
        j = 0
        if TGV:
            (x_new, xk, x, v_new, v, Kyk1, gradx, Ax, Kyk2, symgrad_v) = arrays
            (x_new_part, xk_part, x_part, v_new_part, v_part, Kyk1_part,
             gradx_part, Ax_part, Kyk2_part, symgrad_v_part) = parts
        else:
            (x_new, xk, x, Kyk1, gradx, Ax) = arrays
            (x_new_part, xk_part, x_part, Kyk1_part, gradx_part,
             Ax_part) = parts
        for i in range(self.num_dev):
            idx_start = i*self.par_slices
            idx_stop = (i+1)*self.par_slices+self.overlap
            x_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], x_part[i].data,
                    x[idx_start:idx_stop, ...],
                    wait_for=x_part[i].events, is_blocking=False))
            Kyk1_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], Kyk1_part[i].data,
                    Kyk1[idx_start:idx_stop, ...],
                    wait_for=Kyk1_part[i].events, is_blocking=False))
            xk_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], xk_part[i].data,
                    xk[idx_start:idx_stop, ...],
                    wait_for=xk_part[i].events, is_blocking=False))
            self.coil_buf_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], self.coil_buf_part[i].data,
                    self.C[idx_start:idx_stop, ...],
                    wait_for=self.coil_buf_part[i].events, is_blocking=False))
            self.grad_buf_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], self.grad_buf_part[i].data,
                    self.grad_x[idx_start:idx_stop, ...],
                    wait_for=self.grad_buf_part[i].events, is_blocking=False))
        for i in range(self.num_dev):
            x_new_part[i].add_event(
                self.update_primal(
                    x_new_part[i], x_part[i], Kyk1_part[i], xk_part[i],
                    tau, delta, i, 0))
            gradx_part[i].add_event(
                self.f_grad(
                    gradx_part[i], x_new_part[i], i, 0))
            Ax_part[i].add_event(
                self.operator_forward(
                    Ax_part[i], x_new_part[i], i, 0))
        for i in range(self.num_dev):
            idx_start = (i+1+self.num_dev-1)*self.par_slices
            idx_stop = (i+2+self.num_dev-1)*self.par_slices
            if idx_stop == self.NSlice:
                idx_start -= self.overlap
            else:
                idx_stop += self.overlap
            x_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], x_part[i+self.num_dev].data,
                    x[idx_start:idx_stop, ...],
                    wait_for=x_part[i+self.num_dev].events, is_blocking=False))
            Kyk1_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], Kyk1_part[i+self.num_dev].data,
                    Kyk1[idx_start:idx_stop, ...],
                    wait_for=Kyk1_part[i+self.num_dev].events,
                    is_blocking=False))
            xk_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], xk_part[i+self.num_dev].data,
                    xk[idx_start:idx_stop, ...],
                    wait_for=xk_part[i+self.num_dev].events,
                    is_blocking=False))
            self.coil_buf_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], self.coil_buf_part[i+self.num_dev].data,
                    self.C[idx_start:idx_stop, ...],
                    wait_for=self.coil_buf_part[i+self.num_dev].events,
                    is_blocking=False))
            self.grad_buf_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], self.grad_buf_part[i+self.num_dev].data,
                    self.grad_x[idx_start:idx_stop, ...],
                    wait_for=self.grad_buf_part[i+self.num_dev].events,
                    is_blocking=False))
        for i in range(self.num_dev):
            x_new_part[i+self.num_dev].add_event(
                self.update_primal(
                    x_new_part[i+self.num_dev], x_part[self.num_dev+i],
                    Kyk1_part[self.num_dev+i], xk_part[self.num_dev+i],
                    tau, delta, i, 1))
            gradx_part[i+self.num_dev].add_event(
                self.f_grad(
                    gradx_part[i+self.num_dev], x_new_part[i+self.num_dev],
                    i, 1))
            Ax_part[i+self.num_dev].add_event(
                self.operator_forward(
                    Ax_part[i+self.num_dev], x_new_part[i+self.num_dev], i, 1))
    # Stream
        for j in range(2*self.num_dev,
                       int(self.NSlice/(2*self.par_slices*self.num_dev) +
                           (2*self.num_dev-1))):
            for i in range(self.num_dev):
                # Get Data
                self.queue[4*(self.num_dev-1)+3].finish()
                if i > 1:
                    self.queue[4*(i-1)+2].finish()
                idx_start = (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev) *
                              self.par_slices))
                idx_stop = ((i+1)*self.par_slices +
                            (2*self.num_dev*(j-2*self.num_dev)) *
                            self.par_slices+self.overlap)
                x_new_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], x_new[idx_start:idx_stop, ...],
                        x_new_part[i].data,
                        wait_for=x_new_part[i].events, is_blocking=False))
                gradx_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], gradx[idx_start:idx_stop, ...],
                        gradx_part[i].data,
                        wait_for=gradx_part[i].events, is_blocking=False))
                Ax_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], Ax[idx_start:idx_stop, ...],
                        Ax_part[i].data,
                        wait_for=Ax_part[i].events, is_blocking=False))
            for i in range(self.num_dev):
                # Put Data
                idx_start = (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev+1) *
                              self.par_slices))
                idx_stop = ((i+1)*self.par_slices +
                            (2*self.num_dev*(j-2*self.num_dev+1)) *
                            self.par_slices+self.overlap)
                x_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], x_part[i].data,
                        x[idx_start:idx_stop, ...],
                        wait_for=x_part[i].events, is_blocking=False))
                Kyk1_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], Kyk1_part[i].data,
                        Kyk1[idx_start:idx_stop, ...],
                        wait_for=Kyk1_part[i].events, is_blocking=False))
                xk_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], xk_part[i].data,
                        xk[idx_start:idx_stop, ...],
                        wait_for=xk_part[i].events, is_blocking=False))
                self.coil_buf_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], self.coil_buf_part[i].data,
                        self.C[idx_start:idx_stop, ...],
                        wait_for=self.coil_buf_part[i].events,
                        is_blocking=False))
                self.grad_buf_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], self.grad_buf_part[i].data,
                        self.grad_x[idx_start:idx_stop, ...],
                        wait_for=self.grad_buf_part[i].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                x_new_part[i].add_event(
                    self.update_primal(
                        x_new_part[i], x_part[i], Kyk1_part[i], xk_part[i],
                        tau, delta, i, 0))
                gradx_part[i].add_event(
                    self.f_grad(
                        gradx_part[i], x_new_part[i], i, 0))
                Ax_part[i].add_event(
                    self.operator_forward(
                        Ax_part[i], x_new_part[i], i, 0))
            for i in range(self.num_dev):
                self.queue[4*(self.num_dev-1)+2].finish()
                if i > 1:
                    self.queue[4*(i-1)+3].finish()
                idx_start = (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                             self.par_slices)
                idx_stop = ((i+1)*self.par_slices +
                            (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                            self.par_slices+self.overlap)
                x_new_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], x_new[idx_start:idx_stop, ...],
                        x_new_part[i+self.num_dev].data,
                        wait_for=x_new_part[i+self.num_dev].events,
                        is_blocking=False))
                gradx_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], gradx[idx_start:idx_stop, ...],
                        gradx_part[i+self.num_dev].data,
                        wait_for=gradx_part[i+self.num_dev].events,
                        is_blocking=False))
                Ax_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], Ax[idx_start:idx_stop, ...],
                        Ax_part[i+self.num_dev].data,
                        wait_for=Ax_part[i+self.num_dev].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                idx_start = (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev+1) +
                              self.num_dev)*self.par_slices)
                idx_stop = ((i+1)*self.par_slices +
                            (2*self.num_dev*(j-2*self.num_dev+1) +
                             self.num_dev)*self.par_slices)
                if idx_stop == self.NSlice:
                    idx_start -= self.overlap
                else:
                    idx_stop += self.overlap
                x_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], x_part[i+self.num_dev].data,
                        x[idx_start:idx_stop, ...],
                        wait_for=x_part[i+self.num_dev].events,
                        is_blocking=False))
                Kyk1_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], Kyk1_part[i+self.num_dev].data,
                        Kyk1[idx_start:idx_stop, ...],
                        wait_for=Kyk1_part[i+self.num_dev].events,
                        is_blocking=False))
                xk_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], xk_part[i+self.num_dev].data,
                        xk[idx_start:idx_stop, ...],
                        wait_for=xk_part[i+self.num_dev].events,
                        is_blocking=False))
                self.coil_buf_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1],
                        self.coil_buf_part[i+self.num_dev].data,
                        self.C[idx_start:idx_stop, ...],
                        wait_for=self.coil_buf_part[i+self.num_dev].events,
                        is_blocking=False))
                self.grad_buf_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1],
                        self.grad_buf_part[i+self.num_dev].data,
                        self.grad_x[idx_start:idx_stop, ...],
                        wait_for=self.grad_buf_part[i+self.num_dev].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                x_new_part[i+self.num_dev].add_event(
                    self.update_primal(
                        x_new_part[i+self.num_dev], x_part[self.num_dev+i],
                        Kyk1_part[self.num_dev+i], xk_part[self.num_dev+i],
                        tau, delta, i, 1))
                gradx_part[i+self.num_dev].add_event(
                    self.f_grad(
                        gradx_part[i+self.num_dev], x_new_part[i+self.num_dev],
                        i, 1))
                Ax_part[i+self.num_dev].add_event(
                    self.operator_forward(
                        Ax_part[i+self.num_dev], x_new_part[i+self.num_dev],
                        i, 1))
    # Collect last block
        if j < 2*self.num_dev:
            j = 2*self.num_dev
        else:
            j += 1
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+3].finish()
            if i > 1:
                self.queue[4*(i-1)+2].finish()
            idx_start = (i*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev)*self.par_slices))
            idx_stop = ((i+1)*self.par_slices +
                        (2*self.num_dev*(j-2*self.num_dev)) *
                        self.par_slices+self.overlap)
            x_new_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+2], x_new[idx_start:idx_stop, ...],
                    x_new_part[i].data,
                    wait_for=x_new_part[i].events, is_blocking=False))
            gradx_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+2], gradx[idx_start:idx_stop, ...],
                    gradx_part[i].data,
                    wait_for=gradx_part[i].events, is_blocking=False))
            Ax_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+2], Ax[idx_start:idx_stop, ...],
                    Ax_part[i].data,
                    wait_for=Ax_part[i].events, is_blocking=False))
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+2].finish()
            if i > 1:
                self.queue[4*(i-1)+3].finish()
            idx_start = (i*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev) +
                          self.num_dev)*self.par_slices)
            idx_stop = ((i+1)*self.par_slices +
                        (2*self.num_dev*(j-2*self.num_dev) +
                         self.num_dev)*self.par_slices)
            if idx_stop == self.NSlice:
                idx_start -= self.overlap
            else:
                idx_stop += self.overlap
            x_new_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+3], x_new[idx_start:idx_stop, ...],
                    x_new_part[i+self.num_dev].data,
                    wait_for=x_new_part[i+self.num_dev].events,
                    is_blocking=False))
            gradx_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+3], gradx[idx_start:idx_stop, ...],
                    gradx_part[i+self.num_dev].data,
                    wait_for=gradx_part[i+self.num_dev].events,
                    is_blocking=False))
            Ax_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+3], Ax[idx_start:idx_stop, ...],
                    Ax_part[i+self.num_dev].data,
                    wait_for=Ax_part[i+self.num_dev].events,
                    is_blocking=False))
        if TGV:
            j = 0
            for i in range(self.num_dev):
                idx_start = (self.NSlice)-((i+1)*self.par_slices)-self.overlap
                idx_stop = (self.NSlice)-(i*self.par_slices)
                v_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], v_part[i].data,
                        v[idx_start:idx_stop, ...],
                        wait_for=v_part[i].events, is_blocking=False))
                Kyk2_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], Kyk2_part[i].data,
                        Kyk2[idx_start:idx_stop, ...],
                        wait_for=Kyk2_part[i].events, is_blocking=False))
            for i in range(self.num_dev):
                v_new_part[i].add_event(
                    self.update_v(
                        v_new_part[i], v_part[i], Kyk2_part[i], tau, i, 0))
                symgrad_v_part[i].add_event(
                    self.sym_grad(
                        symgrad_v_part[i], v_new_part[i], i, 0))
            for i in range(self.num_dev):
                idx_start = (self.NSlice) -\
                            ((i+2+self.num_dev-1)*self.par_slices)
                idx_stop = (self.NSlice)-((i+1+self.num_dev-1)*self.par_slices)
                if idx_start == 0:
                    idx_stop += self.overlap
                else:
                    idx_start -= self.overlap
                v_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], v_part[i+self.num_dev].data,
                        v[idx_start:idx_stop, ...],
                        wait_for=v_part[i+self.num_dev].events,
                        is_blocking=False))
                Kyk2_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], Kyk2_part[i+self.num_dev].data,
                        Kyk2[idx_start:idx_stop, ...],
                        wait_for=Kyk2_part[i+self.num_dev].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                v_new_part[i+self.num_dev].add_event(
                    self.update_v(
                        v_new_part[i+self.num_dev],
                        v_part[self.num_dev+i],
                        Kyk2_part[self.num_dev+i], tau, i, 1))
                symgrad_v_part[i+self.num_dev].add_event(
                    self.sym_grad(
                        symgrad_v_part[i+self.num_dev],
                        v_new_part[i+self.num_dev], i, 1))
        # Stream
            for j in range(2*self.num_dev,
                           int(self.NSlice/(2*self.par_slices*self.num_dev) +
                               (2*self.num_dev-1))):
                for i in range(self.num_dev):
                    # Get Data
                    self.queue[4*(self.num_dev-1)+3].finish()
                    if i > 1:
                        self.queue[4*(i-1)+2].finish()
                    idx_start = (
                        self.NSlice -
                        ((i+1)*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev))*self.par_slices) -
                        self.overlap)
                    idx_stop = (
                        self.NSlice -
                        (i*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev)*self.par_slices)))
                    v_new_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+2], v_new[idx_start:idx_stop, ...],
                            v_new_part[i].data,
                            wait_for=v_new_part[i].events, is_blocking=False))
                    symgrad_v_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+2],
                            symgrad_v[idx_start:idx_stop, ...],
                            symgrad_v_part[i].data,
                            wait_for=symgrad_v_part[i].events,
                            is_blocking=False))
                for i in range(self.num_dev):
                    # Put Data
                    idx_start = (
                        self.NSlice -
                        ((i+1)*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev+1)) *
                         self.par_slices) -
                        self.overlap)
                    idx_stop = (
                        self.NSlice -
                        (i*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev+1) *
                          self.par_slices)))
                    v_part[i].add_event(cl.enqueue_copy(
                        self.queue[4*i], v_part[i].data,
                        v[idx_start:idx_stop, ...],
                        wait_for=v_part[i].events,
                        is_blocking=False))
                for i in range(self.num_dev):
                    Kyk2_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i], Kyk2_part[i].data,
                            Kyk2[idx_start:idx_stop, ...],
                            wait_for=Kyk2_part[i].events, is_blocking=False))
                    v_new_part[i].add_event(
                        self.update_v(
                            v_new_part[i], v_part[i], Kyk2_part[i],
                            tau, i, 0))
                    symgrad_v_part[i].add_event(
                        self.sym_grad(
                            symgrad_v_part[i], v_new_part[i], i, 0))
                for i in range(self.num_dev):
                    self.queue[4*(self.num_dev-1)+2].finish()
                    if i > 1:
                        self.queue[4*(i-1)+3].finish()
                    idx_start = (
                        self.NSlice -
                        ((i+1)*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                         self.par_slices)-self.overlap)
                    idx_stop = (
                        self.NSlice -
                        (i*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                         self.par_slices))
                    v_new_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+3], v_new[idx_start:idx_stop, ...],
                            v_new_part[i+self.num_dev].data,
                            wait_for=v_new_part[i+self.num_dev].events,
                            is_blocking=False))
                    symgrad_v_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+3],
                            symgrad_v[idx_start:idx_stop, ...],
                            symgrad_v_part[i+self.num_dev].data,
                            wait_for=symgrad_v_part[i+self.num_dev].events,
                            is_blocking=False))
                for i in range(self.num_dev):
                    idx_start = (self.NSlice -
                                 ((i+1)*self.par_slices +
                                  (2*self.num_dev*(j-2*self.num_dev+1) +
                                   self.num_dev)*self.par_slices))
                    idx_stop = (self.NSlice -
                                (i*self.par_slices +
                                 (2*self.num_dev*(j-2*self.num_dev+1) +
                                  self.num_dev)*self.par_slices))
                    if idx_start == 0:
                        idx_stop += self.overlap
                    else:
                        idx_start -= self.overlap
                    v_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+1], v_part[i+self.num_dev].data,
                            v[idx_start:idx_stop, ...],
                            wait_for=v_part[i+self.num_dev].events,
                            is_blocking=False))
                    Kyk2_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+1], Kyk2_part[i+self.num_dev].data,
                            Kyk2[idx_start:idx_stop, ...],
                            wait_for=Kyk2_part[i+self.num_dev].events,
                            is_blocking=False))
                for i in range(self.num_dev):
                    v_new_part[i+self.num_dev].add_event(
                        self.update_v(
                            v_new_part[i+self.num_dev], v_part[self.num_dev+i],
                            Kyk2_part[self.num_dev+i], tau, i, 1))
                    symgrad_v_part[i+self.num_dev].add_event(
                        self.sym_grad(
                            symgrad_v_part[i+self.num_dev],
                            v_new_part[i+self.num_dev], i, 1))
        # Collect last block
            if j < 2*self.num_dev:
                j = 2*self.num_dev
            else:
                j += 1
            for i in range(self.num_dev):
                self.queue[4*(self.num_dev-1)+3].finish()
                if i > 1:
                    self.queue[4*(i-1)+2].finish()
                idx_start = (
                    self.NSlice -
                    ((i+1)*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev)) *
                     self.par_slices)-self.overlap)
                idx_stop = (
                    self.NSlice -
                    (i*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev)*self.par_slices)))
                v_new_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], v_new[idx_start:idx_stop, ...],
                        v_new_part[i].data,
                        wait_for=v_new_part[i].events,
                        is_blocking=False))
                symgrad_v_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], symgrad_v[idx_start:idx_stop, ...],
                        symgrad_v_part[i].data,
                        wait_for=symgrad_v_part[i].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                self.queue[4*(self.num_dev-1)+2].finish()
                if i > 1:
                    self.queue[4*(i-1)+3].finish()
                idx_start = (
                    self.NSlice -
                    ((i+1)*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                     self.par_slices))
                idx_stop = (
                    self.NSlice -
                    (i*self.par_slices +
                     (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                     self.par_slices))
                if idx_start == 0:
                    idx_stop += self.overlap
                else:
                    idx_start -= self.overlap
                v_new_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], v_new[idx_start:idx_stop, ...],
                        v_new_part[i+self.num_dev].data,
                        wait_for=v_new_part[i+self.num_dev].events,
                        is_blocking=False))
                symgrad_v_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], symgrad_v[idx_start:idx_stop, ...],
                        symgrad_v_part[i+self.num_dev].data,
                        wait_for=symgrad_v_part[i+self.num_dev].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                self.queue[4*i+2].finish()
                self.queue[4*i+3].finish()

    def stream_dual_update(self, arrays, parts, pars, TGV=True):
        if TGV:
            (ynorm, lhs, beta_line, tau_new, omega,
             alpha, beta, theta_line) = pars
            (z1, gradx, gradx_xold, v_new, v, r, Ax, Axold, res, Kyk1,
             z1_new, r_new, Kyk1_new, z2, symgrad_v, symgrad_v_vold, Kyk2,
             z2_new, Kyk2_new) = arrays
            (z1_part, z1_new_part, z2_part, z2_new_part, gradx_part,
             gradx_xold_part, v_new_part, v_part,
             r_part, r_new_part, Ax_part, Axold_part, res_part, Kyk1_part,
             Kyk1_new_part, Kyk2_part, Kyk2_new_part,
             symgrad_v_part, symgrad_v_vold_part) = parts
        else:
            (ynorm, lhs, beta_line, tau_new, omega, alpha, theta_line) = pars
            (z1, gradx, gradx_xold, r, Ax, Axold, res, Kyk1,
             z1_new, r_new, Kyk1_new) = arrays
            (z1_part, z1_new_part, gradx_part, gradx_xold_part,
             r_part, r_new_part, Ax_part, Axold_part, res_part, Kyk1_part,
             Kyk1_new_part) = parts

        j = 0
        last = 0
        for i in range(self.num_dev):
            idx_start = (self.NSlice-((i+1)*self.par_slices)-self.overlap)
            idx_stop = (self.NSlice-(i*self.par_slices))
            z1_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], z1_part[i].data,
                    z1[idx_start:idx_stop, ...],
                    wait_for=z1_part[i].events, is_blocking=False))
            gradx_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], gradx_part[i].data,
                    gradx[idx_start:idx_stop, ...],
                    wait_for=gradx_part[i].events, is_blocking=False))
            gradx_xold_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], gradx_xold_part[i].data,
                    gradx_xold[idx_start:idx_stop, ...],
                    wait_for=gradx_xold_part[i].events, is_blocking=False))
            r_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], r_part[i].data,
                    r[idx_start:idx_stop, ...],
                    wait_for=r_part[i].events, is_blocking=False))
            Ax_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], Ax_part[i].data,
                    Ax[idx_start:idx_stop, ...],
                    wait_for=Ax_part[i].events, is_blocking=False))
            Axold_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], Axold_part[i].data,
                    Axold[idx_start:idx_stop, ...],
                    wait_for=Axold_part[i].events, is_blocking=False))
            res_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], res_part[i].data,
                    res[idx_start:idx_stop, ...],
                    wait_for=res_part[i].events, is_blocking=False))
            Kyk1_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], Kyk1_part[i].data,
                    Kyk1[idx_start:idx_stop, ...],
                    wait_for=Kyk1_part[i].events, is_blocking=False))
            self.coil_buf_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], self.coil_buf_part[i].data,
                    self.C[idx_start:idx_stop, ...],
                    wait_for=self.coil_buf_part[i].events, is_blocking=False))
            self.grad_buf_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i], self.grad_buf_part[i].data,
                    self.grad_x[idx_start:idx_stop, ...],
                    wait_for=self.grad_buf_part[i].events, is_blocking=False))
            if TGV:
                v_new_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], v_new_part[i].data,
                        v_new[idx_start:idx_stop, ...],
                        wait_for=v_new_part[i].events, is_blocking=False))
                v_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], v_part[i].data,
                        v[idx_start:idx_stop, ...],
                        wait_for=v_part[i].events, is_blocking=False))
        for i in range(self.num_dev):
            if i == 0:
                last = 1
            else:
                last = 0
            if TGV:
                z1_new_part[i].add_event(
                    self.update_z1(
                        z1_new_part[i], z1_part[i], gradx_part[i],
                        gradx_xold_part[i], v_new_part[i], v_part[i],
                        beta_line*tau_new, theta_line, alpha, omega, i, 0))
            else:
                z1_new_part[i].add_event(
                    self.update_z1_tv(
                        z1_new_part[i], z1_part[i], gradx_part[i],
                        gradx_xold_part[i],
                        beta_line*tau_new, theta_line, alpha, omega, i, 0))
            r_new_part[i].add_event(
                self.update_r(
                    r_new_part[i], r_part[i], Ax_part[i], Axold_part[i],
                    res_part[i], beta_line*tau_new, theta_line,
                    self.irgn_par["lambd"], i, 0))
            Kyk1_new_part[i].add_event(
                self.operator_adjoint(
                    Kyk1_new_part[i], r_new_part[i], z1_new_part[i],
                    i, 0, last))
        last = 0
        for i in range(self.num_dev):
            idx_start = (self.NSlice - ((i+2+self.num_dev-1)*self.par_slices))
            idx_stop = (self.NSlice - ((i+1+self.num_dev-1)*self.par_slices))
            if idx_start == 0:
                idx_stop += self.overlap
            else:
                idx_start -= self.overlap
            z1_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], z1_part[i+self.num_dev].data,
                    z1[idx_start:idx_stop, ...],
                    wait_for=z1_part[i+self.num_dev].events,
                    is_blocking=False))
            gradx_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], gradx_part[i+self.num_dev].data,
                    gradx[idx_start:idx_stop, ...],
                    wait_for=gradx_part[i+self.num_dev].events,
                    is_blocking=False))
            gradx_xold_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], gradx_xold_part[i+self.num_dev].data,
                    gradx_xold[idx_start:idx_stop, ...],
                    wait_for=gradx_xold_part[i+self.num_dev].events,
                    is_blocking=False))
            r_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], r_part[i+self.num_dev].data,
                    r[idx_start:idx_stop, ...],
                    wait_for=r_part[i+self.num_dev].events, is_blocking=False))
            Ax_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], Ax_part[i+self.num_dev].data,
                    Ax[idx_start:idx_stop, ...],
                    wait_for=Ax_part[i+self.num_dev].events,
                    is_blocking=False))
            Axold_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], Axold_part[i+self.num_dev].data,
                    Axold[idx_start:idx_stop, ...],
                    wait_for=Axold_part[i+self.num_dev].events,
                    is_blocking=False))
            res_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], res_part[i+self.num_dev].data,
                    res[idx_start:idx_stop, ...],
                    wait_for=res_part[i+self.num_dev].events,
                    is_blocking=False))
            Kyk1_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], Kyk1_part[i+self.num_dev].data,
                    Kyk1[idx_start:idx_stop, ...],
                    wait_for=Kyk1_part[i+self.num_dev].events,
                    is_blocking=False))
            self.coil_buf_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], self.coil_buf_part[i+self.num_dev].data,
                    self.C[idx_start:idx_stop, ...],
                    wait_for=self.coil_buf_part[i+self.num_dev].events,
                    is_blocking=False))
            self.grad_buf_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+1], self.grad_buf_part[i+self.num_dev].data,
                    self.grad_x[idx_start:idx_stop, ...],
                    wait_for=self.grad_buf_part[i+self.num_dev].events,
                    is_blocking=False))
            if TGV:
                v_new_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], v_new_part[i+self.num_dev].data,
                        v_new[idx_start:idx_stop, ...],
                        wait_for=v_new_part[i+self.num_dev].events,
                        is_blocking=False))
                v_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], v_part[i+self.num_dev].data,
                        v[idx_start:idx_stop, ...],
                        wait_for=v_part[i+self.num_dev].events,
                        is_blocking=False))
        for i in range(self.num_dev):
            if TGV:
                z1_new_part[i+self.num_dev].add_event(
                    self.update_z1(
                        z1_new_part[i+self.num_dev], z1_part[self.num_dev+i],
                        gradx_part[self.num_dev+i],
                        gradx_xold_part[self.num_dev+i],
                        v_new_part[self.num_dev+i], v_part[self.num_dev+i],
                        beta_line*tau_new, theta_line, alpha, omega, i, 1))
            else:
                z1_new_part[i+self.num_dev].add_event(
                    self.update_z1_tv(
                        z1_new_part[i+self.num_dev], z1_part[self.num_dev+i],
                        gradx_part[self.num_dev+i],
                        gradx_xold_part[self.num_dev+i],
                        beta_line*tau_new, theta_line, alpha, omega, i, 1))
            r_new_part[i+self.num_dev].add_event(
                self.update_r(
                    r_new_part[i+self.num_dev], r_part[self.num_dev+i],
                    Ax_part[self.num_dev+i], Axold_part[self.num_dev+i],
                    res_part[self.num_dev+i], beta_line*tau_new,
                    theta_line, self.irgn_par["lambd"], i, 1))
            Kyk1_new_part[i+self.num_dev].add_event(
                self.operator_adjoint(
                    Kyk1_new_part[i+self.num_dev], r_new_part[i+self.num_dev],
                    z1_new_part[i+self.num_dev], i, 1, last))
        # Stream
        for j in range(2*self.num_dev,
                       int(self.NSlice/(2*self.par_slices*self.num_dev) +
                           (2*self.num_dev-1))):
            for i in range(self.num_dev):
                # Get Data
                self.queue[4*(self.num_dev-1)+3].finish()
                if i > 1:
                    self.queue[4*(i-1)+2].finish()
                idx_start = (self.NSlice -
                             ((i+1)*self.par_slices +
                              (2*self.num_dev*(j-2*self.num_dev)) *
                              self.par_slices)-self.overlap)
                idx_stop = (self.NSlice -
                            (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev) *
                              self.par_slices)))
                z1_new_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], z1_new[idx_start:idx_stop, ...],
                        z1_new_part[i].data,
                        wait_for=z1_new_part[i].events, is_blocking=False))
                r_new_part[i].add_event(
                    cl.enqueue_copy
                    (self.queue[4*i+2], r_new[idx_start:idx_stop, ...],
                     r_new_part[i].data,
                     wait_for=r_new_part[i].events, is_blocking=False))
                Kyk1_new_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], Kyk1_new[idx_start:idx_stop, ...],
                        Kyk1_new_part[i].data,
                        wait_for=Kyk1_new_part[i].events,
                        is_blocking=False))
                ynorm += ((
                    clarray.vdot(r_new_part[i][self.overlap:, ...] -
                                 r_part[i][self.overlap:, ...],
                                 r_new_part[i][self.overlap:, ...] -
                                 r_part[i][self.overlap:, ...],
                                 queue=self.queue[4*i]) +
                    clarray.vdot(z1_new_part[i][self.overlap:, ...] -
                                 z1_part[i][self.overlap:, ...],
                                 z1_new_part[i][self.overlap:, ...] -
                                 z1_part[i][self.overlap:, ...],
                                 queue=self.queue[4*i]))).get()
                lhs += ((
                    clarray.vdot(Kyk1_new_part[i][self.overlap:, ...] -
                                 Kyk1_part[i][self.overlap:, ...],
                                 Kyk1_new_part[i][self.overlap:, ...] -
                                 Kyk1_part[i][self.overlap:, ...],
                                 queue=self.queue[4*i]))).get()
            for i in range(self.num_dev):
                # Put Data
                idx_start = (self.NSlice -
                             ((i+1)*self.par_slices +
                              (2*self.num_dev*(j-2*self.num_dev+1)) *
                              self.par_slices)-self.overlap)
                idx_stop = (self.NSlice -
                            (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev+1) *
                              self.par_slices)))
                z1_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], z1_part[i].data,
                        z1[idx_start:idx_stop, ...],
                        wait_for=z1_part[i].events, is_blocking=False))
                gradx_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], gradx_part[i].data,
                        gradx[idx_start:idx_stop, ...],
                        wait_for=gradx_part[i].events, is_blocking=False))
                gradx_xold_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], gradx_xold_part[i].data,
                        gradx_xold[idx_start:idx_stop, ...],
                        wait_for=gradx_xold_part[i].events, is_blocking=False))
                r_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], r_part[i].data,
                        r[idx_start:idx_stop, ...],
                        wait_for=r_part[i].events, is_blocking=False))
                Ax_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], Ax_part[i].data,
                        Ax[idx_start:idx_stop, ...],
                        wait_for=Ax_part[i].events, is_blocking=False))
                Axold_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], Axold_part[i].data,
                        Axold[idx_start:idx_stop, ...],
                        wait_for=Axold_part[i].events, is_blocking=False))
                res_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], res_part[i].data,
                        res[idx_start:idx_stop, ...],
                        wait_for=res_part[i].events, is_blocking=False))
                Kyk1_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], Kyk1_part[i].data,
                        Kyk1[idx_start:idx_stop, ...],
                        wait_for=Kyk1_part[i].events, is_blocking=False))
                self.coil_buf_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], self.coil_buf_part[i].data,
                        self.C[idx_start:idx_stop, ...],
                        wait_for=self.coil_buf_part[i].events,
                        is_blocking=False))
                self.grad_buf_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], self.grad_buf_part[i].data,
                        self.grad_x[idx_start:idx_stop, ...],
                        wait_for=self.grad_buf_part[i].events,
                        is_blocking=False))
                if TGV:
                    v_new_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i], v_new_part[i].data,
                            v_new[idx_start:idx_stop, ...],
                            wait_for=v_new_part[i].events, is_blocking=False))
                    v_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i], v_part[i].data,
                            v[idx_start:idx_stop, ...],
                            wait_for=v_part[i].events, is_blocking=False))
            for i in range(self.num_dev):
                if TGV:
                    z1_new_part[i].add_event(
                        self.update_z1(
                            z1_new_part[i], z1_part[i], gradx_part[i],
                            gradx_xold_part[i], v_new_part[i], v_part[i],
                            beta_line*tau_new, theta_line, alpha, omega, i, 0))
                else:
                    z1_new_part[i].add_event(
                        self.update_z1_tv(
                            z1_new_part[i], z1_part[i], gradx_part[i],
                            gradx_xold_part[i],
                            beta_line*tau_new, theta_line, alpha, omega, i, 0))
                r_new_part[i].add_event(
                    self.update_r(
                        r_new_part[i], r_part[i], Ax_part[i], Axold_part[i],
                        res_part[i], beta_line*tau_new, theta_line,
                        self.irgn_par["lambd"], i, 0))
                Kyk1_new_part[i].add_event(
                    self.operator_adjoint(
                        Kyk1_new_part[i], r_new_part[i], z1_new_part[i],
                        i, 0, last))
            for i in range(self.num_dev):
                # Get Data
                self.queue[4*(self.num_dev-1)+2].finish()
                if i > 1:
                    self.queue[4*(i-1)+3].finish()
                idx_start = (self.NSlice -
                             ((i+1)*self.par_slices +
                              (2*self.num_dev*(j-2*self.num_dev) +
                               self.num_dev)*self.par_slices)-self.overlap)
                idx_stop = (self.NSlice -
                            (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev)+self.num_dev) *
                             self.par_slices))
                z1_new_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], z1_new[idx_start:idx_stop, ...],
                        z1_new_part[i+self.num_dev].data,
                        wait_for=z1_new_part[i+self.num_dev].events,
                        is_blocking=False))
                r_new_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], r_new[idx_start:idx_stop, ...],
                        r_new_part[i+self.num_dev].data,
                        wait_for=r_new_part[i+self.num_dev].events,
                        is_blocking=False))
                Kyk1_new_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], Kyk1_new[idx_start:idx_stop, ...],
                        Kyk1_new_part[i+self.num_dev].data,
                        wait_for=Kyk1_new_part[i+self.num_dev].events,
                        is_blocking=False))
                ynorm += ((
                    clarray.vdot(
                        r_new_part[i+self.num_dev][self.overlap:, ...] -
                        r_part[i+self.num_dev][self.overlap:, ...],
                        r_new_part[i+self.num_dev][self.overlap:, ...] -
                        r_part[i+self.num_dev][self.overlap:, ...],
                        queue=self.queue[4*i+1]) +
                    clarray.vdot(
                        z1_new_part[i+self.num_dev][self.overlap:, ...] -
                        z1_part[i+self.num_dev][self.overlap:, ...],
                        z1_new_part[i+self.num_dev][self.overlap:, ...] -
                        z1_part[i+self.num_dev][self.overlap:, ...],
                        queue=self.queue[4*i+1]))).get()
                lhs += ((
                    clarray.vdot(
                        Kyk1_new_part[i+self.num_dev][self.overlap:, ...] -
                        Kyk1_part[i+self.num_dev][self.overlap:, ...],
                        Kyk1_new_part[i+self.num_dev][self.overlap:, ...] -
                        Kyk1_part[i+self.num_dev][self.overlap:, ...],
                        queue=self.queue[4*i+1]))).get()
            for i in range(self.num_dev):
                # Put Data
                idx_start = (self.NSlice -
                             ((i+1)*self.par_slices +
                              (2*self.num_dev*(j-2*self.num_dev+1) +
                               self.num_dev)*self.par_slices))
                idx_stop = (self.NSlice -
                            (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev+1) +
                              self.num_dev)*self.par_slices))
                if idx_start == 0:
                    idx_stop += self.overlap
                else:
                    idx_start -= self.overlap
                z1_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], z1_part[i+self.num_dev].data,
                        z1[idx_start:idx_stop, ...],
                        wait_for=z1_part[i+self.num_dev].events,
                        is_blocking=False))
                gradx_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], gradx_part[i+self.num_dev].data,
                        gradx[idx_start:idx_stop, ...],
                        wait_for=gradx_part[i+self.num_dev].events,
                        is_blocking=False))
                gradx_xold_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1],
                        gradx_xold_part[i+self.num_dev].data,
                        gradx_xold[idx_start:idx_stop, ...],
                        wait_for=gradx_xold_part[i+self.num_dev].events,
                        is_blocking=False))
                r_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], r_part[i+self.num_dev].data,
                        r[idx_start:idx_stop, ...],
                        wait_for=r_part[i+self.num_dev].events,
                        is_blocking=False))
                Ax_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], Ax_part[i+self.num_dev].data,
                        Ax[idx_start:idx_stop, ...],
                        wait_for=Ax_part[i+self.num_dev].events,
                        is_blocking=False))
                Axold_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], Axold_part[i+self.num_dev].data,
                        Axold[idx_start:idx_stop, ...],
                        wait_for=Axold_part[i+self.num_dev].events,
                        is_blocking=False))
                res_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], res_part[i+self.num_dev].data,
                        res[idx_start:idx_stop, ...],
                        wait_for=res_part[i+self.num_dev].events,
                        is_blocking=False))
                Kyk1_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], Kyk1_part[i+self.num_dev].data,
                        Kyk1[idx_start:idx_stop, ...],
                        wait_for=Kyk1_part[i+self.num_dev].events,
                        is_blocking=False))
                self.coil_buf_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1],
                        self.coil_buf_part[i+self.num_dev].data,
                        self.C[idx_start:idx_stop, ...],
                        wait_for=self.coil_buf_part[i+self.num_dev].events,
                        is_blocking=False))
                self.grad_buf_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1],
                        self.grad_buf_part[i+self.num_dev].data,
                        self.grad_x[idx_start:idx_stop, ...],
                        wait_for=self.grad_buf_part[i+self.num_dev].events,
                        is_blocking=False))
                if TGV:
                    v_new_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+1], v_new_part[i+self.num_dev].data,
                            v_new[idx_start:idx_stop, ...],
                            wait_for=v_new_part[i+self.num_dev].events,
                            is_blocking=False))
                    v_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+1], v_part[i+self.num_dev].data,
                            v[idx_start:idx_stop, ...],
                            wait_for=v_part[i+self.num_dev].events,
                            is_blocking=False))
            for i in range(self.num_dev):
                if TGV:
                    z1_new_part[i+self.num_dev].add_event(
                        self.update_z1(
                            z1_new_part[i+self.num_dev],
                            z1_part[self.num_dev+i],
                            gradx_part[self.num_dev+i],
                            gradx_xold_part[self.num_dev+i],
                            v_new_part[self.num_dev+i], v_part[self.num_dev+i],
                            beta_line*tau_new, theta_line, alpha, omega, i, 1))
                else:
                    z1_new_part[i+self.num_dev].add_event(
                        self.update_z1_tv(
                            z1_new_part[i+self.num_dev],
                            z1_part[self.num_dev+i],
                            gradx_part[self.num_dev+i],
                            gradx_xold_part[self.num_dev+i],
                            beta_line*tau_new, theta_line, alpha, omega, i, 1))
                r_new_part[i+self.num_dev].add_event(
                    self.update_r(
                        r_new_part[i+self.num_dev], r_part[self.num_dev+i],
                        Ax_part[self.num_dev+i], Axold_part[self.num_dev+i],
                        res_part[self.num_dev+i], beta_line*tau_new,
                        theta_line, self.irgn_par["lambd"], i, 1))
                Kyk1_new_part[i+self.num_dev].add_event(
                    self.operator_adjoint(
                        Kyk1_new_part[i+self.num_dev],
                        r_new_part[i+self.num_dev],
                        z1_new_part[i+self.num_dev], i, 1, last))
        # Collect last block
        if j < 2*self.num_dev:
            j = 2*self.num_dev
        else:
            j += 1
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+3].finish()
            if i > 1:
                self.queue[4*(i-1)+2].finish()
            idx_start = (self.NSlice -
                         ((i+1)*self.par_slices +
                          (2*self.num_dev*(j-2*self.num_dev)) *
                          self.par_slices)-self.overlap)
            idx_stop = (self.NSlice -
                        (i*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev)*self.par_slices)))
            z1_new_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+2], z1_new[idx_start:idx_stop, ...],
                    z1_new_part[i].data,
                    wait_for=z1_new_part[i].events, is_blocking=False))
            r_new_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+2], r_new[idx_start:idx_stop, ...],
                    r_new_part[i].data,
                    wait_for=r_new_part[i].events, is_blocking=False))
            Kyk1_new_part[i].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+2], Kyk1_new[idx_start:idx_stop, ...],
                    Kyk1_new_part[i].data,
                    wait_for=Kyk1_new_part[i].events, is_blocking=False))
            ynorm += ((
                clarray.vdot(r_new_part[i][self.overlap:, ...] -
                             r_part[i][self.overlap:, ...],
                             r_new_part[i][self.overlap:, ...] -
                             r_part[i][self.overlap:, ...],
                             queue=self.queue[4*i]) +
                clarray.vdot(z1_new_part[i][self.overlap:, ...] -
                             z1_part[i][self.overlap:, ...],
                             z1_new_part[i][self.overlap:, ...] -
                             z1_part[i][self.overlap:, ...],
                             queue=self.queue[4*i]))).get()
            lhs += ((
                clarray.vdot(Kyk1_new_part[i][self.overlap:, ...] -
                             Kyk1_part[i][self.overlap:, ...],
                             Kyk1_new_part[i][self.overlap:, ...] -
                             Kyk1_part[i][self.overlap:, ...],
                             queue=self.queue[4*i]))).get()
        for i in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+2].finish()
            if i > 1:
                self.queue[4*(i-1)+3].finish()
            idx_start = (self.NSlice -
                         ((i+1)*self.par_slices +
                          (2*self.num_dev*(j-2*self.num_dev) +
                           self.num_dev)*self.par_slices))
            idx_stop = (self.NSlice -
                        (i*self.par_slices +
                         (2*self.num_dev*(j-2*self.num_dev) +
                          self.num_dev)*self.par_slices))
            if idx_start == 0:
                idx_stop += self.overlap
                ynorm += ((
                    clarray.vdot(
                        r_new_part[i][:self.par_slices, ...] -
                        r_part[i][:self.par_slices, ...],
                        r_new_part[i][:self.par_slices, ...] -
                        r_part[i][:self.par_slices, ...],
                        queue=self.queue[4*i]) +
                    clarray.vdot(
                        z1_new_part[i][:self.par_slices, ...] -
                        z1_part[i][:self.par_slices, ...],
                        z1_new_part[i][:self.par_slices, ...] -
                        z1_part[i][:self.par_slices, ...],
                        queue=self.queue[4*i]))).get()
                lhs += ((
                    clarray.vdot(
                        Kyk1_new_part[i][:self.par_slices, ...] -
                        Kyk1_part[i][:self.par_slices, ...],
                        Kyk1_new_part[i][:self.par_slices, ...] -
                        Kyk1_part[i][:self.par_slices, ...],
                        queue=self.queue[4*i]))).get()
            else:
                idx_start -= self.overlap
                ynorm += ((
                    clarray.vdot(
                        r_new_part[i+self.num_dev][self.overlap:, ...] -
                        r_part[i+self.num_dev][self.overlap:, ...],
                        r_new_part[i+self.num_dev][self.overlap:, ...] -
                        r_part[i+self.num_dev][self.overlap:, ...],
                        queue=self.queue[4*i+1]) +
                    clarray.vdot(
                        z1_new_part[i+self.num_dev][self.overlap:, ...] -
                        z1_part[i+self.num_dev][self.overlap:, ...],
                        z1_new_part[i+self.num_dev][self.overlap:, ...] -
                        z1_part[i+self.num_dev][self.overlap:, ...],
                        queue=self.queue[4*i+1]))).get()
                lhs += ((
                    clarray.vdot(
                        Kyk1_new_part[i+self.num_dev][self.overlap:, ...] -
                        Kyk1_part[i+self.num_dev][self.overlap:, ...],
                        Kyk1_new_part[i+self.num_dev][self.overlap:, ...] -
                        Kyk1_part[i+self.num_dev][self.overlap:, ...],
                        queue=self.queue[4*i+1]))).get()
            z1_new_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+3], z1_new[idx_start:idx_stop, ...],
                    z1_new_part[i+self.num_dev].data,
                    wait_for=z1_new_part[i+self.num_dev].events,
                    is_blocking=False))
            r_new_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+3], r_new[idx_start:idx_stop, ...],
                    r_new_part[i+self.num_dev].data,
                    wait_for=r_new_part[i+self.num_dev].events,
                    is_blocking=False))
            Kyk1_new_part[i+self.num_dev].add_event(
                cl.enqueue_copy(
                    self.queue[4*i+3], Kyk1_new[idx_start:idx_stop, ...],
                    Kyk1_new_part[i+self.num_dev].data,
                    wait_for=Kyk1_new_part[i+self.num_dev].events,
                    is_blocking=False))

        if TGV:
            j = 0
            first = 0
            for i in range(self.num_dev):
                idx_start = i*self.par_slices
                idx_stop = (i+1)*self.par_slices+self.overlap
                z2_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], z2_part[i].data,
                        z2[idx_start:idx_stop, ...],
                        wait_for=z2_part[i].events, is_blocking=False))
                symgrad_v_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], symgrad_v_part[i].data,
                        symgrad_v[idx_start:idx_stop, ...],
                        wait_for=symgrad_v_part[i].events, is_blocking=False))
                symgrad_v_vold_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], symgrad_v_vold_part[i].data,
                        symgrad_v_vold[idx_start:idx_stop, ...],
                        wait_for=symgrad_v_vold_part[i].events,
                        is_blocking=False))
                Kyk2_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], Kyk2_part[i].data,
                        Kyk2[idx_start:idx_stop, ...],
                        wait_for=Kyk2_part[i].events, is_blocking=False))
                z1_new_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i], z1_new_part[i].data,
                        z1_new[idx_start:idx_stop, ...],
                        wait_for=z1_new_part[i].events, is_blocking=False))
            for i in range(self.num_dev):
                if i == 0:
                    first = 1
                else:
                    first = 0
                z2_new_part[i].add_event(
                    self.update_z2(
                        z2_new_part[i], z2_part[i], symgrad_v_part[i],
                        symgrad_v_vold_part[i],
                        beta_line*tau_new, theta_line, beta, i, 0))
                Kyk2_new_part[i].add_event(
                    self.update_Kyk2(
                        Kyk2_new_part[i], z2_new_part[i], z1_new_part[i],
                        i, 0, first))
            first = 0
            for i in range(self.num_dev):
                idx_start = (i+1+self.num_dev-1)*self.par_slices
                idx_stop = (i+2+self.num_dev-1)*self.par_slices
                if idx_stop == self.NSlice:
                    idx_start -= self.overlap
                else:
                    idx_stop += self.overlap
                z2_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], z2_part[i+self.num_dev].data,
                        z2[idx_start:idx_stop, ...],
                        wait_for=z2_part[i+self.num_dev].events,
                        is_blocking=False))
                symgrad_v_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], symgrad_v_part[i+self.num_dev].data,
                        symgrad_v[idx_start:idx_stop, ...],
                        wait_for=symgrad_v_part[i+self.num_dev].events,
                        is_blocking=False))
                symgrad_v_vold_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1],
                        symgrad_v_vold_part[i+self.num_dev].data,
                        symgrad_v_vold[idx_start:idx_stop, ...],
                        wait_for=symgrad_v_vold_part[i+self.num_dev].events,
                        is_blocking=False))
                Kyk2_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], Kyk2_part[i+self.num_dev].data,
                        Kyk2[idx_start:idx_stop, ...],
                        wait_for=Kyk2_part[i+self.num_dev].events,
                        is_blocking=False))
                z1_new_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+1], z1_new_part[i+self.num_dev].data,
                        z1_new[idx_start:idx_stop, ...],
                        wait_for=z1_new_part[i+self.num_dev].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                z2_new_part[i+self.num_dev].add_event(
                    self.update_z2(
                        z2_new_part[i+self.num_dev], z2_part[self.num_dev+i],
                        symgrad_v_part[self.num_dev+i],
                        symgrad_v_vold_part[self.num_dev+i],
                        beta_line*tau_new, theta_line, beta, i, 1))
                Kyk2_new_part[i+self.num_dev].add_event(
                    self.update_Kyk2(
                        Kyk2_new_part[i+self.num_dev],
                        z2_new_part[i+self.num_dev],
                        z1_new_part[i+self.num_dev], i, 1, first))

            # Stream
            for j in range(2*self.num_dev,
                           int(self.NSlice/(2*self.par_slices*self.num_dev) +
                               (2*self.num_dev-1))):
                for i in range(self.num_dev):
                    # Get Data
                    self.queue[4*(self.num_dev-1)+3].finish()
                    if i > 1:
                        self.queue[4*(i-1)+2].finish()
                    idx_start = (i*self.par_slices +
                                 (2*self.num_dev*(j-2*self.num_dev) *
                                  self.par_slices))
                    idx_stop = ((i+1)*self.par_slices +
                                (2*self.num_dev*(j-2*self.num_dev)) *
                                self.par_slices+self.overlap)
                    z2_new_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+2], z2_new[idx_start:idx_stop, ...],
                            z2_new_part[i].data,
                            wait_for=z2_new_part[i].events,
                            is_blocking=False))
                    Kyk2_new_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+2],
                            Kyk2_new[idx_start:idx_stop, ...],
                            Kyk2_new_part[i].data,
                            wait_for=Kyk2_new_part[i].events,
                            is_blocking=False))
                    ynorm += ((
                        clarray.vdot(
                            z2_new_part[i][:self.par_slices, ...] -
                            z2_part[i][:self.par_slices, ...],
                            z2_new_part[i][:self.par_slices, ...] -
                            z2_part[i][:self.par_slices, ...],
                            queue=self.queue[4*i]))).get()
                    lhs += ((
                        clarray.vdot(
                            Kyk2_new_part[i][:self.par_slices, ...] -
                            Kyk2_part[i][:self.par_slices, ...],
                            Kyk2_new_part[i][:self.par_slices, ...] -
                            Kyk2_part[i][:self.par_slices, ...],
                            queue=self.queue[4*i]))).get()
                for i in range(self.num_dev):
                    # Put Data
                    idx_start = (i*self.par_slices +
                                 (2*self.num_dev*(j-2*self.num_dev+1) *
                                  self.par_slices))
                    idx_stop = ((i+1)*self.par_slices +
                                (2*self.num_dev*(j-2*self.num_dev+1)) *
                                self.par_slices+self.overlap)
                    z2_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i], z2_part[i].data,
                            z2[idx_start:idx_stop, ...],
                            wait_for=z2_part[i].events, is_blocking=False))
                    symgrad_v_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i], symgrad_v_part[i].data,
                            symgrad_v[idx_start:idx_stop, ...],
                            wait_for=symgrad_v_part[i].events,
                            is_blocking=False))
                    symgrad_v_vold_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i], symgrad_v_vold_part[i].data,
                            symgrad_v_vold[idx_start:idx_stop, ...],
                            wait_for=symgrad_v_vold_part[i].events,
                            is_blocking=False))
                    Kyk2_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i], Kyk2_part[i].data,
                            Kyk2[idx_start:idx_stop, ...],
                            wait_for=Kyk2_part[i].events,
                            is_blocking=False))
                    z1_new_part[i].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i], z1_new_part[i].data,
                            z1_new[idx_start:idx_stop, ...],
                            wait_for=z1_new_part[i].events, is_blocking=False))
                for i in range(self.num_dev):
                    z2_new_part[i].add_event(
                        self.update_z2(
                            z2_new_part[i], z2_part[i], symgrad_v_part[i],
                            symgrad_v_vold_part[i],
                            beta_line*tau_new, theta_line, beta, i, 0))
                    Kyk2_new_part[i].add_event(
                        self.update_Kyk2(
                            Kyk2_new_part[i], z2_new_part[i],
                            z1_new_part[i], i, 0, first))
                for i in range(self.num_dev):
                    # Get Data
                    self.queue[4*(self.num_dev-1)+2].finish()
                    if i > 1:
                        self.queue[4*(i-1)+3].finish()
                    idx_start = (i*self.par_slices +
                                 (2*self.num_dev*(j-2*self.num_dev) +
                                  self.num_dev)*self.par_slices)
                    idx_stop = ((i+1)*self.par_slices +
                                (2*self.num_dev*(j-2*self.num_dev) +
                                 self.num_dev)*self.par_slices+self.overlap)
                    z2_new_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+3], z2_new[idx_start:idx_stop, ...],
                            z2_new_part[i+self.num_dev].data,
                            wait_for=z2_new_part[i+self.num_dev].events,
                            is_blocking=False))
                    Kyk2_new_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+3],
                            Kyk2_new[idx_start:idx_stop, ...],
                            Kyk2_new_part[i+self.num_dev].data,
                            wait_for=Kyk2_new_part[i+self.num_dev].events,
                            is_blocking=False))
                    ynorm += ((
                        clarray.vdot(
                          z2_new_part[i+self.num_dev][:self.par_slices, ...] -
                          z2_part[i+self.num_dev][:self.par_slices, ...],
                          z2_new_part[i+self.num_dev][:self.par_slices, ...] -
                          z2_part[i+self.num_dev][:self.par_slices, ...],
                          queue=self.queue[4*i+1]))).get()
                    lhs += ((
                      clarray.vdot(
                        Kyk2_new_part[i+self.num_dev][:self.par_slices, ...] -
                        Kyk2_part[i+self.num_dev][:self.par_slices, ...],
                        Kyk2_new_part[i+self.num_dev][:self.par_slices, ...] -
                        Kyk2_part[i+self.num_dev][:self.par_slices, ...],
                        queue=self.queue[4*i+1]))).get()
                for i in range(self.num_dev):
                    # Put Data
                    idx_start = (i*self.par_slices +
                                 (2*self.num_dev*(j-2*self.num_dev+1) +
                                  self.num_dev)*self.par_slices)
                    idx_stop = ((i+1)*self.par_slices +
                                (2*self.num_dev*(j-2*self.num_dev+1) +
                                 self.num_dev)*self.par_slices)
                    if idx_stop == self.NSlice:
                        idx_start -= self.overlap
                    else:
                        idx_stop += self.overlap
                    z2_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+1], z2_part[i+self.num_dev].data,
                            z2[idx_start:idx_stop, ...],
                            wait_for=z2_part[i+self.num_dev].events,
                            is_blocking=False))
                    symgrad_v_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+1],
                            symgrad_v_part[i+self.num_dev].data,
                            symgrad_v[idx_start:idx_stop, ...],
                            wait_for=symgrad_v_part[i+self.num_dev].events,
                            is_blocking=False))
                    symgrad_v_vold_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                          self.queue[4*i+1],
                          symgrad_v_vold_part[i+self.num_dev].data,
                          symgrad_v_vold[idx_start:idx_stop, ...],
                          wait_for=symgrad_v_vold_part[i+self.num_dev].events,
                          is_blocking=False))
                    Kyk2_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+1], Kyk2_part[i+self.num_dev].data,
                            Kyk2[idx_start:idx_stop, ...],
                            wait_for=Kyk2_part[i+self.num_dev].events,
                            is_blocking=False))
                    z1_new_part[i+self.num_dev].add_event(
                        cl.enqueue_copy(
                            self.queue[4*i+1],
                            z1_new_part[i+self.num_dev].data,
                            z1_new[idx_start:idx_stop, ...],
                            wait_for=z1_new_part[i+self.num_dev].events,
                            is_blocking=False))
                for i in range(self.num_dev):
                    z2_new_part[i+self.num_dev].add_event(
                        self.update_z2(
                            z2_new_part[i+self.num_dev],
                            z2_part[self.num_dev+i],
                            symgrad_v_part[self.num_dev+i],
                            symgrad_v_vold_part[self.num_dev+i],
                            beta_line*tau_new, theta_line, beta, i, 1))
                    Kyk2_new_part[i+self.num_dev].add_event(
                        self.update_Kyk2(
                            Kyk2_new_part[i+self.num_dev],
                            z2_new_part[i+self.num_dev],
                            z1_new_part[i+self.num_dev], i, 1, first))
            # Collect last block
            if j < 2*self.num_dev:
                j = 2*self.num_dev
            else:
                j += 1
            for i in range(self.num_dev):
                self.queue[4*(self.num_dev-1)+3].finish()
                if i > 1:
                    self.queue[4*(i-1)+2].finish()
                idx_start = (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev) *
                              self.par_slices))
                idx_stop = ((i+1)*self.par_slices +
                            (2*self.num_dev*(j-2*self.num_dev)) *
                            self.par_slices+self.overlap)
                z2_new_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2], z2_new[idx_start:idx_stop, ...],
                        z2_new_part[i].data,
                        wait_for=z2_new_part[i].events, is_blocking=False))
                Kyk2_new_part[i].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+2],  Kyk2_new[idx_start:idx_stop, ...],
                        Kyk2_new_part[i].data,
                        wait_for=Kyk2_new_part[i].events,
                        is_blocking=False))
                ynorm += ((
                    clarray.vdot(
                        z2_new_part[i][:self.par_slices, ...] -
                        z2_part[i][:self.par_slices, ...],
                        z2_new_part[i][:self.par_slices, ...] -
                        z2_part[i][:self.par_slices, ...],
                        queue=self.queue[4*i]))).get()
                lhs += ((
                    clarray.vdot(
                        Kyk2_new_part[i][:self.par_slices, ...] -
                        Kyk2_part[i][:self.par_slices, ...],
                        Kyk2_new_part[i][:self.par_slices, ...] -
                        Kyk2_part[i][:self.par_slices, ...],
                        queue=self.queue[4*i]))).get()
            for i in range(self.num_dev):
                self.queue[4*(self.num_dev-1)+2].finish()
                if i > 1:
                    self.queue[4*(i-1)+3].finish()
                idx_start = (i*self.par_slices +
                             (2*self.num_dev*(j-2*self.num_dev) +
                              self.num_dev)*self.par_slices)
                idx_stop = ((i+1)*self.par_slices +
                            (2*self.num_dev*(j-2*self.num_dev) +
                             self.num_dev)*self.par_slices)
                if idx_stop == self.NSlice:
                    idx_start -= self.overlap
                    ynorm += ((
                        clarray.vdot(
                            z2_new_part[i+self.num_dev][self.overlap:, ...] -
                            z2_part[i+self.num_dev][self.overlap:, ...],
                            z2_new_part[i+self.num_dev][self.overlap:, ...] -
                            z2_part[i+self.num_dev][self.overlap:, ...],
                            queue=self.queue[4*i+1]))).get()
                    lhs += ((
                        clarray.vdot(
                            Kyk2_new_part[i+self.num_dev][self.overlap:, ...] -
                            Kyk2_part[i+self.num_dev][self.overlap:, ...],
                            Kyk2_new_part[i+self.num_dev][self.overlap:, ...] -
                            Kyk2_part[i+self.num_dev][self.overlap:, ...],
                            queue=self.queue[4*i+1]))).get()
                else:
                    idx_stop += self.overlap
                    ynorm += ((
                        clarray.vdot(
                          z2_new_part[i+self.num_dev][:self.par_slices, ...] -
                          z2_part[i+self.num_dev][:self.par_slices, ...],
                          z2_new_part[i+self.num_dev][:self.par_slices, ...] -
                          z2_part[i+self.num_dev][:self.par_slices, ...],
                          queue=self.queue[4*i+1]))).get()
                    lhs += ((
                      clarray.vdot(
                        Kyk2_new_part[i+self.num_dev][:self.par_slices, ...] -
                        Kyk2_part[i+self.num_dev][:self.par_slices, ...],
                        Kyk2_new_part[i+self.num_dev][:self.par_slices, ...] -
                        Kyk2_part[i+self.num_dev][:self.par_slices, ...],
                        queue=self.queue[4*i+1]))).get()
                self.queue[4*i+2].finish()
                z2_new_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], z2_new[idx_start:idx_stop, ...],
                        z2_new_part[i+self.num_dev].data,
                        wait_for=z2_new_part[i+self.num_dev].events,
                        is_blocking=False))
                Kyk2_new_part[i+self.num_dev].add_event(
                    cl.enqueue_copy(
                        self.queue[4*i+3], Kyk2_new[idx_start:idx_stop, ...],
                        Kyk2_new_part[i+self.num_dev].data,
                        wait_for=Kyk2_new_part[i+self.num_dev].events,
                        is_blocking=False))
            for i in range(self.num_dev):
                self.queue[4*i].finish()
                self.queue[4*i+1].finish()
                self.queue[4*i+2].finish()
                self.queue[4*i+3].finish()
        return (ynorm, lhs)
