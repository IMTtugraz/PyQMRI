#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import time
import sys
from pkg_resources import resource_filename
import pyopencl.array as clarray
import h5py
import pyqmri.operator as operator
import pyqmri.streaming as streaming
from pyqmri._helper_fun import CLProgram as Program
from pyqmri._helper_fun import _utils as utils
DTYPE = np.complex64
DTYPE_real = np.float32


class ModelReco:
    def __init__(self, par, trafo=1, imagespace=False, reg_type='TGV',
                 config='', model=None):

        par["overlap"] = 1
        self.overlap = par["overlap"]
        self.par_slices = par["par_slices"]
        self.par = par
        self.C = np.require(
            np.transpose(par["C"], [1, 0, 2, 3]), requirements='C',
            dtype=DTYPE)
        self.unknowns_TGV = par["unknowns_TGV"]
        self.unknowns_H1 = par["unknowns_H1"]
        self.unknowns = par["unknowns"]
        self.NSlice = par["NSlice"]
        self.NScan = par["NScan"]
        self.dimX = par["dimX"]
        self.dimY = par["dimY"]
        self.scale = 1

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
        self.irgn_par = utils.read_config(config, reg_type)
        self.model = model
        self.reg_type = reg_type
        self.num_dev = len(par["num_dev"])
        if self.NSlice/(self.num_dev*self.par_slices) < 2:
            raise ValueError(
                "Number of Slices devided by parallel "
                "computed slices and devices needs to be larger two.")
        if self.NSlice % self.par_slices:
            raise ValueError(
                "Number of Slices devided by parallel "
                "computed slices needs to be an integer.")
        self.prg = []
        for j in range(self.num_dev):
            self.prg.append(
                Program(
                    self.ctx[j],
                    open(
                        resource_filename(
                            'pyqmri',
                            'kernels/OpenCL_Kernels_streamed.c')).read()))

        self.tmp_img = []

        self.unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        self.grad_shape = self.unknown_shape + (4,)
        self._imagespace = imagespace
        if imagespace:
            self.data_shape = (self.NSlice, self.NScan,
                               self.dimY, self.dimX)
            self.C = []
            self.NC = 1
            self.N = self.dimX
            self.Nproj = self.dimY
            self.dat_trans_axes = [1, 0, 2, 3]
            self.op = operator.OperatorImagespaceStreamed(par, self.prg)
        else:
            self.NC = par["NC"]
            self.dat_trans_axes = [2, 0, 1, 3, 4]

            self.data_shape = (self.NSlice, self.NScan,
                               self.NC, self.Nproj, self.N)
            self.data_shape_T = self.data_shape
            self.op = operator.OperatorKspaceStreamed(par,
                                                      self.prg,
                                                      trafo=trafo)
            self._expdim_dat = 2
            self._expdim_C = 1

        self.grad_op, self.symgrad_op, self.v = self._setupLinearOps()

        self.step_val = None
        self.pdop = None
        self.model_partial_der = None
        self.grad_buf = None
        self.delta = None
        self.delta_max = None
        self.omega = None
        self.gamma = None
        self.data = None  # Needs to be set outside

    def update_Kyk2(self, outp, inp, par=None, idx=0, idxq=0,
                    bound_cond=0, wait_for=[]):
        return self.prg[idx].update_Kyk2(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data, inp[1].data,
            np.int32(self.unknowns), np.int32(bound_cond), np.float32(self.dz),
            wait_for=outp.events + inp[0].events + inp[1].events+wait_for)

    def update_primal(self, outp, inp, par=None, idx=0, idxq=0,
                      bound_cond=0, wait_for=[]):
        return self.prg[idx].update_primal(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data, inp[1].data, inp[2].data,
            np.float32(par[0]),
            np.float32(par[0]/par[1]), np.float32(1/(1+par[0]/par[1])),
            self.min_const[idx].data, self.max_const[idx].data,
            self.real_const[idx].data, np.int32(self.unknowns),
            wait_for=(outp.events +
                      inp[0].events+inp[1].events +
                      inp[2].events+wait_for))

    def update_z1(self, outp, inp, par=None, idx=0, idxq=0,
                  bound_cond=0, wait_for=[]):
        return self.prg[idx].update_z1(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data, inp[1].data,
            inp[2].data, inp[3].data, inp[4].data,
            np.float32(par[0]), np.float32(par[1]),
            np.float32(1/par[2]), np.int32(self.unknowns_TGV),
            np.int32(self.unknowns_H1), np.float32(1 / (1 + par[0] / par[3])),
            wait_for=(outp.events+inp[0].events+inp[1].events +
                      inp[2].events+inp[3].events+inp[4].events+wait_for))

    def update_z1_tv(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=[]):
        return self.prg[idx].update_z1_tv(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data, inp[0].data, inp[0].data,
            np.float32(par[0]),
            np.float32(par[1]),
            np.float32(1/par[2]), np.int32(self.unknowns_TGV),
            np.int32(self.unknowns_H1), np.float32(1 / (1 + par[0] / par[3])),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def update_z2(self, outp, inp, par=None, idx=0, idxq=0,
                  bound_cond=0, wait_for=[]):
        return self.prg[idx].update_z2(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data, inp[1].data, inp[2].data,
            np.float32(par[0]),
            np.float32(par[1]),
            np.float32(1/par[2]), np.int32(self.unknowns),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def update_r(self, outp, inp, par=None, idx=0, idxq=0,
                 bound_cond=0, wait_for=[]):
        return self.prg[idx].update_r(
            self.queue[4*idx+idxq], (outp.size,), None,
            outp.data, inp[0].data,
            inp[1].data, inp[2].data, inp[3].data,
            np.float32(par[0]), np.float32(par[1]),
            np.float32(1/(1+par[0]/self.irgn_par["lambd"])),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def update_v(self, outp, inp, par=None, idx=0, idxq=0,
                 bound_cond=0, wait_for=[]):
        return self.prg[idx].update_v(
            self.queue[4*idx+idxq], (outp[..., 0].size,), None,
            outp.data, inp[0].data, inp[1].data, np.float32(par[0]),
            wait_for=outp.events+inp[0].events+inp[1].events+wait_for)

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

    def _setupLinearOps(self):
        grad_op = operator.OperatorFiniteGradientStreamed(self.par, self.prg)
        symgrad_op = None
        v = None
        if self.reg_type == 'TGV':
            symgrad_op = operator.OperatorFiniteSymGradientStreamed(
                self.par, self.prg)
            v = np.zeros(
                ([self.par["unknowns"], self.par["NSlice"],
                  self.par["dimY"], self.par["dimX"], 4]),
                dtype=DTYPE)
        return grad_op, symgrad_op, v

    def execute(self, reco_2D=0):
        if reco_2D:
            NotImplementedError("2D currently not implemented, "
                                "3D can be used with a single slice.")
        else:
            self.irgn_par["lambd"] *= 1e2/np.sqrt(self.par["SNR_est"])
            self.delta = self.irgn_par["delta"]
            self.delta_max = self.irgn_par["delta_max"]
            self.gamma = self.irgn_par["gamma"]
            self.omega = self.irgn_par["omega"]
            self._setup_reg_tmp_arrays()
            self.execute_3D()

###############################################################################
# Start a 3D Reconstruction, set TV to True to perform TV instead of TGV#######
# Precompute Model and Gradient values for xk #################################
# Call inner optimization #####################################################
# input: bool to switch between TV (1) and TGV (0) regularization #############
# output: optimal value of x ##################################################
###############################################################################
    def execute_3D(self):

        iters = self.irgn_par["start_iters"]
        result = np.copy(self.model.guess)
        self.data = np.require(
            np.transpose(self.data, self.dat_trans_axes), requirements='C')
        self.grad_x = np.nan_to_num(self.model.execute_gradient(result))

        self._balanceModelGradients(result, 0)
        self.grad_op.updateRatio(result)
        self._updateIRGNRegPar(result, 0)

        for ign in range(self.irgn_par["max_gn_it"]):
            start = time.time()

            if ign > 0:
                self.grad_x = np.nan_to_num(
                    self.model.execute_gradient(result))
                self._balanceModelGradients(result, ign)
                self.grad_op.updateRatio(result)

            self.step_val = np.nan_to_num(
                self.model.execute_forward(result))

            self.step_val = np.require(
                np.transpose(
                    self.step_val, [1, 0, 2, 3]), requirements='C')

            self.grad_x = np.require(
                np.transpose(self.grad_x, [2, 0, 1, 3, 4]), requirements='C')

            self._updateIRGNRegPar(result, ign)

            result = self.irgn_solve_3D(result, iters, ign)

            iters = np.fmin(iters * 2, self.irgn_par["max_iters"])

            end = time.time() - start

            self.gn_res.append(self.fval)
            print("-" * 75)
            print("GN-Iter: %d  Elapsed time: %f seconds" % (ign, end))
            print("-" * 75)
#            if np.abs(self.fval_old - self.fval) / self.fval_init < \
#               self.irgn_par["tol"]:
#                print("Terminated at GN-iteration %d because "
#                      "the energy decrease was less than %.3e" %
#                      (ign, np.abs(self.fval_old - self.fval) /
#                       self.fval_init))
#                self.calc_residual(
#                    np.require(np.transpose(result, [1, 0, 2, 3]),
#                               requirements='C'),
#                    ign+1)
#                self.savetofile(ign, self.model.rescale(result))
#                break
            self.fval_old = self.fval
            self.savetofile(ign, self.model.rescale(result))

        self.calc_residual(
            np.require(
                np.transpose(result, [1, 0, 2, 3]),
                requirements='C'),
            ign+1)

    def _updateIRGNRegPar(self, result, ign):
        self.irgn_par["delta_max"] = (self.delta_max /
                                      1e3 *
                                      np.linalg.norm(result))
        self.irgn_par["delta"] = np.minimum(
            self.delta /
            (1e3) *
            np.linalg.norm(result)*self.irgn_par["delta_inc"]**ign,
            self.irgn_par["delta_max"])
        self.irgn_par["gamma"] = np.maximum(
            self.gamma * self.irgn_par["gamma_dec"]**ign,
            self.irgn_par["gamma_min"])
        self.irgn_par["omega"] = np.maximum(
            self.omega * self.irgn_par["omega_dec"]**ign,
            self.irgn_par["omega_min"])

    def _balanceModelGradients(self, result, ind):
        scale = np.reshape(
            self.grad_x,
            (self.par["unknowns"],
             self.par["NScan"] * self.par["NSlice"] *
             self.par["dimY"] * self.par["dimX"]))
        scale = np.linalg.norm(scale, axis=-1)
        scale = 1e3 / scale
        if not np.mod(ind, 1):
            for uk in range(self.par["unknowns"]):
                self.model.constraints[uk].update(scale[uk])
                result[uk, ...] *= self.model.uk_scale[uk]
                self.grad_x[uk] /= self.model.uk_scale[uk]
                self.model.uk_scale[uk] *= scale[uk]
                result[uk, ...] /= self.model.uk_scale[uk]
                self.grad_x[uk] *= self.model.uk_scale[uk]

    def _setup_reg_tmp_arrays(self):
        if self.reg_type == 'TV':
            self.tau = np.float32(1/np.sqrt(8))
            self.beta_line = 400
            self.theta_line = np.float32(1.0)
        elif self.reg_type == 'TGV':
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
            raise NotImplementedError("Not implemented")
        self._setupstreamingops()

        self.r = np.zeros_like(self.data, dtype=DTYPE)
        self.r = np.require(np.transpose(self.r, self.dat_trans_axes),
                            requirements='C')
        self.z1 = np.zeros(
            ([self.NSlice, self.unknowns, self.dimY, self.dimX, 4]),
            dtype=DTYPE)

###############################################################################
# New .hdf5 save files ########################################################
###############################################################################
    def savetofile(self, myit, result):
        f = h5py.File(self.par["outdir"]+"output_" + self.par["fname"], "a")
        if self.reg_type == 'TGV':
            f.create_dataset("tgv_result_iter_"+str(myit), result.shape,
                             dtype=DTYPE, data=result)
            f.attrs['res_tgv_iter_'+str(myit)] = self.fval
        else:
            f.create_dataset("tv_result_"+str(myit), result.shape,
                             dtype=DTYPE, data=result)
            f.attrs['res_tv_iter_'+str(myit)] = self.fval
        f.close()

###############################################################################
# Precompute constant terms of the GN linearization step ######################
# input: linearization point x ################################################
# numeber of innner iterations iters ##########################################
# Data ########################################################################
# bool to switch between TV (1) and TGV (0) regularization ####################
# output: optimal value of x for the inner GN step ############################
###############################################################################
###############################################################################
    def irgn_solve_3D(self, x, iters, GN_it):
        x = np.require(np.transpose(x, [1, 0, 2, 3]), requirements='C')
        DGk = np.zeros(self.data_shape, dtype=DTYPE)

        self.op.fwd(
            [DGk],
            [[x, self.C, self.grad_x]])

        b = self.calc_residual(x, GN_it)

        res = self.data - b + DGk

        if self.reg_type == 'TV':
            x = self.tv_solve_3D(x, res, iters)
        elif self.reg_type == 'TGV':
            x = self.tgv_solve_3D(x, res, iters)
        x = np.require(np.transpose(x, [1, 0, 2, 3]), requirements='C')
        return x

    def calc_residual(self, x, GN_it):
        grad = np.zeros(self.z1.shape, dtype=DTYPE)
        self.grad_op.fwd([grad], [[x]])

        if self._imagespace is False:
            b = np.zeros(self.data_shape_T, dtype=DTYPE)
            self.op.FTstr.eval(
                [b],
                [[np.expand_dims(self.step_val, self._expdim_dat) *
                  np.expand_dims(self.C, self._expdim_C)]])
        else:
            b = self.step_val

        datacost = self.irgn_par["lambd"] / 2 *\
            np.linalg.norm(self.data - b)**2
        L2Cost = np.linalg.norm(x)/(2.0*self.irgn_par["delta"])

        if self.reg_type == 'TV':
            regcost = self.irgn_par["gamma"]*np.sum(np.abs(
                    grad[:, :self.unknowns_TGV]))

        elif self.reg_type == 'TGV':
            sym_grad = np.zeros_like(self.z2)
            self.symgrad_op.fwd([sym_grad], [[self.v]])
            regcost = self.irgn_par["gamma"] * np.sum(
                  np.abs(grad[:, :self.par["unknowns_TGV"]] -
                         self.v)) + self.irgn_par["gamma"] * 2 * np.sum(
                             np.abs(sym_grad))
            del sym_grad

        self.fval = (datacost +
                     regcost +
                     self.irgn_par["omega"] / 2 *
                     np.linalg.norm(grad[:, self.par["unknowns_TGV"]:])**2)
        del grad
        if GN_it == 0:
            self.fval_init = self.fval
        print("-" * 75)
        print("Costs of Data: %f" % (datacost))
        print("Costs of T(G)V: %f" % (regcost))
        print("Costs of L2 Term: %f" % (L2Cost))
        print("-" * 75)
        print("Function value at GN-Step %i: %f" %
              (GN_it, 1e3*self.fval / self.fval_init))
        print("-" * 75)
        return b

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
        ynorm1 = np.float32(0.0)
        lhs1 = np.float32(0.0)
        ynorm2 = np.float32(0.0)
        lhs2 = np.float32(0.0)
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

        # Warmup
        self.stream_initial_1.eval(
            [Axold, Kyk1, symgrad_v_vold],
            [[x, self.C, self.grad_x], [r, z1, self.C, self.grad_x, []], [v]],
            [self.grad_op._ratio])
        self.stream_initial_2.eval(
            [gradx_xold, Kyk2],
            [[x], [z2, z1, []]])
        # Start Iterations
        for myit in range(iters):
            self.update_primal_1.eval(
                [x_new, gradx, Ax],
                [[x, Kyk1, xk], [], [[], self.C, self.grad_x]],
                [tau, delta])
            self.update_primal_2.eval(
                [v_new, symgrad_v],
                [[v, Kyk2], []],
                [tau])

            beta_new = beta_line*(1+mu*tau)
            tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
            beta_line = beta_new

            while True:
                theta_line = tau_new/tau
                (lhs1, ynorm1) = self.update_dual_1.evalwithnorm(
                    [z1_new, r_new, Kyk1_new],
                    [[z1, gradx, gradx_xold, v_new, v],
                     [r, Ax, Axold, res],
                     [[], [], self.C, self.grad_x, Kyk1]],
                    [beta_line*tau_new, theta_line,
                     alpha, omega, self.grad_op._ratio])
                (lhs2, ynorm2) = self.update_dual_2.evalwithnorm(
                    [z2_new, Kyk2_new],
                    [[z2, symgrad_v, symgrad_v_vold], [[], z1_new, Kyk2]],
                    [beta_line*tau_new, theta_line, beta])

                if np.sqrt(beta_line)*tau_new*(abs(lhs1+lhs2)**(1/2)) <= \
                   (abs(ynorm1+ynorm2)**(1/2))*delta_line:
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
                        np.vdot(gradx[:, :self.unknowns_TGV],
                                gradx[:, :self.unknowns_TGV])).real

                    dual = (
                        - delta/2*np.vdot(-Kyk1.flatten(), -Kyk1.flatten())
                        - np.vdot(xk.flatten(), -Kyk1.flatten())
                        + np.sum(Kyk2)
                        - 1/(2*self.irgn_par["lambd"])
                        * np.vdot(r.flatten(), r.flatten())
                        - np.vdot(res.flatten(), r.flatten())
                        - 1 / (2 * self.irgn_par["omega"])
                        * np.vdot(z1[:, :self.unknowns_TGV],
                                  z1[:, :self.unknowns_TGV])).real
                else:
                    primal_new = (
                        self.irgn_par["lambd"]/2 *
                        np.vdot(Axold-res, Axold-res) +
                        alpha*np.sum(abs((gradx[:, :self.unknowns_TGV]-v))) +
                        beta*np.sum(abs(symgrad_v)) +
                        1/(2*delta)*np.vdot(x_new-xk, x_new-xk)).real

                    dual = (
                        - delta/2*np.vdot(-Kyk1.flatten(), -Kyk1.flatten())
                        - np.vdot(xk.flatten(), -Kyk1.flatten())
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
        ynorm1 = np.float32(0.0)
        lhs1 = np.float32(0.0)
        ynorm2 = np.float32(0.0)
        lhs2 = np.float32(0.0)
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

        # Warmup
        self.stream_initial_1.eval(
            [Axold, Kyk1],
            [[x, self.C, self.grad_x], [r, z1, self.C, self.grad_x, []]])
        self.stream_initial_2.eval(
            [gradx_xold],
            [[x]])

        for myit in range(iters):
            self.update_primal_1.eval(
                [x_new, gradx, Ax],
                [[x, Kyk1, xk], [], [[], self.C, self.grad_x]],
                [tau, delta])

            beta_new = beta_line*(1+mu*tau)
            tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
            beta_line = beta_new

            while True:
                theta_line = tau_new/tau

                (lhs1, ynorm1) = self.update_dual_1.evalwithnorm(
                    [z1_new, r_new, Kyk1_new],
                    [[z1, gradx, gradx_xold],
                     [r, Ax, Axold, res],
                     [[], [], self.C, self.grad_x, Kyk1]],
                    [beta_line*tau_new, theta_line,
                     alpha, omega])

                if np.sqrt(beta_line)*tau_new*(abs(lhs1+lhs2)**(1/2)) <= \
                   (abs(ynorm1+ynorm2)**(1/2))*delta_line:
                    break
                else:
                    tau_new = tau_new*mu_line

            (Kyk1, Kyk1_new, Axold, Ax, z1, z1_new, r, r_new, gradx_xold,
             gradx, tau) = (
             Kyk1_new, Kyk1, Ax, Axold, z1_new, z1, r_new, r, gradx,
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
                        np.vdot(gradx[:, :self.unknowns_TGV],
                                gradx[:, :self.unknowns_TGV])).real

                    dual = (
                        -delta/2*np.vdot(Kyk1, Kyk1) - np.vdot(xk, (-1)*Kyk1)
                        - 1/(2*self.irgn_par["lambd"])*np.vdot(r, r)
                        - np.vdot(res, r)
                        - 1 / (2 * self.irgn_par["omega"])
                        * np.vdot(z1[:, :self.unknowns_TGV],
                                  z1[:, :self.unknowns_TGV])).real
                else:
                    primal_new = (
                        self.irgn_par["lambd"]/2 *
                        np.vdot(Axold-res, Axold-res) +
                        alpha*np.sum(abs((gradx[:, :self.unknowns_TGV]))) +
                        1/(2*delta)*np.vdot(x_new-xk, x_new-xk)).real

                    dual = (
                        -delta/2*np.vdot(Kyk1, Kyk1) - np.vdot(xk, (-1)*Kyk1)
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

    def _setupstreamingops(self):
        if self.reg_type == 'TGV':
            symgrad_shape = self.unknown_shape + (8,)

        self.stream_initial_1 = self._defineoperator(
            [],
            [],
            [[]],
            reverse_dir=True)
        self.stream_initial_1 += self.op.fwdstr
        self.stream_initial_1 += self.op.adjstrKyk1
        if self.reg_type == 'TGV':
            self.stream_initial_1 += self.symgrad_op._stream_symgrad

        if self.reg_type == 'TGV':
            self.stream_Kyk2 = self._defineoperator(
                [self.update_Kyk2],
                [self.grad_shape],
                [[symgrad_shape,
                  self.grad_shape,
                  self.grad_shape]])

            self.stream_initial_2 = self._defineoperator(
                [],
                [],
                [[]])

            self.stream_initial_2 += self.grad_op._stream_grad
            self.stream_initial_2 += self.stream_Kyk2
        self.stream_primal = self._defineoperator(
            [self.update_primal],
            [self.unknown_shape],
            [[self.unknown_shape,
              self.unknown_shape,
              self.unknown_shape]])

        self.update_primal_1 = self._defineoperator(
            [],
            [],
            [[]])

        self.update_primal_1 += self.stream_primal
        self.update_primal_1 += self.grad_op._stream_grad
        self.update_primal_1 += self.op.fwdstr

        self.update_primal_1.connectouttoin(0, (1, 0))
        self.update_primal_1.connectouttoin(0, (2, 0))

        if self.reg_type == 'TGV':
            self.stream_update_v = self._defineoperator(
                [self.update_v],
                [self.grad_shape],
                [[self.grad_shape,
                  self.grad_shape]])

            self.update_primal_2 = self._defineoperator(
                [],
                [],
                [[]],
                reverse_dir=True)

            self.update_primal_2 += self.stream_update_v
            self.update_primal_2 += self.symgrad_op._stream_symgrad
            self.update_primal_2.connectouttoin(0, (1, 0))

        if self.reg_type == 'TV':
            self.stream_update_z1 = self._defineoperator(
                [self.update_z1],
                [self.grad_shape],
                [[self.grad_shape,
                  self.grad_shape,
                  self.grad_shape]])
        else:
            self.stream_update_z1 = self._defineoperator(
                [self.update_z1],
                [self.grad_shape],
                [[self.grad_shape,
                 self. grad_shape,
                 self. grad_shape,
                 self. grad_shape,
                 self. grad_shape]])

        self.stream_update_r = self._defineoperator(
            [self.update_r],
            [self.data_shape],
            [[self.data_shape,
              self.data_shape,
              self.data_shape,
              self.data_shape]])

        self.update_dual_1 = self._defineoperator(
            [],
            [],
            [[]],
            reverse_dir=True,
            posofnorm=[False, False, True])

        self.update_dual_1 += self.stream_update_z1
        self.update_dual_1 += self.stream_update_r
        self.update_dual_1 += self.op.adjstrKyk1
        self.update_dual_1.connectouttoin(0, (2, 1))
        self.update_dual_1.connectouttoin(1, (2, 0))

        del self.stream_update_z1, self.stream_update_r, \
            self.stream_update_v, self.stream_primal

        if self.reg_type == 'TGV':
            self.stream_update_z2 = self._defineoperator(
                [self.update_z2],
                [symgrad_shape],
                [[symgrad_shape,
                  symgrad_shape,
                  symgrad_shape]])

            self.update_dual_2 = self._defineoperator(
                [],
                [],
                [[]],
                posofnorm=[False, True])

            self.update_dual_2 += self.stream_update_z2
            self.update_dual_2 += self.stream_Kyk2
            self.update_dual_2.connectouttoin(0, (1, 0))
            del self.stream_Kyk2, self.stream_update_z2

    def _defineoperator(self,
                        functions,
                        outp,
                        inp,
                        reverse_dir=False,
                        posofnorm=[],
                        slices=None):
        if slices is None:
            slices = self.NSlice
        return streaming.Stream(
            functions,
            outp,
            inp,
            self.par_slices,
            self.overlap,
            slices,
            self.queue,
            self.num_dev,
            reverse_dir,
            posofnorm)
