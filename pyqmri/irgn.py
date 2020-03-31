#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This module holds the classes for IRGN Optimization without streaming.

Attribues:
  DTYPE (complex64):
    Complex working precission. Currently single precission only.
  DTYPE_real (float32):
    Real working precission. Currently single precission only.
"""
from __future__ import division
import time
import numpy as np

from pkg_resources import resource_filename
import pyopencl as cl
import pyopencl.array as clarray
import h5py

import pyqmri.operator as operator
import pyqmri.solver as optimizer
from pyqmri._helper_fun import CLProgram as Program
from pyqmri._helper_fun import _utils as utils

DTYPE = np.complex64
DTYPE_real = np.float32


class IRGNOptimizer:
    """ IRGN Optimization

    This Class performs IRGN Optimization either with TGV or TV regularization
    and a combination of TGV/TV + H1.

    Attributes:
      par (dict): A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      gn_res ((list of) float):
        The residual values for of each Gauss-Newton step. Each iteration
        appends its value to the list.
    """

    def __init__(self, par, trafo=1, imagespace=False, SMS=0, reg_type='TGV',
                 config='', model=None, streamed=False):
        self.par = par
        self._fval_old = 0
        self._fval = 0
        self._fval_init = 0
        self._ctx = par["ctx"]
        self._queue = par["queue"]
        self.gn_res = []
        self.irgn_par = utils.read_config(config, reg_type)
        utils.save_config(self.irgn_par, par["outdir"], reg_type)
        self.model = model
        self.reg_type = reg_type
        self._prg = []
        self.num_dev = len(par["num_dev"])
        self.NSlice = par["NSlice"]
        self.streamed = streamed
        self._imagespace = imagespace
        self._SMS = SMS
        if streamed and par["NSlice"]/(self.num_dev*par["par_slices"]) < 2:
            raise ValueError(
                "Number of Slices devided by parallel "
                "computed slices and devices needs to be larger two.\n"
                "Current values are %i total Slices, %i parallel slices and "
                "%i compute devices."
                % (par["NSlice"], par["par_slices"], self.num_dev))
        if streamed and par["NSlice"] % par["par_slices"]:
            raise ValueError(
                "Number of Slices devided by parallel "
                "computed slices needs to be an integer.\n"
                "Current values are %i total Slices with %i parallel slices."
                % (par["NSlice"], par["par_slices"]))
        if DTYPE == np.complex128:
            if streamed:
                kernname = 'kernels/OpenCL_Kernels_double_streamed.c'
            else:
                kernname = 'kernels/OpenCL_Kernels_double.c'
            for j in range(self.num_dev):
                self._prg.append(Program(
                    self._ctx[j],
                    open(
                        resource_filename(
                            'pyqmri', kernname)
                        ).read()))
        else:
            if streamed:
                kernname = 'kernels/OpenCL_Kernels_streamed.c'
            else:
                kernname = 'kernels/OpenCL_Kernels.c'
            for j in range(self.num_dev):
                self._prg.append(Program(
                    self._ctx[j],
                    open(
                        resource_filename(
                            'pyqmri', kernname)).read()))

        self._imagespace = imagespace

        if imagespace:
            self._coils = []
            self.sliceaxis = 1
        else:
            self.data_shape = (par["NScan"], par["NC"],
                               par["NSlice"], par["Nproj"], par["N"])
            if self.streamed:
                self.data_trans_axes = (2, 0, 1, 3, 4)
                self._coils = np.require(
                    np.swapaxes(par["C"], 0, 1), requirements='C',
                    dtype=DTYPE)

                if SMS:
                    self.data_shape = (par["packs"]*par["numofpacks"],
                                       par["NScan"],
                                       par["NC"], par["dimY"], par["dimX"])
                    self.data_shape_T = (par["NScan"], par["NC"],
                                         par["packs"]*par["numofpacks"],
                                         par["dimY"], par["dimX"])
                    self._expdim_dat = 1
                    self._expdim_C = 0
                else:
                    self.data_shape = (par["NSlice"], par["NScan"], par["NC"],
                                       par["Nproj"], par["N"])
                    self.data_shape_T = self.data_shape
                    self._expdim_dat = 2
                    self._expdim_C = 1
            else:
                self._coils = clarray.to_device(self._queue[0],
                                                self.par["C"])

        self._op, self._FT = operator.Operator.MRIOperatorFactory(
            par,
            self._prg,
            DTYPE,
            DTYPE_real,
            trafo,
            imagespace,
            SMS,
            streamed
            )

        self.grad_op, self.symgrad_op, self.v = self._setupLinearOps(
            DTYPE,
            DTYPE_real)

        self.pdop = optimizer.PDBaseSolver.factory(
            self._prg,
            self._queue,
            self.par,
            self.irgn_par,
            self._fval_init,
            self._coils,
            linops=(self._op, self.grad_op, self.symgrad_op),
            model=model,
            reg_type=self.reg_type,
            SMS=self._SMS,
            streamed=self.streamed,
            imagespace=self._imagespace
            )

        self.step_val = None
        self.model_partial_der = None
        self.modelgrad = None
        self.delta = None
        self.delta_max = None
        self.omega = None
        self.gamma = None
        self.data = None  # Needs to be set outside

    def _setupLinearOps(self, DTYPE, DTYPE_real):
        grad_op = operator.Operator.GradientOperatorFactory(
            self.par,
            self._prg,
            DTYPE,
            DTYPE_real,
            self.streamed)
        symgrad_op = None
        v = None
        if self.reg_type == 'TGV':
            symgrad_op = operator.Operator.SymGradientOperatorFactory(
                self.par,
                self._prg,
                DTYPE,
                DTYPE_real,
                self.streamed)
            v = np.zeros(
                ([self.par["unknowns"], self.par["NSlice"],
                  self.par["dimY"], self.par["dimX"], 4]),
                dtype=DTYPE)
            if self.streamed:
                v = np.require(np.swapaxes(v, 0, 1), requirements='C')
        return grad_op, symgrad_op, v

###############################################################################
# Start a 3D Reconstruction, set TV to True to perform TV instead of TGV#######
# Precompute Model and Gradient values for xk #################################
# Call inner optimization #####################################################
# input: bool to switch between TV (1) and TGV (0) regularization #############
# output: optimal value of x ##################################################
###############################################################################
    def _executeIRGN3D(self):
        iters = self.irgn_par["start_iters"]
        result = np.copy(self.model.guess)

        if self.streamed:
            self.data = np.require(
                np.transpose(self.data, self.data_trans_axes),
                requirements='C')

        self.step_val = np.nan_to_num(self.model.execute_forward(result))

        if self.streamed:
            if self._SMS is False:
                self.step_val = np.require(
                    np.swapaxes(self.step_val, 0, 1), requirements='C')

        for ign in range(self.irgn_par["max_gn_it"]):
            start = time.time()
            self.modelgrad = np.nan_to_num(
                self.model.execute_gradient(result))

            self._balanceModelGradients(result)
            # self.pdop.grad_op.updateRatio(result)

            self.step_val = np.nan_to_num(self.model.execute_forward(result))

            if self.streamed:
                if self._SMS is False:
                    self.step_val = np.require(
                        np.swapaxes(self.step_val, 0, 1), requirements='C')
                self.modelgrad = np.require(
                    np.transpose(self.modelgrad, self.data_trans_axes),
                    requirements='C')
                self.pdop.model = self.model
                self.pdop.modelgrad = self.modelgrad
            else:
                self.modelgrad = clarray.to_device(
                    self._queue[0],
                    self.modelgrad)
                self.pdop.model = self.model
                self.pdop.modelgrad = self.modelgrad

            self._updateIRGNRegPar(result, ign)
            self.pdop.updateRegPar(self.irgn_par)

            result = self._irgnSolve3D(result, iters, self.data, ign)

            iters = np.fmin(iters * 2, self.irgn_par["max_iters"])

            end = time.time() - start
            self.gn_res.append(self._fval)
            print("-" * 75)
            print("GN-Iter: %d  Elapsed time: %f seconds" % (ign, end))
            print("-" * 75)
            self._fval_old = self._fval
            self._saveToFile(ign, self.model.rescale(result))
        self._calcResidual(result, self.data, ign+1)

    def _updateIRGNRegPar(self, result, ign):
        self.irgn_par["delta"] = np.minimum(
            self.delta
            * self.irgn_par["delta_inc"]**ign,
            self.irgn_par["delta_max"])
        self.irgn_par["gamma"] = np.maximum(
            self.gamma * self.irgn_par["gamma_dec"]**ign,
            self.irgn_par["gamma_min"])
        self.irgn_par["omega"] = np.maximum(
            self.omega * self.irgn_par["omega_dec"]**ign,
            self.irgn_par["omega_min"])

    def _balanceModelGradients(self, result):
        scale = np.reshape(
            self.modelgrad,
            (self.par["unknowns"],
             self.par["NScan"] * self.par["NSlice"] *
             self.par["dimY"] * self.par["dimX"]))
        scale = np.linalg.norm(scale, axis=-1)
        print("Initial Norm: ", np.linalg.norm(scale))
        print("Initial Ratio: ", scale)
        scale /= np.linalg.norm(scale)/np.sqrt(self.par["unknowns"])
        scale = 1 / scale
        for uk in range(self.par["unknowns"]):
            self.model.constraints[uk].update(scale[uk])
            result[uk, ...] *= self.model.uk_scale[uk]
            self.modelgrad[uk] /= self.model.uk_scale[uk]
            self.model.uk_scale[uk] *= scale[uk]
            result[uk, ...] /= self.model.uk_scale[uk]
            self.modelgrad[uk] *= self.model.uk_scale[uk]
        scale = np.reshape(
            self.modelgrad,
            (self.par["unknowns"],
             self.par["NScan"] * self.par["NSlice"] *
             self.par["dimY"] * self.par["dimX"]))
        scale = np.linalg.norm(scale, axis=-1)
        print("Norm after rescale: ", np.linalg.norm(scale))
        print("Ratio after rescale: ", scale)

###############################################################################
# New .hdf5 save files ########################################################
###############################################################################
    def _saveToFile(self, myit, result):
        f = h5py.File(self.par["outdir"]+"output_" + self.par["fname"], "a")
        if self.reg_type == 'TGV':
            f.create_dataset("tgv_result_iter_"+str(myit), result.shape,
                             dtype=DTYPE, data=result)
            f.attrs['res_tgv_iter_'+str(myit)] = self._fval
        else:
            f.create_dataset("tv_result_"+str(myit), result.shape,
                             dtype=DTYPE, data=result)
            f.attrs['res_tv_iter_'+str(myit)] = self._fval
        f.close()

###############################################################################
# Precompute constant terms of the GN linearization step ######################
# input: linearization point x ################################################
# iters: number of innner iterations iters ####################################
# data: the input data ########################################################
# TV: bool to switch between TV (1) and TGV (0) regularization ################
# output: optimal value of x for the inner GN step ############################
###############################################################################
###############################################################################
    def _irgnSolve3D(self, x, iters, data, GN_it):
        b = self._calcResidual(x, data, GN_it)

        if self.streamed:
            x = np.require(np.swapaxes(x, 0, 1), requirements='C')
            res = data - b + self._op.fwdoop(
                [[x, self._coils, self.modelgrad]])
        else:
            tmpx = clarray.to_device(self._queue[0], x)
            res = data - b + self._op.fwdoop(
                [tmpx, self._coils, self.modelgrad]).get()
            del tmpx

        tmpres = self.pdop.run(x, res, iters)
        for key in tmpres:
            if key == 'x':
                if type(tmpres[key]) == np.ndarray:
                    x = tmpres["x"]
                else:
                    x = tmpres["x"].get()
            if key == 'v':
                if type(tmpres[key]) == np.ndarray:
                    self.v = tmpres["v"]
                else:
                    self.v = tmpres["v"].get()
        if self.streamed:
            x = np.require(np.swapaxes(x, 0, 1), requirements='C')

        return x

    def _calcResidual(self, x, data, GN_it):
        if self.streamed:
            b, grad, sym_grad = self._calcFwdGNPartStreamed(x)
            grad_tv = grad[:, :self.par["unknowns_TGV"]]
            grad_H1 = grad[:, self.par["unknowns_TGV"]:]
        else:
            b, grad, sym_grad = self._calcFwdGNPartLinear(x)
            grad_tv = grad[:self.par["unknowns_TGV"]]
            grad_H1 = grad[self.par["unknowns_TGV"]:]
        del grad

        datacost = self.irgn_par["lambd"] / 2 * np.linalg.norm(data - b)**2
        L2Cost = np.linalg.norm(x)/(2.0*self.irgn_par["delta"])
        if self.reg_type == 'TV':
            regcost = self.irgn_par["gamma"] * \
                np.sum(np.abs(grad_tv))
        else:
            regcost = self.irgn_par["gamma"] * np.sum(
                  np.abs(grad_tv -
                         self.v)) + self.irgn_par["gamma"] * 2 * np.sum(
                             np.abs(sym_grad))
            del sym_grad

        self._fval = (datacost +
                      regcost +
                      L2Cost +
                      self.irgn_par["omega"] / 2 *
                      np.linalg.norm(grad_H1)**2)
        del grad_tv, grad_H1

        if GN_it == 0:
            self._fval_init = self._fval
            self.pdop._fval_init = self._fval

        print("-" * 75)
        print("Costs of Data: %f" % (datacost * 1e3))
        print("Costs of T(G)V: %f" % (regcost * 1e3))
        print("Costs of L2 Term: %f" % (L2Cost * 1e3))
        print("-" * 75)
        print("Function value at GN-Step %i: %f" %
              (GN_it, 1e3*self._fval / self._fval_init))
        print("-" * 75)
        return b

    def _calcFwdGNPartLinear(self, x):
        if self._imagespace is False:
            b = clarray.empty(self._queue[0], self.data_shape, dtype=DTYPE)
            self._FT.FFT(b, clarray.to_device(
                self._queue[0],
                (self.step_val[:, None, ...] *
                 self.par["C"]))).wait()
            b = b.get()
        else:
            b = self.step_val

        x = clarray.to_device(self._queue[0], np.require(x, requirements="C"))
        grad = clarray.to_device(self._queue[0],
                                 np.zeros(x.shape+(4,), dtype=DTYPE))
        grad.add_event(
            self.grad_op.fwd(
                grad,
                x,
                wait_for=grad.events +
                x.events))
        x = x.get()
        grad = grad.get()
        sym_grad = None
        if self.reg_type == 'TGV':
            v = clarray.to_device(self._queue[0], self.v)
            sym_grad = clarray.to_device(self._queue[0],
                                         np.zeros(x.shape+(8,), dtype=DTYPE))
            sym_grad.add_event(
                self.symgrad_op.fwd(
                    sym_grad,
                    v,
                    wait_for=sym_grad.events +
                    v.events))
            sym_grad = sym_grad.get()

        return b, grad, sym_grad

    def _calcFwdGNPartStreamed(self, x):
        x = np.require(np.swapaxes(x, 0, 1), requirements='C')
        if self._imagespace is False:
            b = np.zeros(self.data_shape_T, dtype=DTYPE)
            if self._SMS is True:
                self._op.FTstr.eval(
                    [b],
                    [[np.expand_dims(self.step_val, self._expdim_dat) *
                      np.expand_dims(self.par["C"], self._expdim_C)]])
                b = np.require(
                    np.transpose(
                        b,
                        self.data_trans_axes),
                    requirements='C')
            else:
                self._op.FTstr.eval(
                    [b],
                    [[np.expand_dims(self.step_val, self._expdim_dat) *
                      np.expand_dims(self._coils, self._expdim_C)]])
        else:
            b = self.step_val
        grad = np.zeros(x.shape+(4,), dtype=DTYPE)
        self.grad_op.fwd([grad], [[x]])

        sym_grad = None
        if self.reg_type == 'TGV':
            sym_grad = np.zeros(x.shape+(8,), dtype=DTYPE)
            self.symgrad_op.fwd([sym_grad], [[self.v]])

        return b, grad, sym_grad

    def execute(self, reco_2D=0):
        if reco_2D:
            print("2D currently not implemented, \
                  3D can be used with a single slice.")
            raise NotImplementedError
        else:
            # self.irgn_par["lambd"] *= (
            #                             (self.par["SNR_est"]))
            self.gamma = self.irgn_par["gamma"]
            self.delta = self.irgn_par["delta"]
            self.omega = self.irgn_par["omega"]
            self._executeIRGN3D()
