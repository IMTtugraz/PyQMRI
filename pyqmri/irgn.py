#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the classes for IRGN Optimization without streaming."""
from __future__ import division
import time
import numpy as np

from pkg_resources import resource_filename
import pyopencl.array as clarray
import h5py

import pyqmri.operator as operator
import pyqmri.solver as optimizer
from pyqmri._helper_fun import CLProgram as Program
from pyqmri._helper_fun import _utils as utils
from scipy import linalg as spl
import faulthandler; faulthandler.enable()


class IRGNOptimizer:
    """Main IRGN Optimization class.

    This Class performs IRGN Optimization either with TGV or TV regularization.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      model : pyqmri.model
        Which model should be used for fitting.
        Expects a pyqmri.model instance.
      trafo : int, 1
        Select radial (1, default) or cartesian (0) sampling for the fft.
      imagespace : bool, false
        Perform the fitting from k-space data (false, default) or from the
        image series (true)
      SMS : int, 0
        Select if simultaneous multi-slice acquisition was used (1) or
        standard slice-by-slice acquisition was done (0, default).
      reg_type : str, "TGV"
        Select between "TGV" (default) or "TV" regularization.
      config : str, ''
        Name of config file. If empty, default config file will be generated.
      streamed : bool, false
        Select between standard reconstruction (false)
        or streamed reconstruction (true) for large volumetric data which
        does not fit on the GPU memory at once.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.

    Attributes
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      gn_res : list of floats
        The residual values for of each Gauss-Newton step. Each iteration
        appends its value to the list.
      irgn_par : dict
        The parameters read from the config file to guide the IRGN
        optimization process
    """

    def __init__(self, par, model, trafo=1, imagespace=False, SMS=0,
                 reg_type='TGV', config='', streamed=False,
                 DTYPE=np.complex64, DTYPE_real=np.float32):
        self.par = par
        self.gn_res = []
        self.irgn_par = utils.read_config(config, reg_type)
        utils.save_config(self.irgn_par, par["outdir"], reg_type)
        num_dev = len(par["num_dev"])
        self._fval_old = 0
        self._fval = 0
        self._fval_init = 0
        self._ctx = par["ctx"]
        self._queue = par["queue"]
        self._model = model
        self._reg_type = reg_type
        self._prg = []

        self._DTYPE = DTYPE
        self._DTYPE_real = DTYPE_real

        self._streamed = streamed
        self._imagespace = imagespace
        self._SMS = SMS
        if streamed and par["NSlice"]/(num_dev*par["par_slices"]) < 2:
            raise ValueError(
                "Number of Slices devided by parallel "
                "computed slices and devices needs to be larger two.\n"
                "Current values are %i total Slices, %i parallel slices and "
                "%i compute devices."
                % (par["NSlice"], par["par_slices"], num_dev))
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
            for j in range(num_dev):
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
            for j in range(num_dev):
                self._prg.append(Program(
                    self._ctx[j],
                    open(
                        resource_filename(
                            'pyqmri', kernname)).read()))

        if imagespace:
            self._coils = []
            self.sliceaxis = 1
            if self._streamed:
                self._data_trans_axes = (1, 0, 2, 3)
                self._grad_trans_axes = (2, 0, 1, 3, 4)
        else:
            if SMS:
                self._data_shape = (par["NScan"], par["NC"],
                                    par["packs"]*par["numofpacks"],
                                    par["Nproj"], par["N"])
            else:
                if par["is3D"] and trafo:
                    self._data_shape = (par["NScan"], par["NC"],
                                        1, par["Nproj"], par["N"])
                else:
                    self._data_shape = (par["NScan"], par["NC"],
                                        par["NSlice"], par["Nproj"], par["N"])
            if self._streamed:
                self._data_trans_axes = (2, 0, 1, 3, 4)
                self._grad_trans_axes = (2, 0, 1, 3, 4)
                self._coils = np.require(
                    np.swapaxes(par["C"], 0, 1), requirements='C',
                    dtype=DTYPE)

                if SMS:
                    self._data_shape = (par["packs"]*par["numofpacks"],
                                        par["NScan"],
                                        par["NC"], par["dimY"], par["dimX"])
                    self._data_shape_T = (par["NScan"], par["NC"],
                                          par["packs"]*par["numofpacks"],
                                          par["dimY"], par["dimX"])
                    self._expdim_dat = 1
                    self._expdim_C = 0
                else:
                    self._data_shape = (par["NSlice"], par["NScan"], par["NC"],
                                        par["Nproj"], par["N"])
                    self._data_shape_T = self._data_shape
                    self._expdim_dat = 2
                    self._expdim_C = 1
            else:
                self._coils = clarray.to_device(self._queue[0],
                                                self.par["C"])

        self._MRI_operator, self._FT = operator.Operator.MRIOperatorFactory(
            par,
            self._prg,
            DTYPE,
            DTYPE_real,
            trafo,
            imagespace,
            SMS,
            streamed
            )

        grad_op, symgrad_op, self._v = self._setupLinearOps(
            DTYPE,
            DTYPE_real)

        if not reg_type == "H1":
            self._pdop = optimizer.PDBaseSolver.factory(
                self._prg,
                self._queue,
                self.par,
                self.irgn_par,
                self._fval_init,
                self._coils,
                linops=(self._MRI_operator, grad_op, symgrad_op),
                model=model,
                reg_type=self._reg_type,
                SMS=self._SMS,
                streamed=self._streamed,
                imagespace=self._imagespace,
                DTYPE=DTYPE,
                DTYPE_real=DTYPE_real
                )
        else:
            self._pdop = optimizer.CGSolver_H1(
                self._prg,
                self._queue,
                self.par,
                self.irgn_par,
                self._coils,
                linops=(self._MRI_operator, grad_op)
                )

        self._gamma = None
        self._delta = None
        self._omega = None
        self._step_val = None
        self._modelgrad = None

    def _setupLinearOps(self, DTYPE, DTYPE_real):
        grad_op = operator.Operator.GradientOperatorFactory(
            self.par,
            self._prg,
            DTYPE,
            DTYPE_real,
            self._streamed)
        symgrad_op = None
        v = None
        if self._reg_type == 'TGV':
            symgrad_op = operator.Operator.SymGradientOperatorFactory(
                self.par,
                self._prg,
                DTYPE,
                DTYPE_real,
                self._streamed)
            v = np.zeros(
                ([self.par["unknowns_TGV"], self.par["NSlice"],
                  self.par["dimY"], self.par["dimX"], 4]),
                dtype=DTYPE)
            if self._streamed:
                v = np.require(np.swapaxes(v, 0, 1), requirements='C')
        return grad_op, symgrad_op, v

    def execute(self, data):
        """Start the IRGN optimization.

        This method performs iterative regularized Gauss-Newton optimization
        and calls the inner loop after precomputing the current linearization
        point. Results of the fitting process are saved after each
        linearization step to the output folder.

        Parameters
        ----------
          data : numpy.array
            the data to perform optimization/fitting on.
        """
        self.const_save = self._model.constraints
        
        self._gamma = self.irgn_par["gamma"]
        self._delta = self.irgn_par["delta"]
        self._omega = self.irgn_par["omega"]
        self.lambd = self.irgn_par["lambd"]

        iters = self.irgn_par["start_iters"]
        result = np.copy(self._model.guess)

        if "inc" in self.irgn_par.keys():
            inc = int(self.irgn_par["inc"])
        else:
            inc = 2

        if self._streamed:
            data = np.require(
                np.transpose(data, self._data_trans_axes),
                requirements='C')
            if self._SMS is False:
                self._step_val = np.require(
                    np.swapaxes(self._step_val, 0, 1), requirements='C')
                
        # data_save = data

        for ign in range(self.irgn_par["max_gn_it"]):
            start = time.time()


            self._modelgrad = np.nan_to_num(
                self._model.execute_gradient(result))
            self._step_val = np.nan_to_num(
                self._model.execute_forward(result))
                
            # Precond only for image data!!!!
            if ign < 0:
                self.precond = True
                self._pdop.precond = True
                self._pdop._grad_op.precond = True
                if self._reg_type == 'TGV':
                    self._pdop._symgrad_op.precond = True
                if False:
                    self._balanceModelGradientsAverage(result, ign)
                else:
                    self._balanceModelGradients(result, ign)
                result = self.applyPrecond(result)
                self._pdop.irgn = self
            else:
                self.precond = False
                self._pdop.precond = False
                self._pdop._grad_op.precond = False
                if self._reg_type == 'TGV':
                    self._pdop._symgrad_op.precond = False
                self._balanceModelGradientsNorm(result, ign)
            
            self._updateIRGNRegPar(ign)
            if self.precond:
                # self.irgn_par["gamma"] *= 1e-20
                self._pdop._grad_op.updateRatio(np.require(self.UTE#np.repeat(np.diag(np.ones(self.k,))[..., None], self.UTE.shape[-1],axis=-1)
                                          ,requirements='C'))

            if self._streamed:
                if self._SMS is False:
                    self._step_val = np.require(
                        np.swapaxes(self._step_val, 0, 1), requirements='C')
                self._modelgrad = np.require(
                    np.transpose(self._modelgrad, self._grad_trans_axes),
                    requirements='C')
                self._pdop.model = self._model
                self._pdop.modelgrad = self._modelgrad
            else:
                _jacobi = np.sum(
                    np.abs(
                        self._modelgrad)**2, 1).astype(self._DTYPE_real)
                _jacobi[_jacobi == 0] = 1e-8
                self._pdop.model = self._model
                self._pdop.modelgrad = clarray.to_device(
                    self._queue[0],
                    self._modelgrad)

            self._pdop.updateRegPar(self.irgn_par)

            result = self._irgnSolve3D(result, iters, data, ign)
            if self.precond:
                result = self.removePrecond(result)

            iters = np.fmin(iters * inc, self.irgn_par["max_iters"])

            end = time.time() - start
            self.gn_res.append(self._fval)
            print("-" * 75)
            print("GN-Iter: %d  Elapsed time: %f seconds" % (ign, end))
            print("-" * 75)
            self._fval_old = self._fval
            self._saveToFile(ign, self._model.rescale(result)["data"])
            if ign > 1:
                if (np.abs(self.gn_res[-1]-self.gn_res[-2])/self.gn_res[0]
                    < self.irgn_par["rtol"]):
                    print(
            "Terminated after GN iteration %d because the energy "
            "decrease in the GN-problem was %.3e which is below the "
            "relative tolerance of %.3e" 
            % (ign, np.abs(self.gn_res[-1]-self.gn_res[-2])/self.gn_res[0], 
                             self.irgn_par["rtol"]))
                    break
        if self.precond:            
            self._calcResidual(self.applyPrecond(result), data, ign+1)
        else:
            self._calcResidual((result), data, ign+1)
            
    def _updateIRGNRegPar(self, ign):

        try:
            self.irgn_par["delta"] = np.minimum(
                self._delta
                * self.irgn_par["delta_inc"]**ign,
                self.irgn_par["delta_max"])*self.irgn_par["lambd"]
        except OverflowError:
            self.irgn_par["delta"] = np.minimum(
                np.finfo(self._delta).max,
                self.irgn_par["delta_max"])

        
        self.irgn_par["gamma"] = np.maximum(
            self._gamma * self.irgn_par["gamma_dec"]**ign,
            self.irgn_par["gamma_min"])/self.irgn_par["lambd"]
        self.irgn_par["omega"] = np.maximum(
            self._omega * self.irgn_par["omega_dec"]**ign,
            self.irgn_par["omega_min"])/self.irgn_par["lambd"]

    def _balanceModelGradients(self, result, ign):
        
        jacobi = self._modelgrad.reshape(*self._modelgrad.shape[:2], -1)
        
        self.k = np.minimum(*self._modelgrad.shape[:2])
        
        U = np.zeros((jacobi.shape[-1], self.k, self._modelgrad.shape[0]), dtype=self.par["DTYPE"])
        V = np.zeros((jacobi.shape[-1], self._modelgrad.shape[1], self.k), dtype=self.par["DTYPE"])
        E = np.zeros((jacobi.shape[-1], self.k), dtype=self.par["DTYPE"])
        self.UTE = np.zeros((jacobi.shape[-1], self.k, self._modelgrad.shape[0]), dtype=self.par["DTYPE"])
        self.EU = np.zeros((jacobi.shape[-1], self.k, self._modelgrad.shape[0]), dtype=self.par["DTYPE"])
        self.UT = np.zeros((jacobi.shape[-1], self.k, self._modelgrad.shape[0]), dtype=self.par["DTYPE"])
        
        jacobi = np.require(jacobi.T, requirements='C')
        
        cutoff = np.inf

        maxval = 0
        minval = 1e30
        
        for j in range(jacobi.shape[0]):
            V[j], E[j], U[j] = spl.svd(jacobi[j], full_matrices=False)
            if np.any(E[j].imag!=0):
                print("Non-zero complex eigenvalue")
            einv = 1/E[j]
            einv[~np.isfinite(einv)] = cutoff*einv[0]
            einv[einv/einv[0] > cutoff] = cutoff*einv[0]

            V[j] = V[j]@np.diag(E[j])@np.diag(einv)
            maxval = np.maximum(maxval, np.max((E[j])))
            minval = np.minimum(minval, np.min((E[j])))
  
            self.UTE[j] = (np.conj(U[j].T)@np.diag(einv))
            self.UT[j] = np.conj(U[j].T)@np.diag(einv)
            self.EU[j] = np.diag(1/einv)@U[j]
            
        print("Maximum Eigenvalue: ", maxval)
        print("Minimum Eigenvalue: ", minval)
        print("Condition number: ", maxval/minval)
        
        print("Mean Eigenvalues: ", np.mean(E, axis=0).real)
            
        
        V = np.require(np.transpose(V, (2,1,0)), requirements='C')

        self._modelgrad = V.reshape(*V.shape[:2],*self._modelgrad.shape[2:])
        
        self.UTE = np.require(self.UTE.transpose(1,2,0), requirements='C')
        self.UT = np.require(self.UT.transpose(1,2,0), requirements='C')
        self.EU = np.require(self.EU.transpose(1,2,0), requirements='C')
        
        self._pdop.EU = clarray.to_device(self._pdop._queue[0], self.EU)
        self._pdop.UTE = clarray.to_device(self._pdop._queue[0], self.UTE)
        
    def _balanceModelGradientsAverage(self, result, ign):
        
        
        scale = self._modelgrad.reshape(self.par["unknowns"], self.par["NScan"], -1)

        tmp = np.quantile(scale, 0.85, axis=-1)
        V, eigvals, U = spl.svd(tmp.T)
        
        jacobi = self._modelgrad.reshape(*self._modelgrad.shape[:2], -1)
        
        self.k = np.minimum(*self._modelgrad.shape[:2])
        
        self.UTE = np.zeros((jacobi.shape[-1], self.k, self._modelgrad.shape[0]), dtype=self.par["DTYPE"])
        self.EU = np.zeros((jacobi.shape[-1], self.k, self._modelgrad.shape[0]), dtype=self.par["DTYPE"])
        
        jacobi = np.require(jacobi.T, requirements='C')
        
        cutoff = np.inf
        
        einv = 1/(eigvals)
        einv[~np.isfinite(einv)] = cutoff*einv[0]
        einv[einv/einv[0] > cutoff] = cutoff*einv[0]
        
        for j in range(jacobi.shape[0]):

            
            jacobi[j] = jacobi[j]@np.conj(U.T)@np.conj(np.diag(einv))
            
            self.UTE[j] = (np.conj(U.T)@np.conj(np.diag(einv)))
            self.EU[j] = np.diag(1/einv)@U
            
            
        print("Average Eigenvalues: ", eigvals)


        self._modelgrad = np.require(jacobi.T.reshape(*self._modelgrad.shape), requirements='C')
        
        self.UTE = np.require(self.UTE.transpose(1,2,0), requirements='C')
        self.EU = np.require(self.EU.transpose(1,2,0), requirements='C')
        
        self._pdop.EU = clarray.to_device(self._pdop._queue[0], self.EU)
        self._pdop.UTE = clarray.to_device(self._pdop._queue[0], self.UTE)


    def applyPrecond(self, inp):
        tmp_step_val = np.require(inp.reshape(inp.shape[0], -1).T, requirements='C')
        precond_stepval = np.zeros((np.prod(inp.shape[1:]), self.k), dtype=self.par["DTYPE"])
        
        for j in range(tmp_step_val.shape[0]):
            precond_stepval[j] = self.EU[...,j ]@tmp_step_val[j]
                                                 
        return np.require(precond_stepval.T.reshape(self.k, *inp.shape[1:]), requirements='C', dtype=self.par["DTYPE"])
    
    def removePrecond(self, inp):
        tmp_step_val = np.require(inp.reshape(inp.shape[0], -1).T, requirements='C')
        precond_stepval = np.zeros((np.prod(inp.shape[1:]), self.k), dtype=self.par["DTYPE"])
        for j in range(tmp_step_val.shape[0]):
            precond_stepval[j] = self.UTE[..., j]@tmp_step_val[j]
        return np.require(precond_stepval.T.reshape(self.k, *inp.shape[1:]), requirements='C', dtype=self.par["DTYPE"])
    
    def _balanceModelGradientsNorm(self, result, ign):
        scale = self._modelgrad.reshape(self.par["unknowns"], -1)
        scale = np.linalg.norm(scale, axis=-1)
        print("Initial Norm: ", np.linalg.norm(scale))
        print("Initial Ratio: ", scale)
        scale /= 1e2/np.sqrt(self.par["unknowns"])
        scale = 1/scale
        scale[~np.isfinite(scale)] = 1e2/np.sqrt(self.par["unknowns"])
        for uk in range(self.par["unknowns"]):
            self._model.constraints[uk].update(scale[uk])
            result[uk, ...] *= self._model.uk_scale[uk]
            self._modelgrad[uk] /= self._model.uk_scale[uk]
            self._model.uk_scale[uk] *= scale[uk]
            result[uk, ...] /= self._model.uk_scale[uk]
            self._modelgrad[uk] *= self._model.uk_scale[uk]
        scale = self._modelgrad.reshape(self.par["unknowns"], -1)
        scale = np.linalg.norm(scale, axis=-1)
        print("Norm after rescale: ", np.linalg.norm(scale))
        print("Ratio after rescale: ", np.abs(scale))


###############################################################################
# New .hdf5 save files ########################################################
###############################################################################
    def _saveToFile(self, myit, result):
        f = h5py.File(self.par["outdir"]+"output_" + self.par["fname"] + ".h5",
                      "a")
        if self._reg_type == 'TGV':
            f.create_dataset("tgv_result_iter_"+str(myit), result.shape,
                             dtype=self._DTYPE, data=result)
            f.attrs['res_tgv_iter_'+str(myit)] = self._fval
        else:
            f.create_dataset("tv_result_"+str(myit), result.shape,
                             dtype=self._DTYPE, data=result)
            f.attrs['res_tv_iter_'+str(myit)] = self._fval
        f.attrs['datacost_iter_'+str(myit)] = self._datacost
        f.attrs['regcost_iter_'+str(myit)] = self._regcost 
        f.attrs['data_norm'] = self.par["dscale"]
        # f.attrs['L2Cost_iter_'+str(myit)] = self._L2Cost 
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
        if self._streamed:
            x = np.require(np.swapaxes(x, 0, 1), requirements='C')
            res = data - b + self._MRI_operator.fwdoop(
                [[x, self._coils, self._pdop.modelgrad]])
        else:
            tmpx = clarray.to_device(self._queue[0], x)
            res = data - b + self._MRI_operator.fwdoop(
                [tmpx, self._coils, self._pdop.modelgrad]).get()
            del tmpx   
        
        tmpres = self._pdop.run((x, self._v), res, iters)
        for key in tmpres:
            if key == 'x':
                if isinstance(tmpres[key], np.ndarray):
                    x = tmpres["x"]
                else:
                    x = tmpres["x"].get()
            if key == 'v':
                if isinstance(tmpres[key], np.ndarray):
                    self._v = tmpres["v"]
                else:
                    self._v = tmpres["v"].get()
        if self._streamed:
            x = np.require(np.swapaxes(x, 0, 1), requirements='C')

        return x

    def _calcResidual(self, x, data, GN_it):
        if self._streamed:
            b, grad, sym_grad = self._calcFwdGNPartStreamed(x)
        else:
            b, grad, sym_grad = self._calcFwdGNPartLinear(x)
        grad_tv = grad[:self.par["unknowns_TGV"]]*self.par["weights"][:,None,None,None,None]
        if sym_grad is not None:
            sym_grad *= self.par["weights"][:,None,None,None,None]
        grad_H1 = grad[self.par["unknowns_TGV"]:]

        del grad
        self._datacost = self.irgn_par["lambd"] / 2 * np.linalg.norm(data - b)**2
#        self._L2Cost = np.linalg.norm(x)/(2.0*self.irgn_par["delta"])

        if self._reg_type == 'TV':
            self._regcost = self.irgn_par["gamma"] * \
                np.sum(np.abs(np.linalg.norm(grad_tv, axis=0)))
        elif self._reg_type == 'TGV':
            self._regcost = self.irgn_par["gamma"] * np.sum(
                  np.abs(np.linalg.norm(grad_tv -
                         self._v, axis=0))) + self.irgn_par["gamma"] * 2 * np.sum(
                             np.abs(np.linalg.norm(sym_grad, axis=0)))
            del sym_grad
        else:
            self.regcost = self.irgn_par["gamma"]/2 * \
                np.linalg.norm(grad_tv)**2

        self._fval = (self._datacost +
                      self._regcost +
#                      self._L2Cost +
                      self.irgn_par["omega"] / 2 *
                      np.linalg.norm(grad_H1.flatten())**2)
        del grad_tv, grad_H1

        if GN_it == 0:
            self._fval_init = self._fval
            self._pdop.setFvalInit(self._fval)
        
        if GN_it >= 0:
            print("-" * 75)
            print("Initial Cost: %f" % (self._fval_init))
            print("Costs of Data: %f" % (1e3*self._datacost / self._fval_init))
            print("Costs of T(G)V: %f" % (1e3*self._regcost / self._fval_init))
    #        print("Costs of L2 Term: %f" % (1e3*self._L2Cost / self._fval_init))
            print("-" * 75)
            print("Function value at GN-Step %i: %f" %
                  (GN_it, 1e3*self._fval / self._fval_init))
            print("-" * 75)
            return b
        return None

    def _calcFwdGNPartLinear(self, x):
        if self._imagespace is False:
            b = clarray.zeros(self._queue[0],
                              self._data_shape,
                              dtype=self._DTYPE)
            self._FT.FFT(b, clarray.to_device(
                self._queue[0],
                (self._step_val[:, None, ...] *
                 self.par["C"]))).wait()
            b = b.get()
        else:
            b = self._step_val

        x = clarray.to_device(self._queue[0], np.require(x, requirements="C"))
        grad = clarray.to_device(self._queue[0],
                                 np.zeros(x.shape+(4,), dtype=self._DTYPE))
        self._pdop._grad_op.fwd(
            grad,
            x,
            wait_for=grad.events +
            x.events).wait()
        x = x.get()
        grad = grad.get()
        sym_grad = None
        if self._reg_type == 'TGV':
            v = clarray.to_device(self._queue[0], self._v)
            sym_grad = clarray.to_device(self._queue[0],
                                         np.zeros(x.shape+(8,),
                                                  dtype=self._DTYPE))

            self._pdop._symgrad_op.fwd(
                sym_grad,
                v,
                wait_for=sym_grad.events +
                v.events).wait()
            sym_grad = sym_grad.get()

        return b, grad, sym_grad

    def _calcFwdGNPartStreamed(self, x):
        x = np.require(np.swapaxes(x, 0, 1), requirements='C')
        if self._imagespace is False:
            b = np.zeros(self._data_shape_T, dtype=self._DTYPE)
            if self._SMS is True:
                self._MRI_operator.FTstr.eval(
                    [b],
                    [[np.expand_dims(self._step_val, self._expdim_dat) *
                      np.expand_dims(self.par["C"], self._expdim_C)]])
                b = np.require(
                    np.transpose(
                        b,
                        self._data_trans_axes),
                    requirements='C')
            else:
                self._MRI_operator.FTstr.eval(
                    [b],
                    [[np.expand_dims(self._step_val, self._expdim_dat) *
                      np.expand_dims(self._coils, self._expdim_C)]])
        else:
            b = self._step_val
        grad = np.zeros(x.shape+(4,), dtype=self._DTYPE)
        self._pdop._grad_op.fwd([grad], [[x]])

        sym_grad = None
        if self._reg_type == 'TGV':
            sym_grad = np.zeros(x.shape+(8,), dtype=self._DTYPE)
            self._pdop._symgrad_op.fwd([sym_grad], [[self._v]])

        return b, grad, sym_grad
