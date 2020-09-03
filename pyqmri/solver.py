#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the classes for different numerical Optimizer."""

from __future__ import division
import sys
import numpy as np
from pkg_resources import resource_filename
import pyopencl as cl
import pyopencl.array as clarray
import pyqmri.operator as operator
from pyqmri._helper_fun import CLProgram as Program
import pyqmri.streaming as streaming


class CGSolver:
    """
    Conjugate Gradient Optimization Algorithm.

    This Class performs a CG reconstruction on single precission complex input
    data.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      NScan : int
        Number of Scan which should be used internally. Do not
        need to be the same number as in par["NScan"]
      trafo : bool
        Switch between radial (1) and Cartesian (0) fft.
      SMS : bool
        Simultaneouos Multi Slice. Switch between noraml (0)
        and slice accelerated (1) reconstruction.
    """

    def __init__(self, par, NScan=1, trafo=1, SMS=0):
        self._NSlice = par["NSlice"]
        NScan_save = par["NScan"]
        par["NScan"] = NScan
        self._NScan = NScan
        self._dimX = par["dimX"]
        self._dimY = par["dimY"]
        self._NC = par["NC"]
        self._queue = par["queue"][0]
        file = open(
            resource_filename(
                'pyqmri', 'kernels/OpenCL_Kernels.c'))
        self._prg = Program(
            par["ctx"][0],
            file.read())
        file.close()
        self._coils = cl.Buffer(par["ctx"][0],
                                cl.mem_flags.READ_ONLY |
                                cl.mem_flags.COPY_HOST_PTR,
                                hostbuf=par["C"].data)
        self._DTYPE = par["DTYPE"]
        self._DTYPE_real = par["DTYPE_real"]

        self.__op, FT = operator.Operator.MRIOperatorFactory(
            par,
            [self._prg],
            self._DTYPE,
            self._DTYPE_real,
            trafo=trafo,
            SMS=SMS
            )

        if SMS:
            self._tmp_sino = clarray.empty(
                self._queue,
                (self._NScan, self._NC,
                 int(self._NSlice/par["MB"]), par["Nproj"], par["N"]),
                self._DTYPE, "C")
        else:
            self._tmp_sino = clarray.empty(
                self._queue,
                (self._NScan, self._NC,
                 self._NSlice, par["Nproj"], par["N"]),
                self._DTYPE, "C")
        self._FT = FT.FFT
        self._FTH = FT.FFTH
        self._tmp_result = clarray.empty(
            self._queue,
            (self._NScan, self._NC,
             self._NSlice, self._dimY, self._dimX),
            self._DTYPE, "C")
        par["NScan"] = NScan_save
        self._scan_offset = 0

    def __del__(self):
        """Destructor.

        Releases GPU memory arrays.
        """
        del self._coils
        del self._tmp_result
        del self._queue
        del self._tmp_sino
        del self._FT
        del self._FTH

    def run(self, data, iters=30, lambd=1e-5, tol=1e-8, guess=None,
            scan_offset=0):
        """
        Start the CG reconstruction.

        All attributes after data are considered keyword only.

        Parameters
        ----------
          data : numpy.array
            The complex k-space data which serves as the basis for the images.
          iters : int
            Maximum number of CG iterations
          lambd : float
            Weighting parameter for the Tikhonov regularization
          tol : float
            Termination criterion. If the energy decreases below this
            threshold the algorithm is terminated.
          guess : numpy.array
            An optional initial guess for the images. If None, zeros is used.

        Returns
        -------
          numpy.Array:
              The result of the image reconstruction.
        """
        self._scan_offset = scan_offset
        if guess is not None:
            x = clarray.to_device(self._queue, guess)
        else:
            x = clarray.zeros(self._queue,
                              (self._NScan, 1,
                               self._NSlice, self._dimY, self._dimX),
                              self._DTYPE, "C")
        b = clarray.empty(self._queue,
                          (self._NScan, 1,
                           self._NSlice, self._dimY, self._dimX),
                          self._DTYPE, "C")
        Ax = clarray.empty(self._queue,
                           (self._NScan, 1,
                            self._NSlice, self._dimY, self._dimX),
                           self._DTYPE, "C")

        data = clarray.to_device(self._queue, data)
        self._operator_rhs(b, data)
        res = b
        p = res
        delta = np.linalg.norm(res.get())**2/np.linalg.norm(b.get())**2

        for i in range(iters):
            self._operator_lhs(Ax, p)
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
            beta = (clarray.vdot(res_new, res_new) /
                    clarray.vdot(res, res)).real.get()
            p = res_new+beta*p
            (res, res_new) = (res_new, res)
        del Ax, b, res, p, data, res_new
        return np.squeeze(x.get())

    def eval_fwd_kspace_cg(self, y, x, wait_for=None):
        """Apply forward operator for image reconstruction.

        Parameters
        ----------
          y : PyOpenCL.Array
            The result of the computation
          x : PyOpenCL.Array
            The input array
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        return self._prg.operator_fwd_cg(self._queue,
                                         (self._NSlice, self._dimY,
                                          self._dimX),
                                         None,
                                         y.data, x.data, self._coils,
                                         np.int32(self._NC),
                                         np.int32(self._NScan),
                                         wait_for=wait_for)

    def _operator_lhs(self, out, x, wait_for=None):
        """Compute the left hand side of the CG equation.

        Parameters
        ----------
          out : PyOpenCL.Array
            The result of the computation
          x : PyOpenCL.Array
            The input array
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        self._tmp_result.add_event(self.eval_fwd_kspace_cg(
            self._tmp_result, x, wait_for=self._tmp_result.events+x.events))
        self._tmp_sino.add_event(self._FT(
            self._tmp_sino, self._tmp_result, scan_offset=self._scan_offset))
        return self._operator_rhs(out, self._tmp_sino)

    def _operator_rhs(self, out, x, wait_for=None):
        """Compute the right hand side of the CG equation.

        Parameters
        ----------
          out : PyOpenCL.Array
            The result of the computation
          x : PyOpenCL.Array
            The input array
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        self._tmp_result.add_event(self._FTH(
            self._tmp_result, x, wait_for=wait_for+x.events,
            scan_offset=self._scan_offset))
        return self._prg.operator_ad_cg(self._queue,
                                        (self._NSlice, self._dimY,
                                         self._dimX),
                                        None,
                                        out.data, self._tmp_result.data,
                                        self._coils, np.int32(self._NC),
                                        np.int32(self._NScan),
                                        wait_for=(self._tmp_result.events +
                                                  out.events+wait_for))


class PDBaseSolver:
    """Primal Dual splitting optimization.

    This Class performs a primal-dual variable splitting based reconstruction
    on single precission complex input data.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      irgn_par : dict
        A python dict containing the regularization
        parameters for a given gauss newton step.
      queue : list of PyOpenCL.Queues
        A list of PyOpenCL queues to perform the optimization.
      tau : float
        Estimate of the initial step size based on the
        operator norm of the linear operator.
      fval : float
        Estimate of the initial cost function value to
        scale the displayed values.
      prg : PyOpenCL Program A PyOpenCL Program containing the
        kernels for optimization.
      reg_type : string String to choose between "TV" and "TGV"
        optimization.
      data_operator : PyQMRI Operator The operator to traverse from
        parameter to data space.
      coil : PyOpenCL Buffer or empty list
        coil buffer, empty list if image based fitting is used.
      model : PyQMRI.Model
        Instance of a PyQMRI.Model to perform plotting

    Attributes
    ----------
      delta : float
        Regularization parameter for L2 penalty on linearization point.
      omega : float
        Not used. Should be set to 0
      lambd : float
        Regularization parameter in front of data fidelity term.
      tol : float
        Relative toleraze to stop iterating
      stag : float
        Stagnation detection parameter
      display_iterations : bool
        Switch between plotting (true) of intermediate results
      mu : float
        Strong convecity parameter (inverse of delta).
      tau : float
        Estimated step size based on operator norm of regularization.
      beta_line : float
        Ratio between dual and primal step size
      theta_line : float
        Line search parameter
      unknwons_TGV : int
        Number of T(G)V unknowns
      unknowns_H1 : int
        Number of H1 unknowns (should be 0 for now)
      unknowns : int
        Total number of unknowns (T(G)V+H1)
      num_dev : int
        Total number of compute devices
      dz : float
        Ratio between 3rd dimension and isotropic 1st and 2nd image dimension.
      model : PyQMRI.Model
        The model which should be fitted
      modelgrad : PyOpenCL.Array or numpy.Array
        The partial derivatives evaluated at the linearization point.
        This variable is set in the PyQMRI.irgn Class.
      min_const : list of float
        list of minimal values, one for each unknown
      max_const : list of float
        list of maximal values, one for each unknown
      real_const : list of int
        list if a unknown is constrained to real values only. (1 True, 0 False)
    """

    def __init__(self,
                 par,
                 irgn_par,
                 queue,
                 tau,
                 fval,
                 prg,
                 coil,
                 model,
                 DTYPE=np.complex64,
                 DTYPE_real=np.float32):
        self._DTYPE = DTYPE
        self._DTYPE_real = DTYPE_real
        self.delta = irgn_par["delta"]
        self.omega = irgn_par["omega"]
        self.lambd = irgn_par["lambd"]
        self.tol = irgn_par["tol"]
        self.stag = irgn_par["stag"]
        self.display_iterations = irgn_par["display_iterations"]
        self.mu = 1 / self.delta
        self.tau = tau
        self.beta_line = 1e3  # 1e10#1e12
        self.theta_line = DTYPE_real(1.0)
        self.unknowns_TGV = par["unknowns_TGV"]
        self.unknowns_H1 = par["unknowns_H1"]
        self.unknowns = par["unknowns"]
        self.num_dev = len(par["num_dev"])
        self.dz = par["dz"]
        self._fval_init = fval
        self._prg = prg
        self._queue = queue
        self.model = model
        self._coils = coil
        self.modelgrad = None
        self.min_const = None
        self.max_const = None
        self.real_const = None
        self._kernelsize = (par["par_slices"] + par["overlap"], par["dimY"],
                            par["dimX"])

    @staticmethod
    def factory(
            prg,
            queue,
            par,
            irgn_par,
            init_fval,
            coils,
            linops,
            model,
            reg_type='TGV',
            SMS=False,
            streamed=False,
            imagespace=False,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        """
        Generate a PDSolver object.

        Parameters
        ----------
          prg : PyOpenCL.Program
            A PyOpenCL Program containing the
            kernels for optimization.
          queue : list of PyOpenCL.Queues
            A list of PyOpenCL queues to perform the optimization.
          par : dict
            A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          irgn_par : dict
            A python dict containing the regularization
            parameters for a given gauss newton step.
          init_fval : float
            Estimate of the initial cost function value to
            scale the displayed values.
          coils : PyOpenCL Buffer or empty list
            The coils used for reconstruction.
          linops : list of PyQMRI Operator
            The linear operators used for fitting.
          model : PyQMRI.Model
            The model which should be fitted
          reg_type : string, "TGV"
            String to choose between "TV" and "TGV"
            optimization.
          SMS : bool, false
            Switch between standard (false) and SMS (True) fitting.
          streamed : bool, false
            Switch between streamed (1) and normal (0) reconstruction.
          imagespace : bool, false
            Switch between k-space (false) and imagespace based fitting (true).
          DTYPE : numpy.dtype, numpy.complex64
             Complex working precission.
          DTYPE_real : numpy.dtype, numpy.float32
            Real working precission.
        """
        if reg_type == 'TV':
            if streamed:
                if SMS:
                    pdop = PDSolverStreamedTVSMS(
                        par,
                        irgn_par,
                        queue,
                        DTYPE_real(1 / np.sqrt(8)),
                        init_fval,
                        prg,
                        linops,
                        coils,
                        model,
                        imagespace=imagespace,
                        DTYPE=DTYPE,
                        DTYPE_real=DTYPE_real)
                else:
                    pdop = PDSolverStreamedTV(
                        par,
                        irgn_par,
                        queue,
                        DTYPE_real(1 / np.sqrt(8)),
                        init_fval,
                        prg,
                        linops,
                        coils,
                        model,
                        imagespace=imagespace,
                        DTYPE=DTYPE,
                        DTYPE_real=DTYPE_real)
            else:
                pdop = PDSolverTV(par,
                                  irgn_par,
                                  queue,
                                  DTYPE_real(1 / np.sqrt(8)),
                                  init_fval, prg,
                                  linops,
                                  coils,
                                  model,
                                  DTYPE=DTYPE,
                                  DTYPE_real=DTYPE_real
                                  )

        elif reg_type == 'TGV':
            L = DTYPE_real(0.5 * (18.0 + np.sqrt(33)))
            if streamed:
                if SMS:
                    pdop = PDSolverStreamedTGVSMS(
                        par,
                        irgn_par,
                        queue,
                        DTYPE_real(1 / np.sqrt(L)),
                        init_fval,
                        prg,
                        linops,
                        coils,
                        model,
                        imagespace=imagespace,
                        DTYPE=DTYPE,
                        DTYPE_real=DTYPE_real)
                else:
                    pdop = PDSolverStreamedTGV(
                        par,
                        irgn_par,
                        queue,
                        DTYPE_real(1 / np.sqrt(L)),
                        init_fval,
                        prg,
                        linops,
                        coils,
                        model,
                        imagespace=imagespace,
                        DTYPE=DTYPE,
                        DTYPE_real=DTYPE_real)
            else:
                pdop = PDSolverTGV(
                    par,
                    irgn_par,
                    queue,
                    DTYPE_real(1 / np.sqrt(L)),
                    init_fval,
                    prg,
                    linops,
                    coils,
                    model,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
        else:
            raise NotImplementedError
        return pdop

    def __del__(self):
        """Destructor.

        Releases GPU memory arrays.
        """

    def run(self, inp, data, iters):
        """
        Optimization with 3D T(G)V regularization.

        Parameters
        ----------
          x (numpy.array):
            Initial guess for the unknown parameters
          x (numpy.array):
            The complex valued data to fit.
          iters : int
            Number of primal-dual iterations to run

        Returns
        -------
          tupel:
            A tupel of all primal variables (x,v in the Paper). If no
            streaming is used, the two entries are opf class PyOpenCL.Array,
            otherwise Numpy.Array.
        """
        self._updateConstraints()
        tau = self.tau
        tau_new = self._DTYPE_real(0)

        theta_line = self.theta_line
        beta_line = self.beta_line
        beta_new = self._DTYPE_real(0)
        mu_line = self._DTYPE_real(0.5)
        delta_line = self._DTYPE_real(1)
        ynorm = self._DTYPE_real(0.0)
        lhs = self._DTYPE_real(0.0)
        primal = self._DTYPE_real(0.0)
        primal_new = self._DTYPE_real(0)
        dual = self._DTYPE_real(0.0)
        gap_init = self._DTYPE_real(0.0)
        gap_old = self._DTYPE_real(0.0)
        gap = self._DTYPE_real(0.0)

        (primal_vars,
         primal_vars_new,
         tmp_results_forward,
         tmp_results_forward_new,
         dual_vars,
         dual_vars_new,
         tmp_results_adjoint,
         tmp_results_adjoint_new,
         data) = self._setupVariables(inp, data)

        self._updateInitial(
            out_fwd=tmp_results_forward,
            out_adj=tmp_results_adjoint,
            in_primal=primal_vars,
            in_dual=dual_vars
            )

        for i in range(iters):
            self._updatePrimal(
                out_primal=primal_vars_new,
                out_fwd=tmp_results_forward_new,
                in_primal=primal_vars,
                in_precomp_adj=tmp_results_adjoint,
                tau=tau
                )

            beta_new = beta_line * (1 + self.mu * tau)
            tau_new = tau * np.sqrt(beta_line / beta_new*(1 + theta_line))
            beta_line = beta_new

            while True:
                theta_line = tau_new / tau
                lhs, ynorm = self._updateDual(
                    out_dual=dual_vars_new,
                    out_adj=tmp_results_adjoint_new,
                    in_primal=primal_vars,
                    in_primal_new=primal_vars_new,
                    in_dual=dual_vars,
                    in_precomp_fwd=tmp_results_forward,
                    in_precomp_fwd_new=tmp_results_forward_new,
                    in_precomp_adj=tmp_results_adjoint,
                    data=data,
                    beta=beta_line,
                    tau=tau_new,
                    theta=theta_line
                    )
                if lhs <= ynorm * delta_line:
                    break
                tau_new = tau_new * mu_line

            tau = tau_new
            for j, k in zip(primal_vars_new,
                            tmp_results_adjoint_new):
                (primal_vars[j],
                 primal_vars_new[j],
                 tmp_results_adjoint[k],
                 tmp_results_adjoint_new[k]) = (
                         primal_vars_new[j],
                         primal_vars[j],
                         tmp_results_adjoint_new[k],
                         tmp_results_adjoint[k],
                         )

            for j, k in zip(dual_vars_new,
                            tmp_results_forward_new):
                (dual_vars[j],
                 dual_vars_new[j],
                 tmp_results_forward[k],
                 tmp_results_forward_new[k]) = (
                         dual_vars_new[j],
                         dual_vars[j],
                         tmp_results_forward_new[k],
                         tmp_results_forward[k]
                         )

            if not np.mod(i, 10):
                if self.display_iterations:
                    if isinstance(primal_vars["x"], np.ndarray):
                        self.model.plot_unknowns(
                            np.swapaxes(primal_vars["x"], 0, 1))
                    else:
                        self.model.plot_unknowns(primal_vars["x"].get())
                primal_new, dual, gap = self._calcResidual(
                    in_primal=primal_vars,
                    in_dual=dual_vars,
                    in_precomp_fwd=tmp_results_forward,
                    in_precomp_adj=tmp_results_adjoint,
                    data=data)

                if i == 0:
                    gap_init = gap
                if np.abs(primal - primal_new)/self._fval_init <\
                   self.tol:
                    print("Terminated at iteration %d because the energy "
                          "decrease in the primal problem was less than %.3e" %
                          (i,
                           np.abs(primal - primal_new)/self._fval_init))
                    return primal_vars_new
                if gap > gap_old * self.stag and i > 1:
                    print("Terminated at iteration %d "
                          "because the method stagnated" % (i))
                    return primal_vars_new
                if np.abs((gap - gap_old) / gap_init) < self.tol:
                    print("Terminated at iteration %d because the "
                          "relative energy decrease of the PD gap was "
                          "less than %.3e"
                          % (i, np.abs((gap - gap_old) / gap_init)))
                    return primal_vars_new
                primal = primal_new
                gap_old = gap
                sys.stdout.write(
                    "Iteration: %04d ---- Primal: %2.2e, "
                    "Dual: %2.2e, Gap: %2.2e, Beta: %2.2e \r" %
                    (i, 1000*primal / self._fval_init,
                     1000*dual / self._fval_init,
                     1000*gap / self._fval_init,
                     beta_line))
                sys.stdout.flush()

        return primal_vars

    def _updateInitial(
            self,
            out_fwd,
            out_adj,
            in_primal,
            in_dual):
        pass

    def _updatePrimal(
            self,
            out_primal,
            out_fwd,
            in_primal,
            in_precomp_adj,
            tau):
        pass

    def _updateDual(self,
                    out_dual,
                    out_adj,
                    in_primal,
                    in_primal_new,
                    in_dual,
                    in_precomp_fwd,
                    in_precomp_fwd_new,
                    in_precomp_adj,
                    data,
                    beta,
                    tau,
                    theta
                    ):
        return ({}, {})

    def _calcResidual(
                    self,
                    in_primal,
                    in_dual,
                    in_precomp_fwd,
                    in_precomp_adj,
                    data):
        return ({}, {}, {})

    def _setupVariables(self, inp, data):
        return ({}, {}, {}, {}, {}, {}, {}, {}, {})

    def _updateConstraints(self):
        num_const = (len(self.model.constraints))
        min_const = np.zeros((num_const), dtype=self._DTYPE_real)
        max_const = np.zeros((num_const), dtype=self._DTYPE_real)
        real_const = np.zeros((num_const), dtype=np.int32)
        for j in range(num_const):
            min_const[j] = self._DTYPE_real(self.model.constraints[j].min)
            max_const[j] = self._DTYPE_real(self.model.constraints[j].max)
            real_const[j] = np.int32(self.model.constraints[j].real)

        self.min_const = []
        self.max_const = []
        self.real_const = []
        for j in range(self.num_dev):
            self.min_const.append(
                clarray.to_device(self._queue[4*j], min_const))
            self.max_const.append(
                clarray.to_device(self._queue[4*j], max_const))
            self.real_const.append(
                clarray.to_device(self._queue[4*j], real_const))

    def updateRegPar(self, irgn_par):
        """Update the regularization parameters.

          Performs an update of the regularization parameters as these usually
          vary from one to another Gauss-Newton step.

        Parameters
        ----------
          irgn_par (dic): A dictionary containing the new parameters.
        """
        self.alpha = irgn_par["gamma"]
        self.beta = irgn_par["gamma"] * 2
        self.delta = irgn_par["delta"]
        self.omega = irgn_par["omega"]
        self.lambd = irgn_par["lambd"]
        self.mu = 1/self.delta

    def update_primal(self, outp, inp, par, idx=0, idxq=0,
                      bound_cond=0, wait_for=None):
        """Primal update of the x variable in the Primal-Dual Algorithm.

        Parameters
        ----------
          outp : PyOpenCL.Array
            The result of the update step
          inp : PyOpenCL.Array
            The previous values of x
          par : list
            List of necessary parameters for the update
          idx : int
            Index of the device to use
          idxq : int
            Index of the queue to use
          bound_cond : int
            Apply boundary condition (1) or not (0).
          wait_for : list of PyOpenCL.Events, None
            A optional list for PyOpenCL.Events to wait for

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        return self._prg[idx].update_primal_LM(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, inp[1].data, inp[2].data, inp[3].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[0]/par[1]),
            self.min_const[idx].data, self.max_const[idx].data,
            self.real_const[idx].data, np.int32(self.unknowns),
            wait_for=(outp.events +
                      inp[0].events+inp[1].events +
                      inp[2].events+wait_for))

    def update_v(self, outp, inp, par=None, idx=0, idxq=0,
                 bound_cond=0, wait_for=None):
        """Primal update of the v variable in Primal-Dual Algorithm.

        Parameters
        ----------
          outp : PyOpenCL.Array
            The result of the update step
          inp : PyOpenCL.Array
            The previous values of x
          par : list
            List of necessary parameters for the update
          idx : int
            Index of the device to use
          idxq : int
            Index of the queue to use
          bound_cond : int
            Apply boundary condition (1) or not (0).
          wait_for : list of PyOpenCL.Events, None
            A optional list for PyOpenCL.Events to wait for

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        return self._prg[idx].update_v(
            self._queue[4*idx+idxq], (outp[..., 0].size,), None,
            outp.data, inp[0].data, inp[1].data, self._DTYPE_real(par[0]),
            wait_for=outp.events+inp[0].events+inp[1].events+wait_for)

    def update_z1(self, outp, inp, par=None, idx=0, idxq=0,
                  bound_cond=0, wait_for=None):
        """Dual update of the z1 variable in Primal-Dual Algorithm for TGV.

        Parameters
        ----------
          outp : PyOpenCL.Array
            The result of the update step
          inp : PyOpenCL.Array
            The previous values of x
          par : list
            List of necessary parameters for the update
          idx : int
            Index of the device to use
          idxq : int
            Index of the queue to use
          bound_cond : int
            Apply boundary condition (1) or not (0).
          wait_for : list of PyOpenCL.Events, None
            A optional list for PyOpenCL.Events to wait for

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []

        return self._prg[idx].update_z1(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, inp[1].data,
            inp[2].data, inp[3].data, inp[4].data,
            self._DTYPE_real(par[0]), self._DTYPE_real(par[1]),
            self._DTYPE_real(1/par[2]), np.int32(self.unknowns_TGV),
            np.int32(self.unknowns_H1),
            self._DTYPE_real(1 / (1 + par[0] / par[3])),
            wait_for=(outp.events+inp[0].events+inp[1].events +
                      inp[2].events+inp[3].events+inp[4].events+wait_for))

    def update_z1_tv(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=None):
        """Dual update of the z1 variable in Primal-Dual Algorithm for TV.

        Parameters
        ----------
          outp : PyOpenCL.Array
            The result of the update step
          inp : PyOpenCL.Array
            The previous values of x
          par : list
            List of necessary parameters for the update
          idx : int
            Index of the device to use
          idxq : int
            Index of the queue to use
          bound_cond : int
            Apply boundary condition (1) or not (0).
          wait_for : list of PyOpenCL.Events, None
            A optional list for PyOpenCL.Events to wait for

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        return self._prg[idx].update_z1_tv(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, inp[1].data, inp[2].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[1]),
            self._DTYPE_real(1/par[2]), np.int32(self.unknowns_TGV),
            np.int32(self.unknowns_H1),
            self._DTYPE_real(1 / (1 + par[0] / par[3])),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def update_z2(self, outp, inp, par=None, idx=0, idxq=0,
                  bound_cond=0, wait_for=None):
        """Dual update of the z2 variable in Primal-Dual Algorithm for TGV.

        Parameters
        ----------
          outp : PyOpenCL.Array
            The result of the update step
          inp : PyOpenCL.Array
            The previous values of x
          par : list
            List of necessary parameters for the update
          idx : int
            Index of the device to use
          idxq : int
            Index of the queue to use
          bound_cond : int
            Apply boundary condition (1) or not (0).
          wait_for : list of PyOpenCL.Events, None
            A optional list for PyOpenCL.Events to wait for

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        return self._prg[idx].update_z2(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, inp[1].data, inp[2].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[1]),
            self._DTYPE_real(1/par[2]), np.int32(self.unknowns),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def update_Kyk2(self,
                    outp,
                    inp,
                    par=None,
                    idx=0,
                    idxq=0,
                    bound_cond=0,
                    wait_for=None
                    ):
        """Precompute the v-part of the Adjoint Linear operator.

        Parameters
        ----------
          outp : PyOpenCL.Array
            The result of the update step
          inp : PyOpenCL.Array
            The previous values of x
          par : list
            List of necessary parameters for the update
          idx : int
            Index of the device to use
          idxq : int
            Index of the queue to use
          bound_cond : int
            Apply boundary condition (1) or not (0).
          wait_for : list of PyOpenCL.Events, None
            A optional list for PyOpenCL.Events to wait for

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        return self._prg[idx].update_Kyk2(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, inp[1].data,
            np.int32(self.unknowns),
            par[idx].data,
            np.int32(bound_cond),
            self._DTYPE_real(self.dz),
            wait_for=outp.events + inp[0].events + inp[1].events+wait_for)

    def update_r(self, outp, inp, par=None, idx=0, idxq=0,
                 bound_cond=0, wait_for=None):
        """Update the data dual variable r.

        Parameters
        ----------
          outp : PyOpenCL.Array
            The result of the update step
          inp : PyOpenCL.Array
            The previous values of x
          par : list
            List of necessary parameters for the update
          idx : int
            Index of the device to use
          idxq : int
            Index of the queue to use
          bound_cond : int
            Apply boundary condition (1) or not (0).
          wait_for : list of PyOpenCL.Events, None
            A optional list for PyOpenCL.Events to wait for

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        return self._prg[idx].update_r(
            self._queue[4*idx+idxq], (outp.size,), None,
            outp.data, inp[0].data,
            inp[1].data, inp[2].data, inp[3].data,
            self._DTYPE_real(par[0]), self._DTYPE_real(par[1]),
            self._DTYPE_real(1/(1+par[0]/par[2])),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def setFvalInit(self, fval):
        """Set the initial value of the cost function.

        Parameters
        ----------
          fval : float
            The initial cost of the optimization problem
        """
        self._fval_init = fval


class PDSolverTV(PDBaseSolver):
    """Primal Dual splitting optimization for TV.

    This Class performs a primal-dual variable splitting based reconstruction
    on single precission complex input data.

    Parameters
    ----------
        par : dict
          A python dict containing the necessary information to
          setup the object. Needs to contain the number of slices (NSlice),
          number of scans (NScan), image dimensions (dimX, dimY), number of
          coils (NC), sampling points (N) and read outs (NProj)
          a PyOpenCL queue (queue) and the complex coil
          sensitivities (C).
        irgn_par : dict
          A python dict containing the regularization
          parameters for a given gauss newton step.
        queue : list of PyOpenCL.Queues
          A list of PyOpenCL queues to perform the optimization.
        tau : float
          Estimated step size based on operator norm of regularization.
        fval : float
          Estimate of the initial cost function value to
          scale the displayed values.
        prg : PyOpenCL.Program
          A PyOpenCL Program containing the
          kernels for optimization.
        linops : list of PyQMRI Operator
          The linear operators used for fitting.
        coils : PyOpenCL Buffer or empty list
          The coils used for reconstruction.
        model : PyQMRI.Model
          The model which should be fitted

    Attributes
    ----------
      alpha : float
        TV regularization weight
    """

    def __init__(self,
                 par,
                 irgn_par,
                 queue,
                 tau,
                 fval,
                 prg,
                 linop,
                 coils,
                 model,
                 **kwargs
                 ):
        super().__init__(
            par,
            irgn_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            model,
            **kwargs)
        self.alpha = irgn_par["gamma"]
        self._op = linop[0]
        self._grad_op = linop[1]

    def _setupVariables(self, inp, data):

        data = clarray.to_device(self._queue[0], data.astype(self._DTYPE))

        primal_vars = {}
        primal_vars_new = {}
        tmp_results_adjoint = {}
        tmp_results_adjoint_new = {}

        primal_vars["x"] = clarray.to_device(self._queue[0], inp)
        primal_vars["xk"] = primal_vars["x"].copy()
        primal_vars_new["x"] = clarray.empty_like(primal_vars["x"])

        tmp_results_adjoint["Kyk1"] = clarray.empty_like(primal_vars["x"])
        tmp_results_adjoint_new["Kyk1"] = clarray.empty_like(primal_vars["x"])

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}
        tmp_results_forward_new = {}
        dual_vars["r"] = clarray.zeros(
            self._queue[0],
            data.shape,
            dtype=self._DTYPE)
        dual_vars_new["r"] = clarray.empty_like(dual_vars["r"])

        dual_vars["z1"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape+(4,),
                                        dtype=self._DTYPE)
        dual_vars_new["z1"] = clarray.empty_like(dual_vars["z1"])

        tmp_results_forward["gradx"] = clarray.empty_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx"] = clarray.empty_like(
            dual_vars["z1"])
        tmp_results_forward["Ax"] = clarray.empty_like(data)
        tmp_results_forward_new["Ax"] = clarray.empty_like(data)

        return (primal_vars,
                primal_vars_new,
                tmp_results_forward,
                tmp_results_forward_new,
                dual_vars,
                dual_vars_new,
                tmp_results_adjoint,
                tmp_results_adjoint_new,
                data)

    def _updateInitial(self,
                       out_fwd, out_adj,
                       in_primal, in_dual):
        out_adj["Kyk1"].add_event(
            self._op.adjKyk1(out_adj["Kyk1"],
                             [in_dual["r"], in_dual["z1"],
                              self._coils,
                              self.modelgrad,
                              self._grad_op.ratio]))

        out_fwd["Ax"].add_event(self._op.fwd(
            out_fwd["Ax"], [in_primal["x"], self._coils, self.modelgrad]))
        out_fwd["gradx"].add_event(
            self._grad_op.fwd(out_fwd["gradx"], in_primal["x"]))

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        out_primal["x"].add_event(self.update_primal(
            outp=out_primal["x"],
            inp=(in_primal["x"], in_precomp_adj["Kyk1"],
                 in_primal["xk"], self.jacobi),
            par=(tau, self.delta)))

        out_fwd["gradx"].add_event(
            self._grad_op.fwd(
                out_fwd["gradx"], out_primal["x"]))

        out_fwd["Ax"].add_event(
            self._op.fwd(out_fwd["Ax"],
                         [out_primal["x"],
                          self._coils,
                          self.modelgrad]))

    def _updateDual(self,
                    out_dual, out_adj,
                    in_primal,
                    in_primal_new,
                    in_dual,
                    in_precomp_fwd,
                    in_precomp_fwd_new,
                    in_precomp_adj,
                    data,
                    beta,
                    tau,
                    theta):
        out_dual["z1"].add_event(
            self.update_z1_tv(
                outp=out_dual["z1"],
                inp=(
                        in_dual["z1"],
                        in_precomp_fwd_new["gradx"],
                        in_precomp_fwd["gradx"],
                    ),
                par=(beta*tau, theta, self.alpha, self.omega)
                )
            )

        out_dual["r"].add_event(
            self.update_r(
                outp=out_dual["r"],
                inp=(
                        in_dual["r"],
                        in_precomp_fwd_new["Ax"],
                        in_precomp_fwd["Ax"],
                        data
                     ),
                par=(beta*tau, theta, self.lambd)
                )
            )

        out_adj["Kyk1"].add_event(
            self._op.adjKyk1(
                out_adj["Kyk1"],
                [out_dual["r"], out_dual["z1"],
                 self._coils,
                 self.modelgrad,
                 self._grad_op.ratio]))

        ynorm = (
            (
                clarray.vdot(
                    out_dual["r"] - in_dual["r"],
                    out_dual["r"] - in_dual["r"]
                    )
                + clarray.vdot(
                    out_dual["z1"] - in_dual["z1"],
                    out_dual["z1"] - in_dual["z1"]
                    )
            )**(1 / 2)).real
        lhs = np.sqrt(beta) * tau * (
            (
                clarray.vdot(
                    out_adj["Kyk1"] - in_precomp_adj["Kyk1"],
                    out_adj["Kyk1"] - in_precomp_adj["Kyk1"]
                    )
            )**(1 / 2)).real
        return lhs.get(), ynorm.get()

    def _calcResidual(
            self,
            in_primal,
            in_dual,
            in_precomp_fwd,
            in_precomp_adj,
            data):

        primal_new = (
            self.lambd / 2 * clarray.vdot(
                in_precomp_fwd["Ax"] - data,
                in_precomp_fwd["Ax"] - data)
            + self.alpha * clarray.sum(
                abs(in_precomp_fwd["gradx"])
                )
            + 1 / (2 * self.delta) * clarray.vdot(
                (in_primal["x"] - in_primal["xk"])*self.jacobi,
                in_primal["x"] - in_primal["xk"]
                )
            ).real

        dual = (
            -self.delta / 2 * clarray.vdot(
                - in_precomp_adj["Kyk1"],
                - in_precomp_adj["Kyk1"])
            - clarray.vdot(
                in_primal["xk"],
                - in_precomp_adj["Kyk1"]
                )
            - 1 / (2 * self.lambd) * clarray.vdot(
                in_dual["r"],
                in_dual["r"])
            - clarray.vdot(data, in_dual["r"])
            ).real

        if self.unknowns_H1 > 0:
            primal_new += (
                 self.omega / 2 * clarray.vdot(
                     in_precomp_fwd["gradx"][self.unknowns_TGV:],
                     in_precomp_fwd["gradx"][self.unknowns_TGV:]
                     )
                 ).real

            dual += (
                - 1 / (2 * self.omega) * clarray.vdot(
                    in_dual["z1"][self.unknowns_TGV:],
                    in_dual["z1"][self.unknowns_TGV:]
                    )
                ).real
        gap = np.abs(primal_new - dual)
        return primal_new.get(), dual.get(), gap.get()


class PDSolverTGV(PDBaseSolver):
    """TGV Primal Dual splitting optimization.

    This Class performs a primal-dual variable splitting based reconstruction
    on single precission complex input data.

    Parameters
    ----------
        par : dict
          A python dict containing the necessary information to
          setup the object. Needs to contain the number of slices (NSlice),
          number of scans (NScan), image dimensions (dimX, dimY), number of
          coils (NC), sampling points (N) and read outs (NProj)
          a PyOpenCL queue (queue) and the complex coil
          sensitivities (C).
        irgn_par : dict
          A python dict containing the regularization
          parameters for a given gauss newton step.
        queue : list of PyOpenCL.Queues
          A list of PyOpenCL queues to perform the optimization.
        tau : float
          Estimated step size based on operator norm of regularization.
        fval : float
          Estimate of the initial cost function value to
          scale the displayed values.
        prg : PyOpenCL.Program
          A PyOpenCL Program containing the
          kernels for optimization.
        linops : list of PyQMRI Operator
          The linear operators used for fitting.
        coils : PyOpenCL Buffer or empty list
          The coils used for reconstruction.
        model : PyQMRI.Model
          The model which should be fitted

    Attributes
    ----------
      alpha : float
        alpha0 parameter for TGV regularization weight
      beta : float
        alpha1 parameter for TGV regularization weight
    """

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 linop, coils, model, **kwargs):
        super().__init__(
            par,
            irgn_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            model,
            **kwargs)
        self.alpha = irgn_par["gamma"]
        self.beta = irgn_par["gamma"] * 2
        self._op = linop[0]
        self._grad_op = linop[1]
        self._symgrad_op = linop[2]

    def _setupVariables(self, inp, data):

        data = clarray.to_device(self._queue[0], data.astype(self._DTYPE))

        primal_vars = {}
        primal_vars_new = {}
        tmp_results_adjoint = {}
        tmp_results_adjoint_new = {}

        primal_vars["x"] = clarray.to_device(self._queue[0], inp)
        primal_vars["xk"] = primal_vars["x"].copy()
        primal_vars_new["x"] = clarray.empty_like(primal_vars["x"])
        primal_vars["v"] = clarray.zeros(self._queue[0],
                                         primal_vars["x"].shape+(4,),
                                         dtype=self._DTYPE)
        primal_vars_new["v"] = clarray.empty_like(primal_vars["v"])

        tmp_results_adjoint["Kyk1"] = clarray.empty_like(primal_vars["x"])
        tmp_results_adjoint_new["Kyk1"] = clarray.empty_like(primal_vars["x"])
        tmp_results_adjoint["Kyk2"] = clarray.empty_like(primal_vars["v"])
        tmp_results_adjoint_new["Kyk2"] = clarray.empty_like(primal_vars["v"])

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}
        tmp_results_forward_new = {}
        dual_vars["r"] = clarray.zeros(
            self._queue[0],
            data.shape,
            dtype=self._DTYPE)
        dual_vars_new["r"] = clarray.empty_like(dual_vars["r"])

        dual_vars["z1"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape+(4,),
                                        dtype=self._DTYPE)
        dual_vars_new["z1"] = clarray.empty_like(dual_vars["z1"])
        dual_vars["z2"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape+(8,),
                                        dtype=self._DTYPE)
        dual_vars_new["z2"] = clarray.empty_like(dual_vars["z2"])

        tmp_results_forward["gradx"] = clarray.empty_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx"] = clarray.empty_like(
            dual_vars["z1"])
        tmp_results_forward["symgradx"] = clarray.empty_like(
            dual_vars["z2"])
        tmp_results_forward_new["symgradx"] = clarray.empty_like(
            dual_vars["z2"])
        tmp_results_forward["Ax"] = clarray.empty_like(data)
        tmp_results_forward_new["Ax"] = clarray.empty_like(data)

        return (primal_vars,
                primal_vars_new,
                tmp_results_forward,
                tmp_results_forward_new,
                dual_vars,
                dual_vars_new,
                tmp_results_adjoint,
                tmp_results_adjoint_new,
                data)

    def _updateInitial(self,
                       out_fwd, out_adj,
                       in_primal, in_dual):
        out_adj["Kyk1"].add_event(
            self._op.adjKyk1(out_adj["Kyk1"],
                             [in_dual["r"], in_dual["z1"],
                              self._coils,
                              self.modelgrad,
                              self._grad_op.ratio]))

        out_adj["Kyk2"].add_event(
            self.update_Kyk2(
                outp=out_adj["Kyk2"],
                inp=(in_dual["z2"], in_dual["z1"]),
                par=[self._symgrad_op.ratio]))

        out_fwd["Ax"].add_event(self._op.fwd(
            out_fwd["Ax"], [in_primal["x"], self._coils, self.modelgrad]))
        out_fwd["gradx"].add_event(
            self._grad_op.fwd(out_fwd["gradx"], in_primal["x"]))
        out_fwd["symgradx"].add_event(
            self._symgrad_op.fwd(out_fwd["symgradx"], in_primal["v"]))

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        out_primal["x"].add_event(self.update_primal(
            outp=out_primal["x"],
            inp=(in_primal["x"], in_precomp_adj["Kyk1"],
                 in_primal["xk"], self.jacobi),
            par=(tau, self.delta)))

        out_primal["v"].add_event(self.update_v(
            outp=out_primal["v"],
            inp=(in_primal["v"], in_precomp_adj["Kyk2"]),
            par=(tau,)))

        out_fwd["gradx"].add_event(
            self._grad_op.fwd(
                out_fwd["gradx"], out_primal["x"]))

        out_fwd["symgradx"].add_event(
            self._symgrad_op.fwd(
                out_fwd["symgradx"], out_primal["v"]))
        out_fwd["Ax"].add_event(
            self._op.fwd(out_fwd["Ax"],
                         [out_primal["x"],
                          self._coils,
                          self.modelgrad]))

    def _updateDual(self,
                    out_dual, out_adj,
                    in_primal,
                    in_primal_new,
                    in_dual,
                    in_precomp_fwd,
                    in_precomp_fwd_new,
                    in_precomp_adj,
                    data,
                    beta,
                    tau,
                    theta):
        out_dual["z1"].add_event(
            self.update_z1(
                outp=out_dual["z1"],
                inp=(
                        in_dual["z1"],
                        in_precomp_fwd_new["gradx"],
                        in_precomp_fwd["gradx"],
                        in_primal_new["v"],
                        in_primal["v"]
                    ),
                par=(beta*tau, theta, self.alpha, self.omega)
                )
            )
        out_dual["z2"].add_event(
            self.update_z2(
                outp=out_dual["z2"],
                inp=(
                        in_dual["z2"],
                        in_precomp_fwd_new["symgradx"],
                        in_precomp_fwd["symgradx"]
                    ),
                par=(beta*tau, theta, self.beta)))
        out_dual["r"].add_event(
            self.update_r(
                outp=out_dual["r"],
                inp=(
                        in_dual["r"],
                        in_precomp_fwd_new["Ax"],
                        in_precomp_fwd["Ax"],
                        data
                     ),
                par=(beta*tau, theta, self.lambd)
                )
            )

        out_adj["Kyk1"].add_event(
            self._op.adjKyk1(
                out_adj["Kyk1"],
                [out_dual["r"], out_dual["z1"],
                 self._coils,
                 self.modelgrad,
                 self._grad_op.ratio]))
        out_adj["Kyk2"].add_event(
            self.update_Kyk2(
                outp=out_adj["Kyk2"],
                inp=(out_dual["z2"], out_dual["z1"]),
                par=[self._symgrad_op.ratio]))

        ynorm = (
            (
                clarray.vdot(
                    out_dual["r"] - in_dual["r"],
                    out_dual["r"] - in_dual["r"]
                    )
                + clarray.vdot(
                    out_dual["z1"] - in_dual["z1"],
                    out_dual["z1"] - in_dual["z1"]
                    )
                + clarray.vdot(
                    out_dual["z2"] - in_dual["z2"],
                    out_dual["z2"] - in_dual["z2"]
                    )
            )**(1 / 2)).real
        lhs = np.sqrt(beta) * tau * (
            (
                clarray.vdot(
                    out_adj["Kyk1"] - in_precomp_adj["Kyk1"],
                    out_adj["Kyk1"] - in_precomp_adj["Kyk1"]
                    )
                + clarray.vdot(
                    out_adj["Kyk2"] - in_precomp_adj["Kyk2"],
                    out_adj["Kyk2"] - in_precomp_adj["Kyk2"]
                    )
            )**(1 / 2)).real
        return lhs.get(), ynorm.get()

    def _calcResidual(
            self,
            in_primal,
            in_dual,
            in_precomp_fwd,
            in_precomp_adj,
            data):

        primal_new = (
            self.lambd / 2 * clarray.vdot(
                in_precomp_fwd["Ax"] - data,
                in_precomp_fwd["Ax"] - data)
            + self.alpha * clarray.sum(
                abs(in_precomp_fwd["gradx"] - in_primal["v"])
                )
            + self.beta * clarray.sum(
                abs(in_precomp_fwd["symgradx"])
                )
            + 1 / (2 * self.delta) * clarray.vdot(
                (in_primal["x"] - in_primal["xk"])*self.jacobi,
                in_primal["x"] - in_primal["xk"]
                )
            ).real

        dual = (
            -self.delta / 2 * clarray.vdot(
                - in_precomp_adj["Kyk1"]*self.jacobi,
                - in_precomp_adj["Kyk1"])
            - clarray.vdot(
                in_primal["xk"],
                - in_precomp_adj["Kyk1"]
                )
            + clarray.sum(in_precomp_adj["Kyk2"])
            - 1 / (2 * self.lambd) * clarray.vdot(
                in_dual["r"],
                in_dual["r"])
            - clarray.vdot(data, in_dual["r"])
            ).real

        if self.unknowns_H1 > 0:
            primal_new += (
                 self.omega / 2 * clarray.vdot(
                     in_precomp_fwd["gradx"][self.unknowns_TGV:],
                     in_precomp_fwd["gradx"][self.unknowns_TGV:]
                     )
                 ).real

            dual += (
                - 1 / (2 * self.omega) * clarray.vdot(
                    in_dual["z1"][self.unknowns_TGV:],
                    in_dual["z1"][self.unknowns_TGV:]
                    )
                ).real
        gap = np.abs(primal_new - dual)
        return primal_new.get(), dual.get(), gap.get()


class PDSolverStreamed(PDBaseSolver):
    """Streamed version of the PD Solver.

    This class is the base class for the streamed array optimization.

    Parameters
    ----------
        par : dict
          A python dict containing the necessary information to
          setup the object. Needs to contain the number of slices (NSlice),
          number of scans (NScan), image dimensions (dimX, dimY), number of
          coils (NC), sampling points (N) and read outs (NProj)
          a PyOpenCL queue (queue) and the complex coil
          sensitivities (C).
        irgn_par : dict
          A python dict containing the regularization
          parameters for a given gauss newton step.
        queue : list of PyOpenCL.Queues
          A list of PyOpenCL queues to perform the optimization.
        tau : float
          Estimated step size based on operator norm of regularization.
        fval : float
          Estimate of the initial cost function value to
          scale the displayed values.
        prg : PyOpenCL.Program
          A PyOpenCL Program containing the
          kernels for optimization.
        coils : PyOpenCL Buffer or empty list
          The coils used for reconstruction.
        model : PyQMRI.Model
          The model which should be fitted
        imagespace : bool, false
          Switch between imagespace (True) and k-space (false) based
          fitting.

    Attributes
    ----------
      unknown_shape : tuple of int
        Size of the unknown array
      model_deriv_shape : tuple of int
        Size of the partial derivative array of the unknowns
      grad_shape : tuple of int
        Size of the finite difference based gradient
      symgrad_shape : tuple of int, None
        Size of the finite difference based symmetrized gradient. Defaults
        to None in TV based optimization.
      data_shape : tuple of int
        Size of the data to be fitted
      data_trans_axes : list of int
        Order of transpose of data axis, requried for streaming
      data_shape_T : tuple of int
        Size of transposed data.
    """

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 coils, model, imagespace=False, **kwargs):
        super().__init__(
            par,
            irgn_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            model,
            **kwargs)

        self._op = None
        self.symgrad_shape = None
        self._packs = None
        self._numofpacks = None

        self.unknown_shape = (par["NSlice"], par["unknowns"],
                              par["dimY"], par["dimX"])
        self.model_deriv_shape = (par["NSlice"], par["unknowns"],
                                  par["NScan"],
                                  par["dimY"], par["dimX"])

        self.grad_shape = self.unknown_shape + (4,)

        self._NSlice = par["NSlice"]
        self._par_slices = par["par_slices"]
        self._overlap = par["overlap"]

        self._symgrad_op = None
        self._grad_op = None

        if imagespace:
            self.data_shape = (par["NSlice"], par["NScan"],
                               par["dimY"], par["dimX"])
            self.dat_trans_axes = [1, 0, 2, 3]
        else:
            self.dat_trans_axes = [2, 0, 1, 3, 4]
            self.data_shape = (par["NSlice"], par["NScan"],
                               par["NC"], par["Nproj"], par["N"])
            self.data_shape_T = self.data_shape
            self._expdim_dat = 2
            self._expdim_C = 1

    def _setup_reg_tmp_arrays(self, reg_type, SMS=False):
        if reg_type == 'TV':
            pass
        elif reg_type == 'TGV':
            self.v = np.zeros(
                self.grad_shape,
                dtype=self._DTYPE)
            self.z2 = np.zeros(
                self.symgrad_shape,
                dtype=self._DTYPE)
        else:
            raise NotImplementedError("Not implemented")
        self._setupstreamingops(reg_type, SMS=SMS)

        self.r = np.zeros(
                self.data_shape,
                dtype=self._DTYPE)
        self.z1 = np.zeros(
            self.grad_shape,
            dtype=self._DTYPE)

    def _setupstreamingops(self, reg_type, SMS=False):
        if not SMS:
            self.stream_initial_1 = self._defineoperator(
                [],
                [],
                [[]],
                reverse_dir=True)
            self.stream_initial_1 += self._op.fwdstr
            self.stream_initial_1 += self._op.adjstrKyk1
            if reg_type == 'TGV':
                self.stream_initial_1 += \
                    self._symgrad_op.getStreamedSymGradientObject()

        if reg_type == 'TGV':
            self.stream_Kyk2 = self._defineoperator(
                [self.update_Kyk2],
                [self.grad_shape],
                [[self.symgrad_shape,
                  self.grad_shape,
                  self.grad_shape]])

            self.stream_initial_2 = self._defineoperator(
                [],
                [],
                [[]])

            self.stream_initial_2 += self._grad_op.getStreamedGradientObject()
            self.stream_initial_2 += self.stream_Kyk2

        self.stream_primal = self._defineoperator(
            [self.update_primal],
            [self.unknown_shape],
            [[self.unknown_shape,
              self.unknown_shape,
              self.unknown_shape,
              self.unknown_shape]])

        self.update_primal_1 = self._defineoperator(
            [],
            [],
            [[]])

        self.update_primal_1 += self.stream_primal
        self.update_primal_1 += self._grad_op.getStreamedGradientObject()
        self.update_primal_1.connectouttoin(0, (1, 0))

        if not SMS:
            self.update_primal_1 += self._op.fwdstr
            self.update_primal_1.connectouttoin(0, (2, 0))

        if reg_type == 'TGV':
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
            self.update_primal_2 += \
                self._symgrad_op.getStreamedSymGradientObject()
            self.update_primal_2.connectouttoin(0, (1, 0))
            self.stream_update_z1 = self._defineoperator(
                [self.update_z1],
                [self.grad_shape],
                [[self.grad_shape,
                  self.grad_shape,
                  self.grad_shape,
                  self.grad_shape,
                  self.grad_shape]],
                reverse_dir=True,
                posofnorm=[False])
        else:
            self.stream_update_z1 = self._defineoperator(
                [self.update_z1_tv],
                [self.grad_shape],
                [[self.grad_shape,
                  self. grad_shape,
                  self. grad_shape]],
                reverse_dir=True,
                posofnorm=[False])
        if not SMS:
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
            self.update_dual_1 += self._op.adjstrKyk1
            self.update_dual_1.connectouttoin(0, (2, 1))
            self.update_dual_1.connectouttoin(1, (2, 0))
            del self.stream_update_z1, self.stream_update_r, self.stream_primal

        else:
            self.stream_update_r = self._defineoperator(
                [self.update_r],
                [self.data_shape],
                [[self.data_shape,
                  self.data_shape,
                  self.data_shape,
                  self.data_shape]],
                slices=self._packs*self._numofpacks,
                reverse_dir=True,
                posofnorm=[False])
            del self.stream_primal

        if reg_type == 'TGV':
            self.stream_update_z2 = self._defineoperator(
                [self.update_z2],
                [self.symgrad_shape],
                [[self.symgrad_shape,
                  self.symgrad_shape,
                  self.symgrad_shape]])
            self.update_dual_2 = self._defineoperator(
                [],
                [],
                [[]],
                posofnorm=[False, True])

            self.update_dual_2 += self.stream_update_z2
            self.update_dual_2 += self.stream_Kyk2
            self.update_dual_2.connectouttoin(0, (1, 0))
            del self.stream_Kyk2, self.stream_update_z2, self.stream_update_v

    def _defineoperator(self,
                        functions,
                        outp,
                        inp,
                        reverse_dir=False,
                        posofnorm=None,
                        slices=None):
        if slices is None:
            slices = self._NSlice
        return streaming.Stream(
            functions,
            outp,
            inp,
            self._par_slices,
            self._overlap,
            slices,
            self._queue,
            self.num_dev,
            reverse_dir,
            posofnorm,
            DTYPE=self._DTYPE)


class PDSolverStreamedTGV(PDSolverStreamed):
    """Streamed TGV optimization.

    This class performes streamd TGV optimization.

    Parameters
    ----------
        par : dict
          A python dict containing the necessary information to
          setup the object. Needs to contain the number of slices (NSlice),
          number of scans (NScan), image dimensions (dimX, dimY), number of
          coils (NC), sampling points (N) and read outs (NProj)
          a PyOpenCL queue (queue) and the complex coil
          sensitivities (C).
        irgn_par : dict
          A python dict containing the regularization
          parameters for a given gauss newton step.
        queue : list of PyOpenCL.Queues
          A list of PyOpenCL queues to perform the optimization.
        tau : float
          Estimated step size based on operator norm of regularization.
        fval : float
          Estimate of the initial cost function value to
          scale the displayed values.
        prg : PyOpenCL.Program
          A PyOpenCL Program containing the
          kernels for optimization.
        linops : list of PyQMRI Operator
          The linear operators used for fitting.
        coils : PyOpenCL Buffer or empty list
          The coils used for reconstruction.
        model : PyQMRI.Model
          The model which should be fitted
        imagespace : bool, false
          Switch between imagespace (True) and k-space (false) based
          fitting.
        SMS : bool, false
          Switch between SMS (True) and standard (false) reconstruction.

    Attributes
    ----------
      alpha : float
        alpha0 parameter for TGV regularization weight
      beta : float
        alpha1 parameter for TGV regularization weight
      symgrad_shape : tuple of int
        Size of the symmetrized gradient
    """

    def __init__(self,
                 par,
                 irgn_par,
                 queue,
                 tau,
                 fval,
                 prg,
                 linop,
                 coils,
                 model,
                 imagespace=False,
                 SMS=False,
                 **kwargs):

        super().__init__(
            par,
            irgn_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            model,
            imagespace=imagespace,
            **kwargs)

        self.alpha = irgn_par["gamma"]
        self.beta = irgn_par["gamma"] * 2
        self._op = linop[0]
        self._grad_op = linop[1]
        self._symgrad_op = linop[2]

        self.symgrad_shape = self.unknown_shape + (8,)

        self._setup_reg_tmp_arrays("TGV", SMS=SMS)

    def _setupVariables(self, inp, data):

        primal_vars = {}
        primal_vars_new = {}
        tmp_results_adjoint = {}
        tmp_results_adjoint_new = {}

        primal_vars["x"] = inp
        primal_vars["xk"] = primal_vars["x"].copy()
        primal_vars_new["x"] = np.zeros_like(primal_vars["x"])
        primal_vars["v"] = np.zeros(
                                        primal_vars["x"].shape+(4,),
                                        dtype=self._DTYPE)
        primal_vars_new["v"] = np.zeros_like(primal_vars["v"])

        tmp_results_adjoint["Kyk1"] = np.zeros_like(primal_vars["x"])
        tmp_results_adjoint_new["Kyk1"] = np.zeros_like(primal_vars["x"])
        tmp_results_adjoint["Kyk2"] = np.zeros_like(primal_vars["v"])
        tmp_results_adjoint_new["Kyk2"] = np.zeros_like(primal_vars["v"])

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}
        tmp_results_forward_new = {}
        dual_vars["r"] = np.zeros(
            data.shape,
            dtype=self._DTYPE)
        dual_vars_new["r"] = np.zeros_like(dual_vars["r"])

        dual_vars["z1"] = np.zeros(
                                            primal_vars["x"].shape+(4,),
                                            dtype=self._DTYPE)
        dual_vars_new["z1"] = np.zeros_like(dual_vars["z1"])
        dual_vars["z2"] = np.zeros(
                                            primal_vars["x"].shape+(8,),
                                            dtype=self._DTYPE)
        dual_vars_new["z2"] = np.zeros_like(dual_vars["z2"])

        tmp_results_forward["gradx"] = np.zeros_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx"] = np.zeros_like(
            dual_vars["z1"])
        tmp_results_forward["symgradx"] = np.zeros_like(
            dual_vars["z2"])
        tmp_results_forward_new["symgradx"] = np.zeros_like(
            dual_vars["z2"])
        tmp_results_forward["Ax"] = np.zeros_like(data)
        tmp_results_forward_new["Ax"] = np.zeros_like(data)

        return (primal_vars,
                primal_vars_new,
                tmp_results_forward,
                tmp_results_forward_new,
                dual_vars,
                dual_vars_new,
                tmp_results_adjoint,
                tmp_results_adjoint_new,
                data)

    def _updateInitial(self,
                       out_fwd, out_adj,
                       in_primal, in_dual):

        self.stream_initial_1.eval(
            [out_fwd["Ax"],
             out_adj["Kyk1"],
             out_fwd["symgradx"]],
            [
                [in_primal["x"], self._coils, self.modelgrad],
                [in_dual["r"], in_dual["z1"],
                 self._coils, self.modelgrad, []],
                [in_primal["v"]]],
            [[],
             [self._grad_op.ratio],
             []])

        self.stream_initial_2.eval(
            [out_fwd["gradx"],
             out_adj["Kyk2"]],
            [[in_primal["x"]],
             [in_dual["z2"], in_dual["z1"], []]],
            [[],
             self._symgrad_op.ratio])

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        self.update_primal_1.eval(
            [out_primal["x"],
             out_fwd["gradx"],
             out_fwd["Ax"]],
            [[in_primal["x"],
              in_precomp_adj["Kyk1"],
              in_primal["xk"],
              self.jacobi],
             [],
             [[], self._coils, self.modelgrad]],
            [[tau, self.delta],
             [],
             []])

        self.update_primal_2.eval(
            [out_primal["v"],
             out_fwd["symgradx"]],
            [[in_primal["v"], in_precomp_adj["Kyk2"]],
             []],
            [[tau],
             []])

    def _updateDual(self,
                    out_dual, out_adj,
                    in_primal,
                    in_primal_new,
                    in_dual,
                    in_precomp_fwd,
                    in_precomp_fwd_new,
                    in_precomp_adj,
                    data,
                    beta,
                    tau,
                    theta):

        (lhs1, ynorm1) = self.update_dual_1.evalwithnorm(
            [out_dual["z1"],
             out_dual["r"],
             out_adj["Kyk1"]],
            [[in_dual["z1"], in_precomp_fwd_new["gradx"],
              in_precomp_fwd["gradx"], in_primal_new["v"], in_primal["v"]],
             [in_dual["r"], in_precomp_fwd_new["Ax"],
              in_precomp_fwd["Ax"], data],
             [[], [], self._coils, self.modelgrad, in_precomp_adj["Kyk1"]]],
            [
                [beta*tau, theta, self.alpha, self.omega],
                [beta * tau, theta, self.lambd],
                [self._grad_op.ratio]
            ])
        (lhs2, ynorm2) = self.update_dual_2.evalwithnorm(
            [out_dual["z2"],
             out_adj["Kyk2"]],
            [[in_dual["z2"], in_precomp_fwd_new["symgradx"],
              in_precomp_fwd["symgradx"]],
             [[], out_dual["z1"], in_precomp_adj["Kyk2"]]],
            [[beta*tau, theta, self.beta],
             self._symgrad_op.ratio])

        ynorm = np.abs(ynorm1 + ynorm2)**(1/2)
        lhs = np.sqrt(beta) * tau * np.abs(lhs1 + lhs2)**(1/2)

        return lhs, ynorm

    def _calcResidual(
            self,
            in_primal,
            in_dual,
            in_precomp_fwd,
            in_precomp_adj,
            data):
        primal_new = (
            self.lambd / 2 * np.vdot(
                in_precomp_fwd["Ax"] - data,
                in_precomp_fwd["Ax"] - data)
            + self.alpha * np.sum(
                abs(in_precomp_fwd["gradx"] - in_primal["v"])
                )
            + self.beta * np.sum(
                abs(in_precomp_fwd["symgradx"])
                )
            + 1 / (2 * self.delta) * np.vdot(
                (in_primal["x"] - in_primal["xk"])*self.jacobi,
                in_primal["x"] - in_primal["xk"]
                )
            ).real

        dual = (
            -self.delta / 2 * np.vdot(
                - in_precomp_adj["Kyk1"]/self.jacobi,
                - in_precomp_adj["Kyk1"])
            - np.vdot(
                in_primal["xk"],
                - in_precomp_adj["Kyk1"]
                )
            + np.sum(in_precomp_adj["Kyk2"])
            - 1 / (2 * self.lambd) * np.vdot(
                in_dual["r"],
                in_dual["r"])
            - np.vdot(data, in_dual["r"])
            ).real

        if self.unknowns_H1 > 0:
            primal_new += (
                 self.omega / 2 * np.vdot(
                     in_precomp_fwd["gradx"][self.unknowns_TGV:],
                     in_precomp_fwd["gradx"][self.unknowns_TGV:]
                     )
                 ).real

            dual += (
                - 1 / (2 * self.omega) * np.vdot(
                    in_dual["z1"][self.unknowns_TGV:],
                    in_dual["z1"][self.unknowns_TGV:]
                    )
                ).real
        gap = np.abs(primal_new - dual)
        return primal_new, dual, gap


class PDSolverStreamedTGVSMS(PDSolverStreamedTGV):
    """Streamed TGV optimization for SMS data.

    Parameters
    ----------
        par : dict
          A python dict containing the necessary information to
          setup the object. Needs to contain the number of slices (NSlice),
          number of scans (NScan), image dimensions (dimX, dimY), number of
          coils (NC), sampling points (N) and read outs (NProj)
          a PyOpenCL queue (queue) and the complex coil
          sensitivities (C).
        irgn_par : dict
          A python dict containing the regularization
          parameters for a given gauss newton step.
        queue : list of PyOpenCL.Queues
          A list of PyOpenCL queues to perform the optimization.
        tau : float
          Estimated step size based on operator norm of regularization.
        fval : float
          Estimate of the initial cost function value to
          scale the displayed values.
        prg : PyOpenCL.Program
          A PyOpenCL Program containing the
          kernels for optimization.
        linops : list of PyQMRI Operator
          The linear operators used for fitting.
        coils : PyOpenCL Buffer or empty list
          The coils used for reconstruction.
        model : PyQMRI.Model
          The model which should be fitted
        imagespace : bool, false
          Switch between imagespace (True) and k-space (false) based
          fitting.
        SMS : bool, false
          Switch between SMS (True) and standard (false) reconstruction.

    Attributes
    ----------
      alpha : float
        alpha0 parameter for TGV regularization weight
      beta : float
        alpha1 parameter for TGV regularization weight
      symgrad_shape : tuple of int
        Size of the symmetrized gradient
    """

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 linop, coils, model, imagespace=False, **kwargs):

        self._packs = par["packs"]
        self._numofpacks = par["numofpacks"]
        self.data_shape = (self._packs*self._numofpacks, par["NScan"],
                           par["NC"], par["dimY"], par["dimX"])
        self.data_shape_T = (par["NScan"], par["NC"],
                             self._packs*self._numofpacks,
                             par["dimY"], par["dimX"])
        self._expdim_dat = 1
        self._expdim_C = 0

        super().__init__(
            par,
            irgn_par,
            queue,
            tau,
            fval,
            prg,
            linop,
            coils,
            model,
            imagespace=imagespace,
            SMS=True,
            **kwargs)

    def _updateInitial(self,
                       out_fwd, out_adj,
                       in_primal, in_dual):
        out_fwd["Ax"] = self._op.fwdoop(
            [[in_primal["x"], self._coils, self.modelgrad]])
        self._op.adjKyk1(
            [out_adj["Kyk1"]],
            [[in_dual["r"], in_dual["z1"], self._coils, self.modelgrad, []]],
            [[self._grad_op.ratio]])

        self._symgrad_op.fwd(
            [out_fwd["symgradx"]],
            [[in_primal["v"]]])

        self.stream_initial_2.eval(
            [out_fwd["gradx"],
             out_adj["Kyk1"]],
            [[in_primal["x"]],
             [in_dual["z2"], in_dual["z1"], []]],
            [[],
             self._symgrad_op.ratio])

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        self.update_primal_1.eval(
            [out_primal["x"],
             out_fwd["gradx"]],
            [[in_primal["x"],
              in_precomp_adj["Kyk1"],
              in_primal["xk"],
              self.jacobi],
             []],
            [[tau, self.delta],
             []])
        out_fwd["Ax"] = self._op.fwdoop(
            [[out_primal["x"], self._coils, self.modelgrad]])

        self.update_primal_2.eval(
            [out_primal["v"],
             out_fwd["symgradx"]],
            [[in_primal["v"], in_precomp_adj["Kyk2"]],
             []],
            [[tau],
             []])

    def _updateDual(self,
                    out_dual, out_adj,
                    in_primal,
                    in_primal_new,
                    in_dual,
                    in_precomp_fwd,
                    in_precomp_fwd_new,
                    in_precomp_adj,
                    data,
                    beta,
                    tau,
                    theta):

        (lhs1, ynorm1) = self.stream_update_z1.evalwithnorm(
            [out_dual["z1"]],
            [[in_dual["z1"],
              in_precomp_fwd_new["gradx"],
              in_precomp_fwd["gradx"],
              in_primal_new["v"], in_primal["v"]]],
            [[beta*tau, theta,
              self.alpha, self.omega]])
        (lhs2, ynorm2) = self.stream_update_r.evalwithnorm(
            [out_dual["r"]],
            [[in_dual["r"], in_precomp_fwd_new["Ax"],
              in_precomp_fwd["Ax"], data]],
            [[beta*tau, theta,
              self.lambd]])
        (lhs3, ynorm3) = self._op.adjKyk1(
            [out_adj["Kyk1"]],
            [[out_dual["r"],
              out_dual["z1"],
              self._coils, self.modelgrad, in_precomp_adj["Kyk1"]]],
            [[self._grad_op.ratio]])

        (lhs4, ynorm4) = self.update_dual_2.evalwithnorm(
            [out_dual["z2"],
             out_adj["Kyk2"]],
            [[in_dual["z2"], in_precomp_fwd_new["symgradx"],
              in_precomp_fwd["symgradx"]],
             [[], out_dual["z1"], in_precomp_adj["Kyk2"]]],
            [[beta*tau, theta, self.beta],
             self._symgrad_op.ratio])

        ynorm = np.abs(ynorm1+ynorm2+ynorm3+ynorm4)**(1/2)
        lhs = np.sqrt(beta)*tau*np.abs(lhs1+lhs2+lhs3+lhs4)**(1/2)

        return lhs, ynorm


class PDSolverStreamedTV(PDSolverStreamed):
    """Streamed TV optimization.

    Parameters
    ----------
        par : dict
          A python dict containing the necessary information to
          setup the object. Needs to contain the number of slices (NSlice),
          number of scans (NScan), image dimensions (dimX, dimY), number of
          coils (NC), sampling points (N) and read outs (NProj)
          a PyOpenCL queue (queue) and the complex coil
          sensitivities (C).
        irgn_par : dict
          A python dict containing the regularization
          parameters for a given gauss newton step.
        queue : list of PyOpenCL.Queues
          A list of PyOpenCL queues to perform the optimization.
        tau : float
          Estimated step size based on operator norm of regularization.
        fval : float
          Estimate of the initial cost function value to
          scale the displayed values.
        prg : PyOpenCL.Program
          A PyOpenCL Program containing the
          kernels for optimization.
        linops : list of PyQMRI Operator
          The linear operators used for fitting.
        coils : PyOpenCL Buffer or empty list
          The coils used for reconstruction.
        model : PyQMRI.Model
          The model which should be fitted
        imagespace : bool, false
          Switch between imagespace (True) and k-space (false) based
          fitting.
        SMS : bool, false
          Switch between SMS (True) and standard (false) reconstruction.

    Attributes
    ----------
      alpha : float
        alpha0 parameter for TGV regularization weight
      symgrad_shape : tuple of int
        Size of the symmetrized gradient
    """

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 linop, coils, model, imagespace=False, SMS=False, **kwargs):

        super().__init__(
            par,
            irgn_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            model,
            imagespace=imagespace,
            **kwargs)

        self.alpha = irgn_par["gamma"]
        self._op = linop[0]
        self._grad_op = linop[1]
        self._symgrad_op = linop[2]

        self._setup_reg_tmp_arrays("TV", SMS=SMS)

    def _setupVariables(self, inp, data):

        primal_vars = {}
        primal_vars_new = {}
        tmp_results_adjoint = {}
        tmp_results_adjoint_new = {}

        primal_vars["x"] = inp
        primal_vars["xk"] = primal_vars["x"].copy()
        primal_vars_new["x"] = np.zeros_like(primal_vars["x"])

        tmp_results_adjoint["Kyk1"] = np.zeros_like(primal_vars["x"])
        tmp_results_adjoint_new["Kyk1"] = np.zeros_like(primal_vars["x"])

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}
        tmp_results_forward_new = {}
        dual_vars["r"] = np.zeros(
            data.shape,
            dtype=self._DTYPE)
        dual_vars_new["r"] = np.zeros_like(dual_vars["r"])

        dual_vars["z1"] = np.zeros(primal_vars["x"].shape+(4,),
                                   dtype=self._DTYPE)
        dual_vars_new["z1"] = np.zeros_like(dual_vars["z1"])

        tmp_results_forward["gradx"] = np.zeros_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx"] = np.zeros_like(
            dual_vars["z1"])
        tmp_results_forward["Ax"] = np.zeros_like(data)
        tmp_results_forward_new["Ax"] = np.zeros_like(data)

        return (primal_vars,
                primal_vars_new,
                tmp_results_forward,
                tmp_results_forward_new,
                dual_vars,
                dual_vars_new,
                tmp_results_adjoint,
                tmp_results_adjoint_new,
                data)

    def _updateInitial(self,
                       out_fwd, out_adj,
                       in_primal, in_dual):

        self.stream_initial_1.eval(
            [out_fwd["Ax"],
             out_adj["Kyk1"]],
            [
                [in_primal["x"], self._coils, self.modelgrad],
                [in_dual["r"], in_dual["z1"],
                 self._coils, self.modelgrad, []]],
            [[],
             [self._grad_op.ratio]])

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        self.update_primal_1.eval(
            [out_primal["x"],
             out_fwd["gradx"],
             out_fwd["Ax"]],
            [[in_primal["x"],
              in_precomp_adj["Kyk1"],
              in_primal["xk"],
              self.jacobi],
             [],
             [[], self._coils, self.modelgrad]],
            [[tau, self.delta],
             [],
             []])

    def _updateDual(self,
                    out_dual, out_adj,
                    in_primal,
                    in_primal_new,
                    in_dual,
                    in_precomp_fwd,
                    in_precomp_fwd_new,
                    in_precomp_adj,
                    data,
                    beta,
                    tau,
                    theta):

        (lhs1, ynorm1) = self.update_dual_1.evalwithnorm(
            [out_dual["z1"],
             out_dual["r"],
             out_adj["Kyk1"]],
            [[in_dual["z1"], in_precomp_fwd_new["gradx"],
              in_precomp_fwd["gradx"]],
             [in_dual["r"], in_precomp_fwd_new["Ax"],
              in_precomp_fwd["Ax"], data],
             [[], [], self._coils, self.modelgrad, in_precomp_adj["Kyk1"]]],
            [
                [beta*tau, theta, self.alpha, self.omega],
                [beta * tau, theta, self.lambd],
                [self._grad_op.ratio]
            ])

        ynorm = np.abs(ynorm1)**(1/2)
        lhs = np.sqrt(beta) * tau * np.abs(lhs1)**(1/2)

        return lhs, ynorm

    def _calcResidual(
            self,
            in_primal,
            in_dual,
            in_precomp_fwd,
            in_precomp_adj,
            data):

        primal_new = (
            self.lambd / 2 * np.vdot(
                in_precomp_fwd["Ax"] - data,
                in_precomp_fwd["Ax"] - data)
            + self.alpha * np.sum(
                abs(in_precomp_fwd["gradx"])
                )
            + 1 / (2 * self.delta) * np.vdot(
                (in_primal["x"] - in_primal["xk"])*self.jacobi,
                in_primal["x"] - in_primal["xk"]
                )
            ).real

        dual = (
            -self.delta / 2 * np.vdot(
                - in_precomp_adj["Kyk1"]/self.jacobi,
                - in_precomp_adj["Kyk1"])
            - np.vdot(
                in_primal["xk"],
                - in_precomp_adj["Kyk1"]
                )
            - 1 / (2 * self.lambd) * np.vdot(
                in_dual["r"],
                in_dual["r"])
            - np.vdot(data, in_dual["r"])
            ).real

        if self.unknowns_H1 > 0:
            primal_new += (
                 self.omega / 2 * np.vdot(
                     in_precomp_fwd["gradx"][self.unknowns_TGV:],
                     in_precomp_fwd["gradx"][self.unknowns_TGV:]
                     )
                 ).real

            dual += (
                - 1 / (2 * self.omega) * np.vdot(
                    in_dual["z1"][self.unknowns_TGV:],
                    in_dual["z1"][self.unknowns_TGV:]
                    )
                ).real
        gap = np.abs(primal_new - dual)
        return primal_new, dual, gap


class PDSolverStreamedTVSMS(PDSolverStreamedTV):
    """Streamed TV optimization for SMS data.

    Parameters
    ----------
        par : dict
          A python dict containing the necessary information to
          setup the object. Needs to contain the number of slices (NSlice),
          number of scans (NScan), image dimensions (dimX, dimY), number of
          coils (NC), sampling points (N) and read outs (NProj)
          a PyOpenCL queue (queue) and the complex coil
          sensitivities (C).
        irgn_par : dict
          A python dict containing the regularization
          parameters for a given gauss newton step.
        queue : list of PyOpenCL.Queues
          A list of PyOpenCL queues to perform the optimization.
        tau : float
          Estimated step size based on operator norm of regularization.
        fval : float
          Estimate of the initial cost function value to
          scale the displayed values.
        prg : PyOpenCL.Program
          A PyOpenCL Program containing the
          kernels for optimization.
        linops : list of PyQMRI Operator
          The linear operators used for fitting.
        coils : PyOpenCL Buffer or empty list
          The coils used for reconstruction.
        model : PyQMRI.Model
          The model which should be fitted
        imagespace : bool, false
          Switch between imagespace (True) and k-space (false) based
          fitting.

    Attributes
    ----------
      alpha : float
        alpha0 parameter for TGV regularization weight
      symgrad_shape : tuple of int
        Size of the symmetrized gradient
    """

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 linop, coils, model, imagespace=False, **kwargs):
        self._packs = par["packs"]
        self._numofpacks = par["numofpacks"]
        self.data_shape = (self._packs*self._numofpacks, par["NScan"],
                           par["NC"], par["dimY"], par["dimX"])
        self.data_shape_T = (par["NScan"], par["NC"],
                             self._packs*self._numofpacks,
                             par["dimY"], par["dimX"])

        self._expdim_dat = 1
        self._expdim_C = 0

        super().__init__(
            par,
            irgn_par,
            queue,
            tau,
            fval,
            prg,
            linop,
            coils,
            model,
            imagespace=imagespace,
            SMS=True,
            **kwargs)

    def _updateInitial(self,
                       out_fwd, out_adj,
                       in_primal, in_dual):
        out_fwd["Ax"] = self._op.fwdoop(
            [[in_primal["x"], self._coils, self.modelgrad]])
        self._op.adjKyk1(
            [out_adj["Kyk1"]],
            [[in_dual["r"], in_dual["z1"], self._coils, self.modelgrad, []]],
            [[self._grad_op.ratio]])

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        self.update_primal_1.eval(
            [out_primal["x"],
             out_fwd["gradx"]],
            [[in_primal["x"],
              in_precomp_adj["Kyk1"],
              in_primal["xk"],
              self.modelgrad],
             []],
            [[tau, self.delta],
             []])
        out_fwd["Ax"] = self._op.fwdoop(
            [[out_primal["x"], self._coils, self.modelgrad]])

    def _updateDual(self,
                    out_dual, out_adj,
                    in_primal,
                    in_primal_new,
                    in_dual,
                    in_precomp_fwd,
                    in_precomp_fwd_new,
                    in_precomp_adj,
                    data,
                    beta,
                    tau,
                    theta):

        (lhs1, ynorm1) = self.stream_update_z1.evalwithnorm(
            [out_dual["z1"]],
            [[in_dual["z1"],
              in_precomp_fwd_new["gradx"],
              in_precomp_fwd["gradx"]]],
            [[beta*tau, theta,
              self.alpha, self.omega]])
        (lhs2, ynorm2) = self.stream_update_r.evalwithnorm(
            [out_dual["r"]],
            [[in_dual["r"], in_precomp_fwd_new["Ax"],
              in_precomp_fwd["Ax"], data]],
            [[beta*tau, theta,
              self.lambd]])
        (lhs3, ynorm3) = self._op.adjKyk1(
            [out_adj["Kyk1"]],
            [[out_dual["r"],
              out_dual["z1"],
              self._coils, self.modelgrad, in_precomp_adj["Kyk1"]]],
            [[self._grad_op.ratio]])

        ynorm = np.abs(ynorm1+ynorm2+ynorm3)**(1/2)
        lhs = np.sqrt(beta)*tau*np.abs(lhs1+lhs2+lhs3)**(1/2)

        return lhs, ynorm
