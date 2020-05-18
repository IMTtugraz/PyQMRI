#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This module holds the classes for different numerical Optimizer.

Attribues:
  DTYPE (complex64):
    Complex working precission. Currently single precission only.
  DTYPE_real (float32):
    Real working precission. Currently single precission only.
"""

from __future__ import division
import numpy as np
from pkg_resources import resource_filename
import pyopencl as cl
import pyopencl.array as clarray
import pyqmri.operator as operator
from pyqmri._helper_fun import CLProgram as Program
import sys
import pyqmri.streaming as streaming
DTYPE = np.complex64
DTYPE_real = np.float32


class CGSolver:
    """
    Conjugate Gradient Optimization Algorithm.

    This Class performs a CG reconstruction on single precission complex input
    data.
    """

    def __init__(self, par, NScan=1, trafo=1, SMS=0):
        """
        CG reconstruction Object.

        Args
        ----
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          NScan (int): Number of Scan which should be used internally. Do not
            need to be the same number as in par["NScan"]
          trafo (bool): Switch between radial (1) and Cartesian (0) fft.
          SMS (bool): Simultaneouos Multi Slice. Switch between noraml (0)
          and slice accelerated (1) reconstruction.
        """
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
        if SMS:
            self._op = operator.OperatorKspaceSMS(par, self._prg, trafo=trafo)
            self._tmp_sino = clarray.empty(
                self._queue,
                (self._NScan, self._NC,
                 int(self._NSlice/par["MB"]), par["Nproj"], par["N"]),
                DTYPE, "C")
        else:
            self._op = operator.OperatorKspace(par, self._prg, trafo=trafo)
            self._tmp_sino = clarray.empty(
                self._queue,
                (self._NScan, self._NC,
                 self._NSlice, par["Nproj"], par["N"]),
                DTYPE, "C")
        self._FT = self._op.NUFFT.FFT
        self._FTH = self._op.NUFFT.FFTH
        self._tmp_result = clarray.empty(
            self._queue,
            (self._NScan, self._NC,
             self._NSlice, self._dimY, self._dimX),
            DTYPE, "C")
        par["NScan"] = NScan_save

    def __del__(self):
        """
        Destructor.

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

        Args
        ----
          data (complex64):
            The complex k-space data which serves as the basis for the images.
          iters (int):
            Maximum number of CG iterations
          lambd (float):
            Weighting parameter for the Tikhonov regularization
          tol (float):
            Termination criterion. If the energy decreases below this
            threshold the algorithm is terminated.
          guess (complex64):
            An optional initial guess for the images. If None, zeros is used.
        """
        self.scan_offset = scan_offset
        if guess is not None:
            x = clarray.to_device(self._queue, guess)
        else:
            x = clarray.zeros(self._queue,
                              (self._NScan, 1,
                               self._NSlice, self._dimY, self._dimX),
                              DTYPE, "C")
        b = clarray.empty(self._queue,
                          (self._NScan, 1,
                           self._NSlice, self._dimY, self._dimX),
                          DTYPE, "C")
        Ax = clarray.empty(self._queue,
                           (self._NScan, 1,
                            self._NSlice, self._dimY, self._dimX),
                           DTYPE, "C")

        data = clarray.to_device(self._queue, data)
        self._operator_rhs(b, data)
        res = b
        p = res
        delta = np.linalg.norm(res.get())**2/np.linalg.norm(b.get())**2

#        print("Initial Residuum: ", delta)

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
#            print("Res after iteration %i: %1.3e." % (i, delta))
            beta = (clarray.vdot(res_new, res_new) /
                    clarray.vdot(res, res)).real.get()
            p = res_new+beta*p
            (res, res_new) = (res_new, res)
        del Ax, b, res, p, data, res_new
        return np.squeeze(x.get())

    def eval_fwd_kspace_cg(self, y, x, wait_for=[]):
        """
        Apply forward operator for image reconstruction.

        Args
        ----
          y (PyOpenCL.Array):
            The result of the computation
          x (PyOpenCL.Array):
            The input array
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        """
        return self._prg.operator_fwd_cg(self._queue,
                                         (self._NSlice, self._dimY,
                                          self._dimX),
                                         None,
                                         y.data, x.data, self._coils,
                                         np.int32(self._NC),
                                         np.int32(self._NScan),
                                         wait_for=wait_for)

    def _operator_lhs(self, out, x, wait_for=[]):
        """
        Compute the left hand side of the CG equation.

        Args
        ----
          out (PyOpenCL.Array):
            The result of the computation
          x (PyOpenCL.Array):
            The input array
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        """
        self._tmp_result.add_event(self.eval_fwd_kspace_cg(
            self._tmp_result, x, wait_for=self._tmp_result.events+x.events))
        self._tmp_sino.add_event(self._FT(
            self._tmp_sino, self._tmp_result, scan_offset=self.scan_offset))
        return self._operator_rhs(out, self._tmp_sino)

    def _operator_rhs(self, out, x, wait_for=[]):
        """
        Compute the right hand side of the CG equation.

        Args
        ----
          out (PyOpenCL.Array):
            The result of the computation
          x (PyOpenCL.Array):
            The input array
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        """
        self._tmp_result.add_event(self._FTH(
            self._tmp_result, x, wait_for=wait_for+x.events,
            scan_offset=self.scan_offset))
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
    """
    Primal Dual splitting optimization.

    This Class performs a primal-dual variable splitting based reconstruction
    on single precission complex input data.
    """

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 coil, model):
        """
        PD reconstruction Object.

        Args
        ----
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          irgn_par (dict): A python dict containing the regularization
            parameters for a given gauss newton step.
          queue (list): A list of PyOpenCL queues to perform the optimization.
          tau (float): Estimate of the initial step size based on the
            operator norm of the linear operator.
          fval (float): Estimate of the initial cost function value to
            scale the displayed values.
          prg (PyOpenCL Program): A PyOpenCL Program containing the
            kernels for optimization.
          reg_type (string): String to choose between "TV" and "TGV"
            optimization.
          data_operator (PyQMRI Operator): The operator to traverse from
            parameter to data space.
          coils (PyOpenCL Buffer or empty List): optional coil buffer.
          NScan (int): Number of Scan which should be used internally. Do not
            need to be the same number as in par["NScan"]
          trafo (bool): Switch between radial (1) and Cartesian (0) fft.
          and slice accelerated (1) reconstruction.
        """
        self.delta = irgn_par["delta"]
        self.omega = irgn_par["omega"]
        self.lambd = irgn_par["lambd"]
        self.tol = irgn_par["tol"]
        self.stag = irgn_par["stag"]
        self.display_iterations = irgn_par["display_iterations"]
        self.mu = 1 / self.delta
        self.tau = tau
        self.beta_line = 1
        self.theta_line = np.float32(1.0)
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
            imagespace=False):
        """
        Generate a PDSolver object.

        Args
        ----
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          irgn_par (dict): A python dict containing the regularization
            parameters for a given gauss newton step.
          queue (list): A list of PyOpenCL queues to perform the optimization.
          tau (float): Estimate of the initial step size based on the
            operator norm of the linear operator.
          fval (float): Estimate of the initial cost function value to
            scale the displayed values.
          prg (PyOpenCL Program): A PyOpenCL Program containing the
            kernels for optimization.
          reg_type (string): String to choose between "TV" and "TGV"
            optimization.
          data_operator (PyQMRI Operator): The operator to traverse from
            parameter to data space.
          coils (PyOpenCL Buffer or empty List): optional coil buffer.
          NScan (int): Number of Scan which should be used internally. Do not
            need to be the same number as in par["NScan"]
          trafo (bool): Switch between radial (1) and Cartesian (0) fft.
          and slice accelerated (1) reconstruction.
        """
        if reg_type == 'TV':
            if streamed:
                if SMS:
                    pdop = PDSolverStreamedTVSMS(
                        par,
                        irgn_par,
                        queue,
                        np.float32(1 / np.sqrt(8)),
                        init_fval,
                        prg,
                        linops,
                        coils,
                        model,
                        imagespace=imagespace)
                else:
                    pdop = PDSolverStreamedTV(
                        par,
                        irgn_par,
                        queue,
                        np.float32(1 / np.sqrt(8)),
                        init_fval,
                        prg,
                        linops,
                        coils,
                        model,
                        imagespace=imagespace)
            else:
                pdop = PDSolverTV(par,
                                  irgn_par,
                                  queue,
                                  np.float32(1 / np.sqrt(8)),
                                  init_fval, prg,
                                  linops,
                                  coils,
                                  model)

        elif reg_type == 'TGV':
            L = np.float32(0.5 * (18.0 + np.sqrt(33)))
            if streamed:
                if SMS:
                    pdop = PDSolverStreamedTGVSMS(
                        par,
                        irgn_par,
                        queue,
                        np.float32(1 / np.sqrt(L)),
                        init_fval,
                        prg,
                        linops,
                        coils,
                        model,
                        imagespace=imagespace)
                else:
                    pdop = PDSolverStreamedTGV(
                        par,
                        irgn_par,
                        queue,
                        np.float32(1 / np.sqrt(L)),
                        init_fval,
                        prg,
                        linops,
                        coils,
                        model,
                        imagespace=imagespace)
            else:
                pdop = PDSolverTGV(
                    par,
                    irgn_par,
                    queue,
                    np.float32(1 / np.sqrt(L)),
                    init_fval,
                    prg,
                    linops,
                    coils,
                    model)
        else:
            raise NotImplementedError
        return pdop

    def __del__(self):
        """
        Destructor.

        Releases GPU memory arrays.
        """

    def run(self, inp, data, iters):
        """
        Optimization with 3D T(G)V regularization.

        Args
        ----
          x (numpy.array):
            Initial guess for the unknown parameters
          x (numpy.array):
            The complex valued data to fit.
          iters (int):
            Number of primal-dual iterations to run
        """
        self._updateConstraints()

        tau = self.tau
        tau_new = np.float32(0)

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
            tau_new = tau * np.sqrt(beta_line / beta_new * (1 + theta_line))
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
                else:
                    tau_new = tau_new * mu_line

            tau = tau_new
            self.beta_line = beta_line
            self.tau = tau
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
                    if type(primal_vars["x"]) is np.ndarray:
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
                     self.beta_line))
                sys.stdout.flush()

        return primal_vars

    def _updateInitial(self, outp, inp):
        pass

    def _updatePrimal(self, outp, inp, tau):
        pass

    def _updateDual(self, outp, inp, par):
        pass

    def _calcResidual(
                    primal_vars,
                    dual_vars,
                    Axold,
                    tmp_results_adjoint,
                    tmp_results_forward):
        pass

    def _setupVariables(inps, data):
        pass

    def _updateConstraints(self):
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
                clarray.to_device(self._queue[4*j], min_const))
            self.max_const.append(
                clarray.to_device(self._queue[4*j], max_const))
            self.real_const.append(
                clarray.to_device(self._queue[4*j], real_const))

    def updateRegPar(self, irgn_par):
        """
        Update the regularization parameters.

          Performs an update of the regularization parameters as these usually
          vary from one to another Gauss-Newton step.

        Args
        ----
          irgn_par (dic): A dictionary containing the new parameters.
        """
        self.alpha = irgn_par["gamma"]
        self.beta = irgn_par["gamma"] * 2
        self.delta = irgn_par["delta"]
        self.omega = irgn_par["omega"]
        self.lambd = irgn_par["lambd"]
        self.mu = 1/self.delta

    def update_primal(self, outp, inp, par=None, idx=0, idxq=0,
                      bound_cond=0, wait_for=[]):
        """
        Primal update of the x variable in the Primal-Dual Algorithm.

        Args
        ----
          x_new (PyOpenCL.Array): The result of the update step
          x (PyOpenCL.Array): The previous values of x
          Kyk (PyOpenCL.Array): x-Part of the precomputed result of
            the adjoint linear operator applied to y
          xk (PyOpenCL.Array): The linearization point of the Gauss-Newton step
          tau (float): Primal step size
          delta (float): Regularization parameter for the step-size constrained
          wait_for (list): A optional list for PyOpenCL.events to wait for
        """
        return self._prg[idx].update_primal_LM(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, inp[1].data, inp[2].data, inp[3].data,
            np.float32(par[0]),
            np.float32(par[0]/par[1]),
            self.min_const[idx].data, self.max_const[idx].data,
            self.real_const[idx].data, np.int32(self.unknowns),
            wait_for=(outp.events +
                      inp[0].events+inp[1].events +
                      inp[2].events+wait_for))

    def update_v(self, outp, inp, par=None, idx=0, idxq=0,
                 bound_cond=0, wait_for=[]):
        """
        Primal update of the v variable in Primal-Dual Algorithm.

        Args
        ----
          v_new (PyOpenCL.Array): The result of the update step
          v (PyOpenCL.Array): The previous values of v
          Kyk2 (PyOpenCL.Array): v-Part of the precomputed result of
            the adjoint linear operator applied to y
          tau (float): Primal step size
          wait_for (list): A optional list for PyOpenCL.events to wait for
        """
        return self._prg[idx].update_v(
            self._queue[4*idx+idxq], (outp[..., 0].size,), None,
            outp.data, inp[0].data, inp[1].data, np.float32(par[0]),
            wait_for=outp.events+inp[0].events+inp[1].events+wait_for)

    def update_z1(self, outp, inp, par=None, idx=0, idxq=0,
                  bound_cond=0, wait_for=[]):
        """
        Dual update of the z1 variable in Primal-Dual Algorithm for TGV.

        Args
        ----
          z_new (PyOpenCL.Array): The result of the update step
          z (PyOpenCL.Array): The previous values of z
          gx (PyOpenCL.Array): Linear Operator applied to the new x
          gx_ (PyOpenCL.Array): Linear Operator applied to the previous x
          vx (PyOpenCL.Array): Linear Operator applied to the new v
          vx_ (PyOpenCL.Array): Linear Operator applied to the previous v
          sigma (float): Dual step size
          theta (float): Variable controling the extrapolation step of the
            PD-algorithm
          alpha (float): Regularization parameter of the first TGV functional
          omega (float): Optimal regularization parameter for H1 regularization
          wait_for (list): A optional list for PyOpenCL.events to wait for
        """
        return self._prg[idx].update_z1(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, inp[1].data,
            inp[2].data, inp[3].data, inp[4].data,
            np.float32(par[0]), np.float32(par[1]),
            np.float32(1/par[2]), np.int32(self.unknowns_TGV),
            np.int32(self.unknowns_H1), np.float32(1 / (1 + par[0] / par[3])),
            wait_for=(outp.events+inp[0].events+inp[1].events +
                      inp[2].events+inp[3].events+inp[4].events+wait_for))

    def update_z1_tv(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=[]):
        """
        Dual update of the z1 variable in Primal-Dual Algorithm for TV.

        Args
        ----
          z_new (PyOpenCL.Array): The result of the update step
          z (PyOpenCL.Array): The previous values of z
          gx (PyOpenCL.Array): Linear Operator applied to the new x
          gx_ (PyOpenCL.Array): Linear Operator applied to the previous x
          sigma (float): Dual step size
          theta (float): Variable controling the extrapolation step of the
            PD-algorithm
          alpha (float): Regularization parameter of the first TGV functional
          omega (float): Optimal regularization parameter for H1 regularization
          wait_for (list): A optional list for PyOpenCL.events to wait for
        """
        return self._prg[idx].update_z1_tv(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, inp[1].data, inp[2].data,
            np.float32(par[0]),
            np.float32(par[1]),
            np.float32(1/par[2]), np.int32(self.unknowns_TGV),
            np.int32(self.unknowns_H1), np.float32(1 / (1 + par[0] / par[3])),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def update_z2(self, outp, inp, par=None, idx=0, idxq=0,
                  bound_cond=0, wait_for=[]):
        """
        Dual update of the z2 variable in Primal-Dual Algorithm for TGV.

        Args
        ----
          z_new (PyOpenCL.Array): The result of the update step
          z (PyOpenCL.Array): The previous values of z
          gx (PyOpenCL.Array): Linear Operator applied to the new x
          gx_ (PyOpenCL.Array): Linear Operator applied to the previous x
          sigma (float): Dual step size
          theta (float): Variable controling the extrapolation step of the
            PD-algorithm
          beta (float): Regularization parameter of the second TGV functional
          wait_for (list): A optional list for PyOpenCL.events to wait for
        """
        return self._prg[idx].update_z2(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, inp[1].data, inp[2].data,
            np.float32(par[0]),
            np.float32(par[1]),
            np.float32(1/par[2]), np.int32(self.unknowns),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def update_Kyk2(self, outp, inp, par=None, idx=0, idxq=0,
                    bound_cond=0, wait_for=[]):
        """
        Precompute the v-part of the Adjoint Linear operator.

        Args
        ----
          div (PyOpenCL.Array): The result of the computation
          u (PyOpenCL.Array): Dual Variable of the symmetrized gradient of TGV
          z (PyOpenCL.Array): Dual Variable of the gradient of TGV
          wait_for (list): A optional list for PyOpenCL.events to wait for
        """
        return self._prg[idx].update_Kyk2(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, inp[1].data,
            np.int32(self.unknowns), np.int32(bound_cond), np.float32(self.dz),
            wait_for=outp.events + inp[0].events + inp[1].events+wait_for)

    def update_r(self, outp, inp, par=None, idx=0, idxq=0,
                 bound_cond=0, wait_for=[]):
        """
        Update the data dual variable r.

        Args
        ----
          r_new (PyOpenCL.Array): New value of the data dual variable
          r (PyOpenCL.Array): Old value of the data dual variable
          A (PyOpenCL.Array): Precomputed Linear Operator applied to the new x
          A_ (PyOpenCL.Array):Precomputed Linear Operator applied to the old x
          res (PyOpenCL.Array): Precomputed data to compare to
          sigma (float): Dual step size
          theta (float): Variable controling the extrapolation step of the
            PD-algorithm
          lambd (float): Regularization parameter in fron tof the data fidelity
            term
          wait_for (list): A optional list for PyOpenCL.events to wait for
        """
        return self._prg[idx].update_r(
            self._queue[4*idx+idxq], (outp.size,), None,
            outp.data, inp[0].data,
            inp[1].data, inp[2].data, inp[3].data,
            np.float32(par[0]), np.float32(par[1]),
            np.float32(1/(1+par[0]/par[2])),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def update_primal_explicit(self, x_new, x, Kyk, xk, ATd,
                               tau, delta, lambd, wait_for=[]):
        """
        Explicit update of the primal variable.

        Args
        ----
          x_new (PyOpenCL.Array): The result of the update step
          x (PyOpenCL.Array): The previous values of x
          Kyk (PyOpenCL.Array): x-Part of the precomputed result of
            the adjoint linear operator applied to y
          xk (PyOpenCL.Array): The linearization point of the Gauss-Newton step
          ATd (PyOpenCL.Array): Precomputed result of the adjoint linear
            operator applied to the data
          tau (float): Primal step size
          delta (float): Regularization parameter for the step-size constrained
          wait_for (list): A optional list for PyOpenCL.events to wait for
        """
        return self._prg.update_primal_explicit(
            self._queue, x.shape[1:], None, x_new.data, x.data, Kyk.data,
            xk.data, ATd.data, np.float32(tau),
            np.float32(1 / delta), np.float32(lambd), self.min_const.data,
            self.max_const.data,
            self.real_const.data, np.int32(self.unknowns),
            wait_for=(
                x_new.events + x.events + Kyk.events +
                xk.events + ATd.events + wait_for))


class PDSolverTV(PDBaseSolver):
    """
    Primal Dual splitting optimization for TV.

    This Class performs a primal-dual variable splitting based reconstruction
    on single precission complex input data.
    """

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 linop, coils, model):
        """
        TV PD reconstruction Object.

        Args
        ----
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          irgn_par (dict): A python dict containing the regularization
            parameters for a given gauss newton step.
          queue (list): A list of PyOpenCL queues to perform the optimization.
          tau (float): Estimate of the initial step size based on the
            operator norm of the linear operator.
          fval (float): Estimate of the initial cost function value to
            scale the displayed values.
          prg (PyOpenCL Program): A PyOpenCL Program containing the
            kernels for optimization.
          reg_type (string): String to choose between "TV" and "TGV"
            optimization.
          data_operator (PyQMRI Operator): The operator to traverse from
            parameter to data space.
          coils (PyOpenCL Buffer or empty List): optional coil buffer.
          NScan (int): Number of Scan which should be used internally. Do not
            need to be the same number as in par["NScan"]
          trafo (bool): Switch between radial (1) and Cartesian (0) fft.
          and slice accelerated (1) reconstruction.
        """
        super().__init__(
            par,
            irgn_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            model)
        self.alpha = irgn_par["gamma"]
        self._op = linop[0]
        self.grad_op = linop[1]

    def _setupVariables(self, inp, data):

        data = clarray.to_device(self._queue[0], data.astype(DTYPE))

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
            dtype=DTYPE)
        dual_vars_new["r"] = clarray.empty_like(dual_vars["r"])

        dual_vars["z1"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape+(4,),
                                        dtype=DTYPE)
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
                              self.grad_op._ratio]))

        out_fwd["Ax"].add_event(self._op.fwd(
            out_fwd["Ax"], [in_primal["x"], self._coils, self.modelgrad]))
        out_fwd["gradx"].add_event(
            self.grad_op.fwd(out_fwd["gradx"], in_primal["x"]))

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        out_primal["x"].add_event(self.update_primal(
            outp=out_primal["x"],
            inp=(in_primal["x"], in_precomp_adj["Kyk1"],
                 in_primal["xk"], self.modelgrad),
            par=(tau, self.delta)))

        out_fwd["gradx"].add_event(
            self.grad_op.fwd(
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
                 self.grad_op._ratio]))

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
                in_primal["x"] - in_primal["xk"],
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
    """
    TGV Primal Dual splitting optimization.

    This Class performs a primal-dual variable splitting based reconstruction
    on single precission complex input data.
    """

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 linop, coils, model):
        """
        TGV PD reconstruction Object.

        Args
        ----
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          irgn_par (dict): A python dict containing the regularization
            parameters for a given gauss newton step.
          queue (list): A list of PyOpenCL queues to perform the optimization.
          tau (float): Estimate of the initial step size based on the
            operator norm of the linear operator.
          fval (float): Estimate of the initial cost function value to
            scale the displayed values.
          prg (PyOpenCL Program): A PyOpenCL Program containing the
            kernels for optimization.
          reg_type (string): String to choose between "TV" and "TGV"
            optimization.
          data_operator (PyQMRI Operator): The operator to traverse from
            parameter to data space.
          coils (PyOpenCL Buffer or empty List): optional coil buffer.
          NScan (int): Number of Scan which should be used internally. Do not
            need to be the same number as in par["NScan"]
          trafo (bool): Switch between radial (1) and Cartesian (0) fft.
          and slice accelerated (1) reconstruction.
        """
        super().__init__(
            par,
            irgn_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            model)
        self.alpha = irgn_par["gamma"]
        self.beta = irgn_par["gamma"] * 2
        self._op = linop[0]
        self.grad_op = linop[1]
        self.symgrad_op = linop[2]

    def _setupVariables(self, inp, data):

        data = clarray.to_device(self._queue[0], data.astype(DTYPE))

        primal_vars = {}
        primal_vars_new = {}
        tmp_results_adjoint = {}
        tmp_results_adjoint_new = {}

        primal_vars["x"] = clarray.to_device(self._queue[0], inp)
        primal_vars["xk"] = primal_vars["x"].copy()
        primal_vars_new["x"] = clarray.empty_like(primal_vars["x"])
        primal_vars["v"] = clarray.zeros(self._queue[0],
                                         primal_vars["x"].shape+(4,),
                                         dtype=DTYPE)
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
            dtype=DTYPE)
        dual_vars_new["r"] = clarray.empty_like(dual_vars["r"])

        dual_vars["z1"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape+(4,),
                                        dtype=DTYPE)
        dual_vars_new["z1"] = clarray.empty_like(dual_vars["z1"])
        dual_vars["z2"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape+(8,),
                                        dtype=DTYPE)
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
                              self.grad_op._ratio]))

        out_adj["Kyk2"].add_event(
            self.update_Kyk2(
                outp=out_adj["Kyk2"],
                inp=(in_dual["z2"], in_dual["z1"])))

        out_fwd["Ax"].add_event(self._op.fwd(
            out_fwd["Ax"], [in_primal["x"], self._coils, self.modelgrad]))
        out_fwd["gradx"].add_event(
            self.grad_op.fwd(out_fwd["gradx"], in_primal["x"]))
        out_fwd["symgradx"].add_event(
            self.symgrad_op.fwd(out_fwd["symgradx"], in_primal["v"]))

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        out_primal["x"].add_event(self.update_primal(
            outp=out_primal["x"],
            inp=(in_primal["x"], in_precomp_adj["Kyk1"],
                 in_primal["xk"], self.modelgrad),
            par=(tau, self.delta)))

        out_primal["v"].add_event(self.update_v(
            outp=out_primal["v"],
            inp=(in_primal["v"], in_precomp_adj["Kyk2"]),
            par=(tau,)))

        out_fwd["gradx"].add_event(
            self.grad_op.fwd(
                out_fwd["gradx"], out_primal["x"]))

        out_fwd["symgradx"].add_event(
            self.symgrad_op.fwd(
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
                 self.grad_op._ratio]))
        out_adj["Kyk2"].add_event(
            self.update_Kyk2(
                outp=out_adj["Kyk2"],
                inp=(out_dual["z2"], out_dual["z1"])))

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
                in_primal["x"] - in_primal["xk"],
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
    """
    Streamed version of the PD Solver.

    This class is the base class for the streamed array optimization.
    """

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 linop, coils, model, imagespace=False):

        super().__init__(
            par,
            irgn_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            model)

        self.unknown_shape = (par["NSlice"], par["unknowns"],
                              par["dimY"], par["dimX"])
        self.model_deriv_shape = (par["NSlice"], par["unknowns"],
                                  par["NScan"],
                                  par["dimY"], par["dimX"])

        self.grad_shape = self.unknown_shape + (4,)

        self.NSlice = par["NSlice"]
        self.par_slices = par["par_slices"]
        self.overlap = par["overlap"]

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
                dtype=DTYPE)
            self.z2 = np.zeros(
                self.symgrad_shape,
                dtype=DTYPE)
        else:
            raise NotImplementedError("Not implemented")
        self._setupstreamingops(reg_type, SMS=SMS)

        self.r = np.zeros(
                self.data_shape,
                dtype=DTYPE)
        self.z1 = np.zeros(
            self.grad_shape,
            dtype=DTYPE)

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
                self.stream_initial_1 += self.symgrad_op._stream_symgrad

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

            self.stream_initial_2 += self.grad_op._stream_grad
            self.stream_initial_2 += self.stream_Kyk2

        self.stream_primal = self._defineoperator(
            [self.update_primal],
            [self.unknown_shape],
            [[self.unknown_shape,
              self.unknown_shape,
              self.unknown_shape,
              self.model_deriv_shape]])

        self.update_primal_1 = self._defineoperator(
            [],
            [],
            [[]])

        self.update_primal_1 += self.stream_primal
        self.update_primal_1 += self.grad_op._stream_grad
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
            self.update_primal_2 += self.symgrad_op._stream_symgrad
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
                slices=self.packs*self.numofpacks,
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
            self._queue,
            self.num_dev,
            reverse_dir,
            posofnorm)


class PDSolverStreamedTGV(PDSolverStreamed):
    """Streamed TGV optimization."""

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
                 SMS=False):

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
            imagespace=imagespace)

        self.alpha = irgn_par["gamma"]
        self.beta = irgn_par["gamma"] * 2
        self._op = linop[0]
        self.grad_op = linop[1]
        self.symgrad_op = linop[2]

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
                                        dtype=DTYPE)
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
            dtype=DTYPE)
        dual_vars_new["r"] = np.zeros_like(dual_vars["r"])

        dual_vars["z1"] = np.zeros(
                                            primal_vars["x"].shape+(4,),
                                            dtype=DTYPE)
        dual_vars_new["z1"] = np.zeros_like(dual_vars["z1"])
        dual_vars["z2"] = np.zeros(
                                            primal_vars["x"].shape+(8,),
                                            dtype=DTYPE)
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
             [self.grad_op._ratio],
             []])

        self.stream_initial_2.eval(
            [out_fwd["gradx"],
             out_adj["Kyk2"]],
            [[in_primal["x"]],
             [in_dual["z2"], in_dual["z1"], []]])

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
              self.modelgrad],
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
                [self.grad_op._ratio]
            ])
        (lhs2, ynorm2) = self.update_dual_2.evalwithnorm(
            [out_dual["z2"],
             out_adj["Kyk2"]],
            [[in_dual["z2"], in_precomp_fwd_new["symgradx"],
              in_precomp_fwd["symgradx"]],
             [[], out_dual["z1"], in_precomp_adj["Kyk2"]]],
            [[beta*tau, theta, self.beta],
             []])

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
                in_primal["x"] - in_primal["xk"],
                in_primal["x"] - in_primal["xk"]
                )
            ).real

        dual = (
            -self.delta / 2 * np.vdot(
                - in_precomp_adj["Kyk1"],
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
    """Streamed TGV optimization for SMS data."""

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 linop, coils, model, imagespace=False):

        self.packs = par["packs"]
        self.numofpacks = par["numofpacks"]
        self.data_shape = (self.packs*self.numofpacks, par["NScan"],
                           par["NC"], par["dimY"], par["dimX"])
        self.data_shape_T = (par["NScan"], par["NC"],
                             self.packs*self.numofpacks,
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
            SMS=True)

    def _updateInitial(self,
                       out_fwd, out_adj,
                       in_primal, in_dual):
        out_fwd["Ax"] = self._op.fwdoop(
            [[in_primal["x"], self._coils, self.modelgrad]])
        self._op.adjKyk1(
            [out_adj["Kyk1"]],
            [[in_dual["r"], in_dual["z1"], self._coils, self.modelgrad, []]],
            [[self.grad_op._ratio]])

        self.symgrad_op.fwd(
            [out_fwd["symgradx"]],
            [[in_primal["v"]]])

        self.stream_initial_2.eval(
            [out_fwd["gradx"],
             out_adj["Kyk1"]],
            [[in_primal["x"]],
             [in_dual["z2"], in_dual["z1"], []]])

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
            [[self.grad_op._ratio]])

        (lhs4, ynorm4) = self.update_dual_2.evalwithnorm(
            [out_dual["z2"],
             out_adj["Kyk2"]],
            [[in_dual["z2"], in_precomp_fwd_new["symgradx"],
              in_precomp_fwd["symgradx"]],
             [[], out_dual["z1"], in_precomp_adj["Kyk2"]]],
            [[beta*tau, theta, self.beta],
             []])

        ynorm = np.abs(ynorm1+ynorm2+ynorm3+ynorm4)**(1/2)
        lhs = np.sqrt(beta)*tau*np.abs(lhs1+lhs2+lhs3+lhs4)**(1/2)

        return lhs, ynorm


class PDSolverStreamedTV(PDSolverStreamed):
    """Streamed TV optimization."""

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 linop, coils, model, imagespace=False, SMS=False):

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
            imagespace=imagespace)

        self.alpha = irgn_par["gamma"]
        self.beta = irgn_par["gamma"] * 2
        self._op = linop[0]
        self.grad_op = linop[1]
        self.symgrad_op = linop[2]

        self.symgrad_shape = self.unknown_shape + (8,)

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
            dtype=DTYPE)
        dual_vars_new["r"] = np.zeros_like(dual_vars["r"])

        dual_vars["z1"] = np.zeros(primal_vars["x"].shape+(4,),
                                   dtype=DTYPE)
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
             [self.grad_op._ratio]])

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
              self.modelgrad],
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
                [self.grad_op._ratio]
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
                in_primal["x"] - in_primal["xk"],
                in_primal["x"] - in_primal["xk"]
                )
            ).real

        dual = (
            -self.delta / 2 * np.vdot(
                - in_precomp_adj["Kyk1"],
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
    """Streamed TGV optimization for SMS data."""

    def __init__(self, par, irgn_par, queue, tau, fval, prg,
                 linop, coils, model, imagespace=False):



        self.packs = par["packs"]
        self.numofpacks = par["numofpacks"]
        self.data_shape = (self.packs*self.numofpacks, par["NScan"],
                           par["NC"], par["dimY"], par["dimX"])
        self.data_shape_T = (par["NScan"], par["NC"],
                             self.packs*self.numofpacks,
                             par["dimY"], par["dimX"])

        self._setupstreamingops = self._setupstreamingopsSMS

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
            SMS=True)

    def _updateInitial(self,
                       out_fwd, out_adj,
                       in_primal, in_dual):
        out_fwd["Ax"] = self._op.fwdoop(
            [[in_primal["x"], self._coils, self.modelgrad]])
        self._op.adjKyk1(
            [out_adj["Kyk1"]],
            [[in_dual["r"], in_dual["z1"], self._coils, self.modelgrad, []]],
            [[self.grad_op._ratio]])

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
            [[self.grad_op._ratio]])

        ynorm = np.abs(ynorm1+ynorm2+ynorm3)**(1/2)
        lhs = np.sqrt(beta)*tau*np.abs(lhs1+lhs2+lhs3)**(1/2)

        return lhs, ynorm
