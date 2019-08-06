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
#import sys
DTYPE = np.complex64
DTYPE_real = np.float32


class CGSolver:
    """ Conjugate Gradient Optimization Algorithm

    This Class performs a CG reconstruction on single precission complex input
    data.
    """
    def __init__(self, par, NScan=1, trafo=1, SMS=0):
        """ Setup a CG reconstruction Object

        Args:
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
        self._prg = Program(
            par["ctx"][0],
            open(
                resource_filename(
                    'pyqmri', 'kernels/OpenCL_Kernels.c')).read())
        self._coil_buf = cl.Buffer(par["ctx"][0],
                                   cl.mem_flags.READ_ONLY |
                                   cl.mem_flags.COPY_HOST_PTR,
                                   hostbuf=par["C"].data)
        if SMS:
            self._op = operator.OperatorKspaceSMS(par, self._prg, trafo)
        else:
            self._op = operator.OperatorKspace(par, self._prg, trafo)
        self._FT = self._op.NUFFT.FFT
        self._FTH = self._op.NUFFT.FFTH
        self._tmp_result = clarray.empty(
                          self._queue,
                          (self._NScan, self._NC,
                           self._NSlice, self._dimY, self._dimX),
                          DTYPE, "C")
        self._tmp_sino = clarray.empty(
                          self._queue,
                          (self._NScan, self._NC,
                           self._NSlice, par["Nproj"], par["N"]),
                          DTYPE, "C")
        par["NScan"] = NScan_save

    def __del__(self):
        """ Destructor
        Releases GPU memory arrays.
        """
        del self._coil_buf
        del self._tmp_result
        del self._queue
        del self._tmp_sino
        del self._FT
        del self._FTH

    def run(self, data, iters=30, lambd=1e-5, tol=1e-8, guess=None):
        """ Start the CG reconstruction

        All attributes after data are considered keyword only.

        Args:
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
        """ Apply forward operator for image reconstruction.

        Args:
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
                                         y.data, x.data, self._coil_buf,
                                         np.int32(self._NC),
                                         np.int32(self._NScan),
                                         wait_for=wait_for)

    def operator_lhs(self, out, x, wait_for=[]):
        """ Compute the left hand side of the CG equation

        Args:
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
            self._tmp_sino, self._tmp_result))
        return self.operator_rhs(out, self._tmp_sino)

    def operator_rhs(self, out, x, wait_for=[]):
        """ Compute the right hand side of the CG equation

        Args:
          out (PyOpenCL.Array):
            The result of the computation
          x (PyOpenCL.Array):
            The input array
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        """
        self._tmp_result.add_event(self._FTH(
            self._tmp_result, x, wait_for=wait_for+x.events))
        return self._prg.operator_ad_cg(self._queue,
                                        (self._NSlice, self._dimY,
                                         self._dimX),
                                        None,
                                        out.data, self._tmp_result.data,
                                        self._coil_buf, np.int32(self._NC),
                                        np.int32(self._NScan),
                                        wait_for=(self._tmp_result.events +
                                                  out.events+wait_for))
#
#
#class PDSolver:
#    """ Conjugate Gradient Optimization Algorithm
#
#    This Class performs a CG reconstruction on single precission complex input
#    data.
#    """
#    def __init__(self, par, irgn_par, tau, A, AT, updatePrimal, updateDual):
#        """ Setup a CG reconstruction Object
#
#        Args:
#          par (dict): A python dict containing the necessary information to
#            setup the object. Needs to contain the number of slices (NSlice),
#            number of scans (NScan), image dimensions (dimX, dimY), number of
#            coils (NC), sampling points (N) and read outs (NProj)
#            a PyOpenCL queue (queue) and the complex coil
#            sensitivities (C).
#          NScan (int): Number of Scan which should be used internally. Do not
#            need to be the same number as in par["NScan"]
#          trafo (bool): Switch between radial (1) and Cartesian (0) fft.
#          SMS (bool): Simultaneouos Multi Slice. Switch between noraml (0)
#          and slice accelerated (1) reconstruction.
#        """
#        self.alpha = irgn_par["gamma"]
#        self.beta = irgn_par["gamma"] * 2
#        self.delta = irgn_par["delta"]
#        self.omega = irgn_par["omega"]
#        self.mu = 1 / self.delta
#        self.tau = tau
#        self.beta_line = 400
#        self.theta_line = np.float32(1.0)
#        self.A = A
#        self.AT = AT
#        self.updatePrimal = updatePrimal
#        self.updateDual = updateDual
#
#    def __del__(self):
#        """ Destructor
#        Releases GPU memory arrays.
#        """
#
#    def run(self, guess, data, iters):
#
#        tau = self.tau
#        tau_new = np.float32(0)
#
#        primal = []
#        primal_new = []
#        AT = []
#        AT_new = []
#        for j in range(len(guess)):
#            primal.append(clarray.to_device(self._queue, guess[j]))
#            primal_new.append(clarray.empty_like(primal[j]))
#            AT.append(clarray.empty_like(primal[j]))
#            AT_new.append(clarray.empty_like(primal[j]))
#
#        xk = primal[0].copy()
#
#        dual = []
#        dual_new = []
#        A = []
#        A_new = []
#        for j in range(len(self.A)):
#            dual.append(clarray.to_device(self._queue,
#                                          np.zeros(self.A[j].out_shape,
#                                                   dtype=DTYPE)))
#            dual_new.append(clarray.empty_like(dual[j]))
#            A.append(clarray.empty_like(dual[j]))
#            A_new.append(clarray.empty_like(dual[j]))
#
#        data = clarray.to_device(self._queue, data.astype(DTYPE))
#
#        theta_line = self.theta_line
#        beta_line = self.beta_line
#        beta_new = np.float32(0)
#        mu_line = np.float32(0.5)
#        delta_line = np.float32(1)
#        ynorm = np.float32(0.0)
#        lhs = np.float32(0.0)
#        primal = np.float32(0.0)
#        primal_new = np.float32(0)
#        dual = np.float32(0.0)
#        gap_init = np.float32(0.0)
#        gap_old = np.float32(0.0)
#        gap = np.float32(0.0)
#
#        self._eval_const()
#
#        for j in range(len(self.A)):
#            A[j].add_event(self.A[j](A[j], primal))
#
#        for j in range(len(self.AT)):
#            AT[j].add_event(self.AT[j](AT[j],
#                                       dual))
#
#        for i in range(iters):
#            for j in range(len(primal)):
#                primal_new[j].add_event(self.updatePrimal[j](
#                    primal_new[j], primal, AT, tau, reg_par))
#
#            beta_new = beta_line * (1 + self.mu * tau)
#            tau_new = tau * np.sqrt(beta_line / beta_new * (1 + theta_line))
#            beta_line = beta_new
#
#            for j in range(len(self.A)):
#                A_new[j].add_event(self.A[j](A_new[j], primal_new))
#
#            while True:
#                theta_line = tau_new / tau
#                for j in range(len(dual)):
#                    dual_new[j].add_event(self.updateDual[j](
#                    primal_new[j], primal, AT, tau, reg_par))
#
#                z1_new.add_event(self.update_z1(
#                    z1_new, z1, gradx, gradx_xold, v_new, v,
#                    beta_line * tau_new, theta_line, self.alpha,
#                    self.omega))
#                z2_new.add_event(self.update_z2(
#                    z2_new, z2, symgrad_v, symgrad_v_vold,
#                    beta_line * tau_new, theta_line, self.beta))
#                r_new.add_event(self.update_r(
#                    r_new, r, Ax, Axold, data,
#                    beta_line * tau_new, theta_line, self.irgn_par["lambd"]))
#
#                for j in range(len(self.AT)):
#                    AT_new[j].add_event(self.AT[j](AT_new[j],
#                                                   dual_new))
#
#                ynorm = (
#                    (clarray.vdot(r_new - r, r_new - r) +
#                     clarray.vdot(z1_new - z1, z1_new - z1) +
#                     clarray.vdot(z2_new - z2, z2_new - z2))**(1 / 2)).real
#                lhs = np.sqrt(beta_line) * tau_new * (
#                    (clarray.vdot(Kyk1_new - Kyk1, Kyk1_new - Kyk1) +
#                     clarray.vdot(Kyk2_new - Kyk2, Kyk2_new - Kyk2))**(1 / 2)
#                    ).real
#
#                if lhs <= ynorm * delta_line:
#                    break
#                else:
#                    tau_new = tau_new * mu_line
#
#            (Kyk1, Kyk1_new, Kyk2, Kyk2_new, Axold, Ax, z1, z1_new,
#             z2, z2_new, r, r_new, gradx_xold, gradx, symgrad_v_vold,
#             symgrad_v, tau) = (
#             Kyk1_new, Kyk1, Kyk2_new, Kyk2, Ax, Axold, z1_new, z1,
#             z2_new, z2, r_new, r, gradx, gradx_xold, symgrad_v,
#             symgrad_v_vold, tau_new)
#
#            if not np.mod(i, 50):
#                if self.irgn_par["display_iterations"]:
#                    self.model.plot_unknowns(x_new.get())
#                if self.par["unknowns_H1"] > 0:
#                    primal_new = (
#                        self.irgn_par["lambd"] / 2 *
#                        clarray.vdot(Axold - data, Axold - data) +
#                        self.alpha * clarray.sum(
#                            abs((gradx[:self.par["unknowns_TGV"]] - v))) +
#                        self.beta * clarray.sum(abs(symgrad_v)) +
#                        1 / (2 * self.delta) * clarray.vdot(
#                            x_new - xk, x_new - xk) +
#                        self.irgn_par["omega"] / 2 *
#                        clarray.vdot(gradx[self.par["unknowns_TGV"]:],
#                                     gradx[self.par["unknowns_TGV"]:])).real
#
#                    dual = (
#                        -self.delta / 2 * clarray.vdot(-Kyk1, -Kyk1)
#                        - clarray.vdot(xk, (-Kyk1)) + clarray.sum(Kyk2)
#                        - 1 / (2 * self.irgn_par["lambd"]) * clarray.vdot(r, r)
#                        - clarray.vdot(data, r)
#                        - 1 / (2 * self.irgn_par["omega"])
#                        * clarray.vdot(z1[self.par["unknowns_TGV"]:],
#                                       z1[self.par["unknowns_TGV"]:])).real
#                else:
#                    primal_new = (
#                        self.irgn_par["lambd"] / 2 *
#                        clarray.vdot(Axold - data, Axold - data) +
#                        self.alpha * clarray.sum(
#                            abs((gradx - v))) +
#                        self.beta * clarray.sum(abs(symgrad_v)) +
#                        1 / (2 * self.delta) * clarray.vdot(
#                            x_new - xk, x_new - xk)).real
#
#                    dual = (
#                        -self.delta / 2 * clarray.vdot(-Kyk1, -Kyk1)
#                        - clarray.vdot(xk, (-Kyk1)) + clarray.sum(Kyk2)
#                        - 1 / (2 * self.irgn_par["lambd"]) * clarray.vdot(r, r)
#                        - clarray.vdot(data, r)).real
#
#                gap = np.abs(primal_new - dual)
#                if i == 0:
#                    gap_init = gap.get()
#                if np.abs(primal - primal_new)/self._fval_init <\
#                   self.irgn_par["tol"]:
#                    print("Terminated at iteration %d because the energy "
#                          "decrease in the primal problem was less than %.3e" %
#                          (i,
#                           np.abs(primal - primal_new).get()/self._fval_init))
#                    self.v = v_new.get()
#                    return (x_new.get(), v_new.get())
#                if gap > gap_old * self.irgn_par["stag"] and i > 1:
#                    self.v = v_new.get()
#                    print("Terminated at iteration %d "
#                          "because the method stagnated" % (i))
#                    return (x_new.get(), v_new.get())
#                if np.abs((gap - gap_old) / gap_init) < self.irgn_par["tol"]:
#                    self.v = v_new.get()
#                    print("Terminated at iteration %d because the "
#                          "relative energy decrease of the PD gap was "
#                          "less than %.3e"
#                          % (i, np.abs((gap - gap_old).get() / gap_init)))
#                    return (x_new.get(), v_new.get())
#                primal = primal_new
#                gap_old = gap
#                sys.stdout.write(
#                    "Iteration: %04d ---- Primal: %2.2e, "
#                    "Dual: %2.2e, Gap: %2.2e \r" %
#                    (i, 1000*primal.get() / self._fval_init,
#                     1000*dual.get() / self._fval_init,
#                     1000*gap.get() / self._fval_init))
#                sys.stdout.flush()
#
#            (x, x_new) = (x_new, x)
#            (v, v_new) = (v_new, v)
#        return (x.get(), v.get())
#
#    def _eval_const(self):
#        num_const = (len(self.model.constraints))
#        self.min_const = np.zeros((num_const), dtype=np.float32)
#        self.max_const = np.zeros((num_const), dtype=np.float32)
#        self.real_const = np.zeros((num_const), dtype=np.int32)
#        for j in range(num_const):
#            self.min_const[j] = np.float32(self.model.constraints[j].min)
#            self.max_const[j] = np.float32(self.model.constraints[j].max)
#            self.real_const[j] = np.int32(self.model.constraints[j].real)
#        self.min_const = clarray.to_device(self._queue, self.min_const)
#        self.max_const = clarray.to_device(self._queue, self.max_const)
#        self.real_const = clarray.to_device(self._queue, self.real_const)
