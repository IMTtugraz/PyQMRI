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
DTYPE = np.complex64
DTYPE_real = np.float32


class CGSolver:
    """ Conjugate Gradient Optimization Algorithm

    This Class performs a CG reconstruction on single precission complex input
    data.
    """
    def __init__(self, par, NScan=1, trafo=1):
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
        self._coil_buf = cl.Buffer(par["ctx"][0],
                                   cl.mem_flags.READ_ONLY |
                                   cl.mem_flags.COPY_HOST_PTR,
                                   hostbuf=par["C"].data)
        self._op = operator.OperatorKspace(par, self._prg, trafo=trafo)
        self._FT = self._op.NUFFT.FFT
        self._FTH = self._op.NUFFT.FFTH
        self._grad = operator.OperatorFiniteGradient(par, self._prg)
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

    def run(self, data, iters=50, lambd=1e-9, tol=1e-8, guess=None):
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

        for i in range(iters):
            self.operator_lhs(Ax, p)
            Ax = Ax + lambd*p
            alpha = (clarray.vdot(res, res) /
                     (clarray.vdot(p, Ax))).real.get()
            x = (x + alpha*p)
            res_new = res - alpha*Ax
            delta = np.linalg.norm(res_new.get())**2 /\
                np.linalg.norm(b.get())**2
#            print("Residum: %f" % (delta))
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


class PDSolver:
    """ Conjugate Gradient Optimization Algorithm

    This Class performs a CG reconstruction on single precission complex input
    data.
    """
    def __init__(self, par, irgn_par, queue, tau, fval, prg, reg_type,
                 data_operator, coil_buffer):
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
          and slice accelerated (1) reconstruction.
        """
        self.alpha = irgn_par["gamma"]
        self.beta = irgn_par["gamma"] * 2
        self.delta = irgn_par["delta"]
        self.omega = irgn_par["omega"]
        self.lambd = irgn_par["lambd"]
        self.tol = irgn_par["tol"]
        self.stag = irgn_par["stag"]
        self.display_iterations = irgn_par["display_iterations"]
        self.mu = 1 / self.delta
        self.tau = tau
        self.beta_line = 400
        self.theta_line = np.float32(1.0)
        self.unknowns_TGV = par["unknowns_TGV"]
        self.unknowns_H1 = par["unknowns_H1"]
        self.unknowns = par["unknowns"]
        self.dz = par["dz"]
        self._fval_init = fval
        self._prg = prg
        self._queue = queue
        self._op = data_operator
        self._coil_buf = coil_buffer
        self.grad_buf = None
        self.model = None
        self.grad_op = None
        self.symgrad_op = None
        self.min_const = None
        self.max_const = None
        self.real_const = None
        self.irgn_par = {}
        if reg_type == 'TV':
            self.run = self.runTV3D
        elif reg_type == 'TGV':
            self.run = self.runTGV3D
        else:
            raise ValueError("Unknown Regularization Type")

    def __del__(self):
        """ Destructor
        Releases GPU memory arrays.
        """

    def runTGV3D(self, x, data, iters):
        self._updateConstraints()
        tau = self.tau
        tau_new = np.float32(0)

        x = clarray.to_device(self._queue, x)
        v = clarray.zeros(self._queue, x.shape+(4,), dtype=DTYPE)

        x_new = clarray.empty_like(x)
        v_new = clarray.empty_like(v)

        r = clarray.zeros(self._queue, data.shape, dtype=DTYPE)
        z1 = clarray.zeros_like(v)
        z2 = clarray.zeros(self._queue, x.shape+(8,), dtype=DTYPE)

        r_new = clarray.empty_like(r)
        z1_new = clarray.empty_like(z1)
        z2_new = clarray.empty_like(z2)

        xk = x.copy()
        data = clarray.to_device(self._queue, data.astype(DTYPE))

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

        Kyk1 = clarray.empty_like(x)
        Kyk1_new = clarray.empty_like(x)
        Kyk2 = clarray.empty_like(z1)
        Kyk2_new = clarray.empty_like(z1)
        gradx = clarray.empty_like(z1)
        gradx_xold = clarray.empty_like(z1)
        symgrad_v = clarray.empty_like(z2)
        symgrad_v_vold = clarray.empty_like(z2)
        Axold = clarray.empty_like(data)
        Ax = clarray.empty_like(data)

        Axold.add_event(self._op.fwd(
            Axold, [x, self._coil_buf, self.grad_buf]))
        gradx_xold.add_event(self.grad_op.fwd(gradx_xold, x))

        symgrad_v_vold.add_event(self.symgrad_op.fwd(symgrad_v_vold, v))

        Kyk1.add_event(self._op.adjKyk1(Kyk1,
                                        [r, z1,
                                         self._coil_buf,
                                         self.grad_buf,
                                         self.grad_op._ratio]))
        Kyk2.add_event(self.update_Kyk2(Kyk2, z2, z1))
        for i in range(iters):
            x_new.add_event(self.update_primal(x_new, x, Kyk1,
                                               xk, tau, self.delta))
            v_new.add_event(self.update_v(v_new, v, Kyk2, tau))

            beta_new = beta_line * (1 + self.mu * tau)
            tau_new = tau * np.sqrt(beta_line / beta_new * (1 + theta_line))
            beta_line = beta_new

            gradx.add_event(self.grad_op.fwd(gradx, x_new))
            symgrad_v.add_event(self.symgrad_op.fwd(symgrad_v, v_new))
            Ax.add_event(self._op.fwd(Ax,
                                      [x_new, self._coil_buf, self.grad_buf]))

            while True:
                theta_line = tau_new / tau

                z1_new.add_event(self.update_z1(
                    z1_new, z1, gradx, gradx_xold, v_new, v,
                    beta_line * tau_new, theta_line, self.alpha,
                    self.omega))
                z2_new.add_event(self.update_z2(
                    z2_new, z2, symgrad_v, symgrad_v_vold,
                    beta_line * tau_new, theta_line, self.beta))
                r_new.add_event(self.update_r(
                    r_new, r, Ax, Axold, data,
                    beta_line * tau_new, theta_line, self.lambd))

                Kyk1_new.add_event(self._op.adjKyk1(Kyk1_new,
                                                    [r_new, z1_new,
                                                     self._coil_buf,
                                                     self.grad_buf,
                                                     self.grad_op._ratio]))
                Kyk2_new.add_event(self.update_Kyk2(Kyk2_new, z2_new, z1_new))

                ynorm = (
                    (clarray.vdot(r_new - r, r_new - r) +
                     clarray.vdot(z1_new - z1, z1_new - z1) +
                     clarray.vdot(z2_new - z2, z2_new - z2))**(1 / 2)).real
                lhs = np.sqrt(beta_line) * tau_new * (
                    (clarray.vdot(Kyk1_new - Kyk1, Kyk1_new - Kyk1) +
                     clarray.vdot(Kyk2_new - Kyk2, Kyk2_new - Kyk2))**(1 / 2)
                    ).real

                if lhs <= ynorm * delta_line:
                    break
                else:
                    tau_new = tau_new * mu_line

            (Kyk1, Kyk1_new, Kyk2, Kyk2_new, Axold, Ax, z1, z1_new,
             z2, z2_new, r, r_new, gradx_xold, gradx, symgrad_v_vold,
             symgrad_v, tau) = (
                 Kyk1_new, Kyk1, Kyk2_new, Kyk2, Ax, Axold, z1_new, z1,
                 z2_new, z2, r_new, r, gradx, gradx_xold, symgrad_v,
                 symgrad_v_vold, tau_new)

            if not np.mod(i, 50):
                if self.display_iterations:
                    self.model.plot_unknowns(x_new.get())
                if self.unknowns_H1 > 0:
                    primal_new = (
                        self.lambd / 2 *
                        clarray.vdot(Axold - data, Axold - data) +
                        self.alpha * clarray.sum(
                            abs((gradx[:self.unknowns_TGV] - v))) +
                        self.beta * clarray.sum(abs(symgrad_v)) +
                        1 / (2 * self.delta) * clarray.vdot(
                            x_new - xk, x_new - xk) +
                        self.omega / 2 *
                        clarray.vdot(gradx[self.unknowns_TGV:],
                                     gradx[self.unknowns_TGV:])).real

                    dual = (
                        -self.delta / 2 * clarray.vdot(-Kyk1, -Kyk1)
                        - clarray.vdot(xk, (-Kyk1)) + clarray.sum(Kyk2)
                        - 1 / (2 * self.lambd) * clarray.vdot(r, r)
                        - clarray.vdot(data, r)
                        - 1 / (2 * self.omega)
                        * clarray.vdot(z1[self.unknowns_TGV:],
                                       z1[self.unknowns_TGV:])).real
                else:
                    primal_new = (
                        self.lambd / 2 *
                        clarray.vdot(Axold - data, Axold - data) +
                        self.alpha * clarray.sum(
                            abs((gradx - v))) +
                        self.beta * clarray.sum(abs(symgrad_v)) +
                        1 / (2 * self.delta) * clarray.vdot(
                            x_new - xk, x_new - xk)).real

                    dual = (
                        -self.delta / 2 * clarray.vdot(-Kyk1, -Kyk1)
                        - clarray.vdot(xk, (-Kyk1)) + clarray.sum(Kyk2)
                        - 1 / (2 * self.lambd) * clarray.vdot(r, r)
                        - clarray.vdot(data, r)).real

                gap = np.abs(primal_new - dual)
                if i == 0:
                    gap_init = gap.get()
                if np.abs(primal - primal_new)/self._fval_init <\
                   self.tol:
                    print("Terminated at iteration %d because the energy "
                          "decrease in the primal problem was less than %.3e" %
                          (i,
                           np.abs(primal - primal_new).get()/self._fval_init))
                    return x_new.get()
                if gap > gap_old * self.stag and i > 1:
                    print("Terminated at iteration %d "
                          "because the method stagnated" % (i))
                    return x_new.get()
                if np.abs((gap - gap_old) / gap_init) < self.tol:
                    print("Terminated at iteration %d because the "
                          "relative energy decrease of the PD gap was "
                          "less than %.3e"
                          % (i, np.abs((gap - gap_old).get() / gap_init)))
                    return x_new.get()
                primal = primal_new
                gap_old = gap
                sys.stdout.write(
                    "Iteration: %04d ---- Primal: %2.2e, "
                    "Dual: %2.2e, Gap: %2.2e \r" %
                    (i, 1000*primal.get() / self._fval_init,
                     1000*dual.get() / self._fval_init,
                     1000*gap.get() / self._fval_init))
                sys.stdout.flush()

            (x, x_new) = (x_new, x)
            (v, v_new) = (v_new, v)
        return x.get()

    def runTGV3DExplicit(self, x, res, iters):
        self._updateConstraints()
        alpha = self.irgn_par["gamma"]
        beta = self.irgn_par["gamma"] * 2

        tau = self.tau
        tau_new = np.float32(0)

        x = clarray.to_device(self._queue, x)
        xk = x.copy()
        x_new = clarray.empty_like(x)
        ATd = clarray.empty_like(x)

        z1 = clarray.zeros(self._queue, x.shape+(4,), dtype=DTYPE)
        z1_new = clarray.empty_like(z1)
        z2 = clarray.zeros(self._queue, x.shape+(8,), dtype=DTYPE)
        z2_new = clarray.empty_like(z2)
        v = clarray.zeros(self._queue, x.shape+(4,), dtype=DTYPE)
        v_new = clarray.empty_like(v)
        res = clarray.to_device(self._queue, res.astype(DTYPE))

        delta = self.irgn_par["delta"]
        omega = self.omega
        mu = 1 / delta

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

        Kyk1 = clarray.empty_like(x)
        Kyk1_new = clarray.empty_like(x)
        Kyk2 = clarray.empty_like(z1)
        Kyk2_new = clarray.empty_like(z1)
        gradx = clarray.empty_like(z1)
        gradx_xold = clarray.empty_like(z1)
        symgrad_v = clarray.empty_like(z2)
        symgrad_v_vold = clarray.empty_like(z2)
        AT = clarray.empty_like(res)

        AT.add_event(self._op.fwd(AT, [x, self._coil_buf, self.grad_buf]))
        ATd.add_event(self._op.adj(ATd, [x, self._coil_buf, self.grad_buf]))

        Kyk1.add_event(self.grad_op.adj(Kyk1, z1))
        Kyk2.add_event(self.update_Kyk2(Kyk2, z2, z1))

        for i in range(iters):

            x_new.add_event(
                self._op.adj(AT,
                             [x_new, self._coil_buf, self.grad_buf]))
            x_new.add_event(self.update_primal_explicit(
                x_new, x, Kyk1, xk, ATd,
                tau, delta, self.lambd))
            v_new.add_event(self.update_v(v_new, v, Kyk2, tau))

            beta_new = beta_line * (1 + mu * tau)
            tau_new = tau * np.sqrt(beta_line / beta_new * (1 + theta_line))

            beta_line = beta_new

            gradx.add_event(self.grad_op.fwd(gradx, x_new))
            gradx_xold.add_event(self.grad_op.fwd(gradx_xold, x))
            symgrad_v.add_event(self.symgrad_op.fwd(symgrad_v, v_new))
            symgrad_v_vold.add_event(self.symgrad_op.fwd(symgrad_v_vold, v))

            AT.add_event(self._op.fwd(AT,
                                      [x_new, self._coil_buf, self.grad_buf]))

            while True:
                theta_line = tau_new / tau
                z1_new.add_event(self.update_z1(
                    z1_new, z1, gradx, gradx_xold, v_new, v,
                    beta_line * tau_new, theta_line, alpha, omega))
                z2_new.add_event(self.update_z2(
                    z2_new, z2, symgrad_v, symgrad_v_vold,
                    beta_line * tau_new, theta_line, beta))

                Kyk1_new.add_event(self.grad_op.adj(Kyk1_new, z1_new))
                Kyk2_new.add_event(self.update_Kyk2(Kyk2_new, z2_new, z1_new))

                ynorm = ((clarray.vdot(z1_new - z1, z1_new - z1) +
                          clarray.vdot(z2_new - z2, z2_new - z2)
                          )**(1 / 2)).real
                lhs = 1e2 * np.sqrt(beta_line) * tau_new * (
                    (clarray.vdot(Kyk1_new - Kyk1, Kyk1_new - Kyk1) +
                     clarray.vdot(Kyk2_new - Kyk2, Kyk2_new - Kyk2)
                     )**(1 / 2)).real
                if lhs <= ynorm * delta_line:
                    break
                else:
                    tau_new = tau_new * mu_line

            (Kyk1, Kyk1_new, Kyk2, Kyk2_new, z1, z1_new, z2, z2_new, tau) = (
                Kyk1_new, Kyk1, Kyk2_new, Kyk2,
                z1_new, z1, z2_new, z2, tau_new)

            if not np.mod(i, 50):
                self.model.plot_unknowns(x_new.get())
                primal_new = (
                    self.lambd / 2 *
                    clarray.vdot(AT - res, AT - res) +
                    alpha * clarray.sum(
                        abs((gradx[:self.unknowns_TGV] - v))) +
                    beta * clarray.sum(abs(symgrad_v)) +
                    1 / (2 * delta) * clarray.vdot(x_new - xk, x_new - xk)
                    ).real

                dual = (
                    -delta / 2 * clarray.vdot(-Kyk1, -Kyk1)
                    - clarray.vdot(xk, (-Kyk1))
                    + clarray.sum(Kyk2)).real

                gap = np.abs(primal_new - dual)
                if i == 0:
                    gap_init = gap.get()
                if np.abs(primal - primal_new)/self._fval_init <\
                   self.tol:
                    print("Terminated at iteration %d because the energy "
                          "decrease in the primal problem was less than %.3e" %
                          (i,
                           np.abs(primal - primal_new).get()/self._fval_init))
                    return x_new.get()
                if gap > gap_old * self.stag and i > 1:
                    print("Terminated at iteration %d "
                          "because the method stagnated" % (i))
                    return x_new.get()
                if np.abs((gap - gap_old) / gap_init) < self.tol:
                    print("Terminated at iteration %d because the "
                          "relative energy decrease of the PD gap was "
                          "less than %.3e"
                          % (i, np.abs((gap - gap_old).get() / gap_init)))
                    return x_new.get()
                primal = primal_new
                gap_old = gap
                sys.stdout.write(
                    "Iteration: %04d ---- Primal: %2.2e, "
                    "Dual: %2.2e, Gap: %2.2e \r" %
                    (i, 1000*primal.get() / self._fval_init,
                     1000*dual.get() / self._fval_init,
                     1000*gap.get() / self._fval_init))
                sys.stdout.flush()

            (x, x_new) = (x_new, x)
            (v, v_new) = (v_new, v)

        self.v = v.get()
        self.z1 = z1.get()
        self.z2 = z2.get()
        return x.get()

    def runTV3D(self, x, data, iters):
        self._updateConstraints()

        tau = self.tau
        tau_new = np.float32(0)

        x = clarray.to_device(self._queue, x)
        x_new = clarray.empty_like(x)

        xk = x.copy()

        r = clarray.zeros(self._queue, data.shape, dtype=DTYPE)
        z1 = clarray.zeros(self._queue, x.shape+(4,), dtype=DTYPE)

        r_new = clarray.empty_like(r)
        z1_new = clarray.empty_like(z1)
        data = clarray.to_device(self._queue, data.astype(DTYPE))

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

        Kyk1 = clarray.empty_like(x)
        Kyk1_new = clarray.empty_like(x)
        gradx = clarray.empty_like(z1)
        gradx_xold = clarray.empty_like(z1)
        Axold = clarray.empty_like(data)
        Ax = clarray.empty_like(data)

        Axold.add_event(self._op.fwd(Axold,
                                     [x, self._coil_buf, self.grad_buf]))
        Kyk1.add_event(self._op.adjKyk1(Kyk1,
                                        [r, z1,
                                         self._coil_buf,
                                         self.grad_buf,
                                         self.grad_op._ratio]))

        for i in range(iters):
            x_new.add_event(self.update_primal(x_new, x, Kyk1, xk, tau,
                                               self.delta))

            beta_new = beta_line * (1 + self.mu * tau)
            tau_new = tau * np.sqrt(beta_line / beta_new * (1 + theta_line))
            beta_line = beta_new

            gradx.add_event(self.grad_op.fwd(
                gradx, x_new, wait_for=gradx.events + x_new.events))
            gradx_xold.add_event(self.grad_op.fwd(
                gradx_xold, x, wait_for=gradx_xold.events + x.events))
            Ax.add_event(self._op.fwd(Ax,
                                      [x_new, self._coil_buf, self.grad_buf]))

            while True:
                theta_line = tau_new / tau
                z1_new.add_event(self.update_z1_tv(
                    z1_new, z1, gradx, gradx_xold,
                    beta_line * tau_new, theta_line, self.alpha, self.omega))
                r_new.add_event(self.update_r(
                    r_new, r, Ax, Axold, data,
                    beta_line * tau_new, theta_line, self.lambd))

                Kyk1_new.add_event(self._op.adjKyk1(Kyk1_new,
                                                    [r_new, z1_new,
                                                     self._coil_buf,
                                                     self.grad_buf,
                                                     self.grad_op._ratio]))

                ynorm = ((clarray.vdot(r_new - r, r_new - r) +
                          clarray.vdot(z1_new - z1, z1_new - z1))**(1 / 2)
                         ).real
                lhs = np.sqrt(beta_line) * tau_new * (
                    (clarray.vdot(Kyk1_new - Kyk1, Kyk1_new - Kyk1))**(1 / 2)
                    ).real
                if lhs <= ynorm * delta_line:
                    break
                else:
                    tau_new = tau_new * mu_line

            (Kyk1, Kyk1_new, Axold, Ax, z1, z1_new, r, r_new, gradx_xold,
             gradx, tau) = (
                 Kyk1_new, Kyk1, Ax, Axold, z1_new, z1, r_new, r, gradx,
                 gradx_xold, tau_new)

            if not np.mod(i, 50):
                if self.display_iterations:
                    self.model.plot_unknowns(x_new.get())
                if self.unknowns_H1 > 0:
                    primal_new = (
                        self.lambd / 2 *
                        clarray.vdot(Axold - data, Axold - data) +
                        self.alpha * clarray.sum(
                            abs((gradx[:self.unknowns_TGV]))) +
                        1 / (2 * self.delta) *
                        clarray.vdot(x_new - xk, x_new - xk) +
                        self.omega / 2 *
                        clarray.vdot(gradx[self.unknowns_TGV:],
                                     gradx[self.unknowns_TGV:])).real
                    dual = (
                        -self.delta / 2 * clarray.vdot(-Kyk1, -Kyk1)
                        - clarray.vdot(xk, (-Kyk1))
                        - 1 / (2 * self.lambd) * clarray.vdot(r, r)
                        - clarray.vdot(data, r)
                        - 1 / (2 * self.omega)
                        * clarray.vdot(z1[self.unknowns_TGV:],
                                       z1[self.unknowns_TGV:])).real
                else:
                    primal_new = (
                        self.lambd / 2 *
                        clarray.vdot(Axold - data, Axold - data) +
                        self.alpha * clarray.sum(
                            abs((gradx[:self.unknowns_TGV]))) +
                        1 / (2 * self.delta) * clarray.vdot(x_new - xk,
                                                            x_new - xk)
                        ).real
                    dual = (
                        -self.delta / 2 * clarray.vdot(-Kyk1, -Kyk1)
                        - clarray.vdot(xk, (-Kyk1))
                        - 1 / (2 * self.lambd) * clarray.vdot(r, r)
                        - clarray.vdot(data, r)).real
                gap = np.abs(primal_new - dual)

                if i == 0:
                    gap_init = gap
                if np.abs(primal - primal_new)/self._fval_init < \
                   self.tol:
                    print("Terminated at iteration %d because the energy "
                          "decrease in the primal problem was less than %.3e" %
                          (i, np.abs(primal - primal_new).get() /
                           self._fval_init))
                    return x_new.get()
                if (gap > gap_old * self.stag) and i > 1:
                    print("Terminated at iteration %d "
                          "because the method stagnated" % (i))
                    return x_new.get()
                if np.abs((gap - gap_old) / gap_init) < self.tol:
                    print("Terminated at iteration %d because the relative "
                          "energy decrease of the PD gap was less than %.3e" %
                          (i, np.abs((gap - gap_old).get() / gap_init)))
                    return x_new.get()
                primal = primal_new
                gap_old = gap
                sys.stdout.write(
                    "Iteration: %04d ---- Primal: %2.2e, "
                    "Dual: %2.2e, Gap: %2.2e \r" %
                    (i, 1000 * primal.get() / self._fval_init,
                     1000 * dual.get() / self._fval_init,
                     1000 * gap.get() / self._fval_init))
                sys.stdout.flush()

            (x, x_new) = (x_new, x)

        return x.get()

    def _updateConstraints(self):
        num_const = (len(self.model.constraints))
        self.min_const = np.zeros((num_const), dtype=np.float32)
        self.max_const = np.zeros((num_const), dtype=np.float32)
        self.real_const = np.zeros((num_const), dtype=np.int32)
        for j in range(num_const):
            self.min_const[j] = np.float32(self.model.constraints[j].min)
            self.max_const[j] = np.float32(self.model.constraints[j].max)
            self.real_const[j] = np.int32(self.model.constraints[j].real)
        self.min_const = clarray.to_device(self._queue, self.min_const)
        self.max_const = clarray.to_device(self._queue, self.max_const)
        self.real_const = clarray.to_device(self._queue, self.real_const)

    def updateRegPar(self, irgn_par):
        self.alpha = irgn_par["gamma"]
        self.beta = irgn_par["gamma"] * 2
        self.delta = irgn_par["delta"]
        self.omega = irgn_par["omega"]
        self.lambd = irgn_par["lambd"]
        self.mu = 1/self.delta

    def update_primal(self, x_new, x, Kyk, xk, tau, delta, wait_for=[]):
        return self._prg.update_primal(
            self._queue, x.shape[1:], None, x_new.data, x.data, Kyk.data,
            xk.data, np.float32(tau),
            np.float32(tau / delta), np.float32(1 / (1 + tau / delta)),
            self.min_const.data, self.max_const.data,
            self.real_const.data, np.int32(self.unknowns),
            wait_for=(x_new.events + x.events +
                      Kyk.events + xk.events + wait_for))

    def update_v(self, v_new, v, Kyk2, tau, wait_for=[]):
        return self._prg.update_v(
            self._queue, (v[..., 0].size,), None, v_new.data, v.data,
            Kyk2.data, np.float32(tau),
            wait_for=v_new.events + v.events + Kyk2.events + wait_for)

    def update_z1(self, z_new, z, gx, gx_, vx, vx_,
                  sigma, theta, alpha, omega, wait_for=[]):
        return self._prg.update_z1(
            self._queue, z.shape[1:-1], None, z_new.data, z.data, gx.data,
            gx_.data, vx.data, vx_.data, np.float32(sigma), np.float32(theta),
            np.float32(1 / alpha), np.int32(self.unknowns_TGV),
            np.int32(self.unknowns_H1),
            np.float32(1 / (1 + sigma / omega)),
            wait_for=(
                z_new.events + z.events + gx.events +
                gx_.events + vx.events + vx_.events + wait_for))

    def update_z1_tv(self, z_new, z, gx, gx_,
                     sigma, theta, alpha, omega, wait_for=[]):
        return self._prg.update_z1_tv(
            self._queue, z.shape[1:-1], None, z_new.data, z.data, gx.data,
            gx_.data, np.float32(sigma), np.float32(theta),
            np.float32(1 / alpha), np.int32(self.unknowns_TGV),
            np.int32(self.unknowns_H1),
            np.float32(1 / (1 + sigma / omega)),
            wait_for=(
                z_new.events + z.events + gx.events + gx_.events + wait_for))

    def update_z2(self, z_new, z, gx, gx_, sigma, theta, beta, wait_for=[]):
        return self._prg.update_z2(
            self._queue, z.shape[1:-1], None, z_new.data, z.data,
            gx.data, gx_.data, np.float32(sigma), np.float32(theta),
            np.float32(1 / beta), np.int32(self.unknowns_TGV),
            wait_for=(
                z_new.events + z.events + gx.events + gx_.events + wait_for))

    def update_Kyk2(self, div, u, z, wait_for=[]):
        return self._prg.update_Kyk2(
            self._queue, u.shape[1:-1], None, div.data, u.data, z.data,
            np.int32(self.unknowns_TGV), np.float32(self.dz),
            wait_for=div.events + u.events + z.events + wait_for)

    def update_r(self, r_new, r, A, A_, res, sigma, theta, lambd, wait_for=[]):
        return self._prg.update_r(
            self._queue, (r.size,), None, r_new.data, r.data, A.data, A_.data,
            res.data, np.float32(sigma), np.float32(theta),
            np.float32(1 / (1 + sigma / lambd)),
            wait_for=r_new.events + r.events + A.events + A_.events + wait_for)

    def update_primal_explicit(self, x_new, x, Kyk, xk, ATd,
                               tau, delta, lambd, wait_for=[]):
        return self._prg.update_primal_explicit(
            self._queue, x.shape[1:], None, x_new.data, x.data, Kyk.data,
            xk.data, ATd.data, np.float32(tau),
            np.float32(1 / delta), np.float32(lambd), self.min_const.data,
            self.max_const.data,
            self.real_const.data, np.int32(self.unknowns),
            wait_for=(
                x_new.events + x.events + Kyk.events +
                xk.events + ATd.events + wait_for))
