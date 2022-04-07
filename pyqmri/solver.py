#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the classes for different numerical Optimizer."""

from __future__ import division
import sys
import numpy as np
import scipy.special as sps
from pkg_resources import resource_filename
import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.reduction as clred
import pyqmri.operator as operator
from pyqmri._helper_fun import CLProgram as Program
import pyqmri.streaming as streaming
import faulthandler; faulthandler.enable()


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
        if par["DTYPE"] == np.complex64:
            file = open(
                resource_filename(
                    'pyqmri', 'kernels/OpenCL_Kernels.c'))
        else:
            file = open(
                resource_filename(
                    'pyqmri', 'kernels/OpenCL_Kernels_double.c'))
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
            self._tmp_sino = clarray.zeros(
                self._queue,
                (self._NScan, self._NC,
                 int(self._NSlice/par["MB"]), par["Nproj"], par["N"]),
                self._DTYPE, "C")
        else:
            self._tmp_sino = clarray.zeros(
                self._queue,
                (self._NScan, self._NC,
                 self._NSlice, par["Nproj"], par["N"]),
                self._DTYPE, "C")
        self._FT = FT.FFT
        self._FTH = FT.FFTH
        self._tmp_result = clarray.zeros(
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
        b = clarray.zeros(self._queue,
                          (self._NScan, 1,
                           self._NSlice, self._dimY, self._dimX),
                          self._DTYPE, "C")
        Ax = clarray.zeros(self._queue,
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
            p = res_new + beta * p
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

class CGSolver_H1:
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
      irgn_par : dict
        A python dict containing the regularization
        parameters for a given gauss newton step.
      queue : list of PyOpenCL.Queues
        A list of PyOpenCL queues to perform the optimization.
      prg : PyOpenCL Program A PyOpenCL Program containing the
        kernels for optimization.
      linops : PyQMRI Operator The operator to traverse from
        parameter to data space.
      coils : PyOpenCL Buffer or empty list
        coil buffer, empty list if image based fitting is used.
    """

    def __init__(self, prg, queue, par, irgn_par, coils, linops):
        self._NSlice = par["NSlice"]
        self._NScan = par["NScan"]
        self._dimX = par["dimX"]
        self._dimY = par["dimY"]
        self._NC = par["NC"]
        self._DTYPE = par["DTYPE"]
        self._DTYPE_real = par["DTYPE_real"]
        self.unknowns = par["unknowns"]
        self._op = linops[0]
        self._grad_op = linops[1]
        self._queue = queue
        self.tol = irgn_par["tol"]
        self._coils = coils
        self.display_iterations = irgn_par["display_iterations"]
        self._prg = prg
        self.num_dev = len(par["num_dev"])

        self._tmp_result = None


    def run(self, guess, data, iters=30):
        """
        Start the CG reconstruction.

        All attributes after data are considered keyword only.

        Parameters
        ----------
          guess : numpy.array
            An optional initial guess for the images. If None, zeros is used.
          data : numpy.array
            The complex k-space data which serves as the basis for the images.
          iters : int
            Maximum number of CG iterations

        Returns
        -------
          dict of numpy.Array:
              The result of the fitting.
        """
        self._updateConstraints()
        self._tmp_result = clarray.zeros(
                    self._queue[0],
                    data.shape,
                    self._DTYPE, "C")
        if guess is not None:
            x = clarray.to_device(self._queue[0], guess[0])
        else:
            x = clarray.zeros(self._queue[0],
                              (self.unknowns,
                               self._NSlice, self._dimY, self._dimX),
                              self._DTYPE, "C")
        x_old = x.copy()
        y = x.copy()
        b = clarray.zeros(self._queue[0],
                          (self.unknowns,
                           self._NSlice, self._dimY, self._dimX),
                          self._DTYPE, "C")
        Ax = clarray.zeros(self._queue[0],
                           (self.unknowns,
                            self._NSlice, self._dimY, self._dimX),
                           self._DTYPE, "C")

        tau1 = self.lambd*(self.power_iteration(x))
        tau2 = self.alpha*(self.power_iteration_grad(x))
        print("Tau1: ", tau1, " Tau2: ", tau2)
        tau = self._DTYPE_real(1/(tau1+tau2+1/self.delta))

        x0 = x.copy()
        data = clarray.to_device(self._queue[0], data)
        self._operator_rhs(b, data).wait()
        # theta = 0.1


        for i in range(iters):
            y = x + self._DTYPE_real((iters-2)/(iters+1))*(x-x_old)
            self._operator_lhs(Ax, y).wait()
            # imgrad = self._gradop.fwdoop(y)
            # tmpsum = np.sum(np.abs(imgrad.get()),axis=-1)**2
            # tmpsum = clarray.to_device(self._queue[0],tmpsum)
            # grad = self._DTYPE_real(self.lambd)*(Ax-b) - self._DTYPE_real(self.alpha)*self._gradop.adjoop(imgrad)/(1+tmpsum) + self._DTYPE_real(1/self.delta)*(y-x0)
            grad = self._DTYPE_real(self.lambd)*(Ax-b) - self._DTYPE_real(self.alpha)*self._grad_op.adjoop(self._grad_op.fwdoop(y)) + self._DTYPE_real(1/self.delta)*(y-x0)

            x_old = (y - tau*grad)
            (x, x_old) = (x_old, x)
            self.update_box(x, [x], []).wait()

            # sigma = np.linalg.norm((x-x_old).get())/np.sqrt(
            #     self.lambd**4*np.linalg.norm(self._op.fwdoop([x-x_old, self._coils, self.modelgrad]).get())**2
            #     + self.alpha**4*np.linalg.norm(self._gradop.fwdoop(x-x_old).get())**2
            #     + 1/self.delta**4*np.linalg.norm((x-x_old-x0).get())**2)

            # if theta*tau >= sigma:
            #     tau = sigma
            # elif tau >= sigma > theta*tau:
            #     tau = np.sqrt(theta*tau)
            # else:
            #     tau = np.sqrt(tau)
            # tau = self._DTYPE_real(tau)



            if not np.mod(i+1, 10):
                datres = self._DTYPE_real(self.lambd/2)*np.linalg.norm(self._op.fwdoop([x, self._coils, self.modelgrad]).get()-data.get())**2
                regres = self._DTYPE_real(self.alpha/2)*np.linalg.norm(self._grad_op.fwdoop(x).get())**2
                l2res = self._DTYPE_real(1/self.delta/2)*np.linalg.norm((x-x0).get())**2
                sys.stdout.write(
                    "Iteration: %04d ---- data: %2.2e, H1: %2.2e, L2: %2.2e\r" %
                    (i+1, datres, regres, l2res))
                sys.stdout.flush()
                if self.display_iterations:
                    if isinstance(x, np.ndarray):
                        self.model.plot_unknowns(
                            np.swapaxes(x, 0, 1))
                    else:
                        self.model.plot_unknowns(x.get())
        return {'x':(x.get())}

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
        (self._op.fwd(
            self._tmp_result, [x, self._coils, self.modelgrad], wait_for=self._tmp_result.events+x.events)).wait()
        return self._operator_rhs(out, self._tmp_result)

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
        return self._op.adj(out, [x, self._coils, self.modelgrad])

    def updateRegPar(self, irgn_par):
        """Update the regularization parameters.

          Performs an update of the regularization parameters as these usually
          vary from one to another Gauss-Newton step.

        Parameters
        ----------
          irgn_par (dic): A dictionary containing the new parameters.
        """
        self.alpha = irgn_par["gamma"]
        self.delta = irgn_par["delta"]
        self.lambd = 1

    def setFvalInit(self, fval):
        """Set the initial value of the cost function.

        Parameters
        ----------
          fval : float
            The initial cost of the optimization problem
        """
        self._fval_init = fval

    def update_box(self, outp, inp, par, idx=0, idxq=0,
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
        return self._prg[idx].update_box(
            self._queue[4*idx+idxq],
            outp.shape[1:], None,
            outp.data, inp[0].data,
            self.min_const[idx].data, self.max_const[idx].data,
            self.real_const[idx].data, np.int32(self.unknowns),
            wait_for=(outp.events +
                      inp[0].events + wait_for))


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

    def power_iteration(self, x, num_simulations=50):
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k = clarray.to_device(self._queue[0], (np.random.randn(*(x.shape))+1j*np.random.randn(*(x.shape))).astype(self._DTYPE))
        b_k1 = clarray.zeros_like(b_k)

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            self._operator_lhs(b_k1, b_k).wait()

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1.get())

            # re normalize the vector
            b_k = b_k1 / self._DTYPE_real(b_k1_norm)

            lmax = np.abs((clarray.vdot(b_k1,b_k)/np.linalg.norm(b_k.get())**2).get())

        return lmax

    def power_iteration_grad(self, x, num_simulations=50):
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k = clarray.to_device(self._queue[0], (np.random.randn(*(x.shape))+1j*np.random.randn(*(x.shape))).astype(self._DTYPE))
        b_k1 = clarray.zeros_like(b_k)

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            b_k1 = self._grad_op.adjoop(self._grad_op.fwdoop(b_k))

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1.get())

            # re normalize the vector
            b_k = b_k1 / self._DTYPE_real(b_k1_norm)

            lmax = np.abs((clarray.vdot(b_k1,b_k)/np.linalg.norm(b_k.get())**2).get())

        return lmax


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
        self.lambd = 1#irgn_par["lambd"]
        self.rtol = irgn_par["rtol"]
        self.atol = irgn_par["atol"]
        self.stag = irgn_par["stag"]
        self.display_iterations = irgn_par["display_iterations"]
        self.mu = 1 / self.delta
        self.tau = tau
        self.beta_line = irgn_par["beta"]
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
        self.tmp_par_array = None
        self.precond = False
        
        self._kernelsize = (par["par_slices"] + par["overlap"], par["dimY"],
                            par["dimX"])
        if self._DTYPE is np.complex64:
            self.abskrnl = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="hypot(x[i].s0,x[i].s1)",
                arguments="__global float2 *x")
            self.abskrnldiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="hypot(x[i].s0-y[i].s0,x[i].s1-y[i].s1)",
                arguments="__global float2 *x, __global float2 *y")
            self.normkrnl = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="pown(x[i].s0,2)+pown(x[i].s1,2)",
                arguments="__global float2 *x")
            self.normkrnlweighted = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="(pown(x[i].s0,2)+pown(x[i].s1,2))*w[i]",
                arguments="__global float2 *x, __global float *w")
            self.normkrnldiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="pown(x[i].s0-y[i].s0,2)+pown(x[i].s1-y[i].s1,2)",
                arguments="__global float2 *x, __global float2 *y")
            self.normkrnlweighteddiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="(pown(x[i].s0-y[i].s0,2)+pown(x[i].s1-y[i].s1,2))*w[i]",
                arguments="__global float2 *x, __global float2 *y, __global float *w")
        elif self._DTYPE is np.complex128:
            self.abskrnl = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="hypot(x[i].s0,x[i].s1)",
                arguments="__global double2 *x")
            self.abskrnldiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="hypot(x[i].s0-y[i].s0,x[i].s1-y[i].s1)",
                arguments="__global double2 *x, __global double2 *y")
            self.normkrnl = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="pown(x[i].s0,2)+pown(x[i].s1,2)",
                arguments="__global double2 *x")
            self.normkrnlweighted = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="(pown(x[i].s0,2)+pown(x[i].s1,2))*w[i]",
                arguments="__global double2 *x, __global double *w")
            self.normkrnldiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="pown(x[i].s0-y[i].s0,2)+pown(x[i].s1-y[i].s1,2)",
                arguments="__global double2 *x, __global double2 *y")
            self.normkrnlweighteddiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="(pown(x[i].s0-y[i].s0,2)+pown(x[i].s1-y[i].s1,2))*w[i]",
                arguments="__global double2 *x, __global double2 *y, __global double *w")

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
            L = irgn_par["gamma"]*np.max(par["weights"])*((0.5 * (18.0 + np.sqrt(33)))**2)
            if streamed:
                if SMS:
                    pdop = PDSolverStreamedTGVSMS(
                        par,
                        irgn_par,
                        queue,
                        DTYPE_real(1 / (L)),
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
                        DTYPE_real(1 / (L)),
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
        elif reg_type == 'ICTV':
            if streamed:
              raise NotImplementedError
            if SMS:
              raise NotImplementedError
            pdop = PDSolverICTV(par,
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
        elif reg_type == 'ICTGV':
            if streamed:
              raise NotImplementedError
            if SMS:
              raise NotImplementedError
            L = irgn_par["gamma"]*np.max(par["weights"])*((0.5 * (18.0 + np.sqrt(33)))**2)
            pdop = PDSolverICTGV(par,
                              irgn_par,
                              queue,
                              DTYPE_real(1 / L),
                              init_fval, prg,
                              linops,
                              coils,
                              model,
                              DTYPE=DTYPE,
                              DTYPE_real=DTYPE_real
                              )
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
        
        l_max = 1#np.sqrt(self.power_iteration(inp[0], data.shape))
        l_max += self.alpha*((0.5 * (18.0 + np.sqrt(33)))**2)
        print("Estimated L: ", l_max)
        print()
        
        tau = 1/np.sqrt(l_max)
        tau_new = self._DTYPE_real(0)

        theta_line = self.theta_line
        beta_line = self.beta_line
        beta_new = self._DTYPE_real(0)
        mu_line = self._DTYPE_real(0.75)
        delta_line = self._DTYPE_real(0.95)
        ynorm = self._DTYPE_real(0.0)
        lhs = self._DTYPE_real(0.0)
        primal = [0]
        dual = [0]
        gap = [0]

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
                print(
                    "\033[ALine search: %04d ---- LHS: %2.2e, "
                    "RHS: %2.2e\r" %
                    (i+1, lhs, ynorm))
                sys.stdout.flush()
                if lhs <= ynorm * delta_line:
                    break
                tau_new = tau_new * mu_line

            tau = tau_new 
            
            (primal_vars, primal_vars_new, dual_vars, dual_vars_new,
             tmp_results_adjoint, tmp_results_adjoint_new,
             tmp_results_forward, tmp_results_forward_new) = (
             primal_vars_new, primal_vars, dual_vars_new, dual_vars,
             tmp_results_adjoint_new, tmp_results_adjoint,
             tmp_results_forward_new, tmp_results_forward)
                     
            primal_val, dual_val, gap_val = self._calcResidual(
                in_primal=primal_vars,
                in_dual=dual_vars,
                in_precomp_fwd=tmp_results_forward,
                in_precomp_adj=tmp_results_adjoint,
                data=data)
            primal.append(primal_val)
            dual.append(dual_val)
            gap.append(gap_val)

            # self.tau = tau
            
            if not np.mod(i+1, 100):
                if self.display_iterations:
                    if isinstance(primal_vars["x"], np.ndarray):
                        self.model.plot_unknowns(
                            np.swapaxes(primal_vars["x"], 0, 1))
                    else:
                        if self.precond:
                            # self.model.plot_unknowns(np.concatenate((self.irgn.removePrecond(primal_vars["x"].get()),primal_vars["x"].get()), axis=0))
                            self.model.plot_unknowns(self.irgn.removePrecond(primal_vars["x"].get()))
                        else:
                            self.model.plot_unknowns(primal_vars["x"].get())
            if (
                len(gap)>40 and
                np.abs(np.mean(gap[-40:-20]) - np.mean(gap[-20:]))
                 /np.mean(gap[-40:]) < self.stag
                ):
                print()
                print(
        "Terminated at iteration %d "
        "because the method stagnated. Relative difference: %.3e" %
        (i+1, np.abs(np.mean(gap[-20:-10]) - np.mean(gap[-10:]))
                 /np.mean(gap[-20:-10])))
                return primal_vars
            if np.abs((gap[-1] - gap[-2]) / gap[1]) < self.rtol:
                print()
                print(
        "Terminated at iteration %d because the energy "
        "decrease in the PD-gap was %.3e which is below the "
        "relative tolerance of %.3e"
        % (i+1, np.abs((gap[-1] - gap[-2]) / gap[1]), self.rtol))
                return primal_vars
            if np.abs((gap[-1]) / gap[1]) < self.atol:
                print()
                print(
        "Terminated at iteration %d because the energy "
        "decrease in the PD-gap was %.3e which is below the "
        "absolute tolerance of %.3e" 
        % (i+1, np.abs(gap[-1] / gap[1]), self.atol))
                return primal_vars
            
            
            print(
                "Iteration: %04d ---- Primal: %2.2e, "
                "Dual: %2.2e, Gap: %2.2e, Beta: %2.2e \r" %
                (i+1, 1000*primal[-1] / gap[1],
                 1000*dual[-1] / gap[1],
                 1000*gap[-1] / gap[1],
                 beta_line), end="")

        print()
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
                clarray.to_device(self._queue[4 * j], min_const))
            self.max_const.append(
                clarray.to_device(self._queue[4 * j], max_const))
            self.real_const.append(
                clarray.to_device(self._queue[4 * j], real_const))

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
        self.lambd = 1#irgn_par["lambd"]
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
            
        if not self.precond:
            return self._prg[idx].update_primal(
            self._queue[4 * idx + idxq],
                self._kernelsize, None,
                outp.data, inp[0].data, inp[1].data, inp[2].data,
                self._DTYPE_real(par[0]),
            self._DTYPE_real(par[0] / par[1]),
                self.min_const[idx].data, self.max_const[idx].data,
                self.real_const[idx].data, np.int32(self.unknowns),
                wait_for=(outp.events +
                          inp[0].events+inp[1].events +
                          inp[2].events+ wait_for))
            
        if self.tmp_par_array is None:
            self.tmp_par_array = clarray.zeros_like(outp)       
            
        outp.add_event(self._prg[idx].update_primal_precond(
                    self._queue[4*idx+idxq],
                    self._kernelsize, None,
                    outp.data, inp[0].data, inp[1].data, inp[2].data, 
                    self._DTYPE_real(par[0]),
                    self._DTYPE_real(par[0]/par[1]),
                    self.min_const[idx].data, self.max_const[idx].data,
                    self.real_const[idx].data, np.int32(self.unknowns),
                    wait_for=(outp.events +
                              inp[0].events+inp[1].events +
                              inp[2].events+ wait_for)))
        
        self.tmp_par_array.add_event(self._prg[idx].squarematvecmult(self._queue[4*idx+idxq], self._kernelsize, None,
            self.tmp_par_array.data, self.UTE.data, outp.data, 
            np.int32(self.unknowns),
            wait_for=self.tmp_par_array.events + outp.events + wait_for))
        
        self.tmp_par_array.add_event(self._prg[idx].update_box(
                    self._queue[4*idx+idxq],
                    self._kernelsize, None,
                    self.tmp_par_array.data,
                    self.min_const[idx].data, self.max_const[idx].data,
                    self.real_const[idx].data, np.int32(self.unknowns),
                    wait_for=(self.tmp_par_array.events 
                              + wait_for)))
            
        return self._prg[idx].squarematvecmult(self._queue[4*idx+idxq], self._kernelsize, None,
            outp.data, self.EU.data, self.tmp_par_array.data, 
            np.int32(self.unknowns),
            wait_for=self.tmp_par_array.events + outp.events + wait_for)

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
            self._queue[4*idx+idxq], (outp.size,), None,
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
            par[4][4*idx].data,
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
            par[4][4*idx].data,
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
            self._DTYPE_real(1/par[2]), np.int32(self.unknowns_TGV),
            par[3][4*idx].data,
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
            np.int32(self.unknowns_TGV),
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


    def power_iteration(self, x, data_shape, num_simulations=50):
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        x = clarray.to_device(self._queue[0], x)
        y = clarray.zeros(self._queue[0], data_shape, self._DTYPE)
        b_k = clarray.to_device(self._queue[0], 
                                (np.random.randn(*(x.shape))
                                 +1j*np.random.randn(*(x.shape))
                                 ).astype(self._DTYPE))
        b_k1 = clarray.zeros_like(b_k)
    
        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            # self._operator_lhs(b_k1, b_k).wait()
            self._op.fwd(y, [x, self._coils, self.modelgrad])
            self._op.adj(b_k1, [y, self._coils, self.modelgrad])
    
            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1.get())
    
            # re normalize the vector
            b_k = b_k1 / self._DTYPE_real(b_k1_norm)
            
            lmax = np.abs((clarray.vdot(b_k1,b_k)/np.linalg.norm(b_k.get())**2).get())
    
        return lmax

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

        primal_vars["x"] = clarray.to_device(self._queue[0], inp[0])
        primal_vars["xk"] = primal_vars["x"].copy()
        primal_vars_new["x"] = clarray.zeros_like(primal_vars["x"])
        primal_vars_new["xk"] = primal_vars["x"].copy()

        tmp_results_adjoint["Kyk1"] = clarray.zeros_like(primal_vars["x"])
        tmp_results_adjoint_new["Kyk1"] = clarray.zeros_like(primal_vars["x"])

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}
        tmp_results_forward_new = {}
        dual_vars["r"] = clarray.zeros(
            self._queue[0],
            data.shape,
            dtype=self._DTYPE)
        dual_vars_new["r"] = clarray.zeros_like(dual_vars["r"])

        dual_vars["z1"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape + (4,),
                                        dtype=self._DTYPE)
        dual_vars_new["z1"] = clarray.zeros_like(dual_vars["z1"])

        tmp_results_forward["gradx"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward["Ax"] = clarray.zeros_like(data)
        tmp_results_forward_new["Ax"] = clarray.zeros_like(data)

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


        out_fwd["Ax"].add_event(self._op.fwd(
            out_fwd["Ax"], [in_primal["x"], self._coils, self.modelgrad]))
        out_fwd["gradx"].add_event(
            self._grad_op.fwd(out_fwd["gradx"], in_primal["x"]))
        if not self.precond:
            out_adj["Kyk1"].add_event(
                self._op.adjKyk1(out_adj["Kyk1"],
                                  [in_dual["r"], in_dual["z1"],
                                  self._coils,
                                  self.modelgrad]))
        else:
            (self._op.adj(
                out_adj["Kyk1"], [in_dual["r"], self._coils, self.modelgrad])).wait()
            
            out_adj["Kyk1"] = out_adj["Kyk1"] - self._grad_op.adjoop(in_dual["z1"])

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        out_primal["x"].add_event(self.update_primal(
            outp=out_primal["x"],
            inp=(in_primal["x"], in_precomp_adj["Kyk1"],
                 in_primal["xk"]),
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
                par=(beta*tau, theta, self.alpha, self.omega, self._op.ratio)
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
        if not self.precond:
            out_adj["Kyk1"].add_event(
                self._op.adjKyk1(
                    out_adj["Kyk1"],
                    [out_dual["r"], out_dual["z1"],
                      self._coils,
                      self.modelgrad]))
        else:
            (self._op.adj(
                out_adj["Kyk1"], [out_dual["r"], self._coils, self.modelgrad])).wait()
            out_adj["Kyk1"] = out_adj["Kyk1"] - self._grad_op.adjoop(out_dual["z1"])

        ynorm = (
            (
                self.normkrnldiff(out_dual["r"], in_dual["r"])
                + self.normkrnldiff(out_dual["z1"], in_dual["z1"])
            )**(1 / 2))
        lhs = np.sqrt(beta) * tau * (
            (
                self.normkrnldiff(out_adj["Kyk1"], in_precomp_adj["Kyk1"])
            )**(1 / 2))
        return lhs.get(), ynorm.get()

    def _calcResidual(
            self,
            in_primal,
            in_dual,
            in_precomp_fwd,
            in_precomp_adj,
            data):

        primal_new = (
            self.lambd / 2 *
            self.normkrnldiff(in_precomp_fwd["Ax"], data)
            + self.alpha * self.abskrnl(in_precomp_fwd["gradx"])
            + 1 / (2 * self.delta) *
                self.normkrnldiff(in_primal["x"],
                                      in_primal["xk"])
            ).real

        dual = (
            -self.delta / 2 * self.normkrnl(in_precomp_adj["Kyk1"])
            - clarray.vdot(
                in_primal["xk"],
                - in_precomp_adj["Kyk1"]
                )
            - 1 / (2 * self.lambd) * self.normkrnl(in_dual["r"])
            - clarray.vdot(data, in_dual["r"])
            ).real

        if self.unknowns_H1 > 0:
            primal_new += (
                 self.omega / 2 * self.normkrnl(
                     in_precomp_fwd["gradx"][self.unknowns_TGV:]
                     )
                 ).real

            dual += (
                - 1 / (2 * self.omega) * self.normkrnl(
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

        primal_vars["x"] = clarray.to_device(self._queue[0], inp[0])
        primal_vars["xk"] = primal_vars["x"].copy()
        primal_vars_new["x"] = clarray.zeros_like(primal_vars["x"])
        primal_vars["v"] = clarray.to_device(self._queue[0], np.zeros_like(inp[1]))
        primal_vars_new["v"] = clarray.zeros_like(primal_vars["v"])
        primal_vars_new["xk"] = primal_vars["x"].copy()

        tmp_results_adjoint["Kyk1"] = clarray.zeros_like(primal_vars["x"])
        tmp_results_adjoint_new["Kyk1"] = clarray.zeros_like(primal_vars["x"])
        tmp_results_adjoint["Kyk2"] = clarray.zeros_like(primal_vars["v"])
        tmp_results_adjoint_new["Kyk2"] = clarray.zeros_like(primal_vars["v"])

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}
        tmp_results_forward_new = {}
        dual_vars["r"] = clarray.zeros(
            self._queue[0],
            data.shape,
            dtype=self._DTYPE)
        dual_vars_new["r"] = clarray.zeros_like(dual_vars["r"])

        dual_vars["z1"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape+(4,),
                                        dtype=self._DTYPE)
        dual_vars_new["z1"] = clarray.zeros_like(dual_vars["z1"])
        dual_vars["z2"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape+(8,),
                                        dtype=self._DTYPE)
        dual_vars_new["z2"] = clarray.zeros_like(dual_vars["z2"])

        tmp_results_forward["gradx"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward["symgradx"] = clarray.zeros_like(
            dual_vars["z2"])
        tmp_results_forward_new["symgradx"] = clarray.zeros_like(
            dual_vars["z2"])
        tmp_results_forward["Ax"] = clarray.zeros_like(data)
        tmp_results_forward_new["Ax"] = clarray.zeros_like(data)

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

        out_adj["Kyk2"].add_event(
            self.update_Kyk2(
                outp=out_adj["Kyk2"],
                inp=(in_dual["z2"], in_dual["z1"])))

        out_fwd["Ax"].add_event(
            self._op.fwd(
            out_fwd["Ax"], [in_primal["x"], self._coils, self.modelgrad]))
        out_fwd["gradx"].add_event(
            self._grad_op.fwd(out_fwd["gradx"], in_primal["x"]))
        out_fwd["symgradx"].add_event(
            self._symgrad_op.fwd(out_fwd["symgradx"], in_primal["v"]))
        
        if not self.precond:
            out_adj["Kyk1"].add_event(
                self._op.adjKyk1(out_adj["Kyk1"],
                                  [in_dual["r"], in_dual["z1"],
                                  self._coils,
                                  self.modelgrad]))
        else:
            (self._op.adj(
                out_adj["Kyk1"], [in_dual["r"], self._coils, self.modelgrad])).wait()
            out_adj["Kyk1"] = out_adj["Kyk1"] - self._grad_op.adjoop(in_dual["z1"])
        
    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        out_primal["x"].add_event(
            self.update_primal(
            outp=out_primal["x"],
            inp=(in_primal["x"], in_precomp_adj["Kyk1"],
                 in_primal["xk"]),
            par=(tau, self.delta)))

        out_primal["v"].add_event(
            self.update_v(
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
                par=(beta*tau, theta, self.alpha, self.omega, self._op.ratio)
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
                par=(beta*tau, theta, self.beta, self._op.ratio)))
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
        if not self.precond:
            out_adj["Kyk1"].add_event(
                self._op.adjKyk1(
                    out_adj["Kyk1"],
                    [out_dual["r"], out_dual["z1"],
                      self._coils,
                      self.modelgrad]))
        else:
            (self._op.adj(
                out_adj["Kyk1"], [out_dual["r"], self._coils, self.modelgrad])).wait()
            out_adj["Kyk1"] = out_adj["Kyk1"] - self._grad_op.adjoop(out_dual["z1"]) 
                
        out_adj["Kyk2"].add_event(
            self.update_Kyk2(
                outp=out_adj["Kyk2"],
                inp=(out_dual["z2"], out_dual["z1"])))
    

        ynorm = (
            self.normkrnldiff(out_dual["r"], in_dual["r"],
                        wait_for=out_dual["r"].events + in_dual["r"].events).get()
            + self.normkrnldiff(out_dual["z1"], in_dual["z1"],
                        wait_for=out_dual["z1"].events + in_dual["z1"].events).get()
            + self.normkrnldiff(out_dual["z2"], in_dual["z2"],
                        wait_for=out_dual["z2"].events + in_dual["z2"].events).get()
            )**(1/2)

        lhs = np.sqrt(beta) * tau * (
            self.normkrnldiff(out_adj["Kyk1"], in_precomp_adj["Kyk1"],
              wait_for=out_adj["Kyk1"].events + in_precomp_adj["Kyk1"].events).get()
            + self.normkrnldiff(out_adj["Kyk2"], in_precomp_adj["Kyk2"],
              wait_for=out_adj["Kyk2"].events + in_precomp_adj["Kyk2"].events).get()
            )**(1/2)

        return lhs, ynorm

    def _calcResidual(
            self,
            in_primal,
            in_dual,
            in_precomp_fwd,
            in_precomp_adj,
            data):

        primal_new = (
            self.lambd / 2 * self.normkrnldiff(in_precomp_fwd["Ax"], data)
            + self.alpha * self.abskrnldiff(in_precomp_fwd["gradx"],
                                              in_primal["v"])
            + self.beta * self.abskrnl(in_precomp_fwd["symgradx"])
            + 1 / (2 * self.delta) * self.normkrnldiff(in_primal["x"],
                                      in_primal["xk"]).get()

            ).real

        dual = (
            -self.delta / 2 * self.normkrnl(in_precomp_adj["Kyk1"])
            - clarray.vdot(
                in_primal["xk"],
                -in_precomp_adj["Kyk1"]
                )
            - clarray.sum(in_precomp_adj["Kyk2"])
            - 1 / (2 * self.lambd) * self.normkrnl(in_dual["r"])
            - clarray.vdot(data, in_dual["r"])
            ).real

        if self.unknowns_H1 > 0:
            primal_new += (
                 self.omega / 2 * self.normkrnl(
                     in_precomp_fwd["gradx"][self.unknowns_TGV:]
                     )
                 ).get()

            dual += (
                - 1 / (2 * self.omega) * self.normkrnl(
                    in_dual["z1"][self.unknowns_TGV:]
                    )
                ).get()
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
            DTYPE=self._DTYPE,
            DTYPE_real=self._DTYPE_real)


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

        primal_vars["x"] = inp[0]
        primal_vars["xk"] = primal_vars["x"].copy()
        primal_vars_new["x"] = np.zeros_like(primal_vars["x"])
        primal_vars["v"] = np.zeros(
                                        primal_vars["x"].shape+(4,),
                                        dtype=self._DTYPE)
        primal_vars_new["v"] = np.zeros_like(primal_vars["v"])
        primal_vars_new["xk"] = primal_vars["x"].copy()

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
            [[],[],[]])

        self.stream_initial_2.eval(
            [out_fwd["gradx"],
             out_adj["Kyk2"]],
            [[in_primal["x"]],
             [in_dual["z2"], in_dual["z1"], []]],
            [[],[]])

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
              in_primal["xk"]],
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
                [beta*tau, theta, self.alpha, self.omega, self._op.ratio],
                [beta * tau, theta, self.lambd],
                []
            ])
        (lhs2, ynorm2) = self.update_dual_2.evalwithnorm(
            [out_dual["z2"],
             out_adj["Kyk2"]],
            [[in_dual["z2"], in_precomp_fwd_new["symgradx"],
              in_precomp_fwd["symgradx"]],
             [[], out_dual["z1"], in_precomp_adj["Kyk2"]]],
            [[beta*tau, theta, self.beta, self._op.ratio], []])

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
                (in_primal["x"] - in_primal["xk"]),
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
            [[in_dual["r"], in_dual["z1"], self._coils, self.modelgrad, []]])

        self._symgrad_op.fwd(
            [out_fwd["symgradx"]],
            [[in_primal["v"]]])

        self.stream_initial_2.eval(
            [out_fwd["gradx"],
             out_adj["Kyk1"]],
            [[in_primal["x"]],
             [in_dual["z2"], in_dual["z1"], []]],
            [[], []])

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        self.update_primal_1.eval(
            [out_primal["x"],
             out_fwd["gradx"]],
            [[in_primal["x"],
              in_precomp_adj["Kyk1"],
              in_primal["xk"]],
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
              self.alpha, self.omega, self._op.ratio]])
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
              self._coils, self.modelgrad, in_precomp_adj["Kyk1"]]])

        (lhs4, ynorm4) = self.update_dual_2.evalwithnorm(
            [out_dual["z2"],
             out_adj["Kyk2"]],
            [[in_dual["z2"], in_precomp_fwd_new["symgradx"],
              in_precomp_fwd["symgradx"]],
             [[], out_dual["z1"], in_precomp_adj["Kyk2"]]],
            [[beta*tau, theta, self.beta, self._op.ratio]])

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

        primal_vars["x"] = inp[0]
        primal_vars["xk"] = primal_vars["x"].copy()
        primal_vars_new["x"] = np.zeros_like(primal_vars["x"])
        primal_vars_new["xk"] = primal_vars["x"].copy()

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
            [[], []])

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
              in_primal["xk"]],
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
                [beta*tau, theta, self.alpha, self.omega, self._op.ratio],
                [beta * tau, theta, self.lambd],
                []
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
                (in_primal["x"] - in_primal["xk"]),
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
            [[in_dual["r"], in_dual["z1"], self._coils, self.modelgrad, []]])

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
              self.alpha, self.omega, self._op.ratio]])
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
              self._coils, self.modelgrad, in_precomp_adj["Kyk1"]]])

        ynorm = np.abs(ynorm1+ynorm2+ynorm3)**(1/2)
        lhs = np.sqrt(beta)*tau*np.abs(lhs1+lhs2+lhs3)**(1/2)

        return lhs, ynorm


class PDSoftSenseBaseSolver:
    """Primal Dual Soft-SENSE optimization.

    This Class performs a primal-dual algorithm for solving a Soft-SENSE
    reconstruction on complex input data

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      pdsose_par : dict
        A python dict containing the required
        parameters for the regularized Soft-SENSE reconstruction.
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
      coil : PyOpenCL Buffer or empty list
         The coils used for reconstruction.

    Attributes
    ----------
      lambd : float
        Regularization parameter in front of data fidelity term.
      tol : float
        Relative toleraze to stop iterating
      stag : float
        Stagnation detection parameter
      adaptive_stepsize : bool
        Use adaptive step size
      tau : float
        Estimated step size based on operator norm of regularization.
      unknowns_TGV : int
        Number of T(G)V unknowns
      unknowns : int
        Total number of unknowns --> Reflects the number of cmaps for
        Soft-SENSE
      num_dev : int
        Total number of compute devices
      dz : float
        Ratio between 3rd dimension and isotropic 1st and 2nd image dimension.
    """

    def __init__(self,
                 par,
                 pdsose_par,
                 queue,
                 tau,
                 fval,
                 prg,
                 coils,
                 DTYPE=np.complex64,
                 DTYPE_real=np.float32):

        self._DTYPE = DTYPE
        self._DTYPE_real = DTYPE_real
        self.sigma = self._DTYPE_real(1 / np.sqrt(12))
        self.lambd = self._DTYPE_real(pdsose_par["lambd"])
        self.adaptive_stepsize = pdsose_par["adaptive_stepsize"]
        self.stag = pdsose_par["stag"]
        self.tol = pdsose_par["tol"]
        self.tau = tau
        self.unknowns = par["unknowns"]
        self.unknowns_TGV = par["unknowns_TGV"]
        self.num_dev = len(par["num_dev"])
        self.dz = par["dz"]
        self._fval_init = fval
        self._prg = prg
        self._queue = queue
        self._coils = coils
        self._kernelsize = (par["par_slices"] + par["overlap"], par["dimY"],
                            par["dimX"])
        if self._DTYPE is np.complex64:
            self.abskrnl = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="hypot(x[i].s0,x[i].s1)",
                arguments="__global float2 *x")
            self.abskrnldiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="hypot(x[i].s0-y[i].s0,x[i].s1-y[i].s1)",
                arguments="__global float2 *x, __global float2 *y")
            self.normkrnl = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="pown(x[i].s0,2)+pown(x[i].s1,2)",
                arguments="__global float2 *x")
            self.normkrnlweighted = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="(pown(x[i].s0,2)+pown(x[i].s1,2))*w[i]",
                arguments="__global float2 *x, __global float *w")
            self.normkrnldiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="pown(x[i].s0-y[i].s0,2)+pown(x[i].s1-y[i].s1,2)",
                arguments="__global float2 *x, __global float2 *y")
            self.normkrnlweighteddiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="(pown(x[i].s0-y[i].s0,2)+pown(x[i].s1-y[i].s1,2))*w[i]",
                arguments="__global float2 *x, __global float2 *y, __global float *w")
        elif self._DTYPE is np.complex128:
            self.abskrnl = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="hypot(x[i].s0,x[i].s1)",
                arguments="__global double2 *x")
            self.abskrnldiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="hypot(x[i].s0-y[i].s0,x[i].s1-y[i].s1)",
                arguments="__global double2 *x, __global double2 *y")
            self.normkrnl = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="pown(x[i].s0,2)+pown(x[i].s1,2)",
                arguments="__global double2 *x")
            self.normkrnlweighted = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="(pown(x[i].s0,2)+pown(x[i].s1,2))*w[i]",
                arguments="__global double2 *x, __global double *w")
            self.normkrnldiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="pown(x[i].s0-y[i].s0,2)+pown(x[i].s1-y[i].s1,2)",
                arguments="__global double2 *x, __global double2 *y")
            self.normkrnlweighteddiff = clred.ReductionKernel(
                par["ctx"][0], self._DTYPE_real, 0,
                reduce_expr="a+b",
                map_expr="(pown(x[i].s0-y[i].s0,2)+pown(x[i].s1-y[i].s1,2))*w[i]",
                arguments="__global double2 *x, __global double2 *y, __global double *w")

    @staticmethod
    def factory(
            prg,
            queue,
            par,
            pdsose_par,
            init_fval,
            coils,
            linops,
            reg_type='TGV',
            streamed=False,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        """
        Generate a PDSoftSenseSolver object.

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
          pdsose_par : dict
            A python dict containing the parameters for the
            regularized Soft-SENSE reconstruction
          init_fval : float
            Estimate of the initial cost function value to
            scale the displayed values.
          coils : PyOpenCL Buffer or empty list
            The coils used for reconstruction.
          linops : list of PyQMRI Operator
            The linear operators used for fitting.
          reg_type : string, "TGV"
            String to choose between "TV" and "TGV"
            optimization.
          streamed : bool, false
            Switch between streamed (1) and normal (0) reconstruction.
          DTYPE : numpy.dtype, numpy.complex64
             Complex working precission.
          DTYPE_real : numpy.dtype, numpy.float32
            Real working precission.
        """
        if reg_type == 'TV':
            if not streamed:
                pdop = PDSoftSenseSolverTV(
                    par,
                    pdsose_par,
                    queue,
                    np.float32(1 / np.sqrt(12)),
                    init_fval,
                    prg,
                    linops,
                    coils,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
            else:
                pdop = PDSoftSenseSolverStreamedTV(
                    par,
                    pdsose_par,
                    queue,
                    np.float32(1 / np.sqrt(12)),
                    init_fval,
                    prg,
                    linops,
                    coils,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
        elif reg_type == 'TGV':
            if not streamed:
                pdop = PDSoftSenseSolverTGV(
                    par,
                    pdsose_par,
                    queue,
                    np.float32(1 / np.sqrt(12)),
                    init_fval,
                    prg,
                    linops,
                    coils,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
            else:
                pdop = PDSoftSenseSolverStreamedTGV(
                    par,
                    pdsose_par,
                    queue,
                    np.float32(1 / np.sqrt(12)),
                    init_fval,
                    prg,
                    linops,
                    coils,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
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

        Parameters
        ----
          inp (numpy.array):
            Initial guess for the reconstruction
          data (numpy.array):
            The complex valued (undersampled) kspace data.
          iters (int):
            Number of primal-dual iterations to run

        Returns
        -------
          tupel:
            Primal variable x. If no streaming is used, the two
            entries are opf class PyOpenCL.Array, otherwise Numpy.Array.
        """

        tau = self.tau
        sigma = self.sigma
        theta = np.float32(1.0)

        primal = [0]
        dual = [0]
        gap = [0]

        (primal_vars,
         primal_vars_new,
         tmp_results_forward,
         dual_vars,
         dual_vars_new,
         tmp_results_adjoint,
         data) = self._setupVariables(inp, data)

        self._updateInitial(
            out_fwd=tmp_results_forward,
            out_adj=tmp_results_adjoint,
            in_primal=primal_vars,
            in_dual=dual_vars
        )

        for i in range(iters):
            self._updateDual(
                out_dual=dual_vars_new,
                out_adj=tmp_results_adjoint,
                in_primal=primal_vars_new,
                in_dual=dual_vars,
                in_precomp_fwd=tmp_results_forward,
                data=data,
                sigma=sigma
            )

            self._updatePrimal(
                out_primal=primal_vars_new,
                out_fwd=tmp_results_forward,
                in_primal=primal_vars,
                in_precomp_adj=tmp_results_adjoint,
                tau=tau,
                theta=theta
            )

            if self.adaptive_stepsize:
                tau, sigma = self._updateStepSize(
                    primal_vars=primal_vars,
                    primal_vars_new=primal_vars_new,
                    tmp_results_forward=tmp_results_forward,
                    theta=theta,
                    tau=tau,
                    sigma=sigma
                )

            self._extrapolatePrimal(
                out_primal=primal_vars,
                out_fwd=tmp_results_forward,
                in_primal=primal_vars_new,
                theta=theta)

            for j in primal_vars_new:
                (primal_vars[j],
                 primal_vars_new[j]) = \
                    (primal_vars_new[j],
                     primal_vars[j])

            for k in dual_vars_new:
                (dual_vars[k],
                 dual_vars_new[k]) = \
                    (dual_vars_new[k],
                     dual_vars[k])

            primal_val, dual_val, gap_val = self._calcResidual(
                in_primal=primal_vars,
                in_dual=dual_vars,
                in_precomp_fwd=tmp_results_forward,
                data=data)
            primal.append(primal_val)
            dual.append(dual_val)
            gap.append(gap_val)

            if self.adaptive_stepsize and not np.mod(i+1, 10):
                print("Iteration: %d \n Current step size: %f" % (i+1, tau))

            if np.abs(primal[-2] - primal[-1]) / primal[1] < \
                    self.tol:
                print(
                    "Terminated at iteration %d because the energy "
                    "decrease in the primal problem was %.3e which is below the "
                    "relative tolerance of %.3e" %
                    (i + 1,
                     np.abs(primal[-2] - primal[-1]) / primal[1],
                     self.tol))
                return primal_vars, i
            if np.abs(np.abs(dual[-2] - dual[-1]) / dual[1]) < \
                    self.tol:
                print(
                    "Terminated at iteration %d because the energy "
                    "decrease in the dual problem was %.3e which is below the "
                    "relative tolerance of %.3e" %
                    (i + 1,
                     np.abs(np.abs(dual[-2] - dual[-1]) / dual[1]),
                     self.tol))
                return primal_vars, i
            if (
                    len(gap) > 40 and
                    np.abs(np.mean(gap[-40:-20]) - np.mean(gap[-20:]))
                    / np.mean(gap[-40:]) < self.stag
            ):
                print(
                    "Terminated at iteration %d "
                    "because the method stagnated. Relative difference: %.3e" %
                    (i + 1, np.abs(np.mean(gap[-20:-10]) - np.mean(gap[-10:]))
                     / np.mean(gap[-20:-10])))
                return primal_vars, i
            if np.abs((gap[-1] - gap[-2]) / gap[1]) < self.tol:
                print(
                    "Terminated at iteration %d because the energy "
                    "decrease in the PD-gap was %.3e which is below the "
                    "relative tolerance of %.3e"
                    % (i + 1, np.abs((gap[-1] - gap[-2]) / gap[1]), self.tol))
                return primal_vars, i
            sys.stdout.write(
                "Iteration: %04d ---- Primal: %2.2e, "
                "Dual: %2.2e, Gap: %2.2e\r" %
                (i + 1, 1000 * primal[-1] / gap[1],
                 1000 * dual[-1] / gap[1],
                 1000 * gap[-1] / gap[1]))
            sys.stdout.flush()

        return primal_vars, i

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
            tau,
            theta):
        pass

    def _updateDual(self,
                    out_dual,
                    out_adj,
                    in_primal,
                    in_dual,
                    in_precomp_fwd,
                    data,
                    sigma):
        pass

    def _updateStepSize(self,
                        primal_vars,
                        primal_vars_new,
                        tmp_results_forward,
                        theta,
                        tau,
                        sigma):
        return {}, {}

    def _extrapolatePrimal(self,
                           out_primal,
                           out_fwd,
                           in_primal,
                           theta):
        pass

    def _calcResidual(self,
                      in_primal,
                      in_dual,
                      in_precomp_fwd,
                      data):
        return {}, {}, {}

    def _setupVariables(self,
                        inp,
                        data):
        pass

    def extrapolate_x(self, outp, inp, par=None, idx=0, idxq=0,
                      bound_cond=0, wait_for=None):
        """Extrapolation step of the x variable in the Primal-Dual Algorithm.

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
        return self._prg[idx].extrapolate_x(
            self._queue[4 * idx + idxq], (outp.size,), None,
            outp.data,
            inp[0].data,
            inp[1].data,
            self._DTYPE_real(par[0]),
            wait_for=(outp.events + inp[0].events
                      + inp[1].events + wait_for))

    def extrapolate_v(self, outp, inp, par=None, idx=0, idxq=0,
                      bound_cond=0, wait_for=None):
        """Extrapolation step of the v variable in the Primal-Dual Algorithm.

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
        return self._prg[idx].extrapolate_v(
            self._queue[4 * idx + idxq], (outp[..., 0].size,), None,
            outp.data,
            inp[0].data,
            inp[1].data,
            self._DTYPE_real(par[0]),
            wait_for=(outp.events + inp[0].events
                      + inp[1].events + wait_for))

    def update_x(self, outp, inp, par=None, idx=0, idxq=0,
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
        return self._prg[idx].update_x(
            self._queue[4 * idx + idxq], (outp.size,), None,
            outp.data,
            inp[0].data,
            inp[1].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[1]),
            wait_for=(outp.events + inp[0].events
                      + inp[1].events + wait_for))

    def update_x_tgv(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=None):
        """Primal update of the x variable in the Primal-Dual Algorithm for TGV.

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
        return self._prg[idx].update_x_explicit(
            self._queue[4 * idx + idxq], (outp.size,), None,
            outp.data,
            inp[0].data,
            inp[1].data,
            inp[2].data,
            self._DTYPE_real(par[0]),         # tau
            self._DTYPE_real(par[1]),         # theta
            wait_for=outp.events + inp[0].events +
            inp[1].events + inp[2].events + wait_for)

    def update_y(self, outp, inp, par=None, idx=0, idxq=0,
                 bound_cond=0, wait_for=None):
        """Update the data dual variable y.

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
        return self._prg[idx].update_y(
            self._queue[4 * idx + idxq], (outp.size,), None,
            outp.data,
            inp[0].data,
            inp[1].data,
            inp[2].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(1 / (1 + par[0] / par[1])),
            wait_for=(outp.events + inp[0].events +
                      inp[1].events + inp[2].events + wait_for))

    def update_z_tv(self, outp, inp, par=None, idx=0, idxq=0,
                    bound_cond=0, wait_for=None):
        """
        Dual update of the z variable in Primal-Dual Algorithm for TV.

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
        return self._prg[idx].update_z_tv(
            self._queue[4 * idx + idxq],
            self._kernelsize, None,
            outp.data,
            inp[0].data,
            inp[1].data,
            self._DTYPE_real(par[0]),
            np.int32(self.unknowns),
            wait_for=(outp.events + inp[0].events +
                      inp[1].events + wait_for))

    def update_Kyk2(self, outp, inp, par=None, idx=0, idxq=0,
                    bound_cond=0, wait_for=None):
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
            self._queue[4 * idx + idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, inp[1].data,
            np.int32(self.unknowns_TGV),
            self._grad_op.ratio[idx].data,
            self._symgrad_op.ratio[idx].data,
            np.int32(bound_cond),
            self._DTYPE_real(self.dz),
            wait_for=(outp.events + inp[0].events
                      + inp[1].events + wait_for))

    def update_z1_tgv(self, outp, inp, par=None, idx=0, idxq=0,
                      bound_cond=0, wait_for=None):
        """
        Dual update of the z1 variable in Primal-Dual Algorithm for TGV.

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
        return self._prg[idx].update_z1_tgv(
            self._queue[4 * idx + idxq],
            self._kernelsize, None,
            outp.data,
            inp[0].data,
            inp[1].data,
            inp[2].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(1 / par[1]),
            np.int32(self.unknowns),
            wait_for=(outp.events + inp[0].events + inp[1].events +
                      inp[2].events + wait_for))

    def update_z2_tgv(self, outp, inp, par=None, idx=0, idxq=0,
                      bound_cond=0, wait_for=None):
        """
        Dual update of the z2 variable in Primal-Dual Algorithm for TGV.

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
        return self._prg[idx].update_z2_tgv(
            self._queue[4 * idx + idxq],
            self._kernelsize, None,
            outp.data,
            inp[0].data,
            inp[1].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(1 / par[1]),
            np.int32(self.unknowns),
            wait_for=(outp.events + inp[0].events +
                      inp[1].events + wait_for))

    def update_v(self, outp, inp, par=None, idx=0, idxq=0,
                 bound_cond=0, wait_for=None):
        """
        Primal update of the v variable in Primal-Dual Algorithm for TGV.

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
            self._queue[4 * idx + idxq], (outp.size,), None,
            outp.data, inp[0].data, inp[1].data,
            self._DTYPE_real(par[0]),
            wait_for=outp.events + inp[0].events + inp[1].events + wait_for)

    def set_fval_init(self, fval):
        """Set the initial value of the cost function.

        Parameters
        ----------
          fval : float
            The initial cost of the optimization problem
        """
        self._fval_init = fval

    def set_tau(self, tau):
        """Set the step size for the primal update.

        Parameters
        ----------
          tau : float
            Step size for the primal update
        """
        self.tau = tau

    def set_sigma(self, sigma):
        """Set the step size for the dual update.

        Parameters
        ----------
          sigma : float
            Step size for the dual update
        """
        self.sigma = sigma


class PDSoftSenseSolverTV(PDSoftSenseBaseSolver):
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
        pdsose_par : dict
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
    """

    def __init__(self,
                 par,
                 pdsose_par,
                 queue,
                 tau,
                 fval,
                 prg,
                 linop,
                 coils,
                 **kwargs
                 ):

        super().__init__(
            par,
            pdsose_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            **kwargs)
        self._op = linop[0]
        self._grad_op = linop[1]

    def _setupVariables(self, inp, data):
        data = clarray.to_device(self._queue[0], data.astype(self._DTYPE))

        primal_vars = {}
        primal_vars_new = {}
        tmp_results_adjoint = {}

        primal_vars["x"] = clarray.to_device(self._queue[0], inp)
        primal_vars_new["x"] = clarray.zeros_like(primal_vars["x"])

        tmp_results_adjoint["Kyk1"] = clarray.zeros_like(primal_vars["x"])

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}

        dual_vars["y"] = clarray.zeros(
            self._queue[0],
            data.shape,
            dtype=self._DTYPE
        )
        dual_vars["z"] = clarray.zeros(self._queue[0],
                                       primal_vars["x"].shape + (4,),
                                       dtype=self._DTYPE)

        dual_vars_new["y"] = clarray.zeros_like(dual_vars["y"])
        dual_vars_new["z"] = clarray.zeros_like(dual_vars["z"])

        tmp_results_forward["Kx"] = clarray.zeros_like(data)
        tmp_results_forward["gradx"] = clarray.zeros_like(dual_vars["z"])

        return (primal_vars,
                primal_vars_new,
                tmp_results_forward,
                dual_vars,
                dual_vars_new,
                tmp_results_adjoint,
                data)

    def _updateInitial(self,
                       out_fwd, out_adj,
                       in_primal, in_dual):
        out_adj["Kyk1"].add_event(
            self._op.adjKyk1(
                out=out_adj["Kyk1"],
                inp=[
                    in_dual["y"],
                    self._coils,
                    in_dual["z"],
                    self._grad_op.ratio]))

        out_fwd["Kx"].add_event(self._op.fwd(
            out_fwd["Kx"], [in_primal["x"], self._coils]))

        out_fwd["gradx"].add_event(self._grad_op.fwd(
            out_fwd["gradx"], in_primal["x"]))

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau, theta):
        out_primal["x"].add_event(self.update_x(
            outp=out_primal["x"],
            inp=(in_primal["x"], in_precomp_adj["Kyk1"]),
            par=[tau, 0]))

    def _updateDual(self,
                    out_dual,
                    out_adj,
                    in_primal,
                    in_dual,
                    in_precomp_fwd,
                    data,
                    sigma):
        out_dual["z"].add_event(
            self.update_z_tv(
                outp=out_dual["z"],
                inp=(in_dual["z"], in_precomp_fwd["gradx"]),
                par=[sigma]))

        out_dual["y"].add_event(
            self.update_y(
                outp=out_dual["y"],
                inp=(in_dual["y"], in_precomp_fwd["Kx"], data),
                par=[sigma, self.lambd]))

        out_adj["Kyk1"].add_event(
            self._op.adjKyk1(
                out=out_adj["Kyk1"],
                inp=[
                    out_dual["y"],
                    self._coils,
                    out_dual["z"],
                    self._grad_op.ratio]))

    def _updateStepSize(self,
                        primal_vars,
                        primal_vars_new,
                        tmp_results_forward,
                        theta,
                        tau,
                        sigma):
        diffx = primal_vars_new["x"] - primal_vars["x"]
        nx = clarray.vdot(diffx, diffx).real ** (1 / 2)
        nx = nx.get()

        fwddiffx = self._op.fwdoop([diffx, self._coils])
        graddiffx = self._grad_op.fwdoop(diffx)
        nKx = (
            clarray.vdot(
                fwddiffx,
                fwddiffx
            )
            + clarray.vdot(
                graddiffx,
                graddiffx
            )
        ).real ** (1 / 2)
        nKx = nKx.get()
        n = nx / nKx if nKx != 0 else 0
        s = sigma * tau  # np.sqrt(sigma * tau)
        fac = theta * sigma * tau  # np.sqrt(theta * sigma * tau)

        if fac >= n > 0:
            s = n
        elif s >= n > fac:
            s = np.sqrt(fac)
        else:
            s = np.sqrt(s)

        return s, s

    def _extrapolatePrimal(self,
                           out_primal,
                           out_fwd,
                           in_primal,
                           theta):
        out_primal["x"].add_event(
            self.extrapolate_x(
                outp=out_primal["x"],
                inp=(in_primal["x"], out_primal["x"]),
                par=[theta]))

        out_fwd["Kx"].add_event(self._op.fwd(
            out_fwd["Kx"], [out_primal["x"], self._coils]))

        out_fwd["gradx"].add_event(self._grad_op.fwd(
            out_fwd["gradx"], out_primal["x"]))

    def _calcResidual(self,
                      in_primal,
                      in_dual,
                      in_precomp_fwd,
                      data):

        Kx = self._op.fwdoop([in_primal["x"], self._coils])
        gradx = self._grad_op.fwdoop(in_primal["x"])

        primal = (
            self.lambd / 2 *
            self.normkrnldiff(Kx, data)
            + self.abskrnl(gradx)
        ).real

        dual = (
            1 / (2 * self.lambd) * clarray.vdot(
                in_dual["y"],
                in_dual["y"]
            )
        ).real

        gap = np.abs(primal - dual)
        return primal.get(), dual.get(), gap.get()


class PDSoftSenseSolverTGV(PDSoftSenseBaseSolver):
    """Primal Dual splitting optimization for TGV.

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
        pdsose_par : dict
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

    Attributes
    ----------
      alpha : float
        TV regularization weight
    """

    def __init__(self,
                 par,
                 pdsose_par,
                 queue,
                 tau,
                 fval,
                 prg,
                 linop,
                 coils,
                 **kwargs):

        super().__init__(
            par,
            pdsose_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            **kwargs)

        self.alpha_0 = self._DTYPE_real(pdsose_par["alpha0"])      # alpha_0
        self.alpha_1 = self._DTYPE_real(pdsose_par["alpha1"])      # alpha_1

        self._op = linop[0]
        self._grad_op = linop[1]
        self._symgrad_op = linop[2]

    def _setupVariables(self, inp, data):
        data = clarray.to_device(self._queue[0], data.astype(self._DTYPE))

        primal_vars = {}
        primal_vars_new = {}
        tmp_results_adjoint = {}

        primal_vars["x"] = clarray.to_device(self._queue[0], inp)
        primal_vars["v"] = clarray.zeros(self._queue[0],
                                         primal_vars["x"].shape + (4,),
                                         dtype=self._DTYPE)
        primal_vars_new["x"] = clarray.zeros_like(primal_vars["x"])
        primal_vars_new["v"] = clarray.zeros_like(primal_vars["v"])

        tmp_results_adjoint["Kyk1"] = clarray.zeros_like(primal_vars["x"])
        tmp_results_adjoint["Kyk2"] = clarray.zeros_like(primal_vars["v"])

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}

        dual_vars["y"] = clarray.zeros(
            self._queue[0],
            data.shape,
            dtype=self._DTYPE
        )
        dual_vars["z1"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape + (4,),
                                        dtype=self._DTYPE)

        dual_vars["z2"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape + (8,),
                                        dtype=self._DTYPE)

        dual_vars_new["y"] = clarray.zeros_like(dual_vars["y"])
        dual_vars_new["z1"] = clarray.zeros_like(dual_vars["z1"])
        dual_vars_new["z2"] = clarray.zeros_like(dual_vars["z2"])

        tmp_results_forward["Kx"] = clarray.zeros_like(data)
        tmp_results_forward["gradx"] = clarray.zeros_like(dual_vars["z1"])
        tmp_results_forward["symgradv"] = clarray.zeros_like(dual_vars["z2"])

        return (primal_vars,
                primal_vars_new,
                tmp_results_forward,
                dual_vars,
                dual_vars_new,
                tmp_results_adjoint,
                data)

    def _updateInitial(self,
                       out_fwd,
                       out_adj,
                       in_primal,
                       in_dual):

        out_adj["Kyk1"].add_event(
            self._op.adjKyk1(
                out=out_adj["Kyk1"],
                inp=[
                    in_dual["y"],
                    self._coils,
                    in_dual["z1"],
                    self._grad_op.ratio]))

        out_adj["Kyk2"].add_event(
            self.update_Kyk2(
                outp=out_adj["Kyk2"],
                inp=(in_dual["z2"], in_dual["z1"])))

        out_fwd["Kx"].add_event(self._op.fwd(
            out_fwd["Kx"], [in_primal["x"], self._coils]))

        out_fwd["gradx"].add_event(self._grad_op.fwd(
            out_fwd["gradx"], in_primal["x"]))

        out_fwd["symgradv"].add_event(self._symgrad_op.fwd(
            out_fwd["symgradv"], in_primal["v"]))

    def _updatePrimal(self,
                      out_primal,
                      out_fwd,
                      in_primal,
                      in_precomp_adj,
                      tau,
                      theta):

        out_primal["x"].add_event(self.update_x(
            outp=out_primal["x"],
            inp=(in_primal["x"], in_precomp_adj["Kyk1"]),
            par=[tau, 0]))

        out_primal["v"].add_event(self.update_v(
            outp=out_primal["v"],
            inp=(in_primal["v"], in_precomp_adj["Kyk2"]),
            par=[tau, 0]))

    def _updateDual(self,
                    out_dual,
                    out_adj,
                    in_primal,
                    in_dual,
                    in_precomp_fwd,
                    data,
                    sigma):

        out_dual["y"].add_event(
            self.update_y(
                outp=out_dual["y"],
                inp=(in_dual["y"], in_precomp_fwd["Kx"], data),
                par=[sigma, self.lambd]))

        out_dual["z1"].add_event(
            self.update_z1_tgv(
                outp=out_dual["z1"],
                inp=(in_dual["z1"], in_precomp_fwd["gradx"], in_primal["v"]),
                par=[sigma, self.alpha_1]))

        out_adj["Kyk1"].add_event(
            self._op.adjKyk1(
                out=out_adj["Kyk1"],
                inp=[
                    out_dual["y"],
                    self._coils,
                    out_dual["z1"],
                    self._grad_op.ratio]))

        out_dual["z2"].add_event(
            self.update_z2_tgv(
                outp=out_dual["z2"],
                inp=(in_dual["z2"], in_precomp_fwd["symgradv"]),
                par=[sigma, self.alpha_0]))

        out_adj["Kyk2"].add_event(
            self.update_Kyk2(
                outp=out_adj["Kyk2"],
                inp=(out_dual["z2"], out_dual["z1"])))

    def _updateStepSize(self,
                        primal_vars,
                        primal_vars_new,
                        tmp_results_forward,
                        theta,
                        tau,
                        sigma):
        diffx = primal_vars_new["x"] - primal_vars["x"]
        diffv = primal_vars_new["v"] - primal_vars["v"]
        nx = (
            clarray.vdot(
                diffx,
                diffx
            )
            + clarray.vdot(
                diffv,
                diffv,
            )
        ).real ** (1 / 2)
        nx = nx.get()
        fwddiffx = self._op.fwdoop([diffx, self._coils])
        graddiffx = self._grad_op.fwdoop(diffx)
        symgraddiffv = self._symgrad_op.fwdoop(diffv)
        nKx = (
            clarray.vdot(
                fwddiffx,
                fwddiffx
            )
            + clarray.vdot(
                graddiffx - diffv,
                graddiffx - diffv
            )
            + clarray.vdot(
                symgraddiffv,
                symgraddiffv
            )
        ).real ** (1 / 2)
        nKx = nKx.get()
        n = nx / nKx if nKx != 0 else 0
        s = sigma * tau  # np.sqrt(sigma * tau)
        fac = theta * sigma * tau  # np.sqrt(theta * sigma * tau)

        if fac >= n > 0:
            s = n
        elif s >= n > fac:
            s = np.sqrt(fac)
        else:
            s = np.sqrt(s)

        return s, s

    def _extrapolatePrimal(self,
                           out_primal,
                           out_fwd,
                           in_primal,
                           theta):
        out_primal["x"].add_event(
            self.extrapolate_x(
                outp=out_primal["x"],
                inp=(in_primal["x"], out_primal["x"]),
                par=[theta]))

        out_primal["v"].add_event(
            self.extrapolate_v(
                outp=out_primal["v"],
                inp=(in_primal["v"], out_primal["v"]),
                par=[theta]))

        out_fwd["Kx"].add_event(self._op.fwd(
            out_fwd["Kx"], [out_primal["x"], self._coils]))

        out_fwd["gradx"].add_event(self._grad_op.fwd(
            out_fwd["gradx"], out_primal["x"]))

        out_fwd["symgradv"].add_event(self._symgrad_op.fwd(
            out_fwd["symgradv"], out_primal["v"]))

    def _calcResidual(
            self,
            in_primal,
            in_dual,
            in_precomp_fwd,
            data):

        Kx = self._op.fwdoop([in_primal["x"], self._coils])
        gradx = self._grad_op.fwdoop(in_primal["x"])
        symgradv = self._symgrad_op.fwdoop(in_primal["v"])

        primal = (
            self.lambd / 2 * self.normkrnldiff(Kx, data)
            + self.alpha_1 * self.abskrnldiff(gradx, in_primal["v"])
            + self.alpha_0 * self.abskrnl(symgradv)
        ).real

        dual = (
            1 / (2 * self.lambd) * clarray.vdot(
                in_dual["y"],
                in_dual["y"]
            )
        ).real

        gap = np.abs(primal - dual)
        return primal.get(), dual.get(), gap.get()


class PDSoftSenseBaseSolverStreamed(PDSoftSenseBaseSolver):
    """
    Streamed version of the PD Soft-SENSE Solver.

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
        pdsose_par : dict
          A python dict containing the required
          parameters for the regularized Soft-SENSE reconstruction.
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

    Attributes
    ----------
      unknown_shape : tuple of int
        Size of the unknown array
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

    def __init__(self,
                 par,
                 pdsose_par,
                 queue,
                 tau,
                 fval,
                 prg,
                 coils,
                 **kwargs
                 ):

        super().__init__(
            par,
            pdsose_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            **kwargs)

        self._op = None
        self._grad_op = None
        self._symgrad_op = None

        self.unknown_shape = (par["NSlice"], par["NMaps"],
                              par["dimY"], par["dimX"])

        self.grad_shape = self.unknown_shape + (4,)

        self.symgrad_shape = None

        self._NSlice = par["NSlice"]
        self._par_slices = par["par_slices"]
        self._overlap = par["overlap"]

        self.dat_trans_axes = [2, 0, 1, 3, 4]
        self.data_shape = (par["NSlice"], par["NScan"],
                           par["NC"], par["Nproj"], par["N"])

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


class PDSoftSenseSolverStreamedTV(PDSoftSenseBaseSolverStreamed):
    """Streamed PD Soft-SENSE TV version.

    Parameters
    ----------
        par : dict
          A python dict containing the necessary information to
          setup the object. Needs to contain the number of slices (NSlice),
          number of scans (NScan), image dimensions (dimX, dimY), number of
          coils (NC), sampling points (N) and read outs (NProj)
          a PyOpenCL queue (queue) and the complex coil
          sensitivities (C).
        pdsose_par : dict
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
    """

    def __init__(self,
                 par,
                 ss_par,
                 queue,
                 tau,
                 fval,
                 prg,
                 linop,
                 coils,
                 **kwargs
                 ):

        super().__init__(
            par,
            ss_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            **kwargs)

        self._op = linop[0]
        self._grad_op = linop[1]

        self._setupstreamingops('TV')

    def _setupstreamingops(self, reg_type):

        self.stream_initial = self._defineoperator(
            [],
            [],
            [[]],
            reverse_dir=True)
        self.stream_initial += self._op.fwdstr
        self.stream_initial += self._op.adjstrKyk1

        self.stream_update_x = self._defineoperator(
            [self.update_x],
            [self.unknown_shape],
            [[self.unknown_shape,
              self.unknown_shape]])

        self.update_primal = self._defineoperator(
            [],
            [],
            [[]])

        self.update_primal += self.stream_update_x

        self.stream_update_y = self._defineoperator(
            [self.update_y],
            [self.data_shape],
            [[self.data_shape,
              self.data_shape,
              self.data_shape]])

        self.stream_update_z1 = self._defineoperator(
            [self.update_z_tv],
            [self.grad_shape],
            [[self.grad_shape,
              self. grad_shape]],
            reverse_dir=True)

        self.update_dual = self._defineoperator(
            [],
            [],
            [[]],
            reverse_dir=True)

        self.update_dual += self.stream_update_z1
        self.update_dual += self.stream_update_y
        self.update_dual += self._op.adjstrKyk1

        self.update_dual.connectouttoin(0, (2, 2))
        self.update_dual.connectouttoin(1, (2, 0))

        self.stream_extrapolate_x = self._defineoperator(
            [self.extrapolate_x],
            [self.unknown_shape],
            [[self.unknown_shape,
              self.unknown_shape]])

        self.extrapolate_primal = self._defineoperator(
            [],
            [],
            [[]])

        self.extrapolate_primal += self.stream_extrapolate_x
        self.extrapolate_primal += self._op.fwdstr
        self.extrapolate_primal += self._grad_op.getStreamedGradientObject()

        self.extrapolate_primal.connectouttoin(0, (1, 0))
        self.extrapolate_primal.connectouttoin(0, (2, 0))

    def _setupVariables(self, inp, data):
        primal_vars = {}
        primal_vars_new = {}
        tmp_results_adjoint = {}

        primal_vars["x"] = inp.copy()
        primal_vars_new["x"] = np.zeros_like(primal_vars["x"])

        tmp_results_adjoint["Kyk1"] = np.zeros_like(primal_vars["x"])

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}

        dual_vars["y"] = np.zeros(
            data.shape,
            dtype=self._DTYPE
        )
        dual_vars_new["y"] = np.zeros_like(dual_vars["y"])

        dual_vars["z"] = np.zeros(
            primal_vars["x"].shape + (4,),
            dtype=self._DTYPE
        )
        dual_vars_new["z"] = np.zeros_like(dual_vars["z"])

        tmp_results_forward["Kx"] = np.zeros_like(data)
        tmp_results_forward["gradx"] = np.zeros_like(dual_vars["z"])

        return (primal_vars,
                primal_vars_new,
                tmp_results_forward,
                dual_vars,
                dual_vars_new,
                tmp_results_adjoint,
                data)

    def _updateInitial(self,
                       out_fwd,
                       out_adj,
                       in_primal,
                       in_dual):

        self.stream_initial.eval(
            [
                out_fwd["Kx"],
                out_adj["Kyk1"]
            ],
            [
                [in_primal["x"], self._coils],
                [in_dual["y"], self._coils, in_dual["z"]]
            ],
            [
                [],
                [self._grad_op.ratio]
            ]
        )

    def _updatePrimal(self,
                      out_primal,
                      out_fwd,
                      in_primal,
                      in_precomp_adj,
                      tau,
                      theta):

        self.update_primal.eval(
            [
                out_primal["x"]
            ],
            [
                [in_primal["x"], in_precomp_adj["Kyk1"]],

            ],
            [
                [tau, 0]
            ]
        )

    def _updateDual(self,
                    out_dual,
                    out_adj,
                    in_primal,
                    in_dual,
                    in_precomp_fwd,
                    data,
                    sigma):

        self.update_dual.eval(
            [
                out_dual["z"],
                out_dual["y"],
                out_adj["Kyk1"]
            ],
            [
                [in_dual["z"], in_precomp_fwd["gradx"]],
                [in_dual["y"], in_precomp_fwd["Kx"], data],
                [[], self._coils, []]
            ],
            [
                [sigma, ],
                [sigma, self.lambd],
                [self._grad_op.ratio]
            ]
        )

    def _updateStepSize(self,
                        primal_vars,
                        primal_vars_new,
                        tmp_results_forward,
                        theta,
                        tau,
                        sigma):

        diffx = primal_vars_new["x"] - primal_vars["x"]
        nx = np.vdot(diffx, diffx).real ** (1 / 2)
        fwddiffx = self._op.fwdoop([[diffx, self._coils]])
        graddiffx = self._grad_op.fwdoop([diffx])
        nKx = (
            np.vdot(fwddiffx, fwddiffx) +
            np.vdot(graddiffx, graddiffx)
        ).real ** (1 / 2)
        n = nx / nKx if nKx != 0 else 0
        s = sigma * tau  # np.sqrt(sigma * tau)
        fac = theta * sigma * tau  # np.sqrt(theta * sigma * tau)

        if fac >= n > 0:
            s = n
        elif s >= n > fac:
            s = np.sqrt(fac)
        else:
            s = np.sqrt(s)

        return s, s

    def _extrapolatePrimal(self,
                           out_primal,
                           out_fwd,
                           in_primal,
                           theta):
        self.extrapolate_primal.eval(
            [
                out_primal["x"],
                out_fwd["Kx"],
                out_fwd["gradx"]
            ],
            [
                [in_primal["x"], out_primal["x"]],
                [[], self._coils],
                [[]]
            ],
            [
                [theta, ],
                [],
                []
            ]
        )

    def _calcResidual(self,
                      in_primal,
                      in_dual,
                      in_precomp_fwd,
                      data):

        Kx = self._op.fwdoop([[in_primal["x"], self._coils]])
        gradx = self._grad_op.fwdoop([in_primal["x"]])

        primal = (
            self.lambd / 2 * np.vdot(
                (Kx - data),
                (Kx - data))
            + np.sum(
                abs(gradx)
            )
        ).real

        dual = (
            1 / (2 * self.lambd) * np.vdot(
                in_dual["y"],
                in_dual["y"]
            )
        ).real

        gap = np.abs(primal - dual)
        return primal, dual, gap


class PDSoftSenseSolverStreamedTGV(PDSoftSenseBaseSolverStreamed):
    """Streamed PD Soft-SENSE TGV version.

    Parameters
    ----------
        par : dict
          A python dict containing the necessary information to
          setup the object. Needs to contain the number of slices (NSlice),
          number of scans (NScan), image dimensions (dimX, dimY), number of
          coils (NC), sampling points (N) and read outs (NProj)
          a PyOpenCL queue (queue) and the complex coil
          sensitivities (C).
        pdsose_par : dict
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

    Attributes
    ----------
      alpha_0 : float
        alpha0 parameter for TGV regularization weight
      alpha_1 : float
        alpha1 parameter for TGV regularization weight
      symgrad_shape : tuple of int
        Size of the symmetrized gradient
    """

    def __init__(self,
                 par,
                 ss_par,
                 queue,
                 tau,
                 fval,
                 prg,
                 linop,
                 coils,
                 **kwargs):

        super().__init__(
            par,
            ss_par,
            queue,
            tau,
            fval,
            prg,
            coils,
            **kwargs)

        self.alpha_0 = ss_par["alpha0"]
        self.alpha_1 = ss_par["alpha1"]

        self._op = linop[0]
        self._grad_op = linop[1]
        self._symgrad_op = linop[2]

        self.symgrad_shape = self.unknown_shape + (8,)

        self._setupstreamingops('TGV')

    def _setupstreamingops(self, reg_type):

        self.stream_initial_1 = self._defineoperator(
            [],
            [],
            [[]],
            reverse_dir=True)
        self.stream_initial_1 += self._op.fwdstr
        self.stream_initial_1 += self._op.adjstrKyk1
        self.stream_initial_1 += self._symgrad_op.getStreamedSymGradientObject()

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

        self.stream_update_x = self._defineoperator(
            [self.update_x],
            [self.unknown_shape],
            [[self.unknown_shape,
              self.unknown_shape]])

        self.update_primal_1 = self._defineoperator(
            [],
            [],
            [[]])

        self.update_primal_1 += self.stream_update_x

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

        self.stream_update_z1 = self._defineoperator(
            [self.update_z1_tgv],
            [self.grad_shape],
            [[self.grad_shape,
              self.grad_shape,
              self.grad_shape]],
            reverse_dir=True)

        self.stream_update_y = self._defineoperator(
            [self.update_y],
            [self.data_shape],
            [[self.data_shape,
              self.data_shape,
              self.data_shape]])

        self.update_dual_1 = self._defineoperator(
            [],
            [],
            [[]],
            reverse_dir=True)

        self.update_dual_1 += self.stream_update_z1
        self.update_dual_1 += self.stream_update_y
        self.update_dual_1 += self._op.adjstrKyk1
        self.update_dual_1.connectouttoin(0, (2, 2))
        self.update_dual_1.connectouttoin(1, (2, 0))

        self.stream_update_z2 = self._defineoperator(
            [self.update_z2_tgv],
            [self.symgrad_shape],
            [[self.symgrad_shape,
              self.symgrad_shape]])
        self.update_dual_2 = self._defineoperator(
            [],
            [],
            [[]])

        self.update_dual_2 += self.stream_update_z2
        self.update_dual_2 += self.stream_Kyk2

        self.update_dual_2.connectouttoin(0, (1, 0))

        self.stream_extrapolate_x = self._defineoperator(
            [self.extrapolate_x],
            [self.unknown_shape],
            [[self.unknown_shape,
              self.unknown_shape]])

        self.extrapolate_primal_1 = self._defineoperator(
            [],
            [],
            [[]])

        self.extrapolate_primal_1 += self.stream_extrapolate_x
        self.extrapolate_primal_1 += self._op.fwdstr
        self.extrapolate_primal_1 += self._grad_op.getStreamedGradientObject()

        self.extrapolate_primal_1.connectouttoin(0, (1, 0))
        self.extrapolate_primal_1.connectouttoin(0, (2, 0))

        self.stream_extrapolate_v = self._defineoperator(
            [self.extrapolate_v],
            [self.grad_shape],
            [[self.grad_shape,
              self.grad_shape]])

        self.extrapolate_primal_2 = self._defineoperator(
            [],
            [],
            [[]],
            reverse_dir=True)

        self.extrapolate_primal_2 += self.stream_extrapolate_v
        self.extrapolate_primal_2 += self._symgrad_op.getStreamedSymGradientObject()

        self.extrapolate_primal_2.connectouttoin(0, (1, 0))

        self.step_size_update = self._defineoperator(
            [],
            [],
            [[]],
            reverse_dir=True)
        self.step_size_update += self._op.fwdstr
        self.step_size_update += self._grad_op.getStreamedGradientObject()
        self.step_size_update += self._symgrad_op.getStreamedSymGradientObject()

    def _setupVariables(self, inp, data):
        primal_vars = {}
        primal_vars_new = {}
        tmp_results_adjoint = {}

        primal_vars["x"] = inp
        primal_vars["v"] = np.zeros(
            primal_vars["x"].shape + (4,), dtype=self._DTYPE)

        primal_vars_new["x"] = np.zeros_like(primal_vars["x"])
        primal_vars_new["v"] = np.zeros_like(primal_vars["v"])

        tmp_results_adjoint["Kyk1"] = np.zeros_like(primal_vars["x"])
        tmp_results_adjoint["Kyk2"] = np.zeros_like(primal_vars["v"])

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}

        dual_vars["y"] = np.zeros(
            data.shape,
            dtype=self._DTYPE
        )

        dual_vars["z1"] = np.zeros(
            primal_vars["x"].shape + (4,),
            dtype=self._DTYPE
        )

        dual_vars["z2"] = np.zeros(
            primal_vars["x"].shape + (8,),
            dtype=self._DTYPE
        )
        dual_vars_new["y"] = np.zeros_like(dual_vars["y"])
        dual_vars_new["z1"] = np.zeros_like(dual_vars["z1"])
        dual_vars_new["z2"] = np.zeros_like(dual_vars["z2"])

        tmp_results_forward["Kx"] = np.zeros_like(data)
        tmp_results_forward["gradx"] = np.zeros_like(dual_vars["z1"])
        tmp_results_forward["symgradv"] = np.zeros_like(dual_vars["z2"])

        return (primal_vars,
                primal_vars_new,
                tmp_results_forward,
                dual_vars,
                dual_vars_new,
                tmp_results_adjoint,
                data)

    def _updateInitial(self,
                       out_fwd,
                       out_adj,
                       in_primal,
                       in_dual):

        self.stream_initial_1.eval(
            [
                out_fwd["Kx"],
                out_adj["Kyk1"],
                out_fwd["symgradv"]
            ],
            [
                [in_primal["x"], self._coils],
                [in_dual["y"], self._coils, in_dual["z1"]],
                [in_primal["v"]]
            ],
            [
                [],
                [self._grad_op.ratio],
                []
            ]
        )
        self.stream_initial_2.eval(
            [
                out_fwd["gradx"],
                out_adj["Kyk2"]
            ],
            [
                [in_primal["x"]],
                [in_dual["z2"], in_dual["z1"], []]
            ],
            [
                [],
                self._symgrad_op.ratio
            ]
        )

    def _updatePrimal(self,
                      out_primal,
                      out_fwd,
                      in_primal,
                      in_precomp_adj,
                      tau,
                      theta):

        self.update_primal_1.eval(
            [
                out_primal["x"]
            ],
            [
                [in_primal["x"], in_precomp_adj["Kyk1"]]
            ],
            [
                [tau, 0]
            ]
        )

        self.update_primal_2.eval(
            [
                out_primal["v"]
            ],
            [
                [in_primal["v"], in_precomp_adj["Kyk2"]]
            ],
            [
                [tau, 0]
            ]
        )

    def _updateDual(self,
                    out_dual,
                    out_adj,
                    in_primal,
                    in_dual,
                    in_precomp_fwd,
                    data,
                    sigma):

        self.update_dual_1.eval(
            [
                out_dual["z1"],
                out_dual["y"],
                out_adj["Kyk1"]
            ],
            [
                [in_dual["z1"], in_precomp_fwd["gradx"], in_primal["v"]],
                [in_dual["y"], in_precomp_fwd["Kx"], data],
                [[], self._coils, []]
            ],
            [
                [sigma, self.alpha_1],
                [sigma, self.lambd],
                [self._grad_op.ratio]
            ]
        )

        self.update_dual_2.eval(
            [
                out_dual["z2"],
                out_adj["Kyk2"]
            ],
            [
                [in_dual["z2"], in_precomp_fwd["symgradv"]],
                [[], out_dual["z1"], []]
            ],
            [
                [sigma, self.alpha_0],
                self._symgrad_op.ratio
            ]
        )

    def _updateStepSize(self,
                        primal_vars,
                        primal_vars_new,
                        tmp_results_forward,
                        theta,
                        tau,
                        sigma):
        diffx = primal_vars_new["x"] - primal_vars["x"]
        diffv = primal_vars_new["v"] - primal_vars["v"]
        nx = (
            np.vdot(
                diffx,
                diffx
            )
            + np.vdot(
                diffv,
                diffv,
            )
        ).real ** (1 / 2)

        fwddiffx = np.zeros(self.data_shape, dtype=self._DTYPE)
        graddiffx = np.zeros(self.grad_shape, dtype=self._DTYPE)
        symgraddiffv = np.zeros(self.symgrad_shape, dtype=self._DTYPE)

        self.step_size_update.eval(
            [
                fwddiffx,
                graddiffx,
                symgraddiffv
            ],
            [
                [diffx, self._coils],
                [diffx],
                [diffv]
            ],
            [
                [],
                [],
                []
            ]
        )

        nKx = (
            np.vdot(
                fwddiffx,
                fwddiffx
            )
            + np.vdot(
                graddiffx - diffv,
                graddiffx - diffv
            )
            + np.vdot(
                symgraddiffv,
                symgraddiffv
            )
        ).real ** (1 / 2)
        n = nx / nKx if nKx != 0 else 0
        s = sigma * tau  # np.sqrt(sigma * tau)
        fac = theta * sigma * tau  # np.sqrt(theta * sigma * tau)

        if fac >= n > 0:
            s = n
        elif s >= n > fac:
            s = np.sqrt(fac)
        else:
            s = np.sqrt(s)

        return s, s

    def _extrapolatePrimal(self,
                           out_primal,
                           out_fwd,
                           in_primal,
                           theta):
        self.extrapolate_primal_1.eval(
            [
                out_primal["x"],
                out_fwd["Kx"],
                out_fwd["gradx"]
            ],
            [
                [in_primal["x"], out_primal["x"]],
                [[], self._coils],
                [[]]
            ],
            [
                [theta, ],
                [],
                []
            ]
        )

        self.extrapolate_primal_2.eval(
            [
                out_primal["v"],
                out_fwd["symgradv"]
            ],
            [
                [in_primal["v"], out_primal["v"]],
                [[]]
            ],
            [
                [theta, ],
                []
            ]
        )

    def _calcResidual(self,
                      in_primal,
                      in_dual,
                      in_precomp_fwd,
                      data):

        Kx = self._op.fwdoop([[in_primal["x"], self._coils]])
        gradx = self._grad_op.fwdoop([in_primal["x"]])
        symgradv = self._symgrad_op.fwdoop([in_primal["v"]])

        primal = (
            self.lambd / 2 * np.vdot(
                (Kx - data),
                (Kx - data))
            + self.alpha_1 * np.sum(
                abs(gradx - in_primal["v"])
            )
            + self.alpha_0 * np.sum(
                abs(symgradv)
            )
        ).real

        dual = (
            1 / (2 * self.lambd) * np.vdot(
                in_dual["y"],
                in_dual["y"]
            )
        ).real

        gap = np.abs(primal - dual)
        return primal, dual, gap



class PDSolverICTV(PDBaseSolver):
    """Primal Dual splitting optimization for IC-TV.

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
        
        s = irgn_par["s"]
        if s == 0:
            self.gamma_1 = 1
            self.gamma_2 = 1e10
        elif s == 1:
            self.gamma_1 = 1e10
            self.gamma_2 = 1
        else:
            self.gamma_1 = s/np.minimum(s, 1-s)
            self.gamma_2 = (1-s)/np.minimum(s, 1-s)
        
        
        self._op = linop[0]
        self._grad_op_1 = linop[1][0]
        self._grad_op_2 = linop[1][0]

    def _setupVariables(self, inp, data):

        data = clarray.to_device(self._queue[0], data.astype(self._DTYPE))

        primal_vars = {}
        primal_vars_new = {}
        tmp_results_adjoint = {}
        tmp_results_adjoint_new = {}

        primal_vars["x"] = clarray.to_device(self._queue[0], inp[0])
        primal_vars_new["x"] = clarray.zeros_like(primal_vars["x"])
        primal_vars["v"] = clarray.zeros_like(primal_vars["x"])
        primal_vars_new["v"] = clarray.zeros_like(primal_vars["x"])

        tmp_results_adjoint["Kyk1"] = clarray.zeros_like(primal_vars["x"])
        tmp_results_adjoint_new["Kyk1"] = clarray.zeros_like(primal_vars["x"])

        tmp_results_adjoint["Kyk2"] = clarray.zeros(
                                        self._queue[0],
                                        primal_vars["x"].shape+(4,),
                                        dtype=self._DTYPE)
        tmp_results_adjoint_new["Kyk2"] = clarray.zeros(
                                        self._queue[0],
                                        primal_vars["x"].shape+(4,),
                                        dtype=self._DTYPE)

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}
        tmp_results_forward_new = {}
        dual_vars["r"] = clarray.zeros(
            self._queue[0],
            data.shape,
            dtype=self._DTYPE)
        dual_vars_new["r"] = clarray.zeros_like(dual_vars["r"])

        dual_vars["z1"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape + (4,),
                                        dtype=self._DTYPE)
        dual_vars_new["z1"] = clarray.zeros_like(dual_vars["z1"])

        dual_vars["z2"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape + (4,),
                                        dtype=self._DTYPE)
        dual_vars_new["z2"] = clarray.zeros_like(dual_vars["z2"])

        tmp_results_forward["gradx1"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx1"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward["gradx2"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx2"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward["gradx3"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx3"] = clarray.zeros_like(
            dual_vars["z1"])

        tmp_results_forward["Ax"] = clarray.zeros_like(data)
        tmp_results_forward_new["Ax"] = clarray.zeros_like(data)

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

        out_fwd["Ax"].add_event(self._op.fwd(
            out_fwd["Ax"], [in_primal["x"], self._coils, self.modelgrad]))


        out_fwd["gradx1"].add_event(
            self._grad_op_1.fwd(out_fwd["gradx1"], in_primal["x"]))
        out_fwd["gradx2"].add_event(
            self._grad_op_1.fwd(out_fwd["gradx2"], in_primal["v"]))
        out_fwd["gradx3"].add_event(
            self._grad_op_2.fwd(out_fwd["gradx3"], in_primal["v"]))

        (self._op.adj(
            out_adj["Kyk1"], [in_dual["r"], self._coils, self.modelgrad])).wait()
        out_adj["Kyk1"] = out_adj["Kyk1"] - self._grad_op_1.adjoop(in_dual["z1"])

        out_adj["Kyk2"] = self._grad_op_1.adjoop(in_dual["z1"]) - self._grad_op_2.adjoop(in_dual["z2"])

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        out_primal["x"].add_event(self.update_primal(
            outp=out_primal["x"],
            inp=(in_primal["x"], in_precomp_adj["Kyk1"]),
            par=(tau, self.delta)))

        out_primal["v"].add_event(self.update_v(
            outp=out_primal["v"],
            inp=(in_primal["v"], in_precomp_adj["Kyk2"]),
            par=(tau,)))

        out_fwd["gradx1"].add_event(
            self._grad_op_1.fwd(
                out_fwd["gradx1"], out_primal["x"]))
        out_fwd["gradx2"].add_event(
            self._grad_op_1.fwd(
                out_fwd["gradx2"], out_primal["v"]))

        out_fwd["gradx3"].add_event(
            self._grad_op_2.fwd(
                out_fwd["gradx3"], out_primal["v"]))


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
            self.update_z1_ictv(
                outp=out_dual["z1"],
                inp=(
                        in_dual["z1"],
                        in_precomp_fwd_new["gradx1"],
                        in_precomp_fwd["gradx1"],
                        in_precomp_fwd_new["gradx2"],
                        in_precomp_fwd["gradx2"]
                    ),
                par=(beta*tau, theta, self.alpha*self.gamma_1)
                )
            )
        out_dual["z2"].add_event(
            self.update_z2_ictv(
                outp=out_dual["z2"],
                inp=(
                        in_dual["z2"],
                        in_precomp_fwd_new["gradx3"],
                        in_precomp_fwd["gradx3"]
                    ),
                par=(beta*tau, theta, self.alpha*self.gamma_2)
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

        (self._op.adj(
            out_adj["Kyk1"], [out_dual["r"], self._coils, self.modelgrad])).wait()
        out_adj["Kyk1"] = out_adj["Kyk1"] - self._grad_op_1.adjoop(out_dual["z1"])
    
        out_adj["Kyk2"] = self._grad_op_1.adjoop(out_dual["z1"]) - self._grad_op_2.adjoop(out_dual["z2"])

        ynorm = (
            (
                self.normkrnldiff(out_dual["r"], in_dual["r"])
                + self.normkrnldiff(out_dual["z1"], in_dual["z1"])
                + self.normkrnldiff(out_dual["z1"], in_dual["z1"])
            )**(1 / 2))
        lhs = np.sqrt(beta) * tau * (
            (
                self.normkrnldiff(out_adj["Kyk1"], in_precomp_adj["Kyk1"])
                + self.normkrnldiff(out_adj["Kyk2"], in_precomp_adj["Kyk2"])
            )**(1 / 2))
        return lhs.get(), ynorm.get()

    def _calcResidual(
            self,
            in_primal,
            in_dual,
            in_precomp_fwd,
            in_precomp_adj,
            data):
        
        primal_new = (
            self.lambd / 2 *
            self.normkrnldiff(in_precomp_fwd["Ax"], data)
            + self.alpha * self.gamma_1 * self.abskrnldiff(in_precomp_fwd["gradx1"],in_precomp_fwd["gradx2"])
            + self.alpha * self.gamma_2 * self.abskrnl(in_precomp_fwd["gradx3"])
            ).real

        dual = (
            - clarray.sum(in_precomp_adj["Kyk1"])
            - clarray.sum(in_precomp_adj["Kyk2"])
            - 1 / (2 * self.lambd) * self.normkrnl(in_dual["r"])
            - clarray.vdot(data, in_dual["r"])
            ).real

        if self.unknowns_H1 > 0:
            primal_new += (
                 self.omega / 2 * self.normkrnl(
                     in_precomp_fwd["gradx"][self.unknowns_TGV:]
                     )
                 ).real

            dual += (
                - 1 / (2 * self.omega) * self.normkrnl(
                    in_dual["z1"][self.unknowns_TGV:]
                    )
                ).real
        gap = np.abs(primal_new - dual)
        return primal_new.get(), dual.get(), gap.get()

    def update_z2_ictv(self, outp, inp, par=None, idx=0, idxq=0,
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
            
        return self._prg[idx].update_z2_ictv(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, 
            inp[1].data, 
            inp[2].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[1]),
            self._DTYPE_real(1/par[2]), np.int32(self.unknowns_TGV),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def update_z1_ictv(self, outp, inp, par=None, idx=0, idxq=0,
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
            
        return self._prg[idx].update_z1_ictv(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, 
            inp[0].data, 
            inp[1].data, 
            inp[2].data,
            inp[3].data, 
            inp[4].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[1]),
            self._DTYPE_real(1/par[2]), np.int32(self.unknowns_TGV),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

        

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
            
        if not self.precond:
            return self._prg[idx].update_primal_imagerecon(
            self._queue[4 * idx + idxq],
                self._kernelsize, None,
                outp.data, inp[0].data, inp[1].data,
                self._DTYPE_real(par[0]),
                self.min_const[idx].data, self.max_const[idx].data,
                self.real_const[idx].data, np.int32(self.unknowns),
                wait_for=(outp.events +
                          inp[0].events+inp[1].events +
                          wait_for))
        
        
class PDSolverICTGV(PDBaseSolver):
    """Primal Dual splitting optimization for IC-TV.

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
        self.beta = 2*irgn_par["gamma"]
        
        s = irgn_par["s"]
        if s == 0:
            self.gamma_1 = 1
            self.gamma_2 = 1e10
        elif s == 1:
            self.gamma_1 = 1e10
            self.gamma_2 = 1
        else:
            self.gamma_1 = s/np.minimum(s, 1-s)
            self.gamma_2 = (1-s)/np.minimum(s, 1-s)
        
        
        self._op = linop[0]
        self._grad_op_1 = linop[1][0]
        self._grad_op_2 = linop[1][1]
        self._symgrad_op_1 = linop[2][0]
        self._symgrad_op_2 = linop[2][1]

    def _setupVariables(self, inp, data):

        data = clarray.to_device(self._queue[0], data.astype(self._DTYPE))

        primal_vars = {}
        primal_vars_new = {}
        tmp_results_adjoint = {}
        tmp_results_adjoint_new = {}

        primal_vars["x"] = clarray.to_device(self._queue[0], inp[0])
        primal_vars_new["x"] = clarray.zeros_like(primal_vars["x"])
        primal_vars["v"] = clarray.zeros_like(primal_vars["x"])
        primal_vars_new["v"] = clarray.zeros_like(primal_vars["x"])
        
        primal_vars["w1"] = clarray.zeros(self._queue[0],primal_vars["x"].shape+(4,), dtype=self._DTYPE)
        primal_vars_new["w1"] = clarray.zeros_like(primal_vars["w1"])
        primal_vars["w2"] = clarray.zeros_like(primal_vars["w1"])
        primal_vars_new["w2"] = clarray.zeros_like(primal_vars["w1"])
        

        tmp_results_adjoint["Kyk1"] = clarray.zeros_like(primal_vars["x"])
        tmp_results_adjoint_new["Kyk1"] = clarray.zeros_like(primal_vars["x"])

        tmp_results_adjoint["Kyk2"] = clarray.zeros_like(primal_vars["x"])
        tmp_results_adjoint_new["Kyk2"] = clarray.zeros_like(primal_vars["x"])
        
        tmp_results_adjoint["Kyk3"] = clarray.zeros(
                                        self._queue[0],
                                        primal_vars["x"].shape+(4,),
                                        dtype=self._DTYPE)
        tmp_results_adjoint_new["Kyk3"] = clarray.zeros(
                                        self._queue[0],
                                        primal_vars["x"].shape+(4,),
                                        dtype=self._DTYPE)
        
        tmp_results_adjoint["Kyk4"] = clarray.zeros(
                                        self._queue[0],
                                        primal_vars["x"].shape+(4,),
                                        dtype=self._DTYPE)
        tmp_results_adjoint_new["Kyk4"] = clarray.zeros(
                                        self._queue[0],
                                        primal_vars["x"].shape+(4,),
                                        dtype=self._DTYPE)

        dual_vars = {}
        dual_vars_new = {}
        tmp_results_forward = {}
        tmp_results_forward_new = {}
        dual_vars["r"] = clarray.zeros(
            self._queue[0],
            data.shape,
            dtype=self._DTYPE)
        dual_vars_new["r"] = clarray.zeros_like(dual_vars["r"])

        dual_vars["z1"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape + (4,),
                                        dtype=self._DTYPE)
        dual_vars_new["z1"] = clarray.zeros_like(dual_vars["z1"])

        dual_vars["z2"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape + (4,),
                                        dtype=self._DTYPE)
        dual_vars_new["z2"] = clarray.zeros_like(dual_vars["z2"])
        
        dual_vars["z3_diag"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape + (4,),
                                        dtype=self._DTYPE)
        dual_vars_new["z3_diag"] = clarray.zeros_like(dual_vars["z3_diag"])
        
        dual_vars["z4_diag"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape + (4,),
                                        dtype=self._DTYPE)
        dual_vars_new["z4_diag"] = clarray.zeros_like(dual_vars["z4_diag"])
        
        dual_vars["z3_offdiag"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape + (8,),
                                        dtype=self._DTYPE)
        dual_vars_new["z3_offdiag"] = clarray.zeros_like(dual_vars["z3_offdiag"])
        
        dual_vars["z4_offdiag"] = clarray.zeros(self._queue[0],
                                        primal_vars["x"].shape + (8,),
                                        dtype=self._DTYPE)
        dual_vars_new["z4_offdiag"] = clarray.zeros_like(dual_vars["z4_offdiag"])

        tmp_results_forward["gradx1"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx1"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward["gradx2"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx2"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward["gradx3"] = clarray.zeros_like(
            dual_vars["z1"])
        tmp_results_forward_new["gradx3"] = clarray.zeros_like(
            dual_vars["z1"])
        
        tmp_results_forward["symgradx1_diag"] = clarray.zeros_like(
            dual_vars["z3_diag"])
        tmp_results_forward_new["symgradx1_diag"] = clarray.zeros_like(
            dual_vars["z3_diag"])
        
        tmp_results_forward["symgradx2_diag"] = clarray.zeros_like(
            dual_vars["z4_diag"])
        tmp_results_forward_new["symgradx2_diag"] = clarray.zeros_like(
            dual_vars["z4_diag"])
        
        tmp_results_forward["symgradx1_offdiag"] = clarray.zeros_like(
            dual_vars["z3_offdiag"])
        tmp_results_forward_new["symgradx1_offdiag"] = clarray.zeros_like(
            dual_vars["z3_offdiag"])
        
        tmp_results_forward["symgradx2_offdiag"] = clarray.zeros_like(
            dual_vars["z4_offdiag"])
        tmp_results_forward_new["symgradx2_offdiag"] = clarray.zeros_like(
            dual_vars["z4_offdiag"])
        

        tmp_results_forward["Ax"] = clarray.zeros_like(data)
        tmp_results_forward_new["Ax"] = clarray.zeros_like(data)

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

        out_fwd["Ax"].add_event(self._op.fwd(
            out_fwd["Ax"], [in_primal["x"], self._coils, self.modelgrad]))


        out_fwd["gradx1"].add_event(
            self._grad_op_1.fwd(out_fwd["gradx1"], in_primal["x"]))
        out_fwd["gradx2"].add_event(
            self._grad_op_1.fwd(out_fwd["gradx2"], in_primal["v"]))
        out_fwd["gradx3"].add_event(
            self._grad_op_2.fwd(out_fwd["gradx3"], in_primal["v"]))
        
        symgradevent = self._symgrad_op_1.fwd([out_fwd["symgradx1_diag"], out_fwd["symgradx1_offdiag"]], in_primal["w1"])
        out_fwd["symgradx1_diag"].add_event(symgradevent)
        out_fwd["symgradx1_offdiag"].add_event(symgradevent)
        symgradevent = self._symgrad_op_2.fwd([out_fwd["symgradx2_diag"], out_fwd["symgradx2_offdiag"]], in_primal["w2"])
        out_fwd["symgradx2_diag"].add_event(symgradevent)
        out_fwd["symgradx2_offdiag"].add_event(symgradevent)            
                     
        out_fwd["gradx3"].add_event(
            self._grad_op_2.fwd(out_fwd["gradx3"], in_primal["v"]))

        (self._op.adj(
            out_adj["Kyk1"], [in_dual["r"], self._coils, self.modelgrad])).wait()
        out_adj["Kyk1"] = out_adj["Kyk1"] - self._grad_op_1.adjoop(in_dual["z1"])

        out_adj["Kyk2"] = self._grad_op_1.adjoop(in_dual["z1"]) - self._grad_op_2.adjoop(in_dual["z2"])
        
        out_adj["Kyk3"] = - in_dual["z1"] - self._symgrad_op_1.adjoop([in_dual["z3_diag"], in_dual["z3_offdiag"]])
        out_adj["Kyk4"] = - in_dual["z2"] - self._symgrad_op_2.adjoop([in_dual["z4_diag"], in_dual["z4_offdiag"]])
        

    def _updatePrimal(self,
                      out_primal, out_fwd,
                      in_primal, in_precomp_adj,
                      tau):
        out_primal["x"].add_event(self.update_primal(
            outp=out_primal["x"],
            inp=(in_primal["x"], in_precomp_adj["Kyk1"]),
            par=(tau, self.delta)))

        out_primal["v"].add_event(self.update_v(
            outp=out_primal["v"],
            inp=(in_primal["v"], in_precomp_adj["Kyk2"]),
            par=(tau,)))
        
        out_primal["w1"].add_event(self.update_v(
            outp=out_primal["w1"],
            inp=(in_primal["w1"], in_precomp_adj["Kyk3"]),
            par=(tau,)))
        
        out_primal["w2"].add_event(self.update_v(
            outp=out_primal["w2"],
            inp=(in_primal["w2"], in_precomp_adj["Kyk4"]),
            par=(tau,)))

        out_fwd["gradx1"].add_event(
            self._grad_op_1.fwd(
                out_fwd["gradx1"], out_primal["x"]))
        out_fwd["gradx2"].add_event(
            self._grad_op_1.fwd(
                out_fwd["gradx2"], out_primal["v"]))

        out_fwd["gradx3"].add_event(
            self._grad_op_2.fwd(
                out_fwd["gradx3"], out_primal["v"]))
        
        symgradevent = self._symgrad_op_1.fwd([out_fwd["symgradx1_diag"], out_fwd["symgradx1_offdiag"]], out_primal["w1"])
        out_fwd["symgradx1_diag"].add_event(symgradevent)
        out_fwd["symgradx1_offdiag"].add_event(symgradevent)
        symgradevent = self._symgrad_op_2.fwd([out_fwd["symgradx2_diag"], out_fwd["symgradx2_offdiag"]], out_primal["w2"])
        out_fwd["symgradx2_diag"].add_event(symgradevent)
        out_fwd["symgradx2_offdiag"].add_event(symgradevent)  


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
            self.update_z1_ictgv(
                outp=out_dual["z1"],
                inp=(
                        in_dual["z1"],
                        in_precomp_fwd_new["gradx1"],
                        in_precomp_fwd["gradx1"],
                        in_precomp_fwd_new["gradx2"],
                        in_precomp_fwd["gradx2"],
                        in_primal_new["w1"],
                        in_primal["w1"]
                    ),
                par=(beta*tau, theta, self.alpha*self.gamma_1)
                )
            )
        out_dual["z2"].add_event(
            self.update_z2_ictgv(
                outp=out_dual["z2"],
                inp=(
                        in_dual["z2"],
                        in_precomp_fwd_new["gradx3"],
                        in_precomp_fwd["gradx3"],
                        in_primal_new["w2"],
                        in_primal["w2"]
                    ),
                par=(beta*tau, theta, self.alpha*self.gamma_2)
                )
            )
        
        z3_events = self.update_z_sympart(
                outp=[out_dual["z3_diag"],out_dual["z3_offdiag"]],
                inp=(
                        in_dual["z3_diag"],
                        in_dual["z3_offdiag"],
                        in_precomp_fwd_new["symgradx1_diag"],
                        in_precomp_fwd_new["symgradx1_offdiag"],
                        in_precomp_fwd["symgradx1_diag"],
                        in_precomp_fwd["symgradx1_offdiag"]
                    ),
                par=(beta*tau, theta, self.beta*self.gamma_1)
                )
        out_dual["z3_diag"].add_event(z3_events)
        out_dual["z3_offdiag"].add_event(z3_events)
        z4_events = self.update_z_sympart(
                outp=[out_dual["z4_diag"],out_dual["z4_offdiag"]],
                inp=(
                        in_dual["z4_diag"],
                        in_dual["z4_offdiag"],
                        in_precomp_fwd_new["symgradx2_diag"],
                        in_precomp_fwd_new["symgradx2_offdiag"],
                        in_precomp_fwd["symgradx2_diag"],
                        in_precomp_fwd["symgradx2_offdiag"]
                    ),
                par=(beta*tau, theta, self.beta*self.gamma_2)
                )
        out_dual["z4_diag"].add_event(z4_events)
        out_dual["z4_offdiag"].add_event(z4_events)

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

        (self._op.adj(
            out_adj["Kyk1"], [out_dual["r"], self._coils, self.modelgrad])).wait()
        out_adj["Kyk1"] = out_adj["Kyk1"] - self._grad_op_1.adjoop(out_dual["z1"])
    
        out_adj["Kyk2"] = self._grad_op_1.adjoop(out_dual["z1"]) - self._grad_op_2.adjoop(out_dual["z2"])
        
        out_adj["Kyk3"] = - out_dual["z1"] - self._symgrad_op_1.adjoop([out_dual["z3_diag"], out_dual["z3_offdiag"]])
        out_adj["Kyk4"] = - out_dual["z2"] - self._symgrad_op_2.adjoop([out_dual["z4_diag"], out_dual["z4_offdiag"]])

        ynorm = (
            (
                self.normkrnldiff(out_dual["r"], in_dual["r"])
                + self.normkrnldiff(out_dual["z1"], in_dual["z1"])
                + self.normkrnldiff(out_dual["z1"], in_dual["z1"])
                + self.normkrnldiff(out_dual["z3_diag"], in_dual["z3_diag"])
                + self.normkrnldiff(out_dual["z3_offdiag"], in_dual["z3_offdiag"])
                + self.normkrnldiff(out_dual["z4_diag"], in_dual["z4_diag"])
                + self.normkrnldiff(out_dual["z4_offdiag"], in_dual["z4_offdiag"])
            )**(1 / 2))
        lhs = np.sqrt(beta) * tau * (
            (
                self.normkrnldiff(out_adj["Kyk1"], in_precomp_adj["Kyk1"])
                + self.normkrnldiff(out_adj["Kyk2"], in_precomp_adj["Kyk2"])
                + self.normkrnldiff(out_adj["Kyk3"], in_precomp_adj["Kyk3"])
                + self.normkrnldiff(out_adj["Kyk4"], in_precomp_adj["Kyk4"])
            )**(1 / 2))
        return lhs.get(), ynorm.get()

    def _calcResidual(
            self,
            in_primal,
            in_dual,
            in_precomp_fwd,
            in_precomp_adj,
            data):
        
        primal_new = (
            self.lambd / 2 *
            self.normkrnldiff(in_precomp_fwd["Ax"], data)
            + self.alpha * self.gamma_1 * self.abskrnldiff(in_precomp_fwd["gradx1"],in_precomp_fwd["gradx2"]+in_primal["w1"])
            + self.alpha * self.gamma_2 * self.abskrnldiff(in_precomp_fwd["gradx3"], in_primal["w2"])
            + self.beta * self.gamma_1* self.abskrnl(in_precomp_fwd["symgradx1_diag"])
            + self.beta * self.gamma_1* self.abskrnl(in_precomp_fwd["symgradx1_offdiag"])
            + self.beta * self.gamma_2* self.abskrnl(in_precomp_fwd["symgradx2_diag"])
            + self.beta * self.gamma_2* self.abskrnl(in_precomp_fwd["symgradx2_offdiag"])
            ).real

        dual = (
            - clarray.sum(in_precomp_adj["Kyk1"])
            - clarray.sum(in_precomp_adj["Kyk2"])
            - clarray.sum(in_precomp_adj["Kyk3"])
            - clarray.sum(in_precomp_adj["Kyk4"])
            - 1 / (2 * self.lambd) * self.normkrnl(in_dual["r"])
            - clarray.vdot(data, in_dual["r"])
            ).real

        if self.unknowns_H1 > 0:
            primal_new += (
                 self.omega / 2 * self.normkrnl(
                     in_precomp_fwd["gradx"][self.unknowns_TGV:]
                     )
                 ).real

            dual += (
                - 1 / (2 * self.omega) * self.normkrnl(
                    in_dual["z1"][self.unknowns_TGV:]
                    )
                ).real
        gap = np.abs(primal_new - dual)
        return primal_new.get(), dual.get(), gap.get()

    def update_z2_ictgv(self, outp, inp, par=None, idx=0, idxq=0,
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
            
        return self._prg[idx].update_z2_ictgv(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, inp[0].data, 
            inp[1].data, 
            inp[2].data,
            inp[3].data, 
            inp[4].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[1]),
            self._DTYPE_real(1/par[2]), np.int32(self.unknowns_TGV),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+
                      inp[3].events+inp[4].events + wait_for))

    def update_z1_ictgv(self, outp, inp, par=None, idx=0, idxq=0,
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
            
        return self._prg[idx].update_z1_ictgv(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp.data, 
            inp[0].data, 
            inp[1].data, 
            inp[2].data,
            inp[3].data, 
            inp[4].data,
            inp[5].data, 
            inp[6].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[1]),
            self._DTYPE_real(1/par[2]), np.int32(self.unknowns_TGV),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+
                      inp[3].events+inp[4].events+
                      inp[5].events+inp[6].events+ wait_for))
    
    def update_z_sympart(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=None):
        """Dual update of the z variable for the symmetrized gradient in Primal-Dual Algorithm forg TV.

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
            
        return self._prg[idx].update_z3_ictgv(
            self._queue[4*idx+idxq],
            self._kernelsize, None,
            outp[0].data, 
            outp[1].data, 
            inp[0].data, 
            inp[1].data, 
            inp[2].data,
            inp[3].data, 
            inp[4].data,
            inp[5].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[1]),
            self._DTYPE_real(1/par[2]), np.int32(self.unknowns_TGV),
            wait_for=(outp[0].events+outp[1].events+inp[0].events +
                      inp[1].events+inp[2].events+
                      inp[3].events+inp[4].events+inp[5].events+
                      wait_for))

        

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
            
        if not self.precond:
            return self._prg[idx].update_primal_imagerecon(
            self._queue[4 * idx + idxq],
                self._kernelsize, None,
                outp.data, inp[0].data, inp[1].data,
                self._DTYPE_real(par[0]),
                self.min_const[idx].data, self.max_const[idx].data,
                self.real_const[idx].data, np.int32(self.unknowns),
                wait_for=(outp.events +
                          inp[0].events+inp[1].events +
                          wait_for))