#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the classes for Soft Sense Optimization."""
from __future__ import division
import time
import numpy as np
import h5py
from PIL import Image
from pathlib import Path

from pkg_resources import resource_filename
import pyopencl.array as clarray

import pyqmri.operator as operator
import pyqmri.solver as optimizer
from pyqmri._helper_fun import CLProgram as Program
from pyqmri._helper_fun import _utils as utils
from pyqmri._my_helper_fun.display_data import img_montage
from pyqmri._my_helper_fun.helpers import sqrt_sum_of_squares, normalize_imgs, calc_image_metrics


class SoftSenseOptimizer:
    """Main Soft Sense Optimization class.

    This Class performs Soft Sense Optimization either with TGV, TV or without regularization.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      reg_type : str, "TGV"
        Select between "TGV" (default), "TV" or "" (without) regularization.
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
      ss_par : dict
        The parameters read from the config file to guide the Soft Sense
        optimization process
    """

    def __init__(self,
                 par,
                 myargs,
                 reg_type='TGV',
                 streamed=False,
                 DTYPE=np.complex64,
                 DTYPE_real=np.float32):

        self.args = myargs
        self.par = par
        self.pdsose_par = utils.read_config(
            myargs.config,
            optimizer="SSense",
            reg_type=reg_type
        )

        utils.save_config(self.pdsose_par, str(par["outdir"]), reg_type)
        num_dev = len(par["num_dev"])
        self._fval = 0
        self._fval_init = 0
        self._ctx = par["ctx"]
        self._queue = par["queue"]
        self._reg_type = reg_type
        self._prg = []

        self._DTYPE = DTYPE
        self._DTYPE_real = DTYPE_real

        self._elapsed_time = None
        self._i_term = -1

        self._streamed = streamed
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

        self._data_shape = (par["NScan"], par["NC"],
                            par["NSlice"], par["dimY"], par["dimX"])
        self._unknown_shape = (par["NMaps"], par["NSlice"],
                               par["dimY"], par["dimX"])

        if self._streamed:
            self._data_trans_axes = (2, 0, 1, 3, 4)
            self._cmaps = np.require(
                np.transpose(par["C"], self._data_trans_axes), requirements='C',
                dtype=DTYPE)
            self._unknown_shape = (par["NSlice"], par["NMaps"],
                                   par["dimY"], par["dimX"])
            self._data_shape = (par["NSlice"], par["NScan"], par["NC"],
                                par["dimY"], par["dimX"])
        else:
            self._cmaps = clarray.to_device(self._queue[0], self.par["C"])

        self._MRI_operator, self._FT = operator.Operator.SoftSenseOperatorFactory(
            par,
            self._prg,
            DTYPE,
            DTYPE_real,
            streamed
            )

        self._grad_op, self._symgrad_op, self._v = self._setup_linear_ops(
            DTYPE,
            DTYPE_real)

        self._pdop = optimizer.PDSoftSenseBaseSolver.factory(
            self._prg,
            self._queue,
            self.par,
            self.pdsose_par,
            self._fval_init,
            self._cmaps,
            linops=(self._MRI_operator, self._grad_op, self._symgrad_op),
            reg_type=self._reg_type,
            streamed=self._streamed,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real
        )

    def _setup_linear_ops(self, DTYPE, DTYPE_real):
        grad_op, _ = operator.Operator.GradientOperatorFactory(
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
                ([self.par["unknowns"], self.par["NSlice"],
                  self.par["dimY"], self.par["dimX"], 4]),
                dtype=DTYPE)
            if self._streamed:
                v = np.require(np.swapaxes(v, 0, 1), requirements='C')
        return grad_op, symgrad_op, v

    def _save_imgs(self, x, fname='', max_val=None, min_val=None):
        if fname:
            filename = fname
        elif "fname" in self.par.keys():
            filename = self.par["fname"]
        else:
            filename = self._reg_type + '_recon_R_{:.2f}'.format(self.par["R"]) + '_lambda_' + '{:.0e}'.format(self.ss_par["lambd"]) \
                       + '_noise_' + '{}'.format(self.ss_par["noise"]) + '_' + self.ss_par["mask_file"]
        path = self.par["outdir"] / 'imgs'
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        x = sqrt_sum_of_squares(x)[[int(np.shape(x)[-3]/2)], ...]
        # x1 = sqrt_sum_of_squares(x)[[3], ...]

        for i, img in enumerate(x):
            img_max = max_val if max_val else img.max()
            img_min = min_val if min_val else img.min()
            img_rescaled = (255.0 / img_max * (img - img_min)).astype(np.uint8)

            im = Image.fromarray(img_rescaled)
            file_dir = path / (filename + '_' + str(i) + '.png')
            im.save(file_dir)

    def _save_data(self, x):
        if "fname" in self.par.keys():
            filename = self.par["fname"]
        else:
            filename = 'Recon_R_{:.2f}'.format(self.par["R"]) + '_lambda_' + '{:.0e}'.format(self.ss_par["lambd"]) \
                       + '_noise_' + '{}'.format(self.ss_par["noise"]) + '_' + self.ss_par["mask_file"]
        path = self.par["outdir"] / 'data' / self._reg_type
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        file_dir = path / (filename + '.hdf5')

        with h5py.File(file_dir, 'w') as f:
            dset = f.create_dataset(self._reg_type+'_result', x.shape,
                                    dtype=self._DTYPE, data=x)
            dset.attrs["R"] = self.par["R"]
            dset.attrs["lambd"] = self.ss_par["lambd"]
            dset.attrs["streamed"] = self._streamed
            dset.attrs["elapsed_time"] = self._elapsed_time
            if self._reg_type == 'TGV':
                dset.attrs["alpha0"] = self.ss_par["alpha0"]
                dset.attrs["alpha1"] = self.ss_par["alpha1"]

    def _power_iterations(self, x, cmap, op, iters=10):
        x = np.require(x.astype(self._DTYPE), requirements='C')

        if len(cmap) > 0:
            cmap = cmap.astype(self._DTYPE)
            if self._streamed:
                y = op.adjoop([[op.fwdoop([[x, cmap]]), cmap]])
            else:
                x = clarray.to_device(self._queue[0], x)
                y = op.adjoop([op.fwdoop([x, cmap]), cmap]).get()
        else:
            if self._streamed:
                y = op.adjoop([op.fwdoop([x])])
            else:
                x = clarray.to_device(self._queue[0], x)
                y = op.adjoop(op.fwdoop(x)).get()

        l1 = []
        for i in range(iters):
            y_norm = np.linalg.norm(y)
            x = y / y_norm if y_norm != 0 else y
            if self._streamed:
                y = op.adjoop([[op.fwdoop([[x, cmap]]), cmap]]) if len(cmap) > 0 \
                    else op.adjoop([op.fwdoop([x])])
            else:
                x = clarray.to_device(self._queue[0], x)
                y = op.adjoop([op.fwdoop([x, cmap]), cmap]).get() if len(cmap) > 0 \
                    else op.adjoop(op.fwdoop(x)).get()

                if not isinstance(x, np.ndarray):
                    x = x.get()
                l1.append(np.vdot(y, x))

        return np.sqrt(np.max(np.abs(l1)))

    def _calc_step_size(self):
        if self._streamed:
            x = np.random.randn(self.par["NSlice"], 1, self.par["dimY"], self.par["dimX"]) + \
                1j * np.random.randn(self.par["NSlice"], 1, self.par["dimY"], self.par["dimX"])
        else:
            x = np.random.randn(1, self.par["NSlice"], self.par["dimY"], self.par["dimX"]) + \
                1j * np.random.randn(1, self.par["NSlice"], self.par["dimY"], self.par["dimX"])

        opnorm_fwd = self._power_iterations(x, self._cmaps, self._MRI_operator)

        K_ssense = np.array([opnorm_fwd, opnorm_fwd])

        if self._reg_type != 'NoReg':
            opnorm_grad = self._power_iterations(x, [], self._grad_op)

            K_ssense_tv = np.array([[opnorm_fwd, opnorm_fwd],
                                    [opnorm_grad, 0],
                                    [0, opnorm_grad]])

        if self._reg_type == 'TGV':
            if self._streamed:
                x_symgrad = np.random.randn(self.par["NSlice"], 1, self.par["dimY"], self.par["dimX"], 4) + \
                            1j * np.random.randn(self.par["NSlice"], 1, self.par["dimY"], self.par["dimX"], 4)
            else:
                x_symgrad = np.random.randn(1, self.par["NSlice"], self.par["dimY"], self.par["dimX"], 4) + \
                            1j * np.random.randn(1, self.par["NSlice"], self.par["dimY"], self.par["dimX"], 4)

            opnorm_symgrad = self._power_iterations(x_symgrad, [], self._symgrad_op)

            K_ssense_tgv = np.array([[opnorm_fwd, opnorm_fwd, 0, 0],
                                     [opnorm_grad, 0, -1, 0],
                                     [0, 0, opnorm_symgrad, 0],
                                     [0, opnorm_grad, 0, -1],
                                     [0, 0, 0, opnorm_symgrad]])

        tau = 1 / np.sqrt(np.vdot(K_ssense, K_ssense))

        if self._reg_type == 'TV':
            tau = 1 / np.sqrt(np.vdot(K_ssense_tv, K_ssense_tv))
        if self._reg_type == 'TGV':
            tau = 1 / np.sqrt(np.vdot(K_ssense_tgv, K_ssense_tgv))

        if tau < 0.1 or tau > 1.0:
            raise
        sigma = tau
        self._pdop.tau = tau
        self._pdop.sigma = sigma

        print("-" * 75)
        print("Calculated step size: %f" % tau)
        print("-" * 75)

    def _calculate_cost(self, x, data):
        if self._streamed:
            fwd = self._MRI_operator.fwdoop([[x, self._cmaps]])
            grad = self._grad_op.fwdoop([x])
            if self._v is not None:
                symgrad = self._symgrad_op.fwdoop([self._v])
        else:
            x = clarray.to_device(self._queue[0], x)
            fwd = self._MRI_operator.fwdoop([x, self._cmaps]).get()
            grad = self._grad_op.fwdoop(x).get()
            if self._v is not None:
                v = clarray.to_device(self._queue[0], self._v)
                symgrad = self._symgrad_op.fwdoop(v).get()

        data_cost = np.linalg.norm(fwd - data) ** 2
        # data_cost = np.vdot((out_fwd - ksp), (out_fwd - ksp)).real

        reg_cost = 0
        if self._reg_type == 'TV':
            data_cost *= (self.ss_par["lambd"] * 0.5)
            reg_cost = np.sum(np.abs(grad))
        if self._reg_type == 'TGV':
            data_cost *= self.ss_par["lambd"]
            reg_cost = self.ss_par['alpha1'] * np.sum(np.abs(grad - self._v)) \
                + self.ss_par['alpha0'] * np.sum(np.abs(symgrad))

        self._fval = data_cost + reg_cost
        self._fval_init = self._fval
        self._pdop.setFvalInit(self._fval)

        print("-" * 75)
        print("Initial Cost: %f" % self._fval_init)
        print("Costs of Data: %f" % data_cost)
        if self._reg_type != 'NoReg':
            print("Costs of T(G)V: %f" % reg_cost)
        print("-" * 75)

    def _solve(self, x, data, iters):
        tmpres, i = self._pdop.run(x, data, iters)
        for key in tmpres:
            if key == 'x':
                if isinstance(tmpres[key], np.ndarray):
                    x = tmpres["x"]
                else:
                    x = tmpres["x"].get()

        if self._streamed:
            x = np.require(np.swapaxes(x, 0, 1), requirements='C')
            x = np.require(np.swapaxes(x, -3, -1), requirements='C')

        self._i_term = i

        return x

    def execute(self, data):
        """Start the Soft Sense optimization.

        Parameters
        ----------
          data : numpy.array
            the data to perform optimization on.
        """
        x = np.require(
            np.zeros(self._unknown_shape, self._DTYPE)
            + 1j * np.zeros(self._unknown_shape, self._DTYPE),
            requirements='C')

        if self._streamed:
            data = np.require(
                np.transpose(data, self._data_trans_axes),
                requirements='C')

            img_init = self._MRI_operator.adjoop([[data, self._cmaps]])
            img_init_ = np.swapaxes(img_init, 0, 1)
            img_montage(sqrt_sum_of_squares(img_init_), 'Initial image(s) view dim YZ')
            if self.args.reco_slices == -1:
                img_init_ = np.swapaxes(img_init_, -3, -1)
                img_montage(sqrt_sum_of_squares(img_init_), 'Initial image(s) view dim YX')
            img_montage(sqrt_sum_of_squares(img_init_)[[int(np.shape(img_init_)[-3] / 2)], ...],
                        'Initial image view dim YX')

        else:
            data_ = clarray.to_device(self._queue[0], data)
            img_init = self._MRI_operator.adjoop([data_, self._cmaps]).get()
            img_montage(sqrt_sum_of_squares(img_init), 'Initial image(s)')

        iters = self.pdsose_par["max_iters"] if "max_iters" in self.pdsose_par.keys() else 1000

        self._calculate_cost(img_init, data)

        if not self.pdsose_par["adaptive_stepsize"]:
            try:
                self._calc_step_size()
            except Exception:
                print("Calculating step size with power iterations failed. Falling back to default.")
                self._pdop.tau = self._DTYPE_real(1 / np.sqrt(12))
                self._pdop.sigma = self._DTYPE_real(1 / np.sqrt(12))
        else:
            self._pdop.tau = self._DTYPE_real(0.9)
            self._pdop.sigma = self._DTYPE_real(0.9)

        start_time = time.time()
        result = self._solve(x, data, iters)
        self._elapsed_time = time.time() - start_time

        print("-" * 75)
        print("Elapsed time PD algorithm: {:.2f} seconds".format(self._elapsed_time))
        print("-" * 75)

        # self._save_data(result)
#        self._save_imgs(np.squeeze(sqrt_sum_of_squares(result)))

        return result



