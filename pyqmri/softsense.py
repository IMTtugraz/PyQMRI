#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the classes for Soft Sense Optimization."""
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

    def __init__(self, par,
                 reg_type='TGV',
                 config='',
                 streamed=False,
                 DTYPE=np.complex64,
                 DTYPE_real=np.float32):
        self.par = par
        self.ss_par = utils.read_config(config, optimizer="SSense", reg_type=reg_type)
        utils.save_config(self.ss_par, par["outdir"], reg_type)
        num_dev = len(par["num_dev"])
        self._fval_old = 0
        self._fval = 0
        self._fval_init = 0
        self._ctx = par["ctx"]
        self._queue = par["queue"]
        self._reg_type = reg_type
        self._prg = []

        self._DTYPE = DTYPE
        self._DTYPE_real = DTYPE_real

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
                            par["NSlice"], par["Nproj"], par["N"])
        if self._streamed:
            self._data_trans_axes = (2, 0, 1, 3, 4)
            self._coils = np.require(
                np.swapaxes(par["C"], 0, 1), requirements='C',
                dtype=DTYPE)

            self._data_shape = (par["NSlice"], par["NScan"], par["NC"],
                                par["Nproj"], par["N"])
        else:
            self._coils = clarray.to_device(self._queue[0],
                                            self.par["C"])

        self._MRI_operator, self._FT = operator.Operator.SoftSenseOperatorFactory(
            par,
            self._prg,
            DTYPE,
            DTYPE_real,
            streamed
            )

        self._grad_op, self._symgrad_op, self._v = self._setupLinearOps(
            DTYPE,
            DTYPE_real)

        if self.ss_par["linesearch"]:
            pass
        else:
            self._pdop = optimizer.PDSoftSenseSolver.factory(
                self._prg,
                self._queue,
                self.par,
                self.ss_par,
                self._fval_init,
                self._coils,
                linops=(self._MRI_operator, self._grad_op, self._symgrad_op),
                reg_type=self._reg_type,
                streamed=self._streamed,
                DTYPE=DTYPE,
                DTYPE_real=DTYPE_real
            )

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
                ([self.par["unknowns"], self.par["NSlice"],
                  self.par["dimY"], self.par["dimX"], 4]),
                dtype=DTYPE)
            if self._streamed:
                v = np.require(np.swapaxes(v, 0, 1), requirements='C')
        return grad_op, symgrad_op, v

    def execute(self, data):
        """Start the Soft Sense optimization.

        Parameters
        ----------
          data : numpy.array
            the data to perform optimization on.
        """
        pass




