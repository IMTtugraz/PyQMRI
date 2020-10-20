#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the classes for different linear Operators."""
from abc import ABC, abstractmethod
import pyopencl.array as clarray
import numpy as np
from pyqmri.transforms import PyOpenCLnuFFT as CLnuFFT
import pyqmri.streaming as streaming


class Operator(ABC):
    """Abstract base class for linear Operators used in the optimization.

    This class serves as the base class for all linear operators used in
    the varous optimization algorithms. it requires to implement a forward
    and backward application in and out of place.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator.
      DTYPE : numpy.dtype, numpy.complex64
         Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
          Real working precission.

    Attributes
    ----------
      NScan : int
        Number of total measurements (Scans)
      NC : int
        Number of complex coils
      NSlice : int
        Number ofSlices
      dimX : int
        X dimension of the parameter maps
      dimY : int
        Y dimension of the parameter maps
      N : int
        N number of samples per readout
      Nproj : int
        Number of rreadouts
      unknowns_TGV : int
        Number of unknowns which should be regularized with TGV. It is assumed
        that these occure first in the unknown vector. Currently at least 1
        TGV unknown is required.
      unknowns_H1 : int
        Number of unknowns which should be regularized with H1. It is assumed
        that these occure after all TGV unknowns in the unknown vector.
        Currently this number can be zero which implies that no H1
        regularization is used.
      unknowns : int
        The sum of TGV and H1 unknowns.
      ctx : list of PyOpenCL.Context
        The context for the PyOpenCL computations. If streamed operations are
        used a list of ctx is required. One for each computation device.
      queue : list of PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels. If streamed operations
        are used a list of queues is required. Four for each computation
        device.
      dz : float
        The ratio between the physical X,Y dimensions vs the Z dimension.
        This allows for anisotrpic regularization along the Z dimension.
      num_dev : int
        Number of compute devices
      NUFFT : PyQMRI.transforms.PyOpenCLnuFFT
        A PyOpenCLnuFFT object to perform forward and backword transformations
        from image to k-space and vice versa.
      prg : PyOpenCL.Program
        The PyOpenCL program containing all compiled kernels.
      self.DTYPE : numpy.dtype
        Complex working precission. Currently single precission only.
      self.DTYPE_real : numpy.dtype
        Real working precission. Currently single precission only.
    """

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        self.NSlice = par["NSlice"]
        self.NScan = par["NScan"]
        self.dimX = par["dimX"]
        self.dimY = par["dimY"]
        self.N = par["N"]
        self.NC = par["NC"]
        self.Nproj = par["Nproj"]
        self.ctx = par["ctx"]
        self.queue = par["queue"]
        self.unknowns_TGV = par["unknowns_TGV"]
        self.unknowns_H1 = par["unknowns_H1"]
        self.unknowns = par["unknowns"]
        self._dz = par["dz"]
        self.num_dev = len(par["num_dev"])
        self._tmp_result = []
        self.NUFFT = []
        self.prg = prg
        self.DTYPE = DTYPE
        self.DTYPE_real = DTYPE_real
        self.par_slices = self.NSlice
        self._overlap = 0

    @abstractmethod
    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex measurement space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        ...

    @abstractmethod
    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex measurement space data which is used as input.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        ...

    @abstractmethod
    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        ...

    @abstractmethod
    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex measurement space which is used as input.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        ...

    @staticmethod
    def MRIOperatorFactory(par,
                           prg,
                           DTYPE,
                           DTYPE_real,
                           trafo=False,
                           imagespace=False,
                           SMS=False,
                           streamed=False):
        """MRI forward/adjoint operator factory method.

        Parameters
        ----------
          par : dict A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          prg : PyOpenCL.Program
            The PyOpenCL.Program object containing the necessary kernels to
            execute the linear Operator.
          DTYPE : numpy.dtype, numpy.complex64
             Complex working precission.
          DTYPE_real : numpy.dtype, numpy.float32
            Real working precission.
          trafo : bool, false
            Select between radial (True) or cartesian FFT (false).
          imagespace : bool, false
            Select between fitting in imagespace (True) or k-space (false).
          SMS : bool, false
            Select between simulatneous multi-slice reconstruction or standard.
          streamed : bool, false
            Use standard reconstruction (false) or streaming of memory blocks
            to the compute device (true). Only use this if data does not
            fit in one block.

        Returns
        -------
          PyQMRI.Operator
            A specialized instance of a PyQMRI.Operator to perform forward
            and ajoint operations for fitting.
          PyQMRI.NUFFT
            An instance of the used (nu-)FFT if k-space fitting is performed,
            None otherwise.
        """
        if streamed:
            if imagespace:
                op = OperatorImagespaceStreamed(
                    par, prg,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
                FT = None
            else:
                if SMS:
                    op = OperatorKspaceSMSStreamed(
                        par,
                        prg,
                        DTYPE=DTYPE,
                        DTYPE_real=DTYPE_real)
                else:
                    op = OperatorKspaceStreamed(
                        par,
                        prg,
                        trafo=trafo,
                        DTYPE=DTYPE,
                        DTYPE_real=DTYPE_real)
                FT = op.NUFFT
        else:
            if imagespace:
                op = OperatorImagespace(
                    par, prg[0],
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
                FT = None
            else:
                if SMS:
                    op = OperatorKspaceSMS(
                        par,
                        prg[0],
                        DTYPE=DTYPE,
                        DTYPE_real=DTYPE_real)
                else:
                    op = OperatorKspace(
                        par,
                        prg[0],
                        trafo=trafo,
                        DTYPE=DTYPE,
                        DTYPE_real=DTYPE_real)
                FT = op.NUFFT
        return op, FT

    @staticmethod
    def GradientOperatorFactory(par,
                                prg,
                                DTYPE,
                                DTYPE_real,
                                streamed=False):
        """Gradient forward/adjoint operator factory method.

        Parameters
        ----------
          par : dict A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          prg : PyOpenCL.Program
            The PyOpenCL.Program object containing the necessary kernels to
            execute the linear Operator.
          DTYPE : numpy.dtype, numpy.complex64
             Complex working precission.
          DTYPE_real : numpy.dtype, numpy.float32
            Real working precission.
          streamed : bool, false
            Use standard reconstruction (false) or streaming of memory blocks
            to the compute device (true). Only use this if data does not
            fit in one block.

        Returns
        -------
          PyQMRI.Operator
            A specialized instance of a PyQMRI.Operator to perform forward
            and ajoint gradient calculations.
        """
        if streamed:
            op = OperatorFiniteGradientStreamed(par,
                                                prg,
                                                DTYPE,
                                                DTYPE_real)
        else:
            op = OperatorFiniteGradient(par,
                                        prg[0],
                                        DTYPE,
                                        DTYPE_real)
        return op

    @staticmethod
    def SymGradientOperatorFactory(par,
                                   prg,
                                   DTYPE,
                                   DTYPE_real,
                                   streamed=False):
        """Symmetrized Gradient forward/adjoint operator factory method.

        Parameters
        ----------
          par : dict A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          prg : PyOpenCL.Program
            The PyOpenCL.Program object containing the necessary kernels to
            execute the linear Operator.
          DTYPE : numpy.dtype, numpy.complex64
             Complex working precission.
          DTYPE_real : numpy.dtype, numpy.float32
            Real working precission.
          streamed : bool, false
            Use standard reconstruction (false) or streaming of memory blocks
            to the compute device (true). Only use this if data does not
            fit in one block.

        Returns
        -------
          PyQMRI.Operator
            A specialized instance of a PyQMRI.Operator to perform forward
            and ajoint symmetriced gradient calculations.
        """
        if streamed:
            op = OperatorFiniteSymGradientStreamed(par,
                                                   prg,
                                                   DTYPE,
                                                   DTYPE_real)
        else:
            op = OperatorFiniteSymGradient(par,
                                           prg[0],
                                           DTYPE,
                                           DTYPE_real)
        return op

    def _defineoperator(self,
                        functions,
                        outp,
                        inp,
                        reverse_dir=False,
                        posofnorm=None,
                        slices=None):
        if slices is None:
            slices = self.NSlice
        return streaming.Stream(
            functions,
            outp,
            inp,
            self.par_slices,
            self._overlap,
            slices,
            self.queue,
            self.num_dev,
            reverse_dir,
            posofnorm,
            DTYPE=self.DTYPE)


class OperatorImagespace(Operator):
    """Imagespace based Operator.

    This class serves as linear operator between parameter and imagespace.

    Use this operator if you want to perform complex parameter fitting from
    complex image space data without the need of performing FFTs.

    Parameters
    ----------
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator.
      DTYPE : numpy.dtype, numpy.complex64
         Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
          Real working precission.

    Attributes
    ----------
    ctx : PyOpenCL.Context
      The context for the PyOpenCL computations.
    queue : PyOpenCL.Queue
      The computation Queue for the PyOpenCL kernels.
    """

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex measurement space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        return self.prg.operator_fwd_imagespace(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, inp[0].data, inp[2].data,
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=inp[0].events + out.events + wait_for)

    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        tmp_result = clarray.empty(
            self.queue, (self.NScan, self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")
        tmp_result.add_event(self.prg.operator_fwd_imagespace(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            tmp_result.data, inp[0].data, inp[2].data,
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=inp[0].events + wait_for))
        return tmp_result

    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex measurement space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        return self.prg.operator_ad_imagespace(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, inp[0].data, inp[2].data,
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + inp[0].events + out.events)

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex measurement space which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        out = clarray.empty(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            dtype=self.DTYPE)
        self.prg.operator_ad_imagespace(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, inp[0].data, inp[2].data,
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + inp[0].events + out.events).wait()
        return out

    def adjKyk1(self, out, inp, **kwargs):
        """Apply the linear operator from image space to parameter space.

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex image space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        return self.prg.update_Kyk1_imagespace(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, inp[0].data, inp[3].data, inp[1].data,
            np.int32(self.NScan),
            inp[4].data,
            np.int32(self.unknowns),
            self.DTYPE_real(self._dz),
            wait_for=(inp[0].events + out.events
                      + inp[1].events + wait_for))


class OperatorKspace(Operator):
    """k-Space based Operator.

    This class serves as linear operator between parameter and k-space.

    Use this operator if you want to perform complex parameter fitting from
    complex k-space data. The type of fft is defined through the NUFFT object.
    The NUFFT object can also be used for simple Cartesian FFTs.

    Parameters
    ----------
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.
      trafo : bool, true
        Switch between cartesian (false) and non-cartesian FFT (True, default).

    Attributes
    ----------
    ctx : PyOpenCL.Context
      The context for the PyOpenCL computations.
    queue : PyOpenCL.Queue
      The computation Queue for the PyOpenCL kernels.
    NUFFT : PyQMRI.PyOpenCLnuFFT
      The (nu) FFT used for fitting.
    """

    def __init__(self, par, prg, DTYPE=np.complex64,
                 DTYPE_real=np.float32, trafo=True):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self._tmp_result = clarray.empty(
            self.queue, (self.NScan, self.NC,
                         self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")
        if not trafo:
            self.Nproj = self.dimY
            self.N = self.dimX
        self.NUFFT = CLnuFFT.create(self.ctx,
                                    self.queue,
                                    par,
                                    radial=trafo,
                                    DTYPE=DTYPE,
                                    DTYPE_real=DTYPE_real)

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex measurement space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        self._tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self._tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=(self._tmp_result.events + inp[0].events
                          + wait_for)))
        return self.NUFFT.FFT(
            out,
            self._tmp_result,
            wait_for=wait_for +
            self._tmp_result.events)

    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        self._tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self._tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=(self._tmp_result.events + inp[0].events
                          + wait_for)))
        tmp_sino = clarray.empty(
            self.queue,
            (self.NScan, self.NC, self.NSlice, self.Nproj, self.N),
            self.DTYPE, "C")
        tmp_sino.add_event(
            self.NUFFT.FFT(tmp_sino, self._tmp_result))
        return tmp_sino

    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex measurement space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        self._tmp_result.add_event(
            self.NUFFT.FFTH(
                self._tmp_result, inp[0], wait_for=(wait_for
                                                    + inp[0].events)))
        return self.prg.operator_ad(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self._tmp_result.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=self._tmp_result.events + out.events)

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex measurement space which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        self._tmp_result.add_event(
            self.NUFFT.FFTH(
                self._tmp_result, inp[0], wait_for=(wait_for
                                                    + inp[0].events)))
        out = clarray.empty(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            dtype=self.DTYPE)
        self.prg.operator_ad(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self._tmp_result.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=(self._tmp_result.events
                      + out.events)).wait()
        return out

    def adjKyk1(self, out, inp, **kwargs):
        """Apply the linear operator from parameter space to k-space.

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is used as input.
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        self._tmp_result.add_event(
            self.NUFFT.FFTH(
                self._tmp_result, inp[0], wait_for=(wait_for
                                                    + inp[0].events)))
        return self.prg.update_Kyk1(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self._tmp_result.data, inp[2].data,
            inp[3].data, inp[1].data, np.int32(self.NC),
            np.int32(self.NScan),
            inp[4].data,
            np.int32(self.unknowns), self.DTYPE_real(self._dz),
            wait_for=(self._tmp_result.events +
                      out.events + inp[1].events))


class OperatorKspaceSMS(Operator):
    """k-Space based Operator for SMS reconstruction.

    This class serves as linear operator between parameter and k-space.
    It implements simultaneous-multi-slice (SMS) reconstruction.

    Use this operator if you want to perform complex parameter fitting from
    complex k-space data measured with SMS. Currently only Cartesian FFTs are
    supported.

    Parameters
    ----------
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.

    Attributes
    ----------
      packs : int
        Number of SMS packs.
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      NUFFT : PyQMRI.PyOpenCLnuFFT
        The (nu) FFT used for fitting.
    """

    def __init__(self, par, prg, DTYPE=np.complex64,
                 DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.packs = par["packs"]*par["numofpacks"]
        self._tmp_result = clarray.empty(
            self.queue, (self.NScan, self.NC,
                         self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")

        self.Nproj = self.dimY
        self.N = self.dimX
        self.NUFFT = CLnuFFT.create(self.ctx,
                                    self.queue,
                                    par,
                                    radial=False,
                                    SMS=True,
                                    DTYPE=DTYPE,
                                    DTYPE_real=DTYPE_real)

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex measurement space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        self._tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self._tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=(self._tmp_result.events + inp[0].events
                          + wait_for)))
        return self.NUFFT.FFT(
            out,
            self._tmp_result,
            wait_for=self._tmp_result.events + out.events)

    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        self._tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self._tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=(self._tmp_result.events + inp[0].events
                          + wait_for)))
        tmp_sino = clarray.empty(
            self.queue,
            (self.NScan, self.NC, self.packs, self.Nproj, self.N),
            self.DTYPE, "C")
        tmp_sino.add_event(
            self.NUFFT.FFT(tmp_sino, self._tmp_result))
        return tmp_sino

    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex measurement space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        self._tmp_result.add_event(
            self.NUFFT.FFTH(
                self._tmp_result, inp[0], wait_for=(wait_for
                                                    + inp[0].events)))
        return self.prg.operator_ad(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self._tmp_result.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=self._tmp_result.events + out.events)

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex measurement space which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        self._tmp_result.add_event(
            self.NUFFT.FFTH(
                self._tmp_result, inp[0], wait_for=(wait_for
                                                    + inp[0].events)))
        out = clarray.empty(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            dtype=self.DTYPE)
        self.prg.operator_ad(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self._tmp_result.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=self._tmp_result.events + out.events).wait()
        return out

    def adjKyk1(self, out, inp, **kwargs):
        """Apply the linear operator from parameter space to k-space.

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is used as input.
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        self._tmp_result.add_event(
            self.NUFFT.FFTH(
                self._tmp_result, inp[0], wait_for=(wait_for
                                                    + inp[0].events)))
        return self.prg.update_Kyk1(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self._tmp_result.data, inp[2].data,
            inp[3].data, inp[1].data, np.int32(self.NC),
            np.int32(self.NScan),
            inp[4].data,
            np.int32(self.unknowns), self.DTYPE_real(self._dz),
            wait_for=(self._tmp_result.events +
                      out.events + inp[1].events))


class OperatorImagespaceStreamed(Operator):
    """The streamed version of the Imagespace based Operator.

    This class serves as linear operator between parameter and imagespace.
    All calculations are performed in a streamed fashion.

    Use this operator if you want to perform complex parameter fitting from
    complex image space data without the need of performing FFTs.
    In contrast to non-streaming classes no out of place operations
    are implemented.

    Parameters
    ----------
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.

    Attributes
    ----------
      overlap : int
        Number of slices that overlap between adjacent blocks.
      par_slices : int
        Number of slices per streamed block
      fwdstr : PyQMRI.Stream
        The streaming object to perform the forward evaluation
      adjstr : PyQMRI.Stream
        The streaming object to perform the adjoint evaluation
      adjstrKyk1 : PyQMRI.Stream
        The streaming object to perform the adjoint evaluation including z1
        of the algorithm.
      unknown_shape : tuple of int
        Size of the parameter maps
      data_shape : tuple of int
        Size of the data
    """

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        par["overlap"] = 1
        self._overlap = par["overlap"]
        self.par_slices = par["par_slices"]
        self.unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        coil_shape = []
        model_grad_shape = (self.NSlice, self.unknowns,
                            self.NScan, self.dimY, self.dimX)
        self.data_shape = (self.NSlice, self.NScan, self.dimY, self.dimX)
        grad_shape = self.unknown_shape + (4,)

        self.fwdstr = self._defineoperator(
            [self._fwdstreamed],
            [self.data_shape],
            [[self.unknown_shape,
              coil_shape,
              model_grad_shape]])

        self.adjstrKyk1 = self._defineoperator(
            [self._adjstreamedKyk1],
            [self.unknown_shape],
            [[self.data_shape,
              grad_shape,
              coil_shape,
              model_grad_shape,
              self.unknown_shape]])

        self.adjstr = self._defineoperator(
            [self._adjstreamed],
            [self.unknown_shape],
            [[self.data_shape,
              coil_shape,
              model_grad_shape]])

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex measurement space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        self.fwdstr.eval(out, inp)

    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        tmp_result = np.zeros(self.data_shape, dtype=self.DTYPE)
        self.fwdstr.eval([tmp_result], inp)
        return tmp_result

    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex measurement space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        self.adjstr.eval(out, inp)

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex measurement space which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        tmp_result = np.zeros(self.unknown_shape, dtype=self.DTYPE)
        self.adjstr.eval([tmp_result], inp)
        return tmp_result

    def adjKyk1(self, out, inp):
        """Apply the linear operator from parameter space to image space.

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Parameters
        ----------
          out : numpy.Array
            The complex parameter space data which is used as input.
          inp : numpy.Array
            The complex parameter space data which is used as input.
        """
        self.adjstrKyk1.eval(out, inp)

    def _fwdstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return (self.prg[idx].operator_fwd_imagespace(
            self.queue[4*idx+idxq],
            (self.par_slices+self._overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[2].data,
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=outp.events+inp[0].events+inp[2].events+wait_for))

    def _adjstreamedKyk1(self, outp, inp, par=None, idx=0, idxq=0,
                         bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.prg[idx].update_Kyk1_imagespace(
            self.queue[4*idx+idxq],
            (self.par_slices+self._overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[3].data,
            inp[1].data,
            np.int32(self.NScan),
            par[0][idx].data, np.int32(self.unknowns),
            np.int32(bound_cond), self.DTYPE_real(self._dz),
            wait_for=(outp.events+inp[0].events+inp[1].events +
                      inp[3].events+wait_for))

    def _adjstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.prg[idx].operator_ad_imagespace(
            self.queue[4*idx+idxq],
            (self.par_slices+self._overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[2].data,
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=(outp.events+inp[0].events +
                      inp[2].events+wait_for))


class OperatorKspaceStreamed(Operator):
    """The streamed version of the k-space based Operator.

    This class serves as linear operator between parameter and k-space.
    All calculations are performed in a streamed fashion.

    Use this operator if you want to perform complex parameter fitting from
    complex k-space data without the need of performing FFTs.
    In contrast to non-streaming classes no out of place operations
    are implemented.

    Parameters
    ----------
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.
      trafo : bool, true
        Switch between cartesian (false) and non-cartesian FFT (True, default).

    Attributes
    ----------
      overlap : int
        Number of slices that overlap between adjacent blocks.
      par_slices : int
        Number of slices per streamed block.
      fwdstr : PyQMRI.Stream
        The streaming object to perform the forward evaluation.
      adjstr : PyQMRI.Stream
        The streaming object to perform the adjoint evaluation.
      adjstrKyk1 : PyQMRI.Stream
        The streaming object to perform the adjoint evaluation including z1
        of the algorithm.
      NUFFT : list of PyQMRI.transforms.PyOpenCLnuFFT
        A list of NUFFT objects. One for each context.
      FTstr : PyQMRI.Stream
        A streamed version of the used (non-uniform) FFT, applied forward.
      unknown_shape : tuple of int
        Size of the parameter maps
      data_shape : tuple of int
        Size of the data
    """

    def __init__(self, par, prg,
                 DTYPE=np.complex64, DTYPE_real=np.float32, trafo=True):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self._overlap = par["overlap"]
        self.par_slices = par["par_slices"]
        if not trafo:
            self.Nproj = self.dimY
            self.N = self.dimX
        for j in range(self.num_dev):
            for i in range(2):
                self._tmp_result.append(
                    clarray.empty(
                        self.queue[4*j+i],
                        (self.par_slices+self._overlap, self.NScan,
                         self.NC, self.dimY, self.dimX),
                        self.DTYPE, "C"))
                self.NUFFT.append(
                    CLnuFFT.create(self.ctx[j],
                                   self.queue[4*j+i], par,
                                   radial=trafo,
                                   streamed=True,
                                   DTYPE=DTYPE,
                                   DTYPE_real=DTYPE_real))

        self.unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        coil_shape = (self.NSlice, self.NC, self.dimY, self.dimX)
        model_grad_shape = (self.NSlice, self.unknowns,
                            self.NScan, self.dimY, self.dimX)
        self.data_shape = (self.NSlice, self.NScan, self.NC, self.Nproj,
                           self.N)
        trans_shape = (self.NSlice, self.NScan,
                       self.NC, self.dimY, self.dimX)
        grad_shape = self.unknown_shape + (4,)

        self.fwdstr = self._defineoperator(
            [self._fwdstreamed],
            [self.data_shape],
            [[self.unknown_shape,
              coil_shape,
              model_grad_shape]])

        self.adjstrKyk1 = self._defineoperator(
            [self._adjstreamedKyk1],
            [self.unknown_shape],
            [[self.data_shape,
              grad_shape,
              coil_shape,
              model_grad_shape,
              self.unknown_shape]])

        self.adjstr = self._defineoperator(
            [self._adjstreamed],
            [self.unknown_shape],
            [[self.data_shape,
              coil_shape,
              model_grad_shape]])

        self.FTstr = self._defineoperator(
            [self._FT],
            [self.data_shape],
            [[trans_shape]])

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex measurement space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        self.fwdstr.eval(out, inp)

    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        tmp_result = np.zeros(self.data_shape, dtype=self.DTYPE)
        self.fwdstr.eval([tmp_result], inp)
        return tmp_result

    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex measurement space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        self.adjstr.eval(out, inp)

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex measurement space which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        tmp_result = np.zeros(self.unknown_shape, dtype=self.DTYPE)
        self.adjstr.eval([tmp_result], inp)
        return tmp_result

    def adjKyk1(self, out, inp):
        """Apply the linear operator from parameter space to k-space.

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Parameters
        ----------
          out : numpy.Array
            The complex parameter space data which is used as input.
          inp : numpy.Array
            The complex parameter space data which is used as input.
        """
        self.adjstrKyk1.eval(out, inp)

    def _fwdstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        self._tmp_result[2*idx+idxq].add_event(self.prg[idx].operator_fwd(
            self.queue[4*idx+idxq],
            (self.par_slices+self._overlap, self.dimY, self.dimX), None,
            self._tmp_result[2*idx+idxq].data, inp[0].data,
            inp[1].data,
            inp[2].data,
            np.int32(self.NC),
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=(self._tmp_result[2*idx+idxq].events +
                      inp[0].events+wait_for)))
        return self.NUFFT[2*idx+idxq].FFT(
            outp, self._tmp_result[2*idx+idxq],
            wait_for=outp.events+wait_for+self._tmp_result[2*idx+idxq].events)

    def _adjstreamedKyk1(self, outp, inp, par=None, idx=0, idxq=0,
                         bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        self._tmp_result[2*idx+idxq].add_event(
            self.NUFFT[2*idx+idxq].FFTH(
                self._tmp_result[2*idx+idxq], inp[0],
                wait_for=(wait_for+inp[0].events +
                          self._tmp_result[2*idx+idxq].events)))
        return self.prg[idx].update_Kyk1(
            self.queue[4*idx+idxq],
            (self.par_slices+self._overlap, self.dimY, self.dimX), None,
            outp.data, self._tmp_result[2*idx+idxq].data,
            inp[2].data,
            inp[3].data,
            inp[1].data, np.int32(self.NC), np.int32(self.NScan),
            par[0][idx].data, np.int32(self.unknowns),
            np.int32(bound_cond), self.DTYPE_real(self._dz),
            wait_for=(
                self._tmp_result[2*idx+idxq].events +
                outp.events+inp[1].events +
                inp[2].events + inp[3].events + wait_for))

    def _adjstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        self._tmp_result[2*idx+idxq].add_event(
            self.NUFFT[2*idx+idxq].FFTH(
                self._tmp_result[2*idx+idxq], inp[0],
                wait_for=(wait_for+inp[0].events +
                          self._tmp_result[2*idx+idxq].events)))
        return self.prg[idx].operator_ad(
            self.queue[4*idx+idxq],
            (self.par_slices+self._overlap, self.dimY, self.dimX), None,
            outp.data, self._tmp_result[2*idx+idxq].data,
            inp[1].data,
            inp[2].data,
            np.int32(self.NC),
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=(self._tmp_result[2*idx+idxq].events +
                      inp[1].events+inp[2].events+wait_for))

    def _FT(self, outp, inp, par=None, idx=0, idxq=0,
            bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.NUFFT[2*idx+idxq].FFT(outp, inp[0])


class OperatorKspaceSMSStreamed(Operator):
    """The streamed version of the k-space based SMS Operator.

    This class serves as linear operator between parameter and k-space.
    It implements simultaneous-multi-slice (SMS) reconstruction.

    All calculations are performed in a streamed fashion.

    Use this operator if you want to perform complex parameter fitting from
    complex k-space data measured with SMS. Currently only Cartesian FFTs are
    supported.

    Parameters
    ----------
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.


    Attributes
    ----------
      overlap : int
        Number of slices that overlap between adjacent blocks.
      par_slices : int
        Number of slices per streamed block
      packs : int
        Number of packs to stream
      fwdstr : PyQMRI.Stream
        The streaming object to perform the forward evaluation
      adjstr : PyQMRI.Stream
        The streaming object to perform the adjoint evaluation
      NUFFT : list of PyQMRI.transforms.PyOpenCLnuFFT
        A list of NUFFT objects. One for each context.
      FTstr : PyQMRI.Stream
        A streamed version of the used (non-uniform) FFT, applied forward.
      FTHstr : PyQMRI.Stream
        A streamed version of the used (non-uniform) FFT, applied adjoint.
      updateKyk1SMSstreamed
      dat_trans_axes : list of int
        Order in which the data needs to be transformed during the SMS
        reconstruction and streaming.
    """

    def __init__(self, par, prg, DTYPE=np.complex64,
                 DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self._overlap = par["overlap"]
        self.par_slices = par["par_slices"]
        self.packs = par["packs"]*par["numofpacks"]

        self.Nproj = self.dimY
        self.N = self.dimX

        for j in range(self.num_dev):
            for i in range(2):
                self._tmp_result.append(
                    clarray.empty(
                        self.queue[4*j+i],
                        (self.par_slices+self._overlap, self.NScan,
                         self.NC, self.dimY, self.dimX),
                        self.DTYPE, "C"))
                self.NUFFT.append(
                    CLnuFFT.create(self.ctx[j],
                                   self.queue[4*j+i], par,
                                   radial=False,
                                   SMS=True,
                                   streamed=True,
                                   DTYPE=DTYPE,
                                   DTYPE_real=DTYPE_real))

        unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        coil_shape = (self.NSlice, self.NC, self.dimY, self.dimX)
        model_grad_shape = (self.NSlice, self.unknowns,
                            self.NScan, self.dimY, self.dimX)
        data_shape = (self.NSlice, self.NScan, self.NC, self.dimY, self.dimX)
        data_shape_T = (self.NScan, self.NC, self.packs,
                        self.dimY, self.dimX)
        trans_shape_T = (self.NScan,
                         self.NC, self.NSlice, self.dimY, self.dimX)
        grad_shape = unknown_shape + (4,)
        self.dat_trans_axes = [2, 0, 1, 3, 4]

        self.fwdstr = self._defineoperator(
            [self._fwdstreamed],
            [data_shape],
            [[unknown_shape,
              coil_shape,
              model_grad_shape]])

        self.adjstr = self._defineoperator(
            [self._adjstreamed],
            [unknown_shape],
            [[data_shape,
              coil_shape,
              model_grad_shape]])

        self.FTstr = self._defineoperatorSMS(
            [self._FT],
            [data_shape_T],
            [[trans_shape_T]])
        self.FTHstr = self._defineoperatorSMS(
            [self._FTH],
            [trans_shape_T],
            [[data_shape_T]])

        self._tmp_fft1 = np.zeros((self.NSlice, self.NScan, self.NC,
                                   self.dimY, self.dimX),
                                  dtype=self.DTYPE)
        self._tmp_fft2 = np.zeros((self.NScan, self.NC, self.NSlice,
                                   self.dimY, self.dimX),
                                  dtype=self.DTYPE)
        self._tmp_transformed = np.zeros((self.NScan, self.NC, self.packs,
                                          self.dimY, self.dimX),
                                         dtype=self.DTYPE)
        self._tmp_Kyk1 = np.zeros(unknown_shape,
                                  dtype=self.DTYPE)

        self._updateKyk1SMSStreamed = self._defineoperator(
            [self._updateKyk1SMS],
            [unknown_shape],
            [[unknown_shape,
              grad_shape,
              unknown_shape]],
            reverse_dir=True,
            posofnorm=[True])

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex measurement space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        self.fwdstr.eval([self._tmp_fft1], inp)
        self._tmp_fft2 = np.require(
            np.transpose(
                self._tmp_fft1, (1, 2, 0, 3, 4)),
            requirements='C')
        self.FTstr.eval(
            [self._tmp_transformed],
            [[self._tmp_fft2]])
        out[0][...] = np.copy(np.require(
            np.transpose(
                self._tmp_transformed,
                self.dat_trans_axes),
            requirements='C'))

    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        self.fwdstr.eval([self._tmp_fft1], inp)
        self._tmp_fft2 = np.require(
            np.transpose(
                self._tmp_fft1, (1, 2, 0, 3, 4)),
            requirements='C')
        self.FTstr.eval(
            [self._tmp_transformed],
            [[self._tmp_fft2]])
        return np.require(
            np.transpose(
                self._tmp_transformed,
                self.dat_trans_axes),
            requirements='C')

    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        Parameters
        ----------
          out : numpy.Array
            The complex parameter space data which is used as input.
          inp : numpy.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          tupel of floats:
            The lhs and rhs for the line search of the primal-dual algorithm.
        """
        self._tmp_transformed = np.require(
            np.transpose(
                inp[0][0], (1, 2, 0, 3, 4)),
            requirements='C')
        self.FTHstr.eval(
            [self._tmp_fft2],
            [[self._tmp_transformed]])
        self._tmp_fft1 = np.require(
            np.transpose(
                self._tmp_fft2, self.dat_trans_axes),
            requirements='C')
        self.adjstr.eval(out, [[self._tmp_fft1]+inp[0][1:]])

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex measurement space which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        self._tmp_transformed = np.require(
            np.transpose(
                inp[0][0], (1, 2, 0, 3, 4)),
            requirements='C')
        self.FTHstr.eval(
            [self._tmp_fft2],
            [[self._tmp_transformed]])
        self._tmp_fft1 = np.require(
            np.transpose(
                self._tmp_fft2, self.dat_trans_axes),
            requirements='C')
        self.adjstr.eval([self._tmp_Kyk1], [[self._tmp_fft1]+inp[0][1:]])
        return self._tmp_Kyk1

    def adjKyk1(self, out, inp, **kwargs):
        """Apply the linear operator from parameter space to k-space.

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Parameters
        ----------
          out : numpy.Array
            The complex parameter space data which is used as input.
          inp : numpy.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          tupel of floats:
            The lhs and rhs for the line search of the primal-dual algorithm.
        """
        self._tmp_transformed = np.require(
            np.transpose(
                inp[0][0], (1, 2, 0, 3, 4)),
            requirements='C')
        self.FTHstr.eval(
            [self._tmp_fft2],
            [[self._tmp_transformed]])
        self._tmp_fft1 = np.require(
            np.transpose(
                self._tmp_fft2, self.dat_trans_axes),
            requirements='C')
        self.adjstr.eval([self._tmp_Kyk1], [[self._tmp_fft1]+inp[0][2:-1]])
        return self._updateKyk1SMSStreamed.evalwithnorm(
            out,
            [[self._tmp_Kyk1]+[inp[0][1]]+[inp[0][-1]]], kwargs["par"])

    def _fwdstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.prg[idx].operator_fwd(
            self.queue[4*idx+idxq],
            (self.par_slices+self._overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[1].data,
            inp[2].data,
            np.int32(self.NC),
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def _adjstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.prg[idx].operator_ad(
            self.queue[4*idx+idxq],
            (self.par_slices+self._overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[1].data,
            inp[2].data,
            np.int32(self.NC), np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=(
                inp[0].events +
                outp.events+inp[1].events +
                inp[2].events + wait_for))

    def _FT(self, outp, inp, par=None, idx=0, idxq=0,
            bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.NUFFT[2*idx+idxq].FFT(outp, inp[0])

    def _FTH(self, outp, inp, par=None, idx=0, idxq=0,
             bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.NUFFT[2*idx+idxq].FFTH(outp, inp[0])

    def _updateKyk1SMS(self, outp, inp, par=None, idx=0, idxq=0,
                       bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.prg[idx].update_Kyk1SMS(
            self.queue[4*idx+idxq],
            (self.par_slices+self._overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[1].data,
            par[0][idx].data, np.int32(self.unknowns),
            np.int32(bound_cond), self.DTYPE_real(self._dz),
            wait_for=(
                inp[0].events +
                outp.events+inp[1].events +
                wait_for))

    def _defineoperatorSMS(self,
                           functions,
                           outp,
                           inp,
                           reverse_dir=False,
                           posofnorm=None):
        return streaming.Stream(
            functions,
            outp,
            inp,
            1,
            0,
            self.NScan,
            self.queue,
            self.num_dev,
            reverse_dir,
            posofnorm,
            DTYPE=self.DTYPE)


class OperatorFiniteGradient(Operator):
    """Gradient operator.

    This class implements the finite difference gradient operation and
    the adjoint (negative divergence).

    Parameters
    ----------
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.

    Attributes
    ----------
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      ratio : list of PyOpenCL.Array
        Ratio between the different unknowns
    """

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.ratio = clarray.to_device(
            self.queue,
            (par["weights"]).astype(
                     dtype=self.DTYPE_real))
        self._weights = par["weights"]

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        return self.prg.gradient(
            self.queue, inp.shape[1:], None, out.data, inp.data,
            np.int32(self.unknowns),
            self.ratio.data, self.DTYPE_real(self._dz),
            wait_for=out.events + inp.events + wait_for)

    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        tmp_result = clarray.empty(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 4),
            self.DTYPE, "C")
        tmp_result.add_event(self.prg.gradient(
            self.queue, inp.shape[1:], None, tmp_result.data, inp.data,
            np.int32(self.unknowns),
            self.ratio.data, self.DTYPE_real(self._dz),
            wait_for=tmp_result.events + inp.events + wait_for))
        return tmp_result

    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex measurement space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        return self.prg.divergence(
            self.queue, inp.shape[1:-1], None, out.data, inp.data,
            np.int32(self.unknowns), self.ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=out.events + inp.events + wait_for)

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex measurement space which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        tmp_result = clarray.empty(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")
        tmp_result.add_event(self.prg.divergence(
            self.queue, inp.shape[1:-1], None, tmp_result.data, inp.data,
            np.int32(self.unknowns), self.ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=tmp_result.events + inp.events + wait_for))
        return tmp_result


class OperatorFiniteSymGradient(Operator):
    """Symmetrized gradient operator.

    This class implements the finite difference symmetrized gradient
    operation and the adjoint (negative symmetrized divergence).

    Parameters
    ----------
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.

    Attributes
    ----------
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      ratio : list of PyOpenCL.Array
        Ratio between the different unknowns
    """

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.ratio = clarray.to_device(
            self.queue,
            (par["weights"]).astype(
                     dtype=self.DTYPE_real))
        self._weights = par["weights"]

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        return self.prg.sym_grad(
            self.queue, inp.shape[1:-1], None, out.data, inp.data,
            np.int32(self.unknowns_TGV),
            self.ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=out.events + inp.events + wait_for)

    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        tmp_result = clarray.empty(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 8),
            self.DTYPE, "C")
        tmp_result.add_event(self.prg.sym_grad(
            self.queue, inp.shape[1:-1], None, tmp_result.data, inp.data,
            np.int32(self.unknowns_TGV),
            self.ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=tmp_result.events + inp.events + wait_for))
        return tmp_result

    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex measurement space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        return self.prg.sym_divergence(
            self.queue, inp.shape[1:-1], None, out.data, inp.data,
            np.int32(self.unknowns_TGV),
            self.ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=out.events + inp.events + wait_for)

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex measurement space which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        if "wait_for" in kwargs.keys():
            wait_for = kwargs["wait_for"]
        else:
            wait_for = []
        tmp_result = clarray.empty(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 4),
            self.DTYPE, "C")
        tmp_result.add_event(self.prg.sym_divergence(
            self.queue, inp.shape[1:-1], None, tmp_result.data, inp.data,
            np.int32(self.unknowns_TGV),
            self.ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=tmp_result.events + inp.events + wait_for))
        return tmp_result


class OperatorFiniteGradientStreamed(Operator):
    """Streamed gradient operator.

    This class implements the finite difference gradient
    operation and the adjoint (negative divergence).

    Parameters
    ----------
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.

    Attributes
    ----------
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      par_slices : int
        Slices to parallel transfer to the compute device.
      ratio : list of PyOpenCL.Array
        Ratio between the different unknowns
    """

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)

        self._weights = par["weights"]
        par["overlap"] = 1
        self._overlap = par["overlap"]
        self.par_slices = par["par_slices"]

        self.ratio = []
        for j in range(self.num_dev):
            self.ratio.append(
                clarray.to_device(
                    self.queue[4*j],
                    (par["weights"]).astype(
                        dtype=self.DTYPE_real)))

        self.unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        self._grad_shape = self.unknown_shape + (4,)

        self._stream_grad = self._defineoperator(
            [self._grad],
            [self._grad_shape],
            [[self.unknown_shape]])

        self._stream_div = self._defineoperator(
            [self._div],
            [self.unknown_shape],
            [[self._grad_shape]],
            reverse_dir=True)

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        self._stream_grad.eval(out, inp)

    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        out = np.zeros(self._grad_shape, dtype=self.DTYPE)
        self._stream_grad.eval([out], inp)
        return out

    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex measurement space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        self._stream_div.eval(out, inp)

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex measurement space which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        out = np.zeros(self.unknown_shape, dtype=self.DTYPE)
        self._stream_div.eval([out], inp)
        return out

    def _grad(self, outp, inp, par=None, idx=0, idxq=0,
              bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.prg[idx].gradient(
            self.queue[4*idx+idxq],
            (self._overlap+self.par_slices, self.dimY, self.dimX),
            None, outp.data, inp[0].data,
            np.int32(self.unknowns),
            self.ratio[idx].data, self.DTYPE_real(self._dz),
            wait_for=outp.events + inp[0].events + wait_for)

    def _div(self, outp, inp, par=None, idx=0, idxq=0,
             bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.prg[idx].divergence(
            self.queue[4*idx+idxq],
            (self._overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data, np.int32(self.unknowns),
            self.ratio[idx].data, np.int32(bound_cond),
            self.DTYPE_real(self._dz),
            wait_for=outp.events + inp[0].events + wait_for)

    def getStreamedGradientObject(self):
        """Access privat stream gradient object.

        Returns
        -------
          PyqMRI.Streaming.Stream:
              A PyQMRI streaming object for the gradient computation.
        """
        return self._stream_grad


class OperatorFiniteSymGradientStreamed(Operator):
    """Streamed symmetrized gradient operator.

    This class implements the finite difference symmetrized gradient
    operation and the adjoint (negative symmetrized divergence).

    Parameters
    ----------
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.

    Attributes
    ----------
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      par_slices : int
        Slices to parallel transfer to the compute device.
      ratio : list of PyOpenCL.Array
        Ratio between the different unknowns
    """

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)

        par["overlap"] = 1
        self._overlap = par["overlap"]
        self.par_slices = par["par_slices"]

        unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        self._grad_shape = unknown_shape + (4,)
        self._symgrad_shape = unknown_shape + (8,)

        self.ratio = []
        for j in range(self.num_dev):
            self.ratio.append(
                clarray.to_device(
                    self.queue[4*j],
                    (par["weights"]).astype(
                        dtype=self.DTYPE_real)))

        self._stream_symgrad = self._defineoperator(
            [self._symgrad],
            [self._symgrad_shape],
            [[self._grad_shape]],
            reverse_dir=True)

        self._stream_symdiv = self._defineoperator(
            [self._symdiv],
            [self._grad_shape],
            [[self._symgrad_shape]])

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        self._stream_symgrad.eval(out, inp)

    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        out = np.zeros(self._symgrad_shape, dtype=self.DTYPE)
        self._stream_symgrad.eval([out], inp)
        return out

    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex measurement space data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        self._stream_symdiv.eval(out, inp)

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex measurement space which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        out = np.zeros(self._grad_shape, dtype=self.DTYPE)
        self._stream_symdiv.eval([out], inp)
        return out

    def _symgrad(self, outp, inp, par=None, idx=0, idxq=0,
                 bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.prg[idx].sym_grad(
            self.queue[4*idx+idxq],
            (self._overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data, np.int32(self.unknowns),
            self.ratio[idx].data,
            self.DTYPE_real(self._dz),
            wait_for=outp.events + inp[0].events + wait_for)

    def _symdiv(self, outp, inp, par=None, idx=0, idxq=0,
                bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.prg[idx].sym_divergence(
            self.queue[4*idx+idxq],
            (self._overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            np.int32(self.unknowns),
            self.ratio[idx].data,
            np.int32(bound_cond),
            self.DTYPE_real(self._dz),
            wait_for=outp.events + inp[0].events + wait_for)

    def getStreamedSymGradientObject(self):
        """Access privat stream symmetrized gradient object.

        Returns
        -------
          PyqMRI.Streaming.Stream:
              A PyQMRI streaming object for the symmetrized gradient
              computation.
        """
        return self._stream_symgrad
