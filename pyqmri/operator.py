#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the classes for different linear Operators."""
from abc import ABC, abstractmethod
import pyopencl.array as clarray
import numpy as np
import scipy.special as sps
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
        self._unknown_shape = (self.unknowns,
                               self.NSlice,
                               self.dimY,
                               self.dimX)
        self._overlap = 0
        self.ratio = []
        self._weights = par["weights"]
        for j in range(self.num_dev):
            self.ratio.append(
                clarray.to_device(
                    self.queue[4*j],
                    (par["weights"]).astype(
                        dtype=self.DTYPE_real)))
            
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
                           streamed=False,
                           imagerecon=False):
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
        if imagerecon:
            op = OperatorKspaceImageRecon(
                par,
                prg[0],
                trafo=trafo,
                DTYPE=DTYPE,
                DTYPE_real=DTYPE_real)
            FT = op.NUFFT
            return op, FT
        
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
    
    def updateRatio(self, inp):
        for j in range(self.num_dev):
            self.ratio = clarray.to_device(
                self.queue[4*j],
                (self._weights*np.array(inp)).astype(
                         dtype=self.DTYPE_real))

    @staticmethod
    def GradientOperatorFactory(par,
                                prg,
                                DTYPE,
                                DTYPE_real,
                                streamed=False,
                                spacetimederivatives="",
                                **kwargs):
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
            if "IC" in spacetimederivatives:
              raise NotImplementedError
            op = OperatorFiniteGradientStreamed(par,
                                                prg,
                                                DTYPE,
                                                DTYPE_real)
        else:
            if "IC" in spacetimederivatives:
              op = OperatorFiniteSpaceTimeGradient(par,
                                                  prg[0],
                                                  DTYPE,
                                                  DTYPE_real,
                                                  kwargs["mu_1"],
                                                  kwargs["dt"],
                                                  kwargs["tsweight"])
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
                                   streamed=False,                                
                                   spacetimederivatives="",
                                   **kwargs):
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
            if "IC" in spacetimederivatives:
              raise NotImplementedError
            op = OperatorFiniteSymGradientStreamed(par,
                                                   prg,
                                                   DTYPE,
                                                   DTYPE_real)
        else:
            if "IC" in spacetimederivatives:
              op = OperatorFiniteSpaceTimeSymGradient(par,
                                                  prg[0],
                                                  DTYPE,
                                                  DTYPE_real,
                                                  kwargs["mu_1"],
                                                  kwargs["dt"],
                                                  kwargs["tsweight"])
            else:
                op = OperatorFiniteSymGradient(par,
                                               prg[0],
                                               DTYPE,
                                               DTYPE_real)
        return op

    @staticmethod
    def SoftSenseOperatorFactory(par,
                                prg,
                                DTYPE,
                                DTYPE_real,
                                streamed=False):
        """Sense forward/adjoint operator factory method."""
        if streamed:
            op = OperatorSoftSenseStreamed(par,
                                           prg,
                                           DTYPE,
                                           DTYPE_real)

        else:
            op = OperatorSoftSense(par,
                                   prg[0],
                                   DTYPE,
                                   DTYPE_real)
        return op, op.NUFFT

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
            DTYPE=self.DTYPE,
            DTYPE_real=self.DTYPE_real)


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
        self.queue = self.queue
        self.ctx = self.ctx[0]
        self._out_shape_fwd = (self.NScan, self.NSlice, self.dimY, self.dimX)

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
            self.queue[0], (self.NSlice, self.dimY, self.dimX), None,
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
        tmp_result = clarray.zeros(
            self.queue[0], (self.NScan, self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")
        self.prg.operator_fwd_imagespace(
            self.queue[0], (self.NSlice, self.dimY, self.dimX), None,
            tmp_result.data, inp[0].data, inp[2].data,
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=inp[0].events + wait_for).wait()
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
        out = clarray.zeros(
            self.queue[0], (self.unknowns, self.NSlice, self.dimY, self.dimX),
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
            self.queue[0], (self.NSlice, self.dimY, self.dimX), None,
            out.data, inp[0].data, inp[3].data, inp[1].data,
            np.int32(self.NScan),
            np.int32(self.unknowns),
            self.DTYPE_real(self._dz),
            self.ratio[0].data,
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
        # self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self._tmp_result = clarray.zeros(
            self.queue[0], (self.NScan, self.NC,
                         self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")
        if not trafo:
            self.Nproj = self.dimY
            self.N = self.dimX
        if par["is3D"] and trafo:
            self._out_shape_fwd = (self.NScan, self.NC,
                         1, self.Nproj, self.N)
        else:
            self._out_shape_fwd = (self.NScan, self.NC,
                         self.NSlice, self.Nproj, self.N)
        self.NUFFT = CLnuFFT.create(self.ctx,
                                    self.queue[0],
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
                self.queue[0],
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
            wait_for=wait_for + self._tmp_result.events + out.events)

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
                self.queue[0],
                (self.NSlice, self.dimY, self.dimX),
                None,
                self._tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=(self._tmp_result.events + inp[0].events
                          + wait_for)))
        tmp_sino = clarray.zeros(
            self.queue[0],
            self._out_shape_fwd,
            self.DTYPE, "C")
        self.NUFFT.FFT(tmp_sino, self._tmp_result,
                       wait_for=tmp_sino.events).wait()
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
            self.queue[0], (self.NSlice, self.dimY, self.dimX), None,
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
        out = clarray.zeros(
            self.queue[0], (self.unknowns, self.NSlice, self.dimY, self.dimX),
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
            self.queue[0], (self.NSlice, self.dimY, self.dimX), None,
            out.data, self._tmp_result.data, inp[2].data,
            inp[3].data, inp[1].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns), self.DTYPE_real(self._dz),
            self.ratio[0].data,
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
        # self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.packs = par["packs"]*par["numofpacks"]
        self._tmp_result = clarray.zeros(
            self.queue[0], (self.NScan, self.NC,
                         self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")
        self._out_shape_fwd = (self.NScan, self.NC,
                               self.packs, self.Nproj, self.N)

        self.Nproj = self.dimY
        self.N = self.dimX
        self.NUFFT = CLnuFFT.create(self.ctx,
                                    self.queue[0],
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
                self.queue[0],
                (self.NSlice, self.dimY, self.dimX),
                None,
                self._tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=(self._tmp_result.events + inp[0].events +
                          inp[1].events + inp[2].events
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
                self.queue[0],
                (self.NSlice, self.dimY, self.dimX),
                None,
                self._tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=(self._tmp_result.events + inp[0].events +
                          inp[1].events + inp[2].events
                          + wait_for)))
        tmp_sino = clarray.zeros(
            self.queue[0],
            (self.NScan, self.NC, self.packs, self.Nproj, self.N),
            self.DTYPE, "C")

        self.NUFFT.FFT(tmp_sino, self._tmp_result).wait()
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
            self.queue[0], (self.NSlice, self.dimY, self.dimX), None,
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
        out = clarray.zeros(
            self.queue[0], (self.unknowns, self.NSlice, self.dimY, self.dimX),
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
            self.queue[0], (self.NSlice, self.dimY, self.dimX), None,
            out.data, self._tmp_result.data, inp[2].data,
            inp[3].data, inp[1].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns), self.DTYPE_real(self._dz),
            self.ratio[0].data,
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
        self._unknown_shape = (self.unknowns,
                       self.par_slices+self._overlap,
                       self.dimY,
                       self.dimX)
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
            np.int32(self.unknowns),
            np.int32(bound_cond), self.DTYPE_real(self._dz),
            self.ratio[idx].data,
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
        self._unknown_shape = (self.unknowns,
                       self.par_slices+self._overlap,
                       self.dimY,
                       self.dimX)
        if not trafo:
            self.Nproj = self.dimY
            self.N = self.dimX
        for j in range(self.num_dev):
            for i in range(2):
                self._tmp_result.append(
                    clarray.zeros(
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
            np.int32(self.unknowns),
            np.int32(bound_cond), self.DTYPE_real(self._dz),
            self.ratio[idx].data,
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
        self._unknown_shape = (self.unknowns,
                       self.par_slices+self._overlap,
                       self.dimY,
                       self.dimX)
        self.Nproj = self.dimY
        self.N = self.dimX

        for j in range(self.num_dev):
            for i in range(2):
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
            np.int32(self.unknowns),
            np.int32(bound_cond), self.DTYPE_real(self._dz),
            self.ratio[idx].data,
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
        self.tmp_grad_array = None
        self.precond = False

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
            
        if not self.precond:
            return self.prg.gradient(
            self.queue, inp.shape[1:], None, out.data, inp.data,
            np.int32(self.unknowns),
            self.DTYPE_real(self._dz),
            self.ratio[0].data,
            wait_for=out.events + inp.events + wait_for)
            
        if self.tmp_grad_array is None:
            self.tmp_grad_array = clarray.zeros_like(inp)
            
        self.prg.squarematvecmult(self.queue, inp.shape[1:], None,
            self.tmp_grad_array.data, self.precondmat.data, inp.data, np.int32(self.unknowns),
            wait_for=self.tmp_grad_array.events + inp.events + wait_for)
        
        return self.prg.gradient(
            self.queue, inp.shape[1:], None, out.data, self.tmp_grad_array.data,
            np.int32(self.unknowns),
            self.DTYPE_real(self._dz),
            self.ratio[0].data,
            wait_for=out.events + self.tmp_grad_array.events + wait_for)

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
        tmp_result = clarray.zeros(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 4),
            self.DTYPE, "C")
        
        if not self.precond:
            self.prg.gradient(
                self.queue, inp.shape[1:], None, tmp_result.data, inp.data,
                np.int32(self.unknowns),
                self.DTYPE_real(self._dz),
                self.ratio[0].data,
                wait_for=tmp_result.events + inp.events + wait_for).wait()
            return tmp_result
            
        if self.tmp_grad_array is None:
            self.tmp_grad_array = clarray.zeros_like(inp)
        self.prg.squarematvecmult(self.queue, inp.shape[1:], None,
            self.tmp_grad_array.data, self.precondmat.data, inp.data, np.int32(self.unknowns),
            wait_for=self.tmp_grad_array.events + inp.events + wait_for)
        
        self.prg.gradient(
            self.queue, inp.shape[1:], None, tmp_result.data, self.tmp_grad_array.data,
            np.int32(self.unknowns),
            self.DTYPE_real(self._dz),
            self.ratio[0].data,
            wait_for=tmp_result.events + self.tmp_grad_array.events + wait_for).wait()
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
            
        if not self.precond:
            return self.prg.divergence(
                    self.queue, inp.shape[1:-1], None, out.data, inp.data,
                    np.int32(self.unknowns),
                    self.DTYPE_real(self._dz),
                    self.ratio[0].data,
                    wait_for=out.events + inp.events + wait_for)
            
        if self.tmp_grad_array is None:
            self.tmp_grad_array = clarray.zeros_like(out)
        self.prg.divergence(
                    self.queue, inp.shape[1:-1], None, self.tmp_grad_array.data, inp.data,
                    np.int32(self.unknowns),
                    self.DTYPE_real(self._dz),
                    self.ratio[0].data,
                    wait_for=self.tmp_grad_array.events + inp.events + wait_for)
            
        return self.prg.squarematvecmult_conj(self.queue, inp.shape[1:-1], None,
            out.data, self.precondmat.data, self.tmp_grad_array.data, np.int32(self.unknowns),
            wait_for=self.tmp_grad_array.events + out.events + wait_for)

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
        tmp_result = clarray.zeros(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")
        
        if not self.precond:
            self.prg.divergence(
                        self.queue, inp.shape[1:-1], None, tmp_result.data, inp.data,
                        np.int32(self.unknowns),
                        self.DTYPE_real(self._dz),
                        self.ratio[0].data,
                        wait_for=tmp_result.events + inp.events + wait_for).wait()
            return tmp_result
        
        if self.tmp_grad_array is None:
            self.tmp_grad_array = clarray.zeros_like(tmp_result)
            
        self.prg.divergence(
                    self.queue, inp.shape[1:-1], None, self.tmp_grad_array.data, inp.data,
                    np.int32(self.unknowns),
                    self.DTYPE_real(self._dz),
                    self.ratio[0].data,
                    wait_for=self.tmp_grad_array.events + inp.events + wait_for).wait()
            
        self.prg.squarematvecmult_conj(self.queue, inp.shape[1:-1], None,
            tmp_result.data, self.precondmat.data, self.tmp_grad_array.data, np.int32(self.unknowns),
            wait_for=self.tmp_grad_array.events + tmp_result.events + wait_for).wait()
        return tmp_result

    def updatePrecondMat(self, inp):
        self.precondmat = clarray.to_device(
            self.queue,
            inp)

    def updateRatio(self, inp):      
        self.ratio = (
            clarray.to_device(
                self.queue,
                self._weights*inp.astype(
                    dtype=self.DTYPE_real)))

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
        tmp_result = clarray.zeros(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 8),
            self.DTYPE, "C")
        self.prg.sym_grad(
            self.queue, inp.shape[1:-1], None, tmp_result.data, inp.data,
            np.int32(self.unknowns_TGV),
            self.DTYPE_real(self._dz),
            wait_for=tmp_result.events + inp.events + wait_for).wait()
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
        tmp_result = clarray.zeros(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 4),
            self.DTYPE, "C")
        self.prg.sym_divergence(
            self.queue, inp.shape[1:-1], None, tmp_result.data, inp.data,
            np.int32(self.unknowns_TGV),
            self.DTYPE_real(self._dz),
            wait_for=tmp_result.events + inp.events + wait_for).wait()
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
        self._unknown_shape = (self.unknowns,
               self.par_slices+self._overlap,
               self.dimY,
               self.dimX)

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
            self.DTYPE_real(self._dz),
            wait_for=outp.events + inp[0].events + wait_for)

    def _div(self, outp, inp, par=None, idx=0, idxq=0,
             bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.prg[idx].divergence(
            self.queue[4*idx+idxq],
            (self._overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data, np.int32(self.unknowns),
            np.int32(bound_cond),
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

    def updateRatio(self, inp):
        for j in range(self.num_dev):
            self.ratio = clarray.to_device(
                self.queue[4*j],
                (self._weights*inp).astype(
                          dtype=self.DTYPE_real))
    
    def updatePrecondMat(self, inp):
        raise(NotImplementedError("Streamed and Preconditioning is not implemented."))

    # def updateRatio(self, inp):
    #     x = np.require(np.swapaxes(inp, 0, 1), requirements='C')
    #     grad = np.zeros(x.shape + (4,), dtype=self.DTYPE)
    #     for i in range(self.num_dev):
    #         for j in range(x.shape[1]):
    #             self.ratio[i][j] = 1
    #     self.fwd([grad], [[x]])
    #     grad = np.require(np.swapaxes(grad, 0, 1),
    #                       requirements='C')

    #     scale = np.reshape(
    #         inp, (self.unknowns,
    #               self.NSlice * self.dimY * self.dimX))
    #     grad = np.reshape(
    #         grad, (self.unknowns,
    #                self.NSlice *
    #                self.dimY *
    #                self.dimX * 4))

    #     print("Total Norm of grad x pre: ", np.sum(np.abs(grad)))
    #     gradnorm = np.sum(np.abs(grad), axis=-1)
    #     print("Norm of grad x pre: ", gradnorm)
    #     gradnorm /= np.sum(gradnorm)/self.unknowns
    #     scale = 1/gradnorm
    #     scale[~np.isfinite(scale)] = 1

    #     for i in range(self.num_dev):
    #         for j in range(inp.shape[0])[:self.unknowns_TGV]:
    #             self.ratio[i][j] = scale[j] * self._weights[j]
    #     for i in range(self.num_dev):
    #         for j in range(inp.shape[0])[self.unknowns_TGV:]:
    #             self.ratio[i][j] = scale[j] * self._weights[j]

    #     grad = np.zeros(x.shape + (4,), dtype=self.DTYPE)
    #     self.fwd([grad], [[x]])
    #     grad = np.require(np.swapaxes(grad, 0, 1),
    #                       requirements='C')

    #     grad = np.reshape(
    #         grad, (self.unknowns,
    #                self.NSlice *
    #                self.dimY *
    #                self.dimX * 4))
    #     print("Norm of grad x post: ",  np.sum(np.abs(grad), axis=-1))
    #     print("Total Norm of grad x post: ",  np.sum(np.abs(grad)))

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

        self._weights = par["weights"]
        par["overlap"] = 1
        self._overlap = par["overlap"]
        self.par_slices = par["par_slices"]
        self._unknown_shape = (self.unknowns,
                       self.par_slices+self._overlap,
                       self.dimY,
                       self.dimX)

        unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        self._grad_shape = unknown_shape + (4,)
        self._symgrad_shape = unknown_shape + (8,)

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

    def updateRatio(self, inp):
        pass
        # for j in range(self.num_dev):
        #     self.ratio = clarray.to_device(
        #         self.queue[4*j],
        #         (self._weights*inp).astype(
        #                  dtype=self.DTYPE_real))


class OperatorSoftSense(Operator):
    """Soft-SENSE Operator.

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
                 DTYPE_real=np.float32, trafo=False):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]

        self.NMaps = par["NMaps"]

        self._tmp_result = clarray.zeros(
            self.queue, (self.NScan, self.NC,
                         self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")
        if not trafo:
            self.Nproj = self.dimY
            self.N = self.dimX

        self._out_shape_fwd = (self.NScan, self.NC,
                               self.NSlice, self.Nproj, self.N)

        # in case of 3D mask include in operator and mask fft with ones
        self.mask = np.array([])
        self._mask_tmp = par["mask"].copy()

        if self._mask_tmp.ndim > 2:
            self.mask = clarray.to_device(self.queue, par["mask"])
            par["mask"] = np.require(
                np.ones((par["dimY"], par["dimX"]), dtype=par["DTYPE_real"]),
                requirements='C')

        self.NUFFT = CLnuFFT.create(self.ctx,
                                    self.queue,
                                    par,
                                    radial=trafo,
                                    DTYPE=DTYPE,
                                    DTYPE_real=DTYPE_real)

        par["mask"] = self._mask_tmp.copy()
        del self._mask_tmp

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear Soft-SENSE operator from image space to kspace
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex image data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex kspace data which is used as input.
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
            self.prg.operator_fwd_ssense(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self._tmp_result.data,
                inp[0].data,
                inp[1].data,
                np.int32(self.NC),
                np.int32(self.NMaps),
                wait_for=(self._tmp_result.events + inp[0].events + wait_for)))
        if self.mask.size != 0:
            out.add_event(
                self.NUFFT.FFT(
                    out,
                    self._tmp_result,
                    wait_for=wait_for + self._tmp_result.events))

            return self.NUFFT.prg.masking(
                self.NUFFT.queue,
                (self._tmp_result.size,),
                None,
                out.data,
                self.mask.data,
                wait_for=(wait_for+out.events+self._tmp_result.events))

        return self.NUFFT.FFT(
                    out,
                    self._tmp_result,
                    wait_for=wait_for + self._tmp_result.events)

    def fwdoop(self, inp, **kwargs):
        """Forward operator application out-of-place.

        Apply the linear Soft-SENSE operator from image space to kspace
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method needs to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex image data which is used as input.
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
            self.prg.operator_fwd_ssense(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self._tmp_result.data,
                inp[0].data,
                inp[1].data,
                np.int32(self.NC),
                np.int32(self.NMaps),
                wait_for=(self._tmp_result.events + inp[0].events
                          + wait_for)))
        tmp_sino = clarray.zeros(
            self.queue,
            self._out_shape_fwd,
            self.DTYPE, "C")
        self.NUFFT.FFT(tmp_sino, self._tmp_result,
                       wait_for=tmp_sino.events).wait()
        if self.mask.size != 0:
            tmp_sino.add_event(
                self.NUFFT.prg.masking(
                    self.NUFFT.queue,
                    (tmp_sino.size,),
                    None,
                    tmp_sino.data,
                    self.mask.data,
                    wait_for=(wait_for+tmp_sino.events)
                )
            )
        return tmp_sino

    def adj(self, out, inp, **kwargs):
        """Adjoint operator application in-place.

        Apply the linear Soft-SENSE operator from kspace to image space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex image data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex kspace data which is used as input.
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
        if self.mask.size != 0:
            inp[0].add_event(
                self.NUFFT.prg.masking(
                    self.NUFFT.queue,
                    (inp[0].size,),
                    None,
                    inp[0].data,
                    self.mask.data,
                    wait_for=(wait_for+inp[0].events)
                )
            )
        self._tmp_result.add_event(
            self.NUFFT.FFTH(
                self._tmp_result,
                inp[0],
                wait_for=(wait_for + inp[0].events)))
        return self.prg.operator_ad_ssense(
            self.queue,
            (self.NSlice, self.dimY, self.dimX),
            None,
            out.data,
            self._tmp_result.data,
            inp[1].data,
            np.int32(self.NC),
            np.int32(self.NMaps),
            wait_for=self._tmp_result.events + out.events)

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear Soft-SENSE operator from kspace to image space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method needs to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex kspace data which is used as input.
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
        if self.mask.size != 0:
            inp[0].add_event(
                self.NUFFT.prg.masking(
                    self.NUFFT.queue,
                    (inp[0].size,),
                    None,
                    inp[0].data,
                    self.mask.data,
                    wait_for=(wait_for+inp[0].events)
                )
            )
        self._tmp_result.add_event(
            self.NUFFT.FFTH(
                self._tmp_result,
                inp[0],
                wait_for=(wait_for + inp[0].events)))
        out = clarray.empty(
            self.queue,
            (self.NMaps, self.NSlice, self.dimY, self.dimX),
            dtype=self.DTYPE)
        self.prg.operator_ad_ssense(
            out.queue,
            (self.NSlice, self.dimY, self.dimX),
            None,
            out.data,
            self._tmp_result.data,
            inp[1].data,
            np.int32(self.NC),
            np.int32(self.NMaps),
            wait_for=(self._tmp_result.events + out.events)).wait()
        return out

    def adjKyk1(self, out, inp, **kwargs):
        """Apply the linear operator from kspace to image space.

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
        if self.mask.size != 0:
            inp[0].add_event(
                self.NUFFT.prg.masking(
                    self.NUFFT.queue,
                    (inp[0].size,),
                    None,
                    inp[0].data,
                    self.mask.data,
                    wait_for=(wait_for+inp[0].events)
                )
            )
        self._tmp_result.add_event(
            self.NUFFT.FFTH(
                self._tmp_result,
                inp[0],
                wait_for=(wait_for + inp[0].events)))
        return self.prg.update_Kyk1_ssense(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data,
            self._tmp_result.data,
            inp[1].data,
            inp[2].data,
            np.int32(self.NC),
            np.int32(self.NMaps),
            inp[3].data,
            self.DTYPE_real(self._dz),
            wait_for=(self._tmp_result.events +
                      out.events + inp[2].events + inp[3].events))


class OperatorSoftSenseStreamed(Operator):
    """ The streamed version of the Soft-SENSE Operator

    Attributes:
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
      NUFFT : list of PyQMRI.transforms.PyOpenCLnuFFT
        A list of NUFFT objects. One for each context.
      FTstr : PyQMRI.Stream
        A streamed version of the used (non-uniform) FFT, applied forward.
      unknown_shape : tuple of int
        Size of the reconstructed images
      data_shape : tuple of int
        Size of the data
    """

    def __init__(self,
                 par,
                 prg,
                 DTYPE=np.complex64,
                 DTYPE_real=np.float32,
                 trafo=False):
        super().__init__(par,
                         prg,
                         DTYPE,
                         DTYPE_real)
        self._overlap = par["overlap"]
        self.par_slices = par["par_slices"]
        self.NMaps = par["NMaps"]

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

        self.unknown_shape = (self.NSlice, self.NMaps, self.dimY, self.dimX)
        coil_shape = (self.NSlice, self.NMaps, self.NC, self.dimY, self.dimX)
        self.data_shape = (self.NSlice, self.NScan, self.NC, self.Nproj,
                           self.N)
        trans_shape = (self.NSlice, self.NScan,
                       self.NC, self.dimY, self.dimX)
        grad_shape = self.unknown_shape + (4,)

        self.fwdstr = self._defineoperator(
            [self._fwdstreamed],
            [self.data_shape],
            [[self.unknown_shape,
              coil_shape]])

        self.adjstrKyk1 = self._defineoperator(
            [self._adjstreamedKyk1],
            [self.unknown_shape],
            [[self.data_shape,
              coil_shape,
              grad_shape]])

        self.adjstr = self._defineoperator(
            [self._adjstreamed],
            [self.unknown_shape],
            [[self.data_shape,
              coil_shape]])

        self.FTstr = self._defineoperator(
            [self.FT],
            [self.data_shape],
            [[trans_shape]])

    def fwd(self, out, inp, **kwargs):
        """Forward operator application in-place.

        Apply the linear Soft-SENSE operator from image space to kspace
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex image data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex kspace data which is used as input.
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

        Apply the linear Soft-SENSE operator from image space to kspace
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method needs to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex image data which is used as input.
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

        Apply the linear Soft-SENSE operator from kspace to image space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex image data which is the result of the
            computation.
          inp : PyOpenCL.Array
            The complex kspace data which is used as input.
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event
            A PyOpenCL event to wait for.
        """
        self.adjstr.eval(out, inp)

    def adjoop(self, inp, **kwargs):
        """Adjoint operator application out-of-place.

        Apply the linear Soft-SENSE operator from kspace to image space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method needs to generate a temporary array and will return it as
        the result.

        Parameters
        ----------
          inp : PyOpenCL.Array
            The complex kspace data which is used as input.
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
        """Apply the linear operator from kspace to image space.

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Parameters
        ----------
          out : PyOpenCL.Array
            The complex parameter space data which is used as input.
          inp : PyOpenCL.Array
            The complex parameter space data which is used as input.
        """
        self.adjstrKyk1.eval(out, inp)

    def _fwdstreamed(self, out, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        self._tmp_result[2*idx+idxq].add_event(
            self.prg[idx].operator_fwd_ssense(
                self.queue[4*idx+idxq],
                (self.par_slices+self._overlap, self.dimY, self.dimX), None,
                self._tmp_result[2*idx+idxq].data,
                inp[0].data,
                inp[1].data,
                np.int32(self.NC),
                np.int32(self.NMaps),
                wait_for=(self._tmp_result[2*idx+idxq].events +
                          inp[0].events+wait_for)))

        return self.NUFFT[2*idx+idxq].FFT(
                out,
                self._tmp_result[2*idx+idxq],
                wait_for=out.events+wait_for+self._tmp_result[2*idx+idxq].events)

    def _adjstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []

        self._tmp_result[2*idx+idxq].add_event(
            self.NUFFT[2*idx+idxq].FFTH(
                self._tmp_result[2*idx+idxq], inp[0],
                wait_for=(wait_for+inp[0].events +
                          self._tmp_result[2*idx+idxq].events)))

        return self.prg[idx].operator_ad_ssense(
            self.queue[4*idx+idxq],
            (self.par_slices+self._overlap, self.dimY, self.dimX),
            None,
            outp.data,
            self._tmp_result[2*idx+idxq].data,
            inp[1].data,
            np.int32(self.NC),
            np.int32(self.NMaps),
            wait_for=(self._tmp_result[2*idx+idxq].events + inp[1].events+wait_for))

    def _adjstreamedKyk1(self, outp, inp, par=None, idx=0, idxq=0,
                         bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []

        self._tmp_result[2*idx+idxq].add_event(
            self.NUFFT[2*idx+idxq].FFTH(
                self._tmp_result[2*idx+idxq], inp[0],
                wait_for=(wait_for+inp[0].events +
                          self._tmp_result[2*idx+idxq].events)))

        return self.prg[idx].update_Kyk1_ssense(
            self.queue[4*idx+idxq],
            (self.par_slices+self._overlap, self.dimY, self.dimX), None,
            outp.data, self._tmp_result[2*idx+idxq].data,
            inp[1].data,
            inp[2].data,
            np.int32(self.NC),
            np.int32(self.NMaps),
            par[0][idx].data,
            np.int32(bound_cond),
            self.DTYPE_real(self._dz),
            wait_for=(
                self._tmp_result[2*idx+idxq].events +
                outp.events+inp[1].events +
                inp[2].events + wait_for))

    def FT(self, outp, inp, par=None, idx=0, idxq=0,
           bound_cond=0, wait_for=None):
        if wait_for is None:
            wait_for = []
        return self.NUFFT[2*idx+idxq].FFT(outp, inp[0])



class OperatorFiniteSpaceTimeGradient(Operator):
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

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32, mu1=1, dt=1, tsweight=1):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.ratio = None
        self.tmp_grad_array = None
        self.precond = False
        self.timeSpaceWeight = tsweight
        
        self.mu1 = mu1
        self.dt = dt
        
        self.mu1, self.dt = self.computeTimeSpaceWeight(self.mu1, self.dt, tsweight)
        
        self.dt = self.dt.astype(DTYPE_real)
        self.dt = clarray.to_device(self.queue, self.dt)
        

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
            
        return self.prg.gradient_w_time(
        self.queue, inp.shape[1:], None, out.data, inp.data,
        np.int32(self.unknowns),
        self.DTYPE_real(self._dz),
        self.DTYPE_real(self.mu1),
        self.dt.data,
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
        tmp_result = clarray.zeros(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 4),
            self.DTYPE, "C")
        
        self.prg.gradient_w_time(
            self.queue, inp.shape[1:], None, tmp_result.data, inp.data,
            np.int32(self.unknowns),
            self.DTYPE_real(self._dz),
            self.DTYPE_real(self.mu1),
            self.dt.data,
            wait_for=tmp_result.events + inp.events + wait_for).wait()
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
            
        return self.prg.divergence_w_time(
                self.queue, inp.shape[1:-1], None, out.data, inp.data,
                np.int32(self.unknowns),
                self.DTYPE_real(self._dz),
                self.DTYPE_real(self.mu1),
                self.dt.data,
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
        tmp_result = clarray.zeros(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")

        self.prg.divergence_w_time(
                    self.queue, inp.shape[1:-1], None, tmp_result.data, inp.data,
                    np.int32(self.unknowns),
                    self.DTYPE_real(self._dz),
                    self.DTYPE_real(self.mu1),
                    self.dt.data,
                    wait_for=tmp_result.events + inp.events + wait_for).wait()
        return tmp_result


    def updateRatio(self, inp):
        self.precondmat = clarray.to_device(
            self.queue,
            inp)
        
    def computeTimeSpaceWeight(self, ds, dt, t):
        timeSpaceRatio = 1/t
        
        if timeSpaceRatio > 0 and timeSpaceRatio <= 1:
            tmp = sps.ellipk(1 - timeSpaceRatio**2)
            mu2 = np.pi / (2*tmp)
            mu1 = timeSpaceRatio *  mu2
        elif timeSpaceRatio > 1:
            timeSpaceRatio = 1/timeSpaceRatio
            tmp = sps.ellipk(1 - timeSpaceRatio**2)
            mu1 = np.pi / (2*tmp)
            mu2 = timeSpaceRatio * mu1
        else:
            raise ValueError("Invalid value for time/space weighting")
            
        return ds/mu1, dt/mu2
        
class OperatorFiniteSpaceTimeSymGradient(Operator):
    """Symmetrized gradient operator.

    This class implements the finite difference symmetrized gradient operation and
    the adjoint (negative symmetrized divergence).

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

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32, mu1=1, dt=1, tsweight=1):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.ratio = None
        self.tmp_grad_array = None
        self.precond = False
        self.timeSpaceWeight = tsweight
        
        self.mu1 = mu1
        self.dt = dt
        
        self.mu1, self.dt = self.computeTimeSpaceWeight(self.mu1, self.dt, tsweight)
        
        self.dt = self.dt.astype(DTYPE_real)
        self.dt = clarray.to_device(self.queue, self.dt)
        

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
            
        return self.prg.sym_grad_w_time(
        self.queue, inp.shape[1:-1], None, out[0].data, out[1].data, inp.data,
        np.int32(self.unknowns),
        self.DTYPE_real(self._dz),
        self.DTYPE_real(self.mu1),
        self.dt.data,
        wait_for=out[0].events + out[1].events + inp.events + wait_for)
            

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
        tmp_result1 = clarray.zeros(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 4),
            self.DTYPE, "C")
        tmp_result2 = clarray.zeros(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 8),
            self.DTYPE, "C")
        
        self.prg.sym_grad_w_time(
            self.queue, inp.shape[1:-1], None, tmp_result1.data,tmp_result2.data, inp.data,
            np.int32(self.unknowns),
            self.DTYPE_real(self._dz),
            self.DTYPE_real(self.mu1),
            self.dt.data,
            wait_for=tmp_result1.events + tmp_result2.events + inp.events + wait_for)
        return [tmp_result1, tmp_result2]

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
            
        return self.prg.sym_divergence_w_time(
                self.queue, inp[0].shape[1:-1], None, out.data, inp[0].data, inp[1].data,
                np.int32(self.unknowns),
                self.DTYPE_real(self._dz),
                self.DTYPE_real(self.mu1),
                self.dt.data,
                wait_for=out.events + inp[0].events + inp[1].events + wait_for)

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
        tmp_result = clarray.zeros(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX, 4),
            self.DTYPE, "C")

        self.prg.sym_divergence_w_time(
                    self.queue, inp[0].shape[1:-1], None, tmp_result.data, inp[0].data, inp[1].data,
                    np.int32(self.unknowns),
                    self.DTYPE_real(self._dz),
                    self.DTYPE_real(self.mu1),
                    self.dt.data,
                    wait_for=tmp_result.events + inp[0].events + inp[1].events + wait_for).wait()
        return tmp_result


    def updateRatio(self, inp):
        self.precondmat = clarray.to_device(
            self.queue,
            inp)
        
    def computeTimeSpaceWeight(self, ds, dt, t):
        timeSpaceRatio = 1/t
        
        if timeSpaceRatio > 0 and timeSpaceRatio <= 1:
            tmp = sps.ellipk(1 - timeSpaceRatio**2)
            mu2 = np.pi / (2*tmp)
            mu1 = timeSpaceRatio *  mu2
        elif timeSpaceRatio > 1:
            timeSpaceRatio = 1/timeSpaceRatio
            tmp = sps.ellipk(1 - timeSpaceRatio**2)
            mu1 = np.pi / (2*tmp)
            mu2 = timeSpaceRatio * mu1
        else:
            raise ValueError("Invalid value for time/space weighting")
            
        return ds/mu1, dt/mu2    
    
class OperatorKspaceImageRecon(OperatorKspace):
    """k-Space based Operator for simple image reconstruction.

    This class serves as linear operator between iamge and k-space.

    Use this operator if you want to perform complex image fitting from
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
        super().__init__(par, prg, DTYPE, DTYPE_real, trafo)
        self.queue = self.queue[0]
        
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
            self.prg.operator_fwd_imagerecon(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self._tmp_result.data, inp[0].data,
                inp[1].data,
                np.int32(self.NC),
                np.int32(self.NScan),
                wait_for=(self._tmp_result.events + inp[0].events
                          + wait_for)))

        return self.NUFFT.FFT(
            out,
            self._tmp_result,
            wait_for=wait_for + self._tmp_result.events + out.events)

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
            self.prg.operator_fwd_imagerecon(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self._tmp_result.data, inp[0].data,
                inp[1].data,
                np.int32(self.NC),
                np.int32(self.NScan),
                wait_for=(self._tmp_result.events + inp[0].events
                          + wait_for)))
        tmp_sino = clarray.zeros(
            self.queue,
            self._out_shape_fwd,
            self.DTYPE, "C")
        self.NUFFT.FFT(tmp_sino, self._tmp_result,
                       wait_for=tmp_sino.events).wait()
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

        return self.prg.operator_ad_imagerecon(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self._tmp_result.data, inp[1].data,
            np.int32(self.NC),
            np.int32(self.NScan),
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
        out = clarray.zeros(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            dtype=self.DTYPE)
        self.prg.operator_ad(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self._tmp_result.data, inp[1].data,
            np.int32(self.NC),
            np.int32(self.NScan),
            wait_for=(self._tmp_result.events
                      + out.events)).wait()
        return out

    def adjKyk1(self, out, inp, **kwargs):
        raise NotImplementedError("adjKyk1 not implemented ofr IC recon.")