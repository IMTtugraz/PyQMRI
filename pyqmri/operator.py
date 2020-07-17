#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This module holds the classes for different linear Operator.

Attribues:
  self.DTYPE (complex64):
    Complex working precission. Currently single precission only.
  self.DTYPE_real (float32):
    Real working precission. Currently single precission only.
"""
from abc import ABC, abstractmethod
import pyopencl.array as clarray
import numpy as np
from pyqmri.transforms import PyOpenCLnuFFT as CLnuFFT
import pyqmri.streaming as streaming


class Operator(ABC):
    """ Abstract base class for linear Operators used in the optimization.

    This class serves as the base class for all linear operators used in
    the varous optimization algorithms. it requires to implement a forward
    and backward application in and out of place.

    Attributes:
      NScan (int):
        Number of total measurements (Scans)
      NC (int):
        Number of complex coils
      NSlice (int):
        Number ofSlices
      dimX (int):
        X dimension of the parameter maps
      dimY (int):
        Y dimension of the parameter maps
      N (int):
        N number of samples per readout
      Nproj (int):
        Number of rreadouts
      unknowns_TGV (int):
        Number of unknowns which should be regularized with TGV. It is assumed
        that these occure first in the unknown vector. Currently at least 1
        TGV unknown is required.
      unknowns_H1 (int):
        Number of unknowns which should be regularized with H1. It is assumed
        that these occure after all TGV unknowns in the unknown vector.
        Currently this number can be zero which implies that no H1
        regularization is used.
      unknowns (int):
        The sum of TGV and H1 unknowns.
      ctx ((list of) PyOpenCL.Context):
        The context for the PyOpenCL computations. If streamed operations are
        used a list of ctx is required. One for each computation device.
      queue ((list of) PyOpenCL.Queue):
        The computation Queue for the PyOpenCL kernels. If streamed operations
        are used a list of queues is required. Four for each computation
        device.
      dz (float):
        The ratio between the physical X,Y dimensions vs the Z dimension.
        This allows for anisotrpic regularization along the Z dimension.
      num_dev (int):
        Number of compute devices
      tmp_result (list):
        A placeholder for an list of temporary PyOpenCL.Arrays if streamed
        operators are used. In the case of one large block of data this
        reduces to a single PyOpenCL.Array.
      NUFFT (PyQMRI.transforms.PyOpenCLnuFFT):
        A PyOpenCLnuFFT object to perform forward and backword transformations
        from image to k-space and vice versa.
      prg (PyOpenCL.Program):
        The PyOpenCL program containing all compiled kernels.
    """

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        """ Setup a Operator object
        Args:
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          prg (PyOpenCL.Program):
            The PyOpenCL.Program object containing the necessary kernels to
            execute the linear Operator.
        """
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
        self.tmp_result = []
        self.NUFFT = []
        self.prg = prg
        self.DTYPE = DTYPE
        self.DTYPE_real = DTYPE_real
        self.par_slices = self.NSlice
        self.overlap = 0

    @abstractmethod
    def fwd(self, out, inp, wait_for=[]):
        """ Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        Args:
          out (PyOpenCL.Array):
            The complex measurement space data which is the result of the \
            computation.
          inp (PyOpenCL.Array):
            The complex parameter space data which is used as input.
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        Returns:
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        ...

    @abstractmethod
    def adj(self, out, inp, wait_for=[]):
        """ Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        Args:
          out (PyOpenCL.Array):
            The complex parameter space data which is the result of the \
            computation.
          inp (PyOpenCL.Array):
            The complex measurement space data which is used as input.
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        Returns:
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        ...

    @abstractmethod
    def fwdoop(self, inp, wait_for=[]):
        """ Apply the linear operator from parameter space to measurement space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Args:
          inp (PyOpenCL.Array):
            The complex parameter space data which is used as input.
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        Returns:
          PyOpenCL.Array: A PyOpenCL array containing the result of the
          computation.
        """
        ...

    @abstractmethod
    def adjoop(self, inp, wait_for=[]):
        """ Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        This method need to generate a temporary array and will return it as
        the result.

        Args:
          inp (PyOpenCL.Array):
            The complex measurement space which is used as input.
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        Returns:
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
        """MRI forward/adjoint operator factory method."""
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
                        trafo=trafo,
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
                        trafo=trafo,
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
        """Gradient forward/adjoint operator factory method."""
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
        """Symmetrized Gradient forward/adjoint operator factory method."""
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
            self.overlap,
            slices,
            self.queue,
            self.num_dev,
            reverse_dir,
            posofnorm,
            DTYPE=self.DTYPE)


class OperatorImagespace(Operator):
    """ Imagespace based Operator
    This class serves as linear operator between parameter and imagespace.

    Use this operator if you want to perform complex parameter fitting from
    complex image space data without the need of performing FFTs.
    """

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]

    def fwd(self, out, inp, wait_for=[]):
        return self.prg.operator_fwd_imagespace(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, inp[0].data, inp[2].data,
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=inp[0].events + out.events + wait_for)

    def fwdoop(self, inp, wait_for=[]):
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

    def adj(self, out, inp, wait_for=[]):
        return self.prg.operator_ad_imagespace(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, inp[0].data, inp[2].data,
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + inp[0].events + out.events)

    def adjoop(self, inp, wait_for=[]):
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

    def adjKyk1(self, out, inp, wait_for=[]):
        """ Apply the linear operator from image space to parameter space

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Args:
          out (PyOpenCL.Array):
            The complex parameter space data which is the result of the
            computation.
          inp (PyOpenCL.Array):
            The complex image space data which is used as input.
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        Returns:
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        return self.prg.update_Kyk1_imagespace(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, inp[0].data, inp[3].data, inp[1].data,
            np.int32(self.NScan),
            inp[4].data,
            np.int32(self.unknowns),
            self.DTYPE_real(self._dz),
            wait_for=inp[0].events + out.events + inp[1].events + wait_for)


class OperatorKspace(Operator):
    """ k-Space based Operator

    This class serves as linear operator between parameter and k-space.

    Use this operator if you want to perform complex parameter fitting from
    complex k-space data. The type of fft is defined through the NUFFT object.
    The NUFFT object can also be used for simple Cartesian FFTs.
    """

    def __init__(self, par, prg, DTYPE=np.complex64,
                 DTYPE_real=np.float32, trafo=True):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.tmp_result = clarray.empty(
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

    def fwd(self, out, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=self.tmp_result.events + inp[0].events + wait_for))
        return self.NUFFT.FFT(
            out,
            self.tmp_result,
            wait_for=wait_for +
            self.tmp_result.events)

    def fwdoop(self, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=self.tmp_result.events + inp[0].events + wait_for))
        tmp_sino = clarray.empty(
            self.queue,
            (self.NScan, self.NC, self.NSlice, self.Nproj, self.N),
            self.DTYPE, "C")
        tmp_sino.add_event(
            self.NUFFT.FFT(tmp_sino, self.tmp_result))
        return tmp_sino

    def adj(self, out, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.NUFFT.FFTH(
                self.tmp_result, inp[0], wait_for=wait_for + inp[0].events))
        return self.prg.operator_ad(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + self.tmp_result.events + out.events)

    def adjoop(self, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.NUFFT.FFTH(
                self.tmp_result, inp[0], wait_for=wait_for + inp[0].events))
        out = clarray.empty(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            dtype=self.DTYPE)
        self.prg.operator_ad(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + self.tmp_result.events + out.events).wait()
        return out

    def adjKyk1(self, out, inp, wait_for=[]):
        """ Apply the linear operator from parameter space to k-space

        This method fully implements the combined linear operator
        fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Args:
          out (PyOpenCL.Array):
            The complex parameter space data which is used as input.
          inp (PyOpenCL.Array):
            The complex parameter space data which is used as input.
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        Returns:
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        self.tmp_result.add_event(
            self.NUFFT.FFTH(
                self.tmp_result, inp[0], wait_for=wait_for + inp[0].events))
        return self.prg.update_Kyk1(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result.data, inp[2].data,
            inp[3].data, inp[1].data, np.int32(self.NC),
            np.int32(self.NScan),
            inp[4].data,
            np.int32(self.unknowns), self.DTYPE_real(self._dz),
            wait_for=(self.tmp_result.events +
                      out.events + inp[1].events + wait_for))


class OperatorKspaceFieldMap(Operator):
    """ k-Space based Operator

    This class serves as linear operator between parameter and k-space.

    Use this operator if you want to perform complex parameter fitting from
    complex k-space data. The type of fft is defined through the NUFFT object.
    The NUFFT object can also be used for simple Cartesian FFTs.
    """

    def __init__(self, par, prg, DTYPE=np.complex64,
                 DTYPE_real=np.float32, trafo=True):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.tmp_result = clarray.empty(
            self.queue, (self.NScan, self.NC,
                         self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")
        self.tmp_result2 = clarray.empty(
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
                                  fft_dim=[2],
                                  DTYPE=DTYPE,
                                  DTYPE_real=DTYPE_real)

        self.phase_map = clarray.to_device(self.queue,
                                           par["phase_map"])
        self.phase_mapadj = clarray.to_device(
            self.queue,
            np.require(np.transpose(
                np.conj(par["phase_map"]), (0, 1, 3, 2)),
              requirements='C'))

    def fwd(self, out, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=self.tmp_result.events + inp[0].events + wait_for))

        self.tmp_result2.add_event(self.prg.addfield(
                            self.queue,
                            (self.NSlice, self.dimY, self.dimX),
                            None,
                            self.tmp_result2.data,
                            self.tmp_result.data,
                            self.phase_map.data,
                            np.int32(self.NC),
                            np.int32(self.NScan),
                            wait_for=self.tmp_result.events+out.events))
        import ipdb
        ipdb.set_trace()

        return self.NUFFT.FFT(
            out,
            self.tmp_result2,
            wait_for=wait_for +
            self.tmp_result2.events)

    def fwdoop(self, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=self.tmp_result.events + inp[0].events + wait_for))
        tmp_sino = clarray.empty(
            self.queue,
            (self.NScan, self.NC, self.NSlice, self.Nproj, self.N),
            self.DTYPE, "C")

        self.prg.addfield(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result2.data,
                self.tmp_result.data,
                self.phase_map.data,
                np.int32(self.NC),
                np.int32(self.NScan)).wait()
        self.NUFFT.FFT(
            tmp_sino,
            self.tmp_result2,
            wait_for=wait_for +
            self.tmp_result2.events).wait()
        return tmp_sino

    def adj(self, out, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.NUFFT.FFTH(
                self.tmp_result,
                inp[0],
                wait_for=wait_for + inp[0].events + self.tmp_result.events))
        self.tmp_result2.add_event(
            self.prg.addfield(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result2.data,
                self.tmp_result.data,
                self.phase_mapadj.data,
                np.int32(self.NC),
                np.int32(self.NScan),
                wait_for=self.tmp_result.events+inp[0].events))
        return self.prg.operator_ad(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result2.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + self.tmp_result2.events + out.events)

    def adjoop(self, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.NUFFT.FFTH(
                self.tmp_result,
                inp[0],
                wait_for=wait_for + inp[0].events + self.tmp_result.events))
        self.tmp_result2.add_event(
            self.prg.addfield(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_resut2.data,
                self.tmp_result.data,
                self.phase_mapadj.data,
                np.int32(self.NC),
                np.int32(self.NScan),
                wait_for=self.tmp_result.events+inp[0].events))
        out = clarray.empty(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            dtype=self.DTYPE)
        self.prg.operator_ad(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result2.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + self.tmp_result2.events + out.events).wait()
        return out

    def adjKyk1(self, out, inp, wait_for=[]):
        """ Apply the linear operator from parameter space to k-space

        This method fully implements the combined linear operator
        fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Args:
          out (PyOpenCL.Array):
            The complex parameter space data which is used as input.
          inp (PyOpenCL.Array):
            The complex parameter space data which is used as input.
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        Returns:
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        self.tmp_result.add_event(
            self.NUFFT.FFTH(
                self.tmp_result,
                inp[0],
                wait_for=wait_for + inp[0].events + self.tmp_result.events))
        import ipdb
        ipdb.set_trace()
        self.tmp_result2.add_event(
            self.prg.addfield(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result2.data,
                self.tmp_result.data,
                self.phase_mapadj.data,
                np.int32(self.NC),
                np.int32(self.NScan),
                wait_for=self.tmp_result2.events+self.tmp_result.events))
        return self.prg.update_Kyk1(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result2.data, inp[2].data,
            inp[3].data, inp[1].data, np.int32(self.NC),
            np.int32(self.NScan),
            inp[4].data,
            np.int32(self.unknowns), self.DTYPE_real(self._dz),
            wait_for=(self.tmp_result2.events +
                      out.events + inp[1].events + wait_for))


class OperatorKspaceSMS(Operator):
    """ k-Space based Operator for SMS reconstruction

    This class serves as linear operator between parameter and k-space.
    It implements simultaneous-multi-slice (SMS) reconstruction.

    Use this operator if you want to perform complex parameter fitting from
    complex k-space data measured with SMS. Currently only Cartesian FFTs are
    supported.

    Attributes:
      packs (int):
        Number of SMS packs.
    """

    def __init__(self, par, prg, DTYPE=np.complex64,
                 DTYPE_real=np.float32, trafo=False):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.packs = par["packs"]*par["numofpacks"]
        self.tmp_result = clarray.empty(
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
                                  SMS=True,
                                  DTYPE=DTYPE,
                                  DTYPE_real=DTYPE_real)

    def fwd(self, out, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=self.tmp_result.events + inp[0].events + wait_for))
        return self.NUFFT.FFT(
            out,
            self.tmp_result,
            wait_for=wait_for +
            self.tmp_result.events)

    def fwdoop(self, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=self.tmp_result.events + inp[0].events + wait_for))
        tmp_sino = clarray.empty(
            self.queue,
            (self.NScan, self.NC, self.packs, self.Nproj, self.N),
            self.DTYPE, "C")
        tmp_sino.add_event(
            self.NUFFT.FFT(tmp_sino, self.tmp_result))
        return tmp_sino

    def adj(self, out, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.NUFFT.FFTH(
                self.tmp_result, inp[0], wait_for=wait_for + inp[0].events))
        return self.prg.operator_ad(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + self.tmp_result.events + out.events)

    def adjoop(self, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.NUFFT.FFTH(
                self.tmp_result, inp[0], wait_for=wait_for + inp[0].events))
        out = clarray.empty(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            dtype=self.DTYPE)
        self.prg.operator_ad(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + self.tmp_result.events + out.events).wait()
        return out

    def adjKyk1(self, out, inp, wait_for=[]):
        """ Apply the linear operator from parameter space to k-space

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Args:
          out (PyOpenCL.Array):
            The complex parameter space data which is used as input.
          inp (PyOpenCL.Array):
            The complex parameter space data which is used as input.
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        Returns:
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        self.tmp_result.add_event(
            self.NUFFT.FFTH(
                self.tmp_result, inp[0], wait_for=wait_for + inp[0].events))
        return self.prg.update_Kyk1(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result.data, inp[2].data,
            inp[3].data, inp[1].data, np.int32(self.NC),
            np.int32(self.NScan),
            inp[4].data,
            np.int32(self.unknowns), self.DTYPE_real(self._dz),
            wait_for=(self.tmp_result.events +
                      out.events + inp[1].events + wait_for))


class OperatorKspaceSMSFieldMap(Operator):
    """ k-Space based Operator for SMS reconstruction

    This class serves as linear operator between parameter and k-space.
    It implements simultaneous-multi-slice (SMS) reconstruction.

    Use this operator if you want to perform complex parameter fitting from
    complex k-space data measured with SMS. Currently only Cartesian FFTs are
    supported.

    Attributes:
      packs (int):
        Number of SMS packs.
    """

    def __init__(self, par, prg, DTYPE=np.complex64,
                 DTYPE_real=np.float32, trafo=False):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.packs = par["packs"]*par["numofpacks"]
        self.tmp_result = clarray.empty(
            self.queue, (self.NScan, self.NC,
                         self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")
        if not trafo:
            self.Nproj = self.dimY
            self.N = self.dimX
        self.NUFFTx = CLnuFFT.create(self.ctx,
                                   self.queue,
                                   par,
                                   radial=trafo,
                                   SMS=True,
                                   fft_dim=1,
                                   DTYPE=DTYPE,
                                   DTYPE_real=DTYPE_real)
        self.NUFFTy = CLnuFFT.create(self.ctx,
                                   self.queue,
                                   par,
                                   radial=trafo,
                                   fft_dim=2,
                                   DTYPE=DTYPE,
                                   DTYPE_real=DTYPE_real)
        self.NUFFTy.mask = clarray.to_device(self.queue,
                                             np.ones_like(par["mask"]))
        self.phase_map = clarray.to_device(self.queue, par["phase_map"])

    def fwd(self, out, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=self.tmp_result.events + inp[0].events + wait_for))
        self.NUFFTy.FFT(
            self.tmp_result,
            self.tmp_result,
            wait_for=self.tmp_result.events).wait()
        self.tmp_result = self.tmp_result*self.phase_map
        return self.NUFFT.FFTx(
            out,
            self.tmp_result,
            wait_for=wait_for +
            self.tmp_result.events)

    def fwdoop(self, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result.data, inp[0].data,
                inp[1].data,
                inp[2].data, np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=self.tmp_result.events + inp[0].events + wait_for))
        tmp_sino = clarray.empty(
            self.queue,
            (self.NScan, self.NC, self.packs, self.Nproj, self.N),
            self.DTYPE, "C")

        self.NUFFTy.FFT(
            self.tmp_result,
            self.tmp_result,
            wait_for=self.tmp_result.events).wait()
        self.tmp_result = self.tmp_result*self.phase_map
        self.NUFFT.FFTx(
            tmp_sino,
            self.tmp_result,
            wait_for=wait_for +
            self.tmp_result.events).wait()
        return tmp_sino

    def adj(self, out, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.NUFFTx.FFTH(
                self.tmp_result,
                inp[0], wait_for=wait_for + inp[0].events)).wait()
        self.tmp_result = self.tmp_result*self.phase_map.conj()
        self.NUFFTy.FFT(
            self.tmp_result,
            self.tmp_result,
            wait_for=self.tmp_result.events).wait()
        return self.prg.operator_ad(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + self.tmp_result.events + out.events)

    def adjoop(self, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.NUFFTx.FFTH(
                self.tmp_result, inp[0], wait_for=wait_for + inp[0].events))
        self.tmp_result = self.tmp_result*self.phase_map.conj()
        self.NUFFTy.FFT(
            self.tmp_result,
            self.tmp_result,
            wait_for=self.tmp_result.events).wait()
        out = clarray.empty(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            dtype=self.DTYPE)
        self.prg.operator_ad(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result.data, inp[1].data,
            inp[2].data, np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + self.tmp_result.events + out.events).wait()
        return out

    def adjKyk1(self, out, inp, wait_for=[]):
        """ Apply the linear operator from parameter space to k-space

        This method fully implements the combined linear operator
        fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Args:
          out (PyOpenCL.Array):
            The complex parameter space data which is used as input.
          inp (PyOpenCL.Array):
            The complex parameter space data which is used as input.
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.
        Returns:
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        self.tmp_result.add_event(
            self.NUFFTx.FFTH(
                self.tmp_result,
                inp[0], wait_for=wait_for + inp[0].events)).wait()
        self.tmp_result = self.tmp_result*self.phase_map.conj()
        self.NUFFTy.FFT(
            self.tmp_result,
            self.tmp_result,
            wait_for=self.tmp_result.events).wait()
        return self.prg.update_Kyk1(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result.data, inp[2].data,
            inp[3].data, inp[1].data, np.int32(self.NC),
            np.int32(self.NScan),
            inp[4].data,
            np.int32(self.unknowns), self.DTYPE_real(self._dz),
            wait_for=(self.tmp_result.events +
                      out.events + inp[1].events + wait_for))


class OperatorImagespaceStreamed(Operator):
    """ The streamed version of the Imagespace based Operator

    This class serves as linear operator between parameter and imagespace.
    All calculations are performed in a streamed fashion.

    Use this operator if you want to perform complex parameter fitting from
    complex image space data without the need of performing FFTs.
    In contrast to non-streaming classes no out of place operations
    are implemented.

    Attributes:
      overlap (int):
        Number of slices that overlap between adjacent blocks.
      par_slices (int):
        Number of slices per streamed block
      fwdstr (PyQMRI.Stream):
        The streaming object to perform the forward evaluation
      adjstr (PyQMRI.Stream):
        The streaming object to perform the adjoint evaluation
    """

    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        par["overlap"] = 1
        self.overlap = par["overlap"]
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

    def fwd(self, out, inp, wait_for=[]):
        self.fwdstr.eval(out, inp)

    def fwdoop(self, inp, wait_for=[]):
        tmp_result = np.zeros(self.data_shape, dtype=self.DTYPE)
        self.fwdstr.eval([tmp_result], inp)
        return tmp_result

    def adj(self, out, inp, wait_for=[]):
        self.adjstr.eval(out, inp)

    def adjoop(self, inp, wait_for=[]):
        tmp_result = np.zeros(self.unknown_shape, dtype=self.DTYPE)
        self.adjstr.eval([tmp_result], inp)
        return tmp_result

    def adjKyk1(self, out, inp):
        """ Apply the linear operator from parameter space to image space

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Args:
          out (Numpy.Array):
            The complex parameter space data which is used as input.
          inp (Numpy.Array):
            The complex parameter space data which is used as input.
        """
        self.adjstrKyk1.eval(out, inp)

    def _fwdstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=[]):
        return (self.prg[idx].operator_fwd_imagespace(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[2].data,
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=outp.events+inp[0].events+inp[2].events+wait_for))

    def _adjstreamedKyk1(self, outp, inp, par=None, idx=0, idxq=0,
                         bound_cond=0, wait_for=[]):
        return self.prg[idx].update_Kyk1_imagespace(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[3].data,
            inp[1].data,
            np.int32(self.NScan),
            par[0][idx].data, np.int32(self.unknowns),
            np.int32(bound_cond), self.DTYPE_real(self._dz),
            wait_for=(outp.events+inp[0].events+inp[1].events +
                      inp[3].events+wait_for))

    def _adjstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=[]):
        return self.prg[idx].operator_ad_imagespace(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[2].data,
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=(outp.events+inp[0].events +
                      inp[2].events+wait_for))


class OperatorKspaceStreamed(Operator):
    """ The streamed version of the k-space based Operator

    This class serves as linear operator between parameter and k-space.
    All calculations are performed in a streamed fashion.

    Use this operator if you want to perform complex parameter fitting from
    complex k-space data without the need of performing FFTs.
    In contrast to non-streaming classes no out of place operations
    are implemented.

    Attributes:
      overlap (int):
        Number of slices that overlap between adjacent blocks.
      par_slices (int):
        Number of slices per streamed block
      fwdstr (PyQMRI.Stream):
        The streaming object to perform the forward evaluation
      adjstr (PyQMRI.Stream):
        The streaming object to perform the adjoint evaluation
      NUFFT (list of PyQMRI.transforms.PyOpenCLnuFFT):
        A list of NUFFT objects. One for each context.
      FTstr (PyQMRI.Stream):
        A streamed version of the used (non-uniform) FFT, applied forward.
    """

    def __init__(self, par, prg,
                 DTYPE=np.complex64, DTYPE_real=np.float32, trafo=True):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.overlap = par["overlap"]
        self.par_slices = par["par_slices"]
        if not trafo:
            self.Nproj = self.dimY
            self.N = self.dimX
        for j in range(self.num_dev):
            for i in range(2):
                self.tmp_result.append(
                    clarray.empty(
                        self.queue[4*j+i],
                        (self.par_slices+self.overlap, self.NScan,
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
            [self.FT],
            [self.data_shape],
            [[trans_shape]])

    def fwd(self, out, inp, wait_for=[]):
        self.fwdstr.eval(out, inp)

    def fwdoop(self, inp, wait_for=[]):
        tmp_result = np.zeros(self.data_shape, dtype=self.DTYPE)
        self.fwdstr.eval([tmp_result], inp)
        return tmp_result

    def adj(self, out, inp, wait_for=[]):
        self.adjstr.eval(out, inp)

    def adjoop(self, inp, wait_for=[]):
        tmp_result = np.zeros(self.unknown_shape, dtype=self.DTYPE)
        self.adjstr.eval([tmp_result], inp)
        return tmp_result

    def adjKyk1(self, out, inp):
        """ Apply the linear operator from parameter space to k-space

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Args:
          out (Numpy.Array):
            The complex parameter space data which is used as input.
          inp (Numpy.Array):
            The complex parameter space data which is used as input.
        """
        self.adjstrKyk1.eval(out, inp)

    def _fwdstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=[]):
        self.tmp_result[2*idx+idxq].add_event(self.prg[idx].operator_fwd(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            self.tmp_result[2*idx+idxq].data, inp[0].data,
            inp[1].data,
            inp[2].data,
            np.int32(self.NC),
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=(self.tmp_result[2*idx+idxq].events +
                      inp[0].events+wait_for)))
        return self.NUFFT[2*idx+idxq].FFT(
            outp, self.tmp_result[2*idx+idxq],
            wait_for=outp.events+wait_for+self.tmp_result[2*idx+idxq].events)

    def _adjstreamedKyk1(self, outp, inp, par=None, idx=0, idxq=0,
                         bound_cond=0, wait_for=[]):
        self.tmp_result[2*idx+idxq].add_event(
            self.NUFFT[2*idx+idxq].FFTH(
                self.tmp_result[2*idx+idxq], inp[0],
                wait_for=(wait_for+inp[0].events +
                          self.tmp_result[2*idx+idxq].events)))
        return self.prg[idx].update_Kyk1(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            outp.data, self.tmp_result[2*idx+idxq].data,
            inp[2].data,
            inp[3].data,
            inp[1].data, np.int32(self.NC), np.int32(self.NScan),
            par[0][idx].data, np.int32(self.unknowns),
            np.int32(bound_cond), self.DTYPE_real(self._dz),
            wait_for=(
                self.tmp_result[2*idx+idxq].events +
                outp.events+inp[1].events +
                inp[2].events + inp[3].events + wait_for))

    def _adjstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=[]):
        self.tmp_result[2*idx+idxq].add_event(
            self.NUFFT[2*idx+idxq].FFTH(
                self.tmp_result[2*idx+idxq], inp[0],
                wait_for=(wait_for+inp[0].events +
                          self.tmp_result[2*idx+idxq].events)))
        return self.prg[idx].operator_ad(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            outp.data, self.tmp_result[2*idx+idxq].data,
            inp[1].data,
            inp[2].data,
            np.int32(self.NC),
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=(self.tmp_result[2*idx+idxq].events +
                      inp[1].events+inp[2].events+wait_for))

    def FT(self, outp, inp, par=None, idx=0, idxq=0,
           bound_cond=0, wait_for=[]):
        return self.NUFFT[2*idx+idxq].FFT(outp, inp[0])


class OperatorKspaceSMSStreamed(Operator):
    """ The streamed version of the k-space based SMS Operator


    This class serves as linear operator between parameter and k-space.
    It implements simultaneous-multi-slice (SMS) reconstruction.

    All calculations are performed in a streamed fashion.

    Use this operator if you want to perform complex parameter fitting from
    complex k-space data measured with SMS. Currently only Cartesian FFTs are
    supported.

    Attributes:
      overlap (int):
        Number of slices that overlap between adjacent blocks.
      par_slices (int):
        Number of slices per streamed block
      fwdstr (PyQMRI.Stream):
        The streaming object to perform the forward evaluation
      adjstr (PyQMRI.Stream):
        The streaming object to perform the adjoint evaluation
      NUFFT (list of PyQMRI.transforms.PyOpenCLnuFFT):
        A list of NUFFT objects. One for each context.
      FTstr (PyQMRI.Stream):
        A streamed version of the used (non-uniform) FFT, applied forward.
      FTHstr (PyQMRI.Stream):
        A streamed version of the used (non-uniform) FFT, applied adjoint.
      updateKyk1SMSstreamed
    """

    def __init__(self, par, prg, DTYPE=np.complex64,
                 DTYPE_real=np.float32, trafo=True):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.overlap = par["overlap"]
        self.par_slices = par["par_slices"]
        self.packs = par["packs"]*par["numofpacks"]
        if not trafo:
            self.Nproj = self.dimY
            self.N = self.dimX

        for j in range(self.num_dev):
            for i in range(2):
                self.tmp_result.append(
                    clarray.empty(
                        self.queue[4*j+i],
                        (self.par_slices+self.overlap, self.NScan,
                         self.NC, self.dimY, self.dimX),
                        self.DTYPE, "C"))
                self.NUFFT.append(
                    CLnuFFT.create(self.ctx[j],
                                 self.queue[4*j+i], par,
                                 radial=trafo,
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

    def fwd(self, out, inp, wait_for=[]):
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

    def fwdoop(self, inp, wait_for=[]):
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

    def adj(self, out, inp, par=None):
        """ Apply the linear operator from measurement space to parameter space
        If streamed operations are used the PyOpenCL.Arrays are replaced
        by Numpy.Array
        Args:
          out (Numpy.Array):
            The complex parameter space data which is used as input.
          inp (Numpy.Array):
            The complex parameter space data which is used as input.
          par (list):
            List of constant parameters for the streamed function.
        Returns:
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


    def adjoop(self, inp, wait_for=[]):
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

    def adjKyk1(self, out, inp, par=None):
        """ Apply the linear operator from parameter space to k-space

        This method fully implements the combined linear operator
        consisting of the data part as well as the TGV regularization part.

        Args:
          out (Numpy.Array):
            The complex parameter space data which is used as input.
          inp (Numpy.Array):
            The complex parameter space data which is used as input.
          par (list):
            List of constant parameters for the streamed function.
        Returns:
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
            [[self._tmp_Kyk1]+[inp[0][1]]+[inp[0][-1]]], par)

    def _fwdstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=[]):
        return self.prg[idx].operator_fwd(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[1].data,
            inp[2].data,
            np.int32(self.NC),
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=(outp.events+inp[0].events +
                      inp[1].events+inp[2].events+wait_for))

    def _adjstreamed(self, outp, inp, par=None, idx=0, idxq=0,
                     bound_cond=0, wait_for=[]):
        return self.prg[idx].operator_ad(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
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
            bound_cond=0, wait_for=[]):
        return self.NUFFT[2*idx+idxq].FFT(outp, inp[0])

    def _FTH(self, outp, inp, par=None, idx=0, idxq=0,
             bound_cond=0, wait_for=[]):
        return self.NUFFT[2*idx+idxq].FFTH(outp, inp[0])

    def _updateKyk1SMS(self, outp, inp, par=None, idx=0, idxq=0,
                       bound_cond=0, wait_for=[]):
        return self.prg[idx].update_Kyk1SMS(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
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
                           posofnorm=[]):
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
    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self._ratio = clarray.to_device(
            self.queue,
            (par["weights"]).astype(
                     dtype=self.DTYPE_real))
        self._weights = par["weights"]

    def fwd(self, out, inp, wait_for=[]):
        return self.prg.gradient(
            self.queue, inp.shape[1:], None, out.data, inp.data,
            np.int32(self.unknowns),
            self._ratio.data, self.DTYPE_real(self._dz),
            wait_for=out.events + inp.events + wait_for)

    def fwdoop(self, inp, wait_for=[]):
        tmp_result = clarray.empty(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 4),
            self.DTYPE, "C")
        tmp_result.add_event(self.prg.gradient(
            self.queue, inp.shape[1:], None, tmp_result.data, inp.data,
            np.int32(self.unknowns),
            self._ratio.data, self.DTYPE_real(self._dz),
            wait_for=tmp_result.events + inp.events + wait_for))
        return tmp_result

    def adj(self, out, inp, wait_for=[]):
        return self.prg.divergence(
            self.queue, inp.shape[1:-1], None, out.data, inp.data,
            np.int32(self.unknowns), self._ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=out.events + inp.events + wait_for)

    def adjoop(self, inp, wait_for=[]):
        tmp_result = clarray.empty(
            self.queue, (self.unknowns, self.NSlice, self.dimY, self.dimX),
            self.DTYPE, "C")
        tmp_result.add_event(self.prg.divergence(
            self.queue, inp.shape[1:-1], None, tmp_result.data, inp.data,
            np.int32(self.unknowns), self._ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=tmp_result.events + inp.events + wait_for))
        return tmp_result

    def updateRatio(self, inp):
        x = clarray.to_device(self.queue, inp)
        grad = clarray.to_device(
            self.queue, np.zeros(x.shape + (4,),
                                 dtype=self.DTYPE))
        self._ratio = clarray.to_device(
            self.queue,
            (self._weights *
             np.ones(
                 self.unknowns)).astype(
                     self.DTYPE_real))
        grad.add_event(
            self.fwd(
                grad,
                x,
                wait_for=grad.events +
                x.events))
        x = x.get()
        grad = grad.get()
        scale = np.reshape(
            x, (self.unknowns,
                self.NSlice * self.dimY * self.dimX))
        grad = np.reshape(
            grad, (self.unknowns,
                   self.NSlice *
                   self.dimY *
                   self.dimX * 4))
        print("Total Norm of grad x pre: ", np.sum(np.abs(grad)))
        gradnorm = np.sum(np.abs(grad), axis=-1)
        print("Norm of grad x pre: ", gradnorm)
        gradnorm /= np.sum(gradnorm)/self.unknowns
        scale = 1/gradnorm
        scale[~np.isfinite(scale)] = 1
        # print("Scale: ", scale)
        for j in range(x.shape[0])[:self.unknowns_TGV]:
            self._ratio[j] = scale[j] * self._weights[j]
#        sum_scale = np.sqrt(np.sum(np.abs(
#            scale[self.unknowns_TGV:])**2/(1000)))
        for j in range(x.shape[0])[self.unknowns_TGV:]:
            self._ratio[j] = scale[j] * self._weights[j]
        # print("Ratio: ", self._ratio)
        x = clarray.to_device(self.queue, x)
        grad = clarray.to_device(
            self.queue, np.zeros(x.shape + (4,),
                                 dtype=self.DTYPE))
        grad.add_event(
            self.fwd(
                grad,
                x,
                wait_for=grad.events +
                x.events))
        x = x.get()
        grad = grad.get()
        grad = np.reshape(
            grad, (self.unknowns,
                   self.NSlice *
                   self.dimY *
                   self.dimX * 4))
        print("Norm of grad x post: ",  np.sum(np.abs(grad), axis=-1))
        print("Total Norm of grad x post: ",  np.sum(np.abs(grad)))


class OperatorFiniteSymGradient(Operator):
    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self._ratio = clarray.to_device(
            self.queue,
            (par["weights"]).astype(
                     dtype=self.DTYPE_real))
        self._weights = par["weights"]

    def fwd(self, out, inp, wait_for=[]):
        return self.prg.sym_grad(
            self.queue, inp.shape[1:-1], None, out.data, inp.data,
            np.int32(self.unknowns_TGV),
            self._ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=out.events + inp.events + wait_for)

    def fwdoop(self, inp, wait_for=[]):
        tmp_result = clarray.empty(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 8),
            self.DTYPE, "C")
        tmp_result.add_event(self.prg.sym_grad(
            self.queue, inp.shape[1:-1], None, tmp_result.data, inp.data,
            np.int32(self.unknowns_TGV),
            self._ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=tmp_result.events + inp.events + wait_for))
        return tmp_result

    def adj(self, out, inp, wait_for=[]):
        return self.prg.sym_divergence(
            self.queue, inp.shape[1:-1], None, out.data, inp.data,
            np.int32(self.unknowns_TGV),
            self._ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=out.events + inp.events + wait_for)

    def adjoop(self, inp, wait_for=[]):
        tmp_result = clarray.empty(
            self.queue, (self.unknowns,
                         self.NSlice, self.dimY, self.dimX, 4),
            self.DTYPE, "C")
        tmp_result.add_event(self.prg.sym_divergence(
            self.queue, inp.shape[1:-1], None, tmp_result.data, inp.data,
            np.int32(self.unknowns_TGV),
            self._ratio.data,
            self.DTYPE_real(self._dz),
            wait_for=tmp_result.events + inp.events + wait_for))
        return tmp_result

    def updateRatio(self, inp):
        for i in range(len(inp)):
            self._ratio[i] = inp[i]

class OperatorFiniteGradientStreamed(Operator):
    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)

        self._weights = par["weights"]
        par["overlap"] = 1
        self.overlap = par["overlap"]
        self.par_slices = par["par_slices"]

        self._ratio = []
        for j in range(self.num_dev):
            self._ratio.append(
                clarray.to_device(
                    self.queue[4*j],
                    (par["weights"]).astype(
                        dtype=self.DTYPE_real)))

        self.unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        self.grad_shape = self.unknown_shape + (4,)

        self._stream_grad = self._defineoperator(
            [self._grad],
            [self.grad_shape],
            [[self.unknown_shape]])

        self._stream_div = self._defineoperator(
            [self._div],
            [self.unknown_shape],
            [[self.grad_shape]],
            reverse_dir=True)

    def fwd(self, out, inp, wait_for=[]):
        self._stream_grad.eval(out, inp)

    def fwdoop(self, inp, wait_for=[]):
        out = np.zeros(self.grad_shape, dtype=self.DTYPE)
        self._stream_grad.eval([out], inp)
        return out

    def adj(self, out, inp, wait_for=[]):
        self._stream_div.eval(out, inp)

    def adjoop(self, inp, wait_for=[]):
        out = np.zeros(self.unknown_shape, dtype=self.DTYPE)
        self._stream_div.eval([out], inp)
        return out

    def updateRatio(self, inp):
        x = np.require(np.swapaxes(inp, 0, 1), requirements='C')
        grad = np.zeros(x.shape + (4,), dtype=self.DTYPE)
        for i in range(self.num_dev):
            for j in range(x.shape[1]):
                self._ratio[i][j] = 1
        self.fwd([grad], [[x]])
        grad = np.require(np.swapaxes(grad, 0, 1),
                          requirements='C')

        scale = np.reshape(
            inp, (self.unknowns,
                  self.NSlice * self.dimY * self.dimX))
        grad = np.reshape(
            grad, (self.unknowns,
                   self.NSlice *
                   self.dimY *
                   self.dimX * 4))

        print("Total Norm of grad x pre: ", np.sum(np.abs(grad)))
        gradnorm = np.sum(np.abs(grad), axis=-1)
        print("Norm of grad x pre: ", gradnorm)
        gradnorm /= np.sum(gradnorm)/self.unknowns
        scale = 1/gradnorm
        scale[~np.isfinite(scale)] = 1

        for i in range(self.num_dev):
            for j in range(inp.shape[0])[:self.unknowns_TGV]:
                self._ratio[i][j] = scale[j] * self._weights[j]
        for i in range(self.num_dev):
            for j in range(inp.shape[0])[self.unknowns_TGV:]:
                self._ratio[i][j] = scale[j] * self._weights[j]

        grad = np.zeros(x.shape + (4,), dtype=self.DTYPE)
        self.fwd([grad], [[x]])
        grad = np.require(np.swapaxes(grad, 0, 1),
                          requirements='C')

        grad = np.reshape(
            grad, (self.unknowns,
                   self.NSlice *
                   self.dimY *
                   self.dimX * 4))
        print("Norm of grad x post: ",  np.sum(np.abs(grad), axis=-1))
        print("Total Norm of grad x post: ",  np.sum(np.abs(grad)))


    def _grad(self, outp, inp, par=None, idx=0, idxq=0,
              bound_cond=0, wait_for=[]):
        return self.prg[idx].gradient(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX),
            None, outp.data, inp[0].data,
            np.int32(self.unknowns),
            self._ratio[idx].data, self.DTYPE_real(self._dz),
            wait_for=outp.events + inp[0].events + wait_for)

    def _div(self, outp, inp, par=None, idx=0, idxq=0,
             bound_cond=0, wait_for=[]):
        return self.prg[idx].divergence(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data, np.int32(self.unknowns),
            self._ratio[idx].data, np.int32(bound_cond),
            self.DTYPE_real(self._dz),
            wait_for=outp.events + inp[0].events + wait_for)


class OperatorFiniteSymGradientStreamed(Operator):
    def __init__(self, par, prg, DTYPE=np.complex64, DTYPE_real=np.float32):
        super().__init__(par, prg, DTYPE, DTYPE_real)

        par["overlap"] = 1
        self.overlap = par["overlap"]
        self.par_slices = par["par_slices"]

        unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        self.grad_shape = unknown_shape + (4,)
        self.symgrad_shape = unknown_shape + (8,)

        self._ratio = []
        for j in range(self.num_dev):
            self._ratio.append(
                clarray.to_device(
                    self.queue[4*j],
                    (par["weights"]).astype(
                        dtype=self.DTYPE_real)))

        self._stream_symgrad = self._defineoperator(
            [self._symgrad],
            [self.symgrad_shape],
            [[self.grad_shape]],
            reverse_dir=True)

        self._stream_symdiv = self._defineoperator(
            [self._symdiv],
            [self.grad_shape],
            [[self.symgrad_shape]])

    def fwd(self, out, inp, wait_for=[]):
        self._stream_symgrad.eval(out, inp)

    def fwdoop(self, inp, wait_for=[]):
        out = np.zeros(self.symgrad_shape, dtype=self.DTYPE)
        self._stream_symgrad.eval([out], inp)
        return out

    def adj(self, out, inp, wait_for=[]):
        self._stream_symdiv.eval(out, inp)

    def adjoop(self, inp, wait_for=[]):
        out = np.zeros(self.grad_shape, dtype=self.DTYPE)
        self._stream_symdiv.eval([out], inp)
        return out

    def _symgrad(self, outp, inp, par=None, idx=0, idxq=0,
                 bound_cond=0, wait_for=[]):
        return self.prg[idx].sym_grad(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data, np.int32(self.unknowns),
            self._ratio[idx].data,
            self.DTYPE_real(self._dz),
            wait_for=outp.events + inp[0].events + wait_for)

    def _symdiv(self, outp, inp, par=None, idx=0, idxq=0,
                bound_cond=0, wait_for=[]):
        return self.prg[idx].sym_divergence(
            self.queue[4*idx+idxq],
            (self.overlap+self.par_slices, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            np.int32(self.unknowns),
            self._ratio[idx].data,
            np.int32(bound_cond),
            self.DTYPE_real(self._dz),
            wait_for=outp.events + inp[0].events + wait_for)

    def updateRatio(self, inp):
        for j in range(self.num_dev):
            for i in range(len(inp[j])):
                self._ratio[j][i] = inp[j][i]
