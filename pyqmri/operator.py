#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This module holds the classes for different linear Operator.

Attribues:
  DTYPE (complex64):
    Complex working precission. Currently single precission only.
"""
from abc import ABC, abstractmethod
import pyopencl.array as clarray
import numpy as np
from pyqmri.transforms.pyopencl_nufft import PyOpenCLFFT as CLFFT
import pyqmri.streaming as streaming
DTYPE = np.complex64


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
      NUFFT (PyQMRI.transforms.PyOpenCLFFT):
        A PyOpenCLFFT object to perform forward and backword transformations
        from image to k-space and vice versa.
      prg (PyOpenCL.Program):
        The PyOpenCL program containing all compiled kernels.
    """
    def __init__(self, par, prg):
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
        self.dz = par["dz"]
        self.num_dev = len(par["num_dev"])
        self.tmp_result = []
        self.NUFFT = []
        self.prg = prg

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

    def _defineoperator(self,
                        functions,
                        outp,
                        inp,
                        reverse_dir=False,
                        posofnorm=[],
                        slices=None):
        if slices is None:
            slices = self.NSlice
        return streaming.stream(
            functions,
            outp,
            inp,
            self.par_slices,
            self.overlap,
            slices,
            self.queue,
            self.num_dev,
            reverse_dir,
            posofnorm)


class OperatorImagespace(Operator):
    """ Imagespace based Operator
    This class serves as linear operator between parameter and imagespace.

    Use this operator if you want to perform complex parameter fitting from
    complex image space data without the need of performing FFTs.
    """
    def __init__(self, par, prg):
        super().__init__(par, prg)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]

    def fwd(self, out, inp, wait_for=[]):
        return self.prg.operator_fwd_imagespace(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, inp[0].data, inp[2],
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=inp[0].events + out.events + wait_for)

    def fwdoop(self, inp, wait_for=[]):
        tmp_result = clarray.empty(
              self.queue, (self.NScan, self.NSlice, self.dimY, self.dimX),
              DTYPE, "C")
        tmp_result.add_event(self.prg.operator_fwd_imagespace(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            tmp_result.data, inp[0].data, inp[1],
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=inp[0].events + wait_for))
        return tmp_result

    def adj(self, out, inp, wait_for=[]):
        return self.prg.operator_ad_imagespace(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, inp[0].data, inp[2],
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + inp[0].events + out.events)

    def adjoop(self, inp, wait_for=[]):
        out = clarray.empty(
            self.queue, (self.unkowns, self.NSlice, self.dimY, self.dimX),
            dtype=DTYPE)
        self.prg.operator_ad_imagespace(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, inp[0].data, inp[1],
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
            out.data, inp[0].data, inp[3], inp[1].data,
            np.int32(self.NScan),
            inp[4].data,
            np.int32(self.unknowns),
            np.float32(self.dz),
            wait_for=inp[0].events + out.events + inp[1].events + wait_for)


class OperatorKspace(Operator):
    """ k-Space based Operator

    This class serves as linear operator between parameter and k-space.

    Use this operator if you want to perform complex parameter fitting from
    complex k-space data. The type of fft is defined through the NUFFT object.
    The NUFFT object can also be used for simple Cartesian FFTs.
    """
    def __init__(self, par, prg, trafo=True):
        super().__init__(par, prg)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.tmp_result = clarray.empty(
            self.queue, (self.NScan, self.NC,
                         self.NSlice, self.dimY, self.dimX),
            DTYPE, "C")
        self.NUFFT = CLFFT.create(self.ctx,
                                  self.queue,
                                  par,
                                  radial=trafo)

    def fwd(self, out, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.prg.operator_fwd(
                self.queue,
                (self.NSlice, self.dimY, self.dimX),
                None,
                self.tmp_result.data, inp[0].data,
                inp[1],
                inp[2], np.int32(self.NC),
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
                inp[1],
                inp[2], np.int32(self.NC),
                np.int32(self.NScan),
                np.int32(self.unknowns),
                wait_for=self.tmp_result.events + inp[0].events + wait_for))
        tmp_sino = clarray.empty(
            self.queue,
            (self.NScan, self.NC, self.NSlice, self.Nproj, self.N),
            DTYPE, "C")
        tmp_sino.add_event(
            self.NUFFT.FFT(tmp_sino, self.tmp_result))
        return tmp_sino

    def adj(self, out, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.NUFFT.FFTH(
                self.tmp_result, inp[0], wait_for=wait_for + inp[0].events))
        return self.prg.operator_ad(
            self.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result.data, inp[1],
            inp[2], np.int32(self.NC),
            np.int32(self.NScan),
            np.int32(self.unknowns),
            wait_for=wait_for + self.tmp_result.events + out.events)

    def adjoop(self, inp, wait_for=[]):
        self.tmp_result.add_event(
            self.NUFFT.FFTH(
                self.tmp_result, inp[0], wait_for=wait_for + inp[0].events))
        out = clarray.empty(
            self.queue, (self.unkowns, self.NSlice, self.dimY, self.dimX),
            dtype=DTYPE)
        self.prg.operator_ad(
            out.queue, (self.NSlice, self.dimY, self.dimX), None,
            out.data, self.tmp_result.data, inp[1],
            inp[2], np.int32(self.NC),
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
            out.data, self.tmp_result.data, inp[2],
            inp[3], inp[1].data, np.int32(self.NC),
            np.int32(self.NScan),
            inp[4].data,
            np.int32(self.unknowns), np.float32(self.dz),
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
    def __init__(self, par, prg):
        super().__init__(par, prg)
        par["overlap"] = 1
        self.overlap = par["overlap"]
        self.par_slices = par["par_slices"]
        unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        coil_shape = []
        model_grad_shape = (self.NSlice, self.unknowns,
                            self.NScan, self.dimY, self.dimX)
        data_shape = (self.NSlice, self.NScan, self.dimY, self.dimX)
        grad_shape = unknown_shape + (4,)

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
              grad_shape,
              coil_shape,
              model_grad_shape,
              unknown_shape]])

    def fwd(self, out, inp, wait_for=[]):
        self.fwdstr.eval(out, inp)

    def fwdoop(self, inp, wait_for=[]):
        tmp_result = np.array(
              self.queue, (self.NSlice, self.NScan, self.dimY, self.dimX),
              DTYPE, "C")
        self.fwdstr.eval([tmp_result], inp)
        return tmp_result

    def adj(self, out, inp, wait_for=[]):
        raise NotImplementedError

    def adjoop(self, inp, wait_for=[]):
        raise NotImplementedError

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
        self.adjstr.eval(out, inp)

    def _fwdstreamed(self, outp, inp, par=[], idx=0, idxq=0,
                     bound_cond=0, wait_for=[]):
        return (self.prg[idx].operator_fwd_imagespace(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[2].data,
            np.int32(self.NScan), np.int32(self.unknowns),
            wait_for=outp.events+inp[0].events+inp[2].events+wait_for))

    def _adjstreamed(self, outp, inp, par=[], idx=0, idxq=0,
                     bound_cond=0, wait_for=[]):
        return self.prg[idx].update_Kyk1_imagespace(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[3].data,
            inp[1].data,
            np.int32(self.NScan),
            par[-1][idx].data, np.int32(self.unknowns),
            np.int32(bound_cond), np.float32(self.dz),
            wait_for=(outp.events+inp[0].events+inp[1].events +
                      inp[3].events+wait_for))


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
      NUFFT (list of PyQMRI.transforms.PyOpenCLFFT):
        A list of NUFFT objects. One for each context.
      FTstr (PyQMRI.Stream):
        A streamed version of the used (non-uniform) FFT, applied forward.

    """
    def __init__(self, par, prg, trafo=True):
        super().__init__(par, prg)
        par["overlap"] = 1
        self.overlap = par["overlap"]
        self.par_slices = par["par_slices"]
        for j in range(self.num_dev):
            for i in range(2):
                self.tmp_result.append(
                    clarray.empty(
                        self.queue[4*j+i],
                        (self.par_slices+self.overlap, self.NScan,
                         self.NC, self.dimY, self.dimX),
                        DTYPE, "C"))
                self.NUFFT.append(
                    CLFFT.create(self.ctx[j],
                                 self.queue[4*j+i], par,
                                 radial=trafo,
                                 streamed=True))

        unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        coil_shape = (self.NSlice, self.NC, self.dimY, self.dimX)
        model_grad_shape = (self.NSlice, self.unknowns,
                            self.NScan, self.dimY, self.dimX)
        data_shape = (self.NSlice, self.NScan, self.NC, self.Nproj, self.N)
        trans_shape = (self.NSlice, self.NScan,
                       self.NC, self.dimY, self.dimX)
        grad_shape = unknown_shape + (4,)

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
              grad_shape,
              coil_shape,
              model_grad_shape,
              unknown_shape]])
        self.FTstr = self._defineoperator(
            [self.FT],
            [data_shape],
            [[trans_shape]])

    def fwd(self, out, inp, wait_for=[]):
        self.fwdstr.eval(out, inp)

    def fwdoop(self, inp, wait_for=[]):
        tmp_result = np.array(
              self.queue, (self.NSlice, self.NScan, self.dimY, self.dimX),
              DTYPE, "C")
        self.fwdstr.eval([tmp_result], inp)
        return tmp_result

    def adj(self, out, inp, wait_for=[]):
        raise NotImplementedError

    def adjoop(self, inp, wait_for=[]):
        raise NotImplementedError

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
        self.adjstr.eval(out, inp)

    def _fwdstreamed(self, outp, inp, par=[], idx=0, idxq=0,
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

    def _adjstreamed(self, outp, inp, par=[], idx=0, idxq=0,
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
            par[-1][idx].data, np.int32(self.unknowns),
            np.int32(bound_cond), np.float32(self.dz),
            wait_for=(
                self.tmp_result[2*idx+idxq].events +
                outp.events+inp[1].events +
                inp[2].events + inp[3].events + wait_for))

    def FT(self, outp, inp, par=[], idx=0, idxq=0,
           bound_cond=0, wait_for=[]):
        return self.NUFFT[2*idx+idxq].FFT(outp, inp[0])
