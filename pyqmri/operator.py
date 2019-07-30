#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import pyopencl.array as clarray
import numpy as np
from pyqmri.transforms.pyopencl_nufft import PyOpenCLFFT as CLFFT
import pyqmri.streaming as streaming
DTYPE = np.complex64


class Operator(ABC):
    def __init__(self, par, prg):
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
        ...

    @abstractmethod
    def adj(self, out, inp, wait_for=[]):
        ...

    @abstractmethod
    def fwdoop(self, out, inp, wait_for=[]):
        ...

    @abstractmethod
    def adjoop(self, out, inp, wait_for=[]):
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
            return self.prg.update_Kyk1_imagespace(
                self.queue, (self.NSlice, self.dimY, self.dimX), None,
                out.data, inp[0].data, inp[3], inp[1].data,
                np.int32(self.NScan),
                inp[4].data,
                np.int32(self.unknowns),
                np.float32(self.dz),
                wait_for=inp[0].events + out.events + inp[1].events + wait_for)


class OperatorKspace(Operator):
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


class OperatorKspaceSMS(Operator):
    def __init__(self, par, prg, trafo=False):
        super().__init__(par, prg)
        self.queue = self.queue[0]
        self.ctx = self.ctx[0]
        self.packs = par["packs"]*par["numofpacks"]
        self.tmp_result = clarray.empty(
            self.queue, (self.NScan, self.NC,
                         self.NSlice, self.dimY, self.dimX),
            DTYPE, "C")
        self.NUFFT = CLFFT.create(self.ctx,
                                  self.queue,
                                  par,
                                  radial=trafo,
                                  SMS=True)

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
                    (self.NScan, self.NC, self.packs, self.Nproj, self.N),
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

    def adjKyk1(self, out, inp, wait_for=[]):
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

    def adjKyk1(self, out, inp, wait_for=[]):
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


class OperatorKspaceSMSStreamed(Operator):
    def __init__(self, par, prg, trafo=True):
        super().__init__(par, prg)
        par["overlap"] = 1
        self.overlap = par["overlap"]
        self.par_slices = 1
        self.packs = par["packs"]*par["numofpacks"]
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
                                 SMS=True,
                                 streamed=True))

        unknown_shape = (self.NSlice, self.unknowns, self.dimY, self.dimX)
        coil_shape = (self.NSlice, self.NC, self.dimY, self.dimX)
        model_grad_shape = (self.NSlice, self.unknowns,
                            self.NScan, self.dimY, self.dimX)
        data_shape = (self.NSlice, self.NScan, self.NC, self.dimY, self.dimX)
        data_shape_T = (self.NScan, self.NC, self.packs,
                        self.Nproj, self.N)
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
            [self.FT],
            [data_shape_T],
            [[trans_shape_T]])
        self.FTHstr = self._defineoperatorSMS(
            [self.FTH],
            [trans_shape_T],
            [[data_shape_T]])

        self.tmp_fft1 = np.zeros((self.NSlice, self.NScan, self.NC,
                                 self.dimY, self.dimX),
                                 dtype=DTYPE)
        self.tmp_fft2 = np.zeros((self.NScan, self.NC, self.NSlice,
                                  self.dimY, self.dimX),
                                 dtype=DTYPE)
        self.tmp_transformed = np.zeros((self.NScan, self.NC, self.packs,
                                         self.dimY, self.dimX),
                                        dtype=DTYPE)
        self.tmp_Kyk1 = np.zeros(unknown_shape,
                                 dtype=DTYPE)

        self.update_Kyk1SMS_streamed = self._defineoperator(
            [self.update_Kyk1SMS],
            [unknown_shape],
            [[unknown_shape,
              grad_shape,
              unknown_shape]],
            reverse_dir=True,
            posofnorm=[True])

    def fwd(self, out, inp, wait_for=[]):
        raise NotImplementedError

    def fwdoop(self, inp, wait_for=[]):
        self.fwdstr.eval([self.tmp_fft1], inp)
        self.tmp_fft2 = np.require(
            np.transpose(
                self.tmp_fft1, (1, 2, 0, 3, 4)),
            requirements='C')
        self.FTstr.eval(
            [self.tmp_transformed],
            [[self.tmp_fft2]])
        return np.require(
            np.transpose(
                self.tmp_transformed,
                self.dat_trans_axes),
            requirements='C')

    def adj(self, out, inp, par=[], wait_for=[]):
        self.tmp_transformed = np.require(
            np.transpose(
                inp[0][0], (1, 2, 0, 3, 4)),
            requirements='C')
        self.FTHstr.eval(
            [self.tmp_fft2],
            [[self.tmp_transformed]])
        self.tmp_fft1 = np.require(
            np.transpose(
                self.tmp_fft2, self.dat_trans_axes),
            requirements='C')
        self.adjstr.eval([self.tmp_Kyk1], [[self.tmp_fft1]+inp[0][2:-1]])
        self.update_Kyk1SMS_streamed.eval(
            out,
            [[self.tmp_Kyk1]+[inp[0][1]]+[inp[0][-1]]], par)

    def adjoop(self, inp, wait_for=[]):
        raise NotImplementedError

    def adjKyk1(self, out, inp, par=[], wait_for=[]):
        self.tmp_transformed = np.require(
            np.transpose(
                inp[0][0], (1, 2, 0, 3, 4)),
            requirements='C')
        self.FTHstr.eval(
            [self.tmp_fft2],
            [[self.tmp_transformed]])
        self.tmp_fft1 = np.require(
            np.transpose(
                self.tmp_fft2, self.dat_trans_axes),
            requirements='C')
        self.adjstr.eval([self.tmp_Kyk1], [[self.tmp_fft1]+inp[0][2:-1]])
        return self.update_Kyk1SMS_streamed.evalwithnorm(
            out,
            [[self.tmp_Kyk1]+[inp[0][1]]+[inp[0][-1]]], par)

    def _fwdstreamed(self, outp, inp, par=[], idx=0, idxq=0,
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

    def _adjstreamed(self, outp, inp, par=[], idx=0, idxq=0,
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

    def FT(self, outp, inp, par=[], idx=0, idxq=0,
           bound_cond=0, wait_for=[]):
        return self.NUFFT[2*idx+idxq].FFT(outp, inp[0])

    def FTH(self, outp, inp, par=[], idx=0, idxq=0,
            bound_cond=0, wait_for=[]):
        return self.NUFFT[2*idx+idxq].FFTH(outp, inp[0])

    def update_Kyk1SMS(self, outp, inp, par=[], idx=0, idxq=0,
                       bound_cond=0, wait_for=[]):
        return self.prg[idx].update_Kyk1SMS(
            self.queue[4*idx+idxq],
            (self.par_slices+self.overlap, self.dimY, self.dimX), None,
            outp.data, inp[0].data,
            inp[1].data,
            par[-1][idx].data, np.int32(self.unknowns),
            np.int32(bound_cond), np.float32(self.dz),
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
        return streaming.stream(
            functions,
            outp,
            inp,
            1,
            0,
            self.NScan,
            self.queue,
            self.num_dev,
            reverse_dir,
            posofnorm)
