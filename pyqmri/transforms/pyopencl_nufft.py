#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
from gpyfft.fft import FFT
from pkg_resources import resource_filename
from pyqmri._helper_fun._calckbkernel import calckbkernel
from pyqmri._helper_fun import CLProgram as Program


class PyOpenCLFFT(object):
    def __init__(self, ctx, queue, DTYPE, DTYPE_real):
        super().__init__()
        self.DTYPE = DTYPE
        self.DTYPE_real = DTYPE_real
        self.ctx = ctx
        self.queue = queue

    @staticmethod
    def create(ctx,
               queue,
               par,
               kwidth=5,
               fft_dim=(
                   1,
                   2),
               klength=200,
               DTYPE=np.complex64,
               DTYPE_real=np.float32,
               radial=False,
               SMS=False,
               streamed=False):

        if not streamed:
            if radial is True and SMS is False:
                obj = PyOpenCLRadialNUFFT(
                    ctx,
                    queue,
                    par,
                    kwidth=5,
                    fft_dim=(
                        1,
                        2),
                    klength=200,
                    DTYPE=np.complex64,
                    DTYPE_real=np.float32)
            elif SMS is True and radial is False:
                obj = PyOpenCLSMSNUFFT(
                    ctx,
                    queue,
                    par,
                    fft_dim=(
                        1,
                        2),
                    DTYPE=np.complex64,
                    DTYPE_real=np.float32)
            elif SMS is False and radial is False:
                obj = PyOpenCLCartNUFFT(
                    ctx,
                    queue,
                    par,
                    fft_dim=(
                        1,
                        2),
                    DTYPE=np.complex64,
                    DTYPE_real=np.float32)
            else:
                raise AssertionError("Combination of Radial "
                                     "and SMS not allowed")
            if DTYPE == np.complex128:
                print('Using double precission')
                obj.prg = Program(
                    obj.ctx,
                    open(
                        resource_filename(
                            'pyqmri',
                            'kernels/OpenCL_gridding_double.c')).read())
            else:
                print('Using single precission')
                obj.prg = Program(
                    obj.ctx,
                    open(
                        resource_filename(
                            'pyqmri',
                            'kernels/OpenCL_gridding_single.c')).read())
        else:
            if radial is True and SMS is False:
                obj = PyOpenCLRadialNUFFTStreamed(
                    ctx,
                    queue,
                    par,
                    kwidth=5,
                    fft_dim=(
                        1,
                        2),
                    klength=200,
                    DTYPE=np.complex64,
                    DTYPE_real=np.float32)
            elif SMS is True and radial is False:
                obj = PyOpenCLSMSNUFFTStreamed(
                    ctx,
                    queue,
                    par,
                    fft_dim=(
                        1,
                        2),
                    DTYPE=np.complex64,
                    DTYPE_real=np.float32)
            elif SMS is False and radial is False:
                obj = PyOpenCLCartNUFFTStreamed(
                    ctx,
                    queue,
                    par,
                    fft_dim=(
                        1,
                        2),
                    DTYPE=np.complex64,
                    DTYPE_real=np.float32)
            else:
                raise AssertionError("Combination of Radial "
                                     "and SMS not allowed")
            if DTYPE == np.complex128:
                print('Using double precission')
                obj.prg = Program(
                    obj.ctx,
                    open(
                      resource_filename(
                        'pyqmri',
                        'kernels/OpenCL_gridding_slicefirst_double.c')).read())
            else:
                print('Using single precission')
                obj.prg = Program(
                    obj.ctx,
                    open(
                      resource_filename(
                        'pyqmri',
                        'kernels/OpenCL_gridding_slicefirst_single.c')).read())
        return obj


class PyOpenCLRadialNUFFT(PyOpenCLFFT):
    def __init__(
            self,
            ctx,
            queue,
            par,
            kwidth=5,
            fft_dim=(
                1,
                2),
            klength=200,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        super().__init__(ctx, queue, DTYPE, DTYPE_real)
        self.traj = par["traj"]
        self.dcf = par["dcf"]
        self.ogf = par["N"]/par["dimX"]
        self.fft_shape = (
            par["NScan"] *
            par["NC"] *
            par["NSlice"],
            int(par["dimY"]*self.ogf),
            int(par["dimX"]*self.ogf))
        (self.kerneltable, self.kerneltable_FT, self.u) = calckbkernel(
            kwidth, self.ogf, par["N"], klength)
        self.kernelpoints = self.kerneltable.size
        self.fft_scale = DTYPE_real(
            np.sqrt(np.prod(self.fft_shape[fft_dim[0]:])))
        self.deapo = 1 / self.kerneltable_FT.astype(DTYPE_real)
        self.kwidth = kwidth / 2
        self.cl_kerneltable = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.kerneltable.astype(DTYPE_real).data)
        self.deapo_cl = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.deapo.data)
        self.dcf = clarray.to_device(self.queue, self.dcf)
        self.traj = clarray.to_device(self.queue, self.traj)
        self.tmp_fft_array = (
            clarray.empty(
                self.queue,
                (self.fft_shape),
                dtype=DTYPE))
        self.check = np.ones(par["N"], dtype=DTYPE_real)
        self.check[1::2] = -1
        self.check = clarray.to_device(self.queue, self.check)
        self.par_fft = int(self.fft_shape[0] / par["NScan"])
        self.fft = FFT(ctx, queue, self.tmp_fft_array[
            0:int(self.fft_shape[0] / par["NScan"]), ...],
                       out_array=self.tmp_fft_array[
                           0:int(self.fft_shape[0] / par["NScan"]), ...],
                       axes=fft_dim)
        self.gridsize = par["N"]

    def __del__(self):
        del self.traj
        del self.dcf
        del self.tmp_fft_array
        del self.cl_kerneltable
        del self.deapo_cl
        del self.check
        del self.queue
        del self.ctx
        del self.prg
        del self.fft

    def FFTH(self, sg, s, wait_for=[]):
        # Zero tmp arrays
        self.tmp_fft_array.add_event(
            self.prg.zero_tmp(
                self.queue,
                (self.tmp_fft_array.size,
                 ),
                None,
                self.tmp_fft_array.data,
                wait_for=s.events +
                sg.events +
                self.tmp_fft_array.events +
                wait_for))
        # Grid k-space
        self.tmp_fft_array.add_event(
            self.prg.grid_lut(
                self.queue,
                (s.shape[0], s.shape[1] * s.shape[2],
                 s.shape[-2] * self.gridsize),
                None,
                self.tmp_fft_array.data,
                s.data,
                self.traj.data,
                np.int32(self.gridsize),
                self.DTYPE_real(self.kwidth / self.gridsize),
                self.dcf.data,
                self.cl_kerneltable,
                np.int32(self.kernelpoints),
                wait_for=(wait_for + sg.events +
                          s.events + self.tmp_fft_array.events)))
        # FFT
        self.tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self.tmp_fft_array.data,
                self.check.data))
        for j in range(s.shape[0]):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    result=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    forward=False)[0])
        self.tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self.tmp_fft_array.data,
                self.check.data))
        return self.prg.deapo_adj(
                  self.queue,
                  (sg.shape[0] * sg.shape[1] *
                   sg.shape[2], sg.shape[3], sg.shape[4]),
                  None,
                  sg.data,
                  self.tmp_fft_array.data,
                  self.deapo_cl,
                  np.int32(self.tmp_fft_array.shape[-1]),
                  self.DTYPE_real(self.fft_scale),
                  self.DTYPE_real(self.ogf),
                  wait_for=(wait_for + sg.events + s.events +
                            self.tmp_fft_array.events))

    def FFT(self, s, sg, wait_for=[]):
        # Zero tmp arrays
        self.tmp_fft_array.add_event(
            self.prg.zero_tmp(
                self.queue,
                (self.tmp_fft_array.size,
                 ),
                None,
                self.tmp_fft_array.data,
                wait_for=s.events +
                sg.events +
                self.tmp_fft_array.events +
                wait_for))
        # Deapodization and Scaling
        self.tmp_fft_array.add_event(
            self.prg.deapo_fwd(
                self.queue,
                (sg.shape[0] * sg.shape[1] * sg.shape[2],
                 sg.shape[3], sg.shape[4]),
                None,
                self.tmp_fft_array.data,
                sg.data,
                self.deapo_cl,
                np.int32(self.tmp_fft_array.shape[-1]),
                self.DTYPE_real(1 / self.fft_scale),
                self.DTYPE_real(self.ogf),
                wait_for=wait_for + sg.events + self.tmp_fft_array.events))
        # FFT
        self.tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self.tmp_fft_array.data,
                self.check.data))
        for j in range(s.shape[0]):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    result=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    forward=True)[0])
        self.tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self.tmp_fft_array.data,
                self.check.data))
        # Resample on Spoke
        return self.prg.invgrid_lut(
            self.queue,
            (s.shape[0], s.shape[1] * s.shape[2], s.shape[-2] * self.gridsize),
            None,
            s.data,
            self.tmp_fft_array.data,
            self.traj.data,
            np.int32(self.gridsize),
            self.DTYPE_real(self.kwidth / self.gridsize),
            self.dcf.data,
            self.cl_kerneltable,
            np.int32(self.kernelpoints),
            wait_for=s.events + wait_for + self.tmp_fft_array.events)


class PyOpenCLCartNUFFT(PyOpenCLFFT):
    def __init__(
            self,
            ctx,
            queue,
            par,
            fft_dim=(
                1,
                2),
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        super().__init__(ctx, queue, DTYPE, DTYPE_real)
        self.fft_shape = (
            par["NScan"] *
            par["NC"] *
            par["NSlice"],
            par["dimY"],
            par["dimX"])
        self.fft_scale = DTYPE_real(
            np.sqrt(np.prod(self.fft_shape[fft_dim[0]:])))
        self.tmp_fft_array = (
            clarray.empty(
                self.queue,
                self.fft_shape,
                dtype=DTYPE))
        self.par_fft = int(self.fft_shape[0] / par["NScan"])
        self.mask = clarray.to_device(self.queue, par["mask"])
        self.fft = FFT(ctx, queue, self.tmp_fft_array[
            0:int(self.fft_shape[0] / par["NScan"]), ...],
                       out_array=self.tmp_fft_array[
                           0:int(self.fft_shape[0] / par["NScan"]), ...],
                       axes=fft_dim)

    def __del__(self):
        del self.tmp_fft_array
        del self.queue
        del self.ctx
        del self.prg
        del self.mask
        del self.fft

    def FFTH(self, sg, s, wait_for=[]):

        self.tmp_fft_array.add_event(
            self.prg.maskingcpy(
                self.queue,
                (self.tmp_fft_array.shape),
                None,
                self.tmp_fft_array.data,
                s.data,
                self.mask.data,
                wait_for=s.events))

        for j in range(s.shape[0]):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    result=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    forward=False)[0])
        return (
            self.prg.copy(
                self.queue,
                (sg.size,
                 ),
                None,
                sg.data,
                self.tmp_fft_array.data,
                self.DTYPE_real(
                    self.fft_scale)))

    def FFT(self, s, sg, wait_for=[]):
        self.tmp_fft_array.add_event(
            self.prg.copy(
                self.queue,
                (sg.size,
                 ),
                None,
                self.tmp_fft_array.data,
                sg.data,
                self.DTYPE_real(
                    1 /
                    self.fft_scale)))

        for j in range(s.shape[0]):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    result=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    forward=True)[0])

        return (
            self.prg.maskingcpy(
                self.queue,
                (self.tmp_fft_array.shape),
                None,
                s.data,
                self.tmp_fft_array.data,
                self.mask.data,
                wait_for=s.events+self.tmp_fft_array.events))


class PyOpenCLSMSNUFFT(PyOpenCLFFT):
    def __init__(
            self,
            ctx,
            queue,
            par,
            fft_dim=(
                1,
                2),
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        super().__init__(ctx, queue, DTYPE, DTYPE_real)
        self.fft_shape = (
            par["NScan"] *
            par["NC"] *
            par["NSlice"],
            par["dimY"],
            par["dimX"])

        self.packs = int(par["packs"])
        self.MB = int(par["MB"])
        self.shift = clarray.to_device(
            self.queue, par["shift"].astype(np.int32))

        self.fft_shape = (
            int(self.fft_shape[0] / self.MB),
            self.fft_shape[1], self.fft_shape[2])

        self.fft_scale = DTYPE_real(
            np.sqrt(np.prod(self.fft_shape[fft_dim[0]:])))
        self.tmp_fft_array = (
            clarray.empty(
                self.queue,
                self.fft_shape,
                dtype=DTYPE))
        self.par_fft = int(self.fft_shape[0] / par["NScan"])
        self.mask = clarray.to_device(self.queue, par["mask"])
        self.fft = FFT(ctx, queue, self.tmp_fft_array[
            0:int(self.fft_shape[0] / par["NScan"]), ...],
                       out_array=self.tmp_fft_array[
                           0:int(self.fft_shape[0] / par["NScan"]), ...],
                       axes=fft_dim)

    def __del__(self):
        del self.tmp_fft_array
        del self.queue
        del self.ctx
        del self.prg
        del self.fft

    def FFTH(self, sg, s, wait_for=[]):
        self.tmp_fft_array.add_event(
            self.prg.maskingcpy(
                self.queue,
                (self.tmp_fft_array.shape),
                None,
                self.tmp_fft_array.data,
                s.data,
                self.mask.data,
                wait_for=s.events))
        for j in range(s.shape[0]):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    result=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    forward=False)[0])
        return (self.prg.copy_SMS_adj(self.queue,
                                      (sg.shape[0] * sg.shape[1],
                                       sg.shape[-2],
                                          sg.shape[-1]),
                                      None,
                                      sg.data,
                                      self.tmp_fft_array.data,
                                      self.shift.data,
                                      np.int32(self.packs),
                                      np.int32(self.MB),
                                      self.DTYPE_real(self.fft_scale),
                                      np.int32(sg.shape[2]/self.packs/self.MB),
                                      wait_for=self.tmp_fft_array.events))

    def FFT(self, s, sg, wait_for=[]):
        self.tmp_fft_array.add_event(
            self.prg.copy_SMS_fwd(
                self.queue,
                (sg.shape[0] * sg.shape[1], sg.shape[-2], sg.shape[-1]),
                None,
                self.tmp_fft_array.data,
                sg.data,
                self.shift.data,
                np.int32(self.packs),
                np.int32(self.MB),
                self.DTYPE_real(1 / self.fft_scale),
                np.int32(sg.shape[2]/self.packs/self.MB),
                wait_for=self.tmp_fft_array.events+sg.events))

        for j in range(s.shape[0]):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    result=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    forward=True)[0])

        return (
            self.prg.maskingcpy(
                self.queue,
                (self.tmp_fft_array.shape),
                None,
                s.data,
                self.tmp_fft_array.data,
                self.mask.data,
                wait_for=s.events+self.tmp_fft_array.events))


class PyOpenCLRadialNUFFTStreamed(PyOpenCLFFT):
    def __init__(
            self,
            ctx,
            queue,
            par,
            kwidth=5,
            fft_dim=(
                1,
                2),
            klength=200,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):

        super().__init__(ctx, queue, DTYPE, DTYPE_real)
        self.NScan = par["NScan"]
        self.traj = par["traj"]
        self.dcf = par["dcf"]
        self.ogf = par["N"]/par["dimX"]
        self.fft_shape = (par["NScan"] *
                          par["NC"] *
                          (par["par_slices"] +
                           par["overlap"]),
                          int(par["dimY"]*self.ogf),
                          int(par["dimX"]*self.ogf))
        (self.kerneltable, self.kerneltable_FT, self.u) = calckbkernel(
            kwidth, self.ogf, par["N"], klength)
        self.kernelpoints = self.kerneltable.size

        self.fft_scale = DTYPE_real(self.fft_shape[-1])
        self.deapo = 1 / self.kerneltable_FT.astype(DTYPE_real)
        self.kwidth = kwidth / 2
        self.cl_kerneltable = cl.Buffer(
            self.queue.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.kerneltable.astype(DTYPE_real).data)
        self.deapo_cl = cl.Buffer(
            self.queue.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.deapo.data)
        self.dcf = clarray.to_device(self.queue, self.dcf)
        self.traj = clarray.to_device(self.queue, self.traj)
        self.tmp_fft_array = (
            clarray.empty(
                self.queue,
                self.fft_shape,
                dtype=DTYPE))
        self.check = np.ones(par["N"], dtype=DTYPE_real)
        self.check[1::2] = -1
        self.check = clarray.to_device(self.queue, self.check)
        self.fft = FFT(ctx, queue, self.tmp_fft_array[
            0:int(self.fft_shape[0] / self.NScan), ...],
                       out_array=self.tmp_fft_array[
                           0:int(self.fft_shape[0] / self.NScan), ...],
                       axes=fft_dim)
        self.gridsize = par["N"]
        self.par_fft = int(self.fft_shape[0] / self.NScan)

    def __del__(self):
        del self.traj
        del self.dcf
        del self.tmp_fft_array
        del self.cl_kerneltable
        del self.deapo_cl
        del self.check
        del self.queue
        del self.ctx
        del self.prg
        del self.fft

    def FFTH(self, sg, s, wait_for=[]):
        # Zero tmp arrays
        self.tmp_fft_array.add_event(
            self.prg.zero_tmp(
                self.queue,
                (self.tmp_fft_array.size,
                 ),
                None,
                self.tmp_fft_array.data,
                wait_for=self.tmp_fft_array.events +
                wait_for))
        # Grid k-space
        self.tmp_fft_array.add_event(
            self.prg.grid_lut(
                 self.queue,
                 (s.shape[0], s.shape[1] * s.shape[2],
                  s.shape[-2] * self.gridsize),
                 None,
                 self.tmp_fft_array.data,
                 s.data,
                 self.traj.data,
                 np.int32(self.gridsize),
                 np.int32(sg.shape[2]),
                 self.DTYPE_real(self.kwidth / self.gridsize),
                 self.dcf.data,
                 self.cl_kerneltable,
                 np.int32(self.kernelpoints),
                 wait_for=(wait_for + sg.events + s.events +
                           self.tmp_fft_array.events)))
        # FFT
        self.tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self.tmp_fft_array.data,
                self.check.data))
        for j in range(int(self.fft_shape[0] / self.par_fft)):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * self.par_fft:
                        (j + 1) * self.par_fft, ...],
                    result=self.tmp_fft_array[
                        j * self.par_fft:
                        (j + 1) * self.par_fft, ...],
                    forward=False)[0])
        self.tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self.tmp_fft_array.data,
                self.check.data))
        # Deapodization and Scaling
        return self.prg.deapo_adj(self.queue,
                                  (sg.shape[0] * sg.shape[1] * sg.shape[2],
                                   sg.shape[3], sg.shape[4]),
                                  None,
                                  sg.data,
                                  self.tmp_fft_array.data,
                                  self.deapo_cl,
                                  np.int32(self.tmp_fft_array.shape[-1]),
                                  self.DTYPE_real(self.fft_scale),
                                  self.DTYPE_real(self.ogf),
                                  wait_for=(wait_for + sg.events + s.events +
                                            self.tmp_fft_array.events))

    def FFT(self, s, sg, wait_for=[]):
        # Zero tmp arrays
        self.tmp_fft_array.add_event(
            self.prg.zero_tmp(
                self.queue,
                (self.tmp_fft_array.size,
                 ),
                None,
                self.tmp_fft_array.data,
                wait_for=self.tmp_fft_array.events +
                wait_for))
        # Deapodization and Scaling
        self.tmp_fft_array.add_event(
            self.prg.deapo_fwd(
                self.queue,
                (sg.shape[0] * sg.shape[1] * sg.shape[2], sg.shape[3],
                 sg.shape[4]),
                None,
                self.tmp_fft_array.data,
                sg.data,
                self.deapo_cl,
                np.int32(self.tmp_fft_array.shape[-1]),
                self.DTYPE_real(1 / self.fft_scale),
                self.DTYPE_real(self.ogf),
                wait_for=wait_for + sg.events + self.tmp_fft_array.events))
        # FFT
        self.tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self.tmp_fft_array.data,
                self.check.data))
        for j in range(int(self.fft_shape[0] / self.par_fft)):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * self.par_fft:
                        (j + 1) * self.par_fft, ...],
                    result=self.tmp_fft_array[
                        j * self.par_fft:
                        (j + 1) * self.par_fft, ...],
                    forward=True)[0])
        self.tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self.tmp_fft_array.data,
                self.check.data))
        # Resample on Spoke

        return self.prg.invgrid_lut(
            self.queue,
            (s.shape[0], s.shape[1] * s.shape[2],
             s.shape[-2] * self.gridsize),
            None,
            s.data,
            self.tmp_fft_array.data,
            self.traj.data,
            np.int32(self.gridsize),
            np.int32(s.shape[2]),
            self.DTYPE_real(self.kwidth / self.gridsize),
            self.dcf.data,
            self.cl_kerneltable,
            np.int32(self.kernelpoints),
            wait_for=(s.events + wait_for + self.tmp_fft_array.events +
                      sg.events))


class PyOpenCLCartNUFFTStreamed(PyOpenCLFFT):
    def __init__(
            self,
            ctx,
            queue,
            par,
            kwidth=5,
            fft_dim=(
                1,
                2),
            klength=200,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):

        super().__init__(ctx, queue, DTYPE, DTYPE_real)
        self.NScan = par["NScan"]

        self.fft_shape = (par["NScan"] *
                          par["NC"] *
                          (par["par_slices"] +
                           par["overlap"]), par["dimY"], par["dimX"])
        self.par_fft = int(self.fft_shape[0] / self.NScan)
        self.fft_scale = DTYPE_real(
            np.sqrt(np.prod(self.fft_shape[fft_dim[0]:])))
        self.tmp_fft_array = (
            clarray.zeros(
                self.queue,
                self.fft_shape,
                dtype=DTYPE))
        self.fft = []
        self.mask = clarray.to_device(self.queue, par["mask"])
        self.fft = FFT(ctx, queue, self.tmp_fft_array[
            0:self.par_fft, ...],
                       out_array=self.tmp_fft_array[
                           0:self.par_fft, ...],
                       axes=fft_dim)

    def __del__(self):
        del self.tmp_fft_array
        del self.queue
        del self.ctx
        del self.prg
        del self.mask
        del self.fft

    def FFTH(self, sg, s, wait_for=[]):
        self.tmp_fft_array.add_event(
            self.prg.maskingcpy(
                self.queue,
                (self.tmp_fft_array.shape),
                None,
                self.tmp_fft_array.data,
                s.data,
                self.mask.data,
                wait_for=s.events))

        for j in range(int(self.fft_shape[0] / self.par_fft)):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * self.par_fft:
                        (j + 1) * self.par_fft, ...],
                    result=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    forward=False)[0])
        # Scaling
        return (
            self.prg.copy(
                self.queue,
                (sg.size,
                 ),
                None,
                sg.data,
                self.tmp_fft_array.data,
                self.DTYPE_real(
                    self.fft_scale)))

    def FFT(self, s, sg, wait_for=[]):

        self.tmp_fft_array.add_event(
            self.prg.copy(
                self.queue,
                (sg.size,
                 ),
                None,
                self.tmp_fft_array.data,
                sg.data,
                self.DTYPE_real(
                    1 /
                    self.fft_scale)))
        for j in range(int(self.fft_shape[0] / self.par_fft)):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    result=self.tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    forward=True)[0])

        return (
            self.prg.maskingcpy(
                self.queue,
                (self.tmp_fft_array.shape),
                None,
                s.data,
                self.tmp_fft_array.data,
                self.mask.data,
                wait_for=s.events+self.tmp_fft_array.events))


class PyOpenCLSMSNUFFTStreamed(PyOpenCLFFT):
    def __init__(
            self,
            ctx,
            queue,
            par,
            kwidth=5,
            fft_dim=(
                1,
                2),
            klength=200,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):

        super().__init__(ctx, queue, DTYPE, DTYPE_real)
        self.NScan = par["NScan"]

        self.MB = int(par["MB"])
        self.fft_shape = (par["NSlice"] *
                          par["NC"], par["dimY"], par["dimX"])
        self.packs = int(par["packs"])
        self.MB = int(par["MB"])
        self.shift = clarray.to_device(
            self.queue, par["shift"].astype(np.int32))
        self.fft_shape = (
            int(self.fft_shape[0] / self.MB),
            self.fft_shape[1], self.fft_shape[2])
        self.par_fft = int(self.fft_shape[0])
        self.scanind = 0

        self.fft_scale = DTYPE_real(
            np.sqrt(np.prod(self.fft_shape[fft_dim[0]:])))
        self.tmp_fft_array = (
            clarray.zeros(
                self.queue,
                self.fft_shape,
                dtype=DTYPE))
        self.fft = []
        self.mask = clarray.to_device(self.queue, par["mask"])
        self.fft = FFT(ctx, queue, self.tmp_fft_array[
            0:self.par_fft, ...],
                       out_array=self.tmp_fft_array[
                           0:self.par_fft, ...],
                       axes=fft_dim)

    def __del__(self):
        del self.tmp_fft_array
        del self.queue
        del self.ctx
        del self.prg
        del self.fft

    def FFTH(self, sg, s, wait_for=[]):
        self.tmp_fft_array.add_event(
            self.prg.maskingcpy(
                self.queue,
                (self.tmp_fft_array.shape),
                None,
                self.tmp_fft_array.data,
                s.data,
                self.mask.data,
                wait_for=s.events))
        self.tmp_fft_array.add_event(
            self.fft.enqueue_arrays(
                data=self.tmp_fft_array,
                result=self.tmp_fft_array,
                forward=False)[0])
        return (self.prg.copy_SMS_adj(self.queue,
                                      (sg.shape[0] * sg.shape[1],
                                       sg.shape[-2],
                                          sg.shape[-1]),
                                      None,
                                      sg.data,
                                      self.tmp_fft_array.data,
                                      self.shift.data,
                                      np.int32(self.packs),
                                      np.int32(self.MB),
                                      self.DTYPE_real(self.fft_scale),
                                      np.int32(sg.shape[2]/self.packs/self.MB),
                                      wait_for=self.tmp_fft_array.events))

    def FFT(self, s, sg, wait_for=[]):
        self.tmp_fft_array.add_event(
            self.prg.copy_SMS_fwd(
                self.queue,
                (sg.shape[0] * sg.shape[1], sg.shape[-2], sg.shape[-1]),
                None,
                self.tmp_fft_array.data,
                sg.data,
                self.shift.data,
                np.int32(self.packs),
                np.int32(self.MB),
                self.DTYPE_real(1 / self.fft_scale),
                np.int32(sg.shape[2]/self.packs/self.MB),
                wait_for=self.tmp_fft_array.events+sg.events))

        self.tmp_fft_array.add_event(
            self.fft.enqueue_arrays(
                data=self.tmp_fft_array,
                result=self.tmp_fft_array,
                forward=True)[0])
        return (
            self.prg.maskingcpy(
                self.queue,
                (self.tmp_fft_array.shape),
                None,
                s.data,
                self.tmp_fft_array.data,
                self.mask.data,
                wait_for=s.events+self.tmp_fft_array.events))
