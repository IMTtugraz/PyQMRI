#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2019 Oliver Maier

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

import pyopencl as cl
import pyopencl.array as clarray
from gpyfft.fft import FFT
from pkg_resources import resource_filename
from pyqmri._helper_fun._calckbkernel import calckbkernel


class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build("-cl-mad-enable -cl-fast-relaxed-math")
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
            self.__dict__[kernel.function_name] = kernel


class PyOpenCLNUFFT:
    def __init__(
            self,
            ctx,
            queue,
            par,
            kwidth=5,
            overgridfactor=2,
            fft_dim=(
                1,
                2),
            klength=200,
            DTYPE=np.complex64,
            DTYPE_real=np.float32,
            radial=True):
        self.radial = radial
        self.DTYPE = DTYPE
        self.DTYPE_real = DTYPE_real
        self.fft_shape = (par["NScan"] *
                          par["NC"] *
                          (par["par_slices"] +
                           par["overlap"]), par["N"], par["N"])
        self.traj = par["traj"]
        self.dcf = par["dcf"]
        self.ctx = ctx
        self.queue = queue
        if self.radial:
            (self.kerneltable, self.kerneltable_FT, self.u) = calckbkernel(
                kwidth, overgridfactor, par["N"], klength)
            self.kernelpoints = self.kerneltable.size
            self.overgridfactor = overgridfactor
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
                0:int(self.fft_shape[0] / par["NScan"]), ...],
                           out_array=self.tmp_fft_array[
                               0:int(self.fft_shape[0] / par["NScan"]), ...],
                           axes=fft_dim)
            self.gridsize = par["N"]
            self.fwd_NUFFT = self.NUFFT
            self.adj_NUFFT = self.NUFFTH
        else:
            self.fwd_NUFFT = self.FFT
            self.adj_NUFFT = self.FFTH
            self.fft_scale = DTYPE_real(
                np.sqrt(np.prod(self.fft_shape[fft_dim[0]:])))
            self.tmp_fft_array = (
                clarray.zeros(
                    self.queue,
                    self.fft_shape,
                    dtype=DTYPE))
            self.fft = []
            self.fft_shape = self.fft_shape
            self.par_fft = int(self.fft_shape[0] / par["NScan"])
            self.mask = clarray.to_device(self.queue, par["mask"])
            self.fft = FFT(ctx, queue, self.tmp_fft_array[
                0:int(self.fft_shape[0] / par["NScan"]), ...],
                           out_array=self.tmp_fft_array[
                               0:int(self.fft_shape[0] / par["NScan"]), ...],
                           axes=fft_dim)
        print('Using single precission')
        self.prg = Program(
            self.ctx,
            open(
              resource_filename(
                'pyqmri',
                'kernels/OpenCL_gridding_slicefirst_single.c')).read())

    def __del__(self):
        if self.radial:
            del self.traj
            del self.dcf
            del self.tmp_fft_array
            del self.cl_kerneltable
            del self.fft
            del self.deapo_cl
        else:
            del self.tmp_fft_array
            del self.fft

    def NUFFTH(self, sg, s, wait_for=[]):
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
        for j in range(s.shape[1]):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * s.shape[0] * s.shape[2]:
                        (j + 1) * s.shape[0] * s.shape[2], ...],
                    result=self.tmp_fft_array[
                        j * s.shape[0] * s.shape[2]:
                        (j + 1) * s.shape[0] * s.shape[2], ...],
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
                                  self.DTYPE_real(self.overgridfactor),
                                  wait_for=(wait_for + sg.events + s.events +
                                            self.tmp_fft_array.events))

    def NUFFT(self, s, sg, wait_for=[]):
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
                self.DTYPE_real(self.overgridfactor),
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
        for j in range(s.shape[1]):
            self.tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self.tmp_fft_array[
                        j * s.shape[0] * s.shape[2]:
                        (j + 1) * s.shape[0] * s.shape[2], ...],
                    result=self.tmp_fft_array[
                        j * s.shape[0] * s.shape[2]:
                        (j + 1) * s.shape[0] * s.shape[2], ...],
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

    def FFTH(self, sg, s, wait_for=[]):

        self.tmp_fft_array.add_event(
            self.prg.copy(
                self.queue,
                (s.size,
                 ),
                None,
                self.tmp_fft_array.data,
                s.data,
                self.DTYPE_real(1)))

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
                self.gueue,
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
            self.prg.copy(
                self.queue,
                (s.size,
                 ),
                None,
                s.data,
                self.tmp_fft_array.data,
                self.DTYPE_real(1)))
