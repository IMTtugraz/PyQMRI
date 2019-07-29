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
DTYPE = np.complex64


class stream:
    def __init__(self,
                 fun,
                 outp_shape,
                 inp_shape,
                 par_slices,
                 overlap,
                 NSlice,
                 queue,
                 num_dev,
                 reverse=False,
                 lhs=[]):

        self.fun = fun
        self.num_dev = num_dev
        self.slices = par_slices
        self.overlap = overlap
        self.queue = queue
        self.reverse = reverse
        self.NSlice = NSlice
        self.num_fun = len(self.fun)

        self.lhs = lhs
        self.at_end = False
        self._resetindex()

        self.inp = []
        self.outp = []
        self._alloctmparrays(inp_shape, outp_shape)

    def __del__(self):
#        for j in range(len(self.inp)-1, 0, -1):
#            for i in range(len(self.inp[0])-1, 0, -1):
#                for k in range(len(self.inp[0][0])-1, 0, -1):
#                    del self.inp[j][i][k]
#        for j in range(len(self.outp)-1, 0, -1):
#            for i in range(len(self.outp[0])-1, 0, -1):
#                del self.outp[j][i]
        del self.queue

    def _alloctmparrays(self,
                        inp_shape,
                        outp_shape):
        block_size = self.slices+self.overlap
        for j in range(self.num_fun):
            self.inp.append([])
            for i in range(2*self.num_dev):
                self.inp[j].append([])
                for k in range(len(inp_shape[j])):
                    if not inp_shape[j][k] == []:
                        self.inp[j][i].append(
                            clarray.empty(
                                self.queue[4*int(i/2)],
                                ((block_size, )+inp_shape[j][k][1:]),
                                dtype=DTYPE))
                    else:
                        self.inp[j][i].append([])

        for j in range(self.num_fun):
            self.outp.append([])
            for i in range(2*self.num_dev):
                self.outp[j].append(
                    clarray.empty(
                        self.queue[4*int(i/2)],
                        ((block_size, )+outp_shape[j][1:]),
                        dtype=DTYPE))

    def __add__(self, other):
        for j in range(other.num_fun):
            self.fun.append(other.fun[j])
            self.inp.append(other.inp[j])
            self.outp.append(other.outp[j])
        self.num_fun += other.num_fun
        return self

    def _getindtodev(self):
        if self.reverse:
            tmp_return = slice(self.idx_todev_start,
                               self.idx_todev_stop)
            self.idx_todev_start -= self.slices
            self.idx_todev_stop -= self.slices
            if self.idx_todev_start < 0:
                self.idx_todev_start = 0
                self.idx_todev_stop = self.slices + self.overlap
            return tmp_return
        else:
            tmp_return = slice(self.idx_todev_start,
                               self.idx_todev_stop)
            self.idx_todev_start += self.slices
            self.idx_todev_stop += self.slices
            if self.idx_todev_stop > self.NSlice:
                self.idx_todev_start = (self.NSlice -
                                        self.slices -
                                        self.overlap)
                self.idx_todev_stop = self.NSlice
            return tmp_return

    def _getindtohost(self):
        if self.reverse:
            tmp_return = slice(self.idx_tohost_start,
                               self.idx_tohost_stop)
            self.idx_tohost_start -= self.slices
            self.idx_tohost_stop -= self.slices
            self.at_end = False
            if self.idx_tohost_start < 0:
                self.at_end = True
                self.idx_tohost_start = 0
                self.idx_tohost_stop = self.slices + self.overlap
            return tmp_return
        else:
            tmp_return = slice(self.idx_tohost_start,
                               self.idx_tohost_stop)
            self.idx_tohost_start += self.slices
            self.idx_tohost_stop += self.slices
            self.at_end = False
            if self.idx_tohost_stop > self.NSlice:
                self.at_end = True
                self.idx_tohost_start = (self.NSlice -
                                         self.slices -
                                         self.overlap)
                self.idx_tohost_stop = self.NSlice
            return tmp_return

    def eval(self, outp, inp, par=[]):
        # Reset Array Index
        self._resetindex()
        # Warmup Queue 1
        self._streamtodevice(inp, 0)
        self._startcomputation(par, bound_cond=1, odd=0)
        # Warmup Queue 1
        self._streamtodevice(inp, 1)
        self._startcomputation(par, bound_cond=0, odd=1)
        # Start Streaming
        islice = 2*self.slices*self.num_dev
        odd = True
        while islice < self.NSlice:
            # Collect Previous Block
            odd = not odd
            self._streamtohost(outp, odd)
            # Stream new Block
            self._streamtodevice(inp, odd)
            # Start Computation
            self._startcomputation(par, bound_cond=0, odd=odd)
            islice += self.num_dev*self.slices

        # Collect last block
        if odd:
            self._streamtohost(outp, 0)
            self._streamtohost(outp, 1)
        elif (self.NSlice/(self.slices*self.num_dev) <
              2*(self.slices*self.num_dev)):
            self._streamtohost(outp, 1)
            self._streamtohost(outp, 0)
        else:
            self._streamtohost(outp, 0)
        # Wait for all Queues to finish
        for i in range(self.num_dev):
            self.queue[4*i].finish()
            self.queue[4*i+1].finish()
            self.queue[4*i+2].finish()
            self.queue[4*i+3].finish()

    def evalwithnorm(self, outp, inp, par=[]):
        # Reset Array Index
        self._resetindex()
        rhs = 0
        lhs = 0
        # Warmup Queue 1
        self._streamtodevice(inp, 0)
        self._startcomputation(par, bound_cond=1, odd=0)
        # Warmup Queue 2
        self._streamtodevice(inp, 1)
        self._startcomputation(par, bound_cond=0, odd=1)

        # Start Streaming
        islice = 2*self.slices*self.num_dev
        odd = True
        while islice < self.NSlice:
            odd = not odd
            # Collect Previous Block
            (rhs, lhs) = self._streamtohostnorm(
                outp,
                rhs,
                lhs,
                odd)
            # Stream new Block
            self._streamtodevice(inp, odd)
            # Start Computation
            self._startcomputation(par, bound_cond=0, odd=odd)
            islice += self.num_dev*self.slices
        # Collect last block
        if odd:
            (rhs, lhs) = self._streamtohostnorm(outp, rhs, lhs, 0)
            (rhs, lhs) = self._streamtohostnorm(outp, rhs, lhs, 1)
        elif (self.NSlice/(self.slices*self.num_dev) <
              2*(self.slices*self.num_dev)):
            (rhs, lhs) = self._streamtohostnorm(outp, rhs, lhs, 1)
            (rhs, lhs) = self._streamtohostnorm(outp, rhs, lhs, 0)
        else:
            (rhs, lhs) = self._streamtohostnorm(outp, rhs, lhs, 0)
        # Wait for all Queues to finish
        for i in range(self.num_dev):
            self.queue[4*i].finish()
            self.queue[4*i+1].finish()
            self.queue[4*i+2].finish()
            self.queue[4*i+3].finish()
        return (lhs, rhs)

    def _streamtodevice(self, inp, odd):
        for idev in range(self.num_dev):
            idx = self._getindtodev()
            for ifun in range(self.num_fun):
                if not inp[ifun] == []:
                    for iinp in range(len(self.inp[ifun][idev])):
                        if not inp[ifun][iinp] == []:
                            self.inp[
                                ifun][
                                    idev*self.num_dev+odd][
                                        iinp].add_event(
                                cl.enqueue_copy(
                                  self.queue[4*idev+odd],
                                  self.inp[
                                      ifun][
                                          idev*self.num_dev+odd][
                                              iinp].data,
                                  inp[
                                      ifun][
                                          iinp][idx, ...],
                                  wait_for=self.inp[
                                      ifun][
                                          idev*self.num_dev+odd][
                                              iinp].events,
                                  is_blocking=False))
            self.queue[4*idev+odd].flush()

    def _startcomputation(self, par=[], bound_cond=0, odd=0):
        for idev in range(self.num_dev):
            for ifun in range(self.num_fun):
                self.outp[
                    ifun][
                        idev*self.num_dev+odd].add_event(
                    self.fun[ifun](
                        self.outp[ifun][idev*self.num_dev+odd],
                        self.inp[ifun][idev*self.num_dev+odd][:],
                        par,
                        idev,
                        odd,
                        bound_cond))
            bound_cond = 0
            self.queue[4*idev+odd].flush()

    def _streamtohost(self, outp, odd):
        for idev in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+3-odd].finish()
            if idev > 1:
                self.queue[4*(idev-1)+2+odd].finish()
            idx = self._getindtohost()
            for ifun in range(self.num_fun):
                self.outp[ifun][idev*self.num_dev+odd].add_event(
                    cl.enqueue_copy(
                      self.queue[4*idev+2+odd],
                      outp[ifun][idx, ...],
                      self.outp[ifun][idev*self.num_dev+odd].data,
                      wait_for=self.outp[ifun][idev*self.num_dev+odd].events,
                      is_blocking=False))
            self.queue[4*idev+2+odd].flush()

    def _streamtohostnorm(self, outp, rhs, lhs, odd):
        for idev in range(self.num_dev):
            self.queue[4*(self.num_dev-1)+3-odd].finish()
            if idev > 1:
                self.queue[4*(idev-1)+2+odd].finish()
            idx = self._getindtohost()
            for ifun in range(self.num_fun):
                self.outp[ifun][idev*self.num_dev+odd].add_event(
                    cl.enqueue_copy(
                      self.queue[4*idev+2+odd],
                      outp[ifun][idx, ...],
                      self.outp[ifun][idev*self.num_dev+odd].data,
                      wait_for=self.outp[ifun][idev*self.num_dev+odd].events,
                      is_blocking=False))
                if self.reverse:
                    if not self.at_end:
                        (rhs, lhs) = self._calcnormreverse(
                            rhs, lhs, idev, ifun, odd)
                    else:
                        (rhs, lhs) = self._calcnormforward(
                            rhs, lhs, idev, ifun, odd)
                else:
                    if not self.at_end:
                        (rhs, lhs) = self._calcnormforward(
                            rhs, lhs, idev, ifun, odd)
                    else:
                        (rhs, lhs) = self._calcnormreverse(
                            rhs, lhs, idev, ifun, odd)
            self.queue[4*idev+odd].flush()
            self.queue[4*idev+2+odd].flush()
        return (rhs, lhs)

    def _resetindex(self):
        if self.reverse:
            self.idx_todev_start = self.NSlice - (self.slices + self.overlap)
            self.idx_todev_stop = self.NSlice
            self.idx_tohost_start = self.NSlice - (self.slices +
                                                   self.overlap)
            self.idx_tohost_stop = self.NSlice
        else:
            self.idx_todev_start = 0
            self.idx_todev_stop = (self.slices + self.overlap)
            self.idx_tohost_start = 0
            self.idx_tohost_stop = (self.slices + self.overlap)

    def connectouttoin(self, outpos, inpos):
        for j in range(2*self.num_dev):
            self.inp[inpos[0]][j][inpos[1]] = self.outp[outpos][j]

    def resettmparrays(self):
        for j in range(self.num_fun):
            for i in range(2*self.num_dev):
                self.outp[j][i] = clarray.empty_like(self.outp[j][i])

    def _calcnormreverse(self, rhs, lhs, idev, ifun, odd=0):
        if self.lhs[ifun] is False:
            rhs += clarray.vdot(
                self.outp[
                    ifun][
                        idev*self.num_dev+odd][self.overlap:, ...] -
                self.inp[
                    ifun][
                        idev*self.num_dev+odd][0][self.overlap:, ...],
                self.outp[
                    ifun][
                        idev*self.num_dev+odd][self.overlap:, ...] -
                self.inp[
                    ifun][
                        idev*self.num_dev+odd][0][self.overlap:, ...]
                ).get()
        else:
            lhs += clarray.vdot(
                self.outp[
                    ifun][
                        idev*self.num_dev+odd][self.overlap:, ...] -
                self.inp[
                    ifun][
                        idev*self.num_dev+odd][-1][self.overlap:, ...],
                self.outp[
                    ifun][
                        idev*self.num_dev+odd][self.overlap:, ...] -
                self.inp[
                    ifun][
                        idev*self.num_dev+odd][-1][self.overlap:, ...]
                ).get()
        return (rhs, lhs)

    def _calcnormforward(self, rhs, lhs, idev, ifun, odd=0):
        if self.lhs[ifun] is False:
            rhs += clarray.vdot(
                self.outp[
                    ifun][
                        idev*self.num_dev+odd][:self.slices, ...] -
                self.inp[
                    ifun][
                        idev*self.num_dev+odd][0][:self.slices, ...],
                self.outp[
                    ifun][
                        idev*self.num_dev+odd][:self.slices, ...] -
                self.inp[
                    ifun][
                        idev*self.num_dev+odd][0][:self.slices, ...]
                ).get()
        else:
            lhs += clarray.vdot(
                self.outp[
                    ifun][
                        idev*self.num_dev+odd][:self.slices, ...] -
                self.inp[
                    ifun][
                        idev*self.num_dev+odd][-1][:self.slices, ...],
                self.outp[
                    ifun][
                        idev*self.num_dev+odd][:self.slices, ...] -
                self.inp[
                    ifun][
                        idev*self.num_dev+odd][-1][:self.slices, ...]
                ).get()
        return (rhs, lhs)
