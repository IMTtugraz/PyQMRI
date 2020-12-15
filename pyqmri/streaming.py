#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the class for streaming operations on the GPU."""

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray


class Stream:
    """Basic streaming Class.

    This Class is responsible for performing asynchroneous transfer
    and computation on the GPU for arbitrary large numpy data.

    Parameters
    ----------
     fun : list of functions
        This list contains all functions that should be executed on the
        GPU. The functions are executed in order from first to last
        element of the list.
      outp_shape (list of tuple):
        The shape of the output array. Slice dimension is assumed to
        be the same as number of parallel slices plus overlap.
      inp_shape (list of list of tuple):
        The shape of the input arrays. Slice dimension is assumed to
        be the same as number of parallel slices plus overlap.
      par_slices : int
        Number of slices computed in one transfer on the GPU.
      overlap : int
        Overlap of adjacent blocks
      nslice : int
        Total number of slices
      queue : list of PyOpenCL.Queue
        The OpenCL queues used for transfer and computation. 4 queues are
        used per device.
      num_dev : int
        Number of computation devices.
      reverse : bool, false
        Indicator of the streaming direction. If False, streaming will
        start at the first and end at the last slice. If True streaming
        will be performed vice versa
      lhs : list of bool, None
        Indicator for the norm calculation in the line search of TGV.
        lhs refers to left hand side. Needs to be passed if a norm should
        be computed.
      DTYPE : numpy.dype, numpy.complex64
        Complex data type.

    Attributes
    ----------
      fun : list of functions
        This list contains all functions that should be executed on the GPU.
        The functions are executed in order from first to last element of the
        list.
      num_dev : int
        Number of computation devices.
      slices : int
        Number of slices computed in one transfer on the GPU.
      overlap : int
        Overlap of adjacent blocks
      queue : list of PyOpenCL.Queue
        The OpenCL queues used for transfer and computation. 4 queues are used
        per device.
      reverse : bool
        Indicator of the streaming direction. If False, streaming will start
        at the first and end at the last slice. If True streaming will be
        performed vice versa
      NSlice : int
        Total number of slices
      num_fun : int
        Total number of functions to stream (length of fun)
      lhs : list of bool, None
        Indicator for the norm calculation in the line search of TGV.
        lhs refers to left hand side.
      at_end : bool
        Specifies if the end of the data slice dimension is reached
      inp (list of list of list of PyOpenCL.Array):
        For each function a list of devices and a list of inputs is generated.
        E.g. for one function which needs two inputs and one computation device
        the list would have dimensions [1][1][2]
      outp (list of list of PyOpenCL.Array):
        For each function a list of devices with a single output is generated.
        E.g. for one function and one device
        the list would have dimension [1][1]
    """

    def __init__(self,
                 fun,
                 outp_shape,
                 inp_shape,
                 par_slices,
                 overlap,
                 nslice,
                 queue,
                 num_dev,
                 reverse=False,
                 lhs=None,
                 DTYPE=np.complex64):
        self.fun = fun
        self.num_dev = num_dev
        self.slices = par_slices
        self.overlap = overlap
        self.queue = queue
        self.reverse = reverse
        self.nslice = nslice
        self.num_fun = len(self.fun)
        self.dtype = DTYPE

        self.lhs = lhs
        self.at_end = False
        self.idx_todev_start = 0
        self.idx_todev_stop = 0
        self.idx_tohost_start = 0
        self.idx_tohost_stop = 0
        self._resetindex()

        self.inp = []
        self.outp = []

        self._alloctmparrays(inp_shape, outp_shape)

    def __add__(self, other):
        """Overloading add.

        Concatinates the functions, inputs and outputs of one stremaing
        object to another.

        Parameters
        ----------
          other (class stream):
            The object which should be added.

        Returns
        -------
          The combined objects
        """
        for j in range(other.num_fun):
            self.fun.append(other.fun[j])
            self.inp.append(other.inp[j])
            self.outp.append(other.outp[j])
        self.num_fun += other.num_fun
        return self

    def __del__(self):
        """Delete the Queue."""
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
                    if not len(inp_shape[j][k]) == 0:
                        self.inp[j][i].append(
                            clarray.empty(
                                self.queue[4*int(i/2)],
                                ((block_size, )+inp_shape[j][k][1:]),
                                dtype=self.dtype))
                    else:
                        self.inp[j][i].append([])

        for j in range(self.num_fun):
            self.outp.append([])
            for i in range(2*self.num_dev):
                self.outp[j].append(
                    clarray.empty(
                        self.queue[4*int(i/2)],
                        ((block_size, )+outp_shape[j][1:]),
                        dtype=self.dtype))

    def _getindtodev(self):
        if self.reverse:
            tmp_return = slice(self.idx_todev_start,
                               self.idx_todev_stop)
            self.idx_todev_start -= self.slices
            self.idx_todev_stop -= self.slices
            if self.idx_todev_start < 0:
                self.idx_todev_start = 0
                self.idx_todev_stop = self.slices + self.overlap
        else:
            tmp_return = slice(self.idx_todev_start,
                               self.idx_todev_stop)
            self.idx_todev_start += self.slices
            self.idx_todev_stop += self.slices
            if self.idx_todev_stop > self.nslice:
                self.idx_todev_start = (self.nslice -
                                        self.slices -
                                        self.overlap)
                self.idx_todev_stop = self.nslice
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
        else:
            tmp_return = slice(self.idx_tohost_start,
                               self.idx_tohost_stop)
            self.idx_tohost_start += self.slices
            self.idx_tohost_stop += self.slices
            self.at_end = False
            if self.idx_tohost_stop > self.nslice:
                self.at_end = True
                self.idx_tohost_start = (self.nslice -
                                         self.slices -
                                         self.overlap)
                self.idx_tohost_stop = self.nslice
        return tmp_return

    def eval(self, outp, inp, par=None):
        """Evaluate all functions of the object.

        Perform asynchroneous evaluation of the functions stored in
        fun.

        Parameters
        ----------
          outp (list of np.arrays):
            Result of the computation for each function as numpy array
          inp (list of list of np.arrays):
            For each function contains a list of numpy arrays used as input.
          par (list of list of parameters):
            Optional list of parameters which should be passed to a function.
        """
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
        while islice+self.overlap < self.nslice:
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
        else:
            self._streamtohost(outp, 1)
            self._streamtohost(outp, 0)
        # Wait for all Queues to finish
        for i in range(self.num_dev):
            self.queue[4*i].finish()
            self.queue[4*i+1].finish()
            self.queue[4*i+2].finish()
            self.queue[4*i+3].finish()

    def evalwithnorm(self, outp, inp, par=None):
        """Evaluate all functions of the object and returns norms.

        Perform asynchroneous evaluation of the functions stored in
        fun. Same as eval but also computes the norm relevant for the
        linesearch in the TGV algorithm.

        Parameters
        ----------
          outp : list of np.arrays
            Result of the computation for each function as numpy array
          inp (list of list of np.arrays):
            For each function contains a list of numpy arrays used as input.
          par : list of list of parameters
            Optional list of parameters which should be passed to a function.

        Returns
        -------
          tuple of floats:
            (lhs, rhs) The lhs and rhs for the linesearch used in the TGV
            algorithm.
        """
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
        while islice + self.overlap < self.nslice:
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
            islice += self.num_dev*self.slices
            self._startcomputation(par, bound_cond=0, odd=odd)

        # Collect last block
        if odd:
            (rhs, lhs) = self._streamtohostnorm(outp, rhs, lhs, 0)
            (rhs, lhs) = self._streamtohostnorm(outp, rhs, lhs, 1)
        else:
            (rhs, lhs) = self._streamtohostnorm(outp, rhs, lhs, 1)
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
                if not len(inp[ifun]) == 0:
                    for iinp in range(len(self.inp[ifun][idev])):
                        if not len(inp[ifun][iinp]) == 0:
                            self.inp[
                                ifun][
                                    2*idev+odd][
                                        iinp].add_event(
                                            cl.enqueue_copy(
                                                self.queue[4*idev+odd],
                                                self.inp[
                                                    ifun][
                                                        2*idev+odd][
                                                            iinp].data,
                                                inp[
                                                    ifun][
                                                        iinp][idx, ...],
                                                wait_for=self.inp[
                                                    ifun][
                                                        2*idev+odd][
                                                            iinp].events,
                                                is_blocking=False))
                            self.queue[4*idev+odd].flush()

    def _startcomputation(self, par=None, bound_cond=0, odd=0):
        if par is None:
            par = []
            for ifun in range(self.num_fun):
                par.append([])
        for idev in range(self.num_dev):
            for ifun in range(self.num_fun):
                for inps in self.inp[ifun][2*idev+odd]:
                    for inp in inps:
                        for event in inp.events:
                            event.wait()
                self.outp[
                    ifun][
                        2*idev+odd].add_event(
                            self.fun[ifun](
                                self.outp[ifun][2*idev+odd],
                                self.inp[ifun][2*idev+odd][:],
                                par[ifun],
                                idev,
                                odd,
                                bound_cond=bound_cond))
                self.queue[4*idev+odd].flush()
            bound_cond = 0

    def _streamtohost(self, outp, odd):
        for idev in range(self.num_dev):
            self.queue[4*idev+3-odd].finish()
            if self.num_dev > 1:
                self.queue[4*np.mod(idev-1, self.num_dev)+2+odd].finish()
                self.queue[4*np.mod(idev-1, self.num_dev)+3-odd].finish()
            idx = self._getindtohost()
            for ifun in range(self.num_fun):
                self.outp[ifun][2*idev+odd].add_event(
                    cl.enqueue_copy(
                        self.queue[4*idev+2+odd],
                        outp[ifun][idx, ...],
                        self.outp[ifun][2*idev+odd].data,
                        wait_for=self.outp[ifun][2*idev+odd].events,
                        is_blocking=False))
                self.queue[4*idev+2+odd].flush()

    def _streamtohostnorm(self, outp, rhs, lhs, odd):
        for idev in range(self.num_dev):
            self.queue[4*idev+3-odd].finish()
            if self.num_dev > 1:
                self.queue[4*np.mod(idev-1, self.num_dev)+2+odd].finish()
                self.queue[4*np.mod(idev-1, self.num_dev)+3-odd].finish()
            idx = self._getindtohost()
            for ifun in range(self.num_fun):
                self.outp[ifun][2*idev+odd].add_event(
                    cl.enqueue_copy(
                        self.queue[4*idev+2+odd],
                        outp[ifun][idx, ...],
                        self.outp[ifun][2*idev+odd].data,
                        wait_for=self.outp[ifun][2*idev+odd].events,
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
            self.idx_todev_start = self.nslice - (self.slices + self.overlap)
            self.idx_todev_stop = self.nslice
            self.idx_tohost_start = self.nslice - (self.slices +
                                                   self.overlap)
            self.idx_tohost_stop = self.nslice
        else:
            self.idx_todev_start = 0
            self.idx_todev_stop = (self.slices + self.overlap)
            self.idx_tohost_start = 0
            self.idx_tohost_stop = (self.slices + self.overlap)

    def connectouttoin(self, outpos, inpos):
        """Connect output to input of functions within the object.

        This function can be used to connect the output of a function to the
        input of another one used in the same stream object.

        Parameters
        ----------
          outpos : int
            The position in the list of outputs which should be connected to
            an input
          inpos (list of list of np.arrays:
            The position in the list of inputs which should be connected
            with an output
        """
        for j in range(2*self.num_dev):
            self.inp[inpos[0]][j][inpos[1]] = self.outp[outpos][j]

    def _calcnormreverse(self, rhs, lhs, idev, ifun, odd=0):
        if self.lhs[ifun] is False:
            rhs += clarray.vdot(
                self.outp[
                    ifun][
                        2*idev+odd][self.overlap:, ...] -
                self.inp[
                    ifun][
                        2*idev+odd][0][self.overlap:, ...],
                self.outp[
                    ifun][
                        2*idev+odd][self.overlap:, ...] -
                self.inp[
                    ifun][
                        2*idev+odd][0][self.overlap:, ...]
                ).get()
        else:
            lhs += clarray.vdot(
                self.outp[
                    ifun][
                        2*idev+odd][self.overlap:, ...] -
                self.inp[
                    ifun][
                        2*idev+odd][-1][self.overlap:, ...],
                self.outp[
                    ifun][
                        2*idev+odd][self.overlap:, ...] -
                self.inp[
                    ifun][
                        2*idev+odd][-1][self.overlap:, ...]
                ).get()
        return (rhs, lhs)

    def _calcnormforward(self, rhs, lhs, idev, ifun, odd=0):
        if self.lhs[ifun] is False:
            rhs += clarray.vdot(
                self.outp[
                    ifun][
                        2*idev+odd][:self.slices, ...] -
                self.inp[
                    ifun][
                        2*idev+odd][0][:self.slices, ...],
                self.outp[
                    ifun][
                        2*idev+odd][:self.slices, ...] -
                self.inp[
                    ifun][
                        2*idev+odd][0][:self.slices, ...]
                ).get()
        else:
            lhs += clarray.vdot(
                self.outp[
                    ifun][
                        2*idev+odd][:self.slices, ...] -
                self.inp[
                    ifun][
                        2*idev+odd][-1][:self.slices, ...],
                self.outp[
                    ifun][
                        2*idev+odd][:self.slices, ...] -
                self.inp[
                    ifun][
                        2*idev+odd][-1][:self.slices, ...]
                ).get()
        return (rhs, lhs)
