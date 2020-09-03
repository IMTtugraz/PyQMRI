#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base for PyOpenCL programs."""
import pyopencl as cl


class CLProgram():
    """Base class for PyOpenCL kernels.

    Parameters
    ----------
      ctx : PyOpenCL.Context
        The context to compile the code in.
      code : string
        The Kernel to compile
    """

    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build("-cl-mad-enable -cl-fast-relaxed-math")
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
            self.__dict__[kernel.function_name] = kernel
