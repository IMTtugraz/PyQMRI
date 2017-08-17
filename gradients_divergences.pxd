#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:23:19 2017

@author: omaier
"""

cimport numpy as np
np.import_array()
import numpy as np

ctypedef np.complex128_t DTYPE_t

cdef np.ndarray[DTYPE_t, ndim=3] bdiv_1(np.ndarray[DTYPE_t, ndim=4] v, float dx=*, float dy=*)
cdef np.ndarray[DTYPE_t, ndim=4] fgrad_1(np.ndarray[DTYPE_t, ndim=3] u,float dx=*, float dy=*)
cdef np.ndarray[DTYPE_t, ndim=4] fdiv_2(np.ndarray[DTYPE_t, ndim=4] x,float dx=*, float dy=*)
cdef np.ndarray[DTYPE_t, ndim=4] sym_bgrad_2(np.ndarray[DTYPE_t, ndim=4] x, float dx=*, float dy=*)

cdef bdiv_3(np.ndarray[DTYPE_t, ndim=5] v, float dx=*, float dy=*, float dz = *)
cdef fgrad_3(np.ndarray[DTYPE_t, ndim=4] u,float dx=*, float dy=*, float dz = *)
cdef fdiv_3(np.ndarray[DTYPE_t, ndim=5] x,float dx=*, float dy=*, float dz=*)
cdef sym_bgrad_3(np.ndarray[DTYPE_t, ndim=5] x, float dx=*, float dy=*, float dz=*)