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

cdef np.ndarray[DTYPE_t, ndim=3] bdiv_1(np.ndarray[DTYPE_t, ndim=4] v, int dx=*, int dy=*)
cdef np.ndarray[DTYPE_t, ndim=4] fgrad_1(np.ndarray[DTYPE_t, ndim=3] u,int dx=*, int dy=*)
cdef np.ndarray[DTYPE_t, ndim=4] fdiv_2(np.ndarray[DTYPE_t, ndim=4] x,int dx=*, int dy=*)
cdef np.ndarray[DTYPE_t, ndim=4] sym_bgrad_2(np.ndarray[DTYPE_t, ndim=4] x, int dx=*, int dy=*)

cdef bdiv_3(np.ndarray[DTYPE_t, ndim=5] v, int dx=*, int dy=*, int dz = *)
cdef fgrad_3(np.ndarray[DTYPE_t, ndim=4] u,int dx=*, int dy=*, int dz = *)
cdef fdiv_3(np.ndarray[DTYPE_t, ndim=5] x,int dx=*, int dy=*, int dz=*)
cdef sym_bgrad_3(np.ndarray[DTYPE_t, ndim=5] x, int dx=*, int dy=*, int dz=*)