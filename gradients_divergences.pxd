#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:23:19 2017

@author: omaier
"""

cimport numpy as np
np.import_array()
import numpy as np

ctypedef np.complex64_t DTYPE_t

cpdef np.ndarray[DTYPE_t, ndim=3] bdiv_1(np.ndarray[DTYPE_t, ndim=4] v, float dx=*, float dy=*, np.ndarray[DTYPE_t, ndim=3] scale=*)
cpdef np.ndarray[DTYPE_t, ndim=4] fgrad_1(np.ndarray[DTYPE_t, ndim=3] u,float dx=*, float dy=*, np.ndarray[DTYPE_t, ndim=3] scale=*)
cpdef np.ndarray[DTYPE_t, ndim=4] fdiv_2(np.ndarray[DTYPE_t, ndim=4] x,float dx=*, float dy=*)
cpdef np.ndarray[DTYPE_t, ndim=4] sym_bgrad_2(np.ndarray[DTYPE_t, ndim=4] x, float dx=*, float dy=*)

cpdef bdiv_3(np.ndarray[DTYPE_t, ndim=5] v, float dx=*, float dy=*, float dz = *, np.ndarray[DTYPE_t, ndim=4] scale=*)
cpdef fgrad_3(np.ndarray[DTYPE_t, ndim=4] u,float dx=*, float dy=*, float dz = *, np.ndarray[DTYPE_t, ndim=4] scale=*)
cpdef fdiv_3(np.ndarray[DTYPE_t, ndim=5] x,float dx=*, float dy=*, float dz=*)
cpdef sym_bgrad_3(np.ndarray[DTYPE_t, ndim=5] x, float dx=*, float dy=*, float dz=*)
