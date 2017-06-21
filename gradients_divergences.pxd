#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:23:19 2017

@author: omaier
"""
import numpy as np
cimport numpy as np
np.import_array()


DTYPE = np.complex128
ctypedef np.complex_t DTYPE_t

cpdef bdiv_1(np.ndarray[DTYPE_t, ndim=4] v, int dx=*, int dy=*)
cpdef fgrad_1(np.ndarray[DTYPE_t, ndim=3] u,int dx=*, int dy=*)
cpdef fdiv_2(np.ndarray[DTYPE_t, ndim=4] x,int dx=*, int dy=*)
cpdef sym_bgrad_2(np.ndarray[DTYPE_t, ndim=4] x, int dx=*, int dy=*)

cpdef bdiv_3(np.ndarray[DTYPE_t, ndim=5] v, int dx=*, int dy=*, int dz = *)
cpdef fgrad_3(np.ndarray[DTYPE_t, ndim=4] u,int dx=*, int dy=*, int dz = *)
cpdef fdiv_3(np.ndarray[DTYPE_t, ndim=5] x,int dx=*, int dy=*, int dz=*)
cpdef sym_bgrad_3(np.ndarray[DTYPE_t, ndim=5] x, int dx=*, int dy=*, int dz=*)