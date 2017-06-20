#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:21:57 2017

@author: omaier
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["gradients_divergences.pyx", "VFA_Model_Reco.pyx","IRLL_Model.pyx"], gdb_debug=True),
    include_dirs = [numpy.get_include()]
)
