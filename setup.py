#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:21:57 2017

@author: omaier
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

ext_modules=[ Extension("*",
              ["./src/*.pyx"],
              libraries=["m","stdc++"],
              extra_compile_args = ["-ffast-math","-O3",'-fopenmp','-ggdb'],
              extra_link_args=['-fopenmp'],
              include_dirs = [numpy.get_include()])]

if __name__ == '__main__':
  setup(
      ext_modules = cythonize(ext_modules)

      )
