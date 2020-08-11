#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:56:16 2018

@author: omaier
"""
import numpy as np
from scipy.special import i0 as i0
#
# function y = kb(u,w,beta)
#
# Computes the Kaiser-Bessel function used for gridding, namely
#
# y = f(u,w,beta) = I0 [ beta*sqrt(1-(2u/w)^2) ]/w
#
# where I0 is the zero-order modified Bessel function
#  of the first kind.
#
# INPUT:
#  u = vector of k-space locations for calculation.
#  w = width parameter - see Jackson et al.
# beta = beta parameter - see Jackson et al.
#
# OUTPUT:
#  y = vector of Kaiser-Bessel values.
#

# B. Hargreaves Oct, 2003.
# Adapted for Python by O. Maier


def kaiser_bessel(u, width, beta):
    """
    Kaiser-Bessel window precomputation.

    Args
    ----
      u (numpy.array):
        Kernel Radii
      width (int):
        Kernel width
      beta (float):
        Scale for the argument of the modified bessel function of oder 0,
        see Jackson '91 and Beatty et al.

    Returns
    -------
      numpy.array
        The Kaiser-Bessel window
    """
    assert np.size(width) == 1, 'width should be a single scalar value.'

    # if np.size(uz) > 0:  # Calculate y at indices uz.
    x = beta * np.sqrt(1 - (2 * u / width) ** 2)
    # Argument - see Jackson '91.
    y = i0(x) / width

    return y
