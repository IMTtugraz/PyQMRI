#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:56:16 2018

@author: omaier
"""
import numpy as np
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


def kb(u, w, beta, G):
    if (np.size(w) > 1):
        raise('w should be a single scalar value.')

    y = 0 * u  # Allocate space.
    uz = np.where(np.abs(u) <= w / (2 * G))			# Indices where u<w/2.

    if (np.size(uz) > 0):			# Calculate y at indices uz.
        # Argument - see Jackson '91.
        x = beta * np.sqrt(1 - (2 * u[uz] * G / w)**2)
        y[uz] = G * np.i0(x) / w

    return (y)
