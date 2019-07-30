#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:05:45 2018

@author: omaier

Copyright 2019 Oliver Maier

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import numpy as np
from pyqmri._helper_fun._kb import kb
# function [kern,kbu] = calckbkernel(kwidth,overgridfactor,klength)
# Function calculates the appropriate Kaiser-Bessel kernel
# for gridding, using the approach of Jackson et al.
# INPUT:
#  width = kernel width in grid samples.
#  overgridfactor = over-gridding factor.
#  klength = kernel look-up-table length.
# OUTPUT:
#  kern = kernel values for klength values
#  of u, uniformly spaced from 0 to kwidth/2.
#  kbu = u values.
# Original Skript by B. Hargreaves
# Adapted for Python by O. Maier


def calckbkernel(kwidth, overgridfactor, G, klength=32):
    if (klength < 2):
        klength = 2
        print('Warning:  klength must be 2 or more - using 2.')

    a = overgridfactor
    w = kwidth
    # From Beatty et al.
    beta = np.pi * np.sqrt(w**2 / a**2 * (a - 0.5)**2 - 0.8)

# Kernel radii - grid samples.
    u = np.linspace(0, np.floor(klength * w / 2), int(np.ceil(klength * w / 2))
                    ) / (np.floor(klength * w / 2)) * w / 2 / G

    kern = kb(u, kwidth, beta, G)
    kern = kern / kern[u == 0]  # Normalize.

    ft_y = np.flip(kern)
    ft_y = np.concatenate((ft_y[:-1], kern))
    ft_y = np.pad(ft_y, int((G * klength - ft_y.size) / 2), 'constant')
    ft_y = np.abs(np.fft.fftshift(np.fft.ifft(
        np.fft.ifftshift(ft_y))) * ft_y.size)
    x = np.linspace(-int(G / (2 * a)), int(G / (2 * a)) - 1, int(G / (a)))

    ft_y = ft_y[(ft_y.size / 2 - x).astype(int)]
    h = np.sinc(x / (G * klength))**2

    kern_ft = (ft_y * h)
    kern_ft = (kern_ft / np.max((kern_ft)))

    return (kern, kern_ft, u)
