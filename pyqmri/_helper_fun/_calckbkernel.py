#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute Kaiser-Bessel gridding kernel.

Original Skript by B. Hargreaves
Adapted for Python by O. Maier

function [kern,kbu] = calckbkernel(kwidth,overgridfactor,klength)
Function calculates the appropriate Kaiser-Bessel kernel
for gridding, using the approach of Jackson et al.
INPUT:
  width = kernel width in grid samples.
  overgridfactor = over-gridding factor.
  klength = kernel look-up-table length.
OUTPUT:
  kern = kernel values for klength values
  of u, uniformly spaced from 0 to kwidth/2.
  kbu = u values.
"""
import numpy as np
import warnings

def calckbkernel(
        kernelwidth,
        overgridfactor,
        gridsize,
        kernellength=32,
        **kwargs):
    """
    Calculate the appropriate Kaiser-Bessel gridding kernel.
    Function calculates the appropriate Kaiser-Bessel kernel for gridding,
    using the approach of Jackson et al. Original Skript by B. Hargreaves
    Adapted for Python by O. Maier
    Args
    ----
        kernelwidth (int):
            kernel width
        overgridfactor (float):
            over-gridding factor.
        gridsize (int):
            gridsize of oversampled grid
        kernellength (int):
            kernel look-up-table length.
    Returns
    -------
        kern:
            kernel values for kernellength values of u,
            uniformly spaced from 0 to kernelwidth/2.
        kern_ft:
            normalized Fourier transform of kernel used for deapodization
        kbu:
            position of sampling points of kernel.
            linspace from 0 to length of kernel
    """
    if kernellength < 2:
        kernellength = 2
        warnings.warn(
            'Warning:  kernellength must be 2 or more. Default to 2.'
            )

    # From Beatty et al. -
    # Rapid Gridding Reconstruction With a Minimal Oversampling Ratio -
    # equation [5]
    beta = np.pi * np.sqrt(
        (kernelwidth / overgridfactor) ** 2
        * (overgridfactor - 0.5) ** 2
        - 0.8
        )
    # Kernel radii.
    u = np.linspace(
        0,
        (kernelwidth/2),
        int(np.ceil(kernellength*kernelwidth/2)))

    kern = _kb(u, kernelwidth, beta)
    kern = kern / kern[u == 0]  # Normalize.

    ft_y = np.flip(kern)
    if np.mod(kernelwidth, 2):
        ft_y = np.concatenate((ft_y[:-1], kern))
    else:
        ft_y = np.concatenate((ft_y, kern))

    ft_y = np.abs(
      np.fft.fftshift(
          np.fft.ifft(
                  ft_y, gridsize * kernellength
              )
          )
      * ft_y.size
      )

    x = np.linspace(
        -int(np.floor(gridsize/(2*overgridfactor))),
        int(np.floor(gridsize/(2*overgridfactor)))-1,
        int(np.floor(gridsize/(overgridfactor)))
        )

    ft_y = ft_y[(ft_y.size / 2 + x).astype(int)]
    h = np.sinc(x / (gridsize * kernellength)) ** 2

    kern_ft = ft_y * h
    kern_ft = kern_ft / np.max(kern_ft)

    return kern, kern_ft

def _kb(u, w, beta):
    # if np.size(w) > 1:
    #     raise TypeError('w should be a single scalar value.')

    # y = 0 * u  # Allocate space.
    # uz = np.where(np.abs(u) <= w / (2 * G))			# Indices where u<w/2.

    # if np.size(uz) > 0:			# Calculate y at indices uz.
    #     # Argument - see Jackson '91.
    #     x = beta * np.sqrt(1 - (2 * u[uz] * G / w)**2)
    #     y[uz] = G * np.i0(x) / w

    # return y
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
    assert np.size(w) == 1, 'width should be a single scalar value.'

    # if np.size(uz) > 0:  # Calculate y at indices uz.
    x = beta * np.sqrt(1 - (2 * u / w) ** 2)
    # Argument - see Jackson '91.
    y = np.i0(x) / w
    return y