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


def calckbkernel(kwidth, overgridfactor, G, klength=32):
    """Precompute the Kaiser-Bessel Kerenl used for gridding.

    This function precomputes a Kaiser-Bessel kernel for a given width.
    To save memory, only half the kernel is computed due to its symmetry
    around zero.

    Parameters
    ----------
      kwidth : float
        The width of the kernel in Gridunits
      overgridfactor : float
        The overgridfactor compared to the desired image matrix.
      G : int
          The number of acquired data points per projection.
      klength : int, 32
        The number of discretization points of the kernel.

    Returns
    -------
        numpy.array
            The precomputed Kaiser-Bessel kernel
        numpy.array
            The fourier transformation of the kernel

    """
    if klength < 2:
        klength = 2
        print('Warning:  klength must be 2 or more - using 2.')

    a = overgridfactor
    w = kwidth
    # From Beatty et al.
    beta = np.pi * np.sqrt(w**2 / a**2 * (a - 0.5)**2 - 0.8)

    # Kernel radii - grid samples.
    u = np.linspace(0, np.floor(klength * w / 2), int(np.ceil(klength * w / 2))
                    ) / (np.floor(klength * w / 2)) * w / 2 / G

    kern = _kb(u, kwidth, beta, G)
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

    return (kern, kern_ft)


def _kb(u, w, beta, G):
    if np.size(w) > 1:
        raise TypeError('w should be a single scalar value.')

    y = 0 * u  # Allocate space.
    uz = np.where(np.abs(u) <= w / (2 * G))			# Indices where u<w/2.

    if np.size(uz) > 0:			# Calculate y at indices uz.
        # Argument - see Jackson '91.
        x = beta * np.sqrt(1 - (2 * u[uz] * G / w)**2)
        y[uz] = G * np.i0(x) / w

    return y
