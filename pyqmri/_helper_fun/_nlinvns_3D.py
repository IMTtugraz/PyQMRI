#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-linear Inversion.

nlinvns
 Written and invented
 by Martin Uecker <muecker@gwdg.de> in 2008-09-22

 Modifications by Tilman Sumpf 2012 <tsumpf@gwdg.de>:
     removed fftshift during reconstruction (ns = "no shift")
     added switch to return coil profiles
     added switch to force the image estimate to be real
     use of vectorized operation rather than "for" loops

 Version 0.1

 Biomedizinische NMR Forschungs GmbH am
 Max-Planck-Institut fuer biophysikalische Chemie
Adapted for Python by O. Maier
"""
from time import perf_counter
import numpy as np

from pyqmri.transforms import PyOpenCLnuFFT
import pyopencl.array as cla


def nlinvns(Y, n, par, *arg, DTYPE=np.complex64, DTYPE_real=np.float32):
    """Non-linear inversen based Coil sensitivity estimation.

    Parameters
    ----------
      Y : numpy.array
        Data used for estimation
      n : int
        number of Gausse-Newton iteration steps
      realConstr : bool
        Real value constraint on the image. Should be set to false usually.
      returnProfiles : bool
        Return coil profiles. Should be set to True usually.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.

    Returns
    -------
    numpy.array :
        The results of each iteration containing the reconstructed image at
        position 0 and 1, followed by the complex coil sensitivities.
    """
    nrarg = len(arg)
    if nrarg == 2:
        returnProfiles = arg[0]
        realConstr = arg[1]
    elif nrarg < 2:
        realConstr = False
        if nrarg < 1:
            returnProfiles = 0

    print('Start...')

    alpha = 0.001

    [c, z, y, x] = Y.shape
    
    par["NC"] = c
    par["fft_dim"] = [-3, -2, -1]
    par["mask"] = np.ones(Y.shape[1:], dtype=par["DTYPE_real"])
    
    FFT = PyOpenCLnuFFT.create(
        par["ctx"][0], par["queue"][0], par,
        DTYPE=par["DTYPE"],
        DTYPE_real=par["DTYPE_real"],
        radial=False, SMS=False)
       

    if returnProfiles:
        R = np.zeros([c + 2, n, z, y, x], DTYPE)

    else:
        R = np.zeros([2, n, z, y, x], DTYPE)

    # initialization x-vector
    X0 = np.array(np.zeros([c + 1, z, y, x]), DTYPE)
    X0[0] = 1  # object part

    # initialize mask and weights
    P = np.ones(Y[0].shape, dtype=DTYPE)
    P[Y[0] == 0] = 0

    W = _weights(z, y, x)

    W = _fftshift2(W).astype(DTYPE_real)

    # normalize data vector
    yscale = 100 / np.sqrt(_scal(Y, Y))
    YS = Y * yscale

    XT = np.zeros([c + 1, z, y, x], dtype=DTYPE)
    XN = np.copy(X0)
    
    start = perf_counter()  
       
    for i in range(0, n):

        # the application of the weights matrix to XN
        # is moved out of the operator and the derivative
        XT[0] = np.copy(XN[0])
        XT[1:] = _apweightsns(W, np.copy(XN[1:]), FFT)

        RES = (YS - _opns(P, XT, FFT))

        print(np.round(np.linalg.norm(RES)))

        # calculate rhs
        r = _derHns(P, W, XT, RES, realConstr, FFT)

        r = np.array(r + alpha * (X0 - XN), dtype=DTYPE)

        z = np.zeros_like(r)
        d = np.copy(r)
        dnew = np.linalg.norm(r)**2
        dnot = np.copy(dnew)

        for j in range(0, 100):

            # regularized normal equations
            q = _derHns(P, W, XT, _derns(P, W, XT, d, FFT), realConstr, FFT) + alpha * d
            # np.nan_to_num(q)

            a = dnew / np.real(_scal(d, q))
            z = z + a * d
            r = r - a * q
            # np.nan_to_num(r)
            dold = np.copy(dnew)
            dnew = np.linalg.norm(r)**2

            d = d * ((dnew / dold)) + r
            # np.nan_to_num(d)
            if np.sqrt(dnew) < (1e-4 * dnot):
                break

        print('(', j, ')')

        XN = XN + z

        alpha = alpha / 3

        # postprocessing

        CR = _apweightsns(W, XN[1:, :, :], FFT)

        if returnProfiles:
            R[2:, i] = CR / yscale

        C = (np.conj(CR) * CR).sum(0)

        R[0, i] = (XN[0] * np.sqrt(C) / yscale)
        R[1, i] = np.copy(XN[0])

    end = perf_counter()
    print('done in', round((end - start)), 's')
    return R


def _scal(a, b):
    v = np.sum(np.conj(a) * b)
    return v


def _apweightsns(W, CT, fft):
    C = _nsIfft(W * CT, fft)
    return C


def _apweightsnsH(W, CT, fft):
    C = np.conj(W) * _nsFft(CT, fft)
    return C


def _opns(P, X, fft):
    K = X[0] * X[1:]
    K = P * _nsFft(K, fft)
    return K


def _derns(P, W, X0, DX, fft):
    K = X0[0] * _apweightsns(W, DX[1:], fft)
    K = K + (DX[0] * X0[1:])
    K = P * _nsFft(K, fft)
    return K


def _derHns(P, W, X0, DK, realConstr, fft):

    K = _nsIfft(P * DK, fft)

    if realConstr:
        DXrho = np.sum(np.real(K * np.conj(X0[1:])), 0)
    else:
        DXrho = np.sum(K * np.conj(X0[1:]), 0)

    DXc = _apweightsnsH(W, (K * np.conj(X0[0])), fft)
    DX = np.concatenate(
        (DXrho[None, ...], DXc), axis=0)
    return DX


def _nsFft(M, fft):
    # K = np.fft.fftn(M, axes=(-3,-2,-1), norm="ortho")
    tmp = cla.to_device(fft.queue, M)
    K = cla.empty_like(tmp)
    fft.FFT(K,tmp)
    # K = fft(M)
    return K.get()


def _nsIfft(M, fft):
    # K = np.fft.ifftn(M, axes=(-3,-2,-1), norm="ortho")
    tmp = cla.to_device(fft.queue, M)
    K = cla.empty_like(tmp)
    fft.FFTH(K,tmp)
    # K = fft(M)
    return K.get()


def _weights(z, y, x):
    W = np.zeros([z, y, x])
    for k in range(z):
        for i in range(y):
            for j in range(x):
                d = ((i) / y - 0.5)**2 + ((j) / x - 0.5)**2 + ((k) / z - 0.5)**2
                W[k, i, j] = 1 / (1 + 220 * d)**16
    return W


def _fftshift2(I):
    if I.ndim >= 3:
        S = np.fft.fftshift(I, axes=(-3,-2,-1))
    else:
        S = np.fft.fftshift(np.fft.fftshift(I, 1), 0)
    return S
