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
import time
import numpy as np
import pyfftw


def nlinvns(Y, n, *arg, DTYPE=np.complex64, DTYPE_real=np.float32):
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

    alpha = 1

    [c, y, x] = Y.shape

    if returnProfiles:
        R = np.zeros([c + 2, n, y, x], DTYPE)

    else:
        R = np.zeros([2, n, y, x], DTYPE)

    # initialization x-vector
    X0 = np.array(np.zeros([c + 1, y, x]), DTYPE_real)
    X0[0, :, :] = 1  # object part

    # initialize mask and weights
    P = np.ones(Y[0, :, :].shape, dtype=DTYPE_real)
    P[Y[0, :, :] == 0] = 0

    W = _weights(y, x)

    W = _fftshift2(W)

    # normalize data vector
    yscale = 100 / np.sqrt(_scal(Y, Y))
    YS = Y * yscale

    XT = np.zeros([c + 1, y, x], dtype=DTYPE)
    XN = np.copy(X0)

    start = time.clock()
    for i in range(0, n):

        # the application of the weights matrix to XN
        # is moved out of the operator and the derivative
        XT[0, :, :] = np.copy(XN[0, :, :])
        XT[1:, :, :] = _apweightsns(W, np.copy(XN[1:, :, :]))

        RES = (YS - _opns(P, XT))

        print(np.round(np.linalg.norm(RES)))

        # calculate rhs
        r = _derHns(P, W, XT, RES, realConstr)

        r = np.array(r + alpha * (X0 - XN), dtype=DTYPE)

        z = np.zeros_like(r)
        d = np.copy(r)
        dnew = np.linalg.norm(r)**2
        dnot = np.copy(dnew)

        for j in range(0, 500):

            # regularized normal equations
            q = _derHns(P, W, XT, _derns(P, W, XT, d), realConstr) + alpha * d
            np.nan_to_num(q)

            a = dnew / np.real(_scal(d, q))
            z = z + a * (d)
            r = r - a * q
            np.nan_to_num(r)
            dold = np.copy(dnew)
            dnew = np.linalg.norm(r)**2

            d = d * ((dnew / dold)) + r
            np.nan_to_num(d)
            if np.sqrt(dnew) < (1e-2 * dnot):
                break

        print('(', j, ')')

        XN = XN + z

        alpha = alpha / 3

        # postprocessing

        CR = _apweightsns(W, XN[1:, :, :])

        if returnProfiles:
            R[2:, i, :, :] = CR / yscale

        C = (np.conj(CR) * CR).sum(0)

        R[0, i, :, :] = (XN[0, :, :] * np.sqrt(C) / yscale)
        R[1, i, :, :] = np.copy(XN[0, :, :])

    end = time.clock()
    print('done in', round((end - start)), 's')
    return R


def _scal(a, b):
    v = np.sum(np.conj(a) * b)
    return v


def _apweightsns(W, CT):
    C = _nsIfft(W * CT)
    return C


def _apweightsnsH(W, CT):
    C = np.conj(W) * _nsFft(CT)
    return C


def _opns(P, X):
    K = X[0, :, :] * X[1:, :, :]
    K = P * _nsFft(K)
    return K


def _derns(P, W, X0, DX):
    K = X0[0, :, :] * _apweightsns(W, DX[1:, :, :])
    K = K + (DX[0, :, :] * X0[1:, :, :])
    K = P * _nsFft(K)
    return K


def _derHns(P, W, X0, DK, realConstr):

    K = _nsIfft(P * DK)

    if realConstr:
        DXrho = np.sum(np.real(K * np.conj(X0[1:, :, :])), 0)
    else:
        DXrho = np.sum(K * np.conj(X0[1:, :, :]), 0)

    DXc = _apweightsnsH(W, (K * np.conj(X0[0, :, :])))
    DX = np.concatenate(
        (DXrho[None, ...], DXc), axis=0)
    return DX


def _nsFft(M):
    si = M.shape
    a = 1 / (np.sqrt((si[M.ndim - 1])) * np.sqrt((si[M.ndim - 2])))
    K = pyfftw.interfaces.numpy_fft.fft2(
        M, norm=None).dot(a)
    return K


def _nsIfft(M):
    si = M.shape
    a = np.sqrt(si[M.ndim - 1]) * np.sqrt(si[M.ndim - 2])
    K = np.array(pyfftw.interfaces.numpy_fft.ifft2(M, norm=None).dot(a))
    return K  # .T


def _weights(y, x):
    W = np.zeros([y, x])
    for i in range(0, y):
        for j in range(0, x):
            d = ((i) / y - 0.5)**2 + ((j) / x - 0.5)**2
            W[i, j] = 1 / (1 + 220 * d)**16
    return W


def _fftshift2(I):
    if I.ndim >= 3:
        S = np.fft.fftshift(np.fft.fftshift(I, -2), -1)
    else:
        S = np.fft.fftshift(np.fft.fftshift(I, 1), 0)
    return S
