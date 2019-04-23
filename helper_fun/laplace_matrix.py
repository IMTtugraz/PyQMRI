#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:26:06 2017

@author: omaier
"""

import scipy.sparse as sp


def laplace_matrix(n, m, u, scale, bcond):

    A = sp.lil_matrix((n * m, n * m))
    u = u.flatten()

    # Inner diagonal cubes

    for j in range(n * m):
        A[j, j] = 4  # Set diagonals

    for j in range(n * m - 1):
        A[j + 1, j] = -1
        A[j, j + 1] = -1

    # Off diagonal cubes

    for j in range(n * (m - 1)):
        A[n + j, j] = -1
        A[j, n + j] = -1

    # Periodic boundary extension
    if 'per' in bcond:

        for j in range(n, n * m - 1, n):
            A[j + 1, j] = 0
            A[j, j + 1] = 0

        for j in range(n):  # Top right and lower left block (periodic row)
            A[j, n * (m - 1) + j] = -1
            A[n * (m - 1) + j, j] = -1

        for j in range(0, n * m, n):  # Peroidic column
            A[j, j + (n - 1)] = -1
            A[j + (n - 1), j] = -1

    # Symmetric boundary extension
    elif 'sym' in bcond:

        for j in range(
                0, n * m - 1, n):  # Set boundary of inner diagonal cubes
            A[j, j] = 3
            A[j + n - 1, j + n - 1] = 3

        for j in range(n):  # Reduce first and last diagonal cube
            A[j, j] = A[j, j] - 1
            A[n * (m - 1) + j, n * (m - 1) + j] = A[n *
                                                    (m - 1) + j, n * (m - 1) + j] - 1

        for j in range(n, n * m - 1, n):
            A[j + 1, j] = 0
            A[j, j + 1] = 0

    # Add diagonal entries

        for j in range(n * m):
            A[j, j] = scale * A[j, j] + u[j]

    return A.T
