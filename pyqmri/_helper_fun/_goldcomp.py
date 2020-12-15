#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Density compensation for non-uniform acquried data."""
import numpy as np


def cmp(k):
    """Golden angle density compensation.

    Simple linear ramp based density compensation function

    Parameters
    ----------
      k : numpy.array
        Trajectory which should be density compensated.

    Returns
    -------
      numpy.array :
          The density compensation array.
    """
    if len(np.shape(k)) == 2:
        nspokes, N = np.shape(k)
    elif len(np.shape(k)) == 3:
        _, nspokes, N = np.shape(k)
    else:
        return -5

    w = np.abs(np.linspace(-N / 2, N / 2, N))  # -N/2 N/2
    w = w * (np.pi / 4) / nspokes  # no scaling seems to work better??
    w = np.repeat(w, nspokes, 0)
    w = np.reshape(w, (N, nspokes)).T

    return np.array(w)
