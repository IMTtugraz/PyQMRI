#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the template base class model."""
from abc import ABC, abstractmethod
import numpy as np


DTYPE = np.complex64
DTYPE_real = np.float32


class constraints:
    """Constraints for a parameter.

    This class holds min/max and real value constrains for a parameter. It
    supports updating these based on the current estimated sclaing between
    each partial derivative.

    Parameters
    ----------
      min_val : float, -numpy.inf
        The minimum value.
      max_val : float, numpy.inf
        The maximum value.
      real_const : bool, false
        Constrain to real values (true) or complex values (false).

    Attributes
    ----------
      constraints : list of pyqmri.models.template.constrains
        An empy list of constrains objects.
      NScan : int
        Number of scans (dynamics).
      NSlice : int
        Number of slices.
      dimX, dimY : int
        The image dimensions.
      figure : matplotlib.pyplot.figure, None
        The placeholder figure object
    """

    def __init__(self, min_val=-np.inf, max_val=np.inf,
                 real_const=False):
        self.min = min_val
        self.max = max_val
        self.real = real_const

    def update(self, scale):
        """Update the constrains based on current scaling factor.

        Parameters
        ----------
          scale : float
            The new scaling factor which should be used.
        """
        self.min = self.min / scale
        self.max = self.max / scale


class BaseModel(ABC):
    """Base model for MRI parameter quantification.

    This class holds the base model to derive other signal models from.
    It defines abstract a forward application of the analytical signal
    expression    and the partial derivatives with respesct to each parameter
    of interest.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the sequence related parametrs,
        e.g. Number of Scans, Slices, and image dimension.

    Attributes
    ----------
      constraints : list of pyqmri.models.template.constrains
        An empy list of constrains objects.
      NScan : int
        Number of scans (dynamics).
      NSlice : int
        Number of slices.
      dimX, dimY : int
        The image dimensions.
      figure : matplotlib.pyplot.figure, None
        The placeholder figure object
    """

    def __init__(self, par):
        super().__init__()
        self.constraints = []
        self.uk_scale = []
        self.NScan = par["NScan"]
        self.NSlice = par["NSlice"]
        self.dimX = par["dimX"]
        self.dimY = par["dimY"]
        self.figure = None

    def rescale(self, x):
        """Rescale the unknowns with the scaling factors.

        Rescales each unknown with the corresponding scaling factor.

        Parameters
        ----------
          x : numpy.array
            The array of unknowns to be rescaled

        Returns
        -------
          numpy.array:
            The rescaled unknowns
        """
        tmp_x = np.copy(x)
        for i in range(x.shape[0]):
            tmp_x[i] *= self.uk_scale[i]
        return tmp_x

    def execute_forward(self, x, islice=None):
        """Execute the signal model from parameter to imagespace.

        This function exectues the given signal model to generate an image
        series, given estimated parameters.

        Parameters
        ----------
          x : numpy.array
            The array of quantitative parameters to be fitted
          islice : int, None
            Currently unused.
        """
        # if islice is None:
        return self._execute_forward_3D(x)

    def execute_gradient(self, x, islice=None):
        """Execute the partial derivatives of the signal model.

        This function exectues the partial derivatives with respect to each
        unknown parameter, based on the signal model.

        Parameters
        ----------
          x : numpy.array
            The array of quantitative parameters to be fitted
          islice : int, None
            Currently unused.
        """
        # if islice is None:
        return self._execute_gradient_3D(x)

    @abstractmethod
    def _execute_forward_3D(self, x):
        ...

    @abstractmethod
    def _execute_gradient_3D(self, x):
        ...

    @abstractmethod
    def plot_unknowns(self, x, dim_2D=False):
        """Plot the unkowns in an interactive figure.

        This function can be used to plot intermediate results during the
        optimization process.

        Parameters
        ----------
          x : numpy.array
            The array of unknowns to be displayed
          dim_2D : bool, false
            Currently unused.
        """
        ...

    @abstractmethod
    def computeInitialGuess(self, *args):
        """Initialize unknown array for the fitting.

        This function provides an initial guess for the fitting.
        """
        ...
