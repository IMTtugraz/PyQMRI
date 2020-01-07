#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This module holds the base model class as well as the class for constraints
  on the unknowns.

Attribues:
  DTYPE (complex64):
    Complex working precission. Currently single precission only.
  DTYPE_real (float32):
    Real working precission. Currently single precission only.
"""
from abc import ABC, abstractmethod
import numpy as np


DTYPE = np.complex64
DTYPE_real = np.float32


class constraints:
    """ Box and real value constraints on the unknowns

    This Class implements box and real value constraints on the unknowns. If
    the unknown is set to be complex, the box constraints act on the magnitude
    of the complex value.

    Attributes:
      min (float): Minimum value
      max (float): Maximum value
      real (bool): Flag if unknown is real (True) or complex (False, default)
    """
    def __init__(self, min_val=-np.inf, max_val=np.inf,
                 real_const=False):
        self.min = min_val
        self.max = max_val
        self.real = real_const

    def update(self, scale):
        """ Updates the constraints according to the scaling of the unknowns

        Args:
          scale (float): The scale which is applied to the unknown
        """
        self.min = self.min / scale
        self.max = self.max / scale


class BaseModel(ABC):
    """ This class serves as abstract base class for all models.

    The base class of all models. It holds all common parameters and implements
    abstract methods which every dervied class needs to specialice.

    Attributes:
      constraints (list of constraints): A list of constraints. One for each
        unknown.
      uk_scale (list of float): A list of scaling factors,
        one for each unknown.
      NScan (bool): Number of Scans
      NSlice (int): Number of Slices
      dimY (int): Imagedimension Y
      dimX (int): Imagedimenson X
      figure (matplotlib.Figure): A Figure instance for plotting.
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
        """ Rescales each unknown by its scaling factor.

        The Rescale function is implemented to rescale simple scaling,
        e.g.:  x*x_scale. More complex transformations need to be implemented
        on a per model basis.

        Args:
          x (numpy.Array): The array of unknowns to be rescaled
        Returns:
          numpy.Array: The rescaled array of unknowns
        """
        tmp_x = np.copy(x)
        for i in range(x.shape[0]):
            tmp_x[i] *= self.uk_scale[i]
        return tmp_x

    def execute_forward(self, x, islice=None):
        """ The forward signal model

        Internally class the pyqmri.model.MODEL.__execute_forward_3D function
        which needs to be implemented in each new model by the user.

        Args:
          x (numpy.Array): The array of unknowns
          islice (int): Optinal slice index if 2D reconstruction is performed.
        Returns:
          numpy.Array: A image series based on the current parameter estimated
            and the sequence related parameters.
        """
        if islice is None:
            return self._execute_forward_3D(x)
        else:
            return self._execute_forward_2D(x, islice)

    def execute_gradient(self, x, islice=None):
        """ The partial derivative with respect to each unknown

        Internally class the pyqmri.model.MODEL._execute_gradient_3D function
        which needs to be implemented in each new model by the user.

        Args:
          x (numpy.Array): The array of unknowns
          islice (int): Optinal slice index if 2D reconstruction is performed.
        Returns:
          numpy.Array: The partial derivatives of each unknown.
        """
        if islice is None:
            return self._execute_gradient_3D(x)
        else:
            return self._execute_gradient_2D(x, islice)


    @abstractmethod
    def _execute_forward_3D(self, x):
        """ Abstract method for 3D forward evaluation of the signal equation.

          This method needs to be implemented in each model.

        Args:
          x (numpy.Array): The array of unknowns
        Returns:
          numpy.Array: A image series based on the current parameter estimated
            and the sequence related parameters.
        """
        ...

    @abstractmethod
    def _execute_gradient_3D(self, x):
        """ Abstract method for 3D evaluation of the partial derivatives.

          This method needs to be implemented in each model.

        Args:
          x (numpy.Array): The array of unknowns
        Returns:
          numpy.Array: The partial derivatives of each unknown.
        """
        ...

    @abstractmethod
    def plot_unknowns(self, x, dim_2D=False):
        """ Abstract method for ploting intermediate results

        Args:
          x (numpy.Array): The array of unknowns
        """
        ...

    @abstractmethod
    def _set_init_scales(self):
        ...
