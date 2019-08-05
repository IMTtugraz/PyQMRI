#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np


DTYPE = np.complex64
DTYPE_real = np.float32


class constraints:
    def __init__(self, min_val=-np.inf, max_val=np.inf,
                 real_const=False):
        self.min = min_val
        self.max = max_val
        self.real = real_const

    def update(self, scale):
        self.min = self.min / scale
        self.max = self.max / scale


class BaseModel(ABC):
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
        tmp_x = np.copy(x)
        for i in range(x.shape[0]):
            tmp_x[i] *= self.uk_scale[i]
        return tmp_x

    def execute_forward(self, x, islice=None):
        if islice is None:
            return self._execute_forward_3D(x)
        else:
            return self._execute_forward_2D(x, islice)

    def execute_gradient(self, x, islice=None):
        if islice is None:
            return self._execute_gradient_3D(x)
        else:
            return self._execute_gradient_2D(x, islice)

    @abstractmethod
    def _execute_forward_2D(self, x, islice):
        ...

    @abstractmethod
    def _execute_gradient_2D(self, x, islice):
        ...

    @abstractmethod
    def _execute_forward_3D(self, x):
        ...

    @abstractmethod
    def _execute_gradient_3D(self, x):
        ...

    @abstractmethod
    def plot_unknowns(self, x, dim_2D=False):
        ...

    @abstractmethod
    def _set_init_scales(self):
        ...
