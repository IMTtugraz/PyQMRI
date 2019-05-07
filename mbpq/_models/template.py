#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 08:55:35 2019

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
