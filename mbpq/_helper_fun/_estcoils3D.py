#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:28:16 2019

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
import numpy as np
import sys
import ipyparallel as ipp
from mbpq._helper_fun import _nlinvns3D as nlinvns

# % Estimates sensitivities and complex image.
# %(see Martin Uecker: Image reconstruction by regularized nonlinear
# %inversion joint estimation of coil sensitivities and image content)
DTYPE = np.complex64
DTYPE_real = np.float32


def estcoils3D(data, par, file, args):
    ###########################################################################
    # Initiate parallel interface #############################################
    ###########################################################################
    c = ipp.Client()
    nlinvNewtonSteps = 6
    nlinvRealConstr = False
    par["C"] = np.zeros(
        (par["NScan"], par["NC"], par["NSlice"], par["dimY"], par["dimX"]),
        dtype=DTYPE)
    par["phase_map"] = np.zeros(
        (par["NScan"], par["NSlice"], par["dimY"], par["dimX"]), dtype=DTYPE)

    result = []
    for i in range(1):
        sys.stdout.write(
            "Computing coil sensitivity map of Scan %i \r" %
            (i))
        sys.stdout.flush()

        # RADIAL PART
        dview = c[int(np.floor(i * len(c) / par["NScan"]))]
        result.append(
            dview.apply_async(
                nlinvns.nlinvns,
                data[i, ...],
                nlinvNewtonSteps,
                True,
                nlinvRealConstr))

    for i in range(1):
        par["C"][i, ...] = result[i].get()[2:, -1, ...]
        sys.stdout.write("Scan %i done \r"
                         % (i))
        sys.stdout.flush()
        if not nlinvRealConstr:
            par["phase_map"][i, ...] = np.exp(
                1j * np.angle(result[i].get()[0, -1, ...]))

            # standardize coil sensitivity profiles
    sumSqrC = np.sqrt(
        np.sum(
            (par["C"] *
             np.conj(
                par["C"])),
            0))  # 4, 9, 128, 128
    par["InScale"] = sumSqrC
    if par["NC"] == 1:
        par["C"] = sumSqrC
    else:
        par["C"] = par["C"] / sumSqrC
