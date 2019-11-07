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
import pyopencl.array as clarray
from pyqmri._helper_fun import _nlinvns as nlinvns
from pyqmri._helper_fun import _goldcomp as goldcomp
from pyqmri._helper_fun import _utils as utils

# % Estimates sensitivities and complex image.
# %(see Martin Uecker: Image reconstruction by regularized nonlinear
# %inversion joint estimation of coil sensitivities and image content)
DTYPE = np.complex64
DTYPE_real = np.float32


def est_coils(data, par, file, args, off):
    ###########################################################################
    # Initiate parallel interface #############################################
    ###########################################################################
    c = ipp.Client()
    nlinvNewtonSteps = 6
    nlinvRealConstr = False
    if "Coils_real" in list(file.keys()):
        print("Using precomputed coil sensitivities")
        slices_coils = file['Coils_real'][()].shape[1]
        par["C"] = file['Coils_real'][
            :,
            int(slices_coils / 2) - int(np.floor((par["NSlice"]) / 2)) + off:
            int(slices_coils / 2) + int(np.ceil(par["NSlice"] / 2)) + off,
            ...] + 1j * file['Coils_imag'][
            :,
            int(slices_coils / 2) - int(np.floor((par["NSlice"]) / 2)) + off:
            int(slices_coils / 2) + int(np.ceil(par["NSlice"] / 2)) + off,
            ...]
    elif "Coils" in list(file.keys()):
        if args.trafo and not file['Coils'].shape[1] >= par["NSlice"]:

            traj_coil = np.reshape(
                par["traj"], (par["NScan"] * par["Nproj"], par["N"]))
            dcf_coil = np.sqrt(
                np.array(
                    goldcomp.cmp(traj_coil),
                    dtype=DTYPE))

            par["C"] = np.zeros(
                (par["NC"], par["NSlice"], par["dimY"], par["dimX"]),
                dtype=DTYPE)
            par["phase_map"] = np.zeros(
                (par["NSlice"], par["dimY"], par["dimX"]), dtype=DTYPE)

            par_coils = {}
            par_coils["traj"] = traj_coil
            par_coils["dcf"] = dcf_coil
            par_coils["N"] = par["N"]
            par_coils["NScan"] = 1
            par_coils["NC"] = 1
            par_coils["NSlice"] = 1
            par_coils["ctx"] = par["ctx"]
            par_coils["queue"] = par["queue"]
            par_coils["dimX"] = par["dimX"]
            par_coils["dimY"] = par["dimY"]
            FFT = utils.NUFFT(par_coils)

            result = []
            for i in range(0, (par["NSlice"])):
                sys.stdout.write(
                    "Computing coil sensitivity map of slice %i \r" %
                    (i))
                sys.stdout.flush()

                # RADIAL PART
                combinedData = np.transpose(
                    data[:, :, i, :, :], (1, 0, 2, 3))
                combinedData = np.require(
                    np.reshape(
                        combinedData,
                        (1,
                         par["NC"],
                            1,
                            par["NScan"] * par["Nproj"],
                            par["N"])),
                    requirements='C') * dcf_coil
                tmp_coilData = clarray.zeros(
                    FFT.queue, (1, 1, 1, par["dimY"], par["dimX"]),
                    dtype=DTYPE)
                coilData = np.zeros(
                    (par["NC"], par["dimY"], par["dimX"]), dtype=DTYPE)
                for j in range(par["NC"]):
                    tmp_combinedData = clarray.to_device(
                        FFT.queue, combinedData[None, :, j, ...])
                    FFT.FFTH(tmp_coilData, tmp_combinedData)
                    coilData[j, ...] = np.squeeze(tmp_coilData.get())

                combinedData = np.require(
                    np.fft.fft2(
                        coilData,
                        norm=None) /
                    np.sqrt(
                        par["dimX"] *
                        par["dimY"]),
                    dtype=DTYPE,
                    requirements='C')

                dview = c[int(np.floor(i * len(c) / par["NSlice"]))]
                result.append(
                    dview.apply_async(
                        nlinvns.nlinvns,
                        combinedData,
                        nlinvNewtonSteps,
                        True,
                        nlinvRealConstr))

            for i in range(par["NSlice"]):
                par["C"][:, i, :, :] = result[i].get()[2:, -1, :, :]
                sys.stdout.write("slice %i done \r"
                                 % (i))
                sys.stdout.flush()
                if not nlinvRealConstr:
                    par["phase_map"][i, :, :] = np.exp(
                        1j * np.angle(result[i].get()[0, -1, :, :]))
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
                par["C"] = par["C"] / \
                    np.tile(sumSqrC, (par["NC"], 1, 1, 1))
            del file['Coils']
            del FFT
            file.create_dataset(
                "Coils",
                par["C"].shape,
                dtype=par["C"].dtype,
                data=par["C"])
            file.flush()
        elif not args.trafo and not \
                file['Coils'].shape[1] >= par["NSlice"]:

            par["C"] = np.zeros(
                (par["NC"], par["NSlice"], par["dimY"], par["dimX"]),
                dtype=DTYPE)
            par["phase_map"] = np.zeros(
                (par["NSlice"], par["dimY"], par["dimX"]), dtype=DTYPE)

            result = []
            combinedData = np.sum(data, 0)
            for i in range(0, (par["NSlice"])):
                sys.stdout.write(
                    "Computing coil sensitivity map of slice %i \r" %
                    (i))
                sys.stdout.flush()
                tmp = combinedData[:, i, ...]
                dview = c[int(np.floor(i * len(c) / par["NSlice"]))]
                result.append(
                    dview.apply_async(
                        nlinvns.nlinvns,
                        tmp,
                        nlinvNewtonSteps,
                        True,
                        nlinvRealConstr))

            for i in range(par["NSlice"]):
                par["C"][:, i, :, :] = result[i].get()[2:, -1, :, :]
                sys.stdout.write("slice %i done \r"
                                 % (i))
                sys.stdout.flush()
                if not nlinvRealConstr:
                    par["phase_map"][i, :, :] = np.exp(
                        1j * np.angle(result[i].get()[0, -1, :, :]))

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
                par["C"] = par["C"] / \
                    np.tile(sumSqrC, (par["NC"], 1, 1, 1))
            del file['Coils']

            file.create_dataset(
                "Coils",
                par["C"].shape,
                dtype=par["C"].dtype,
                data=par["C"])
            file.flush()
        else:
            print("Using precomputed coil sensitivities")
            slices_coils = file['Coils'].shape[1]
            par["C"] = \
                file['Coils'][
                    :,
                    int(slices_coils / 2) -
                    int(np.floor((par["NSlice"]) / 2)) + off:
                    int(slices_coils / 2) +
                    int(np.ceil(par["NSlice"] / 2)) + off, ...]

    else:
        if args.trafo:

            traj_coil = np.reshape(
                par["traj"], (par["NScan"] * par["Nproj"], par["N"]))
            dcf_coil = np.sqrt(np.array(goldcomp.cmp(traj_coil), dtype=DTYPE))

            par["C"] = np.zeros(
                (par["NC"],
                 par["NSlice"],
                    par["dimY"],
                    par["dimX"]),
                dtype=DTYPE)
            par["phase_map"] = np.zeros(
                (par["NSlice"], par["dimY"], par["dimX"]), dtype=DTYPE)

            par_coils = {}
            par_coils["traj"] = traj_coil
            par_coils["dcf"] = dcf_coil
            par_coils["dimX"] = par["dimX"]
            par_coils["dimY"] = par["dimY"]
            par_coils["N"] = par["N"]
            par_coils["NScan"] = 1
            par_coils["NC"] = 1
            par_coils["NSlice"] = 1
            par_coils["ctx"] = par["ctx"]
            par_coils["queue"] = par["queue"]

            FFT = utils.NUFFT(par_coils)

            result = []
            for i in range(0, (par["NSlice"])):
                sys.stdout.write(
                    "Computing coil sensitivity map of slice %i \r" %
                    (i))
                sys.stdout.flush()

                combinedData = np.transpose(data[:, :, i, :, :], (1, 0, 2, 3))
                combinedData = np.require(
                    np.reshape(
                        combinedData,
                        (1,
                         par["NC"],
                            1,
                            par["NScan"] * par["Nproj"],
                            par["N"])),
                    requirements='C') * dcf_coil
                tmp_coilData = clarray.zeros(
                    FFT.queue, (1, 1, 1, par["dimY"], par["dimX"]),
                    dtype=DTYPE)
                coilData = np.zeros(
                    (par["NC"], par["dimY"], par["dimX"]), dtype=DTYPE)
                for j in range(par["NC"]):
                    tmp_combinedData = clarray.to_device(
                        FFT.queue, combinedData[None, :, j, ...])
                    FFT.FFTH(tmp_coilData, tmp_combinedData)
                    coilData[j, ...] = np.squeeze(tmp_coilData.get())

                combinedData = np.require(
                    np.fft.fft2(
                        coilData,
                        norm=None) /
                    np.sqrt(
                        par["dimX"] *
                        par["dimY"]),
                    dtype=DTYPE,
                    requirements='C')

#                result = nlinvns.nlinvns(
#                            combinedData,
#                            nlinvNewtonSteps,
#                            True,
#                            nlinvRealConstr)
#
#                par["C"][:, i, :, :] = result[2:, -1, :, :]
#                sys.stdout.write("slice %i done \r"
#                                 % (i))
#                sys.stdout.flush()
#                if not nlinvRealConstr:
#                    par["phase_map"][i, :, :] = np.exp(
#                        1j * np.angle(result[0, -1, :, :]))

                dview = c[int(np.floor(i * len(c) / par["NSlice"]))]
                result.append(
                    dview.apply_async(
                        nlinvns.nlinvns,
                        combinedData,
                        nlinvNewtonSteps,
                        True,
                        nlinvRealConstr))

            for i in range(par["NSlice"]):
                par["C"][:, i, :, :] = result[i].get()[2:, -1, :, :]
                sys.stdout.write("slice %i done \r"
                                 % (i))
                sys.stdout.flush()
                if not nlinvRealConstr:
                    par["phase_map"][i, :, :] = np.exp(
                        1j * np.angle(result[i].get()[0, -1, :, :]))

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
                par["C"] = par["C"] / np.tile(sumSqrC, (par["NC"], 1, 1, 1))
            del FFT
        else:

            par["C"] = np.zeros(
                (par["NC"],
                 par["NSlice"],
                    par["dimY"],
                    par["dimX"]),
                dtype=DTYPE)
            par["phase_map"] = np.zeros(
                (par["NSlice"], par["dimY"], par["dimX"]), dtype=DTYPE)

            result = []
            combinedData = np.sum(data, 0)
            for i in range(0, (par["NSlice"])):
                sys.stdout.write(
                    "Computing coil sensitivity map of slice %i \r" %
                    (i))
                sys.stdout.flush()

#                result = nlinvns.nlinvns(
#                            combinedData[:, i, ...],
#                            nlinvNewtonSteps,
#                            True,
#                            nlinvRealConstr)
#
#                par["C"][:, i, :, :] = result[2:, -1, :, :]
#                sys.stdout.write("slice %i done \r"
#                                 % (i))
#                sys.stdout.flush()
#                if not nlinvRealConstr:
#                    par["phase_map"][i, :, :] = np.exp(
#                        1j * np.angle(result[0, -1, :, :]))

                # RADIAL PART
                tmp = combinedData[:, i, ...]
                dview = c[int(np.floor(i * len(c) / par["NSlice"]))]
                result.append(
                    dview.apply_async(
                        nlinvns.nlinvns,
                        tmp,
                        nlinvNewtonSteps,
                        True,
                        nlinvRealConstr))

            for i in range(par["NSlice"]):
                par["C"][:, i, :, :] = result[i].get()[2:, -1, :, :]
                sys.stdout.write("slice %i done \r"
                                 % (i))
                sys.stdout.flush()
                if not nlinvRealConstr:
                    par["phase_map"][i, :, :] = np.exp(
                        1j * np.angle(result[i].get()[0, -1, :, :]))

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
                par["C"] = par["C"] / np.tile(sumSqrC, (par["NC"], 1, 1, 1))
        file.create_dataset(
            "Coils",
            par["C"].shape,
            dtype=par["C"].dtype,
            data=par["C"])
        file.flush()
