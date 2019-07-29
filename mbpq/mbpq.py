#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
import os
import h5py
import time
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
import importlib

import pyopencl as cl
import argparse

from mbpq._helper_fun import _goldcomp as goldcomp
from mbpq._helper_fun._est_coils import est_coils
from mbpq._helper_fun import _utils as utils

DTYPE = np.complex64
DTYPE_real = np.float32


def _setupOCL(myargs, par):
    platforms = cl.get_platforms()
    par["GPU"] = False
    par["Platform_Indx"] = 0
    if myargs.use_GPU:
        for j in range(len(platforms)):
            if platforms[j].get_devices(device_type=cl.device_type.GPU):
                print("GPU OpenCL platform <%s> found "
                      "with %i device(s) and OpenCL-version <%s>"
                      % (str(platforms[j].get_info(cl.platform_info.NAME)),
                         len(platforms[j].get_devices(
                            device_type=cl.device_type.GPU)),
                         str(platforms[j].get_info(cl.platform_info.VERSION))))
                par["GPU"] = True
                par["Platform_Indx"] = j
    if not par["GPU"]:
        if myargs.use_GPU:
            print("No GPU OpenCL platform found. Falling back to CPU.")
        for j in range(len(platforms)):
            if platforms[j].get_devices(device_type=cl.device_type.CPU):
                print("CPU OpenCL platform <%s> found "
                      "with %i device(s) and OpenCL-version <%s>"
                      % (str(platforms[j].get_info(cl.platform_info.NAME)),
                         len(platforms[j].get_devices(
                            device_type=cl.device_type.GPU)),
                         str(platforms[j].get_info(cl.platform_info.VERSION))))
                par["GPU"] = False
                par["Platform_Indx"] = j

    par["ctx"] = []
    par["queue"] = []
    if type(myargs.devices) == int:
        myargs.devices = [myargs.devices]
    if myargs.streamed:
        if len(myargs.devices) == 1 and myargs.devices[0] == -1:
            num_dev = []
            for j in range(len(platforms[par["Platform_Indx"]].get_devices())):
                num_dev.append(j)
            par["num_dev"] = num_dev
        else:
            num_dev = myargs.devices
            par["num_dev"] = num_dev
    else:
        num_dev = myargs.devices
        par["num_dev"] = num_dev
    for device in num_dev:
        dev = []
        dev.append(platforms[par["Platform_Indx"]].get_devices()[device])
        tmp = cl.Context(dev)
        par["ctx"].append(tmp)
        par["queue"].append(
         cl.CommandQueue(
          tmp,
          platforms[par["Platform_Indx"]].get_devices()[device],
          properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
          | cl.command_queue_properties.PROFILING_ENABLE))
        par["queue"].append(
         cl.CommandQueue(
          tmp,
          platforms[par["Platform_Indx"]].get_devices()[device],
          properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
          | cl.command_queue_properties.PROFILING_ENABLE))
        par["queue"].append(
         cl.CommandQueue(
          tmp,
          platforms[par["Platform_Indx"]].get_devices()[device],
          properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
          | cl.command_queue_properties.PROFILING_ENABLE))
        par["queue"].append(
         cl.CommandQueue(
          tmp,
          platforms[par["Platform_Indx"]].get_devices()[device],
          properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
          | cl.command_queue_properties.PROFILING_ENABLE))


def _genImages(myargs, par, data):
    #    #    par['C'] = np.require(par['C'][:,16,...][:,None,...],
    #                               requirements='C')
    #    FFT = utils.NUFFT(par, trafo=myargs.trafo, SMS=myargs.sms)
    #    #    data = data/(FFT.deblurring[None, None, None, :, None])
    #    import pyopencl.array as clarray
    #
    #    def nFTH(x, fft, par):
    #        siz = np.shape(x)
    #        MB = int(par["MB"])
    #        result = np.zeros(
    #            (par["NC"], par["NSlice"], par["NScan"],
    #             par["dimY"], par["dimX"]), dtype=DTYPE)
    #        tmp_result = clarray.empty(fft.queue, (par["NScan"], 1, MB,
    #                                   par["dimY"], par["dimX"]), dtype=DTYPE)
    #        for j in range(siz[1]):
    #            for k in range(siz[2]):
    #                inp = clarray.to_device(fft.queue,
    #                                        np.require(x[:, j, k, ...]
    #                                                    [:, None, None, ...],
    #                                                   requirements='C'))
    #                fft.FFTH(tmp_result, inp)
    #                if myargs.sms:
    #                    ind = slice(k, siz[2]+k+1, siz[2])
    #                    result[j, ind, ...] = np.transpose(
    #                        (tmp_result.get()), (1, 2, 0, 3, 4))
    #                else:
    #                    result[j, k, ...] = np.squeeze(tmp_result.get())
    #        return np.transpose(result, (2, 0, 1, 3, 4))
    #    images = np.require(np.sum(nFTH(data, FFT, par) *
    #                               (np.conj(par["C"])), axis=1),
    #                        requirements='C')
    #    del FFT, nFTH

    #    del par["file"]["images_cg"]
    if "images_cg" not in list(par["file"].keys()):
        from mbpq.solver import CGSolver
        images = np.zeros((par["NScan"],
                           par["NSlice"],
                           par["dimY"],
                           par["dimX"]), dtype=DTYPE)
        if par["NScan"]/10 >= 1:
            cgs = CGSolver(par, 10, myargs.trafo, myargs.sms)
            for j in range(int(par["NScan"]/10)):
                if par["NSlice"] == 1:
                    images[10*j:10*(j+1), ...] = cgs.run(
                        data[10*j:10*(j+1), ...],
                        tol=1e-3)[:, None, ...]
                else:
                    images[10*j:10*(j+1), ...] = cgs.run(
                        data[10*j:10*(j+1), ...],
                        tol=1e-3)
            del cgs
        if np.mod(par["NScan"], 10):
            cgs = CGSolver(par, np.mod(par["NScan"], 10),
                           myargs.trafo, myargs.sms)
            if par["NSlice"] == 1:
                if np.mod(par["NScan"], 10) == 1:
                    images[-np.mod(par["NScan"], 10):, ...] = cgs.run(
                            data[-np.mod(par["NScan"], 10):, ...],
                            tol=1e-3)
                else:
                    images[-np.mod(par["NScan"], 10):, ...] = cgs.run(
                            data[-np.mod(par["NScan"], 10):, ...],
                            tol=1e-3)[:, None, ...]
            else:
                images[-np.mod(par["NScan"], 10):, ...] = cgs.run(
                        data[-np.mod(par["NScan"], 10):, ...],
                        tol=1e-3)
            del cgs
        par["file"].create_dataset("images_cg", images.shape,
                                   dtype=DTYPE, data=images)
    else:
        images = par["file"]['images_cg']
        if images.shape[1] < par["NSlice"]:
            del par["file"]["images_cg"]
            images = _genImages(myargs, par, data)
        else:
            print("Using precomputed images")
            slices_images = par["file"]['images_cg'][()].shape[1]
            images = \
                par["file"]['images_cg'][
                    :, int(slices_images / 2) - int(
                        np.floor((par["NSlice"]) / 2)):int(
                            slices_images / 2) + int(
                                np.ceil(par["NSlice"] / 2)), ...].astype(DTYPE)
#    mask = np.ones_like(images)
#    for j in range(mask.shape[0]):
#        mask[j, images[0, ...] < 20] = 0
#    images = images*mask
#    images[mask == 1] = np.log(np.abs(images[mask == 1])).astype(DTYPE)
    return images


def _estScaleNorm(myargs, par, images, data):
    if myargs.imagespace:
        dscale = DTYPE_real(np.sqrt(2*1e3*par["NSlice"]/par["MB"]) /
                            (np.linalg.norm(images.flatten())))
        par["dscale"] = dscale
        images = images*dscale
    else:
        dscale = (DTYPE_real(np.sqrt(2*1e3*par["NSlice"]/par["MB"])) /
                  (np.linalg.norm(data.flatten())))
        par["dscale"] = dscale
        data = data*dscale

    if myargs.trafo:
        center = int(par["N"]*0.1)
        sig = []
        noise = []
        ind = np.zeros((par["N"]), dtype=bool)
        ind[int(par["N"]/2-center):int(par["N"]/2+center)] = 1
        for j in range(par["Nproj"]):
            sig.append(np.sum(
                data[..., int(par["NSlice"]/2), j, ind] *
                np.conj(data[..., int(par["NSlice"]/2), j, ind])))
            noise.append(np.sum(
                data[..., int(par["NSlice"]/2), j, ~ind] *
                np.conj(data[..., int(par["NSlice"]/2), j, ~ind])))
        sig = (np.sum(np.array(sig)))/np.sum(ind)
        noise = (np.sum(np.array(noise)))/np.sum(~ind)
        SNR_est = np.abs(sig/noise)
        par["SNR_est"] = SNR_est
        print("Estimated SNR from kspace", SNR_est)
    else:
        centerX = int(par["dimX"]*0.2)
        centerY = int(par["dimY"]*0.2)
        ind = np.zeros((par["dimY"], par["dimX"]), dtype=bool)
        ind[int(par["N"]/2-centerY):int(par["N"]/2+centerY),
            int(par["N"]/2-centerX):int(par["N"]/2+centerX)] = 1
        ind = np.fft.fftshift(ind)
        sig = np.sum(
            data[..., int(par["NSlice"]/2/par["MB"]), ind] *
            np.conj(
                data[..., int(par["NSlice"]/2/par["MB"]), ind]))/np.sum(ind)
        noise = np.sum(
            data[..., int(par["NSlice"]/2/par["MB"]), ~ind] *
            np.conj(
                data[..., int(par["NSlice"]/2/par["MB"]), ~ind]))/np.sum(~ind)
        SNR_est = np.abs(sig/noise)
        par["SNR_est"] = SNR_est
        print("Estimated SNR from kspace", SNR_est)
    return data, images


def _readInput(myargs, par):
    if myargs.file == '':
        select_file = True
        while select_file is True:
            root = Tk()
            root.withdraw()
            root.update()
            file = filedialog.askopenfilename()
            root.destroy()
            if not file == () and \
               not file.endswith((('.h5'), ('.hdf5'))):
                print("Please specify a h5 file. Press cancel to exit.")
            elif file == ():
                print("Exiting...")
                return 0
            else:
                select_file = False
    else:
        if not myargs.file.endswith((('.h5'), ('.hdf5'))):
            print("Please specify a h5 file. ")
            return 0
        file = myargs.file

    name = os.path.normpath(file)
    par["fname"] = name.split(os.sep)[-1]
    outdir = time.strftime("%Y-%m-%d  %H-%M-%S_MRI_" +
                           par["fname"][:-3])
    if not os.path.exists('./output'):
        os.makedirs('./output')
    if not os.path.exists('./output/' + outdir):
        os.makedirs("output/" + outdir)
    par["outdir"] = './output/'+outdir+'/'
    par["file"] = h5py.File(file)


def start_recon(myargs):
    sig_model_path = os.path.normpath(myargs.sig_model)
    if len(sig_model_path.split(os.sep)) > 1:
        spec = importlib.util.spec_from_file_location(
            sig_model_path.split(os.sep)[-1], sig_model_path)
        sig_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sig_model)
    else:
        sig_model = importlib.import_module(
            "mbpq._models."+str(sig_model_path))
    if int(myargs.streamed) == 1:
        import mbpq._irgn._reco_streamed as optimizer
    else:
        import mbpq._irgn._reco as optimizer
    np.seterr(divide='ignore', invalid='ignore')

# Create par struct to store everyting
    par = {}
###############################################################################
# Select input file ###########################################################
###############################################################################
    _readInput(myargs, par)
###############################################################################
# Read Data ###################################################################
###############################################################################
    reco_Slices = myargs.slices
    dimX, dimY, NSlice = ((par["file"].attrs['image_dimensions']).astype(int))
    if reco_Slices == -1:
        reco_Slices = NSlice
    off = 0
    if myargs.sms:
        data = par["file"]['real_dat'][()].astype(DTYPE)\
               + 1j*par["file"]['imag_dat'][()].astype(DTYPE)
    else:
        data = par["file"]['real_dat'][
          ..., int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:
          int(NSlice/2)+int(np.ceil(reco_Slices/2))+off, :, :].astype(DTYPE)\
               + 1j*par["file"]['imag_dat'][
          ..., int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:
          int(NSlice/2)+int(np.ceil(reco_Slices/2))+off, :, :].astype(DTYPE)
    dimreduction = 0
    if myargs.trafo:
        par["traj"] = par["file"]['real_traj'][()].astype(DTYPE) + \
                      1j*par["file"]['imag_traj'][()].astype(DTYPE)

        par["dcf"] = np.sqrt(np.array(goldcomp.cmp(
                         par["traj"]), dtype=DTYPE_real)).astype(DTYPE_real)
        par["dcf"] = np.require(np.abs(par["dcf"]),
                                DTYPE_real, requirements='C')
    else:
        par["traj"] = None
        par["dcf"] = None
    if np.max(utils.prime_factors(data.shape[-1])) > 13:
        print("Samples along the spoke need to have their largest prime factor"
              " to be 13 or lower. Finding next smaller grid.")
        dimreduction = 2
        while np.max(utils.prime_factors(data.shape[-1]-dimreduction)) > 13:
            dimreduction += 2
        print('Decrease grid size by %i' % dimreduction)
        dimX -= dimreduction
        dimY -= dimreduction
        if myargs.trafo:
            data = np.require(data[..., int(dimreduction/2):
                                   data.shape[-1]-int(dimreduction/2)],
                              requirements='C')
            par["traj"] = np.require(par["traj"][
                ..., int(dimreduction/2):
                par["traj"].shape[-1]-int(dimreduction/2)], requirements='C')
            par["dcf"] = np.sqrt(np.array(goldcomp.cmp(par["traj"]),
                                          dtype=DTYPE_real)).astype(DTYPE_real)
            par["dcf"] = np.require(np.abs(par["dcf"]),
                                    DTYPE_real, requirements='C')
        else:
            data = np.require(data[...,
                                   int(dimreduction/2):
                                   data.shape[-1]-int(dimreduction/2),
                                   int(dimreduction/2):
                                   data.shape[-1]-int(dimreduction/2)],
                              requirements='C')

###############################################################################
# FA correction ###############################################################
###############################################################################
    if "fa_corr" in list(par["file"].keys()):
        print("Using provied flip angle correction.")
        if myargs.sms:
            par["fa_corr"] = np.flip(par["file"]['fa_corr'][()].astype(DTYPE),
                                     0)[...]
        else:
            NSlice_fa, dimY_fa, dimX_fa = par["file"]['fa_corr'][()].shape
            par["fa_corr"] = np.flip(
                par["file"]['fa_corr'][()].astype(DTYPE),
                0)[
                  int(NSlice_fa/2)-int(np.floor((reco_Slices)/2)):
                  int(NSlice_fa/2)+int(np.ceil(reco_Slices/2)),
                  ...]
        par["fa_corr"][par["fa_corr"] == 0] = 0
        par["fa_corr"] = par["fa_corr"][
           ...,
           int(dimreduction/2):par["fa_corr"].shape[-2]-int(dimreduction/2),
           int(dimreduction/2):par["fa_corr"].shape[-1]-int(dimreduction/2)]
        # Recheck if shifted/transposed correctly
        par["fa_corr"] = np.require((np.transpose(par["fa_corr"], (0, 2, 1))),
                                    requirements='C')
    elif "interpol_fa" in list(par["file"].keys()):
        print("Using provied flip angle correction.")
        if myargs.sms:
            par["fa_corr"] = np.flip(
                par["file"]['interpol_fa'][()].astype(DTYPE),
                0)[...]
        else:
            NSlice_fa, dimY_fa, dimX_fa = par["file"]['interpol_fa'][()].shape
            par["fa_corr"] = np.flip(
                par["file"]['interpol_fa'][()].astype(DTYPE),
                0)[
                  int(NSlice_fa/2)-int(np.floor((reco_Slices)/2)):
                  int(NSlice_fa/2)+int(np.ceil(reco_Slices/2)),
                  ...]
        par["fa_corr"][par["fa_corr"] == 0] = 0
        par["fa_corr"] = par["fa_corr"][
           ...,
           int(dimreduction/2):par["fa_corr"].shape[-2]-int(dimreduction/2),
           int(dimreduction/2):par["fa_corr"].shape[-1]-int(dimreduction/2)]
        # Recheck if shifted/transposed correctly
        par["fa_corr"] = np.require((np.transpose(par["fa_corr"], (0, 2, 1))),
                                    requirements='C')
    else:
        print("No flip angle correction provided/used.")

    if data.ndim == 5:
        [NScan, NC, reco_Slices, Nproj, N] = data.shape
    elif data.ndim == 4 and "IRLL" in myargs.sig_model:
        print("4D Data passed and IRLL model used. Reordering Projections "
              "into 8 Spokes/Frame")
        [NC, reco_Slices, Nproj, N] = data.shape
        Nproj_new = 13
        NScan = np.floor_divide(Nproj, Nproj_new)
        par["Nproj_measured"] = Nproj
        Nproj = Nproj_new
        data = np.require(np.transpose(np.reshape(data[..., :Nproj*NScan, :],
                                       (NC, reco_Slices, NScan, Nproj, N)),
                                       (2, 0, 1, 3, 4)), requirements='C')
        par["traj"] = np.require(np.reshape(par["traj"][:Nproj*NScan, :],
                                 (NScan, Nproj, N)), requirements='C')
        par["dcf"] = np.sqrt(np.array(goldcomp.cmp(par["traj"]),
                                      dtype=DTYPE_real)).astype(DTYPE_real)
        par["dcf"] = np.require(np.abs(par["dcf"]), DTYPE_real,
                                requirements='C')
    else:
        print("Wrong data dimension / model inkompatible. Returning")
        return
###############################################################################
# Set sequence related parameters #############################################
###############################################################################
    for att in par["file"].attrs:
        par[att] = par["file"].attrs[att]

    par["NC"] = NC
    par["dimY"] = dimY
    par["dimX"] = dimX
    if myargs.sms:
        par["NSlice"] = NSlice
        par["packs"] = int(par["packs"])
        par["numofpacks"] = int(NSlice/(int(par["packs"])*int(par["MB"])))
    else:
        par["NSlice"] = reco_Slices
        par["packs"] = 1
        par["MB"] = 1
    par["NScan"] = NScan
    par["N"] = N
    par["Nproj"] = Nproj
    par["unknowns_TGV"] = sig_model.unknowns_TGV
    par["unknowns_H1"] = sig_model.unknowns_H1
    par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]
    par["imagespace"] = myargs.imagespace
    if myargs.weights is None:
        par["weights"] = np.ones((par["unknowns"]), dtype=np.float32)
    else:
        par["weights"] = np.array(myargs.weights, dtype=np.float32)
    if myargs.streamed:
        par["par_slices"] = myargs.par_slices
    if not myargs.trafo:
        tmp = np.ones_like(np.abs(data[0, 0,  ...]))
        tmp[np.abs(data[0, 0,  ...]) == 0] = 0
        par['mask'] = np.reshape(
            tmp,
            (data[0, 0, ...].shape)).astype(DTYPE_real)
        del tmp
    else:
        par['mask'] = None
###############################################################################
# ratio of z direction to x,y, important for finite differences ###############
###############################################################################
    par["dz"] = myargs.dz
###############################################################################
# Create OpenCL Context and Queues ############################################
###############################################################################
    _setupOCL(myargs, par)
###############################################################################
# Coil Sensitivity Estimation #################################################
###############################################################################

    est_coils(data, par, par["file"], myargs)
###############################################################################
# Standardize data ############################################################
###############################################################################
    [NScan, NC, NSlice, Nproj, N] = data.shape
    if myargs.trafo:
        if par["file"].attrs['data_normalized_with_dcf']:
            pass
        else:
            data = data*(par["dcf"])

    if NC == 1:
        par['C'] = np.ones((data[0, ...].shape), dtype=DTYPE)
    else:
        par['C'] = par['C'].astype(DTYPE)
###############################################################################
# Reconstruct images using CG-SENSE  ##########################################
###############################################################################
    images = _genImages(myargs, par, data)
###############################################################################
# Scale data norm  ############################################################
###############################################################################
    data, images = _estScaleNorm(myargs, par, images, data)
###############################################################################
# initialize operator  ########################################################
###############################################################################
    opt = optimizer.ModelReco(par, myargs.trafo,
                              imagespace=myargs.imagespace,
                              SMS=myargs.sms)
    if myargs.imagespace:
        opt.data = images
    else:
        opt.data = data
    f = h5py.File(par["outdir"]+"output_" + par["fname"], "a")
    f.create_dataset("images_ifft_", data=images)
    f.attrs['data_norm'] = par["dscale"]
    f.close()
###############################################################################
# Start Reco ##################################################################
###############################################################################
    if myargs.type == '3D':
        #######################################################################
        # Init forward model and initial guess ################################
        #######################################################################
        model = sig_model.Model(par, images)
        par["file"].close()
        #######################################################################
        # IRGN - TGV Reco #####################################################
        #######################################################################
        if "TGV" in myargs.reg or myargs.reg == 'all':
            opt.model = model
            ###################################################################
            # IRGN Params #####################################################
            ###################################################################
            opt.irgn_par = utils.read_config(myargs.config, "3D_TGV")
            opt.execute(TV=0, imagespace=myargs.imagespace)
            plt.close('all')
        #######################################################################
        # IRGN - TV Reco ######################################################
        #######################################################################
        if "TV" in myargs.reg or myargs.reg == 'all':
            opt.model = model
            ###################################################################
            # IRGN Params #####################################################
            ###################################################################
            opt.irgn_par = utils.read_config(myargs.config, "3D_TV")
            opt.execute(TV=1, imagespace=myargs.imagespace)
            plt.close('all')
        del opt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(recon_type='3D', reg_type='TGV', slices=1, trafo=False,
         streamed=False,
         par_slices=1, data='', model='VFA', config='default',
         imagespace=False,
         OCL_GPU=True, sms=False, devices=0, dz=1, weights=None):
    """
    Start a 3D model based reconstruction. Data can also be selected at
    start up. \n\n
    recon_type: 3D (2D currently not supported but 3D works on one slice also)
    \n
    reg_type: TGV (default) or TV\n
    slices: 1 (pass -1 for all). Slices are picked symmetrically from
    the volume center\n
    trafo: 1 (radial, default), 0 Cartesian\n
    streamed: 0 (default, everything on the GPU), 1 streamed packages\n
    par_slices: 1 number of slices per package. Volume devided by GPU's and
    par_slices must be an even number!
    (Stream packages to the GPU)\n
    data: '' (Select at start) or pass the full path to the .h5 raw data\n
    model: 'VFA' or pass the path to your own model file.\n
    config: 'default' or the name of your own .ini\n
    (Assumed to be in the current folder)\n
    imagespace: 0 (perform k-space reconstruction, default) or 1\n
    OCL_GPU: 1 (Use GPU, default, CAVE: CPU FFT not working)\n
    sms: 1 use Simultaneous Multi Slice Recon (default 0)
    devices: 1 Device ID of device(s) to use for streaming\n
    dz: Ratio of physical Z to X/Y dimension.
    X/Y is assumed to be isotropic.\n
    """
    parser = argparse.ArgumentParser(description="T1 quantification from VFA "
                                                 "data. By default runs 3D "
                                                 "regularization for TGV.")
    parser.add_argument(
      '--recon_type', default=recon_type, dest='type',
      help='Choose reconstruction type (currently only 3D)')
    parser.add_argument(
      '--reg_type', default=reg_type, dest='reg',
      help="Choose regularization type (default: TGV) "
           "options are: TGV, TV, all")
    parser.add_argument(
      '--slices', default=slices, dest='slices', type=int,
      help="Number of reconstructed slices (default=40). "
           "Symmetrical around the center slice.")
    parser.add_argument(
      '--trafo', default=trafo, dest='trafo', type=str2bool,
      help='Choos between radial (1, default) and Cartesian (0) sampling. ')
    parser.add_argument(
      '--streamed', default=streamed, dest='streamed', type=str2bool,
      help='Enable streaming of large data arrays (e.g. >10 slices).')
    parser.add_argument(
      '--par_slices', default=par_slices, dest='par_slices', type=int,
      help='number of slices per package. Volume devided by GPU\'s and'
           ' par_slices must be an even number!')
    parser.add_argument(
      '--data', default=data, dest='file',
      help="Full path to input data. "
           "If not provided, a file dialog will open.")
    parser.add_argument(
      '--model', default=model, dest='sig_model',
      help="Name of the signal model to use. Defaults to VFA. "
           "Please put your signal model file in the Model subfolder.")
    parser.add_argument(
      '--config', default=config, dest='config',
      help="Name of config file to use (assumed to be in the same folder). "
           "If not specified, use default parameters.")
    parser.add_argument(
      '--imagespace', default=imagespace, dest='imagespace', type=str2bool,
      help="Select if Reco is performed on images (1) or on kspace (0) data. "
           "Defaults to 0")
    parser.add_argument(
      '--sms', default=sms, dest='sms', type=str2bool,
      help="Switch to SMS reconstruction")
    parser.add_argument(
      '--OCL_GPU', default=OCL_GPU, dest='use_GPU', type=str2bool,
      help="Select if CPU or GPU should be used as OpenCL platform. "
           "Defaults to GPU (1). CAVE: CPU FFT not working")
    parser.add_argument(
      '--devices', default=devices, dest='devices', type=int,
      help="Device ID of device(s) to use for streaming. "
           "-1 selects all available devices", nargs='*')
    parser.add_argument(
      '--dz', default=dz, dest='dz', type=float,
      help="Ratio of physical Z to X/Y dimension. "
           "X/Y is assumed to be isotropic. Defaults to 1")
    parser.add_argument(
      '--weights', default=weights, dest='weights', type=float,
      help="Ratio of unkowns to each other. Defaults to 1. "
           "If passed, needs to be in the same size as the number of unknowns",
           nargs='*')
    args = parser.parse_args()
    start_recon(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="T1 quantification from VFA "
                                                 "data. By default runs 3D "
                                                 "regularization for TGV.")
    parser.add_argument(
      '--recon_type', default='3D', dest='type',
      help='Choose reconstruction type (currently only 3D)')
    parser.add_argument(
      '--reg_type', default='TGV', dest='reg',
      help="Choose regularization type (default: TGV) "
           "options are: TGV, TV, all")
    parser.add_argument(
      '--slices', default=-1, dest='slices', type=int,
      help="Number of reconstructed slices (default=40). "
           "Symmetrical around the center slice.")
    parser.add_argument(
      '--trafo', default=False, dest='trafo', type=str2bool,
      help='Choos between radial (1, default) and Cartesian (0) sampling. ')
    parser.add_argument(
      '--streamed', default=False, dest='streamed', type=str2bool,
      help='Enable streaming of large data arrays (e.g. >10 slices).')
    parser.add_argument(
      '--par_slices', default=1, dest='par_slices', type=int,
      help='number of slices per package. Volume devided by GPU\'s and'
           ' par_slices must be an even number!')
    parser.add_argument(
      '--data', default='', dest='file',
      help="Full path to input data. "
           "If not provided, a file dialog will open.")
    parser.add_argument(
      '--model', default='ImageReco', dest='sig_model',
      help='Name of the signal model to use. Defaults to VFA. \
 Please put your signal model file in the Model subfolder.')
    parser.add_argument(
      '--config', default='default', dest='config',
      help='Name of config file to use (assumed to be in the same folder). \
 If not specified, use default parameters.')
    parser.add_argument(
      '--imagespace', default=False, dest='imagespace', type=str2bool,
      help="Select if Reco is performed on images (1) or on kspace (0) data. "
           "Defaults to 0")
    parser.add_argument(
      '--sms', default=False, dest='sms', type=str2bool,
      help="Switch to SMS reconstruction")
    parser.add_argument(
      '--OCL_GPU', default=True, dest='use_GPU', type=str2bool,
      help="Select if CPU or GPU should be used as OpenCL platform. "
           "Defaults to GPU (1). CAVE: CPU FFT not working")
    parser.add_argument(
      '--devices', default=0, dest='devices', type=int,
      help="Device ID of device(s) to use for streaming. "
           "-1 selects all available devices", nargs='*')
    parser.add_argument(
      '--dz', default=1, dest='dz', type=float,
      help="Ratio of physical Z to X/Y dimension. "
           "X/Y is assumed to be isotropic. Defaults to  1")
    parser.add_argument(
      '--weights', default=None, dest='weights', type=float,
      help="Ratio of unkowns to each other. Defaults to 1. "
           "If passed, needs to be in the same size as the number of unknowns",
           nargs='*')
    args = parser.parse_args()
    start_recon(args)
