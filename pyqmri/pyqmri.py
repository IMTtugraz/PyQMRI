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
import time
import os
import h5py

from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
import importlib

import pyopencl as cl
import pyopencl.array as clarray


import argparse

from pyqmri._helper_fun import _goldcomp as goldcomp
from pyqmri._helper_fun._est_coils import est_coils
from pyqmri._helper_fun import _utils as utils

DTYPE = np.complex64
DTYPE_real = np.float32


def start_recon(args):
    sig_model_path = os.path.normpath(args.sig_model)
    if len(sig_model_path.split(os.sep)) > 1:
        spec = importlib.util.spec_from_file_location(
            sig_model_path.split(os.sep)[-1], sig_model_path)
        sig_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sig_model)
    else:
        sig_model = importlib.import_module(
            "pyqmri._models."+str(sig_model_path))
    if int(args.streamed) == 1:
        import pyqmri._irgn._reco_streamed as optimizer
    else:
        import pyqmri._irgn._reco as optimizer
    np.seterr(divide='ignore', invalid='ignore')

# Create par struct to store everyting
    par = {}
###############################################################################
# Select input file ###########################################################
###############################################################################
    if args.file == '':
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
        if not args.file.endswith((('.h5'), ('.hdf5'))):
            print("Please specify a h5 file. ")
            return 0
        file = args.file

    name = os.path.normpath(file)
    fname = name.split(os.sep)[-1]

    par["file"] = h5py.File(file)
#    del par["file"]["Coils"]
###############################################################################
# Read Data ###################################################################
###############################################################################
    reco_Slices = args.slices
    dimX, dimY, NSlice = ((par["file"].attrs['image_dimensions']).astype(int))
    if reco_Slices == -1:
        reco_Slices = NSlice
    off = 0

    data = (par["file"]['real_dat'][
      ..., int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:
      int(NSlice/2)+int(np.ceil(reco_Slices/2))+off, :, :].astype(DTYPE)
            + 1j*par["file"]['imag_dat'][
               ..., int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:
               int(NSlice/2)+int(np.ceil(reco_Slices/2))+off,
               :, :].astype(DTYPE))

    dimreduction = 0
    if args.trafo:
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
        data = np.require(data[..., int(dimreduction/2):
                               data.shape[-1]-int(dimreduction/2)],
                          requirements='C')
        dimX -= dimreduction
        dimY -= dimreduction
        par["traj"] = np.require(par["traj"][
            ..., int(dimreduction/2):
            par["traj"].shape[-1]-int(dimreduction/2)], requirements='C')
        par["dcf"] = np.sqrt(np.array(goldcomp.cmp(par["traj"]),
                                      dtype=DTYPE_real)).astype(DTYPE_real)
        par["dcf"] = np.require(np.abs(par["dcf"]),
                                DTYPE_real, requirements='C')
###############################################################################
# FA correction ###############################################################
###############################################################################
    if "fa_corr" in list(par["file"].keys()):
        print("Using provied flip angle correction.")
        par["fa_corr"] = np.flip(par["file"]['fa_corr'][()].astype(DTYPE), 0)[
            int(NSlice/2)-int(np.floor((reco_Slices)/2)):
            int(NSlice/2)+int(np.ceil(reco_Slices/2)),
            ...]

        par["fa_corr"] = par["fa_corr"][
            :,
            int(dimreduction/2):par["fa_corr"].shape[-2]-int(dimreduction/2),
            int(dimreduction/2):par["fa_corr"].shape[-1]-int(dimreduction/2)]

        # Recheck if shifted/transposed correctly
        par["fa_corr"] = np.require((np.transpose(par["fa_corr"], (0, 2, 1))),
                                    requirements='C')
    elif "interpol_fa" in list(par["file"].keys()):
        print("Using provied flip angle correction.")
        par["fa_corr"] = np.flip(par["file"]['interpol_fa'][()].astype(DTYPE),
                                 0)[
            int(NSlice/2)-int(np.floor((reco_Slices)/2)):
            int(NSlice/2)+int(np.ceil(reco_Slices/2)),
            int(dimreduction/2):-int(dimreduction/2),
            int(dimreduction/2):-int(dimreduction/2)]

        # Recheck if shifted/transposed correctly
        par["fa_corr"] = np.require((np.transpose(par["fa_corr"], (0, 2, 1))),
                                    requirements='C')
    else:
        print("No flip angle correction provided/used.")
    if data.ndim == 5:
        [NScan, NC, reco_Slices, Nproj, N] = data.shape
    elif data.ndim == 4 and "IRLL" in args.sig_model:
        print("4D Data passed and IRLL model used. Reordering Projections "
              "into 8 Spokes/Frame")
        [NC, reco_Slices, Nproj, N] = data.shape
        Nproj_new = 8
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
    par["NSlice"] = reco_Slices
    par["NScan"] = NScan
    par["N"] = N
    par["Nproj"] = Nproj
    if args.streamed:
        par["par_slices"] = args.par_slices

    par["unknowns_TGV"] = sig_model.unknowns_TGV
    par["unknowns_H1"] = sig_model.unknowns_H1
    par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]

    if not args.trafo:
        tmp = np.ones_like(np.abs(data))
        tmp[np.abs(data) == 0] = 0
        par['mask'] = np.reshape(tmp, (data.shape)).astype(DTYPE_real)
        del tmp
    else:
        par['mask'] = None
###############################################################################
# Create OpenCL Context and Queues ############################################
###############################################################################
    platforms = cl.get_platforms()
    par["GPU"] = False
    par["Platform_Indx"] = 0
    if args.use_GPU:
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
        if args.use_GPU:
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
    if type(args.devices) == int:
        args.devices = [args.devices]
    if args.streamed:
        if len(args.devices) == 1 and args.devices[0] == -1:
            num_dev = []
            for j in range(len(platforms[par["Platform_Indx"]].get_devices())):
                num_dev.append(j)
            par["num_dev"] = num_dev
        else:
            num_dev = args.devices
            par["num_dev"] = num_dev
    else:
        num_dev = args.devices
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
###############################################################################
# Coil Sensitivity Estimation #################################################
###############################################################################
    est_coils(data, par, par["file"], args)
###############################################################################
# Standardize data ############################################################
###############################################################################
    [NScan, NC, NSlice, Nproj, N] = data.shape
    if args.trafo:
        if par["file"].attrs['data_normalized_with_dcf']:
            pass
        else:
            data = data*(par["dcf"])
###############################################################################
# generate nFFT  ##############################################################
###############################################################################
    if NC == 1:
        par['C'] = np.ones((data[0, ...].shape), dtype=DTYPE)
    else:
        par['C'] = par['C'].astype(DTYPE)

    FFT = utils.NUFFT(par, trafo=args.trafo)

    def nFTH(x, fft, par):
        siz = np.shape(x)
        result = np.zeros((par["NC"], par["NSlice"], par["NScan"],
                           par["dimY"], par["dimX"]), dtype=DTYPE)
        tmp_result = clarray.empty(fft.queue, (par["NScan"], 1, 1,
                                   par["dimY"], par["dimX"]), dtype=DTYPE)
        for j in range(siz[1]):
            for k in range(siz[2]):
                inp = clarray.to_device(fft.queue,
                                        np.require(x[:, j, k, ...]
                                                    [:, None, None, ...],
                                                   requirements='C'))
                fft.adj_NUFFT(tmp_result, inp)
                result[j, k, ...] = np.squeeze(tmp_result.get())
        return np.transpose(result, (2, 0, 1, 3, 4))

    images = np.require(np.sum(nFTH(data, FFT, par) *
                               (np.conj(par["C"])), axis=1),
                        requirements='C')
    del FFT, nFTH

###############################################################################
# Scale data norm  ############################################################
###############################################################################
    if args.imagespace:
        dscale = DTYPE_real(np.sqrt(2*1e3*NSlice) /
                            (np.linalg.norm(images.flatten())))
        par["dscale"] = dscale
        images = images*dscale
    else:
        dscale = (DTYPE_real(np.sqrt(2*1e3*NSlice)) /
                  (np.linalg.norm(data.flatten())))
        par["dscale"] = dscale
        data = data*dscale

    if args.trafo:
        center = int(N*0.1)
        sig = []
        noise = []
        ind = np.zeros((N), dtype=bool)
        ind[int(par["N"]/2-center):int(par["N"]/2+center)] = 1
        for j in range(Nproj):
            sig.append(np.sum(data[..., int(NSlice/2), j, ind] *
                              np.conj(data[..., int(NSlice/2), j, ind])))
            noise.append(np.sum(data[..., int(NSlice/2), j, ~ind] *
                                np.conj(data[..., int(NSlice/2), j, ~ind])))
        sig = (np.sum(np.array(sig)))/np.sum(ind)
        noise = (np.sum(np.array(noise)))/np.sum(~ind)
        SNR_est = np.abs(sig/noise)
        par["SNR_est"] = SNR_est
        print("Estimated SNR from kspace", SNR_est)
    else:
        center = int(N*0.1)
        ind = np.zeros((dimY, dimX), dtype=bool)
        ind[int(par["N"]/2-center):int(par["N"]/2+center),
            int(par["N"]/2-center):int(par["N"]/2+center)] = 1
        ind = np.fft.fftshift(ind)
        sig = np.sum(data[..., int(NSlice/2), ind] *
                     np.conj(data[..., int(NSlice/2), ind]))/np.sum(ind)
        noise = np.sum(data[..., int(NSlice/2), ~ind] *
                       np.conj(data[..., int(NSlice/2), ~ind]))/np.sum(~ind)
        SNR_est = np.abs(sig/noise)
        par["SNR_est"] = SNR_est
        print("Estimated SNR from kspace", SNR_est)

    opt = optimizer.ModelReco(par, args.trafo,
                              imagespace=args.imagespace)

    if args.imagespace:
        opt.data = images
    else:
        opt.data = data
###############################################################################
# ratio of z direction to x,y, important for finite differences ###############
###############################################################################
    opt.dz = args.dz
###############################################################################
# Start Reco ##################################################################
###############################################################################
    if args.type == '3D':
        #######################################################################
        # Init forward model and initial guess ################################
        #######################################################################
        model = sig_model.Model(par, images)
        par["file"].close()
        #######################################################################
        # IRGN - TGV Reco #####################################################
        #######################################################################
        if "TGV" in args.reg or args.reg == 'all':
            result_tgv = []
            opt.model = model
            ###################################################################
            # IRGN Params #####################################################
            ###################################################################
            opt.irgn_par = utils.read_config(args.config, "3D_TGV")
            opt.execute(TV=0, imagespace=args.imagespace)
            result_tgv.append(opt.result)
            plt.close('all')
            res_tgv = opt.gn_res
            res_tgv = np.array(res_tgv)/(opt.irgn_par["lambd"]*NSlice)
        #######################################################################
        # IRGN - TV Reco ######################################################
        #######################################################################
        if "TV" in args.reg or args.reg == 'all':
            result_tv = []
            opt.model = model
            ###################################################################
            # IRGN Params #####################################################
            ###################################################################
            opt.irgn_par = utils.read_config(args.config, "3D_TV")
            opt.execute(TV=1, imagespace=args.imagespace)
            result_tv.append(opt.result)
            plt.close('all')
            res_tv = opt.gn_res
            res_tv = np.array(res_tv)/(opt.irgn_par["lambd"]*NSlice)
        del opt
###############################################################################
# New .hdf5 save files ########################################################
###############################################################################
    outdir = time.strftime("%Y-%m-%d  %H-%M-%S_MRI_"+args.reg+"_"+args.type+"_"
                           + fname[:-3])
    if not os.path.exists('./output'):
        os.makedirs('./output')
    if not os.path.exists('./output/' + outdir):
        os.makedirs("output/" + outdir)
    cwd = os.getcwd()
    os.chdir("output/" + outdir)
    f = h5py.File("output_" + fname, "w")
    f.create_dataset("images_ifft_", images.shape, dtype=DTYPE, data=images)
    if "TGV" in args.reg or args.reg == 'all':
        for i in range(len(result_tgv)):
            f.create_dataset("tgv_full_result_"+str(i), result_tgv[i].shape,
                             dtype=DTYPE, data=result_tgv[i])
            f.attrs['res_tgv'+str(i)] = res_tgv
            for j in range(len(model.uk_scale)):
                model.uk_scale[j] = 1
            image_final = model.execute_forward(result_tgv[i][-1, ...])
            f.create_dataset("sim_images_"+str(i), image_final.shape,
                             dtype=DTYPE, data=image_final)
    if "TV" in args.reg or args.reg == 'all':
        for i in range(len(result_tv)):
            f.create_dataset("tv_full_result_"+str(i), result_tv[i].shape,
                             dtype=DTYPE, data=result_tv[i])
            f.attrs['res_tv'] = res_tv

    if "imagespace" in args.reg or args.reg == 'all':
        for i in range(len(result_tgv)):
            f.create_dataset("imagespace_full_result_"+str(i),
                             result_tgv[i].shape,
                             dtype=DTYPE, data=result_tgv[i])
            f.attrs['imagespace_tgv'] = res_tgv
        f.flush()
    f.attrs['data_norm'] = dscale
    f.close()
    os.chdir(cwd)


def main(recon_type='3D', reg_type='TGV', slices=1, trafo=1, streamed=0,
         par_slices=1, data='', model='VFA', config='default', imagespace=0,
         OCL_GPU=1, devices=0, dz=1):
    """
    Start a 3D model based reconstruction. Data can also be selected at
    start up.

    Parameters
    ----------
    recon_type
        3D (2D currently not supported but 3D works on one slice also)
    reg_type
        TGV or TV, defaults to TGV
    slices
        The number of slices to reconsturct. Slices are picked symmetrically
        from the volume center. Pass -1 to select all slices available.
        Defaults to 1
    trafo
      Choos between Radial (1) or Cartesian (0) FFT
    streamed
      Toggle between streaming slices to the GPU (1) or computing
      everything with a single memory transfer (0). Defaults to 0
    par_slices
      Number of slices per streamed package. Volume devided by GPU's and
      par_slices must be an even number! Defaults to 1
    data
      The path to the .h5 file containing the data to reconstruct.
      If left empty, a GUI will open and asks for data file selection. This
      is also the default behaviour.
    model
      The name of the model which should be used to fit the data. Defaults to
      'VFA'. A path to your own model file can be passed. See the Model Class
      for further information on how to setup your own model.
    config
      The path to the confi gfile used for the IRGN reconstruction. If
      not specified the default config file will be used. If no default
      config file is present in the current working directory one will be
      generated.
    imagespace
      Select between fitting in imagespace (1) or in k-space (0). Defaults to 0
    OCL_GPU
      Select between GPU (1) or CPU (0) OpenCL devices. Defaults to GPU
      CAVE: CPU FFT not working.
    devices
      The device ID of device(s) to use for streaming/reconstruction
    dz
      Ratio of physical Z to X/Y dimension. X/Y is assumed to be isotropic.
    """
    parser = argparse.ArgumentParser(description="T1 quantification from VFA "
                                                 "data. By default runs 3D "
                                                 "regularization for TGV and "
                                                 "TV.")
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
      '--trafo', default=trafo, dest='trafo', type=int,
      help='Choos between radial (1, default) and Cartesian (0) sampling. ')
    parser.add_argument(
      '--streamed', default=streamed, dest='streamed', type=int,
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
      '--imagespace', default=imagespace, dest='imagespace', type=int,
      help="Select if Reco is performed on images (1) or on kspace (0) data. "
           "Defaults to 0")
    parser.add_argument(
      '--OCL_GPU', default=OCL_GPU, dest='use_GPU', type=int,
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
    args = parser.parse_args()
    start_recon(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="T1 quantification from VFA "
                                                 "data. By default runs 3D "
                                                 "regularization for TGV and "
                                                 "TV.")
    parser.add_argument(
      '--recon_type', default='3D', dest='type',
      help='Choose reconstruction type (currently only 3D)')
    parser.add_argument(
      '--reg_type', default='TGV', dest='reg',
      help="Choose regularization type (default: TGV) "
           "options are: TGV, TV, all")
    parser.add_argument(
      '--slices', default=1, dest='slices', type=int,
      help="Number of reconstructed slices (default=40). "
           "Symmetrical around the center slice.")
    parser.add_argument(
      '--trafo', default=1, dest='trafo', type=int,
      help='Choos between radial (1, default) and Cartesian (0) sampling. ')
    parser.add_argument(
      '--streamed', default=0, dest='streamed', type=int,
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
      '--model', default='VFA', dest='sig_model',
      help='Name of the signal model to use. Defaults to VFA. \
 Please put your signal model file in the Model subfolder.')
    parser.add_argument(
      '--config', default='default', dest='config',
      help='Name of config file to use (assumed to be in the same folder). \
 If not specified, use default parameters.')
    parser.add_argument(
      '--imagespace', default=0, dest='imagespace', type=int,
      help="Select if Reco is performed on images (1) or on kspace (0) data. "
           "Defaults to 0")
    parser.add_argument(
      '--imagespace', default=0, dest='imagespace', type=int,
      help='Select if Reco is performed on images (1) or on kspace (0) data. \
 Defaults to 0')
    parser.add_argument(
      '--devices', default=0, dest='devices', type=int,
      help="Device ID of device(s) to use for streaming. "
           "-1 selects all available devices", nargs='*')
    parser.add_argument(
      '--dz', default=1, dest='dz', type=float,
      help="Ratio of physical Z to X/Y dimension. "
           "X/Y is assumed to be isotropic. Defaults to  1")
    args = parser.parse_args()
    start_recon(args)
