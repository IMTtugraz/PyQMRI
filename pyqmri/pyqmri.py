#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module handling the start up of the fitting procedure."""
import sys
import argparse
import os
import time
import importlib

import numpy as np
from tkinter import filedialog
from tkinter import Tk

import matplotlib.pyplot as plt

import h5py
import pyopencl as cl
import pyopencl.array as clarray


from pyqmri._helper_fun import _goldcomp as goldcomp
from pyqmri._helper_fun._est_coils import est_coils
from pyqmri._helper_fun import _utils as utils
from pyqmri.solver import CGSolver
from pyqmri.irgn import IRGNOptimizer


def _choosePlatform(myargs, par):
    platforms = cl.get_platforms()
    use_GPU = False
    par["Platform_Indx"] = 0
    if myargs.use_GPU:
        for j, platfrom in enumerate(platforms):
            if platfrom.get_devices(device_type=cl.device_type.GPU):
                print("GPU OpenCL platform <%s> found "
                      "with %i device(s) and OpenCL-version <%s>"
                      % (str(platfrom.get_info(cl.platform_info.NAME)),
                         len(platfrom.get_devices(
                             device_type=cl.device_type.GPU)),
                         str(platfrom.get_info(cl.platform_info.VERSION))))
                use_GPU = True
                par["Platform_Indx"] = j
    if not use_GPU:
        if myargs.use_GPU:
            print("No GPU OpenCL platform found. Falling back to CPU.")
        for j, platfrom in enumerate(platforms):
            if platfrom.get_devices(device_type=cl.device_type.CPU):
                print("CPU OpenCL platform <%s> found "
                      "with %i device(s) and OpenCL-version <%s>"
                      % (str(platfrom.get_info(cl.platform_info.NAME)),
                         len(platfrom.get_devices(
                             device_type=cl.device_type.GPU)),
                         str(platfrom.get_info(cl.platform_info.VERSION))))
                use_GPU = False
                par["Platform_Indx"] = j
    return platforms


def _precoompFFT(data, par):
    full_dimY = (np.all(np.abs(data[0, 0, 0, :, 0])) or
                 np.all(np.abs(data[0, 0, 0, :, 1])))
    full_dimX = (np.all(np.abs(data[0, 0, 0, 0, :])) or
                 np.all(np.abs(data[0, 0, 0, 1, :])))

    if full_dimY and not full_dimX:
        print("Image Dimensions Y seems fully sampled. "
              "Precompute FFT along Y")
        data = np.fft.ifft(data, axis=-2, norm='ortho')
        par["fft_dim"] = [-1]

    elif full_dimX and not full_dimY:
        print("Image Dimensions X seems fully sampled. "
              "Precompute FFT along X")
        data = np.fft.ifft(data, axis=-1, norm='ortho')
        data = np.require(
            np.moveaxis(data, -1, -2),
            requirements='C')
        par["C"] = np.require(
            np.moveaxis(par["C"], -1, -2),
            requirements='C')
        par["mask"] = np.require(
            np.moveaxis(par["mask"], -1, -2),
            requirements='C')
        dimX = par["dimX"]
        dimY = par["dimY"]
        par["dimX"] = dimY
        par["dimY"] = dimX
        par["N"] = dimY
        par["Nproj"] = dimX
        par["transpXY"] = True
        par["fft_dim"] = [-1]

    elif full_dimX and full_dimY:
        print("Image Dimensions X and Y seem fully sampled. "
              "Precompute FFT along X and Y")
        data = np.fft.ifft2(data, norm='ortho')
        par["fft_dim"] = None
    else:
        par["fft_dim"] = [-2, -1]

    return np.require(data.astype(par["DTYPE"]),
                      requirements='C')


def _setupOCL(myargs, par):
    platforms = _choosePlatform(myargs, par)
    par["ctx"] = []
    par["queue"] = []
    if isinstance(myargs.devices, int):
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
        tmpxtx = cl.Context(dev)
        par["ctx"].append(tmpxtx)
        for j in range(4):
            par["queue"].append(
                cl.CommandQueue(
                   tmpxtx,
                   platforms[par["Platform_Indx"]].get_devices()[device],
                   properties=(
                     cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
                   )
                )


def _genImages(myargs, par, data, off):
    if not myargs.usecg:
        FFT = utils.NUFFT(par, trafo=myargs.trafo, SMS=myargs.sms)

        def nFTH(x, fft, par):
            siz = np.shape(x)
            result = np.zeros(
                (par["NScan"], par["NC"], par["NSlice"],
                 par["dimY"], par["dimX"]), dtype=par["DTYPE"])
            tmp_result = clarray.empty(fft.queue,
                                       (1, 1, par["NSlice"],
                                        par["dimY"], par["dimX"]),
                                       dtype=par["DTYPE"])
            start = time.time()
            for j in range(siz[0]):
                for k in range(siz[1]):
                    inp = clarray.to_device(fft.queue,
                                            np.require(x[j, k, ...]
                                                        [None, None, ...],
                                                       requirements='C'))
                    fft.FFTH(tmp_result, inp, scan_offset=j).wait()
                    result[j, k, ...] = np.squeeze(tmp_result.get())
            end = time.time()-start
            print("FT took %f s" % end)
            return result
        images = np.require(np.sum(nFTH(data, FFT, par) *
                                   (np.conj(par["C"])), axis=1),
                            requirements='C')
        del FFT, nFTH

    else:
        tol = 1e-6
        par_scans = 4
        lambd = 1e-3
        if "images" not in list(par["file"].keys()):
            images = np.zeros((par["NScan"],
                               par["NSlice"],
                               par["dimY"],
                               par["dimX"]), dtype=par["DTYPE"])
            if par["NScan"]/par_scans >= 1:
                cgs = CGSolver(par, par_scans, myargs.trafo, myargs.sms)
                for j in range(int(par["NScan"]/par_scans)):
                    if par["NSlice"] == 1:
                        images[par_scans*j:par_scans*(j+1), ...] = cgs.run(
                            data[par_scans*j:par_scans*(j+1), ...],
                            tol=tol, lambd=lambd,
                            scan_offset=par_scans*j)[:, None, ...]
                    else:
                        images[par_scans*j:par_scans*(j+1), ...] = cgs.run(
                            data[par_scans*j:par_scans*(j+1), ...],
                            tol=tol, lambd=lambd, scan_offset=par_scans*j)
                del cgs
            if np.mod(par["NScan"], par_scans):
                cgs = CGSolver(par, np.mod(par["NScan"], par_scans),
                               myargs.trafo, myargs.sms)
                if par["NSlice"] == 1:
                    if np.mod(par["NScan"], par_scans) == 1:
                        images[-np.mod(par["NScan"], par_scans):, ...] = \
                            cgs.run(
                                data[-np.mod(par["NScan"], par_scans):, ...],
                                tol=tol, lambd=lambd,
                                scan_offset=par["NScan"]-np.mod(
                                    par["NScan"], par_scans))
                    else:
                        images[-np.mod(par["NScan"], par_scans):, ...] = \
                            cgs.run(
                                data[-np.mod(par["NScan"], par_scans):, ...],
                                tol=tol, lambd=lambd,
                                scan_offset=par["NScan"]-np.mod(
                                    par["NScan"], par_scans))[:, None, ...]
                else:
                    images[-np.mod(par["NScan"], par_scans):, ...] = \
                        cgs.run(
                            data[-np.mod(par["NScan"], par_scans):, ...],
                            tol=tol, lambd=lambd,
                            scan_offset=par["NScan"]-np.mod(
                                    par["NScan"], par_scans))
                del cgs
            par["file"].create_dataset("images", images.shape,
                                       dtype=par["DTYPE"], data=images)
        else:
            images = par["file"]['images']
            if images.shape[1] < par["NSlice"]:
                del par["file"]["images"]
                images = _genImages(myargs, par, data, off)
            else:
                print("Using precomputed images")
                slices_images = par["file"]['images'][()].shape[1]
                images = \
                    par["file"]['images'][
                      :,
                      int(slices_images / 2) - int(
                        np.floor((par["NSlice"]) / 2)) + off:int(
                          slices_images / 2) + int(
                            np.ceil(par["NSlice"] / 2)) + off,
                      ...].astype(par["DTYPE"])
    return images


def _estScaleNorm(myargs, par, images, data):

    if myargs.trafo:
        center = int(par["N"]*0.1)
        ind = np.zeros((par["N"]), dtype=bool)
        ind[int(par["N"]/2-center):int(par["N"]/2+center)] = 1
        inds = np.fft.fftshift(ind)
        dims = tuple(range(data.ndim - 3)) + (-2,)
        sig = np.max(
            np.sum(data[..., ind], dims) *
            np.conj(np.sum(data[..., ind], dims)))
        noise = np.std(
            np.sum(data[..., inds], dims) *
            np.conj(np.sum(data[..., inds], dims)))

    else:
        centerX = int(par["dimX"]*0.1)
        centerY = int(par["dimY"]*0.1)
        ind = np.zeros((par["dimY"], par["dimX"]), dtype=bool)
        ind[int(par["dimY"]/2-centerY):int(par["dimY"]/2+centerY),
            int(par["dimX"]/2-centerX):int(par["dimX"]/2+centerX)] = 1
        if par["fft_dim"] is not None:
            for shiftdim in par["fft_dim"]:
                ind = np.fft.fftshift(ind, axes=shiftdim)
            sig = np.sum(
                np.abs(data[..., ind])**2)
            noise = np.sum(
                np.abs(data[..., ~ind])**2)
        else:
            tmp = np.fft.fft2(data, norm='ortho')
            ind = np.fft.fftshift(ind)
            sig = np.sum(
                np.abs(tmp[..., ind])**2)
            noise = np.sum(
                np.abs(tmp[..., ~ind])**2)

    SNR_est = (np.abs(sig/noise))
    par["SNR_est"] = SNR_est
    print("Estimated SNR from kspace", SNR_est)

    dscale = par["DTYPE_real"](1 / np.linalg.norm(np.abs(data)))
    print("Data scale: ", dscale)
    par["dscale"] = dscale
    images = images*dscale
    data = data*dscale

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
            if file and \
               not file.endswith((('.h5'), ('.hdf5'))):
                print("Please specify a h5 file. Press cancel to exit.")
            elif not file:
                print("Exiting...")
                sys.exit()
            else:
                select_file = False
    else:
        if not myargs.file.endswith((('.h5'), ('.hdf5'))):
            print("Please specify a h5 file. ")
            sys.exit()
        file = myargs.file
    name = os.path.normpath(file)
    par["fname"] = name.split(os.sep)[-1]

    if myargs.outdir == '':
        outdir = os.sep.join(name.split(os.sep)[:-1]) + os.sep + \
            "PyQMRI_out" + \
            os.sep + myargs.sig_model + os.sep + \
            time.strftime("%Y-%m-%d  %H-%M-%S") + os.sep
    else:
        outdir = myargs.outdir + os.sep + "PyQMRI_out" + \
            os.sep + myargs.sig_model + os.sep + par["fname"] + os.sep + \
            time.strftime("%Y-%m-%d  %H-%M-%S") + os.sep
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    par["outdir"] = outdir
    par["file"] = h5py.File(file, 'a')


def _read_data_from_file(par, myargs):
    reco_Slices = myargs.slices
    dimX, dimY, NSlice = ((par["file"].attrs['image_dimensions']).astype(int))
    if reco_Slices == -1:
        reco_Slices = NSlice
    off = 0

    if myargs.sms:
        data = par["file"]['real_dat'][()].astype(par["DTYPE"])\
               + 1j*par["file"]['imag_dat'][()].astype(par["DTYPE"])
    else:
        data = par["file"]['real_dat'][
          ...,
          int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:
          int(NSlice/2)+int(np.ceil(reco_Slices/2))+off, :, :
              ].astype(par["DTYPE"])\
               + 1j*par["file"]['imag_dat'][
          ...,
          int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:
          int(NSlice/2)+int(np.ceil(reco_Slices/2))+off, :, :
              ].astype(par["DTYPE"])

    dimreduction = 0
    if myargs.trafo:
        par["traj"] = par["file"]['real_traj'][()].astype(par["DTYPE"]) + \
                      1j*par["file"]['imag_traj'][()].astype(par["DTYPE"])

        par["dcf"] = np.sqrt(goldcomp.cmp(
                         par["traj"]))
        par["dcf"] = np.require(np.abs(par["dcf"]),
                                par["DTYPE_real"], requirements='C')
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
            par["dcf"] = np.sqrt(goldcomp.cmp(par["traj"]))
            par["dcf"] = np.require(np.abs(par["dcf"]),
                                    par["DTYPE_real"], requirements='C')
        else:
            data = np.require(data[...,
                                   int(dimreduction/2):
                                   data.shape[-1]-int(dimreduction/2),
                                   int(dimreduction/2):
                                   data.shape[-1]-int(dimreduction/2)],
                              requirements='C')
    return data, dimX, dimY, NSlice, reco_Slices, dimreduction, off


def _read_flip_angle_correction_data(par, myargs, dimreduction, reco_Slices):
    if "fa_corr" in list(par["file"].keys()):
        NSlice_fa, _, _ = par["file"]['fa_corr'][()].shape
    elif "interpol_fa" in list(par["file"].keys()):
        NSlice_fa, _, _ = par["file"]['interpol_fa'][()].shape

    par["fa_corr"] = np.flip(
        par["file"]['fa_corr'][()].astype(par["DTYPE"]),
        0)[
          int(NSlice_fa/2)-int(np.floor((reco_Slices)/2)):
          int(NSlice_fa/2)+int(np.ceil(reco_Slices/2)),
          ...]
    par["fa_corr"][par["fa_corr"] == 0] = 1
    par["fa_corr"] = par["fa_corr"][
       ...,
       int(dimreduction/2):par["fa_corr"].shape[-2]-int(dimreduction/2),
       int(dimreduction/2):par["fa_corr"].shape[-1]-int(dimreduction/2)]
    par["fa_corr"] = np.require((np.transpose(par["fa_corr"], (0, 2, 1))),
                                requirements='C')


def _check_data_size(par, data, sigmodel):
    if data.ndim == 5:
        pass
    elif data.ndim == 4 and "IRLL" in sigmodel:
        Nproj_new = 13
        print("4D Data passed and IRLL model used. Reordering Projections "
              "into "+str(Nproj_new)+" Spokes/Frame")
        [NC, reco_Slices, Nproj, N] = data.shape
        NScan = np.floor_divide(Nproj, Nproj_new)
        par["Nproj_measured"] = Nproj
        Nproj = Nproj_new
        data = np.require(np.transpose(np.reshape(data[..., :Nproj*NScan, :],
                                                  (NC, reco_Slices, NScan,
                                                   Nproj, N)),
                                       (2, 0, 1, 3, 4)), requirements='C')
        par["traj"] = np.require(np.reshape(par["traj"][:Nproj*NScan, :],
                                            (NScan, Nproj, N)),
                                 requirements='C')
        par["dcf"] = np.sqrt(goldcomp.cmp(par["traj"]))
        par["dcf"] = np.require(np.abs(par["dcf"]), par["DTYPE_real"],
                                requirements='C')
    elif data.ndim == 4 and "ImageReco" in sigmodel:
        data = data[None]
    else:
        raise ValueError(
            "Wrong data dimension / model incompatible.")
    return data


def _populate_par_w_sequencepar(par,
                                datashape,
                                imagedims,
                                SMS,
                                imagespace,
                                streamded,
                                par_slices,
                                trafo):
    for att in par["file"].attrs:
        par[att] = par["file"].attrs[att]
    par["NC"] = datashape[1]
    par["dimY"] = imagedims[-2]
    par["dimX"] = imagedims[-1]
    if SMS:
        par["NSlice"] = imagedims[0]
        par["packs"] = int(par["packs"])
        par["numofpacks"] = int(
            imagedims[0]/(int(par["packs"])*int(par["MB"])))
    else:
        par["NSlice"] = datashape[2]
        par["packs"] = 1
        par["MB"] = 1
    par["NScan"] = datashape[0]
    par["N"] = datashape[-1]
    par["Nproj"] = datashape[-2]
    par["imagespace"] = imagespace
    par["fft_dim"] = (-2, -1)
    if streamded:
        par["par_slices"] = par_slices
        par["overlap"] = 1
    else:
        par["par_slices"] = datashape[2]
        par["overlap"] = 0
    par["transpXY"] = False


def _import_sigmodel(modelpath):
    sig_model_path = os.path.normpath(modelpath)
    if len(sig_model_path.split(os.sep)) > 1:
        if os.path.splitext(sig_model_path)[-1] == '':
            spec = importlib.util.spec_from_file_location(
                    sig_model_path.split(os.sep)[-1]+'.py',
                    sig_model_path+'.py')
        elif os.path.splitext(sig_model_path)[-1] == '.py':
            spec = importlib.util.spec_from_file_location(
                    sig_model_path.split(os.sep)[-1], sig_model_path)
        else:
            raise argparse.ArgumentTypeError(
                "Specified model file does not end with .py nor does it have "
                "no extension at all. Please specify a valid python file.")
        sig_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sig_model)
    else:
        sig_model = importlib.import_module(
            "pyqmri.models."+str(sig_model_path))
    return sig_model


def _start_recon(myargs):
    sig_model = _import_sigmodel(myargs.sig_model)
# Create par struct to store relevant parameteres for reconstruction/fitting
    par = {}
###############################################################################
# Define precision ############################################################
###############################################################################
    if myargs.double_precision is True:
        par["DTYPE"] = np.complex128
        par["DTYPE_real"] = np.float64
    else:
        par["DTYPE"] = np.complex64
        par["DTYPE_real"] = np.float32
###############################################################################
# Select input file ###########################################################
###############################################################################
    _readInput(myargs, par)
###############################################################################
# Read Data ###################################################################
###############################################################################
    data, dimX, dimY, NSlice, reco_Slices, dimreduction, off = \
        _read_data_from_file(par, myargs)
###############################################################################
# Flip angle correction #######################################################
###############################################################################
    if "fa_corr" in list(par["file"].keys()) or \
            "interpol_fa" in list(par["file"].keys()):
        print("Using provied flip angle correction.")
        _read_flip_angle_correction_data(
            par,
            myargs,
            dimreduction,
            reco_Slices
            )
    else:
        print("No flip angle correction provided/used.")
###############################################################################
# Check size compatibility ####################################################
###############################################################################
    data = _check_data_size(par, data, myargs.sig_model)
###############################################################################
# Set sequence related parameters #############################################
###############################################################################
    _populate_par_w_sequencepar(par,
                                data.shape,
                                (NSlice, dimY, dimX),
                                myargs.sms,
                                myargs.imagespace,
                                myargs.streamed,
                                myargs.par_slices,
                                myargs.trafo)
    if not myargs.trafo:
        tmpmask = np.ones(data.shape[2:])
        tmpmask[np.abs(data[0, 0, ...]) == 0] = 0
        par['mask'] = np.reshape(
            tmpmask,
            (data.shape[2:])).astype(par["DTYPE_real"])
        del tmpmask
    else:
        par['mask'] = None
###############################################################################
# ratio of z direction to x,y, important for finite differences ###############
###############################################################################
    par["dz"] = par["DTYPE_real"](myargs.dz)
###############################################################################
# Create OpenCL Context and Queues ############################################
###############################################################################
    _setupOCL(myargs, par)
###############################################################################
# Coil Sensitivity Estimation #################################################
###############################################################################
    est_coils(data, par, par["file"], myargs, off)
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
        par['C'] = np.ones((data[0, ...].shape), dtype=par["DTYPE"])
    else:
        par['C'] = par['C'].astype(par["DTYPE"])
###############################################################################
# Init forward model and initial guess ########################################
###############################################################################
    if myargs.sig_model == "GeneralModel":
        par["modelfile"] = myargs.modelfile
        par["modelname"] = myargs.modelname
    model = sig_model.Model(par)
###############################################################################
# Reconstruct images using CG-SENSE  ##########################################
###############################################################################
    if myargs.trafo is False:
        data = _precoompFFT(data, par)
    images = _genImages(myargs, par, data, off)
###############################################################################
# Scale data norm  ############################################################
###############################################################################
    data, images = _estScaleNorm(myargs, par, images, data)
    if np.allclose(myargs.weights, -1):
        par["weights"] = np.ones((par["unknowns"]), dtype=par["DTYPE_real"])
    else:
        par["weights"] = np.array(myargs.weights, dtype=par["DTYPE_real"])
###############################################################################
# Compute initial guess #######################################################
###############################################################################
    model.computeInitialGuess(
        images,
        par["dscale"])
###############################################################################
# initialize operator  ########################################################
###############################################################################
    opt = IRGNOptimizer(par,
                        model,
                        trafo=myargs.trafo,
                        imagespace=myargs.imagespace,
                        SMS=myargs.sms,
                        config=myargs.config,
                        streamed=myargs.streamed,
                        reg_type=myargs.reg,
                        DTYPE=par["DTYPE"],
                        DTYPE_real=par["DTYPE_real"])
    f = h5py.File(par["outdir"]+"output_" + par["fname"], "a")
    f.create_dataset("images_ifft", data=images)
    f.attrs['data_norm'] = par["dscale"]
    f.close()
    par["file"].close()
###############################################################################
# Start Reco ##################################################################
###############################################################################
    if myargs.imagespace is True:
        opt.execute(images)
    else:
        opt.execute(data)
    plt.close('all')


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def run(reg_type='TGV',
        slices=1,
        trafo=True,
        streamed=False,
        par_slices=1,
        data='',
        model='GeneralModel',
        config='default',
        imagespace=False,
        sms=False,
        devices=0,
        dz=1,
        weights=-1,
        useCGguess=True,
        out='',
        modelfile="models.ini",
        modelname="VFA-E1",
        double_precision=False):
    """
    Start a 3D model based reconstruction.

    Start a 3D model based reconstruction.
    If no data path is given, a file dialog can be used to select data
    at start up. If no other parameters are passed, T1 from a single slice
    of radially acquired variable flip angle data will be quantified.

    If no config file is passed, a default one will be generated in the
    current folder, the script is run in.

    Parameters
    ----------
      reg_type : str, TGV
        TGV or TV, defaults to TGV
      slices : int, 1
        The number of slices to reconsturct. Slices are picked symmetrically
        from the volume center. Pass -1 to select all slices available.
        Defaults to 1
      trafo : bool, True
        Choos between Radial (1) or Cartesian (0) FFT
      streamed : bool, False
        Toggle between streaming slices to the GPU (1) or computing
        everything with a single memory transfer (0). Defaults to 0
      par_slices : int, 1
        Number of slices per streamed package. Volume devided by GPU's and
        par_slices must be an even number! Defaults to 1
      data : str, ''
        The path to the .h5 file containing the data to reconstruct.
        If left empty, a GUI will open and asks for data file selection. This
        is also the default behaviour.
      model : str, GeneralModel
        The name of the model which should be used to fit the data. Defaults to
        'VFA'. A path to your own model file can be passed. See the Model Class
        for further information on how to setup your own model.
      config : str, default
        The path to the confi gfile used for the IRGN reconstruction. If
        not specified the default config file will be used. If no default
        config file is present in the current working directory one will be
        generated.
      imagespace : bool, False
        Select between fitting in imagespace (1) or in k-space (0).
        Defaults to 0
      sms : bool, False
        use Simultaneous Multi Slice Recon (1) or normal reconstruction (0).
        Defaults to 0
      devices : list of int, 0
        The device ID of device(s) to use for streaming/reconstruction
      dz : float, 1
        Ratio of physical Z to X/Y dimension. X/Y is assumed to be isotropic.
      useCGguess : bool, True
        Switch between CG sense and simple FFT as initial guess for the images.
      out : str, ''
        Output directory. Defaults to the location of the input file.
      modelpath : str, models.ini
        Path to the .mod file for the generative model.
      modelname : str, VFA-E1
        Name of the model in the .mod file to use.
      weights : list of float, -1
        Optional weights for each unknown. Defaults to -1, i.e. no additional
        weights are used.
      double_precision : bool, False
        Enable double precission computation.
    """
    params = [('--recon_type', "TGV"),
              ('--reg_type', str(reg_type)),
              ('--slices', str(slices)),
              ('--trafo', str(trafo)),
              ('--streamed', str(streamed)),
              ('--par_slices', str(par_slices)),
              ('--data', str(data)),
              ('--config', str(config)),
              ('--imagespace', str(imagespace)),
              ('--sms', str(sms)),
              ('--OCL_GPU', "True"),
              ('--devices', str(devices)),
              ('--dz', str(dz)),
              ('--weights', str(weights)),
              ('--useCGguess', str(useCGguess)),
              ('--model', str(model)),
              ('--modelfile', str(modelfile)),
              ('--modelname', str(modelname)),
              ('--outdir', str(out)),
              ('--double_precision', str(double_precision))
              ]

    sysargs = sys.argv[1:]
    for par_name, par_value in params:
        if par_name not in sysargs:
            sysargs.append(par_name)
            sysargs.append(par_value)
    argsrun, unknown = _parseArguments(sysargs)
    if unknown:
        print("Unknown command line arguments passed: " + str(unknown) + "."
              " These will be ignored for fitting.")
    _start_recon(argsrun)


def _parseArguments(args):
    argparmain = argparse.ArgumentParser(
        description="T1 quantification from VFA "
                    "data. By default runs 3D "
                    "regularization for TGV.")
    argparmain.add_argument(
      '--recon_type', dest='type',
      help='Choose reconstruction type (currently only 3D)')
    argparmain.add_argument(
      '--reg_type', dest='reg',
      help="Choose regularization type (default: TGV) "
           "options are: TGV, TV, all")
    argparmain.add_argument(
      '--slices', dest='slices', type=int,
      help="Number of reconstructed slices (default=40). "
           "Symmetrical around the center slice.")
    argparmain.add_argument(
      '--trafo', dest='trafo', type=_str2bool,
      help='Choos between radial (1, default) and Cartesian (0) sampling. ')
    argparmain.add_argument(
      '--streamed', dest='streamed', type=_str2bool,
      help='Enable streaming of large data arrays (e.g. >10 slices).')
    argparmain.add_argument(
      '--par_slices', dest='par_slices', type=int,
      help='number of slices per package. Volume devided by GPU\'s and'
           ' par_slices must be an even number!')
    argparmain.add_argument(
      '--data', dest='file',
      help="Full path to input data. "
           "If not provided, a file dialog will open.")
    argparmain.add_argument(
      '--config', dest='config',
      help='Name of config file to use (assumed to be in the same folder). \
 If not specified, use default parameters.')
    argparmain.add_argument(
      '--imagespace', dest='imagespace', type=_str2bool,
      help="Select if Reco is performed on images (1) or on kspace (0) data. "
           "Defaults to 0")
    argparmain.add_argument(
      '--sms', dest='sms', type=_str2bool,
      help="Switch to SMS reconstruction")
    argparmain.add_argument(
      '--OCL_GPU', dest='use_GPU', type=_str2bool,
      help="Select if CPU or GPU should be used as OpenCL platform. "
           "Defaults to GPU (1). CAVE: CPU FFT not working")
    argparmain.add_argument(
      '--devices', dest='devices', type=int,
      help="Device ID of device(s) to use for streaming. "
           "-1 selects all available devices", nargs='*')
    argparmain.add_argument(
      '--dz', dest='dz', type=float,
      help="Ratio of physical Z to X/Y dimension. "
           "X/Y is assumed to be isotropic. Defaults to  1")
    argparmain.add_argument(
      '--weights', dest='weights', type=float,
      help="Ratio of unkowns to each other. Defaults to 1. "
           "If passed, needs to be in the same size as the number of unknowns",
           nargs='*')
    argparmain.add_argument(
      '--useCGguess', dest='usecg', type=_str2bool,
      help="Switch between CG sense and simple FFT as "
           "initial guess for the images.")
    argparmain.add_argument('--out', dest='outdir', type=str,
                            help="Set output directory. Defaults to the input "
                            "file directory")
    argparmain.add_argument(
      '--model', dest='sig_model',
      help='Name of the signal model to use. Defaults to VFA. \
 Please put your signal model file in the Model subfolder.')
    argparmain.add_argument(
      '--modelfile', dest='modelfile', type=str,
      help="Path to the model file.")
    argparmain.add_argument(
      '--modelname', dest='modelname', type=str,
      help="Name of the model to use.")
    argparmain.add_argument('--outdir', dest='outdir', type=str,
                            help="The path to the output directory. Defaults "
                            "to the location of the input.")
    argparmain.add_argument(
      '--double_precision', dest='double_precision', type=_str2bool,
      help="Switch between single (False, default) and double "
           "precision (True). Usually, single precision gives high enough "
           "accuracy.")

    arguments, unknown = argparmain.parse_known_args(args)
    return arguments, unknown


if __name__ == '__main__':
    run()
