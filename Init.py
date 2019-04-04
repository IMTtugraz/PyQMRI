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

from helper_fun import goldcomp as goldcomp
from helper_fun.est_coils import est_coils
from helper_fun import utils

DTYPE = np.complex64
DTYPE_real = np.float32


def main(args):
    sig_model = importlib.import_module("Models."+str(args.sig_model))
    if int(args.streamed) == 1:
        import IRGN.Model_Reco_OpenCL_streamed as Model_Reco
    else:
        import IRGN.Model_Reco_OpenCL as Model_Reco
    np.seterr(divide='ignore', invalid='ignore')

# Create par struct to store everyting
    par = {}
###############################################################################
# Select input file ###########################################################
###############################################################################
    if args.file == '':
        root = Tk()
        root.withdraw()
        root.update()
        file = filedialog.askopenfilename()
        root.destroy()
    else:
        file = args.file

    name = file.split('/')[-1]

    par["file"] = h5py.File(file)
###############################################################################
# Read Data ###################################################################
###############################################################################
    reco_Slices = args.slices
    dimX, dimY, NSlice = ((par["file"].attrs['image_dimensions']).astype(int))
    if reco_Slices == -1:
        reco_Slices = NSlice
    off = 0

    if args.sms:
        data = par["file"]['real_dat'][()].astype(DTYPE)\
               + 1j*par["file"]['imag_dat'][()].astype(DTYPE)
    else:
        data = par["file"]['real_dat'][
          ..., int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:
          int(NSlice/2)+int(np.ceil(reco_Slices/2))+off, :, :].astype(DTYPE)\
               + 1j*par["file"]['imag_dat'][
          ..., int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:
          int(NSlice/2)+int(np.ceil(reco_Slices/2))+off, :, :].astype(DTYPE)


##    data /= dimX
##
#    images = par["file"]["GT/SI"][()].astype(DTYPE)
##
#    del par["file"]['Coils']
##
##    data = data+np.max(np.abs(data))*0.0001*(np.random.standard_normal(data.shape).astype(DTYPE_real)+1j*np.random.standard_normal(data.shape).astype(DTYPE_real))
#
##    norm_coils = par["file"]["GT/sensitivities/real_dat"][2:] + 1j*par["file"]["GT/sensitivities/imag_dat"][2:]
##    norm_coils = par["file"]["Coils_real"][...] + 1j*par["file"]["Coils_imag"][...]
##    norm_coils = np.array((norm_coils[0],norm_coils[-2],norm_coils[-1]))
##    norm = np.sqrt(np.sum(np.abs(norm_coils)**2,0,keepdims=True))
##    norm_coils = (norm_coils)/norm
###    norm_coils = norm_coils/np.max(np.abs(norm_coils))
##    norm_coils = (norm_coils)/np.linalg.norm(norm_coils)*128
##    norm_coils = np.abs(norm_coils).astype(DTYPE)
#
#
#    data = np.require(np.fft.fft2(images[:,None,...],norm='ortho'),requirements='C')
##    data = data+np.max(np.abs(data))*0.0001*(np.random.standard_normal(data.shape).astype(DTYPE_real)+1j*np.random.standard_normal(data.shape).astype(DTYPE_real))
##    del par["file"]["GT/sensitivities/real_dat"]
##    del par["file"]["GT/sensitivities/imag_dat"]
#    del par["file"]["real_dat"]
#    del par["file"]["imag_dat"]
#    par["file"].create_dataset("real_dat",data.shape,dtype=DTYPE_real,data=np.real(data))
#    par["file"].create_dataset("imag_dat",data.shape,dtype=DTYPE_real,data=np.imag(data))
##    par["file"].create_dataset("GT/sensitivities/real_dat",norm_coils.shape,dtype=DTYPE_real,data=np.real(norm_coils))
##    par["file"].create_dataset("GT/sensitivities/imag_dat",norm_coils.shape,dtype=DTYPE_real,data=np.imag(norm_coils))

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
        print('Samples along the spoke need to have their largest prime factor\
 to be 13 or lower. Finding next smaller grid. ')
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
        if args.sms:
            par["fa_corr"] = np.flip(par["file"]['fa_corr'][()].astype(DTYPE),
                                     0)[...]
        else:
            par["fa_corr"] = np.flip(par["file"]['fa_corr'][()].astype(DTYPE),
                                     0)[
                                 int(NSlice/2)-int(np.floor((reco_Slices)/2)):
                                 int(NSlice/2)+int(np.ceil(reco_Slices/2)),
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
    elif data.ndim == 4 and "IRLL" in args.sig_model:
        print("4D Data passed and IRLL model used. Reordering Projections \
 into 8 Spokes/Frame")
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
    if args.sms:
        par["NSlice"] = NSlice
        par["packs"] = reco_Slices
    else:
        par["NSlice"] = reco_Slices
    par["NScan"] = NScan
    par["N"] = N
    par["Nproj"] = Nproj

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
                print("GPU OpenCL platform <%s> found\
 with %i device(s) and OpenCL-version <%s>"
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
                print("CPU OpenCL platform <%s> found\
 with %i device(s) and OpenCL-version <%s>"
                      % (str(platforms[j].get_info(cl.platform_info.NAME)),
                         len(platforms[j].get_devices(
                            device_type=cl.device_type.GPU)),
                         str(platforms[j].get_info(cl.platform_info.VERSION))))
                par["GPU"] = False
                par["Platform_Indx"] = j

    par["ctx"] = []
    par["queue"] = []
    num_dev = len(platforms[par["Platform_Indx"]].get_devices())
    par["num_dev"] = num_dev
    for device in range(num_dev):
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
###############################################################################
# Coil Sensitivity Estimation #################################################
###############################################################################
    est_coils(data, par, par["file"], args)
###############################################################################
# Standardize data norm #######################################################
###############################################################################
    [NScan, NC, NSlice, Nproj, N] = data.shape
    if args.trafo:
        if par["file"].attrs['data_normalized_with_dcf']:
            pass
        else:
            data = data*(par["dcf"])
    dscale = np.sqrt(NSlice) * \
                    (DTYPE(np.sqrt(2*1e3)) / (np.linalg.norm(data.flatten())))
    par["dscale"] = dscale
    data *= dscale
###############################################################################
# generate nFFT  ##############################################################
###############################################################################
    if NC == 1:
        par['C'] = np.ones((data[0, ...].shape), dtype=DTYPE)
    else:
        par['C'] = par['C'].astype(DTYPE)



    FFT = utils.NUFFT(par, trafo=args.trafo, SMS=args.sms)

    def nFTH(x, fft, par):
        siz = np.shape(x)
        result = np.zeros((par["NC"], par["NSlice"], par["NScan"],
                           par["dimY"], par["dimX"]), dtype=DTYPE)
        tmp_result = clarray.empty(fft.queue[0], (par["NScan"], 1, 1,
                                   par["dimY"], par["dimX"]), dtype=DTYPE)
        for j in range(siz[1]):
            for k in range(siz[2]):
                inp = clarray.to_device(fft.queue[0],
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
#    par['C'] = par["file"]["GT/sensitivities/real_dat"][()] + 1j*par["file"]["GT/sensitivities/imag_dat"][()]
    opt = Model_Reco.Model_Reco(par, args.trafo,
                                imagespace=args.imagespace, SMS=args.sms)

#    images = par["file"]["GT/SI"][()].astype(DTYPE)
#    dscale = np.sqrt(NSlice)*DTYPE(np.sqrt(2*1e3)) /\
#        (np.linalg.norm(images.flatten()))
#    images *= dscale
    if args.imagespace:
        opt.data = images
    else:
        opt.data = data
###############################################################################
# ratio of z direction to x,y, important for finite differences ###############
###############################################################################
    opt.dz = 1
###############################################################################
# Start Reco ##################################################################
###############################################################################
    if args.type == '3D':
        #######################################################################
        # Init forward model and initial guess ################################
        #######################################################################
        model = sig_model.Model(par, images)
        # Close File after everything was read
#        test = model.execute_forward(model.guess)
#        ratio = np.mean((images[:20]))/np.mean((test[:20]))
#        print(ratio.real)
#        model.guess[0] *= ratio.real
#        test = model.execute_forward(model.guess)
#        print((np.mean((images[:20]))/np.mean((test[:20]))).real)
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
                           + name[:-3])
#    outdir = "noise_prop_test"
    if not os.path.exists('./output'):
        os.makedirs('./output')
    if not os.path.exists('./output/' + outdir):
        os.makedirs("output/" + outdir)
    cwd = os.getcwd()
    os.chdir("output/" + outdir)
    f = h5py.File("output_" + name, "w")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T1 quantification from VFA \
 data. By default runs 3D regularization for TGV and TV.')
    parser.add_argument(
      '--recon_type', default='3D', dest='type',
      help='Choose reconstruction type (currently only 3D)')
    parser.add_argument(
      '--reg_type', default='TGV', dest='reg',
      help="Choose regularization type (default: TGV) \
      options are: TGV, TV, all")
    parser.add_argument(
      '--slices', default=180, dest='slices', type=int,
      help='Number of reconstructed slices (default=40). \
      Symmetrical around the center slice.')
    parser.add_argument(
      '--trafo', default=1, dest='trafo', type=int,
      help='Choos between radial (1, default) and Cartesian (0) sampling. ')
    parser.add_argument(
      '--streamed', default=1, dest='streamed', type=int,
      help='Enable streaming of large data arrays (>10 slices).')
    parser.add_argument(
      '--data', default='', dest='file',
      help='Full path to input data. \
      If not provided, a file dialog will open.')
    parser.add_argument(
      '--model', default='VFA', dest='sig_model',
      help='Name of the signal model to use. Defaults to VFA. \
 Please put your signal model file in the Model subfolder.')
    parser.add_argument(
      '--config', default='test', dest='config',
      help='Name of config file to use (assumed to be in the same folder). \
 If not specified, use default parameters.')
    parser.add_argument(
      '--sms', default=0, dest='sms', type=int,
      help='Simultanious Multi Slice, defaults to off (0). \
      Can only be used with Cartesian sampling.')
    parser.add_argument(
      '--imagespace', default=0, dest='imagespace', type=int,
      help='Select if Reco is performed on images (1) or on kspace (0) data. \
 Defaults to 0')
    parser.add_argument(
      '--OCL_GPU', default=1, dest='use_GPU', type=int,
      help='Select if CPU or GPU should be used as OpenCL platform. \
 Defaults to GPU (1). CAVE: CPU FFT not working')
    args = parser.parse_args()
    main(args)
