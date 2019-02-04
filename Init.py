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

from helper_fun import  goldcomp as goldcomp
from helper_fun.est_coils import est_coils
from helper_fun import utils

DTYPE = np.complex64
DTYPE_real = np.float32

def main(args):
    sig_model = importlib.import_module("Models."+str(args.sig_model))
    if int(args.streamed)==1:
      import IRGN.Model_Reco_OpenCL_streamed as Model_Reco
    else:
      import IRGN.Model_Reco_OpenCL as Model_Reco
    np.seterr(divide='ignore', invalid='ignore')
################################################################################
### Select input file ##########################################################
################################################################################
    if args.file == '':
      root = Tk()
      root.withdraw()
      root.update()
      file = filedialog.askopenfilename()
      root.destroy()
    else:
      file = args.file

    name = file.split('/')[-1]

    file = h5py.File(file)
#    del file['Coils']
################################################################################
### Read Data ##################################################################
################################################################################
    #Create par struct to store everyting
    par={}
    reco_Slices = args.slices
    dimX, dimY, NSlice = ((file.attrs['image_dimensions']).astype(int))
    if reco_Slices==-1:
      reco_Slices=NSlice
    off = 0

    if args.sms:
      data = file['real_dat'][()].astype(DTYPE)\
       +1j*file['imag_dat'][()].astype(DTYPE)
    else:
      data = file['real_dat'][...,int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:int(NSlice/2)+int(np.ceil(reco_Slices/2))+off,:,:].astype(DTYPE)\
       +1j*file['imag_dat'][...,int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:int(NSlice/2)+int(np.ceil(reco_Slices/2))+off,:,:].astype(DTYPE)

#    print(data.shape)
#    check = np.outer((-1)**(np.linspace(1,100,100)),(-1)**(np.linspace(1,100,100)))
#    data = np.fft.ifft(np.fft.fft(data,axis=-1)*check,axis=2)
#    data = np.require(np.fft.fft2(np.fft.ifftshift(np.fft.ifft2(data),(-1,-2))),dtype=DTYPE,requirements='C')
#    data = np.require(np.transpose(data,(0,1,4,3,2)),requirements='C')
#    data = data[:,0,...][:,None,...]
#    data = np.repeat(data[:,None,...],2,1)

    if args.trafo:
      par["traj"] = file['real_traj'][()].astype(DTYPE) + \
             1j*file['imag_traj'][()].astype(DTYPE)

      par["dcf"] = np.sqrt(np.array(goldcomp.cmp( par["traj"]),dtype=DTYPE_real)).astype(DTYPE_real)
      par["dcf"] = np.require(np.abs(par["dcf"]),DTYPE_real,requirements='C')
    else:
      par["traj"]=None
      par["dcf"] = None
    if np.max(utils.prime_factors(data.shape[-1]))>13:
      raise ValueError('Samples along the spoke need to have their largest prime factor to be 13 or lower.')
#      print(data.shape[-1])
#      data = np.require(data[...,3:-3],requirements='C')
#      dimX -=3
#      dimY -=3
#      traj = traj[...,3:-3]
#      dcf = dcf[...,3:-3]
################################################################################
### FA correction ##############################################################
################################################################################
    try:
      if args.sms:
        par["fa_corr"] = np.flip(file['fa_corr'][()].astype(DTYPE),0)[...]
      else:
        par["fa_corr"] = np.flip(file['fa_corr'][()].astype(DTYPE),0)[int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...]
      par["fa_corr"][par["fa_corr"]==0] = 1
    except:
        print("No flip angle correction provided.")
    try:
      [NScan,NC,reco_Slices,Nproj, N] = data.shape
    except:
      [NC,reco_Slices,Nproj, N] = data.shape
      Nproj_new = 2
      NScan = np.floor_divide(Nproj,Nproj_new)
      par["Nproj_measured"] = Nproj
      Nproj = Nproj_new
      data = np.require(np.transpose(np.reshape(data[...,:Nproj*NScan,:],\
                                     (NC,reco_Slices,NScan,Nproj,N)),(2,0,1,3,4)),requirements='C')
      par["traj"] =np.require(np.reshape(par["traj"][:Nproj*NScan,:],(NScan,Nproj,N)),requirements='C')
      par["dcf"] = np.array(goldcomp.cmp(par["traj"]),dtype=DTYPE)
################################################################################
### Set sequence related parameters ############################################
################################################################################
    for att in file.attrs:
      par[att] = file.attrs[att]

    par["NC"]          = NC
    par["dimY"]        = dimY
    par["dimX"]        = dimX
    if args.sms:
      par["NSlice"] = NSlice
      par["packs"] = reco_Slices
    else:
      par["NSlice"]      = reco_Slices
    par["NScan"]       = NScan
    par["N"] = N
    par["Nproj"] = Nproj

    par["unknowns_TGV"] = sig_model.unknowns_TGV
    par["unknowns_H1"] = sig_model.unknowns_H1
    par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]
    if not args.trafo:
      tmp = np.ones_like(np.abs(data))
      tmp[np.abs(data)==0] = 0
      par['mask'] = np.reshape(tmp,(data.shape)).astype(DTYPE_real)
      del tmp
    else:
      par['mask']=None

################################################################################
### Coil Sensitivity Estimation ################################################
################################################################################
    est_coils(data,par,file,args)
################################################################################
### Standardize data norm ######################################################
################################################################################
    [NScan,NC,NSlice,Nproj, N] = data.shape
    if args.trafo:
      if file.attrs['data_normalized_with_dcf']:
          pass
      else:
          data = data*(par["dcf"])
#### Close File after everything was read
    file.close()
#    dscale = np.sqrt(NSlice)*DTYPE(np.sqrt(2*1e3))/(np.linalg.norm(data[:10].flatten()))
#    par["dscale"] = dscale
#    data[:10] = data[:10]* dscale
#    dscale = np.sqrt(NSlice)*DTYPE(np.sqrt(2*1e3))/(np.linalg.norm(data[10:].flatten()))
#    par["dscale"] = dscale
#    data[10:] = data[10:]* dscale
#    scale = np.median(np.abs(data[5:]))/np.median(np.abs(data[:5]))/4.8432
#    data[:5]*=scale
    dscale = np.sqrt(NSlice)*DTYPE(np.sqrt(2*1e3))/(np.linalg.norm(data.flatten()))
    par["dscale"] = dscale
    data*=dscale

################################################################################
### generate nFFT  #############################################################
################################################################################
    if NC ==1:
      par['C'] = np.ones((data[0,...].shape),dtype=DTYPE)
    else:
      par['C'] = par['C'].astype(DTYPE)


    (ctx,queue,FFT) = utils.NUFFT(par,trafo=args.trafo,SMS=args.sms)
    def nFTH(x,fft,par):
      siz = np.shape(x)
      result = np.zeros((par["NC"],par["NSlice"],par["NScan"],par["dimY"],par["dimX"]),dtype=DTYPE)
      tmp_result = clarray.zeros(fft.queue[0],(par["NScan"],1,1,par["dimY"],par["dimX"]),dtype=DTYPE)
      for j in range(siz[1]):
        for k in range(siz[2]):
          inp = clarray.to_device(fft.queue[0],np.require(x[:,j,k,...][:,None,None,...],requirements='C'))
          fft.adj_NUFFT(tmp_result,inp)
          result[j,k,...] = np.squeeze(tmp_result.get())
      return np.transpose(result,(2,0,1,3,4))
    try:
      par["fa_corr"] = np.require((np.transpose(par["fa_corr"],(0,2,1))),requirements='C')
    except:
      pass
    images= np.require(np.sum(nFTH(data,FFT,par)*(np.conj(par["C"])),axis = 1),requirements='C')
    del FFT,ctx,queue

################################################################################
### Create OpenCL Context and Queues ###########################################
################################################################################
    platforms = cl.get_platforms()

    ctx = []
    queue = []
    num_dev = 1#len(platforms[0].get_devices())
    for device in range(num_dev):
      tmp = cl.Context(
              dev_type=cl.device_type.GPU,
              properties=[(cl.context_properties.PLATFORM, platforms[1])])
      ctx.append(tmp)
      queue.append(cl.CommandQueue(tmp,platforms[1].get_devices()[0],properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE | cl.command_queue_properties.PROFILING_ENABLE))
      queue.append(cl.CommandQueue(tmp, platforms[1].get_devices()[0],properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE | cl.command_queue_properties.PROFILING_ENABLE))
      queue.append(cl.CommandQueue(tmp, platforms[1].get_devices()[0],properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE | cl.command_queue_properties.PROFILING_ENABLE))



    opt = Model_Reco.Model_Reco(par,ctx,queue,args.trafo,imagespace=args.imagespace,SMS=args.sms)
    if args.imagespace:
      opt.data =  images
    else:
      opt.data =  data

################################################################################
### ratio of z direction to x,y, important for finite differences ##############
################################################################################
    opt.dz = 1
################################################################################
### Start Reco #################################################################
################################################################################
    if args.type=='3D':
################################################################################
### IRGN - TGV Reco ############################################################
################################################################################
      if "TGV" in args.reg or args.reg=='all':
        result_tgv = []
################################################################################
### Init forward model and initial guess #######################################
################################################################################
        model = sig_model.Model(par,images)
        opt.model = model
################################################################################
##IRGN Params ##################################################################
################################################################################
        opt.irgn_par = utils.read_config(args.config,"3D_TGV")
        opt.execute(TV=0,imagespace=args.imagespace)
        result_tgv.append(opt.result)
        plt.close('all')
        res_tgv = opt.gn_res
        res_tgv = np.array(res_tgv)/(opt.irgn_par["lambd"]*NSlice)
################################################################################
#### IRGN - TV referenz ########################################################
################################################################################
      if "TV" in args.reg or args.reg=='all':
        result_tv = []
################################################################################
### Init forward model and initial guess #######################################
#############################################################re#################
        model = sig_model.Model(par,images)
        opt.model = model
################################################################################
##IRGN Params ##################################################################
################################################################################
        opt.irgn_par = utils.read_config(args.config,"3D_TV")
        opt.execute(TV=1,imagespace=args.imagespace)
        result_tv.append(opt.result)
        plt.close('all')
        res_tv = opt.gn_res
        res_tv = np.array(res_tv)/(opt.irgn_par["lambd"]*NSlice)
      del opt
###############################################################################
## New .hdf5 save files #######################################################
###############################################################################
    outdir = time.strftime("%Y-%m-%d  %H-%M-%S_MRI_"+args.reg+"_"+args.type+"_"+name[:-3])
    if not os.path.exists('./output'):
        os.makedirs('./output')
    os.makedirs("output/"+ outdir)

    os.chdir("output/"+ outdir)
    f = h5py.File("output_"+name,"w")

    if "TGV" in args.reg or args.reg=='all':
      for i in range(len(result_tgv)):
        f.create_dataset("tgv_full_result_"+str(i),result_tgv[i].shape,\
                                     dtype=DTYPE,data=result_tgv[i])
        f.attrs['res_tgv'] = res_tgv
    if "TV" in args.reg or args.reg=='all':
      for i in range(len(result_tv)):
        f.create_dataset("tv_full_result_"+str(i),result_tv[i].shape,\
                                         dtype=DTYPE,data=result_tv[i])
        f.attrs['res_tv'] = res_tv

    if "imagespace" in args.reg or args.reg=='all':
      for i in range(len(result_tgv)):
        f.create_dataset("imagespace_full_result_"+str(i),result_tgv[i].shape,\
                                     dtype=DTYPE,data=result_tgv[i])
        f.attrs['imagespace_tgv'] = res_tgv
      f.attrs['data_norm'] = dscale
      f.flush()
    f.close()

    os.chdir('..')
    os.chdir('..')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T1 quantification from VFA data. By default runs 3D regularization for TGV and TV.')
    parser.add_argument('--recon_type', default='3D', dest='type',help='Choose reconstruction type (currently only 3D)')
    parser.add_argument('--reg_type', default='TGV', dest='reg',help="Choose regularization type (default: TGV)\
                                                                     options are: TGV, TV, all")
    parser.add_argument('--slices',default=4, dest='slices', type=int, help='Number of reconstructed slices (default=40). Symmetrical around the center slice.')
    parser.add_argument('--trafo', default=1, dest='trafo', type=int, help='Choos between radial (1, default) and Cartesian (0) sampling. ')
    parser.add_argument('--streamed', default=1, dest='streamed', type=int, help='Enable streaming of large data arrays (>10 slices).')
    parser.add_argument('--data',default='',dest='file',help='Full path to input data. If not provided, a file dialog will open.')
    parser.add_argument('--model',default='VFA',dest='sig_model',help='Name of the signal model to use. Defaults to VFA. Please name your signal model file MODEL_model.')
    parser.add_argument('--config',default='test',dest='config',help='Name of config file to use (assumed to be in the same folder). If not specified, use default parameters.')
    parser.add_argument('--sms',default=0, dest='sms', type=int, help='Simultanious Multi Slice, defaults to off (0). Can only be used with Cartesian sampling.')
    parser.add_argument('--imagespace',default=0,dest='imagespace',type=int,help='Select if Reco is performed on images (1) or on kspace (0) data. Defaults to 0')
    args = parser.parse_args()

    main(args)