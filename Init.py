import numpy as np

import time
import os
import argparse
import h5py
import sys
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier as nlinvns


import goldcomp as goldcomp

import matplotlib.pyplot as plt
import ipyparallel as ipp

from gridroutines import gridding
import pyopencl as cl
import pyopencl.array as clarray
import importlib
import configparser

DTYPE = np.complex64
DTYPE_real = np.float32

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def NUFFT(N,NScan,NC,NSlice,traj,dcf,trafo=1,mask=None):
  platforms = cl.get_platforms()
  ctx = cl.Context(
            dev_type=cl.device_type.GPU,
            properties=[(cl.context_properties.PLATFORM, platforms[1])])
  queue=[]
  queue.append(cl.CommandQueue(ctx,platforms[1].get_devices()[0]))
  if trafo:
    FFT = gridding(ctx,queue,4,2,N,NScan,(NScan*NC*NSlice,N,N),(1,2),traj.astype(DTYPE),np.require(np.abs(dcf),DTYPE_real,requirements='C'),N,1000,DTYPE,DTYPE_real,radial=trafo)
  else:
    FFT = gridding(ctx,queue,4,2,N,NScan,(NScan*NC*NSlice,N,N),(1,2),traj,dcf,N,1000,DTYPE,DTYPE_real,radial=trafo,mask=mask)
  return (ctx,queue[0],FFT)

def gen_default_config():

  config = configparser.ConfigParser()

  config['DEFAULT'] = {}
  config['DEFAULT']["max_iters"] = '300'
  config['DEFAULT']["start_iters"] = '100'
  config['DEFAULT']["max_gn_it"] = '13'
  config['DEFAULT']["lambd"] = '1e2'
  config['DEFAULT']["gamma"] = '1e0'
  config['DEFAULT']["delta"] = '1e-1'
  config['DEFAULT']["display_iterations"] = 'True'
  config['DEFAULT']["gamma_min"] = '0.18'
  config['DEFAULT']["delta_max"] = '1e2'
  config['DEFAULT']["tol"] = '5e-3'
  config['DEFAULT']["stag"] = '1'
  config['DEFAULT']["delta_inc"] = '2'
  config['DEFAULT']["gamma_dec"] = '0.7'

  config['3D_TGV'] = {}
  config['3D_TGV']["max_iters"] = '300'
  config['3D_TGV']["start_iters"] = '100'
  config['3D_TGV']["max_gn_it"] = '13'
  config['3D_TGV']["lambd"] = '1e2'
  config['3D_TGV']["gamma"] = '2e-3'
  config['3D_TGV']["delta"] = '1e-1'
  config['3D_TGV']["display_iterations"] = 'True'
  config['3D_TGV']["gamma_min"] = '0.8e-3'
  config['3D_TGV']["delta_max"] = '1e2'
  config['3D_TGV']["tol"] = '5e-3'
  config['3D_TGV']["stag"] = '1'
  config['3D_TGV']["delta_inc"] = '2'
  config['3D_TGV']["gamma_dec"] = '0.7'

  config['3D_TV'] = {}
  config['3D_TV']["max_iters"] = '300'
  config['3D_TV']["start_iters"] = '100'
  config['3D_TV']["max_gn_it"] = '13'
  config['3D_TV']["lambd"] = '1e2'
  config['3D_TV']["gamma"] = '2e-3'
  config['3D_TV']["delta"] = '1e-1'
  config['3D_TV']["display_iterations"] = 'True'
  config['3D_TV']["gamma_min"] = '0.8e-3'
  config['3D_TV']["delta_max"] = '1e2'
  config['3D_TV']["tol"] = '5e-3'
  config['3D_TV']["stag"] = '1'
  config['3D_TV']["delta_inc"] = '2'
  config['3D_TV']["gamma_dec"] = '0.7'

  with open('default.ini', 'w') as configfile:
    config.write(configfile)

def read_config(conf_file,reg_type="DEFAULT"):
  config = configparser.ConfigParser()
  try:
    config.read(conf_file+".ini")
  except:
    raise
  else:
    params = {}
    for key in config[reg_type]:
      if key in {'max_gn_it','max_iters','start_iters'}:
        params[key] = int(config[reg_type][key])
      elif key == 'display_iterations':
        params[key] = config[reg_type].getboolean(key)
      else:
        params[key] = float(config[reg_type][key])
    return params


def main(args):
    sig_model = importlib.import_module(str(args.sig_model)+"_model")
    if args.streamed:
      import Model_Reco_OpenCL_streamed_Kyk2_sep as Model_Reco
    else:
      import Model_Reco_OpenCL as Model_Reco
    DTYPE = np.complex64
    np.seterr(divide='ignore', invalid='ignore')
################################################################################
### Initiate parallel interface ################################################
################################################################################
    c = ipp.Client()
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

#    if os.path.isfile(file[:-3]+"_sense.h5"):
#      coil_file = h5py.File(file[:-3]+"_sense.h5")
#      coils = coil_file['real_dat'][()]+1j*coil_file['imag_dat'][()]
#      coils = np.transpose(coils,(0,3,1,2))

    file = h5py.File(file)
#    file['Coils'][...] = coils.astype(DTYPE)


################################################################################
### Check if file contains all necessary information ###########################
################################################################################
#    if args.trafo:
#      test_data = ['dcf', 'fa_corr','imag_dat', 'imag_traj', 'real_dat', 'real_traj']
#      test_attributes = ['image_dimensions','flip_angle(s)','TR',\
#                         'data_normalized_with_dcf']
#    else:
#      test_data = ['fa_corr','imag_dat', 'real_dat']
#      test_attributes = ['image_dimensions','flip_angle(s)','TR']
#
#
#
#    for datasets in test_data:
#        if not (datasets in list(file.keys())):
#            file.close()
#            raise NameError("Error: '" + datasets + \
#                            "' data was not provided/wrongly named!")
#    for attributes in test_attributes:
#        if not (attributes in list(file.attrs)):
#            file.close()
#            raise NameError("Error: '" + attributes + \
#                            "' was not provided/wrongly as an attribute!")


################################################################################
### Read Data ##################################################################
################################################################################
    reco_Slices = args.slices
    dimX, dimY, NSlice = (file.attrs['image_dimensions']).astype(int)
    dimX-=3
    dimY-=3
    off = 0

    data = file['real_dat'][...,int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:int(NSlice/2)+int(np.ceil(reco_Slices/2))+off,:,3:-3].astype(DTYPE)\
       +1j*file['imag_dat'][...,int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:int(NSlice/2)+int(np.ceil(reco_Slices/2))+off,:,3:-3].astype(DTYPE)

#    data = np.require(np.transpose(data,(0,1,4,3,2)),requirements='C')
#    data = data[:,0,...]
#    data = np.repeat(data[:,None,...],2,1)


    if args.trafo:
      traj = file['real_traj'][()].astype(DTYPE) + \
             1j*file['imag_traj'][()].astype(DTYPE)
      traj = traj[...,3:-3]
      dcf = np.array(goldcomp.cmp(traj),dtype=DTYPE)
    else:
      traj=None
      dcf = None


    if np.max(prime_factors(data.shape[-1]))>13:
      raise ValueError('Samples along the spoke need to have their largest prime factor to be 13 or lower.')

    #Create par struct to store everyting
    par={}


################################################################################
### FA correction ##############################################################
################################################################################
    try:
      par["fa_corr"] = np.flip(file['fa_corr'][()].astype(DTYPE),0)[int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...]
      par["fa_corr"][par["fa_corr"]==0] = 1
    except:
        pass
    try:
      [NScan,NC,reco_Slices,Nproj, N] = data.shape
    except:
      [NC,reco_Slices,Nproj, N] = data.shape
      Nproj_new = 13

      NScan = np.floor_divide(Nproj,Nproj_new)
      par["Nproj_measured"] = Nproj
      Nproj = Nproj_new
      data = np.require(np.transpose(np.reshape(data[...,:Nproj*NScan,:],\
                                     (NC,reco_Slices,NScan,Nproj,N)),(2,0,1,3,4)),requirements='C')
      traj =np.require(np.reshape(traj[:Nproj*NScan,:],(NScan,Nproj,N)),requirements='C')
      dcf = np.array(goldcomp.cmp(traj),dtype=DTYPE)



#    del file['Coils']
################################################################################
### Set sequence related parameters ############################################
################################################################################
    for att in file.attrs:
      par[att] = file.attrs[att]
    par["NC"]          = NC
    par["dimY"]        = dimY
    par["dimX"]        = dimX
    par["NSlice"]      = reco_Slices
    par["NScan"]       = NScan
    par["N"] = N
    par["Nproj"] = Nproj

    #### TEST
    par["unknowns_TGV"] = 2
    par["unknowns_H1"] = 0
    par["unknowns"] = 2
    if not args.trafo:
      par['mask'] = np.ones_like(np.abs(data)).astype(int)
      (par['mask'])[np.abs(data)==0] = 0
      par['mask'] = np.reshape(par['mask'],(NScan*NC*reco_Slices,dimY,dimX))



################################################################################
### Coil Sensitivity Estimation ################################################
################################################################################
#% Estimates sensitivities and complex image.
#%(see Martin Uecker: Image reconstruction by regularized nonlinear
#%inversion joint estimation of coil sensitivities and image content)
    class B(Exception):
      pass
    try:
      if not file['Coils'][()].shape[1] >= reco_Slices:
        if args.trafo:
          nlinvNewtonSteps = 6
          nlinvRealConstr  = False

          traj_coil = np.reshape(traj,(NScan*Nproj,N))
          dcf_coil = np.repeat(dcf,NScan,0)

          par["C"] = np.zeros((NC,reco_Slices,dimY,dimX), dtype=DTYPE)
          par["phase_map"] = np.zeros((reco_Slices,dimY,dimX), dtype=DTYPE)

          (ctx,queue,FFT) = NUFFT(N,1,1,1,traj_coil,dcf_coil)

          result = []
          for i in range(0,(reco_Slices)):
            sys.stdout.write("Computing coil sensitivity map of slice %i \r" \
                           %(i))
            sys.stdout.flush()

            ##### RADIAL PART
            combinedData = np.transpose(data[:,:,i,:,:],(1,0,2,3))
            combinedData = np.reshape(combinedData,(1,NC,1,NScan*Nproj,N))
            tmp_coilData = clarray.zeros(FFT.queue[0],(1,1,1,dimY,dimX),dtype=DTYPE)
            coilData = np.zeros((NC,dimY,dimX),dtype=DTYPE)
            for j in range(NC):
                tmp_combinedData = clarray.to_device(FFT.queue[0],combinedData[None,:,j,...])
                FFT.adj_NUFFT(tmp_coilData,tmp_combinedData)
                coilData[j,...] = np.squeeze(tmp_coilData.get())

            combinedData = np.fft.fft2(coilData,norm=None)/np.sqrt(dimX*dimY)

            dview = c[int(np.floor(i*len(c)/NSlice))]
            result.append(dview.apply_async(nlinvns.nlinvns, combinedData,
                                            nlinvNewtonSteps, True, nlinvRealConstr))

          for i in range(reco_Slices):
            par["C"][:,i,:,:] = result[i].get()[2:,-1,:,:]
            sys.stdout.write("slice %i done \r" \
                           %(i))
            sys.stdout.flush()
            if not nlinvRealConstr:
              par["phase_map"][i,:,:] = np.exp(1j * np.angle( result[i].get()[0,-1,:,:]))
              par["C"][:,i,:,:] = par["C"][:,i,:,:]* np.exp(1j *\
                   np.angle( result[i].get()[1,-1,:,:]))

              # standardize coil sensitivity profiles
          sumSqrC = np.sqrt(np.sum((par["C"] * np.conj(par["C"])),0)) #4, 9, 128, 128
          if NC == 1:
            par["C"] = sumSqrC
          else:
            par["C"] = par["C"] / np.tile(sumSqrC, (NC,1,1,1))
          del file['Coils']
          del FFT,ctx,queue
        else:
          nlinvNewtonSteps = 6
          nlinvRealConstr  = False

          par["C"] = np.zeros((NC,reco_Slices,dimY,dimX), dtype=DTYPE)
          par["phase_map"] = np.zeros((reco_Slices,dimY,dimX), dtype=DTYPE)

          result = []
          tmp =  np.transpose(np.fft.ifft(np.fft.fft(np.mean(data,0),axis=-3),axis=-1),(0,3,1,2))
          for i in range(0,(reco_Slices)):
            sys.stdout.write("Computing coil sensitivity map of slice %i \r" \
                           %(i))
            sys.stdout.flush()

            ##### RADIAL PART
            combinedData = tmp[:,i,...]

            dview = c[int(np.floor(i*len(c)/NSlice))]
            result.append(dview.apply_async(nlinvns.nlinvns, combinedData,
                                            nlinvNewtonSteps, True, nlinvRealConstr))

          for i in range(reco_Slices):
            par["C"][:,i,:,:] = result[i].get()[2:,-1,:,:]
            sys.stdout.write("slice %i done \r" \
                           %(i))
            sys.stdout.flush()
            if not nlinvRealConstr:
              par["phase_map"][i,:,:] = np.exp(1j * np.angle( result[i].get()[0,-1,:,:]))
              par["C"][:,i,:,:] = par["C"][:,i,:,:]* np.exp(1j *\
                   np.angle( result[i].get()[1,-1,:,:]))

              # standardize coil sensitivity profiles
          sumSqrC = np.sqrt(np.sum((par["C"] * np.conj(par["C"])),0)) #4, 9, 128, 128
          if NC == 1:
            par["C"] = sumSqrC
          else:
            par["C"] = par["C"] / np.tile(sumSqrC, (NC,1,1,1))
          del file['Coils']

        file.create_dataset("Coils",par["C"].shape,dtype=par["C"].dtype,data=par["C"])
        file.flush()

      else:
        print("Using precomputed coil sensitivities")
        slices_coils = file['Coils'][()].shape[1]
        par["C"] = file['Coils'][:,int(slices_coils/2)-int(np.floor((reco_Slices)/2)):int(slices_coils/2)+int(np.ceil(reco_Slices/2)),...]

    except:
      if args.trafo:
        nlinvNewtonSteps = 6
        nlinvRealConstr  = False

        traj_coil = np.reshape(traj,(NScan*Nproj,N))
        dcf_coil = np.repeat(dcf,NScan,0)

        par["C"] = np.zeros((NC,reco_Slices,dimY,dimX), dtype=DTYPE)
        par["phase_map"] = np.zeros((reco_Slices,dimY,dimX), dtype=DTYPE)

        (ctx,queue,FFT) = NUFFT(N,1,1,1,traj_coil,dcf_coil)

        result = []
        for i in range(0,(reco_Slices)):
          sys.stdout.write("Computing coil sensitivity map of slice %i \r" \
                         %(i))
          sys.stdout.flush()

          ##### RADIAL PART
          combinedData = np.transpose(data[:,:,i,:,:],(1,0,2,3))
          combinedData = np.reshape(combinedData,(1,NC,1,NScan*Nproj,N))
          tmp_coilData = clarray.zeros(FFT.queue[0],(1,1,1,dimY,dimX),dtype=DTYPE)
          coilData = np.zeros((NC,dimY,dimX),dtype=DTYPE)
          for j in range(NC):
              tmp_combinedData = clarray.to_device(FFT.queue[0],combinedData[None,:,j,...])
              FFT.adj_NUFFT(tmp_coilData,tmp_combinedData)
              coilData[j,...] = np.squeeze(tmp_coilData.get())

          combinedData = np.fft.fft2(coilData,norm=None)/np.sqrt(dimX*dimY)

          dview = c[int(np.floor(i*len(c)/NSlice))]
          result.append(dview.apply_async(nlinvns.nlinvns, combinedData,
                                          nlinvNewtonSteps, True, nlinvRealConstr))

        for i in range(reco_Slices):
          par["C"][:,i,:,:] = result[i].get()[2:,-1,:,:]
          sys.stdout.write("slice %i done \r" \
                         %(i))
          sys.stdout.flush()
          if not nlinvRealConstr:
            par["phase_map"][i,:,:] = np.exp(1j * np.angle( result[i].get()[0,-1,:,:]))
            par["C"][:,i,:,:] = par["C"][:,i,:,:]* np.exp(1j *\
                 np.angle( result[i].get()[1,-1,:,:]))

            # standardize coil sensitivity profiles
        sumSqrC = np.sqrt(np.sum((par["C"] * np.conj(par["C"])),0)) #4, 9, 128, 128
        if NC == 1:
          par["C"] = sumSqrC
        else:
          par["C"] = par["C"] / np.tile(sumSqrC, (NC,1,1,1))
        del FFT,ctx,queue
      else:
        nlinvNewtonSteps = 6
        nlinvRealConstr  = False

        par["C"] = np.zeros((NC,reco_Slices,dimY,dimX), dtype=DTYPE)
        par["phase_map"] = np.zeros((reco_Slices,dimY,dimX), dtype=DTYPE)

        result = []
        for i in range(0,(reco_Slices)):
          sys.stdout.write("Computing coil sensitivity map of slice %i \r" \
                         %(i))
          sys.stdout.flush()

          ##### RADIAL PART
          combinedData =  np.fft.fftshift(np.mean(data[:,:,i,:,:],0),(-1,-2))
#          combinedData = np.mean(combinedData,0)

          dview = c[int(np.floor(i*len(c)/NSlice))]
          result.append(dview.apply_async(nlinvns.nlinvns, combinedData,
                                          nlinvNewtonSteps, True, nlinvRealConstr))

        for i in range(reco_Slices):
          par["C"][:,i,:,:] = result[i].get()[2:,-1,:,:]
          sys.stdout.write("slice %i done \r" \
                         %(i))
          sys.stdout.flush()
          if not nlinvRealConstr:
            par["phase_map"][i,:,:] = np.exp(1j * np.angle( result[i].get()[0,-1,:,:]))
            par["C"][:,i,:,:] = par["C"][:,i,:,:]* np.exp(1j *\
                 np.angle( result[i].get()[1,-1,:,:]))

            # standardize coil sensitivity profiles
        sumSqrC = np.sqrt(np.sum((par["C"] * np.conj(par["C"])),0)) #4, 9, 128, 128
        if NC == 1:
          par["C"] = sumSqrC
        else:
          par["C"] = par["C"] / np.tile(sumSqrC, (NC,1,1,1))
      file.create_dataset("Coils",par["C"].shape,dtype=par["C"].dtype,data=par["C"])
      file.flush()


##################
#    par["C"] = np.repeat(0.5*np.ones_like(par["C"][0,...])[None,...],2,0)
################

################################################################################
### Standardize data norm ######################################################
################################################################################
    [NScan,NC,NSlice,Nproj, N] = data.shape
    if args.trafo:
      if file.attrs['data_normalized_with_dcf']:
          pass
      else:
          data = data*np.sqrt(dcf)
#### Close File after everything was read
    file.close()
    dscale = np.sqrt(NSlice)*DTYPE(np.sqrt(2*1e3))/(np.linalg.norm(data.flatten()))
    par["dscale"] = dscale
    data = data* dscale
################################################################################
### generate nFFT  #############################################################
################################################################################

    if args.trafo:
      (ctx,queue,FFT) = NUFFT(N,NScan,1,1,traj,np.sqrt(dcf))

      def nFTH(x,fft,dcf,NScan,NC,NSlice,dimY,dimX):
        siz = np.shape(x)
        result = np.zeros((NC,NSlice,NScan,dimY,dimX),dtype=DTYPE)
        tmp_result = clarray.zeros(fft.queue[0],(NScan,1,1,dimY,dimX),dtype=DTYPE)
        for j in range(siz[1]):
          for k in range(siz[2]):
            inp = clarray.to_device(fft.queue[0],np.require(x[:,j,k,...][:,None,None,...],requirements='C'))
            fft.adj_NUFFT(tmp_result,inp)
            result[j,k,...] = np.squeeze(tmp_result.get())
        return np.transpose(result,(2,0,1,3,4))

      images= np.require(np.sum(nFTH(data,FFT,dcf,NScan,NC,\
                     NSlice,dimY,dimX)*(np.conj(par["C"])),axis = 1),requirements='C')
      del FFT,ctx,queue
      try:
        par["fa_corr"] = np.require((np.transpose(par["fa_corr"],(0,2,1))),requirements='C')
      except:
        pass
    else:
#      (ctx,queue,FFT) = NUFFT(N,NScan,NC,NSlice,traj,dcf,trafo=0,mask=par['mask'])
      images= (np.sum(np.fft.ifft2(data,norm='ortho')*(np.conj(par["C"])),axis = 1))

    par['C'] = par['C'].astype(DTYPE)


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
    if args.trafo:
      opt = Model_Reco.Model_Reco(par,ctx,queue,traj,np.sqrt(dcf),args.trafo)
      opt.data =  data
    else:
      opt = Model_Reco.Model_Reco(par,ctx,queue,None,None,args.trafo)
      opt.data =  data

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
        opt.dz = 1

    ################################################################################
    ##IRGN Params
        try:
          opt.irgn_par = read_config(args.config,"3D_TGV")
        except:
          print("Config file not readable or not found. Falling back to default.")
          gen_default_config()
          opt.irgn_par = read_config("default","3D_TGV")

        opt.execute_3D()
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
    #############################################################re###################
        model = sig_model.Model(par,images)
        opt.model = model
        opt.dz = 1
    ################################################################################
    ##IRGN Params
        try:
          opt.irgn_par = read_config(args.config,"3D_TV")
        except:
          print("Config file not readable or not found. Falling back to default.")
          gen_default_config()
          opt.irgn_par = read_config("default","3D_TV")
        opt.execute_3D(1)
        result_tv.append(opt.result)
        plt.close('all')

        res_tv = opt.gn_res
        res_tv = np.array(res_tv)/(opt.irgn_par["lambd"]*NSlice)

      del opt
#    else:
#################################################################################
#### IRGN - TGV Reco ############################################################
#################################################################################
#      if "TGV" in args.reg or args.reg=='all':
#        result_tgv = []
#    ################################################################################
#    ### Init forward model and initial guess #######################################
#    ################################################################################
#        model = sig_model.Model(par,images)
#        opt.model = model
#        opt.dz = 1
#
#    ################################################################################
#    ##IRGN Params
#        irgn_par = {}
#        if args.def_par:
#          irgn_par = user_input()
#        else:
#          irgn_par["max_iters"] = 300
#          irgn_par["start_iters"] = 100
#          irgn_par["max_gn_it"] = 13
#          irgn_par["lambd"] = 1e2
#          irgn_par["gamma"] = 1e0
#          irgn_par["delta"] = 1e-1
#          irgn_par["display_iterations"] = True
#          irgn_par["gamma_min"] = 0.18
#          irgn_par["delta_max"] = 1e2
#          irgn_par["tol"] = 5e-3
#          irgn_par["stag"] = 1
#          irgn_par["delta_inc"] = 2
#          irgn_par["gamma_dec"] = 0.7
#        opt.irgn_par = irgn_par
#
#        opt.execute_2D()
#        result_tgv.append(opt.result)
#        plt.close('all')
#
#
#        res_tgv = opt.gn_res
#        res_tgv = np.array(res_tgv)/(irgn_par["lambd"]*NSlice)
#  ################################################################################
#  #### IRGN - TV referenz ########################################################
#  ################################################################################
#      if "TV" in args.reg or args.reg=='all':
#        result_tv = []
#    ################################################################################
#    ### Init forward model and initial guess #######################################
#    #############################################################re###################
#        model = sig_model.Model(par,images)
#        opt.model = model
#    ################################################################################
#    ##IRGN Params
#        irgn_par = {}
#        if args.def_par:
#          irgn_par = user_input()
#        else:
#          irgn_par["max_iters"] = 300
#          irgn_par["start_iters"] = 100
#          irgn_par["max_gn_it"] = 13
#          irgn_par["lambd"] = 1e2
#          irgn_par["gamma"] = 1e0
#          irgn_par["delta"] = 1e-1
#          irgn_par["display_iterations"] = True
#          irgn_par["gamma_min"] = 0.23
#          irgn_par["delta_max"] = 1e2
#          irgn_par["tol"] = 5e-3
#          irgn_par["stag"] = 1
#          irgn_par["delta_inc"] = 2
#          irgn_par["gamma_dec"] = 0.7
#        opt.irgn_par = irgn_par
#
#        opt.execute_2D(1)
#        opt.result[:,1,...] = -par["TR"]/np.log(opt.result[:,1,...])
#        result_tv.append(opt.result)
#        plt.close('all')
#
#        res_tv = opt.gn_res
#        res_tv = np.array(res_tv)/(irgn_par["lambd"]*NSlice)
#
#      del opt
###############################################################################
## New .hdf5 save files #######################################################
###############################################################################
    outdir = time.strftime("%Y-%m-%d  %H-%M-%S_MRI_"+args.reg+"_"+args.type+"_"+name[:-3])
    if not os.path.exists('./output'):
        os.makedirs('./output')
    os.makedirs("output/"+ outdir)

    os.chdir("output/"+ outdir)
    f = h5py.File("output_"+name,"w")

    for i in range(len(result_tgv)):
      if "TGV" in args.reg or args.reg=='all':
        f.create_dataset("tgv_full_result_"+str(i),result_tgv[i].shape,\
                                     dtype=DTYPE,data=result_tgv[i])
        f.attrs['res_tgv'] = res_tgv
      if "TV" in args.reg or args.reg=='all':
        f.create_dataset("tv_full_result_"+str(i),result_tv[i].shape,\
                                         dtype=DTYPE,data=result_tv[i])
        f.attrs['res_tv'] = res_tv
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
    parser.add_argument('--slices',default=1, dest='slices', type=int, help='Number of reconstructed slices (default=40). Symmetrical around the center slice.')
    parser.add_argument('--trafo', default=1, dest='trafo',  help='Choos between radial (1, default) and Cartesian (0) sampling. ')
    parser.add_argument('--streamed', default=0, dest='streamed',  help='Enable streaming of large data arrays (>10 slices).')
    parser.add_argument('--data',default='',dest='file',help='Full path to input data. If not provided, a file dialog will open.')
    parser.add_argument('--model',default='VFA_exp',dest='sig_model',help='Name of the signal model to use. Defaults to VFA. Please name your signal model file MODEL_model.')
    parser.add_argument('--config',default='default',dest='config',help='Name of config file to use (assumed to be in the same folder). If not specified, use default parameters.')
    args = parser.parse_args()

    main(args)