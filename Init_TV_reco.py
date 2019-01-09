import numpy as np
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
import configparser
import TV_Reco_OpenCL as Model_Reco

import multislice_viewer as msv
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
    file = h5py.File(file)


################################################################################
### Read Data ##################################################################
################################################################################
    reco_Slices = args.slices
    dimX, dimY, NSlice = ((file.attrs['image_dimensions']).astype(int))
    off = 0

    data = file['real_dat'][-1,...,int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:int(NSlice/2)+int(np.ceil(reco_Slices/2))+off,:,:].astype(DTYPE)\
       +1j*file['imag_dat'][-1,...,int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:int(NSlice/2)+int(np.ceil(reco_Slices/2))+off,:,:].astype(DTYPE)


    if args.trafo:
      traj = file['real_traj'][-1,...].astype(DTYPE) + \
             1j*file['imag_traj'][-1,...].astype(DTYPE)

      dcf = np.array(goldcomp.cmp(traj),dtype=DTYPE)
    else:
      traj=None
      dcf = None

    if np.max(prime_factors(data.shape[-1]))>13:
      print(data.shape[-1])
      data = np.require(data[...,3:-3],requirements='C')
      dimX -=3
      dimY -=3
      traj = traj[...,3:-3]
      dcf = dcf[...,3:-3]

    #Create par struct to store everyting
    par={}


################################################################################
### FA correction ##############################################################
################################################################################
    [NC,reco_Slices,Nproj, N] = data.shape
    NScan = 1
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

    if not args.trafo:
      tmp = np.ones_like(np.abs(data))
      tmp[np.abs(data)==0] = 0
      par['mask'] = np.reshape(tmp,(NScan,NC,reco_Slices,dimY,dimX)).astype(DTYPE_real)
      del tmp



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
          nlinvNewtonSteps = 8
          nlinvRealConstr  = False

          traj_coil = np.reshape(traj,(NScan*Nproj,N))
          dcf_coil = (np.array(goldcomp.cmp(traj_coil),dtype=DTYPE))

          par["C"] = np.zeros((NC,reco_Slices,dimY,dimX), dtype=DTYPE)
          par["phase_map"] = np.zeros((reco_Slices,dimY,dimX), dtype=DTYPE)

          (ctx,queue,FFT) = NUFFT(N,1,1,1,traj_coil,np.sqrt(dcf_coil))

          result = []
          for i in range(0,(reco_Slices)):
            sys.stdout.write("Computing coil sensitivity map of slice %i \r" \
                           %(i))
            sys.stdout.flush()

            ##### RADIAL PART
            combinedData = np.transpose(data[:,:,i,:,:],(1,0,2,3))
            combinedData = np.require(np.reshape(combinedData,(1,NC,1,NScan*Nproj,N)),requirements='C')
            tmp_coilData = clarray.zeros(FFT.queue[0],(1,1,1,dimY,dimX),dtype=DTYPE)
            coilData = np.zeros((NC,dimY,dimX),dtype=DTYPE)
            for j in range(NC):
                tmp_combinedData = clarray.to_device(FFT.queue[0],combinedData[None,:,j,...])
                FFT.adj_NUFFT(tmp_coilData,tmp_combinedData)
                coilData[j,...] = np.squeeze(tmp_coilData.get())

            combinedData = np.require(np.fft.fft2(coilData,norm=None)/np.sqrt(dimX*dimY),dtype=DTYPE,requirements='C')

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
          sumSqrC = np.sqrt(np.sum((par["C"] * np.conj(par["C"])),0))
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
          tmp =  np.sum(data,0)
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
          sumSqrC = np.sqrt(np.sum((par["C"] * np.conj(par["C"])),0))
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
        nlinvNewtonSteps = 8
        nlinvRealConstr  = False

        traj_coil = np.reshape(traj,(NScan*Nproj,N))
        dcf_coil = np.repeat(dcf,NScan,0)

        par["C"] = np.zeros((NC,reco_Slices,dimY,dimX), dtype=DTYPE)
        par["phase_map"] = np.zeros((reco_Slices,dimY,dimX), dtype=DTYPE)

        (ctx,queue,FFT) = NUFFT(N,1,1,1,traj_coil,np.sqrt(dcf_coil))

        result = []
        for i in range(0,(reco_Slices)):
          sys.stdout.write("Computing coil sensitivity map of slice %i \r" \
                         %(i))
          sys.stdout.flush()

          combinedData = np.transpose(data[:,:,i,:,:],(1,0,2,3))
          combinedData = np.require(np.reshape(combinedData,(1,NC,1,NScan*Nproj,N)),requirements='C')
          tmp_coilData = clarray.zeros(FFT.queue[0],(1,1,1,dimY,dimX),dtype=DTYPE)
          coilData = np.zeros((NC,dimY,dimX),dtype=DTYPE)
          for j in range(NC):
              tmp_combinedData = clarray.to_device(FFT.queue[0],combinedData[None,:,j,...])
              FFT.adj_NUFFT(tmp_coilData,tmp_combinedData)
              coilData[j,...] = np.squeeze(tmp_coilData.get())
#
          combinedData = np.require(np.fft.fft2(coilData,norm=None)/np.sqrt(dimX*dimY),dtype=DTYPE,requirements='C')

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
        sumSqrC = np.sqrt(np.sum((par["C"] * np.conj(par["C"])),0))
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
        tmp =  np.sum(data,0)
        for i in range(0,(reco_Slices)):
          sys.stdout.write("Computing coil sensitivity map of slice %i \r" \
                         %(i))
          sys.stdout.flush()

          ##### RADIAL PART
          combinedData =  tmp[:,i,...]

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
        sumSqrC = np.sqrt(np.sum((par["C"] * np.conj(par["C"])),0)) #4, 9, 128, 128
        if NC == 1:
          par["C"] = sumSqrC
        else:
          par["C"] = par["C"] / np.tile(sumSqrC, (NC,1,1,1))
      file.create_dataset("Coils",par["C"].shape,dtype=par["C"].dtype,data=par["C"])
      file.flush()


################################################################################
### Standardize data norm ######################################################
################################################################################
    [NC,NSlice,Nproj, N] = data.shape
    NScan = 1
    if args.trafo:
      if file.attrs['data_normalized_with_dcf']:
          pass
      else:
          data = data*np.sqrt(dcf)
#### Close File after everything was read
    file.close()
    dscale = np.sqrt(NSlice)*DTYPE(np.sqrt(2*1e3))/(np.linalg.norm(data.flatten()))
    par["dscale"] = dscale
    data*=dscale

################################################################################
### generate nFFT  #############################################################
################################################################################
    if NC ==1:
      par['C'] = np.ones_like(par['C']).astype(DTYPE)
    else:
      par['C'] = par['C'].astype(DTYPE)

    if args.trafo:
      (ctx,queue,FFT) = NUFFT(N,NScan,1,1,traj,np.sqrt(dcf))

      def nFTH(x,fft,dcf,NScan,NC,NSlice,dimY,dimX):
        siz = np.shape(x)
        result = np.zeros((NC,NSlice,NScan,dimY,dimX),dtype=DTYPE)
        tmp_result = clarray.zeros(fft.queue[0],(NScan,1,1,dimY,dimX),dtype=DTYPE)
        for j in range(siz[0]):
          for k in range(siz[1]):
            inp = clarray.to_device(fft.queue[0],np.require(x[j,k,...][None,None,None,...],requirements='C'))
            fft.adj_NUFFT(tmp_result,inp)
            result[j,k,...] = np.squeeze(tmp_result.get())
        return np.squeeze(np.transpose(result,(2,0,1,3,4)))

      images= np.require(np.sum(nFTH(data,FFT,dcf,NScan,NC,\
                     NSlice,dimY,dimX)*(np.conj(par["C"])),axis = 0),requirements='C')
      del FFT,ctx,queue
      try:
        par["fa_corr"] = np.require((np.transpose(par["fa_corr"],(0,2,1))),requirements='C')
      except:
        pass
    else:
      (ctx,queue,FFT) = NUFFT(N,NScan,1,1,traj,dcf,trafo=0,mask=par['mask'])

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


    msv.imshow(np.abs(np.squeeze(images)))
    plt.pause(1e-3)
    input("Press Enter to continue...")
    plt.close('all')



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
      opt.data =  data[None,...]
      opt.guess = images
    else:
      opt = Model_Reco.Model_Reco(par,ctx,queue,None,None,args.trafo)
      opt.data =  data


################################################################################
#### IRGN - TV referenz ########################################################
################################################################################
    result_tv = []
################################################################################
### Init forward model and initial guess #######################################
#############################################################re#################
    opt.dz = 1
################################################################################
##IRGN Params
    try:
      opt.irgn_par = read_config(args.config,"3D_TV")
    except:
      print("Config file not readable or not found. Falling back to default.")
      gen_default_config()
      opt.irgn_par = read_config("default","3D_TV")
    opt.irgn_par["max_iters"]=int(5e2)
    opt.irgn_par["lambd"]=(2e0)
    opt.irgn_par["gamma"]=(1e-2)
    opt.irgn_par["tol"] = 1e-3
    opt.execute_3D()
    result_tv.append(opt.result)

    res_tv = opt.gn_res
    res_tv = np.array(res_tv)/(opt.irgn_par["lambd"]*NSlice)
    msv.imshow(np.abs(np.squeeze(images)))
    input("Done! Press Enter to continue...")
    plt.close('all')
    del opt




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T1 quantification from VFA data. By default runs 3D regularization for TGV and TV.')
    parser.add_argument('--slices',default=4, dest='slices', type=int, help='Number of reconstructed slices (default=40). Symmetrical around the center slice.')
    parser.add_argument('--trafo', default=1, dest='trafo', type=int, help='Choos between radial (1, default) and Cartesian (0) sampling. ')
    parser.add_argument('--data',default='',dest='file',help='Full path to input data. If not provided, a file dialog will open.')
    parser.add_argument('--config',default='test',dest='config',help='Name of config file to use (assumed to be in the same folder). If not specified, use default parameters.')
    args = parser.parse_args()

    main(args)