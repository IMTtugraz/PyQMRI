import numpy as np

import time
import os
import sys
import h5py
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier as nlinvns

import Model_Reco_OpenCL as Model_Reco

#from pynfft.nfft import NFFT

import VFA_model as VFA_model
import goldcomp

import pyopencl as cl

DTYPE = np.complex64
np.seterr(divide='ignore', invalid='ignore')

import ipyparallel as ipp

c = ipp.Client()

import pyopencl.array as clarray
from gridroutines import gridding

DTYPE = np.complex64
DTYPE_real = np.float32

def NUFFT(N,NScan,NC,NSlice,traj,dcf,trafo=1):
  platforms = cl.get_platforms()
  ctx = cl.Context(
            dev_type=cl.device_type.GPU,
            properties=[(cl.context_properties.PLATFORM, platforms[1])])
  queue=[]
  queue.append(cl.CommandQueue(ctx,platforms[1].get_devices()[0]))
  if trafo:
    FFT = gridding(ctx,queue,4,2,N,NScan,(NScan*NC*NSlice,N,N),(1,2),traj.astype(DTYPE),np.require(np.abs(dcf),DTYPE_real,requirements='C'),N,1000,DTYPE,DTYPE_real,radial=trafo)
  else:
    FFT = gridding(ctx,queue,4,2,N,NScan,(NScan*NC*NSlice,N,N),(1,2),traj,dcf,N,1000,DTYPE,DTYPE_real,radial=trafo)
  return (ctx,queue[0],FFT)

################################################################################
### Select input file ##########################################################
################################################################################

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

name = file.split('/')[-1][:-3] + "_5_angle.h5"
file = h5py.File(file)
print("Starting computation for "+name)
################################################################################
### Check if file contains all necessary information ###########################
################################################################################
test_data = ['dcf', 'fa_corr','imag_dat', 'imag_traj', 'real_dat', 'real_traj']
test_attributes = ['image_dimensions','flip_angle(s)','TR',\
                   'data_normalized_with_dcf']



for datasets in test_data:
  if not (datasets in list(file.keys())):
    file.close()
    raise NameError("Error: '" + datasets + \
                    "' data was not provided/wrongly named!")
for attributes in test_attributes:
  if not (attributes in list(file.attrs)):
    file.close()
    raise NameError("Error: '" + attributes + \
                    "' was not provided/wrongly as an attribute!")

################################################################################
### Read Data ##################################################################
################################################################################
dimX, dimY, NSlice = (file.attrs['image_dimensions']).astype(int)
reco_Slices = 2

data = file['real_dat'][:,:,int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...].astype(DTYPE) +\
       1j*file['imag_dat'][:,:,int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...].astype(DTYPE)


traj = file['real_traj'][()].astype(DTYPE) + \
       1j*file['imag_traj'][()].astype(DTYPE)


dcf = np.array(goldcomp.cmp(traj),dtype=DTYPE)

#Create par struct to store everyting
class struct:
    pass
par = struct()

################################################################################
### FA correction ##############################################################
################################################################################

par.fa_corr = np.flip(file['fa_corr'][()].astype(DTYPE),0)[int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...]
par.fa_corr[par.fa_corr==0] = 1

#par.fa_corr =file['interpol_fa'][()].astype(DTYPE)[int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...]
#par.fa_corr[par.fa_corr==0] = 1


[NScan,NC,reco_Slices,Nproj, N] = data.shape
################################################################################
### Set sequence related parameters ############################################
################################################################################

par.fa = file.attrs['flip_angle(s)']*np.pi/180


par.TR          = file.attrs['TR']
par.NC          = NC
par.dimY        = dimY
par.dimX        = dimX
par.NSlice      = reco_Slices
par.NScan       = NScan
par.N = N
par.Nproj = Nproj

#### TEST
par.unknowns_TGV = 2
par.unknowns_H1 = 0
par.unknowns = 2



if file.attrs['data_normalized_with_dcf']:
  pass
else:
  data = data*np.sqrt(dcf)

################################################################################
### Estimate coil sensitivities ################################################
################################################################################
class B(Exception):
  pass
try:
  if not file['Coils'][()].shape[1] >= reco_Slices:
    nlinvNewtonSteps = 6
    nlinvRealConstr  = False

    traj_coil = np.reshape(traj,(NScan*Nproj,N))
#    coil_plan = NFFT((dimY,dimX),NScan*Nproj*N)
#    coil_plan.x = np.transpose(np.array([np.imag(traj_coil.flatten()),\
#                                         np.real(traj_coil.flatten())]))
#    coil_plan.precompute()
    dcf_coil = np.array(goldcomp.cmp(traj_coil),dtype=DTYPE)

    par.C = np.zeros((NC,reco_Slices,dimY,dimX), dtype=DTYPE)
    par.phase_map = np.zeros((reco_Slices,dimY,dimX), dtype=DTYPE)

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
#          coil_plan.f = combinedData[j,:,:]*np.repeat(np.sqrt(dcf),NScan,axis=0)
#          coilData[j,:,:] = coil_plan.adjoint()
          FFT.adj_NUFFT(tmp_coilData,tmp_combinedData)
          coilData[j,...] = tmp_coilData.get()

      combinedData = np.fft.fft2(coilData,norm=None)/np.sqrt(dimX*dimY)

      dview = c[int(np.floor(i*len(c)/NSlice))]
      result.append(dview.apply_async(nlinvns.nlinvns, combinedData,
                                      nlinvNewtonSteps, True, nlinvRealConstr))

    for i in range(reco_Slices):
      par.C[:,i,:,:] = result[i].get()[2:,-1,:,:]
      sys.stdout.write("slice %i done \r" \
                     %(i))
      sys.stdout.flush()
      if not nlinvRealConstr:
        par.phase_map[i,:,:] = np.exp(1j * np.angle( result[i].get()[0,-1,:,:]))
        par.C[:,i,:,:] = par.C[:,i,:,:]* np.exp(1j *\
             np.angle( result[i].get()[1,-1,:,:]))

        # standardize coil sensitivity profiles
    sumSqrC = np.sqrt(np.sum((par.C * np.conj(par.C)),0)) #4, 9, 128, 128
    if NC == 1:
      par.C = sumSqrC
    else:
      par.C = par.C / np.tile(sumSqrC, (NC,1,1,1))
    del file['Coils']
    file.create_dataset("Coils",par.C.shape,dtype=par.C.dtype,data=par.C)
    file.flush()
    del FFT,ctx,queue
  else:
    print("Using precomputed coil sensitivities")
    slices_coils = file['Coils'][()].shape[1]
    par.C = file['Coils'][:,int(slices_coils/2)-int(np.floor((reco_Slices)/2)):int(slices_coils/2)+int(np.ceil(reco_Slices/2)),...]

except:
  nlinvNewtonSteps = 6
  nlinvRealConstr  = False

  traj_coil = np.reshape(traj,(NScan*Nproj,N))
#    coil_plan = NFFT((dimY,dimX),NScan*Nproj*N)
#    coil_plan.x = np.transpose(np.array([np.imag(traj_coil.flatten()),\
#                                         np.real(traj_coil.flatten())]))
#    coil_plan.precompute()
  dcf_coil = np.array(goldcomp.cmp(traj_coil),dtype=DTYPE)

  par.C = np.zeros((NC,reco_Slices,dimY,dimX), dtype=DTYPE)
  par.phase_map = np.zeros((reco_Slices,dimY,dimX), dtype=DTYPE)

  (ctx,queue,FFT) = NUFFT(N,1,1,1,traj_coil,np.ones_like(dcf_coil))

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
#          coil_plan.f = combinedData[j,:,:]*np.repeat(np.sqrt(dcf),NScan,axis=0)
#          coilData[j,:,:] = coil_plan.adjoint()
        FFT.adj_NUFFT(tmp_coilData,tmp_combinedData)
        coilData[j,...] = tmp_coilData.get()

    combinedData = np.fft.fft2(coilData,norm=None)/np.sqrt(dimX*dimY)

    dview = c[int(np.floor(i*len(c)/NSlice))]
    result.append(dview.apply_async(nlinvns.nlinvns, combinedData,
                                    nlinvNewtonSteps, True, nlinvRealConstr))

  for i in range(reco_Slices):
    par.C[:,i,:,:] = result[i].get()[2:,-1,:,:]
    sys.stdout.write("slice %i done \r" \
                   %(i))
    sys.stdout.flush()
    if not nlinvRealConstr:
      par.phase_map[i,:,:] = np.exp(1j * np.angle( result[i].get()[0,-1,:,:]))
      par.C[:,i,:,:] = par.C[:,i,:,:]* np.exp(1j *\
           np.angle( result[i].get()[1,-1,:,:]))

      # standardize coil sensitivity profiles
  sumSqrC = np.sqrt(np.sum((par.C * np.conj(par.C)),0)) #4, 9, 128, 128
  if NC == 1:
    par.C = sumSqrC
  else:
    par.C = par.C / np.tile(sumSqrC, (NC,1,1,1))
  file.create_dataset("Coils",par.C.shape,dtype=par.C.dtype,data=par.C)
  file.flush()
  del FFT,ctx,queue
#### Close File after everything was read
file.close()
[NScan,NC,NSlice,Nproj, N] = data.shape
################################################################################
### Standardize data norm ######################################################
################################################################################

#data = data*(10/NScan)
#data = data/(NC*NScan*Nproj*NSlice)
dscale = np.sqrt(NSlice)*np.sqrt(2*1e3)/(np.linalg.norm(data.flatten()))
par.dscale = dscale

################################################################################
### generate nFFT for radial cases #############################################
################################################################################

#def nfft(NScan,NC,dimX,dimY,N,Nproj,traj):
#  plan = []
#  traj_x = np.imag(traj)
#  traj_y = np.real(traj)
#  for i in range(NScan):
#      plan.append([])
#      points = np.transpose(np.array([traj_x[i,:,:].flatten(),\
#                                      traj_y[i,:,:].flatten()]))
#      for j in range(NC):
#          plan[i].append(NFFT([dimX,dimY],N*Nproj))
#          plan[i][j].x = points
#          plan[i][j].precompute()

#  return plan

(ctx,queue,FFT) = NUFFT(N,NScan,1,1,traj,dcf)

#def nFT(x,plan,dcf,NScan,NC,NSlice,Nproj,N,dimX):
#  siz = np.shape(x)
#  result = np.zeros((NScan,NC,NSlice,Nproj*N),dtype=DTYPE)
#  for i in range(siz[0]):
#    for j in range(siz[1]):
#      for k in range(siz[2]):
#        plan[i][j].f_hat = x[i,j,k,:,:]/dimX
#        result[i,j,k,:] = plan[i][j].trafo()*np.sqrt(dcf).flatten()
#
#  return result


def nFTH(x,fft,dcf,NScan,NC,NSlice,dimY,dimX):
  siz = np.shape(x)
  result = np.zeros((NC,NSlice,NScan,dimY,dimX),dtype=DTYPE)
  tmp_result = clarray.zeros(fft.queue[0],(NScan,1,1,dimY,dimX),dtype=DTYPE)
#  for i in range(siz[0]):
  for j in range(siz[1]):
    for k in range(siz[2]):
#        plan[i][j].f = x[i,j,k,:,:]*np.sqrt(dcf)
#        result[i,j,k,:,:] = plan[i][j].adjoint()
      inp = clarray.to_device(fft.queue[0],np.require(x[:,j,k,...][:,None,None,...],requirements='C'))
      fft.adj_NUFFT(tmp_result,inp)
#      print(tmp_result.get().shape)
      result[j,k,...] = np.squeeze(tmp_result.get())


  return np.transpose(result,(2,0,1,3,4))


#plan = nfft(NScan,NC,dimX,dimY,N,Nproj,traj)

data = data* dscale

data_save = data

test = nFTH(data_save,FFT,dcf,NScan,NC,\
                     NSlice,dimY,dimX)


images= np.require(np.sum(nFTH(data_save,FFT,dcf,NScan,NC,\
                     NSlice,dimY,dimX)*(np.conj(par.C)),axis = 1),requirements='C')
del FFT,ctx,queue
traj_save = np.copy(traj)

#par.C = np.require(np.transpose(par.C,(0,1,3,2)),requirements='C')
par.fa_corr = np.require((np.transpose(par.fa_corr,(0,2,1))),requirements='C')
################################################################################
### Init forward model and initial guess #######################################
################################################################################
model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,NSlice,Nproj)


################################################################################
### IRGN - TGV Reco ############################################################
################################################################################

#platforms = cl.get_platforms()
#
#ctx = []
#queue = []
#num_dev = len(platforms[0].get_devices())
#for device in platforms[0].get_devices():# range(num_dev):
#  tmp = cl.Context(
#          dev_type=cl.device_type.GPU,
#          properties=[(cl.context_properties.PLATFORM, platforms[0])])
#  ctx.append(tmp)
#  queue.append(cl.CommandQueue(tmp,device,properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE))#))#,  platforms[0].get_devices()[device]))
#  queue.append(cl.CommandQueue(tmp, device,properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE))#))#, platforms[0].get_devices()[device]))#properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE))
#  queue.append(cl.CommandQueue(tmp, device,properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE))

#import Model_Reco_OpenCL_streamed_slicesfirst as Model_Reco

platforms = cl.get_platforms()

ctx = []
queue = []
num_dev = 1#len(platforms[0].get_devices())
for device in range(num_dev):
  tmp = cl.Context(
          dev_type=cl.device_type.GPU,
          properties=[(cl.context_properties.PLATFORM, platforms[1])])
  ctx.append(tmp)
  queue.append(cl.CommandQueue(tmp,platforms[1].get_devices()[0]))#))#,  platforms[0].get_devices()[device]))
  queue.append(cl.CommandQueue(tmp, platforms[1].get_devices()[0]))#))#, platforms[0].get_devices()[device]))#properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE))
  queue.append(cl.CommandQueue(tmp, platforms[1].get_devices()[0]))



opt = Model_Reco.Model_Reco(par,ctx,queue,traj,np.sqrt(dcf))
#import Model_Reco_OpenCL as Model_Reco2
#opt2 = Model_Reco2.Model_Reco(par,ctx,queue,traj,np.sqrt(dcf))
#
#
#import pyopencl.array as clarray
#xx = clarray.to_device(queue[0],np.random.randn(4,2,256,256)+1j*np.random.randn(4,2,256,256)).astype(DTYPE)
#yy =clarray.to_device(queue[0],np.random.randn(4,2,256,256,4)+1j*np.random.randn(4,2,256,256,4)).astype(DTYPE)
#test1 = clarray.zeros_like(yy)
#test2 =  clarray.zeros_like(xx)
##
##
###opt.NUFFT[0].fwd_NUFFT(test1,xx)
###opt.NUFFT[0].adj_NUFFT(test2,yy)
#opt.f_grad(test1,xx)
#opt.bdiv(test2,yy)
#
##
##opt.f_grad(test1,xx)
##opt.bdiv(test2,yy)
##
##
#test1 = test1.get()
#test2 = test2.get()
#yy = yy.get()
#xx = xx.get()
##
#
##a = np.sum(np.conj((test1[...,0:3]))*yy[...,0:3]+2*np.conj(test1[...,3:6])*yy[...,3:6])
#a = np.vdot(test1,yy)
#b = np.vdot(-xx[...],test2[...])
#
#
##
#print(np.abs(a-b)/np.size(xx))

#data =np.fft.fftshift( np.fft.fft(np.fft.fftshift(data,2),axis=2,norm='ortho'),2)

opt.data =  data
opt.model = model
opt.images = images
#opt.nfftplan = plan
opt.dcf = np.sqrt(dcf)
opt.dcf_flat = np.sqrt(dcf).flatten()
result_tgv = []
#IRGN Params
################################################################################
#IRGN Params
irgn_par = {}
irgn_par["max_iters"] = 300
irgn_par["start_iters"] = 100
irgn_par["max_GN_it"] = 12
irgn_par["lambd"] = 1e2
irgn_par["gamma"] = 1e0
irgn_par["delta"] = 1e-1
irgn_par["display_iterations"] = True
irgn_par["gamma_min"] = 0.24
irgn_par["delta_max"] = 1e-1*2**7
irgn_par["tol"] = 5e-3
irgn_par["stag"] = 1e1
irgn_par["delta_inc"] = 2
irgn_par["gamma_dec"] = 0.7
opt.irgn_par = irgn_par


opt.execute_3D()






#import cProfile
#import pstats
#cProfile.run('opt.execute_3D()','profile_stats')
#p = pstats.Stats('profile_stats')
#p.sort_stats('cumulative').print_stats(20)

result_tgv.append(opt.result)
res = opt.gn_res
res = np.array(res)/(irgn_par["lambd"]*NSlice)


################################################################################
### New .hdf5 save files #######################################################
################################################################################
outdir = time.strftime("%Y-%m-%d  %H-%M-%S_MRI_joint_"+name[:-3])
if not os.path.exists('./output'):
    os.makedirs('./output')
os.makedirs("output/"+ outdir)

os.chdir("output/"+ outdir)
f = h5py.File("output_"+name,"w")

for i in range(len(result_tgv)):
  dset_result=f.create_dataset("tgv_full_result_"+str(i),result_tgv[i].shape,\
                               dtype=DTYPE,data=result_tgv[i])
#  dset_result_ref=f.create_dataset("tv_full_result_"+str(i),result_tv[i].shape,\
#                                   dtype=DTYPE,data=result_tv[i])
#  dset_result_ref=f.create_dataset("wt_full_result_"+str(i),result_wt[i].shape,\
#                                   dtype=DTYPE,data=result_wt[i])
#  dset_T1=f.create_dataset("T1_final_"+str(i),np.squeeze(result_tgv[i][-1,1,...]).shape,\
#                           dtype=DTYPE,\
#                           data=np.squeeze(result_tgv[i][-1,1,...]))
#  dset_M0=f.create_dataset("M0_final_"+str(i),np.squeeze(result_tgv[i][-1,0,...]).shape,\
#                           dtype=DTYPE,\
#                           data=np.squeeze(result_tgv[i][-1,0,...]))
#  dset_T1_ref=f.create_dataset("T1_ref_"+str(i),np.squeeze(result_ref[i][-1,1,...]).shape\
#                               ,dtype=DTYPE,\
#                               data=np.squeeze(result_ref[i][-1,1,...]))
#  dset_M0_ref=f.create_dataset("M0_ref_"+str(i),np.squeeze(result_ref[i][-1,0,...]).shape\
#                               ,dtype=DTYPE,\
#                               data=np.squeeze(result_ref[i][-1,0,...]))
  #f.create_dataset("T1_guess",np.squeeze(model.T1_guess).shape,\
  #                 dtype=np.float64,data=np.squeeze(model.T1_guess))
  #f.create_dataset("M0_guess",np.squeeze(model.M0_guess).shape,\
  #                 dtype=np.float64,data=np.squeeze(model.M0_guess))
  f.attrs['data_norm'] = dscale
  f.attrs['dcf_scaling'] = (N*(np.pi/(4*Nproj)))
  f.attrs['IRGN_TGV_res'] = res
  f.attrs['dscale'] = dscale
  f.flush()
f.close()

os.chdir('..')
#os.chdir('..')