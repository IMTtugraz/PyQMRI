import numpy as np

import time
import os
import h5py
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier as nlinvns

import Model_Reco_OpenCL as Model_Reco

from pynfft.nfft import NFFT

import VFA_model as VFA_model
import goldcomp

import pyopencl as cl

DTYPE = np.complex64
np.seterr(divide='ignore', invalid='ignore')

import ipyparallel as ipp

c = ipp.Client()

################################################################################
### Select input file ##########################################################
################################################################################

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

name = file.split('/')[-1]
file = h5py.File(file)

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
reco_Slices = 1
dimX, dimY, NSlice = (file.attrs['image_dimensions']).astype(int)

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


[NScan,NC,NSlice,Nproj, N] = data.shape
################################################################################
### Set sequence related parameters ############################################
################################################################################

par.fa = file.attrs['flip_angle(s)']*np.pi/180


par.TR          = file.attrs['TR']
par.NC          = NC
par.dimY        = dimY
par.dimX        = dimX
par.NSlice      = NSlice
par.NScan       = NScan
par.N = N
par.Nproj = Nproj

#### TEST
par.unknowns_TGV = 2
par.unknowns_H1 = 0
par.unknowns = 2

################################################################################
### Estimate coil sensitivities ################################################
################################################################################

nlinvNewtonSteps = 6
nlinvRealConstr  = False

traj_coil = np.reshape(traj,(NScan*Nproj,N))
coil_plan = NFFT((dimY,dimX),NScan*Nproj*N)
coil_plan.x = np.transpose(np.array([np.imag(traj_coil.flatten()),\
                                     np.real(traj_coil.flatten())]))
coil_plan.precompute()

par.C = np.zeros((NC,NSlice,dimY,dimX), dtype=DTYPE)
par.phase_map = np.zeros((NSlice,dimY,dimX), dtype=DTYPE)
result = []
for i in range(0,(NSlice)):
  print('deriving M(TI(1)) and coil profiles')


  ##### RADIAL PART
  combinedData = np.transpose(data[:,:,i,:,:],(1,0,2,3))
  combinedData = np.reshape(combinedData,(NC,NScan*Nproj,N))
  coilData = np.zeros((NC,dimY,dimX),dtype=DTYPE)
  for j in range(NC):
      coil_plan.f = combinedData[j,:,:]*np.repeat(np.sqrt(dcf),NScan,axis=0)
      coilData[j,:,:] = coil_plan.adjoint()

  combinedData = np.fft.fft2(coilData,norm=None)/np.sqrt(dimX*dimY)

  dview = c[int(np.floor(i*len(c)/NSlice))]
  result.append(dview.apply_async(nlinvns.nlinvns, combinedData,
                                  nlinvNewtonSteps, True, nlinvRealConstr))

for i in range(NSlice):
  par.C[:,i,:,:] = result[i].get()[2:,-1,:,:]

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
################################################################################
### Standardize data norm ######################################################
################################################################################
if file.attrs['data_normalized_with_dcf']:
  pass
else:
  data = data*np.sqrt(dcf)
#### Close File after everything was read
file.close()

#data = data/(NC*NScan*Nproj*NSlice)
dscale = np.sqrt(NSlice)*np.sqrt(2*1e3)/(np.linalg.norm(data.flatten()))
par.dscale = dscale

################################################################################
### generate nFFT for radial cases #############################################
################################################################################

def nfft(NScan,NC,dimX,dimY,N,Nproj,traj):
  plan = []
  traj_x = np.imag(traj)
  traj_y = np.real(traj)
  for i in range(NScan):
      plan.append([])
      points = np.transpose(np.array([traj_x[i,:,:].flatten(),\
                                      traj_y[i,:,:].flatten()]))
      for j in range(NC):
          plan[i].append(NFFT([dimX,dimY],N*Nproj))
          plan[i][j].x = points
          plan[i][j].precompute()

  return plan

def nFT(x,plan,dcf,NScan,NC,NSlice,Nproj,N,dimX):
  siz = np.shape(x)
  result = np.zeros((NScan,NC,NSlice,Nproj*N),dtype=DTYPE)
  for i in range(siz[0]):
    for j in range(siz[1]):
      for k in range(siz[2]):
        plan[i][j].f_hat = x[i,j,k,:,:]/dimX
        result[i,j,k,:] = plan[i][j].trafo()*np.sqrt(dcf).flatten()

  return result


def nFTH(x,plan,dcf,NScan,NC,NSlice,dimY,dimX):
  siz = np.shape(x)
  result = np.zeros((NScan,NC,NSlice,dimY,dimX),dtype=DTYPE)
  for i in range(siz[0]):
    for j in range(siz[1]):
      for k in range(siz[2]):
        plan[i][j].f = x[i,j,k,:,:]*np.sqrt(dcf)
        result[i,j,k,:,:] = plan[i][j].adjoint()

  return result/dimX

#
#
#import pyopencl.array as clarray
#r_struct = radon_struct(queue, img_shape, 21,
#                            n_detectors=512)
#scale = radon_normest(queue, r_struct)
#test_adj(queue,r_struct)
#
#def nFT_2D(x):
#  result = np.zeros((NScan,NC,NSlice,Nproj,N),dtype=DTYPE)
#  for i in range(NScan):
#    for j in range(NC):
#      for k in range(NSlice):
#        tmp_img_real = clarray.to_device(queue,np.require(np.real(x[i,j,k,...]),np.float32,"F"))
#        tmp_img_imag = clarray.to_device(queue,np.require(np.iamg(x[i,j,k,...]),np.float32,"F"))
#        tmp_sino_real = clarray.zeros(queue,r_struct[2],np.float32,"F")
#        tmp_sino_imag = clarray.zeros(queue,r_struct[2],np.float32,"F")
#        (radon(tmp_sino_real,tmp_img_real,r_struct))
#        (radon(tmp_sino_imag,tmp_img_imag,r_struct)).wait()
#        result[i,j,k,...] = np.reshape(tmp_sino_real.get()+1j*tmp_sino_imag,(Nproj,N))
#
#  return result/scale
#
#
#
#def nFTH_2D(x):
#  result = np.zeros((NScan,NC,NSlice,dimY,dimX),dtype=np.float32)
#  for i in range(NScan):
#    for j in range(NC):
#      for k in range(NSlice):
#        tmp_sino_real = clarray.to_device(queue,np.require(np.real(x[i,j,k,...].T),np.float32,"F"))
#        tmp_sino_imag = clarray.to_device(queue,np.require(np.imag(x[i,j,k,...].T),np.float32,"F"))
#        tmp_img_real = clarray.zeros(queue,r_struct[1],np.float32,"F")
#        tmp_img_imag = clarray.zeros(queue,r_struct[1],np.float32,"F")
#        (radon_ad(tmp_img_real,tmp_sino_real,r_struct))
#        (radon_ad(tmp_img_imag,tmp_sino_imag,r_struct)).wait()
#        result[i,j,k,...] = np.reshape((tmp_img_real.get()+1j*tmp_img_imag),(dimY,dimX))
#
#  return result/(scale)


plan = nfft(NScan,NC,dimX,dimY,N,Nproj,traj)

data = data* dscale

data_save = data

images= (np.sum(nFTH(data_save,plan,dcf,NScan,NC,\
                     NSlice,dimY,dimX)*(np.conj(par.C)),axis = 1))


test = nFTH(data_save/np.sqrt(dcf),plan,dcf,NScan,NC,\
                     NSlice,dimY,dimX)

traj_save = np.copy(traj)


################################################################################
### Init forward model and initial guess #######################################
################################################################################
model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,par.phase_map,1,Nproj)

par.U = np.ones((data).shape, dtype=bool)
par.U[abs(data) == 0] = False
#
### 89 spk
#shift_read = np.array((0.0316,    0.0310 ,   0.0324  ,  0.0322 ,   0.0329  ,  0.0341 ,   0.0346   , 0.0334 ,   0.0323  ,  0.0280))
#shift_phase = np.array((0.0887,    0.0870  ,  0.0897 ,   0.0915  ,  0.0928  ,  0.0964  ,  0.0962 ,   0.0989 ,   0.1052 ,   0.1034))
#    ## 34 spk
##shift_read = np.array((0.0311,    0.0313,    0.0319,    0.0327,    0.0334,    0.0340,    0.0350,    0.0344,    0.0315,    0.0318))
##shift_phase = np.array((0.0902,    0.0882,    0.0898,    0.0911,    0.0929,    0.0947,    0.0971,    0.0997,    0.1028,    0.1037))
#
#angles = np.zeros((par.NScan,par.Nproj))
#sorted_angles = np.zeros((par.NScan,par.Nproj),dtype=np.int32)
#GA = 111.246117975/180*np.pi
#offset = par.Nproj*GA
#for n in range(par.NScan):
#  for ip in range(par.Nproj):
#      angles[n,ip] = np.mod(-np.pi/2 + np.mod(offset*n,2*np.pi) + (ip)*GA,2*np.pi)
#  sorted_angles[n,:] = np.argsort(angles[n,:])
#
#
#deltak = np.zeros((par.NScan, par.Nproj))
#samples_data = np.zeros((par.NScan, par.Nproj, N))
#for n in range(par.NScan):
#  for ip in range(par.Nproj):
#    deltak[n,ip] = ((np.cos(2*angles[n,ip]) + 1) * shift_read[n]+(-np.cos(2*angles[n,ip]) + 1) * shift_phase[n])/2
#    samples_data[n,ip,:] = np.linspace(-N/2,N/2-1,N)-deltak[n,ip]
#
#
#### -0.5....0.5 or -1...1 ????
#samples= np.linspace(-N/2,N/2-1,N)
#data = np.copy(data_save)/np.sqrt(dcf)
#for j in range(par.NScan):
#  for i in range(par.Nproj):
#    for k in range(NC):
#      for l in range(NSlice):
#        data[j,k,l,i,:] = np.interp(samples_data[j,i],samples,data[j,k,l,i,:])
#
#
#for j in range(par.NScan):
#  data[j,...] = np.transpose(data[j,:,:,sorted_angles[j,:],:],(1,2,0,3))
#  angles[j,:] = angles[j,sorted_angles[j,:]]
#  traj[j,...] = traj_save[j,sorted_angles[j,:],:]



data = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(data_save/np.sqrt(dcf),-1),axis=-1)/(512),-1)

#for j in range(par.NScan):
#  for i in range(par.Nproj):
#        data[j,:,:,i,:] = data[j,:,:,i,:]*np.exp(1j*deltak[j,i]*samples)


data = np.linalg.norm(data_save)/np.linalg.norm(data)*data



################################################################################
### IRGN - TGV Reco ############################################################
################################################################################
#
#from skimage.transform import iradon
#
#test2 = np.transpose(data[0,0,0,...].real,(1,0))
#test = iradon(test2,theta=angles[0,...]*180/np.pi,filter=None,output_size=256)
#test2 = np.transpose(data[0,0,0,...].imag,(1,0))
#test3 = iradon(test2,theta=angles[0,...]*180/np.pi,filter=None,output_size=256)
#
#numpyradon = test+1J*test3
#
#plt.imshow(np.abs(test))

platforms = cl.get_platforms()

ctx = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[1])])



queue = cl.CommandQueue(ctx)
opt = Model_Reco.Model_Reco(par,ctx,queue,traj,model,angles)


#import pyopencl.array as clarray

#xx = clarray.to_device(opt.queue,np.random.random_sample((opt.unknowns,opt.NSlice,opt.dimX,opt.dimY)).astype(DTYPE)+1j*np.random.random_sample((opt.unknowns,opt.NSlice,opt.dimX,opt.dimY)).astype(DTYPE))
#
#yy = np.random.random_sample((opt.unknowns,opt.NSlice,opt.dimX,opt.dimY,4)).astype(DTYPE)+1j*np.random.random_sample((opt.unknowns,opt.NSlice,opt.dimX,opt.dimY,4)).astype(DTYPE)
#yy = clarray.to_device(opt.queue,yy)
#
#
##yy = clarray.to_device(opt.queue,np.random.random_sample(opt.z1.shape).astype(DTYPE))
#
#tmp1 = clarray.zeros_like(xx)
#tmp2 = clarray.zeros_like(yy)
#opt.bdiv(tmp1,yy)
#opt.sym_grad(tmp2,xx)
#
#tmp = tmp2.get()[...,:3]*yy.get()[...,:3]+2*tmp2.get()[...,3:6]*yy.get()[...,3:6]
#
#a = np.vdot(xx.get().flatten(),-tmp1.get().flatten())
#b = np.vdot(tmp2.get().flatten(),yy.get().flatten())
#b=np.sum(tmp)
#test = np.abs(a-b)
#print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))

opt.data =  data
opt.images = images
opt.nfftplan = plan
opt.dcf = np.sqrt(dcf)
opt.dcf_flat = np.sqrt(dcf).flatten()

#IRGN Params
################################################################################
#IRGN Params
irgn_par = struct()
irgn_par.start_iters = 100
irgn_par.max_iters = 300
irgn_par.max_GN_it = 20
irgn_par.lambd = 1e2
irgn_par.gamma = 1e-1   #### 5e-2   5e-3 phantom ##### brain 1e-2
irgn_par.delta = 1e-1#### 8spk in-vivo 1e-2
irgn_par.omega = 0e-10
irgn_par.display_iterations = True
irgn_par.gamma_min = 3e-3
irgn_par.delta_max = 1e2
irgn_par.tol = 1e-5
irgn_par.stag = 1.00
irgn_par.delta_inc = 2
irgn_par.gamma_dec = 0.5
opt.irgn_par = irgn_par



#model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images2,par.phase_map,1)
#opt.model = model
import cProfile
import pstats
cProfile.run('opt.execute_3D()','profile_stats')
p = pstats.Stats('profile_stats')

#p.strip_dirs().sort_stats(-1)
#p.sort_stats('name')
#p.print_stats()

p.sort_stats('cumulative').print_stats(20)


#
#
#images2= (np.sum(opt.FTH(data[:,:,0,...])[:,:,None,...]*(np.conj(opt.par.C)),axis = 1))
#test2 = opt.FT(opt.FTH(data[:,:,0,...]))
#
#
#test = np.squeeze(np.stack((np.abs(images[2,:,...]),np.abs(images2[2,:,...])),axis=1))
#msv.imshow(np.abs(test))
#
#
#################################################################################
#### IRGN - Tikhonov referenz ###################################################
#################################################################################
#
#opt_t = Model_Reco_Tikh.Model_Reco(par)
#
#opt_t.par = par
#opt_t.data =  data_save
#opt_t.images = images
#opt_t.nfftplan = plan
#opt_t.dcf = np.sqrt(dcf)
#opt_t.dcf_flat = np.sqrt(dcf).flatten()
#opt_t.model = model
#opt_t.traj = traj
#
#################################################################################
###IRGN Params
#irgn_par = struct()
#irgn_par.start_iters = 10
#irgn_par.max_iters = 1000
#irgn_par.max_GN_it = 10
#irgn_par.lambd = 1e2
#irgn_par.gamma = 1e-2  #### 5e-2   5e-3 phantom ##### brain 1e-2
#irgn_par.delta = 1e-3  #### 8spk in-vivo 1e-2
#irgn_par.omega = 1e0
#irgn_par.display_iterations = True
#
#opt_t.irgn_par = irgn_par
#
#opt_t.execute_2D()
#
#################################################################################
#### New .hdf5 save files #######################################################
#################################################################################
#outdir = time.strftime("%Y-%m-%d  %H-%M-%S_2D_"+name[:-3])
#if not os.path.exists('./output'):
#    os.makedirs('./output')
#os.makedirs("output/"+ outdir)
#
#os.chdir("output/"+ outdir)
#
#f = h5py.File("output_"+name,"w")
#dset_result=f.create_dataset("full_result",opt.result.shape,\
#                             dtype=np.complex64,data=opt.result)
#dset_result_ref=f.create_dataset("ref_full_result",opt_t.result.shape,\
#                                 dtype=np.complex64,data=opt_t.result)
#dset_T1=f.create_dataset("T1_final",np.squeeze(opt.result[-1,1,...]).shape,\
#                         dtype=np.complex64,\
#                         data=np.squeeze(opt.result[-1,1,...]))
#dset_M0=f.create_dataset("M0_final",np.squeeze(opt.result[-1,0,...]).shape,\
#                         dtype=np.complex64,\
#                         data=np.squeeze(opt.result[-1,0,...]))
#dset_T1_ref=f.create_dataset("T1_ref",np.squeeze(opt_t.result[-1,1,...]).shape\
#                             ,dtype=np.complex64,\
#                             data=np.squeeze(opt_t.result[-1,1,...]))
#dset_M0_ref=f.create_dataset("M0_ref",np.squeeze(opt_t.result[-1,0,...]).shape\
#                             ,dtype=np.complex64,\
#                             data=np.squeeze(opt_t.result[-1,0,...]))
##f.create_dataset("T1_guess",np.squeeze(model.T1_guess).shape,\
##                 dtype=np.float64,data=np.squeeze(model.T1_guess))
##f.create_dataset("M0_guess",np.squeeze(model.M0_guess).shape,\
##                 dtype=np.float64,data=np.squeeze(model.M0_guess))
#dset_result.attrs['data_norm'] = dscale
#dset_result.attrs['dcf_scaling'] = (N*(np.pi/(4*Nproj)))
#f.flush()
#f.close()
#
#os.chdir('..')
#os.chdir('..')
#
#
