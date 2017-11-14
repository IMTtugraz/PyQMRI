import pyfftw
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier as nlinvns

import pyximport; pyximport.install()


import Model_Reco as Model_Reco
import Model_Reco_old as Model_Reco_Tikh

import scipy.io as sio

import multiprocessing as mp
import mkl

from pynfft.nfft import NFFT

import VFA_model as VFA_model

import h5py  

DTYPE = np.complex64
np.seterr(divide='ignore', invalid='ignore')# TODO:
  
mkl.set_num_threads(mp.cpu_count())  
os.system("taskset -p 0xff %d" % os.getpid()) 
  
  
  
plt.ion()
pyfftw.interfaces.cache.enable()

################################################################################
### Read input data ############################################################
################################################################################

#root = Tk()
#root.withdraw()
#root.update()
#file = filedialog.askopenfilename()
#root.destroy()
#
#data = sio.loadmat(file)
#data = data['data_mid'].astype(DTYPE)
##data = data['data']
#
#data = np.transpose(data)

##### Read H5
root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

name = file.split('/')[-1]

file = h5py.File(file)
data = file['real_dat'][()].astype(DTYPE) + 1j*file['imag_dat'][()].astype(DTYPE)

#root = Tk()
#root.withdraw()
#root.update()
#file = filedialog.askopenfilename()
#root.destroy()

#file = h5py.File(file)

traj = file['real_traj'][()].astype(DTYPE) + 1j*file['imag_traj'][()].astype(DTYPE)



dcf = file['dcf'][()].astype(DTYPE)
#dcf = dcf[0,:,:]
#root = Tk()
#root.withdraw()
#root.update()
#file = filedialog.askopenfilename()
#root.destroy()
#
#traj = sio.loadmat(file)
#traj = file['traj'].astype(DTYPE)
#
#traj = np.transpose(traj)
#
#root = Tk()
#root.withdraw()
#root.update()
#file = filedialog.askopenfilename()
#root.destroy()
#
#dcf = sio.loadmat(file)
#dcf = dcf['dcf'].astype(DTYPE)
#
#dcf = np.transpose(dcf)
#dcf = dcf/np.max(dcf)

#data = np.fft.fft(data,axis=2).astype(DTYPE)
data = data[:,:,32,:,:]
data = data[:,:,None,:,:]
dimX = 256#192
dimY = 256#192


#NSlice = 1
[NScan,NC,NSlice,Nproj, N] = data.shape
#[NScan,NC,NSlice,dimY,dimX] = data.shape


#Create par struct to store everyting
class struct:
    pass
par = struct()


################################################################################
### FA correction ##############################################################
################################################################################

par.fa_corr = file['fa_corr'][()].astype(DTYPE)#np.ones([NSlice,dimX,dimY],dtype=DTYPE)
par.fa_corr = par.fa_corr[32,:,:]
par.fa_corr[par.fa_corr==0] = 1
#
#root = Tk()
#root.withdraw()
#root.update()
#file = filedialog.askopenfilename()
#root.destroy()
#
#fa_corr = sio.loadmat(file)
#fa_corr = fa_corr['fa_mid_3mm']
#
#fa_corr = np.transpose(fa_corr)
#fa_corr[[fa_corr==0]] = 1*np.pi/180
#par.fa_corr = fa_corr*180/np.pi#[16,:,:]#

################################################################################
### Estimate coil sensitivities ################################################
################################################################################

nlinvNewtonSteps = 6
nlinvRealConstr  = False

traj_coil = np.reshape(traj,(NScan*Nproj,N))
coil_plan = NFFT((dimY,dimX),NScan*Nproj*N)
coil_plan.x = np.transpose(np.array([np.imag(traj_coil.flatten()),np.real(traj_coil.flatten())]))
coil_plan.precompute()
        
par.C = np.zeros((NC,NSlice,dimY,dimX), dtype=DTYPE)       
par.phase_map = np.zeros((NSlice,dimY,dimX), dtype=DTYPE)   
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
   
  ### CARTESIAN PART    
#  combinedData = np.squeeze(np.sum(data[:,:,i,:,:],0))/NSlice

  
  
  
  

  """shape combinData(128, 128, 4)"""
            
  nlinvout = nlinvns.nlinvns(combinedData, nlinvNewtonSteps,
                     True, nlinvRealConstr)

  # coil sensitivities are stored in par.C
  par.C[:,i,:,:] = nlinvout[2:,-1,:,:]

  if not nlinvRealConstr:
    par.phase_map[i,:,:] = np.exp(1j * np.angle(nlinvout[0,-1,:,:]))
    par.C[:,i,:,:] = par.C[:,i,:,:]* np.exp(1j * np.angle(nlinvout[1,-1,:,:]))
    
    # standardize coil sensitivity profiles
sumSqrC = np.sqrt(np.sum((par.C * np.conj(par.C)),0)) #4, 9, 128, 128
if NC == 1:
  par.C = sumSqrC 
else:
  par.C = par.C / np.tile(sumSqrC, (NC,1,1,1)) 
  
  
data = data*np.sqrt(dcf) ## only in-vivo

################################################################################ 
### Artificial Subsampling #####################################################
################################################################################
#data_full = np.copy(data)
#traj_full = np.copy(traj)
#dcf_full = np.copy(dcf)
#
#
## Choose undersampling mode
#Nproj = 5
#
#for i in range(NScan):
#  data[i,:,:,:Nproj,:] = data[i,:,:,i*Nproj:(i+1)*Nproj,:]
#  traj[i,:Nproj,:] = traj[i,i*Nproj:(i+1)*Nproj,:]
#
#
#data = data[:,:,:,:Nproj,:]
#traj = traj[:,:Nproj,:]
#dcf = dcf[:Nproj,:]




#print("**undersampling")
#  
#undersampling_mode = 1
#
#def one():
#    # Fully Sampled
#    global uData
#    par.AF = 1
#    par.ACL = 32
#    uData = data
#
#def two():
#    # radial Pattern
#    global uData
#    AF = 6
#    par.AF = AF
#    ACL = 32
#    par.ACL = ACL
#    uData = np.zeros(data.shape)
#    uData      = optimizedPattern(data,AF,ACL); #data?
#    
#def three():
#    # Random Pattern %% Vorerst nicht portieren
#    uData = np.zeros_like(data)
#    uData[:,:,:,:,list(range(0,dimY,3))] = data[:,:,:,:,list(range(0,dimY,3))]
#    print(" Random Pattern")
#    
#options = {1 : one,
#           2 : two,
#           3 : three,}
#
#options[undersampling_mode]()


################################################################################
### Set sequence related parameters ############################################
################################################################################

#FA = np.array([1,2,4,5,7,9,11,14,17,23],np.complex128)*np.pi/180
#par.fa = np.array([1,2,3,4,5,6,7,9,11,13],DTYPE)*np.pi/180
par.fa = np.array([1,3,5,7,9,11,13,15,17,19],DTYPE)*np.pi/180
#par.fa = np.array([1,2,4,5,7,9,11,14,17,20],np.complex128)*np.pi/180

par.TR          = 5.0#3.4 #TODO
#par.TE          = list(range(20,40*20+1,20)) #TODO
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
### Standardize data norm ######################################################
################################################################################


dscale = np.sqrt(NSlice)*np.complex128(1)/(np.linalg.norm(data.flatten()))
par.dscale = dscale

################################################################################
### generate FFTW for cartesian cases ##########################################
################################################################################

data = pyfftw.byte_align(data)
#
#fftw_ksp = pyfftw.empty_aligned((dimX,dimY),dtype=DTYPE)
#fftw_img = pyfftw.empty_aligned((dimX,dimY),dtype=DTYPE)
#
#fft_forward = pyfftw.FFTW(fftw_img,fftw_ksp,axes=(0,1))
#fft_back = pyfftw.FFTW(fftw_ksp,fftw_img,axes=(0,1),direction='FFTW_BACKWARD')
#
#
#
#def FT(x):
#  siz = np.shape(x)
#  result = np.zeros_like(x,dtype=DTYPE)
#  for i in range(siz[0]):
#    for j in range(siz[1]):
#      for k in range(siz[2]):
#        result[i,j,k,:,:] = fft_forward(x[i,j,k,:,:])/np.sqrt(siz[4]*(siz[3]))
#      
#  return result
#
#
#def FTH(x):
#  siz = np.shape(x)
#  result = np.zeros_like(x,dtype=DTYPE)
#  for i in range(siz[0]):
#    for j in range(siz[1]):
#      for k in range(siz[2]):
#        result[i,j,k,:,:] = fft_back(x[i,j,k,:,:])*np.sqrt(siz[4]*(siz[3]))
#      
#  return result

################################################################################
### generate nFFT for radial cases #############################################
################################################################################

def nfft(NScan,NC,dimX,dimY,N,Nproj,traj):
  plan = []
  traj_x = np.imag(traj)
  traj_y = np.real(traj)  
  for i in range(NScan):
      plan.append([])
      points = np.transpose(np.array([traj_x[i,:,:].flatten(),traj_y[i,:,:].flatten()]))      
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


plan = nfft(NScan,NC,dimX,dimY,N,Nproj,traj)

data = data* dscale

images= (np.sum(nFTH(data,plan,dcf*N*(np.pi/(4*Nproj)),NScan,NC,NSlice,dimY,dimX)*(np.conj(par.C)),axis = 1))


################################################################################
### Init forward model and initial guess #######################################
################################################################################
model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,par.phase_map,1)



par.U = np.ones((data).shape, dtype=bool)
par.U[abs(data) == 0] = False
################################################################################
### IRGN - TGV Reco ############################################################
################################################################################

opt = Model_Reco.Model_Reco(par)

opt.par = par
opt.data =  data
opt.images = images
#opt.fft_forward = fft_forward
#opt.fft_back = fft_back
opt.nfftplan = plan
opt.dcf = np.sqrt(dcf*(N*(np.pi/(4*Nproj))))
opt.dcf_flat = np.sqrt(dcf*(N*(np.pi/(4*Nproj)))).flatten()
opt.model = model
opt.traj = traj 

################################################################################
#IRGN Params
irgn_par = struct()
irgn_par.start_iters = 10
irgn_par.max_iters = 1000
irgn_par.max_GN_it = 8
irgn_par.lambd = 1e2
irgn_par.gamma = 5e-2   #### 5e-2   5e-3 phantom ##### brain 1e-2
irgn_par.delta = 1e0  #### 8spk in-vivo 1e-2
irgn_par.omega = 1e-8
irgn_par.display_iterations = True

opt.irgn_par = irgn_par

opt.execute_2D()



################################################################################
### IRGN - Tikhonov referenz ###################################################
################################################################################
#
#opt_t = Model_Reco_Tikh.Model_Reco(par)
#
#opt_t.par = par
#opt_t.data =  data
#opt_t.images = images
##opt_t.fft_forward = fft_forward
##opt_t.fft_back = fft_back
#opt_t.nfftplan = plan
#opt_t.dcf = np.sqrt(dcf*(N*(np.pi/(4*Nproj))))
#opt_t.dcf_flat = np.sqrt(dcf*(N*(np.pi/(4*Nproj)))).flatten()
#opt_t.model = model
#opt_t.traj = traj 
#
##################################################################################
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

################################################################################
### Profiling ##################################################################
################################################################################

#import cProfile
#cProfile.run("opt.execute_2D()","eval_speed_3")
##
#import pstats
#p=pstats.Stats("eval_speed_3")
#p.sort_stats('time').print_stats(20)
#
#p=pstats.Stats("eval_speed_2")
#p.sort_stats('time').print_stats(20)
#
#p=pstats.Stats("eval_speed")
#p.sort_stats('time').print_stats(20)
#




################################################################################
### Old .mat save files ########################################################
################################################################################

#outdir = time.strftime("%Y-%m-%d  %H-%M-%S")
#if not os.path.exists('./output'):
#    os.makedirs('./output')
#os.makedirs("output/"+ outdir)
#
#os.chdir("output/"+ outdir)
#
#sio.savemat("resultinvivo_21spk_pro.mat",{"result":opt.result,"model":model})
#sio.savemat("resultinvivo_21spk_pro_tikh.mat",{"result":opt_t.result,"model":model})
#import pickle
#with open("par" + ".p", "wb") as pickle_file:
#    pickle.dump(par, pickle_file)
#
#os.chdir('..')
#os.chdir('..')
#with open("par.txt", "rb") as myFile:
#    par = pickle.load(myFile)
#par.dump("par.dat")


################################################################################
### New .hdf5 save files #######################################################
################################################################################
outdir = time.strftime("%Y-%m-%d  %H-%M-%S_"+name[:-3])
if not os.path.exists('./output'):
    os.makedirs('./output')
os.makedirs("output/"+ outdir)

os.chdir("output/"+ outdir)  

f = h5py.File("output_"+name,"w")
dset_result = f.create_dataset("full_result",opt.result.shape,dtype=np.complex64,data=opt.result)
#dset_result_ref = f.create_dataset("ref_full_result",opt_t.result.shape,dtype=np.complex64,data=opt_t.result)
dset_T1 = f.create_dataset("T1_final",np.squeeze(opt.result[-1,1,...]).shape,dtype=np.complex64,data=np.squeeze(opt.result[-1,1,...]))
dset_M0 = f.create_dataset("M0_final",np.squeeze(opt.result[-1,0,...]).shape,dtype=np.complex64,data=np.squeeze(opt.result[-1,0,...]))
#dset_T1_ref = f.create_dataset("T1_ref",np.squeeze(opt_t.result[-1,1,...]).shape,dtype=np.complex64,data=np.squeeze(opt_t.result[-1,1,...]))
#dset_M0_ref = f.create_dataset("M0_ref",np.squeeze(opt_t.result[-1,0,...]).shape,dtype=np.complex64,data=np.squeeze(opt_t.result[-1,0,...]))
f.create_dataset("T1_guess",np.squeeze(model.T1_guess).shape,dtype=np.float64,data=np.squeeze(model.T1_guess))
f.create_dataset("M0_guess",np.squeeze(model.M0_guess).shape,dtype=np.float64,data=np.squeeze(model.M0_guess))
dset_result.attrs['data_norm'] = dscale
dset_result.attrs['dcf_scaling'] = (N*(np.pi/(4*Nproj)))
f.flush()
f.close()

os.chdir('..')
os.chdir('..')



################################################################################
### 3D viewer  ##########s######################################################
################################################################################              
def multi_slice_viewer(volume):

  if volume.ndim<=3: 
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
  elif volume.ndim==4:
    fig, ax = plt.subplots(int(np.floor(np.sqrt(volume.shape[0]))),int(np.ceil(np.sqrt(volume.shape[0]))))
    ax = ax.flatten()
    ni = int(np.ceil(np.sqrt(volume.shape[0])))
    nj = int(np.floor(np.sqrt(volume.shape[0])))
    for j in range(nj):
      for i in range(ni):
        if i+ni*j >= volume.shape[0]:
          ax[i+ni*j].volume = np.zeros_like(volume[0])
        else:
          ax[i+ni*j].volume = volume[i+(j*ni)]
          ax[i+ni*j].index = volume[i+(j*ni)].shape[0] // 2
          ax[i+ni*j].imshow(volume[i+(j*ni),ax[i+ni*j].index])
  else:
    raise NameError('Unsupported Dimensions')
  fig.canvas.mpl_connect('scroll_event', process_scroll)
#  axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
#  axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
#  bnext = Button(axnext,'Next')
#  bprev = Button(axprev,'Prev')
def process_scroll(event):
  fig = event.canvas.figure
  ax = fig.axes
  for i in range(len(ax)):
    volume = ax[i].volume
    if (int((ax[i].index - event.step) >= volume.shape[0]) or 
           int((ax[i].index - event.step) < 0)):
           pass
    else:
      ax[i].index = int((ax[i].index - event.step) % volume.shape[0])
      ax[i].images[0].set_array(volume[ax[i].index])
      fig.canvas.draw()


