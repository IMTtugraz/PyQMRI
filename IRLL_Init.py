import numpy as np

import time
import os
import h5py
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier as nlinvns

import Model_Reco as Model_Reco
import Model_Reco_old as Model_Reco_Tikh

from pynfft.nfft import NFFT

import IRLL_Model_new as IRLL_Model

DTYPE = np.complex64
np.seterr(divide='ignore', invalid='ignore')

os.system("taskset -p 0xff %d" % os.getpid())

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
test_data = ['dcf', 'fa_corr', 'imag_dat', 'imag_traj', 'real_dat',
             'real_traj']
test_attributes = ['image_dimensions', 'tau', 'gradient_delay',
                   'flip_angle(s)', 'time_per_slice']

for datasets in test_data:
  if not (datasets in list(file.keys())):
    file.close()
    raise NameError("Error: '" + datasets +
                    "' data was not provided/wrongly named!")
for attributes in test_attributes:
  if not (attributes in list(file.attrs)):
    file.close()
    raise NameError("Error: '" + attributes +
                    "' was not provided/wrongly named as an attribute!")

################################################################################
### Read Data ##################################################################
################################################################################

data = file['real_dat'][()].astype(DTYPE) +\
       1j*file['imag_dat'][()].astype(DTYPE)

traj = file['real_traj'][()].astype(DTYPE) + \
       1j*file['imag_traj'][()].astype(DTYPE)

dcf = file['dcf'][()].astype(DTYPE)

dimX, dimY, NSlice = (file.attrs['image_dimensions']).astype(int)

############### Set number of Slices ###########################################
reco_Slices = 1

class struct:
    pass


par = struct()

################################################################################
### FA correction ##############################################################
################################################################################

par.fa_corr = file['fa_corr'][()].astype(DTYPE)

################################################################################
### Pick slices for reconstruction #############################################
################################################################################

data = data[None,:,int(NSlice/2)-\
            int(np.ceil(reco_Slices/2)):int(NSlice/2)+\
            int(np.floor(reco_Slices/2)),:,:]

  
par.fa_corr = np.flip(par.fa_corr,axis=0)[int(NSlice/2)-\
            int(np.ceil(reco_Slices/2)):int(NSlice/2)+\
            int(np.floor(reco_Slices/2)),:,:]

[NScan,NC,NSlice,Nproj, N] = data.shape

################################################################################
### Set sequence related parameters ############################################
################################################################################

par.tau         = file.attrs['tau']
par.td          = file.attrs['gradient_delay']
par.NC          = NC
par.dimY        = dimY
par.dimX        = dimX
par.fa          = file.attrs['flip_angle(s)']/180*np.pi
par.NSlice      = NSlice
par.NScan       = NScan
par.N = N
par.Nproj = Nproj

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

################################################################################
### Reorder acquired Spokes   ##################################################
################################################################################

data = data*np.sqrt(dcf)


Nproj_new = 21

NScan = np.floor_divide(Nproj,Nproj_new)
Nproj = Nproj_new

par.Nproj = Nproj
par.NScan = NScan

data = np.transpose(np.reshape(data[:,:,:,:Nproj*NScan,:],\
                               (NC,NSlice,NScan,Nproj,N)),(2,0,1,3,4))
traj =np.reshape(traj[:Nproj*NScan,:],(NScan,Nproj,N))
dcf = dcf[:Nproj,:]

################################################################################
### Calcualte wait time   ######################################################
################################################################################
par.TR = file.attrs['time_per_slice']-(par.tau*Nproj*NScan+par.td)

################################################################################
### Standardize data norm ######################################################
################################################################################

#### Close File after everything was read
file.close()

dscale = np.sqrt(NSlice)*DTYPE(1)/(np.linalg.norm(data.flatten()))
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


plan = nfft(NScan,NC,dimX,dimY,N,Nproj,traj)

data = data* dscale

images= (np.sum(nFTH(data,plan,dcf*N*(np.pi/(4*Nproj)),NScan,NC,NSlice,\
                     dimY,dimX)*(np.conj(par.C)),axis = 1))



################################################################################
### Init forward model and initial guess #######################################
################################################################################
model = IRLL_Model.IRLL_Model(par.fa,par.fa_corr,par.TR,par.tau,par.td,\
                              NScan,NSlice,dimY,dimX,Nproj)



par.U = np.ones((data).shape, dtype=bool)
par.U[abs(data) == 0] = False
################################################################################
### IRGN - TGV Reco ############################################################
################################################################################


opt = Model_Reco.Model_Reco(par)

opt.par = par
opt.data =  data
opt.images = images
opt.nfftplan = plan
opt.dcf = np.sqrt(dcf)
opt.dcf_flat = np.sqrt(dcf).flatten()
opt.model = model
opt.traj = traj

################################################################################
#IRGN Params
irgn_par = struct()
irgn_par.start_iters = 10
irgn_par.max_iters = 1000
irgn_par.max_GN_it = 10
irgn_par.lambd = 1e2
irgn_par.gamma = 1e-3   #### 5e-2   5e-3 phantom ##### brain 1e-3
irgn_par.delta = 1e1  #### 8spk in-vivo 5e2
irgn_par.omega = 1e-14
irgn_par.display_iterations = True

opt.irgn_par = irgn_par

opt.execute_2D()

################################################################################
### IRGN - Tikhonov referenz ###################################################
################################################################################

opt_t = Model_Reco_Tikh.Model_Reco(par)

opt_t.par = par
opt_t.data =  data
opt_t.images = images
#opt_t.fft_forward = fft_forward
#opt_t.fft_back = fft_back
opt_t.nfftplan = plan
opt_t.dcf = np.sqrt(dcf*(N*(np.pi/(4*Nproj))))
opt_t.dcf_flat = np.sqrt(dcf*(N*(np.pi/(4*Nproj)))).flatten()
opt_t.model = model
opt_t.traj = traj

################################################################################
##IRGN Params
irgn_par = struct()
irgn_par.start_iters = 10
irgn_par.max_iters = 1000
irgn_par.max_GN_it = 10
irgn_par.lambd = 1e2
irgn_par.gamma = 1e-2  #### 5e-2   5e-3 phantom ##### brain 1e-2
irgn_par.delta = 1e-3  #### 8spk in-vivo 1e-2
irgn_par.omega = 1e0
irgn_par.display_iterations = True

opt_t.irgn_par = irgn_par

opt_t.execute_2D()

################################################################################
### New .hdf5 save files #######################################################
################################################################################
outdir = time.strftime("%Y-%m-%d  %H-%M-%S_"+name[:-3])
if not os.path.exists('./output'):
    os.makedirs('./output')
os.makedirs("output/"+ outdir)

os.chdir("output/"+ outdir)

f = h5py.File("output_"+name,"w")
dset_result=f.create_dataset("full_result",opt.result.shape,\
                             dtype=np.complex64,data=opt.result)
dset_result_ref=f.create_dataset("ref_full_result",opt_t.result.shape,\
                                 dtype=np.complex64,data=opt_t.result)
dset_T1=f.create_dataset("T1_final",np.squeeze(opt.result[-1,1,...]).shape,\
                         dtype=np.complex64,\
                         data=np.squeeze(opt.result[-1,1,...]))
dset_M0=f.create_dataset("M0_final",np.squeeze(opt.result[-1,0,...]).shape,\
                         dtype=np.complex64,\
                         data=np.squeeze(opt.result[-1,0,...]))
dset_T1_ref=f.create_dataset("T1_ref",np.squeeze(opt_t.result[-1,1,...]).shape\
                             ,dtype=np.complex64,\
                             data=np.squeeze(opt_t.result[-1,1,...]))
dset_M0_ref=f.create_dataset("M0_ref",np.squeeze(opt_t.result[-1,0,...]).shape\
                             ,dtype=np.complex64,\
                             data=np.squeeze(opt_t.result[-1,0,...]))
#f.create_dataset("T1_guess",np.squeeze(model.T1_guess).shape,\
#                 dtype=np.float64,data=np.squeeze(model.T1_guess))
#f.create_dataset("M0_guess",np.squeeze(model.M0_guess).shape,\
#                 dtype=np.float64,data=np.squeeze(model.M0_guess))
dset_result.attrs['data_norm'] = dscale
dset_result.attrs['dcf_scaling'] = (N*(np.pi/(4*Nproj)))
f.flush()
f.close()

os.chdir('..')
os.chdir('..')