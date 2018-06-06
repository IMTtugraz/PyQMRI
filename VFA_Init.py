import numpy as np

import time
import os
import h5py
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier as nlinvns

import Model_Reco as Model_Reco

import VFA_model as VFA_model
import goldcomp

import primaldualtoolbox

DTYPE = np.complex64
np.seterr(divide='ignore', invalid='ignore')

################################################################################
### Initiate parallel interface ################################################
################################################################################

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
reco_Slices = 40

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
### Coil Sensitivity Estimation ################################################
################################################################################
#% Estimates sensitivities and complex image.
#%(see Martin Uecker: Image reconstruction by regularized nonlinear
#%inversion joint estimation of coil sensitivities and image content)

nlinvNewtonSteps = 6
nlinvRealConstr  = False

traj_x = np.real(np.asarray(traj))
traj_y = np.imag(np.asarray(traj))

config = {'osf' : 2,
            'sector_width' : 8,
            'kernel_width' : 3,
            'img_dim' : dimX}

points = (np.array([traj_x.flatten(),traj_y.flatten()]))
op = primaldualtoolbox.mri.MriRadialOperator(config)
op.setTrajectory(points)
op.setDcf(np.repeat(np.sqrt(dcf),NScan,axis=0).flatten().astype(np.float32)[None,...])
op.setCoilSens(np.ones((1,dimX,dimY),dtype=DTYPE))


par.C = np.zeros((NC,NSlice,dimY,dimX), dtype=DTYPE)
par.phase_map = np.zeros((NSlice,dimY,dimX), dtype=DTYPE)

result = []
for i in range(NSlice):
  print('deriving M(TI(1)) and coil profiles')


  ##### RADIAL PART
  combinedData = np.transpose(data[:,:,i,:,:],(1,0,2,3))
  combinedData = np.reshape(combinedData,(NC,NScan*Nproj*N))
  coilData = np.zeros((NC,dimY,dimX),dtype=DTYPE)
  for j in range(NC):
      coilData[j,:,:] = op.adjoint(combinedData[j,:]*(np.repeat(np.sqrt(dcf),NScan,axis=0).flatten())[None,...])

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
sumSqrC = np.sqrt(np.sum((par.C * np.conj(par.C)),0))
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
data = data
dscale = np.sqrt(NSlice)*DTYPE(np.sqrt(2*1e3))/(np.linalg.norm(data.flatten()))
par.dscale = dscale
################################################################################
### generate nFFT  #############################################################
################################################################################

def gpuNUFFT(NScan,NSlice,dimX,traj,dcf,Coils):
  plan = []

  traj_x = np.real(np.asarray(traj))
  traj_y = np.imag(np.asarray(traj))

  config = {'osf' : 2,
            'sector_width' : 8,
            'kernel_width' : 3,
            'img_dim' : dimX}

  for i in range(NScan):
    plan.append([])
    points = (np.array([traj_x[i,:,:].flatten(),traj_y[i,:,:].flatten()]))
    for j in range(NSlice):
      op = primaldualtoolbox.mri.MriRadialOperator(config)
      op.setTrajectory(points)
      op.setDcf(dcf.flatten().astype(np.float32)[None,...])
      op.setCoilSens(np.require(Coils[:,j,...],DTYPE,'C'))
      plan[i].append(op)


  return plan

def nFT_gpu(plan,x):
    result = np.zeros((NScan,NC,NSlice,Nproj*N),dtype=DTYPE)
    for scan in range(NScan):
      for islice in range(NSlice):
        result[scan,:,islice,...] = plan[scan][islice].forward(np.require(x[scan,:,islice,...],DTYPE,'C'))

    return np.reshape(result,[NScan,NC,NSlice,Nproj,N])




def nFTH_gpu(x,Coils):
    traj_x = np.real(np.asarray(traj))
    traj_y = np.imag(np.asarray(traj))

    config = {'osf' : 2,
            'sector_width' : 8,
            'kernel_width' : 3,
            'img_dim' : dimX}
    result = np.zeros((NScan,NSlice,dimX,dimY),dtype=DTYPE)
    x = np.require(np.reshape(x,(NScan,NC,NSlice,Nproj*N)))
    for scan in range(NScan):
      points = (np.array([traj_x[scan,:,:].flatten(),traj_y[scan,:,:].flatten()]))
      for islice in range(NSlice):
          op = primaldualtoolbox.mri.MriRadialOperator(config)
          op.setTrajectory(points)
          op.setDcf(dcf.flatten().astype(np.float32)[None,...])
          op.setCoilSens(np.require(Coils[:,islice,...],DTYPE,'C'))
          result[scan,islice,...] = op.adjoint(np.require(x[scan,:,islice,...],DTYPE,'C'))

    return result



data = data* dscale
images= nFTH_gpu(data,par.C)

del op

################################################################################
### IRGN - TGV Reco ############################################################
################################################################################
import matplotlib.pyplot as plt
opt = Model_Reco.Model_Reco(par)
result_tgv = []
################################################################################
### Init forward model and initial guess #######################################
################################################################################
model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,\
                        par.phase_map,NSlice,Nproj)
opt.par = par
opt.data =  data
opt.images = images
opt.dcf = (dcf)
opt.dcf_flat = (dcf).flatten()
opt.model = model
opt.traj = traj
opt.dz = 1

################################################################################
##IRGN Params
irgn_par = struct()
irgn_par.max_iters = 300
irgn_par.start_iters = 100
irgn_par.max_GN_it = 13
irgn_par.lambd = 1e2
irgn_par.gamma = 1e0
irgn_par.delta = 1e-1
irgn_par.omega = 0e-10
irgn_par.display_iterations = True
irgn_par.gamma_min = 0.18
irgn_par.delta_max = 1e2
irgn_par.tol = 5e-3
irgn_par.stag = 1
irgn_par.delta_inc = 2
irgn_par.gamma_dec = 0.7
opt.irgn_par = irgn_par
opt.ratio = 2e2

opt.execute_2D()
result_tgv.append(opt.result)
plt.close('all')


res = opt.gn_res
res = np.array(res)/(irgn_par.lambd*NSlice)
scale_E1_TGV = opt.model.T1_sc
################################################################################
#### IRGN - TV referenz ########################################################
################################################################################
result_tv = []
################################################################################
### Init forward model and initial guess #######################################
#############################################################re###################
model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,\
                        par.phase_map,NSlice,Nproj)
opt.model = model
################################################################################
##IRGN Params
irgn_par = struct()
irgn_par.max_iters = 300
irgn_par.start_iters = 100
irgn_par.max_GN_it = 13
irgn_par.lambd = 1e2
irgn_par.gamma = 1e0
irgn_par.delta = 1e-1
irgn_par.omega = 0e-10
irgn_par.display_iterations = True
irgn_par.gamma_min = 0.23
irgn_par.delta_max = 1e2
irgn_par.tol = 5e-3
irgn_par.stag = 1.00
irgn_par.delta_inc = 2
irgn_par.gamma_dec = 0.7
opt.irgn_par = irgn_par
opt.ratio = 2e2

opt.execute_2D(1)
result_tv.append(opt.result)
plt.close('all')

scale_E1_ref = opt.model.T1_sc
res_tv = opt.gn_res
res_tv = np.array(res_tv)/(irgn_par.lambd*NSlice)

################################################################################
#### IRGN - WT referenz ########################################################
################################################################################
result_wt = []
################################################################################
### Init forward model and initial guess #######################################
################################################################################
model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,\
               par.phase_map,NSlice,Nproj)
opt.par = par
opt.data =  data
opt.images = images
opt.dcf = (dcf)
opt.dcf_flat = (dcf).flatten()
opt.model = model
opt.traj = traj

opt.dz = 1

################################################################################
##IRGN Params
irgn_par = struct()
irgn_par.max_iters = 300
irgn_par.start_iters = 100
irgn_par.max_GN_it = 13
irgn_par.lambd = 1e2
irgn_par.gamma = 1e0
irgn_par.delta = 1e-1
irgn_par.omega = 0e-10
irgn_par.display_iterations = True
irgn_par.gamma_min = 0.37
irgn_par.delta_max = 1e2
irgn_par.tol = 5e-3
irgn_par.stag = 1.00
irgn_par.delta_inc = 2
irgn_par.gamma_dec = 0.7
opt.irgn_par = irgn_par
opt.ratio = 2e2

opt.execute_2D(2)
result_wt.append(opt.result)
plt.close('all')


res = opt.gn_res
res = np.array(res)/(irgn_par.lambd*NSlice)
del opt
###############################################################################
## New .hdf5 save files #######################################################
###############################################################################
outdir = time.strftime("%Y-%m-%d  %H-%M-%S_MRI_joint_2D_"+name[:-3])
if not os.path.exists('./output'):
    os.makedirs('./output')
os.makedirs("output/"+ outdir)

os.chdir("output/"+ outdir)
f = h5py.File("output_"+name,"w")

for i in range(len(result_tgv)):
  dset_result=f.create_dataset("tgv_full_result_"+str(i),result_tgv[i].shape,\
                               dtype=DTYPE,data=result_tgv[i])
  dset_result_ref=f.create_dataset("tv_full_result_"+str(i),result_tv[i].shape,\
                                   dtype=DTYPE,data=result_tv[i])
  dset_result_ref=f.create_dataset("wt_full_result_"+str(i),result_wt[i].shape,\
                                   dtype=DTYPE,data=result_wt[i])
  f.attrs['data_norm'] = dscale
  f.attrs['dcf_scaling'] = (N*(np.pi/(4*Nproj)))
  f.attrs['E1_scale_TGV'] =scale_E1_TGV
  f.attrs['E1_scale_ref'] =scale_E1_ref
  f.attrs['M0_scale'] = model.M0_sc
  f.attrs['IRGN_TGV_res'] = res
  f.attrs['IRGN_TV_res'] = res_tv
  f.attrs['dscale'] = dscale
  f.flush()
f.close()

os.chdir('..')
os.chdir('..')