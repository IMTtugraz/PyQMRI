import numpy as np

import time
import os
import h5py  
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier as nlinvns

import Model_Reco as Model_Reco
import Model_Reco_old as Model_Reco_Tikh

#from pynfft.nfft import NFFT

import VFA_model as VFA_model
import goldcomp

import primaldualtoolbox

DTYPE = np.complex64
np.seterr(divide='ignore', invalid='ignore')
   
#os.system("taskset -p 0xfff %d" % os.getpid()) 

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
data = data
dscale = np.sqrt(NSlice)*DTYPE(np.sqrt(2*1e3))/(np.linalg.norm(data.flatten()))
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


#plan = gpuNUFFT(NScan,NSlice,dimX,traj,dcf,par.C)

data = data* dscale

data_save = data

#images= (np.sum(nFTH(data_save,plan,dcf,NScan,NC,\
#                     NSlice,dimY,dimX)*(np.conj(par.C)),axis = 1))

images= nFTH_gpu(data,par.C)

#del plan
del op


################################################################################
### Init forward model and initial guess #######################################
################################################################################
model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,\
                            par.phase_map,NSlice,Nproj)

#test_T1 = np.reshape(np.linspace(10,5500,dimX*dimY*NSlice),(NSlice,dimX,dimY))/model.T1_sc
#G_x = model.execute_forward_3D(np.array([1*np.ones((NSlice,dimY,dimX),dtype=DTYPE),np.exp(-model.TR/(test_T1*np.ones((NSlice,dimY,dimX),dtype=DTYPE)))],dtype=DTYPE))
##test_T1*np.ones((NSlice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE))#
#model.M0_sc = model.M0_sc*np.max(np.abs(images))/np.max(np.abs(G_x))

par.U = np.ones((data).shape, dtype=bool)
par.U[abs(data) == 0] = False
################################################################################
### IRGN - TGV Reco ############################################################
################################################################################

opt = Model_Reco.Model_Reco(par)

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
irgn_par.max_GN_it = 20
irgn_par.lambd = 1e2
irgn_par.gamma = 1e2   #### 5e-2   5e-3 phantom ##### brain 1e-2
irgn_par.delta = 1e-2 #### 8spk in-vivo 1e-2
irgn_par.omega = 0e-10
irgn_par.display_iterations = True
irgn_par.gamma_min = 2e-1  
irgn_par.delta_max = 1e0
irgn_par.tol = 5e-3
irgn_par.stag = 1.00
irgn_par.delta_inc = 2
irgn_par.gamma_dec = 0.5
opt.irgn_par = irgn_par

opt.execute_3D()

result_tgv = opt.result
res = opt.gn_res
res = np.array(res)/(irgn_par.lambd*NSlice)
scale_E1_TGV = opt.model.T1_sc
del opt

###############################################################################
## IRGN - Tikhonov referenz ###################################################
###############################################################################
model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,\
                            par.phase_map,NSlice,Nproj)

opt_t = Model_Reco_Tikh.Model_Reco(par)

opt_t.par = par
opt_t.data =  data
opt_t.images = images
opt_t.dcf = (dcf)
opt_t.dcf_flat = (dcf).flatten()
opt_t.model = model
opt_t.traj = traj 

###############################################################################
#IRGN Params
irgn_par = struct()
irgn_par.start_iters = 100
irgn_par.max_iters = 300
irgn_par.max_GN_it = 20
irgn_par.lambd = 1e2
irgn_par.gamma = 1e-2  #### 5e-2   5e-3 phantom ##### brain 1e-2
irgn_par.delta = 1e-4  #### 8spk in-vivo 1e-2
irgn_par.omega = 0e0
irgn_par.display_iterations = True
irgn_par.gamma_min = 1e-6
irgn_par.delta_max = 1e0
irgn_par.tol = 1e-5
irgn_par.stag = 1.05
irgn_par.delta_inc = 2
irgn_par.gamma_dec = 0.5
opt_t.irgn_par = irgn_par


opt_t.execute_3D()

result_ref = opt_t.result
scale_E1_ref = opt_t.model.T1_sc
del opt_t
###############################################################################
## New .hdf5 save files #######################################################
###############################################################################
outdir = time.strftime("%Y-%m-%d  %H-%M-%S_"+name[:-3])
if not os.path.exists('./output'):
    os.makedirs('./output')
os.makedirs("output/"+ outdir)

os.chdir("output/"+ outdir)  
f = h5py.File("output_"+name,"w")
dset_result=f.create_dataset("full_result",result_tgv.shape,\
                             dtype=DTYPE,data=result_tgv)
dset_result_ref=f.create_dataset("ref_full_result",result_ref.shape,\
                                 dtype=DTYPE,data=result_ref)
dset_T1=f.create_dataset("T1_final",np.squeeze(result_tgv[-1,1,...]).shape,\
                         dtype=DTYPE,\
                         data=np.squeeze(result_tgv[-1,1,...]))
dset_M0=f.create_dataset("M0_final",np.squeeze(result_tgv[-1,0,...]).shape,\
                         dtype=DTYPE,\
                         data=np.squeeze(result_tgv[-1,0,...]))
dset_T1_ref=f.create_dataset("T1_ref",np.squeeze(result_ref[-1,1,...]).shape\
                             ,dtype=DTYPE,\
                             data=np.squeeze(result_ref[-1,1,...]))
dset_M0_ref=f.create_dataset("M0_ref",np.squeeze(result_ref[-1,0,...]).shape\
                             ,dtype=DTYPE,\
                             data=np.squeeze(result_ref[-1,0,...]))
#f.create_dataset("T1_guess",np.squeeze(model.T1_guess).shape,\
#                 dtype=np.float64,data=np.squeeze(model.T1_guess))
#f.create_dataset("M0_guess",np.squeeze(model.M0_guess).shape,\
#                 dtype=np.float64,data=np.squeeze(model.M0_guess))
f.attrs['data_norm'] = dscale
f.attrs['dcf_scaling'] = (N*(np.pi/(4*Nproj)))
f.attrs['E1_scale_TGV'] =scale_E1_TGV
f.attrs['E1_scale_ref'] =scale_E1_ref
f.attrs['M0_scale'] = model.M0_sc
f.attrs['IRGN_TGV_res'] = res
f.flush()
f.close()

os.chdir('..')
os.chdir('..')
