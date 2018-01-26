import numpy as np
import reikna

import time
import os
import h5py
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier as nlinvns

import Model_Reco as Model_Reco
import Model_Reco_old as Model_Reco_Tikh


api = reikna.cluda.ocl_api()
thr = api.Thread.create()



import IRLL_Model as IRLL_Model



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
test_data = [ 'fa_corr', 'imag_dat', 'real_dat',
             ]
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
data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data,axes=(-1,-2))),axes=-1)
test = np.zeros((50,13,1,256,256),dtype=DTYPE)
test[:,:,:,24:-24,22:-22] = data
data = DTYPE(np.fft.fft2(test))
       

dimX, dimY, NSlice = (file.attrs['image_dimensions']).astype(int)

############### Set number of Slices ###########################################
reco_Slices = 1

os_slices = 0
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

[NScan,NC,NSlice,dimY,dimX] = data.shape

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

par.unknowns_TGV = 2
par.unknowns_H1 = 0
par.unknowns = 2


################################################################################
### Estimate coil sensitivities ################################################
################################################################################


nlinvNewtonSteps = 7
nlinvRealConstr  = False
        
for i in range(0,(NSlice)):
  print('deriving M(TI(1)) and coil profiles')
  
  
  ##### RADIAL PART
  combinedData = np.mean(data,0)
  coilData = np.zeros((NC,dimY,dimX),dtype=DTYPE)
            
  nlinvout = nlinvns.nlinvns(np.squeeze(combinedData), nlinvNewtonSteps,
                     True, nlinvRealConstr) #(6, 9, 128, 128)

  #% coil sensitivities are stored in par.C
  par.C = np.zeros(nlinvout[2:,-1,:,:].shape, dtype='complex128')
  par.C[:,:,:] = nlinvout[2:,-1,:,:]

  if not nlinvRealConstr:
    par.phase_map = np.exp(1j * np.angle(nlinvout[0,-1,:,:]))
    par.C = par.C* np.exp(1j * np.angle(nlinvout[1,-1,:,:]))
    
    # standardize coil sensitivity profiles
  sumSqrC = np.sqrt(np.sum((par.C * np.conj(par.C)),0)) #4, 9, 128, 128
  if NC == 1:
    par.C = sumSqrC 
  else:
    par.C = par.C / np.tile(sumSqrC, (NC,1,1)) 
  
  par.C = np.expand_dims(par.C,axis=1)

################################################################################
### Calcualte wait time   ######################################################
################################################################################
par.TR = file.attrs['TR']

################################################################################
### Standardize data norm ######################################################
################################################################################

#### Close File after everything was read
file.close()
################################################################################
### Scale Data #################################################################
################################################################################

dscale = np.sqrt(NSlice)*DTYPE(np.sqrt(200))/(np.linalg.norm(data.flatten()))
par.dscale = dscale
data = data*dscale

################################################################################
### generate nFFT for radial cases #############################################
################################################################################

#print("**undersampling")
#  
#undersampling_mode = 1
#
#def one():
#    # Fully Sampled
#    global uData
#    par.AF = 1
#    par.ACL = 32
#    uData = data[:,:,None,:,:]
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


######################################################################## 
#### generate FFTW


#fftw_ksp = pyfftw.empty_aligned((NScan,NC,dimY,dimX),dtype=DTYPE)
#fftw_img = pyfftw.empty_aligned((NScan,NC,dimY,dimX),dtype=DTYPE)
#
fft = reikna.fft.FFT(np.squeeze(data),axes=(2,3)).compile(thr)


def FT(x):
  x_dev = thr.to_device(x)
  result_dev = thr.empty_like(x_dev)
  fft(result_dev,x_dev)
  return result_dev.get()


def FTH(x):
  x_dev = thr.to_device(x)
  result_dev = thr.empty_like(x_dev)
  fft(result_dev,x_dev,1)
      
  return result_dev.get()


images= (np.sum(FTH(data[:,:,0,:,:])[:,:,None,:,:]*(np.conj(par.C)),axis = 1))


################################################################################
### Init forward model and initial guess #######################################
################################################################################
model = IRLL_Model.IRLL_Model(par.fa,par.fa_corr,par.TR,par.tau,par.td,\
                              NScan,NSlice,dimY,dimX,1,NScan,1)

G_x = model.execute_forward_3D(np.array([1*np.ones((NSlice,dimY,dimX),dtype=DTYPE),1500/model.T1_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE))
model.M0_sc = model.M0_sc*np.max(np.abs(images))/np.max(np.abs(G_x))

par.U = np.ones((data).shape, dtype=bool)
par.U[abs(data) == 0] = False
################################################################################
### IRGN - TGV Reco ############################################################
################################################################################
opt = Model_Reco.Model_Reco(par)

opt.par = par
opt.data =  data
opt.images = images
opt.fft_forward = FT
opt.fft_back = FTH

opt.model = model


########################################################################
#IRGN Params
irgn_par = struct()
irgn_par.start_iters = 100
irgn_par.max_iters = 1000
irgn_par.max_GN_it = 30
irgn_par.lambd = 1e2
irgn_par.gamma = 5e-2 #### 5e-2   5e-3 phantom ##### brain 1e-3
irgn_par.delta = 1e2 ### 8spk in-vivo 5e2
irgn_par.omega = 1e-10
irgn_par.display_iterations = True
irgn_par.gamma_min = 1e-2
irgn_par.delta_max = 1e4
irgn_par.tol = 1e-4
irgn_par.stag = 1.2
irgn_par.delta_inc = 5
opt.irgn_par = irgn_par

opt.execute_2D_cart()

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