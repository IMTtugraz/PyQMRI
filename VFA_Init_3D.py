import pyfftw
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.io as sio
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier as nlinvns
import Model_Reco as Model_Reco
import multiprocessing as mp

import mkl
from pynfft.nfft import NFFT
from optimizedPattern import optimizedPattern
import VFA_model
np.seterr(divide='ignore', invalid='ignore')# TODO:
  
mkl.set_num_threads(mp.cpu_count())  
os.system("taskset -p 0xff %d" % os.getpid()) 
  

import ipyparallel as ipp

c = ipp.Client()

import h5py  
  
plt.ion()
pyfftw.interfaces.cache.enable()

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

data = h5py.File(file)
data_real = data['real_dat_comp'][()].astype("double")

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

data = h5py.File(file)
data_imag = data['imag_dat_comp'][()].astype("double")

#data = np.transpose(data)

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

traj = sio.loadmat(file)
traj = traj['traj']

traj = np.transpose(traj)

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

dcf = sio.loadmat(file)
dcf = dcf['dcf']

dcf = np.transpose(dcf)
#dcf = dcf/np.max(dcf)

data = data_real+1j*data_imag
data = np.fft.fftshift(data,axes=2)
data = data[:,:,24,:,:]
#data = data[:,:,0,:,:]
dimX = 256
dimY = 256
data = data*np.sqrt(dcf)

data = data[:,:,None,:,:]

#NSlice = 1
[NScan,NC,NSlice,Nproj, N] = data.shape
#[NScan,NC,NSlice,dimY,dimX] = data.shape




#Create par struct to store everyting
class struct:
    pass
par = struct()

par.NScan         = NScan 
#no b1 correction              
par.B1_correction = False 

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

fa_corr = sio.loadmat(file)
fa_corr = fa_corr['fa_corr']

fa_corr = np.transpose(fa_corr)
par.fa_corr =fa_corr
par.fa_corr[par.fa_corr==0] = 1
par.fa_corr = par.fa_corr[20,:,:]



################################################################### 
## Coil sensitivity estimate

#% Estimates sensitivities and complex image.
#%(see Martin Uecker: Image reconstruction by regularized nonlinear
#%inversion joint estimation of coil sensitivities and image content)
nlinvNewtonSteps = 6
nlinvRealConstr  = False

traj_coil = np.reshape(traj,(NScan*Nproj,N))
coil_plan = NFFT((dimY,dimX),NScan*Nproj*N)
coil_plan.x = np.transpose(np.array([np.imag(traj_coil.flatten()),np.real(traj_coil.flatten())]))
coil_plan.precompute()
       
par.C = np.zeros((NC,NSlice,dimY,dimX), dtype="complex128")       
par.phase_map = np.zeros((NSlice,dimY,dimX), dtype="complex128")   

result = []
for i in range(NSlice):
  print('deriving M(TI(1)) and coil profiles')

  
  ##### RADIAL PART
  combinedData = np.transpose(data[:,:,i,:,:],(1,0,2,3))
  combinedData = np.reshape(combinedData,(NC,NScan*Nproj,N))
  coilData = np.zeros((NC,dimY,dimX),dtype='complex128')
  for j in range(NC):
      coil_plan.f = combinedData[j,:,:]*np.repeat(np.sqrt(dcf),NScan,axis=0)
      coilData[j,:,:] = coil_plan.adjoint()
      
  combinedData = np.fft.fft2(coilData,norm=None)/np.sqrt(dimX*dimY)     
  ### CARTESIAN PART    
#  combinedData = np.squeeze(np.sum(data[:,:,i,:,:],0))/NSlice



  """shape combinData(128, 128, 4)"""
  #            print('nlivout')np.
  #            print(combinedData[0,0,0])
  dview = c[int(np.floor(i*len(c)/NSlice))]
  result.append(dview.apply_async(nlinvns.nlinvns, combinedData, nlinvNewtonSteps,
                     True, nlinvRealConstr))

  
  
            
#  nlinvout = nlinvns.nlinvns(combinedData, nlinvNewtonSteps,
#                     True, nlinvRealConstr) #(6, 9, 128, 128)

  #% coil sensitivities are stored in par.C

  

  
for i in range(NSlice):  
  par.C[:,i,:,:] = result[i].get()[2:,-1,:,:]

  if not nlinvRealConstr:
    par.phase_map[i,:,:] = np.exp(1j * np.angle( result[i].get()[0,-1,:,:]))
    par.C[:,i,:,:] = par.C[:,i,:,:]* np.exp(1j * np.angle( result[i].get()[1,-1,:,:]))
    
    # standardize coil sensitivity profiles
sumSqrC = np.sqrt(np.sum((par.C * np.conj(par.C)),0)) #4, 9, 128, 128
if NC == 1:
  par.C = sumSqrC 
else:
  par.C = par.C / np.tile(sumSqrC, (NC,1,1,1)) 


################################################################### 
## Choose undersampling mode
Nproj = 21

for i in range(NScan):
  data[i,:,:,:Nproj,:] = data[i,:,:,i*Nproj:(i+1)*Nproj,:]
  traj[i,:Nproj,:] = traj[i,i*Nproj:(i+1)*Nproj,:]


data = data[:,:,:,:Nproj,:]
traj = traj[:,:Nproj,:]
dcf = dcf[:Nproj,:]






print("**undersampling")
  
undersampling_mode = 1

def one():
    # Fully Sampled
    global uData
    par.AF = 1
    par.ACL = 32
    uData = data

def two():
    # radial Pattern
    global uData
    AF = 6
    par.AF = AF
    ACL = 32
    par.ACL = ACL
    uData = np.zeros(data.shape)
    uData      = optimizedPattern(data,AF,ACL); #data?
    
def three():
    # Random Pattern %% Vorerst nicht portieren
    uData = np.zeros_like(data)
    uData[:,:,:,:,list(range(0,dimY,3))] = data[:,:,:,:,list(range(0,dimY,3))]
    print(" Random Pattern")
    
options = {1 : one,
           2 : two,
           3 : three,}

options[undersampling_mode]()


######################################################################## 
## struct par init

FA = np.array([2,3,4,5,7,9,11,14,17,22],np.complex128)
#FA = np.array([1,3,5,7,9,11,13,15,17],np.complex128)
fa = FA    #  % flip angle in rad FA siehe FLASH phantom generierung
#alpha = [1,3,5,7,9,11,13,15,17,19]*pi/180;

par.TR          = 5.0#3.4 #TODO
par.TE          = list(range(20,40*20+1,20)) #TODO
par.NC          = NC
par.dimY        = dimY
par.dimX        = dimX
par.fa          = fa
par.NSlice      = NSlice
par.NScan       = NScan
par.N = N
par.Nproj = Nproj


par.unknowns = 2



##### No FA correction
#par.fa_corr = np.ones([NSlice,dimX,dimY],dtype='complex128')

#root = Tk()
#root.withdraw()
#root.update()
#file = filedialog.askopenfilename()
#root.destroy()

#fa_corr = sio.loadmat(file)
#fa_corr = fa_corr['fa_mid_3mm']

#fa_corr = np.transpose(fa_corr)
#fa_corr[[fa_corr==0]] = 1
#par.fa_corr = fa_corr[None,:,:]*180/np.pi

'''standardize the data'''


dscale = np.sqrt(NSlice)*np.complex128(0.1)/(np.linalg.norm(uData.flatten()))
par.dscale = dscale

######################################################################## 
#### generate FFTW

uData = pyfftw.byte_align(uData)

fftw_ksp = pyfftw.empty_aligned((dimX,dimY),dtype='complex128')
fftw_img = pyfftw.empty_aligned((dimX,dimY),dtype='complex128')

fft_forward = pyfftw.FFTW(fftw_img,fftw_ksp,axes=(0,1))
fft_back = pyfftw.FFTW(fftw_ksp,fftw_img,axes=(0,1),direction='FFTW_BACKWARD')


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
  result = np.zeros((NScan,NC,NSlice,Nproj*N),dtype='complex128')
  for i in range(siz[0]):
    for j in range(siz[1]): 
      for k in range(siz[2]):
        plan[i][j].f_hat = x[i,j,k,:,:]/dimX
        result[i,j,k,:] = plan[i][j].trafo()*np.sqrt(dcf).flatten()
      
  return np.reshape(result,[NScan,NC,NSlice,Nproj,N])


def nFTH(x,plan,dcf,NScan,NC,NSlice,dimY,dimX):
  siz = np.shape(x)
  result = np.zeros((NScan,NC,NSlice,dimY,dimX),dtype='complex128')
  for i in range(siz[0]):
    for j in range(siz[1]):  
      for k in range(siz[2]):
        plan[i][j].f = x[i,j,k,:]*np.sqrt(dcf)
        result[i,j,k,:,:] = plan[i][j].adjoint()
      
  return result/dimX



def FT(x):
  siz = np.shape(x)
  result = np.zeros_like(x,dtype='complex128')
  for i in range(siz[0]):
    for j in range(siz[1]):
      for k in range(siz[2]):
        result[i,j,k,:,:] = fft_forward(x[i,j,k,:,:])/np.sqrt(siz[4]*(siz[3]))
      
  return result


def FTH(x):
  siz = np.shape(x)
  result = np.zeros_like(x,dtype='complex128')
  for i in range(siz[0]):
    for j in range(siz[1]):
      for k in range(siz[2]):
        result[i,j,k,:,:] = fft_back(x[i,j,k,:,:])*np.sqrt(siz[4]*(siz[3]))
      
  return result


plan = nfft(NScan,NC,dimX,dimY,N,Nproj,traj)
#

uData = np.reshape(uData,(NScan,NC,NSlice,Nproj,N))* dscale

images= (np.sum(nFTH(uData,plan,dcf* N*np.pi/(4*Nproj),NScan,NC,NSlice,dimY,dimX)[:None,:,:,:]*(np.conj(par.C)),axis = 1))

#images= (np.sum(FTH(uData*dscale)*(np.conj(par.C)),axis = 1))


########################################################################
#Init Forward Model
model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,par.phase_map)



par.U = np.ones((uData).shape, dtype=bool)
par.U[abs(uData) == 0] = False
########################################################################
#Init optimizer
opt = Model_Reco.Model_Reco(par)

opt.par = par
opt.data =  uData
#model.data = uData*dscale
opt.images = images
opt.fft_forward = fft_forward
opt.fft_back = fft_back
opt.nfftplan = plan
opt.dcf = np.sqrt(dcf* N*np.pi/(4*Nproj))
opt.dcf_flat =np.sqrt( dcf.flatten()* N*np.pi/(4*Nproj))
opt.model = model


opt.traj = traj

#
##
#xx = np.random.randn(NScan,NC,NSlice,dimX,dimY)
#yy = np.random.randn(NScan,NC,NSlice,Nproj*N)
#a = np.vdot(xx,nFTH(yy,plan,dcf,NScan,NC,NSlice,dimY,dimX))
#b = np.vdot(nFT(xx,plan,dcf,NScan,NC,NSlice,Nproj,N,dimX),yy)
#test = np.abs(a-b)
#print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,test))



########################################################################
#IRGN Params
irgn_par = struct()
irgn_par.start_iters = 10
irgn_par.max_iters = 1000
irgn_par.max_GN_it = 10
irgn_par.lambd = 1e6
irgn_par.gamma = 1e-8
irgn_par.delta = 1e4
irgn_par.display_iterations = True

opt.irgn_par = irgn_par


opt.dz = 3.0

opt.execute_2D()

#
#import cProfile
#cProfile.run("opt.execute_2D()","eval_speed")
##
#import pstats
#
#p=pstats.Stats("eval_speed")
#p.sort_stats('time').print_stats(20)






########################################################################
#starte reco

#result,guess,par = DESPOT1_Model_Reco(par)

outdir = time.strftime("%Y-%m-%d  %H-%M-%S")
if not os.path.exists('./output'):
    os.makedirs('./output')
os.makedirs("output/"+ outdir)

os.chdir("output/"+ outdir)

sio.savemat("result.mat",{"result":opt.result,"model":model})

import pickle
with open("par" + ".p", "wb") as pickle_file:
    pickle.dump(par, pickle_file)
#with open("par.txt", "rb") as myFile:
    #par = pickle.load(myFile)
#par.dump("par.dat")

