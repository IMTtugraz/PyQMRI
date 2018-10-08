#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:11:18 2018

@author: omaier
"""

import numpy as np

import h5py
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier_3D as nlinvns

import goldcomp
import sys

import multislice_viewer as msv
import pyopencl.array as clarray

DTYPE = np.complex64
np.seterr(divide='ignore', invalid='ignore')

################################################################################
### Select input file ##########################################################
################################################################################

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

file = h5py.File(file)

################################################################################
### Read Data ##################################################################
################################################################################
dimX, dimY, NSlice = (file.attrs['image_dimensions']).astype(int)
reco_Slices = 10

data = file['real_dat'][:,:,int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...].astype(DTYPE) +\
       1j*file['imag_dat'][:,:,int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...].astype(DTYPE)


traj = file['real_traj'][()].astype(DTYPE) + \
       1j*file['imag_traj'][()].astype(DTYPE)


dcf = np.array(goldcomp.cmp(traj),dtype=DTYPE)


[NScan,NC,NSlice,Nproj, N] = data.shape

(ctx,queue,FFT) = nlinvns.NUFFT(N,NScan,NC,NSlice,traj,dcf)

data = np.fft.fft(np.fft.fftshift(data,2),axis=2,norm='ortho')
data = np.fft.fftshift(data,2).astype(DTYPE)


nlinvNewtonSteps = 6
nlinvRealConstr  = False

traj_coil = np.reshape(traj,(NScan*Nproj,N))
dcf_coil = np.array(goldcomp.cmp(traj_coil),dtype=DTYPE)

C = np.zeros((NC,reco_Slices,dimY,dimX), dtype=DTYPE)
phase_map = np.zeros((reco_Slices,dimY,dimX), dtype=DTYPE)

sys.stdout.write("Computing coil sensitivity maps of 3D volume\r")
sys.stdout.flush()

##### RADIAL PART
combinedData = np.transpose(data,(1,2,0,3,4))
combinedData = np.reshape(combinedData,(NC,NSlice,NScan*Nproj,N))*np.sqrt(dcf_coil)

result = nlinvns.nlinvns(combinedData,nlinvNewtonSteps,traj_coil,dcf_coil,1,True,nlinvRealConstr)

C= result[2:,-1,...]
if not nlinvRealConstr:
  phase_map = np.exp(1j * np.angle( result[0,-1,...]))
  C = C* np.exp(1j *\
       np.angle( result[1,-1,...]))

    # standardize coil sensitivity profiles
sumSqrC = np.sqrt(np.sum((C * np.conj(C)),0)) #4, 9, 128, 128
if NC == 1:
  C = sumSqrC
else:
  C = C / np.tile(sumSqrC, (NC,1,1,1))
#

data = clarray.to_device(FFT.queue[0],data)
img = clarray.zeros(FFT.queue[0],(NScan,NC,NSlice,dimY,dimX),dtype=DTYPE)

FFT.adj_NUFFT(img,data)

test = np.sum(img.get()*np.conj(C),1)
