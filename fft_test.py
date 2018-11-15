#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:31:39 2018

@author: omaier
"""

import gridroutines as gridding
import gridroutines as gridding_ref
import numpy as np

import pyopencl as cl
import pyopencl.array as clarray
import time
import matplotlib.pyplot as plt
import multislice_viewer as msv

from reikna.fft import FFT, FFTShift

from tkinter import filedialog, Tk
import h5py

DTYPE = np.complex128
DTYPE_real = np.float64

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()
file = h5py.File(file)


reco_Slices = 5
dimX, dimY, NSlice = ((file.attrs['image_dimensions']).astype(int))
off = 0

data = file['real_dat'][...,int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:int(NSlice/2)+int(np.ceil(reco_Slices/2))+off,:,:].astype(DTYPE)\
   +1j*file['imag_dat'][...,int(NSlice/2)-int(np.floor((reco_Slices)/2))+off:int(NSlice/2)+int(np.ceil(reco_Slices/2))+off,:,:].astype(DTYPE)



platforms = cl.get_platforms()

ctx = []
queue = []
num_dev = len(platforms[1].get_devices())
for device in platforms[1].get_devices():
  tmp = cl.Context(
          dev_type=cl.device_type.ALL,
          properties=[(cl.context_properties.PLATFORM, platforms[1])])
  ctx.append(tmp)
  queue.append(cl.CommandQueue(tmp, device))
  queue.append(cl.CommandQueue(tmp, device))



[NScan,NC,NSlice,Nproj, N] = data.shape#(10,10,10,100,100)
kernel_width = 4

mask =  np.ones_like(np.abs(data))
mask[np.abs(data)==0] = 0
mask = np.reshape(mask,(NScan*NC*NSlice,N,N)).astype(DTYPE_real)


NUFFT = gridding.gridding(ctx[0],queue,4,2,N,NScan,(NScan*NC*NSlice,N,N),(1,2),None,None,N,1000,DTYPE,DTYPE_real,radial=0,mask=np.ones_like(mask))


test = (np.random.randn(NScan,NC,NSlice,N,N)+1j*np.random.randn(NScan,NC,NSlice,N,N)).astype(DTYPE)
test2 = (np.random.randn(NScan,NC,NSlice,N,N)+1j*np.random.randn(NScan,NC,NSlice,N,N)).astype(DTYPE)

test_cl = clarray.to_device(queue[0],test)
test2_cl = clarray.to_device(queue[0],test2)
test_res1 = clarray.zeros_like(test_cl)
test_res2 = clarray.zeros_like(test_cl)

NUFFT.fwd_NUFFT(test_res1,test_cl)
NUFFT.adj_NUFFT(test_res2,test2_cl)

test_res1 = test_res1.get()
test_res2 = test_res2.get()

a = np.vdot(test_res1.flatten(),test2.flatten())
b = np.vdot(test.flatten(),test_res2.flatten())


test = np.abs(a-b)
print("test deriv-op-adjointness streamed:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)/(N**2)))

