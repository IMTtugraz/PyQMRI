#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:31:39 2018

@author: omaier
"""

import gridroutines as gridding
import numpy as np

import pyopencl as cl
import pyopencl.array as clarray
import time
import matplotlib.pyplot as plt
import multislice_viewer as msv

from reikna.fft import FFT, FFTShift

platforms = cl.get_platforms()

ctx = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[1])])

queue = cl.CommandQueue(ctx)

DTYPE = np.complex64
DTYPE_real = np.float32

[NScan,NC,NSlice,Nproj, N] = data.shape
G = data.shape[-1]
osr = 2
N = G/osr
kernel_width = 4

nsamples = data.shape[-1]*data.shape[-2]

gridsize=data.shape[-1]

tmp_data =(data).astype(DTYPE)


cl_data = clarray.to_device(queue, tmp_data)

fft_size = (NScan,NC,NSlice,int(G),int(G))


test_gridder = gridding.gridding(ctx,queue,kernel_width,osr,G,(NScan,NC,NSlice,int(G),int(G)),(3,4),traj.astype(DTYPE),np.sqrt(np.abs(np.require(np.abs(dcf),DTYPE_real,requirements='C'))),gridsize,1000
                                 ,DTYPE,DTYPE_real)



cl_out = clarray.zeros(queue,(NScan,NC,NSlice,int(N),int(N)),dtype=DTYPE)

tic = time.time()
test_gridder.adj_NUFFT(cl_out,cl_data)
toc = time.time()

print("OCL time grid: %fms" %((toc-tic)*1000))

test = cl_out.get()
#msv.imshow(np.abs(test[:,:,0,...]))


#print("OCL time grid: %fms" %((toc-tic)*1000))
#test = cl_out_r.get()+1j*cl_out_i.get()
#test = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(test,axes=(-2,-1)),norm='ortho'),axes=(-2,-1))
#test = test[:,:,:,int(G/2-N/2):int(G/2+N/2),int(G/2-N/2):int(G/2+N/2)]
test2 = np.sum(test*np.conj(np.transpose(par.C,(0,1,3,2))),1)
msv.imshow(np.abs(np.squeeze(test2)))
msv.imshow(np.angle(np.squeeze(test2)))

#tic = time.time()
#test_gridder.cl_invgridlut(cl_out,test,cl_traj,gridsize,kernel_width/2,cl_dcf)
#toc = time.time()
#print("OCL time regrid: %fms" %((toc-tic)*1000))




cl_traj = clarray.to_device(test_gridder.queue,traj.astype(DTYPE))

test_data = np.reshape(np.random.randn((NScan*NC*NSlice*data.shape[-2]*gridsize))+1j*np.random.randn((NScan*NC*NSlice*data.shape[-2]*gridsize)),(NScan,NC,NSlice,data.shape[-2],gridsize))
test_img = np.reshape(np.random.randn(int(NScan*NC*NSlice*gridsize**2/4))+1j*np.random.randn(int(NScan*NC*NSlice*gridsize**2/4)),(NScan,NC,NSlice,int(gridsize/2),int(gridsize/2)))


data_test = clarray.to_device(test_gridder.queue,test_data.astype(DTYPE))


cl_out = clarray.zeros(queue,(NScan,NC,NSlice,int(N),int(N)),dtype=DTYPE)
tic = time.time()
test_gridder.adj_NUFFT(cl_out,data_test)
toc = time.time()
test1 = cl_out.get()
print("OCL time adj: %fms" %((toc-tic)*1000))



cl_out = clarray.zeros(queue,(NScan,NC,NSlice,data.shape[-2],gridsize),dtype=DTYPE)
img = clarray.to_device(queue,np.require(test_img,dtype=DTYPE,requirements='C'))
tic = time.time()
test_gridder.fwd_NUFFT(cl_out,img)
toc = time.time()
print("OCL time fwd: %fms" %((toc-tic)*1000))



a = np.vdot(test1.flatten(),test_img.flatten())
b = np.vdot(test_data.flatten(),cl_out.get().flatten())

test = np.abs(a-b)
print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)/(gridsize**2)))


