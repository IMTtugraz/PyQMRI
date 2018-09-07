#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:31:39 2018

@author: omaier
"""

import gridroutines_slicefirst as gridding
import numpy as np

import pyopencl as cl
import pyopencl.array as clarray
import time
import matplotlib.pyplot as plt
import multislice_viewer as msv

from reikna.fft import FFT, FFTShift




platforms = cl.get_platforms()

#ctx = cl.Context(
#        dev_type=cl.device_type.ALL,
#        properties=[(cl.context_properties.PLATFORM, platforms[0])])
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

DTYPE = np.complex64
DTYPE_real = np.float32

[NScan,NC,NSlice,Nproj, N] = data.shape
G = data.shape[-1]
osr = 2
N = G/osr
kernel_width = 4

nsamples = data.shape[-1]*data.shape[-2]

gridsize=data.shape[-1]

tmp_data =np.require(((data).astype(DTYPE)).transpose(2,0,1,3,4),requirements='C')

par_slices = 1
test_gridder = []
for j in range(num_dev):
  test_gridder.append(gridding.gridding(ctx[j],queue[num_dev*j:(num_dev*j+2)],kernel_width,osr,G,NScan,(NScan*NC*par_slices,int(G),int(G)),(1,2),traj.astype(DTYPE),np.sqrt(np.abs(np.require(np.abs(dcf),DTYPE_real,requirements='C'))),gridsize,1000
                                   ,DTYPE,DTYPE_real))



outp = np.zeros((NSlice,NScan,NC,int(N),int(N)),dtype=DTYPE)



def FTH(outp,tmp_data):
#  tic_tot = time.time()
  cl_out = []
  for j in range(num_dev):
    cl_out.append(clarray.zeros(queue[2*j],(par_slices,NScan,NC,int(N),int(N)),dtype=DTYPE))
    cl_out.append(clarray.zeros(queue[2*j+1],(par_slices,NScan,NC,int(N),int(N)),dtype=DTYPE))
#  tic = time.time()
  cl_data = []
  for i in range(num_dev):
    cl_data.append(clarray.to_device(queue[2*i], np.require(tmp_data[i*par_slices:(i+1)*par_slices,...],requirements='C')))
#  toc = time.time()
#  print("Data1 to GPU: %fms" %((toc-tic)*1000))
#  tic = time.time()
  for i in range(num_dev):
    cl_out[2*i].add_event(test_gridder[i].adj_NUFFT(cl_out[2*i],cl_data[i],0))
#  toc = time.time()
#  print("Compute1 FFT GPU: %fms" %((toc-tic)*1000))
#  tic = time.time()

  for i in range(num_dev):
    cl_data.append(clarray.to_device(queue[2*i+1], np.require(tmp_data[(i+1+num_dev-1)*par_slices:(i+2+num_dev-1)*par_slices,...],requirements='C')))
#  toc = time.time()
#  print("Data2 to GPU: %fms" %((toc-tic)*1000))
#  tic = time.time()
  for i in range(num_dev):
    test_gridder[i].adj_NUFFT(cl_out[2*i+1],cl_data[num_dev+i],1)
#  toc = time.time()
#  print("Data2 FFT GPU: %fms" %((toc-tic)*1000))
  for j in range(2*num_dev,int(NSlice/(2*par_slices*num_dev)+(2*num_dev-1))):
#      tic = time.time()
      for i in range(num_dev):
        cl_out[2*i].get(ary=outp[i*par_slices+(2*num_dev*(j-2*num_dev)*par_slices):(i+1)*par_slices+(2*num_dev*(j-2*num_dev))*par_slices,...])
        cl_data[i] = clarray.to_device(queue[2*i], np.require(tmp_data[i*par_slices+(2*num_dev*(j-2*num_dev+1)*par_slices):(i+1)*par_slices+(2*num_dev*(j-2*num_dev+1))*par_slices,...],requirements='C'))
#      toc = time.time()
#      print("Data1 get and put GPU: %fms" %((toc-tic)*1000))
#      tic = time.time()
      for i in range(num_dev):
        test_gridder[i].adj_NUFFT(cl_out[2*i],cl_data[i],0)
#      toc = time.time()
#      print("Data1 FFT GPU: %fms" %((toc-tic)*1000))
#      tic = time.time()
      for i in range(num_dev):
        cl_out[2*i+1].get(ary=outp[i*par_slices+(2*num_dev*(j-2*num_dev)+num_dev)*par_slices:(i+1)*par_slices+(2*num_dev*(j-2*num_dev)+num_dev)*par_slices,...])
        cl_data[i+num_dev] = clarray.to_device(queue[2*i+1], np.require(tmp_data[i*par_slices+(2*num_dev*(j-2*num_dev+1)+num_dev)*par_slices:
          (i+1)*par_slices+(2*num_dev*(j-2*num_dev+1)+num_dev)*par_slices,...],requirements='C'))
#      toc = time.time()
#      print("Data2 get and put GPU: %fms" %((toc-tic)*1000))
#      tic = time.time()
      for i in range(num_dev):
        test_gridder[i].adj_NUFFT(cl_out[2*i+1],cl_data[i+num_dev],1)
#      toc = time.time()
#      print("Data2 FFT GPU: %fms" %((toc-tic)*1000))
#
#  tic = time.time()
  j+=1
  for i in range(num_dev):
    cl_out[2*i].get(ary=outp[i*par_slices+(2*num_dev*(j-2*num_dev)*par_slices):(i+1)*par_slices+(2*num_dev*(j-2*num_dev))*par_slices,...])
    cl_out[2*i+1].get(ary=outp[i*par_slices+(2*num_dev*(j-2*num_dev)+num_dev)*par_slices:(i+1)*par_slices+(2*num_dev*(j-2*num_dev)+num_dev)*par_slices,...])
#  toc = time.time()
#  print("Get last GPU: %fms" %((toc-tic)*1000))
#  toc_tot = time.time()
#  print("Total transform time on GPU: %fms" %((toc_tot-tic_tot)*1000))



def FT(outp,tmp_data):
#  tic_tot = time.time()
  cl_out = []
  for j in range(num_dev):
    cl_out.append(clarray.zeros(queue[2*j],(par_slices,NScan,NC,Nproj,G),dtype=DTYPE))
    cl_out.append(clarray.zeros(queue[2*j+1],(par_slices,NScan,NC,Nproj,G),dtype=DTYPE))
#  tic = time.time()
  cl_data = []
  for i in range(num_dev):
    cl_data.append(clarray.to_device(queue[2*i], np.require(tmp_data[i*par_slices:(i+1)*par_slices,...],requirements='C')))
#  toc = time.time()
#  print("Data1 to GPU: %fms" %((toc-tic)*1000))
#  tic = time.time()
  for i in range(num_dev):
    test_gridder[i].fwd_NUFFT(cl_out[2*i],cl_data[i],0)
#  toc = time.time()
#  print("Compute1 FFT GPU: %fms" %((toc-tic)*1000))
#  tic = time.time()
  for i in range(num_dev):
    cl_data.append(clarray.to_device(queue[2*i+1], np.require(tmp_data[(i+1+num_dev-1)*par_slices:(i+2+num_dev-1)*par_slices,...],requirements='C')))
#  toc = time.time()
#  print("Data2 to GPU: %fms" %((toc-tic)*1000))
#  tic = time.time()
  for i in range(num_dev):
    test_gridder[i].fwd_NUFFT(cl_out[2*i+1],cl_data[num_dev+i],1)
#  toc = time.time()
#  print("Data2 FFT GPU: %fms" %((toc-tic)*1000))
  for j in range(2*num_dev,int(NSlice/(2*par_slices*num_dev)+(2*num_dev-1))):
#      tic = time.time()
      for i in range(num_dev):
        cl_out[2*i].get(ary=outp[i*par_slices+(2*num_dev*(j-2*num_dev)*par_slices):(i+1)*par_slices+(2*num_dev*(j-2*num_dev))*par_slices,...])
        cl_data[i] = clarray.to_device(queue[2*i], np.require(tmp_data[i*par_slices+(2*num_dev*(j-2*num_dev+1)*par_slices):(i+1)*par_slices+(2*num_dev*(j-2*num_dev+1))*par_slices,...],requirements='C'))
#      toc = time.time()
#      print("Data1 get and put GPU: %fms" %((toc-tic)*1000))
#      tic = time.time()
      for i in range(num_dev):
        test_gridder[i].fwd_NUFFT(cl_out[2*i],cl_data[i],0)
#      toc = time.time()
#      print("Data1 FFT GPU: %fms" %((toc-tic)*1000))
#      tic = time.time()
      for i in range(num_dev):
        cl_out[2*i+1].get(ary=outp[i*par_slices+(2*num_dev*(j-2*num_dev)+num_dev)*par_slices:(i+1)*par_slices+(2*num_dev*(j-2*num_dev)+num_dev)*par_slices,...])
        cl_data[i+num_dev] = clarray.to_device(queue[2*i+1], np.require(tmp_data[i*par_slices+(2*num_dev*(j-2*num_dev+1)+num_dev)*par_slices:
          (i+1)*par_slices+(2*num_dev*(j-2*num_dev+1)+num_dev)*par_slices,...],requirements='C'))
#      toc = time.time()
#      print("Data2 get and put GPU: %fms" %((toc-tic)*1000))
#      tic = time.time()
      for i in range(num_dev):
        test_gridder[i].fwd_NUFFT(cl_out[2*i+1],cl_data[i+num_dev],1)
#      toc = time.time()
#      print("Data2 FFT GPU: %fms" %((toc-tic)*1000))
#
#  tic = time.time()
  j+=1
  for i in range(num_dev):
    cl_out[2*i].get(ary=outp[i*par_slices+(2*num_dev*(j-2*num_dev)*par_slices):(i+1)*par_slices+(2*num_dev*(j-2*num_dev))*par_slices,...])
    cl_out[2*i+1].get(ary=outp[i*par_slices+(2*num_dev*(j-2*num_dev)+num_dev)*par_slices:(i+1)*par_slices+(2*num_dev*(j-2*num_dev)+num_dev)*par_slices,...])
#  toc = time.time()
#  print("Get last GPU: %fms" %((toc-tic)*1000))
#  toc_tot = time.time()
#  print("Total transform time on GPU: %fms" %((toc_tot-tic_tot)*1000))
#print("OCL time grid: %fms" %((toc-tic)*1000))
#test = cl_out_r.get()+1j*cl_out_i.get()
#test = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(test,axes=(-2,-1)),norm='ortho'),axes=(-2,-1))
#test = test[:,:,:,int(G/2-N/2):int(G/2+N/2),int(G/2-N/2):int(G/2+N/2)]
#test2 = np.sum(test*np.conj(np.transpose(par.C,(0,1,3,2))),1)
#msv.imshow(np.abs(np.squeeze(test2)))
#msv.imshow(np.angle(np.squeeze(test2)))

#tic = time.time()
#test_gridder.cl_invgridlut(cl_out,test,cl_traj,gridsize,kernel_width/2,cl_dcf)
#toc = time.time()
#print("OCL time regrid: %fms" %((toc-tic)*1000))



cl_traj = clarray.to_device(test_gridder[0].queue[0],traj.astype(DTYPE))

test_data = np.reshape(np.random.randn((NScan*NC*NSlice*data.shape[-2]*gridsize))+1j*np.random.randn((NScan*NC*NSlice*data.shape[-2]*gridsize)),(NSlice,NScan,NC,data.shape[-2],gridsize)).astype(DTYPE)
test_img = np.reshape(np.random.randn(int(NScan*NC*NSlice*gridsize**2/4))+1j*np.random.randn(int(NScan*NC*NSlice*gridsize**2/4)),(NSlice,NScan,NC,int(gridsize/2),int(gridsize/2))).astype(DTYPE)


#data_test = clarray.to_device(test_gridder[0].queue[0],test_data.astype(DTYPE))


#cl_out = clarray.zeros(queue[0],(par_slices,NScan,NC,int(N),int(N)),dtype=DTYPE)
outp1 = np.zeros_like(test_img)
outp2 = np.zeros_like(test_data)

FTH(outp1,test_data)
FT(outp2,test_img)


#cl_out = clarray.zeros(queue[0],(par_slices,NScan,NC,data.shape[-2],gridsize),dtype=DTYPE)
#img = clarray.to_device(queue[0],np.require(test_img,dtype=DTYPE,requirements='C'))



a = np.vdot(outp1.flatten(),test_img.flatten())
b = np.vdot(test_data.flatten(),outp2.flatten())

test = np.abs(a-b)
print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)/(gridsize**2)))


