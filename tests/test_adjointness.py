#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:02:51 2018

@author: omaier
"""
import numpy as np


DTYPE = np.complex64

x = np.ones((10,2,256,256,4))
data = np.ones((10,2,256,256,8))#,np.ones((10, 5, 10, 34, 512))


xx = np.array((np.random.random_sample(np.shape(x))+1j*np.random.random_sample(np.shape(x)))).astype(DTYPE)
##xx[...,3:] = 0
yy = np.array((np.random.random_sample(np.shape(data))+1j*np.random.random_sample(np.shape(data)))).astype(DTYPE)
##yy[...,6:] = 0
test1 = np.zeros_like(xx)
test2 = np.zeros_like(yy)


#opt.operator_forward_full(test1,yy)
#opt.operator_adjoint_full(test2,xx)



opt.symgrad_streamed(test2,xx)
opt.symdiv_streamed(test1,yy)

#test3 = gd.bdiv_3(np.transpose(yy,(1,-1,0,2,3)))
#test4 = gd.fgrad_3(np.transpose(xx,(1,0,2,3)))
#
#
#
b = np.sum(np.conj((test2[...,0:3]))*yy[...,0:3]+2*np.conj(test2[...,3:6])*yy[...,3:6])
a = np.vdot(xx[...].flatten(),(-test1[...]).flatten())
#b = np.vdot(test2.flatten(),yy.flatten())
test = np.abs(a-b)
print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))
#
#test4 = np.transpose(test4,(2,0,3,4,1))
#test3 = np.transpose(test3,(1,0,2,3))
##b = np.sum(np.conj((test4[...,0:3]))*yy[...,0:3]+2*np.conj(test4[...,3:6])*yy[...,3:6])
#a = np.vdot(xx[...].flatten(),(-test3).flatten())
#b = np.vdot(test2.flatten(),yy.flatten())
#test = np.abs(a-b)
#print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))
#



import pyopencl.array as clarray
#xx = clarray.to_device(queue[0],np.random.randn(12,2,256,256)+1j*np.random.randn(12,2,256,256)).astype(DTYPE)
#yy =clarray.to_device(queue[0],np.random.randn(12,2,256,256,4)+1j*np.random.randn(12,2,256,256,4)).astype(DTYPE)
xx = clarray.to_device(queue[0],xx)
yy = clarray.to_device(queue[0],yy)
test2 = clarray.zeros_like(yy)
test1 =  clarray.zeros_like(xx)
#
#
##opt.NUFFT[0].fwd_NUFFT(test1,xx)
##opt.NUFFT[0].adj_NUFFT(test2,yy)
opt.NUFFT.adj_NUFFT(test1,yy)
opt.NUFFT.fwd_NUFFT(test2,xx)

#
#opt.f_grad(test1,xx)
#opt.bdiv(test2,yy)
#
#
test5 = test1.get()
test6 = test2.get()
yy = yy.get()
xx = xx.get()
#

#b = np.sum(np.conj((test5[...,0:3]))*yy[...,0:3]+2*np.conj(test5[...,3:6])*yy[...,3:6])
a = np.vdot(xx.flatten(),(test5).flatten())
b = np.vdot(test6.flatten(),yy.flatten())
test = np.abs(a-b)

#
print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))

