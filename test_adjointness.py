#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:02:51 2018

@author: omaier
"""
import numpy as np
import gradients_divergences_old as gd

DTYPE = np.complex64

x = np.ones((4,2,256,256,4))
data = np.ones((4,2,256,256,8))


xx = np.array((np.random.random_sample(np.shape(x))+1j*np.random.random_sample(np.shape(x)))).astype(DTYPE)
yy = np.array((np.random.random_sample(np.shape(data))+1j*np.random.random_sample(np.shape(data)))).astype(DTYPE)
test1 = np.zeros_like(xx)
test2 = np.zeros_like(yy)


opt.symdiv_streamed(test1,yy)
opt.symgrad_streamed(test2,xx)



#test3 = gd.bdiv_3(np.transpose(yy,(1,-1,0,2,3)))
#test4 = gd.fgrad_3(np.transpose(xx,(1,0,2,3)))

a = np.vdot(xx.flatten(),(-test1).flatten())
b = np.vdot(test2.flatten(),yy.flatten())
test = np.abs(a-b)
print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))




import pyopencl.array as clarray
#xx = clarray.to_device(queue[0],np.random.randn(12,2,256,256)+1j*np.random.randn(12,2,256,256)).astype(DTYPE)
#yy =clarray.to_device(queue[0],np.random.randn(12,2,256,256,4)+1j*np.random.randn(12,2,256,256,4)).astype(DTYPE)
xx = clarray.to_device(queue[0],np.require(np.transpose(xx,(1,0,2,3,4)),requirements='C'))
yy = clarray.to_device(queue[0],np.require(np.transpose(yy,(1,0,2,3,4)),requirements='C'))
test4 = clarray.zeros_like(yy)
test3 =  clarray.zeros_like(xx)
#
#
##opt.NUFFT[0].fwd_NUFFT(test1,xx)
##opt.NUFFT[0].adj_NUFFT(test2,yy)
opt2.sym_grad(test4,xx)
opt2.sym_bdiv(test3,yy)

#
#opt.f_grad(test1,xx)
#opt.bdiv(test2,yy)
#
#
test3 = test3.get()
test4 = test4.get()
yy = yy.get()
xx = xx.get()
#

#a = np.sum(np.conj((test1[...,0:3]))*yy[...,0:3]+2*np.conj(test1[...,3:6])*yy[...,3:6])
b = np.vdot(test4,yy)
a = np.vdot(-xx[...],test3[...])
test = np.abs(a-b)

#
print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))

