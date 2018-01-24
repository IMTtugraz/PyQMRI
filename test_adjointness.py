#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:02:51 2018

@author: omaier
"""
import numpy as np
import gradients_divergences_old as gd

DTYPE = np.complex128

x = np.ones((2,256,256))
data = np.ones((2,2,256,256))

scale = np.ones((2,1,1),dtype=DTYPE)

xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
yy = np.random.random_sample(np.shape(data)).astype(DTYPE)
a = np.vdot(xx.flatten(),(-gd.bdiv_1(yy)/scale).flatten())
b = np.vdot(gd.fgrad_1(xx/scale).flatten(),yy.flatten())
test = np.abs(a-b)
print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))