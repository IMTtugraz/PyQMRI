#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:02:45 2018

@author: omaier
"""

import VFA_model as model
import numpy as np
import multislice_viewer as msv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

fa = np.arange(1,30,0.1)/180*np.pi
fa_corr = 1
TR = 5
images = np.ones((len(fa),1,2,2))
NSlice = 1
Nproj = 1

test = model.VFA_Model(fa,fa_corr,TR,images,0,NSlice,Nproj)

M0 = np.ones((1,2,2))
T1 = np.exp(-TR/np.reshape(np.arange(100,2000,1900/(2*2)),[1,2,2]))
inp = np.array((M0,T1))
test.M0_sc = 1
test.T1_sc = 1

test2 = test.execute_forward_2D(inp,0)
test2 = np.reshape(test2,[len(fa),2*2])
T1 = (np.round(np.reshape(np.arange(100,2000,1900/(2*2)),[1,2,2]))).astype(np.int32)
fa = np.arange(1,30,0.1)
ax = plt.plot(fa,test2,linewidth=4)
plt.title('Simulated signal evolution for VFA')


for j in range(len(ax)):
   ax[j].set_label('T1 '+str(T1.flatten()[j]) +' ms')
   ax[j].convert_xunits(fa)
plt.legend(loc='best')
plt.xlabel('Fa in deg')
plt.ylabel('Signal in a.u.')
plt.tight_layout()
plt.savefig('sim_vfa',dpi=300)