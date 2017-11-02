#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:07:27 2017

@author: omaier
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tkinter import filedialog
from tkinter import Tk

import h5py  
import scipy.io as sio
from matplotlib import cm
import Compute_mask as masking

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

name = file.split('/')[-1]
fname = name.split('.')[0]
ftype = name.split('.')[-1]

if ftype == 'h5':
  file = h5py.File(file)
  names = []
  data = []
  for name in file:
    names.append(name)
    data.append(file[name][()])
  M0 = data[7]
  T1 = (data[9])
elif ftype =='mat':
  result = sio.loadmat(file)
  result = result['result']
  M0 = result[-1,0,...]
  T1 = (result[-1,1,...])*10000
else:
  print("Unsupported format")  

dz = 1
mask = masking.compute(M0)

[z,y,x] = T1.shape
z = z*dz

T1 = T1*mask
M0 = M0*mask

T1_plot=[]

T1_plot.append(np.squeeze(T1[20,:,:,]).T)
T1_plot.append(np.flip(np.squeeze(T1[:,120,:]).T,1))
T1_plot.append([])
T1_plot.append(np.squeeze(T1[:,:,120]))
T1_plot.append(np.zeros((20,20)))
T1_plot.append([])
T1_min = 300
T1_max = 3000

M0_plot=[]

M0_plot.append(np.squeeze(M0[20,:,:,]).T)
M0_plot.append(np.flip(np.squeeze(M0[:,120,:]).T,1))
M0_plot.append([])
M0_plot.append(np.squeeze(M0[:,:,120]))
M0_plot.append(np.zeros((20,20)))
M0_plot.append([])
M0_min = 0
M0_max = np.abs(np.max(M0))

fig = plt.figure(figsize = (8,8))
fig.subplots_adjust(hspace=0, wspace=0)
fig.patch.set_facecolor(cm.viridis.colors[0])
gs = gridspec.GridSpec(2,3, width_ratios=[x/z,1,1/10],height_ratios=[x/z,1])
#ax = [fig.add_subplot(2,2,i+1) for i in range(4)]
ax = []
plot_extend_x = [1,z/x,1/10,1,z/x,1/10]
plot_extend_y = [1,1,1,z/x,z/x,z/x]

for i in range(6):
  ax.append(plt.subplot(gs[i]))  
  if i==2 or i==5:
    ax[i].axis('off')    
  else:
    im = ax[i].imshow(np.abs(T1_plot[i]),vmin=T1_min,vmax=T1_max, extent=[0,plot_extend_x[i],0,plot_extend_y[i]],aspect=1,cmap=cm.viridis)
    ax[i].axis('off')

    
    
#  ax[i].axis('scaled')
ax[0].set_anchor("SE")  
ax[1].set_anchor("SW")  
ax[2].set_anchor("C")  
ax[3].set_anchor("NE")  
ax[4].set_anchor("NW")  
ax[5].set_anchor("C")  
cax = plt.subplot(gs[:,2])
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=12,colors='white')
for spine in cbar.ax.spines:
  cbar.ax.spines[spine].set_color('white')
#fig.colorbar(im, pad=0)
plt.show()  


fig = plt.figure(figsize = (8,8))
fig.subplots_adjust(hspace=0, wspace=0)
fig.patch.set_facecolor(cm.viridis.colors[0])
gs = gridspec.GridSpec(2,3, width_ratios=[x/z,1,1/10],height_ratios=[x/z,1])
#ax = [fig.add_subplot(2,2,i+1) for i in range(4)]
ax = []
plot_extend_x = [1,z/x,1/10,1,z/x,1/10]
plot_extend_y = [1,1,1,z/x,z/x,z/x]

for i in range(6):
  ax.append(plt.subplot(gs[i]))  
  if i==2 or i==5:
    ax[i].axis('off')    
  else:
    im = ax[i].imshow(np.abs(M0_plot[i]),vmin=M0_min,vmax=M0_max, extent=[0,plot_extend_x[i],0,plot_extend_y[i]],aspect=1,cmap=cm.viridis)
    ax[i].axis('off')

    
    
#  ax[i].axis('scaled')
ax[0].set_anchor("SE")  
ax[1].set_anchor("SW")  
ax[2].set_anchor("C")  
ax[3].set_anchor("NE")  
ax[4].set_anchor("NW")  
ax[5].set_anchor("C")  
cax = plt.subplot(gs[:,2])
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=12,colors='white')
for spine in cbar.ax.spines:
  cbar.ax.spines[spine].set_color('white')
#fig.colorbar(im, pad=0)
plt.show()  






T1_plot=[]

T1_plot.append(np.squeeze(T1[20,6:-6,6:-6]).T)
T1_plot.append(np.squeeze(T1_ref).T)
T1_plot.append([])
T1_min = 300
T1_max = 3000


fig = plt.figure(figsize = (8,4))
fig.subplots_adjust(hspace=0, wspace=0)
fig.patch.set_facecolor(cm.viridis.colors[0])
gs = gridspec.GridSpec(1,3, width_ratios=[1,1,1/10],height_ratios=[1])
#ax = [fig.add_subplot(2,2,i+1) for i in range(4)]
ax = []
plot_extend_x = [1,1,1/10]
plot_extend_y = [1,1,1]

for i in range(3):
  ax.append(plt.subplot(gs[i]))  
  if i==2 or i==5:
    ax[i].axis('off')    
  else:
    im = ax[i].imshow(np.abs(T1_plot[i]),vmin=T1_min,vmax=T1_max, extent=[0,plot_extend_x[i],0,plot_extend_y[i]],aspect=1,cmap=cm.viridis)
    ax[i].axis('off')

    
    
#  ax[i].axis('scaled')
ax[0].set_anchor("SE")  
ax[1].set_anchor("SW")  
ax[2].set_anchor("C")  
ax[3].set_anchor("NE")  
ax[4].set_anchor("NW")  
ax[5].set_anchor("C")  
cax = plt.subplot(gs[:,2])
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=12,colors='white')
for spine in cbar.ax.spines:
  cbar.ax.spines[spine].set_color('white')
#fig.colorbar(im, pad=0)
plt.show()  