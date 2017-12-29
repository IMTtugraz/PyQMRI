#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:07:27 2017

@author: omaier
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

from tkinter import filedialog
from tkinter import Tk

import h5py  

from matplotlib import cm
import Compute_mask as masking

from itertools import compress

import os

root = Tk()
root.withdraw()
root.update()
root_dir = filedialog.askdirectory()
root.destroy()

filenames = []

for root, dirs, files in os.walk(root_dir, topdown=False):
    for name in files:
        filenames.append(os.path.join(root, name))
        
filenames.reverse()
NResults = len(filenames)-1    
M0_tgv = []
M0_tikh = []
T1_tgv = []
T1_tikh = []
plot_names = []


full_name = filenames[0].split('/')[-1]
fname = full_name.split('.')[0]
ftype = full_name.split('.')[-1]

plot_names.append("Reference")
plot_names.append(" ")


file = h5py.File(filenames[0])
names = []
data = []
for name in file:
  names.append(name)
  data.append(file[name][()])
  
M0_ref = data[0]
T1_ref = data[1]

for files in filenames[1:]:
  full_name = files.split('/')[-1]
  fname = full_name.split('.')[0]
  ftype = full_name.split('.')[-1]
  
  plot_names.append(fname[-2:] + " Spokes TGV")  
  plot_names.append(fname[-2:] + " Spokes Tikh")  



  file = h5py.File(files)
  names = []
  data = []
  for name in file:
    names.append(name)
    data.append(file[name][()])
  M0_tgv.append(data[0])
  M0_tikh.append(data[2])
  T1_tgv.append(-5/np.log(data[3]))
  T1_tikh.append(-5/np.log(data[5]))
  



dz = 1



mask = np.ones_like(masking.compute(M0_tgv[0]))

[z,y,x] = M0_ref.shape
z = z*dz

T1_plot=[]
M0_plot=[]
T1_min = 0
T1_max = 1500
M0_min = 0
M0_max = np.abs(np.max(M0_ref))

T1_plot.append(T1_ref[int(z/2),:,:,].T)
T1_plot.append(np.zeros((y,x)))

M0_plot.append(M0_ref[int(z/2),:,:,].T)
M0_plot.append(np.zeros((y,x)))

T1_err=[]

T1_err_min = 0
T1_err_max = 30

for i in range(NResults):
  T1_tgv[i] = T1_tgv[i]*mask
  M0_tgv[i] = M0_tgv[i]*mask
  T1_tikh[i] = T1_tikh[i]*mask
  M0_tikh[i] = M0_tikh[i]*mask

  T1_plot.append(np.squeeze(T1_tgv[i][int(z/2),:,:]).T)
  T1_plot.append(np.squeeze(T1_tikh[i][int(z/2),:,:]).T)

  M0_plot.append(np.squeeze(M0_tgv[i][int(z/2),:,:]).T)
  M0_plot.append(np.squeeze(M0_tikh[i][int(z/2),:,:]).T)
  
  T1_err.append(np.squeeze(np.abs(T1_tgv[i][int(z/2),:,:]-T1_ref[int(z/2),:,:])/np.abs(T1_ref[int(z/2),:,:])).T*100)
  T1_err.append(np.squeeze(np.abs(T1_tikh[i][int(z/2),:,:]-T1_ref[int(z/2),:,:])/np.abs(T1_ref[int(z/2),:,:])).T*100)  

  
  


fig = plt.figure(figsize = (8,8))
fig.subplots_adjust(hspace=0.15, wspace=0)
upper_bg = patches.Rectangle((0, 0.5), width=1, height=0.5, 
                             transform=fig.transFigure,      # use figure coordinates
                             facecolor=cm.viridis.colors[0],               # define color
                             edgecolor='none',               # remove edges
                             zorder=0)                       # send it to the background
lower_bg = patches.Rectangle((0, 0), width=1.0, height=0.5, 
                             transform=fig.transFigure,      # use figure coordinates
                             facecolor=cm.inferno.colors[0],              # define color
                             edgecolor='none',               # remove edges
                             zorder=0)  
fig.patches.extend([upper_bg, lower_bg])

width_ratio = np.ones((1,(NResults+1)),dtype=int).tolist()[0]
width_ratio.append(1/10)
height_ratio = np.ones((1,4),dtype=int).tolist()[0]
height_ratio.insert(2,1/10)
gs = gridspec.GridSpec(5,NResults+2, width_ratios=width_ratio,height_ratios=height_ratio)





ax = []


for i in range((NResults+1)):
  ax.append(plt.subplot(gs[i]))
  im = ax[i].imshow(np.abs(T1_plot[2*i]),vmin=T1_min,vmax=T1_max,aspect=1,cmap=cm.viridis)
  ax[i].axis('off')
#  ax[i].set_anchor("N")   
  ax[i].set_title(plot_names[2*i],color='white')

for i in range((NResults+1)):
  ax.append(plt.subplot(gs[i+NResults+2]))
  im = ax[i+NResults+1].imshow(np.abs(T1_plot[2*i+1]),vmin=T1_min,vmax=T1_max,aspect=1,cmap=cm.viridis)
  ax[i+NResults+1].axis('off')
#  ax[i+NResults].set_anchor("S")    
  ax[i+NResults+1].set_title(plot_names[2*i+1],color='white')
  
cax = plt.subplot(gs[:2,-1])
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=12,colors='white')
for spine in cbar.ax.spines:
  cbar.ax.spines[spine].set_color('white')

for i in range((NResults)):
  ax.append(plt.subplot(gs[i+3*(NResults+2)+1]))
  im = ax[i+2*(NResults+1)].imshow(np.abs(T1_err[2*i]),vmin=T1_err_min,vmax=T1_err_max,aspect=1,cmap=cm.inferno)
  ax[i+2*(NResults+1)].axis('off')
  ax[i+2*(NResults+1)].set_anchor("N")   
  ax[i+2*(NResults+1)].set_title(plot_names[2*(i+1)],color='white')

for i in range((NResults)):
  ax.append(plt.subplot(gs[i+4*(NResults+2)+1]))
  im = ax[i+3*(NResults+1)-1].imshow(np.abs(T1_err[2*i+1]),vmin=T1_err_min,vmax=T1_err_max,aspect=1,cmap=cm.inferno)
  ax[i+3*(NResults+1)-1].axis('off')
  ax[i+3*(NResults+1)-1].set_anchor("S")   
  ax[i+3*(NResults+1)-1].set_title(plot_names[2*(i+1)+1],color='white')

cax = plt.subplot(gs[3:,-1])
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=12,colors='white')
for spine in cbar.ax.spines:
  cbar.ax.spines[spine].set_color('white')
plt.show()  
  
  
