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
import cv2
import pandas as pd

from matplotlib import cm
import Compute_mask as masking

from itertools import compress

import os

root = Tk()
root.withdraw()
root.update()
root_dir = filedialog.askdirectory()
root.destroy()
filenames.reverse()

filenames = []

for root, dirs, files in os.walk(root_dir, topdown=False):
    for name in files:
        filenames.append(os.path.join(root, name))
filenames.sort()        
NResults = len(filenames) 
M0_tgv = []
M0_tikh = []
T1_tgv = []
T1_tikh = []
plot_names = []
NRef = 0
if "IRLL" in filenames[0]:
  tr = 5.5
  save_name = "IRLL"
else:
  tr = 5
  save_name = "VFA"

for files in filenames:
  full_name = files.split('/')[-1]
  fname = full_name.split('.')[0]
  ftype = full_name.split('.')[-1]
  

  file = h5py.File(files)
  names = []
  data = []
  for name in file:
    names.append(name)
    data.append(file[name][()])  
  if "ref" in files:
    T1_ref = np.flip(data[names.index('t1_ref_l2')],axis=0)
    M0_ref = data[names.index('m0_ref_l2')]
    plot_names.append("Reference")
    plot_names.append(" ")
    NRef = 1
  else:
    M0_tgv.append(data[names.index('M0_final')])
#    M0_tikh.append(data[names.index('M0_ref')])
    if "IRLL" in files:
      T1_tgv.append(data[names.index("T1_final")]*5500)
    else:
      T1_tgv.append(-tr/np.log(data[names.index('T1_final')]))
#    T1_tikh.append(-tr/np.log(data[names.index('T1_ref')])) 
    plot_names.append(fname[-5:].split('_')[0] +" spk "+ fname[-5:].split('_')[1] + " TGV")  
#    plot_names.append(fname[-5:].split('_')[0] +" spk "+ fname[-5:].split('_')[1] + " Tikh")  


dz = 1



mask = (masking.compute(M0_tgv[0]))

[z,y,x] = M0_tgv[0].shape
z = z*dz

T1_plot=[]
M0_plot=[]
T1_min = 0
T1_max = 3000
M0_min = 0
M0_max = np.abs(np.max(M0_tgv[0]))

half_im_size = 106
plot_err = False


if "Reference" in plot_names:
  dimz, dimy, dimx =   T1_ref.shape
  T1_plot.append(T1_ref[int(dimz/2),int(dimy/2)-half_im_size:int(dimy/2)+half_im_size,int(dimx/2)-half_im_size:int(dimx/2)+half_im_size].T)
  T1_plot.append(np.zeros((dimy,dimx)))
  
#  M0_plot.append(M0_ref[int(dimz/2),int(dimy/2)-half_im_size:int(dimy/2)+half_im_size,int(dimx/2)-half_im_size:int(dimx/2)+half_im_size].T)
#  M0_plot.append(np.zeros((dimy,dimx)))

T1_err=[]

T1_err_min = 0
T1_err_max = 30
mask = mask[0,int(y/2)-half_im_size:int(y/2)+half_im_size,int(x/2)-half_im_size:int(x/2)+half_im_size]

for i in range(NResults-NRef):
  slices, x, y = T1_tgv[i].shape
  T1_tgv[i] = T1_tgv[i][int(z/2)+2,int(y/2)-half_im_size:int(y/2)+half_im_size,int(x/2)-half_im_size:int(x/2)+half_im_size]*mask
  M0_tgv[i] = M0_tgv[i][int(z/2)+2,int(y/2)-half_im_size:int(y/2)+half_im_size,int(x/2)-half_im_size:int(x/2)+half_im_size]*mask
#  T1_tikh[i] = T1_tikh[i][int(z/2),int(y/2)-half_im_size:int(y/2)+half_im_size,int(x/2)-half_im_size:int(x/2)+half_im_size]*mask
#  M0_tikh[i] = M0_tikh[i][int(z/2),int(y/2)-half_im_size:int(y/2)+half_im_size,int(x/2)-half_im_size:int(x/2)+half_im_size]*mask

  T1_plot.append(np.squeeze(T1_tgv[i][...]).T)
#  T1_plot.append(np.squeeze(T1_tikh[i][...]).T)

#  M0_plot.append(np.squeeze(M0_tgv[i][...]).T)
#  M0_plot.append(np.squeeze(M0_tikh[i][...]).T)
  
  if "Reference" in plot_names:
    T1_err.append(np.squeeze(np.abs(T1_tgv[i]-T1_ref[-1,int(dimy/2)-half_im_size:int(dimy/2)+half_im_size,int(dimx/2)-half_im_size:int(dimx/2)+half_im_size])\
                             /np.abs(T1_ref[-1,int(dimy/2)-half_im_size:int(dimy/2)+half_im_size,int(dimx/2)-half_im_size:int(dimx/2)+half_im_size])).T*100)
#    T1_err.append(np.squeeze(np.abs(T1_tikh[i]-T1_ref[-1,int(dimy/2)-half_im_size:int(dimy/2)+half_im_size,int(dimx/2)-half_im_size:int(dimx/2)+half_im_size])\
#                             /np.abs(T1_ref[-1,int(dimy/2)-half_im_size:int(dimy/2)+half_im_size,int(dimx/2)-half_im_size:int(dimx/2)+half_im_size])).T*100)  
    
if "Reference" in plot_names and plot_err:  
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
    
  ax = []
  width_ratio = np.ones((1,(NResults)),dtype=int).tolist()[0]
  width_ratio.append(1/10)
  height_ratio = np.ones((1,4),dtype=int).tolist()[0]
  height_ratio.insert(2,1/10)
  gs = gridspec.GridSpec(5,NResults+1, width_ratios=width_ratio,height_ratios=height_ratio)  
  
  for i in range((NResults)):
    ax.append(plt.subplot(gs[i]))
    im = ax[i].imshow(np.abs(T1_plot[i]),vmin=T1_min,vmax=T1_max,aspect=1,cmap=cm.viridis)
    ax[i].axis('off')
  #  ax[i].set_anchor("N")   
    ax[i].set_title(plot_names[i],color='white')
  
    
  cax = plt.subplot(gs[:2,-1])
  cbar = fig.colorbar(im, cax=cax)
  cbar.ax.tick_params(labelsize=12,colors='white')
  for spine in cbar.ax.spines:
    cbar.ax.spines[spine].set_color('white')
  
  for i in range((NResults-NRef)):
    ax.append(plt.subplot(gs[i+(NResults+1)+1]))
    im = ax[i+(NResults)].imshow(np.abs(T1_err[i]),vmin=T1_err_min,vmax=T1_err_max,aspect=1,cmap=cm.inferno)
    ax[i+(NResults)].axis('off')
    ax[i+(NResults)].set_anchor("N")   
    ax[i+(NResults)].set_title(plot_names[(i+1)],color='white')
  
  
  cax = plt.subplot(gs[3:,-1])
  cbar = fig.colorbar(im, cax=cax)
  cbar.ax.tick_params(labelsize=12,colors='white')
  for spine in cbar.ax.spines:
    cbar.ax.spines[spine].set_color('white')
  plt.show()  

else:
  fig = plt.figure(figsize = (2*int(NResults),5))
  fig.subplots_adjust(hspace=0.15, wspace=0)
  upper_bg = patches.Rectangle((0, 0), width=1, height=1, 
                               transform=fig.transFigure,      # use figure coordinates
                               facecolor=cm.viridis.colors[0],               # define color
                               edgecolor='none',               # remove edges
                               zorder=0)     
  fig.patches.extend([upper_bg])

  ax = []
  width_ratio = np.ones((1,(NResults)),dtype=int).tolist()[0]
  width_ratio.append(1/20)
  height_ratio = np.ones((1,2),dtype=int).tolist()[0]
  height_ratio.insert(2,1/10)
  gs = gridspec.GridSpec(3,NResults+1, width_ratios=width_ratio,height_ratios=height_ratio)  
  
  for i in range((NResults)):
    ax.append(plt.subplot(gs[i]))
    im = ax[i].imshow(np.abs(T1_plot[i]),vmin=T1_min,vmax=T1_max,aspect=1,cmap=cm.viridis)
    ax[i].axis('off')  
    ax[i].set_title(plot_names[i],color='white')
  
    
  cax = plt.subplot(gs[:2,-1])
  cbar = fig.colorbar(im, cax=cax)
  cbar.ax.tick_params(labelsize=8,colors='white')
  for spine in cbar.ax.spines:
    cbar.ax.spines[spine].set_color('white')
  plt.show()  





  
roi_num = int(input("Enter the number of desired ROIs: "))
if roi_num > 0:
  if not "Reference" in plot_names:
    T1_ref = T1_tgv[0]
  mean_TGV = []
  mean_Tikh = []
  std_TGV = []
  std_Tikh =  []
  col_names = []
  selector = cv2.cvtColor(np.abs(T1_ref[...].T/3000).astype(np.float32),cv2.COLOR_GRAY2BGR)
  
  for j in range(roi_num):
    r = (cv2.selectROI(selector,False))
    col_names.append("ROI "+str(j+1))
    for i in range((NResults)):
      mean_TGV.append(np.abs(np.mean(T1_plot[i][int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])])))
#      mean_Tikh.append(np.abs(np.mean(T1_plot[2*i+1][int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])])))
      std_TGV.append(np.abs(np.std(T1_plot[i][int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])])))
#      std_Tikh.append(np.abs(np.std(T1_plot[2*i+1][int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])])))
    rects = patches.Rectangle((int(r[0]),int(r[1])),
                                   int(r[2]),int(r[3]),linewidth=1,edgecolor='r',facecolor='none')
    posx = int(r[1]-5)
    posy = int(r[0])
    ax[0].text(posy,posx,str(j+1),color='red')
    ax[0].add_patch(rects) 
     
  mean_TGV = np.round(pd.DataFrame(np.reshape(np.asarray(mean_TGV),(roi_num,NResults)).T,index=plot_names,columns=col_names),decimals=0)
#  mean_Tikh =  np.round(pd.DataFrame(np.reshape(np.asarray(mean_Tikh),(roi_num,NResults)).T,index=plot_names[1::2],columns=col_names),decimals=0)
  std_TGV =  np.round(pd.DataFrame(np.reshape(np.asarray(std_TGV),(roi_num,NResults)).T,index=plot_names,columns=col_names),decimals=0)
#  std_Tikh =  np.round(pd.DataFrame(np.reshape(np.asarray(std_Tikh),(roi_num,NResults)).T,index=plot_names[1::2],columns=col_names),decimals=0)
  
  f = open("test.tex","w")
  f.write(mean_TGV.to_latex())
#  f.write(mean_Tikh.to_latex())
  f.write(std_TGV.to_latex())
#  f.write(std_Tikh.to_latex())
  f.flush()
  f.close()
  
  
  from pandas.plotting import table
  
  
  fig_table = plt.figure(figsize = (16,4))
  fig_table.subplots_adjust(hspace=0, wspace=0.5)
  
  gs = gridspec.GridSpec(2,2)
  
  ax_table = []
  my_tabs = []
  my_tabs.append(mean_TGV)
#  my_tabs.append(mean_Tikh)
  my_tabs.append(std_TGV)
#  my_tabs.append(std_Tikh)
  for i in range(len(my_tabs)):
    ax_table.append(plt.subplot(gs[i]))
    table(ax_table[i],np.round(my_tabs[i],1),loc='center')
    ax_table[i].axis('off')
    if i==0:
      ax_table[i].set_title("Mean")
    else:
      ax_table[i].set_title("Standardeviation")
      

plt.savefig('/media/omaier/a3c6e764-0f9b-44b3-b888-26da7d3fe6e7/Papers/Parameter_Mapping/2Dvs3D_'+save_name+'.eps', format='eps', dpi=300)