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
import pandas as pd

import os
from matplotlib import cm
import Compute_mask as masking
import multislice_viewer as msv

import cv2

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
NResults = len(filenames)
M0_tgv = []
M0_tikh = []
T1_tgv = []
T1_tikh = []
plot_names = []


#full_name = filenames[0].split('/')[-1]
#fname = full_name.split('.')[0]
#ftype = full_name.split('.')[-1]
#
#plot_names.append("Reference")
#plot_names.append(" ")


#file = h5py.File(filenames[0])
#names = []
#data = []
#for name in file:
#  names.append(name)
#  data.append(file[name][()])
#  
#M0_ref = data[0]
#T1_ref = data[1]

if "IRLL" in filenames[0]:
  tr = 1000
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
    T1_ref = data[names.index('t1_corr')]
    M0_ref = data[names.index('m0_corr')]
    plot_names.append("Reference")
    plot_names.append(" ")
    NResults -=1
  else:
    M0_tgv.append(data[names.index('M0_final')])
#    M0_tikh.append(data[names.index('M0_ref')])
    T1_tgv.append(-tr/np.log(data[names.index('T1_final')]))
#    T1_tikh.append(-1000/np.log(data[names.index('T1_ref')]))
    plot_names.append(fname[-2:] + " Spokes TGV")  
    plot_names.append(fname[-2:] + " Spokes Tikh")    
dz = 1



mask = (masking.compute(M0_tgv[0]))

[z,y,x] = M0_tgv[0].shape
z = z*dz

T1_plot=[]
M0_plot=[]
T1_min = 300
T1_max = 3000
M0_min = 0
M0_max = np.abs(np.max(M0_tgv[0]))



for files in filenames: 
  if "ref" in files:
    fig = plt.figure(figsize = (8,8))
    ax_ref = (fig.add_axes([0,0,1,1]))
#    T1_min = 300
#    T1_max = 3000 
    ax_ref.imshow(np.abs(T1_ref.T),vmin=T1_min,vmax=T1_max,aspect=1,cmap=cm.viridis)
  
ax = []
for j in range(NResults): 
  T1 = T1_tgv[j]*mask
  T1_plot=[]
  
  T1_plot.append(np.squeeze(T1[int(z/2),:,:,]).T)
  T1_plot.append(np.flip((T1[:,int(x/2+3),:]).T,1))
  T1_plot.append([])
  T1_plot.append((T1[:,:,int(y/2-15)]))
  T1_plot.append(np.zeros((20,20)))
  T1_plot.append([])
#  T1_min = 300
#  T1_max = 3000
  
  fig = plt.figure(figsize = (8,8))
  fig.subplots_adjust(hspace=0, wspace=0)
  fig.patch.set_facecolor(cm.viridis.colors[0])
  upper_bg = patches.Rectangle((0, 0), width=1, height=1, 
                               transform=fig.transFigure,      # use figure coordinates
                               facecolor=cm.viridis.colors[0],               # define color
                               edgecolor='none',               # remove edges
                               zorder=0)     
  fig.patches.extend([upper_bg])
  gs = gridspec.GridSpec(2,3, width_ratios=[x/z,1,1/10],height_ratios=[x/z,1])
  #ax = [fig.add_subplot(2,2,i+1) for i in range(4)]

#  plot_extend_x = [1,z/x,1/10,1,z/x,1/10]
#  plot_extend_y = [1,1,1,z/x,z/x,z/x]
#  extent=[0,plot_extend_x[i],0,plot_extend_y[i]]
  
  for i in range(6):
    ax.append(plt.subplot(gs[i]))  
    if i==2 or i==5:
      ax[i+j*6].axis('off')    
    else:
      im = ax[i+j*6].imshow(np.abs(T1_plot[i]),vmin=T1_min,vmax=T1_max, aspect=1,cmap=cm.viridis)
      ax[i+j*6].axis('off')

  #  ax[i].axis('scaled')
  ax[0+j*6].set_anchor("SE")  
  ax[1+j*6].set_anchor("SW")  
  ax[2+j*6].set_anchor("C")  
  ax[3+j*6].set_anchor("NE")  
  ax[4+j*6].set_anchor("NW")  
  ax[5+j*6].set_anchor("C")  
  cax = plt.subplot(gs[:,2])
  cbar = fig.colorbar(im, cax=cax)
  cbar.ax.tick_params(labelsize=12,colors='white')
  for spine in cbar.ax.spines:
    cbar.ax.spines[spine].set_color('white')
  #fig.colorbar(im, pad=0)
  plt.show()  



plt.savefig('/media/omaier/a3c6e764-0f9b-44b3-b888-26da7d3fe6e7/Papers/Parameter_Mapping/3D_'+save_name+'.eps', format='eps', dpi=1000)
#
#if int(input("Enter 0 for no ref or 1 for ref: ")):
#  T1_ref = np.flip(T1_ref,axis=0)
#  T1_plot_ref=[]
#  for j in range(NResults):
#    T1 = np.squeeze(T1_tgv[j][int(z/2),6:-6,6:-6]*mask[int(z/2),6:-6,6:-6])
#    T1_plot_ref.append(np.squeeze(T1_ref*mask[int(z/2),6:-6,6:-6]).T)
#    T1_plot_ref.append(np.squeeze(T1).T)
#    T1_plot_ref.append((((T1_ref*mask[int(z/2),6:-6,6:-6])-T1)/((T1_ref*mask[int(z/2),6:-6,6:-6]))*100).T)
#    T1_plot_ref.append([])
#    T1_min = 300
#    T1_max = 3000
#    
#    
#    fig = plt.figure(figsize = (12,4))
#    fig.subplots_adjust(hspace=0, wspace=0)
#    fig.patch.set_facecolor(cm.viridis.colors[0])
#    gs = gridspec.GridSpec(1,4, width_ratios=[1,1,1,1/10],height_ratios=[1])
#    #ax = [fig.add_subplot(2,2,i+1) for i in range(4)]
#    ax_diff = []
#    
#    for i in range(4):
#      ax_diff.append(plt.subplot(gs[i]))  
#      if i==3 or i==6:
#        ax_diff[i].axis('off')    
#      else:
#        im = ax_diff[i].imshow(np.abs(T1_plot_ref[i+j*4]),vmin=T1_min,vmax=T1_max,aspect=1,cmap=cm.viridis)
#        ax_diff[i].axis('off')
#  
#  
#    cax = plt.subplot(gs[:,3])
#    cbar = fig.colorbar(im, cax=cax)
#    cbar.ax.tick_params(labelsize=12,colors='white')
#    for spine in cbar.ax.spines:
#      cbar.ax.spines[spine].set_color('white')
#    #fig.colorbar(im, pad=0)
#    plt.show()  
  
offset = 0

  
roi_num = int(input("Enter the number of desired ROIs: "))
if roi_num > 0:
  mean_TGV = []
  std_TGV = []
  col_names = []
  selector = cv2.cvtColor(np.abs(T1_ref.T/np.max(T1_ref)).astype(np.float32),cv2.COLOR_GRAY2BGR)
  for j in range(roi_num):
    r = (cv2.selectROI(selector))
    col_names.append("ROI "+str(j+1))
    mean_TGV.append(np.abs(np.mean(T1_ref[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])])))
    std_TGV.append(np.abs(np.std(T1_ref[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])])))   
    for i in range(NResults):
      mean_TGV.append(np.abs(np.mean(T1_tgv[i][int(z/2),int(r[1]+offset):int(r[1]+r[3]+offset), int(r[0]):int(r[0]+r[2]+offset)])))
      std_TGV.append(np.abs(np.std(T1_tgv[i][int(z/2),int(r[1]+offset):int(r[1]+r[3]+offset), int(r[0]+offset):int(r[0]+r[2]+offset)])))
    rects = patches.Rectangle((int(r[0]),int(r[1])),
                                   int(r[2]),int(r[3]),linewidth=3,edgecolor='r',facecolor='none')
    posx = int(r[1]-5)
    posy = int(r[0])
    ax_ref.text(posy,posx,str(j+1),color='red')
    ax_ref.add_patch(rects) 
     
  mean_TGV = np.round(pd.DataFrame(np.reshape(np.asarray(mean_TGV),(roi_num,NResults+1)).T,index=['Reference','5s_new','5s','8s','8s_new'],columns=col_names),decimals=0)
  std_TGV =  np.round(pd.DataFrame(np.reshape(np.asarray(std_TGV),(roi_num,NResults+1)).T,index=['Reference','5s_new','5s','8s','8s_new'],columns=col_names),decimals=0)
  
  f = open("test.tex","w")
  f.write(mean_TGV.to_latex())
  f.write(std_TGV.to_latex())
  f.flush()
  f.close()
  
  
  from pandas.plotting import table
  
  
  fig_table = plt.figure(figsize = (16,4))
  fig_table.subplots_adjust(hspace=0, wspace=0.5)
  
  gs = gridspec.GridSpec(2,1)
  
  ax_table = []
  my_tabs = []
  my_tabs.append(mean_TGV)
  my_tabs.append(std_TGV)
  for i in range(2):
    ax_table.append(plt.subplot(gs[i]))
    table(ax_table[i],np.round(my_tabs[i],1),loc='center')
    ax_table[i].axis('off')
    if i<1:
      ax_table[i].set_title("Mean")
    else:
      ax_table[i].set_title("Standardeviation")