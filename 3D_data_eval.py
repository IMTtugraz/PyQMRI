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
filenames.sort() 
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
  tr = 5000
  save_name = "IRLL"
else:
  tr = 5.38
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
    T1_ref = data[names.index('T1_ref')]
    M0_ref = data[names.index('M0_ref')]
    plot_names.append("Reference")
    plot_names.append(" ")
    NResults -=1
  else:
    if "IRLL" in files:
#      T1_tgv.append(data[names.index("T1_final")]*5500)
#      T1_tikh.append(data[names.index("T1_ref")]*5500)      
      T1_tgv.append((data[names.index('T1_final')])*tr)
#      T1_tikh.append((data[names.index('T1_ref')])*tr)       
      M0_tgv.append(data[names.index('M0_final')])
#      M0_tikh.append(data[names.index('M0_ref')])      
    else:
      scale = file['full_result'].attrs['E1_scale']  
      T1_tgv.append(-tr/np.log(data[names.index('full_result')]*scale))
#      T1_tikh.append(-tr/np.log(data[names.index('T1_ref')]*scale)) 
      M0_tgv.append(data[names.index('full_result')])
#      M0_tikh.append(data[names.index('M0_ref')])

    plot_names.append(fname[-5:].split('_')[1] + " TGV")  
    plot_names.append(fname[-5:].split('_')[0] + " Tikh")  
  file.close()
#    
for i in range(len(T1_tgv)):
  T1_tgv[i] = np.flip(T1_tgv[i],axis=0)
  M0_tgv[i] = np.flip(M0_tgv[i],axis=0)
  for j in range(T1_tgv[i].shape[0]):
    if np.sum(T1_tgv[i][j,1,...]):
      T1_tgv[i] = T1_tgv[i][j,1,...]
      M0_tgv[i] = M0_tgv[i][j,0,...]
      break
    
    
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


ax_ref = []
#T1_ref = test

for files in filenames: 
  if "ref" in files:
    if len(T1_ref.shape) == 2:
      dimz = 1
      dimy, dimx =   T1_ref.shape
      T1_ref = T1_ref[None,...]
    else:      
      dimz, dimy, dimx =   T1_ref.shape
    T1_plot.append(T1_ref)
    T1_plot.append(np.zeros((dimy,dimx)))    
    T1_ref = ((T1_ref))
    T1 = T1_ref*mask[int(z/2),...]
    T1_plot=[]
    
    T1_plot.append(np.squeeze(T1[int(dimz/2),:,:,].T))
    T1_plot.append(np.flip((T1[:,int(x/2+10),:].T),1))
    T1_plot.append([])
    T1_plot.append((T1[:,:,int(y/2-15)]))
    T1_plot.append(np.zeros((20,20)))
    T1_plot.append([])
    #  T1_min = 300
    #  T1_max = 3000
    
    fig_ref = plt.figure(figsize = (8,8))
    fig_ref.subplots_adjust(hspace=0, wspace=0)
    fig_ref.patch.set_facecolor(cm.viridis.colors[0])
    upper_bg = patches.Rectangle((0, 0), width=1, height=1, 
                                 transform=fig_ref.transFigure,      # use figure coordinates
                                 facecolor=cm.viridis.colors[0],               # define color
                                 edgecolor='none',               # remove edges
                                 zorder=0)     
    fig_ref.patches.extend([upper_bg])
    gs = gridspec.GridSpec(2,3, width_ratios=[x/z,1,1/10],height_ratios=[x/z,1])

    
    for i in range(6):
      ax_ref.append(plt.subplot(gs[i]))  
      if i==2 or i==5:
        ax_ref[i].axis('off')    
      else:
        im = ax_ref[i].imshow(np.abs(T1_plot[i]),vmin=T1_min,vmax=T1_max, aspect=1,cmap=cm.viridis)
        ax_ref[i].axis('off')
    
    #  ax[i].axis('scaled')
    ax_ref[0].set_anchor("SE")  
    ax_ref[1].set_anchor("SW")  
    ax_ref[2].set_anchor("C")  
    ax_ref[3].set_anchor("NE")  
    ax_ref[4].set_anchor("NW")  
    ax_ref[5].set_anchor("C")  
    cax = plt.subplot(gs[:,2])
    cbar = fig_ref.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=12,colors='white')
    for spine in cbar.ax.spines:
      cbar.ax.spines[spine].set_color('white')
    #fig.colorbar(im, pad=0)
    plt.show()  
    plt.savefig('/media/data/Papers/Parameter_Mapping/3D_'+save_name+'_ref.eps', format='eps', dpi=1000)

  
ax = []
for j in range(NResults): 
  T1 = T1_tgv[j]*mask
  T1_plot=[]
  
  T1_plot.append(np.squeeze(T1[int(z/2),:,:,]).T)
  T1_plot.append(np.flip((T1[:,int(x/2+10),:]).T,1))
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
  plt.savefig('/media/data/Papers/Parameter_Mapping/3D_'+save_name+'_'+str(j)+'.svg', format='svg', dpi=1000)




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
import scipy.stats as stat
from matplotlib.path import Path



import polyroi as polyroi

roi = polyroi.polyroi(T1_ref,int(z/2))
#test = np.array(roi.select_roi())
 

coord_map = np.vstack((np.repeat(np.arange(0,256,1)[None,:],256,0).flatten(), np.repeat(np.arange(0,256,1)[None,:].T,256,1).flatten())).T
#polypath = Path((test).astype(int))
#mask = polypath.contains_points(coord_map).reshape(y,x)

  
roi_num = int(input("Enter the number of desired ROIs: "))
if roi_num > 0:
  if not "Reference" in plot_names:
    T1_ref = T1_tgv[0]
  mean_TGV = []
  std_TGV = []
  col_names = []
  statistic = []
  selector = cv2.cvtColor(np.abs(T1_ref[int(z/2),:,:].T/np.max(3000)).astype(np.float32),cv2.COLOR_GRAY2BGR)
  cv2.namedWindow('ROISelector', cv2.WINDOW_NORMAL)
  for j in range(roi_num):
#    r = (cv2.selectROI('ROISelector',selector,fromCenter=False))
    r = np.array(roi.select_roi()) 
    col_names.append("ROI "+str(j+1))
    polypath = Path((r).astype(int))
    mask = polypath.contains_points(coord_map).reshape(y,x)
#    mean_TGV.append(np.abs(np.mean(T1_ref[int(z/2),int(r[0]):int(r[0]+r[2]), int(r[1]):int(r[1]+r[3])])))
#    std_TGV.append(np.abs(np.std(T1_ref[int(z/2),int(r[0]):int(r[0]+r[2]), int(r[1]):int(r[1]+r[3])])))      
    mean_TGV.append(np.abs(np.mean(T1_ref[int(z/2),mask.T])))
    std_TGV.append(np.abs(np.std(T1_ref[int(z/2),mask.T])))   
    for i in range(NResults):
      mean_TGV.append(np.abs(np.mean(T1_tgv[i][int(z/2),mask.T])))
      std_TGV.append(np.abs(np.std(T1_tgv[i][int(z/2),mask.T])))
      statistic.append(stat.ttest_ind(np.abs(T1_tgv[i][int(z/2),mask.T]).flatten(),
                                      np.abs((T1_ref[int(z/2),mask.T]).flatten())
                                      ,equal_var=False))
#      statistic.append(stat.normaltest(np.abs(T1_tgv[0][int(z/2),int(r[0]):int(r[0]+r[2]), int(r[1]):int(r[1]+r[3])]).flatten()))
#    rects = patches.Rectangle((int(r[0]),int(r[1])),
#                                   int(r[2]),int(r[3]),linewidth=3,edgecolor='r',facecolor='none')
    rects = patches.Polygon(r,linewidth=3,edgecolor='r',facecolor='none')      
    posx = int(r[0][0])
    posy = int(r[0][1]-5)
    ax_ref[0].text(posx,posy,str(j+1),color='red')
    ax_ref[0].add_patch(rects) 

  mean_TGV = np.round(pd.DataFrame(np.reshape(np.asarray(mean_TGV),(roi_num,NResults+1)).T,index=['Reference','TGV_89','TGV_55','TGV_34','TGV_21','TGV_13','TGV_08'],columns=col_names),decimals=0)
  std_TGV =  np.round(pd.DataFrame(np.reshape(np.asarray(std_TGV),(roi_num,NResults+1)).T,index=['Reference','TGV_89','TGV_55','TGV_34','TGV_21','TGV_13','TGV_08'],columns=col_names),decimals=0)
  
  f = open("3Drois.tex","w")
  f.write(mean_TGV.to_latex())
  f.write(std_TGV.to_latex())
  f.flush()
  f.close()
  
  f = open("VFA_Invivo_Satistic.tex","w")
  f.write(pd.DataFrame(statistic).to_latex())
  f.flush()
  f.close()
#  
  
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
      


mask2 = (masking.skullstrip(M0_ref))      
      
from matplotlib.colors import LogNorm, PowerNorm
hist_fig = plt.figure(figsize = (8,4))
hist_fig.subplots_adjust(hspace=0.5, wspace=0.5)
gs_hist = gridspec.GridSpec(2,3)#, width_ratios=[0.5,0.5], height_ratios=[1])
ax_hist = []

cont_fig = plt.figure(figsize = (8,4))
cont_fig.subplots_adjust(hspace=0.5, wspace=0.5)
gs_cont = gridspec.GridSpec(2,3)#, width_ratios=[0.5,0.5], height_ratios=[1])
ax_cont = []
myhist = []
mycont = []
V = np.logspace(np.log10(10),4+np.log10(5),num=3*(4+np.log10(5)-np.log10(10)+1),base=10,dtype='int')
#V2 = np.logspace(np.log10(10),4+np.log10(5),num=2*(4+np.log10(5)-np.log10(10)+1),base=10,dtype='int')/2
#V = np.vstack((V2,V1))
#V = V.flatten(order='F')
# int(np.sqrt(np.sum(mask2/z)))
for i in range(NResults):
  ax_hist.append(hist_fig.add_subplot(gs_hist[i]))
  myhist.append(ax_hist[i].hist2d(np.abs(T1_ref[mask2>0].flatten()),np.abs(T1_tgv[i][mask2>0].flatten()),
                      bins=100,range=[[0,4500],[0,4500]],norm=LogNorm()))
  data = myhist[i][0]
  data[data<0] = 0
  myhist[i][3].set_data(data.T)
  xticks = np.linspace(*ax_hist[i].get_xlim())
  ax_hist[i].plot(xticks, xticks,color='r',linestyle='--')
  cbar = hist_fig.colorbar(myhist[0][3])
  ax_hist[i].set_xlabel('Ref T1 in ms')
  ax_hist[i].set_ylabel(plot_names[2+(2*i)]+' in ms')

  ax_cont.append(cont_fig.add_subplot(gs_cont[i]))  
  mycont.append(ax_cont[i].contourf(data.T,V,extent=[myhist[i][2].min(),myhist[i][2].max(),
                               myhist[i][1].min(),myhist[i][1].max()],norm=PowerNorm(0.22),cmap='gnuplot2'))
  xticks = np.linspace(*ax_hist[i].get_xlim())
  ax_cont[i].plot(xticks, xticks,color='r',linestyle='--')
  cbar = cont_fig.colorbar(mycont[0],ax=ax_cont[i])
  ax_cont[i].set_xlabel('Ref T1 in ms')
  ax_cont[i].set_ylabel(plot_names[2+(2*i)]+' in ms')
#
#f = open('acc_test.img','wb')  
#images = np.zeros((NResults,256,256),dtype='float32')
#for i in range(NResults):
#  images[i,...] = np.abs(T1_tgv[i][int(z/2),...]).astype(np.float32)
#  
#f.write(images.tobytes())
#f.flush()
#f.close()

plt.savefig('/media/data/Papers/Parameter_Mapping/2D_Histogram'+save_name+'_'+str(j)+'.eps', format='eps', dpi=1000)