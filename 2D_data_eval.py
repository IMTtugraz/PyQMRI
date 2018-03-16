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
full_res = []
NRef = 0
if "IRLL" in filenames[0]:
  tr = 3000
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
    T1_ref = np.flip(data[names.index('T1_referenz')],axis=0)
    M0_ref = data[names.index('M0_referenz')]
    plot_names.append("Reference")
    plot_names.append(" ")
    NRef = 1
#    mask = data[names.index('mask')]
  else:
          
    if "IRLL" in files:
#      T1_tgv.append(data[names.index("T1_final")]*5500)
#      T1_tikh.append(data[names.index("T1_ref")]*5500)      
      T1_tgv.append((data[names.index('T1_final')])*tr)
      T1_tikh.append((data[names.index('T1_ref')])*tr)       
      full_res.append(data[names.index('full_result')])
      M0_tgv.append(data[names.index('M0_final')])
      M0_tikh.append(data[names.index('M0_ref')])      
    else:
      scale = file.attrs['E1_scale']  
      T1_tgv.append(-tr/np.log(data[names.index('full_result')]*scale))
      T1_tikh.append(-tr/np.log(data[names.index('T1_ref')]*scale)) 
      M0_tgv.append(data[names.index('full_result')])
      M0_tikh.append(data[names.index('M0_ref')])

    plot_names.append(fname[-5:].split('_')[1] + " TGV")  
    plot_names.append(fname[-5:].split('_')[0] + " Tikh")  
  file.close()  
  
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

mid_x = int(x/2)#55
mid_y = int(y/2)#55
offset = 0
plot_err = True

pos_ref = 0
pos = 0
mask = mask[int(z/2)+pos,int(y/2)-mid_y+offset:int(y/2)+mid_y+offset,int(x/2)-mid_x+offset:int(x/2)+mid_x+offset]

if "Reference" in plot_names:
  if len(T1_ref.shape) == 2:
    dimz = 1
    dimy, dimx =   T1_ref.shape
    T1_ref = T1_ref[None,...]
  else:      
    dimz, dimy, dimx =   T1_ref.shape
  T1_ref = ((T1_ref[int(dimz/2)+pos_ref,int(dimy/2)-mid_y+offset:int(dimy/2)+mid_y+offset,int(dimx/2)-mid_x+offset:int(dimx/2)+mid_x+offset]*mask).T)
  T1_plot.append(T1_ref)
  T1_plot.append(np.zeros((dimy,dimx)))
  
#  M0_plot.append(M0_ref[int(dimz/2),int(dimy/2)-half_im_size:int(dimy/2)+half_im_size,int(dimx/2)-half_im_size:int(dimx/2)+half_im_size].T)
#  M0_plot.append(np.zeros((dimy,dimx)))

T1_err=[]

T1_err_min = 0
T1_err_max = 30


for i in range(NResults-NRef):
  slices, x, y = T1_tgv[i].shape
  T1_tikh[i] = np.reshape(T1_tikh[i],(slices,x,y))
  M0_tikh[i] = np.reshape(M0_tikh[i],(slices,x,y))
  
  T1_tgv[i] = T1_tgv[i][int(np.floor(slices/2))+pos,int(y/2)-mid_y+offset:int(y/2)+mid_y+offset,int(x/2)-mid_x+offset:int(x/2)+mid_x+offset]*mask
  M0_tgv[i] = M0_tgv[i][int(np.floor(slices/2))+pos,int(y/2)-mid_y+offset:int(y/2)+mid_y+offset,int(x/2)-mid_x+offset:int(x/2)+mid_x+offset]*mask
  T1_tikh[i] = T1_tikh[i][int(np.floor(slices/2))+pos,int(y/2)-mid_y+offset:int(y/2)+mid_y+offset,int(x/2)-mid_x+offset:int(x/2)+mid_x+offset]*mask
  M0_tikh[i] = M0_tikh[i][int(np.floor(slices/2))+pos,int(y/2)-mid_y+offset:int(y/2)+mid_y+offset,int(x/2)-mid_x+offset:int(x/2)+mid_x+offset]*mask

  T1_plot.append(np.squeeze(T1_tgv[i][...]).T)
  T1_plot.append(np.squeeze(T1_tikh[i][...]).T)

#  M0_plot.append(np.squeeze(M0_tgv[i][...]).T)
#  M0_plot.append(np.squeeze(M0_tikh[i][...]).T)
  
  if "Reference" in plot_names:
    T1_err.append(np.squeeze(np.abs(T1_tgv[i]-T1_ref.T)\
                             /np.abs(T1_ref.T)).T*100)
    T1_err.append(np.squeeze(np.abs(T1_tikh[i]-T1_ref.T)\
                             /np.abs(T1_ref.T)).T*100)  
    
if "Reference" in plot_names and plot_err:  
  fig = plt.figure(figsize = (8,8))
  fig.subplots_adjust(hspace=0, wspace=0.1)
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
  height_ratio.insert(2,1/4)
  gs = gridspec.GridSpec(5,NResults+1, width_ratios=width_ratio,height_ratios=height_ratio)  
  
  for i in range((NResults)):
    ax.append(plt.subplot(gs[i]))
    im = ax[i].imshow(np.abs(T1_plot[2*i]),vmin=T1_min,vmax=T1_max,aspect=1,cmap=cm.viridis)
    ax[i].axis('off')
  #  ax[i].set_anchor("N")   
    ax[i].set_title(plot_names[2*i],color='white',fontsize=16)
  
  for i in range((NResults)):
    ax.append(plt.subplot(gs[i+NResults+1]))
    im = ax[i+NResults].imshow(np.abs(T1_plot[2*i+1]),vmin=T1_min,vmax=T1_max,aspect=1,cmap=cm.viridis)
    ax[i+NResults].axis('off')
    ax[i+NResults].set_anchor("S")    
    ax[i+NResults].set_title(plot_names[2*i+1],color='white',fontsize=16)
    
  cax = plt.subplot(gs[:2,-1])
  cbar = fig.colorbar(im, cax=cax)
  cbar.ax.tick_params(labelsize=16,colors='white')
  for spine in cbar.ax.spines:
    cbar.ax.spines[spine].set_color('white')
  
  for i in range((NResults-NRef)):
    ax.append(plt.subplot(gs[i+3*(NResults+1)+1]))
    im = ax[i+2*(NResults)].imshow(np.abs(T1_err[2*i]),vmin=T1_err_min,vmax=T1_err_max,aspect=1,cmap=cm.inferno)
    ax[i+2*(NResults)].axis('off')
    ax[i+2*(NResults)].set_anchor("N")   
    ax[i+2*(NResults)].set_title(plot_names[2*(i+1)],color='white',fontsize=16)
  
  for i in range((NResults-NRef)):
    ax.append(plt.subplot(gs[i+4*(NResults+1)+1]))
    im = ax[i+3*(NResults)-1].imshow(np.abs(T1_err[2*i+1]),vmin=T1_err_min,vmax=T1_err_max,aspect=1,cmap=cm.inferno)
    ax[i+3*(NResults)-1].axis('off')
    ax[i+3*(NResults)-1].set_anchor("S")   
    ax[i+3*(NResults)-1].set_title(plot_names[2*(i+1)+1],color='white',fontsize=16)
  
  cax = plt.subplot(gs[3:,-1])
  cbar = fig.colorbar(im, cax=cax)
  cbar.ax.tick_params(labelsize=16,colors='white')
  for spine in cbar.ax.spines:
    cbar.ax.spines[spine].set_color('white')
  plt.show()  

else:
  fig = plt.figure(figsize = (2*int(NResults),5))
  fig.subplots_adjust(hspace=0, wspace=0.1)
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
#  height_ratio.insert(2,1/10000)
  gs = gridspec.GridSpec(2,NResults+1, width_ratios=width_ratio,height_ratios=height_ratio)  
  
  for i in range((NResults)):
    ax.append(plt.subplot(gs[i]))
    im = ax[i].imshow(np.abs(T1_plot[2*i]),vmin=T1_min,vmax=T1_max,aspect=1,cmap=cm.viridis)
    ax[i].axis('off')  
    ax[i].set_title(plot_names[2*i],color='white',fontsize=16)
  
  for i in range((NResults)):
    ax.append(plt.subplot(gs[i+NResults+1]))
    im = ax[i+NResults].imshow(np.abs(T1_plot[2*i+1]),vmin=T1_min,vmax=T1_max,aspect=1,cmap=cm.viridis)
    ax[i+NResults].axis('off')    
    ax[i+NResults].set_title(plot_names[2*i+1],color='white',fontsize=16)
    
  cax = plt.subplot(gs[:2,-1])
  cbar = fig.colorbar(im, cax=cax)
  cbar.ax.tick_params(labelsize=16,colors='white')
  for spine in cbar.ax.spines:
    cbar.ax.spines[spine].set_color('white')
  plt.show()  



import scipy.stats as stat
  

roi_num = int(input("Enter the number of desired ROIs: "))
if roi_num > 0:
  if not "Reference" in plot_names:
    T1_ref = T1_tgv[0]
  mean_TGV = []
  mean_Tikh = []
  std_TGV = []
  std_Tikh =  []
  col_names = []
  statistic = []
  selector = cv2.cvtColor(np.abs(np.squeeze(T1_ref[...])/3000).astype(np.float32),cv2.COLOR_GRAY2BGR)
  
  for j in range(roi_num):
    r = (cv2.selectROI(selector,False))
    col_names.append("ROI "+str(j+1))
    mean_TGV.append(np.abs(np.mean(T1_ref.T[int(r[0]):int(r[0]+r[2]), int(r[1]):int(r[1]+r[3])])))   
    std_TGV.append(np.abs(np.std(T1_ref.T[int(r[0]):int(r[0]+r[2]), int(r[1]):int(r[1]+r[3])])))       
    for i in range((NResults-1)):    
      mean_TGV.append(np.abs(np.mean(T1_tgv[i][int(r[0]):int(r[0]+r[2]), int(r[1]):int(r[1]+r[3])])))
      mean_Tikh.append(np.abs(np.mean(T1_tikh[i][int(r[0]):int(r[0]+r[2]), int(r[1]):int(r[1]+r[3])])))
      std_TGV.append(np.abs(np.std(T1_tgv[i][int(r[0]):int(r[0]+r[2]), int(r[1]):int(r[1]+r[3])])))
      std_Tikh.append(np.abs(np.std(T1_tikh[i][int(r[0]):int(r[0]+r[2]), int(r[1]):int(r[1]+r[3])])))
      statistic.append(stat.ttest_ind(np.abs(T1_tgv[0][int(r[0]):int(r[0]+r[2]), int(r[1]):int(r[1]+r[3])]).flatten(),
                                      np.abs((T1_ref[int(r[0]):int(r[0]+r[2]), int(r[1]):int(r[1]+r[3])])).flatten()))
    rects = patches.Rectangle((int(r[0]),int(r[1])),
                                   int(r[2]),int(r[3]),linewidth=1,edgecolor='r',facecolor='none')
    posx = int(r[1]-5)
    posy = int(r[0])
    ax[0].text(posy,posx,str(j+1),color='red')
    ax[0].add_patch(rects) 
  
  mean_TGV = np.round(pd.DataFrame(np.reshape(np.asarray(mean_TGV),(roi_num,NResults)).T,index=plot_names[0::2],columns=col_names),decimals=0)
  mean_Tikh =  np.round(pd.DataFrame(np.reshape(np.asarray(mean_Tikh),(roi_num,NResults-1)).T,index=plot_names[3::2],columns=col_names),decimals=0)
  std_TGV =  np.round(pd.DataFrame(np.reshape(np.asarray(std_TGV),(roi_num,NResults)).T,index=plot_names[0::2],columns=col_names),decimals=0)
  std_Tikh =  np.round(pd.DataFrame(np.reshape(np.asarray(std_Tikh),(roi_num,NResults-1)).T,index=plot_names[3::2],columns=col_names),decimals=0)
  
  f = open("test.tex","w")
  f.write(mean_TGV.to_latex())
  f.write(mean_Tikh.to_latex())
  f.write(std_TGV.to_latex())
  f.write(std_Tikh.to_latex())
  f.flush()
  f.close()
  
  
  from pandas.plotting import table
  
  
  fig_table = plt.figure(figsize = (16,4))
  fig_table.subplots_adjust(hspace=0, wspace=0.5)
  
  gs = gridspec.GridSpec(2,2)
  
  ax_table = []
  my_tabs = []
  my_tabs.append(mean_TGV)
  my_tabs.append(mean_Tikh)
  my_tabs.append(std_TGV)
  my_tabs.append(std_Tikh)
  for i in range(len(my_tabs)):
    ax_table.append(plt.subplot(gs[i]))
    table(ax_table[i],np.round(my_tabs[i],1),loc='center')
    ax_table[i].axis('off')
    if i<2:
      ax_table[i].set_title("Mean")
    else:
      ax_table[i].set_title("Standardeviation")
 
fig_lines = plt.figure(figsize=(16,8))
fig_lines.subplots_adjust(hspace=0, wspace=0.5)
#plt.subplot(211)
line_pos = 84
plt.plot(T1_ref[line_pos,:])
ax[0].hlines(line_pos,1,x-1,colors='w')
k= 1
for i in range(NResults-1):
  plt.plot(T1_tgv[i][:,line_pos],dashes=[k*2,2,k*2,2])
  k+=1
  plt.plot(T1_tikh[i][:,line_pos],dashes=[k*2,2,k*2,2])
  k+=1
# 
#plt.legend(plot_names[0::2])
#plt.ylabel('T1 in ms')
del plot_names[1]      
#plt.subplot(212)
#
#line_pos = 84
#plt.plot(T1_ref[line_pos,:])
#ax[0].hlines(line_pos,1,x-1,colors='w')
#k= 1
#for i in range(NResults-1):
##  plt.plot(T1_tgv[i][:,line_pos],dashes=[k*2,2,k*2,2])
##  k+=1
#  plt.plot(T1_tikh[i][:,line_pos],dashes=[k*2,2,k*2,2])
#  k+=1
# 
plt.legend(plot_names)
plt.xlabel('Pixel')
plt.ylabel('T1 in ms')
plt.savefig('/media/data/Papers/Parameter_Mapping/Lineplot'+save_name+'.svg', format='svg', dpi=1000)

plt.savefig('/media/data/Papers/Parameter_Mapping/2Dvs3D_'+save_name+'.svg', format='svg', dpi=1000)