#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:26:42 2018

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



root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

file = h5py.File(file)


data = file['tgv_full_result_0'][()]
data = 1/data[-1,1,...]

[z,y,x] = data.shape

from matplotlib.path import Path



import polyroi as polyroi
ref_pos = 0 #int(z/2)
roi = polyroi.polyroi(data,ref_pos,np.max(np.abs(data)))
#roi = polyroi.polyroi(data,ref_pos)
#test = np.array(roi.select_roi())


coord_map = np.vstack((np.repeat(np.arange(0,x,1)[None,:],x,0).flatten(), np.repeat(np.arange(0,y,1)[None,:].T,y,1).flatten())).T
#polypath = Path((test).astype(int))
#mask = polypath.contains_points(coord_map).reshape(y,x)


roi_num = 14#int(input("Enter the number of desired ROIs: "))
if roi_num > 0:
  mean_TGV = []
  std_TGV = []
  cv2.namedWindow('ROISelector', cv2.WINDOW_NORMAL)
  for j in range(roi_num):
    r = np.array(roi.select_roi())
    polypath = Path((r).astype(int))
    mask = polypath.contains_points(coord_map).reshape(y,x)
    mean_TGV.append(np.abs(np.mean(data[ref_pos,mask.T])))
    std_TGV.append(np.abs(np.std(data[ref_pos,mask.T])))


  mean_TGV = np.round(pd.DataFrame(np.reshape(np.asarray(mean_TGV),(roi_num,1)).T),decimals=0)
  std_TGV =  np.round(pd.DataFrame(np.reshape(np.asarray(std_TGV),(roi_num,1)).T),decimals=0)


#  f = open("3Drois.tex","w")
#  f.write(mean_TGV.to_latex())
#  f.write(std_TGV.to_latex())
#  f.flush()
#  f.close()


  from pandas.plotting import table


  fig_table = plt.figure(figsize = (8,4))
  fig_table.subplots_adjust(hspace=0, wspace=0.5)

  gs = gridspec.GridSpec(2,1)

  ax_table = []
  my_tabs = []
  my_tabs.append(mean_TGV)
  my_tabs.append(std_TGV)
  for i in range(len(my_tabs)):
    ax_table.append(plt.subplot(gs[i]))
    table(ax_table[i],np.round(my_tabs[i],1),loc='center')
    ax_table[i].axis('off')
    if i<1:
      ax_table[i].set_title("Mean")
    else:
      ax_table[i].set_title("Standardeviation")
