#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:35:41 2018

@author: omaier
"""

from scipy.interpolate import griddata
import numpy as np
import polyroi as polyroi
from scipy.ndimage.filters import gaussian_filter

from tkinter import filedialog
from tkinter import Tk

import h5py  
import cv2
import os
DTYPE = np.complex64
################################################################################
### Select input file ##########################################################
################################################################################

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

name = file.split('/')[-1]
file = h5py.File(file)

################################################################################
### Check if file contains all necessary information ###########################
################################################################################
test_data = ['dcf', 'fa_corr', 'imag_dat', 'imag_traj', 'real_dat',
             'real_traj']
test_attributes = ['image_dimensions', 'tau', 'gradient_delay',
                   'flip_angle(s)', 'time_per_slice']
dimX, dimY, NSlice = (file.attrs['image_dimensions']).astype(int)
reco_Slices = NSlice
for datasets in test_data:
  if not (datasets in list(file.keys())):
    file.close()
    raise NameError("Error: '" + datasets +
                    "' data was not provided/wrongly named!")
for attributes in test_attributes:
  if not (attributes in list(file.attrs)):
    file.close()
    raise NameError("Error: '" + attributes +
                    "' was not provided/wrongly named as an attribute!")


################################################################################
### FA correction ##############################################################
################################################################################

fa_corr = file['fa_corr'][()].astype(DTYPE)[int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...]
fa_corr[fa_corr==0] = 1

fa_corr = fa_corr[30-2:30+3,...]

from matplotlib.path import Path
[z,y,x] = fa_corr.shape
grid = []
cv2.namedWindow('ROISelector', cv2.WINDOW_NORMAL)
for i in range(z):
  grid.append([])
  while True:
    roi = polyroi.polyroi(fa_corr,i,np.max(np.abs(fa_corr)))
    coord_map = np.vstack((np.repeat(np.arange(0,x,1)[None,:],y,0).flatten(), np.repeat(np.arange(0,y,1)[None,:].T,x,1).flatten())).T
    r = np.array(roi.select_roi()) 
    polypath = Path((r).astype(int))
    mask = polypath.contains_points(coord_map).reshape(y,x)
    grid[i].append(mask)
    cont = int(input("continue? [1]yes [0]no"))
    if cont ==1:
      break

mask = np.ones((z,y,x),bool)      
for i in range(z):
  for j in range(len(grid[i])):
    mask[i,...] = np.logical_and(mask[i,...],np.invert(grid[i][j]))
    
interpol_fa = np.zeros((z,y,x),DTYPE)    
for j in range(z)    :
  points = np.argwhere(np.squeeze(mask[j,...].T)>0)
  values = fa_corr[j,mask[j,...].T]
  
  grid_x, grid_y = np.mgrid[0:x:1, 0:y:1]
  interpol_fa[j,...] = griddata(points, values, ( grid_x, grid_y), method='cubic')
  
interpol_fa = gaussian_filter(np.abs(interpol_fa),3)

fa_corr = file['fa_corr'][()].astype(DTYPE)[int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...]
fa_corr[fa_corr==0] = 1
fa_corr[30-2:30+3,...] = interpol_fa


dset_result=file.create_dataset("interpol_fa",fa_corr.shape,\
                             dtype=DTYPE,data=fa_corr)
file.flush()
file.close()
