#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:55:39 2017

@author: omaier
"""
import matplotlib.pyplot as plt
import numpy as np

################################################################################
### 3D viewer  ##########s######################################################
################################################################################              
def imshow(volume):

  if volume.ndim<=3: 
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
  elif volume.ndim==4:
    fig, ax = plt.subplots(int(np.ceil(np.sqrt(volume.shape[0]))),int(np.ceil(np.sqrt(volume.shape[0]))))
    ax = ax.flatten()
    ni = int(np.ceil(np.sqrt(volume.shape[0])))
    nj = int(np.ceil(np.sqrt(volume.shape[0])))
    for j in range(nj):
      for i in range(ni):
        if i+ni*j >= volume.shape[0]:
          ax[i+ni*j].volume = np.zeros_like(volume[0])
        else:
          ax[i+ni*j].volume = volume[i+(j*ni)]
          ax[i+ni*j].index = volume[i+(j*ni)].shape[0] // 2
          ax[i+ni*j].imshow(volume[i+(j*ni),ax[i+ni*j].index])
  else:
    raise NameError('Unsupported Dimensions')
  fig.canvas.mpl_connect('scroll_event', process_scroll)
#  axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
#  axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
#  bnext = Button(axnext,'Next')
#  bprev = Button(axprev,'Prev')
def process_scroll(event):
  fig = event.canvas.figure
  ax = fig.axes
  for i in range(len(ax)):
    volume = ax[i].volume
    if (int((ax[i].index - event.step) >= volume.shape[0]) or 
           int((ax[i].index - event.step) < 0)):
           pass
    else:
      ax[i].index = int((ax[i].index - event.step) % volume.shape[0])
      ax[i].images[0].set_array(volume[ax[i].index])
      fig.canvas.draw()
