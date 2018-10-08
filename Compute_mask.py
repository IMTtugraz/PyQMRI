#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:16:33 2017

@author: omaier
"""
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_fill_holes, binary_closing
from skimage.morphology import remove_small_objects
import numpy as np
from skimage import filters
from skimage import morphology
#from scipy import ndimage as ndi
#import matplotlib.pyplot as plt
def compute(image):

  mask = np.zeros_like(image,dtype=bool)
  image = np.abs(image)
  if len(image.shape) > 2:
    thres = threshold_otsu(np.reshape(image,(image.shape[0]*image.shape[1],image.shape[2])))
    for i in range(image.shape[0]):
      mask[i,...] = image[i,...]>=thres*0.2
      mask[i,...] = remove_small_objects(mask[i,...], 250)
      mask[i,...] = binary_closing(mask[i,...],iterations=15)
      mask[i,...] = binary_fill_holes((mask[i,...]))
#      sobel = filters.sobel(mask[i,...]*image[i,...]/np.max(image[i,...]))
#      blurred = filters.gaussian(sobel, sigma=10)
##      light_spots = np.array((image[i,...]*mask[i,...] > 1).nonzero()).T
##      dark_spots = np.array((image[i,...]*mask[i,...] < 0.1).nonzero()).T
##      bool_mask = np.zeros(image[i,...].shape, dtype=np.bool)
##      bool_mask[tuple(light_spots.T)] = True
##      bool_mask[tuple(dark_spots.T)] = True
##      seed_mask, num_seeds = ndi.label(bool_mask)
#      seed_mask = np.zeros(image[i,...].shape, dtype=np.int)
#      seed_mask[0, 0] = 1 # background
#      seed_mask[-1, -1] = 1 # background
#      seed_mask[0, -1] = 1 # background
#      seed_mask[-1, 0] = 1 # background
#      seed_mask[128-40:128+40, 128-40:128+40] = 2 # foreground
#      ws = morphology.watershed(blurred, seed_mask,mask=mask[i,...])
##      plt.imshow(ws)
##      plt.pause(1e-3)
#      background = max(set(ws.ravel()), key=lambda g: np.sum(ws == g))
#      background_mask = (ws == background)
#      mask[i,...] = ~background_mask



  else:
    thres = threshold_otsu(image)
    mask = image>=thres
    mask = binary_closing(mask)
    mask = remove_small_objects(mask, 10000)

  return mask
