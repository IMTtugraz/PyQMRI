#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:16:33 2017

@author: omaier
"""
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_fill_holes, binary_closing
from skimage.morphology import remove_small_objects
from skimage.util import img_as_ubyte
import numpy as np

def compute(image):
  
  mask = np.zeros_like(image,dtype=bool)
  image = np.abs(image)
  if len(image.shape) > 2:
    for i in range(image.shape[0]):
      thres = threshold_otsu(image[i,...])
      mask[i,...] = image[i,...]>=thres
      mask[i,...] = binary_fill_holes((mask[i,...]))
      
  else:
    thres = threshold_otsu(image)
    mask = image>=thres      
  mask = remove_small_objects(mask, 10)
  return mask
