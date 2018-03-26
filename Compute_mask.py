#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:16:33 2017

@author: omaier
"""
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_opening
from skimage.morphology import remove_small_objects
from skimage.util import img_as_ubyte
import numpy as np

def compute(image):
  
  mask = np.zeros_like(image,dtype=bool)
  image = np.abs(image)
  if len(image.shape) > 2:
    for i in range(image.shape[0]):
      thres = threshold_otsu(image[i,...])*1
      mask[i,...] = image[i,...]>=thres
      mask[i,...] = binary_closing(mask[i,...],iterations=2)          
      mask[i,...] = binary_fill_holes((mask[i,...]))

      
  else:
    thres = threshold_otsu(image)
    mask = image>=thres      
    mask = binary_closing(mask)    
  mask = remove_small_objects(mask, 20000)

  return mask

def skullstrip(image):
  
  mask = np.zeros_like(image,dtype=bool)
  image = np.abs(image)
  test = np.ones((4,4))
  test[0,0] = 0
  test[0,-1] = 0
  test[-1,0] = 0
  test[-1,-1] = 0
  if len(image.shape) > 2:
    for i in range(image.shape[0]):
      thres = threshold_otsu(image[i,...])
      mask[i,...] = image[i,...]>=thres*1.1
      mask[i,...] = binary_opening(mask[i,...],test,iterations=6)
      mask[i,...] = binary_closing(mask[i,...],test,iterations=10)          
      mask[i,...] = binary_fill_holes((mask[i,...]))

      
  else:
    thres = threshold_otsu(image)
    mask = image>=thres      
    mask = binary_closing(mask)    
  mask = remove_small_objects(mask, 200)

  return mask