#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:16:33 2017

@author: omaier
"""
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
import numpy as np

def compute(image):
  
  mask = np.zeros_like(image)
  ra