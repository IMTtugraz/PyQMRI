#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:05:59 2018

@author: omaier
"""
import numpy as np

def cmp(k):

  if len(np.shape(k))==2:
    nspkes,N = np.shape(k)#
  elif len(np.shape(k))==3:
    NScan,nspokes,N = np.shape(k)
  else:
    return -5
  
  w = np.abs(np.linspace(-N/2,N/2,N))
  w = w*np.pi/4/nspokes
  w = np.repeat(w,nspokes,0)
  w = np.reshape(w,(N,nspokes)).T
  return w

     
