#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:04:42 2017

@author: omaier
"""
import numpy as np

def exp_mean(x,tau,T1):
  result = np.zeros((np.shape(x)[0]))
  w=0
  for k in range(np.shape(x)[1]): 
    w = 1-np.exp(-tau+(k*np.shape(x)[-1])/T1)     
    V = (1-w**np.shape(x)[-1])/(1-w)     
    for i in range(np.shape(x)[-1]):
      result[k] = result[k] + x[k,i]*w**i/V
  return result
      