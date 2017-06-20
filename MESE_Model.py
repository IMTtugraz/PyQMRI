#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:42:42 2017

@author: omaier
"""

import numpy as np


class MESE_Model:
  def __init__(self):
    self.M0_sc = []
    self.T2_sc= []
    self.TE = []
  def execute_forward_2D(self,x):
    S = np.zeros((np.size(self.TE),np.shape(x)[1],np.shape(x)[2]),dtype='complex128')
    for i in range(np.size(self.TE)):
        S[i,:,:] = x[0,:,:]*self.M0_sc * np.exp(self.TE[i]/(x[1,:,:]*self.T2_sc))
    return S
  def execute_gradient_2D(self,x):
    grad_M0 = np.zeros((np.size(self.TE),np.shape(x)[1],np.shape(x)[2]),dtype='complex128')
    grad_T2 = np.zeros((np.size(self.TE),np.shape(x)[1],np.shape(x)[2]),dtype='complex128')
    for i in range(np.size(self.TE)):  
        grad_M0[i,:,:] = self.M0_sc*np.exp(self.TE[i]/(x[1,:,:]*self.T2_sc))
        grad_T2[i,:,:] = x[0,:,:]*self.M0_sc*self.TE[i]*np.exp(-self.TE[i]/(x[1,:,:]*self.T2_sc))/(x[1,:,:]**2*self.T2_sc)
    grad = np.array([grad_M0,grad_T2])
    grad[~np.isfinite(grad)] = 1e-20
    return grad
  