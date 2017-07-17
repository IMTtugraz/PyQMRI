#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:42:42 2017

@author: omaier
"""

import numpy as np
import time
from scipy.ndimage.filters import gaussian_filter as gf


from createInitGuess_FLASH import createInitGuess_FLASH
#from compute_mask import compute_mask


class VFA_Model:
  def __init__(self,fa,fa_corr,TR,images,phase_map):



    self.TR = TR
    self.images = images
    self.fa = fa
    self.fa_corr = fa_corr
    
#    siz = images.shape
    
    phi_corr = np.zeros_like(images,dtype='complex128')
    for i in range(np.size(fa)):
      phi_corr[i,:,:,:] = fa[i]*fa_corr[i,:,:,:]
    
    self.sin_phi = np.sin(phi_corr)
    self.cos_phi = np.cos(phi_corr)    

    
    th = time.clock()

    [M0_guess, T1_guess, mask_guess] = createInitGuess_FLASH(self.images,phi_corr,self.TR)

    T1_guess[np.isnan(T1_guess)] = np.spacing(1)
    T1_guess[np.isinf(T1_guess)] = np.spacing(1)
    T1_guess[T1_guess<0] = 0 
    T1_guess[T1_guess>5000] = 5000
    T1_guess = np.abs(T1_guess)

    M0_guess[M0_guess<0] = 0 
    M0_guess[np.isnan(M0_guess)] = np.spacing(1)
    M0_guess[np.isinf(M0_guess)] = np.spacing(1)   
    M0_guess = np.abs(M0_guess)
    
#
    hist =  np.histogram(np.abs(M0_guess),int(1e2))
    aa = np.array(hist[0], dtype=np.float64)
    #bb = hist[1] #hist0[1][:-1] + np.diff(hist0[1])/2
    bb = np.array(hist[1][:-1] + np.diff(hist[1])/2, dtype=np.float64)
   
    idx = np.array(aa > 0.01*aa[0],dtype=np.float64)

    M0_guess[M0_guess > bb[int(np.sum(idx))]] = bb[int(np.sum(idx))] #passst
#    print(M0_guess)
    
    
    M0_guess = np.squeeze(M0_guess)

#    mask_guess = compute_mask(M0_guess,False)

#    self.mask = mask_guess#par.mask[:,63] is different
    
    self.T1_sc = np.max(T1_guess)
    self.M0_sc = np.max(np.abs(M0_guess))
    
    #print(mask_guess)
    print('T1 scale: ',self.T1_sc,
                              '/ M0_scale: ',self.M0_sc)
    #print(M0_guess[39,11]) M0 guess is gleich
    self.T1_sc = 5e3
    self.M0_sc = 30
    M0_guess = M0_guess / self.M0_sc
    T1_guess = T1_guess / self.T1_sc

    T1_guess[np.isnan(T1_guess)] = 0;
    M0_guess[np.isnan(M0_guess)] = 0;
    
    self.T1_guess = T1_guess * self.T1_sc
    self.M0_guess = M0_guess * self.M0_sc
#        
    print( 'done in', time.clock() - th)

    result = np.concatenate(((gf(M0_guess,5)*np.exp(1j*np.angle(phase_map)))[None,:,:,:],gf(T1_guess,5)[None,:,:,:]),axis=0)
#    result = np.concatenate((((M0_guess)*np.exp(1j*np.angle(phase_map)))[None,:,:,:],(T1_guess)[None,None,:,:]),axis=0)
#    result = np.array([(0.01+0*M0_guess*np.exp(1j*np.angle(phase_map))),0.3+0*(T1_guess)])
#    result = np.array([1/self.M0_sc*np.ones((siz[1],siz[2],siz[3]),dtype='complex128'),1500/self.T1_sc*np.ones((siz[1],siz[2],siz[3]),dtype='complex128')])
    self.guess = result               
    
    
  def execute_forward_2D(self,x,slice):
    E1 = np.exp(-self.TR/(x[1,:,:]*self.T1_sc))#
#    E1 = x[1,:,:]
    S = x[0,:,:]*self.M0_sc*self.sin_phi[:,slice,:,:]*(1-E1)/(1-E1*self.cos_phi[:,slice,:,:])
    S[~np.isfinite(S)] = 1e-20
    return S
  def execute_gradient_2D(self,x,slice):
    E1 = np.exp(self.TR/(x[1,:,:]*self.T1_sc))  ####no minus!!!  
#    E1 = x[1,:,:]
#    E1[~np.isfinite(E1)] = 0
    grad_M0 = (self.M0_sc*self.sin_phi[:,slice,:,:]*(E1-1))/(E1-self.cos_phi[:,slice,:,:])
    grad_T1 = (-(x[0,:,:]*self.M0_sc*self.TR*E1*(2*self.sin_phi[:,slice,:,:]-2*self.sin_phi[:,slice,:,:]*self.cos_phi[:,slice,:,:]))/
               (2*x[1,:,:]**2*self.T1_sc*(E1-self.cos_phi[:,slice,:,:])**2))
#    grad_M0 = (self.M0_sc*self.sin_phi[:,slice,:,:]*(1 - E1))/(-self.cos_phi[:,slice,:,:]*E1 + 1)
#    grad_T1 = (x[0,:,:]*self.M0_sc*self.sin_phi[:,slice,:,:]*(self.cos_phi[:,slice,:,:] - 1))/(E1*self.cos_phi[:,slice,:,:] - 1)**2
    grad = np.array([grad_M0,grad_T1])
    grad[~np.isfinite(grad)] = 1e-20
    return grad
  
  def execute_forward_3D(self,x):
    E1 = np.exp(-self.TR/(x[1,:,:,:]*self.T1_sc))
    S = x[0,:,:,:]*self.M0_sc*self.sin_phi*(1-E1)/(1-E1*self.cos_phi)
    S[~np.isfinite(S)] = 1e-20
    return S
  def execute_gradient_3D(self,x):
    E1 = np.exp(self.TR/(x[1,:,:]*self.T1_sc))
    
    grad_M0 = (self.M0_sc*self.sin_phi*(E1-1))/(E1-self.cos_phi)
    grad_T1 = (-(x[0,:,:,:]*self.M0_sc*self.TR*E1*(2*self.sin_phi-2*self.sin_phi*self.cos_phi))/
               (2*x[1,:,:,:]**2*self.T1_sc*(E1-self.cos_phi)**2))
    grad = np.array([grad_M0,grad_T1])
    grad[~np.isfinite(grad)] = 1e-20
    return grad
  

    
  