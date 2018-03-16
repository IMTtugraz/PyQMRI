#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:42:42 2017

@author: omaier
"""

import numpy as np
import time
from scipy.ndimage.filters import gaussian_filter as gf
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

plt.ion()

from createInitGuess_FLASH import createInitGuess_FLASH
#from compute_mask import compute_mask
DTYPE = np.complex64

import ipyparallel as ipp



class constraint:
  def __init__(self, min_val=-np.inf, max_val=np.inf, real_const=False):
    self.min = min_val
    self.max = max_val
    self.real = real_const


class VFA_Model:
  def __init__(self,fa,fa_corr,TR,images,phase_map,Nislice):
    self.constraints = []    
    self.TR = TR
    self.images = images
    self.fa = fa
    self.fa_corr = np.ones_like(fa_corr)
    self.Nislice = Nislice
#    c = ipp.Client()
    
    (NScan,Nislice,dimX,dimY) = images.shape
    
    phi_corr = np.zeros_like(images,dtype=DTYPE)
    for i in range(np.size(fa)):
      phi_corr[i,:,:,:] = fa[i]*fa_corr
    self.fa = phi_corr
#    self.sin_phi = np.sin(phi_corr)
#    self.cos_phi = np.cos(phi_corr)    

    
#    th = time.clock()
#
##    [M0_guess, T1_guess, mask_guess] = createInitGuess_FLASH(self.images,phi_corr,self.TR)
#    result = []
#    T1_guess = np.zeros((Nislice,dimY,dimX),DTYPE)
#    M0_guess = np.zeros((Nislice,dimY,dimX),DTYPE)
#    for i in range(Nislice):
#      print("Processing slice: %i" %(i))
#      dview = c[int(np.floor(i*len(c)/Nislice))]
#      result.append(dview.apply_async(createInitGuess_FLASH, images[:,i,...], 
#                                    phi_corr[:,i,...], TR))
#    for i in range(Nislice):  
#      T1_guess[i,:,:] = result[i].get()[1]
#      M0_guess[i,:,:] = result[i].get()[0]
#
#    T1_guess[np.isnan(T1_guess)] = np.spacing(1)
#    T1_guess[np.isinf(T1_guess)] = np.spacing(1)
#    T1_guess = np.abs(T1_guess)
#    T1_guess[T1_guess<0] = 0 
#
#
##    T1_guess[T1_guess>5000] = 5000
##    T1_guess = np.abs(T1_guess)
#    M0_guess = np.abs(M0_guess)
#    M0_guess[M0_guess<0] = 0 
#    M0_guess[np.isnan(M0_guess)] = np.spacing(1)
#    M0_guess[np.isinf(M0_guess)] = np.spacing(1)   
#
#    
##
#    hist =  np.histogram((M0_guess),int(1e3),range=(0,10*(np.max((np.median(M0_guess),np.mean(M0_guess))))))
#    aa = np.array(hist[0], dtype=np.float64)
#    #bb = hist[1] #hist0[1][:-1] + np.diff(hist0[1])/2
#    bb =(hist[1])
#   
#    max_val = aa[np.argmax(aa[1:-1])+1]
#    FWHM = 2*np.abs(np.argwhere(aa[np.argmax(aa[1:-1])+1:]<max_val/2)[0][0])
#    std = FWHM/(2*np.sqrt(2*np.log(2)))
##    
#    
#
#    M0_guess[M0_guess > bb[np.argmax(aa[1:-1])+ 1+ int(5*std)]] =  bb[np.argmax(aa[1:-1])+ 1+ int(5*std)] #passst
##    print(M0_guess)
#    
#    hist =  np.histogram((T1_guess),int(1e3),range=(0,10*(np.max((np.median(T1_guess),np.mean(T1_guess))))))
#    aa = np.array(hist[0], dtype=np.float64)
##    bb = hist[1] #hist0[1][:-1] + np.diff(hist0[1])/2
#    bb =(hist[1])
#   
#    max_val = aa[np.argmax(aa[1:-1])+1]
#    FWHM = 2*np.abs(np.argwhere(aa[np.argmax(aa[1:-1])+1:]<max_val/2)[0][0])
#    std = FWHM/(2*np.sqrt(2*np.log(2)))
##    
#    
#
#    T1_guess[T1_guess > bb[np.argmax(aa[1:-1])+ 1+ int(5*std)]] =  bb[np.argmax(aa[1:-1])+ 1+ int(5*std)] #passst

    self.M0_sc = 1#np.max(np.abs(M0_guess))    
    self.T1_sc = 1#np.max(np.abs(T1_guess))
    self.fa_scale = 1

#    self.T1_guess = np.copy(T1_guess)
#    self.M0_guess = np.copy(M0_guess)
###
    test_T1 = np.reshape(np.linspace(10,5500,dimX*dimY*Nislice),(Nislice,dimX,dimY))
    test_M0 = 1#np.reshape(np.linspace(0,1,dimX*dimY*Nislice),(Nislice,dimX,dimY))
    G_x = self.execute_forward_3D(np.array([test_M0/self.M0_sc*np.ones((Nislice,dimY,dimX),dtype=DTYPE),\
                                            1/self.T1_sc*np.exp(-self.TR/(test_T1*np.ones((Nislice,dimY,dimX),dtype=DTYPE))),\
                                            1/self.fa_scale*np.ones((Nislice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE))
    self.M0_sc = self.M0_sc*np.mean(np.abs(images))/np.mean(np.abs(G_x))
#test_T1*np.ones((Nislice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE))#    
    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.M0_sc*np.ones((Nislice,dimY,dimX),dtype=DTYPE),\
                                               1/self.T1_sc*np.exp(-self.TR/(test_T1*np.ones((Nislice,dimY,dimX),dtype=DTYPE))),\
                                               1/self.fa_scale*np.ones((Nislice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE))
    self.T1_sc = self.T1_sc*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...]))
    self.fa_scale = self.fa_scale*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[2,...]))
    
    self.T1_sc = self.T1_sc / self.M0_sc
    self.fa_scale = self.fa_scale / self.M0_sc
    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.M0_sc*np.ones((Nislice,dimY,dimX),dtype=DTYPE),\
                                               1/self.T1_sc*np.exp(-self.TR/(test_T1*np.ones((Nislice,dimY,dimX),dtype=DTYPE))),\
                                               1/self.fa_scale*np.ones((Nislice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE))
    print('Grad Scaling E1: ', np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...])),
          'Grad Scaling FA: ', np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[2,...])))    

    #print(mask_guess)#
#    self.T1_sc = 3e3
    print('T1 scale: ',self.T1_sc,
          '/ FA_scale: ',self.fa_scale,
          '/ M0_scale: ',self.M0_sc)
    #print(M0_guess[39,11]) M0 guess is gleich


#    self.M0_sc = 50

#    M0_guess = M0_guess / self.M0_sc
#    T1_guess = T1_guess / self.T1_sc
#
#    T1_guess[np.isnan(T1_guess)] = 0;
#    M0_guess[np.isnan(M0_guess)] = 0;
#    
#
#    print( 'done in', time.clock() - th)
#
#    E1 = np.exp(-self.TR/(self.T1_guess[None,:,:,:]))
#    E1[~np.isfinite(E1)] = 1e-20

#    result = np.array(np.concatenate((self.M0_guess[None,:,:,:]/self.M0_sc,E1),axis=0),dtype=DTYPE)
    result = np.array([0.5/self.M0_sc*np.ones((Nislice,dimY,dimX),dtype=DTYPE),\
                       1/self.T1_sc*np.exp(-self.TR/(1500*np.ones((Nislice,dimY,dimX),dtype=DTYPE))),
                       fa_corr/self.fa_scale*np.ones((Nislice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE)
#    result = np.concatenate((((M0_guess)*np.exp(1j*np.angle(phase_map)))[None,:,:,:],(T1_guess)[None,None,:,:]),axis=0)
#    result = np.array([(0.01+0*M0_guess*np.exp(1j*np.angle(phase_map))),0.3+0*(T1_guess)])
#    result = np.array([1/self.M0_sc*np.ones((Nislice,dimY,dimX),dtype=DTYPE),1500/self.T1_sc*np.ones((Nislice,dimY,dimX),dtype=DTYPE)])
    self.guess = result                   
    self.constraints.append(constraint(-300,300,False)  )
#    self.constraints.append(constraint(10/self.T1_sc,5500/self.T1_sc,True))
    self.constraints.append(constraint(np.exp(-self.TR/(50))/self.T1_sc,np.exp(-self.TR/(5500))/self.T1_sc,True))
    self.constraints.append(constraint(0.6/self.fa_scale,1.4/self.fa_scale,True))
  def execute_forward_2D(self,x,islice):
#    E1 = np.exp(-self.TR/(x[1,:,:]*self.T1_sc))
    E1 = x[1,...]*self.T1_sc
    fa_corr = x[2,...]*self.fa_scale
    E1[~np.isfinite(E1)] = 0
    sin_phi = np.sin(self.fa[:,islice,...]*fa_corr)
    cos_phi = np.cos(self.fa[:,islice,...]*fa_corr)

    S = x[0,:,:]*self.M0_sc*(-E1 + 1)*sin_phi/(-E1*cos_phi + 1)
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)
    return S
  def execute_gradient_2D(self,x,islice):
#    E1 = np.exp(-self.TR/(x[1,:,:]*self.T1_sc))  
    E1 = x[1,...]*self.T1_sc
    M0 = x[0,...]
    fa_corr = x[2,...]*self.fa_scale
    E1[~np.isfinite(E1)] = 0
    sin_phi = np.sin(self.fa[:,islice,...]*fa_corr)
    cos_phi = np.cos(self.fa[:,islice,...]*fa_corr)
#    grad_M0 = self.M0_sc*(1 - E1)*self.sin_phi[:,islice,:,:]/(1 -E1*self.cos_phi[:,islice,:,:])
#    grad_T1 = -x[0,...]*self.M0_sc*self.TR*E1*self.sin_phi[:,islice,:,:]/\
#    (x[1,...]**2*self.T1_sc*(1 - E1*self.cos_phi[:,islice,:,:])) + x[0,...]*self.M0_sc*self.TR*\
#    (1 - E1)*E1*self.sin_phi[:,islice,:,:]*self.cos_phi[:,islice,:,:]/(x[1,...]**2*self.T1_sc*(1 - E1*self.cos_phi[:,islice,:,:])**2)
    grad_M0 = self.M0_sc*(-E1 + 1)*sin_phi/(-E1*cos_phi + 1)
    grad_T1 = M0*self.M0_sc*self.T1_sc*(-E1 + 1)*sin_phi*cos_phi/(-E1*cos_phi + 1)**2 -\
    M0*self.M0_sc*self.T1_sc*sin_phi/(-E1*cos_phi + 1)
    grad_FA = -M0*self.M0_sc*E1*self.fa[:,islice,...]*self.fa_scale*(-E1 + 1)*sin_phi**2/\
              (-E1*cos_phi + 1)**2 + M0*self.M0_sc*self.fa[:,islice,...]*self.fa_scale*(-E1 + 1)*cos_phi/(-E1*cos_phi + 1)
    grad = np.array([grad_M0,grad_T1,grad_FA],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
    return grad
  
  def execute_forward_3D(self,x):
#    E1 = np.exp(-self.TR/(x[1,:,:]*self.T1_sc))
    E1 = x[1,...]*self.T1_sc
    fa_corr = x[2,...]*self.fa_scale
    E1[~np.isfinite(E1)] = 0
    sin_phi = np.sin(self.fa*fa_corr)
    cos_phi = np.cos(self.fa*fa_corr)

    S = x[0,:,:]*self.M0_sc*(-E1 + 1)*sin_phi/(-E1*cos_phi + 1)
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)
    return S
  def execute_gradient_3D(self,x):
#    E1 = np.exp(-self.TR/(x[1,:,:]*self.T1_sc))  
    E1 = x[1,...]*self.T1_sc
    M0 = x[0,...]
    fa_corr = x[2,...]*self.fa_scale
    E1[~np.isfinite(E1)] = 0
    sin_phi = np.sin(self.fa*fa_corr)
    cos_phi = np.cos(self.fa*fa_corr)
#    grad_M0 = self.M0_sc*(1 - E1)*self.sin_phi[:,islice,:,:]/(1 -E1*self.cos_phi[:,islice,:,:])
#    grad_T1 = -x[0,...]*self.M0_sc*self.TR*E1*self.sin_phi[:,islice,:,:]/\
#    (x[1,...]**2*self.T1_sc*(1 - E1*self.cos_phi[:,islice,:,:])) + x[0,...]*self.M0_sc*self.TR*\
#    (1 - E1)*E1*self.sin_phi[:,islice,:,:]*self.cos_phi[:,islice,:,:]/(x[1,...]**2*self.T1_sc*(1 - E1*self.cos_phi[:,islice,:,:])**2)
    grad_M0 = self.M0_sc*(-E1 + 1)*sin_phi/(-E1*cos_phi + 1)
    grad_T1 = M0*self.M0_sc*self.T1_sc*(-E1 + 1)*sin_phi*cos_phi/(-E1*cos_phi + 1)**2 -\
    M0*self.M0_sc*self.T1_sc*sin_phi/(-E1*cos_phi + 1)
    grad_FA = -M0*self.M0_sc*E1*self.fa*self.fa_scale*(-E1 + 1)*sin_phi**2/\
              (-E1*cos_phi + 1)**2 + M0*self.M0_sc*self.fa*self.fa_scale*(-E1 + 1)*cos_phi/(-E1*cos_phi + 1)
    grad = np.array([grad_M0,grad_T1,grad_FA],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
    return grad
  
  

  def execute_forward_2D_ns(self,x,islice):
    E1 = np.exp(-self.TR/(x[1,:,:]))#
#    E1 = x[1,:,:]
    S = x[0,:,:]*self.sin_phi[:,islice,:,:]*(1-E1)/(1-E1*self.cos_phi[:,islice,:,:])
    S[~np.isfinite(S)] = 1e-20
    return S
  def execute_gradient_2D_ns(self,x,islice):
    E1 = np.exp(self.TR/(x[1,:,:]))  ####no minus!!!  
#    E1 = x[1,:,:]
#    E1[~np.isfinite(E1)] = 0
    grad_M0 = (self.sin_phi[:,islice,:,:]*(E1-1))/(E1-self.cos_phi[:,islice,:,:])
    grad_T1 = (-(x[0,:,:]*self.TR*E1*(2*self.sin_phi[:,islice,:,:]-2*self.sin_phi[:,islice,:,:]*self.cos_phi[:,islice,:,:]))/
               (2*x[1,:,:]**2*(E1-self.cos_phi[:,islice,:,:])**2))
#    grad_M0 = (self.M0_sc*self.sin_phi[:,islice,:,:]*(1 - E1))/(-self.cos_phi[:,islice,:,:]*E1 + 1)
#    grad_T1 = (x[0,:,:]*self.M0_sc*self.sin_phi[:,islice,:,:]*(self.cos_phi[:,islice,:,:] - 1))/(E1*self.cos_phi[:,islice,:,:] - 1)**2
    grad = np.array([grad_M0,grad_T1])
    grad[~np.isfinite(grad)] = 1e-20
    return grad   
  def plot_unknowns(self,x,dim_2D=False):
      
      if dim_2D:
          plt.figure(1)
          plt.imshow(np.transpose(np.abs(x[0,...]*self.M0_sc)))
          plt.pause(0.05)
          plt.figure(2)
          plt.imshow(np.transpose(np.abs(-self.TR/np.log(x[1,...]*self.T1_sc))))
          plt.pause(0.05)
          plt.figure(3)          
          plt.imshow(np.transpose(np.abs(x[2,...]*self.fa_scale)))
          plt.pause(0.05)          
      else:         
          plt.figure(1)
          plt.imshow(np.transpose(np.abs(x[0,int(self.Nislice/2),...]*self.M0_sc)))
          plt.pause(0.05)
          plt.figure(2)
          plt.imshow(np.transpose(np.abs(-self.TR/np.log(x[1,int(self.Nislice/2),...]*self.T1_sc))))
#          plt.imshow(np.transpose(np.abs(x[1,int(self.Nislice/2),...]*self.T1_sc)))
          plt.pause(0.05)
           
           
             
