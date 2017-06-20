#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:33:09 2017

@author: omaier
"""
import numpy as np



class IRLL_Model:
  def __init__(self,fa,fa_corr,TR,tau,td,NScan,NSlice,dimY,dimX,Nproj):



    self.TR = TR
    self.fa = fa
    self.fa_corr = fa_corr
    
    self.T1_sc = 5000
    self.M0_sc = 50
    
    self.tau = tau
    self.td = td
    self.NLL = NScan
    self.Nproj = Nproj
    self.dimY = dimY
    self.dimX = dimX
    
#    phi_corr = np.zeros_like(fa_corr,dtype='complex128')
    phi_corr = fa*fa_corr
    
    self.sin_phi = np.sin(phi_corr)
    self.cos_phi = np.cos(phi_corr)    

    self.guess = np.array([0.05*np.ones((NSlice,dimY,dimX)),0.3*np.ones((NSlice,dimY,dimX))])               
    
    
  def execute_forward_2D(self,x,islice):
    S = np.zeros((self.NLL,self.Nproj,self.dimY,self.dimX),dtype='complex128')
    for i in range(self.NLL):
      for j in range(self.Nproj):
        n_i = i*self.Nproj+j+1
        S[i,j,:,:] = (x[0,:,:]*self.M0_sc*self.sin_phi[islice,:,:]*(((np.exp(-self.TR/(x[1,:,:]*self.T1_sc)) - 
                    2*np.exp(-self.td/(x[1,:,:]*self.T1_sc)) + (np.exp(-self.TR/(x[1,:,:]*self.T1_sc))
                    *np.exp(-self.td/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:]*((np.exp(-self.tau/(x[1,:,:]
                    *self.T1_sc))*self.cos_phi[islice,:,:])**(self.NLL - 1) - 1)*(np.exp(-self.tau/(x[1,:,:]*
                    self.T1_sc)) - 1))/(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:] - 1)
                    + 1)/(np.exp(-self.TR/(x[1,:,:]*self.T1_sc))*np.exp(-self.td/(x[1,:,:]*self.T1_sc))
                    *self.cos_phi[islice,:,:]*(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))
                    *self.cos_phi[islice,:,:])**(self.NLL - 1) + 1) - (np.exp(-self.tau/(x[1,:,:]*self.T1_sc)) 
                    - 1)/(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:] - 1))*
                    (np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:])**(n_i - 1) 
                    + (np.exp(-self.tau/(x[1,:,:]*self.T1_sc)) - 1)/(np.exp(-self.tau/(x[1,:,:]
                    *self.T1_sc))*self.cos_phi[islice,:,:] - 1)))
  
    S[~np.isfinite(S)] = 1e-20
    return S
  def execute_gradient_2D(self,x,islice):
    grad = np.zeros((2,self.NLL,self.Nproj,self.dimY,self.dimX),dtype='complex128')
    for i in range(self.NLL):
      for j in range(self.Nproj):
        n_i = i*self.Nproj+j+1
        grad[0,i,j,:,:] = (self.M0_sc*self.sin_phi[islice,:,:]*(((np.exp(
                          -self.TR/(x[1,:,:]*self.T1_sc)) - 2*np.exp(-self.td
                          /(x[1,:,:]*self.T1_sc)) + (np.exp(-self.TR/(x[1,:,:]*self.T1_sc))
                          *np.exp(-self.td/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:]
                          *((np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:])
                          **(self.NLL - 1) - 1)*(np.exp(-self.tau/(x[1,:,:]*self.T1_sc)) - 1))
                          /(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:] - 1)
                          + 1)/(np.exp(-self.TR/(x[1,:,:]*self.T1_sc))*np.exp(-self.td/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:]*(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:])**(self.NLL - 1) + 1) - (np.exp(-self.tau/(x[1,:,:]
                          *self.T1_sc))- 1)/(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:] - 1))*(np.exp(-self.tau/(x[1,:,:]
                          *self.T1_sc))*self.cos_phi[islice,:,:])**(n_i - 1) 
                          + (np.exp(-self.tau/(x[1,:,:]*self.T1_sc)) - 1)/(np.exp(-self.tau/(x[1,:,:]
                          *self.T1_sc))*self.cos_phi[islice,:,:] - 1)))
      
        grad[1,i,j,:,:] = (x[0,:,:]*self.M0_sc*self.sin_phi[islice,:,:]
                          *((np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:])
                          **(n_i - 1)*(((self.TR*np.exp(-self.TR/(x[1,:,:]*self.T1_sc)))
                          /(x[1,:,:]**2*self.T1_sc) - (2*self.td*np.exp(-self.td/(x[1,:,:]
                          *self.T1_sc)))/(x[1,:,:]**2*self.T1_sc) + (self.tau*np.exp(-(self.TR 
                          + self.tau + self.td)/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:]
                          *((np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:])
                          **(self.NLL - 1) - 1))/(x[1,:,:]**2*self.T1_sc*(np.exp(-self.tau
                          /(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:] - 1)) 
                          - (self.tau*np.exp(-(self.TR + self.tau + self.td)/(x[1,:,:]
                          *self.T1_sc))*self.cos_phi[islice,:,:]**2*((np.exp(-self.tau
                          /(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:])**(self.NLL - 1) - 1)
                          *(np.exp(-self.tau/(x[1,:,:]*self.T1_sc)) - 1))/(x[1,:,:]**2*self.T1_sc
                          *(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:] - 1)**2)
                          + (self.TR*np.exp(-(self.TR + self.td)/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:]*((np.exp(-self.tau/(x[1,:,:]
                          *self.T1_sc))*self.cos_phi[islice,:,:])**(self.NLL - 1) - 1)
                          *(np.exp(-self.tau/(x[1,:,:]*self.T1_sc)) - 1))/(x[1,:,:]
                          **2*self.T1_sc*(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:] - 1)) + (self.td
                          *np.exp(-(self.TR + self.td)/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:]*((np.exp(-self.tau/(x[1,:,:]
                          *self.T1_sc))*self.cos_phi[islice,:,:])**(self.NLL - 1) - 1)
                          *(np.exp(-self.tau/(x[1,:,:]*self.T1_sc)) - 1))
                          /(x[1,:,:]**2*self.T1_sc*(np.exp(-self.tau/(x[1,:,:]
                          *self.T1_sc))*self.cos_phi[islice,:,:] - 1)) + (self.tau
                          *np.exp(-(self.TR + self.tau + self.td)/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:]**2*(self.NLL - 1)*(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:])**(self.NLL - 2)*(np.exp(-self.tau
                          /(x[1,:,:]*self.T1_sc)) - 1))/(x[1,:,:]**2*self.T1_sc
                          *(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:] - 1)))
                          /(np.exp(-(self.TR + self.td)/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:]
                          *(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:])
                          **(self.NLL - 1) + 1) + self.tau/(x[1,:,:]**2*self.T1_sc
                          *(np.exp(self.tau/(x[1,:,:]*self.T1_sc)) - self.cos_phi[islice,:,:])) 
                          + (self.tau*np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:]
                          *(np.exp(-self.tau/(x[1,:,:]*self.T1_sc)) - 1))/(x[1,:,:]
                          **2*self.T1_sc*(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:] - 1)**2) - (np.exp(-(self.TR - self.tau + self.td)
                          /(x[1,:,:]*self.T1_sc))*(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:])**self.NLL*(self.TR - self.tau 
                          + self.td + self.NLL*self.tau)*(np.exp(-self.TR/(x[1,:,:]
                          *self.T1_sc)) - 2*np.exp(-self.td/(x[1,:,:]*self.T1_sc)) 
                          + (np.exp(-(self.TR + self.td)/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:]*((np.exp(-self.tau/(x[1,:,:]
                          *self.T1_sc))*self.cos_phi[islice,:,:])**(self.NLL - 1) - 1)
                          *(np.exp(-self.tau/(x[1,:,:]*self.T1_sc)) - 1))
                          /(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:] - 1) + 1))
                          /(x[1,:,:]**2*self.T1_sc*(np.exp(-(self.TR + self.td)
                          /(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:]
                          *(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:])
                          **(self.NLL - 1) + 1)**2)) - self.tau/(x[1,:,:]**2*self.T1_sc
                          *(np.exp(self.tau/(x[1,:,:]*self.T1_sc)) - self.cos_phi[islice,:,:])) 
                          - (self.tau*np.exp(-self.tau/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:]*(np.exp(-self.tau/(x[1,:,:]*self.T1_sc)) - 1))
                          /(x[1,:,:]**2*self.T1_sc*(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))
                          *self.cos_phi[islice,:,:] - 1)**2) + (self.tau*np.exp(-self.tau
                          /(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:]*((np.exp(-self.TR
                          /(x[1,:,:]*self.T1_sc)) - 2*np.exp(-self.td/(x[1,:,:]*self.T1_sc)) 
                          + (np.exp(-(self.TR + self.td)/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:]
                          *((np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:])
                          **(self.NLL - 1) - 1)*(np.exp(-self.tau/(x[1,:,:]*self.T1_sc)) - 1))
                          /(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:] - 1) + 1)
                          /(np.exp(-(self.TR + self.td)/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:]
                          *(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:])
                          **(self.NLL - 1) + 1) - (np.exp(self.tau/(x[1,:,:]*self.T1_sc)) - 1)
                          /(np.exp(self.tau/(x[1,:,:]*self.T1_sc)) - self.cos_phi[islice,:,:]))
                          *(n_i - 1)*(np.exp(-self.tau/(x[1,:,:]*self.T1_sc))*self.cos_phi[islice,:,:])
                          **(n_i - 2))/(x[1,:,:]**2*self.T1_sc)))                     
                            
                            
                                       
    return np.mean(grad,axis=2)
  
  

    
  