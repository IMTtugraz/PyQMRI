#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:33:09 2017

@author: omaier
"""
# cython: profile=False
# filename: IRLL_Model.pyx
#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
np.import_array()
import matplotlib.pyplot as plt
plt.ion()

DTYPE = np.complex64


class IRLL_Model:
  
  
  def __init__(self, fa, fa_corr, TR,tau,td,
               NScan,NSlice,dimY,dimX, Nproj):


    self.NSlice = NSlice
    self.TR = TR
    self.fa = fa
    self.fa_corr = fa_corr
    
    self.T1_sc = 5000#5000
    self.M0_sc = 1#50
    
    self.tau = tau
    self.td = td
    self.NLL = NScan
    self.Nproj = Nproj
    self.dimY = dimY
    self.dimX = dimX
    
#    phi_corr = np.zeros_like(fa_corr,dtype='complex128')
    phi_corr = np.real(fa)*np.real(fa_corr) + 1j*np.imag(fa)*np.imag(fa_corr)

    
    self.sin_phi = np.sin(phi_corr)
    self.cos_phi = np.cos(phi_corr)    

    self.guess = np.array([0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),1e-5/self.T1_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE)])               
    self.min_T1 = 50/self.T1_sc
    self.max_T1 = 5000/self.T1_sc

  def execute_forward_2D(self, x, islice):
    S = np.zeros((self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
    sin_phi = self.sin_phi
    cos_phi = self.cos_phi    
    N = self.NLL
    T1 = x[1,...]
    M0 = x[0,...]
    for i in range(self.NLL):
      cosEtau = cos_phi[islice,...]*np.exp(-tau/(T1*T1_sc))        
      cosEtauN = cosEtau**(N-1)           
      Etr = np.exp(-TR/(T1*T1_sc))
      Etd = np.exp(-td/(T1*T1_sc))
      Etau = np.exp(-tau/(T1*T1_sc))
      F = (1 - Etau)/(-cosEtau + 1)
      Q = (-cos_phi[islice,...]*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr)/(cos_phi[islice,...]*cosEtauN*Etr*Etd + 1)
      Q_F = Q-F
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            S[i,j,...] = M0*M0_sc*sin_phi[islice,...]*((cosEtau)**(n - 1)*(Q_F) + F)
    
    return np.mean(S,axis=1)
  def execute_gradient_2D(self, x, islice):
    grad = np.zeros((2,self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
    sin_phi = self.sin_phi
    cos_phi = self.cos_phi    
    N = self.NLL  
    T1 = x[1,...]
    ####Precompute
    cosEtau = cos_phi[islice,...]*exp(-tau/(T1*T1_sc))        
    cosEtauN = cosEtau**(N-1)           
    Etr = exp(-TR/(T1*T1_sc))
    Etd = exp(-td/(T1*T1_sc))
    Etau = exp(-tau/(T1*T1_sc))
    F = (1 - Etau)/(-cosEtau + 1)
    Q = (-cos_phi[islice,...]*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr)/(cos_phi[islice,...]*cosEtauN*Etr*Etd + 1)
    Q_F = Q-F
    tmp1 = ((-TR*cos*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(T1**2*T1_sc*(-cosEtau + 1)) + TR*Etr/(T1**2*T1_sc) -\
              cos**2*tau*(1 - Etau)*(-cosEtauN + 1)*Etr*Etau*Etd/(T1**2*T1_sc*(-cosEtau + 1)**2) + \
              cos*tau*cosEtauN*(1 - Etau)*(N - 1)*Etr*Etd/(T1**2*T1_sc*(-cosEtau + 1)) + \
              cos*tau*(-cosEtauN + 1)*Etr*Etau*Etd/(T1**2*T1_sc*(-cosEtau + 1)) - cos*td*(1 - Etau)\
              *(-cosEtauN + 1)*Etr*Etd/(T1**2*T1_sc*(-cosEtau + 1)) - 2*td*Etd/(T1**2*T1_sc))/\
              (cos*cosEtauN*Etr*Etd + 1) + (-TR*cos*cosEtauN*Etr*Etd/(T1**2*T1_sc) -\
               cos*tau*cosEtauN*(N - 1)*Etr*Etd/(T1**2*T1_sc) - cos*td*cosEtauN*Etr*Etd/(T1**2*T1_sc))*\
               (-cos*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(-cosEtau + 1) + 1 - 2*Etd + Etr)/(cos*cosEtauN*Etr*\
                   Etd + 1)**2 - cos*tau*(1 - Etau)*Etau/(T1**2*T1_sc*(-cosEtau + 1)**2) + tau*Etau/(T1**2*T1_sc*(-cosEtau + 1)))
    tmp2 = cos*tau*(1 - Etau)*Etau/(T1**2*T1_sc*(-cosEtau + 1)**2) + \
                tau*(cosEtau)**(n - 1)*(n - 1)*(-(1 - Etau)/\
                (-cosEtau + 1) + (-cos*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(-cosEtau + 1) + 1 - 2*Etd +\
                Etr)/(cos*cosEtauN*Etr*Etd + 1))/(T1**2*T1_sc) - tau*Etau/(T1**2*T1_sc*(-cosEtau + 1))        
    
    
    for i in range(NLL):  
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            
            grad[0,i,j,...] = M0_sc*sin_phi[islice,...]*((cosEtau)**(n - 1)*(Q_F) + F)
            
            grad[1,i,j,...] =x[0,...]*M0_sc*sin_phi[islice,...]*((cosEtau)**(n - 1)*tmp1 + tmp2)
            
    return np.mean(grad,axis=2)
  
  
  def execute_forward_3D(self, x):
    S = np.zeros((self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
    sin_phi = self.sin_phi
    cos_phi = self.cos_phi    
    N = self.NLL
    for i in range(self.NLL):
      cosEtau = cos_phi*exp(-tau/(T1*T1_sc))        
      cosEtauN = cosEtau**(N-1)           
      Etr = exp(-TR/(T1*T1_sc))
      Etd = exp(-td/(T1*T1_sc))
      Etau = exp(-tau/(T1*T1_sc))
      F = (1 - Etau)/(-cosEtau + 1)
      Q = (-cos_phi*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr)/(cos_phi*cosEtauN*Etr*Etd + 1)
      Q_F = Q-F
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            S[i,j,...] = x[0,...]*M0_sc*sin_phi*((cosEtau)**(n - 1)*(Q_F) + F)
    
    return np.mean(S,axis=1)
#    return np.average(np.array(S),axis=1,weights=self.calc_weights(x[1,:,:,:]))
  def execute_gradient_3D(self, x):
    grad = np.zeros((2,self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
    sin_phi = self.sin_phi
    cos_phi = self.cos_phi    
    N = self.NLL  
    
    ####Precompute
    cosEtau = cos_phi*exp(-tau/(T1*T1_sc))        
    cosEtauN = cosEtau**(N-1)           
    Etr = exp(-TR/(T1*T1_sc))
    Etd = exp(-td/(T1*T1_sc))
    Etau = exp(-tau/(T1*T1_sc))
    F = (1 - Etau)/(-cosEtau + 1)
    Q = (-cos_phi*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr)/(cos_phi*cosEtauN*Etr*Etd + 1)
    Q_F = Q-F
    tmp1 = ((-TR*cos*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(T1**2*T1_sc*(-cosEtau + 1)) + TR*Etr/(T1**2*T1_sc) -\
              cos**2*tau*(1 - Etau)*(-cosEtauN + 1)*Etr*Etau*Etd/(T1**2*T1_sc*(-cosEtau + 1)**2) + \
              cos*tau*cosEtauN*(1 - Etau)*(N - 1)*Etr*Etd/(T1**2*T1_sc*(-cosEtau + 1)) + \
              cos*tau*(-cosEtauN + 1)*Etr*Etau*Etd/(T1**2*T1_sc*(-cosEtau + 1)) - cos*td*(1 - Etau)\
              *(-cosEtauN + 1)*Etr*Etd/(T1**2*T1_sc*(-cosEtau + 1)) - 2*td*Etd/(T1**2*T1_sc))/\
              (cos*cosEtauN*Etr*Etd + 1) + (-TR*cos*cosEtauN*Etr*Etd/(T1**2*T1_sc) -\
               cos*tau*cosEtauN*(N - 1)*Etr*Etd/(T1**2*T1_sc) - cos*td*cosEtauN*Etr*Etd/(T1**2*T1_sc))*\
               (-cos*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(-cosEtau + 1) + 1 - 2*Etd + Etr)/(cos*cosEtauN*Etr*\
                   Etd + 1)**2 - cos*tau*(1 - Etau)*Etau/(T1**2*T1_sc*(-cosEtau + 1)**2) + tau*Etau/(T1**2*T1_sc*(-cosEtau + 1)))
    tmp2 = cos*tau*(1 - Etau)*Etau/(T1**2*T1_sc*(-cosEtau + 1)**2) + \
                tau*(cosEtau)**(n - 1)*(n - 1)*(-(1 - Etau)/\
                (-cosEtau + 1) + (-cos*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(-cosEtau + 1) + 1 - 2*Etd +\
                Etr)/(cos*cosEtauN*Etr*Etd + 1))/(T1**2*T1_sc) - tau*Etau/(T1**2*T1_sc*(-cosEtau + 1))        
    
    
    for i in range(NLL):  
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            
            grad[0,i,j,...] = M0_sc*sin_phi*((cosEtau)**(n - 1)*(Q_F) + F)
            
            grad[1,i,j,...] =x[0,...]*M0_sc*sin_phi*((cosEtau)**(n - 1)*tmp1 + tmp2)
            
    return np.mean(grad,axis=2)
                                         
#    return np.average(np.array(grad),axis=2,weights=np.tile(self.calc_weights(x[1,:,:,:]),(2,1,1,1,1,1)))


           
#  cpdef calc_weights(self,DTYPE_t[:,:,::1] x):
#      cdef int i=0,j=0
#      cdef np.ndarray[ndim=3,dtype=DTYPE_t] T1 = np.array(x)*self.T1_sc
#      cdef np.ndarray[ndim=3,dtype=DTYPE_t] w = np.zeros_like(T1)
#      cdef np.ndarray[ndim=3,dtype=DTYPE_t] V = np.ones_like(T1)
#      cdef np.ndarray[ndim=5,dtype=DTYPE_t] result = np.zeros((self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
#      for i in range(self.NLL):
##           w = 1-np.exp(-(self.tau)/T1)
##           V[~(w==1)] = (1-w[~(w==1)]**self.Nproj)/(1-w[~(w==1)])
#           for j in range(self.Nproj):
#               result[i,j,:,:,:] = np.exp(-(self.td+self.tau*j+self.tau*self.Nproj*i)/T1)/np.exp(-(self.td+self.tau*self.Nproj*i)/T1)
#           result[i,:,:,:,:] = result[i,:,:,:,:]/np.sum(result[i,:,:,:,:],0)
#            
#      return np.squeeze(result)
  def plot_unknowns(self,x,dim_2D=False):
      
      if dim_2D:
          plt.figure(1)
          plt.imshow(np.transpose(np.abs(x[0,...]*self.M0_sc)))
          plt.pause(0.05)
          plt.figure(2)
          plt.imshow(np.transpose(np.abs(x[1,...]*self.T1_sc)))
        #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
          plt.pause(0.05)          
      else:         
          plt.figure(1)
          plt.imshow(np.transpose(np.abs(x[0,int(self.NSlice/2),...]*self.M0_sc)))
          plt.pause(0.05)
          plt.figure(2)
          plt.imshow(np.transpose(np.abs(x[1,int(self.NSlice/2),...]*self.T1_sc)))
        #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
          plt.pause(0.05)
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           