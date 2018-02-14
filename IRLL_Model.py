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
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt
import numexpr as ne
plt.ion()

DTYPE = np.complex64
class constraint:
  def __init__(self, min_val=-np.inf, max_val=np.inf, real_const=False):
    self.min = min_val
    self.max = max_val
    self.real = real_const

class IRLL_Model:
  
  
  def __init__(self, fa, fa_corr, TR,tau,td,
               NScan,NSlice,dimY,dimX, Nproj,Nproj_measured,scale):

    self.constraints = []
    self.NSlice = NSlice
    self.TR = TR
    self.fa = fa
    self.fa_corr = fa_corr
    
    self.T1_sc = 3000#5000
    self.M0_sc = 1#50
    self.Nproj_measured = Nproj_measured
    
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

    self.guess = np.array([1e-5/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),1500/self.T1_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE)])               
    self.constraints.append(constraint(-300,300,False)  )
    self.constraints.append(constraint(10/self.T1_sc, 5500/self.T1_sc,True))

  def execute_forward_2D(self, x, islice):
    S = np.zeros((self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
    sin_phi = self.sin_phi[islice,...]
    cos_phi = self.cos_phi[islice,...]   
    N = self.Nproj_measured
    T1 = x[1,...]
    M0 = x[0,...]          
    Etr = np.exp(-TR/(T1*T1_sc))
    Etd = np.exp(-td/(T1*T1_sc))
    Etau = np.exp(-tau/(T1*T1_sc))
    cosEtau = cos_phi*Etau        
    cosEtauN = cosEtau**(N-1)   
    F = (1 - Etau)/(-cosEtau + 1)
    Q = (-cos_phi*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr*Etd)/(cos_phi*cosEtauN*Etr*Etd + 1)
    Q_F = Q-F
    for i in range(self.NLL):
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            S[i,j,...] = M0*M0_sc*sin_phi*((cosEtau)**(n - 1)*(Q_F) + F)
    
    return np.mean(S,axis=1)
  def execute_gradient_2D(self, x, islice):
    grad = np.zeros((2,self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
    sin_phi = self.sin_phi[islice,...]
    cos_phi = self.cos_phi[islice,...]    
    N = self.Nproj_measured
    T1 = x[1,...]
    M0 = x[0,...]
    ####Precompute
    Etr = np.exp(-TR/(T1*T1_sc))
    Etd = np.exp(-td/(T1*T1_sc))
    Etau = np.exp(-tau/(T1*T1_sc))
    cosEtau = cos_phi*Etau        
    cosEtauN = cosEtau**(N-1)       
    F = (1 - Etau)/(-cosEtau + 1)
    Q = (-cos_phi*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr*Etd)/(cos_phi*cosEtauN*Etr*Etd + 1)
    Q_F = Q-F
    tmp1 = ((TR*Etr*Etd/(T1**2*T1_sc) - TR*(1 - Etau)*\
                (-(Etau*cos_phi)**(N - 1) + 1)*Etr*Etd*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)) + tau*(Etau*cos_phi)**(N - 1)*\
                (1 - Etau)*(N - 1)*Etr*Etd*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)) + tau*(-(Etau*cos_phi)**(N - 1) + 1)*\
                Etr*Etau*Etd*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)) - tau*(1 - Etau)*(-(Etau*cos_phi)**(N - 1) + 1)\
                *Etr*Etau*Etd*cos_phi**2/(T1**2*T1_sc*(1 - Etau*cos_phi)**2) - 2*td*Etd/(T1**2*T1_sc) + td*Etr*Etd/(T1**2*T1_sc)\
                - td*(1 - Etau)*(-(Etau*cos_phi)**(N - 1) + 1)*Etr*Etd*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)))/\
      ((Etau*cos_phi)**(N - 1)*Etr*Etd*cos_phi + 1) + (-TR*(Etau*cos_phi)**(N - 1)*Etr*Etd*cos_phi/(T1**2*T1_sc) - \
       tau*(Etau*cos_phi)**(N - 1)*(N - 1)*Etr*Etd*cos_phi/(T1**2*T1_sc) - td*(Etau*cos_phi)**(N - 1)*Etr*Etd*cos_phi/\
       (T1**2*T1_sc))*(1 - 2*Etd + Etr*Etd - (1 - Etau)*(-(Etau*cos_phi)**(N - 1) + 1)*Etr*Etd*cos_phi/(1 - Etau*cos_phi))/\
       ((Etau*cos_phi)**(N - 1)*Etr*Etd*cos_phi + 1)**2 + tau*Etau/(T1**2*T1_sc*(1 - Etau*cos_phi)) - \
       tau*(1 - Etau)*Etau*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)**2))
    tmp2 = ((1 - 2*Etd + Etr*Etd - (1 - Etau)*(-(Etau*cos_phi)**(N - 1) + 1)*Etr*Etd*cos_phi/(1 - Etau*cos_phi))/((Etau*cos_phi)**\
         (N - 1)*Etr*Etd*cos_phi + 1) - (1 - Etau)/(1 - Etau*cos_phi))/(T1**2*T1_sc)
    tmp3 = - tau*Etau/(T1**2*T1_sc*(1 - Etau*cos_phi))\
         + tau*(1 - Etau)*Etau*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)**2)
                
    
    
    for i in range(self.NLL):  
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            
            grad[0,i,j,...] = M0_sc*sin_phi*((cosEtau)**(n - 1)*(Q_F) + F)
            
            grad[1,i,j,...] =M0*M0_sc*((Etau*cos_phi)**(n - 1)*tmp1 + tau*(Etau*cos_phi)**(n - 1)*(n - 1)*\
       tmp2 +tmp3)*sin_phi
            
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
    N = self.Nproj_measured
    T1 = x[1,...]
    M0 = x[0,...]          
    Etr = np.exp(-TR/(T1*T1_sc))
    Etd = np.exp(-td/(T1*T1_sc))
    Etau = np.exp(-tau/(T1*T1_sc))
    cosEtau = cos_phi*Etau        
    cosEtauN = cosEtau**(N-1)   
    F = (1 - Etau)/(-cosEtau + 1)
    Q = (-cos_phi*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr*Etd)/(cos_phi*cosEtauN*Etr*Etd + 1)
    Q_F = Q-F
    def numexpeval_S(M0,M0_sc,sin_phi,cosEtau,n,Q_F,F):
      return ne.evaluate("M0*M0_sc*sin_phi*((cosEtau)**(n - 1)*(Q_F) + F)")    
    for i in range(self.NLL):
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            S[i,j,...] = numexpeval_S(M0,M0_sc,sin_phi,cosEtau,n,Q_F,F)
    
    return np.mean(S,axis=1)
  
  def execute_gradient_3D(self, x):
    grad = np.zeros((2,self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
    sin_phi = self.sin_phi
    cos_phi = self.cos_phi
    N = self.Nproj_measured
    T1 = x[1,...]
    M0 = x[0,...]
    ####Precompute
    Etr = np.exp(-TR/(T1*T1_sc))
    Etd = np.exp(-td/(T1*T1_sc))
    Etau = np.exp(-tau/(T1*T1_sc))
    cosEtau = cos_phi*Etau        
    cosEtauN = cosEtau**(N-1)       
    F = (1 - Etau)/(-cosEtau + 1)
    Q = (-cos_phi*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr*Etd)/(cos_phi*cosEtauN*Etr*Etd + 1)
    Q_F = Q-F
    tmp1 = ((TR*Etr*Etd/(T1**2*T1_sc) - TR*(1 - Etau)*\
                (-(Etau*cos_phi)**(N - 1) + 1)*Etr*Etd*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)) + tau*(Etau*cos_phi)**(N - 1)*\
                (1 - Etau)*(N - 1)*Etr*Etd*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)) + tau*(-(Etau*cos_phi)**(N - 1) + 1)*\
                Etr*Etau*Etd*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)) - tau*(1 - Etau)*(-(Etau*cos_phi)**(N - 1) + 1)\
                *Etr*Etau*Etd*cos_phi**2/(T1**2*T1_sc*(1 - Etau*cos_phi)**2) - 2*td*Etd/(T1**2*T1_sc) + td*Etr*Etd/(T1**2*T1_sc)\
                - td*(1 - Etau)*(-(Etau*cos_phi)**(N - 1) + 1)*Etr*Etd*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)))/\
      ((Etau*cos_phi)**(N - 1)*Etr*Etd*cos_phi + 1) + (-TR*(Etau*cos_phi)**(N - 1)*Etr*Etd*cos_phi/(T1**2*T1_sc) - \
       tau*(Etau*cos_phi)**(N - 1)*(N - 1)*Etr*Etd*cos_phi/(T1**2*T1_sc) - td*(Etau*cos_phi)**(N - 1)*Etr*Etd*cos_phi/\
       (T1**2*T1_sc))*(1 - 2*Etd + Etr*Etd - (1 - Etau)*(-(Etau*cos_phi)**(N - 1) + 1)*Etr*Etd*cos_phi/(1 - Etau*cos_phi))/\
       ((Etau*cos_phi)**(N - 1)*Etr*Etd*cos_phi + 1)**2 + tau*Etau/(T1**2*T1_sc*(1 - Etau*cos_phi)) - \
       tau*(1 - Etau)*Etau*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)**2))
    tmp2 = ((1 - 2*Etd + Etr*Etd - (1 - Etau)*(-(Etau*cos_phi)**(N - 1) + 1)*Etr*Etd*cos_phi/(1 - Etau*cos_phi))/((Etau*cos_phi)**\
         (N - 1)*Etr*Etd*cos_phi + 1) - (1 - Etau)/(1 - Etau*cos_phi))/(T1**2*T1_sc)
    tmp3 = - tau*Etau/(T1**2*T1_sc*(1 - Etau*cos_phi))\
         + tau*(1 - Etau)*Etau*cos_phi/(T1**2*T1_sc*(1 - Etau*cos_phi)**2)
                
    def numexpeval_M0(M0_sc,sin_phi,cosEtau,n,Q_F,F):
      return ne.evaluate("M0_sc*sin_phi*((cosEtau)**(n - 1)*(Q_F) + F)")
    def numexpeval_T1(M0,M0_sc,Etau,cos_phi,sin_phi,cosEtau,n,Q_F,F,tmp1,tmp2,tmp3,tau):
      return ne.evaluate("M0*M0_sc*((Etau*cos_phi)**(n - 1)*tmp1 + tau*(Etau*cos_phi)**(n - 1)*(n - 1)*\
                          tmp2 +tmp3)*sin_phi")
    
    for i in range(self.NLL):  
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            
            grad[0,i,j,...] = numexpeval_M0(M0_sc,sin_phi,cosEtau,n,Q_F,F)
            
            grad[1,i,j,...] = numexpeval_T1(M0,M0_sc,Etau,cos_phi,sin_phi,cosEtau,n,Q_F,F,tmp1,tmp2,tmp3,tau)
            
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
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
