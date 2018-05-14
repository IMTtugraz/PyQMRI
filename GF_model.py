#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:39:42 2018

@author: omaier
"""

import numpy as np
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

plt.ion()
DTYPE = np.complex64

import ipyparallel as ipp



class constraint:
  def __init__(self, min_val=-np.inf, max_val=np.inf, real_const=False):
    self.min = min_val
    self.max = max_val
    self.real = real_const

class GF_Model:
  def __init__(self,fa,tau,images,NSlice):
    self.constraints = []    
    self.tau = tau
    self.fa = fa
    self.NSlice = NSlice
#    c = ipp.Client()
    
    (NScan,NSlice,dimX,dimY) = images.shape
    
    
    self.cos_phi = np.cos(fa)    
    
    fftlength = 32
    self.fftlength = 2*fftlength
    phi = np.arange(-1,1,1/fftlength)*np.pi
    z = np.exp(1j*phi)
    z = np.reshape(z,(2*fftlength,1,1,1))
    self.z = z
    


    self.M0_sc = 1
    self.T1_sc = 1
    self.T2_sc = 1

    test_T1 = np.reshape(np.linspace(10,5500,dimX*dimY*NSlice),(NSlice,dimX,dimY))
    test_T2 = np.reshape(np.linspace(1,5500,dimX*dimY*NSlice),(NSlice,dimX,dimY))
    test_M0 = 1#np.reshape(np.linspace(0,1,dimX*dimY*Nislice),(Nislice,dimX,dimY))
    G_x = self.execute_forward_3D(np.array([test_M0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),\
                                            1/self.T1_sc*np.exp(-self.tau/(test_T1*np.ones((NSlice,dimY,dimX),dtype=DTYPE))),\
                                            1/self.T2_sc*np.exp(-self.tau/(test_T2*np.ones((NSlice,dimY,dimX),dtype=DTYPE)))],dtype=DTYPE))
    self.M0_sc = self.M0_sc*np.mean(np.abs(images))/np.mean(np.abs(G_x))

#
    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),\
                                            1/self.T1_sc*np.exp(-self.tau/(test_T1*np.ones((NSlice,dimY,dimX),dtype=DTYPE))),\
                                            1/self.T2_sc*np.exp(-self.tau/(test_T2*np.ones((NSlice,dimY,dimX),dtype=DTYPE)))],dtype=DTYPE))
    self.T1_sc = self.T1_sc*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...]))
    self.T1_sc = self.T1_sc / self.M0_sc
    self.T2_sc = self.T2_sc*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[2,...]))
    self.T2_sc = self.T2_sc / self.M0_sc    
    
    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),\
                                            1/self.T1_sc*np.exp(-self.tau/(test_T1*np.ones((NSlice,dimY,dimX),dtype=DTYPE))),\
                                            1/self.T2_sc*np.exp(-self.tau/(test_T2*np.ones((NSlice,dimY,dimX),dtype=DTYPE)))],dtype=DTYPE))
    print('Grad Scaling T1', np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...])),
          'Grad Scaling T2', np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[2,...])))    


    print('T1 scale: ',self.T1_sc,
                              '/ T2_scale: ',self.T2_sc,
                              '/ M0_scale: ',self.M0_sc)
    
    result = np.array([0.5/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),\
                       1/self.T1_sc*np.exp(-self.tau/(1500*np.ones((NSlice,dimY,dimX),dtype=DTYPE))),\
                       1/self.T2_sc*np.exp(-self.tau/(1500*np.ones((NSlice,dimY,dimX),dtype=DTYPE)))],dtype=DTYPE)
    self.guess = result                   
    self.constraints.append(constraint(-300,300,False)  )
    self.constraints.append(constraint(np.exp(-self.tau/(50))/self.T1_sc,np.exp(-self.tau/(5500))/self.T1_sc,True))
    self.constraints.append(constraint(np.exp(-self.tau/(5))/self.T2_sc,np.exp(-self.tau/(5000))/self.T2_sc,True))
    
  def execute_forward_2D(self,x,islice):
    E1 = x[1,...]
    E2 = x[2,...]
    M0 = x[0,...]
    T1_sc = self.T1_sc
    T2_sc = self.T2_sc
    M0_sc = self.M0_sc
    z = self.z
    
    E1[~np.isfinite(E1)] = 0
    E2[~np.isfinite(E2)] = 0   
    
    S = (M0*M0_sc*((((E2*T2_sc*z + 1)*(E1*E2*T1_sc*T2_sc*z**2 - self.cos_phi*(E1*T1_sc + E2*T2_sc)*z + 1))/((E2*T2_sc*z - 1)*(E1*E2*T1_sc*T2_sc*z**2 + self.cos_phi*(E1*T1_sc - E2*T2_sc)*z - 1)))**(1/2) + 1))/2
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)
    S = np.fft.fft(S,axis=0)/self.fftlength

    return S
  def execute_gradient_2D(self,x,islice):
    E1 = x[1,...]
    E2 = x[2,...]
    M0 = x[0,...]
    T1_sc = self.T1_sc
    T2_sc = self.T2_sc
    M0_sc = self.M0_sc
    z = self.z
    
    E1[~np.isfinite(E1)] = 0
    E2[~np.isfinite(E2)] = 0

    grad_M0 = (M0_sc*((((E2*T2_sc*z + 1)*(E1*E2*T1_sc*T2_sc*z**2 - self.cos_phi*(E1*T1_sc + E2*T2_sc)*z + 1))/((E2*T2_sc*z - 1)*(E1*E2*T1_sc*T2_sc*z**2 + self.cos_phi*(E1*T1_sc - E2*T2_sc)*z - 1)))**(1/2) + 1))/2

    grad_T1 = -(M0*M0_sc*(((- E2*T1_sc*T2_sc*z**2 + T1_sc*self.cos_phi*z)*(E2*T2_sc*z + 1))/((E2*T2_sc*z - 1)*(E1*E2*T1_sc*T2_sc*z**2 + self.cos_phi*(E1*T1_sc - E2*T2_sc)*z - 1)) + ((E2*T1_sc*T2_sc*z**2 + T1_sc*self.cos_phi*z)*(E2*T2_sc*z + 1)*(E1*E2*T1_sc*T2_sc*z**2 - self.cos_phi*(E1*T1_sc + E2*T2_sc)*z + 1))/((E2*T2_sc*z - 1)*(E1*E2*T1_sc*T2_sc*z**2 + self.cos_phi*(E1*T1_sc - E2*T2_sc)*z - 1)**2)))/(4*(((E2*T2_sc*z + 1)*(E1*E2*T1_sc*T2_sc*z**2 - self.cos_phi*(E1*T1_sc + E2*T2_sc)*z + 1))/((E2*T2_sc*z - 1)*(E1*E2*T1_sc*T2_sc*z**2 + self.cos_phi*(E1*T1_sc - E2*T2_sc)*z - 1)))**(1/2))

    grad_T2 = (M0*M0_sc*T2_sc*z*(self.cos_phi - 1)*(E1*T1_sc*z + 1)*(E1*T1_sc*E2**2*T2_sc**2*z**3 - self.cos_phi*E2**2*T2_sc**2*z**2 + E1*T1_sc*self.cos_phi*z - 1))/(2*(E2*T2_sc*z - 1)**2*(E1*T1_sc*z*self.cos_phi - E2*T2_sc*z*self.cos_phi + E1*E2*T1_sc*T2_sc*z**2 - 1)**2*(-((E2*T2_sc*z + 1)*(E1*T1_sc*z*self.cos_phi + E2*T2_sc*z*self.cos_phi - E1*E2*T1_sc*T2_sc*z**2 - 1))/((E2*T2_sc*z - 1)*(E1*T1_sc*z*self.cos_phi - E2*T2_sc*z*self.cos_phi + E1*E2*T1_sc*T2_sc*z**2 - 1)))**(1/2))



    grad = np.array([grad_M0,grad_T1,grad_T2],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
    return grad
  
  def execute_forward_3D(self,x):
    E1 = x[1,...]
    E2 = x[2,...]
    M0 = x[0,...]
    T1_sc = self.T1_sc
    T2_sc = self.T2_sc
    M0_sc = self.M0_sc
    z = self.z
    
    E1[~np.isfinite(E1)] = 0
    E2[~np.isfinite(E2)] = 0   
    
    S = (M0*M0_sc*((((E2*T2_sc*z + 1)*(E1*E2*T1_sc*T2_sc*z**2 - self.cos_phi*(E1*T1_sc + E2*T2_sc)*z + 1))/((E2*T2_sc*z - 1)*(E1*E2*T1_sc*T2_sc*z**2 + self.cos_phi*(E1*T1_sc - E2*T2_sc)*z - 1)))**(1/2) + 1))/2
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)
    S = np.fft.fft(S,axis=0)

    return S
  def execute_gradient_3D(self,x):
    E1 = x[1,...]
    E2 = x[2,...]
    M0 = x[0,...]
    T1_sc = self.T1_sc
    T2_sc = self.T2_sc
    M0_sc = self.M0_sc
    z = self.z
    
    E1[~np.isfinite(E1)] = 0
    E2[~np.isfinite(E2)] = 0

    grad_M0 = (M0_sc*((((E2*T2_sc*z + 1)*(E1*E2*T1_sc*T2_sc*z**2 - self.cos_phi*(E1*T1_sc + E2*T2_sc)*z + 1))/((E2*T2_sc*z - 1)*(E1*E2*T1_sc*T2_sc*z**2 + self.cos_phi*(E1*T1_sc - E2*T2_sc)*z - 1)))**(1/2) + 1))/2

    grad_T1 = -(M0*M0_sc*(((- E2*T1_sc*T2_sc*z**2 + T1_sc*self.cos_phi*z)*(E2*T2_sc*z + 1))/((E2*T2_sc*z - 1)*(E1*E2*T1_sc*T2_sc*z**2 + self.cos_phi*(E1*T1_sc - E2*T2_sc)*z - 1)) + ((E2*T1_sc*T2_sc*z**2 + T1_sc*self.cos_phi*z)*(E2*T2_sc*z + 1)*(E1*E2*T1_sc*T2_sc*z**2 - self.cos_phi*(E1*T1_sc + E2*T2_sc)*z + 1))/((E2*T2_sc*z - 1)*(E1*E2*T1_sc*T2_sc*z**2 + self.cos_phi*(E1*T1_sc - E2*T2_sc)*z - 1)**2)))/(4*(((E2*T2_sc*z + 1)*(E1*E2*T1_sc*T2_sc*z**2 - self.cos_phi*(E1*T1_sc + E2*T2_sc)*z + 1))/((E2*T2_sc*z - 1)*(E1*E2*T1_sc*T2_sc*z**2 + self.cos_phi*(E1*T1_sc - E2*T2_sc)*z - 1)))**(1/2))

    grad_T2 = (M0*M0_sc*T2_sc*z*(self.cos_phi - 1)*(E1*T1_sc*z + 1)*(E1*T1_sc*E2**2*T2_sc**2*z**3 - self.cos_phi*E2**2*T2_sc**2*z**2 + E1*T1_sc*self.cos_phi*z - 1))/(2*(E2*T2_sc*z - 1)**2*(E1*T1_sc*z*self.cos_phi - E2*T2_sc*z*self.cos_phi + E1*E2*T1_sc*T2_sc*z**2 - 1)**2*(-((E2*T2_sc*z + 1)*(E1*T1_sc*z*self.cos_phi + E2*T2_sc*z*self.cos_phi - E1*E2*T1_sc*T2_sc*z**2 - 1))/((E2*T2_sc*z - 1)*(E1*T1_sc*z*self.cos_phi - E2*T2_sc*z*self.cos_phi + E1*E2*T1_sc*T2_sc*z**2 - 1)))**(1/2))

    
    
    grad = np.array([grad_M0,grad_T1,grad_T2],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
    return grad
  
 
  def plot_unknowns(self,x,dim_2D=False):
      
      if dim_2D:
          plt.figure(1)
          plt.imshow(np.transpose(np.abs(x[0,...]*self.M0_sc)))
          plt.pause(0.05)
          plt.figure(2)
          plt.imshow(np.transpose(np.abs(-self.tau/np.log(x[1,...]*self.T1_sc))))
          plt.pause(0.05)          
          plt.figure(3)
          plt.imshow(np.transpose(np.abs(-self.tau/np.log(x[2,...]*self.T2_sc))))
          plt.pause(0.05)           
      else:         
          plt.figure(1)
          plt.imshow(np.transpose(np.abs(x[0,int(self.Nislice/2),...]*self.M0_sc)))
          plt.pause(0.05)
          plt.figure(2)
          plt.imshow(np.transpose(np.abs(-self.tau/np.log(x[1,int(self.Nislice/2),...]*self.T1_sc))))
#          plt.imshow(np.transpose(np.abs(x[1,int(self.Nislice/2),...]*self.T1_sc)))
          plt.pause(0.05)
          plt.figure(3)
          plt.imshow(np.transpose(np.abs(-self.tau/np.log(x[2,int(self.Nislice/2),...]*self.T2_sc))))
          plt.pause(0.05)            
           
             
