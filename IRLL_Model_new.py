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
import matplotlib.pyplot as plt
plt.ion()
import numexpr as ne
DTYPE = np.complex64
class constraint:
  def __init__(self, min_val=-np.inf, max_val=np.inf, real_const=False):
    self.min = min_val
    self.max = max_val
    self.real = real_const
  def update(self,scale):
    self.min = self.min/scale
    self.max = self.max/scale


class IRLL_Model:
  
  
  def __init__(self, fa, fa_corr, TR,tau,td,
               NScan,NSlice,dimY,dimX, Nproj,Nproj_measured,scale,images):

    self.constraints = []
    self.NSlice = NSlice
    self.TR = TR
    self.fa = fa
    self.fa_corr = fa_corr
    
    self.T1_sc = 1#5000
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
    
    self.M0_sc = 1
    self.T1_sc = 1
    self.scale = 100

###
###
    test_T1 = np.exp(-self.scale/np.reshape(np.linspace(10,5500,dimX*dimY*NSlice),(NSlice,dimX,dimY)))
    test_M0 = 1#np.reshape(np.linspace(0,1,dimX*dimY*Nislice),(Nislice,dimX,dimY))
#    G_x = self.execute_forward_3D(np.array([test_M0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),1/self.T1_sc*test_T1*np.ones((NSlice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE))
#    self.M0_sc = self.M0_sc*np.median(np.abs(images))/np.median(np.abs(G_x))
#test_T1*np.ones((Nislice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE))#    
    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),1/self.T1_sc*test_T1*np.ones((NSlice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE))
    self.T1_sc = self.T1_sc*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...]))
    
    self.T1_sc = self.T1_sc #/ np.sqrt(self.M0_sc)
    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),1/self.T1_sc*test_T1*np.ones((NSlice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE))
    print('Grad Scaling', np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...])))    
    print('T1 scale: ',self.T1_sc,
                              '/ M0_scale: ',self.M0_sc)
    
    self.guess = np.array([1/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),np.exp(-self.scale/1500)/self.T1_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE)])               
    self.constraints.append(constraint(-300,300,False)  )
    self.constraints.append(constraint(np.exp(-self.scale/10)/self.T1_sc, np.exp(-self.scale/5500)/self.T1_sc,True))

  def plot_unknowns(self,x,dim_2D=False):
      
      if dim_2D:
          plt.figure(1)
          plt.imshow(np.transpose(np.abs(x[0,...]*self.M0_sc)))
          plt.pause(0.05)
          plt.figure(2)
          plt.imshow(np.transpose(-self.scale/np.log(np.abs(x[1,...]*self.T1_sc))))
          plt.pause(0.05)          
        #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=1000)
#          plt.pause(0.05)
#          plt.figure(3)
#          plt.imshow(np.transpose(np.abs(x[2,...])))
#        #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=1000)
#          plt.pause(0.05)             
      else:         
          plt.figure(1)
          plt.imshow(np.transpose(np.abs(x[0,int(self.NSlice/2),...]*self.M0_sc)))
          plt.pause(0.05)
          plt.figure(2)
          plt.imshow(np.transpose(-self.scale/np.log(np.abs(x[1,int(self.NSlice/2),...]*self.T1_sc))))
        #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=1000)
          plt.pause(0.05)
           
           
           

  def execute_forward_2D(self, x, islice):
    S = np.zeros((self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
#    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
    sin_phi = self.sin_phi[islice,...]#np.sin(self.fa*x[2,...])
    cos_phi = self.cos_phi[islice,...]#np.cos(self.fa*x[2,...])+
    N = self.Nproj_measured
    scale = self.scale
    Efit = x[1,...]*self.T1_sc
    Etau = Efit**(tau/scale)     
    Etr = Efit**(TR/scale)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = Efit**(td/scale)#np.exp(-td/(x[1,...]*T1_sc))   
    M0 = x[0,...]
    M0_sc = self.M0_sc
        

    F = (1 - Etau)/(1-Etau*cos_phi)
    Q = (-Etr*Etd*F*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi + Etr*Etd - 2*Etd + 1)/(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1)
    Q_F = Q-F   
  
    for i in range(self.NLL):
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            S[i,j,...] = M0*M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
    
    return np.array(np.mean(S,axis=1,dtype=np.complex256),dtype=DTYPE)  
  def execute_gradient_2D(self, x, islice):
    grad = np.zeros((2,self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
#    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
    sin_phi = self.sin_phi[islice,...]#np.sin(self.fa*x[2,...])
    cos_phi = self.cos_phi[islice,...]#np.cos(self.fa*x[2,...])+
    N = self.Nproj_measured
    T1_sc =self.T1_sc
    scale = self.scale
    Efit = x[1,...]*self.T1_sc
    Etau = Efit**(tau/scale)     
    Etr = Efit**(TR/scale)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = Efit**(td/scale)#np.exp(-td/(x[1,...]*T1_sc))   

    M0 = x[0,...]
    M0_sc = self.M0_sc
    
  
    F = (1 - Etau)/(1-Etau*cos_phi)
    Q = (-Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)/(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1)
    Q_F = Q-F  
    
    tmp1 = ((-TR*Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                *cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)) + TR*Etr*Etd/(x[1,...]*scale) - tau*Etr*Etau*Etd*(-Etau + 1)*\
                (-(Etau*cos_phi)**(N - 1) + 1)*cos_phi**2/(x[1,...]*scale*(-Etau*cos_phi + 1)**2) + \
                tau*Etr*Etau*Etd*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)) + \
                tau*Etr*Etd*(Etau*cos_phi)**(N - 1)*(N - 1)*(-Etau + 1)*cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)) - \
                td*Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)) +\
                td*Etr*Etd/(x[1,...]*scale) - 2*td*Etd/(x[1,...]*scale))/(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1) +\
                (-TR*Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi/(x[1,...]*scale) - tau*Etr*Etd*(Etau*cos_phi)**(N - 1)*(N - 1)\
                 *cos_phi/(x[1,...]*scale) - td*Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi/(x[1,...]*scale))*(-Etr*Etd*(-Etau + 1)\
                          *(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)/\
                 (Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1)**2 - tau*Etau*(-Etau + 1)*cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)**2) \
                 + tau*Etau/(x[1,...]*scale*(-Etau*cos_phi + 1)))
    
    tmp2 = tau*Etau*(-Etau + 1)*cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)**2) - \
                 tau*Etau/(x[1,...]*scale*(-Etau*cos_phi + 1))

    tmp3 =  (-(-Etau + 1)/(-Etau*cos_phi + 1) \
                           + (-Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)\
                           /(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1))/(x[1,...]*scale)
    for i in range(self.NLL):  
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            
            grad[0,i,j,...] = M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
            
            grad[1,i,j,...] =M0*M0_sc*((Etau*cos_phi)**(n - 1)*tmp1 + tmp2 + tau*(Etau*cos_phi)**(n - 1)*(n - 1)*tmp3)*sin_phi
                 
    return np.array(np.mean(grad,axis=2,dtype=np.complex256),dtype=DTYPE)
           
  def execute_forward_3D(self, x):
    S = np.zeros((self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)

#    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
    sin_phi = self.sin_phi#np.sin(self.fa*x[2,...])
    cos_phi = self.cos_phi#np.cos(self.fa*x[2,...])+
    N = self.Nproj_measured
    scale = self.scale
    Efit = x[1,...]*self.T1_sc
    Etau = Efit**(tau/scale)     
    Etr = Efit**(TR/scale)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = Efit**(td/scale)#np.exp(-td/(x[1,...]*T1_sc))   
    M0 = x[0,...]
    M0_sc = self.M0_sc
        

    F = (1 - Etau)/(1-Etau*cos_phi)
    Q = (-Etr*Etd*F*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi + Etr*Etd - 2*Etd + 1)/(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1)
    Q_F = Q-F   
  
    def numexpeval_S(M0,M0_sc,sin_phi,cos_phi,n,Q_F,F,Etau):
      return ne.evaluate("M0*M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi")    
    for i in range(self.NLL):
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            S[i,j,...] = numexpeval_S(M0,M0_sc,sin_phi,cos_phi,n,Q_F,F,Etau)
    
    return np.array(np.mean(S,axis=1,dtype=np.complex256),dtype=DTYPE)   
  
  def execute_gradient_3D(self, x):
    grad = np.zeros((2,self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
#    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
    sin_phi = self.sin_phi#np.sin(self.fa*x[2,...])
    cos_phi = self.cos_phi#np.cos(self.fa*x[2,...])+
    N = self.Nproj_measured
    T1_sc =self.T1_sc
    scale = self.scale
    Efit = x[1,...]*self.T1_sc
    Etau = Efit**(tau/scale)     
    Etr = Efit**(TR/scale)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = Efit**(td/scale)#np.exp(-td/(x[1,...]*T1_sc))   

    M0 = x[0,...]
    M0_sc = self.M0_sc
  
    F = (1 - Etau)/(1-Etau*cos_phi)
    Q = (-Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)/(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1)
    Q_F = Q-F  
    
    tmp1 = ((-TR*Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                *cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)) + TR*Etr*Etd/(x[1,...]*scale) - tau*Etr*Etau*Etd*(-Etau + 1)*\
                (-(Etau*cos_phi)**(N - 1) + 1)*cos_phi**2/(x[1,...]*scale*(-Etau*cos_phi + 1)**2) + \
                tau*Etr*Etau*Etd*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)) + \
                tau*Etr*Etd*(Etau*cos_phi)**(N - 1)*(N - 1)*(-Etau + 1)*cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)) - \
                td*Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)) +\
                td*Etr*Etd/(x[1,...]*scale) - 2*td*Etd/(x[1,...]*scale))/(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1) +\
                (-TR*Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi/(x[1,...]*scale) - tau*Etr*Etd*(Etau*cos_phi)**(N - 1)*(N - 1)\
                 *cos_phi/(x[1,...]*scale) - td*Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi/(x[1,...]*scale))*(-Etr*Etd*(-Etau + 1)\
                          *(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)/\
                 (Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1)**2 - tau*Etau*(-Etau + 1)*cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)**2) \
                 + tau*Etau/(x[1,...]*scale*(-Etau*cos_phi + 1)))
    
    tmp2 = tau*Etau*(-Etau + 1)*cos_phi/(x[1,...]*scale*(-Etau*cos_phi + 1)**2) - \
                 tau*Etau/(x[1,...]*scale*(-Etau*cos_phi + 1))

    tmp3 =  (-(-Etau + 1)/(-Etau*cos_phi + 1) \
                           + (-Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)\
                           /(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1))/(x[1,...]*scale)
                           
    def numexpeval_M0(M0_sc,sin_phi,cos_phi,n,Q_F,F,Etau):
      return ne.evaluate("M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi")
    def numexpeval_T1(M0,M0_sc,Etau,cos_phi,sin_phi,n,tmp1,tmp2,tmp3,tau):
      return ne.evaluate("M0*M0_sc*((Etau*cos_phi)**(n - 1)*tmp1 + tmp2 + tau*(Etau*cos_phi)**(n - 1)*(n - 1)*tmp3)*sin_phi")
                           
    for i in range(self.NLL):  
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            
            grad[0,i,j,...] = numexpeval_M0(M0_sc,sin_phi,cos_phi,n,Q_F,F,Etau)         
            grad[1,i,j,...] = numexpeval_T1(M0,M0_sc,Etau,cos_phi,sin_phi,n,tmp1,tmp2,tmp3,tau)
            
    print('Grad Scaling', np.linalg.norm(np.abs(np.mean(grad,axis=2)[0,...]))/np.linalg.norm(np.abs(np.mean(grad,axis=2)[1,...])))          
                     
    return np.array(np.mean(grad,axis=2,dtype=np.complex256),dtype=DTYPE)             
           
           
           
           
           
           
           
           