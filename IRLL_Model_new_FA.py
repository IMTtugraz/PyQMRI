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

DTYPE = np.complex64

class constraint:
  def __init__(self, min_val=-np.inf, max_val=np.inf, real_const=False):
    self.min = min_val
    self.max = max_val
    self.real = real_const


class IRLL_Model:
  
  
  def __init__(self, fa, fa_corr, TR,tau,td,
               NScan,NSlice,dimY,dimX, Nproj):


    self.constraints = []
    self.NSlice = NSlice
    self.TR = TR
    self.fa = fa
    self.fa_corr = np.ones_like(fa_corr,DTYPE)
    
    self.T1_sc = 1#2000
    self.M0_sc = 1#50
    
    self.tau = tau
    self.td = td
    self.NLL = NScan
    self.Nproj = Nproj
    self.dimY = dimY
    self.dimX = dimX
    
    phi_corr = np.zeros_like(fa_corr,dtype=DTYPE)
    phi_corr = np.real(fa)*np.real(fa_corr) + 1j*np.imag(fa)*np.imag(fa_corr)

    
    self.sin_phi = np.sin(phi_corr)
    self.cos_phi = np.cos(phi_corr)    

    self.guess = np.array([0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),\
                           np.exp(-200/(3000/self.T1_sc))*np.ones((NSlice,dimY,dimX),dtype=DTYPE),\
                           DTYPE(fa)*np.ones((NSlice,dimY,dimX),dtype=DTYPE)])  
    
    self.constraints.append(constraint(-300,300,False)  )
    self.constraints.append(constraint(np.exp(-200/(50)), np.exp(-200/(5000)),True))
    self.constraints.append(constraint(fa/1.4,fa*1.4,True))
#    self.min_T1 = np.exp(-200/(50/self.T1_sc))
#    self.max_T1 = np.exp(-200/(5000/self.T1_sc))

  def plot_unknowns(self,x,dim_2D=False):
      
      if dim_2D:
          plt.figure(1)
          plt.imshow(np.transpose(np.abs(x[0,...]*self.M0_sc)))
          plt.pause(0.05)
          plt.figure(2)
          plt.imshow(np.transpose(-200/np.log(np.abs(x[1,...]*self.T1_sc))))
          plt.pause(0.05)          
        #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
          plt.figure(3)
          plt.imshow(np.transpose(np.abs(x[2,...])))
          plt.pause(0.05)          
#        #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
#          plt.pause(0.05)             
      else:         
          plt.figure(1)
          plt.imshow(np.transpose(np.abs(x[0,int(self.NSlice/2),...]*self.M0_sc)))
          plt.pause(0.05)
          plt.figure(2)
          plt.imshow(np.transpose(-200/np.log(np.abs(x[1,int(self.NSlice/2),...]*self.T1_sc))))
        #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
          plt.pause(0.05)
          plt.figure(3)
          plt.imshow(np.transpose(np.abs(x[2,int(self.NSlice/2),...])))
          plt.pause(0.05)              
           
           
           
           
           
           

  def execute_forward_2D(self, x, islice):
    S = np.zeros((self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
#    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
#    fa = self.fa
#    fa_corr = x[2,...]
    sin_phi = np.sin(x[2,...])#np.sin(self.fa*x[2,...])
    cos_phi = np.cos(x[2,...])#np.cos(self.fa*x[2,...])
    beta = 180/5
    cos_phi_alpha = np.cos(beta*x[2,...])
    N = self.NLL*self.Nproj
    Etau =x[1,...]**(tau/200) #np.exp(-tau/(x[1,...]*T1_sc))    
    Etr = x[1,...]**(TR/200)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = x[1,...]**(td/200)#np.exp(-td/(x[1,...]*T1_sc))    
#    cosEtau = cos_phi*Etau        
#    cosEtauN = cosEtau**(N-1)           

    F = (-Etau + 1)/(-Etau*cos_phi + 1)
    Q = (Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(-Etau*cos_phi + 1) + Etd*(-Etr + 1)*cos_phi_alpha - Etd + 1)/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1)
    Q_F = Q-F   
  
    for i in range(self.NLL):
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            S[i,j,...] = x[0,...]*M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
    
    return np.mean(S,axis=1)
  
  def execute_gradient_2D(self, x, islice):
    grad = np.zeros((3,self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
#    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
#    fa = self.fa
#    fa_corr = x[2,...]
    sin_phi = np.sin(x[2,...])#np.sin(self.fa*x[2,...])
    cos_phi = np.cos(x[2,...])#np.cos(self.fa*x[2,...])
    beta = 180/5
    cos_phi_alpha = np.cos(beta*x[2,...])
    sin_phi_alpha = np.cos(beta*x[2,...])
    N = self.NLL*self.Nproj
    Efit = x[1,...]
    M0 = x[0,...]
    Etau =x[1,...]**(tau/200) #np.exp(-tau/(x[1,...]*T1_sc))    
    Etr = x[1,...]**(TR/200)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = x[1,...]**(td/200)#np.exp(-td/(x[1,...]*T1_sc))    
#    cosEtau = cos_phi*Etau        
#    cosEtauN = cosEtau**(N-1)           

    F = (-Etau + 1)/(-Etau*cos_phi + 1)
    Q = (Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(-Etau*cos_phi + 1) + Etd*(-Etr + 1)*cos_phi_alpha - Etd + 1)/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1)
    Q_F = Q-F   
    tmp1 = ((Etr*Etau*Etd*tau*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                *cos_phi**2*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)**2) -\
                Etr*Etau*Etd*tau*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/\
                (200*Efit*(-Etau*cos_phi + 1)) + Etr*Etd*TR*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                *cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) - Etr*Etd*TR*cos_phi_alpha/(200*Efit) -\
                Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(-Etau + 1)*(N - 1)*cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) \
                + Etr*Etd*td*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) +\
                Etd*td*(-Etr + 1)*cos_phi_alpha/(200*Efit) - Etd*td/(200*Efit))/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*\
                       cos_phi*cos_phi_alpha + 1) + (Etr*Etd*TR*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha/(200*Efit) +\
                                                    Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(N - 1)*cos_phi*cos_phi_alpha/(200*Efit)\
                                                    + Etr*Etd*td*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha/(200*Efit))*Q**2 -\
                Etau*tau*(-Etau + 1)*cos_phi/(200*Efit*(-Etau*cos_phi + 1)**2) + Etau*tau/(200*Efit*(-Etau*cos_phi + 1)))
    tmp2 = Etau*tau*(-Etau + 1)*cos_phi/(200*Efit*(-Etau*cos_phi + 1)**2) - Etau*tau/(200*Efit*(-Etau*cos_phi + 1))
    tmp3 = -Etau*(-Etau + 1)*sin_phi/(-Etau*cos_phi + 1)**2
    tmp4 = (Etau*(-Etau + 1)*sin_phi/(-Etau*cos_phi + 1)**2 +\
                       (-Etr*Etau*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*sin_phi*cos_phi*cos_phi_alpha/\
                        (-Etau*cos_phi + 1)**2 - Etr*Etd*beta*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*\
                        sin_phi_alpha*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd*(Etau*cos_phi)**(N - 1)*(-Etau + 1)*(N - 1)*\
                        sin_phi*cos_phi_alpha/(-Etau*cos_phi + 1) - Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                        *sin_phi*cos_phi_alpha/(-Etau*cos_phi + 1) - Etd*beta*(-Etr + 1)*sin_phi_alpha)/\
                        (-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1) + (-Etr*Etd*beta*(Etau*cos_phi)**(N - 1)\
                         *sin_phi_alpha*cos_phi - Etr*Etd*(Etau*cos_phi)**(N - 1)*(N - 1)*sin_phi*cos_phi_alpha - \
                         Etr*Etd*(Etau*cos_phi)**(N - 1)*sin_phi*cos_phi_alpha)*Q**2)

    for i in range(self.NLL):
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            
            grad[0,i,j,...] = M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
            
            grad[1,i,j,...] = M0*M0_sc*((Etau*cos_phi)**(n - 1)*tmp1 + tmp2 +\
                              tau*(Etau*cos_phi)**(n - 1)*(n - 1)*(Q_F)/(200*Efit))*sin_phi
            grad[2,i,j,...] = M0*M0_sc*((Etau*cos_phi)**(n - 1)*(Q_F) + F)*cos_phi + \
                              M0*M0_sc*(tmp3 - (Etau*cos_phi)**(n - 1)*(n - 1)*(Q_F)*sin_phi/cos_phi
                                        + (Etau*cos_phi)**(n - 1)*tmp4)*sin_phi
            
    return np.mean(grad,axis=2)
             
           
           
  def execute_forward_3D(self, x):
    S = np.zeros((self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
#    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
#    fa = self.fa
#    fa_corr = x[2,...]
    sin_phi = np.sin(x[2,...])#np.sin(self.fa*x[2,...])
    cos_phi = np.cos(x[2,...])#np.cos(self.fa*x[2,...])
    beta = 180/5
    cos_phi_alpha = np.cos(beta*x[2,...])
    N = self.NLL*self.Nproj
    Etau =x[1,...]**(tau/200) #np.exp(-tau/(x[1,...]*T1_sc))    
    Etr = x[1,...]**(TR/200)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = x[1,...]**(td/200)#np.exp(-td/(x[1,...]*T1_sc))    
#    cosEtau = cos_phi*Etau        
#    cosEtauN = cosEtau**(N-1)           

    F = (-Etau + 1)/(-Etau*cos_phi + 1)
    Q = (Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(-Etau*cos_phi + 1) + Etd*(-Etr + 1)*cos_phi_alpha - Etd + 1)/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1)
    Q_F = Q-F   
  
    for i in range(self.NLL):
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            S[i,j,...] = x[0,...]*M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
    
    return np.mean(S,axis=1)
  
  def execute_gradient_3D(self, x):
    grad = np.zeros((3,self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
    M0_sc = self.M0_sc
#    T1_sc = self.T1_sc
    TR = self.TR
    tau = self.tau
    td = self.td
#    fa = self.fa
#    fa_corr = x[2,...]
    sin_phi = np.sin(x[2,...])#np.sin(self.fa*x[2,...])
    cos_phi = np.cos(x[2,...])#np.cos(self.fa*x[2,...])
    beta = 180/5
    cos_phi_alpha = np.cos(beta*x[2,...])
    sin_phi_alpha = np.cos(beta*x[2,...])
    N = self.NLL*self.Nproj
    Efit = x[1,...]
    M0 = x[0,...]
    Etau =x[1,...]**(tau/200) #np.exp(-tau/(x[1,...]*T1_sc))    
    Etr = x[1,...]**(TR/200)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = x[1,...]**(td/200)#np.exp(-td/(x[1,...]*T1_sc))    
#    cosEtau = cos_phi*Etau        
#    cosEtauN = cosEtau**(N-1)           

    F = (-Etau + 1)/(-Etau*cos_phi + 1)
    Q = (Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(-Etau*cos_phi + 1) + Etd*(-Etr + 1)*cos_phi_alpha - Etd + 1)/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1)
    Q_F = Q-F   
    tmp1 = ((Etr*Etau*Etd*tau*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                *cos_phi**2*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)**2) -\
                Etr*Etau*Etd*tau*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/\
                (200*Efit*(-Etau*cos_phi + 1)) + Etr*Etd*TR*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                *cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) - Etr*Etd*TR*cos_phi_alpha/(200*Efit) -\
                Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(-Etau + 1)*(N - 1)*cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) \
                + Etr*Etd*td*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) +\
                Etd*td*(-Etr + 1)*cos_phi_alpha/(200*Efit) - Etd*td/(200*Efit))/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*\
                       cos_phi*cos_phi_alpha + 1) + (Etr*Etd*TR*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha/(200*Efit) +\
                                                    Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(N - 1)*cos_phi*cos_phi_alpha/(200*Efit)\
                                                    + Etr*Etd*td*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha/(200*Efit))*Q**2 -\
                Etau*tau*(-Etau + 1)*cos_phi/(200*Efit*(-Etau*cos_phi + 1)**2) + Etau*tau/(200*Efit*(-Etau*cos_phi + 1)))
    tmp2 = Etau*tau*(-Etau + 1)*cos_phi/(200*Efit*(-Etau*cos_phi + 1)**2) - Etau*tau/(200*Efit*(-Etau*cos_phi + 1))
    tmp3 = -Etau*(-Etau + 1)*sin_phi/(-Etau*cos_phi + 1)**2
    tmp4 = (Etau*(-Etau + 1)*sin_phi/(-Etau*cos_phi + 1)**2 +\
                       (-Etr*Etau*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*sin_phi*cos_phi*cos_phi_alpha/\
                        (-Etau*cos_phi + 1)**2 - Etr*Etd*beta*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*\
                        sin_phi_alpha*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd*(Etau*cos_phi)**(N - 1)*(-Etau + 1)*(N - 1)*\
                        sin_phi*cos_phi_alpha/(-Etau*cos_phi + 1) - Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                        *sin_phi*cos_phi_alpha/(-Etau*cos_phi + 1) - Etd*beta*(-Etr + 1)*sin_phi_alpha)/\
                        (-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1) + (-Etr*Etd*beta*(Etau*cos_phi)**(N - 1)\
                         *sin_phi_alpha*cos_phi - Etr*Etd*(Etau*cos_phi)**(N - 1)*(N - 1)*sin_phi*cos_phi_alpha - \
                         Etr*Etd*(Etau*cos_phi)**(N - 1)*sin_phi*cos_phi_alpha)*Q**2)

    for i in range(self.NLL):
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            
            grad[0,i,j,...] = M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
            
            grad[1,i,j,...] = M0*M0_sc*((Etau*cos_phi)**(n - 1)*tmp1 + tmp2 +\
                              tau*(Etau*cos_phi)**(n - 1)*(n - 1)*(Q_F)/(200*Efit))*sin_phi
            grad[2,i,j,...] = M0*M0_sc*((Etau*cos_phi)**(n - 1)*(Q_F) + F)*cos_phi + \
                              M0*M0_sc*(tmp3 - (Etau*cos_phi)**(n - 1)*(n - 1)*(Q_F)*sin_phi/cos_phi
                                        + (Etau*cos_phi)**(n - 1)*tmp4)*sin_phi
            
    return np.mean(grad,axis=2)
             
           
           
#           ############################################## fa_corr fit
##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Fri Jun  9 11:33:09 2017
#
#@author: omaier
#"""
## cython: profile=False
## filename: IRLL_Model.pyx
##cython: boundscheck=False, wraparound=False, nonecheck=False
#
#import numpy as np
#import matplotlib.pyplot as plt
#plt.ion()
#
#DTYPE = np.complex64
#
#
#class IRLL_Model:
#  
#  
#  def __init__(self, fa, fa_corr, TR,tau,td,
#               NScan,NSlice,dimY,dimX, Nproj):
#
#
#    self.NSlice = NSlice
#    self.TR = TR
#    self.fa = fa
#    self.fa_corr = np.ones_like(fa_corr,DTYPE)
#    
#    self.T1_sc = 1#2000
#    self.M0_sc = 1#50
#    
#    self.tau = tau
#    self.td = td
#    self.NLL = NScan
#    self.Nproj = Nproj
#    self.dimY = dimY
#    self.dimX = dimX
#    
#    phi_corr = np.zeros_like(fa_corr,dtype=DTYPE)
#    phi_corr = np.real(fa)*np.real(fa_corr) + 1j*np.imag(fa)*np.imag(fa_corr)
#
#    
#    self.sin_phi = np.sin(phi_corr)
#    self.cos_phi = np.cos(phi_corr)    
#
#    self.guess = np.array([0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),\
#                           np.exp(-200/(3000/self.T1_sc))*np.ones((NSlice,dimY,dimX),dtype=DTYPE),\
#                           np.ones((NSlice,dimY,dimX),dtype=DTYPE)])               
#    self.min_T1 = np.exp(-200/(50/self.T1_sc))
#    self.max_T1 = np.exp(-200/(2000/self.T1_sc))
#
#  def plot_unknowns(self,x,dim_2D=False):
#      
#      if dim_2D:
#          plt.figure(1)
#          plt.imshow(np.transpose(np.abs(x[0,...]*self.M0_sc)))
#          plt.pause(0.05)
#          plt.figure(2)
#          plt.imshow(np.transpose(-200/np.log(np.abs(x[1,...]*self.T1_sc))))
#          plt.pause(0.05)          
#        #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
#          plt.figure(3)
#          plt.imshow(np.transpose(np.abs(x[2,...])))
#          plt.pause(0.05)          
##        #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
##          plt.pause(0.05)             
#      else:         
#          plt.figure(1)
#          plt.imshow(np.transpose(np.abs(x[0,int(self.NSlice/2),...]*self.M0_sc)))
#          plt.pause(0.05)
#          plt.figure(2)
#          plt.imshow(np.transpose(-200/np.log(np.abs(x[1,int(self.NSlice/2),...]*self.T1_sc))))
#        #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
#          plt.pause(0.05)
#           
#           
#           
#           
#           
#           
#           
#
#  def execute_forward_2D(self, x, islice):
#    S = np.zeros((self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
#    M0_sc = self.M0_sc
##    T1_sc = self.T1_sc
#    TR = self.TR
#    tau = self.tau
#    td = self.td
#    fa = self.fa
##    fa_corr = x[2,...]
#    sin_phi = np.sin(fa*x[2,...])#np.sin(self.fa*x[2,...])
#    cos_phi = np.cos(fa*x[2,...])#np.cos(self.fa*x[2,...])
#    beta = 180/5
#    cos_phi_alpha = np.cos(fa*beta*x[2,...])
#    N = self.NLL*self.Nproj
#    Etau =x[1,...]**(tau/200) #np.exp(-tau/(x[1,...]*T1_sc))    
#    Etr = x[1,...]**(TR/200)#np.exp(-TR/(x[1,...]*T1_sc))
#    Etd = x[1,...]**(td/200)#np.exp(-td/(x[1,...]*T1_sc))    
##    cosEtau = cos_phi*Etau        
##    cosEtauN = cosEtau**(N-1)           
#
#    F = (-Etau + 1)/(-Etau*cos_phi + 1)
#    Q = (Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(-Etau*cos_phi + 1) + Etd*(-Etr + 1)*cos_phi_alpha - Etd + 1)/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1)
#    Q_F = Q-F   
#  
#    for i in range(self.NLL):
#      for j in range(self.Nproj):
#            n = i*self.Nproj+j+1
#            S[i,j,...] = x[0,...]*M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
#    
#    return np.mean(S,axis=1)
#  
#  def execute_gradient_2D(self, x, islice):
#    grad = np.zeros((3,self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
#    M0_sc = self.M0_sc
##    T1_sc = self.T1_sc
#    TR = self.TR
#    tau = self.tau
#    td = self.td
#    fa = self.fa
##    fa_corr = x[2,...]
#    sin_phi = np.sin(fa*x[2,...])#np.sin(self.fa*x[2,...])
#    cos_phi = np.cos(fa*x[2,...])#np.cos(self.fa*x[2,...])
#    beta = 180/5
#    cos_phi_alpha = np.cos(fa*beta*x[2,...])
#    sin_phi_alpha = np.cos(fa*beta*x[2,...])
#    N = self.NLL*self.Nproj
#    Efit = x[1,...]
#    M0 = x[0,...]
#    Etau =x[1,...]**(tau/200) #np.exp(-tau/(x[1,...]*T1_sc))    
#    Etr = x[1,...]**(TR/200)#np.exp(-TR/(x[1,...]*T1_sc))
#    Etd = x[1,...]**(td/200)#np.exp(-td/(x[1,...]*T1_sc))    
##    cosEtau = cos_phi*Etau        
##    cosEtauN = cosEtau**(N-1)           
#
#    F = (-Etau + 1)/(-Etau*cos_phi + 1)
#    Q = (Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(-Etau*cos_phi + 1) + Etd*(-Etr + 1)*cos_phi_alpha - Etd + 1)/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1)
#    Q_F = Q-F   
#    tmp1 = ((Etr*Etau*Etd*tau*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
#                *cos_phi**2*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)**2) -\
#                Etr*Etau*Etd*tau*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/\
#                (200*Efit*(-Etau*cos_phi + 1)) + Etr*Etd*TR*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
#                *cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) - Etr*Etd*TR*cos_phi_alpha/(200*Efit) -\
#                Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(-Etau + 1)*(N - 1)*cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) \
#                + Etr*Etd*td*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) +\
#                Etd*td*(-Etr + 1)*cos_phi_alpha/(200*Efit) - Etd*td/(200*Efit))/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*\
#                       cos_phi*cos_phi_alpha + 1) + (Etr*Etd*TR*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha/(200*Efit) +\
#                                                    Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(N - 1)*cos_phi*cos_phi_alpha/(200*Efit)\
#                                                    + Etr*Etd*td*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha/(200*Efit))*Q**2 -\
#                Etau*tau*(-Etau + 1)*cos_phi/(200*Efit*(-Etau*cos_phi + 1)**2) + Etau*tau/(200*Efit*(-Etau*cos_phi + 1)))
#    tmp2 = Etau*tau*(-Etau + 1)*cos_phi/(200*Efit*(-Etau*cos_phi + 1)**2) - Etau*tau/(200*Efit*(-Etau*cos_phi + 1))
#    tmp3 = -Etau*fa*(-Etau + 1)*sin_phi/(-Etau*cos_phi + 1)**2
#    tmp4 = (Etau*fa*(-Etau + 1)*sin_phi/(-Etau*cos_phi + 1)**2 + \
#                       (-Etr*Etau*Etd*fa*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*sin_phi*cos_phi*cos_phi_alpha/\
#                        (-Etau*cos_phi + 1)**2 - Etr*Etd*beta*fa*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*\
#                        sin_phi_alpha*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd*fa*(Etau*cos_phi)**(N - 1)*(-Etau + 1)*(N - 1)\
#                        *sin_phi*cos_phi_alpha/(-Etau*cos_phi + 1) - Etr*Etd*fa*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
#                        *sin_phi*cos_phi_alpha/(-Etau*cos_phi + 1) - Etd*beta*fa*(-Etr + 1)*sin_phi_alpha)/\
#                        (-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1) + \
#                        (-Etr*Etd*beta*fa*(Etau*cos_phi)**(N - 1)*sin_phi_alpha*cos_phi - Etr*Etd*fa*(Etau*cos_phi)**(N - 1)\
#                         *(N - 1)*sin_phi*cos_phi_alpha - Etr*Etd*fa*(Etau*cos_phi)**(N - 1)*sin_phi*cos_phi_alpha)*Q**2)
#
#    for i in range(self.NLL):
#      for j in range(self.Nproj):
#            n = i*self.Nproj+j+1
#            
#            grad[0,i,j,...] = M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
#            
#            grad[1,i,j,...] = M0*M0_sc*((Etau*cos_phi)**(n - 1)*tmp1 + tmp2 +\
#                              tau*(Etau*cos_phi)**(n - 1)*(n - 1)*(Q_F)/(200*Efit))*sin_phi
#            grad[2,i,j,...] = M0*M0_sc*fa*((Etau*cos_phi)**(n - 1)*(Q_F) + F)*cos_phi + \
#                              M0*M0_sc*(tmp3 - fa*(Etau*cos_phi)**(n - 1)*(n - 1)*(Q_F)*sin_phi/cos_phi +\
#                                        (Etau*cos_phi)**(n - 1)*tmp4)*sin_phi
#    return np.mean(grad,axis=2)
#             
#           
#           
#  def execute_forward_3D(self, x):
#    S = np.zeros((self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
#    M0_sc = self.M0_sc
##    T1_sc = self.T1_sc
#    TR = self.TR
#    tau = self.tau
#    td = self.td
#    fa = self.fa
##    fa_corr = x[2,...]
#    sin_phi = np.sin(fa*x[2,...])#np.sin(self.fa*x[2,...])
#    cos_phi = np.cos(fa*x[2,...])#np.cos(self.fa*x[2,...])
#    beta = 180/5
#    cos_phi_alpha = np.cos(fa*beta*x[2,...])
#    N = self.NLL*self.Nproj
#    Etau =x[1,...]**(tau/200) #np.exp(-tau/(x[1,...]*T1_sc))    
#    Etr = x[1,...]**(TR/200)#np.exp(-TR/(x[1,...]*T1_sc))
#    Etd = x[1,...]**(td/200)#np.exp(-td/(x[1,...]*T1_sc))    
##    cosEtau = cos_phi*Etau        
##    cosEtauN = cosEtau**(N-1)           
#
#    F = (-Etau + 1)/(-Etau*cos_phi + 1)
#    Q = (Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(-Etau*cos_phi + 1) + Etd*(-Etr + 1)*cos_phi_alpha - Etd + 1)/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1)
#    Q_F = Q-F   
#  
#    for i in range(self.NLL):
#      for j in range(self.Nproj):
#            n = i*self.Nproj+j+1
#            S[i,j,...] = x[0,...]*M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
#    
#    return np.mean(S,axis=1)
#  
#  def execute_gradient_3D(self, x):
#    grad = np.zeros((2,self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
#    M0_sc = self.M0_sc
##    T1_sc = self.T1_sc
#    TR = self.TR
#    tau = self.tau
#    td = self.td
#    fa = self.fa
##    fa_corr = x[2,...]
#    sin_phi = np.sin(fa*x[2,...])#np.sin(self.fa*x[2,...])
#    cos_phi = np.cos(fa*x[2,...])#np.cos(self.fa*x[2,...])
#    beta = 180/5
#    cos_phi_alpha = np.cos(fa*beta*x[2,...])
#    sin_phi_alpha = np.cos(fa*beta*x[2,...])
#    N = self.NLL*self.Nproj
#    Efit = x[1,...]
#    M0 = x[0,...]
#    Etau =x[1,...]**(tau/200) #np.exp(-tau/(x[1,...]*T1_sc))    
#    Etr = x[1,...]**(TR/200)#np.exp(-TR/(x[1,...]*T1_sc))
#    Etd = x[1,...]**(td/200)#np.exp(-td/(x[1,...]*T1_sc))    
##    cosEtau = cos_phi*Etau        
##    cosEtauN = cosEtau**(N-1)           
#
#    F = (-Etau + 1)/(-Etau*cos_phi + 1)
#    Q = (Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(-Etau*cos_phi + 1) + Etd*(-Etr + 1)*cos_phi_alpha - Etd + 1)/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1)
#    Q_F = Q-F   
#    tmp1 = ((Etr*Etau*Etd*tau*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
#                *cos_phi**2*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)**2) -\
#                Etr*Etau*Etd*tau*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/\
#                (200*Efit*(-Etau*cos_phi + 1)) + Etr*Etd*TR*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
#                *cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) - Etr*Etd*TR*cos_phi_alpha/(200*Efit) -\
#                Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(-Etau + 1)*(N - 1)*cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) \
#                + Etr*Etd*td*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi*cos_phi_alpha/(200*Efit*(-Etau*cos_phi + 1)) +\
#                Etd*td*(-Etr + 1)*cos_phi_alpha/(200*Efit) - Etd*td/(200*Efit))/(-Etr*Etd*(Etau*cos_phi)**(N - 1)*\
#                       cos_phi*cos_phi_alpha + 1) + (Etr*Etd*TR*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha/(200*Efit) +\
#                                                    Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(N - 1)*cos_phi*cos_phi_alpha/(200*Efit)\
#                                                    + Etr*Etd*td*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha/(200*Efit))*Q**2 -\
#                Etau*tau*(-Etau + 1)*cos_phi/(200*Efit*(-Etau*cos_phi + 1)**2) + Etau*tau/(200*Efit*(-Etau*cos_phi + 1)))
#    tmp2 = Etau*tau*(-Etau + 1)*cos_phi/(200*Efit*(-Etau*cos_phi + 1)**2) - Etau*tau/(200*Efit*(-Etau*cos_phi + 1))
#    tmp3 = -Etau*fa*(-Etau + 1)*sin_phi/(-Etau*cos_phi + 1)**2
#    tmp4 = (Etau*fa*(-Etau + 1)*sin_phi/(-Etau*cos_phi + 1)**2 + \
#                       (-Etr*Etau*Etd*fa*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*sin_phi*cos_phi*cos_phi_alpha/\
#                        (-Etau*cos_phi + 1)**2 - Etr*Etd*beta*fa*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*\
#                        sin_phi_alpha*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd*fa*(Etau*cos_phi)**(N - 1)*(-Etau + 1)*(N - 1)\
#                        *sin_phi*cos_phi_alpha/(-Etau*cos_phi + 1) - Etr*Etd*fa*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
#                        *sin_phi*cos_phi_alpha/(-Etau*cos_phi + 1) - Etd*beta*fa*(-Etr + 1)*sin_phi_alpha)/\
#                        (-Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi*cos_phi_alpha + 1) + \
#                        (-Etr*Etd*beta*fa*(Etau*cos_phi)**(N - 1)*sin_phi_alpha*cos_phi - Etr*Etd*fa*(Etau*cos_phi)**(N - 1)\
#                         *(N - 1)*sin_phi*cos_phi_alpha - Etr*Etd*fa*(Etau*cos_phi)**(N - 1)*sin_phi*cos_phi_alpha)*Q**2)
#
#    for i in range(self.NLL):
#      for j in range(self.Nproj):
#            n = i*self.Nproj+j+1
#            
#            grad[0,i,j,...] = M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
#            
#            grad[1,i,j,...] = M0*M0_sc*((Etau*cos_phi)**(n - 1)*tmp1 + tmp2 +\
#                              tau*(Etau*cos_phi)**(n - 1)*(n - 1)*(Q_F)/(200*Efit))*sin_phi
#            grad[2,i,j,...] = M0*M0_sc*fa*((Etau*cos_phi)**(n - 1)*(Q_F) + F)*cos_phi + \
#                              M0*M0_sc*(tmp3 - fa*(Etau*cos_phi)**(n - 1)*(n - 1)*(Q_F)*sin_phi/cos_phi +\
#                                        (Etau*cos_phi)**(n - 1)*tmp4)*sin_phi
#    return np.mean(grad,axis=2)
#             
#           
#           
           
           
           
           
           
           
           
           
           
           
           