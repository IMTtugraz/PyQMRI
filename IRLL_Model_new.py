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
               NScan,NSlice,dimY,dimX, Nproj, Nproj_measured, scale):

    self.constraints = []
    self.NSlice = NSlice
    self.TR = TR
    self.fa = fa
    self.fa_corr = np.ones_like(fa_corr,DTYPE)
    
    self.T1_sc = 1#1000
    self.M0_sc = 1/scale#50
    
    self.tau = tau
    self.td = td
    self.NLL = NScan
    self.Nproj = Nproj
    self.Nproj_measured = Nproj
    self.dimY = dimY
    self.dimX = dimX
    
    phi_corr = np.zeros_like(fa_corr,dtype=DTYPE)
    phi_corr = np.real(fa)*np.real(fa_corr) + 1j*np.imag(fa)*np.imag(fa_corr)

    
    self.sin_phi = np.sin(phi_corr)
    self.cos_phi = np.cos(phi_corr)    

    self.guess = np.array([1e-3/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),\
                           np.exp(-100/(1000/self.T1_sc))*np.ones((NSlice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE)
#                           np.ones((NSlice,dimY,dimX),dtype=DTYPE)])               
    self.constraints.append(constraint(-300,300,False)  )
    self.constraints.append(constraint(np.exp(-100/(10)), np.exp(-100/(5500)),True))

#  def execute_forward_2D(self, x, islice):
#    S = np.zeros((self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
#    M0_sc = self.M0_sc
#    T1_sc = self.T1_sc
#    TR = self.TR
#    tau = self.tau
#    td = self.td
#    sin_phi = self.sin_phi#np.sin(self.fa*x[2,...])
#    cos_phi = self.cos_phi#np.cos(self.fa*x[2,...])
#    N = self.NLL
#    cosEtau = cos_phi[islice,...]*np.exp(-tau/(x[1,...]*T1_sc))        
#    cosEtauN = cosEtau**(N-1)           
#    Etr = np.exp(-TR/(x[1,...]*T1_sc))
#    Etd = np.exp(-td/(x[1,...]*T1_sc))
#    Etau = np.exp(-tau/(x[1,...]*T1_sc))
#    F = (1 - Etau)/(-cosEtau + 1)
#    Q = (-cos_phi[islice,...]*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr)/(cos_phi[islice,...]*cosEtauN*Etr*Etd + 1)
#    Q_F = Q-F   
#  
#    for i in range(self.NLL):
#      for j in range(self.Nproj):
#            n = i*self.Nproj+j+1
#            S[i,j,...] = x[0,...]*M0_sc*sin_phi[islice,...]*((cosEtau)**(n - 1)*(Q_F) + F)
#    
#    return np.mean(S,axis=1)
#  def execute_gradient_2D(self, x, islice):
#    grad = np.zeros((2,self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
#    M0_sc = self.M0_sc
#    T1_sc = self.T1_sc
#    TR = self.TR
#    tau = self.tau
#    td = self.td
#    sin_phi = self.sin_phi#np.sin(self.fa*x[2,...])
#    cos_phi = self.cos_phi#np.cos(self.fa*x[2,...]) 
##    fa = self.fa
#    N = self.NLL  
#    
#    ####Precompute
#    cosEtau = cos_phi[islice,...]*np.exp(-tau/(x[1,...]*T1_sc))        
#    cosEtauN = cosEtau**(N-1)           
#    Etr = np.exp(-TR/(x[1,...]*T1_sc))
#    Etd = np.exp(-td/(x[1,...]*T1_sc))
#    Etau = np.exp(-tau/(x[1,...]*T1_sc))
#    F = (1 - Etau)/(-cosEtau + 1)
#    cos_phi_Etau_tmp = cos_phi[islice,...]*cosEtauN*Etr*Etd    
#    Q = (-cos_phi[islice,...]*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr)/(cos_phi_Etau_tmp + 1)
#    Q_F = Q-F
#    T1sqrt = x[1,...]**2*T1_sc
#    tmp1 = ((-TR*cos_phi[islice,...]*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(T1sqrt*(-cosEtau + 1)) + TR*Etr/(T1sqrt) -\
#              cos_phi[islice,...]**2*tau*(1 - Etau)*(-cosEtauN + 1)*Etr*Etau*Etd/(T1sqrt*(-cosEtau + 1)**2) + \
#              cos_phi[islice,...]*tau*cosEtauN*(1 - Etau)*(N - 1)*Etr*Etd/(T1sqrt*(-cosEtau + 1)) + \
#              cos_phi[islice,...]*tau*(-cosEtauN + 1)*Etr*Etau*Etd/(T1sqrt*(-cosEtau + 1)) - cos_phi[islice,...]*td*(1 - Etau)\
#              *(-cosEtauN + 1)*Etr*Etd/(T1sqrt*(-cosEtau + 1)) - 2*td*Etd/(T1sqrt))/\
#              (cos_phi_Etau_tmp + 1) + (-TR*cos_phi_Etau_tmp/(T1sqrt) -\
#               cos_phi[islice,...]*tau*cosEtauN*(N - 1)*Etr*Etd/(T1sqrt) - cos_phi[islice,...]*td*cosEtauN*Etr*Etd/(T1sqrt))*\
#               (-cos_phi[islice,...]*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(-cosEtau + 1) + 1 - 2*Etd + Etr)/(cos_phi[islice,...]*cosEtauN*Etr*\
#                   Etd + 1)**2 - cos_phi[islice,...]*tau*(1 - Etau)*Etau/(T1sqrt*(-cosEtau + 1)**2) + tau*Etau/(T1sqrt*(-cosEtau + 1)))
#    tmp2 = cos_phi[islice,...]*tau*(1 - Etau)*Etau/(T1sqrt*(-cosEtau + 1)**2) - tau*Etau/(T1sqrt*(-cosEtau + 1))      
#    
##    tmp_sin = fa*(1 - Etau)*Etau*sin_phi[islice,...]/(1 - cosEtau)**2
#
##    tmp3 = (tmp_sin + (-fa*cosEtauN*(1 - Etau)*(N - 1)\
##                 *Etr*Etd*sin_phi[islice,...]/(1 - cosEtau) + fa*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd*sin_phi[islice,...]\
##                 /(1 - cosEtau) + fa*(1 - Etau)*(-cosEtauN + 1)*Etr*Etau*Etd*sin_phi[islice,...]*cos_phi[islice,...]\
##                 /(1 - cosEtau)**2)/(cosEtauN*Etr*Etd*cos_phi[islice,...] + 1) + (fa*cosEtauN*(N - 1)\
##                  *Etr*Etd*sin_phi[islice,...] + fa*cosEtauN*Etr*Etd*sin_phi[islice,...])*\
##            (1 - 2*Etd + Etr - (1 - Etau)*(-cosEtauN + 1)*Etr*Etd*cos_phi[islice,...]/(1 - cosEtau))/\
##            (cos_phi_Etau_tmp + 1)**2)
#
#    for i in range(self.NLL):  
#      for j in range(self.Nproj):
#            n = i*self.Nproj+j+1
#            
#            grad[0,i,j,...] = M0_sc*sin_phi[islice,...]*((cosEtau)**(n - 1)*(Q_F) + F)
#            
#            grad[1,i,j,...] =x[0,...]*M0_sc*sin_phi[islice,...]*((cosEtau)**(n - 1)*tmp1 + tmp2 \
#                +tau*(cosEtau)**(n - 1)*(n - 1)*(-(1 - Etau)/(-cosEtau + 1) + (-cos_phi[islice,...]*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(-cosEtau + 1) + 1 - 2*Etd +\
#                Etr)/(cos_phi_Etau_tmp + 1))/(T1sqrt))
##            grad[2,i,j,...] = x[0,...]*M0_sc*fa*((cosEtau)**(n - 1)*(Q_F) + F)*cos_phi[islice,...]\
##            + x[0,...]*M0_sc*(-fa*(cosEtau)**(n - 1)*(n - 1)*(Q_F)*sin_phi[islice,...]/cos_phi[islice,...] - \
##            tmp_sin + (cosEtau)**(n - 1)*tmp3)*sin_phi[islice,...]
#            
#    return np.mean(grad,axis=2)
  
  
#  def execute_forward_3D(self, x):
#    S = np.zeros((self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
#    M0_sc = self.M0_sc
#    T1_sc = self.T1_sc
#    TR = self.TR
#    tau = self.tau
#    td = self.td
#    sin_phi = self.sin_phi
#    cos_phi = self.cos_phi    
#    N = self.NLL
#    for i in range(self.NLL):
#      cosEtau = cos_phi*np.exp(-tau/(x[1,...]*T1_sc))        
#      cosEtauN = cosEtau**(N-1)           
#      Etr = np.exp(-TR/(x[1,...]*T1_sc))
#      Etd = np.exp(-td/(x[1,...]*T1_sc))
#      Etau = np.exp(-tau/(x[1,...]*T1_sc))
#      F = (1 - Etau)/(-cosEtau + 1)
#      Q = (-cos_phi*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr)/(cos_phi*cosEtauN*Etr*Etd + 1)
#      Q_F = Q-F
#      for j in range(self.Nproj):
#            n = i*self.Nproj+j+1
#            S[i,j,...] = x[0,...]*M0_sc*sin_phi*((cosEtau)**(n - 1)*(Q_F) + F)
#    
#    return np.mean(S,axis=1)
##    return np.average(np.array(S),axis=1,weights=self.calc_weights(x[1,:,:,:]))
#  def execute_gradient_3D(self, x):
#    grad = np.zeros((2,self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
#    M0_sc = self.M0_sc
#    T1_sc = self.T1_sc
#    TR = self.TR
#    tau = self.tau
#    td = self.td
#    sin_phi = self.sin_phi
#    cos_phi = self.cos_phi    
#    N = self.NLL  
#    
#    ####Precompute
#    cosEtau = cos_phi*np.exp(-tau/(x[1,...]*T1_sc))        
#    cosEtauN = cosEtau**(N-1)           
#    Etr = np.exp(-TR/(x[1,...]*T1_sc))
#    Etd = np.exp(-td/(x[1,...]*T1_sc))
#    Etau = np.exp(-tau/(x[1,...]*T1_sc))
#    F = (1 - Etau)/(-cosEtau + 1)
#    cos_phi_Etau_tmp = cos_phi*cosEtauN*Etr*Etd
#    Q = (-cos_phi*F*(-cosEtauN + 1)*Etr*Etd + 1 - 2*Etd + Etr)/(cos_phi_Etau_tmp + 1)
#    Q_F = Q-F
#    T1sqrt = x[1,...]**2*T1_sc
#    tmp1 = ((-TR*cos_phi*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(T1sqrt*(-cosEtau + 1)) + TR*Etr/(T1sqrt) -\
#              cos_phi**2*tau*(1 - Etau)*(-cosEtauN + 1)*Etr*Etau*Etd/(T1sqrt*(-cosEtau + 1)**2) + \
#              cos_phi*tau*cosEtauN*(1 - Etau)*(N - 1)*Etr*Etd/(T1sqrt*(-cosEtau + 1)) + \
#              cos_phi*tau*(-cosEtauN + 1)*Etr*Etau*Etd/(T1sqrt*(-cosEtau + 1)) - cos_phi*td*(1 - Etau)\
#              *(-cosEtauN + 1)*Etr*Etd/(T1sqrt*(-cosEtau + 1)) - 2*td*Etd/(T1sqrt))/\
#              (cos_phi_Etau_tmp + 1) + (-TR*cos_phi_Etau_tmp/(T1sqrt) -\
#               cos_phi*tau*cosEtauN*(N - 1)*Etr*Etd/(T1sqrt) - cos_phi*td*cosEtauN*Etr*Etd/(T1sqrt))*\
#               (-cos_phi*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(-cosEtau + 1) + 1 - 2*Etd + Etr)/(cos_phi*cosEtauN*Etr*\
#                   Etd + 1)**2 - cos_phi*tau*(1 - Etau)*Etau/(T1sqrt*(-cosEtau + 1)**2) + tau*Etau/(T1sqrt*(-cosEtau + 1)))
#    tmp2 = cos_phi*tau*(1 - Etau)*Etau/(T1sqrt*(-cosEtau + 1)**2) - tau*Etau/(T1sqrt*(-cosEtau + 1))        
#    
#    
#    for i in range(self.NLL):  
#      for j in range(self.Nproj):
#            n = i*self.Nproj+j+1
#            
#            grad[0,i,j,...] = M0_sc*sin_phi*((cosEtau)**(n - 1)*(Q_F) + F)
#            
#            grad[1,i,j,...] =x[0,...]*M0_sc*sin_phi*((cosEtau)**(n - 1)*tmp1 + tmp2 + \
#                tau*(cosEtau)**(n - 1)*(n - 1)*(-F + (-cos_phi*(1 - Etau)*(-cosEtauN + 1)*Etr*Etd/(-cosEtau + 1) + 1 - 2*Etd +\
#                Etr)/(cos_phi_Etau_tmp + 1))/(T1sqrt))
#            
#    return np.mean(grad,axis=2)
                                         
#    return np.average(np.array(grad),axis=2,weights=np.tile(self.calc_weights(x[1,:,:,:]),(2,1,1,1,1,1)))


           
#  cpdef calc_weights(self,DTYPE_t[:,:,::1] x):
#      cdef int i=0,j=0
#      cdef np.ndarray[ndim=3,dtype=DTYPE_t] T1 = np.array(x)*self.T1_sc
#      cdef np.ndarray[ndim=3,dtype=DTYPE_t] w = np.zeros_like(T1)
#      cdef np.ndarray[ndim=3,dtype=DTYPE_t] V = np.ones_like(T1)
#      cdef np.ndarray[ndim=5,dtype=DTYPE_t] result = np.zeros((self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
#      for i in range(self.NLL):
##           w = 1-np.np.exp(-(self.tau)/T1)
##           V[~(w==1)] = (1-w[~(w==1)]**self.Nproj)/(1-w[~(w==1)])
#           for j in range(self.Nproj):
#               result[i,j,:,:,:] = np.np.exp(-(self.td+self.tau*j+self.tau*self.Nproj*i)/T1)/np.np.exp(-(self.td+self.tau*self.Nproj*i)/T1)
#           result[i,:,:,:,:] = result[i,:,:,:,:]/np.sum(result[i,:,:,:,:],0)
#            
#      return np.squeeze(result)
  def plot_unknowns(self,x,dim_2D=False):
      
      if dim_2D:
          plt.figure(1)
          plt.imshow(np.transpose(np.abs(x[0,...]*self.M0_sc)))
          plt.pause(0.05)
          plt.figure(2)
          plt.imshow(np.transpose(-100/np.log(np.abs(x[1,...]*self.T1_sc))))
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
          plt.imshow(np.transpose(-100/np.log(np.abs(x[1,int(self.NSlice/2),...]*self.T1_sc))))
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
    Etau =x[1,...]**(tau/100) #np.exp(-tau/(x[1,...]*T1_sc))    
    Etr = x[1,...]**(TR/100)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = x[1,...]**(td/100)#np.exp(-td/(x[1,...]*T1_sc)) 
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

    Efit = x[1,...]           
    Etau =x[1,...]**(tau/100) #np.exp(-tau/(x[1,...]*T1_sc))    
    Etr = x[1,...]**(TR/100)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = x[1,...]**(td/100)#np.exp(-td/(x[1,...]*T1_sc))   

    M0 = x[0,...]
    M0_sc = self.M0_sc
    
  
    F = (1 - Etau)/(1-Etau*cos_phi)
    Q = (-Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)/(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1)
    Q_F = Q-F  
    
    tmp1 = ((-Etr*Etau*Etd*tau*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                *cos_phi**2/(100*Efit*(-Etau*cos_phi + 1)**2) + Etr*Etau*Etd*tau*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(100\
                            *Efit*(-Etau*cos_phi + 1)) - Etr*Etd*TR*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(100*Efit*\
                                  (-Etau*cos_phi + 1)) + Etr*Etd*TR/(100*Efit) + Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(-Etau + 1)*\
                            (N - 1)*cos_phi/(100*Efit*(-Etau*cos_phi + 1)) - Etr*Etd*td*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                            *cos_phi/(100*Efit*(-Etau*cos_phi + 1)) + Etr*Etd*td/(100*Efit) - Etd*td/(50*Efit))/(Etr*Etd*\
                                     (Etau*cos_phi)**(N - 1)*cos_phi + 1) + (-Etr*Etd*TR*(Etau*cos_phi)**(N - 1)*\
                                     cos_phi/(100*Efit) - Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(N - 1)*cos_phi/(100*Efit) - \
                                     Etr*Etd*td*(Etau*cos_phi)**(N - 1)*cos_phi/(100*Efit))*(-Etr*Etd*(-Etau + 1)*\
                                                (-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)\
                            /(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1)**2 - Etau*tau*(-Etau + 1)*cos_phi/\
                            (100*Efit*(-Etau*cos_phi + 1)**2) + Etau*tau/(100*Efit*(-Etau*cos_phi + 1)))
    tmp2 = Etau*tau*(-Etau + 1)\
                            *cos_phi/(100*Efit*(-Etau*cos_phi + 1)**2) - Etau*tau/(100*Efit*(-Etau*cos_phi + 1))
    tmp3 = (-(-Etau + 1)/(-Etau*cos_phi + 1) + (-Etr*Etd*(-Etau + 1)*\
                                                 (-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)\
                            /(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1))/(100*Efit)
    for i in range(self.NLL):  
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            
            grad[0,i,j,...] = M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
            
            grad[1,i,j,...] = M0*M0_sc*((Etau*cos_phi)**(n - 1)*tmp1 + tmp2 + \
                            tau*(Etau*cos_phi)**(n - 1)*(n - 1)*tmp3)*sin_phi
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
    Etau =x[1,...]**(tau/100) #np.exp(-tau/(x[1,...]*T1_sc))    
    Etr = x[1,...]**(TR/100)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = x[1,...]**(td/100)#np.exp(-td/(x[1,...]*T1_sc)) 
    M0 = x[0,...]
    M0_sc = self.M0_sc
        

    F = (1 - Etau)/(1-Etau*cos_phi)
    Q = (-Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)/(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1)
    Q_F = Q-F   
  
    for i in range(self.NLL):
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            S[i,j,...] = M0*M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
    
    return np.mean(S,axis=1)
  
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

    Efit = x[1,...]           
    Etau =x[1,...]**(tau/100) #np.exp(-tau/(x[1,...]*T1_sc))    
    Etr = x[1,...]**(TR/100)#np.exp(-TR/(x[1,...]*T1_sc))
    Etd = x[1,...]**(td/100)#np.exp(-td/(x[1,...]*T1_sc))   

    M0 = x[0,...]
    M0_sc = self.M0_sc
    
  
    F = (1 - Etau)/(1-Etau*cos_phi)
    Q = (-Etr*Etd*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)/(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1)
    Q_F = Q-F  
    
    tmp1 = ((-Etr*Etau*Etd*tau*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                *cos_phi**2/(100*Efit*(-Etau*cos_phi + 1)**2) + Etr*Etau*Etd*tau*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(100\
                            *Efit*(-Etau*cos_phi + 1)) - Etr*Etd*TR*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(100*Efit*\
                                  (-Etau*cos_phi + 1)) + Etr*Etd*TR/(100*Efit) + Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(-Etau + 1)*\
                            (N - 1)*cos_phi/(100*Efit*(-Etau*cos_phi + 1)) - Etr*Etd*td*(-Etau + 1)*(-(Etau*cos_phi)**(N - 1) + 1)\
                            *cos_phi/(100*Efit*(-Etau*cos_phi + 1)) + Etr*Etd*td/(100*Efit) - Etd*td/(50*Efit))/(Etr*Etd*\
                                     (Etau*cos_phi)**(N - 1)*cos_phi + 1) + (-Etr*Etd*TR*(Etau*cos_phi)**(N - 1)*\
                                     cos_phi/(100*Efit) - Etr*Etd*tau*(Etau*cos_phi)**(N - 1)*(N - 1)*cos_phi/(100*Efit) - \
                                     Etr*Etd*td*(Etau*cos_phi)**(N - 1)*cos_phi/(100*Efit))*(-Etr*Etd*(-Etau + 1)*\
                                                (-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)\
                            /(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1)**2 - Etau*tau*(-Etau + 1)*cos_phi/\
                            (100*Efit*(-Etau*cos_phi + 1)**2) + Etau*tau/(100*Efit*(-Etau*cos_phi + 1)))
    tmp2 = Etau*tau*(-Etau + 1)\
                            *cos_phi/(100*Efit*(-Etau*cos_phi + 1)**2) - Etau*tau/(100*Efit*(-Etau*cos_phi + 1))
    tmp3 = (-(-Etau + 1)/(-Etau*cos_phi + 1) + (-Etr*Etd*(-Etau + 1)*\
                                                 (-(Etau*cos_phi)**(N - 1) + 1)*cos_phi/(-Etau*cos_phi + 1) + Etr*Etd - 2*Etd + 1)\
                            /(Etr*Etd*(Etau*cos_phi)**(N - 1)*cos_phi + 1))/(100*Efit)
    for i in range(self.NLL):  
      for j in range(self.Nproj):
            n = i*self.Nproj+j+1
            
            grad[0,i,j,...] = M0_sc*((Etau*cos_phi)**(n - 1)*Q_F + F)*sin_phi
            
            grad[1,i,j,...] = M0*M0_sc*((Etau*cos_phi)**(n - 1)*tmp1 + tmp2 + \
                            tau*(Etau*cos_phi)**(n - 1)*(n - 1)*tmp3)*sin_phi
    return np.array(np.mean(grad,axis=2,dtype=np.complex256),dtype=DTYPE)
             
           
           
           
           
           
           
           
           