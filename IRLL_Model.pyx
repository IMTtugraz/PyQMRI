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
cimport numpy as np
np.import_array()

cimport cython
from cython.parallel import parallel, prange

DTYPE = np.complex128
ctypedef double complex DTYPE_t

cdef extern from "complex.h":
    DTYPE_t cexp(DTYPE_t z) nogil
    DTYPE_t cpow(DTYPE_t z) nogil
    
@cython.cdivision(True)      
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function    
cdef class IRLL_Model:
  
  cdef public DTYPE_t TR, fa, T1_sc, M0_sc, tau, td
  cdef public int NLL, Nproj, dimY, dimX, NSlice
  cdef public DTYPE_t[:,:,::1] fa_corr, phi_corr, sin_phi, cos_phi
  cdef public DTYPE_t[:,:,:,::1] guess
  
  
  def __init__(self,DTYPE_t fa, DTYPE_t[:,:,::1] fa_corr,DTYPE_t TR,DTYPE_t tau,DTYPE_t td,
               int NScan,int NSlice,int dimY,int dimX, int Nproj):


    self.NSlice = NSlice
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
    phi_corr = np.real(fa)*np.real(fa_corr) + 1j*np.imag(fa)*np.imag(fa_corr)

    
    self.sin_phi = np.sin(phi_corr)
    self.cos_phi = np.cos(phi_corr)    

    self.guess = np.array([0.05*np.ones((NSlice,dimY,dimX),dtype=DTYPE),0.3*np.ones((NSlice,dimY,dimX),dtype=DTYPE)])               
    

  cpdef execute_forward_2D(self,DTYPE_t[:,:,::1] x,int islice):
    cdef DTYPE_t[:,:,:,::1] S = np.zeros((self.NLL,self.Nproj,self.dimY,self.dimX),dtype='complex128')
    cdef int i = 0
    cdef int j = 0
    cdef int n_i = 0
    cdef int m=0, n=0
    cdef DTYPE_t[:,:,::1] sin_phi = self.sin_phi
    cdef DTYPE_t[:,:,::1] cos_phi = self.cos_phi    
    for i in range(self.NLL):
      for j in range(self.Nproj):
        for m in range(self.dimY):
          for n in range(self.dimX):
            n_i = i*self.Nproj+j+1
            S[i,j,m,n] = (x[0,m,n]*self.M0_sc*sin_phi[islice,m,n]*(((cexp(-self.TR/(x[1,m,n]*self.T1_sc)) - 
                    2*cexp(-self.td/(x[1,m,n]*self.T1_sc)) + (cexp(-self.TR/(x[1,m,n]*self.T1_sc))
                    *cexp(-self.td/(x[1,m,n]*self.T1_sc))*cos_phi[islice,m,n]*((cexp(-self.tau/(x[1,m,n]
                    *self.T1_sc))*cos_phi[islice,m,n])**(self.NLL - 1) - 1)*(cexp(-self.tau/(x[1,m,n]*
                    self.T1_sc)) - 1))/(cexp(-self.tau/(x[1,m,n]*self.T1_sc))*cos_phi[islice,m,n] - 1)
                    + 1)/(cexp(-self.TR/(x[1,m,n]*self.T1_sc))*cexp(-self.td/(x[1,m,n]*self.T1_sc))
                    *cos_phi[islice,m,n]*(cexp(-self.tau/(x[1,m,n]*self.T1_sc))
                    *cos_phi[islice,m,n])**(self.NLL - 1) + 1) - (cexp(-self.tau/(x[1,m,n]*self.T1_sc)) 
                    - 1)/(cexp(-self.tau/(x[1,m,n]*self.T1_sc))*cos_phi[islice,m,n] - 1))*
                    (cexp(-self.tau/(x[1,m,n]*self.T1_sc))*cos_phi[islice,m,n])**(n_i - 1) 
                    + (cexp(-self.tau/(x[1,m,n]*self.T1_sc)) - 1)/(cexp(-self.tau/(x[1,m,n]
                    *self.T1_sc))*cos_phi[islice,m,n] - 1)))
    
#    S[~np.isfinite(S)] = 1e-20
    return np.mean(S,axis=1)
  cpdef execute_gradient_2D(self,DTYPE_t[:,:,::1] x,int islice):
    cdef DTYPE_t[:,:,:,:,::1] grad = np.zeros((2,self.NLL,self.Nproj,self.dimY,self.dimX),dtype=DTYPE)
    cdef int i = 0
    cdef int j = 0  
    cdef int n_i = 0
    cdef DTYPE_t M0_sc = self.M0_sc
    cdef DTYPE_t T1_sc = self.T1_sc
    cdef DTYPE_t td = self.td
    cdef int NLL = self.NLL
    cdef DTYPE_t TR = self.TR
    cdef DTYPE_t tau = self.tau
    cdef int Nproj = self.Nproj
    cdef DTYPE_t E1=0+1j*0, TAU1=0+1j*0, TD1=0+1j*0, TRTD1=0+1j*0, TAUp1=0+1j*0, TAU1cosphi=0+1j*0
    cdef DTYPE_t[:,:,::1] sin_phi = self.sin_phi
    cdef DTYPE_t[:,:,::1] cos_phi = self.cos_phi
    cdef DTYPE_t T1
    
#    sin_phi = self.sin_phi
#    cos_phi = self.cos_phi
    cdef int m=0, n=0    
    
    with nogil, parallel():
      for i in range(NLL):
        for j in range(Nproj):
          for m in range(self.dimY):
            for n in prange(self.dimX):
#              T1 = x[1,m,n]
              n_i = i*Nproj+j+1
              E1 = cexp(-TR/(x[1,m,n]*T1_sc))
              TAU1 = cexp(-tau/(x[1,m,n]*T1_sc))
              TD1 = cexp(-td/(x[1,m,n]*T1_sc))
              TRTD1 = cexp(-(TR + td)/(x[1,m,n]*T1_sc))
              TAUp1 = cexp(tau/(x[1,m,n]*T1_sc))
              TAU1cosphi = TAU1*cos_phi[islice,m,n]
              
              grad[0,i,j,m,n] = (M0_sc*sin_phi[islice,m,n]*(((E1 - 2*TD1 + (E1
                                *TD1*cos_phi[islice,m,n]
                                *((TAU1cosphi)
                                **(NLL - 1) - 1)*(TAU1 - 1))
                                /(TAU1cosphi - 1)
                                + 1)/(E1*TD1
                                *cos_phi[islice,m,n]*(TAU1cosphi)**(NLL - 1) + 1) 
                                - (TAU1- 1)/(TAU1cosphi - 1))*(TAU1cosphi)**(n_i - 1) 
                                + (TAU1 - 1)/(TAU1cosphi - 1)))
            
              grad[1,i,j,m,n] = (x[0,m,n]*M0_sc*sin_phi[islice,m,n]
                                *((TAU1cosphi)
                                **(n_i - 1)*(((TR*E1)
                                /(x[1,m,n]**2*T1_sc) - (2*td*TD1)/(x[1,m,n]**2*T1_sc) + 
                                (tau*cexp(-(TR + tau + td)/(x[1,m,n]*T1_sc))*cos_phi[islice,m,n]
                                *((TAU1cosphi)
                                **(NLL - 1) - 1))/(x[1,m,n]**2*T1_sc*(TAU1cosphi - 1)) 
                                - (tau*cexp(-(TR + tau + td)/(x[1,m,n]
                                *T1_sc))*cos_phi[islice,m,n]**2*((TAU1cosphi)**(NLL - 1) - 1)
                                *(TAU1 - 1))/(x[1,m,n]**2*T1_sc
                                *(TAU1cosphi - 1)**2)
                                + (TR*TRTD1
                                *cos_phi[islice,m,n]*((TAU1cosphi)**(NLL - 1) - 1)
                                *(TAU1 - 1))/(x[1,m,n]
                                **2*T1_sc*(TAU1cosphi - 1)) + (td
                                *TRTD1
                                *cos_phi[islice,m,n]*((TAU1cosphi)**(NLL - 1) - 1)
                                *(TAU1 - 1))
                                /(x[1,m,n]**2*T1_sc*(TAU1cosphi - 1)) + (tau
                                *cexp(-(TR + tau + td)/(x[1,m,n]*T1_sc))
                                *cos_phi[islice,m,n]**2*(NLL - 1)*(TAU1cosphi)**
                                (NLL - 2)*(TAU1 - 1))/(x[1,m,n]**2*T1_sc
                                *(TAU1cosphi - 1)))
                                /(TRTD1*cos_phi[islice,m,n]
                                *(TAU1cosphi)
                                **(NLL - 1) + 1) + tau/(x[1,m,n]**2*T1_sc
                                *(TAUp1 - cos_phi[islice,m,n])) 
                                + (tau*TAU1cosphi
                                *(TAU1 - 1))/(x[1,m,n]
                                **2*T1_sc*(TAU1cosphi - 1)**2) - (cexp(-(TR - tau + td)
                                /(x[1,m,n]*T1_sc))*(TAU1cosphi)**NLL*(TR - tau 
                                + td + NLL*tau)*(E1 - 2*TD1 
                                + (TRTD1
                                *cos_phi[islice,m,n]*((TAU1cosphi)**(NLL - 1) - 1)
                                *(TAU1 - 1))
                                /(TAU1cosphi - 1) + 1))
                                /(x[1,m,n]**2*T1_sc*(TRTD1*cos_phi[islice,m,n]
                                *(TAU1cosphi)
                                **(NLL - 1) + 1)**2)) - tau/(x[1,m,n]**2*T1_sc
                                *(TAUp1 - cos_phi[islice,m,n])) 
                                - (tau*TAU1cosphi*(TAU1 - 1))
                                /(x[1,m,n]**2*T1_sc*(TAU1cosphi - 1)**2) + (tau*TAU1cosphi*((E1 - 2*TD1 
                                + (TRTD1*cos_phi[islice,m,n]
                                *((TAU1cosphi)
                                **(NLL - 1) - 1)*(TAU1 - 1))
                                /(TAU1cosphi - 1) + 1)
                                /(TRTD1*cos_phi[islice,m,n]
                                *(TAU1cosphi)
                                **(NLL - 1) + 1) - (TAUp1 - 1)
                                /(TAUp1 - cos_phi[islice,m,n]))
                                *(n_i - 1)*(TAU1cosphi)
                                **(n_i - 2))/(x[1,m,n]**2*T1_sc)))                     
                              
                            
                                       
    return np.mean(grad,axis=2)
  
  
  cpdef execute_forward_3D(self,DTYPE_t[:,:,:,::1] x):
    cdef DTYPE_t[:,:,:,:,::1] S = np.zeros((self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype='complex128')
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef int n_i = 0
    cdef int m=0, n=0
    cdef DTYPE_t[:,:,::1] sin_phi = self.sin_phi
    cdef DTYPE_t[:,:,::1] cos_phi = self.cos_phi    
    with nogil, parallel():
        for i in range(self.NLL):
          for j in range(self.Nproj):
            for k in range(self.NSlice):
              for m in range(self.dimY):
                for n in prange(self.dimX):
                  n_i = i*self.Nproj+j+1
                  S[i,j,k,m,n] = (x[0,k,m,n]*self.M0_sc*sin_phi[k,m,n]*(((cexp(-self.TR/(x[1,k,m,n]*self.T1_sc)) - 
                          2*cexp(-self.td/(x[1,k,m,n]*self.T1_sc)) + (cexp(-self.TR/(x[1,k,m,n]*self.T1_sc))
                          *cexp(-self.td/(x[1,k,m,n]*self.T1_sc))*cos_phi[k,m,n]*((cexp(-self.tau/(x[1,k,m,n]
                          *self.T1_sc))*cos_phi[k,m,n])**(self.NLL - 1) - 1)*(cexp(-self.tau/(x[1,k,m,n]*
                          self.T1_sc)) - 1))/(cexp(-self.tau/(x[1,k,m,n]*self.T1_sc))*cos_phi[k,m,n] - 1)
                          + 1)/(cexp(-self.TR/(x[1,k,m,n]*self.T1_sc))*cexp(-self.td/(x[1,k,m,n]*self.T1_sc))
                          *cos_phi[k,m,n]*(cexp(-self.tau/(x[1,k,m,n]*self.T1_sc))
                          *cos_phi[k,m,n])**(self.NLL - 1) + 1) - (cexp(-self.tau/(x[1,k,m,n]*self.T1_sc)) 
                          - 1)/(cexp(-self.tau/(x[1,k,m,n]*self.T1_sc))*cos_phi[k,m,n] - 1))*
                          (cexp(-self.tau/(x[1,k,m,n]*self.T1_sc))*cos_phi[k,m,n])**(n_i - 1) 
                          + (cexp(-self.tau/(x[1,k,m,n]*self.T1_sc)) - 1)/(cexp(-self.tau/(x[1,k,m,n]
                          *self.T1_sc))*cos_phi[k,m,n] - 1)))
    
#    S[~np.isfinite(S)] = 1e-20
    return np.mean(S,axis=1)
  cpdef execute_gradient_3D(self,DTYPE_t[:,:,:,::1] x):
    cdef DTYPE_t[:,:,:,:,:,::1] grad = np.zeros((2,self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
    cdef int i = 0
    cdef int j = 0  
    cdef int k = 0
    cdef int n_i = 0
    cdef DTYPE_t M0_sc = self.M0_sc
    cdef DTYPE_t T1_sc = self.T1_sc
    cdef DTYPE_t td = self.td
    cdef int NLL = self.NLL
    cdef DTYPE_t TR = self.TR
    cdef DTYPE_t tau = self.tau
    cdef int Nproj = self.Nproj
    cdef DTYPE_t E1=0+1j*0, TAU1=0+1j*0, TD1=0+1j*0, TRTD1=0+1j*0, TAUp1=0+1j*0, TAU1cosphi=0+1j*0
    cdef DTYPE_t[:,:,::1] sin_phi = self.sin_phi
    cdef DTYPE_t[:,:,::1] cos_phi = self.cos_phi
    cdef DTYPE_t T1
    
#    sin_phi = self.sin_phi
#    cos_phi = self.cos_phi
    cdef int m=0, n=0    
    
    with nogil, parallel():
      for i in range(NLL):
        for j in range(Nproj):
          for k in range(self.NSlice):
            for m in range(self.dimY):
              for n in prange(self.dimX):
  #              T1 = x[1,m,n]
                n_i = i*Nproj+j+1
                E1 = cexp(-TR/(x[1,k,m,n]*T1_sc))
                TAU1 = cexp(-tau/(x[1,k,m,n]*T1_sc))
                TD1 = cexp(-td/(x[1,k,m,n]*T1_sc))
                TRTD1 = cexp(-(TR + td)/(x[1,k,m,n]*T1_sc))
                TAUp1 = cexp(tau/(x[1,k,m,n]*T1_sc))
                TAU1cosphi = TAU1*cos_phi[k,m,n]
                
                grad[0,i,j,k,m,n] = (M0_sc*sin_phi[k,m,n]*(((E1 - 2*TD1 + (E1
                                  *TD1*cos_phi[k,m,n]
                                  *((TAU1cosphi)
                                  **(NLL - 1) - 1)*(TAU1 - 1))
                                  /(TAU1cosphi - 1)
                                  + 1)/(E1*TD1
                                  *cos_phi[k,m,n]*(TAU1cosphi)**(NLL - 1) + 1) 
                                  - (TAU1- 1)/(TAU1cosphi - 1))*(TAU1cosphi)**(n_i - 1) 
                                  + (TAU1 - 1)/(TAU1cosphi - 1)))
              
                grad[1,i,j,k,m,n] = (x[0,k,m,n]*M0_sc*sin_phi[k,m,n]
                                  *((TAU1cosphi)
                                  **(n_i - 1)*(((TR*E1)
                                  /(x[1,k,m,n]**2*T1_sc) - (2*td*TD1)/(x[1,k,m,n]**2*T1_sc) + 
                                  (tau*cexp(-(TR + tau + td)/(x[1,k,m,n]*T1_sc))*cos_phi[k,m,n]
                                  *((TAU1cosphi)
                                  **(NLL - 1) - 1))/(x[1,k,m,n]**2*T1_sc*(TAU1cosphi - 1)) 
                                  - (tau*cexp(-(TR + tau + td)/(x[1,k,m,n]
                                  *T1_sc))*cos_phi[k,m,n]**2*((TAU1cosphi)**(NLL - 1) - 1)
                                  *(TAU1 - 1))/(x[1,k,m,n]**2*T1_sc
                                  *(TAU1cosphi - 1)**2)
                                  + (TR*TRTD1
                                  *cos_phi[k,m,n]*((TAU1cosphi)**(NLL - 1) - 1)
                                  *(TAU1 - 1))/(x[1,k,m,n]
                                  **2*T1_sc*(TAU1cosphi - 1)) + (td
                                  *TRTD1
                                  *cos_phi[k,m,n]*((TAU1cosphi)**(NLL - 1) - 1)
                                  *(TAU1 - 1))
                                  /(x[1,k,m,n]**2*T1_sc*(TAU1cosphi - 1)) + (tau
                                  *cexp(-(TR + tau + td)/(x[1,k,m,n]*T1_sc))
                                  *cos_phi[k,m,n]**2*(NLL - 1)*(TAU1cosphi)**
                                  (NLL - 2)*(TAU1 - 1))/(x[1,k,m,n]**2*T1_sc
                                  *(TAU1cosphi - 1)))
                                  /(TRTD1*cos_phi[k,m,n]
                                  *(TAU1cosphi)
                                  **(NLL - 1) + 1) + tau/(x[1,k,m,n]**2*T1_sc
                                  *(TAUp1 - cos_phi[k,m,n])) 
                                  + (tau*TAU1cosphi
                                  *(TAU1 - 1))/(x[1,k,m,n]
                                  **2*T1_sc*(TAU1cosphi - 1)**2) - (cexp(-(TR - tau + td)
                                  /(x[1,k,m,n]*T1_sc))*(TAU1cosphi)**NLL*(TR - tau 
                                  + td + NLL*tau)*(E1 - 2*TD1 
                                  + (TRTD1
                                  *cos_phi[k,m,n]*((TAU1cosphi)**(NLL - 1) - 1)
                                  *(TAU1 - 1))
                                  /(TAU1cosphi - 1) + 1))
                                  /(x[1,k,m,n]**2*T1_sc*(TRTD1*cos_phi[k,m,n]
                                  *(TAU1cosphi)
                                  **(NLL - 1) + 1)**2)) - tau/(x[1,k,m,n]**2*T1_sc
                                  *(TAUp1 - cos_phi[k,m,n])) 
                                  - (tau*TAU1cosphi*(TAU1 - 1))
                                  /(x[1,k,m,n]**2*T1_sc*(TAU1cosphi - 1)**2) + (tau*TAU1cosphi*((E1 - 2*TD1 
                                  + (TRTD1*cos_phi[k,m,n]
                                  *((TAU1cosphi)
                                  **(NLL - 1) - 1)*(TAU1 - 1))
                                  /(TAU1cosphi - 1) + 1)
                                  /(TRTD1*cos_phi[k,m,n]
                                  *(TAU1cosphi)
                                  **(NLL - 1) + 1) - (TAUp1 - 1)
                                  /(TAUp1 - cos_phi[k,m,n]))
                                  *(n_i - 1)*(TAU1cosphi)
                                  **(n_i - 2))/(x[1,k,m,n]**2*T1_sc)))                     
                              
                            
                                       
    return np.mean(grad,axis=2)
    
  