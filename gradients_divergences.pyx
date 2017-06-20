#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:14:08 2017

@author: omaier
"""

import numpy as np
cimport numpy as np

DTYPE = np.complex128
ctypedef np.complex_t DTYPE_t

cpdef bdiv_1(np.ndarray[DTYPE_t, ndim=4] v, int dx=1, int dy=1):


    cdef int n = v.shape[3]
    cdef int m = v.shape[2]
    cdef int k = v.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=3] div_v = np.zeros([k,m,n],dtype=DTYPE)
    
    
    div_v[:,:,0] = v[:,0,:,0]/dx
    div_v[:,:,-1] = -v[:,0,:,-2]/dx
    div_v[:,:,1:-1] = (v[:,0,:,1:-1]-v[:,0,:,:-2])/dx
    
    div_v[:,0,:] = div_v[:,0,:] + v[:,1,0,:]/dy
    div_v[:,-1,:] = div_v[:,-1,:] - v[:,1,-2,:]/dy
    div_v[:,1:-1,:] = div_v[:,1:-1,:] + (v[:,1,1:-1,:] - v[:,1,:-2,:])/dy

    return div_v

cpdef fgrad_1(np.ndarray[DTYPE_t, ndim=3] u,int dx=1, int dy=1):

    
    cdef int n = u.shape[2]
    cdef int m = u.shape[1]
    cdef int k = u.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=4] grad = np.zeros([k,2,m,n],dtype=DTYPE)
     
    grad[:,0,:,:-1] = (u[:,:,1:] - u[:,:,:-1])/dx
    
    grad[:,1,:-1,:] = (u[:,1:,:] - u[:,:-1,:])/dy


    return grad
  
cpdef fdiv_2(np.ndarray[DTYPE_t, ndim=4] x,int dx=1, int dy=1):


      
    cdef int k = np.shape(x)[0]
    cdef int i = np.shape(x)[1]
    cdef int m = np.shape(x)[2]
    cdef int n = np.shape(x)[3]
    
    cdef np.ndarray[DTYPE_t, ndim=4] div_x = np.zeros((k,2,m,n),dtype=DTYPE)
    
    div_x[:,0,:,:-1] = (x[:,0,:,1:]-x[:,0,:,:-1])/dx
    div_x[:,0,:-1,:]  = div_x[:,0,:-1,:] + (x[:,2,1:,:]-x[:,2,:-1,:])/dy
    
    div_x[:,1,:,:-1] = (x[:,2,:,1:]-x[:,2,:,:-1])/dx
    div_x[:,1,:-1,:] = div_x[:,1,:-1,:] + (x[:,1,1:,:]-x[:,1,:-1,:])/dy
    return div_x
    
cpdef sym_bgrad_2(np.ndarray[DTYPE_t, ndim=4] x, int dx=1, int dy=1):


    cdef int k = np.shape(x)[0]
    cdef int i = np.shape(x)[1]
    cdef int m = np.shape(x)[2]
    cdef int n = np.shape(x)[3]
    
    cdef np.ndarray[DTYPE_t, ndim=4] grad_x = np.zeros((k,3,m,n),dtype=DTYPE)
    
    grad_x[:,0,:,0] =x[:,0,:,0]/dx
    grad_x[:,0,:,1:-1] = (x[:,0,:,1:-1] - x[:,0,:,:-2])/dx
    grad_x[:,0,:,-1] = -x[:,0,:,-2]/dx
    
    grad_x[:,1,0,:] = x[:,1,0,:]/dy
    grad_x[:,1,1:-1,:] = (x[:,1,1:-1,:]-x[:,1,:-2,:])/dy
    grad_x[:,1,-1,:] = -x[:,1,-2,:]/dy
    
    grad_x[:,2,0,:] = x[:,0,0,:]/dy
    grad_x[:,2,1:-1,:] = (x[:,0,1:-1,:]-x[:,0,:-2,:])/dy
    grad_x[:,2,-1,:] = -x[:,0,-2,:]/dy
    
    grad_x[:,2,:,0] = grad_x[:,2,:,0]+x[:,1,:,0]/dx
    grad_x[:,2,:,1:-1] = grad_x[:,2,:,1:-1] + (x[:,1,:,1:-1] - x[:,1,:,:-2])/dx
    grad_x[:,2,:,-1] = grad_x[:,2,:,-1] - x[:,1,:,-2]/dx
    
    grad_x[:,2,:,:] = grad_x[:,2,:,:]/2
    
    return grad_x  
################################## 3D Functions  
cpdef bdiv_3(np.ndarray[DTYPE_t, ndim=5] v, int dx=1, int dy=1, int dz = 1):


    cdef int n = v.shape[4]
    cdef int m = v.shape[3]
    cdef int l = v.shape[2]
    cdef int k = v.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=4] div_v = np.zeros([k,l,m,n],dtype=DTYPE)
    
    ## dx
    div_v[:,:,:,0] = v[:,0,:,:,0]/dx
    div_v[:,:,:,-1] = -v[:,0,:,:,-2]/dx
    div_v[:,:,:,1:-1] = (v[:,0,:,:,1:-1]-v[:,0,:,:,:-2])/dx
    ## dy
    div_v[:,:,0,:] = div_v[:,:,0,:] + v[:,1,:,0,:]/dy
    div_v[:,:,-1,:] = div_v[:,:,-1,:] - v[:,1,:,-2,:]/dy
    div_v[:,:,1:-1,:] = div_v[:,:,1:-1,:] + (v[:,1,:,1:-1,:] - v[:,1,:,:-2,:])/dy
    ## dz
    div_v[:,0,:,:] = div_v[:,0,:,:] + v[:,2,0,:,:]/dz
    div_v[:,-1,:,:] = div_v[:,-1,:,:] - v[:,2,-2,:,:]/dz
    div_v[:,1:-1,:,:] = div_v[:,1:-1,:,:] + (v[:,2,1:-1,:,:] - v[:,2,:-2,:,:])/dz   
    return div_v

cpdef fgrad_3(np.ndarray[DTYPE_t, ndim=4] u,int dx=1, int dy=1, int dz = 1):

    
    cdef int n = u.shape[3]
    cdef int m = u.shape[2]
    cdef int l = u.shape[1]
    cdef int k = u.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=5] grad = np.zeros([k,3,l,m,n],dtype=DTYPE)
    
    ##dx
    grad[:,0,:,:,:-1] = (u[:,:,:,1:] - u[:,:,:,:-1])/dx
    ##dy
    grad[:,1,:,:-1,:] = (u[:,:,1:,:] - u[:,:,:-1,:])/dy
    ##dz
    grad[:,2,:-1,:,:] = (u[:,1:,:,:] - u[:,:-1,:,:])/dz
    return grad
  
cpdef fdiv_3(np.ndarray[DTYPE_t, ndim=5] x,int dx=1, int dy=1, int dz=1):

      
    cdef int n = x.shape[4]
    cdef int m = x.shape[3]
    cdef int l = x.shape[2]
    cdef int k = x.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=5] div_x = np.zeros((k,3,l,m,n),dtype=DTYPE)
    
    div_x[:,0,:,:,1:-1] = (x[:,0,:,:,2:]-x[:,0,:,:,1:-1])/dx
    div_x[:,0,:,:,0] = x[:,0,:,:,1]/dx
    div_x[:,0,:,:,-1] = -x[:,0,:,:,-1]/dx
    
    div_x[:,0,:,1:-1,:] = div_x[:,0,:,1:-1,:]+(x[:,3,:,2:,:]-x[:,3,:,1:-1,:])/dy
    div_x[:,0,:,0,:] = div_x[:,0,:,0,:] + x[:,3,:,1,:]/dy
    div_x[:,0,:,-1,:] = div_x[:,0,:,-1,:] - x[:,3,:,-1,:]/dy
    
    div_x[:,0,1:-1,:,:] = div_x[:,0,1:-1,:,:] + (x[:,4,2:,:,:]-x[:,4,1:-1,:,:])/dz
    div_x[:,0,0,:,:] = div_x[:,0,0,:,:] + x[:,4,1,:,:]/dz
    div_x[:,0,-1,:,:] = div_x[:,0,-1,:,:] - x[:,4,-1,:,:]/dz
    
    div_x[:,1,:,:,1:-1] = (x[:,3,:,:,2:]-x[:,3,:,:,1:-1])/dx
    div_x[:,1,:,:,0] = x[:,3,:,:,1]/dx
    div_x[:,1,:,:,-1] = -x[:,3,:,:,-1]/dx
    
    div_x[:,1,:,1:-1,:] = div_x[:,1,:,1:-1,:] +(x[:,1,:,2:,:]-x[:,1,:,1:-1,:])/dy
    div_x[:,1,:,0,:] = div_x[:,1,:,0,:] +x[:,1,:,1,:]/dy
    div_x[:,1,:,-1,:] = div_x[:,1,:,-1,:] - x[:,1,:,-1,:]/dy
    
    div_x[:,1,1:-1,:,:] = div_x[:,1,1:-1,:,:] + (x[:,5,2:,:,:]-x[:,5,1:-1,:,:])/dz
    div_x[:,1,0,:,:] = div_x[:,1,0,:,:] + x[:,5,1,:,:]/dz
    div_x[:,1,-1,:,:] = div_x[:,1,-1,:,:] - x[:,5,-1,:,:]/dz
    
    div_x[:,2,:,:,1:-1] = (x[:,4,:,:,2:]-x[:,4,:,:,1:-1])/dx
    div_x[:,2,:,:,0] = x[:,4,:,:,1]/dx
    div_x[:,2,:,:,-1] = -x[:,4,:,:,-1]/dx
    
    div_x[:,2,:,1:-1,:] = div_x[:,2,:,1:-1,:] + (x[:,5,:,2:,:]-x[:,5,:,1:-1,:])/dy
    div_x[:,2,:,0,:] = div_x[:,2,:,0,:] + x[:,5,:,1,:]/dy
    div_x[:,2,:,-1,:] = div_x[:,2,:,-1,:] - x[:,5,:,-1,:]/dy
    
    div_x[:,2,1:-1,:,:] = div_x[:,2,1:-1,:,:] + (x[:,2,2:,:,:]-x[:,2,1:-1,:,:])/dz
    div_x[:,2,0,:,:] = div_x[:,2,0,:,:] + x[:,2,1,:,:]/dz
    div_x[:,2,-1,:,:] = div_x[:,2,-1,:,:] - x[:,2,-1,:,:]/dz
    
    
    

    return div_x
    
cpdef sym_bgrad_3(np.ndarray[DTYPE_t, ndim=5] x, int dx=1, int dy=1, int dz=1):


    cdef int n = x.shape[4]
    cdef int m = x.shape[3]
    cdef int l = x.shape[2]
    cdef int k = x.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=5] grad_x = np.zeros((k,6,l,m,n),dtype=DTYPE)
    
    grad_x[:,0,:,:,1:] = (x[:,0,:,:,1:]-x[:,0,:,:,:-1])/dx
    
    grad_x[:,1,:,1:,:] = (x[:,1,:,1:,:]-x[:,1,:,:-1,:])/dy
    
    grad_x[:,2,1:,:,:] = (x[:,2,1:,:,:]-x[:,2,:-1,:,:])/dz
    
    grad_x[:,3,:,1:,:] = (x[:,0,:,1:,:]-x[:,0,:,:-1,:])/dy
    grad_x[:,3,:,:,1:] = grad_x[:,3,:,:,1:]+(x[:,1,:,:,1:]-x[:,1,:,:,:-1])/dx    
    grad_x[:,3,:,:,:] = grad_x[:,3,:,:,:]/2
    
    
    grad_x[:,4,1:,:,:] = (x[:,0,1:,:,:]-x[:,0,:-1,:,:])/dz
    grad_x[:,4,:,:,1:] = grad_x[:,4,:,:,1:] + (x[:,2,:,:,1:]-x[:,2,:,:,:-1])/dx    
    grad_x[:,4,:,:,:] = grad_x[:,4,:,:,:]/2
    
    
    grad_x[:,5,1:,:,:] = (x[:,1,1:,:,:]-x[:,1,:-1,:,:])/dz
    grad_x[:,5,:,1:,:] = grad_x[:,5,:,1:,:] + (x[:,2,:,1:,:]-x[:,2,:,:-1,:])/dy
    grad_x[:,5,:,:,:] = grad_x[:,5,:,:,:]/2
    
    
    
    
    
    return grad_x    
  