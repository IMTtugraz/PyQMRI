#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:42:42 2017

@author: omaier
"""

import numpy as np
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

plt.ion()
DTYPE = np.complex64



class constraint:
  def __init__(self, min_val=-np.inf, max_val=np.inf, real_const=False):
    self.min = min_val
    self.max = max_val
    self.real = real_const
  def update(self,scale):
    self.min = self.min/scale
    self.max = self.max/scale


class VFA_Model:
  def __init__(self,fa,fa_corr,TR,images,Nislice,Nproj):
    self.constraints = []
    self.TR = TR
    self.images = images
    self.fa = fa
    self.fa_corr = fa_corr
    self.Nislice = Nislice
    self.figure = None

    (NScan,Nislice,dimX,dimY) = images.shape

    phi_corr = np.zeros_like(images,dtype=DTYPE)
    for i in range(np.size(fa)):
      phi_corr[i,:,:,:] = fa[i]*fa_corr

    self.sin_phi = np.sin(phi_corr)
    self.cos_phi = np.cos(phi_corr)


    self.M0_sc = 1
    self.T1_sc = 1

    test_T1 = np.reshape(np.linspace(10,5500,dimX*dimY*Nislice),(Nislice,dimX,dimY))
    test_M0 = 0.1*np.sqrt((dimX*np.pi/2)/Nproj)
    test_T1 = 1/self.T1_sc*np.exp(-self.TR/(test_T1*np.ones((Nislice,dimY,dimX),dtype=DTYPE)))


    G_x = self.execute_forward_3D(np.array([test_M0/self.M0_sc*np.ones((Nislice,dimY,dimX),dtype=DTYPE),test_T1],dtype=DTYPE))
    self.M0_sc = self.M0_sc*np.median(np.abs(images))/np.median(np.abs(G_x))

    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.M0_sc*np.ones((Nislice,dimY,dimX),dtype=DTYPE),test_T1],dtype=DTYPE))
    self.T1_sc = self.T1_sc*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...]))

    self.T1_sc = self.T1_sc/np.sqrt(self.M0_sc)
    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.M0_sc*np.ones((Nislice,dimY,dimX),dtype=DTYPE),test_T1],dtype=DTYPE))
#    print('Grad Scaling init', np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...])))
#    print('T1 scale: ',self.T1_sc,
#                              '/ M0_scale: ',self.M0_sc)


    result = np.array([1/self.M0_sc*np.ones((Nislice,dimY,dimX),dtype=DTYPE),1/self.T1_sc*np.exp(-self.TR/(800*np.ones((Nislice,dimY,dimX),dtype=DTYPE)))],dtype=DTYPE)
    self.guess = result
    self.constraints.append(constraint(-20/self.M0_sc,20/self.M0_sc,False)  )
    self.constraints.append(constraint(np.exp(-self.TR/(50))/self.T1_sc,np.exp(-self.TR/(5500))/self.T1_sc,True))

  def execute_forward_2D(self,x,islice):
    print('T1_sc: ',self.T1_sc)
    E1 = x[1,...]*self.T1_sc
    S = x[0,:,:]*self.M0_sc*(-E1 + 1)*self.sin_phi[:,islice,:,:]/(-E1*self.cos_phi[:,islice,:,:] + 1)
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)
    return S
  def execute_gradient_2D(self,x,islice):
    E1 = x[1,:,:]*self.T1_sc
    M0 = x[0,...]
    E1[~np.isfinite(E1)] = 0
    grad_M0 = self.M0_sc*(-E1 + 1)*self.sin_phi[:,islice,:,:]/(-E1*self.cos_phi[:,islice,:,:] + 1)
    grad_T1 = M0*self.M0_sc*self.T1_sc*(-E1 + 1)*self.sin_phi[:,islice,:,:]*self.cos_phi[:,islice,:,:]/(-E1*self.cos_phi[:,islice,:,:] + 1)**2 -\
    M0*self.M0_sc*self.T1_sc*self.sin_phi[:,islice,:,:]/(-E1*self.cos_phi[:,islice,:,:] + 1)
    grad = np.array([grad_M0,grad_T1],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
#    print('Grad Scaling', np.linalg.norm(np.abs(grad_M0))/np.linalg.norm(np.abs(grad_T1)))
    return grad

  def execute_forward_3D(self,x):
#    print('T1_sc: ',self.T1_sc)
    E1 = x[1,...]*self.T1_sc
    S = x[0,:,:]*self.M0_sc*(-E1 + 1)*self.sin_phi/(-E1*self.cos_phi + 1)
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)
    return S
  def execute_gradient_3D(self,x):
    E1 = x[1,:,:]*self.T1_sc
    M0 = x[0,...]
    E1[~np.isfinite(E1)] = 0
    grad_M0 = self.M0_sc*(-E1 + 1)*self.sin_phi/(-E1*self.cos_phi + 1)
    grad_T1 = M0*self.M0_sc*self.T1_sc*(-E1 + 1)*self.sin_phi*self.cos_phi/(-E1*self.cos_phi + 1)**2 -\
    M0*self.M0_sc*self.T1_sc*self.sin_phi/(-E1*self.cos_phi + 1)
    grad = np.array([grad_M0,grad_T1],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
#    print('Grad Scaling', np.linalg.norm(np.abs(grad_M0))/np.linalg.norm(np.abs(grad_T1)))
    return grad


  def plot_unknowns(self,x,dim_2D=False):
      M0 = np.abs(x[0,...]*self.M0_sc)
      T1 = np.abs(-self.TR/np.log(x[1,...]*self.T1_sc))
      M0_min = M0.min()
      M0_max = M0.max()
      T1_min = T1.min()
      T1_max = T1.max()

      if dim_2D:
         if not self.figure:
           plt.ion()
           self.figure, self.ax = plt.subplots(1,2,figsize=(12,5))
           self.M0_plot = self.ax[0].imshow((M0))
           self.ax[0].set_title('Proton Density in a.u.')
           self.ax[0].axis('off')
           self.figure.colorbar(self.M0_plot,ax=self.ax[0])
           self.T1_plot = self.ax[1].imshow((T1))
           self.ax[1].set_title('T1 in  ms')
           self.ax[1].axis('off')
           self.figure.colorbar(self.T1_plot,ax=self.ax[1])
           self.figure.tight_layout()
           plt.draw()
           plt.pause(1e-10)
         else:
           self.M0_plot.set_data((M0))
           self.M0_plot.set_clim([M0_min,M0_max])
           self.T1_plot.set_data((T1))
           self.T1_plot.set_clim([T1_min,T1_max])
           plt.draw()
           plt.pause(1e-10)
      else:
         if not self.figure:
           plt.ion()
           self.figure, self.ax = plt.subplots(1,2,figsize=(12,5))
           self.M0_plot=self.ax[0].imshow((M0[int(self.Nislice/2),...]))
           self.ax[0].set_title('Proton Density in a.u.')
           self.ax[0].axis('off')
           self.figure.colorbar(self.M0_plot,ax=self.ax[0])
           self.T1_plot=self.ax[1].imshow((T1[int(self.Nislice/2),...]))
           self.ax[1].set_title('T1 in  ms')
           self.ax[1].axis('off')
           self.figure.colorbar(self.T1_plot,ax=self.ax[1])
           self.figure.tight_layout()
           plt.draw()
           plt.pause(1e-10)
         else:
           self.M0_plot.set_data((M0[int(self.Nislice/2),...]))
           self.M0_plot.set_clim([M0_min,M0_max])
           self.T1_plot.set_data((T1[int(self.Nislice/2),...]))
           self.T1_plot.set_clim([T1_min,T1_max])
           plt.draw()
           plt.pause(1e-10)