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
import matplotlib.gridspec as gridspec
plt.ion()
DTYPE = np.complex64
DTYPE_real = np.float32

unknowns_TGV = 2
unknowns_H1 = 0

class constraint:
  def __init__(self, min_val=-np.inf, max_val=np.inf, real_const=False, pos_real=False):
    self.min = min_val
    self.max = max_val
    self.real = real_const
    self.pos_real = pos_real
  def update(self,scale):
    self.min = self.min/scale
    self.max = self.max/scale


class Model:
  def __init__(self,par,images):
    self.constraints = []
    self.TR = par["TR"]
    self.images = images
    self.fa = par["flip_angle(s)"]
    self.fa_corr = par["fa_corr"]
    self.NSlice = par["NSlice"]
    self.figure = None

    (NScan,NSlice,dimX,dimY) = images.shape

    phi_corr = np.zeros_like(images,dtype=DTYPE)
    for i in range(np.size(par["flip_angle(s)"])):
      phi_corr[i,:,:,:] = par["flip_angle(s)"][i]*np.pi/180*par["fa_corr"]

    self.sin_phi = np.sin(phi_corr)
    self.cos_phi = np.cos(phi_corr)

    self.uk_scale=[]
    self.uk_scale.append(1/np.median(np.abs(images)))
    self.uk_scale.append(1)

    test_T1 = np.reshape(np.linspace(50,5500,dimX*dimY*NSlice),(NSlice,dimX,dimY))

    test_M0 =np.ones((NSlice,dimY,dimX),dtype=DTYPE)#*np.sqrt(par["Nproj"]/34)
    test_T1 = 1/self.uk_scale[1]*np.exp(-self.TR/(test_T1))


    G_x = self.execute_forward_3D(np.array([test_M0,test_T1],dtype=DTYPE))

#    print(np.median(np.abs(G_x)))
    self.uk_scale[0]*=1/np.median(np.abs(G_x))

    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.uk_scale[0],test_T1],dtype=DTYPE))
    self.uk_scale[1] = self.uk_scale[1]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...]))

    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.uk_scale[0],test_T1],dtype=DTYPE))
#    print('Grad Scaling init', np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...])))
    print('T1 scale: ',self.uk_scale[1],
                              '/ M0_scale: ',self.uk_scale[0])


    result = np.array([1/self.uk_scale[0]*np.ones((NSlice,dimY,dimX),dtype=DTYPE),1/self.uk_scale[1]*np.exp(-self.TR/(800*np.ones((NSlice,dimY,dimX),dtype=DTYPE)))],dtype=DTYPE)
    self.guess = result
    self.constraints.append(constraint(1e-4/self.uk_scale[0],1e6/self.uk_scale[0],False)  )
    self.constraints.append(constraint(np.exp(-self.TR/(50))/self.uk_scale[1],np.exp(-self.TR/(5500))/self.uk_scale[1],True))

  def rescale(self,x):
    M0 = x[0,...]*self.uk_scale[0]
    T1 = -self.TR/np.log(x[1,...]*self.uk_scale[1])
    return np.array((M0,T1))

  def execute_forward_2D(self,x,islice):
    print('uk_scale[1]: ',self.uk_scale[1])
    E1 = x[1,...]*self.uk_scale[1]
    S = x[0,:,:]*self.uk_scale[0]*(-E1 + 1)*self.sin_phi[:,islice,:,:]/(-E1*self.cos_phi[:,islice,:,:] + 1)
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)
    return S
  def execute_gradient_2D(self,x,islice):
    E1 = x[1,:,:]*self.uk_scale[1]
    M0 = x[0,...]
    E1[~np.isfinite(E1)] = 0
    grad_M0 = self.uk_scale[0]*(-E1 + 1)*self.sin_phi[:,islice,:,:]/(-E1*self.cos_phi[:,islice,:,:] + 1)
    grad_T1 = M0*self.uk_scale[0]*self.uk_scale[1]*(-E1 + 1)*self.sin_phi[:,islice,:,:]*self.cos_phi[:,islice,:,:]/(-E1*self.cos_phi[:,islice,:,:] + 1)**2 -\
    M0*self.uk_scale[0]*self.uk_scale[1]*self.sin_phi[:,islice,:,:]/(-E1*self.cos_phi[:,islice,:,:] + 1)
    grad = np.array([grad_M0,grad_T1],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
#    print('Grad Scaling', np.linalg.norm(np.abs(grad_M0))/np.linalg.norm(np.abs(grad_T1)))
    return grad

  def execute_forward_3D(self,x):
#    print('uk_scale[1]: ',self.uk_scale[1])
    E1 = x[1,...]*self.uk_scale[1]
    S = x[0,:,:]*self.uk_scale[0]*(-E1 + 1)*self.sin_phi/(-E1*self.cos_phi + 1)
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)
    return S
  def execute_gradient_3D(self,x):
    E1 = x[1,:,:]*self.uk_scale[1]
    M0 = x[0,...]
    E1[~np.isfinite(E1)] = 0
    grad_M0 = self.uk_scale[0]*(-E1 + 1)*self.sin_phi/(-E1*self.cos_phi + 1)
    grad_T1 = M0*self.uk_scale[0]*self.uk_scale[1]*(-E1 + 1)*self.sin_phi*self.cos_phi/(-E1*self.cos_phi + 1)**2 -\
    M0*self.uk_scale[0]*self.uk_scale[1]*self.sin_phi/(-E1*self.cos_phi + 1)
    grad = np.array([grad_M0,grad_T1],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
#    print('Grad Scaling', np.linalg.norm(np.abs(grad_M0))/np.linalg.norm(np.abs(grad_T1)))
    return grad


  def plot_unknowns(self,x,dim_2D=False):
      M0 = np.abs(x[0,...]*self.uk_scale[0])
      T1 = np.abs(-self.TR/np.log(x[1,...]*self.uk_scale[1]))
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
         [z,y,x] = M0.shape
         self.ax = []
         if not self.figure:
           plt.ion()
           self.figure = plt.figure(figsize = (12,6))
           self.figure.subplots_adjust(hspace=0, wspace=0)
           self.gs = gridspec.GridSpec(2,6, width_ratios=[x/(20*z),x/z,1,x/z,1,x/(20*z)],height_ratios=[x/z,1])
           self.figure.tight_layout()
           self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
           for grid in self.gs:
             self.ax.append(plt.subplot(grid))
             self.ax[-1].axis('off')

           self.M0_plot=self.ax[1].imshow((M0[int(self.NSlice/2),...]))
           self.M0_plot_cor=self.ax[7].imshow((M0[:,int(M0.shape[1]/2),...]))
           self.M0_plot_sag=self.ax[2].imshow(np.flip((M0[:,:,int(M0.shape[-1]/2)]).T,1))
           self.ax[1].set_title('Proton Density in a.u.',color='white')
           self.ax[1].set_anchor('SE')
           self.ax[2].set_anchor('SW')
           self.ax[7].set_anchor('NW')
           cax = plt.subplot(self.gs[:,0])
           cbar = self.figure.colorbar(self.M0_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           cax.yaxis.set_ticks_position('left')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')
           plt.draw()
           plt.pause(1e-10)

           self.T1_plot=self.ax[3].imshow((T1[int(self.NSlice/2),...]))
           self.T1_plot_cor=self.ax[9].imshow((T1[:,int(T1.shape[1]/2),...]))
           self.T1_plot_sag=self.ax[4].imshow(np.flip((T1[:,:,int(T1.shape[-1]/2)]).T,1))
           self.ax[3].set_title('T1 in  ms',color='white')
           self.ax[3].set_anchor('SE')
           self.ax[4].set_anchor('SW')
           self.ax[9].set_anchor('NW')
           cax = plt.subplot(self.gs[:,5])
           cbar = self.figure.colorbar(self.T1_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')
           plt.draw()
           plt.pause(1e-10)
         else:
           self.M0_plot.set_data((M0[int(self.NSlice/2),...]))
           self.M0_plot_cor.set_data((M0[:,int(M0.shape[1]/2),...]))
           self.M0_plot_sag.set_data(np.flip((M0[:,:,int(M0.shape[-1]/2)]).T,1))
           self.M0_plot.set_clim([M0_min,M0_max])
           self.M0_plot_cor.set_clim([M0_min,M0_max])
           self.M0_plot_sag.set_clim([M0_min,M0_max])
           self.T1_plot.set_data((T1[int(self.NSlice/2),...]))
           self.T1_plot_cor.set_data((T1[:,int(T1.shape[1]/2),...]))
           self.T1_plot_sag.set_data(np.flip((T1[:,:,int(T1.shape[-1]/2)]).T,1))
           self.T1_plot.set_clim([T1_min,T1_max])
           self.T1_plot_sag.set_clim([T1_min,T1_max])
           self.T1_plot_cor.set_clim([T1_min,T1_max])
           plt.draw()
           plt.pause(1e-10)