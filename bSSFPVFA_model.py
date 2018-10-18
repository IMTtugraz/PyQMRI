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


class constraint:
  def __init__(self, min_val=-np.inf, max_val=np.inf, real_const=False):
    self.min = min_val
    self.max = max_val
    self.real = real_const
  def update(self,scale):
    self.min = self.min/scale
    self.max = self.max/scale


class Model:
  def __init__(self,par,images):
    self.constraints = []
    if len(par['TR'])>1:
      self.par['TR'] = par['TR'][0]
    else:
      self.TR = par['TR']
    self.images = images
    self.fa_fl = par['flip_angle(s)'][:int(par['NScan']/2)]*np.pi/180
    self.fa_bl = par['flip_angle(s)'][int(par['NScan']/2):]*np.pi/180
    self.fa_corr = par['fa_corr']
    self.NSlice = par['NSlice']
    self.figure = None

    (NScan,NSlice,dimY,dimX) = images.shape

    phi_corr = np.zeros_like(images,dtype=DTYPE)
    for i in range(np.size(self.fa_fl)):
      phi_corr[i,:,:,:] = self.fa_fl[i]*self.fa_corr

    for i in range(np.size(self.fa_fl),np.size(self.fa_bl)+np.size(self.fa_fl)):
      phi_corr[i,:,:,:] = self.fa_bl[i-np.size(self.fa_fl)]*self.fa_corr

    self.sin_phi_fl = np.sin(phi_corr[:int(par['NScan']/2)])
    self.cos_phi_fl = np.cos(phi_corr[:int(par['NScan']/2)])

    self.sin_phi_bl = np.sin(phi_corr[int(par['NScan']/2):])
    self.cos_phi_bl = np.cos(phi_corr[int(par['NScan']/2):])


    self.uk_scale = []
    for i in range(4):
      self.uk_scale.append(1)

#    self.uk_scale[0] = 1
#    self.uk_scale[1] = 1
#    self.uk_scale[2] = 1
#    self.uk_scale[3] = 1
#    self.E1 = np.exp(-TR/T1_ref)

    test_T1 = np.reshape(np.linspace(10,5500,dimX*dimY*NSlice),(NSlice,dimX,dimY))
    test_T2 = np.reshape(np.linspace(10,2500,dimX*dimY*NSlice),(NSlice,dimX,dimY))
    test_M0 = 0.1*np.sqrt((dimX*np.pi/2)/par['Nproj'])
    test_T1 = 1/self.uk_scale[1]*np.exp(-self.TR/(test_T1*np.ones((NSlice,dimY,dimX),dtype=DTYPE)))
    test_T2 = 1/self.uk_scale[2]*np.exp(-self.TR/(test_T2*np.ones((NSlice,dimY,dimX),dtype=DTYPE)))
    test_k = 1/self.uk_scale[3]*np.ones_like(test_T1)


    G_x = self.execute_forward_3D(np.array([test_M0/self.uk_scale[0]*np.ones((NSlice,dimY,dimX),dtype=DTYPE),test_T1,test_T2,test_k],dtype=DTYPE))
    self.uk_scale[0] = self.uk_scale[0]*np.median(np.abs(images))/np.median(np.abs(G_x))


    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.uk_scale[0]*np.ones((NSlice,dimY,dimX),dtype=DTYPE),test_T1/self.uk_scale[1],test_T2/self.uk_scale[2],test_k/self.uk_scale[3]],dtype=DTYPE))
    self.uk_scale[1] = self.uk_scale[1]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...]))
    self.uk_scale[2] = self.uk_scale[2]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[2,...]))
    self.uk_scale[3] = self.uk_scale[3]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[3,...]))

    self.uk_scale[1] /= np.sqrt(self.uk_scale[0])
    self.uk_scale[2] /= np.sqrt(self.uk_scale[0])#self.uk_scale[2]/np.sqrt(self.uk_scale[2])
    self.uk_scale[3] /= np.sqrt(self.uk_scale[0])
    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.uk_scale[0]*np.ones((NSlice,dimY,dimX),dtype=DTYPE),test_T1/self.uk_scale[1],test_T2/self.uk_scale[2],test_k/self.uk_scale[3]],dtype=DTYPE))
    print('Grad Scaling init M0/T1: %f,  M0/T2: %f,  M0/k: %f'%(np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...])),np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[2,...])),np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[3,...]))))
    print('T1 scale: ',self.uk_scale[1],'/ T2_scale: ',self.uk_scale[2],
                              '/ M0_scale: ',self.uk_scale[0],'/ k_scale: ',self.uk_scale[3])


    result = np.array([1/self.uk_scale[0]*np.ones((NSlice,dimY,dimX),dtype=DTYPE),1/self.uk_scale[1]*np.exp(-self.TR/(800*np.ones((NSlice,dimY,dimX),dtype=DTYPE))),1/self.uk_scale[2]*np.exp(-self.TR/(80*np.ones((NSlice,dimY,dimX),dtype=DTYPE)))
    ,1/self.uk_scale[3]*np.ones((NSlice,dimY,dimX),dtype=DTYPE)]).astype(DTYPE)
    self.guess = result
    self.constraints.append(constraint(-1000/self.uk_scale[0],1000/self.uk_scale[0],False)  )
    self.constraints.append(constraint(np.exp(-self.TR/(10))/self.uk_scale[1],np.exp(-self.TR/(5500))/self.uk_scale[1],True))
    self.constraints.append(constraint(np.exp(-self.TR/(1))/self.uk_scale[2],np.exp(-self.TR/(2500))/self.uk_scale[2],True))
    self.constraints.append(constraint(0/self.uk_scale[3],10/self.uk_scale[3],True)  )

  def rescale(self,x):
    M0 = x[0,...]*self.uk_scale[0]
    T1 = -self.TR/np.log(x[1,...]*self.uk_scale[1])
    T2 = -self.TR/np.log(x[2,...]*self.uk_scale[2])
    k = x[3,...]*self.uk_scale[3]
    return np.array((M0,T1,T2,k))

  def execute_forward_3D(self,x):
    M0 = x[0,...]
    E1 = x[1,...]
    E2 = x[2,...]
    k = x[3,...]
    M0_sc = self.uk_scale[0]
    T1_sc = self.uk_scale[1]
    T2_sc = self.uk_scale[2]
    S1 = M0*k*self.uk_scale[3]*self.uk_scale[0]*(-E1*T1_sc + 1)*self.sin_phi_fl/(-E1*T1_sc*self.cos_phi_fl + 1)
    S2 = 1/(k*self.uk_scale[3])*M0*M0_sc*(-E1*T1_sc + 1)*self.sin_phi_bl/(-E1*E2*T1_sc*T2_sc - (E1*T1_sc - E2*T2_sc)*self.cos_phi_bl + 1)
    S =(np.concatenate((S1,S2)))
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)
    return S
  def execute_gradient_3D(self,x):
    M0 = x[0,...]
    E1 = x[1,:,:]#self.E1#
    E2 = x[2,...]#self.E2#
    k = x[3,...]
    M0_sc = self.uk_scale[0]
    T1_sc = self.uk_scale[1]
    T2_sc = self.uk_scale[2]
    grad_M0_fl = (k*self.uk_scale[3]*M0_sc*(-E1*T1_sc + 1)*self.sin_phi_fl/(-E1*T1_sc*self.cos_phi_fl + 1))
    grad_T1_fl = M0*k*self.uk_scale[3]*self.uk_scale[0]*self.uk_scale[1]*(-E1*T1_sc + 1)*self.sin_phi_fl*self.cos_phi_fl/(-E1*T1_sc*self.cos_phi_fl + 1)**2 -\
    M0*k*self.uk_scale[3]*self.uk_scale[0]*self.uk_scale[1]*self.sin_phi_fl/(-E1*T1_sc*self.cos_phi_fl + 1)
    grad_T2_fl = np.zeros_like(grad_T1_fl)
    grad_k_fl = (M0*self.uk_scale[3]*M0_sc*(-E1*T1_sc + 1)*self.sin_phi_fl/(-E1*T1_sc*self.cos_phi_fl + 1))

    grad_M0_bl = 1/(k*self.uk_scale[3])*(M0_sc*(-E1*T1_sc + 1)*self.sin_phi_bl/(-E1*E2*T1_sc*T2_sc - (E1*T1_sc - E2*T2_sc)*self.cos_phi_bl + 1))
    grad_T1_bl = 1/(k*self.uk_scale[3])*(-M0*M0_sc*T1_sc*self.sin_phi_bl/(-E1*E2*T1_sc*T2_sc - (E1*T1_sc - E2*T2_sc)*self.cos_phi_bl + 1) + M0*M0_sc*(-E1*T1_sc + 1)*(E2*T1_sc*T2_sc + T1_sc*self.cos_phi_bl)*self.sin_phi_bl/(-E1*E2*T1_sc*T2_sc - (E1*T1_sc - E2*T2_sc)*self.cos_phi_bl + 1)**2)
    grad_T2_bl = 1/(k*self.uk_scale[3])*M0*M0_sc*(-E1*T1_sc + 1)*(E1*T1_sc*T2_sc - T2_sc*self.cos_phi_bl)*self.sin_phi_bl/(-E1*E2*T1_sc*T2_sc - (E1*T1_sc - E2*T2_sc)*self.cos_phi_bl + 1)**2
    grad_k_bl = -(M0_sc*(-E1*T1_sc + 1)*self.sin_phi_bl/(-E1*E2*T1_sc*T2_sc - (E1*T1_sc - E2*T2_sc)*self.cos_phi_bl + 1))/(k**2*self.uk_scale[3])

    grad_M0 = np.concatenate((grad_M0_fl,grad_M0_bl))
    grad_T1 = np.concatenate((grad_T1_fl,grad_T1_bl))
    grad_T2 = np.concatenate((grad_T2_fl,grad_T2_bl))
    grad_k =  np.concatenate((grad_k_fl,grad_k_bl))
    grad = np.array([grad_M0,grad_T1,grad_T2,grad_k],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
    print('Grad Scaling E1', np.linalg.norm(np.abs(grad[0]))/np.linalg.norm(np.abs(grad[1])))
    print('Grad Scaling E2', np.linalg.norm(np.abs(grad[0]))/np.linalg.norm(np.abs(grad[2])))
    print('Grad Scaling k', np.linalg.norm(np.abs(grad[0]))/np.linalg.norm(np.abs(grad[3])))
    return grad


  def plot_unknowns(self,x,dim_2D=False):
      M0 = np.abs(x[0,...]*self.uk_scale[0])
      T1 = np.abs(-self.TR/np.log(x[1,...]*self.uk_scale[1]))
      T2 = np.abs(-self.TR/np.log(x[2,...]*self.uk_scale[2]))
      k = np.abs(x[3,...])*self.uk_scale[3]
      M0_min = M0.min()
      M0_max = M0.max()
      T1_min = T1.min()
      T1_max = T1.max()
      T2_min = T2.min()
      T2_max = T2.max()
      k_min = k.min()
      k_max = k.max()
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
           self.gs = gridspec.GridSpec(2,14, width_ratios=[x/(20*z),x/z,1,x/z,1,x/(20*z),4*x/(20*z),x/z,1,x/(20*z),4*x/(20*z),x/z,1,x/(20*z)],height_ratios=[x/z,1])
           self.figure.tight_layout()
           self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
           for grid in self.gs:
             self.ax.append(plt.subplot(grid))
             self.ax[-1].axis('off')

           self.M0_plot=self.ax[1].imshow((M0[int(self.NSlice/2),...]))
           self.M0_plot_cor=self.ax[15].imshow((M0[:,int(M0.shape[1]/2),...]))
           self.M0_plot_sag=self.ax[2].imshow(np.flip((M0[:,:,int(M0.shape[-1]/2)]).T,1))
           self.ax[1].set_title('Proton Density in a.u.',color='white')
           self.ax[1].set_anchor('SE')
           self.ax[2].set_anchor('SW')
           self.ax[15].set_anchor('NW')
           cax = plt.subplot(self.gs[:,0])
           cbar = self.figure.colorbar(self.M0_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           cax.yaxis.set_ticks_position('left')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')
           self.T1_plot=self.ax[3].imshow((T1[int(self.NSlice/2),...]))
           self.T1_plot_cor=self.ax[17].imshow((T1[:,int(T1.shape[1]/2),...]))
           self.T1_plot_sag=self.ax[4].imshow(np.flip((T1[:,:,int(T1.shape[-1]/2)]).T,1))
           self.ax[3].set_title('T1 in  ms',color='white')
           self.ax[3].set_anchor('SE')
           self.ax[4].set_anchor('SW')
           self.ax[17].set_anchor('NW')
           cax = plt.subplot(self.gs[:,5])
           cbar = self.figure.colorbar(self.T1_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')

           self.T2_plot=self.ax[7].imshow((T2[int(self.NSlice/2),...]))
           self.T2_plot_cor=self.ax[21].imshow((T2[:,int(T2.shape[1]/2),...]))
           self.T2_plot_sag=self.ax[8].imshow(np.flip((T2[:,:,int(T2.shape[-1]/2)]).T,1))
           self.ax[7].set_title('T2 in  ms',color='white')
           self.ax[7].set_anchor('SE')
           self.ax[8].set_anchor('SW')
           self.ax[21].set_anchor('NW')
           cax = plt.subplot(self.gs[:,9])
           cbar = self.figure.colorbar(self.T2_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')

           self.k=self.ax[11].imshow((k[int(self.NSlice/2),...]))
           self.k_cor=self.ax[25].imshow((k[:,int(T2.shape[1]/2),...]))
           self.k_sag=self.ax[12].imshow(np.flip((k[:,:,int(T2.shape[-1]/2)]).T,1))
           self.ax[11].set_title('k in  a.u.',color='white')
           self.ax[11].set_anchor('SE')
           self.ax[12].set_anchor('SW')
           self.ax[25].set_anchor('NW')
           cax = plt.subplot(self.gs[:,13])
           cbar = self.figure.colorbar(self.k, cax=cax)
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
           self.T2_plot.set_data((T2[int(self.NSlice/2),...]))
           self.T2_plot_cor.set_data((T2[:,int(T2.shape[1]/2),...]))
           self.T2_plot_sag.set_data(np.flip((T2[:,:,int(T2.shape[-1]/2)]).T,1))
           self.T2_plot.set_clim([T2_min,T2_max])
           self.T2_plot_sag.set_clim([T2_min,T2_max])
           self.T2_plot_cor.set_clim([T2_min,T2_max])

           self.k.set_data((k[int(self.NSlice/2),...]))
           self.k_cor.set_data((k[:,int(k.shape[1]/2),...]))
           self.k_sag.set_data(np.flip((k[:,:,int(k.shape[-1]/2)]).T,1))
           self.k.set_clim([k_min,k_max])
           self.k_sag.set_clim([k_min,k_max])
           self.k_cor.set_clim([k_min,k_max])

           plt.draw()
           plt.pause(1e-10)