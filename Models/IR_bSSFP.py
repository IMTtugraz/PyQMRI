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


class bSSFP_Model:
  def __init__(self,fa,fa_corr,TR,images,NSlice,Nproj):
    self.constraints = []
    self.TR = TR
    self.images = images
    self.fa = fa
    self.fa_corr = fa_corr
    self.NSlice = NSlice
    self.figure = None
    self.Nproj = Nproj

    (NScan,NSlice,dimX,dimY) = images.shape
    self.NScan = NScan
    self.dimX = dimX
    self.dimY = dimY

    phi_corr = np.zeros_like(images,dtype=DTYPE)

    phi_corr = fa*fa_corr

    self.sin_phi = np.sin(phi_corr)
    self.cos_phi = np.cos(phi_corr)

    self.sin_phi_2 = np.sin(phi_corr/2)
    self.cos_phi_2 = np.cos(phi_corr/2)

    self.M0_sc = 1
    self.T1_sc = 1
    self.T2_sc = 1


    test_T1 = np.reshape(np.linspace(10,5500,dimX*dimY*NSlice),(NSlice,dimX,dimY))
    test_T2 = np.reshape(np.linspace(0,2000,dimX*dimY*NSlice),(NSlice,dimX,dimY))
    test_M0 = 0.1*np.sqrt((dimX*np.pi/2)/Nproj)
    test_T1 = 1/self.T1_sc*test_T1*np.ones((NSlice,dimY,dimX),dtype=DTYPE)
    test_T2 = 1/self.T2_sc*test_T2*np.ones((NSlice,dimY,dimX),dtype=DTYPE)


    G_x = self.execute_forward_3D(np.array([test_M0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),test_T1,test_T2],dtype=DTYPE))
    self.M0_sc = self.M0_sc*np.median(np.abs(images))/np.median(np.abs(G_x))

    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),test_T1,test_T2],dtype=DTYPE))
    self.T1_sc = self.T1_sc*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...]))
    self.T2_sc = self.T2_sc*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[2,...]))

    self.T1_sc /= np.sqrt(self.M0_sc)
    self.T2_sc /= np.sqrt(self.M0_sc)#self.T2_sc/np.sqrt(self.T2_sc)
    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),test_T1/self.T1_sc,test_T2/self.T2_sc],dtype=DTYPE))
    print('Grad Scaling init M0/T1: %f,  M0/T2: %f'%(np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...])),np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[2,...]))))
    print('T1 scale: ',self.T1_sc,'/ T2_scale: ',self.T2_sc,
                              '/ M0_scale: ',self.M0_sc)


    result = np.array([1/self.M0_sc*np.ones((NSlice,dimY,dimX),dtype=DTYPE),1/self.T1_sc*800*np.ones((NSlice,dimY,dimX),dtype=DTYPE),1/self.T2_sc*80*np.ones((NSlice,dimY,dimX),dtype=DTYPE)],dtype=DTYPE)
    self.guess = result
    self.constraints.append(constraint(-10/self.M0_sc,10/self.M0_sc,False)  )
    self.constraints.append(constraint(200/self.T1_sc,5500/self.T1_sc,True))
    self.constraints.append(constraint(0/self.T2_sc,1000/self.T2_sc,True))


  def execute_forward_3D(self,x):
    S = np.zeros((self.NScan,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
    M0 = x[0,...]
    T1 = x[1,...]
    T2 = x[2,...]
    M0_sc = self.M0_sc
    T1_sc = self.T1_sc
    T2_sc = self.T2_sc
    E1 = np.exp(-self.TR/(T1*T1_sc))
    E2 = np.exp(-self.TR/(T2*T2_sc))
    TR = self.TR

    tmp1 = M0*M0_sc*(1 - E1)
    tmp2 = ((T1*T1_sc/(T2*T2_sc) - (T1*T1_sc/(T2*T2_sc) - 1)*self.cos_phi + 1)*self.sin_phi_2/self.sin_phi + 1)
    tmp3 = (self.sin_phi_2**2/(T2*T2_sc) + self.cos_phi_2**2/(T1*T1_sc))
    tmp4 = self.sin_phi/(-(-E2 + E1)*self.cos_phi + 1 - E1*E2)
    for k in range(self.NScan):
      for l in range(self.Nproj):
        n = k*self.Nproj+l+1
        S[k,l,...] = tmp1*(-tmp2*np.exp(-TR*n*tmp3) + 1)*tmp4
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)

    return (np.require(np.mean(S,axis=1),requirements='C'))
  def execute_gradient_3D(self,x):
    grad = np.zeros((3,self.NScan,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
    M0 = x[0,...]
    T1 = x[1,:,:]#self.E1#
    T2 = x[2,...]#self.E2#
    M0_sc = self.M0_sc
    T1_sc = self.T1_sc
    T2_sc = self.T2_sc
    E1 = np.exp(-self.TR/(T1*T1_sc))
    E2 = np.exp(-self.TR/(T2*T2_sc))
    TR = self.TR


    tmp1 = M0_sc*(1 - E1)
    tmp2 = ((T1*T1_sc/(T2*T2_sc) - (T1*T1_sc/(T2*T2_sc) - 1)*self.cos_phi + 1)*self.sin_phi_2/self.sin_phi + 1)
    tmp3 = (self.sin_phi_2**2/(T2*T2_sc) + self.cos_phi_2**2/(T1*T1_sc))
    tmp4 = self.sin_phi/(-(-E2 + E1)*self.cos_phi + 1 - E1*E2)
    tmp5 = (TR*E1*self.cos_phi/(T1**2*T1_sc) + TR*E1*E2/(T1**2*T1_sc))*tmp4**2
    tmp6 = (-T1_sc*self.cos_phi/(T2*T2_sc) + T1_sc/(T2*T2_sc))
    tmp7 = E1*self.sin_phi/(T1**2*T1_sc*(-(-E2 + E1)*self.cos_phi + 1 - E1*E2))
    tmp8 = (-TR*E2*self.cos_phi/(T2**2*T2_sc) + TR*E1*E2/(T2**2*T2_sc))*tmp4**2
    tmp9 = (T1*T1_sc*self.cos_phi/(T2**2*T2_sc) - T1*T1_sc/(T2**2*T2_sc))


    for k in range(self.NScan):
      for l in range(self.Nproj):
        n = k*self.Nproj+l+1
        Etmp = np.exp(-TR*n*tmp3)
        tmp = (-tmp2*Etmp + 1)
        grad[0,k,l,...] =tmp1*tmp*tmp4

        grad[1,k,l,...] = M0*tmp1*tmp*tmp5 + M0*tmp1*(-tmp6*Etmp*self.sin_phi_2/self.sin_phi - TR*n*tmp2*Etmp*self.cos_phi_2**2/(T1**2*T1_sc))*tmp4 - M0*M0_sc*TR*tmp*tmp7

        grad[2,k,l,...] = M0*tmp1*tmp*tmp8 + M0*tmp1*(-tmp9*Etmp*self.sin_phi_2/self.sin_phi - TR*n*tmp2*Etmp*self.sin_phi_2**2/(T2**2*T2_sc))*tmp4

    grad[~np.isfinite(grad)] = 1e-20

    grad = (np.mean(grad,axis=2))


    print('Grad Scaling E1', np.linalg.norm(np.abs(grad[0]))/np.linalg.norm(np.abs(grad[1])))
    print('Grad Scaling E2', np.linalg.norm(np.abs(grad[0]))/np.linalg.norm(np.abs(grad[2])))
    return np.require(grad,requirements='C')


  def plot_unknowns(self,x,dim_2D=False):
      M0 = np.abs(x[0,...]*self.M0_sc)
      T1 = np.abs(x[1,...]*self.T1_sc)
      T2 = np.abs(x[2,...]*self.T2_sc)
      M0_min = M0.min()
      M0_max = M0.max()
      T1_min = T1.min()
      T1_max = T1.max()
      T2_min = T2.min()
      T2_max = T2.max()
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
           self.gs = gridspec.GridSpec(2,10, width_ratios=[x/(20*z),x/z,1,x/z,1,x/(20*z),4*x/(20*z),x/z,1,x/(20*z)],height_ratios=[x/z,1])
           self.figure.tight_layout()
           self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
           for grid in self.gs:
             self.ax.append(plt.subplot(grid))
             self.ax[-1].axis('off')

           self.M0_plot=self.ax[1].imshow((M0[int(self.NSlice/2),...]))
           self.M0_plot_cor=self.ax[11].imshow((M0[:,int(M0.shape[1]/2),...]))
           self.M0_plot_sag=self.ax[2].imshow(np.flip((M0[:,:,int(M0.shape[-1]/2)]).T,1))
           self.ax[1].set_title('Proton Density in a.u.',color='white')
           self.ax[1].set_anchor('SE')
           self.ax[2].set_anchor('SW')
           self.ax[11].set_anchor('NW')
           cax = plt.subplot(self.gs[:,0])
           cbar = self.figure.colorbar(self.M0_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           cax.yaxis.set_ticks_position('left')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')
           self.T1_plot=self.ax[3].imshow((T1[int(self.NSlice/2),...]))
           self.T1_plot_cor=self.ax[13].imshow((T1[:,int(T1.shape[1]/2),...]))
           self.T1_plot_sag=self.ax[4].imshow(np.flip((T1[:,:,int(T1.shape[-1]/2)]).T,1))
           self.ax[3].set_title('T1 in  ms',color='white')
           self.ax[3].set_anchor('SE')
           self.ax[4].set_anchor('SW')
           self.ax[13].set_anchor('NW')
           cax = plt.subplot(self.gs[:,5])
           cbar = self.figure.colorbar(self.T1_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')

           self.T2_plot=self.ax[7].imshow((T2[int(self.NSlice/2),...]))
           self.T2_plot_cor=self.ax[17].imshow((T2[:,int(T2.shape[1]/2),...]))
           self.T2_plot_sag=self.ax[8].imshow(np.flip((T2[:,:,int(T2.shape[-1]/2)]).T,1))
           self.ax[7].set_title('T2 in  ms',color='white')
           self.ax[7].set_anchor('SE')
           self.ax[8].set_anchor('SW')
           self.ax[17].set_anchor('NW')
           cax = plt.subplot(self.gs[:,9])
           cbar = self.figure.colorbar(self.T2_plot, cax=cax)
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
           plt.draw()
           plt.pause(1e-10)