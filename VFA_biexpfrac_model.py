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

    self.images = images
    self.NSlice = par['NSlice']
    self.figure = None

    (NScan,Nislice,dimX,dimY) = images.shape
    self.TE = np.ones((NScan,1,1,1))
    try:
      self.NScan = par["T2PREP"].size
      for i in range(self.NScan):
        self.TE[i,...] = par["T2PREP"][i]*np.ones((1,1,1))
    except:
      self.NScan = par["TE"].size
      for i in range(self.NScan):
        self.TE[i,...] = par["TE"][i]*np.ones((1,1,1))
    self.uk_scale=[]
    self.uk_scale.append(1)
    self.uk_scale.append(1)
    self.uk_scale.append(1)
    self.uk_scale.append(1)

    test_M0 = 1*np.sqrt((dimX*np.pi/2)/par['Nproj'])
    T21 = 1/np.reshape(np.linspace(1e-4,150,dimX*dimY*Nislice),(Nislice,dimX,dimY))
    test_M01 = 1
    T21 = 1/self.uk_scale[2]*T21*np.ones((Nislice,dimY,dimX),dtype=DTYPE)
    T22 = 1/np.reshape(np.linspace(150,1500,dimX*dimY*Nislice),(Nislice,dimX,dimY))
    T22 = 1/self.uk_scale[3]*T22*np.ones((Nislice,dimY,dimX),dtype=DTYPE)
#
    G_x = self.execute_forward_3D(np.array([test_M0/self.uk_scale[0]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),test_M01/self.uk_scale[1]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),T21,T22],dtype=DTYPE))
    self.uk_scale[0] = self.uk_scale[0]*np.max(np.abs(images))/np.median(np.abs(G_x))

    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.uk_scale[0]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),test_M01/self.uk_scale[1]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),T21,T22],dtype=DTYPE))
    self.uk_scale[1] = self.uk_scale[1]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...]))
    self.uk_scale[2] = self.uk_scale[2]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[2,...]))
    self.uk_scale[3] = self.uk_scale[3]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[3,...]))

    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.uk_scale[0]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),test_M01/self.uk_scale[1]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),T21/self.uk_scale[2],T22/self.uk_scale[3]],dtype=DTYPE))
#    print('Grad Scaling init', np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...])))
    print('M0 scale: ',self.uk_scale[0])
    print('T21 scale: ',self.uk_scale[2])
    print('M01 scale: ',self.uk_scale[1])
    print('T22 scale: ',self.uk_scale[3])


    result = np.array([0.1/self.uk_scale[0]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),0.5/self.uk_scale[1]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),((1/10)/self.uk_scale[2]*np.ones((Nislice,dimY,dimX),dtype=DTYPE)),((1/150)/self.uk_scale[3]*np.ones((Nislice,dimY,dimX),dtype=DTYPE))],dtype=DTYPE)
    self.guess = result

    self.constraints.append(constraint(-100/self.uk_scale[0],100/self.uk_scale[0],False)  )
    self.constraints.append(constraint(0/self.uk_scale[1],1/self.uk_scale[1],True)  )
    self.constraints.append(constraint(((1/150)/self.uk_scale[2]),((1/1e-4)/self.uk_scale[2]),True))
    self.constraints.append(constraint(((1/1500)/self.uk_scale[3]),((1/150)/self.uk_scale[3]),True))
  def rescale(self,x):
    M0 = x[0,...]*self.uk_scale[0]
    M01 = x[1,...]*self.uk_scale[1]
    T21 = 1/(x[2,...]*self.uk_scale[2])
    T22 = 1/(x[3,...]*self.uk_scale[3])
    return np.array((M0,M01,T21,T22))

  def execute_forward_2D(self,x,islice):
    M0  = x[0,...]*self.uk_scale[0]
    M01 = x[1,...]*self.uk_scale[1]
    T21 = x[2,...]*self.uk_scale[2]
    T22 = x[3,...]*self.uk_scale[3]
    S = M0*(M01*np.exp(-self.TE*(T21))+(1-M01)*np.exp(-self.TE*(T22)))
    S[~np.isfinite(S)] = 1e-200
    S = np.array(S,dtype=DTYPE)
    return S
  def execute_gradient_2D(self,x,islice):
    M0  = x[0,...]
    M01 = x[1,...]
    T21 = x[2,...]
    T22 = x[3,...]
    grad_M0 =  self.uk_scale[0]*(M01*self.uk_scale[1]*np.exp(-self.TE*(T21*self.uk_scale[2])) + (1-M01*self.uk_scale[1])*np.exp(-self.TE*(T22*self.uk_scale[3])))
    grad_M01 = self.uk_scale[0]*M0*(self.uk_scale[1]*np.exp(-self.TE*(T21*self.uk_scale[2]))-self.uk_scale[1]*np.exp(-self.TE*(T22*self.uk_scale[3])))
    grad_T21 = -self.uk_scale[0]*M0*M01*self.uk_scale[1]*self.TE*self.uk_scale[2]*np.exp(-self.TE*(T21*self.uk_scale[2]))
    grad_T22 = -self.uk_scale[0]*M0*(1-M01*self.uk_scale[1])*self.TE*self.uk_scale[3]*np.exp(-self.TE*(T22*self.uk_scale[3]))
    grad = np.array([grad_M0,grad_M01,grad_T21,grad_T22],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
#    print('Grad Scaling', np.linalg.norm(np.abs(grad_M0))/np.linalg.norm(np.abs(grad_T2)))
    return grad

  def execute_forward_3D(self,x):
    M0  = x[0,...]*self.uk_scale[0]
    M01 = x[1,...]*self.uk_scale[1]
    T21 = x[2,...]*self.uk_scale[2]
    T22 = x[3,...]*self.uk_scale[3]
    S = M0*(M01*np.exp(-self.TE*(T21))+(1-M01)*np.exp(-self.TE*(T22)))
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)
    return S

  def execute_gradient_3D(self,x):
    M0  = x[0,...]
    M01 = x[1,...]
    T21 = x[2,...]
    T22 = x[3,...]
    grad_M0 =  self.uk_scale[0]*(M01*self.uk_scale[1]*np.exp(-self.TE*(T21*self.uk_scale[2])) + (1-M01*self.uk_scale[1])*np.exp(-self.TE*(T22*self.uk_scale[3])))
    grad_M01 = self.uk_scale[0]*M0*(self.uk_scale[1]*np.exp(-self.TE*(T21*self.uk_scale[2]))-self.uk_scale[1]*np.exp(-self.TE*(T22*self.uk_scale[3])))
    grad_T21 = -self.uk_scale[0]*M0*M01*self.uk_scale[1]*self.TE*self.uk_scale[2]*np.exp(-self.TE*(T21*self.uk_scale[2]))
    grad_T22 = -self.uk_scale[0]*M0*(1-M01*self.uk_scale[1])*self.TE*self.uk_scale[3]*np.exp(-self.TE*(T22*self.uk_scale[3]))
    grad = np.array([grad_M0,grad_M01,grad_T21,grad_T22],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
#    print('Grad Scaling', np.linalg.norm(np.abs(grad_M0))/np.linalg.norm(np.abs(grad_ADC)))
    return grad


  def plot_unknowns(self,x,dim_2D=False):
      M0 = np.abs(x[0,...])*self.uk_scale[0]
      M0_min = M0.min()
      M0_max = M0.max()

      M01 = np.abs(x[1,...])*self.uk_scale[1]
      T21 = 1/(np.abs(x[2,...])*self.uk_scale[2])
      M01_min = M01.min()
      M01_max = M01.max()
      T21_min = T21.min()
      T21_max = T21.max()

      T22 = 1/(np.abs(x[3,...])*self.uk_scale[3])
      T22_min = T22.min()
      T22_max = T22.max()

      if dim_2D:
         if not self.figure:
           plt.ion()
           self.figure, self.ax = plt.subplots(1,2,figsize=(12,5))
           self.M01_plot = self.ax[0].imshow((M01))
           self.ax[0].set_title('Proton Density in a.u.')
           self.ax[0].axis('off')
           self.figure.colorbar(self.M01_plot,ax=self.ax[0])
           self.ADC_plot = self.ax[1].imshow((T21))
           self.ax[1].set_title('T21 in  ms')
           self.ax[1].axis('off')
           self.figure.colorbar(self.T21_plot,ax=self.ax[1])
           self.figure.tight_layout()
           plt.draw()
           plt.pause(1e-10)
         else:
           self.M01_plot.set_data((M01))
           self.M01_plot.set_clim([M01_min,M01_max])
           self.T21_plot.set_data((T21))
           self.T21_plot.set_clim([T21_min,T21_max])
           plt.draw()
           plt.pause(1e-10)
      else:
         [z,y,x] = M0.shape
         self.ax = []
         if not self.figure:
           plt.ion()
           self.figure = plt.figure(figsize = (12,6))
           self.figure.subplots_adjust(hspace=0, wspace=0)
           self.gs = gridspec.GridSpec(2,13, width_ratios=[x/(20*z),x/z,1,x/z,1,x/(20*z),x/z*0.25,x/(20*z),x/z,1,x/z,1,x/(20*z)],height_ratios=[x/z,1])
           self.figure.tight_layout()
           self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
           for grid in self.gs:
             self.ax.append(plt.subplot(grid))
             self.ax[-1].axis('off')

           self.M0_plot=self.ax[1].imshow((M0[int(self.NSlice/2),...]))
           self.M0_plot_cor=self.ax[14].imshow((M0[:,int(M0.shape[1]/2),...]))
           self.M0_plot_sag=self.ax[2].imshow(np.flip((M0[:,:,int(M0.shape[-1]/2)]).T,1))
           self.ax[1].set_title('Proton Density in a.u.',color='white')
           self.ax[1].set_anchor('SE')
           self.ax[2].set_anchor('SW')
           self.ax[14].set_anchor('NW')
           cax = plt.subplot(self.gs[:,0])
           cbar = self.figure.colorbar(self.M0_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           cax.yaxis.set_ticks_position('left')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')

           self.M01_plot=self.ax[8].imshow((M01[int(self.NSlice/2),...]))
           self.M01_plot_cor=self.ax[21].imshow((M01[:,int(M01.shape[1]/2),...]))
           self.M01_plot_sag=self.ax[9].imshow(np.flip((M01[:,:,int(M01.shape[-1]/2)]).T,1))
           self.ax[8].set_title('Proton Density in a.u.',color='white')
           self.ax[8].set_anchor('SE')
           self.ax[9].set_anchor('SW')
           self.ax[21].set_anchor('NW')
           cax = plt.subplot(self.gs[:,7])
           cbar = self.figure.colorbar(self.M01_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           cax.yaxis.set_ticks_position('left')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')

           self.T21_plot=self.ax[3].imshow((T21[int(self.NSlice/2),...]))
           self.T21_plot_cor=self.ax[16].imshow((T21[:,int(T21.shape[1]/2),...]))
           self.T21_plot_sag=self.ax[4].imshow(np.flip((T21[:,:,int(T21.shape[-1]/2)]).T,1))
           self.ax[3].set_title('T21 in  ms',color='white')
           self.ax[3].set_anchor('SE')
           self.ax[4].set_anchor('SW')
           self.ax[16].set_anchor('NW')
           cax = plt.subplot(self.gs[:,5])
           cbar = self.figure.colorbar(self.T21_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')


           self.T22_plot=self.ax[10].imshow((T22[int(self.NSlice/2),...]))
           self.T22_plot_cor=self.ax[23].imshow((T22[:,int(T22.shape[1]/2),...]))
           self.T22_plot_sag=self.ax[11].imshow(np.flip((T22[:,:,int(T22.shape[-1]/2)]).T,1))
           self.ax[10].set_title('T22 in  ms',color='white')
           self.ax[10].set_anchor('SE')
           self.ax[11].set_anchor('SW')
           self.ax[23].set_anchor('NW')
           cax = plt.subplot(self.gs[:,12])
           cbar = self.figure.colorbar(self.T22_plot, cax=cax)
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
           self.T21_plot.set_data((T21[int(self.NSlice/2),...]))
           self.T21_plot_cor.set_data((T21[:,int(T21.shape[1]/2),...]))
           self.T21_plot_sag.set_data(np.flip((T21[:,:,int(T21.shape[-1]/2)]).T,1))
           self.T21_plot.set_clim([T21_min,T21_max])
           self.T21_plot_sag.set_clim([T21_min,T21_max])
           self.T21_plot_cor.set_clim([T21_min,T21_max])


           self.M01_plot.set_data((M01[int(self.NSlice/2),...]))
           self.M01_plot_cor.set_data((M01[:,int(M01.shape[1]/2),...]))
           self.M01_plot_sag.set_data(np.flip((M01[:,:,int(M01.shape[-1]/2)]).T,1))
           self.M01_plot.set_clim([M01_min,M01_max])
           self.M01_plot_cor.set_clim([M01_min,M01_max])
           self.M01_plot_sag.set_clim([M01_min,M01_max])
           self.T22_plot.set_data((T22[int(self.NSlice/2),...]))
           self.T22_plot_cor.set_data((T22[:,int(T22.shape[1]/2),...]))
           self.T22_plot_sag.set_data(np.flip((T22[:,:,int(T22.shape[-1]/2)]).T,1))
           self.T22_plot.set_clim([T22_min,T22_max])
           self.T22_plot_sag.set_clim([T22_min,T22_max])
           self.T22_plot_cor.set_clim([T22_min,T22_max])
           plt.draw()
           plt.pause(1e-10)