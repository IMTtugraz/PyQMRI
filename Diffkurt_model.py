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

unknowns_TGV = 3
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

    self.images = images
    self.NSlice = par['NSlice']
    self.figure = None

    (NScan,Nislice,dimX,dimY) = images.shape
    self.b = np.ones((NScan,1,1,1))
    try:
      self.NScan = par["T2PREP"].size
      for i in range(self.NScan):
        self.b[i,...] = par["T2PREP"][i]*np.ones((1,1,1))
    except:
      self.NScan = par["b_value"].size
      for i in range(self.NScan):
        self.b[i,...] = par["b_value"][i]*np.ones((1,1,1))

    self.uk_scale=[]
    self.uk_scale.append(1/np.max(np.abs(images)))
    self.uk_scale.append(1)
    self.uk_scale.append(1)
#
    ADC = np.reshape(np.linspace(1e-4,1e-2,dimX*dimY*Nislice),(Nislice,dimX,dimY))
    kurt = np.reshape(np.linspace(0,1,dimX*dimY*Nislice),(Nislice,dimX,dimY))
    test_M0 = 1#1*np.sqrt((dimX*np.pi/2)/par['Nproj'])
#    print(test_M0)
    ADC = 1/self.uk_scale[1]*ADC*np.ones((Nislice,dimY,dimX),dtype=DTYPE)
    kurt = 1/self.uk_scale[1]*kurt*np.ones((Nislice,dimY,dimX),dtype=DTYPE)
#
#
    G_x = self.execute_forward_3D(np.array([test_M0/self.uk_scale[0]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),ADC,kurt],dtype=DTYPE))
#    print(np.max(np.abs(G_x))/np.max(np.abs(images)))
    self.uk_scale[0] *= 1/np.max(np.abs(G_x))
#
#    test_M0*=self.uk_scale[0]
#    self.uk_scale[0] = 2

    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.uk_scale[0]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),ADC,kurt],dtype=DTYPE))
    self.uk_scale[1] = self.uk_scale[1]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...]))
    self.uk_scale[2] = self.uk_scale[2]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[2,...]))

    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.uk_scale[0]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),ADC/self.uk_scale[1],kurt/self.uk_scale[2]],dtype=DTYPE))
    print('Grad Scaling init', np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...])))
    print('ADC scale: ',self.uk_scale[1])
    print('Kurt scale: ',self.uk_scale[2])
    print('M0 scale: ',self.uk_scale[0])



    result = np.array([0.1/self.uk_scale[0]*np.ones((Nislice,dimY,dimX),dtype=DTYPE),(1e-3/self.uk_scale[1]*np.ones((Nislice,dimY,dimX),dtype=DTYPE)),(0/self.uk_scale[2]*np.ones((Nislice,dimY,dimX),dtype=DTYPE))],dtype=DTYPE)
    self.guess = result

    self.constraints.append(constraint(1e-5/self.uk_scale[0],100/self.uk_scale[0],False)  )
    self.constraints.append(constraint((1e-4/self.uk_scale[1]),(1e-2/self.uk_scale[1]),False))
    self.constraints.append(constraint((0/self.uk_scale[2]),(10/self.uk_scale[2]),True))
  def rescale(self,x):
    M0 = x[0,...]*self.uk_scale[0]
    ADC = (x[1,...]*self.uk_scale[1])
    kurt = (x[2,...]*self.uk_scale[2])
    return np.array((M0,ADC,kurt))

#  def execute_forward_2D(self,x,islice):
#    ADC = x[1,...]*self.uk_scale[1]
#    S = x[0,...]*self.uk_scale[0]*np.exp(-self.TE*(ADC))
#    S[~np.isfinite(S)] = 1e-200
#    S = np.array(S,dtype=DTYPE)
#    return S
#  def execute_gradient_2D(self,x,islice):
#    M0 = x[0,...]
#    ADC = x[1,...]
#    grad_M0 = self.uk_scale[0]*np.exp(-self.TE*(ADC*self.uk_scale[1]))
#    grad_ADC = -M0*self.uk_scale[0]*self.TE*self.uk_scale[1]*np.exp(-self.TE*(ADC*self.uk_scale[1]))
#    grad = np.array([grad_M0,grad_ADC],dtype=DTYPE)
#    grad[~np.isfinite(grad)] = 1e-20
##    print('Grad Scaling', np.linalg.norm(np.abs(grad_M0))/np.linalg.norm(np.abs(grad_T2)))
#    return grad

  def execute_forward_3D(self,x):
    ADC = x[1,...]*self.uk_scale[1]
    kurt= x[2,...]*self.uk_scale[2]

    S = x[0,...]*self.uk_scale[0]*np.exp(0.2*ADC**2*self.b**2*kurt - ADC*self.b)
    S[~np.isfinite(S)] = 1e-20
    S = np.array(S,dtype=DTYPE)
    return S

  def execute_gradient_3D(self,x):
    M0 = x[0,...]
    ADC = x[1,...]
    kurt = x[2,...]
    M0_sc = self.uk_scale[0]
    ADC_sc = self.uk_scale[1]
    kurt_sc = self.uk_scale[2]
    grad_M0 = M0_sc*np.exp(0.2*ADC**2*ADC_sc**2*self.b**2*kurt*kurt_sc - ADC*ADC_sc*self.b)
    grad_ADC = M0*M0_sc*(0.4*ADC*ADC_sc**2*self.b**2*kurt*kurt_sc - ADC_sc*self.b)*np.exp(0.2*ADC**2*ADC_sc**2*self.b**2*kurt*kurt_sc - ADC*ADC_sc*self.b)
    grad_kurt= 0.2*ADC**2*ADC_sc**2*M0*M0_sc*self.b**2*kurt_sc*np.exp(0.2*ADC**2*ADC_sc**2*self.b**2*kurt*kurt_sc - ADC*ADC_sc*self.b)

    grad = np.array([grad_M0,grad_ADC,grad_kurt],dtype=DTYPE)
    grad[~np.isfinite(grad)] = 1e-20
    print('Grad Scaling', np.linalg.norm(np.abs(grad_M0))/np.linalg.norm(np.abs(grad_ADC)),'Grad Scaling', np.linalg.norm(np.abs(grad_M0))/np.linalg.norm(np.abs(grad_kurt)))
    return grad


  def plot_unknowns(self,x,dim_2D=False):
      M0 = np.abs(x[0,...])*self.uk_scale[0]
      ADC = (np.abs(x[1,...])*self.uk_scale[1])
      kurt =  (np.abs(x[2,...])*self.uk_scale[2])
      M0_min = M0.min()
      M0_max = M0.max()
      ADC_min = ADC.min()
      ADC_max = ADC.max()
      kurt_min = kurt.min()
      kurt_max = kurt.max()
      if dim_2D:
         if not self.figure:
           plt.ion()
           self.figure, self.ax = plt.subplots(1,2,figsize=(12,5))
           self.M0_plot = self.ax[0].imshow((M0))
           self.ax[0].set_title('Proton Density in a.u.')
           self.ax[0].axis('off')
           self.figure.colorbar(self.M0_plot,ax=self.ax[0])
           self.ADC_plot = self.ax[1].imshow((ADC))
           self.ax[1].set_title('ADC in  ms')
           self.ax[1].axis('off')
           self.figure.colorbar(self.ADC_plot,ax=self.ax[1])
           self.figure.tight_layout()
           plt.draw()
           plt.pause(1e-10)
         else:
           self.M0_plot.set_data((M0))
           self.M0_plot.set_clim([M0_min,M0_max])
           self.ADC_plot.set_data((ADC))
           self.ADC_plot.set_clim([ADC_min,ADC_max])
           plt.draw()
           plt.pause(1e-10)
      else:
         [z,y,x] = M0.shape
         self.ax = []
         if not self.figure:
           plt.ion()
           self.figure = plt.figure(figsize = (12,6))
           self.figure.subplots_adjust(hspace=0, wspace=0)
           self.gs = gridspec.GridSpec(2,10, width_ratios=[x/(20*z),x/z,1,x/z,1,x/(20*z),x/(2*z),x/z,1,x/(20*z)],height_ratios=[x/z,1])
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

           self.ADC_plot=self.ax[3].imshow((ADC[int(self.NSlice/2),...]))
           self.ADC_plot_cor=self.ax[13].imshow((ADC[:,int(ADC.shape[1]/2),...]))
           self.ADC_plot_sag=self.ax[4].imshow(np.flip((ADC[:,:,int(ADC.shape[-1]/2)]).T,1))
           self.ax[3].set_title('ADC',color='white')
           self.ax[3].set_anchor('SE')
           self.ax[4].set_anchor('SW')
           self.ax[13].set_anchor('NW')
           cax = plt.subplot(self.gs[:,5])
           cbar = self.figure.colorbar(self.ADC_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')
           plt.draw()

           self.kurt_plot=self.ax[7].imshow((kurt[int(self.NSlice/2),...]))
           self.kurt_plot_cor=self.ax[17].imshow((kurt[:,int(ADC.shape[1]/2),...]))
           self.kurt_plot_sag=self.ax[8].imshow(np.flip((kurt[:,:,int(ADC.shape[-1]/2)]).T,1))
           self.ax[7].set_title('Kurtosis',color='white')
           self.ax[7].set_anchor('SE')
           self.ax[8].set_anchor('SW')
           self.ax[17].set_anchor('NW')
           cax = plt.subplot(self.gs[:,9])
           cbar = self.figure.colorbar(self.kurt_plot, cax=cax)
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
           self.ADC_plot.set_data((ADC[int(self.NSlice/2),...]))
           self.ADC_plot_cor.set_data((ADC[:,int(ADC.shape[1]/2),...]))
           self.ADC_plot_sag.set_data(np.flip((ADC[:,:,int(ADC.shape[-1]/2)]).T,1))
           self.ADC_plot.set_clim([ADC_min,ADC_max])
           self.ADC_plot_sag.set_clim([ADC_min,ADC_max])
           self.ADC_plot_cor.set_clim([ADC_min,ADC_max])

           self.kurt_plot.set_data((kurt[int(self.NSlice/2),...]))
           self.kurt_plot_cor.set_data((kurt[:,int(kurt.shape[1]/2),...]))
           self.kurt_plot_sag.set_data(np.flip((kurt[:,:,int(kurt.shape[-1]/2)]).T,1))
           self.kurt_plot.set_clim([kurt_min,kurt_max])
           self.kurt_plot_sag.set_clim([kurt_min,kurt_max])
           self.kurt_plot_cor.set_clim([kurt_min,kurt_max])
           plt.draw()
           plt.pause(1e-10)