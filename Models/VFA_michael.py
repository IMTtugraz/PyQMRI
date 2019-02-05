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

unknowns_TGV = 4
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
  def __init__(self, par,images):

    self.t = np.arange(500)*10/60

    ### Currently fixed variables
    self.A_B = np.ones_like(images[0])
    self.mu_B  = np.ones_like(images[0])*18.0375
    self.A_G = np.ones_like(images[0])*0.0041
    self.mu_G = np.ones_like(images[0])*0.1696
    self.tau = np.ones_like(images[0])*0.5
    self.Tc = np.ones_like(images[0])*1/6

    self.r = 0.4
    self.R10 = 0.05

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
    for j in range(unknowns_TGV-1):
      self.uk_scale.append(1)


    test_M0 =np.ones((NSlice,dimY,dimX),dtype=DTYPE)
    FP = np.ones((NSlice,dimY,dimX),dtype=DTYPE)
    Te = np.ones((NSlice,dimY,dimX),dtype=DTYPE)
    alpha = np.ones((NSlice,dimY,dimX),dtype=DTYPE)


    G_x = self.execute_forward_3D(np.array([test_M0,FP,Te,alpha],dtype=DTYPE))
    self.uk_scale[0]*=1/np.median(np.abs(G_x))

    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.uk_scale[0],FP,Te,alpha],dtype=DTYPE))
    self.uk_scale[1] = self.uk_scale[1]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[1,...]))
    self.uk_scale[2] = self.uk_scale[2]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[2,...]))
    self.uk_scale[3] = self.uk_scale[3]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[3,...]))

    DG_x =  self.execute_gradient_3D(np.array([test_M0/self.uk_scale[0],FP/self.uk_scale[1],
                                               Te/self.uk_scale[2],alpha/self.uk_scale[3]],dtype=DTYPE))
    print('T1 scale: ',self.uk_scale[1],
                              '/ M0_scale: ',self.uk_scale[0])


    result = np.array([1/self.uk_scale[0]*np.ones((NSlice,dimY,dimX),dtype=DTYPE),
                       1/self.uk_scale[1]*FP,1/self.uk_scale[2]*Te,1/self.uk_scale[3]*alpha],dtype=DTYPE)

    self.guess = result
    self.constraints.append(constraint(0,1e6/self.uk_scale[0],False)  )
    self.constraints.append(constraint(0,10/self.uk_scale[1],True))
    self.constraints.append(constraint(0,10/self.uk_scale[1],True))
    self.constraints.append(constraint(0,10/self.uk_scale[1],True))
  def rescale(self,x):
    M0 = x[0,...]*self.uk_scale[0]
    FP = x[1,...]*self.uk_scale[1]
    Te = x[2,...]*self.uk_scale[2]
    alpha = x[3,...]*self.uk_scale[3]
    return np.array((M0,FP,Te,alpha))

  def execute_forward_3D(self,x):
    C = self.dce_fit(x)
    R =self.r*C+self.R10

    E1 = np.exp(-self.TR*R)
    S = x[0,...]*self.uk_scale[0]*(-E1 + 1)*self.sin_phi/(-E1*self.cos_phi + 1)

    S = np.array(S,dtype=DTYPE)
    return S

  def execute_gradient_3D(self,x):
    (C,dC) = self.dce_fit_grad(x)
    R =self.r*C+self.R10
    M0 = x[0,...]
    M0_sc = self.uk_sclae[0]
    E1 = np.exp(-self.TR*R)


    grad_M0 = self.uk_scale[0]*(-E1 + 1)*self.sin_phi/(-E1*self.cos_phi + 1)

    grad_T1 = M0*M0_sc*self.r*self.TR*E1*self.sin_phi/(1 -E1*self.cos_phi) -\
              M0*M0_sc*self.r*self.TR*(1 -E1)*E1*self.sin_phi*self.cos_phi/(1 -E1*self.cos_phi)**2
    grad_T1 *= dC
    grad = np.concatenate((grad_M0,grad_T1))
#    grad[~np.isfinite(grad)] = 1e-20
    return grad



  def plot_unknowns(self,x,dim_2D=False):
      M0 = np.abs(x[0,...]*self.uk_scale[0])
      T1 = np.abs(x[1,...]*self.uk_scale[1])
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


  def dce_fit(self,x):
    A_B = self.A_B
    mu_B  = self.mu_B
    A_G = self.A_G
    mu_G = self.mu_G
    tau = self.tau
    Tc = self.Tc

    Fp = x[1]*self.uk_scale[1]
    Te = x[2]*self.uk_scale[2]
    alpha = x[3]*self.uk_scale[3]

    C_fit = np.zeros((self.t.size,x.shape[1],x.shape[2],x.shape[3]))

    for j in range(len(self.t)):
      ind_low = self.t[j]>tau
      ind_high = self.t[j]<=tau+Tc
      ind = ind_low & ind_high
      if np.any(ind==True):

        t2 = self.t[j]-tau[ind]
        t3 = 1.0/mu_B[ind]
        t4 = 1.0/mu_G[ind]
        t7 = mu_B[ind]*t2
        t5 = np.exp(-t7)
        t6 = 1.0/mu_B[ind]**2
        t8 = t3*t5
        t30 = mu_G[ind]*t2
        t9 = np.exp(-t30)
        t31 = t4*t9
        t10 = t8-t31
        t11 = A_G[ind]*t10
        t13 = t2*t3
        t14 = t6+t13

        C_fit[j,ind]=t11+A_B[ind]*t6-A_G[ind]*(t3-t4)-A_B[ind]*t5*t14

      ind = self.t[j]>(tau+Tc)
      if np.any(ind==True):
        t2 = self.t[j]-tau[ind]
        t3 = 1.0/mu_B[ind]
        t4 = 1.0/mu_G[ind]
        t7 = mu_B[ind]*t2
        t5 = np.exp(-t7)
        t6 = 1.0/mu_B[ind]**2
        t8 = t3*t5
        t30 = mu_G[ind]*t2
        t9 = np.exp(-t30)
        t31 = t4*t9
        t10 = t8-t31
        t11 = A_G[ind]*t10
        t12 = Tc[ind]-self.t[j]+tau[ind]
        t13 = t2*t3
        t14 = t6+t13
        t15 = np.exp(-alpha[ind])
        t16 = t15-1.0
        t17 = 1.0/Te[ind]
        t18 = 1.0/alpha[ind]
        t19 = t16*t17*t18
        t20 = mu_B[ind]+t19
        t21 = 1.0/t20
        t22 = mu_G[ind]+t19
        t23 = 1.0/t22
        t24 = t12*t20
        t25 = np.exp(t24)
        t26 = 1.0/t20**2
        t27 = mu_B[ind]*t12
        t28 = np.exp(t27)
        t42 = t12*t16*t17*t18
        t33 = np.exp(-t42)
        t44 = t12*t21
        t34 = t26-t44
        t46 = t3*t28
        t47 = mu_G[ind]*t12
        t48 = np.exp(t47)
        t49 = t21*t25
        t50 = t12*t22
        t51 = np.exp(t50)
        t57 = A_B[ind]*t26
        t58 = t21-t23
        C_fit[j,ind]=t11-A_G[ind]*(t46-t4*t48)+A_B[ind]*t28*(t6-t3*t12)-t16*t33*\
                        (t57-A_G[ind]*t58+A_G[ind]*(t49-t23*t51)-A_B[ind]*t25*t34)-A_B[ind]*t5*t14

    C_fit=C_fit*Fp
    return C_fit


  def dce_fit_grad(self,x):
    A_B = self.A_B
    mu_B  = self.mu_B
    A_G = self.A_G
    mu_G = self.mu_G
    tau = self.tau
    Tc = self.Tc

    Fp = x[1]*self.uk_scale[1]
    Te = x[2]*self.uk_scale[2]
    alpha = x[3]*self.uk_scale[3]

    dC_fit=np.zeros((x.shape[0],self.t.size,x.shape[1],x.shape[2],x.shape[3]))
    C_fit=np.zeros((self.t.size,x.shape[1],x.shape[2],x.shape[3]))
    for j in range(len(self.t)):
      ind_low = self.t[j]>tau
      ind_high = self.t[j]<=tau+Tc
      ind = ind_low & ind_high
      if np.any(ind==True):
        t2 = self.t[j]-tau[ind]
        t3 = 1.0/mu_B[ind]
        t4 = 1.0/mu_G[ind]
        t7 = mu_B[ind]*t2
        t5 = np.exp(-t7)
        t6 = 1.0/mu_B[ind]**2
        t8 = t3*t5
        t30 = mu_G[ind]*t2
        t9 = np.exp(-t30)
        t31 = t4*t9
        t10 = t8-t31
        t11 = A_G[ind]*t10
        t13 = t2*t3
        t14 = t6+t13
        t29 = 1.0/mu_B[ind]**3
        t32 = 1.0/mu_G[ind]**2
        t35 = t5*t6
        t36 = t2*t3*t5
        t38 = t29*2.0
        t39 = t2*t6
        t40 = t38+t39
        t52 = t9*t32
        t53 = t2*t4*t9
        t54 = t52+t53
        t71 = t5-t9

        t37 = t35+t36
        t41 = A_B[ind]*t5*t40
        t45 = A_B[ind]*t2*t5*t14
        t55 = A_G[ind]*t54
        t72 = A_G[ind]*t71
        t74 = A_B[ind]*t3*t5


        C_fit[j,ind]=t11+A_B[ind]*t6-A_G[ind]*(t3-t4)-A_B[ind]*t5*t14
        dC_fit[0,j,ind]=t6-t5*t14
        dC_fit[1,j,ind]=t41+t45-A_B[ind]*t29*2.0+A_G[ind]*t6-A_G[ind]*t37
        dC_fit[2,j,ind]=-t3+t4+t8-t31
        dC_fit[3,j,ind]=t55-A_G[ind]*t32
        dC_fit[8,j,ind]=t72+t74-A_B[ind]*mu_B[ind]*t5*t14

      ind = self.t[j]>(tau+Tc)
      if np.any(ind==True):
        t2 = self.t[j]-tau[ind]
        t3 = 1.0/mu_B[ind]
        t4 = 1.0/mu_G[ind]
        t7 = mu_B[ind]*t2
        t5 = np.exp(-t7)
        t6 = 1.0/mu_B[ind]**2
        t8 = t3*t5
        t30 = mu_G[ind]*t2
        t9 = np.exp(-t30)
        t31 = t4*t9
        t10 = t8-t31
        t11 = A_G[ind]*t10
        t12 = Tc[ind]-self.t[j]+tau[ind]
        t13 = t2*t3
        t14 = t6+t13
        t15 = np.exp(-alpha[ind])
        t16 = t15-1.0
        t17 = 1.0/Te[ind]
        t18 = 1.0/alpha[ind]
        t19 = t16*t17*t18
        t20 = mu_B[ind]+t19
        t21 = 1.0/t20
        t22 = mu_G[ind]+t19
        t23 = 1.0/t22
        t24 = t12*t20
        t25 = np.exp(t24)
        t26 = 1.0/t20**2
        t27 = mu_B[ind]*t12
        t28 = np.exp(t27)
        t29 = 1.0/mu_B[ind]**3
        t32 = 1.0/mu_G[ind]**2
        t42 = t12*t16*t17*t18
        t33 = np.exp(-t42)
        t44 = t12*t21
        t34 = t26-t44
        t35 = t5*t6
        t36 = t2*t3*t5
        t37 = t35+t36
        t38 = t29*2.0
        t39 = t2*t6
        t40 = t38+t39
        t41 = A_B[ind]*t5*t40
        t43 = 1.0/t20**3
        t45 = A_B[ind]*t2*t5*t14
        t46 = t3*t28
        t47 = mu_G[ind]*t12
        t48 = np.exp(t47)
        t49 = t21*t25
        t50 = t12*t22
        t51 = np.exp(t50)
        t52 = t9*t32
        t53 = t2*t4*t9
        t54 = t52+t53
        t55 = A_G[ind]*t54
        t56 = 1.0/t22**2
        t57 = A_B[ind]*t26
        t58 = t21-t23
        t59 = t23*t51
        t60 = 1.0/Te[ind]**2
        t61 = t16**2
        t62 = t49-t59
        t63 = A_G[ind]*t62
        t65 = A_G[ind]*t58
        t66 = A_B[ind]*t25*t34
        t64 = t57+t63-t65-t66
        t67 = 1.0/alpha[ind]**2
        t68 = t16*t17*t67
        t69 = t15*t17*t18
        t70 = t68+t69
        t71 = t5-t9
        t72 = A_G[ind]*t71
        t73 = t28-t48
        t74 = A_B[ind]*t3*t5
        t75 = t25-t51
        t76 = A_G[ind]*t75
        t77 = A_B[ind]*t21*t25
        t78 = t76+t77-A_B[ind]*t20*t25*t34
        t79 = A_B[ind]*mu_B[ind]*t28*(t6-t3*t12)
        t80 = t17*t18*t33*t61*t64

        C_fit[j,ind]=t11-A_G[ind]*(t46-t4*t48)+A_B[ind]*t28*(t6-t3*t12)-t16*t33*(t57-A_G[ind]*t58+A_G[ind]*(t49-t23*t51)-\
                     A_B[ind]*t25*t34)-A_B[ind]*t5*t14


        dC_fit[:,j,ind]=np.array((-t5*t14+t28*(t6-t3*t12)-t16*t33*(t26-t25*t34),
        t41+t45+A_G[ind]*(t6*t28-t3*t12*t28)-A_G[ind]*t37-A_B[ind]*t28*(t38-t6*t12)+t16*t33*(A_G[ind]*(t25*t26-t12*t21*t25)\
                   +A_B[ind]*t43*2.0-A_G[ind]*t26-A_B[ind]*t25*(t43*2.0-t12*t26)+A_B[ind]*t12*t25*t34)+A_B[ind]*t12*t28*(t6-t3*t12),
            t8-t31-t46+t4*t48+t16*t33*(t21-t23-t49+t59),
            t55-A_G[ind]*(t32*t48-t4*t12*t48)-t16*t33*(A_G[ind]*(t51*t56-t12*t23*t51)-A_G[ind]*t56),
            A_B[ind],##replaced later but saving memory allocations this way
            t79+t80-A_G[ind]*t73-A_B[ind]*t3*t28-t16*t33*t78,
            -t16*t33*(-A_G[ind]*(t16*t18*t26*t60-t16*t18*t56*t60)+A_G[ind]*\
                      (t16*t18*t25*t26*t60-t16*t18*t51*t56*t60-t12*t16*t18*t21*t25*t60+t12*t16*t18*t23*t51*t60)-\
                      A_B[ind]*t25*(t16*t18*t43*t60*2.0-t12*t16*t18*t26*t60)+A_B[ind]*t16*t18*t43*t60*2.0+\
                      A_B[ind]*t12*t16*t18*t25*t34*t60)-t12*t18*t33*t60*t61*t64,
            -t16*t33*(-A_G[ind]*(t26*t70-t56*t70)+A_G[ind]*(t25*t26*t70-t51*t56*t70-t12*t21*t25*t70+t12*t23*t51*t70)-\
                      A_B[ind]*t25*(t43*t70*2.0-t12*t26*t70)+A_B[ind]*t43*t70*2.0+A_B[ind]*t12*t25*t34*t70)+\
                      t15*t33*t64-t16*t33*t64*(t12*t15*t17*t18+t12*t16*t17*t67),
            t72+t74+t79+t80-A_G[ind]*t73-A_B[ind]*t3*t28-t16*t33*t78-A_B[ind]*mu_B[ind]*t5*t14))


    dC_fit=dC_fit*Fp
    dC_fit[4,:]=C_fit
    return (C_fit*Fp,dC_fit)











