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
from Models.Model import BaseModel, constraints, DTYPE, DTYPE_real

unknowns_TGV = 4
unknowns_H1 = 0


class Model(BaseModel):
  def __init__(self, par,images):
    super().__init__(par)
    self.t = par['t']
    parameters = par["file"]["GT/parameters"]

    self.A_B = (parameters[0][...].astype(DTYPE_real))
    self.mu_B  = (parameters[1][...].astype(DTYPE_real))
    self.A_G = (parameters[2][...].astype(DTYPE_real))
    self.mu_G = (parameters[3][...].astype(DTYPE_real))
    self.tau = (parameters[8][...].astype(DTYPE_real))
    self.Tc = (parameters[5][...].astype(DTYPE_real))

    self.FP = (parameters[4][...].astype(DTYPE_real))
    self.Te = (parameters[6][...].astype(DTYPE_real))
    self.Te[self.Te<1/6] = 1/6
    self.alpha = (parameters[7][...].astype(DTYPE_real))

    par["C"] = par["file"]["GT/sensitivities/real_dat"][()] + 1j*par["file"]["GT/sensitivities/imag_dat"][()]
    sumSqrC = np.sqrt(np.sum((par["C"] * np.conj(par["C"])),0))
    par["C"] = par["C"] / np.tile(sumSqrC, (par["NC"],1,1,1))

    self.r = 3.2
    self.R10 = 1
    self.TR = par["TR"]/1000

    self.sin_phi = np.sin(par["flip_angle(s)"]*np.pi/180)
    self.cos_phi = np.cos(par["flip_angle(s)"]*np.pi/180)

    self.uk_scale.append(1/np.median(np.abs(images)))
    for j in range(unknowns_TGV):
      self.uk_scale.append(1)


    self.guess = self._set_init_scales(par["dscale"])

    self.constraints.append(constraints(0,1e6/self.uk_scale[0],False)  )
    self.constraints.append(constraints(0/self.uk_scale[1],100/self.uk_scale[1],True))
    self.constraints.append(constraints(1/6/self.uk_scale[2],20/self.uk_scale[2],True))
    self.constraints.append(constraints(1e-4/self.uk_scale[3],3/self.uk_scale[3],True))

  def _execute_forward_2D(self,x,islice):
    print("2D Functions not implemented")
    raise NotImplementedError
  def _execute_gradient_2D(self,x,islice):
    print("2D Functions not implemented")
    raise NotImplementedError

  def _execute_forward_3D(self,x):
    C = self._dce_fit(x)
    R =self.r*C+self.R10

    E1 = np.exp(-self.TR*R)
    S = x[0]*self.uk_scale[0]*(-E1 + 1)*self.sin_phi/(-E1*self.cos_phi + 1)

    S = np.array(S,dtype=DTYPE)
    S[~np.isfinite(S)] = 1e-20
    return S

  def _execute_gradient_3D(self,x):
    (C,dC) = self._dce_fit_grad(x)
    R =self.r*C+self.R10
    M0 = x[0,...]
    M0_sc = self.uk_scale[0]
    E1 = np.exp(-self.TR*R)


    grad_M0 = self.uk_scale[0]*(-E1 + 1)*self.sin_phi/(-E1*self.cos_phi + 1)

    grad_T1 = self.r*M0*M0_sc*(self.TR*E1*self.sin_phi/(1 -E1*self.cos_phi) -\
              self.TR*(1 -E1)*E1*self.sin_phi*self.cos_phi/(1 -E1*self.cos_phi)**2)
    grad_T1 = grad_T1*dC
    grad = np.concatenate((grad_M0[None,...],grad_T1))
#    grad = grad_T1
    grad[~np.isfinite(grad)] = 1e-20
    print('Grad Scaling FP', np.linalg.norm(np.abs(grad[0]))/np.linalg.norm(np.abs(grad[1])))
    print('Grad Scaling Te', np.linalg.norm(np.abs(grad[0]))/np.linalg.norm(np.abs(grad[2])))
    print('Grad Scaling alpha', np.linalg.norm(np.abs(grad[0]))/np.linalg.norm(np.abs(grad[3])))
    return grad.astype(DTYPE)



  def plot_unknowns(self,x,dim_2D=False):
      M0 = np.abs(x[0,...]*self.uk_scale[0])
      Fp = np.concatenate((np.abs(x[1,...]*self.uk_scale[1]),self.FP),axis=-1)
      M0_min = M0.min()
      M0_max = M0.max()
      Fp_min = 0#Fp.min()
      Fp_max = 1#Fp.max()
      Te = np.concatenate((np.abs(x[2,...]*self.uk_scale[2]),self.Te),axis=-1)
      alpha = np.concatenate((np.abs(x[3,...]*self.uk_scale[3]),self.alpha),axis=-1)
      Te_min = 0#Te.min()
      Te_max = 5#Te.max()
      alpha_min = alpha.min()
      alpha_max = alpha.max()

      if dim_2D:
         if not self.figure:
           plt.ion()
#           self.figure, self.ax = plt.subplots(1,2,figsize=(12,5))
#           self.M0_plot = self.ax[0].imshow((M0))
#           self.ax[0].set_title('Proton Density in a.u.')
#           self.ax[0].axis('off')
#           self.figure.colorbar(self.M0_plot,ax=self.ax[0])
#           self.T1_plot = self.ax[1].imshow((T1))
#           self.ax[1].set_title('T1 in  ms')
#           self.ax[1].axis('off')
#           self.figure.colorbar(self.T1_plot,ax=self.ax[1])
#           self.figure.tight_layout()
           plt.draw()
           plt.pause(1e-10)
         else:
#           self.M0_plot.set_data((M0))
#           self.M0_plot.set_clim([M0_min,M0_max])
#           self.T1_plot.set_data((T1))
#           self.T1_plot.set_clim([T1_min,T1_max])
           plt.draw()
           plt.pause(1e-10)
      else:
         [z,y,x] = M0.shape
         self.ax = []
         if not self.figure:
           plt.ion()
           self.figure = plt.figure(figsize = (12,6))
           self.figure.subplots_adjust(hspace=0, wspace=0)
           self.gs = gridspec.GridSpec(5,6, width_ratios=[x/(20*z),x/z,1,x/z,1,x/(20*z)],height_ratios=[x/z,1,x/z/2,x/z,1])
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
           self.ax[7].set_anchor('NE')
           cax = plt.subplot(self.gs[:2,0])
           cbar = self.figure.colorbar(self.M0_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           cax.yaxis.set_ticks_position('left')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')
           plt.draw()
           plt.pause(1e-10)

           self.Fp_plot=self.ax[3].imshow((Fp[int(self.NSlice/2),...]))
           self.Fp_plot_cor=self.ax[9].imshow((Fp[:,int(Fp.shape[1]/2),...]))
           self.Fp_plot_sag=self.ax[4].imshow(np.flip((Fp[:,:,int(Fp.shape[-1]/2)]).T,1))
           self.ax[3].set_title('Fp in a.u.',color='white')
           self.ax[3].set_anchor('SE')
           self.ax[4].set_anchor('SW')
           self.ax[9].set_anchor('NE')
           cax = plt.subplot(self.gs[:2,5])
           cbar = self.figure.colorbar(self.Fp_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')
           plt.draw()
           plt.pause(1e-10)

           self.Te_plot=self.ax[19].imshow((Te[int(self.NSlice/2),...]))
           self.Te_plot_cor=self.ax[25].imshow((Te[:,int(Te.shape[1]/2),...]))
           self.Te_plot_sag=self.ax[20].imshow(np.flip((Te[:,:,int(Te.shape[-1]/2)]).T,1))
           self.ax[19].set_title('Te',color='white')
           self.ax[19].set_anchor('SE')
           self.ax[20].set_anchor('SW')
           self.ax[25].set_anchor('NE')
           cax = plt.subplot(self.gs[3:,0])
           cbar = self.figure.colorbar(self.Te_plot, cax=cax)
           cbar.ax.tick_params(labelsize=12,colors='white')
           cax.yaxis.set_ticks_position('left')
           for spine in cbar.ax.spines:
            cbar.ax.spines[spine].set_color('white')
           plt.draw()
           plt.pause(1e-10)

           self.alpha_plot=self.ax[21].imshow((alpha[int(self.NSlice/2),...]))
           self.alpha_plot_cor=self.ax[27].imshow((alpha[:,int(alpha.shape[1]/2),...]))
           self.alpha_plot_sag=self.ax[22].imshow(np.flip((alpha[:,:,int(alpha.shape[-1]/2)]).T,1))
           self.ax[21].set_title('alpha',color='white')
           self.ax[21].set_anchor('SE')
           self.ax[22].set_anchor('SW')
           self.ax[27].set_anchor('NE')
           cax = plt.subplot(self.gs[3:,5])
           cbar = self.figure.colorbar(self.alpha_plot, cax=cax)
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

           self.Fp_plot.set_data((Fp[int(self.NSlice/2),...]))
           self.Fp_plot_cor.set_data((Fp[:,int(Fp.shape[1]/2),...]))
           self.Fp_plot_sag.set_data(np.flip((Fp[:,:,int(Fp.shape[-1]/2)]).T,1))
           self.Fp_plot.set_clim([Fp_min,Fp_max])
           self.Fp_plot_sag.set_clim([Fp_min,Fp_max])
           self.Fp_plot_cor.set_clim([Fp_min,Fp_max])

           self.Te_plot.set_data((Te[int(self.NSlice/2),...]))
           self.Te_plot_cor.set_data((Te[:,int(Te.shape[1]/2),...]))
           self.Te_plot_sag.set_data(np.flip((Te[:,:,int(Te.shape[-1]/2)]).T,1))
           self.Te_plot.set_clim([Te_min,Te_max])
           self.Te_plot_cor.set_clim([Te_min,Te_max])
           self.Te_plot_sag.set_clim([Te_min,Te_max])

           self.alpha_plot.set_data((alpha[int(self.NSlice/2),...]))
           self.alpha_plot_cor.set_data((alpha[:,int(alpha.shape[1]/2),...]))
           self.alpha_plot_sag.set_data(np.flip((alpha[:,:,int(alpha.shape[-1]/2)]).T,1))
           self.alpha_plot.set_clim([alpha_min,alpha_max])
           self.alpha_plot_sag.set_clim([alpha_min,alpha_max])
           self.alpha_plot_cor.set_clim([alpha_min,alpha_max])

           plt.draw()
           plt.pause(1e-10)


  def _dce_fit(self,x):
    A_B = self.A_B
    mu_B  = self.mu_B
    A_G = self.A_G
    mu_G = self.mu_G
    tau = self.tau
    Tc = self.Tc

    Fp = x[1]*self.uk_scale[1]
    Te = x[2]*self.uk_scale[2]
    alpha = x[3]*self.uk_scale[3]

    C_fit = np.zeros((self.t.size,x.shape[1],x.shape[2],x.shape[3]),dtype=DTYPE)

    for j in range((self.t).size):
      ind_low = self.t[j]>tau
      ind_high = self.t[j]<=tau+Tc
      ind = ind_low & ind_high
      if np.any(ind==True):

#        t2 = self.t[j]-tau[ind]
#        t3 = 1.0/mu_B[ind]
#        t3[~np.isfinite(t3)]= 0
#        t4 = 1.0/mu_G[ind]
#        t4[~np.isfinite(t4)]= 0
#        t7 = mu_B[ind]*t2
#        t5 = np.exp(-t7)
#        t6 = 1.0/mu_B[ind]**2
#        t6[~np.isfinite(t6)]= 0
#        t8 = t3*t5
#        t30 = mu_G[ind]*t2
#        t9 = np.exp(-t30)
#        t31 = t4*t9
#        t10 = t8-t31
#        t11 = A_G[ind]*t10
#        t13 = t2*t3
#        t14 = t6+t13
#
#        C_fit[j,ind]=t11+A_B[ind]*t6-A_G[ind]*(t3-t4)-A_B[ind]*t5*t14

        t2 = self.t[j]-tau[ind]
        t3 = 1.0/mu_B[ind]
        t3[~np.isfinite(t3)]= 0
        t4 = 1.0/mu_G[ind]
        t4[~np.isfinite(t4)]= 0
        t7 = mu_B[ind]*t2
        t5 = np.exp(-t7)
        t6 = 1.0/mu_B[ind]**2
        t6[~np.isfinite(t6)]= 0
        t8 = t3*t5
        t31 = mu_G[ind]*t2
        t9 = np.exp(-t31)
        t32 = t4*t9
        t10 = t8-t32
        t11 = A_G[ind]*t10
        t28 = t2*t3
        t29 = t6+t28
#        t30 = 1.0/mu_B[ind]**3
#        t30[~np.isfinite(t30)]= 0
#        t33 = 1.0/mu_G[ind]**2
#        t33[~np.isfinite(t33)]= 0
#        t36 = t5*t6
#        t37 = t2*t3*t5
#        t41 = t30*2.0
#        t42 = t2*t6
#        t43 = t41+t42
#        t50 = t9*t33
#        t51 = t2*t4*t9
#        t52 = t50+t51
#        t64 = t5-t9

        C_fit[j,ind]=t11+A_B[ind]*t6-A_G[ind]*(t3-t4)-A_B[ind]*t5*t29

      ind = self.t[j]>(tau+Tc)
      if np.any(ind==True):
#        print(j)
#        t2 = self.t[j]-tau[ind]
#        t3 = 1.0/mu_B[ind]
#        t3[~np.isfinite(t3)]= 0
#        t4 = 1.0/mu_G[ind]
#        t4[~np.isfinite(t4)]= 0
#        t7 = mu_B[ind]*t2
#        t5 = np.exp(-t7)
#        t6 = 1.0/mu_B[ind]**2
#        t6[~np.isfinite(t6)]= 0
#        t8 = t3*t5
#        t30 = mu_G[ind]*t2
#        t9 = np.exp(-t30)
#        t31 = t4*t9
#        t10 = t8-t31
#        t11 = A_G[ind]*t10
#        t12 = Tc[ind]-self.t[j]+tau[ind]
#        t13 = t2*t3
#        t14 = t6+t13
#        t15 = np.exp(-alpha[ind])
#        t16 = t15-1.0
#        t17 = 1.0/Te[ind]
#        t17[~np.isfinite(t17)]= 0
#        t18 = 1.0/alpha[ind]
#        t18[~np.isfinite(t18)]= 0
#        t19 = t16*t17*t18
#        t20 = mu_B[ind]+t19
#        t21 = 1.0/t20
#        t21[~np.isfinite(t21)]= 0
#        t22 = mu_G[ind]+t19
#        t23 = 1.0/t22
#        t23[~np.isfinite(t23)]= 0
#        t24 = t12*t20
#        t25 = np.exp(t24)
#        t26 = 1.0/t20**2
#        t26[~np.isfinite(t26)]= 0
#        t27 = mu_B[ind]*t12
#        t28 = np.exp(t27)
#        t42 = t12*t16*t17*t18
#        t33 = np.exp(-t42)
#        t44 = t12*t21
#        t34 = t26-t44
#        t46 = t3*t28
#        t47 = mu_G[ind]*t12
#        t48 = np.exp(t47)
#        t49 = t21*t25
#        t50 = t12*t22
#        t51 = np.exp(t50)
#        t57 = A_B[ind]*t26
#        t58 = t21-t23
#        C_fit[j,ind]=t11-A_G[ind]*(t46-t4*t48)+A_B[ind]*t28*(t6-t3*t12)-t16*t33*\
#                        (t57-A_G[ind]*t58+A_G[ind]*(t49-t23*t51)-A_B[ind]*t25*t34)-A_B[ind]*t5*t14


        t2 = self.t[j]-tau[ind]
        t3 = 1.0/mu_B[ind]
        t3[~np.isfinite(t3)]= 0
        t4 = 1.0/mu_G[ind]
        t4[~np.isfinite(t4)]= 0
        t7 = mu_B[ind]*t2
        t5 = np.exp(-t7)
        t6 = 1.0/mu_B[ind]**2
        t6[~np.isfinite(t6)]= 0
        t8 = t3*t5
        t31 = mu_G[ind]*t2
        t9 = np.exp(-t31)
        t32 = t4*t9
        t10 = t8-t32
        t11 = A_G[ind]*t10
        t12 = np.exp(-alpha[ind])
        t13 = t12-1.0
        t14 = Tc[ind]-self.t[j]+tau[ind]
        t15 = 1.0/Te[ind]
        t15[~np.isfinite(t15)]= 0
        t16 = 1.0/alpha[ind]
        t16[~np.isfinite(t16)]= 0
        t17 = t13*t15*t16
        t18 = mu_B[ind]+t17
        t19 = 1.0/t18
        t19[~np.isfinite(t19)]= 0
        t20 = mu_G[ind]+t17
        t21 = 1.0/t20
        t21[~np.isfinite(t21)]= 0
        t22 = mu_B[ind]*t14
        t23 = np.exp(t22)
        t35 = t13*t14*t15*t16
        t24 = np.exp(-t35)
        t25 = 1.0/t18**2
        t25[~np.isfinite(t25)]= 0
        t26 = mu_G[ind]*t14
        t27 = np.exp(t26)
        t28 = t2*t3
        t29 = t6+t28
        t30 = 1.0/mu_B[ind]**3
        t30[~np.isfinite(t30)]= 0
        t33 = 1.0/mu_G[ind]**2
        t33[~np.isfinite(t33)]= 0
        t40 = t14*t19
        t34 = t25-t40
#        t41 = t30*2.0
        t46 = t19*t23
        t47 = t19-t21
        t48 = t3*t23
#        t55 = 1.0/alpha[ind]**2;
#        t55[~np.isfinite(t55)]= 0
#        t56 = t13*t15*t55
#        t57 = t12*t15*t16
#        t59 = t13*t14*t15*t55
#        t60 = t12*t14*t15*t16
        t63 = A_B[ind]*t24*t25
#        t66 = mu_B[ind]*t19*t23
#        t67 = t66-mu_G[ind]*t21*t27
#        t68 = A_G[ind]*t67
#        t69 = A_B[ind]*t19*t23
#        t70 = A_G[ind]*t13*t15*t16*t24*t47

        C_fit[j,ind]=t11-t13*(t63+A_G[ind]*(t46-t21*t27)-A_B[ind]*t23*t34-A_G[ind]*t24*t47)-A_G[ind]\
        *(t48-t4*t27)+A_B[ind]*t23*(t6-t3*t14)-A_B[ind]*t5*t29


    C_fit=C_fit*Fp
    return C_fit


  def _dce_fit_grad(self,x):
    A_B = self.A_B
    mu_B  = self.mu_B
    A_G = self.A_G
    mu_G = self.mu_G
    tau = self.tau
    Tc = self.Tc

    Fp = x[1]*self.uk_scale[1]
    Te = x[2]*self.uk_scale[2]
    alpha = x[3]*self.uk_scale[3]

    dC_fit=np.zeros((x.shape[0]-1,self.t.size,x.shape[1],x.shape[2],x.shape[3]),dtype=DTYPE)
    C_fit=np.zeros((self.t.size,x.shape[1],x.shape[2],x.shape[3]),dtype=DTYPE)
    for j in range(len(self.t)):
      ind_low = self.t[j]>tau
      ind_high = self.t[j]<=tau+Tc
      ind = ind_low & ind_high
      if np.any(ind==True):
        t2 = self.t[j]-tau[ind]
        t3 = 1.0/mu_B[ind]
        t3[~np.isfinite(t3)]= 0
        t4 = 1.0/mu_G[ind]
        t4[~np.isfinite(t4)]= 0
        t7 = mu_B[ind]*t2
        t5 = np.exp(-t7)
        t6 = 1.0/mu_B[ind]**2
        t6[~np.isfinite(t6)]= 0
        t8 = t3*t5
        t31 = mu_G[ind]*t2
        t9 = np.exp(-t31)
        t32 = t4*t9
        t10 = t8-t32
        t11 = A_G[ind]*t10
        t28 = t2*t3
        t29 = t6+t28
#        t30 = 1.0/mu_B[ind]**3
#        t30[~np.isfinite(t30)]= 0
#        t33 = 1.0/mu_G[ind]**2
#        t33[~np.isfinite(t33)]= 0
#        t36 = t5*t6
#        t37 = t2*t3*t5
#        t41 = t30*2.0
#        t42 = t2*t6
#        t43 = t41+t42
#        t50 = t9*t33
#        t51 = t2*t4*t9
#        t52 = t50+t51
#        t64 = t5-t9

        C_fit[j,ind]=t11+A_B[ind]*t6-A_G[ind]*(t3-t4)-A_B[ind]*t5*t29
#        dC_fit[0,j,ind]=t6-t5*29
#        dC_fit[1,j,ind]=t44+t45-A_B[ind]*t30*2.0+A_G[ind]*t6-A_G[ind]*t38
#        dC_fit[2,j,ind]=-t3+t4+t8-t32
#        dC_fit[3,j,ind]=t55-A_G[ind]*t33
#        dC_fit[8,j,ind]=t65+t73-A_B[ind]*mu_B[ind]*t5*t29
    for j in range(len(self.t)):
      ind = self.t[j]>(tau+Tc)
      if np.any(ind==True):
        t2 = self.t[j]-tau[ind]
        t3 = 1.0/mu_B[ind]
        t3[~np.isfinite(t3)]= 0
        t4 = 1.0/mu_G[ind]
        t4[~np.isfinite(t4)]= 0
        t7 = mu_B[ind]*t2
        t5 = np.exp(-t7)
        t6 = 1.0/mu_B[ind]**2
        t6[~np.isfinite(t6)]= 0
        t8 = t3*t5
        t31 = mu_G[ind]*t2
        t9 = np.exp(-t31)
        t32 = t4*t9
        t10 = t8-t32
        t11 = A_G[ind]*t10
        t12 = np.exp(-alpha[ind])
        t13 = t12-1.0
        t14 = Tc[ind]-self.t[j]+tau[ind]
        t15 = 1.0/Te[ind]
        t15[~np.isfinite(t15)]= 0
        t16 = 1.0/alpha[ind]
        t16[~np.isfinite(t16)]= 0
        t17 = t13*t15*t16
        t18 = mu_B[ind]+t17
        t19 = 1.0/t18
        t19[~np.isfinite(t19)]= 0
        t20 = mu_G[ind]+t17
        t21 = 1.0/t20
        t21[~np.isfinite(t21)]= 0
        t22 = mu_B[ind]*t14
        t23 = np.exp(t22)
        t35 = t13*t14*t15*t16
        t24 = np.exp(-t35)
        t25 = 1.0/t18**2
        t25[~np.isfinite(t25)]= 0
        t26 = mu_G[ind]*t14
        t27 = np.exp(t26)
        t28 = t2*t3
        t29 = t6+t28
        t30 = 1.0/mu_B[ind]**3
        t30[~np.isfinite(t30)]= 0
        t33 = 1.0/mu_G[ind]**2
        t33[~np.isfinite(t33)]= 0
        t40 = t14*t19
        t34 = t25-t40
#        t41 = t30*2.0
        t46 = t19*t23
        t47 = t19-t21
        t48 = t3*t23
        t55 = 1.0/alpha[ind]**2;
        t55[~np.isfinite(t55)]= 0
        t56 = t13*t15*t55
        t57 = t12*t15*t16
        t59 = t13*t14*t15*t55
        t60 = t12*t14*t15*t16
        t63 = A_B[ind]*t24*t25
#        t66 = mu_B[ind]*t19*t23
#        t67 = t66-mu_G[ind]*t21*t27
#        t68 = A_G[ind]*t67
#        t69 = A_B[ind]*t19*t23
#        t70 = A_G[ind]*t13*t15*t16*t24*t47


#        t36 = t5*t6
#        t37 = t2*t3*t5
#        t38 = t36+t37
#        t42 = t2*t6
#        t43 = t41+t42
#        t44 = A_B[ind]*t5*t43
#        t45 = A_B[ind]*t2*t5*t29
        t39 = 1.0/t18**3;
        t39[~np.isfinite(t39)]= 0
        t49 = 1.0/t20**2;
        t49[~np.isfinite(t49)]= 0
#        t50 = t9*t33
#        t51 = t2*t4*t9
#        t52 = t50+t51
#        t53 = A_G[ind]*t52
        t54 = 1.0/Te[ind]**2
        t54[~np.isfinite(t54)]= 0
        t58 = t56+t57
        t61 = t59+t60
        t62 = t21*t27
#        t64 = t5-t9
#        t65 = A_G[ind]*t64
#        t71 = t68+t69+t70-A_B[ind]*mu_B[ind]*t23*t34-A_B[ind]*t13*t15*t16*t24*t25
#        t72 = t23-t27
#        t73 = A_B[ind]*t3*t5
#        t74 = A_B[ind]*mu_B[ind]*t23*(t6-t3*t14)

        C_fit[j,ind]=t11-t13*(t63+A_G[ind]*(t46-t21*t27)-A_B[ind]*t23*t34-A_G[ind]*t24*t47)-A_G[ind]\
        *(t48-t4*t27)+A_B[ind]*t23*(t6-t3*t14)-A_B[ind]*t5*t29


        dC_fit[:,j,ind]=np.array((#-t5*t29-t13*(t24*t25-t23*t34)+t23*(t6-t3*t14),
#        t44+t45+A_G*(t6*t23-t3*t14*t23)-A_G*t38+t13*(A_G*(t23*t25-t14*t19*t23)+\
#                  A_B*t24*t39*2.0-A_G*t24*t25-A_B*t23*(t39*2.0-t14*t25)+A_B*t14*t23*t34)-A_B*t23*(t41-t6*t14)+A_B*t14*t23*(t6-t3*t14),
#            t8-t32-t48+t4*t27+t13*(-t46+t62+t24*t47),
#            t53-A_G*(t27*t33-t4*t14*t27)-t13*(A_G*(t27*t49-t14*t21*t27)-A_G*t24*t49),
            A_B[ind],##replaced later but saving memory allocations this way
#            t74-A_G*t72-t13*t71-A_B*t3*t23,
            self.uk_scale[2]*(-t13*(A_G[ind]*(t13*t16*t23*t25*t54-t13*t16*t27*t49*t54)-A_B[ind]*t23*(t13*t16*t39*t54*2.0-t13*t14*t16*t25*t54)-A_G[ind]*t24*(t13*t16*t25*t54-t13*t16*t49*t54)+A_B[ind]*t13*t16*t24*t39*t54*2.0+A_B[ind]*t13*t14*t16*t24*t25*t54-A_G[ind]*t13*t14*t16*t24*t47*t54)),
            self.uk_scale[3]*(t12*(t63+A_G[ind]*(t46-t62)-A_B[ind]*t23*t34-A_G[ind]*t24*t47)-t13*(A_G[ind]*(t23*t25*t58-t27*t49*t58)-A_G[ind]*t24*(t25*t58-t49*t58)-A_B[ind]*t23*(t39*t58*2.0-t14*t25*t58)+A_B[ind]*t24*t25*t61+A_B[ind]*t24*t39*t58*2.0-A_G[ind]*t24*t47*t61))))#,
#            t65+t73+t74-A_G*t72-t13*t71-A_B*t3*t23-A_B*mu_B*t5*t29))


    dC_fit=dC_fit*Fp
    dC_fit[0,:]=C_fit*self.uk_scale[1]
    dC_fit[~np.isfinite(dC_fit)] = 0
    return (C_fit*Fp,dC_fit)

  def _set_init_scales(self,dscale):

    self.mask = np.ones_like(self.alpha)
    self.mask[self.alpha<1e-6] = 0


    test_M0 =0.01*dscale*np.ones((self.NSlice,self.dimY,self.dimX),dtype=DTYPE)*self.mask

    FP = 0.5*np.ones((self.NSlice,self.dimY,self.dimX),dtype=DTYPE)*self.mask#parameters[4][...]#
    Te = 5*np.ones((self.NSlice,self.dimY,self.dimX),dtype=DTYPE)*self.mask#parameters[6][...]#5*np.ones((NSlice,dimY,dimX),dtype=DTYPE)#
    Te[Te<1/6] = 1/6
    alpha = 0.4*np.ones((self.NSlice,self.dimY,self.dimX),dtype=DTYPE)*self.mask#parameters[7][...]#
    alpha[alpha<1e-4] = 1e-4
    x = np.array([test_M0,FP,Te,alpha],dtype=DTYPE)

    G_x = self._execute_forward_3D(x)
    self.uk_scale[0]*=1/np.median(np.abs(G_x))

    x[0]/=self.uk_scale[0]

#    res_R1 = par["file"]["GT/dR1"][()]
#    test = self.dce_fit(x)*self.r
#    print("Check model if it is the same up to 5e-5: ",np.allclose(test,res_R1,5e-5))


    DG_x =  self._execute_gradient_3D(x)
    for j in range(1,unknowns_TGV+unknowns_H1):
      self.uk_scale[j] = self.uk_scale[j]*np.linalg.norm(np.abs(DG_x[0,...]))/np.linalg.norm(np.abs(DG_x[j,...]))
      x[j]/=self.uk_scale[j]

    print('Fp scale: ',self.uk_scale[1],
                              '/ M0_scale: ',self.uk_scale[0],'/ Te_scale: ', self.uk_scale[2], '/ alpha_scale: ',self.uk_scale[3])

    return x








