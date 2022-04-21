#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:22:09 2021

@author: eligal
"""
from pyqmri.models.template import BaseModel, constraints
import numpy as np
from sympy import symbols, diff, lambdify
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Model(BaseModel):
    def __init__(self,par):
        super().__init__(par)
        
        full_slices = par["file"]["popt"].shape[1]
        sliceind = slice(int(full_slices / 2) -
                         int(np.floor((par["NSlice"]) / 2)),
                         int(full_slices / 2) +
                         int(np.ceil(par["NSlice"] / 2)))
        
        self.omega = par["omega"]
        
        self.dt = 1/(self.omega[1:]-self.omega[:-1])
        self.dt /= np.max(self.dt)
        while len(self.omega.shape) < 4:
            self.omega = self.omega[..., None]
        
        self.popt = par["file"]["popt"][:,sliceind,:,:]
        self.popt[~np.isfinite(self.popt)] = 0
        self.popt = self.popt.astype(self._DTYPE_real)
        
        
        self.amount_pools = 5
        
        par["unknowns_TGV"] = self.amount_pools*3 + 1
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]
        self.unknowns = par["unknowns"]
        
        self.initFunctions()
        self._ind1 = 0
        self._ind2 = 0
        
        for j in range(par["unknowns"]):
            self.uk_scale.append(1)
            
        par["weights"] = np.array([1]*self.unknowns,dtype=par["DTYPE_real"])
        # # # par["weights"] = np.linspace(1, 1, self.amount_pools*3 + 1).astype(par["DTYPE_real"])
        # # # par["weights"][4:7] *= 4
        # # # par["weights"][5:7] /= 30
        # # par["weights"][4] *= 5
        # par["weights"][5] /= 10
        # par["weights"][7] /= 10
        # par["weights"][8] /= 10
        # # par["weights"][7:9] /= 1e2
        # par["weights"][11] /= 100
        # # par["weights"][12] /= 300
        # # # par["weights"][-3:-1] *= 6
        # # par["weights"][-3] *= 10
        # # par["weights"][-2] /= 10
        # # par["weights"][-1] *= 10
        
        # # par["weights"][0] /= 1e3
        # # par["weights"][1] /= 10
        # # par["weights"][2] /= 10
        # # par["weights"][3] /= 10
        
        par["weights"][4] /= 4
        par["weights"][5] /= 4
        par["weights"][6] *= 4
        par["weights"][7] /= 2
        
        par["weights"][8] /= 1
        par["weights"][9] *= 6
        par["weights"][10] /= 1
        par["weights"][11] *= 4
        
        par["weights"][12] *= 2
        par["weights"][13] /= 1
        par["weights"][14] /= 1
        par["weights"][15] *= 6
        
        
    def rescale(self, x):
        """Rescale the unknowns with the scaling factors.

        Rescales each unknown with the corresponding scaling factor.

        Parameters
        ----------
          x : numpy.array
            The array of unknowns to be rescaled

        Returns
        -------
          numpy.array:
            The rescaled unknowns
        """
        tmp_x = np.copy(x)
        for i in range(x.shape[0]):
            tmp_x[i] *= self.uk_scale[i]
        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return {"data": tmp_x,
                "unknown_name": self.uk_names,
                "real_valued": const}

    
    def _execute_forward_3D(self, x):

        S = self.signal(x,self.uk_scale, self.omega)
        
        return S.astype(self._DTYPE)*self.dscale
    
    def _execute_gradient_3D(self,x):
            
        gradients = []
        for grad_fun in self.gradients:
            gradients.append(grad_fun(x, self.uk_scale, self.omega)*
                             np.ones((self.NScan,)+x.shape[1:]))    
        gradients = np.array(gradients, dtype=self._DTYPE, order='C')
        
        return gradients*self.dscale
    
    def initFunctions(self):

        my_symbol_names = "M0,M0_sc"
        for j in range(self.amount_pools):
            my_symbol_names += (",a_"+str(j)+",a_"+str(j)+"_sc"+
                                ",Gamma_"+str(j)+",Gamma_"+str(j)+"_sc"+
                                ",omega_"+str(j)+",omega_"+str(j)+"_sc"
                                )
        my_symbol_names += ",omega"
                                  
        my_symbols = symbols(my_symbol_names)
        self.uk_names = my_symbols[:-1:2]
        
        def symbolicLorentzian(a, gamma, omega_0, omega):
            return a * (gamma/2)**2/((gamma/2)**2 + (omega_0 - omega)**2)
        
        for j in range(self.amount_pools):
            a = my_symbols[2+6*j]*my_symbols[3+6*j]
            gamma = my_symbols[4+6*j]*my_symbols[5+6*j]
            omega_0 = my_symbols[6+6*j]*my_symbols[7+6*j]
            if j == 0:
                signal = symbolicLorentzian(a, gamma, omega_0, my_symbols[-1])
            else:
                signal += symbolicLorentzian(a, gamma, omega_0, my_symbols[-1])
        signal = my_symbols[0]*my_symbols[1]*(1-signal)
        
        symbolic_gradients = [diff(signal, my_symbols[0])]
        # symbolic_gradients.append(diff(signal, my_symbols[2]))
        for j in range(self.amount_pools):
            symbolic_gradients.append(diff(signal, my_symbols[2+6*j]))
            symbolic_gradients.append(diff(signal, my_symbols[2+6*j+2]))
            symbolic_gradients.append(diff(signal, my_symbols[2+6*j+4]))
                
        params = my_symbols[:-1:2]
        scales = my_symbols[1:-1:2]
        omega = my_symbols[-1]
        
        self.signal = lambdify([params, scales, omega], signal)
        self.gradients = []
        for j in range(len(symbolic_gradients)):
            self.gradients.append(lambdify([params, scales, omega], 
                                           symbolic_gradients[j]))
        
    
    def computeInitialGuess(self, **kwargs):
        self.images = np.abs(kwargs["images"]/kwargs["dscale"])
        self.dscale = kwargs["dscale"]
        # import ipdb
        # import matplotlib.pyplot as plt
        # import pyqmri
        # ipdb.set_trace()
        
        if self.amount_pools==1:
            lb = [0,0,0.1,-4]
            ub = [1e5,1,50,4]
            self.constraints = []
            for min_val,max_val in zip(lb,ub):
                self.constraints.append(constraints(min_val,max_val,True))
            self.guess =  [1,1,20,-1]
        elif self.amount_pools==2:
            lb = [0,
                  0.1,0.5,-2,
                  0,15,-4]
            ub = [1e3,
                  0.9,6,0.2,
                  0.4,100,0]
            self.constraints = []
            for min_val,max_val in zip(lb,ub):
                self.constraints.append(constraints(min_val,max_val,True))
            self.guess =  [1,
                    0.8,2,0,
                    0.1,50,-2]
        elif self.amount_pools==3:
            lb = [0,-np.pi,
                  0,0.1,-0.01,
                  0,2,-0.01,
                  0.001,20,-4]
            ub = [1e3,np.pi,
                  1,2,0.01,
                  1,8,0.01,
                  0.4,100,0]
            self.constraints = []
            for min_val,max_val in zip(lb,ub):
                self.constraints.append(constraints(min_val,max_val,True))
            self.guess =  [1,0,
                   0.8,1,0,
                   0.4,3,0,
                   0.1,50,-2]
        elif self.amount_pools==4:
            lb = [0,
                  0.02,0.3,-1,
                  0.0,0.4,3,
                  0,2,-5,
                  0,1,1]
            ub = [1,
                  1,10,1,
                  0.3,6,6,
                  0.2,7,-2,
                  0.2,4,2.5]
            self.constraints = []
            for min_val,max_val in zip(lb,ub):
                self.constraints.append(constraints(min_val,max_val,True))
            self.guess =  [1,
                    0.9,0.8,0,
                    0.1,2,3.5,
                    0.1,4,-3.5,
                    0.01,2,2.2]
        elif self.amount_pools==5:
            lb = [0,
                  0.02,0.3,-1,
                  0.0,0.4,+3,
                  0.0,0.5,-4,
                  0.0,10,-8,
                  0.0,1e-3,1.7]
            ub = [1e5,
                  1,10,+1,
                  0.2,4,+4,
                  0.6,10,-2.5, #mittlerer Wert war 5
                  1,99,0,
                  0.5,3.5,2.5]
            self.constraints = []
            for min_val,max_val in zip(lb,ub):
                self.constraints.append(constraints(min_val,max_val,True))
                            
            self.guess =  [
                1,
                0.9,1.4,0, #Wasser #[1, #ground truth
                0.025,0.5,3.5, #APT
                0.02,3,-3.5, #NOE #mittlerer Wert war 7
                0.1,25,-4, #MT
                0.01,1.0,2.2] #APT
            # self.guess = self.popt 
        elif self.amount_pools==6:
            lb = [0,
                  0.02,0.3,-1,
                  0,0.4,3,
                  0.0,1,-4.5,
                  0,10,-4,
                  0,0.4,1,
                  0,2,1.5]
            ub = [1,
                  1,10,1,
                  0.2,1.5,4,
                  0.4,5,-2,
                  1,100,-2,
                  0.2,1.5,2.5,
                  0.2,5,5]
            self.constraints = []
            for min_val,max_val in zip(lb,ub):
                self.constraints.append(constraints(min_val,max_val,True))
            self.guess =  [1,
                    0.95,0.5,0,
                    0.025,1,3.5,
                    0.02,7,-3.5,
                    0.1,25,-2,
                    0.01,1,2,
                    0.01,3,3.5]
        else:
            raise AssertionError("number of pools out of range")
        self.guess = np.array(self.guess)[:,None,None,None] * np.ones((self.unknowns, self.NSlice, self.dimY, self.dimX))
        # self.guess[0] = self.popt[0]
        # self.guess[2:] = self.popt[1:]
        self.constraints[0].real = False
        # for const in self.constraints[1::3]:
        #     const.real = False
        # self.guess[0] = args[0][self.omega.squeeze()==0]
        self.guess = self.guess.astype(self._DTYPE)
        # self.guess[0] = (kwargs["images"][0])/self.dscale
        print("Max Signal intensity: ", np.max(np.abs(self.guess[0])))
        # self.guess[1] = np.exp(1j*np.angle(args[0][0]))
        # x = self.guess
        # x_scale = np.max(np.abs(x).reshape(x.shape[0], -1), axis=-1)
        # x_scale[x_scale==0] = 1
        # self.uk_scale = x_scale
        # self.guess = x/x_scale[:,None,None,None]
        # for uk in range(self.unknowns):
        #     self.constraints[uk].update(x_scale[uk])


        
    def plot_unknowns(self, x, dim_2D=False):
        unknowns = self.rescale(x)
        tmp_x = unknowns["data"]
        uknames = unknowns["unknown_name"]

        images = np.abs(self._execute_forward_3D(x) / self.dscale)
        
        # tmp_x[0] = np.abs(tmp_x[0])
        # tmp_x[2::3] = np.abs(tmp_x[2::3])
        tmp_x = np.real(tmp_x)

        if dim_2D:
            pass
        else:
            if not self._figure:
                self.ax = []
                plot_dim = int(np.ceil(np.sqrt(len(self.uk_scale))))
                plt.ion()
                self._figure = plt.figure(figsize=(12, 6))
                self._figure.subplots_adjust(hspace=0.3, wspace=0)
                wd_ratio = np.tile([1, 1 / 20, 1 / (5)], plot_dim)
                self.gs = gridspec.GridSpec(
                    plot_dim+1, 3*plot_dim,
                    width_ratios=wd_ratio, hspace=0.3, wspace=0)
                self._figure.tight_layout()
                self._figure.patch.set_facecolor(plt.cm.viridis.colors[0])
                for grid in self.gs:
                    self.ax.append(plt.subplot(grid))
                    self.ax[-1].axis('off')
                self._plot = []
                for j in range(len(self.uk_scale)):
                    self._plot.append(
                        self.ax[3 * j].imshow(
                            tmp_x[j, int(self.NSlice / 2), ...]))
                    self.ax[3 *
                            j].set_title(uknames[j], color='white')
                    self.ax[3 * j + 1].axis('on')
                    cbar = self._figure.colorbar(
                        self._plot[j], cax=self.ax[3 * j + 1])
                    cbar.ax.tick_params(labelsize=12, colors='white')
                    for spine in cbar.ax.spines:
                        cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)
                self._figure.canvas.mpl_connect(
                    'button_press_event',
                    self.onclick)

                self.plot_ax = plt.subplot(self.gs[-1, :])
                self.plot_ax.set_title("Time course", color='w')
                self.time_course_ref = []

                self.time_course_ref.append(self.plot_ax.plot(
                    self.omega.squeeze(), np.real(
                        self.images[...,
                                    int(self.NSlice/2),
                                    self._ind2, self._ind1]).T,
                    'x')[0])
                self.plot_ax.set_prop_cycle(None)

                self.time_course = self.plot_ax.plot(
                    self.omega.squeeze(), np.real(
                        images[..., int(self.NSlice/2),
                               self._ind2, self._ind1]).T)
                self.plot_ax.set_xlim(np.max(self.omega), np.min(self.omega))
                self.plot_ax.set_ylim(
                    np.minimum(np.real(images[...,
                                              int(self.NSlice/2),
                                              self._ind2,
                                              self._ind1]).min(),
                               np.real(self.images[...,
                                                   int(self.NSlice/2),
                                                   self._ind2,
                                                   self._ind1]).min()),
                    1.2*np.maximum(np.real(images[...,
                                                  int(self.NSlice/2),
                                                  self._ind2,
                                                  self._ind1]).max(),
                                   np.real(self.images[...,
                                                       int(self.NSlice/2),
                                                       self._ind2,
                                                       self._ind1]).max()))
                for spine in self.plot_ax.spines:
                    self.plot_ax.spines[spine].set_color('white')
                self.plot_ax.xaxis.label.set_color('white')
                self.plot_ax.yaxis.label.set_color('white')
                self.plot_ax.tick_params(axis='both', colors='white')

                plt.draw()
                plt.show()
                plt.pause(1e-4)
            else:
                for j in range(len(self.uk_scale)):
                    self._plot[j].set_data(
                        tmp_x[j, int(self.NSlice / 2), ...])
                    self._plot[j].set_clim(
                        [tmp_x[j].min(), tmp_x[j].max()])


                self.time_course[0].set_ydata(
                    np.real(images[
                        ..., int(self.NSlice/2), self._ind2, self._ind1]))
                self.plot_ax.set_ylim(
                    np.minimum(np.real(images[...,
                                              int(self.NSlice/2),
                                              self._ind2,
                                              self._ind1]).min(),
                               np.real(self.images[...,
                                                   int(self.NSlice/2),
                                                   self._ind2,
                                                   self._ind1]).min()),
                    1.2*np.maximum(np.real(images[...,
                                                  int(self.NSlice/2),
                                                  self._ind2,
                                                  self._ind1]).max(),
                                   np.real(self.images[...,
                                                       int(self.NSlice/2),
                                                       self._ind2,
                                                       self._ind1]).max()))
                plt.draw()
                plt.pause(1e-10)

    def onclick(self, event):
        if event.inaxes in self.ax[::3]:
            self._ind1 = int(event.xdata)
            self._ind2 = int(event.ydata)

            self.time_course_ref[0].set_ydata(np.real(
                    self.images[...,
                                int(self.NSlice/2),
                                self._ind2, self._ind1]).T)