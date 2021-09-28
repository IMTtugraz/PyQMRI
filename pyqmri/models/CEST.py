#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:22:09 2021

@author: eligal
"""
from pyqmri.models.template import BaseModel, constraints
import numpy as np
from sympy import symbols, diff, lambdify

import time

class Model(BaseModel):
    def __init__(self,par):
        super().__init__(par)
        
        full_slices = par["file"]["popt"].shape[1]
        sliceind = slice(int(full_slices / 2) -
                         int(np.floor((par["NSlice"]) / 2)),
                         int(full_slices / 2) +
                         int(np.ceil(par["NSlice"] / 2)))
        
        self.omega = par["omega"]
        while len(self.omega.shape) < 4:
            self.omega = self.omega[..., None]
        
        self.popt = par["file"]["popt"][:,sliceind,:,:]
        self.popt[~np.isfinite(self.popt)] = 0
        self.popt = self.popt.astype(self._DTYPE)
        
        
        self.amount_pools = 5
        
        par["unknowns_TGV"] = self.amount_pools*3 + 1
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]
        
        self.initFunctions()
        
        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

    
    def _execute_forward_3D(self, x):
        tic = time.perf_counter()
        

        S = self.signal(x,self.uk_scale, self.omega)

        toc = time.perf_counter()
        print("forward: {e} seconds".format(e=(toc-tic)))
        return S.astype(self._DTYPE)
    
    def _execute_gradient_3D(self,x):
        tic = time.perf_counter()

            
        gradients = []
        for grad_fun in self.gradients:
            gradients.append(grad_fun(x, self.uk_scale, self.omega)*
                             np.ones((1,self.NScan)+x.shape[1:]))    
        gradients = np.array(gradients, dtype=self._DTYPE, order='C')
        
        toc = time.perf_counter()
        print("gradients: {e} seconds".format(e=(toc-tic)))
        return gradients
    
    def initFunctions(self):

        my_symbol_names = "M0,M0_sc"
        for j in range(self.amount_pools):
            my_symbol_names += (",a_"+str(j)+",a_"+str(j)+"_sc"+
                                ",Gamma_"+str(j)+",Gamma_"+str(j)+"_sc"+
                                ",omega_"+str(j)+",omega_"+str(j)+"_sc"
                                )
        my_symbol_names += ",omega"
                                  
        my_symbols = symbols(my_symbol_names)
        
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
        signal = my_symbols[0]*my_symbols[1]-signal
        
        symbolic_gradients = [diff(signal, my_symbols[0])]
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
        
    
    def computeInitialGuess(self,*args):
        if self.amount_pools==1:
            max_y = np.nanmax(self.popt[...,0])
            lb = [max_y,0,15,-4]
            ub = [max_y,0.04,50,0.0]
            self.constraints = []
            for min_val,max_val in zip(lb,ub):
                self.constraints.append(constraints(min_val,max_val))
            self.guess =  [max_y,0.02,25,-2]
        elif self.amount_pools==2:
            lb = [0,
                  0.1,0.5,-0.001,
                  0,15,-4]
            ub = [1,
                  1,6,0.001,
                  0.4,100,0]
            self.constraints = []
            for min_val,max_val in zip(lb,ub):
                self.constraints.append(constraints(min_val,max_val))
            self.guess =  [0.7,
                    0.8,2,0,
                    0.1,50,-2]
        elif self.amount_pools==3:
            lb = [0.5,
                  0,0.1,-0.01,
                  0,2,-0.01,
                  0,15,-4]
            ub = [1,
                  1,2,0.01,
                  1,8,0.01,
                  0.4,100,0]
            self.constraints = []
            for min_val,max_val in zip(lb,ub):
                self.constraints.append(constraints(min_val,max_val))
            self.guess =  [1,
                   0.8,1,0,
                   0.4,3,0,
                   0.1,50,-2]
        elif self.amount_pools==4:
            lb = [0.5,
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
                self.constraints.append(constraints(min_val,max_val))
            self.guess =  [1,
                    0.9,0.8,0,
                    0.1,2,3.5,
                    0.1,4,-3.5,
                    0.01,2,2.2]
        elif self.amount_pools==5:
            lb = [0,
                  0.02,0.3,-1,
                  0.0,0.4,+3,
                  0.0,1,-4.5,
                  0.0,10,-4,
                  0.0,0.4,1]
            ub = [1e3,
                  1e2,10,+1,
                  1e2,4,+4,
                  1e2,7,-2, #mittlerer Wert war 5
                  1e2,100,4,
                  1e2,2.5,2.5]
            self.constraints = []
            for min_val,max_val in zip(lb,ub):
                self.constraints.append(constraints(min_val,max_val,True))
                            
            self.guess =  np.array([1e-3, 0.9,1.4,0, #Wasser #[1, #ground truth
                    0.025,0.5,3.5, #APT
                    0.02,5,-3.5, #NOE #mittlerer Wert war 7
                    0.1,25,-2, #MT
                    0.01,1,2.2], dtype=self._DTYPE)[:,None,None,None] * np.ones((16, self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
            self.guess = self.popt 
        elif self.amount_pools==6:
            lb = [0.5,
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
                self.constraints.append(constraints(min_val,max_val))
            self.guess =  [1,
                    0.95,0.5,0,
                    0.025,1,3.5,
                    0.02,7,-3.5,
                    0.1,25,-2,
                    0.01,1,2,
                    0.01,3,3.5]
        else:
            raise AssertionError("number of pools out of range")