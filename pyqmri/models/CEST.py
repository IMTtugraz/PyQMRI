#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:22:09 2021

@author: eligal
"""
from pyqmri.models.template import BaseModel, constraints
import numpy as np
from sympy import *
import scipy.io
from scipy import optimize
import matplotlib.pyplot as plt
import math
import h5py
import time

class Model(BaseModel):
    def __init__(self,par):
        super().__init__(par)
        
        self.omega = par["omega"]
        self.popt = par["file"]["popt"][()]
        #self.popt = par["popt"]
        self.amount_pools = 5#int((self.popt.shape[0]-3)/3)
        #self.zcorr = par["Z_corrExt"]
        self.guess = self.computeInitialGuess()
        
        par["unknowns_TGV"] = self.amount_pools*3 + 1
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]
        
        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

    
    def _execute_forward_3D(self,x):
        tic = time.perf_counter()
        shape = x.shape
        #print(shape)
        x = x.reshape(shape[0], -1)
        forward = []
        for values in x.T:
            forward.append(self.Lorentzian_multiple(self.omega,values))
        forward = np.array(forward)
        #print(forward.shape)
        forward = forward.transpose()
        forward = forward.reshape((self.NScan,)+shape[1:])
        forward = np.require(forward, requirements='C', dtype=self._DTYPE)
        toc = time.perf_counter()
        print("forward: {e} seconds".format(e=(toc-tic)))
        #print(forward.shape)
        return forward
    
    def Lorentzian_function(self,x,coeffs):
        a = coeffs[0]
        gamma = coeffs[1]
        x0 = coeffs[2]
        return a*(gamma/2)**2/((x-x0)**2+(gamma/2)**2)
    
    def Lorentzian_multiple(self,x,coeffs):
        off = coeffs[0]
        amount = int(len(coeffs[1:])/3)
        func_sum = 0
        counter = 0
        while counter in range(0,amount):
            lorentz = self.Lorentzian_function(x,coeffs[3*counter+1:3*counter+5])
            func_sum = func_sum + lorentz
            counter += 1
        return off - func_sum
        
    def Lorentzian_residuals(self,coeffs,x,data):
        return data - self.Lorentzian_multiple(x,coeffs)
    
    def Lorentzian_fitting(self,npools=5):
        #initial = self.computeInitialGuess(npools)
        ttic = time.perf_counter()
        lb = []
        ub = []
        for constraints in self.constraints:
            lb.append(constraints.min)
            ub.append(constraints.max)
        shape = self.zcorr.shape
        zcorr = self.zcorr.reshape(shape[0]*shape[1]*shape[2],shape[3])
        opt = []
        f = 0
        for data_point in zcorr:
            tic = time.perf_counter()
            f+=1
            if np.isnan(data_point).any():
                opt_point = np.empty((npools*3)+1)
            else:
                opt_point = scipy.optimize.least_squares(self.Lorentzian_residuals,self.guess, 
                                                     bounds=(lb,ub),args=(self.omega,data_point))["x"]
            opt.append(opt_point)
            toc = time.perf_counter()
            print("iteration #{f}: {e} seconds".format(f=f,e=(toc-tic)))
        opt = np.array(opt)
        ttoc = time.perf_counter()
        print("fitting: {e} seconds".format(e=(ttoc-ttic)))
        opt = opt.reshape(shape[0],shape[1],shape[2],opt.shape[-1])
        #plt.plot(x,self.Lorentzian_multiple(self.omega,opt),"g")
        #plt.plot(x,data_point,"r")
        
        return opt
    
    def _execute_gradient_3D(self,x):
        #x = x[...,1:]
        
        #uk_scales
        #maximum of 6 pools?
        #assumed order of input values: M0, Gamma, x0
        tic = time.perf_counter()

        offset,offset_sc,M00,M00_sc,Gamma0,Gamma0_sc,x00,x00_sc,M01,M01_sc,\
            Gamma1,Gamma1_sc,x01,x01_sc,M02,M02_sc,Gamma2,Gamma2_sc,x02,x02_sc,\
                M03,M03_sc,Gamma3,Gamma3_sc,x03,x03_sc,M04,M04_sc,Gamma4,Gamma4_sc,\
                    x04,x04_sc,M05,M05_sc,Gamma5,Gamma5_sc,x05,x05_sc,omega = \
        symbols('offset, offset_sc,'
                'M00, M00_sc, Gamma0, Gamma0_sc, x00, x00_sc,'
                'M01, M01_sc, Gamma1, Gamma1_sc, x01, x01_sc,'
                'M02, M02_sc, Gamma2, Gamma2_sc, x02, x02_sc,'
                'M03, M03_sc, Gamma3, Gamma3_sc, x03, x03_sc,'
                'M04, M04_sc, Gamma4, Gamma4_sc, x04, x04_sc,'
                'M05, M05_sc, Gamma5, Gamma5_sc, x05, x05_sc, omega')
        init_printing(use_unicode=True)
        M0 = [M00,M01,M02,M03,M04,M05]
        M0_sc = [M00_sc,M01_sc,M02_sc,M03_sc,M04_sc,M05_sc]
        Gamma = [Gamma0,Gamma1,Gamma2,Gamma3,Gamma4,Gamma5]
        Gamma_sc = [Gamma0_sc,Gamma1_sc,Gamma2_sc,Gamma3_sc,Gamma4_sc,Gamma5_sc]
        x0 = [x00,x01,x02,x03,x04,x05]
        x0_sc = [x00_sc,x01_sc,x02_sc,x03_sc,x04_sc,x05_sc]
    
        def S(M0,Gamma,x0,M0_sc,Gamma_sc,x0_sc):
            return M0*M0_sc * (Gamma*Gamma_sc/2)**2/((Gamma*Gamma_sc/2)**2 + (x0*x0_sc - omega)**2)
        
        
        nparams = x.shape[0]
        amount = self.amount_pools
        pools = 0
        all_params = []
        scales = []
        all_params.append(offset)
        scales.append(offset_sc)
        for i in range(amount):
            pools += S(M0[i],Gamma[i],x0[i],M0_sc[i],Gamma_sc[i],x0_sc[i])
            all_params.append(M0[i])
            all_params.append(Gamma[i])
            all_params.append(x0[i])
            scales.append(M0_sc[i])
            scales.append(Gamma_sc[i])
            scales.append(x0_sc[i])
            
        pools = offset*offset_sc - pools
        
        sym_grads = np.array([])
        for j in range(amount):
            sym_grads = np.append(sym_grads,[(diff(pools,M0[j])),(diff(pools,Gamma[j])),(diff(pools,x0[j]))])
            
        final_grads = []
        shape = x.shape
        x = x.reshape(shape[0], -1)
        for sym_grad in sym_grads:
            grads = []
            grad_s = lambdify([all_params,scales,omega],sym_grad)
            for values in x.T:
                grad = grad_s(values,self.uk_scale,self.omega)
                grads.append(grad)
            final_grads.append(grads)

        final_grads = np.array(final_grads)
        final_grads = final_grads.transpose((0,2,1))
        final_grads = final_grads.reshape((nparams-1,self.NScan)+shape[1:])
        
        scale_grad = np.ones((1,self.NScan)+shape[1:])*self.uk_scale[0]
        
        final_grads = np.concatenate((scale_grad,final_grads), axis=0)
        final_grads = np.require(final_grads,requirements="C", dtype=self._DTYPE)
        
        
        
        toc = time.perf_counter()
        print("gradients: {e} seconds".format(e=(toc-tic)))
        print(final_grads.shape)
        return final_grads

    
    def computeInitialGuess(self,*args):
        if self.amount_pools==1:
            max_y = np.nanmax(self.popt[...,0])
            lb = [max_y,0,15,-4]
            ub = [max_y,0.04,50,0.0]
            self.constraints = []
            for min_val,max_val in zip(lb,ub):
                self.constraints.append(constraints(min_val,max_val))
            return [max_y,0.02,25,-2]
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
            return [0.7,
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
            return [1,
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
            return [1,
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
            return np.array([1, 0.9,1.4,0, #Wasser #[1, #ground truth
                    0.025,0.5,3.5, #APT
                    0.02,5,-3.5, #NOE #mittlerer Wert war 7
                    0.1,25,-2, #MT
                    0.01,1,2.2], dtype=self._DTYPE)[:,None,None,None] * np.ones((16, self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
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
            return [1,
                    0.95,0.5,0,
                    0.025,1,3.5,
                    0.02,7,-3.5,
                    0.1,25,-2,
                    0.01,1,2,
                    0.01,3,3.5]
        else:
            raise AssertionError("number of pools out of range")
        
    
#test stuff
#ohc06 = scipy.io.loadmat("/home/eligal/OHC06.mat")
#ohc06["NScan"]=ohc06["omega"].shape[0]
#ohc06["NSlice"]=ohc06["Z_corrExt"].shape[-2]
#ohc06["dimX"]=ohc06["Z_corrExt"].shape[0]
#ohc06["dimY"]=ohc06["Z_corrExt"].shape[1]
#ohc06["DTYPE"]=ohc06["Z_corrExt"].dtype
#ohc06["DTYPE_real"]=ohc06["Z_corrExt"].dtype
#test = Model(ohc06)
#initial = test.computeInitialGuess()
#popt=ohc06["popt"][...,1:16]
#constraints = test.constraints
#fit = test.Lorentzian_fitting()
#grads = test._execute_forward_3D(initial)

#pyqmri.run(slices=1,trafo=0,data="/home/eligal/23052018_VFA_08.h5",model="CEST",config="default",useCGguess=False)