

import numpy as np
import time

import gradients_divergences_old as gd
import matplotlib.pyplot as plt

import pynfft.nfft as nfft

import scipy.optimize as op

plt.ion()

DTYPE = np.complex64


class Model_Reco: 
  def __init__(self,par):
    self.par = par
    self.unknowns_TGV = par.unknowns_TGV
    self.unknowns_H1 = par.unknowns_H1
    self.unknowns = par.unknowns
    self.NSlice = par.NSlice
    self.NScan = par.NScan
    self.dimX = par.dimX
    self.dimY = par.dimY
    self.scale = np.sqrt(par.dimX*par.dimY)
    self.NC = par.NC
    self.N = par.N
    self.Nproj = par.Nproj
    print("Please Set Parameters, Data and Initial images")

    
  def irgn_solve_2D(self, x, iters, data):
    

    ###################################
#    ### Adjointness     
    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
    yy = np.random.random_sample(np.shape(data)).astype(DTYPE)
    a = np.vdot(xx.flatten(),self.operator_adjoint_2D(yy).flatten())
    b = np.vdot(self.operator_forward_2D(xx).flatten(),yy.flatten())
    test = np.abs(a-b)
    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))
    x_old = x
    a = self.FT(self.step_val[:,None,:,:]*self.Coils)
    b = self.operator_forward_2D(x)
    res = data - a + b
#    print("Test the norm: %2.2E  a=%2.2E   b=%2.2E" %(np.linalg.norm(res.flatten()),np.linalg.norm(a.flatten()), np.linalg.norm(b.flatten())))
  
    ############ Douglas Rachford
#    x = self.pdr_tgv_solve_2D(x,res,iters)
    ############# PD Solve
#    x = self.tgv_solve_2D(x,res,iters)
    ############# Tikhonov Solve     
    x = self.tikonov_solve(x_old,res,iters)
#    fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.step_val[:,None,:,:]*self.Coils))**2
#           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_1(x)-self.v))
#           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_2(self.v))) 
#           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)  

#    print ("Function value after GN-Step: %f" %fval)
    self.model.plot_unknowns(x,True) 
    return x
   
  def irgn_solve_3D(self,x, iters,data):
    

    ###################################
    ### Adjointness     
    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
    yy = np.random.random_sample(np.shape(data)).astype(DTYPE)
    a = np.vdot(xx.flatten(),self.operator_adjoint_2D(yy).flatten())
    b = np.vdot(self.operator_forward_2D(xx).flatten(),yy.flatten())
    test = np.abs(a-b)
    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))
    x_old = x
    a = self.FT(self.step_val[:,None,:,:]*self.Coils)
    b = self.operator_forward_2D(x)
    res = data - a + b
    print("Test the norm: %2.2E  a=%2.2E   b=%2.2E" %(np.linalg.norm(res.flatten()),np.linalg.norm(a.flatten()), np.linalg.norm(b.flatten())))
  
    x = self.tikonov_solve(x_old,res,iters)
#      
#    fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.step_val[:,None,:,:]*self.Coils))**2
#           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_3(x)-self.v))
#           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_3(self.v))) 
#           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)    
#    print ("Function value after GN-Step: %f" %fval)
    self.model.plot_unknowns(x)  
    return x
        

    
  def execute_2D(self):
      self.init_plan()
      self.FT = self.nFT_2D
      self.FTH = self.nFTH_2D
      gamma = self.irgn_par.gamma
      delta = self.irgn_par.delta


      
      self.result = np.zeros((self.irgn_par.max_GN_it,self.unknowns_TGV+self.unknowns_H1,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
      result = np.copy(self.model.guess)
      for islice in range(self.par.NSlice):
        self.irgn_par.gamma = gamma
        self.irgn_par.delta = delta
        self.Coils = np.squeeze(self.par.C[:,islice,:,:])    
        self.conjCoils = np.conj(self.Coils)   
        self.v = np.zeros(([self.unknowns_TGV,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
        self.r = np.zeros(([self.NScan,self.NC,self.Nproj,self.N]),dtype=DTYPE)
        self.z1 = np.zeros(([self.unknowns_TGV,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
        self.z2 = np.zeros(([self.unknowns_TGV,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
        self.z3 = np.zeros(([self.unknowns_H1,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)  
        iters = self.irgn_par.start_iters        
        self.NSlice = 1
        for i in range(self.irgn_par.max_GN_it):
          start = time.time()       
          self.step_val = self.model.execute_forward_2D(result[:,islice,:,:],islice)
          self.grad_x_2D = self.model.execute_gradient_2D(result[:,islice,:,:],islice)
          self.conj_grad_x_2D = np.conj(self.grad_x_2D)

          result[:,islice,:,:] = self.irgn_solve_2D(np.squeeze(result[:,islice,:,:]), iters, np.squeeze(self.data[:,:,islice,:]))
          self.result[i,:,islice,:,:] = result[:,islice,:,:]
          
          iters = np.fmin(iters*2,self.irgn_par.max_iters)
          self.irgn_par.gamma = self.irgn_par.gamma*0.8
          self.irgn_par.delta = self.irgn_par.delta*2
          
          end = time.time()-start
          print("Elapsed time: %f seconds" %end)
            
        
      
      
     
  def execute_3D(self):
      self.init_plan()     
      self.FT = self.nFT_3D
      self.FTH = self.nFTH_3D
      iters = self.irgn_par.start_iters

      self.v = np.zeros(([self.unknowns,3,self.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)
      self.r = np.zeros_like(self.data,dtype=DTYPE)
      self.z1 = np.zeros(([self.unknowns,3,self.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)
      self.z2 = np.zeros(([self.unknowns,6,self.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)        
      
      self.result = np.zeros((self.irgn_par.max_GN_it+1,self.unknowns,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
      self.result[0,:,:,:,:] = np.copy(self.model.guess)

      self.Coils = np.squeeze(self.par.C)        
      self.conjCoils = np.conj(self.Coils)
      for i in range(self.irgn_par.max_GN_it):
        start = time.time()       

        self.step_val = np.nan_to_num(self.model.execute_forward_3D(self.result[i,:,:,:,:]))
        self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_3D(self.result[i,:,:,:,:]))
        self.conj_grad_x_2D = np.nan_to_num(np.conj(self.grad_x_2D))
          
        self.result[i+1,:,:,:,:] = self.irgn_solve_3D(self.result[i,:,:,:,:], iters, self.data)


        iters = np.fmin(iters*2,self.irgn_par.max_iters)
        self.irgn_par.gamma = self.irgn_par.gamma*0.8 ##0.7
        self.irgn_par.delta = self.irgn_par.delta*2
          
        end = time.time()-start
        print("Elapsed time: %f seconds" %end)
               
      
  def operator_forward_2D(self, x):
    
    return self.FT(np.sum(x[:,None,...]*self.grad_x_2D,axis=0)[:,None,...]*self.Coils)

    
  def operator_adjoint_2D(self, x):
    
    return np.squeeze(np.sum(np.squeeze(np.sum(self.FTH(x)*self.conjCoils,axis=1))*self.conj_grad_x_2D,axis=1)) 

  def operator_forward_3D(self, x):
      
    return self.FT(np.sum(x[:,None,:,:,:]*self.grad_x,axis=0)[:,None,:,:,:]*self.Coils3D)

    
  def operator_adjoint_3D(self, x):
      
    return np.squeeze(np.sum(np.squeeze(np.sum(self.FTH(x)*self.conjCoils3D,axis=1)*self.conj_grad_x),axis=1)) 
    
  def tgv_solve_2D(self, x, res, iters):
    alpha = self.irgn_par.gamma
    beta = self.irgn_par.gamma*2

    #estimate operator norm using power iteration
    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
    yy = self.operator_adjoint_2D(self.operator_forward_2D(xx));
    for i in range(10):
       if not np.isclose(np.linalg.norm(yy.flatten()),0):
           xx = yy/np.linalg.norm(yy.flatten())
       else:
           xx = yy
       yy = self.operator_adjoint_2D(self.operator_forward_2D(xx))
       l1 = np.vdot(yy.flatten(),xx.flatten());
    L = np.max(np.abs(l1)) ## Lipschitz constant estimate       
    print("Operatornorm estimate L: %f ---- "%(L))      
    
    tau = 1/np.sqrt(L+16**2+8**2)
    sigma = 1/np.sqrt(L+16**2+8**2)
    
    xk = x
    x_new = np.zeros_like(x,dtype=DTYPE)
    x_bar = np.copy(x)
    v = np.copy(self.v)
    v_new = np.copy(self.v)
    v_bar = np.copy(self.v)
    
    
    r = np.copy(self.r)
    z1 = np.copy(self.z1)
    z2 = np.copy(self.z2)

#    r_new = np.zeros_like(res,dtype=DTYPE)
#    z1_new = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
#    z2_new = np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)

    

    Kyk1 = np.zeros_like(x,dtype=DTYPE)
    Kyk2 = np.zeros_like(z1,dtype=DTYPE)
    
#    Ax = np.zeros_like(res,dtype=DTYPE)
#    Ax_Axold = np.zeros_like(res,dtype=DTYPE)
#    Axold = np.zeros_like(res,dtype=DTYPE)    
#    tmp = np.zeros_like(res,dtype=DTYPE)    
    
#    Kyk1_new = np.zeros_like(x,dtype=DTYPE)
#    Kyk2_new = np.zeros_like(z1,dtype=DTYPE)
    
    
    delta = self.irgn_par.delta
    mu = 1/delta
    theta = 1
    
#    theta_line = 1.0
#    beta_line = 0.10
#    mu_line = 0.1
#    delta_line = 0.2
    scal = np.zeros((self.par.dimX,self.par.dimY),dtype=DTYPE)
    
#    ynorm = 0.0
#    lhs = 0.0

    primal = 0.0
    dual = 0.0
    gap = 0.0

    
#    gradx = np.zeros_like(z1,dtype=DTYPE)
#    gradx_xold = np.zeros_like(z1,dtype=DTYPE)
#    
#    v_vold = np.zeros_like(v,dtype=DTYPE)
#    symgrad_v = np.zeros_like(z2,dtype=DTYPE)
#    symgrad_v_vold = np.zeros_like(z2,dtype=DTYPE)
#    
#    Axold = self.operator_forward_2D(x)

    
    
    for i in range(iters):
      
#      gradx = gd.fgrad_1(x_new)
#      np.subtract(gradx , gd.fgrad_1(x),gradx_xold)
#      np.subtract(v_new,v,v_vold)
#      symgrad_v = gd.sym_bgrad_2(v_new)
#      np.subtract(symgrad_v , gd.sym_bgrad_2(v),symgrad_v_vold)
#      Ax = self.operator_forward_2D(x_new)
#      np.subtract(Ax,Axold,Ax_Axold)
      
      
      z1 = z1 + sigma*( gd.fgrad_1(x_bar) - v_bar  )
      z1 = z1/np.maximum(1,(np.sqrt(np.sum(z1**2,axis=(0,1)))/alpha))
   
      z2 = z2 + sigma*( gd.sym_bgrad_2(v_bar))
      scal = np.sqrt( np.sum(z2[:,0,:,:]**2 + z2[:,1,:,:]**2 + 2*z2[:,2,:,:]**2,axis=0))
      scal = np.maximum(1,scal/(beta))
      z2 = z2/scal
      

      r =(( r  + sigma*self.operator_forward_2D(x_bar) ) - sigma*res)/(1+sigma/self.irgn_par.lambd)
      
      Kyk1 = self.operator_adjoint_2D(r) - gd.bdiv_1(z1)     
      Kyk2 = -z1 - gd.fdiv_2(z2)
      
      x_new = np.maximum(0,((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta))
      
      
      x_new[0,...] = np.maximum(0,np.minimum(300/self.model.M0_sc,x_new[0,...]))
      x_new[1,...] = np.real(np.maximum(50/self.model.T1_sc,np.minimum(5000/self.model.T1_sc,x_new[1,...])))


      
      v_new = v-tau*Kyk2
      
#      theta =1/np.sqrt(1+2*mu*tau)
#      tau = theta*tau
#      sigma = sigma/tau
      
      x_bar = x_new+theta*(x_new-x)
      v_bar = v_new+theta*(v_new-v)
      

      x = np.copy(x_new)
      v = np.copy(v_new)
        
      if not np.mod(i,20):
        plt.figure(1)
        plt.imshow(np.transpose(np.abs(x[0,:,:]*self.model.M0_sc)))
        plt.pause(0.05)
        plt.figure(2)
#        plt.imshow(np.transpose(np.abs(-self.par.TR/np.log(x[1,:,:]))),vmin=0,vmax=3000)
        plt.imshow(np.transpose(np.abs(x[1,:,:])))
        plt.pause(0.05)
        primal= np.real(self.irgn_par.lambd/2*np.linalg.norm((self.operator_forward_2D(x)-res).flatten())**2+alpha*np.sum(np.abs((gd.fgrad_1(x)-v))) +
                 beta*np.sum(np.abs(gd.sym_bgrad_2(v))) + 1/(2*delta)*np.linalg.norm((x-xk).flatten())**2)
    
        dual = np.real(-delta/2*np.linalg.norm((-Kyk1).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1).flatten()) + np.sum(Kyk2) 
                - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r.flatten())**2 - np.vdot(res.flatten(),r.flatten()))
        gap = np.abs(primal - dual)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal,dual,gap))
        
        
    self.v = v
    self.r = r
    self.z1 = z1
    self.z2 = z2
    return x
 
#  def tgv_solve_3D(self, x, res, iters):
#    alpha = self.irgn_par.gamma
#    beta = self.irgn_par.gamma*2
#    
#    
#    tau = 1/np.sqrt(16**2+8**2)
#    xk = x
#    x_new = np.zeros_like(x,dtype=DTYPE)
#    
#    r = np.zeros_like(res,dtype=DTYPE)
#    z1 = np.zeros(([self.unknowns,3,self.par.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)
#    z2 = np.zeros(([self.unknowns,6,self.par.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)
#    v = np.zeros_like(z1,dtype=DTYPE)
#    
#    r_new = np.zeros_like(r,dtype=DTYPE)
#    z1_new = np.zeros_like(z1,dtype=DTYPE)
#    z2_new = np.zeros_like(z2,dtype=DTYPE)
#    v_new = np.zeros_like(v,dtype=DTYPE)
#    
#
#    Kyk1 = np.zeros_like(x,dtype=DTYPE)
#    Kyk2 = np.zeros_like(z1,dtype=DTYPE)
#    
#    Ax = np.zeros_like(res,dtype=DTYPE)
#    Ax_Axold = np.zeros_like(Ax,dtype=DTYPE)
#    Axold = np.zeros_like(Ax,dtype=DTYPE)    
#    tmp = np.zeros_like(Ax,dtype=DTYPE)    
#    
#    Kyk1_new = np.zeros_like(Kyk1,dtype=DTYPE)
#    Kyk2_new = np.zeros_like(Kyk2,dtype=DTYPE)
#    
#    
#    delta = self.irgn_par.delta
#    mu = 1/delta
#    
#    theta_line = 1
#    beta_line = 1
#    mu_line = 0.5
#    scal = np.zeros((self.par.NSlice,self.par.dimX,self.par.dimY),dtype=DTYPE)
#    
#    ynorm = 0
#    lhs = 0
#
#    primal = 0
#    dual = 0
#    gap = 0
#
#    
#    gradx = np.zeros_like(z1,dtype=DTYPE)
#    gradx_xold = np.zeros_like(z1,dtype=DTYPE)
#    
#    v_old = np.zeros_like(v,dtype=DTYPE)
#    symgrad_v = np.zeros_like(z2,dtype=DTYPE)
#    symgrad_v_vold = np.zeros_like(z2,dtype=DTYPE)
#    
#    Axold = self.operator_forward_3D(x)
#    Kyk1 = self.operator_adjoint_3D(r) - gd.bdiv_3(z1)
#    Kyk2 = -z1 - gd.fdiv_3(z2)
#    
#    
#    for i in range(iters):
#      np.maximum(0,((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta),x_new)
#      
#      
#      np.maximum(0,np.minimum(300/self.model.M0_sc,x_new[0,:,:,:],x_new[0,:,:,:]),x_new[0,:,:,:])
#      np.abs(np.maximum(50/self.model.T1_sc,np.minimum(5000/self.model.T1_sc,
#                                                       x_new[1,:,:,:],x_new[1,:,:,:]),
#                                                       x_new[1,:,:,:]),x_new[1,:,:,:])
#
##      np.abs(x_new[1,:,:],x_new[1,:,:])
#      
#      
#      v_new = v-tau*Kyk2
#      
#      beta_new = beta_line*(1+mu*tau)
#      
#      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
#      beta_line = beta_new
#      
#      gradx = gd.fgrad_3(x_new)
#      gradx_xold = gradx - gd.fgrad_3(x)
#      v_vold = v_new-v
#      symgrad_v = gd.sym_bgrad_3(v_new)
#      symgrad_v_vold = symgrad_v - gd.sym_bgrad_3(v)
#      Ax = self.operator_forward_3D(x_new)
#      Ax_Axold = Ax-Axold
#    
#      while True:
#        
#        theta_line = tau_new/tau
#        
#        z1_new = z1 + beta_line*tau_new*( gradx + theta_line*gradx_xold - v_new - theta_line*v_vold  )
#        z1_new = z1_new/np.maximum(1,(np.sqrt(np.sum(z1_new**2,axis=(0,1)))/alpha))
#     
#        z2_new = z2 + beta_line*tau_new*( symgrad_v + theta_line*symgrad_v_vold )
#        scal = np.sqrt( np.sum(z2_new[:,0,:,:,:]**2 + z2_new[:,1,:,:,:]**2 +
#                    z2_new[:,2,:,:,:]**2+ 2*z2_new[:,3,:,:,:]**2 + 
#                    2*z2_new[:,4,:,:,:]**2+2*z2_new[:,5,:,:,:]**2,axis=0))
#        np.maximum(1,scal/(beta),scal)
#        z2_new = z2_new/scal
#        
#        
#        
#        tmp = Ax+theta_line*Ax_Axold
#        r_new = (( r  + beta_line*tau_new*(tmp) ) - beta_line*tau_new*res)/(1+beta_line*tau_new/self.irgn_par.lambd)
#        
#        Kyk1_new = self.operator_adjoint_3D(r_new)-gd.bdiv_3(z1_new)
#        Kyk2_new = -z1_new -gd.fdiv_3(z2_new)
#
#        
#        ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),(z1_new-z1).flatten(),(z2_new-z2).flatten()]))
#        lhs = np.sqrt(beta_line)*tau_new*np.linalg.norm(np.concatenate([(Kyk1_new-Kyk1).flatten(),(Kyk2_new-Kyk2).flatten()]))        
#        if lhs <= ynorm:
#            break
#        else:
#            tau_new = tau_new*mu_line
#            
#      Kyk1 = (Kyk1_new)
#      Kyk2 =  (Kyk2_new)
#      Axold =(Ax)
#      z1 = (z1_new)
#      z2 = (z2_new)
#      r =  (r_new)
#      tau =  (tau_new)
#        
#  
#      x = x_new
#      v = v_new
#        
#      if not np.mod(i,20):
#        plt.figure(1)
#        plt.imshow(np.transpose(np.abs(x[0,0,:,:]*self.model.M0_sc)))
#        plt.pause(0.05)
#        plt.figure(2)
#        plt.imshow(np.transpose(np.abs(-self.par.TR/np.log(x[1,:,:]))),vmin=0,vmax=3000)
##        plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
#        plt.pause(0.05)
#        primal= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*np.sum(np.abs((gradx-v))) +
#                 beta*np.sum(np.abs(symgrad_v)) + 1/(2*delta)*np.linalg.norm((x-xk).flatten())**2)
#    
#        dual = np.real(-delta/2*np.linalg.norm((-Kyk1_new).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1_new).flatten()) + np.sum(Kyk2_new) 
#                - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r.flatten())**2 - np.vdot(res.flatten(),r.flatten()))
#        gap = np.abs(primal - dual)
#        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal,dual,gap))
#        
#        
#    self.v = v
#    return x  
#  
  
  
  def FT_2D(self,  x):

#    nscan = np.shape(x)[0]
#    NC = np.shape(x)[1]
#    scale =  np.sqrt(np.shape(x)[2]*np.shape(x)[3])
#        
#    result = np.zeros_like(x,dtype=DTYPE)
##    cdef int scan=0
##    cdef int coil=0
#    for scan in range(nscan):
#      for coil in range(NC):
#       result[scan,coil,:,:] = self.fft_forward(x[scan,coil,:,:])/scale
      
    return self.fft_forward(x)/np.sqrt(np.shape(x)[2]*np.shape(x)[3])
 
  
  
      
  def FTH_2D(self, x):
#    nscan = np.shape(x)[0]
#    NC = np.shape(x)[1]
#    scale =  np.sqrt(np.shape(x)[2]*np.shape(x)[3])
#        
#    result = np.zeros_like(x,dtype=DTYPE)
#    cdef int scan=0
#    cdef int coil=0
#    for scan in range(nscan):
#      for coil in range(NC):
#        result[scan,coil,:,:] = self.fft_back(x[scan,coil,:,:])*scale
      
    return self.fft_back(x)*np.sqrt(np.shape(x)[2]*np.shape(x)[3])

  def nFT_2D(self, x):

    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]   
    result = np.zeros((nscan,NC,self.par.Nproj*self.par.N),dtype=DTYPE)
    scal = self.scale
    plan = self._plan
    for scan in range(nscan):
      for coil in range(NC):
          plan[scan].f_hat = x[scan,coil,:,:]/scal
          result[scan,coil,:] = plan[scan].trafo()
      
    return np.reshape(result*self.dcf_flat,[nscan,NC,self.par.Nproj,self.par.N])



  def nFTH_2D(self, x):
    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]     
    result = np.zeros((nscan,NC,self.par.dimX,self.par.dimY),dtype=DTYPE)
    plan = self._plan
    x = x*self.dcf
    for scan in range(nscan):
        for coil in range(NC):  
            plan[scan].f = x[scan,coil,:,:]
            result[scan,coil,:,:] = plan[scan].adjoint()
      
    return result/self.scale
      
  
  
  def nFT_3D(self, x):

    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]   
    NSlice = self.par.NSlice
    result = np.zeros((nscan,NC,NSlice,self.par.Nproj*self.par.N),dtype=DTYPE)
    scan=0
    coil=0
    islice=0
    plan = self._plan    
    scal = self.scale
    for scan in range(nscan):
      for coil in range(NC):
        for islice in range(NSlice):
          plan[scan][coil].f_hat = x[scan,coil,islice,:,:]/scal
          result[scan,coil,islice,:] = plan[scan][coil].trafo()
      
    return np.reshape(result*self.dcf_flat,[nscan,NC,NSlice,self.par.Nproj,self.par.N])



  def  nFTH_3D(self, x):
    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]  
    NSlice = self.par.NSlice
    result = np.zeros((nscan,NC,NSlice,self.par.dimX,self.par.dimY),dtype=DTYPE)
    dcf = self.dcf  
    scan=0
    coil=0
    islice=0    
    plan = self._plan      
    for scan in range(nscan):
        for coil in range(NC): 
          for islice in range(NSlice):
            plan[scan][coil].f = x[scan,coil,islice,:,:]*dcf
            result[scan,coil,islice,:,:] = plan[scan][coil].adjoint()
      
    return result/self.scale


  def init_plan(self):
    plan = []

    traj_x = np.imag(np.asarray(self.traj))
    traj_y = np.real(np.asarray(self.traj))

    for i in range(self.NScan):

#        plan.append([])
        points = np.transpose(np.array([traj_x[i,:,:].flatten(),traj_y[i,:,:].flatten()]))      
        plan.append(nfft.NFFT([self.dimX,self.dimY],self.N*self.Nproj))
        plan[i].x = points
        plan[i].precompute()
    self._plan = plan







  def pdr_tgv_solve_2D(self, x, res, iters):
    
    

    
    alpha = self.irgn_par.gamma
    beta = self.irgn_par.gamma*2
    
    delta = self.irgn_par.delta
    gamma1 = 1/delta    
    

    
    sigma0 = 1e0
    tau0 = 1e0
    
    L = 0
    L1 = 0
    L2 = 0
    L_pow = 0
#    ##estimate operator norm using power iteration
#    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
#    yy = xx+sigma0*tau0*self.operator_adjoint_2D(self.operator_forward_2D(xx));
#    for i in range(10):
#       if not np.isclose(np.linalg.norm(yy.flatten()),0):
#           xx = yy/np.linalg.norm(yy.flatten())
#       else:
#           xx = yy
#       yy = xx+sigma0*tau0*self.operator_adjoint_2D(self.operator_forward_2D(xx))
#       l1 = np.vdot(yy.flatten(),xx.flatten());
#    L_pow = np.max(np.abs(l1)) ## Lipschitz constant estimate   
    
    L1 = np.max(np.abs(self.grad_x[0,:,None,:,:]*self.Coils
                                   *np.conj(self.grad_x[0,:,None,:,:])*np.conj(self.Coils)))
    L2 = np.max(np.abs(self.grad_x[1,:,None,:,:]*self.Coils
                                   *np.conj(self.grad_x[1,:,None,:,:])*np.conj(self.Coils)))
#

    L = np.max((L1,L2))*self.unknowns*self.par.NScan*self.par.NC
    L =1+sigma0*tau0*np.sqrt(L**2+8+16) + (4*sigma0*tau0)**2/(1+4*sigma0*tau0)
    print("Operatornorm estimate L1: %f ---- L2: %f -----  L: %f ---- Lpow: %f "%(L1,L2,L,L_pow))    
#    L = L_pow

    gamma = 2*gamma1/L    #(<2*gamma1/L)
    

    
    L1 = 1+sigma0*tau0*np.sqrt((L1*self.unknowns*self.par.NScan*self.par.NC*np.ones_like(x[0,...]))**2+8+16) + (4*sigma0*tau0)**2/(1+4*sigma0*tau0)
    L2 = 1+sigma0*tau0*np.sqrt((L2*self.unknowns*self.par.NScan*self.par.NC*np.ones_like(x[1,...]))**2+8+16) + (4*sigma0*tau0)**2/(1+4*sigma0*tau0)
    L= np.concatenate((1/L1[None,None,...],1/L2[None,None,...]),axis=0)
    
    theta = 1/np.sqrt(1+sigma0*gamma)

    
    sigma = sigma0
    tau = tau0 

    
    xk = np.copy(x)
    xhat = np.copy(x)    
    x = np.copy(xk)
    v = np.copy(self.v)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)    
    vhat = np.copy(v)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)   
    
    r = np.copy(self.r)#np.zeros_like(res,dtype=DTYPE)
    z1 = np.copy(self.z1)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2 = np.copy(self.z2)#np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)

    rhat = np.copy(self.rhat)#np.zeros_like(res,dtype=DTYPE)
    z1hat = np.copy(self.z1hat)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2hat = np.copy(self.z2hat)#np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    

#    Kd1 = np.zeros_like(res,dtype=DTYPE)
    Kd2 = np.zeros_like(z1,dtype=DTYPE)
#    Kd3 = np.zeros_like(z2,dtype=DTYPE)
    
    
#    b1 = np.zeros_like(x,dtype=DTYPE)    
#    b2 = np.zeros_like(v,dtype=DTYPE)
    
    d1 = np.copy(self.d1)#np.zeros_like(x,dtype=DTYPE)      
    d2 = np.copy(self.d2)#np.zeros_like(v,dtype=DTYPE)
                   
#    T1 = np.zeros_like(d1,dtype=DTYPE)    
#    T2 = np.zeros_like(d2,dtype=DTYPE)                      

    scal = np.zeros((self.par.dimX,self.par.dimY),dtype=DTYPE)


    primal = []
    dual = []
    gap = []
    j = 0
#    plt.close('all')
#    fig1,ax1 = plt.subplots(1,1,num=1)    
#    fig2,ax2 = plt.subplots(1,1,num=2)
#    fig3,ax3 = plt.subplots(1,1,num=3)
#    ax3.set_xscale('log')
#    ax3.set_yscale('log')
#    points = ax3.plot(1,1, 1,1, 1,1)

    for i in range(iters):
      
      #### Prox (id+sigma F) of x_hat
      x = np.maximum(0,(xhat+(sigma*gamma1)*xk)/(1+sigma*gamma1))
      x[0,:,:] = np.minimum(3000/self.model.M0_sc,x[0,:,:])
      x[1,:,:] = np.real(np.maximum(self.model.min_T1,np.minimum(self.model.max_T1,x[1,:,:])))
      v = (vhat)
      
      
      #### Prox (id+tau G) of y_hat
      r = (( rhat - tau*res)/(1+tau/self.irgn_par.lambd)        )
      z1 = (z1hat/np.maximum(1,(np.sqrt(np.sum(z1hat**2,axis=(0,1)))/alpha)))
      scal = np.sqrt( np.sum(z2hat[:,0,:,:]**2 + z2hat[:,1,:,:]**2 + 2*z2hat[:,2,:,:]**2,axis=0) )
      scal = np.maximum(1,scal/(beta))
      z2 = (z2hat/scal)
  
      
      
      #### update b
      #### Accelerated Version
#      b1 = (((1+theta)*x-theta*xhat) - sigma*(self.operator_adjoint_2D((1+theta)*r-rhat) - gd.bdiv_1((1+theta)*z1-z1hat)))
#      b2 = (((1+theta)*v-theta*vhat) - sigma*(-((1+theta)*z1-z1hat) - gd.fdiv_2((1+theta)*z2-z2hat)))
      ### normal Version
#      b1 = ((2*x-xhat) - sigma*(self.operator_adjoint_2D(2*r-rhat) - gd.bdiv_1(2*z1-z1hat)))
#      b2 = ((2*v-vhat) - sigma*(-(2*z1-z1hat) - gd.fdiv_2(2*z2-z2hat))          )
#          
      #### update d
#      Kd1 = (self.operator_forward_2D(d1))
      Kd2 = (gd.fgrad_1(d1)-d2)
#      Kd3 = (gd.sym_bgrad_2(d2))
      
#      T1 = d1 + sigma0*tau0*(self.operator_adjoint_2D(Kd1) - gd.bdiv_1(Kd2))
#      T2 = d2 + sigma0*tau0*(-Kd2 - gd.fdiv_2(Kd3))
      
#      d1 = (d1 + 1/L*(b1-T1))
#      d2 = (d2 + 1/L*(b2-T2))

######################### 1/L Precomputed!!!!
      d1 = (d1 + np.squeeze(L)*((((1+theta)*x-theta*xhat) - 
            sigma*(self.operator_adjoint_2D((1+theta)*r-rhat) - gd.bdiv_1((1+theta)*z1-z1hat)))-
            (d1 + sigma0*tau0*(self.operator_adjoint_2D((self.operator_forward_2D(d1))) - gd.bdiv_1(Kd2)))))
      
      d2 = (d2 + L*((((1+theta)*v-theta*vhat) - sigma*(-((1+theta)*z1-z1hat) -
                        gd.fdiv_2((1+theta)*z2-z2hat)))-(d2 + sigma0*tau0*(-Kd2 - gd.fdiv_2((gd.sym_bgrad_2(d2)))))))

#      d1 = (d1 + 1/L*(((2*x-xhat) - sigma*(self.operator_adjoint_2D(2*r-rhat) - 
#            gd.bdiv_1(2*z1-z1hat)))-(d1 + 
#            sigma0*tau0*(self.operator_adjoint_2D((self.operator_forward_2D(d1))) - 
#                         gd.bdiv_1(Kd2)))))
#      
#      d2 = (d2 + 1/L*(((2*v-vhat) - sigma*(-(2*z1-z1hat) - gd.fdiv_2(2*z2-z2hat)))-
#                      (d2 + sigma0*tau0*(-Kd2 - gd.fdiv_2((gd.sym_bgrad_2(d2)))))))
#      
      #### Accelerated Version
      xhat = (theta*(xhat-x) + d1)
      vhat = (theta*(vhat-v) + d2)
      #### normal Version
#      xhat = ((xhat-x) + d1)
#      vhat = ((vhat-v) + d2)
      #### Accelerated Version      
      rhat = (r + 1/theta*tau*(self.operator_forward_2D(d1)))
      z1hat = (z1 + 1/theta*tau*(gd.fgrad_1(d1)-d2))
      z2hat = (z2 + 1/theta*tau*(gd.sym_bgrad_2(d2)))
      #### normal Version
#      rhat = (r + tau*(self.operator_forward_2D(d1)))
#      z1hat = (z1 + tau*(gd.fgrad_1(d1)-d2))
#      z2hat = (z2 + tau*(gd.sym_bgrad_2(d2)))
      #### Accelerated Version      
      sigma = (theta*sigma)
      tau =(1/theta*tau)
      theta = (1/np.sqrt(1+sigma*gamma))
#      
        
      if not np.mod(i,10):
        plt.figure(1)
        plt.imshow(np.transpose(np.abs(x[0,:,:]*self.model.M0_sc)))
#        fig1.colorbar(ax1)
        plt.pause(0.05)
        plt.figure(2)
#        plt.imshow(np.transpose(np.abs(x[1,:,:]))*self.model.T1_sc)
#        fig2.colorbar(ax2)        
        plt.imshow(np.transpose(np.abs(-self.par.TR/np.log(x[1,:,:]))))
        plt.pause(0.05)
        primal.append(np.real(self.irgn_par.lambd/2*np.linalg.norm((self.operator_forward_2D(x)-res).flatten())**2+alpha*np.sum(np.abs((gd.fgrad_1(x)-v))) +
                 beta*np.sum(np.abs(gd.sym_bgrad_2(v))) + 1/(2*delta)*np.linalg.norm((x-xk).flatten())**2))
#        
#
        dual.append(np.real(-delta/2*np.linalg.norm(-(self.operator_adjoint_2D(r)-gd.bdiv_1(z1)).flatten())**2 
             - np.vdot(xk.flatten(),-(self.operator_adjoint_2D(r)-gd.bdiv_1(z1)).flatten()) + np.sum(-z1 -gd.fdiv_2(z2)) 
             - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r.flatten())**2 - np.vdot(res.flatten(),r.flatten())))
        gap.append(np.abs(primal[j] - dual[j]))
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal[j],dual[j],gap[j]))
        j = j+1
#        points[0].set_data([np.abs(primal),np.arange(0,i+1,10)])
#        points[1].set_data([np.abs(dual),np.arange(0,i+1,10)])
#        points[2].set_data([np.abs(gap),np.arange(0,i+1,10)])
#        plt.pause(0.05)
        
    self.v = v
    self.rhat = rhat
    self.z1hat = z1hat
    self.z2hat = z2hat
    self.z1 = z1
    self.z2 = z2
    self.r = r
    self.d1 = d1
    self.d2 = d2
    return x  
  
  
  def tikonov_solve(self,xk,data,iters):
    x_old = np.concatenate((np.real(xk),np.imag(xk)),axis=0) 
    con_min = np.zeros_like(x_old)
    con_max = np.zeros_like(x_old)
    ATA = np.squeeze(np.zeros((self.par.unknowns,self.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE))
    for i in range(self.par.unknowns):
      ATA[i,...] = np.sum(np.sum(self.FTH(self.FT(self.grad_x_2D[i,:,None,...]*self.Coils))*self.conjCoils,axis=1)*self.conj_grad_x_2D[i,...],axis=0)
    ATd = self.operator_adjoint_2D(data)
    
    for j in range(len(self.model.constraints)):   
      con_min[j,...] = self.model.constraints[j].min*np.ones_like(con_min[j,...])
      con_max[j,...] = self.model.constraints[j].max*np.ones_like(con_max[j,...])
      con_min[j+self.par.unknowns,...] = self.model.constraints[j].min*np.ones_like(con_min[j+self.par.unknowns,...])
      con_max[j+self.par.unknowns,...] = self.model.constraints[j].max*np.ones_like(con_max[j+self.par.unknowns,...])  
      if self.model.constraints[j].real:
        con_min[j+self.par.unknowns,...] = np.zeros_like(con_min[j,...])
        con_max[j+self.par.unknowns,...] = np.zeros_like(con_max[j,...])    
    

    con = np.concatenate((con_min.flatten()[None,:],con_max.flatten()[None,:]),axis=0).T.tolist()

    optres = op.minimize(self.fun_val,x_old,args=(xk,data,ATA,ATd),method='L-BFGS-B',jac=self.lhs_2D,options={'maxiter':iters,'disp':10},bounds=con)
    x = np.squeeze(np.reshape(optres.x,(self.unknowns*2,self.NSlice,self.dimX,self.dimY)))
#    fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.step_val[:,None,:,:]*self.Coils))**2
#           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_1(x)-self.v))
#           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_2(self.v))) 
#           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)  
    fval = self.fun_val(optres.x,xk,data,ATA,ATd)
    x = x[:self.unknowns,...] + 1j*x[self.unknowns:,...]

    print ("Function value after GN-Step: %f" %fval)   
    return x     
      


    
  def fun_val(self,x,xk,data,ATA,ATd):
    x = np.squeeze(np.reshape(x,(self.unknowns*2,self.NSlice,self.dimX,self.dimY)))
    x = x[:self.unknowns,...] + 1j*x[self.unknowns:,...]
    return (self.irgn_par.lambd/2*np.linalg.norm(self.operator_forward_2D(x)-data)**2+
            self.irgn_par.gamma/2*np.linalg.norm((x))**2 + 1/(2*self.irgn_par.delta)*np.linalg.norm(x-xk)**2).flatten()
    
  def lhs_2D(self,x,xk,data,ATA,ATd):
    x = np.squeeze(np.reshape(x,(self.unknowns*2,self.NSlice,self.dimX,self.dimY)))
    x = x[:self.unknowns,...] + 1j*x[self.unknowns:,...]
    tmp = (self.irgn_par.lambd*ATA*x
           +self.irgn_par.gamma*x+1/self.irgn_par.delta*((x)) - 
            self.irgn_par.lambd*ATd-1/self.irgn_par.delta*xk).flatten()
    return np.concatenate((np.real(tmp),np.imag(tmp)),axis=0)
      
  

    

    
    
        