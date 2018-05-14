

import numpy as np
import time

import gradients_divergences_old as gd
import matplotlib.pyplot as plt

#import pynfft.nfft as nfft

import scipy.optimize as op
import primaldualtoolbox

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

    
  def irgn_solve_2D(self, x, iters, data,islice):
    

    ###################################
#    ### Adjointness     
#    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
#    yy = np.random.random_sample(np.shape(data)).astype(DTYPE)
#    a = np.vdot(xx.flatten(),self.operator_adjoint_2D(yy).flatten())
#    b = np.vdot(self.operator_forward_2D(xx).flatten(),yy.flatten())
#    test = np.abs(a-b)
#    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))
    x_old = x
    res = data - self.FT(x[self.unknowns_TGV:]*self.step_val[:,None,...]) + self.operator_forward_2D(x)
  
    x = self.tgv_solve_2D(x_old,res,iters)     
    
    self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_2D(x[:self.unknowns_TGV,...],islice)[:,None,...]*x[self.unknowns_TGV:,...]))**2
           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_1(x[:self.unknowns_TGV,...])-self.v))
           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_2(self.v))) 
           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2
           +self.irgn_par.omega/2*np.linalg.norm(gd.fgrad_1(x[-self.unknowns_H1:,...]))**2)    
    print("-"*80)
    print ("Function value after GN-Step: %f" %(self.fval/self.irgn_par.lambd))
    self.model.plot_unknowns(x,True) 
    return x
   
  def irgn_solve_3D(self,x, iters,data):
    

    ###################################
    ### Adjointness     
#    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
#    yy = np.random.random_sample(np.shape(data)).astype(DTYPE)
#    a = np.vdot(xx.flatten(),self.operator_adjoint_2D(yy).flatten())
#    b = np.vdot(self.operator_forward_2D(xx).flatten(),yy.flatten())
#    test = np.abs(a-b)
#    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))
    x_old = x
    a = self.FT(self.step_val)
    b = self.operator_forward_2D(x)
    res = data - a + b
#    print("Test the norm: %2.2E  a=%2.2E   b=%2.2E" %(np.linalg.norm(res.flatten()),np.linalg.norm(a.flatten()), np.linalg.norm(b.flatten())))
  
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
      gamma = self.irgn_par.gamma
      delta = self.irgn_par.delta

      self.result = np.zeros((self.irgn_par.max_GN_it,self.unknowns_TGV+self.unknowns_H1,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
      result = np.concatenate(((self.model.guess),np.ones_like(self.par.C)),0)
      for islice in range(self.par.NSlice):
        self.irgn_par.gamma = gamma
        self.irgn_par.delta = delta
        self.init_plan(islice)
        self.FT = self.nFT_2D
        self.FTH = self.nFTH_2D        
        
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
          self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_2D(result[:self.unknowns_TGV,islice,:,:],islice))
    
          scale = np.linalg.norm(np.abs(self.grad_x_2D[0,...]))/np.linalg.norm(np.abs(self.grad_x_2D[1,...]))
            
          for j in range(len(self.model.constraints)-1):
            self.model.constraints[j+1].update(scale)
          result[1,islice,:,:] = result[1,islice,:,:]*self.model.T1_sc        
          self.model.T1_sc = self.model.T1_sc*(scale)
          result[1,islice,:,:] = result[1,islice,:,:]/self.model.T1_sc         
          
          self.step_val = self.model.execute_forward_2D(result[:self.unknowns_TGV,islice,:,:],islice)
          self.grad_x_2D = np.zeros((self.unknowns,self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE)
          self.grad_x_2D[:self.unknowns_TGV,...] = result[self.unknowns_TGV:,islice,...]*self.model.execute_gradient_2D(result[:self.unknowns_TGV,islice,:,:],islice)[:,:,None,...]
          
          for j in range(self.NC):
            self.grad_x_2D[self.unknowns_TGV:,:,j,...] = np.repeat(self.step_val[None,...],self.par.NC,0)
            
          self.conj_grad_x_2D = np.conj(self.grad_x_2D)

          result[:,islice,:,:] = self.irgn_solve_2D(np.squeeze(result[:,islice,:,:]), iters, np.squeeze(self.data[:,:,islice,:]),islice)
          self.result[i,:,islice,:,:] = result[:,islice,:,:]

          
          iters = np.fmin(iters*2,self.irgn_par.max_iters)
          self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*self.irgn_par.gamma_dec,self.irgn_par.gamma_min)
          self.irgn_par.delta = np.minimum(self.irgn_par.delta*self.irgn_par.delta_inc,self.irgn_par.delta_max)
          self.irgn_par.omega = np.minimum(self.irgn_par.omega*self.irgn_par.omega_dec,self.irgn_par.omega_min)
          end = time.time()-start
#          print ("Function value after GN-Step %i: %f" %(i,self.fval))
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
        self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*self.irgn_par.gamma_dec,self.irgn_par.gamma_min)
        self.irgn_par.delta = np.minimum(self.irgn_par.delta*self.irgn_par.delta_inc,self.irgn_par.delta_max)
          
        end = time.time()-start
        print ("Function value after GN-Step %i: %f" %(i,self.fval))
        print("Elapsed time: %f seconds" %end)
               
      
  def operator_forward_2D(self, x):
    
    return self.FT(np.sum(x[:,None,None,...]*self.grad_x_2D,axis=0))

    
  def operator_adjoint_2D(self, x):
    
    return np.squeeze(np.sum(np.squeeze((self.FTH(x)))[None,...]*self.conj_grad_x_2D,axis=(1,2))) 

  
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
    result = np.zeros((self.NScan,self.NC,self.par.Nproj*self.par.N),dtype=DTYPE)
    plan = self._plan
    for scan in range(self.NScan):    
      for coil in range(self.NC):
        result[scan,coil,...] = plan[scan].forward(np.require(x[scan,coil,...]))  
    return np.reshape(result,[self.NScan,self.NC,self.par.Nproj,self.par.N])



  def nFTH_2D(self, x):
    result = np.zeros((self.NScan,self.NC,self.par.dimX,self.par.dimY),dtype=DTYPE)
    plan = self._plan
    x_wrk = np.require(np.reshape(x,(self.NScan,self.NC,self.Nproj*self.N)))
    for scan in range(self.NScan):
      for coil in range(self.NC):
            result[scan,coil,...] = plan[scan].adjoint(x_wrk[scan,coil,...][None,...])
      
    return result
  
  
  def  nFT_3D(self, x):

    nscan = self.NScan
    NC = self.NC   
    NSlice = self.NSlice
    result = np.zeros((nscan,NC,NSlice,self.Nproj*self.N),dtype=DTYPE)
    scan=0
    coil=0
    islice=0
    plan = self._plan    

    for scan in range(nscan):
        for islice in range(NSlice):
          result[scan,:,islice,:] = plan[scan][coil].forward(x[scan,islice,...])
      
    return np.reshape(result,[nscan,NC,NSlice,self.par.Nproj,self.par.N])



  def  nFTH_3D(self, x):
    nscan = self.NScan
    NC = self.NC
    NSlice = self.NSlice
    result = np.zeros((nscan,NSlice,self.par.dimX,self.par.dimY),dtype=DTYPE) 
    scan=0
    coil=0
    islice=0    
    plan = self._plan   
    x_wrk = np.reshape(x,(nscan,NC,NSlice,self.N*self.Nproj))
    for scan in range(nscan):
      for islice in range(NSlice):
        result[scan,islice,...] = plan[scan][coil].adjoint(np.require(x_wrk[scan,:,islice,...],DTYPE,'C'))
      
    return result

  def init_plan(self,islice=None):
    plan = []

    traj_x = np.real(np.asarray(self.traj))
    traj_y = np.imag(np.asarray(self.traj))
    
    config = {'osf' : 2,
              'sector_width' : 8,
              'kernel_width' : 3,
              'img_dim' : self.dimX}
    if not (islice==None):
      for i in range(self.NScan):
          points = (np.array([traj_x[i,:,:].flatten(),traj_y[i,:,:].flatten()]))      
          op = primaldualtoolbox.mri.MriRadialOperator(config)
          op.setTrajectory(points)
          op.setDcf(self.dcf_flat.astype(np.float32)[None,...])
          op.setCoilSens(np.ones((1,self.dimY,self.dimX),dtype=DTYPE,order='C'))
          plan.append(op)
      self._plan = plan
    else:
      for i in range(self.NScan):
        plan.append([])
        points = (np.array([traj_x[i,:,:].flatten(),traj_y[i,:,:].flatten()]))      
        for j in range(self.NSlice):
          op = primaldualtoolbox.mri.MriRadialOperator(config)
          op.setTrajectory(points)
          op.setDcf(self.dcf_flat.astype(np.float32)[None,...])
          op.setCoilSens(np.ones((1,self.dimY,self.dimX),dtype=DTYPE,order='C'))
          plan[i].append(op)
      self._plan = plan     


  def tgv_solve_2D(self, x, res, iters):
    alpha = self.irgn_par.gamma
    beta = self.irgn_par.gamma*2
#    
#    xx = np.zeros_like(x,dtype=DTYPE)
#    yy = np.zeros_like(x,dtype=DTYPE)
#    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
#    yy = self.operator_adjoint_2D(self.operator_forward_2D(xx));
#    for j in range(10):
#       if not np.isclose(np.linalg.norm(yy.flatten()),0):
#           xx = yy/np.linalg.norm(yy.flatten())
#       else:
#           xx = yy
#       yy = self.operator_adjoint_2D(self.operator_forward_2D(xx))
#       l1 = np.vdot(yy.flatten(),xx.flatten());
#    L = np.max(np.abs(l1)) ## Lipschitz constant estimate   
    L = (8**2+16**2)
    print('L: %f'%(L))

    
    tau = 1/np.sqrt(L)
    tau_new = 0
    
    xk = np.copy(x)
    x_new = np.zeros_like(x,dtype=DTYPE)
    
    r = self.r#np.zeros_like(res,dtype=DTYPE)
    z1 = self.z1#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2 = self.z2#np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
   
    v = self.v#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    
    r_new = np.zeros_like(res,dtype=DTYPE)
    z1_new = np.zeros(([self.unknowns_TGV,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2_new = np.zeros(([self.unknowns_TGV,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)

    z3_new = np.zeros(([self.unknowns_H1,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)    
    z3 = self.z3#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE) 
      
      
    v_new = np.zeros(([self.unknowns_TGV,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    

    Kyk1 = np.zeros_like(x,dtype=DTYPE)
    Kyk2 = np.zeros_like(z1,dtype=DTYPE)
    
    Ax = np.zeros_like(res,dtype=DTYPE)
    Ax_Axold = np.zeros_like(res,dtype=DTYPE)
    Axold = np.zeros_like(res,dtype=DTYPE)    
    tmp = np.zeros_like(res,dtype=DTYPE)    
    
    Kyk1_new = np.zeros_like(x,dtype=DTYPE)
    Kyk2_new = np.zeros_like(z1,dtype=DTYPE)
    
    
    delta = self.irgn_par.delta
    mu = 1/delta
    
    theta_line = 1.0

    
    beta_line = 400
    beta_new = 0
    
    mu_line = 0.5
    delta_line = 1
    scal = np.zeros((self.par.dimX,self.par.dimY),dtype=DTYPE)
    
    ynorm = 0.0
    lhs = 0.0

    primal = 0.0
    primal_new = 0
    dual = 0.0
    gap_min = 0.0
    gap = 0.0
    

    
    gradx = np.zeros_like(z1,dtype=DTYPE)
    gradx_xold = np.zeros_like(z1,dtype=DTYPE)
    
    v_old = np.zeros_like(v,dtype=DTYPE)
    symgrad_v = np.zeros_like(z2,dtype=DTYPE)
    symgrad_v_vold = np.zeros_like(z2,dtype=DTYPE)
    
    
    Axold = self.operator_forward_2D(x)
    
    if self.unknowns_H1 > 0:
      Kyk1 = self.operator_adjoint_2D(r) - np.concatenate((gd.bdiv_1(z1),(gd.bdiv_1(z3))),0)
    else:
      Kyk1 = self.operator_adjoint_2D(r) - (gd.bdiv_1(z1))
      
    Kyk2 = -z1 - gd.fdiv_2(z2)
    for i in range(iters):
        
      x_new = ((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta)
      
#      if self.unknowns_H1 > 0:
#        x_new[-self.unknowns_H1:,...] = (x_new[-self.unknowns_H1:,...]*(1+tau/delta)+tau*self.irgn_par.omega*self.par.fa)/(1+tau/delta+tau*self.irgn_par.omega)
      
      for j in range(len(self.model.constraints)):   
        x_new[j,...] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,x_new[j,...]))
        if self.model.constraints[j].real:
          x_new[j,...] = np.real(x_new[j,...])



      v_new = v-tau*Kyk2
      
      beta_new = beta_line*(1+mu*tau)
      
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
#      tau_new = tau*np.sqrt(beta_line/beta_new)      
      
#      tau_new = tau*np.sqrt((1+theta_line))     
      
      beta_line = beta_new
      
      gradx = gd.fgrad_1(x_new)
      gradx_xold = gradx - gd.fgrad_1(x)
      v_vold = v_new-v
      symgrad_v = gd.sym_bgrad_2(v_new)
      symgrad_v_vold = symgrad_v - gd.sym_bgrad_2(v)
      Ax = self.operator_forward_2D(x_new)
      Ax_Axold = Ax-Axold
    
      while True:
        
        theta_line = tau_new/tau
        
        z1_new = z1 + beta_line*tau_new*( gradx[:self.unknowns_TGV] + theta_line*gradx_xold[:self.unknowns_TGV]
                                          - v_new - theta_line*v_vold  )
        z1_new = z1_new/np.maximum(1,(np.sqrt(np.sum(z1_new**2,axis=(0,1)))/alpha))
     
        z2_new = z2 + beta_line*tau_new*( symgrad_v + theta_line*symgrad_v_vold )
        scal = np.sqrt( np.sum(z2_new[:,0,:,:]**2 + z2_new[:,1,:,:]**2 + 2*z2_new[:,2,:,:]**2,axis=0) )

        scal = np.maximum(1,scal/(beta))

        z2_new = z2_new/scal
        
        tmp = Ax+theta_line*Ax_Axold
        r_new = (( r  + beta_line*tau_new*(tmp) ) - beta_line*tau_new*res)/(1+beta_line*tau_new/self.irgn_par.lambd)    
        
        
        if self.unknowns_H1 > 0:
          z3_new = z3 + beta_line*tau_new*( gradx[-self.unknowns_H1:,...] + theta_line*gradx_xold[-self.unknowns_H1:,...])  
          z3_new = z3_new/(1+beta_line*tau_new/self.irgn_par.omega)
          Kyk1_new = self.operator_adjoint_2D(r_new) - np.concatenate((gd.bdiv_1(z1_new),(gd.bdiv_1(z3_new))),0)
          ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),(z1_new-z1).flatten(),(z2_new-z2).flatten(),(z3_new-z3).flatten()]))
        else:
          Kyk1_new = self.operator_adjoint_2D(r_new) - (gd.bdiv_1(z1_new))
          ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),(z1_new-z1).flatten(),(z2_new-z2).flatten()]))
        Kyk2_new = -z1_new -gd.fdiv_2(z2_new)

        
        lhs = np.sqrt(beta_line)*tau_new*np.linalg.norm(np.concatenate([(Kyk1_new-Kyk1).flatten(),(Kyk2_new-Kyk2).flatten()]))        
        if lhs <= ynorm*delta_line:
            break
        else:
          tau_new = tau_new*mu_line
             
      Kyk1 = np.copy(Kyk1_new)
      Kyk2 =  np.copy(Kyk2_new)
      Axold =np.copy(Ax)
      z1 = np.copy(z1_new)
      z2 = np.copy(z2_new)
      if self.unknowns_H1 > 0:
        z3 = np.copy(z3_new)
      r =  np.copy(r_new)
      tau =  np.copy(tau_new)
        
        
      if not np.mod(i,20):
          
        self.model.plot_unknowns(x_new,True)
        if self.unknowns_H1 > 0:
          primal_new= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*np.sum(np.abs((gradx[:self.unknowns_TGV]-v))) +
                   beta*np.sum(np.abs(symgrad_v)) + 1/(2*delta)*np.linalg.norm((x_new-xk).flatten())**2
                   +self.irgn_par.omega/2*np.linalg.norm(gradx[-self.unknowns_H1:,...])**2)
      
          dual = np.real(-delta/2*np.linalg.norm((-Kyk1_new).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1_new).flatten()) + np.sum(Kyk2_new) 
                  - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r.flatten())**2 - np.vdot(res.flatten(),r.flatten())
                  - 1/(2*self.irgn_par.omega)*np.linalg.norm(z3.flatten())**2)
        else:
          primal_new= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*np.sum(np.abs((gradx[:self.unknowns_TGV]-v))) +
                   beta*np.sum(np.abs(symgrad_v)) + 1/(2*delta)*np.linalg.norm((x_new-xk).flatten())**2)
      
          dual = np.real(-delta/2*np.linalg.norm((-Kyk1_new).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1_new).flatten()) + np.sum(Kyk2_new) 
                  - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r.flatten())**2 - np.vdot(res.flatten(),r.flatten()))
            
        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<self.irgn_par.lambd*self.irgn_par.tol:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,np.abs(primal-primal_new)/self.irgn_par.lambd))
          self.v = v_new
          self.r = r
          self.z1 = z1
          self.z2 = z2
          return x_new
        if (gap > gap_min*self.irgn_par.stag) and i>1:
          self.v = v
          self.r = r
          self.z1 = z1
          self.z2 = z2
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x
        if np.abs(gap - gap_min)<self.irgn_par.lambd*self.irgn_par.tol and i>1:
          self.v = v_new
          self.r = r
          self.z1 = z1
          self.z2 = z2
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,np.abs(gap - gap_min)/self.irgn_par.lambd))
          return x_new        
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal/self.irgn_par.lambd,dual/self.irgn_par.lambd,gap/self.irgn_par.lambd))
#        print("Norm of primal gradient: %.3e"%(np.linalg.norm(Kyk1)+np.linalg.norm(Kyk2)))
#        print("Norm of dual gradient: %.3e"%(np.linalg.norm(tmp)+np.linalg.norm(gradx[:self.unknowns_TGV] + theta_line*gradx_xold[:self.unknowns_TGV]
#                                          - v_new - theta_line*v_vold)+np.linalg.norm( symgrad_v + theta_line*symgrad_v_vold)))
        
      x = np.copy(x_new)
      v = np.copy(v_new)
#      for j in range(self.par.unknowns_TGV):
#        self.scale_2D[j,...] = np.linalg.norm(x[j,...])
    self.v = v
    self.r = r
    self.z1 = z1
    self.z2 = z2
    if self.unknowns_H1 > 0:
      self.z3 = z3
      
    return x
   
  def tikonov_solve(self,xk,data,iters):
    x_old = np.concatenate((np.real(xk),np.imag(xk)),axis=0)
    con_min = np.zeros_like(x_old)
    con_max = np.zeros_like(x_old)
    
    for j in range(len(self.model.constraints)):   
      con_min[j,...] = self.model.constraints[j].min*np.ones_like(con_min[j,...])
      con_max[j,...] = self.model.constraints[j].max*np.ones_like(con_max[j,...])
      con_min[j+self.par.unknowns,...] = self.model.constraints[j].min*np.ones_like(con_min[j+self.par.unknowns,...])
      con_max[j+self.par.unknowns,...] = self.model.constraints[j].max*np.ones_like(con_max[j+self.par.unknowns,...])  
      if self.model.constraints[j].real:
        con_min[j+self.par.unknowns,...] = np.zeros_like(con_min[j,...])
        con_max[j+self.par.unknowns,...] = np.zeros_like(con_max[j,...])    
    

    con = np.concatenate((con_min.flatten()[None,:],con_max.flatten()[None,:]),axis=0).T.tolist()

    optres = op.minimize(self.fun_val,x_old,args=(xk,data),method='L-BFGS-B',jac=self.lhs_2D,options={'maxiter':iters,'disp':10},bounds=con)
    x = np.squeeze(np.reshape(optres.x,(self.unknowns*2,self.NSlice,self.dimX,self.dimY)))

    self.fval = self.fun_val(optres.x,xk,data)
    x = x[:self.unknowns,...] + 1j*x[self.unknowns:,...]

 
    return x     
      


    
  def fun_val(self,x,xk,data):
    x = np.squeeze(np.reshape(x,(self.unknowns*2,self.NSlice,self.dimX,self.dimY)))
    x = x[:self.unknowns,...] + 1j*x[self.unknowns:,...]
    x = x.astype(DTYPE)
    return (self.irgn_par.lambd/2*np.linalg.norm(self.operator_forward_2D(x)-data)**2+
            self.irgn_par.gamma/2*np.linalg.norm((x))**2 + 1/(2*self.irgn_par.delta)*np.linalg.norm(x-xk)**2).flatten()
    
  def lhs_2D(self,x,xk,data):
    x = np.squeeze(np.reshape(x,(self.unknowns*2,self.NSlice,self.dimX,self.dimY)))
    x = x[:self.unknowns,...] + 1j*x[self.unknowns:,...]
    tmp = (self.irgn_par.lambd*self.operator_adjoint_2D(self.operator_forward_2D(x.astype(DTYPE)))
           +self.irgn_par.gamma*x+1/self.irgn_par.delta*((x)) - 
            self.irgn_par.lambd*self.operator_adjoint_2D(data)-1/self.irgn_par.delta*xk).flatten()
    return np.concatenate((np.real(tmp),np.imag(tmp)),axis=0)
      
  

    

    
    
        