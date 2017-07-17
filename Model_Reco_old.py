

import numpy as np
import time
import decimal
import gradients_divergences_old as gd
import matplotlib.pyplot as plt

#import ipyparallel as ipp
#
#c = ipp.Client()
#lview = c.load_balanced_view()

plt.ion()

DTYPE = np.complex128


class Model_Reco: 
  def __init__(self,par):
    self.par = par
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
    xx = np.random.random_sample(np.shape(x)).astype('complex128')
    yy = np.random.random_sample(np.shape(data)).astype('complex128')
    a = np.vdot(xx.flatten(),self.operator_adjoint_2D(yy).flatten())
    b = np.vdot(self.operator_forward_2D(xx).flatten(),yy.flatten())
    test = np.abs(a-b)
    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,decimal.Decimal(test)))
    x_old = x
    a = self.FT(self.step_val[:,None,:,:]*self.Coils)
    b = self.operator_forward_2D(x)
    res = data - a + b
#    print("Test the norm: %2.2E  a=%2.2E   b=%2.2E" %(np.linalg.norm(res.flatten()),np.linalg.norm(a.flatten()), np.linalg.norm(b.flatten())))
  
#    x = self.pdr_tgv_solve_2D(x,res,iters)
#    x = optimizer.fmin_cg(self.lb_fun,x.flatten(),self.lb_grad,args=(res,x),maxiter=iters)
#    x = np.reshape(x,(self.unknowns,self.par.dimY,self.par.dimX))
    x = self.tgv_solve_2D(x,res,iters)      
    fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.step_val[:,None,:,:]*self.Coils))**2
           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_1(x)-self.v))
           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_2(self.v))) 
           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)    
    print ("Function value after GN-Step: %f" %fval)

    return x
  
  def irgn_solve_3D(self,x, iters,data):
    

    ###################################
    ### Adjointness     
    xx = np.random.random_sample(np.shape(x)).astype('complex128')
    yy = np.random.random_sample(np.shape(data)).astype('complex128')
    a = np.vdot(xx.flatten(),self.operator_adjoint_3D(yy).flatten())
    b = np.vdot(self.operator_forward_3D(xx).flatten(),yy.flatten())
    test = np.abs(a-b)
    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,decimal.Decimal(test)))
    x_old = x
    a = self.FT(self.step_val[:,None,:,:]*self.Coils)
    b = self.operator_forward_3D(x)
    res = data - a + b
    print("Test the norm: %2.2E  a=%2.2E   b=%2.2E" %(np.linalg.norm(res.flatten()),np.linalg.norm(a.flatten()), np.linalg.norm(b.flatten())))
  
    x = self.tgv_solve_3D(x,res,iters)
      
    fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.step_val[:,None,:,:]*self.Coils))**2
           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_3(x)-self.v))
           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_3(self.v))) 
           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)    
    print ("Function value after GN-Step: %f" %fval)

    return x
        

    
  def execute_2D(self):
     
      self.FT = self.nFT_2D
      self.FTH = self.nFTH_2D
      iters = self.irgn_par.start_iters
      self.v = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype='complex128')
      
      self.result = np.zeros((self.irgn_par.max_GN_it,self.unknowns,self.par.NSlice,self.par.dimY,self.par.dimX),dtype='complex128')
      result = np.copy(self.model.guess)
      for islice in range(self.par.NSlice):
        self.Coils = np.squeeze(self.par.C[:,islice,:,:])        
        for i in range(self.irgn_par.max_GN_it):
          start = time.time()       
          self.step_val = self.model.execute_forward_2D(result[:,islice,:,:],islice)
          self.grad_x = self.model.execute_gradient_2D(result[:,islice,:,:],islice)
          self.conj_grad_x = np.conj(self.grad_x)
          
          
          result[:,islice,:,:] = self.irgn_solve_2D(result[:,islice,:,:], iters, self.data[:,:,islice,:])
          self.result[i,:,islice,:,:] = result[:,islice,:,:]
          
          iters = np.fmin(iters*2,self.irgn_par.max_iters)
          self.irgn_par.gamma = self.irgn_par.gamma*0.7
          self.irgn_par.delta = self.irgn_par.delta*2
          
          end = time.time()-start
          print("Elapsed time: %f seconds" %end)
            
        
      
      
     
#  def execute_3D(self):
#     
#      self.FT = self.nFT_3D
#      self.FTH = self.nFTH_3D
#      iters = self.irgn_par.start_iters
#
#      
#      
#      result = np.copy(self.model.guess)
#      self.Coils = np.squeeze(self.par.C)        
#      for i in range(self.irgn_par.max_GN_it):
#        start = time.time()       
#        self.step_val = self.model.execute_forward_3D(result)
#        self.grad_x = self.model.execute_gradient_3D(result)
#        self.conj_grad_x = np.conj(self.grad_x)
#          
#          
#        result = self.irgn_solve_3D(result, iters, self.data)
#          
#          
#        iters = np.fmin(iters*2,self.irgn_par.max_iters)
#        self.irgn_par.gamma = self.irgn_par.gamma*0.8
#        self.irgn_par.delta = self.irgn_par.delta*2
#          
#        end = time.time()-start
#        print("Elapsed time: %f seconds" %end)
#            
#        
#      self.result = result
               
      
  def operator_forward_2D(self, x):
    
    tmp = np.zeros_like(self.grad_x)
      
    for i in range(self.unknowns):
      tmp[i,:,:,:] = x[i,:,:]*self.grad_x[i,:,:,:]
      
    return self.FT(np.sum(tmp,axis=0)[:,None,:,:]*self.Coils)
#      
#      tmp1 = x[0,:,:]*self.grad_x[0,:,:,:]
#      tmp2 = x[1,:,:]*self.grad_x[1,:,:,:]
      
#      tmp1 = np.expand_dims(np.expand_dims(tmp1,axis=1), axis=1)
#      tmp2 = np.expand_dims(np.expand_dims(tmp2,axis=1), axis=1)
      
#      tmp1[~np.isfinite(tmp1)] = 1e-20
#      tmp2[~np.isfinite(tmp2)] = 1e-20
#      
#      return self.FT((tmp1[:,None,:,:]+tmp2[:,None,:,:])*self.Coils)

    
  def operator_adjoint_2D(self, x):
    
#    x[~np.isfinite(x)] = 1e-20
#    fdx = 
      
#    dx = 
      
#     = np.zeros((self.unknowns,self.par.dimX,self.par.dimY),dtype=DTYPE)
#    for i in range(self.unknowns):   
#      dx[i,:,:] = np.sum(fdx,axis=0)
#      dx[0,:,:] = np.sum(self.conj_grad_x[0,:,:,:]*fdx,axis=0)
#      dx[1,:,:] = np.sum(self.conj_grad_x[1,:,:,:]*fdx,axis=0)
      
    return np.squeeze(np.sum(np.squeeze(np.sum(self.FTH(x)*np.conj(self.Coils),axis=1)*self.conj_grad_x),axis=1))
  
#  def operator_forward_3D(self, x):
#    
#    tmp = np.zeros_like(self.grad_x)
#      
#    for i in range(self.unknowns):
#      tmp[i,:,:,:,:] = x[i,:,:,:]*self.grad_x[i,:,:,:,:]
#      
#    return self.FT(np.sum(tmp,axis=0)[:,None,:,:,:]*self.Coils)
##      
##      tmp1 = x[0,:,:]*self.grad_x[0,:,:,:]
##      tmp2 = x[1,:,:]*self.grad_x[1,:,:,:]
#      
##      tmp1 = np.expand_dims(np.expand_dims(tmp1,axis=1), axis=1)
##      tmp2 = np.expand_dims(np.expand_dims(tmp2,axis=1), axis=1)
#      
##      tmp1[~np.isfinite(tmp1)] = 1e-20
##      tmp2[~np.isfinite(tmp2)] = 1e-20
##      
##      return self.FT((tmp1[:,None,:,:]+tmp2[:,None,:,:])*self.Coils)
#
#    
#  def operator_adjoint_3D(self, x):
#    
#    x[~np.isfinite(x)] = 1e-20
#    fdx = self.FTH(x)
#      
#    fdx = np.squeeze(np.sum(fdx*np.conj(self.Coils),axis=1))
#      
#    dx = np.zeros((self.unknowns,self.par.NSlice,self.par.dimX,self.par.dimY),dtype=DTYPE)
#    for i in range(self.unknowns):   
#      dx[i,:,:,:] = np.sum(self.conj_grad_x[i,:,:,:,:]*fdx,axis=0)
##      dx[0,:,:] = np.sum(self.conj_grad_x[0,:,:,:]*fdx,axis=0)
##      dx[1,:,:] = np.sum(self.conj_grad_x[1,:,:,:]*fdx,axis=0)
#      
#    return dx    
    
  def tgv_solve_2D(self, x, res, iters):
    alpha = self.irgn_par.gamma
    beta = self.irgn_par.gamma*2
    
    
    tau = 1/np.sqrt(16**2+8**2)
    xk = x
    x_new = np.zeros_like(x,dtype=DTYPE)
    
    r = np.zeros_like(res,dtype=DTYPE)
    z1 = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2 = np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    v = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    
    r_new = np.zeros_like(res,dtype=DTYPE)
    z1_new = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2_new = np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    v_new = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    

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
    beta_line = 1.0
    mu_line = 0.8
#    delta_line = 0.8
    scal = np.zeros((self.par.dimX,self.par.dimY),dtype=DTYPE)
    
    ynorm = 0.0
    lhs = 0.0

    primal = 0.0
    dual = 0.0
    gap = 0.0

    
    gradx = np.zeros_like(z1,dtype=DTYPE)
    gradx_xold = np.zeros_like(z1,dtype=DTYPE)
    
    v_vold = np.zeros_like(v,dtype=DTYPE)
    symgrad_v = np.zeros_like(z2,dtype=DTYPE)
    symgrad_v_vold = np.zeros_like(z2,dtype=DTYPE)
    
    Axold = self.operator_forward_2D(x)
    Kyk1 = self.operator_adjoint_2D(r) - gd.bdiv_1(z1)
    Kyk2 = -z1 - gd.fdiv_2(z2)
    
    
    for i in range(iters):
      np.maximum(0,((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta),x_new)
      
      
      np.maximum(0,np.minimum(300/self.model.M0_sc,x_new[0,:,:],x_new[0,:,:]),x_new[0,:,:])
      np.abs(np.maximum(50/self.model.T1_sc,np.minimum(5000/self.model.T1_sc,x_new[1,:,:],x_new[1,:,:]),x_new[1,:,:]),x_new[1,:,:])

#      np.abs(x_new[1,:,:],x_new[1,:,:])
      
      
      np.subtract(v,tau*Kyk2,v_new)
      
      beta_new = beta_line*(1+mu*tau)
      
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
#      tau_new = tau*np.sqrt((1+theta_line))     
      
      beta_line = beta_new
      
      gradx = gd.fgrad_1(x_new)
      np.subtract(gradx , gd.fgrad_1(x),gradx_xold)
      np.subtract(v_new,v,v_vold)
      symgrad_v = gd.sym_bgrad_2(v_new)
      np.subtract(symgrad_v , gd.sym_bgrad_2(v),symgrad_v_vold)
      Ax = self.operator_forward_2D(x_new)
      np.subtract(Ax,Axold,Ax_Axold)
    
      while True:
        
        theta_line = tau_new/tau
        
        np.add(z1 , beta_line*tau_new*( gradx + theta_line*gradx_xold - v_new - theta_line*v_vold  ),z1_new)
        np.divide(z1_new,np.maximum(1,(np.sqrt(np.sum(z1_new**2,axis=(0,1)))/alpha)),z1_new)
     
        np.add(z2, beta_line*tau_new*( symgrad_v + theta_line*symgrad_v_vold ))
        np.sqrt( np.sum(z2_new[:,0,:,:]**2 + z2_new[:,1,:,:]**2 + 2*z2_new[:,2,:,:]**2,axis=0) ,scal)
        np.maximum(1,scal/(beta),scal)
        np.divide(z2_new,scal,z2_new)
        
        
        
        np.add(Ax,theta_line*Ax_Axold,tmp)
        np.divide((( r  + beta_line*tau_new*(tmp) ) - beta_line*tau_new*res),(1+beta_line*tau_new/self.irgn_par.lambd),r_new)
        
        np.subtract(self.operator_adjoint_2D(r_new),gd.bdiv_1(z1_new),Kyk1_new)
        np.subtract(-z1_new ,gd.fdiv_2(z2_new),Kyk2_new)

        
        ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),(z1_new-z1).flatten(),(z2_new-z2).flatten()]))
        lhs = np.sqrt(beta_line)*tau_new*np.linalg.norm(np.concatenate([(Kyk1_new-Kyk1).flatten(),(Kyk2_new-Kyk2).flatten()]))        
        if lhs <= ynorm:
            break
        else:
            tau_new = tau_new*mu_line
            
      Kyk1 = (Kyk1_new)
      Kyk2 =  (Kyk2_new)
      Axold =(Ax)
      z1 = (z1_new)
      z2 = (z2_new)
      r =  (r_new)
      tau =  (tau_new)
        
  
      x = x_new
      v = v_new
        
      if not np.mod(i,20):
        plt.figure(1)
        plt.imshow(np.transpose(np.abs(x[0,:,:]*self.model.M0_sc)))
        plt.pause(0.05)
        plt.figure(2)
#        plt.imshow(np.transpose(np.abs(-self.par.TR/np.log(x[1,:,:]))),vmin=0,vmax=3000)
        plt.imshow(np.transpose(np.abs(x[1,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
        plt.pause(0.05)
        primal= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*np.sum(np.abs((gradx-v))) +
                 beta*np.sum(np.abs(symgrad_v)) + 1/(2*delta)*np.linalg.norm((x-xk).flatten())**2)
    
        dual = np.real(-delta/2*np.linalg.norm((-Kyk1_new).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1_new).flatten()) + np.sum(Kyk2_new) 
                - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r.flatten())**2 - np.vdot(res.flatten(),r.flatten()))
        gap = np.abs(primal - dual)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal,dual,gap))
        
        
    self.v = v
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
#    cdef int scan=0
#    cdef int coil=0
    for scan in range(nscan):
      for coil in range(NC):
          self.nfftplan[scan][coil].f_hat = x[scan,coil,:,:]/np.sqrt(self.par.dimX*self.par.dimY)
          result[scan,coil,:] = self.nfftplan[scan][coil].trafo()*self.dcf_flat
      
    return np.reshape(result,(nscan,NC,self.par.Nproj,self.par.N))



  def nFTH_2D(self, x):
    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]     
    result = np.zeros((nscan,NC,self.par.dimX,self.par.dimY),dtype='complex128')
    for scan in range(nscan):
        for coil in range(NC):  
            self.nfftplan[scan][coil].f = x[scan,coil,:]*self.dcf
            result[scan,coil,:,:] = self.nfftplan[scan][coil].adjoint()
      
    return result/np.sqrt(self.par.dimX*self.par.dimY)
      
  
  
  def nFT_3D(self, x):

    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]   
    NSlice = self.par.NSlice
    dimX = self.par.dimX
    dimY = self.par.dimY
    result = np.zeros((nscan,NC,NSlice,self.par.Nproj*self.par.N),dtype=DTYPE)
    scan=0
    coil=0
    islice=0
    for scan in range(nscan):
      for coil in range(NC):
        for islice in range(NSlice):
          self.nfftplan[scan][coil].f_hat = x[scan,coil,islice,:,:]/np.sqrt(dimX*dimY)
          result[scan,coil,islice,:] = self.nfftplan[scan][coil].trafo()*self.dcf_flat
      
    return result



  def nFTH_3D(self, x):
    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]  
    NSlice = self.par.NSlice
    dimX = self.par.dimX
    dimY = self.par.dimY
    result = np.zeros((nscan,NC,NSlice,dimX,dimY),dtype='complex128')
    scan=0
    coil=0
    islice=0    
    for scan in range(nscan):
        for coil in range(NC): 
          for islice in range(NSlice):
            self.nfftplan[scan][coil].f = x[scan,coil,islice,:]*self.dcf
            result[scan,coil,islice,:,:] = self.nfftplan[scan][coil].adjoint()
      
    return result/np.sqrt(dimX*dimY)  






  def pdr_tgv_solve_2D(self, x, res, iters):
    
    

    
    alpha = self.irgn_par.gamma
    beta = self.irgn_par.gamma*2
    
    delta = self.irgn_par.delta
    gamma1 = 1/delta    
    

    
    sigma0 = 1   
    tau0 = 1
    
    L = 0
    L1 = 0
    L2 = 0
    
    ##estimate operator norm using power iteration
    xx = np.random.random_sample(np.shape(x)).astype('complex128')
    yy = 1+sigma0*tau0*self.operator_adjoint_2D(self.operator_forward_2D(xx));
    for i in range(10):
       if not np.isclose(np.linalg.norm(yy.flatten()),0):
           xx = yy/np.linalg.norm(yy.flatten())
       else:
           xx = yy
       yy = 1+sigma0*tau0*self.operator_adjoint_2D(self.operator_forward_2D(xx))
       l1 = np.vdot(yy.flatten(),xx.flatten());
    L = np.max(np.abs(l1))**2 ## Lipschitz constant estimate   
#    L1 = np.max(np.abs(self.grad_x[0,:,None,:,:]*self.Coils
#                                   *np.conj(self.grad_x[0,:,None,:,:])*np.conj(self.Coils)))
#    L2 = np.max(np.abs(self.grad_x[1,:,None,:,:]*self.Coils
#                                   *np.conj(self.grad_x[1,:,None,:,:])*np.conj(self.Coils)))
#
#    L = np.max((L1,L2))*self.unknowns*self.par.NScan*self.par.NC*sigma0*tau0+1
    L =1+sigma0*tau0*(L**2+8+16) + (4*sigma0*tau0)**2/(1+4*sigma0*tau0)
    print("Operatornorm estimate L1: %f ---- L2: %f -----  L: %f "%(L1,L2,L))    
    gamma = 2*gamma1/L   
    
    theta = 1/np.sqrt(1+sigma0*gamma)

    
    sigma = sigma0
    tau = tau0 

    
    xk = x
    xhat = (x)    
    v = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)    
    vhat = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)   
    
    r = np.zeros_like(res,dtype=DTYPE)
    z1 = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2 = np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)

    rhat = np.zeros_like(res,dtype=DTYPE)
    z1hat = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2hat = np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    

    Kd1 = np.zeros_like(res,dtype=DTYPE)
    Kd2 = np.zeros_like(z1,dtype=DTYPE)
    Kd3 = np.zeros_like(z2,dtype=DTYPE)
    
    
    b1 = np.zeros_like(x,dtype=DTYPE)    
    b2 = np.zeros_like(v,dtype=DTYPE)
    
    d1 = np.zeros_like(x,dtype=DTYPE)      
    d2 = np.zeros_like(v,dtype=DTYPE)
                   
    T1 = np.zeros_like(d1,dtype=DTYPE)    
    T2 = np.zeros_like(d2,dtype=DTYPE)                      

    scal = np.zeros((self.par.dimX,self.par.dimY),dtype=DTYPE)


    primal = 0
    dual = 0
    gap = 0




    for i in range(iters):
      
      #### Prox (id+sigma F) of x_hat
      np.maximum(0,(xhat+(sigma*gamma1)*xk)/(1+sigma*gamma1),x)
      np.minimum(300/self.model.M0_sc,x[0,:,:],x[0,:,:])
      np.abs(np.maximum(50/self.model.T1_sc,np.minimum(5000/self.model.T1_sc,x[1,:,:],x[1,:,:]),x[1,:,:]),x[1,:,:])
      v = vhat
      
      
      #### Prox (id+tau G) of y_hat
      r = ( rhat - tau*res)/(1+tau/self.irgn_par.lambd)        
      z1 = z1hat/np.maximum(1,(np.sqrt(np.sum(z1hat**2,axis=(0,1)))/alpha))
      scal = np.sqrt( np.sum(z2hat[:,0,:,:]**2 + z2hat[:,1,:,:]**2 + 2*z2hat[:,2,:,:]**2,axis=0) )
      np.maximum(1,scal/(beta),scal)
      z2 = z2hat/scal      
  
      
      
      #### update b
      #### Accelerated Version
      b1 = ((1+theta)*x-theta*xhat) - sigma*(self.operator_adjoint_2D((1+theta)*r-rhat) - gd.bdiv_1((1+theta)*z1-z1hat))
      b2 = ((1+theta)*v-theta*vhat) - sigma*(-((1+theta)*z1-z1hat) - gd.fdiv_2((1+theta)*z2-z2hat))
      ### normal Version
#      b1 = (2*x-theta*xhat) - sigma*(self.operator_adjoint_2D(2*r-rhat) - gd.bdiv_1(2*z1-z1hat))
#      b2 = (2*v-theta*vhat) - sigma*(-(2*z1-z1hat) - gd.fdiv_2(2*z2-z2hat))          
          
      #### update d
      Kd1 = (self.operator_forward_2D(d1))
      Kd2 = (gd.fgrad_1(d1)-d2)
      Kd3 = (gd.sym_bgrad_2(d2))
      
      T1 = d1 + sigma0*tau0*(self.operator_adjoint_2D(Kd1) - gd.bdiv_1(Kd2))
      T2 = d2 + sigma0*tau0*(-Kd2 - gd.fdiv_2(Kd3))
      
      d1 = d1 + 1/L*(b1-T1)
      d2 = d2 + 1/L*(b2-T2)
      
      #### Accelerated Version
      xhat = theta*(xhat-x) + d1
      vhat = theta*(vhat-v) + d2
      #### normal Version
#      xhat = (xhat-x) + d1
#      vhat = (vhat-v) + d2      
      #### Accelerated Version      
      rhat = r + 1/theta*tau*(self.operator_forward_2D(d1))
      z1hat = z1 + 1/theta*tau*(gd.fgrad_1(d1)-d2)
      z2hat = z2 + 1/theta*tau*(gd.sym_bgrad_2(d2))
      #### normal Version
#      rhat = r + tau*(self.operator_forward_2D(d1))
#      z1hat = z1 + tau*(gd.fgrad_1(d1)-d2)
#      z2hat = z2 + tau*(gd.sym_bgrad_2(d2))
      #### Accelerated Version      
      sigma = theta*sigma
      tau = 1/theta*tau
      theta = 1/np.sqrt(1+sigma*gamma)
      
        
      if not np.mod(i,1):
        plt.figure(1)
        plt.imshow(np.transpose(np.abs(x[0,:,:]*self.model.M0_sc)))
        plt.pause(0.05)
        plt.figure(2)
        plt.imshow(np.transpose(np.abs(x[1,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
        plt.pause(0.05)
        primal= np.real(self.irgn_par.lambd/2*np.linalg.norm((self.operator_forward_2D(x)-res).flatten())**2+alpha*np.sum(np.abs((gd.fgrad_1(x)-v))) +
                 beta*np.sum(np.abs(gd.sym_bgrad_2(v))) + 1/(2*delta)*np.linalg.norm((x-xk).flatten())**2)
        

        dual = np.real(-delta/2*np.linalg.norm(-(self.operator_adjoint_2D(r)-gd.bdiv_1(z1)).flatten())**2 
             - np.vdot(xk.flatten(),-(self.operator_adjoint_2D(r)-gd.bdiv_1(z1)).flatten()) + np.sum(-z1 -gd.fdiv_2(z2)) 
             - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r.flatten())**2 - np.vdot(res.flatten(),r.flatten()))
        gap = np.abs(primal - dual)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal,dual,gap))
        
        
    self.v = v
    return x  
  
  def lb_fun(self,x,res,xk):
    x = np.reshape(x,[self.unknowns,self.par.dimY,self.par.dimX])
    f = (1/2*np.linalg.norm(self.operator_forward_2D(x)-res)**2 + 
        self.irgn_par.gamma*np.log(1+np.sum(np.linalg.norm(gd.fgrad_1(x),axis=0)**2)) 
        + 1/(2*self.irgn_par.delta)*np.linalg.norm(x-xk)**2)
    return f.flatten()
  def lb_grad(self,x,res,xk):
    x = np.reshape(x,[self.unknowns,self.par.dimY,self.par.dimX])
    f =(self.operator_adjoint_2D(self.operator_forward_2D(x)-res) + 
        self.irgn_par.gamma*(-np.sqrt(np.sum(gd.bdiv_1(gd.fgrad_1(x))**2,0)))/(1+np.sum(np.linalg.norm(gd.fgrad_1(x),axis=0)**2)) 
        + 1/(self.irgn_par.delta)*(x-xk))
    return f.flatten()
  
#  def ipalm_solve(self,x,res,iters):
#    
#    alpha = self.irgn_par.gamma
#    beta = self.irgn_par.gamma*2
#    
#    delta = self.irgn_par.gamma
#    
#  
#    L1 = np.max(np.abs(self.grad_x[0,:,None,:,:]*self.Coils
#                                   *np.conj(self.grad_x[0,:,None,:,:])*np.conj(self.Coils)))*self.unknowns*self.par.NScan*self.par.NC
#    L2 = np.max(np.abs(self.grad_x[1,:,None,:,:]*self.Coils
#                                   *np.conj(self.grad_x[1,:,None,:,:])*np.conj(self.Coils)))*self.unknowns*self.par.NScan*self.par.NC
#    
#    
#    eps = 0.5
#    alpha_bar = 0.5*(1-eps)
#    beta_bar = 1
#
#    step_sig1 = (alpha_bar+beta_bar)/(1-eps-2*alpha_bar)*L1
#    step_sig2 = (alpha_bar+beta_bar)/(1-eps-2*alpha_bar)*L2
#                       
#    
#    tau1 = ((1+eps)*step_sig1+(1+beta_bar)*L1)/(1-alpha_bar)
#    tau2 = ((1+eps)*step_sig2+(1+beta_bar)*L2)/(1-alpha_bar)
#    
#    x1 = x[0,:,:]
#    x1k = x1
#    x1old = x1
#    x2 = x[1,:,:]
#    x2k = x2
#    x2old = x2
#    
#    alpha1 = 1
#    beta1 = 1
#    
#    alpha2 = 1
#    beta2 = 1
#    
#    
#    for i in range(iters):
#      ##### M0 update
#      y1 = x1 + alpha1*(x1-x1old)
#      z1 = x1 + beta1*(x1-x1old)
#      
#      tmp_inner = y1-1/tau1*(self.operator_adjoint_2D(self.operator_forward_2D(np.concatenate(z1,x2)))-res)[0,:,:]
#      
#      x1old = x1
#      x1 = np.maximum(0,(np.abs(tmp_inner+tau1/delta*x1k)-alpha*tau1)/(1+tau1/delta))*np.sign(tmp_inner+tau1/delta*x1k)
#      
#      #### T1 update
#      y2 = x2 + alpha2*(x2-x2old)
#      z2 = x2 + beta2*(x2-x2old)
#      
#      tmp_inner = y2-1/tau2*(self.operator_adjoint_2D(self.operator_forward_2D(np.concatenate(x1,z2)))-res)[1,:,:]
#      
#      x2old = x2
#      x2 = np.maximum(0,(np.abs(tmp_inner+tau2/delta*x2k)-alpha*tau2)/(1+tau2/delta))*np.sign(tmp_inner+tau2/delta*x2k)
      
      
      
      
        