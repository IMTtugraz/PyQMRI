#FLASH_SSFP_Model_Reco
import numpy as np
import time
import matplotlib.pyplot as plt
from createInitGuess_FLASH import createInitGuess_FLASH
from compute_mask import compute_mask
import VFA_model
import decimal
import gradients_divergences as gd
import numexpr as ne
import time
import scipy.io as sio

class VFA_Model_Reco:
  def __init__(self):
    self.par = []
    self.data= []
    self.result = []
    self.guess = []
    self.images = []
    self.fft_forward = []
    self.fft_back = []
    self.irgn_par = []
    self.model = []
    self.step_val = []
    self.grad_x = []
    self.conj_grad_x = []
    self.FT_scale = []
    print("Please Set Parameters, Data and Initial images")

  def __init_guess(self):

# =============================================
## initial guess for the parameter maps
# =============================================
### create an initial guess by standard pixel-based fitting of "composite images"
    th = time.clock()

    [M0_guess, T1_guess, mask_guess] = createInitGuess_FLASH(self.images,self.par.fa,self.par.TR,self.par.fa_corr)

    T1_guess[np.isnan(T1_guess)] = np.spacing(1)
    T1_guess[np.isinf(T1_guess)] = np.spacing(1)
    T1_guess[T1_guess<0] = 0 
    T1_guess[T1_guess>5000] = 5000
    T1_guess = np.abs(T1_guess)

    M0_guess[M0_guess<0] = 0 
    M0_guess[np.isnan(M0_guess)] = np.spacing(1)
    M0_guess[np.isinf(M0_guess)] = np.spacing(1)   
    

#    hist =  np.histogram(np.abs(M0_guess),int(1e2))
#    aa = np.array(hist[0], dtype=np.float64)
#    #bb = hist[1] #hist0[1][:-1] + np.diff(hist0[1])/2
#    bb = np.array(hist[1][:-1] + np.diff(hist[1])/2, dtype=np.float64)
#   
#    idx = np.array(aa > 0.01*aa[0],dtype=np.float64)
#
#    M0_guess[M0_guess > bb[int(np.sum(idx))]] = bb[int(np.sum(idx))] #passst
    #print(M0_guess)
    M0_guess = np.squeeze(M0_guess)

    mask_guess = compute_mask(M0_guess,False)

    self.par.mask = mask_guess#par.mask[:,63] is different
    
    self.par.T1_sc = np.max(T1_guess)
    self.par.M0_sc = np.max(np.abs(M0_guess))
    
    #print(mask_guess)
    print('T1 scale: ',self.par.T1_sc,
                              '/ M0_scale: ',self.par.M0_sc)
    #print(M0_guess[39,11]) M0 guess is gleich

    M0_guess = M0_guess / self.par.M0_sc
    T1_guess = T1_guess / self.par.T1_sc

    T1_guess[np.isnan(T1_guess)] = 0;
    M0_guess[np.isnan(M0_guess)] = 0;
    
    self.par.T1_guess = T1_guess * self.par.T1_sc
    self.par.M0_guess = M0_guess * self.par.M0_sc
#        
    print( 'done in', time.clock() - th)

    result = np.array([(M0_guess*np.exp(1j*np.angle(self.par.phase_map)))*mask_guess,np.exp(-self.par.TR/(T1_guess*self.par.T1_sc))*mask_guess])
    self.guess = result
    sio.savemat("Guess.mat",{"Guess":result})
    
    
    
  def init_fwd_model(self):
#    self.par.M0_sc =1
#    self.par.T2_sc = 1200
    model = VFA_model.VFA_Model()
    model.M0_sc = self.par.M0_sc
    model.T1_sc = self.par.T1_sc
    model.sin_phi = np.sin(self.par.phi_corr)
    model.cos_phi = np.cos(self.par.phi_corr)
#    model.TE = self.par.TE
#    self.guess = np.array([0.01*np.ones((self.par.dimX,self.par.dimY),dtype='complex128'),0.2*np.ones((self.par.dimX,self.par.dimY),dtype='complex128')])
    self.model = model
    
  def irgn_solve(self,x,iters):
    
    self.step_val = self.model.execute_forward_2D(x)
    self.grad_x = self.model.execute_gradient_2D(x)
    self.conj_grad_x = np.conj(self.grad_x)
    


    
    ###################################
    ### Adjointness     
    xx = np.random.randn(2,self.par.dimX,self.par.dimY)
    yy = np.random.randn(self.par.NScan,self.par.NC,self.par.NSlice,self.par.dimX,self.par.dimY)
    a = np.vdot(xx.flatten(),self.operator_adjoint(yy).flatten())
    b = np.vdot(self.operator_forward(xx).flatten(),yy.flatten())
    test = np.abs(a-b)
    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,decimal.Decimal(test)))
    x_old = x
    ax = self.FT(self.step_val[:,None,None,:,:]*self.par.C)
    bx = self.operator_forward(x)
    res = self.data - ax + bx
    print("Test the norm: %2.2E  a=%2.2E   b=%2.2E" %(np.linalg.norm(res.flatten()),np.linalg.norm(ax.flatten()), np.linalg.norm(bx.flatten())))

    x = self.tgv_solve(x,res,iters)
    
    fval= (self.irgn_par.lambd/2*np.linalg.norm(self.data - self.FT(np.expand_dims(np.expand_dims(self.step_val,axis=1),axis=1)*self.par.C))**2
          +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_1(x)-self.v))
          +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_2(self.v))) 
          +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)    
    print ("Function value after GN-Step: %f" %fval)
      
    
    
    return x
     

    
    
  def execute(self):
      self.__init_guess()
      self.init_fwd_model()
      iters = self.irgn_par.start_iters
      result = np.copy(self.guess)
      for i in range(self.irgn_par.max_GN_it):    
        start = time.time()
        result = self.irgn_solve(result, iters)
#        name = ["mdl_iter"+str(i)+".mat"]
#        sio.savemat(str(name),{'Model':self.step_val})
#        name = ["grd_iter"+str(i)+".mat"]
#        sio.savemat(str(name),{'Grad':self.grad_x})
#        name = ["res"+str(i)+".mat"]
#        sio.savemat(str(name),{'Result':result})
        iters = np.fmin(iters*2,self.irgn_par.max_iters)
        self.irgn_par.gamma = self.irgn_par.gamma/2
        self.irgn_par.delta = self.irgn_par.delta*2
        end = time.time()-start
        print("Elapsed time: %f seconds" %end)
            
        
      self.result = result
      
          
      
  def operator_forward(self,x):
    
      tmp1 = x[0,:,:]*self.grad_x[0,:,:,:]
      tmp2 = x[1,:,:]*self.grad_x[1,:,:,:]
      
      
      tmp1[~np.isfinite(tmp1)] = 1e-20
      tmp2[~np.isfinite(tmp2)] = 1e-20
      
      return self.FT((tmp1[:,None,None,:,:]+tmp2[:,None,None,:,:])*self.par.C)
    
  def operator_adjoint(self,x):
    
      x[~np.isfinite(x)] = 1e-20
      fdx = self.FTH(x)
      
      fdx = np.squeeze(np.sum((fdx*np.conj(self.par.C)),axis=1))
      
      dx = np.zeros((2,self.par.dimX,self.par.dimY),dtype='complex128')
      
      dx[0,:,:] = np.sum(self.conj_grad_x[0,:,:,:]*fdx,axis=0)
      dx[1,:,:] = np.sum(self.conj_grad_x[1,:,:,:]*fdx,axis=0)
      
      return dx
    
    
  def tgv_solve(self,x,res,iters):
    alpha = self.irgn_par.gamma
    beta = self.irgn_par.gamma*2
    
    
    tau = 1/np.sqrt(16**2+8**2)
    xk = np.copy(x)
    
    r = np.zeros_like(self.data,dtype='complex128')
    z1 = np.zeros(([2,2,self.par.dimX,self.par.dimY]),dtype='complex128')
    z2 = np.zeros(([2,3,self.par.dimX,self.par.dimY]),dtype='complex128')
    v = np.zeros(([2,2,self.par.dimX,self.par.dimY]),dtype='complex128')
    x_new = np.zeros_like(x)
    
    delta = self.irgn_par.delta
    mu = 1/delta
    
    theta_line = 1.0
    beta_line = 1
    mu_line = 0.5
    
    Axold = self.operator_forward(x)
    Kyk1 = self.operator_adjoint(r) - gd.bdiv_1(z1)
    Kyk2 = -z1 - gd.fdiv_2(z2)
    
    for i in range(iters):
      
      np.maximum(0,((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta),x_new)
      np.maximum(0,np.minimum(300/self.par.M0_sc,x_new[0,:,:],x_new[0,:,:]),x_new[0,:,:])
      np.abs(np.maximum(0.9777,np.minimum(0.9991,x_new[1,:,:],x_new[1,:,:]),x_new[1,:,:]),x_new[1,:,:])
      
      
      v_new = v-tau*Kyk2
      
      beta_new = beta_line*(1+mu*tau)
      
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new
      
      gradx = gd.fgrad_1(x_new)
      gradx_xold = gradx - gd.fgrad_1(x)
      v_vold = v_new-v
      symgrad_v = gd.sym_bgrad_2(v_new)
      symgrad_v_vold = symgrad_v - gd.sym_bgrad_2(v)
      Ax = self.operator_forward(x_new)
      Ax_Axold = Ax-Axold
    
      while True:
        
        theta_line = tau_new/tau
        
        z1_new = z1 + beta_line*tau_new*( gradx + theta_line*gradx_xold - v_new - theta_line*v_vold  )
        z1_new = z1_new/np.maximum(1,(np.sqrt(np.sum(z1_new**2,axis=(0,1)))/alpha))
     
        z2_new = z2 + beta_line*tau_new*( symgrad_v + theta_line*symgrad_v_vold )
        scal = np.sqrt( np.sum(z2_new[:,0,:,:]**2 + z2_new[:,1,:,:]**2 + 2*z2_new[:,2,:,:]**2,axis=0) )
        np.maximum(1,scal/(beta),scal)
        z2_new = z2_new/scal
        
        
        
        tmp = Ax+theta_line*Ax_Axold
        r_new = (( r  + beta_line*tau_new*(tmp) ) - beta_line*tau_new*res)/(1+beta_line*tau_new/self.irgn_par.lambd)
        
        Kyk1_new = self.operator_adjoint(r_new)-gd.bdiv_1(z1_new)
        Kyk2_new = -z1_new -gd.fdiv_2(z2_new)

        
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
        
      if not np.mod(i,50):
        plt.figure(1)
        plt.imshow(np.transpose(np.abs(x[0,:,:]*self.par.M0_sc)))
        plt.pause(0.05)
        plt.figure(2)
        plt.imshow(np.transpose(np.abs(-self.par.TR/np.log(x[1,:,:]))))
        plt.pause(0.05)
        primal= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*np.sum(np.abs((gradx-v))) +
                 beta*np.sum(np.abs(symgrad_v)) + 1/(2*delta)*np.linalg.norm((x-xk).flatten())**2)
    
        dual = np.real(-delta/2*np.linalg.norm((-Kyk1_new).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1_new).flatten()) + np.sum(Kyk2_new) 
                - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r.flatten())**2 - np.vdot(res.flatten(),r.flatten()))
        gap = np.abs(primal - dual)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal,dual,gap))
        
        
    self.v = v
    return x
      
  def FT(self, x):
    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]
    scale =  np.sqrt(np.shape(x)[3]*np.shape(x)[4])
        
    result = np.zeros_like(x,dtype='complex128')
#    cdef int scan=0
#    cdef int coil=0
    for scan in range(nscan):
      for coil in range(NC):
        result[scan,coil,:,:,:] = self.fft_forward(x[scan,coil,:,:,:])/scale
      
    return result


  def FTH(self, x):
    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]
    scale =  np.sqrt(np.shape(x)[3]*np.shape(x)[4])
        
    result = np.zeros_like(x,dtype='complex128')
#    cdef int scan=0
#    cdef int coil=0
    for scan in range(nscan):
      for coil in range(NC):
        result[scan,coil,:,:,:] = self.fft_back(x[scan,coil,:,:,:])*scale
      
    return result      
      
      
