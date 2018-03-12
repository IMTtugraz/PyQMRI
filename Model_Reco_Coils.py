

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
  
    x = self.tikonov_solve(x_old,res,iters)     
    
#    self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_2D(x,islice)))**2
#           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_1(x[:self.unknowns_TGV,...])-self.v))
#           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_2(self.v))) 
#           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2
#           +self.irgn_par.omega/2*np.linalg.norm(gd.fgrad_1(x[-self.unknowns_H1:,...]))**2)    
#    print("-"*80)
#    print ("Function value after GN-Step: %f" %(self.fval/self.irgn_par.lambd))
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
      result = np.concatenate(((self.model.guess),self.par.C),0)
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
          
          end = time.time()-start
          print ("Function value after GN-Step %i: %f" %(i,self.fval))
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
      
  

    

    
    
        