
# cython: infer_types=True
# cython: profile=False

from __future__ import division
cimport cython

cimport numpy as np
import numpy as np
from numpy cimport ndarray
import pywt
import time

cimport gradients_divergences as gd

np.import_array()

DTYPE = np.complex64
ctypedef np.complex64_t DTYPE_t
DTYPE_real = np.float32
ctypedef np.float32_t DTYPE_t_real

import primaldualtoolbox



@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.initializedcheck(False)

################################################################################
### Main class to manage parameter reconstruction problem ######################
################################################################################
cdef class Model_Reco:
  cdef list _plan
  cdef public list gn_res
  cdef dict __dict__
  cdef int unknowns
  cdef int NSlice
  cdef int NScan
  cdef int dimY
  cdef int dimX
  cdef int NC
  cdef int N
  cdef int Nproj
  cdef double scale
  cdef public float dz
  cdef double fval
  cdef double fval_min
  cdef DTYPE_t[:,:,:,::1] grad_x_2D
  cdef DTYPE_t[:,:,:,::1] conj_grad_x_2D
  cdef DTYPE_t[:,:,::1] Coils, conjCoils
  cdef DTYPE_t[:,:,:,::1] Coils3D, conjCoils3D
  cdef public str wavelet
  cdef public str border
  cdef DTYPE_t_real[::1] ukscale
  cdef public alpha0_alpha1
  cdef public double ratio


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
    self.dz = 1
    self.fval_min = 0
    self.fval = 0
    self.gn_res = []
    self.wavelet = 'db4'
    self.border = 'symmetric'
    self.ukscale = np.ones(self.unknowns,dtype=DTYPE_real)
    self.alpha0_alpha1 = 2
    self.ratio = 100


################################################################################
### Start a 2D Reconstruction, set TV to True to perform TV instead of TGV######
### Precompute Model and Gradient values for xk ################################
### Call inner optimization ####################################################
### input: bool to switch between TV (1) and TGV (0) regularization ############
### output: optimal value of x #################################################
################################################################################
  cpdef execute_2D(self, int TV=0):
    gamma = self.irgn_par.gamma
    delta = self.irgn_par.delta
    self.result = np.zeros((self.irgn_par.max_GN_it,self.unknowns,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
    result = np.copy(self.model.guess)
    if TV==1:
      for islice in range(self.par.NSlice):
        self.init_plan(islice)
        self.FT = self.nFT_2D
        self.FTH = self.nFTH_2D
        self.irgn_par.gamma = gamma
        self.irgn_par.delta = delta
        self.Coils = np.array(np.squeeze(self.par.C[:,islice,:,:]),order='C')
        self.conjCoils = np.conj(self.Coils)
        self.r = np.zeros(([self.NScan,self.NC,self.Nproj,self.N]),dtype=DTYPE)
        self.z1 = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
        iters = self.irgn_par.start_iters
        for i in range(self.irgn_par.max_GN_it):
          start = time.time()
          self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_2D(result[:,islice,:,:],islice))

          scale = np.linalg.norm(np.abs(self.grad_x_2D[0,...]))/np.linalg.norm(np.abs(self.grad_x_2D[1,...]))
          if scale > 1e3:
            scale = 1

          for j in range(len(self.model.constraints)-1):
            self.model.constraints[j+1].update(scale)

          result[1,islice,:,:] = result[1,islice,:,:]*self.model.T1_sc
          self.model.T1_sc = self.model.T1_sc*(scale)
          result[1,islice,:,:] = result[1,islice,:,:]/self.model.T1_sc
          self.step_val = self.model.execute_forward_2D(result[:,islice,:,:],islice)
          self.grad_x_2D = self.model.execute_gradient_2D(result[:,islice,:,:],islice)
          self.conj_grad_x_2D = np.conj(self.grad_x_2D)

          result[:,islice,:,:] = self.irgn_solve_2D(result[:,islice,:,:], iters, self.data[:,:,islice,:],TV)
          self.result[i,1,islice,:,:] = result[1,islice,:,:]*self.model.T1_sc
          self.result[i,0,islice,:,:] = result[0,islice,:,:]*self.model.M0_sc

          iters = np.fmin(iters*2,self.irgn_par.max_iters)
          self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*self.irgn_par.gamma_dec,self.irgn_par.gamma_min)
          self.irgn_par.delta = np.minimum(self.irgn_par.delta*self.irgn_par.delta_inc,self.irgn_par.delta_max)

          end = time.time()-start
          print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
          print("-"*80)
          self.gn_res.append(self.fval)
          if (np.abs(self.fval_min-self.fval) < self.irgn_par.lambd*self.irgn_par.tol) and i>0:
            print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-self.fval)/self.irgn_par.lambd))
            break
          if i==0:
            self.fval_min = self.fval
          self.fval_min = np.minimum(self.fval,self.fval_min)
    elif TV==0:
      for islice in range(self.par.NSlice):
        self.init_plan(islice)
        self.FT = self.nFT_2D
        self.FTH = self.nFTH_2D
        self.irgn_par.gamma = gamma
        self.irgn_par.delta = delta
        self.Coils = np.array(np.squeeze(self.par.C[:,islice,:,:]),order='C')
        self.conjCoils = np.conj(self.Coils)
        self.v = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
        self.r = np.zeros(([self.NScan,self.NC,self.Nproj,self.N]),dtype=DTYPE)
        self.z1 = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
        self.z2 = np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
        iters = self.irgn_par.start_iters
        for i in range(self.irgn_par.max_GN_it):
          start = time.time()
          self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_2D(result[:,islice,:,:],islice))

          scale = np.linalg.norm(np.abs(self.grad_x_2D[0,...]))/np.linalg.norm(np.abs(self.grad_x_2D[1,...]))
          if scale > 1e3:
            scale = 1

          for j in range(len(self.model.constraints)-1):
            self.model.constraints[j+1].update(scale)

          result[1,islice,:,:] = result[1,islice,:,:]*self.model.T1_sc
          self.model.T1_sc = self.model.T1_sc*(scale)
          result[1,islice,:,:] = result[1,islice,:,:]/self.model.T1_sc
          self.step_val = self.model.execute_forward_2D(result[:,islice,:,:],islice)
          self.grad_x_2D = self.model.execute_gradient_2D(result[:,islice,:,:],islice)
          self.conj_grad_x_2D = np.conj(self.grad_x_2D)

          result[:,islice,:,:] = self.irgn_solve_2D(result[:,islice,:,:], iters, self.data[:,:,islice,:],TV)
          self.result[i,1,islice,:,:] = result[1,islice,:,:]*self.model.T1_sc
          self.result[i,0,islice,:,:] = result[0,islice,:,:]*self.model.M0_sc

          iters = np.fmin(iters*2,self.irgn_par.max_iters)
          self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*self.irgn_par.gamma_dec,self.irgn_par.gamma_min)
          self.irgn_par.delta = np.minimum(self.irgn_par.delta*self.irgn_par.delta_inc,self.irgn_par.delta_max)

          end = time.time()-start
          print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
          print("-"*80)
          self.gn_res.append(self.fval)
          if (np.abs(self.fval_min-self.fval) < self.irgn_par.lambd*self.irgn_par.tol) and i>0:
            print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-self.fval)/self.irgn_par.lambd))
            break
          if i==0:
            self.fval_min = self.fval
          self.fval_min = np.minimum(self.fval,self.fval_min)
    else:
      for islice in range(self.par.NSlice):
        self.init_plan(islice)
        self.FT = self.nFT_2D
        self.FTH = self.nFTH_2D
        self.irgn_par.gamma = gamma
        self.irgn_par.delta = delta
        self.Coils = np.array(np.squeeze(self.par.C[:,islice,:,:]),order='C')
        self.conjCoils = np.conj(self.Coils)
        self.r = np.zeros(([self.NScan,self.NC,self.Nproj,self.N]),dtype=DTYPE)
        self.z1 = pywt.wavedec2(result[:,islice,...],self.wavelet,self.border)
        iters = self.irgn_par.start_iters
        for j in range(len(self.z1)):
          self.z1[j] = np.array(self.z1[j])
          self.z1[j] = np.zeros_like(self.z1[j]).astype(DTYPE)

        iters = self.irgn_par.start_iters
        for i in range(self.irgn_par.max_GN_it):
          start = time.time()
          self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_2D(result[:,islice,:,:],islice))

          scale = np.linalg.norm(np.abs(self.grad_x_2D[0,...]))/np.linalg.norm(np.abs(self.grad_x_2D[1,...]))
          if scale > 1e3:
            scale = 1

          for j in range(len(self.model.constraints)-1):
            self.model.constraints[j+1].update(scale)

          result[1,islice,:,:] = result[1,islice,:,:]*self.model.T1_sc
          self.model.T1_sc = self.model.T1_sc*(scale)
          result[1,islice,:,:] = result[1,islice,:,:]/self.model.T1_sc
          self.step_val = self.model.execute_forward_2D(result[:,islice,:,:],islice)
          self.grad_x_2D = self.model.execute_gradient_2D(result[:,islice,:,:],islice)
          self.conj_grad_x_2D = np.conj(self.grad_x_2D)

          result[:,islice,:,:] = self.irgn_solve_2D(result[:,islice,:,:], iters, self.data[:,:,islice,:],TV)
          self.result[i,1,islice,:,:] = result[1,islice,:,:]*self.model.T1_sc
          self.result[i,0,islice,:,:] = result[0,islice,:,:]*self.model.M0_sc

          iters = np.fmin(iters*2,self.irgn_par.max_iters)
          self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*self.irgn_par.gamma_dec,self.irgn_par.gamma_min)
          self.irgn_par.delta = np.minimum(self.irgn_par.delta*self.irgn_par.delta_inc,self.irgn_par.delta_max)

          end = time.time()-start
          print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
          print("-"*80)
          self.gn_res.append(self.fval)
          if (np.abs(self.fval_min-self.fval) < self.irgn_par.lambd*self.irgn_par.tol) and i>0:
            print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-self.fval)/self.irgn_par.lambd))
            break
          if i==0:
            self.fval_min = self.fval
          self.fval_min = np.minimum(self.fval,self.fval_min)
################################################################################
### Precompute constant terms of the GN linearization step #####################
### input: linearization point x ###############################################
########## numeber of innner iterations iters ##################################
########## Data ################################################################
########## bool to switch between TV (1) and TGV (0) regularization ############
### output: optimal value of x for the inner GN step ###########################
################################################################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=3] irgn_solve_2D(self,np.ndarray[DTYPE_t,ndim=3] x, int iters,np.ndarray[DTYPE_t,ndim=4] data, int TV=0):

    cdef np.ndarray[DTYPE_t,ndim=3] x_old = x
    res = data - self.FT(self.step_val) + self.operator_forward_2D(x)
    if TV==1:
       x = self.tv_solve_2D(x,res,iters)
       self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_2D(x,0)))**2
              +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_1(self.scale_fwd(x))))
              +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)
       print('Norm M0 grad: %f  norm T1 grad: %f' %(np.linalg.norm(gd.fgrad_1(self.scale_fwd(x))[0,...]),np.linalg.norm(gd.fgrad_1(self.scale_fwd(x))[1,...])))
       scale = np.linalg.norm(gd.fgrad_1(self.scale_fwd(x))[0,...])/np.linalg.norm(gd.fgrad_1(self.scale_fwd(x))[1,...])
       if scale == 0 or not np.isfinite(scale):
         self.ratio = self.ratio
       else:
         self.ratio *= scale
    elif TV==0:
      x = self.tgv_solve_2D(x,res,iters)
      self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_2D(x,0)))**2
              +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_1(self.scale_fwd(x))-self.v))
              +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_2(self.v)))
              +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)
      print('Norm M0 grad: %f  norm T1 grad: %f' %(np.linalg.norm(gd.fgrad_1(self.scale_fwd(x))[0,...]),
                                                    np.linalg.norm(gd.fgrad_1(self.scale_fwd(x))[1,...])))
      scale = np.linalg.norm(gd.fgrad_1(self.scale_fwd(x))[0,...])/np.linalg.norm(gd.fgrad_1(self.scale_fwd(x))[1,...])
      if scale == 0 or not np.isfinite(scale):
        self.ratio = self.ratio
      else:
        self.ratio *= scale
    else:
       x = self.wt_solve_2D(x,res,iters)
       grad = pywt.wavedec2(self.scale_fwd(x),self.wavelet,self.border)
       print('Norm M0 grad: %f  norm T1 grad: %f' %(np.linalg.norm(np.array(grad[len(grad)-1])[:,0,...]),np.linalg.norm(np.array(grad[len(grad)-1])[:,1,...])))
       scale = np.linalg.norm(np.array(grad[len(grad)-1])[:,0,...])/np.linalg.norm(np.array(grad[len(grad)-1])[:,1,...])
       if scale == 0 or not np.isfinite(scale):
         self.ratio = self.ratio
       else:
         self.ratio *= scale
       for j in range(len(grad)):
         grad[j] = np.sum(np.abs(np.array(grad[j])))
       self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_2D(x,0)))**2
              +self.irgn_par.gamma*np.abs(np.sum(grad))
              +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)
    print("-"*80)
    print ("Function value after GN-Step: %f" %(self.fval/self.irgn_par.lambd))
    return x
################################################################################
### Start a 3D Reconstruction, set TV to True to perform TV instead of TGV######
### Precompute Model and Gradient values for xk ################################
### Call inner optimization ####################################################
### input: bool to switch between TV (1) and TGV (0) regularization ############
### output: optimal value of x #################################################
################################################################################
  cpdef execute_3D(self, int TV=0):
   self.init_plan()
   self.FT = self.nFT_3D
   self.FTH = self.nFTH_3D
   iters = self.irgn_par.start_iters
   gamma = self.irgn_par.gamma
   delta = self.irgn_par.delta


   self.r = np.zeros_like(self.data,dtype=DTYPE)
   self.z1 = np.zeros(([self.unknowns,3,self.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)


   self.result = np.zeros((self.irgn_par.max_GN_it+1,self.unknowns,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
   self.result[0,:,:,:,:] = np.copy(self.model.guess)

   self.Coils3D = np.squeeze(self.par.C)
   self.conjCoils3D = np.conj(self.Coils3D)
   result = np.copy(self.model.guess)

   if TV==1:
      for i in range(self.irgn_par.max_GN_it):
        start = time.time()
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))

        scale = np.linalg.norm(np.abs(self.grad_x[0,...]))/np.linalg.norm(np.abs(self.grad_x[1,...]))

        for j in range(len(self.model.constraints)-1):
          self.model.constraints[j+1].update(scale)

        result[1,...] = result[1,...]*self.model.T1_sc
        self.model.T1_sc = self.model.T1_sc*(scale)
        result[1,...] = result[1,...]/self.model.T1_sc

        self.step_val = np.nan_to_num(self.model.execute_forward_3D(result))
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))
        self.conj_grad_x = np.nan_to_num(np.conj(self.grad_x))

        result = self.irgn_solve_3D(result, iters, self.data,TV)
        self.result[i+1,0,...] = result[0,...]*self.model.M0_sc
        self.result[i+1,1,...] = result[1,...]*self.model.T1_sc

        iters = np.fmin(iters*2,self.irgn_par.max_iters)
        self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*self.irgn_par.gamma_dec,self.irgn_par.gamma_min)
        self.irgn_par.delta = np.minimum(self.irgn_par.delta*self.irgn_par.delta_inc,self.irgn_par.delta_max)

        end = time.time()-start
        self.gn_res.append(self.fval)
        print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
        print("-"*80)
        if (np.abs(self.fval_min-self.fval) < self.irgn_par.lambd*self.irgn_par.tol) and i>0:
          print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-self.fval)/(self.irgn_par.lambd*self.NSlice)))
          break
        if i==0:
          self.fval_min = self.fval
        self.fval_min = np.minimum(self.fval,self.fval_min)

   elif TV==0:
      self.v = np.zeros(([self.unknowns,3,self.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)
      self.z2 = np.zeros(([self.unknowns,6,self.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)
      for i in range(self.irgn_par.max_GN_it):
        start = time.time()
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))

        scale = np.linalg.norm(np.abs(self.grad_x[0,...]))/np.linalg.norm(np.abs(self.grad_x[1,...]))

        for j in range(len(self.model.constraints)-1):
          self.model.constraints[j+1].update(scale)

        result[1,...] = result[1,...]*self.model.T1_sc
        self.model.T1_sc = self.model.T1_sc*(scale)
        result[1,...] = result[1,...]/self.model.T1_sc

        self.step_val = np.nan_to_num(self.model.execute_forward_3D(result))
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))
        self.conj_grad_x = np.nan_to_num(np.conj(self.grad_x))

        result = self.irgn_solve_3D(result, iters, self.data,TV)
        self.result[i+1,0,...] = result[0,...]*self.model.M0_sc
        self.result[i+1,1,...] = result[1,...]*self.model.T1_sc

        iters = np.fmin(iters*2,self.irgn_par.max_iters)
        self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*self.irgn_par.gamma_dec,self.irgn_par.gamma_min)
        self.irgn_par.delta = np.minimum(self.irgn_par.delta*self.irgn_par.delta_inc,self.irgn_par.delta_max)

        end = time.time()-start
        self.gn_res.append(self.fval)
        print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
        print("-"*80)
        if (np.abs(self.fval_min-self.fval) < self.irgn_par.lambd*self.irgn_par.tol) and i>0:
          print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-self.fval)/(self.irgn_par.lambd*self.NSlice)))
          break
        if i==0:
          self.fval_min = self.fval
        self.fval_min = np.minimum(self.fval,self.fval_min)
   else:
      self.z1 = pywt.wavedec2(result,self.wavelet,self.border)
      for j in range(len(self.z1)):
        self.z1[j] = np.array(self.z1[j])
        self.z1[j] = np.zeros_like(self.z1[j]).astype(DTYPE)
      for i in range(self.irgn_par.max_GN_it):
        start = time.time()
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))

        scale = np.linalg.norm(np.abs(self.grad_x[0,...]))/np.linalg.norm(np.abs(self.grad_x[1,...]))

        for j in range(len(self.model.constraints)-1):
          self.model.constraints[j+1].update(scale)

        result[1,...] = result[1,...]*self.model.T1_sc
        self.model.T1_sc = self.model.T1_sc*(scale)
        result[1,...] = result[1,...]/self.model.T1_sc

        self.step_val = np.nan_to_num(self.model.execute_forward_3D(result))
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))
        self.conj_grad_x = np.nan_to_num(np.conj(self.grad_x))

        result = self.irgn_solve_3D(result, iters, self.data,TV)
        self.result[i+1,0,...] = result[0,...]*self.model.M0_sc
        self.result[i+1,1,...] = result[1,...]*self.model.T1_sc

        iters = np.fmin(iters*2,self.irgn_par.max_iters)
        self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*self.irgn_par.gamma_dec,self.irgn_par.gamma_min)
        self.irgn_par.delta = np.minimum(self.irgn_par.delta*self.irgn_par.delta_inc,self.irgn_par.delta_max)

        end = time.time()-start
        self.gn_res.append(self.fval)
        print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
        print("-"*80)
        if (np.abs(self.fval_min-self.fval) < self.irgn_par.lambd*self.irgn_par.tol) and i>0:
          print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-self.fval)/(self.irgn_par.lambd*self.NSlice)))
          break
        if i==0:
          self.fval_min = self.fval
        self.fval_min = np.minimum(self.fval,self.fval_min)
################################################################################
### Precompute constant terms of the GN linearization step #####################
### input: linearization point x ###############################################
########## numeber of innner iterations iters ##################################
########## Data ################################################################
########## bool to switch between TV (1) and TGV (0) regularization ############
### output: optimal value of x for the inner GN step ###########################
################################################################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=4] irgn_solve_3D(self,np.ndarray[DTYPE_t,ndim=4] x,int iters,np.ndarray[DTYPE_t,ndim=5] data, int TV=0):

    cdef np.ndarray[DTYPE_t,ndim=4] x_old = x
    res = data - self.FT(self.step_val) + self.operator_forward_3D(x)


    if TV==1:
      x = self.tv_solve_3D(x,res,iters)
      grad = gd.fgrad_3(self.scale_fwd(x),1,1,self.dz)
      self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_3D(x)))**2
              +self.irgn_par.gamma*np.sum(np.abs(grad))
              +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)
      print('Norm M0 grad: %f  norm T1 grad: %f' %(np.linalg.norm(grad[0,...]),np.linalg.norm(grad[1,...])))
      scale = np.linalg.norm(grad[0,...])/np.linalg.norm(grad[1,...])
      if scale == 0 or not np.isfinite(scale):
        self.ratio = self.ratio
      else:
        self.ratio *= scale
    elif TV==0:
       x = self.tgv_solve_3D(x,res,iters)
       grad = gd.fgrad_3(self.scale_fwd(x),1,1,self.dz)
       self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_3D(x)))**2
              +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_3(self.scale_fwd(x),1,1,self.dz)-self.v))
              +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_3(self.v,1,1,self.dz)))
              +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)
       print('Norm M0 grad: %f  norm T1 grad: %f' %(np.linalg.norm(grad[0,...]),np.linalg.norm(grad[1,...])))
       scale = np.linalg.norm(grad[0,...])/np.linalg.norm(grad[1,...])
       if scale == 0 or not np.isfinite(scale):
         self.ratio = self.ratio
       else:
         self.ratio *= scale
    else:
       x = self.wt_solve_3D(x,res,iters)
       grad = pywt.wavedec2(self.scale_fwd(x),self.wavelet,self.border)
       print('Norm M0 grad: %f  norm T1 grad: %f' %(np.linalg.norm(np.array(grad[len(grad)-1])[:,0,...]),np.linalg.norm(np.array(grad[len(grad)-1])[:,1,...])))
       scale = np.linalg.norm(np.array(grad[len(grad)-1])[:,0,...])/np.linalg.norm(np.array(grad[len(grad)-1])[:,1,...])
       if scale == 0 or not np.isfinite(scale):
         self.ratio = self.ratio
       else:
         self.ratio *= scale
       for j in range(len(grad)):
         grad[j] = np.sum(np.abs(np.array(grad[j])))
       self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_3D(x)))**2
              +self.irgn_par.gamma*np.abs(np.sum(grad))
              +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)

    print("-"*80)
    print ("Function value after GN-Step: %f" %(self.fval/(self.irgn_par.lambd*self.NSlice)))

    return x
################################################################################
### 2D forward data operator ###################################################
### input: parameter space data x ##############################################
### output: k-space data #######################################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=3] operator_forward_2D(self,np.ndarray[DTYPE_t,ndim=3] x):
    return self.FT(np.sum(x[:,None,...]*self.grad_x_2D,axis=0))
################################################################################
### 2D adjoint data operator ###################################################
### input: k-space data x ######################################################
### output: parameter-space data ###############################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=3] operator_adjoint_2D(self,np.ndarray[DTYPE_t,ndim=4] x):
    return np.squeeze(np.sum(np.squeeze((self.FTH(x)))*self.conj_grad_x_2D,axis=1))
################################################################################
### 3D forward data operator ###################################################
### input: parameter space data x ##############################################
### output: k-space data #######################################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=5] operator_forward_3D(self,np.ndarray[DTYPE_t,ndim=4] x):
    return self.FT(np.sum(x[:,None,:,:,:]*self.grad_x,axis=0))
################################################################################
### 3D adjoint data operator ###################################################
### input: k-space data x ######################################################
### output: parameter-space data ###############################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=4] operator_adjoint_3D(self,np.ndarray[DTYPE_t,ndim=5] x):
    return np.squeeze(np.sum(np.squeeze((self.FTH(x))*self.conj_grad_x),axis=1))

################################################################################
### 2D-TGV-Frobenius optimized with Primal-Dual algorithm with line search #####
### input: initial guess x #####################################################
########## precomputed data combined with constant terms res ###################
########## maximum number of iterations ########################################
### output: optimal values for x ###############################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=3] tgv_solve_2D(self, np.ndarray[DTYPE_t, ndim=3] x, np.ndarray[DTYPE_t, ndim=4] res, int iters):
############# Define ratio between gradient and sym. gradient ##################
    cdef double alpha = self.irgn_par.gamma
    cdef double beta = self.irgn_par.gamma*self.alpha0_alpha1

### Optimal determine the operator norm
#    cdef np.ndarray[DTYPE_t,ndim=3] xx = np.zeros_like(x,dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t,ndim=3] yy = np.zeros_like(x,dtype=DTYPE)
#    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
#    yy = self.operator_adjoint_2D(self.operator_forward_2D(xx));
#    cdef int j = 0
#    for j in range(10):
#       if not np.isclose(np.linalg.norm(yy.flatten()),0):
#           xx = yy/np.linalg.norm(yy.flatten())
#       else:
#           xx = yy
#       yy = self.operator_adjoint_2D(self.operator_forward_2D(xx))
#       l1 = np.vdot(yy.flatten(),xx.flatten());
#    L = np.max(np.abs(l1)) ## Lipschitz constant estimate
#    L = (L+12)
    L = (12)
    print('L: %f'%(L))



    cdef double tau = 1/np.sqrt(L)
    cdef double tau_new = 0
############# Primal variables #################################################
    cdef np.ndarray[DTYPE_t, ndim=3] xk = x
    cdef np.ndarray[DTYPE_t, ndim=3] x_new = np.zeros_like(x,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] v = self.v
    cdef np.ndarray[DTYPE_t, ndim=4] v_new = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
############# Dual variables ###################################################
    cdef np.ndarray[DTYPE_t, ndim=4] r = self.r
    cdef np.ndarray[DTYPE_t, ndim=4] z1 = self.z1
    cdef np.ndarray[DTYPE_t, ndim=4] z2 = self.z2
    cdef np.ndarray[DTYPE_t, ndim=4] r_new = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] z1_new = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] z2_new = np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)


############# Arrays holding intermediate results of the primal update step ####
    cdef np.ndarray[DTYPE_t, ndim=3] Kyk1 = np.zeros_like(x,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] Kyk2 = np.zeros_like(z1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] Kyk1_new = np.zeros_like(x,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] Kyk2_new = np.zeros_like(z1,dtype=DTYPE)


############# Arrays holding intermediate results of the dual update step ######
    cdef np.ndarray[DTYPE_t, ndim=4] Ax = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] Ax_Axold = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] Axold = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] tmp = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] gradx = np.zeros_like(z1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] gradx_xold = np.zeros_like(z1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] v_old = np.zeros_like(v,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] symgrad_v = np.zeros_like(z2,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] symgrad_v_vold = np.zeros_like(z2,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t_real, ndim=2] scal = np.zeros((self.par.dimX,self.par.dimY),dtype=DTYPE_real)

############# Strong convexity parameter #######################################
    cdef double delta = self.irgn_par.delta
    cdef double mu = 1/delta


############# Line search parameters ###########################################
    cdef double theta_line = 1.0
    cdef double beta_line = 400
    cdef double beta_new = 0
    cdef double mu_line = 0.5
    cdef double delta_line = 1
    cdef double ynorm = 0.0
    cdef double lhs = 0.0

############# Primal, Dual and Gap energy ######################################
    cdef double primal = 0.0
    cdef double primal_new = 0
    cdef double dual = 0.0
    cdef double gap_min = 0.0
    cdef double gap = 0.0

    cdef int i=0


    self.set_scale(x)
############# Precompute intermediate results ##################################
    Axold = self.operator_forward_2D(x)
    Kyk1 = self.operator_adjoint_2D(r) - self.scale_adj(gd.bdiv_1(z1))
    Kyk2 = -z1 - gd.fdiv_2(z2)

############# Start main iterations ############################################
    for i in range(iters):


############# Primal updates with box constraint ###############################
      x_new = ((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta)
      for j in range(len(self.model.constraints)):
        x_new[j,...] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,x_new[j,...]))
        if self.model.constraints[j].real:
          x_new[j,...] = np.real(x_new[j,...])
      v_new = v-tau*Kyk2


############# Update step sizes to maximal values ##############################
      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new


############# Precompute Dual intermediate results for line search #############
      gradx = gd.fgrad_1(self.scale_fwd(x_new))
      gradx_xold = gradx - gd.fgrad_1(self.scale_fwd(x))
      v_vold = v_new-v
      symgrad_v = gd.sym_bgrad_2(v_new)
      symgrad_v_vold = symgrad_v - gd.sym_bgrad_2(v)
      Ax = self.operator_forward_2D(x_new)
      Ax_Axold = Ax-Axold


############# Start line serach ################################################
      while True:

        theta_line = tau_new/tau


############# Dual Updates #####################################################
        z1_new = z1 + beta_line*tau_new*( gradx + theta_line*gradx_xold
                                          - v_new - theta_line*v_vold  )
        z1_new = z1_new/np.maximum(1,(np.sqrt(np.sum(np.abs(z1_new)**2,axis=(0,1)))/alpha))

        z2_new = z2 + beta_line*tau_new*( symgrad_v + theta_line*symgrad_v_vold )
        scal = np.sqrt( np.sum(np.abs(z2_new[:,0,:,:])**2 + np.abs(z2_new[:,1,:,:])**2 + 2*np.abs(z2_new[:,2,:,:])**2,axis=0) )
        scal = np.maximum(1,scal/(beta))
        z2_new = z2_new/scal

        tmp = Ax+theta_line*Ax_Axold
        r_new = (( r  + beta_line*tau_new*(tmp) ) - beta_line*tau_new*res)/(1+beta_line*tau_new/self.irgn_par.lambd)

############# Precompute primal intermediate results ###########################
        Kyk1_new = self.operator_adjoint_2D(r_new) - self.scale_adj(gd.bdiv_1(z1_new))
        Kyk2_new = -z1_new - gd.fdiv_2(z2_new)


############# Check line search conditions #####################################
        ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),(z1_new-z1).flatten(),(z2_new-z2).flatten()]))
        lhs = np.sqrt(beta_line)*tau_new*np.linalg.norm(np.concatenate([(Kyk1_new-Kyk1).flatten(),(Kyk2_new-Kyk2).flatten()]))
        if lhs <= ynorm*delta_line:
            break
        else:
          tau_new = tau_new*mu_line


############# Check if algorithm converged/stagnated and plot result ###########
      if not np.mod(i,20):
        if self.irgn_par.display_iterations:
          self.model.plot_unknowns(x_new,True)
        primal_new= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*np.sum(np.abs((gradx-v_new))) +
                   beta*np.sum(np.abs(symgrad_v)) + 1/(2*delta)*np.linalg.norm((x_new-xk).flatten())**2)

        dual = np.real(-delta/2*np.linalg.norm((-Kyk1_new).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1_new).flatten()) + np.sum(Kyk2_new)
                  - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r_new.flatten())**2 - np.vdot(res.flatten(),r_new.flatten()))

        gap = np.abs(primal_new - dual)

        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<self.irgn_par.lambd*self.irgn_par.tol:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,np.abs(primal-primal_new)/self.irgn_par.lambd))
          self.v = v_new
          self.r = r_new
          self.z1 = z1_new
          self.z2 = z2_new
          return x_new
        if (gap > gap_min*self.irgn_par.stag) and i>1:
          self.v = v_new
          self.r = r_new
          self.z1 = z1_new
          self.z2 = z2_new
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x_new
        if np.abs(gap - gap_min)<self.irgn_par.lambd*self.irgn_par.tol and i>1:
          self.v = v_new
          self.r = r_new
          self.z1 = z1_new
          self.z2 = z2_new
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,np.abs(gap - gap_min)/self.irgn_par.lambd))
          return x_new
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal/self.irgn_par.lambd,dual/self.irgn_par.lambd,gap/self.irgn_par.lambd))

############# Update variables #################################################
      Kyk1 = (Kyk1_new)
      Kyk2 =  (Kyk2_new)
      Axold =(Ax)
      z1 = (z1_new)
      z2 = (z2_new)
      r =  (r_new)
      tau =  (tau_new)
      x = (x_new)
      v = (v_new)
############# Return final result ##############################################
    self.v = v
    self.r = r
    self.z1 = z1
    self.z2 = z2
    return x
################################################################################
### 2D-TV optimized with Primal-Dual algorithm with line search ################
### input: initial guess x #####################################################
########## precomputed data combined with constant terms res ###################
########## maximum number of iterations ########################################
### output: optimal values for x ###############################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=3] tv_solve_2D(self, np.ndarray[DTYPE_t, ndim=3] x, np.ndarray[DTYPE_t, ndim=4] res, int iters):
    cdef double alpha = self.irgn_par.gamma

############## Optimal determine operator  norm of #############################
#    cdef np.ndarray[DTYPE_t,ndim=3] xx = np.zeros_like(x,dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t,ndim=3] yy = np.zeros_like(x,dtype=DTYPE)
#    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
#    yy = self.operator_adjoint_2D(self.operator_forward_2D(xx));
#    cdef int j = 0
#    for j in range(10):
#       if not np.isclose(np.linalg.norm(yy.flatten()),0):
#           xx = yy/np.linalg.norm(yy.flatten())
#       else:
#           xx = yy
#       yy = self.operator_adjoint_2D(self.operator_forward_2D(xx))
#       l1 = np.vdot(yy.flatten(),xx.flatten());
#    L = np.max(np.abs(l1)) ## Lipschitz constant estimate
#
#    L = (L+8)
    L= 8
    print('L: %f'%(L))



    cdef double tau = 1/np.sqrt(L)
    cdef double tau_new = 0
############# Primal variables #################################################
    cdef np.ndarray[DTYPE_t, ndim=3] xk = x
    cdef np.ndarray[DTYPE_t, ndim=3] x_new = np.zeros_like(x,dtype=DTYPE)
############# Dual variables ###################################################
    cdef np.ndarray[DTYPE_t, ndim=4] r = self.r
    cdef np.ndarray[DTYPE_t, ndim=4] z1 = self.z1
    cdef np.ndarray[DTYPE_t, ndim=4] r_new = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] z1_new = np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)

############# Arrays holding intermediate results of the primal update step ####
    cdef np.ndarray[DTYPE_t, ndim=3] Kyk1 = np.zeros_like(x,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] Kyk1_new = np.zeros_like(x,dtype=DTYPE)

############# Arrays holding intermediate results of the dual update step ######
    cdef np.ndarray[DTYPE_t, ndim=4] Ax = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] Ax_Axold = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] Axold = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] tmp = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] gradx = np.zeros_like(z1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] gradx_xold = np.zeros_like(z1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t_real, ndim=2] scal = np.zeros((self.par.dimX,self.par.dimY),dtype=DTYPE_real)

############# Strong convexity parameter #######################################
    cdef double delta = self.irgn_par.delta
    cdef double mu = 1/delta

############# Line search parameters ###########################################
    cdef double theta_line = 1.0
    cdef double beta_line = 400
    cdef double beta_new = 0
    cdef double mu_line = 0.5
    cdef double delta_line = 1
    cdef double ynorm = 0.0
    cdef double lhs = 0.0

############# Primal, Dual and Gap energy ######################################
    cdef double primal = 0.0
    cdef double primal_new = 0
    cdef double dual = 0.0
    cdef double gap_min = 0.0
    cdef double gap = 0.0

    cdef int i=0
    self.set_scale(x)
############# Precompute intermediate results ##################################
    Axold = self.operator_forward_2D(x)
    Kyk1 = self.operator_adjoint_2D(r) - self.scale_adj(gd.bdiv_1(z1))

############# Start main iterations ############################################
    for i in range(iters):

############# Primal updates with box constraint ###############################
      x_new = ((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta)
      for j in range(len(self.model.constraints)):
        x_new[j,...] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,x_new[j,...]))
        if self.model.constraints[j].real:
          x_new[j,...] = np.real(x_new[j,...])

############# Update step sizes to maximal values ##############################
      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new

############# Precompute Dual intermediate results for line search #############
      gradx = gd.fgrad_1(self.scale_fwd(x_new))
      gradx_xold = gradx - gd.fgrad_1(self.scale_fwd(x))
      Ax = self.operator_forward_2D(x_new)
      Ax_Axold = Ax-Axold

############# Start line serach ################################################
      while True:

        theta_line = tau_new/tau
############# Dual Updates #####################################################
        z1_new = z1 + beta_line*tau_new*( gradx + theta_line*gradx_xold)
        z1_new = z1_new/np.maximum(1,(np.sqrt(np.sum(np.abs(z1_new)**2,axis=(0,1),keepdims=True))/alpha))

        tmp = Ax+theta_line*Ax_Axold
        r_new = (( r  + beta_line*tau_new*(tmp) ) - beta_line*tau_new*res)/(1+beta_line*tau_new/self.irgn_par.lambd)

############# Precompute primal intermediate results ###########################
        Kyk1_new = self.operator_adjoint_2D(r_new) - self.scale_adj(gd.bdiv_1(z1_new))
############# Check line search conditions #####################################
        ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),(z1_new-z1).flatten()]))
        lhs = np.sqrt(beta_line)*tau_new*np.linalg.norm(np.concatenate([(Kyk1_new-Kyk1).flatten()]))
        if lhs <= ynorm*delta_line:
            break
        else:
          tau_new = tau_new*mu_line

############# Check if algorithm converged/stagnated and plot result ###########
      if not np.mod(i,20):
        if self.irgn_par.display_iterations:
          self.model.plot_unknowns(x_new,True)
        primal_new= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*np.sum(np.abs((gradx)))
                           + 1/(2*delta)*np.linalg.norm((x_new-xk).flatten())**2)

        dual = np.real(-delta/2*np.linalg.norm((-Kyk1_new).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1_new).flatten())
                    - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r_new.flatten())**2 - np.vdot(res.flatten(),r_new.flatten()))

        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<self.irgn_par.lambd*self.irgn_par.tol:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,np.abs(primal-primal_new)/self.irgn_par.lambd))
          self.r = r_new
          self.z1 = z1_new
          return x_new
        if (gap > gap_min*self.irgn_par.stag) and i>1:
          self.r = r_new
          self.z1 = z1_new
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x_new
        if np.abs(gap - gap_min)<self.irgn_par.lambd*self.irgn_par.tol and i>1:
          self.r = r_new
          self.z1 = z1_new
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,np.abs(gap - gap_min)/self.irgn_par.lambd))
          return x_new
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal/self.irgn_par.lambd,dual/self.irgn_par.lambd,gap/self.irgn_par.lambd))

############# Update variables #################################################
      x = (x_new)
      Kyk1 = (Kyk1_new)
      Axold =(Ax)
      z1 = (z1_new)
      r =  (r_new)
      tau =  (tau_new)
############# Return final result ##############################################
    self.r = r
    self.z1 = z1
    return x

################################################################################
### 2D-TV optimized with Primal-Dual algorithm with line search ################
### input: initial guess x #####################################################
########## precomputed data combined with constant terms res ###################
########## maximum number of iterations ########################################
### output: optimal values for x ###############################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=3] wt_solve_2D(self, np.ndarray[DTYPE_t, ndim=3] x, np.ndarray[DTYPE_t, ndim=4] res, int iters):
    cdef double alpha = self.irgn_par.gamma

############## Optimal determine operator  norm of #############################
#    cdef np.ndarray[DTYPE_t,ndim=3] xx = np.zeros_like(x,dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t,ndim=3] yy = np.zeros_like(x,dtype=DTYPE)
#    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
#    yy = self.operator_adjoint_2D(self.operator_forward_2D(xx));
#    cdef int j = 0
#    for j in range(10):
#       if not np.isclose(np.linalg.norm(yy.flatten()),0):
#           xx = yy/np.linalg.norm(yy.flatten())
#       else:
#           xx = yy
#       yy = self.operator_adjoint_2D(self.operator_forward_2D(xx))
#       l1 = np.vdot(yy.flatten(),xx.flatten());
#    L = np.max(np.abs(l1)) ## Lipschitz constant estimate
#
#    L = (L+8)
    L= 8
    print('L: %f'%(L))



    cdef double tau = 1/np.sqrt(L)
    cdef double tau_new = 0
############# Primal variables #################################################
    cdef np.ndarray[DTYPE_t, ndim=3] xk = x
    cdef np.ndarray[DTYPE_t, ndim=3] x_new = np.zeros_like(x,dtype=DTYPE)
############# Dual variables ###################################################
    cdef np.ndarray[DTYPE_t, ndim=4] r = self.r
    cdef list z1 = self.z1
    cdef np.ndarray[DTYPE_t, ndim=4] r_new = np.zeros_like(res,dtype=DTYPE)
    cdef list z1_new = list(np.copy(z1))#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    cdef list diffz = list(np.copy(z1))
############# Arrays holding intermediate results of the primal update step ####
    cdef np.ndarray[DTYPE_t, ndim=3] Kyk1 = np.zeros_like(x,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] Kyk1_new = np.zeros_like(x,dtype=DTYPE)

############# Arrays holding intermediate results of the dual update step ######
    cdef np.ndarray[DTYPE_t, ndim=4] Ax = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] Ax_Axold = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] Axold = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] tmp = np.zeros_like(res,dtype=DTYPE)
    cdef list gradx = []
    cdef list gradx_xold = []
    cdef np.ndarray[DTYPE_t_real, ndim=2] scal = np.zeros((self.par.dimX,self.par.dimY),dtype=DTYPE_real)

############# Strong convexity parameter #######################################
    cdef double delta = self.irgn_par.delta
    cdef double mu = 1/delta

############# Line search parameters ###########################################
    cdef double theta_line = 1.0
    cdef double beta_line = 400
    cdef double beta_new = 0
    cdef double mu_line = 0.5
    cdef double delta_line = 1
    cdef double ynorm = 0.0
    cdef double lhs = 0.0

############# Primal, Dual and Gap energy ######################################
    cdef double primal = 0.0
    cdef double primal_new = 0
    cdef double dual = 0.0
    cdef double gap_min = 0.0
    cdef double gap = 0.0

    cdef int i=0
    self.set_scale(x)
############# Precompute intermediate results ##################################
    Axold = self.operator_forward_2D(x)
    for j in range(len(z1)):
      z1[j] = list(z1[j])
    Kyk1 = self.operator_adjoint_2D(r) + self.scale_adj(pywt.waverec2(z1,self.wavelet,self.border))
    for j in range(len(z1)):
      z1[j] = np.array(z1[j])

    gradx_xold = []
    gradxold = pywt.wavedec2(self.scale_fwd(x),self.wavelet,self.border)
    for j in range(len(gradxold)):
      gradxold[j] = np.array(gradxold[j])
    gradx_xold = list(np.copy(gradxold))
############# Start main iterations ############################################
    for i in range(iters):

############# Primal updates with box constraint ###############################
      x_new = ((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta)
      for j in range(len(self.model.constraints)):
        x_new[j,...] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,x_new[j,...]))
        if self.model.constraints[j].real:
          x_new[j,...] = np.real(x_new[j,...])

############# Update step sizes to maximal values ##############################
      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new

############# Precompute Dual intermediate results for line search #############
      gradx = pywt.wavedec2(self.scale_fwd(x_new),self.wavelet,self.border)
      for j in range(len(gradx)):
        gradx_xold[j] = ((np.array(gradx[j]) - np.array(gradxold[j])))
      Ax = self.operator_forward_2D(x_new)
      Ax_Axold = Ax-Axold

############# Start line serach ################################################
      while True:

        theta_line = tau_new/tau
############# Dual Updates #####################################################
        for j in range(len(z1_new)):
          if j == 0:
            z1_new[j] = (np.array(z1[j]) + beta_line*tau_new*( (np.array(gradx[j])+theta_line*np.array(gradx_xold[j])) ))
            z1_new[j] = (np.array(z1_new[j])/np.maximum(1,(np.sqrt(np.sum(np.abs(np.array(z1_new[j]))**2,axis=(0),keepdims=True))/alpha)))
          else:
            z1_new[j] = (np.array(z1[j]) + beta_line*tau_new*( (np.array(gradx[j])+theta_line*np.array(gradx_xold[j])) ))
            z1_new[j] = (np.array(z1_new[j])/np.maximum(1,(np.sqrt(np.sum(np.abs(np.array(z1_new[j]))**2,axis=(0,1),keepdims=True))/alpha)))
          diffz[j] = np.linalg.norm(np.array(z1_new[j])-np.array(z1[j]))
          z1_new[j] = list(z1_new[j])

        tmp = Ax+theta_line*Ax_Axold
        r_new = (( r  + beta_line*tau_new*(tmp) ) - beta_line*tau_new*res)/(1+beta_line*tau_new/self.irgn_par.lambd)

############# Precompute primal intermediate results ###########################
        Kyk1_new = self.operator_adjoint_2D(r_new) + self.scale_adj(pywt.waverec2(z1_new,self.wavelet,self.border))
        for j in range(len(z1_new)):
          z1_new[j] = np.array(z1_new[j])
############# Check line search conditions #####################################
        ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),np.array(diffz).flatten()]))
        lhs = np.sqrt(beta_line)*tau_new*np.linalg.norm(np.concatenate([(Kyk1_new-Kyk1).flatten()]))
        if lhs <= ynorm*delta_line:
            break
        else:
          tau_new = tau_new*mu_line

############# Check if algorithm converged/stagnated and plot result ###########
      if not np.mod(i,20):
        if self.irgn_par.display_iterations:
          self.model.plot_unknowns(x_new,True)
        gradsum = 0
        for j in range(len(gradxold)):
           gradsum += np.sum(np.abs(np.array(gradx[j])))
        primal_new= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*gradsum
                           + 1/(2*delta)*np.linalg.norm((x_new-xk).flatten())**2)

        dual = np.real(-delta/2*np.linalg.norm((-Kyk1_new).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1_new).flatten())
                    - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r_new.flatten())**2 - np.vdot(res.flatten(),r_new.flatten()))

        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<self.irgn_par.lambd*self.irgn_par.tol:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,np.abs(primal-primal_new)/self.irgn_par.lambd))
          self.r = r_new
          self.z1 = z1_new
          return x_new
        if (gap > gap_min*self.irgn_par.stag) and i>1:
          self.r = r_new
          self.z1 = z1_new
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x_new
        if np.abs(gap - gap_min)<self.irgn_par.lambd*self.irgn_par.tol and i>1:
          self.r = r_new
          self.z1 = z1_new
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,np.abs(gap - gap_min)/self.irgn_par.lambd))
          return x_new
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal/self.irgn_par.lambd,dual/self.irgn_par.lambd,gap/self.irgn_par.lambd))

############# Update variables #################################################
      x = (x_new)
      Kyk1 = (Kyk1_new)
      Axold =(Ax)
      gradxold = (gradx)
      z1 = (z1_new)
      r =  (r_new)
      tau =  (tau_new)
############# Return final result ##############################################
    self.r = r
    self.z1 = z1
    return x
################################################################################
### 3D-TGV-Frobenius optimized with Primal-Dual algorithm with line search #####
### input: initial guess x #####################################################
########## precomputed data combined with constant terms res ###################
########## maximum number of iterations ########################################
### output: optimal values for x ###############################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=4] tgv_solve_3D(self, np.ndarray[DTYPE_t,ndim=4] x, np.ndarray[DTYPE_t,ndim=5] res, int iters):
############# Define ratio between gradient and sym. gradient ##################
    cdef double alpha = self.irgn_par.gamma
    cdef double beta = self.irgn_par.gamma*self.alpha0_alpha1

    cdef float dz = self.dz

    L = (12)


    cdef double tau = 1/np.sqrt(L)
    cdef double tau_new = 0
############# Primal variables #################################################
    cdef np.ndarray[DTYPE_t,ndim=4] xk = x
    cdef np.ndarray[DTYPE_t,ndim=4] x_new = np.zeros_like(x,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] v = np.copy(self.v)
    cdef np.ndarray[DTYPE_t,ndim=5] v_new = np.zeros_like(v,dtype=DTYPE)
############# Dual variables ###################################################
    cdef np.ndarray[DTYPE_t,ndim=5] r = np.copy(self.r)
    cdef np.ndarray[DTYPE_t,ndim=5] z1 = np.copy(self.z1)
    cdef np.ndarray[DTYPE_t,ndim=5] z2 = np.copy(self.z2)
    cdef np.ndarray[DTYPE_t,ndim=5] r_new = np.zeros_like(r,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] z1_new = np.zeros_like(z1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] z2_new = np.zeros_like(z2,dtype=DTYPE)

############# Arrays holding intermediate results of the primal update step ####
    cdef np.ndarray[DTYPE_t,ndim=4] Kyk1 = np.zeros_like(x,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] Kyk2 = np.zeros_like(z1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=4] Kyk1_new = np.zeros_like(Kyk1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] Kyk2_new = np.zeros_like(Kyk2,dtype=DTYPE)

############# Arrays holding intermediate results of the dual update step ######
    cdef np.ndarray[DTYPE_t,ndim=5] Ax = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] Ax_Axold = np.zeros_like(Ax,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] Axold = np.zeros_like(Ax,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] tmp = np.zeros_like(Ax,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] gradx = np.zeros_like(z1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] gradx_xold = np.zeros_like(z1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] v_vold = np.zeros_like(v,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] symgrad_v = np.zeros_like(z2,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] symgrad_v_vold = np.zeros_like(z2,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t_real,ndim=3] scal = np.zeros((self.par.NSlice,self.par.dimX,self.par.dimY),dtype=DTYPE_real)


############# Strong convexity parameter #######################################
    cdef double delta = self.irgn_par.delta
    cdef double mu = 1/delta

    cdef double theta_line = 1.0

############# Line search parameters ###########################################
    cdef double beta_line = 400
    cdef double beta_new = 0
    cdef double mu_line = 0.5
    cdef double delta_line = 1
    cdef double ynorm = 0
    cdef double lhs = 0
############# Primal, Dual and Gap energy ######################################
    cdef double primal = 0.0
    cdef double primal_new = 0
    cdef double dual = 0.0
    cdef double gap_min = 0.0
    cdef double gap = 0.0

    cdef int i=0



    self.set_scale(x)
############# Precompute intermediate results ##################################
    Axold = self.operator_forward_3D(x)
    Kyk1 = self.operator_adjoint_3D(r) - self.scale_adj(gd.bdiv_3(z1,1,1,dz))
    Kyk2 = -z1 - gd.fdiv_3(z2,1,1,dz)

############# Start main iterations ############################################
    for i in range(iters):
############# Primal updates with box constraint ###############################
      x_new = ((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta)
      for j in range(len(self.model.constraints)):
        x_new[j,...] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,x_new[j,...]))
        if self.model.constraints[j].real:
          x_new[j,...] = np.real(x_new[j,...])
      v_new = v-tau*Kyk2
############# Update step sizes to maximal values ##############################
      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new
############# Precompute Dual intermediate results for line search #############
      gradx = gd.fgrad_3(self.scale_fwd(x_new),1,1,dz)
      gradx_xold = gradx - gd.fgrad_3(self.scale_fwd(x),1,1,dz)
      v_vold = v_new-v
      symgrad_v = gd.sym_bgrad_3(v_new,1,1,dz)
      symgrad_v_vold = symgrad_v - gd.sym_bgrad_3(v,1,1,dz)
      Ax = self.operator_forward_3D(x_new)
      Ax_Axold = Ax-Axold
############# Start line serach ################################################
      while True:

        theta_line = tau_new/tau
############# Dual Updates #####################################################
        z1_new = z1 + beta_line*tau_new*( gradx + theta_line*gradx_xold - v_new - theta_line*v_vold  )
        z1_new = z1_new/np.maximum(1,(np.sqrt(np.sum(np.abs(z1_new)**2,axis=(0,1)))/alpha))

        z2_new = z2 + beta_line*tau_new*( symgrad_v + theta_line*symgrad_v_vold )
        scal = np.sqrt( np.sum(np.abs(z2_new[:,0,:,:,:])**2 + np.abs(z2_new[:,1,:,:,:])**2 +
                    np.abs(z2_new[:,2,:,:,:])**2+ 2*np.abs(z2_new[:,3,:,:,:])**2 +
                    2*np.abs(z2_new[:,4,:,:,:])**2+2*np.abs(z2_new[:,5,:,:,:])**2,axis=0))
        scal = np.maximum(1,scal/(beta))
        z2_new = z2_new/scal

        tmp = Ax+theta_line*Ax_Axold
        r_new = (( r  + beta_line*tau_new*(tmp) ) - beta_line*tau_new*res)/(1+beta_line*tau_new/self.irgn_par.lambd)

############# Precompute primal intermediate results ###########################
        Kyk1_new = self.operator_adjoint_3D(r_new) - self.scale_adj(gd.bdiv_3(z1_new,1,1,dz))
        Kyk2_new = -z1_new -gd.fdiv_3(z2_new,1,1,dz)
############# Check line search conditions #####################################
        ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),(z1_new-z1).flatten(),(z2_new-z2).flatten()]))
        lhs = np.sqrt(beta_line)*tau_new*np.linalg.norm(np.concatenate([(Kyk1_new-Kyk1).flatten(),(Kyk2_new-Kyk2).flatten()]))
        if lhs <= ynorm*delta_line:
            break
        else:
            tau_new = tau_new*mu_line



############# Check if algorithm converged/stagnated and plot result ###########
      if not np.mod(i,20):
        if self.irgn_par.display_iterations:
          self.model.plot_unknowns(x)
        primal_new= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*np.sum(np.abs((gradx-v_new))) +
                   beta*np.sum(np.abs(symgrad_v)) + 1/(2*delta)*np.linalg.norm((x_new-xk).flatten())**2)

        dual = np.real(-delta/2*np.linalg.norm((-Kyk1_new).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1_new).flatten()) + np.sum(Kyk2_new)
                  - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r_new.flatten())**2 - np.vdot(res.flatten(),r_new.flatten()))

        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<self.irgn_par.lambd*self.irgn_par.tol:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,np.abs(primal-primal_new)/(self.irgn_par.lambd*self.NSlice)))
          self.v = v_new
          self.r = r_new
          self.z1 = z1_new
          self.z2 = z2_new
          return x_new
        if (gap > gap_min*self.irgn_par.stag) and i>1:
          self.v = v_new
          self.r = r_new
          self.z1 = z1_new
          self.z2 = z2_new
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x_new
        if np.abs(gap - gap_min)<self.irgn_par.lambd*self.irgn_par.tol and i>1:
          self.v = v_new
          self.r = r_new
          self.z1 = z1_new
          self.z2 = z2_new
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,np.abs(gap - gap_min)/(self.irgn_par.lambd*self.NSlice)))
          return x_new
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal/(self.irgn_par.lambd*self.NSlice),dual/(self.irgn_par.lambd*self.NSlice),gap/(self.irgn_par.lambd*self.NSlice)))
############# Update variables #################################################
      Kyk1 = (Kyk1_new)
      Kyk2 =  (Kyk2_new)
      Axold =(Ax)
      z1 = (z1_new)
      z2 = (z2_new)
      r =  (r_new)
      tau =  (tau_new)
      x = x_new
      v = v_new
############# Return final result ##############################################
    self.v = v
    self.z1 = z1
    self.z2 = z2
    self.r = r
    return x

################################################################################
### 3D-TV optimized with Primal-Dual algorithm with line search ################
### input: initial guess x #####################################################
########## precomputed data combined with constant terms res ###################
########## maximum number of iterations ########################################
### output: optimal values for x ###############################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=4] tv_solve_3D(self, np.ndarray[DTYPE_t,ndim=4] x, np.ndarray[DTYPE_t,ndim=5] res, int iters):
    cdef double alpha = self.irgn_par.gamma

    cdef float dz = self.dz

    L = (8)


    cdef double tau = 1/np.sqrt(L)
    cdef double tau_new = 0
############# Primal variables #################################################
    cdef np.ndarray[DTYPE_t,ndim=4] xk = x
    cdef np.ndarray[DTYPE_t,ndim=4] x_new = np.zeros_like(x,dtype=DTYPE)
############# Dual variables ###################################################
    cdef np.ndarray[DTYPE_t,ndim=5] r = np.copy(self.r)
    cdef np.ndarray[DTYPE_t,ndim=5] z1 = np.copy(self.z1)
    cdef np.ndarray[DTYPE_t,ndim=5] r_new = np.zeros_like(r,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] z1_new = np.zeros_like(z1,dtype=DTYPE)

############# Arrays holding intermediate results of the primal update step ####
    cdef np.ndarray[DTYPE_t,ndim=4] Kyk1 = np.zeros_like(x,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=4] Kyk1_new = np.zeros_like(Kyk1,dtype=DTYPE)
############# Arrays holding intermediate results of the dual update step ######
    cdef np.ndarray[DTYPE_t,ndim=5] Ax = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] Ax_Axold = np.zeros_like(Ax,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] Axold = np.zeros_like(Ax,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] tmp = np.zeros_like(Ax,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] gradx = np.zeros_like(z1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=5] gradx_xold = np.zeros_like(z1,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t_real,ndim=3] scal = np.zeros((self.par.NSlice,self.par.dimX,self.par.dimY),dtype=DTYPE_real)

############# Strong convexity parameter #######################################
    cdef double delta = self.irgn_par.delta
    cdef double mu = 1/delta
############# Line search parameters ###########################################
    cdef double theta_line = 1.0
    cdef double beta_line = 400
    cdef double beta_new = 0
    cdef double mu_line = 0.5
    cdef double delta_line = 1
    cdef double ynorm = 0
    cdef double lhs = 0

############# Primal, Dual and Gap energy ######################################
    cdef double primal = 0.0
    cdef double primal_new = 0
    cdef double dual = 0.0
    cdef double gap_min = 0.0
    cdef double gap = 0.0

    cdef int i=0
    self.set_scale(x)
############# Precompute intermediate results ##################################
    Axold = self.operator_forward_3D(x)
    Kyk1 = self.operator_adjoint_3D(r) - self.scale_adj(gd.bdiv_3(z1))

############# Start main iterations ############################################
    for i in range(iters):
############# Primal updates with box constraint ###############################
      x_new = ((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta)
      for j in range(len(self.model.constraints)):
        x_new[j,...] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,x_new[j,...]))
        if self.model.constraints[j].real:
          x_new[j,...] = np.real(x_new[j,...])
############# Update step sizes to maximal values ##############################
      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new
############# Precompute Dual intermediate results for line search #############
      gradx = gd.fgrad_3(self.scale_fwd(x_new),1,1,dz)
      gradx_xold = gradx - gd.fgrad_3(self.scale_fwd(x),1,1,dz)
      Ax = self.operator_forward_3D(x_new)
      Ax_Axold = Ax-Axold
############# Start line serach ################################################
      while True:

        theta_line = tau_new/tau
############# Dual Updates #####################################################
        z1_new = z1 + beta_line*tau_new*( gradx + theta_line*gradx_xold )
        z1_new = z1_new/np.maximum(1,(np.sqrt(np.sum(np.abs(z1_new)**2,axis=(0,1),keepdims=True))/alpha))

        tmp = Ax+theta_line*Ax_Axold
        r_new = (( r  + beta_line*tau_new*(tmp) ) - beta_line*tau_new*res)/(1+beta_line*tau_new/self.irgn_par.lambd)
############# Precompute primal intermediate results ###########################
        Kyk1_new = self.operator_adjoint_3D(r_new) - self.scale_adj(gd.bdiv_3(z1_new))
############# Check line search conditions #####################################
        ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),(z1_new-z1).flatten()]))
        lhs = np.sqrt(beta_line)*tau_new*np.linalg.norm(np.concatenate([(Kyk1_new-Kyk1).flatten()]))
        if lhs <= ynorm*delta_line:
            break
        else:
            tau_new = tau_new*mu_line

############# Check if algorithm converged/stagnated and plot result ###########
      if not np.mod(i,20):
        if self.irgn_par.display_iterations:
          self.model.plot_unknowns(x)
        primal_new= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*np.sum(np.abs((gradx)))
                             + 1/(2*delta)*np.linalg.norm((x_new-xk).flatten())**2)

        dual = np.real(-delta/2*np.linalg.norm((-Kyk1_new).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1_new).flatten())
                  - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r_new.flatten())**2 - np.vdot(res.flatten(),r_new.flatten()))

        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<self.irgn_par.lambd*self.irgn_par.tol:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,np.abs(primal-primal_new)/(self.irgn_par.lambd*self.NSlice)))
          self.r = r_new
          self.z1 = z1_new
          return x_new
        if (gap > gap_min*self.irgn_par.stag) and i>1:
          self.r = r_new
          self.z1 = z1_new
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x_new
        if np.abs(gap - gap_min)<self.irgn_par.lambd*self.irgn_par.tol and i>1:
          self.r = r_new
          self.z1 = z1_new
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,np.abs(gap - gap_min)/(self.irgn_par.lambd*self.NSlice)))
          return x_new
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal/(self.irgn_par.lambd*self.NSlice),dual/(self.irgn_par.lambd*self.NSlice),gap/(self.irgn_par.lambd*self.NSlice)))
############# Update variables #################################################
      x = x_new
      Kyk1 = (Kyk1_new)
      Axold =(Ax)
      z1 = (z1_new)
      r =  (r_new)
      tau =  (tau_new)
############# Return final result ##############################################
    self.z1 = z1
    self.r = r
    return x

################################################################################
### 3D-WT optimized with Primal-Dual algorithm with line search ################
### input: initial guess x #####################################################
########## precomputed data combined with constant terms res ###################
########## maximum number of iterations ########################################
### output: optimal values for x ###############################################
################################################################################
  cdef np.ndarray[DTYPE_t,ndim=4] wt_solve_3D(self, np.ndarray[DTYPE_t, ndim=4] x, np.ndarray[DTYPE_t, ndim=5] res, int iters):
    cdef double alpha = self.irgn_par.gamma

    L= 8
    print('L: %f'%(L))



    cdef double tau = 1/np.sqrt(L)
    cdef double tau_new = 0
############# Primal variables #################################################
    cdef np.ndarray[DTYPE_t, ndim=4] xk = x
    cdef np.ndarray[DTYPE_t, ndim=4] x_new = np.zeros_like(x,dtype=DTYPE)
############# Dual variables ###################################################
    cdef np.ndarray[DTYPE_t, ndim=5] r = self.r
    cdef list z1 = self.z1
    cdef np.ndarray[DTYPE_t, ndim=5] r_new = np.zeros_like(res,dtype=DTYPE)
    cdef list z1_new = list(np.copy(z1))#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    cdef list diffz = list(np.copy(z1))
############# Arrays holding intermediate results of the primal update step ####
    cdef np.ndarray[DTYPE_t, ndim=4] Kyk1 = np.zeros_like(x,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] Kyk1_new = np.zeros_like(x,dtype=DTYPE)

############# Arrays holding intermediate results of the dual update step ######
    cdef np.ndarray[DTYPE_t, ndim=5] Ax = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=5] Ax_Axold = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=5] Axold = np.zeros_like(res,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=5] tmp = np.zeros_like(res,dtype=DTYPE)
    cdef list gradx = []
    cdef list gradx_xold = []
    cdef np.ndarray[DTYPE_t_real, ndim=3] scal = np.zeros((self.par.NSlice,self.par.dimX,self.par.dimY),dtype=DTYPE_real)

############# Strong convexity parameter #######################################
    cdef double delta = self.irgn_par.delta
    cdef double mu = 1/delta

############# Line search parameters ###########################################
    cdef double theta_line = 1.0
    cdef double beta_line = 400
    cdef double beta_new = 0
    cdef double mu_line = 0.5
    cdef double delta_line = 1
    cdef double ynorm = 0.0
    cdef double lhs = 0.0

############# Primal, Dual and Gap energy ######################################
    cdef double primal = 0.0
    cdef double primal_new = 0
    cdef double dual = 0.0
    cdef double gap_min = 0.0
    cdef double gap = 0.0

    cdef int i=0
    self.set_scale(x)
############# Precompute intermediate results ##################################
    Axold = self.operator_forward_3D(x)
    for j in range(len(z1)):
      z1[j] = list(z1[j])
    Kyk1 = self.operator_adjoint_3D(r) + self.scale_adj(pywt.waverec2(z1,self.wavelet,self.border))
    for j in range(len(z1)):
      z1[j] = np.array(z1[j])

    gradx_xold = []
    gradxold = pywt.wavedec2(self.scale_fwd(x),self.wavelet,self.border)
    for j in range(len(gradxold)):
      gradxold[j] = np.array(gradxold[j])
    gradx_xold = list(np.copy(gradxold))
############# Start main iterations ############################################
    for i in range(iters):

############# Primal updates with box constraint ###############################
      x_new = ((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta)
      for j in range(len(self.model.constraints)):
        x_new[j,...] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,x_new[j,...]))
        if self.model.constraints[j].real:
          x_new[j,...] = np.real(x_new[j,...])

############# Update step sizes to maximal values ##############################
      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new

############# Precompute Dual intermediate results for line search #############
      gradx = pywt.wavedec2(self.scale_fwd(x_new),self.wavelet,self.border)
      for j in range(len(gradx)):
        gradx_xold[j] = ((np.array(gradx[j]) - np.array(gradxold[j])))
      Ax = self.operator_forward_3D(x_new)
      Ax_Axold = Ax-Axold

############# Start line serach ################################################
      while True:

        theta_line = tau_new/tau
############# Dual Updates #####################################################
        for j in range(len(z1_new)):
          if j == 0:
            z1_new[j] = (np.array(z1[j]) + beta_line*tau_new*( (np.array(gradx[j])+theta_line*np.array(gradx_xold[j])) ))
            z1_new[j] = (np.array(z1_new[j])/np.maximum(1,(np.sqrt(np.sum(np.abs(np.array(z1_new[j]))**2,axis=(0),keepdims=True))/alpha)))
          else:
            z1_new[j] = (np.array(z1[j]) + beta_line*tau_new*( (np.array(gradx[j])+theta_line*np.array(gradx_xold[j])) ))
            z1_new[j] = (np.array(z1_new[j])/np.maximum(1,(np.sqrt(np.sum(np.abs(np.array(z1_new[j]))**2,axis=(0,1),keepdims=True))/alpha)))
          diffz[j] = np.linalg.norm(np.array(z1_new[j])-np.array(z1[j]))
          z1_new[j] = list(z1_new[j])

        tmp = Ax+theta_line*Ax_Axold
        r_new = (( r  + beta_line*tau_new*(tmp) ) - beta_line*tau_new*res)/(1+beta_line*tau_new/self.irgn_par.lambd)

############# Precompute primal intermediate results ###########################
        Kyk1_new = self.operator_adjoint_3D(r_new) + self.scale_adj(pywt.waverec2(z1_new,self.wavelet,self.border))
        for j in range(len(z1_new)):
          z1_new[j] = np.array(z1_new[j])
############# Check line search conditions #####################################
        ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),np.array(diffz).flatten()]))
        lhs = np.sqrt(beta_line)*tau_new*np.linalg.norm(np.concatenate([(Kyk1_new-Kyk1).flatten()]))
        if lhs <= ynorm*delta_line:
            break
        else:
          tau_new = tau_new*mu_line

############# Check if algorithm converged/stagnated and plot result ###########
      if not np.mod(i,20):
        if self.irgn_par.display_iterations:
          self.model.plot_unknowns(x_new)
        gradsum = 0
        for j in range(len(gradxold)):
           gradsum += np.sum(np.abs(np.array(gradx[j])))
        primal_new= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*gradsum
                           + 1/(2*delta)*np.linalg.norm((x_new-xk).flatten())**2)

        dual = np.real(-delta/2*np.linalg.norm((-Kyk1_new).flatten())**2 - np.vdot(xk.flatten(),(-Kyk1_new).flatten())
                    - 1/(2*self.irgn_par.lambd)*np.linalg.norm(r_new.flatten())**2 - np.vdot(res.flatten(),r_new.flatten()))

        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<self.irgn_par.lambd*self.irgn_par.tol:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,np.abs(primal-primal_new)/(self.irgn_par.lambd*self.NSlice)))
          self.r = r_new
          self.z1 = z1_new
          return x_new
        if (gap > gap_min*self.irgn_par.stag) and i>1:
          self.r = r_new
          self.z1 = z1_new
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x_new
        if np.abs(gap - gap_min)<self.irgn_par.lambd*self.irgn_par.tol and i>1:
          self.r = r_new
          self.z1 = z1_new
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,np.abs(gap - gap_min)/(self.irgn_par.lambd*self.NSlice)))
          return x_new
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal/(self.irgn_par.lambd*self.NSlice),dual/(self.irgn_par.lambd*self.NSlice),gap/(self.irgn_par.lambd*self.NSlice)))

############# Update variables #################################################
      x = (x_new)
      Kyk1 = (Kyk1_new)
      Axold =(Ax)
      gradxold = (gradx)
      z1 = (z1_new)
      r =  (r_new)
      tau =  (tau_new)
############# Return final result ##############################################
    self.r = r
    self.z1 = z1
    return x
################################################################################
### 2D-NUFFT forward over all scans and coils ##################################
################################################################################
  cpdef np.ndarray[DTYPE_t,ndim=4] nFT_2D(self, np.ndarray[DTYPE_t,ndim=3] x):
    cdef np.ndarray[DTYPE_t,ndim=3] result = np.zeros((self.NScan,self.NC,self.par.Nproj*self.par.N),dtype=DTYPE)
    cdef list plan = self._plan
    for scan in range(self.NScan):
      result[scan,...] = plan[scan].forward(np.require(x[scan,...]))
    return np.reshape(result,[self.NScan,self.NC,self.par.Nproj,self.par.N])
################################################################################
### 2D-NUFFT adjoint over all scans and coils ##################################
################################################################################
  cpdef np.ndarray[DTYPE_t,ndim=3] nFTH_2D(self, np.ndarray[DTYPE_t,ndim=4] x):
    cdef np.ndarray[DTYPE_t,ndim=3] result = np.zeros((self.NScan,self.par.dimX,self.par.dimY),dtype=DTYPE)
    cdef list plan = self._plan
    cdef np.ndarray[DTYPE_t,ndim=3] x_wrk = np.require(np.reshape(x,(self.NScan,self.NC,self.Nproj*self.N)))
    for scan in range(self.NScan):
            result[scan,...] = plan[scan].adjoint(x_wrk[scan,...])
    return result
################################################################################
### 3D-NUFFT forward over all scans, coils and slices ##########################
################################################################################
  cpdef  np.ndarray[DTYPE_t,ndim=5] nFT_3D(self, np.ndarray[DTYPE_t,ndim=4] x):

    cdef int nscan = self.NScan
    cdef int NC = self.NC
    cdef int NSlice = self.NSlice
    cdef np.ndarray[DTYPE_t,ndim=4] result = np.zeros((nscan,NC,NSlice,self.Nproj*self.N),dtype=DTYPE)
    cdef int scan=0
    cdef int coil=0
    cdef int islice=0
    cdef list plan = self._plan
    for scan in range(nscan):
        for islice in range(NSlice):
          result[scan,:,islice,...] = plan[scan][coil].forward(x[scan,islice,...])
    return np.reshape(result,[nscan,NC,NSlice,self.par.Nproj,self.par.N])
################################################################################
### 3D-NUFFT adjoint over all scans, coils and slices ##########################
################################################################################
  cpdef  np.ndarray[DTYPE_t,ndim=4] nFTH_3D(self, np.ndarray[DTYPE_t,ndim=5] x):
    cdef int nscan = self.NScan
    cdef int NC = self.NC
    cdef int NSlice = self.NSlice
    cdef np.ndarray[DTYPE_t,ndim=4] result = np.zeros((nscan,NSlice,self.par.dimX,self.par.dimY),dtype=DTYPE)
    cdef int scan=0
    cdef int coil=0
    cdef int islice=0
    cdef list plan = self._plan
    cdef np.ndarray[DTYPE_t,ndim=4] x_wrk = np.reshape(x,(nscan,NC,NSlice,self.N*self.Nproj))
    for scan in range(nscan):
      for islice in range(NSlice):
        result[scan,islice,...] = plan[scan][coil].adjoint(np.require(x_wrk[scan,:,islice,...],DTYPE,'C'))
    return result
################################################################################
### Initialize NUFFT Plan ######################################################
################################################################################
  cdef init_plan(self,islice=None):
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
          op.setCoilSens(np.require(self.par.C[:,islice,...],DTYPE,'C'))
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
          op.setCoilSens(np.require(self.par.C[:,j,...],DTYPE,'C'))
          plan[i].append(op)
      self._plan = plan
################################################################################
### Scale before gradient ######################################################
################################################################################
  cpdef set_scale(self,x):
    for j in range(x.shape[0]):
      self.ukscale[j] = np.linalg.norm(x[j,...])
      print('scale %f at uk %i' %(self.ukscale[j],j))
  cpdef scale_fwd(self,x):
    y = np.copy(x)
    for j in range(x.shape[0]):
      y[j,...] /= self.ukscale[j]
      if j==0:
        y[j,...] *= np.max(self.ukscale)/self.ratio
      else:
        y[j,...] *= np.max(self.ukscale)
    return y
  cpdef scale_adj(self,x):
    y = np.copy(x)
    for j in range(x.shape[0]):
      y[j,...] /= self.ukscale[j]
      if j==0:
        y[j,...] *= np.max(self.ukscale)/self.ratio
      else:
        y[j,...] *= np.max(self.ukscale)
    return y

