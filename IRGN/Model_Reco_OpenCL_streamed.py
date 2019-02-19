from __future__ import division

import numpy as np
import time
import sys

import pyopencl as cl
import pyopencl.array as clarray

import Transforms.gridroutines_slicefirst as NUFFT

DTYPE = np.complex64
DTYPE_real = np.float32

class MyAllocator:
    def __init__(self, context, flags=cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR):
        self.context = context
        self.flags = flags

    def __call__(self, size):
        return cl.Buffer(self.context, self.flags, size)

class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build("-cl-mad-enable -cl-fast-relaxed-math")
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel


class Model_Reco:
  def __init__(self,par,trafo=1,imagespace=False, SMS=False):


    par["par_slices"] = 1
    par["overlap"] = 1
    self.overlap = 1
    self.par_slices = 1
    self.par = par
    self.C = np.require(np.transpose(par["C"],[1,0,2,3]),requirements='C')
    self.unknowns_TGV = par["unknowns_TGV"]
    self.unknowns_H1 = par["unknowns_H1"]
    self.unknowns = par["unknowns"]
    self.NSlice = par["NSlice"]
    self.NScan = par["NScan"]
    self.dimX = par["dimX"]
    self.dimY = par["dimY"]
    self.scale = 1
    self.NC = par["NC"]
    self.N = par["N"]
    self.Nproj = par["Nproj"]
    self.dz = 1
    self.fval_min = 0
    self.fval = 0
    self.ctx = par["ctx"]
    self.queue = par["queue"]
    self.gn_res = []
    self.num_dev = len(self.ctx)
    self.NUFFT = []
    self.ukscale = []
    self.prg = []
    self.alloc=[]
    self.ratio = []
    self.tmp_img=[]
    self.tmp_FT = np.zeros((self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE)
    self.tmp_FTH = np.zeros((self.par_slices+self.overlap,self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE)
    self.tmp_adj = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE)
    self.tmp_out = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE)
    for j in range(self.num_dev):
      self.alloc.append(MyAllocator(self.ctx[j]))
      self.tmp_img.append(clarray.zeros(self.queue[3*j],(self.par_slices+self.overlap,self.NScan,self.NC,self.dimY,self.dimX),DTYPE,"C"))
      self.ratio.append(clarray.to_device(self.queue[3*j],(1*np.ones(self.unknowns)).astype(dtype=DTYPE_real)))
      self.NUFFT.append(NUFFT.gridding(self.ctx[j],self.queue[3*j:3*(j+1)-1],par,radial=trafo,SMS=SMS))
      self.ukscale.append(clarray.to_device(self.queue[3*j],np.ones(self.unknowns,dtype=DTYPE_real)))
      self.prg.append(Program(self.ctx[j],  open('./Kernels/OpenCL_Kernels_streamed.c').read()))



  def operator_forward_full(self, out, x, idx=0,idxq=0,wait_for=[]):
    self.tmp_img[idx].add_event(self.eval_fwd_streamed(self.tmp_img[idx],x,idx,idxq,wait_for=self.tmp_img[idx].events+x.events))
    return  self.NUFFT[idx].fwd_NUFFT(out,self.tmp_img[idx],idxq,wait_for=out.events+wait_for+self.tmp_img[idx].events)

  def operator_adjoint_full(self, out, x,z,idx=0,idxq=0, last=0,wait_for=[]):
    self.tmp_img[idx].add_event(self.NUFFT[idx].adj_NUFFT(self.tmp_img[idx],x,idxq,wait_for=wait_for+x.events+self.tmp_img[idx].events))
    return self.prg[idx].update_Kyk1(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None,
                                 out.data, self.tmp_img[idx].data, self.coil_buf_part[idx+idxq*self.num_dev].data, self.grad_buf_part[idx+idxq*self.num_dev].data, z.data, np.int32(self.NC),
                                 np.int32(self.NScan), self.ukscale[idx].data,
                                 np.float32(np.amax(self.ukscale[idx].get())),self.ratio[idx].data, np.int32(self.unknowns),np.int32(last),np.float32(self.dz),
                                 wait_for=self.tmp_img[idx].events+out.events+z.events+wait_for)



  def eval_const(self):
    num_const = (len(self.model.constraints))
    min_const = np.zeros((num_const),dtype=np.float32)
    max_const = np.zeros((num_const),dtype=np.float32)
    real_const = np.zeros((num_const),dtype=np.int32)
    for j in range(num_const):
        min_const[j] = np.float32(self.model.constraints[j].min)
        max_const[j] = np.float32(self.model.constraints[j].max)
        real_const[j] = np.int32(self.model.constraints[j].real)

    self.min_const = []
    self.max_const = []
    self.real_const = []
    for j in range(self.num_dev):
      self.min_const.append(clarray.to_device(self.queue[3*j], min_const))
      self.max_const.append(clarray.to_device(self.queue[3*j], max_const))
      self.real_const.append(clarray.to_device(self.queue[3*j], real_const))


  def f_grad(self,grad, u, idx=0,idxq=0, wait_for=[]):
    return self.prg[idx].gradient(self.queue[3*idx+idxq],  (self.overlap+self.par_slices,self.dimY,self.dimX), None, grad.data, u.data,
                np.int32(self.unknowns),
                self.ukscale[idx].data,  np.float32(np.amax(self.ukscale[idx].get())),self.ratio[idx].data, np.float32(self.dz),
                wait_for=grad.events + u.events + wait_for)

  def bdiv(self,div, u, idx=0,idxq=0,last=0,wait_for=[]):
    return self.prg[idx].divergence(self.queue[3*idx+idxq], (self.overlap+self.par_slices,self.dimY,self.dimX), None, div.data, u.data,
                np.int32(self.unknowns),
                self.ukscale[idx].data, np.float32(np.amax(self.ukscale[idx].get())),self.ratio[idx].data, np.int32(last),np.float32(self.dz),
                wait_for=div.events + u.events + wait_for)

  def sym_grad(self,sym, w,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].sym_grad(self.queue[3*idx+idxq], (self.overlap+self.par_slices,self.dimY,self.dimX), None, sym.data, w.data,
                np.int32(self.unknowns),np.float32(self.dz),
                wait_for=sym.events + w.events + wait_for)

  def sym_bdiv(self,div, u, idx=0,idxq=0,first=0,wait_for=[]):
    return self.prg[idx].sym_divergence(self.queue[3*idx+idxq], (self.overlap+self.par_slices,self.dimY,self.dimX), None, div.data, u.data,
                np.int32(self.unknowns),np.int32(first),np.float32(self.dz),
                wait_for=div.events + u.events + wait_for)
  def update_Kyk2(self,div, u, z,  idx=0,idxq=0,first=0,wait_for=[]):
    return self.prg[idx].update_Kyk2(self.queue[3*idx+idxq], (self.overlap+self.par_slices,self.dimY,self.dimX), None, div.data, u.data, z.data,
                np.int32(self.unknowns),np.int32(first),np.float32(self.dz),
                wait_for=div.events + u.events + z.events+wait_for)

  def update_primal(self, x_new, x, Kyk, xk, tau, delta,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_primal(self.queue[3*idx+idxq],(self.overlap+self.par_slices,self.dimY,self.dimX), None, x_new.data, x.data, Kyk.data, xk.data, np.float32(tau),
                                  np.float32(tau/delta), np.float32(1/(1+tau/delta)), self.min_const[idx].data, self.max_const[idx].data,
                                  self.real_const[idx].data, np.int32(self.unknowns),
                                  wait_for=x_new.events + x.events + Kyk.events+ xk.events+wait_for
                                  )
  def update_z1(self, z_new, z, gx, gx_, vx, vx_, sigma, theta, alpha,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_z1(self.queue[3*idx+idxq], (self.overlap+self.par_slices,self.dimY,self.dimX), None, z_new.data, z.data, gx.data, gx_.data, vx.data, vx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/alpha), np.int32(self.unknowns),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ vx.events+ vx_.events+wait_for
                                  )
  def update_z1_tv(self, z_new, z, gx, gx_, sigma, theta, alpha,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_z1_tv(self.queue[3*idx+idxq], (self.overlap+self.par_slices,self.dimY,self.dimX), None, z_new.data, z.data, gx.data, gx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/alpha), np.int32(self.unknowns),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ wait_for
                                  )
  def update_z2(self, z_new, z, gx, gx_, sigma, theta, beta,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_z2(self.queue[3*idx+idxq], (self.overlap+self.par_slices,self.dimY,self.dimX), None, z_new.data, z.data, gx.data, gx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/beta),  np.int32(self.unknowns),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ wait_for
                                  )
  def update_r(self, r_new, r, A, A_, res, sigma, theta, lambd,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_r(self.queue[3*idx+idxq], (r.size,), None, r_new.data, r.data, A.data, A_.data, res.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/(1+sigma/lambd)),
                                  wait_for= r_new.events + r.events + A.events+ A_.events+ wait_for
                                  )
  def update_v(self, v_new, v, Kyk2, tau,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_v(self.queue[3*idx+idxq], (v[...,0].size,), None,
                             v_new.data, v.data, Kyk2.data, np.float32(tau),
                                  wait_for= v_new.events + v.events + Kyk2.events+ wait_for
                                  )
  def update_primal_explicit(self, x_new, x, Kyk, xk, ATd, tau, delta, lambd, idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_primal_explicit(self.queue[3*idx+idxq], (self.overlap+self.par_slices,self.dimY,self.dimX), None, x_new.data, x.data, Kyk.data, xk.data, ATd.data, np.float32(tau),
                                  np.float32(1/delta), np.float32(lambd), self.min_const[idx].data, self.max_const[idx].data,
                                  self.real_const[idx].data, np.int32(self.unknowns),
                                  wait_for=x_new.events + x.events + Kyk.events+ xk.events+ATd.events+wait_for
                                  )
################################################################################
### Scale before gradient ######################################################
################################################################################
  def set_scale(self,x):
    for i in range(self.num_dev):
      for j in range(self.unknowns):
        self.ukscale[i][j] = np.linalg.norm(x[:,j,...])
    grad = clarray.to_device(self.queue[3*i],np.zeros_like(self.z1))
    x = clarray.to_device(self.queue[3*i],x)
    grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))
    grad = grad.get()
    for j in range(self.unknowns):
      scale = np.linalg.norm(grad[:,j,...])/np.linalg.norm(grad[:,0,...])
      if np.isfinite(scale) and scale>1e-4:
        self.ratio[i][j] = scale



################################################################################
### Start a 3D Reconstruction, set TV to True to perform TV instead of TGV######
### Precompute Model and Gradient values for xk ################################
### Call inner optimization ####################################################
### input: bool to switch between TV (1) and TGV (0) regularization ############
### output: optimal value of x #################################################
################################################################################
  def execute_3D(self, TV=0):
   self.FT = self.FT_streamed
   iters = self.irgn_par["start_iters"]


   self.r = np.zeros_like(self.data,dtype=DTYPE)
   self.r = np.require(np.transpose(self.r,[2,0,1,3,4]),requirements='C')
   self.z1 = np.zeros(([self.NSlice,self.unknowns,self.dimY,self.dimX,4]),dtype=DTYPE)


   self.result = np.zeros((self.irgn_par["max_gn_it"]+1,self.unknowns,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
   self.result[0,:,:,:,:] = np.copy(self.model.guess)

   result = np.copy(self.model.guess)

   self.v = np.zeros(([self.NSlice,self.unknowns,self.dimY,self.dimX,4]),dtype=DTYPE)
   self.z2 = np.zeros(([self.NSlice,self.unknowns,self.dimY,self.dimX,8]),dtype=DTYPE)
   for i in range(self.irgn_par["max_gn_it"]):
    start = time.time()
    self.grad_x = np.nan_to_num(self.model.execute_gradient(result))

    for uk in range(self.unknowns-1):
      scale = np.linalg.norm(np.abs(self.grad_x[0,...]))/np.linalg.norm(np.abs(self.grad_x[uk+1,...]))
      self.model.constraints[uk+1].update(scale)
      result[uk+1,...] = result[uk+1,...]*self.model.uk_scale[uk+1]
      self.model.uk_scale[uk+1] = self.model.uk_scale[uk+1]*scale
      result[uk+1,...] = result[uk+1,...]/self.model.uk_scale[uk+1]

    self.step_val = np.nan_to_num(self.model.execute_forward(result))
    self.step_val = np.require(np.transpose(self.step_val,[1,0,2,3]),requirements='C')
    self.grad_x = np.nan_to_num(self.model.execute_gradient(result))
    self.grad_x = np.require(np.transpose(self.grad_x,[2,0,1,3,4]),requirements='C')

    self.set_scale(np.require(np.transpose(result,[1,0,2,3]),requirements='C'))

    result = self.irgn_solve_3D(result, iters, self.data,TV)
    self.result[i+1,...] = self.model.rescale(result)


    iters = np.fmin(iters*2,self.irgn_par["max_iters"])
    self.irgn_par["gamma"] = np.maximum(self.irgn_par["gamma"]*self.irgn_par["gamma_dec"],self.irgn_par["gamma_min"])
    self.irgn_par["delta"] = np.minimum(self.irgn_par["delta"]*self.irgn_par["delta_inc"],self.irgn_par["delta_max"])

    end = time.time()-start
    self.gn_res.append(self.fval)
    print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
    print("-"*80)
    if (np.abs(self.fval_min-self.fval) < self.irgn_par["lambd"]*self.irgn_par["tol"]) and i>0:
      print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-self.fval)/(self.irgn_par["lambd"]*self.NSlice)))
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
  def irgn_solve_3D(self,x,iters, data, TV=0):

    x_old = x
    x = np.require(np.transpose(x,[1,0,2,3]),requirements='C')
    data = np.require(np.transpose(data,[2,0,1,3,4]),requirements='C')


    b = np.zeros(data.shape,dtype=DTYPE)
    DGk = np.zeros_like(data.astype(DTYPE))
    self.FT(b,self.step_val[:,:,None,...]*self.C[:,None,...])
    self.operator_forward_streamed(DGk,x)
    res = data - b + DGk

    x = self.tgv_solve_3D(x,res,iters)
    x = clarray.to_device(self.queue[0],x)
    v = clarray.to_device(self.queue[0],self.v)
    grad = clarray.to_device(self.queue[0],np.zeros_like(self.z1))
    sym_grad = clarray.to_device(self.queue[0],np.zeros_like(self.z2))
    grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))
    sym_grad.add_event(self.sym_grad(sym_grad,v,wait_for=sym_grad.events+v.events))
    x = np.require(np.transpose(x.get(),[1,0,2,3]),requirements='C')
    self.FT(b,np.require(np.transpose(self.model.execute_forward(x),[1,0,2,3]),requirements='C')[:,:,None,...]*self.C[:,None,...])
    self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - b)**2
            +self.irgn_par["gamma"]*np.sum(np.abs(grad.get()-self.v))
            +self.irgn_par["gamma"]*(2)*np.sum(np.abs(sym_grad.get()))
            +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2)

    print("-"*80)
    print ("Function value after GN-Step: %f" %(self.fval/(self.irgn_par["lambd"]*self.NSlice)))

    return x

  def tgv_solve_3D(self, x,res, iters):

    alpha = self.irgn_par["gamma"]
    beta = self.irgn_par["gamma"]*2


    L = np.float32(0.5*(18.0 + np.sqrt(33)))
#    print('L: %f'%(L))


    tau = np.float32(1/np.sqrt(L))
    tau_new =np.float32(0)

#    self.set_scale(x)
    xk = x.copy()
    x_new = np.zeros_like(x)

    r = np.zeros_like(self.r)#np.zeros_like(res,dtype=DTYPE)
    r_new = np.zeros_like(r)
    z1 = np.zeros_like(self.z1)#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    z1_new =  np.zeros_like(z1)#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    z2 = np.zeros_like(self.z2)#np.zeros(([self.unknowns,3,self.dimY,self.dimX]),dtype=DTYPE)
    z2_new =  np.zeros_like(z2)
    v = np.zeros_like(self.v)#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    v_new =  np.zeros_like(v)
    res = (res).astype(DTYPE)


    delta = self.irgn_par["delta"]
    mu = 1/delta
    theta_line = np.float32(1.0)
    beta_line = np.float32(400)
    beta_new = np.float32(0)
    mu_line =np.float32( 0.5)
    delta_line = np.float32(1)
    ynorm = np.float32(0.0)
    lhs = np.float32(0.0)
    primal = np.float32(0.0)
    primal_new = np.float32(0)
    dual = np.float32(0.0)
    gap_min = np.float32(0.0)



    self.eval_const()


    Kyk1 = np.zeros_like(x)
    Kyk1_new = np.zeros_like(x)
    Kyk2 = np.zeros_like(z1)
    Kyk2_new = np.zeros_like(z1)
    gradx = np.zeros_like(z1)
    gradx_xold = np.zeros_like(z1)
    symgrad_v = np.zeros_like(z2)
    symgrad_v_vold = np.zeros_like(z2)

    Axold = np.zeros_like(res)
    Ax = np.zeros_like(res)

#### Allocate temporary Arrays
    Axold_part = []
    Kyk1_part = []
    Kyk2_part = []
    for i in range(2*self.num_dev):
      Axold_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
      Kyk1_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
      Kyk2_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))


##### Warmup
    x_part = []
    r_part = []
    z1_part = []
    j=0
    last=0
    for i in range(self.num_dev):
      idx_start = self.NSlice-((i+1)*self.par_slices)-self.overlap
      idx_stop = self.NSlice-(i*self.par_slices)
      if idx_stop==self.NSlice:
        last=1
      else:
        last=0
      x_part.append(clarray.to_device(self.queue[3*i], x[idx_start:idx_stop,...],allocator=self.alloc[i],async_=True))# ))
      r_part.append(clarray.to_device(self.queue[3*i], r[idx_start:idx_stop,...],allocator=self.alloc[i],async_=True))# ))
      z1_part.append(clarray.to_device(self.queue[3*i], z1[idx_start:idx_stop,...],allocator=self.alloc[i],async_=True))# ))
      self.coil_buf_part[i].add_event(cl.enqueue_copy(self.queue[3*i],self.coil_buf_part[i].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[i].events,is_blocking=False))
      self.grad_buf_part[i].add_event(cl.enqueue_copy(self.queue[3*i],self.grad_buf_part[i].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[i].events,is_blocking=False))
      Axold_part[i].add_event(self.operator_forward_full(Axold_part[i],x_part[i],i,0))
      Kyk1_part[i].add_event(self.operator_adjoint_full(Kyk1_part[i],r_part[i],z1_part[i],i,0,last))
    last = 0
    for i in range(self.num_dev):
      idx_start = self.NSlice-((i+2+self.num_dev-1)*self.par_slices)
      idx_stop = self.NSlice-((i+1+self.num_dev-1)*self.par_slices)
      if idx_start==0:
        idx_stop+=self.overlap
      else:
        idx_start-=self.overlap
      x_part.append(clarray.to_device(self.queue[3*i+1], x[idx_start:idx_stop,...],allocator=self.alloc[i],async_=True))# ))
      r_part.append(clarray.to_device(self.queue[3*i+1], r[idx_start:idx_stop,...],allocator=self.alloc[i],async_=True))# ))
      z1_part.append(clarray.to_device(self.queue[3*i+1], z1[idx_start:idx_stop,...],allocator=self.alloc[i],async_=True))# ))
      self.coil_buf_part[self.num_dev+i].add_event(cl.enqueue_copy(self.queue[3*i+1],self.coil_buf_part[self.num_dev+i].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[self.num_dev+i].events,is_blocking=False))
      self.grad_buf_part[self.num_dev+i].add_event(cl.enqueue_copy(self.queue[3*i+1],self.grad_buf_part[self.num_dev+i].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[self.num_dev+i].events,is_blocking=False))

      Axold_part[i+self.num_dev].add_event(self.operator_forward_full(Axold_part[i+self.num_dev],x_part[self.num_dev+i],i,1))
      Kyk1_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_part[i+self.num_dev],r_part[self.num_dev+i],z1_part[self.num_dev+i],i,1,last))


#### Stream
    for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
        last = 0
        for i in range(self.num_dev):
          ### Get Data
          idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices)-self.overlap
          idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices))
          Axold_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Axold[idx_start:idx_stop,...],Axold_part[i].data,wait_for=Axold_part[i].events,is_blocking=False))
          Kyk1_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk1[idx_start:idx_stop,...],Kyk1_part[i].data,wait_for=Kyk1_part[i].events,is_blocking=False))
          ### Put Data
          idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices)-self.overlap
          idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices))


          x_part[i].add_event(cl.enqueue_copy(self.queue[3*i],x_part[i].data,x[idx_start:idx_stop,...],wait_for=x_part[i].events,is_blocking=False))
          r_part[i].add_event(cl.enqueue_copy(self.queue[3*i],r_part[i].data,r[idx_start:idx_stop,...],wait_for=r_part[i].events,is_blocking=False))
          z1_part[i].add_event(cl.enqueue_copy(self.queue[3*i],z1_part[i].data,z1[idx_start:idx_stop,...],wait_for=z1_part[i].events,is_blocking=False))
          self.coil_buf_part[i].add_event(cl.enqueue_copy(self.queue[3*i],self.coil_buf_part[i].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[i].events,is_blocking=False))
          self.grad_buf_part[i].add_event(cl.enqueue_copy(self.queue[3*i],self.grad_buf_part[i].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[i].events,is_blocking=False))
          Axold_part[i].add_event(self.operator_forward_full(Axold_part[i],x_part[i],i,0))
          Kyk1_part[i].add_event(self.operator_adjoint_full(Kyk1_part[i],r_part[i],z1_part[i],i,0,last))
        for i in range(self.num_dev):
          idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)-self.overlap
          idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)
          Axold_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Axold[idx_start:idx_stop,...],Axold_part[i+self.num_dev].data,wait_for=Axold_part[i+self.num_dev].events,is_blocking=False))
          Kyk1_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk1[idx_start:idx_stop,...],Kyk1_part[i+self.num_dev].data,wait_for=Kyk1_part[i+self.num_dev].events,is_blocking=False))
          idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices)
          idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices)
          if idx_start==0:
            idx_stop+=self.overlap
          else:
            idx_start-=self.overlap
          x_part[self.num_dev+i].add_event(cl.enqueue_copy(self.queue[3*i+1],x_part[i].data,x[idx_start:idx_stop,...],wait_for=x_part[i].events,is_blocking=False))
          r_part[self.num_dev+i].add_event(cl.enqueue_copy(self.queue[3*i+1],r_part[i].data,r[idx_start:idx_stop,...],wait_for=r_part[i].events,is_blocking=False))
          z1_part[self.num_dev+i].add_event(cl.enqueue_copy(self.queue[3*i+1],z1_part[i].data,z1[idx_start:idx_stop,...],wait_for=z1_part[i].events,is_blocking=False))

          self.coil_buf_part[self.num_dev+i].add_event(cl.enqueue_copy(self.queue[3*i+1],self.coil_buf_part[self.num_dev+i].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[self.num_dev+i].events,is_blocking=False))
          self.grad_buf_part[self.num_dev+i].add_event(cl.enqueue_copy(self.queue[3*i+1],self.grad_buf_part[self.num_dev+i].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[self.num_dev+i].events,is_blocking=False))
          Axold_part[i+self.num_dev].add_event(self.operator_forward_full(Axold_part[i+self.num_dev],x_part[self.num_dev+i],i,1))
          Kyk1_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_part[i+self.num_dev],r_part[self.num_dev+i],z1_part[self.num_dev+i],i,1,last))
#### Collect last block
    if j<2*self.num_dev:
      j = 2*self.num_dev
    else:
      j+=1
    for i in range(self.num_dev):
      idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices)-self.overlap
      idx_stop =  self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices))
      Axold_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Axold[idx_start:idx_stop,...],Axold_part[i].data,wait_for=Axold_part[i].events,is_blocking=False))
      Kyk1_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk1[idx_start:idx_stop,...],Kyk1_part[i].data,wait_for=Kyk1_part[i].events,is_blocking=False))
      idx_start = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)
      idx_stop = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)
      if idx_start==0:
        idx_stop+=self.overlap
      else:
        idx_start-=self.overlap
      Axold_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Axold[idx_start:idx_stop,...],Axold_part[i+self.num_dev].data,wait_for=Axold_part[i+self.num_dev].events,is_blocking=False))
      Kyk1_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk1[idx_start:idx_stop,...],Kyk1_part[i+self.num_dev].data,wait_for=Kyk1_part[i+self.num_dev].events,is_blocking=False))
    self.queue[3*i+2].finish()

##### Warmup
    z2_part = []
    j=0
    first = 0
    for i in range(self.num_dev):
      idx_start = i*self.par_slices
      idx_stop = (i+1)*self.par_slices+self.overlap
      z1_part[i].add_event(cl.enqueue_copy(self.queue[3*i],z1_part[i].data,z1[idx_start:idx_stop,...],wait_for=z1_part[i].events,is_blocking=False))
      z2_part.append(clarray.to_device(self.queue[3*i], z2[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
      if i == 0:
        first=1
      else:
        first=0
      Kyk2_part[i].add_event((self.update_Kyk2(Kyk2_part[i],z2_part[i],z1_part[i],i,0,first)))
    first = 0
    for i in range(self.num_dev):
      idx_start = (i+1+self.num_dev-1)*self.par_slices
      idx_stop = (i+2+self.num_dev-1)*self.par_slices
      if idx_stop == self.NSlice:
        idx_start -=self.overlap
      else:
        idx_stop +=self.overlap

      z1_part[self.num_dev+i].add_event(cl.enqueue_copy(self.queue[3*i+1],z1_part[i].data,z1[idx_start:idx_stop,...],wait_for=z1_part[i].events,is_blocking=False))
      z2_part.append(clarray.to_device(self.queue[3*i+1], z2[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
      Kyk2_part[i+self.num_dev].add_event(self.update_Kyk2(Kyk2_part[i+self.num_dev],z2_part[self.num_dev+i],z1_part[self.num_dev+i],i,1,first))
#### Stream
    for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
        for i in range(self.num_dev):
          ### Get Data
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices+self.overlap
          Kyk2_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk2[idx_start:idx_stop,...],Kyk2_part[i].data,wait_for=Kyk2_part[i].events,is_blocking=False))
          ### Put Data
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices+self.overlap
          z1_part[i].add_event(cl.enqueue_copy(self.queue[3*i],z1_part[i].data,z1[idx_start:idx_stop,...],wait_for=z1_part[i].events,is_blocking=False))
          z2_part[i].add_event(cl.enqueue_copy(self.queue[3*i],z2_part[i].data,z2[idx_start:idx_stop,...],wait_for=z2_part[i].events,is_blocking=False))
          Kyk2_part[i].add_event(self.update_Kyk2(Kyk2_part[i],z2_part[i],z1_part[i],i,0,first))
        for i in range(self.num_dev):
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices+self.overlap
          Kyk2_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk2[idx_start:idx_stop,...],Kyk2_part[+self.num_dev].data,wait_for=Kyk2_part[i+self.num_dev].events,is_blocking=False))
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
          if idx_stop == self.NSlice:
            idx_start -=self.overlap
          else:
            idx_stop +=self.overlap
          z1_part[self.num_dev+i].add_event(cl.enqueue_copy(self.queue[3*i+1],z1_part[self.num_dev+i].data,z1[idx_start:idx_stop,...],wait_for=z1_part[self.num_dev+i].events,is_blocking=False))
          z2_part[self.num_dev+i].add_event(cl.enqueue_copy(self.queue[3*i+1],z2_part[self.num_dev+i].data,z2[idx_start:idx_stop,...],wait_for=z2_part[self.num_dev+i].events,is_blocking=False))
          Kyk2_part[i+self.num_dev].add_event(self.update_Kyk2(Kyk2_part[i+self.num_dev],z2_part[i],z1_part[self.num_dev+i],i,1,first))
#### Collect last block
    if j<2*self.num_dev:
      j = 2*self.num_dev
    else:
      j+=1
    for i in range(self.num_dev):
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
      Kyk2_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk2[idx_start:idx_stop+self.overlap,...],Kyk2_part[i].data,wait_for=Kyk2_part[i].events,is_blocking=False))

      self.queue[3*i+1].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices

      if idx_stop == self.NSlice:
        idx_start-=self.overlap
      else:
        idx_stop+=self.overlap
      Kyk2_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk2[idx_start:idx_stop,...],Kyk2_part[+self.num_dev].data,wait_for=Kyk2_part[i+self.num_dev].events,is_blocking=False))
    self.queue[3*i+2].finish()




    xk_part = []
    v_part = []
    Ax_old_part = []
    res_part =  []
    for i in range(self.num_dev):
      idx_start = i*self.par_slices
      idx_stop = (i+1)*self.par_slices
      if idx_stop == self.NSlice:
        idx_start -=self.overlap
      else:
        idx_stop +=self.overlap
      xk_part.append(clarray.to_device(self.queue[3*i], xk[idx_start:idx_stop,...],allocator=self.alloc[i]))
      v_part.append(clarray.to_device(self.queue[3*i], v[idx_start:idx_stop,...] ,allocator=self.alloc[i]))
      Ax_old_part.append(clarray.to_device(self.queue[3*i], Axold[idx_start:idx_stop,...] ,allocator=self.alloc[i]))
      res_part.append(clarray.to_device(self.queue[3*i], res[idx_start:idx_stop,...] ,allocator=self.alloc[i] ))
    for i in range(self.num_dev):
      idx_start = (i+1+self.num_dev-1)*self.par_slices
      idx_stop = (i+2+self.num_dev-1)*self.par_slices
      if idx_stop == self.NSlice:
        idx_start -=self.overlap
      else:
        idx_stop +=self.overlap
      xk_part.append(clarray.to_device(self.queue[3*i+1],xk[idx_start:idx_stop,...],allocator=self.alloc[i]))
      v_part.append(clarray.to_device(self.queue[3*i+1], v[idx_start:idx_stop,...] ,allocator=self.alloc[i]))
      Ax_old_part.append(clarray.to_device(self.queue[3*i+1], Axold[idx_start:idx_stop,...],allocator=self.alloc[i] ))
      res_part.append(clarray.to_device(self.queue[3*i+1], res[idx_start:idx_stop,...],allocator=self.alloc[i]  ))




    for myit in range(iters):
  #### Allocate temporary Arrays
      if myit == 0:
        x_new_part = []
        Ax_part = []
        v_new_part = []
        gradx_part= []
        gradx_xold_part = []
        symgrad_v_part = []
        symgrad_v_vold_part = []
        for i in range(2*self.num_dev):
          x_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
          Ax_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
          v_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
          gradx_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
          gradx_xold_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
          symgrad_v_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,8),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
          symgrad_v_vold_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,8),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
      j=0
      for i in range(self.num_dev):
        idx_start = i*self.par_slices
        idx_stop = (i+1)*self.par_slices+self.overlap
        x_part[i].add_event(cl.enqueue_copy(self.queue[3*i],x_part[i].data,x[idx_start:idx_stop,...],wait_for=x_part[i].events,is_blocking=False))
        Kyk1_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Kyk1_part[i].data,Kyk1[idx_start:idx_stop,...],wait_for=Kyk1_part[i].events,is_blocking=False))
        xk_part[i].add_event(cl.enqueue_copy(self.queue[3*i],xk_part[i].data,xk[idx_start:idx_stop,...],wait_for=xk_part[i].events,is_blocking=False))
        self.coil_buf_part[i].add_event(cl.enqueue_copy(self.queue[3*i],self.coil_buf_part[i].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[i].events,is_blocking=False))
        self.grad_buf_part[i].add_event(cl.enqueue_copy(self.queue[3*i],self.grad_buf_part[i].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[i].events,is_blocking=False))
        x_new_part[i].add_event(self.update_primal(x_new_part[i],x_part[i],Kyk1_part[i],xk_part[i],tau,delta,i,0))
        gradx_part[i].add_event(self.f_grad(gradx_part[i],x_new_part[i],i,0))
        gradx_xold_part[i].add_event(self.f_grad(gradx_xold_part[i],x_part[i],i,0))
        Ax_part[i].add_event(self.operator_forward_full(Ax_part[i],x_new_part[i],i,0))
      for i in range(self.num_dev):
        idx_start = (i+1+self.num_dev-1)*self.par_slices
        idx_stop = (i+2+self.num_dev-1)*self.par_slices
        if idx_stop == self.NSlice:
          idx_start -=self.overlap
        else:
          idx_stop +=self.overlap
        x_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],x_part[i+self.num_dev].data,x[idx_start:idx_stop,...],wait_for=x_part[i+self.num_dev].events,is_blocking=False))
        Kyk1_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Kyk1_part[i+self.num_dev].data,Kyk1[idx_start:idx_stop,...],wait_for=Kyk1_part[i+self.num_dev].events,is_blocking=False))
        xk_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],xk_part[i+self.num_dev].data,xk[idx_start:idx_stop,...],wait_for=xk_part[i+self.num_dev].events,is_blocking=False))
        self.coil_buf_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],self.coil_buf_part[i+self.num_dev].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[i+self.num_dev].events,is_blocking=False))
        self.grad_buf_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],self.grad_buf_part[i+self.num_dev].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[i+self.num_dev].events,is_blocking=False))
        x_new_part[i+self.num_dev].add_event(self.update_primal(x_new_part[i+self.num_dev],x_part[self.num_dev+i],Kyk1_part[self.num_dev+i],xk_part[self.num_dev+i],tau,delta,i,1))
        gradx_part[i+self.num_dev].add_event(self.f_grad(gradx_part[i+self.num_dev],x_new_part[i+self.num_dev],i,1))
        gradx_xold_part[i+self.num_dev].add_event(self.f_grad(gradx_xold_part[i+self.num_dev],x_part[self.num_dev+i],i,1))
        Ax_part[i+self.num_dev].add_event(self.operator_forward_full(Ax_part[i+self.num_dev],x_new_part[i+self.num_dev],i,1))
  #### Stream
      for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
          for i in range(self.num_dev):
            ### Get Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices+self.overlap
            x_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],x_new[idx_start:idx_stop,...],x_new_part[i].data,wait_for=x_new_part[i].events,is_blocking=False))
            gradx_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],gradx[idx_start:idx_stop,...],gradx_part[i].data,wait_for=gradx_part[i].events,is_blocking=False))
            gradx_xold_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],gradx_xold[idx_start:idx_stop,...],gradx_xold_part[i].data,wait_for=gradx_xold_part[i].events,is_blocking=False))
            Ax_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Ax[idx_start:idx_stop,...],Ax_part[i].data,wait_for=Ax_part[i].events,is_blocking=False))
            ### Put Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices+self.overlap
            x_part[i].add_event(cl.enqueue_copy(self.queue[3*i],x_part[i].data,x[idx_start:idx_stop,...],wait_for=x_part[i].events,is_blocking=False))
            Kyk1_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Kyk1_part[i].data,Kyk1[idx_start:idx_stop,...],wait_for=Kyk1_part[i].events,is_blocking=False))
            xk_part[i].add_event(cl.enqueue_copy(self.queue[3*i],xk_part[i].data,xk[idx_start:idx_stop,...],wait_for=xk_part[i].events,is_blocking=False))
            self.coil_buf_part[i].add_event(cl.enqueue_copy(self.queue[3*i],self.coil_buf_part[i].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[i].events,is_blocking=False))
            self.grad_buf_part[i].add_event(cl.enqueue_copy(self.queue[3*i],self.grad_buf_part[i].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[i].events,is_blocking=False))
            x_new_part[i].add_event(self.update_primal(x_new_part[i],x_part[i],Kyk1_part[i],xk_part[i],tau,delta,i,0))
            gradx_part[i].add_event(self.f_grad(gradx_part[i],x_new_part[i],i,0))
            gradx_xold_part[i].add_event(self.f_grad(gradx_xold_part[i],x_part[i],i,0))
            Ax_part[i].add_event(self.operator_forward_full(Ax_part[i],x_new_part[i],i,0))
          for i in range(self.num_dev):
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices+self.overlap
            x_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],x_new[idx_start:idx_stop,...],x_new_part[i+self.num_dev].data,wait_for=x_new_part[i+self.num_dev].events,is_blocking=False))
            gradx_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],gradx[idx_start:idx_stop,...],gradx_part[i+self.num_dev].data,wait_for=gradx_part[i+self.num_dev].events,is_blocking=False))
            gradx_xold_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],gradx_xold[idx_start:idx_stop,...],gradx_xold_part[i+self.num_dev].data,wait_for=gradx_xold_part[i+self.num_dev].events,is_blocking=False))
            Ax_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Ax[idx_start:idx_stop,...],Ax_part[i+self.num_dev].data,wait_for=Ax_part[i+self.num_dev].events,is_blocking=False))
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            if idx_stop == self.NSlice:
              idx_start -=self.overlap
            else:
              idx_stop +=self.overlap
            x_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],x_part[i+self.num_dev].data,x[idx_start:idx_stop,...],wait_for=x_part[i+self.num_dev].events,is_blocking=False))
            Kyk1_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Kyk1_part[i+self.num_dev].data,Kyk1[idx_start:idx_stop,...],wait_for=Kyk1_part[i+self.num_dev].events,is_blocking=False))
            xk_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],xk_part[i+self.num_dev].data,xk[idx_start:idx_stop,...],wait_for=xk_part[i+self.num_dev].events,is_blocking=False))
            self.coil_buf_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],self.coil_buf_part[i+self.num_dev].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[i+self.num_dev].events,is_blocking=False))
            self.grad_buf_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],self.grad_buf_part[i+self.num_dev].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[i+self.num_dev].events,is_blocking=False))
            x_new_part[i+self.num_dev].add_event(self.update_primal(x_new_part[i+self.num_dev],x_part[self.num_dev+i],Kyk1_part[self.num_dev+i],xk_part[self.num_dev+i],tau,delta,i,1))
            gradx_part[i+self.num_dev].add_event(self.f_grad(gradx_part[i+self.num_dev],x_new_part[i+self.num_dev],i,1))
            gradx_xold_part[i+self.num_dev].add_event(self.f_grad(gradx_xold_part[i+self.num_dev],x_part[self.num_dev+i],i,1))
            Ax_part[i+self.num_dev].add_event(self.operator_forward_full(Ax_part[i+self.num_dev],x_new_part[i+self.num_dev],i,1))
  #### Collect last block
      if j<2*self.num_dev:
        j = 2*self.num_dev
      else:
        j+=1
      for i in range(self.num_dev):
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices+self.overlap
        x_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],x_new[idx_start:idx_stop,...],x_new_part[i].data,wait_for=x_new_part[i].events,is_blocking=False))
        gradx_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],gradx[idx_start:idx_stop,...],gradx_part[i].data,wait_for=gradx_part[i].events,is_blocking=False))
        gradx_xold_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],gradx_xold[idx_start:idx_stop,...],gradx_xold_part[i].data,wait_for=gradx_xold_part[i].events,is_blocking=False))
        Ax_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Ax[idx_start:idx_stop,...],Ax_part[i].data,wait_for=Ax_part[i].events,is_blocking=False))
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        if idx_stop == self.NSlice:
          idx_start-=self.overlap
        else:
          idx_stop+=self.overlap
        x_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],x_new[idx_start:idx_stop,...],x_new_part[i+self.num_dev].data,wait_for=x_new_part[i+self.num_dev].events,is_blocking=False))
        gradx_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],gradx[idx_start:idx_stop,...],gradx_part[i+self.num_dev].data,wait_for=gradx_part[i+self.num_dev].events,is_blocking=False))
        gradx_xold_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],gradx_xold[idx_start:idx_stop,...],gradx_xold_part[i+self.num_dev].data,wait_for=gradx_xold_part[i+self.num_dev].events,is_blocking=False))
        Ax_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Ax[idx_start:idx_stop,...],Ax_part[i+self.num_dev].data,wait_for=Ax_part[i+self.num_dev].events,is_blocking=False))
      self.queue[3*i+2].finish()


      j=0
      for i in range(self.num_dev):
        idx_start = self.NSlice-((i+1)*self.par_slices)-self.overlap
        idx_stop = self.NSlice-(i*self.par_slices)
        v_part[i].add_event(cl.enqueue_copy(self.queue[3*i],v_part[i].data,v[idx_start:idx_stop,...],wait_for=v_part[i].events,is_blocking=False))
        Kyk2_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Kyk2_part[i].data,Kyk2[idx_start:idx_stop,...],wait_for=Kyk2_part[i].events,is_blocking=False))
        v_new_part[i].add_event(self.update_v(v_new_part[i],v_part[i],Kyk2_part[i],tau,i,0))
        symgrad_v_part[i].add_event(self.sym_grad(symgrad_v_part[i],v_new_part[i],i,0))
        symgrad_v_vold_part[i].add_event(self.sym_grad(symgrad_v_vold_part[i],v_part[i],i,0))
      for i in range(self.num_dev):
        idx_start = self.NSlice-((i+2+self.num_dev-1)*self.par_slices)
        idx_stop = self.NSlice-((i+1+self.num_dev-1)*self.par_slices)
        if idx_start==0:
          idx_stop+=self.overlap
        else:
          idx_start-=self.overlap
        v_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],v_part[i+self.num_dev].data,v[idx_start:idx_stop,...],wait_for=v_part[i+self.num_dev].events,is_blocking=False))
        Kyk2_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Kyk2_part[i+self.num_dev].data,Kyk2[idx_start:idx_stop,...],wait_for=Kyk2_part[i+self.num_dev].events,is_blocking=False))
        v_new_part[i+self.num_dev].add_event(self.update_v(v_new_part[i+self.num_dev],v_part[self.num_dev+i],Kyk2_part[self.num_dev+i],tau,i,1))
        symgrad_v_part[i+self.num_dev].add_event(self.sym_grad(symgrad_v_part[i+self.num_dev],v_new_part[i+self.num_dev],i,1))
        symgrad_v_vold_part[i+self.num_dev].add_event(self.sym_grad(symgrad_v_vold_part[i+self.num_dev],v_part[self.num_dev+i],i,1))
  #### Stream
      for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
          for i in range(self.num_dev):
            ### Get Data
            idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices)-self.overlap
            idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices))
            v_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],v_new[idx_start:idx_stop,...],v_new_part[i].data,wait_for=v_new_part[i].events,is_blocking=False))
            symgrad_v_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],symgrad_v[idx_start:idx_stop,...],symgrad_v_part[i].data,wait_for=symgrad_v_part[i].events,is_blocking=False))
            symgrad_v_vold_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],symgrad_v_vold[idx_start:idx_stop,...],symgrad_v_vold_part[i].data,wait_for=symgrad_v_vold_part[i].events,is_blocking=False))
            ### Put Data
            idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices)-self.overlap
            idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices))
            v_part[i].add_event(cl.enqueue_copy(self.queue[3*i],v_part[i].data,v[idx_start:idx_stop,...],wait_for=v_part[i].events,is_blocking=False))
            Kyk2_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Kyk2_part[i].data,Kyk2[idx_start:idx_stop,...],wait_for=Kyk2_part[i].events,is_blocking=False))
            v_new_part[i].add_event(self.update_v(v_new_part[i],v_part[i],Kyk2_part[i],tau,i,0))
            symgrad_v_part[i].add_event(self.sym_grad(symgrad_v_part[i],v_new_part[i],i,0))
            symgrad_v_vold_part[i].add_event(self.sym_grad(symgrad_v_vold_part[i],v_part[i],i,0))
          for i in range(self.num_dev):
            idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)-self.overlap
            idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)
            v_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],v_new[idx_start:idx_stop,...],v_new_part[i+self.num_dev].data,wait_for=v_new_part[i+self.num_dev].events,is_blocking=False))
            symgrad_v_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],symgrad_v[idx_start:idx_stop,...],symgrad_v_part[i+self.num_dev].data,wait_for=symgrad_v_part[i+self.num_dev].events,is_blocking=False))
            symgrad_v_vold_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],symgrad_v_vold[idx_start:idx_stop,...],symgrad_v_vold_part[i+self.num_dev].data,wait_for=symgrad_v_vold_part[i+self.num_dev].events,is_blocking=False))

            idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices)
            idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices)
            v_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],v_part[i+self.num_dev].data,v[idx_start:idx_stop,...],wait_for=v_part[i+self.num_dev].events,is_blocking=False))
            Kyk2_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Kyk2_part[i+self.num_dev].data,Kyk2[idx_start:idx_stop,...],wait_for=Kyk2_part[i+self.num_dev].events,is_blocking=False))
            v_new_part[i+self.num_dev].add_event(self.update_v(v_new_part[i+self.num_dev],v_part[self.num_dev+i],Kyk2_part[self.num_dev+i],tau,i,1))
            symgrad_v_part[i+self.num_dev].add_event(self.sym_grad(symgrad_v_part[i+self.num_dev],v_new_part[i+self.num_dev],i,1))
            symgrad_v_vold_part[i+self.num_dev].add_event(self.sym_grad(symgrad_v_vold_part[i+self.num_dev],v_part[self.num_dev+i],i,1))
  #### Collect last block
      if j<2*self.num_dev:
        j = 2*self.num_dev
      else:
        j+=1
      for i in range(self.num_dev):
        idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices)-self.overlap
        idx_stop =  self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices))
        v_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],v_new[idx_start:idx_stop,...],v_new_part[i].data,wait_for=v_new_part[i].events,is_blocking=False))
        symgrad_v_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],symgrad_v[idx_start:idx_stop,...],symgrad_v_part[i].data,wait_for=symgrad_v_part[i].events,is_blocking=False))
        symgrad_v_vold_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],symgrad_v_vold[idx_start:idx_stop,...],symgrad_v_vold_part[i].data,wait_for=symgrad_v_vold_part[i].events,is_blocking=False))
        idx_start = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)
        idx_stop = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)
        if idx_start==0:
          idx_stop+=self.overlap
        else:
          idx_start-=self.overlap
        v_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],v_new[idx_start:idx_stop,...],v_new_part[i+self.num_dev].data,wait_for=v_new_part[i+self.num_dev].events,is_blocking=False))
        symgrad_v_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],symgrad_v[idx_start:idx_stop,...],symgrad_v_part[i+self.num_dev].data,wait_for=symgrad_v_part[i+self.num_dev].events,is_blocking=False))
        symgrad_v_vold_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],symgrad_v_vold[idx_start:idx_stop,...],symgrad_v_vold_part[i+self.num_dev].data,wait_for=symgrad_v_vold_part[i+self.num_dev].events,is_blocking=False))
      self.queue[3*i+2].finish()



      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new

      while True:
        theta_line = tau_new/tau
        #### Allocate temporary Arrays
        ynorm = 0
        lhs = 0

        if myit == 0:
          z1_new_part = []
          z2_new_part = []
          r_new_part = []
          Kyk1_new_part = []
          Kyk2_new_part = []
          for i in range(2*self.num_dev):
            z1_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
            z2_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,8),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
            r_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
            Kyk1_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
            Kyk2_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))

        j=0
        last=0
        for i in range(self.num_dev):
          idx_start = self.NSlice-((i+1)*self.par_slices)-self.overlap
          idx_stop = self.NSlice-(i*self.par_slices)
          if idx_stop==self.NSlice:
            last=1
          else:
            last=0

          z1_part[i].add_event(cl.enqueue_copy(self.queue[3*i],z1_part[i].data,z1[idx_start:idx_stop,...],wait_for=z1_part[i].events,is_blocking=False))
          gradx_part[i].add_event(cl.enqueue_copy(self.queue[3*i],gradx_part[i].data,gradx[idx_start:idx_stop,...],wait_for=gradx_part[i].events,is_blocking=False))
          gradx_xold_part[i].add_event(cl.enqueue_copy(self.queue[3*i],gradx_xold_part[i].data,gradx_xold[idx_start:idx_stop,...],wait_for=gradx_xold_part[i].events,is_blocking=False))
          v_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i],v_new_part[i].data,v_new[idx_start:idx_stop,...],wait_for=v_new_part[i].events,is_blocking=False))
          v_part[i].add_event(cl.enqueue_copy(self.queue[3*i],v_part[i].data,v[idx_start:idx_stop,...],wait_for=v_part[i].events,is_blocking=False))
          r_part[i].add_event(cl.enqueue_copy(self.queue[3*i],r_part[i].data,r[idx_start:idx_stop,...],wait_for=r_part[i].events,is_blocking=False))
          Ax_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Ax_part[i].data,Ax[idx_start:idx_stop,...],wait_for=Ax_part[i].events,is_blocking=False))
          Ax_old_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Ax_old_part[i].data,Axold[idx_start:idx_stop,...],wait_for=Ax_old_part[i].events,is_blocking=False))
          res_part[i].add_event(cl.enqueue_copy(self.queue[3*i],res_part[i].data,res[idx_start:idx_stop,...],wait_for=res_part[i].events,is_blocking=False))
          Kyk1_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Kyk1_part[i].data,Kyk1[idx_start:idx_stop,...],wait_for=Kyk1_part[i].events,is_blocking=False))
          self.coil_buf_part[i].add_event(cl.enqueue_copy(self.queue[3*i],self.coil_buf_part[i].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[i].events,is_blocking=False))
          self.grad_buf_part[i][i].add_event(cl.enqueue_copy(self.queue[3*i],self.grad_buf_part[i][i].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[i][i].events,is_blocking=False))

          z1_new_part[ i].add_event(self.update_z1(z1_new_part[ i],z1_part[i],gradx_part[i],gradx_xold_part[i],v_new_part[i],v_part[i], beta_line*tau_new, theta_line, alpha,i,0))
          r_new_part[i].add_event(self.update_r(r_new_part[i],r_part[i],Ax_part[i],Ax_old_part[i],res_part[i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,0))
          Kyk1_new_part[ i].add_event(self.operator_adjoint_full(Kyk1_new_part[ i],r_new_part[ i],z1_new_part[ i],i,0,last))

        last = 0
        for i in range(self.num_dev):
          idx_start = self.NSlice-((i+2+self.num_dev-1)*self.par_slices)
          idx_stop = self.NSlice-((i+1+self.num_dev-1)*self.par_slices)
          if idx_start==0:
            idx_stop+=self.overlap
          else:
            idx_start-=self.overlap
          z1_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],z1_part[i+self.num_dev].data,z1[idx_start:idx_stop,...],wait_for=z1_part[i+self.num_dev].events,is_blocking=False))
          gradx_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],gradx_part[i+self.num_dev].data,gradx[idx_start:idx_stop,...],wait_for=gradx_part[i+self.num_dev].events,is_blocking=False))
          gradx_xold_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],gradx_xold_part[i+self.num_dev].data,gradx_xold[idx_start:idx_stop,...],wait_for=gradx_xold_part[i+self.num_dev].events,is_blocking=False))
          v_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],v_new_part[i+self.num_dev].data,v_new[idx_start:idx_stop,...],wait_for=v_new_part[i+self.num_dev].events,is_blocking=False))
          v_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],v_part[i+self.num_dev].data,v[idx_start:idx_stop,...],wait_for=v_part[i+self.num_dev].events,is_blocking=False))
          r_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],r_part[i+self.num_dev].data,r[idx_start:idx_stop,...],wait_for=r_part[i+self.num_dev].events,is_blocking=False))
          Ax_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Ax_part[i+self.num_dev].data,Ax[idx_start:idx_stop,...],wait_for=Ax_part[i+self.num_dev].events,is_blocking=False))
          Ax_old_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Ax_old_part[i+self.num_dev].data,Axold[idx_start:idx_stop,...],wait_for=Ax_old_part[i+self.num_dev].events,is_blocking=False))
          res_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],res_part[i+self.num_dev].data,res[idx_start:idx_stop,...],wait_for=res_part[i+self.num_dev].events,is_blocking=False))
          Kyk1_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Kyk1_part[i+self.num_dev].data,Kyk1[idx_start:idx_stop,...],wait_for=Kyk1_part[i+self.num_dev].events,is_blocking=False))
          self.coil_buf_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],self.coil_buf_part[i+self.num_dev].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[i+self.num_dev].events,is_blocking=False))
          self.grad_buf_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],self.grad_buf_part[i+self.num_dev].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[i+self.num_dev].events,is_blocking=False))
          z1_new_part[i+self.num_dev].add_event(self.update_z1(z1_new_part[i+self.num_dev],z1_part[self.num_dev+i],gradx_part[self.num_dev+i],gradx_xold_part[self.num_dev+i],v_new_part[self.num_dev+i],v_part[self.num_dev+i], beta_line*tau_new, theta_line, alpha,i,1))
          r_new_part[i+self.num_dev].add_event(self.update_r(r_new_part[i+self.num_dev],r_part[self.num_dev+i],Ax_part[self.num_dev+i],Ax_old_part[self.num_dev+i],res_part[self.num_dev+i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,1))
          Kyk1_new_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_new_part[i+self.num_dev],r_new_part[i+self.num_dev],z1_new_part[i+self.num_dev],i,1,last))

      #### Stream
        for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
          last = 0
          for i in range(self.num_dev):
            ### Get Data
            idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices)-self.overlap
            idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices))
            z1_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],z1_new[idx_start:idx_stop,...],z1_new_part[i].data,wait_for=z1_new_part[i].events,is_blocking=False))
            r_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],r_new[idx_start:idx_stop,...],r_new_part[i].data,wait_for=r_new_part[i].events,is_blocking=False))
            Kyk1_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk1_new[idx_start:idx_stop,...],Kyk1_new_part[i].data,wait_for=Kyk1_new_part[i].events,is_blocking=False))
            ynorm += ((clarray.vdot(r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],queue=self.queue[3*i])+clarray.vdot(z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
            lhs += ((clarray.vdot(Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
            ### Put Data
            idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices)-self.overlap
            idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices))
            z1_part[i].add_event(cl.enqueue_copy(self.queue[3*i],z1_part[i].data,z1[idx_start:idx_stop,...],wait_for=z1_part[i].events,is_blocking=False))
            gradx_part[i].add_event(cl.enqueue_copy(self.queue[3*i],gradx_part[i].data,gradx[idx_start:idx_stop,...],wait_for=gradx_part[i].events,is_blocking=False))
            gradx_xold_part[i].add_event(cl.enqueue_copy(self.queue[3*i],gradx_xold_part[i].data,gradx_xold[idx_start:idx_stop,...],wait_for=gradx_xold_part[i].events,is_blocking=False))
            v_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i],v_new_part[i].data,v_new[idx_start:idx_stop,...],wait_for=v_new_part[i].events,is_blocking=False))
            v_part[i].add_event(cl.enqueue_copy(self.queue[3*i],v_part[i].data,v[idx_start:idx_stop,...],wait_for=v_part[i].events,is_blocking=False))
            r_part[i].add_event(cl.enqueue_copy(self.queue[3*i],r_part[i].data,r[idx_start:idx_stop,...],wait_for=r_part[i].events,is_blocking=False))
            Ax_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Ax_part[i].data,Ax[idx_start:idx_stop,...],wait_for=Ax_part[i].events,is_blocking=False))
            Ax_old_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Ax_old_part[i].data,Axold[idx_start:idx_stop,...],wait_for=Ax_old_part[i].events,is_blocking=False))
            res_part[i].add_event(cl.enqueue_copy(self.queue[3*i],res_part[i].data,res[idx_start:idx_stop,...],wait_for=res_part[i].events,is_blocking=False))
            Kyk1_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Kyk1_part[i].data,Kyk1[idx_start:idx_stop,...],wait_for=Kyk1_part[i].events,is_blocking=False))
            self.coil_buf_part[i].add_event(cl.enqueue_copy(self.queue[3*i],self.coil_buf_part[i].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[i].events,is_blocking=False))
            self.grad_buf_part[i][i].add_event(cl.enqueue_copy(self.queue[3*i],self.grad_buf_part[i][i].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[i][i].events,is_blocking=False))
            z1_new_part[ i].add_event(self.update_z1(z1_new_part[ i],z1_part[i],gradx_part[i],gradx_xold_part[i],v_new_part[i],v_part[i], beta_line*tau_new, theta_line, alpha,i,0))
            r_new_part[i].add_event(self.update_r(r_new_part[i],r_part[i],Ax_part[i],Ax_old_part[i],res_part[i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,0))
            Kyk1_new_part[ i].add_event(self.operator_adjoint_full(Kyk1_new_part[ i],r_new_part[ i],z1_new_part[ i],i,0,last))
          for i in range(self.num_dev):
            ### Get Data
            idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)-self.overlap
            idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)
            z1_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],z1_new[idx_start:idx_stop,...],z1_new_part[i+self.num_dev].data,wait_for=z1_new_part[i+self.num_dev].events,is_blocking=False))
            r_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],r_new[idx_start:idx_stop,...],r_new_part[i+self.num_dev].data,wait_for=r_new_part[i+self.num_dev].events,is_blocking=False))
            Kyk1_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk1_new[idx_start:idx_stop,...],Kyk1_new_part[i+self.num_dev].data,wait_for=Kyk1_new_part[i+self.num_dev].events,is_blocking=False))
            ynorm += ((clarray.vdot(r_new_part[i+self.num_dev][self.overlap:,...]-r_part[i+self.num_dev][self.overlap:,...],r_new_part[i+self.num_dev][self.overlap:,...]-r_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1])+clarray.vdot(z1_new_part[i+self.num_dev][self.overlap:,...]-z1_part[i+self.num_dev][self.overlap:,...],z1_new_part[i+self.num_dev][self.overlap:,...]-z1_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
            lhs += ((clarray.vdot(Kyk1_new_part[i+self.num_dev][self.overlap:,...]-Kyk1_part[i+self.num_dev][self.overlap:,...],Kyk1_new_part[i+self.num_dev][self.overlap:,...]-Kyk1_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
            ### Put Data
            idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices)
            idx_stop = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices)
            if idx_start==0:
              idx_stop+=self.overlap
            else:
              idx_start-=self.overlap
            z1_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],z1_part[i+self.num_dev].data,z1[idx_start:idx_stop,...],wait_for=z1_part[i+self.num_dev].events,is_blocking=False))
            gradx_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],gradx_part[i+self.num_dev].data,gradx[idx_start:idx_stop,...],wait_for=gradx_part[i+self.num_dev].events,is_blocking=False))
            gradx_xold_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],gradx_xold_part[i+self.num_dev].data,gradx_xold[idx_start:idx_stop,...],wait_for=gradx_xold_part[i+self.num_dev].events,is_blocking=False))
            v_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],v_new_part[i+self.num_dev].data,v_new[idx_start:idx_stop,...],wait_for=v_new_part[i+self.num_dev].events,is_blocking=False))
            v_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],v_part[i+self.num_dev].data,v[idx_start:idx_stop,...],wait_for=v_part[i+self.num_dev].events,is_blocking=False))
            r_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],r_part[i+self.num_dev].data,r[idx_start:idx_stop,...],wait_for=r_part[i+self.num_dev].events,is_blocking=False))
            Ax_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Ax_part[i+self.num_dev].data,Ax[idx_start:idx_stop,...],wait_for=Ax_part[i+self.num_dev].events,is_blocking=False))
            Ax_old_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Ax_old_part[i+self.num_dev].data,Axold[idx_start:idx_stop,...],wait_for=Ax_old_part[i+self.num_dev].events,is_blocking=False))
            res_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],res_part[i+self.num_dev].data,res[idx_start:idx_stop,...],wait_for=res_part[i+self.num_dev].events,is_blocking=False))
            Kyk1_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Kyk1_part[i+self.num_dev].data,Kyk1[idx_start:idx_stop,...],wait_for=Kyk1_part[i+self.num_dev].events,is_blocking=False))
            self.coil_buf_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],self.coil_buf_part[i+self.num_dev].data,self.C[idx_start:idx_stop,...],wait_for=self.coil_buf_part[i+self.num_dev].events,is_blocking=False))
            self.grad_buf_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],self.grad_buf_part[i+self.num_dev].data,self.grad_x[idx_start:idx_stop,...],wait_for=self.grad_buf_part[i+self.num_dev].events,is_blocking=False))
            z1_new_part[i+self.num_dev].add_event(self.update_z1(z1_new_part[i+self.num_dev],z1_part[self.num_dev+i],gradx_part[self.num_dev+i],gradx_xold_part[self.num_dev+i],v_new_part[self.num_dev+i],v_part[self.num_dev+i], beta_line*tau_new, theta_line, alpha,i,1))
            r_new_part[i+self.num_dev].add_event(self.update_r(r_new_part[i+self.num_dev],r_part[self.num_dev+i],Ax_part[self.num_dev+i],Ax_old_part[self.num_dev+i],res_part[self.num_dev+i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,1))
            Kyk1_new_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_new_part[i+self.num_dev],r_new_part[i+self.num_dev],z1_new_part[i+self.num_dev],i,1,last))
    #### Collect last block
        if j<2*self.num_dev:
          j = 2*self.num_dev
        else:
          j+=1
        for i in range(self.num_dev):
          idx_start = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices)-self.overlap
          idx_stop =  self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices))
          z1_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],z1_new[idx_start:idx_stop,...],z1_new_part[i].data,wait_for=z1_new_part[i].events,is_blocking=False))
          r_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],r_new[idx_start:idx_stop,...],r_new_part[i].data,wait_for=r_new_part[i].events,is_blocking=False))
          Kyk1_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk1_new[idx_start:idx_stop,...],Kyk1_new_part[i].data,wait_for=Kyk1_new_part[i].events,is_blocking=False))
          ynorm += ((clarray.vdot(r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],queue=self.queue[3*i])+clarray.vdot(z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
          lhs += ((clarray.vdot(Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
          idx_start = self.NSlice-(i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)
          idx_stop = self.NSlice-((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices)
          if idx_start==0:
            idx_stop+=self.overlap
            ynorm += ((clarray.vdot(r_new_part[i][:self.par_slices,...]-r_part[i][:self.par_slices,...],r_new_part[i][:self.par_slices,...]-r_part[i][:self.par_slices,...],queue=self.queue[3*i])+clarray.vdot(z1_new_part[i][:self.par_slices,...]-z1_part[i][:self.par_slices,...],z1_new_part[i][:self.par_slices,...]-z1_part[i][:self.par_slices,...],queue=self.queue[3*i]))).get()
            lhs += ((clarray.vdot(Kyk1_new_part[i][:self.par_slices,...]-Kyk1_part[i][:self.par_slices,...],Kyk1_new_part[i][:self.par_slices,...]-Kyk1_part[i][:self.par_slices,...],queue=self.queue[3*i]))).get()
          else:
            idx_start-=self.overlap
            ynorm += ((clarray.vdot(r_new_part[i+self.num_dev][self.overlap:,...]-r_part[i+self.num_dev][self.overlap:,...],r_new_part[i+self.num_dev][self.overlap:,...]-r_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1])+clarray.vdot(z1_new_part[i+self.num_dev][self.overlap:,...]-z1_part[i+self.num_dev][self.overlap:,...],z1_new_part[i+self.num_dev][self.overlap:,...]-z1_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
            lhs += ((clarray.vdot(Kyk1_new_part[i+self.num_dev][self.overlap:,...]-Kyk1_part[i+self.num_dev][self.overlap:,...],Kyk1_new_part[i+self.num_dev][self.overlap:,...]-Kyk1_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()

          z1_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],z1_new[idx_start:idx_stop,...],z1_new_part[i+self.num_dev].data,wait_for=z1_new_part[i+self.num_dev].events,is_blocking=False))
          r_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],r_new[idx_start:idx_stop,...],r_new_part[i+self.num_dev].data,wait_for=r_new_part[i+self.num_dev].events,is_blocking=False))
          Kyk1_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk1_new[idx_start:idx_stop,...],Kyk1_new_part[i+self.num_dev].data,wait_for=Kyk1_new_part[i+self.num_dev].events,is_blocking=False))
        self.queue[3*i+2].finish()


        j=0
        first = 0
        for i in range(self.num_dev):
          idx_start = i*self.par_slices
          idx_stop = (i+1)*self.par_slices +self.overlap
          z2_part[i].add_event(cl.enqueue_copy(self.queue[3*i],z2_part[i].data,z2[idx_start:idx_stop,...],wait_for=z2_part[i].events,is_blocking=False))
          symgrad_v_part[i].add_event(cl.enqueue_copy(self.queue[3*i],symgrad_v_part[i].data,symgrad_v[idx_start:idx_stop,...],wait_for=symgrad_v_part[i].events,is_blocking=False))
          symgrad_v_vold_part[i].add_event(cl.enqueue_copy(self.queue[3*i],symgrad_v_vold_part[i].data,symgrad_v_vold[idx_start:idx_stop,...],wait_for=symgrad_v_vold_part[i].events,is_blocking=False))
          Kyk2_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Kyk2_part[i].data,Kyk2[idx_start:idx_stop,...],wait_for=Kyk2_part[i].events,is_blocking=False))
          z1_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i],z1_new_part[i].data,z1_new[idx_start:idx_stop,...],wait_for=z1_new_part[i].events,is_blocking=False))
          if i==0:
            first=1
          else:
            first=0
          z2_new_part[ i].add_event(self.update_z2(z2_new_part[ i],z2_part[i],symgrad_v_part[i],symgrad_v_vold_part[i],beta_line*tau_new,theta_line,beta,i,0))
          Kyk2_new_part[ i].add_event(self.update_Kyk2(Kyk2_new_part[ i],z2_new_part[ i],z1_new_part[ i],i,0,first))

        first=0
        for i in range(self.num_dev):
          idx_start = (i+1+self.num_dev-1)*self.par_slices
          idx_stop = (i+2+self.num_dev-1)*self.par_slices
          if idx_stop == self.NSlice:
            idx_start -=self.overlap
          else:
            idx_stop +=self.overlap
          z2_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],z2_part[i+self.num_dev].data,z2[idx_start:idx_stop,...],wait_for=z2_part[i+self.num_dev].events,is_blocking=False))
          symgrad_v_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],symgrad_v_part[i+self.num_dev].data,symgrad_v[idx_start:idx_stop,...],wait_for=symgrad_v_part[i+self.num_dev].events,is_blocking=False))
          symgrad_v_vold_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],symgrad_v_vold_part[i+self.num_dev].data,symgrad_v_vold[idx_start:idx_stop,...],wait_for=symgrad_v_vold_part[i+self.num_dev].events,is_blocking=False))
          Kyk2_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Kyk2_part[i+self.num_dev].data,Kyk2[idx_start:idx_stop,...],wait_for=Kyk2_part[i+self.num_dev].events,is_blocking=False))
          z1_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],z1_new_part[i+self.num_dev].data,z1_new[idx_start:idx_stop,...],wait_for=z1_new_part[i+self.num_dev].events,is_blocking=False))
          z2_new_part[i+self.num_dev].add_event(self.update_z2(z2_new_part[i+self.num_dev],z2_part[self.num_dev+i],symgrad_v_part[self.num_dev+i],symgrad_v_vold_part[self.num_dev+i],beta_line*tau_new,theta_line,beta,i,1))
          Kyk2_new_part[i+self.num_dev].add_event(self.update_Kyk2(Kyk2_new_part[i+self.num_dev],z2_new_part[i+self.num_dev],z1_new_part[i+self.num_dev],i,1,first))

      #### Stream
        for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
          for i in range(self.num_dev):
            ### Get Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices+self.overlap
            z2_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],z2_new[idx_start:idx_stop,...],z2_new_part[i].data,wait_for=z2_new_part[i].events,is_blocking=False))
            Kyk2_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk2_new[idx_start:idx_stop,...],Kyk2_new_part[i].data,wait_for=Kyk2_new_part[i].events,is_blocking=False))
            ynorm += ((clarray.vdot(z2_new_part[i][:self.par_slices,...]-z2_part[i][:self.par_slices,...],z2_new_part[i][:self.par_slices,...]-z2_part[i][:self.par_slices,...],queue=self.queue[3*i]))).get()
            lhs += ((clarray.vdot(Kyk2_new_part[i][:self.par_slices,...]-Kyk2_part[i][:self.par_slices,...],Kyk2_new_part[i][:self.par_slices,...]-Kyk2_part[i][:self.par_slices,...],queue=self.queue[3*i]))).get()

            ### Put Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices+self.overlap
            z2_part[i].add_event(cl.enqueue_copy(self.queue[3*i],z2_part[i].data,z2[idx_start:idx_stop,...],wait_for=z2_part[i].events,is_blocking=False))
            symgrad_v_part[i].add_event(cl.enqueue_copy(self.queue[3*i],symgrad_v_part[i].data,symgrad_v[idx_start:idx_stop,...],wait_for=symgrad_v_part[i].events,is_blocking=False))
            symgrad_v_vold_part[i].add_event(cl.enqueue_copy(self.queue[3*i],symgrad_v_vold_part[i].data,symgrad_v_vold[idx_start:idx_stop,...],wait_for=symgrad_v_vold_part[i].events,is_blocking=False))
            Kyk2_part[i].add_event(cl.enqueue_copy(self.queue[3*i],Kyk2_part[i].data,Kyk2[idx_start:idx_stop,...],wait_for=Kyk2_part[i].events,is_blocking=False))
            z1_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i],z1_new_part[i].data,z1_new[idx_start:idx_stop,...],wait_for=z1_new_part[i].events,is_blocking=False))
            z2_new_part[ i].add_event(self.update_z2(z2_new_part[ i],z2_part[i],symgrad_v_part[i],symgrad_v_vold_part[i],beta_line*tau_new,theta_line,beta,i,0))
            Kyk2_new_part[ i].add_event(self.update_Kyk2(Kyk2_new_part[ i],z2_new_part[ i],z1_new_part[ i],i,0,first))
          for i in range(self.num_dev):
            ### Get Data
            self.queue[3*i+1].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices+self.overlap
            z2_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],z2_new[idx_start:idx_stop,...],z2_new_part[i+self.num_dev].data,wait_for=z2_new_part[i+self.num_dev].events,is_blocking=False))
            Kyk2_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk2_new[idx_start:idx_stop,...],Kyk2_new_part[i+self.num_dev].data,wait_for=Kyk2_new_part[i+self.num_dev].events,is_blocking=False))
            ynorm += ((clarray.vdot(z2_new_part[i+self.num_dev][:self.par_slices,...]-z2_part[i+self.num_dev][:self.par_slices,...],z2_new_part[i+self.num_dev][:self.par_slices,...]-z2_part[i+self.num_dev][:self.par_slices,...],queue=self.queue[3*i+1]))).get()
            lhs += ((clarray.vdot(Kyk2_new_part[i+self.num_dev][:self.par_slices,...]-Kyk2_part[i+self.num_dev][:self.par_slices,...],Kyk2_new_part[i+self.num_dev][:self.par_slices,...]-Kyk2_part[i+self.num_dev][:self.par_slices,...],queue=self.queue[3*i+1]))).get()

            ### Put Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            if idx_stop == self.NSlice:
              idx_start -=self.overlap
            else:
              idx_stop +=self.overlap
            z2_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],z2_part[i+self.num_dev].data,z2[idx_start:idx_stop,...],wait_for=z2_part[i+self.num_dev].events,is_blocking=False))
            symgrad_v_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],symgrad_v_part[i+self.num_dev].data,symgrad_v[idx_start:idx_stop,...],wait_for=symgrad_v_part[i+self.num_dev].events,is_blocking=False))
            symgrad_v_vold_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],symgrad_v_vold_part[i+self.num_dev].data,symgrad_v_vold[idx_start:idx_stop,...],wait_for=symgrad_v_vold_part[i+self.num_dev].events,is_blocking=False))
            Kyk2_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],Kyk2_part[i+self.num_dev].data,Kyk2[idx_start:idx_stop,...],wait_for=Kyk2_part[i+self.num_dev].events,is_blocking=False))
            z1_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+1],z1_new_part[i+self.num_dev].data,z1_new[idx_start:idx_stop,...],wait_for=z1_new_part[i+self.num_dev].events,is_blocking=False))
            z2_new_part[i+self.num_dev].add_event(self.update_z2(z2_new_part[i+self.num_dev],z2_part[self.num_dev+i],symgrad_v_part[self.num_dev+i],symgrad_v_vold_part[self.num_dev+i],beta_line*tau_new,theta_line,beta,i,1))
            Kyk2_new_part[i+self.num_dev].add_event(self.update_Kyk2(Kyk2_new_part[i+self.num_dev],z2_new_part[i+self.num_dev],z1_new_part[i+self.num_dev],i,1,first))
      #### Collect last block
        if j<2*self.num_dev:
          j = 2*self.num_dev
        else:
          j+=1
        for i in range(self.num_dev):
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices+self.overlap
          z2_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],z2_new[idx_start:idx_stop,...],z2_new_part[i].data,wait_for=z2_new_part[i].events,is_blocking=False))
          Kyk2_new_part[i].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk2_new[idx_start:idx_stop,...],Kyk2_new_part[i].data,wait_for=Kyk2_new_part[i].events,is_blocking=False))
          ynorm += ((clarray.vdot(z2_new_part[i][:self.par_slices,...]-z2_part[i][:self.par_slices,...],z2_new_part[i][:self.par_slices,...]-z2_part[i][:self.par_slices,...],queue=self.queue[3*i]))).get()
          lhs += ((clarray.vdot(Kyk2_new_part[i][:self.par_slices,...]-Kyk2_part[i][:self.par_slices,...],Kyk2_new_part[i][:self.par_slices,...]-Kyk2_part[i][:self.par_slices,...],queue=self.queue[3*i]))).get()

          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          if idx_stop == self.NSlice:
            idx_start-=self.overlap
            ynorm += ((clarray.vdot(z2_new_part[i+self.num_dev][self.overlap:,...]-z2_part[i+self.num_dev][self.overlap:,...],z2_new_part[i+self.num_dev][self.overlap:,...]-z2_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
            lhs += ((clarray.vdot(Kyk2_new_part[i+self.num_dev][self.overlap:,...]-Kyk2_part[i+self.num_dev][self.overlap:,...],Kyk2_new_part[i+self.num_dev][self.overlap:,...]-Kyk2_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
          else:
            idx_stop+=self.overlap
            ynorm += ((clarray.vdot(z2_new_part[i+self.num_dev][:self.par_slices,...]-z2_part[i+self.num_dev][:self.par_slices,...],z2_new_part[i+self.num_dev][:self.par_slices,...]-z2_part[i+self.num_dev][:self.par_slices,...],queue=self.queue[3*i+1]))).get()
            lhs += ((clarray.vdot(Kyk2_new_part[i+self.num_dev][:self.par_slices,...]-Kyk2_part[i+self.num_dev][:self.par_slices,...],Kyk2_new_part[i+self.num_dev][:self.par_slices,...]-Kyk2_part[i+self.num_dev][:self.par_slices,...],queue=self.queue[3*i+1]))).get()
          z2_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],z2_new[idx_start:idx_stop,...],z2_new_part[i+self.num_dev].data,wait_for=z2_new_part[i+self.num_dev].events,is_blocking=False))
          Kyk2_new_part[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i+2],Kyk2_new[idx_start:idx_stop,...],Kyk2_new_part[i+self.num_dev].data,wait_for=Kyk2_new_part[i+self.num_dev].events,is_blocking=False))
        self.queue[3*i+2].finish()

        if np.sqrt(beta_line)*tau_new*(abs(lhs)**(1/2)) <= (abs(ynorm)**(1/2))*delta_line:
            break
        else:
          tau_new = tau_new*mu_line
      (Kyk1, Kyk1_new, Kyk2, Kyk2_new, Axold, Ax, z1, z1_new, z2, z2_new, r, r_new) =\
      (Kyk1_new, Kyk1, Kyk2_new, Kyk2, Ax, Axold, z1_new, z1, z2_new, z2, r_new, r)
      tau =  (tau_new)


      if not np.mod(myit,10):
        self.model.plot_unknowns(np.transpose(x_new,[1,0,2,3]))
        primal_new= (self.irgn_par["lambd"]/2*np.vdot(Axold-res,Axold-res)+alpha*np.sum(abs((gradx[:,:self.unknowns_TGV]-v))) +beta*np.sum(abs(symgrad_v)) + 1/(2*delta)*np.vdot(x_new-xk,x_new-xk)).real

        dual = (-delta/2*np.vdot(-Kyk1.flatten(),-Kyk1.flatten())- np.vdot(xk.flatten(),(-Kyk1).flatten()) + np.sum(Kyk2)
                  - 1/(2*self.irgn_par["lambd"])*np.vdot(r.flatten(),r.flatten()) - np.vdot(res.flatten(),r.flatten())).real

        gap = np.abs(primal_new - dual)
        if myit==0:
          gap_min = gap
        if np.abs(primal-primal_new)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"]:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(myit,abs(primal-primal_new)/(self.irgn_par["lambd"]*self.NSlice)))
          self.v = v_new
          self.r = r
          self.z1 = z1
          self.z2 = z2
          return x_new
        if (gap > gap_min*self.irgn_par["stag"]) and myit>1:
          self.v = v_new
          self.r = r
          self.z1 = z1
          self.z2 = z2
          print("Terminated at iteration %d because the method stagnated"%(myit))
          return x_new
        if np.abs(gap - gap_min)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"] and myit>1:
          self.v = v_new
          self.r = r
          self.z1 = z1
          self.z2 = z2
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(myit,abs(gap - gap_min)/(self.irgn_par["lambd"]*self.NSlice)))
          return x_new
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        sys.stdout.write("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f    \r" \
                       %(myit,primal/(self.irgn_par["lambd"]*self.NSlice),dual/(self.irgn_par["lambd"]*self.NSlice),gap/(self.irgn_par["lambd"]*self.NSlice)))
        sys.stdout.flush()

      (x, x_new) = (x_new, x)
      (v, v_new) = (v_new, v)
    self.v = v
    self.r = r
    self.z1 = z1
    self.z2 = z2
    return x

  def FT_streamed(self,outp,inp):
      cl_out = []
      j=0
      for i in range(self.num_dev):
        cl_out.append(clarray.zeros(self.queue[3*i],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE))
        cl_out.append(clarray.zeros(self.queue[3*i+1],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE))
      cl_data = []
      for i in range(self.num_dev):
        idx_start = i*self.par_slices
        idx_stop = (i+1)*self.par_slices+self.overlap
        cl_data.append(clarray.to_device(self.queue[3*i], inp[idx_start:idx_stop,...]))
        cl_out[2*i].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_data[i],0))
      for i in range(self.num_dev):
        idx_start = (i+1+self.num_dev-1)*self.par_slices
        idx_stop = (i+2+self.num_dev-1)*self.par_slices
        if idx_stop == self.NSlice:
          idx_start -= self.overlap
        else:
          idx_stop+=self.overlap
        cl_data.append(clarray.to_device(self.queue[3*i+1], inp[idx_start:idx_stop,...]))
        cl_out[2*i+1].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_data[self.num_dev+i],1))

      for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
          for i in range(self.num_dev):
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices+self.overlap
            cl_out[2*i].add_event(cl.enqueue_copy(self.queue[3*i+2],outp[idx_start:idx_stop,...],cl_out[2*i].data,wait_for=cl_out[2*i].events,is_blocking=False))
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
            idx_stop = ((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices)+self.overlap
            cl_data[i].add_event(cl.enqueue_copy(self.queue[3*i],cl_data[i].data,inp[idx_start:idx_stop,...],wait_for=cl_data[i].events,is_blocking=False))
            cl_out[2*i].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_data[i],0))
          for i in range(self.num_dev):
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices+self.overlap
            cl_out[2*i+1].add_event(cl.enqueue_copy(self.queue[3*i+2],outp[idx_start:idx_stop,...],cl_out[2*i+1].data,wait_for=cl_out[2*i+1].events,is_blocking=False))
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            if idx_stop == self.NSlice:
              idx_start -= self.overlap
            else:
              idx_stop+=self.overlap
            cl_data[i+self.num_dev].add_event(cl.enqueue_copy(self.queue[3*i],cl_data[i+self.num_dev].data,inp[idx_start:idx_stop,...],wait_for=cl_data[i+self.num_dev].events,is_blocking=False))
            cl_out[2*i+1].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_data[i+self.num_dev],1))
      if j< 2*self.num_dev:
        j = 2*self.num_dev
      else:
        j+=1
      for i in range(self.num_dev):
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices+self.overlap
        cl_out[2*i].add_event(cl.enqueue_copy(self.queue[3*i+2],outp[idx_start:idx_stop,...],cl_out[2*i].data,wait_for=cl_out[2*i].events,is_blocking=False))
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        if idx_stop==self.NSlice:
          idx_start-=self.overlap
        else:
          idx_stop+=self.overlap
        cl_out[2*i+1].add_event(cl.enqueue_copy(self.queue[3*i+2],outp[idx_start:idx_stop,...],cl_out[2*i+1].data,wait_for=cl_out[2*i+1].events,is_blocking=False))
      self.queue[3*i+2].finish()


  def eval_fwd_streamed(self,y,x,idx=0,idxq=0,wait_for=[]):

    return self.prg[idx].operator_fwd(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None,
                                 y.data, x.data, self.coil_buf_part[idx+idxq*self.num_dev].data, self.grad_buf_part[idx+idxq*self.num_dev].data,
                                 np.int32(self.NC), np.int32(self.par_slices+self.overlap), np.int32(self.NScan), np.int32(self.unknowns),
                                 wait_for= y.events+x.events+wait_for)

  def operator_forward_streamed(self, outp, inp):
      cl_out = []
      cl_tmp = []
      self.coil_buf_part = []
      self.grad_buf_part = []
      j=0

      for i in range(self.num_dev):
        cl_out.append(clarray.zeros(self.queue[3*i],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE))
        cl_out.append(clarray.zeros(self.queue[3*i+1],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE))
        cl_tmp.append(clarray.zeros(self.queue[3*i],(self.par_slices+self.overlap,self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE))
        cl_tmp.append(clarray.zeros(self.queue[3*i+1],(self.par_slices+self.overlap,self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE))

      cl_data = []
      for i in range(self.num_dev):
        idx_start = i*self.par_slices
        idx_stop = (i+1)*self.par_slices+self.overlap
        cl_data.append(clarray.to_device(self.queue[3*i], inp[idx_start:idx_stop,...]))
        self.coil_buf_part.append(clarray.to_device(self.queue[3*i], self.C[idx_start:idx_stop,...],allocator=self.alloc[i]))
        self.grad_buf_part.append(clarray.to_device(self.queue[3*i], self.grad_x[idx_start:idx_stop,...],allocator=self.alloc[i]))
        cl_tmp[2*i].add_event(self.eval_fwd_streamed(cl_tmp[2*i],cl_data[i],idx=i,idxq=0,wait_for=cl_tmp[2*i].events+cl_data[i].events
              +self.coil_buf_part[i].events+self.grad_buf_part[i].events))
        cl_out[2*i].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_tmp[2*i],0,wait_for=cl_tmp[2*i].events+cl_out[2*i].events))


      for i in range(self.num_dev):
        idx_start = (i+1+self.num_dev-1)*self.par_slices
        idx_stop = (i+2+self.num_dev-1)*self.par_slices
        if idx_stop==self.NSlice:
          idx_start -=self.overlap
        else:
          idx_stop+=self.overlap

        cl_data.append(clarray.to_device(self.queue[3*i+1], inp[idx_start:idx_stop,...]))
        self.coil_buf_part.append(clarray.to_device(self.queue[3*i+1], self.C[idx_start:idx_stop,...],allocator=self.alloc[i]))
        self.grad_buf_part.append(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start:idx_stop,...],allocator=self.alloc[i]))
        cl_tmp[2*i+1].add_event(self.eval_fwd_streamed(cl_tmp[2*i+1],cl_data[self.num_dev+i],idx=i,idxq=1,wait_for=cl_tmp[2*i+1].events+cl_data[self.num_dev+i].events
              +self.coil_buf_part[self.num_dev+i].events+self.grad_buf_part[self.num_dev+i].events))
        cl_out[2*i+1].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_tmp[2*i+1],1,wait_for=cl_tmp[2*i+1].events+cl_out[2*i+1].events))


      for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
          for i in range(self.num_dev):
            self.queue[3*i].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
            cl_out[2*i].get(queue=self.queue[3*i+2],ary=self.tmp_FT)
            outp[idx_start:idx_stop,...]=self.tmp_FT[:self.par_slices,...]

            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
            idx_stop = ((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices)+self.overlap
            cl_data[i] = clarray.to_device(self.queue[3*i],inp[idx_start:idx_stop,...])
            self.coil_buf_part[i]=(clarray.to_device(self.queue[3*i], self.C[idx_start:idx_stop,...]))
            self.grad_buf_part[i]=(clarray.to_device(self.queue[3*i], self.grad_x[idx_start:idx_stop,...]))
            cl_tmp[2*i].add_event(self.eval_fwd_streamed(cl_tmp[2*i],cl_data[i],idx=i,idxq=0,wait_for=cl_tmp[2*i].events+cl_data[i].events
                  +self.coil_buf_part[i].events+self.grad_buf_part[i].events))
            cl_out[2*i].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_tmp[2*i],0,wait_for=cl_tmp[2*i].events+cl_out[2*i].events))
          for i in range(self.num_dev):
            self.queue[3*i+1].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            cl_out[2*i+1].get(queue=self.queue[3*i+2],ary=self.tmp_FT)
            outp[idx_start:idx_stop,...]=self.tmp_FT[:self.par_slices,...]

            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            if idx_stop==self.NSlice:
              idx_start -= self.overlap
            else:
              idx_stop += self.overlap

            cl_data[i+self.num_dev] = clarray.to_device(self.queue[3*i+1], inp[idx_start:idx_stop,...])
            self.coil_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.C[idx_start:idx_stop,...]))
            self.grad_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start:idx_stop,...]))
            cl_tmp[2*i+1].add_event(self.eval_fwd_streamed(cl_tmp[2*i+1],cl_data[self.num_dev+i],idx=i,idxq=1,wait_for=cl_tmp[2*i+1].events+cl_data[self.num_dev+i].events
                  +self.coil_buf_part[self.num_dev+i].events+self.grad_buf_part[self.num_dev+i].events))
            cl_out[2*i+1].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_tmp[2*i+1],1,wait_for=cl_tmp[2*i+1].events+cl_out[2*i+1].events))
      if j<2*self.num_dev:
        j = 2*self.num_dev
      else:
        j+=1
      for i in range(self.num_dev):
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
        self.queue[3*i].finish()
        cl_out[2*i].get(queue=self.queue[3*i+2],ary=self.tmp_FT)
        outp[idx_start:idx_stop,...]=self.tmp_FT[:self.par_slices,...]
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        self.queue[3*i+1].finish()
        cl_out[2*i+1].get(queue=self.queue[3*i+2],ary=self.tmp_FT)
        if idx_stop==self.NSlice:
          outp[idx_start:idx_stop,...]=self.tmp_FT[self.overlap:,...]
        else:
          outp[idx_start:idx_stop,...]=self.tmp_FT[:self.par_slices,...]



  def tv_solve_3D(self, x,res, iters):

    alpha = self.irgn_par["gamma"]


    L = np.float32(8)
#    print('L: %f'%(L))


    tau = np.float32(1/np.sqrt(L))
    tau_new =np.float32(0)

#    self.set_scale(x)
    xk = x.copy()
    x_new = np.zeros_like(x)

    r = self.r#np.zeros_like(res,dtype=DTYPE)
    r_new = np.zeros_like(r)
    z1 = self.z1#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    z1_new =  np.zeros_like(z1)#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    res = (res).astype(DTYPE)


    delta = self.irgn_par["delta"]
    mu = 1/delta
    theta_line = np.float32(1.0)
    beta_line = np.float32(10)
    beta_new = np.float32(0)
    mu_line =np.float32( 0.5)

    delta_line = np.float32(1)

    ynorm = np.float32(0.0)
    lhs = np.float32(0.0)
    primal = np.float32(0.0)
    primal_new = np.float32(0)
    dual = np.float32(0.0)
    gap_min = np.float32(0.0)
    gap = np.float32(0.0)



    self.eval_const()


    Kyk1 = np.zeros_like(x)
    Kyk1_new = np.zeros_like(x)
    gradx = np.zeros_like(z1)
    gradx_xold = np.zeros_like(z1)

    Axold = np.zeros_like(res)
    Ax = np.zeros_like(res)

#### Allocate temporary Arrays
    Axold_part = []
    Axold_tmp = np.zeros((self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE)
    Kyk1_part = []
    Kyk1_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE)
    for i in range(2*self.num_dev):
      Axold_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
      Kyk1_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))


##### Warmup
    x_part = []
    r_part = []
    z1_part = []
    self.coil_buf_part = []
    self.grad_buf_part = []
    j=0
    last=0
    for i in range(self.num_dev):
      idx_start = i*self.par_slices
      idx_stop = (i+1)*self.par_slices
      if idx_start==0:
        idx_stop +=self.overlap
      else:
        idx_start-=self.overlap
      x_part.append(clarray.to_device(self.queue[3*i], x[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
      r_part.append(clarray.to_device(self.queue[3*i], r[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
      z1_part.append(clarray.to_device(self.queue[3*i], z1[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
      self.coil_buf_part.append(clarray.to_device(self.queue[3*i], self.C[idx_start:idx_stop,...],allocator=self.alloc[i]))
      self.grad_buf_part.append(clarray.to_device(self.queue[3*i], self.grad_x[idx_start:idx_stop,...],allocator=self.alloc[i]))
    for i in range(self.num_dev):
      Axold_part[i].add_event(self.operator_forward_full(Axold_part[i],x_part[i],i,0))
      Kyk1_part[i].add_event(self.operator_adjoint_full(Kyk1_part[i],r_part[i],z1_part[i],i,0,last))
    for i in range(self.num_dev):
      idx_start = ((i+1+self.num_dev-1)*self.par_slices)-self.overlap
      idx_stop = (i+2+self.num_dev-1)*self.par_slices
      x_part.append(clarray.to_device(self.queue[3*i+1], x[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
      r_part.append(clarray.to_device(self.queue[3*i+1], r[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
      z1_part.append(clarray.to_device(self.queue[3*i+1], z1[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
      self.coil_buf_part.append(clarray.to_device(self.queue[3*i+1], self.C[idx_start:idx_stop,...],allocator=self.alloc[i]))
      self.grad_buf_part.append(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start:idx_stop,...],allocator=self.alloc[i]))
    for i in range(self.num_dev):
      Axold_part[i+self.num_dev].add_event(self.operator_forward_full(Axold_part[i+self.num_dev],x_part[self.num_dev+i],i,1))
      Kyk1_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_part[i+self.num_dev],r_part[self.num_dev+i],z1_part[self.num_dev+i],i,1,last))
#### Stream
    for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
        for i in range(self.num_dev):
          ### Get Data
          self.queue[3*i].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
          Axold_part[i].get(queue=self.queue[3*i+2],ary=Axold_tmp)
          Kyk1_part[i].get(queue=self.queue[3*i+2],ary=Kyk1_tmp)
          self.queue[3*i+2].finish()
          if idx_start==0:
            Axold[idx_start:idx_stop,...] = Axold_tmp[:self.par_slices,...]
            Kyk1[idx_start:idx_stop,...] = Kyk1_tmp[:self.par_slices,...]
          else:
            Axold[idx_start:idx_stop,...] = Axold_tmp[self.overlap:,...]
            Kyk1[idx_start:idx_stop,...] = Kyk1_tmp[self.overlap:,...]
          ### Put Data
          idx_start = (i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices))-self.overlap
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices
          x_part[i].set(x[idx_start:idx_stop,...],self.queue[3*i])# ))
          r_part[i].set( r[idx_start:idx_stop,...],self.queue[3*i])# ))
          z1_part[i].set( z1[idx_start:idx_stop,...],self.queue[3*i])# ))
          self.coil_buf_part[i].set(self.C[idx_start:idx_stop,...],self.queue[3*i])
          self.grad_buf_part[i].set( self.grad_x[idx_start:idx_stop,...],self.queue[3*i])
        for i in range(self.num_dev):
          Axold_part[i].add_event(self.operator_forward_full(Axold_part[i],x_part[i],i,0))
          Kyk1_part[i].add_event(self.operator_adjoint_full(Kyk1_part[i],r_part[i],z1_part[i],i,0,last))
        for i in range(self.num_dev):
          self.queue[3*i+1].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          Axold_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=Axold_tmp)
          Kyk1_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=Kyk1_tmp)
          self.queue[3*i+2].finish()
          Axold[idx_start:idx_stop,...] = Axold_tmp[self.overlap:,...]
          Kyk1[idx_start:idx_stop,...] = Kyk1_tmp[self.overlap:,...]
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices-self.overlap
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
          x_part[self.num_dev+i].set( x[idx_start:idx_stop,...] ,self.queue[3*i+1])# ))
          r_part[self.num_dev+i].set( r[idx_start:idx_stop,...] ,self.queue[3*i+1])# ))
          z1_part[self.num_dev+i].set( z1[idx_start:idx_stop,...] ,self.queue[3*i+1])# ))
          self.coil_buf_part[self.num_dev+i].set( self.C[idx_start:idx_stop,...] ,self.queue[3*i+1])
          self.grad_buf_part[self.num_dev+i].set( self.grad_x[idx_start:idx_stop,...] ,self.queue[3*i+1])
          if idx_stop==self.NSlice:
            last = 1
        for i in range(self.num_dev):
          Axold_part[i+self.num_dev].add_event(self.operator_forward_full(Axold_part[i+self.num_dev],x_part[self.num_dev+i],i,1))
          Kyk1_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_part[i+self.num_dev],r_part[self.num_dev+i],z1_part[self.num_dev+i],i,1,last))
#### Collect last block
    if j<2*self.num_dev:
      j = 2*self.num_dev
    else:
      j+=1
    for i in range(self.num_dev):
      self.queue[3*i].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
      Axold_part[i].get(queue=self.queue[3*i+2],ary=Axold_tmp)
      Kyk1_part[i].get(queue=self.queue[3*i+2],ary=Kyk1_tmp)
      self.queue[3*i+2].finish()
      if idx_start==0:
        Axold[idx_start:idx_stop,...] = Axold_tmp[:self.par_slices,...]
        Kyk1[idx_start:idx_stop,...] = Kyk1_tmp[:self.par_slices,...]
      else:
        Axold[idx_start:idx_stop,...] = Axold_tmp[self.overlap:,...]
        Kyk1[idx_start:idx_stop,...] = Kyk1_tmp[self.overlap:,...]
      self.queue[3*i+1].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      Axold_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=Axold_tmp)
      Kyk1_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=Kyk1_tmp)
      self.queue[3*i+2].finish()
      Axold[idx_start:idx_stop,...] = Axold_tmp[self.overlap:,...]
      Kyk1[idx_start:idx_stop,...] = Kyk1_tmp[self.overlap:,...]


    xk_part = []
    Ax_old_part = []
    res_part =  []
    for i in range(self.num_dev):
      idx_start = i*self.par_slices
      idx_stop = (i+1)*self.par_slices
      if idx_stop == self.NSlice:
        idx_start -=self.overlap
      else:
        idx_stop +=self.overlap
      xk_part.append(clarray.to_device(self.queue[3*i], xk[idx_start:idx_stop,...],allocator=self.alloc[i]))
      Ax_old_part.append(clarray.to_device(self.queue[3*i], Axold[idx_start:idx_stop,...] ,allocator=self.alloc[i]))
      res_part.append(clarray.to_device(self.queue[3*i], res[idx_start:idx_stop,...] ,allocator=self.alloc[i] ))
    for i in range(self.num_dev):
      idx_start = (i+1+self.num_dev-1)*self.par_slices
      idx_stop = (i+2+self.num_dev-1)*self.par_slices
      if idx_stop == self.NSlice:
        idx_start -=self.overlap
      else:
        idx_stop +=self.overlap
      xk_part.append(clarray.to_device(self.queue[3*i+1],xk[idx_start:idx_stop,...],allocator=self.alloc[i]))
      Ax_old_part.append(clarray.to_device(self.queue[3*i+1], Axold[idx_start:idx_stop,...],allocator=self.alloc[i] ))
      res_part.append(clarray.to_device(self.queue[3*i+1], res[idx_start:idx_stop,...],allocator=self.alloc[i]  ))
    for i in range(self.num_dev):
      self.queue[3*i].finish()
      self.queue[3*i+1].finish()
    for myit in range(iters):
  #### Allocate temporary Arrays
      if myit == 0:
        x_new_part = []
        Ax_part = []
        gradx_part= []
        gradx_xold_part = []
        Ax_tmp = np.zeros((self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE)
        x_new_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE)
        gradx_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE)
        gradx_xold_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE)
        for i in range(2*self.num_dev):
          x_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
          Ax_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
          gradx_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
          gradx_xold_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
      j=0
      for i in range(self.num_dev):
        idx_start = i*self.par_slices
        idx_stop = (i+1)*self.par_slices+self.overlap
        x_part[i].set(x[idx_start:idx_stop,...],self.queue[3*i])
        Kyk1_part[i].set(Kyk1[idx_start:idx_stop,...],self.queue[3*i])
        xk_part[i].set(xk[idx_start:idx_stop,...],self.queue[3*i])
        self.coil_buf_part[i].set(self.C[idx_start:idx_stop,...],self.queue[3*i])
        self.grad_buf_part[i].set(self.grad_x[idx_start:idx_stop,...],self.queue[3*i])
      for i in range(self.num_dev):
        x_new_part[i].add_event(self.update_primal(x_new_part[i],x_part[i],Kyk1_part[i],xk_part[i],tau,delta,i,0))
        gradx_part[i].add_event(self.f_grad(gradx_part[i],x_new_part[i],i,0))
        gradx_xold_part[i].add_event(self.f_grad(gradx_xold_part[i],x_part[i],i,0))
        Ax_part[i].add_event(self.operator_forward_full(Ax_part[i],x_new_part[i],i,0))
      for i in range(self.num_dev):
        idx_start = (i+1+self.num_dev-1)*self.par_slices
        idx_stop = (i+2+self.num_dev-1)*self.par_slices
        if idx_stop == self.NSlice:
          idx_start -=self.overlap
        else:
          idx_stop +=self.overlap
        x_part[i+self.num_dev].set(x[idx_start:idx_stop,...],self.queue[3*i+1],)
        Kyk1_part[i+self.num_dev].set(Kyk1[idx_start:idx_stop,...],self.queue[3*i+1])
        xk_part[i+self.num_dev].set(xk[idx_start:idx_stop,...],self.queue[3*i+1])
        self.coil_buf_part[i+self.num_dev].set(self.C[idx_start:idx_stop,...],self.queue[3*i+1])
        self.grad_buf_part[i+self.num_dev].set(self.grad_x[idx_start:idx_stop,...],self.queue[3*i+1])
      for i in range(self.num_dev):
        x_new_part[i+self.num_dev].add_event(self.update_primal(x_new_part[i+self.num_dev],x_part[self.num_dev+i],Kyk1_part[self.num_dev+i],xk_part[self.num_dev+i],tau,delta,i,1))
        gradx_part[i+self.num_dev].add_event(self.f_grad(gradx_part[i+self.num_dev],x_new_part[i+self.num_dev],i,1))
        gradx_xold_part[i+self.num_dev].add_event(self.f_grad(gradx_xold_part[i+self.num_dev],x_part[self.num_dev+i],i,1))
        Ax_part[i+self.num_dev].add_event(self.operator_forward_full(Ax_part[i+self.num_dev],x_new_part[i+self.num_dev],i,1))
  #### Stream
      for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
#          tic = time.time()
          for i in range(self.num_dev):
            ### Get Data
            self.queue[3*i].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
            x_new_part[i].get(queue=self.queue[3*i+2],ary=x_new_tmp)
            gradx_part[i].get(queue=self.queue[3*i+2],ary=gradx_tmp)
            gradx_xold_part[i].get(queue=self.queue[3*i+2],ary=gradx_xold_tmp)
            Ax_part[i].get(queue=self.queue[3*i+2],ary=Ax_tmp)
            self.queue[3*i+2].finish()
            x_new[idx_start:idx_stop,...]= x_new_tmp[:self.par_slices,...]
            gradx[idx_start:idx_stop,...] = gradx_tmp[:self.par_slices,...]
            gradx_xold[idx_start:idx_stop,...] = gradx_xold_tmp[:self.par_slices,...]
            Ax[idx_start:idx_stop,...]=Ax_tmp[:self.par_slices,...]
            ### Put Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
            idx_stop = ((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices)+self.overlap
            x_part[i].set(x[idx_start:idx_stop,...],self.queue[3*i])
            Kyk1_part[i].set(Kyk1[idx_start:idx_stop,...],self.queue[3*i])
            xk_part[i].set(xk[idx_start:idx_stop,...],self.queue[3*i])
            self.coil_buf_part[i].set(self.C[idx_start:idx_stop,...],self.queue[3*i])
            self.grad_buf_part[i].set(self.grad_x[idx_start:idx_stop,...],self.queue[3*i])
          for i in range(self.num_dev):
            x_new_part[i].add_event(self.update_primal(x_new_part[i],x_part[i],Kyk1_part[i],xk_part[i],tau,delta,i,0))
            gradx_part[i].add_event(self.f_grad(gradx_part[i],x_new_part[i],i,0))
            gradx_xold_part[i].add_event(self.f_grad(gradx_xold_part[i],x_part[i],i,0))
            Ax_part[i].add_event(self.operator_forward_full(Ax_part[i],x_new_part[i],i,0))
          for i in range(self.num_dev):
            self.queue[3*i+1].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            x_new_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=x_new_tmp)
            gradx_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=gradx_tmp)
            gradx_xold_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=gradx_xold_tmp)
            Ax_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=Ax_tmp)
            self.queue[3*i+2].finish()
            x_new[idx_start:idx_stop,...]=x_new_tmp[:self.par_slices,...]
            gradx[idx_start:idx_stop,...] = gradx_tmp[:self.par_slices,...]
            gradx_xold[idx_start:idx_stop,...] = gradx_xold_tmp[:self.par_slices,...]
            Ax[idx_start:idx_stop,...]=Ax_tmp[:self.par_slices,...]
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            if idx_stop == self.NSlice:
              idx_start -=self.overlap
            else:
              idx_stop +=self.overlap
            x_part[i+self.num_dev].set(x[idx_start:idx_stop,...],self.queue[3*i+1],)
            Kyk1_part[i+self.num_dev].set(Kyk1[idx_start:idx_stop,...],self.queue[3*i+1])
            xk_part[i+self.num_dev].set(xk[idx_start:idx_stop,...],self.queue[3*i+1])
            self.coil_buf_part[i+self.num_dev].set(self.C[idx_start:idx_stop,...],self.queue[3*i+1])
            self.grad_buf_part[i+self.num_dev].set(self.grad_x[idx_start:idx_stop,...],self.queue[3*i+1])
          for i in range(self.num_dev):
            x_new_part[i+self.num_dev].add_event(self.update_primal(x_new_part[i+self.num_dev],x_part[self.num_dev+i],Kyk1_part[self.num_dev+i],xk_part[self.num_dev+i],tau,delta,i,1))
            gradx_part[i+self.num_dev].add_event(self.f_grad(gradx_part[i+self.num_dev],x_new_part[i+self.num_dev],i,1))
            gradx_xold_part[i+self.num_dev].add_event(self.f_grad(gradx_xold_part[i+self.num_dev],x_part[self.num_dev+i],i,1))
            Ax_part[i+self.num_dev].add_event(self.operator_forward_full(Ax_part[i+self.num_dev],x_new_part[i+self.num_dev],i,1))
  #### Collect last block
      if j<2*self.num_dev:
        j = 2*self.num_dev
      else:
        j+=1
      for i in range(self.num_dev):
        self.queue[3*i].finish()
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
        x_new_part[i].get(queue=self.queue[3*i+2],ary=x_new_tmp)
        gradx_part[i].get(queue=self.queue[3*i+2],ary=gradx_tmp)
        gradx_xold_part[i].get(queue=self.queue[3*i+2],ary=gradx_xold_tmp)
        Ax_part[i].get(queue=self.queue[3*i+2],ary=Ax_tmp)
        self.queue[3*i+2].finish()
        x_new[idx_start:idx_stop,...]= x_new_tmp[:self.par_slices,...]
        gradx[idx_start:idx_stop,...] = gradx_tmp[:self.par_slices,...]
        gradx_xold[idx_start:idx_stop,...] = gradx_xold_tmp[:self.par_slices,...]
        Ax[idx_start:idx_stop,...]=Ax_tmp[:self.par_slices,...]
        self.queue[3*i+1].finish()
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        x_new_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=x_new_tmp)
        gradx_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=gradx_tmp)
        gradx_xold_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=gradx_xold_tmp)
        Ax_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=Ax_tmp)
        self.queue[3*i+2].finish()
        if idx_stop == self.NSlice:
          x_new[idx_start:idx_stop,...]=x_new_tmp[self.overlap:,...]
          gradx[idx_start:idx_stop,...] = gradx_tmp[self.overlap:,...]
          gradx_xold[idx_start:idx_stop,...] = gradx_xold_tmp[self.overlap:,...]
          Ax[idx_start:idx_stop,...]=Ax_tmp[self.overlap:,...]
        else:
          x_new[idx_start:idx_stop,...]=x_new_tmp[:self.par_slices,...]
          gradx[idx_start:idx_stop,...] = gradx_tmp[:self.par_slices,...]
          gradx_xold[idx_start:idx_stop,...] = gradx_xold_tmp[:self.par_slices,...]
          Ax[idx_start:idx_stop,...]=Ax_tmp[:self.par_slices,...]

      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new

      while True:
        theta_line = tau_new/tau
        #### Allocate temporary Arrays
        ynorm = 0
        lhs = 0
        last=0
        if myit == 0:
          z1_new_part = []
          z1_new_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE)
          r_new_part = []
          r_new_tmp = np.zeros((self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE)
          Kyk1_new_part = []
          Kyk1_new_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE)
          for i in range(2*self.num_dev):
            z1_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
            r_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
            Kyk1_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))

        j=0
        for i in range(self.num_dev):
          idx_start = i*self.par_slices
          idx_stop = (i+1)*self.par_slices #### +1 overlap
          if idx_start == 0:
            idx_stop +=self.overlap
          else:
            idx_start -=self.overlap
          z1_part[i].set(  z1[idx_start:idx_stop,...] ,self.queue[3*i])
          gradx_part[i].set(gradx[idx_start:idx_stop,...] ,self.queue[3*i])
          gradx_xold_part[i].set( gradx_xold[idx_start:idx_stop,...] ,self.queue[3*i])
          r_part[i].set( r[idx_start:idx_stop,...] ,self.queue[3*i])
          Ax_part[i].set( Ax[idx_start:idx_stop,...] ,self.queue[3*i])
          Ax_old_part[i].set(  Axold[idx_start:idx_stop,...] ,self.queue[3*i])
          res_part[i].set(res[idx_start:idx_stop,...] ,self.queue[3*i])
          Kyk1_part[i].set( Kyk1[idx_start:idx_stop,...] ,self.queue[3*i])
          self.coil_buf_part[i].set(self.C[idx_start:idx_stop,...] ,self.queue[3*i])
          self.grad_buf_part[i].set(self.grad_x[idx_start:idx_stop,...],self.queue[3*i])
        for i in range(self.num_dev):
          z1_new_part[ i].add_event(self.update_z1_tv(z1_new_part[ i],z1_part[i],gradx_part[i],gradx_xold_part[i], beta_line*tau_new, theta_line, alpha,i,0))
          r_new_part[i].add_event(self.update_r(r_new_part[i],r_part[i],Ax_part[i],Ax_old_part[i],res_part[i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,0))
          Kyk1_new_part[ i].add_event(self.operator_adjoint_full(Kyk1_new_part[ i],r_new_part[ i],z1_new_part[ i],i,0,last))

        for i in range(self.num_dev):
          idx_start = (i+1+self.num_dev-1)*self.par_slices-self.overlap
          idx_stop = (i+2+self.num_dev-1)*self.par_slices
          z1_part[i+self.num_dev].set( z1[idx_start:idx_stop,...] ,self.queue[3*i+1])
          gradx_part[i+self.num_dev].set( gradx[idx_start:idx_stop,...] ,self.queue[3*i+1])
          gradx_xold_part[i+self.num_dev].set( gradx_xold[idx_start:idx_stop,...] ,self.queue[3*i+1])
          r_part[i+self.num_dev].set(r[idx_start:idx_stop,...] ,self.queue[3*i+1])
          Ax_part[i+self.num_dev].set( Ax[idx_start:idx_stop,...] ,self.queue[3*i+1])
          Ax_old_part[i+self.num_dev].set( Axold[idx_start:idx_stop,...] ,self.queue[3*i+1])
          res_part[i+self.num_dev].set( res[idx_start:idx_stop,...] ,self.queue[3*i+1])
          Kyk1_part[i+self.num_dev].set( Kyk1[idx_start:idx_stop,...] ,self.queue[3*i+1])
          self.coil_buf_part[i+self.num_dev].set(self.C[idx_start:idx_stop,...] ,self.queue[3*i+1])
          self.grad_buf_part[i+self.num_dev].set(self.grad_x[idx_start:idx_stop,...] ,self.queue[3*i+1])
          if idx_stop==self.NSlice:
            last=1

        for i in range(self.num_dev):
          z1_new_part[i+self.num_dev].add_event(self.update_z1_tv(z1_new_part[i+self.num_dev],z1_part[self.num_dev+i],gradx_part[self.num_dev+i],gradx_xold_part[self.num_dev+i], beta_line*tau_new, theta_line, alpha,i,1))
          r_new_part[i+self.num_dev].add_event(self.update_r(r_new_part[i+self.num_dev],r_part[self.num_dev+i],Ax_part[self.num_dev+i],Ax_old_part[self.num_dev+i],res_part[self.num_dev+i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,1))
          Kyk1_new_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_new_part[i+self.num_dev],r_new_part[i+self.num_dev],z1_new_part[i+self.num_dev],i,1,last))
      #### Stream
        for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
#          tic = time.time()
          for i in range(self.num_dev):
            ### Get Data
            self.queue[3*i].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
            z1_new_part[i].get(queue=self.queue[3*i+2],ary=z1_new_tmp)
            r_new_part[i].get(queue=self.queue[3*i+2],ary=r_new_tmp)
            Kyk1_new_part[i].get(queue=self.queue[3*i+2],ary=Kyk1_new_tmp)
            self.queue[3*i+2].finish()
            if idx_start == 0:
              z1_new[idx_start:idx_stop,...] = z1_new_tmp[:self.par_slices,...]
              r_new[idx_start:idx_stop,...] = r_new_tmp[:self.par_slices,...]
              Kyk1_new[idx_start:idx_stop,...] = Kyk1_new_tmp[:self.par_slices,...]
              ynorm += ((clarray.vdot(r_new_part[i][:self.par_slices,...]-r_part[i][:self.par_slices,...],r_new_part[i][:self.par_slices,...]-r_part[i][:self.par_slices,...],queue=self.queue[3*i])+clarray.vdot(z1_new_part[i][:self.par_slices,...]-z1_part[i][:self.par_slices,...],z1_new_part[i][:self.par_slices,...]-z1_part[i][:self.par_slices,...],queue=self.queue[3*i]))).get()
              lhs += ((clarray.vdot(Kyk1_new_part[i][:self.par_slices,...]-Kyk1_part[i][:self.par_slices,...],Kyk1_new_part[i][:self.par_slices,...]-Kyk1_part[i][:self.par_slices,...],queue=self.queue[3*i]))).get()
            else:
              z1_new[idx_start:idx_stop,...] = z1_new_tmp[self.overlap:,...]
              r_new[idx_start:idx_stop,...] = r_new_tmp[self.overlap:,...]
              Kyk1_new[idx_start:idx_stop,...] = Kyk1_new_tmp[self.overlap:,...]
              ynorm += ((clarray.vdot(r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],queue=self.queue[3*i])+clarray.vdot(z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
              lhs += ((clarray.vdot(Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
            self.queue[3*i].finish()
            idx_start = (i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices))-self.overlap
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices
            z1_part[i].set(  z1[idx_start:idx_stop,...] ,self.queue[3*i])
            gradx_part[i].set(gradx[idx_start:idx_stop,...] ,self.queue[3*i])
            gradx_xold_part[i].set( gradx_xold[idx_start:idx_stop,...] ,self.queue[3*i])
            r_part[i].set( r[idx_start:idx_stop,...] ,self.queue[3*i])
            Ax_part[i].set( Ax[idx_start:idx_stop,...] ,self.queue[3*i])
            Ax_old_part[i].set(  Axold[idx_start:idx_stop,...] ,self.queue[3*i])
            res_part[i].set(res[idx_start:idx_stop,...] ,self.queue[3*i])
            Kyk1_part[i].set( Kyk1[idx_start:idx_stop,...] ,self.queue[3*i])
            self.coil_buf_part[i].set(self.C[idx_start:idx_stop,...] ,self.queue[3*i])
            self.grad_buf_part[i].set(self.grad_x[idx_start:idx_stop,...],self.queue[3*i])
            if idx_start==self.NSlice:
              last=1
          for i in range(self.num_dev):
            z1_new_part[ i].add_event(self.update_z1_tv(z1_new_part[ i],z1_part[i],gradx_part[i],gradx_xold_part[i], beta_line*tau_new, theta_line, alpha,i,0))
            r_new_part[i].add_event(self.update_r(r_new_part[i],r_part[i],Ax_part[i],Ax_old_part[i],res_part[i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,0))
            Kyk1_new_part[ i].add_event(self.operator_adjoint_full(Kyk1_new_part[ i],r_new_part[ i],z1_new_part[ i],i,0,last))
          for i in range(self.num_dev):
            ### Get Data
            self.queue[3*i+1].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            z1_new_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=z1_new_tmp)
            r_new_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=r_new_tmp)
            Kyk1_new_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=Kyk1_new_tmp)
            self.queue[3*i+2].finish()
            z1_new[idx_start:idx_stop,...] = z1_new_tmp[self.overlap:,...]
            r_new[idx_start:idx_stop,...] = r_new_tmp[self.overlap:,...]
            Kyk1_new[idx_start:idx_stop,...] = Kyk1_new_tmp[self.overlap:,...]
            ynorm += ((clarray.vdot(r_new_part[i+self.num_dev][self.overlap:,...]-r_part[i+self.num_dev][self.overlap:,...],r_new_part[i+self.num_dev][self.overlap:,...]-r_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1])+clarray.vdot(z1_new_part[i+self.num_dev][self.overlap:,...]-z1_part[i+self.num_dev][self.overlap:,...],z1_new_part[i+self.num_dev][self.overlap:,...]-z1_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
            lhs += ((clarray.vdot(Kyk1_new_part[i+self.num_dev][self.overlap:,...]-Kyk1_part[i+self.num_dev][self.overlap:,...],Kyk1_new_part[i+self.num_dev][self.overlap:,...]-Kyk1_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
            self.queue[3*i+1].finish()
            ### Put Data
            idx_start = (i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices)-self.overlap
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            z1_part[i+self.num_dev].set( z1[idx_start:idx_stop,...] ,self.queue[3*i+1])
            gradx_part[i+self.num_dev].set( gradx[idx_start:idx_stop,...] ,self.queue[3*i+1])
            gradx_xold_part[i+self.num_dev].set( gradx_xold[idx_start:idx_stop,...] ,self.queue[3*i+1])
            r_part[i+self.num_dev].set(r[idx_start:idx_stop,...] ,self.queue[3*i+1])
            Ax_part[i+self.num_dev].set( Ax[idx_start:idx_stop,...] ,self.queue[3*i+1])
            Ax_old_part[i+self.num_dev].set( Axold[idx_start:idx_stop,...] ,self.queue[3*i+1])
            res_part[i+self.num_dev].set( res[idx_start:idx_stop,...] ,self.queue[3*i+1])
            Kyk1_part[i+self.num_dev].set( Kyk1[idx_start:idx_stop,...] ,self.queue[3*i+1])
            self.coil_buf_part[i+self.num_dev].set(self.C[idx_start:idx_stop,...] ,self.queue[3*i+1])
            self.grad_buf_part[i+self.num_dev].set(self.grad_x[idx_start:idx_stop,...] ,self.queue[3*i+1])
          for i in range(self.num_dev):
            z1_new_part[i+self.num_dev].add_event(self.update_z1_tv(z1_new_part[i+self.num_dev],z1_part[self.num_dev+i],gradx_part[self.num_dev+i],gradx_xold_part[self.num_dev+i], beta_line*tau_new, theta_line, alpha,i,1))
            r_new_part[i+self.num_dev].add_event(self.update_r(r_new_part[i+self.num_dev],r_part[self.num_dev+i],Ax_part[self.num_dev+i],Ax_old_part[self.num_dev+i],res_part[self.num_dev+i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,1))
            Kyk1_new_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_new_part[i+self.num_dev],r_new_part[i+self.num_dev],z1_new_part[i+self.num_dev],i,1,last))
            (self.update_z1_tv(z1_new_part[i+self.num_dev],z1_part[self.num_dev+i],gradx_part[self.num_dev+i],gradx_xold_part[self.num_dev+i], beta_line*tau_new, theta_line, alpha,i,1)).wait()
            (self.update_r(r_new_part[i+self.num_dev],r_part[self.num_dev+i],Ax_part[self.num_dev+i],Ax_old_part[self.num_dev+i],res_part[self.num_dev+i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,1)).wait()
            (self.operator_adjoint_full(Kyk1_new_part[i+self.num_dev],r_new_part[i+self.num_dev],z1_new_part[i+self.num_dev],i,1,last)).wait()
    #### Collect last block
        if j<2*self.num_dev:
          j = 2*self.num_dev
        else:
          j+=1
        for i in range(self.num_dev):
          self.queue[3*i].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
          z1_new_part[i].get(queue=self.queue[3*i+2],ary=z1_new_tmp)
          r_new_part[i].get(queue=self.queue[3*i+2],ary=r_new_tmp)
          Kyk1_new_part[i].get(queue=self.queue[3*i+2],ary=Kyk1_new_tmp)
          self.queue[3*i+2].finish()
          if idx_start == 0:
            z1_new[idx_start:idx_stop,...] = z1_new_tmp[:self.par_slices,...]
            r_new[idx_start:idx_stop,...] = r_new_tmp[:self.par_slices,...]
            Kyk1_new[idx_start:idx_stop,...] = Kyk1_new_tmp[:self.par_slices,...]
            ynorm += ((clarray.vdot(r_new_part[i][:self.par_slices,...]-r_part[i][:self.par_slices,...],r_new_part[i][:self.par_slices,...]-r_part[i][:self.par_slices,...],queue=self.queue[3*i])+clarray.vdot(z1_new_part[i][:self.par_slices,...]-z1_part[i][:self.par_slices,...],z1_new_part[i][:self.par_slices,...]-z1_part[i][:self.par_slices,...],queue=self.queue[3*i]))).get()
            lhs += ((clarray.vdot(Kyk1_new_part[i][:self.par_slices,...]-Kyk1_part[i][:self.par_slices,...],Kyk1_new_part[i][:self.par_slices,...]-Kyk1_part[i][:self.par_slices,...],queue=self.queue[3*i]))).get()
          else:
            z1_new[idx_start:idx_stop,...] = z1_new_tmp[self.overlap:,...]
            r_new[idx_start:idx_stop,...] = r_new_tmp[self.overlap:,...]
            Kyk1_new[idx_start:idx_stop,...] = Kyk1_new_tmp[self.overlap:,...]
            ynorm += ((clarray.vdot(r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],queue=self.queue[3*i])+clarray.vdot(z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
            lhs += ((clarray.vdot(Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
          self.queue[3*i].finish()
          self.queue[3*i+1].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          z1_new_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=z1_new_tmp)
          r_new_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=r_new_tmp)
          Kyk1_new_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=Kyk1_new_tmp)
          self.queue[3*i+2].finish()
          z1_new[idx_start:idx_stop,...] = z1_new_tmp[self.overlap:,...]
          r_new[idx_start:idx_stop,...] = r_new_tmp[self.overlap:,...]
          Kyk1_new[idx_start:idx_stop,...] = Kyk1_new_tmp[self.overlap:,...]
          ynorm += ((clarray.vdot(r_new_part[i+self.num_dev][self.overlap:,...]-r_part[i+self.num_dev][self.overlap:,...],r_new_part[i+self.num_dev][self.overlap:,...]-r_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1])+clarray.vdot(z1_new_part[i+self.num_dev][self.overlap:,...]-z1_part[i+self.num_dev][self.overlap:,...],z1_new_part[i+self.num_dev][self.overlap:,...]-z1_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
          lhs += ((clarray.vdot(Kyk1_new_part[i+self.num_dev][self.overlap:,...]-Kyk1_part[i+self.num_dev][self.overlap:,...],Kyk1_new_part[i+self.num_dev][self.overlap:,...]-Kyk1_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
          self.queue[3*i+1].finish()


        if np.sqrt(beta_line)*tau_new*(abs(lhs)**(1/2)) <= (abs(ynorm)**(1/2))*delta_line:
            break
        else:
          tau_new = tau_new*mu_line
      (Kyk1, Kyk1_new,  Axold, Ax, z1, z1_new, r, r_new) =\
      (Kyk1_new, Kyk1,  Ax, Axold, z1_new, z1, r_new, r)
      tau =  (tau_new)


      if not np.mod(myit,1):
        self.model.plot_unknowns(np.transpose(x_new,[1,0,2,3]))
        primal_new= (self.irgn_par["lambd"]/2*np.vdot(Axold-res,Axold-res)+alpha*np.sum(abs((gradx[:,:self.unknowns_TGV]))) + 1/(2*delta)*np.vdot(x_new-xk,x_new-xk)).real

        dual = (-delta/2*np.vdot(-Kyk1,-Kyk1)- np.vdot(xk,(-Kyk1))
                  - 1/(2*self.irgn_par["lambd"])*np.vdot(r,r) - np.vdot(res,r)).real

        gap = np.abs(primal_new - dual)
        if myit==0:
          gap_min = gap
        if np.abs(primal-primal_new)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"]:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(myit,abs(primal-primal_new)/(self.irgn_par["lambd"]*self.NSlice)))
          self.r = r
          self.z1 = z1
          return x_new
#        if (gap > gap_min*self.irgn_par["stag"]) and myit>1:
#          self.v = v_new
#          self.r = r
#          self.z1 = z1
#          self.z2 = z2
#          print("Terminated at iteration %d because the method stagnated"%(myit))
#          return x
        if np.abs(gap - gap_min)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"] and myit>1:
          self.r = r
          self.z1 = z1
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(myit,abs(gap - gap_min)/(self.irgn_par["lambd"]*self.NSlice)))
          return x_new
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        sys.stdout.write("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f    \r" \
                       %(myit,primal/(self.irgn_par["lambd"]*self.NSlice),dual/(self.irgn_par["lambd"]*self.NSlice),gap/(self.irgn_par["lambd"]*self.NSlice)))
        sys.stdout.flush()

      (x, x_new) = (x_new, x)
    self.r = r
    self.z1 = z1
    return x

  def grad_streamed(self,outp,inp):
    x_part = []
    gradx_part= []
    gradx_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE)
    for i in range(2*self.num_dev):
      gradx_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
      x_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
    j=0
    for i in range(self.num_dev):
      idx_start = i*self.par_slices
      idx_stop = (i+1)*self.par_slices+self.overlap
      x_part[i].set(inp[idx_start:idx_stop,...],self.queue[3*i])
    for i in range(self.num_dev):
      (self.f_grad(gradx_part[i],x_part[i],i,0)).wait()
    for i in range(self.num_dev):
      idx_start = (i+1+self.num_dev-1)*self.par_slices
      idx_stop = (i+2+self.num_dev-1)*self.par_slices
      if idx_stop == self.NSlice:
        idx_start -=self.overlap
      else:
        idx_stop +=self.overlap
      x_part[i+self.num_dev].set(inp[idx_start:idx_stop,...],self.queue[3*i+1],)
    for i in range(self.num_dev):
      (self.f_grad(gradx_part[i+self.num_dev],x_part[i+self.num_dev],i,1)).wait()
  #### Stream
    for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
  #          tic = time.time()
        for i in range(self.num_dev):
          ### Get Data
          self.queue[3*i].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
          gradx_part[i].get(queue=self.queue[3*i+2],ary=gradx_tmp)
          self.queue[3*i+2].finish()
          outp[idx_start:idx_stop,...] = gradx_tmp[:self.par_slices,...]
          ### Put Data
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
          idx_stop = ((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices)+self.overlap
          x_part[i].set(inp[idx_start:idx_stop,...],self.queue[3*i])
        for i in range(self.num_dev):
          (self.f_grad(gradx_part[i],x_part[i],i,0)).wait()
        for i in range(self.num_dev):
          self.queue[3*i+1].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          gradx_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=gradx_tmp)
          self.queue[3*i+2].finish()
          outp[idx_start:idx_stop,...] = gradx_tmp[:self.par_slices,...]
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
          if idx_stop == self.NSlice:
            idx_start -=self.overlap
          else:
            idx_stop +=self.overlap
          x_part[i+self.num_dev].set(inp[idx_start:idx_stop,...],self.queue[3*i+1],)
        for i in range(self.num_dev):
          (self.f_grad(gradx_part[i+self.num_dev],x_part[i+self.num_dev],i,1)).wait()
  #### Collect last block
    if j<2*self.num_dev:
      j = 2*self.num_dev
    else:
      j+=1
    for i in range(self.num_dev):
      self.queue[3*i].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
      gradx_part[i].get(queue=self.queue[3*i+2],ary=gradx_tmp)
      self.queue[3*i+2].finish()
      outp[idx_start:idx_stop,...] = gradx_tmp[:self.par_slices,...]
      self.queue[3*i+1].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      gradx_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=gradx_tmp)
      self.queue[3*i+2].finish()
      if idx_stop == self.NSlice:
        outp[idx_start:idx_stop,...] = gradx_tmp[self.overlap:,...]
      else:
        outp[idx_start:idx_stop,...] = gradx_tmp[:self.par_slices,...]
  def div_streamed(self,outp,inp):
    x_part = []
    gradx_part= []
    x_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE)
    for i in range(2*self.num_dev):
      gradx_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
      x_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
    j=0
    for i in range(self.num_dev):
      idx_start = i*self.par_slices
      idx_stop = (i+1)*self.par_slices
      if idx_start == 0:
        idx_stop+=self.overlap
      else:
        idx_start-=self.overlap
      gradx_part[i].set(inp[idx_start:idx_stop,...],self.queue[3*i])
    for i in range(self.num_dev):
      (self.bdiv(x_part[i],gradx_part[i],i,0)).wait()
    for i in range(self.num_dev):
      idx_start = (i+1+self.num_dev-1)*self.par_slices-self.overlap
      idx_stop = (i+2+self.num_dev-1)*self.par_slices
      gradx_part[i+self.num_dev].set(inp[idx_start:idx_stop,...],self.queue[3*i+1])
    for i in range(self.num_dev):
      (self.bdiv(x_part[i+self.num_dev],gradx_part[i+self.num_dev],i,1)).wait()
  ### Stream
    for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
        for i in range(self.num_dev):
          ## Get Data
          self.queue[3*i].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
          x_part[i].get(queue=self.queue[3*i+2],ary=x_tmp)
          self.queue[3*i+2].finish()
          if idx_start==0:
            outp[idx_start:idx_stop,...] = x_tmp[:self.par_slices,...]
          else:
            outp[idx_start:idx_stop,...] = x_tmp[self.overlap:,...]
          ### Put Data
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)-self.overlap
          idx_stop = ((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices)
          gradx_part[i].set(inp[idx_start:idx_stop,...],self.queue[3*i])
        for i in range(self.num_dev):
          (self.bdiv(x_part[i],gradx_part[i],i,0)).wait()
        for i in range(self.num_dev):
          self.queue[3*i+1].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          x_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=x_tmp)
          self.queue[3*i+2].finish()
          outp[idx_start:idx_stop,...] = x_tmp[self.overlap:,...]
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices-self.overlap
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
          gradx_part[i+self.num_dev].set(inp[idx_start:idx_stop,...],self.queue[3*i+1])
        for i in range(self.num_dev):
          (self.bdiv(x_part[i+self.num_dev],gradx_part[i+self.num_dev],i,1,1)).wait()
  #### Collect last block
    if j<2*self.num_dev:
      j = 2*self.num_dev
    else:
      j+=1
    for i in range(self.num_dev):
      self.queue[3*i].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
      x_part[i].get(queue=self.queue[3*i+2],ary=x_tmp)
      self.queue[3*i+2].finish()
      if idx_start==0:
        outp[idx_start:idx_stop,...] = x_tmp[:self.par_slices,...]
      else:
        outp[idx_start:idx_stop,...] = x_tmp[self.overlap:,...]
      self.queue[3*i+1].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      x_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=x_tmp)
      self.queue[3*i+2].finish()
      outp[idx_start:idx_stop,...] = x_tmp[self.overlap:,...]

  def symgrad_streamed(self,outp,inp):
    x_part = []
    gradx_part= []
    gradx_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,8),dtype=DTYPE)
    for i in range(2*self.num_dev):
      gradx_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,8),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
      x_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
    j=0
    for i in range(self.num_dev):
      idx_start = i*self.par_slices
      idx_stop = (i+1)*self.par_slices
      if idx_start == 0:
        idx_stop +=self.overlap
      else:
        idx_start -=self.overlap
      x_part[i].set(inp[idx_start:idx_stop,...],self.queue[3*i])
    for i in range(self.num_dev):
      (self.sym_grad(gradx_part[i],x_part[i],i,0)).wait()
    for i in range(self.num_dev):
      idx_start = (i+1+self.num_dev-1)*self.par_slices-self.overlap
      idx_stop = (i+2+self.num_dev-1)*self.par_slices
      x_part[i+self.num_dev].set(inp[idx_start:idx_stop,...],self.queue[3*i+1],)
    for i in range(self.num_dev):
      (self.sym_grad(gradx_part[i+self.num_dev],x_part[i+self.num_dev],i,1)).wait()
  #### Stream
    for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
  #          tic = time.time()
        for i in range(self.num_dev):
          ### Get Data
          self.queue[3*i].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
          gradx_part[i].get(queue=self.queue[3*i+2],ary=gradx_tmp)
          self.queue[3*i+2].finish()
          if idx_start == 0:
            outp[idx_start:idx_stop,...] = gradx_tmp[:self.par_slices,...]
          else:
            outp[idx_start:idx_stop,...] = gradx_tmp[self.overlap:,...]
          ### Put Data
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)-self.overlap
          idx_stop = ((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices)
          x_part[i].set(inp[idx_start:idx_stop,...],self.queue[3*i])
        for i in range(self.num_dev):
          (self.sym_grad(gradx_part[i],x_part[i],i,0)).wait()
        for i in range(self.num_dev):
          self.queue[3*i+1].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          gradx_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=gradx_tmp)
          self.queue[3*i+2].finish()
          outp[idx_start:idx_stop,...] = gradx_tmp[self.overlap:,...]
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices-self.overlap
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
          x_part[i+self.num_dev].set(inp[idx_start:idx_stop,...],self.queue[3*i+1],)
        for i in range(self.num_dev):
          (self.sym_grad(gradx_part[i+self.num_dev],x_part[i+self.num_dev],i,1)).wait()
  #### Collect last block
    if j<2*self.num_dev:
      j = 2*self.num_dev
    else:
      j+=1
    for i in range(self.num_dev):
      self.queue[3*i].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
      gradx_part[i].get(queue=self.queue[3*i+2],ary=gradx_tmp)
      self.queue[3*i+2].finish()
      if idx_start == 0:
        outp[idx_start:idx_stop,...] = gradx_tmp[:self.par_slices,...]
      else:
        outp[idx_start:idx_stop,...] = gradx_tmp[self.overlap:,...]
      self.queue[3*i+1].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      gradx_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=gradx_tmp)
      self.queue[3*i+2].finish()
      outp[idx_start:idx_stop,...] = gradx_tmp[self.overlap:,...]

  def symdiv_streamed(self,outp,inp):
    x_part = []
    gradx_part= []
    x_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE)
    for i in range(2*self.num_dev):
      gradx_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,8),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
      x_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
    j=0
    for i in range(self.num_dev):
      idx_start = i*self.par_slices
      idx_stop = (i+1)*self.par_slices+self.overlap
      gradx_part[i].set(inp[idx_start:idx_stop,...],self.queue[3*i])
    for i in range(self.num_dev):
      (self.sym_bdiv(x_part[i],gradx_part[i],i,0,1)).wait()
    for i in range(self.num_dev):
      idx_start = (i+1+self.num_dev-1)*self.par_slices
      idx_stop = (i+2+self.num_dev-1)*self.par_slices
      if idx_stop==self.NSlice:
        idx_start-=self.overlap
      else:
        idx_stop+=self.overlap
      gradx_part[i+self.num_dev].set(inp[idx_start:idx_stop,...],self.queue[3*i+1])
    for i in range(self.num_dev):
      (self.sym_bdiv(x_part[i+self.num_dev],gradx_part[i+self.num_dev],i,1,0)).wait()
  ### Stream
    for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
        for i in range(self.num_dev):
          ## Get Data
          self.queue[3*i].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
          x_part[i].get(queue=self.queue[3*i+2],ary=x_tmp)
          self.queue[3*i+2].finish()
          outp[idx_start:idx_stop,...] = x_tmp[:self.par_slices,...]
          ### Put Data
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
          idx_stop = ((i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices)+self.overlap
          gradx_part[i].set(inp[idx_start:idx_stop,...],self.queue[3*i])
        for i in range(self.num_dev):
          (self.sym_bdiv(x_part[i],gradx_part[i],i,0,0)).wait()
        for i in range(self.num_dev):
          self.queue[3*i+1].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          x_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=x_tmp)
          self.queue[3*i+2].finish()
          outp[idx_start:idx_stop,...] = x_tmp[:self.par_slices,...]
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
          if idx_stop==self.NSlice:
            idx_start-=self.overlap
          else:
            idx_stop+=self.overlap
          gradx_part[i+self.num_dev].set(inp[idx_start:idx_stop,...],self.queue[3*i+1])
        for i in range(self.num_dev):
          (self.sym_bdiv(x_part[i+self.num_dev],gradx_part[i+self.num_dev],i,1,0)).wait()
  #### Collect last block
    if j<2*self.num_dev:
      j = 2*self.num_dev
    else:
      j+=1
    for i in range(self.num_dev):
      self.queue[3*i].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
      x_part[i].get(queue=self.queue[3*i+2],ary=x_tmp)
      self.queue[3*i+2].finish()
      outp[idx_start:idx_stop,...] = x_tmp[:self.par_slices,...]
      self.queue[3*i+1].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      x_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=x_tmp)
      self.queue[3*i+2].finish()
      if idx_stop==self.NSlice:
        outp[idx_start:idx_stop,...] = x_tmp[self.overlap:,...]
      else:
        outp[idx_start:idx_stop,...] = x_tmp[:self.par_slices,...]

  def execute(self,TV=0,imagespace=0,reco_2D=0):

    if reco_2D:
      print("2D currently not implemented, 3D can be used with a single slice.")
      return
    else:
      if imagespace:
        print("Streamed imagespace operation is currently not implemented.")
      else:
        self.execute_3D(TV)

