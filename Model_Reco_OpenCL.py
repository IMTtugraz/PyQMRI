
# cython: infer_types=True
# cython: profile=False

from __future__ import division

import numpy as np
import time

import gradients_divergences as gd

import matplotlib.pyplot as plt
plt.ion()

DTYPE = np.complex64


import pynfft.nfft as nfft

import pyopencl as cl
import pyopencl.array as clarray

class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build()
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel


class Model_Reco: 
  def __init__(self,par,ctx,queue):
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
    self.dz = 3
    self.fval_min = 0
    self.fval = 0
    self.ctx = ctx
    self.queue = queue

    
    self.prg = Program(self.ctx, r"""
__kernel void update_p(__global float2 *p, __global float *u,
                       __global float2 *w,
                       const float sigma, const float alphainv) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  // gradient 
  float2 val = -u[i];
  if (x < Nx-1) val.s0 += u[i+1];  else val.s0 = 0.0f;
  if (y < Ny-1) val.s1 += u[i+Nx]; else val.s1 = 0.0f;

  // step
  val = p[i] + sigma*(val - w[i]);

  // reproject
  float fac = hypot(val.s0, val.s1)*alphainv;
  if (fac > 1.0f) p[i] = val/fac; else p[i] = val;
}

__kernel void update_q(__global float3 *q, __global float2 *w,
                       const float sigma, const float alphainv) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  // symmetrized gradient 
  float4 val = (float4)(w[i], w[i]);
  if (x > 0) val.s01 -= w[i-1];  else val.s01 = (float2)(0.0f, 0.0f);
  if (y > 0) val.s23 -= w[i-Nx]; else val.s23 = (float2)(0.0f, 0.0f);
  float3 val2 = (float3)(val.s0, val.s3, 0.5f*(val.s1 + val.s2));

  // step
  val2 = q[i] + sigma*val2;

  // reproject
  float fac = hypot(hypot(val2.s0, val2.s1), 2.0f*val2.s2)*alphainv;
  if (fac > 1.0f) q[i] = val2/fac; else q[i] = val2;
}

__kernel void update_lambda(__global float *lambda, __global float *Ku,
                            __global float *f, const float sigma,
                            const float sigmap1inv) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = y*Nx + x;

  lambda[i] = (lambda[i] + sigma*(Ku[i] - f[i]))*sigmap1inv;
}

__kernel void update_u(__global float *u, __global float *u_,
                       __global float2 *p, __global float *Kstarlambda,
                       const float tau, const float norming) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = y*Nx + x;

  // divergence
  float2 val = p[i];
  if (x == Nx-1) val.s0 = 0.0f;
  if (x > 0) val.s0 -= p[i-1].s0;
  if (y == Ny-1) val.s1 = 0.0f;
  if (y > 0) val.s1 -= p[i-Nx].s1;

  // linear step
  u[i] = u_[i] + tau*(val.s0 + val.s1 - norming*Kstarlambda[i]);
}

__kernel void update_w(__global float2 *w, __global float2 *w_,
                       __global float2 *p, __global float3 *q,
                       const float tau) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = y*Nx + x;

  // divergence
  float3 val0 = -q[i];
  float4 val = (float4)(val0.s0, val0.s2, val0.s2, val0.s1);
  if (x == 0)   val.s01 = 0.0f;
  if (x < Nx-1) val.s01 += (float2)(q[i+1].s0, q[i+1].s2);
  if (y == 0)   val.s23 = 0.0f;
  if (y < Ny-1) val.s23 += (float2)(q[i+Nx].s2, q[i+Nx].s1);

  // linear step
  w[i] = w_[i] + tau*(p[i] + val.s01 + val.s23);
}

__kernel void functional_discrepancy(__global float *accum,
                                __global float *Ku, __global float *f) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  float val = Ku[i] - f[i];
  accum[i] = val*val;
}

__kernel void functional_tgv(__global float *accum, __global float *u,
                        __global float2 *w,
                        const float alpha0, const float alpha1) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  // gradient 
  float2 val = -u[i];
  if (x < Nx-1) val.s0 += u[i+1];  else val.s0 = 0.0f;
  if (y < Ny-1) val.s1 += u[i+Nx]; else val.s1 = 0.0f;

  // symmetrized gradient
  float2 wi = w[i];
  float4 val2 = (float4)(wi, wi);
  if (x > 0) val2.s01 -= w[i-1];  else val2.s01 = (float2)(0.0f, 0.0f);
  if (y > 0) val2.s23 -= w[i-Nx]; else val2.s23 = (float2)(0.0f, 0.0f);
  float3 val3 = (float3)(val2.s0, val2.s3, 0.5f*(val2.s1 + val2.s2));

  val -= wi;
  accum[i] = alpha1*hypot(val.s0, val.s1)
           + alpha0*hypot(hypot(val2.s0, val2.s1), 2.0f*val2.s2);
}


__kernel void radon(__global float *sino, __global float *img,
                    __constant float4 *ofs, const int X,
                    const int Y, const int CoilSlice)
{
  size_t I = get_global_size(2);
  size_t J = get_global_size(1);
  size_t K = get_global_size(0);
  
  size_t i = get_global_id(2);
  size_t j = get_global_id(1);
  size_t k = get_global_id(0);
  
  int scan_index = (int)(k/CoilSlice);

  float4 o = ofs[j+scan_index*J];
  float acc = 0.0f;
  
  for(int y = 0; y < Y; y++) {
    int x_low, x_high;
    float d = y*o.y + o.z;

    // compute bounds
    if (o.x == 0) {
      if ((d > i-1) && (d < i+1)) {
        x_low = 0; x_high = X-1;
      } else {
        img += X; continue;
      }
    } else if (o.x > 0) {
      x_low = (int)((i-1 - d)*o.w);
      x_high = (int)((i+1 - d)*o.w);
    } else {
      x_low = (int)((i+1 - d)*o.w);
      x_high = (int)((i-1 - d)*o.w);
    }
    x_low = max(x_low, 0);
    x_high = min(x_high, X-1);

    // integrate
    for(int x = x_low; x <= x_high; x++) {
      float weight = 1.0 - fabs(x*o.x + d - i);
      if (weight > 0.0f) acc += weight*img[x];
    }
    img += X;
  }
  sino[k*(I*J)+ j*I + i] = acc;
}

__kernel void radon_ad(__global float *img, __global float *sino,
                       __constant float4 *ofs, const int I,
                       const int J, const int CoilSlice)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t K = get_global_size(0);
  
  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);

  float4 c = (float4)(x,y,1,0);
  float acc = 0.0f;
  
  int scan_index = (int)(k/CoilSlice);
  
  for (int j=0; j < J; j++) {
    float i = dot(c, ofs[j+scan_index*J]);
    if ((i > -1) && (i < I)) {
      float i_floor;
      float w = fract(i, &i_floor);
      if (i_floor >= 0)   acc += (1.0f - w)*sino[(int)i_floor];
      if (i_floor <= I-2) acc += w*sino[(int)(i_floor+1)];
    }
    sino += I;
  }
  img[k*(X*Y)+y*X + x] = acc;
}
""")

    print("Please Set Parameters, Data and Initial images")
      
  def radon_struct(self, n_detectors=None,
                   detector_width=1.0, detector_shift=0.0):
      if np.isscalar(self.Nproj):
          angles = np.mod(111.246117975*np.arange(self.Nproj*self.NScan),360)/180*np.pi#linspace(0,pi,angles+1)[:-1]
      if n_detectors is None:
          nd = 2*np.max((self.dimX,self.dimY))#int(ceil(hypot(shape[0],shape[1])))
      else:
          nd = n_detectors
      midpoint_domain = np.array([self.dimX-1, self.dimY-1])/2.0
      midpoint_detectors = (nd-1.0)/2.0
  
      X = np.cos(angles)/detector_width
      Y = np.sin(angles)/detector_width
      Xinv = 1.0/X
  
      # set near vertical lines to vertical
      mask = abs(Xinv) > 10*nd
      X[mask] = 0
      Y[mask] = np.sin(angles[mask]).round()/detector_width
      Xinv[mask] = 0
  
      offset = midpoint_detectors - X*midpoint_domain[0] \
               - Y*midpoint_domain[1] + detector_shift/detector_width
  
      ofs = np.zeros((self.NScan*self.Nproj,4), dtype=np.float32, order='C')
      ofs[:,0] = X; ofs[:,1] = Y; ofs[:,2] = offset; ofs[:,3] = Xinv
      ofs.reshape((self.Nproj,self.NScan,4))
  
      ofs_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ofs.data)
  #    cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()
      
      sinogram_shape = (self.NSlice*self.NC*self.NScan, self.Nproj ,nd)
      
      return (ofs_buf, (self.NSlice*self.NC*self.NScan,self.dimX,self.dimY), sinogram_shape)
  
  def radon(self,sino, img, wait_for=None):
      (ofs_buf, shape, sinogram_shape) = self.r_struct
      
      return self.prg.radon(sino.queue, sinogram_shape, None,
                       sino.data, img.data, ofs_buf,
                       np.int32(shape[-2]), np.int32(shape[-1]), np.int32(self.NC*self.NSlice),
                       wait_for=wait_for)
  
  def radon_ad(self,img, sino, wait_for=None):
      (ofs_buf, shape, sinogram_shape) = self.r_struct
  
      return self.prg.radon_ad(img.queue, shape, None,
                          img.data, sino.data, ofs_buf,
                          np.int32(sinogram_shape[2]),
                          np.int32(sinogram_shape[1]), np.int32(self.NC*self.NSlice),
                          wait_for=wait_for)
  
  def radon_normest(self):
      img2 = np.require(np.random.randn(*(self.r_struct[1])), np.float32, 'C')
      sino2 = np.require(np.random.randn(*(self.r_struct[2])), np.float32, 'C')
      img = clarray.zeros(self.queue, self.r_struct[1], dtype=np.float32, order='C')
      
      
      sino = clarray.to_device(self.queue, sino2)  
      img.add_event(self.radon_ad(img, sino))
      a = np.vdot(img2.flatten(),img.get().flatten())
      
      img = clarray.to_device(self.queue, img2)
      sino = clarray.zeros(self.queue, self.r_struct[2], dtype=np.float32, order='C')
      self.radon(sino, img, wait_for=img.events)
      b = np.vdot(sino.get().flatten(),sino2.flatten())
      print("Ajointness test: %f" %(np.abs(a-b)))
  
      for i in range(10):
          normsqr = np.float(clarray.sum(img).get())
          img /= normsqr
          sino.add_event(self.radon(sino, img, wait_for=img.events))
          img.add_event(self.radon_ad(img, sino, wait_for=sino.events))
  
      return np.sqrt(normsqr)    

    
  def irgn_solve_2D(self, x, iters, data):
    

    ###################################
    ### Adjointness     
#    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
#    yy = np.random.random_sample(np.shape(data)).astype(DTYPE)
#    a = np.vdot(xx.flatten(),self.operator_adjoint_2D(yy).flatten())
#    b = np.vdot(self.operator_forward_2D(xx).flatten(),yy.flatten())
#    test = np.abs(a-b)
#    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))
    x_old = x
#    a = 
#    b = 
    res = data - self.FT(self.step_val[:,None,:,:]*self.Coils) + self.operator_forward_2D(x)
  
    x = self.tgv_solve_2D(x,res,iters)      
    self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_2D(x,0)[:,None,:,:]*self.Coils))**2
           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_1(x[:self.unknowns_TGV,...])-self.v))
           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_2(self.v))) 
           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2
           +self.irgn_par.omega/2*np.linalg.norm(gd.fgrad_1(x[-self.unknowns_H1:,...]))**2)    
    print("-"*80)
    print ("Function value after GN-Step: %f" %(self.fval/self.irgn_par.lambd))

    return x
  
  def irgn_solve_3D(self, x, iters, data):
    

    ###################################
    ### Adjointness     
#    xx = np.random.random_sample(np.shape(x)).astype('complex128')
#    yy = np.random.random_sample(np.shape(data)).astype('complex128')
#    a = np.vdot(xx.flatten(),self.operator_adjoint_3D(yy).flatten())
#    b = np.vdot(self.operator_forward_3D(xx).flatten(),yy.flatten())
#    test = np.abs(a-b)
#    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,decimal.Decimal(test)))

    x_old = x

    res = data - self.FT(self.step_val[:,None,:,:]*self.Coils3D) + self.operator_forward_3D(x)
   
    x = self.tgv_solve_3D(x,res,iters)
      
    self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_3D(x)[:,None,:,:]*self.Coils3D))**2
           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_3(x[:self.unknowns_TGV,...],1,1,self.dz)-self.v))
           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_3(self.v,1,1,self.dz))) 
           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2
           +self.irgn_par.omega/2*np.linalg.norm((x[-self.unknowns_H1:,...]))**2)  
    print("-"*80)
    print ("Function value after GN-Step: %f" %(self.fval/self.irgn_par.lambd))

    return x
        

    
  def execute_2D(self):
      self.r_struct = self.radon_struct()
      print("Radon norm: %f" %(self.radon_normest()))
      
      
#      self.FT = self.nFT_2D
#      self.FTH = self.nFTH_2D
#      gamma = self.irgn_par.gamma
#      delta = self.irgn_par.delta
#      
#      self.result = np.zeros((self.irgn_par.max_GN_it,self.unknowns_TGV+self.unknowns_H1,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
#      result = np.copy(self.model.guess)
#      for islice in range(self.par.NSlice):
#        self.irgn_par.gamma = gamma
#        self.irgn_par.delta = delta
#        self.Coils = np.array(np.squeeze(self.par.C[:,islice,:,:]),order='C')
#        self.conjCoils = np.conj(self.Coils)   
#        self.v = np.zeros(([self.unknowns_TGV,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
#        self.r = np.zeros(([self.NScan,self.NC,self.Nproj,self.N]),dtype=DTYPE)
#        self.z1 = np.zeros(([self.unknowns_TGV,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
#        self.z2 = np.zeros(([self.unknowns_TGV,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
#        self.z3 = np.zeros(([self.unknowns_H1,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)  
#        iters = self.irgn_par.start_iters          
#        for i in range(self.irgn_par.max_GN_it):
#          start = time.time()       
#          self.step_val = self.model.execute_forward_2D(result[:,islice,:,:],islice)
#          self.grad_x_2D = self.model.execute_gradient_2D(result[:,islice,:,:],islice)
#          self.conj_grad_x_2D = np.conj(self.grad_x_2D)
#                        
#          result[:,islice,:,:] = self.irgn_solve_2D(result[:,islice,:,:], iters, self.data[:,:,islice,:])
#          self.result[i,:,islice,:,:] = result[:,islice,:,:]
#          
#          iters = np.fmin(iters*2,self.irgn_par.max_iters)
#          self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*0.8,self.irgn_par.gamma_min)
#          self.irgn_par.delta = np.minimum(self.irgn_par.delta*2, self.irgn_par.delta_max)
#          
#          end = time.time()-start
#          print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
#          print("-"*80)
#          if np.abs(self.fval_min-self.fval) < self.irgn_par.lambd*self.irgn_par.tol:
#            print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-self.fval)/self.irgn_par.lambd))            
#            return
#          self.fval_min = np.minimum(self.fval,self.fval_min)
                 

        
     
  def execute_3D(self):
      self.init_plan()     
      self.FT = self.nFT_3D
      self.FTH = self.nFTH_3D
      iters = self.irgn_par.start_iters
      gamma = self.irgn_par.gamma
      delta = self.irgn_par.delta

      self.v = np.zeros(([self.unknowns_TGV,3,self.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)
      self.r = np.zeros_like(self.data,dtype=DTYPE)
      self.z1 = np.zeros(([self.unknowns_TGV,3,self.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)
      self.z2 = np.zeros(([self.unknowns_TGV,6,self.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)        
      self.z3 = np.zeros(([self.unknowns_H1,3,self.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)        
      
      self.result = np.zeros((self.irgn_par.max_GN_it+1,self.unknowns,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
      self.result[0,:,:,:,:] = np.copy(self.model.guess)

      self.Coils3D = np.squeeze(self.par.C)        
      self.conjCoils3D = np.conj(self.Coils3D)
      for i in range(self.irgn_par.max_GN_it):
        start = time.time()       

        self.step_val = np.nan_to_num(self.model.execute_forward_3D(self.result[i,:,:,:,:]))
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(self.result[i,:,:,:,:]))
        self.conj_grad_x = np.nan_to_num(np.conj(self.grad_x))
          
        self.result[i+1,:,:,:,:] = self.irgn_solve_3D(self.result[i,:,:,:,:], iters, self.data)


        iters = np.fmin(iters*2,self.irgn_par.max_iters)
        self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*0.8,gamma*1e-2)
        self.irgn_par.delta = np.minimum(self.irgn_par.delta*2,delta*1e2)
          
        end = time.time()-start
        print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
        print("-"*80)
        if np.abs(self.fval_min-self.fval) < self.irgn_par.lambd*self.irgn_par.tol*self.NSlice:
          print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-self.fval)))
          return
        self.fval_min = np.minimum(self.fval,self.fval_min)            
               
      
  def   operator_forward_2D(self,  x):

    return self.FT(np.sum(x[:,None,:,:]*self.grad_x_2D,axis=0)[:,None,:,:]*self.Coils)
    
  def   operator_adjoint_2D(self,  x):
      
    return np.squeeze(np.sum(np.squeeze(np.sum(self.FTH(x)*self.conjCoils,axis=1))*self.conj_grad_x_2D,axis=1))      
  
  def   operator_forward_3D(self,  x):
      
    return self.FT(np.sum(x[:,None,:,:,:]*self.grad_x,axis=0)[:,None,:,:,:]*self.Coils3D)

    
  def   operator_adjoint_3D(self,  x):
      
    return np.squeeze(np.sum(np.squeeze(np.sum(self.FTH(x)*self.conjCoils3D,axis=1)*self.conj_grad_x),axis=1))    
  
    
  def tgv_solve_2D(self, x,res, iters):
    alpha = self.irgn_par.gamma
    beta = self.irgn_par.gamma*2
    
    xx = np.zeros_like(x,dtype=DTYPE)
    yy = np.zeros_like(x,dtype=DTYPE)
    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
    yy = self.operator_adjoint_2D(self.operator_forward_2D(xx));
    for j in range(10):
       if not np.isclose(np.linalg.norm(yy.flatten()),0):
           xx = yy/np.linalg.norm(yy.flatten())
       else:
           xx = yy
       yy = self.operator_adjoint_2D(self.operator_forward_2D(xx))
       l1 = np.vdot(yy.flatten(),xx.flatten());
    L = np.max(np.abs(l1)) ## Lipschitz constant estimate   
#    L1 = np.max(np.abs(self.grad_x[0,:,None,:,:]*self.Coils
#                                   *np.conj(self.grad_x[0,:,None,:,:])*np.conj(self.Coils)))
#    L2 = np.max(np.abs(self.grad_x[1,:,None,:,:]*self.Coils
#                                   *np.conj(self.grad_x[1,:,None,:,:])*np.conj(self.Coils)))
#
#    L = np.max((L1,L2))*self.unknowns*self.par.NScan*self.par.NC*sigma0*tau0+1
    L = (L**2+8**2+16**2)
    print('L: %f'%(L))
#    print("Operatornorm estimate L: %f "%(L))   
#    L = 320 #### worked always ;)
    
    
    tau = 1/np.sqrt(L)
    tau_new = 0
    
    xk = x
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
#      x_new[1,:,:] = np.real(np.maximum(self.model.min_T1,np.minimum(self.model.max_T1,x_new[1,...]))) 
#      x_new[2,...] = np.real(np.maximum(np.minimum(self.par.fa*1.4,x_new[2,...]),self.par.fa/1.4))


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
          Z3_new = z3_new/(1+beta_line*tau_new/self.irgn_par.omega)
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
             
      Kyk1 = (Kyk1_new)
      Kyk2 =  (Kyk2_new)
      Axold =(Ax)
      z1 = (z1_new)
      z2 = (z2_new)
      if self.unknowns_H1 > 0:
        z3 = (z3_new)
      r =  (r_new)
      tau =  (tau_new)
        
        
      if not np.mod(i,20):
          
        self.model.plot_unknowns(x_new,True)
        if self.unknowns_H1 > 0:
          primal_new= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*np.sum(np.abs((gradx[:self.unknowns_TGV]-v))) +
                   beta*np.sum(np.abs(symgrad_v)) + 1/(2*delta)*np.linalg.norm((x_new-xk).flatten())**2
                   +self.irgn_par.omega/2*np.linalg.norm(gradx[-self.unknowns_H1:,...]-self.par.fa)**2)
      
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
        
      x = (x_new)
      v = (v_new)
#      for j in range(self.par.unknowns_TGV):
#        self.scale_2D[j,...] = np.linalg.norm(x[j,...])
    self.v = v
    self.r = r
    self.z1 = z1
    self.z2 = z2
    if self.unknowns_H1 > 0:
      self.z3 = z3
    
    return x
  
  def tgv_solve_3D(self,   x,   res,  iters):
    alpha = self.irgn_par.gamma
    beta = self.irgn_par.gamma*2
    
    dz = self.dz
    
    L = (8**2+16**2)

    
    tau = 1/np.sqrt(L)
    tau_new = 0   
    
    xk = x
    x_new = np.zeros_like(x,dtype=DTYPE)
    
    r = np.copy(self.r)#np.zeros_like(res,dtype=DTYPE)
    z1 = np.copy(self.z1)#np.zeros(([self.unknowns,3,self.par.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2 = np.copy(self.z2)#np.zeros(([self.unknowns,6,self.par.NSlice,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    v = np.copy(self.v)#np.zeros_like(z1,dtype=DTYPE)
    
    r_new = np.zeros_like(r,dtype=DTYPE)
    z1_new = np.zeros_like(z1,dtype=DTYPE)
    z2_new = np.zeros_like(z2,dtype=DTYPE)
    v_new = np.zeros_like(v,dtype=DTYPE)
    
    z3_new = np.zeros_like(self.z3,dtype=DTYPE)   
    z3 = self.z3#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)     

    Kyk1 = np.zeros_like(x,dtype=DTYPE)
    Kyk2 = np.zeros_like(z1,dtype=DTYPE)
    
    Ax = np.zeros_like(res,dtype=DTYPE)
    Ax_Axold = np.zeros_like(Ax,dtype=DTYPE)
    Axold = np.zeros_like(Ax,dtype=DTYPE)    
    tmp = np.zeros_like(Ax,dtype=DTYPE)    
    
    Kyk1_new = np.zeros_like(Kyk1,dtype=DTYPE)
    Kyk2_new = np.zeros_like(Kyk2,dtype=DTYPE)
    
    
    delta = self.irgn_par.delta
    mu = 1/delta
    
    theta_line = 1.0

    
    beta_line = 400
    beta_new = 0
    
    mu_line = 0.5
    delta_line = 1
    
    scal = np.zeros((self.par.NSlice,self.par.dimX,self.par.dimY),dtype=DTYPE)
    
    ynorm = 0
    lhs = 0

    primal = 0.0
    primal_new = 0
    dual = 0.0
    gap_min = 0.0
    gap = 0.0

    
    gradx = np.zeros_like(z1,dtype=DTYPE)
    gradx_xold = np.zeros_like(z1,dtype=DTYPE)
    
    v_vold = np.zeros_like(v,dtype=DTYPE)
    symgrad_v = np.zeros_like(z2,dtype=DTYPE)
    symgrad_v_vold = np.zeros_like(z2,dtype=DTYPE)
    
    Axold = self.operator_forward_3D(x)
    if self.unknowns_H1 > 0:
      Kyk1 = self.operator_adjoint_3D(r) - np.concatenate((gd.bdiv_3(z1,1,1,dz),np.zeros_like(gd.bdiv_3(z3,1,1,dz))),0)
    else:
      Kyk1 = self.operator_adjoint_3D(r) - gd.bdiv_3(z1)
    Kyk2 = -z1 - gd.fdiv_3(z2,1,1,dz)     

    
    for i in range(iters):
      x_new = ((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta)      
      if self.unknowns_H1 > 0:
        x_new[-self.unknowns_H1:,...] = x_new[-self.unknowns_H1:,...]*(1+tau/delta)/(1+tau/delta+tau*self.irgn_par.omega)
      
      for j in range(len(self.model.constraints)):   
        x_new[j,...] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,x_new[j,...]))
        if self.model.constraints[j].real:
          x_new[j,...] = np.real(x_new[j,...])

            
      v_new = v-tau*Kyk2   
      beta_new = beta_line*(1+mu*tau)
      
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
#      tau_new = tau*np.sqrt((1+theta_line))
#      tau_new = tau*np.sqrt(beta_line/beta_new)      
      beta_line = beta_new
      
      gradx = gd.fgrad_3(x_new,1,1,dz)
      gradx_xold = gradx - gd.fgrad_3(x,1,1,dz)  
      v_vold = v_new-v
      symgrad_v = gd.sym_bgrad_3(v_new,1,1,dz)
      symgrad_v_vold = symgrad_v - gd.sym_bgrad_3(v,1,1,dz)     
      Ax = self.operator_forward_3D(x_new)
      Ax_Axold = Ax-Axold
      while True:
        
        theta_line = tau_new/tau
        
        z1_new = z1 + beta_line*tau_new*( gradx[:self.unknowns_TGV] + theta_line*gradx_xold[:self.unknowns_TGV] - v_new - theta_line*v_vold  )
        z1_new = z1_new/np.maximum(1,(np.sqrt(np.sum(z1_new**2,axis=(0,1)))/alpha))

        z2_new = z2 + beta_line*tau_new*( symgrad_v + theta_line*symgrad_v_vold )
        scal = np.sqrt( np.sum(z2_new[:,0,:,:,:]**2 + z2_new[:,1,:,:,:]**2 +
                    z2_new[:,2,:,:,:]**2+ 2*z2_new[:,3,:,:,:]**2 + 
                    2*z2_new[:,4,:,:,:]**2+2*z2_new[:,5,:,:,:]**2,axis=0))
        scal = np.maximum(1,scal/(beta))
        z2_new = z2_new/scal
        
        
        tmp = Ax+theta_line*Ax_Axold
        r_new = (( r  + beta_line*tau_new*(tmp) ) - beta_line*tau_new*res)/(1+beta_line*tau_new/self.irgn_par.lambd)
        
        if self.unknowns_H1 > 0:
#          z3_new = z3 + beta_line*tau_new*( gradx[-self.unknowns_H1:,...] + theta_line*gradx_xold[-self.unknowns_H1:,...])  
#          Z3_new = z3_new/(1+beta_line*tau_new/self.irgn_par.omega)
          Kyk1_new = self.operator_adjoint_3D(r_new) - np.concatenate((gd.bdiv_3(z1_new,1,1,dz),np.zeros_like(gd.bdiv_3(z3_new,1,1,dz))),0)
          ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),(z1_new-z1).flatten(),(z2_new-z2).flatten(),(z3_new-z3).flatten()]))
        else:
          Kyk1_new = self.operator_adjoint_3D(r_new) - gd.bdiv_3(z1_new)
          ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),(z1_new-z1).flatten(),(z2_new-z2).flatten()]))
          
        Kyk2_new = -z1_new -gd.fdiv_3(z2_new,1,1,dz)
        
        ynorm = np.linalg.norm(np.concatenate([(r_new-r).flatten(),(z1_new-z1).flatten(),(z2_new-z2).flatten()]))
        lhs = np.sqrt(beta_line)*tau_new*np.linalg.norm(np.concatenate([(Kyk1_new-Kyk1).flatten(),(Kyk2_new-Kyk2).flatten()]))        
        if lhs <= ynorm*delta_line:
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

        
      if not np.mod(i,20):
        self.model.plot_unknowns(x)
        if self.unknowns_H1 > 0:
          primal_new= np.real(self.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2+alpha*np.sum(np.abs((gradx[:self.unknowns_TGV]-v))) +
                   beta*np.sum(np.abs(symgrad_v)) + 1/(2*delta)*np.linalg.norm((x_new-xk).flatten())**2
                   +self.irgn_par.omega/2*np.linalg.norm(x[-self.unknowns_H1:,...])**2)
      
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
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal/(self.irgn_par.lambd*self.par.NSlice),dual/(self.irgn_par.lambd*self.par.NSlice),gap/(self.irgn_par.lambd*self.par.NSlice)))


        
      x = x_new
      v = v_new    
    self.v = v
    self.z1 = z1
    self.z2 = z2
    self.r = r
    return x  
  
  
  
  def   FT_2D(self,  x):
   
    return self.fft_forward(x)/np.sqrt(np.shape(x)[2]*np.shape(x)[3])

      
  def   FTH_2D(self,  x):
      
    return self.fft_back(x)*np.sqrt(np.shape(x)[2]*np.shape(x)[3])


  def   nFT_2D(self,   x):

    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]   
    result = np.zeros((nscan,NC,self.par.Nproj*self.par.N),dtype=DTYPE)
    scal = self.scale
    plan = self._plan
    for scan in range(nscan):
      for coil in range(NC):
          plan[scan][coil].f_hat = x[scan,coil,:,:]/scal
          result[scan,coil,:] = plan[scan][coil].trafo()
      
    return np.reshape(result*self.dcf_flat,[nscan,NC,self.par.Nproj,self.par.N])



  def   nFTH_2D(self,   x):
    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]     
    result = np.zeros((nscan,NC,self.par.dimX,self.par.dimY),dtype=DTYPE)
    dcf = self.dcf
    plan = self._plan
    for scan in range(nscan):
        for coil in range(NC):  
            plan[scan][coil].f = x[scan,coil,:,:]*dcf
            result[scan,coil,:,:] = plan[scan][coil].adjoint()
      
    return result/self.scale
      
  
  
  def    nFT_3D(self,   x):

    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]   
    NSlice = self.par.NSlice
    result = np.zeros((nscan,NC,NSlice,self.par.Nproj*self.par.N),dtype=DTYPE)
    plan = self._plan    
    scal = self.scale
    for scan in range(nscan):
      for coil in range(NC):
        for islice in range(NSlice):
          plan[scan][coil].f_hat = x[scan,coil,islice,:,:]/scal
          result[scan,coil,islice,:] = plan[scan][coil].trafo()
      
    return np.reshape(result*self.dcf_flat,[nscan,NC,NSlice,self.par.Nproj,self.par.N])



  def    nFTH_3D(self,   x):
    nscan = np.shape(x)[0]
    NC = np.shape(x)[1]  
    NSlice = self.par.NSlice
    result = np.zeros((nscan,NC,NSlice,self.par.dimX,self.par.dimY),dtype=DTYPE)
    dcf = self.dcf    
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

        plan.append([])
        points = np.transpose(np.array([traj_x[i,:,:].flatten(),traj_y[i,:,:].flatten()]))      
        for j in range(self.NC):

            plan[i].append(nfft.NFFT([self.dimX,self.dimY],self.N*self.Nproj))
            plan[i][j].x = points
            plan[i][j].precompute()
    self._plan = plan

