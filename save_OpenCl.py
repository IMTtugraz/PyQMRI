# cython: infer_types=True
# cython: profile=False

from __future__ import division

import numpy as np
import time

import gradients_divergences_old as gd

import matplotlib.pyplot as plt
plt.ion()

DTYPE = np.complex64


import pynfft.nfft as nfft

import pyopencl as cl
import pyopencl.array as clarray
import multislice_viewer as msv
import sys

class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build()
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel


class Model_Reco: 
  def __init__(self,par,ctx,queue,traj,model):
    self.par = par
    self.traj = traj
    self.model = model
    self.C = par.C
    self.unknowns_TGV = par.unknowns_TGV
    self.unknowns_H1 = par.unknowns_H1
    self.unknowns = par.unknowns
    self.NSlice = par.NSlice
    self.NScan = par.NScan
    self.dimX = par.dimX
    self.dimY = par.dimY
    self.scale = 1#np.sqrt(par.dimX*par.dimY)
    self.NC = par.NC
    self.N = par.N
    self.Nproj = par.Nproj
    self.dz = 3
    self.fval_min = 0
    self.fval = 0
    self.ctx = ctx
    self.queue = queue             
    self.coil_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.C.data)   
    
    self.update_extra = cl.elementwise.ElementwiseKernel(self.queue.context, 'float *u_, float *u',
                                                'u[i] = 2.0f*u_[i] - u[i]')

    
    self.prg = Program(ctx, r"""
__kernel void update_p(__global float4 *p, __global float2 *u,
                       __global float4 *w,
                       const float sigma, const float alphainv) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  // gradient 
  float4 val = (float4)(-u[i].s0,-u[i].s0,-u[i].s1,-u[i].s1);
  if (x < Nx-1) 
  { val.s0 += u[i+1].s0; val.s2 += u[i+1].s1;}  
  else 
  { val.s0 = 0.0f; val.s2 = 0.0f;}
  
  if (y < Ny-1) 
  { val.s1 += u[i+Nx].s0; val.s3 += u[i+Nx].s1;} 
  else 
  { val.s1 = 0.0f; val.s3 = 0.0f; }

  // step
  val = p[i] + sigma*(val - w[i]);

  // reproject
  float fac = hypot(hypot(val.s0,val.s2), hypot(val.s1,val.s3))*alphainv;
  if (fac > 1.0f) p[i] = val/fac; else p[i] = val;
}

__kernel void update_q(__global float8 *q, __global float4 *w,
                       const float sigma, const float alphainv) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  // symmetrized gradient 
  float8 val = (float8)(w[i].s01, w[i].s01, w[i].s23, w[i].s23);
  if (x > 0) 
  { val.s01 -= w[i-1].s01;  val.s45 -= w[i-1].s23;}
  else 
  { val.s01 = (float2)(0.0f, 0.0f); val.s45 = (float2)(0.0f, 0.0f); }

  if (y > 0) 
  {val.s23 -= w[i-Nx].s01;  val.s67 -= w[i-Nx].s23;}
  else 
  {val.s23 = (float2)(0.0f, 0.0f);val.s67 = (float2)(0.0f, 0.0f);  }
  
  float8 val2 = (float8)(val.s0, val.s3, 0.5f*(val.s1 + val.s2),0.0f,
                         val.s4, val.s7, 0.5f*(val.s5 + val.s6),0.0f);

  // step
  val2 = q[i] + sigma*val2;

  // reproject
  float fac = hypot(hypot(hypot(val2.s0,val2.s4), hypot(val2.s1,val2.s5)), 2.0f*hypot(val2.s2,val.s6))*alphainv;
  if (fac > 1.0f) q[i] = val2/fac; else q[i] = val2;
}

__kernel void update_lambda(__global float2 *lambda, __global float2 *Ku,
                            __global float2 *f, const float sigma,
                            const float sigmap1inv) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  lambda[i] = (lambda[i] + sigma*(Ku[i] - f[i]))*sigmap1inv;
}

__kernel void update_u(__global float2 *u, __global float2 *u_,
                       __global float4 *p, __global float2 *Kstarlambda,
                       const float tau, const float norming) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  // divergence
  float4 val = p[i];
  if (x == Nx-1)
  {
      //real
      val.s0 = 0.0f; 
      //imag
      val.s2 = 0.0f;
  }
  if (x > 0) 
  {
      //real
      val.s0 -= p[i-1].s0; 
      //imag
      val.s2 -= p[i-1].s2;
  }
  if (y == Ny-1) 
  {
      //real
      val.s1 = 0.0f; 
      //imag
      val.s3 = 0.0f;
  }
  if (y > 0) 
  {
      //real
      val.s1 -= p[i-Nx].s1; 
      //imag
      val.s3 -= p[i-Nx].s3;
  }    


  // linear step
  u[i] = u_[i] + tau*(val.s02 + val.s13 - norming*Kstarlambda[i]);

}

__kernel void update_w(__global float4 *w, __global float4 *w_,
                       __global float4 *p, __global float8 *q,
                       const float tau) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  // divergence
  float8 val0 = -q[i];
  float8 val = (float8)(val0.s0, val0.s2, val0.s2, val0.s1,
                        val0.s4, val0.s6, val0.s6, val0.s5);
  if (x == 0)
  {   
      //real
      val.s01 = 0.0f;
      //imag
      val.s45 = 0.0f;
  }
  if (x < Nx-1) 
  {
      //real
      val.s01 += (float2)(q[i+1].s0, q[i+1].s2);
      //imag
      val.s45 += (float2)(q[i+1].s4, q[i+1].s6);
  }
  if (y == 0)   
  {
      //real
      val.s23 = 0.0f;
      //imag
      val.s67 = 0.0f;
  }
  if (y < Ny-1) 
  {
      //real
      val.s23 += (float2)(q[i+Nx].s2, q[i+Nx].s1);
      //imag
      val.s67 += (float2)(q[i+Nx].s6, q[i+Nx].s5);
  }

  // linear step
  //real
  w[i].s01 = w_[i].s01 + tau*(p[i].s01 + val.s01 + val.s23);
  //imag
  w[i].s23 = w_[i].s23 + tau*(p[i].s23 + val.s45 + val.s67);
}

__kernel void functional_discrepancy(__global float *accum,
                                __global float2 *Ku, __global float2 *f) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  float2 val = Ku[i] - f[i];
  accum[i] = hypot(val.x*val.x-val.y*val.y,2*val.x*val.y);
}

__kernel void functional_tgv(__global float *accum, __global float2 *u,
                        __global float4 *w,
                        const float alpha0, const float alpha1) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  // gradient 
  float4 val = (float4)(-u[i].s0,-u[i].s0,-u[i].s1,-u[i].s1);
  if (x < Nx-1) 
  { val.s0 += u[i+1].s0; val.s2 += u[i+1].s1;}  
  else 
  { val.s0 = 0.0f; val.s2 = 0.0f;}
  
  if (y < Ny-1) 
  { val.s1 += u[i+Nx].s0; val.s3 += u[i+Nx].s1;} 
  else 
  { val.s1 = 0.0f; val.s3 = 0.0f; }

  // symmetrized gradient
  float4 wi = w[i];
  float8 val2 = (float8)(wi.s01, wi.s01,wi.s23, wi.s23);
  if (x > 0) 
  { val2.s01 -= w[i-1].s01;  val2.s45 -= w[i-1].s23;}
  else 
  { val2.s01 = (float2)(0.0f, 0.0f); val2.s45 = (float2)(0.0f, 0.0f); }

  if (y > 0) 
  {val2.s23 -= w[i-Nx].s01;  val2.s67 -= w[i-Nx].s23;}
  else 
  {val2.s23 = (float2)(0.0f, 0.0f);val2.s67 = (float2)(0.0f, 0.0f);  }
  float8 val3 = (float8)(val2.s0, val2.s3, 0.5f*(val2.s1 + val2.s2),0.0f,
                         val2.s4, val2.s7, 0.5f*(val2.s5 + val2.s6),0.0f);
  
  

  val -= wi;
  accum[i] = alpha1*hypot(hypot(val.s0,val.s2), hypot(val.s1,val.s3))
           + alpha0*hypot(hypot(hypot(val3.s0,val3.s4), hypot(val3.s1,val3.s5)), 2.0f*hypot(val3.s2,val3.s6));
}


__kernel void radon(__global float2 *sino, __global float2 *img,
                    __constant float4 *ofs, const int X,
                    const int Y, const float scale)
{
  size_t I = get_global_size(0);
  size_t J = get_global_size(1);
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);


  float4 o = ofs[j];
  float2 acc = 0.0f;
  
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
      float2 weight = 1.0 - fabs(x*o.x + d - i);
      if (weight.x > 0.0f) acc += weight*img[x];
    }
    img += X;
  }
  sino[j*I + i] = acc/scale;
}

__kernel void radon_ad(__global float2 *img, __global float2 *sino,
                       __constant float4 *ofs, const int I,
                       const int J, const float scale)
{
  size_t X = get_global_size(0);
  size_t Y = get_global_size(1);
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);


  float4 c = (float4)(x,y,1,0);
  float2 acc = 0.0f;
  for (int j=0; j < J; j++) {
    float i = dot(c, ofs[j]);
    if ((i > -1) && (i < I)) {
      float i_floor;
      float2 w = fract(i, &i_floor);
      if (i_floor >= 0)   acc += (1.0f - w)*sino[(int)i_floor];
      if (i_floor <= I-2) acc += w*sino[(int)(i_floor+1)];
    }
    sino += I;
  }
  img[y*X + x] = acc/scale;
}
""")

    self.r_struct=self.radon_struct()
    self.scale = (self.radon_normest())
    print("Radon Norm: %f" %(self.scale))

    print("Please Set Parameters, Data and Initial images")
      
  def radon_struct(self, n_detectors=None,
                   detector_width=1.0, detector_shift=0.0):

    angles = np.reshape(np.angle(self.traj[0,:,0]),(self.Nproj))
#    angles = np.linspace(0,np.pi,(self.Nproj))
    nd = self.N
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

    ofs = np.zeros((4, len(angles)), dtype=np.float32, order='F')
    ofs[0,:] = X; ofs[1,:] = Y; ofs[2,:] = offset; ofs[3,:] = Xinv

    ofs_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ofs.data)
#    cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()
    
    sinogram_shape = (nd, len(angles))
    
    return (ofs_buf, (self.dimY,self.dimX), sinogram_shape)     
  
  def radon(self,sino, img, scan=0, wait_for=None):
      (ofs_buf, shape, sinogram_shape) = self.r_struct
      
      return self.prg.radon(sino.queue, sinogram_shape, None,
                       sino.data, img.data, ofs_buf,
                       np.int32(shape[0]), np.int32(shape[1]),
                       np.float32(self.scale),
                       wait_for=wait_for)
  
  def radon_ad(self,img, sino, scan=0, wait_for=None):
      (ofs_buf, shape, sinogram_shape) = self.r_struct
  
      return self.prg.radon_ad(img.queue, shape, None,
                          img.data, sino.data, ofs_buf,
                          np.int32(sinogram_shape[0]),
                          np.int32(sinogram_shape[1]),
                          np.float32(self.scale),
                          wait_for=wait_for)
  
  def radon_normest(self):
      img2 = np.require(np.random.randn(*(self.r_struct[1])), DTYPE, 'F')
      sino2 = np.require(np.random.randn(*(self.r_struct[2])), DTYPE, 'F')
      img = clarray.zeros(self.queue, self.r_struct[1], dtype=DTYPE, order='F')
      
      sino = clarray.to_device(self.queue, sino2)  
      img.add_event(self.radon_ad(img, sino))
      a = np.vdot(img2.flatten(),img.get().flatten())
      
      img = clarray.to_device(self.queue, img2)
      sino = clarray.zeros(self.queue, self.r_struct[2], dtype=DTYPE, order='F')
      self.radon(sino, img, wait_for=img.events)
      b = np.vdot(sino.get().flatten(),sino2.flatten())
      print("Ajointness test: %e" %(np.abs(a-b)))
      img = clarray.to_device(self.queue, np.require(np.random.randn(*self.r_struct[1]), DTYPE, 'F'))
      sino = clarray.zeros(self.queue, self.r_struct[2], dtype=DTYPE, order='F') 
      for i in range(10):
          normsqr = np.abs(clarray.sum(img).get())
          img /= normsqr
          sino.add_event(self.radon(sino, img, wait_for=img.events))
          img.add_event(self.radon_ad(img, sino, wait_for=sino.events))
  
      return np.sqrt(normsqr)

    
  def irgn_solve_2D(self, x, iters, data):
    

    ###################################
    ### Adjointness     
    xx = clarray.to_device(self.queue,np.random.random_sample((self.dimX,self.dimY)).astype(DTYPE))
    yy = clarray.to_device(self.queue,np.random.random_sample((self.Nproj,self.N)).T.astype(DTYPE))
    tmp1 = clarray.zeros_like(xx)
    tmp2 = clarray.zeros_like(yy)
    self.operator_adjoint_2D(tmp1,yy).wait()
    self.operator_forward_2D(tmp2,xx).wait()
    a = np.vdot(xx.get().flatten(),tmp1.get().flatten())
    b = np.vdot(tmp2.get().flatten(),yy.get().flatten())
    test = np.abs(a-b)
    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))
#    x_old = np.copy(x)
#    x = clarray.to_device(self.queue,x)
#    a = np.squeeze(self.operator_forward_2D(x).get())
#    b = np.squeeze(self.FT(self.step_val))
#
#    res = np.squeeze(data) - b + a
#    res = res[:,None,...]
  
#    x = self.tgv_solve_2D(x.get(),res,iters)      
    print((data.T).shape)
    alpha = 1
    for j in range(self.NScan):
       (x, values) = self.tgv_radon(np.squeeze(data.T)[:,:,j], (self.dimX,self.dimY), self.Nproj, (4*alpha, alpha), 1001,
                              record_values=True)
    
#    self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_2D(x,0)[:,None,:,:]*self.Coils))**2
#           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_1(x[:self.unknowns_TGV,...])-self.v))
#           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_2(self.v))) 
#           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)    
#    print("-"*80)
#    print ("Function value after GN-Step: %f" %(self.fval/self.irgn_par.lambd))

    return x
  
    
  def execute_2D(self):

      
#      self.FT = self.nFT_2D
#      self.FTH = self.nFTH_2D      

      gamma = self.irgn_par.gamma
      delta = self.irgn_par.delta
      
      self.result = np.zeros((self.irgn_par.max_GN_it,self.unknowns_TGV+self.unknowns_H1,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
      result = np.copy(self.model.guess)
      for islice in range(self.par.NSlice):
        self.irgn_par.gamma = gamma
        self.irgn_par.delta = delta
        self.Coils = np.array(np.squeeze(self.par.C[:,islice,:,:]),order='C')
        self.conjCoils = np.conj(self.Coils)   
        self.v = np.zeros(([self.unknowns_TGV,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
        self.r = np.zeros(([self.NScan,self.NC,self.Nproj,self.N]),dtype=DTYPE)
        self.z1 = np.zeros(([self.unknowns_TGV,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
        self.z2 = np.zeros(([self.unknowns_TGV,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
        self.z3 = np.zeros(([self.unknowns_H1,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)  
        iters = self.irgn_par.start_iters          
        for i in range(self.irgn_par.max_GN_it):
          start = time.time()       
          self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_2D(result[:,islice,:,:],islice))
    
          scale = np.linalg.norm(np.abs(self.grad_x_2D[0,...]))/np.linalg.norm(np.abs(self.grad_x_2D[1,...]))
            
          for j in range(len(self.model.constraints)-1):
            self.model.constraints[j+1].update(scale)
              
          result[1,islice,:,:] = result[1,islice,:,:]*self.model.T1_sc        
          self.model.T1_sc = self.model.T1_sc*(scale)
          result[1,islice,:,:] = result[1,islice,:,:]/self.model.T1_sc          
          self.step_val = self.model.execute_forward_2D(result[:,islice,:,:],islice)
          self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_2D(result[:,islice,:,:],islice).astype(DTYPE))
          self.grad_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grad_x_2D.data)          
          self.conj_grad_x_2D = np.conj(self.grad_x_2D)
                        
                        
          self.test = self.irgn_solve_2D(result[:,islice,:,:], iters, self.data[:,:,islice,:,:])
          self.result[i,:,islice,:,:] = result[:,islice,:,:]
          
          iters = np.fmin(iters*2,self.irgn_par.max_iters)
          self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*0.1,self.irgn_par.gamma_min)
          self.irgn_par.delta = np.minimum(self.irgn_par.delta*self.irgn_par.delta_inc, self.irgn_par.delta_max)
          
          end = time.time()-start
          print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
          print("-"*80)
          if np.abs(self.fval_min-self.fval) < self.irgn_par.lambd*self.irgn_par.tol:
            print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-       self.fval)/self.irgn_par.lambd))            
            return
          self.fval_min = np.minimum(self.fval,self.fval_min)
                 
         
  def eval_fwd(self,y,x,wait_for=None):
 
    return self.prg.operator_fwd(y.queue, (self.dimY,self.dimX), None, 
                                 y.data, x.data, self.coil_buf, self.grad_buf, 
                                 np.int32(self.NC), np.int32(1),
                                 np.int32(self.NScan),np.int32(self.unknowns),
                                 wait_for=wait_for)       
      
  def operator_forward_2D(self, y,x,wait_for=None):
    
#    return self.FT(np.sum(x[:,None,...]*self.grad_x_2D,axis=0)[:,None,...]*self.Coils)
       
#    tmp_img = clarray.to_device(self.queue,np.require(np.reshape(x.get(),(self.unknowns*self.NSlice,self.dimY,self.dimX)),DTYPE,"C"))
#    tmp_result = clarray.zeros(self.queue,(self.NScan*self.NC,self.dimY,self.dimX),DTYPE,"C")
#    tmp_result.add_event(self.eval_fwd(tmp_result,tmp_img))
#    tmp_sino = clarray.zeros(self.queue,self.r_struct[2],DTYPE,"F")#self.FT(np.reshape(tmp_result.get(),(self.NScan,self.NC,self.dimY,self.dimX)))#
    return (self.radon(y,x,wait_for=wait_for))

    
  def operator_adjoint_2D(self, y,x,wait_for=None):
    
#    return np.squeeze(np.sum(np.squeeze(np.sum(self.FTH(x)*self.conjCoils,axis=1))*self.conj_grad_x_2D,axis=1)) 
#    tmp_sino = clarray.to_device(self.queue,np.require(np.reshape(x.get(),(self.NScan*self.NC*self.NSlice,self.Nproj,self.N)),DTYPE,"C"))
#    tmp_sino = clarray.reshape(x,(self.NScan*self.NC,self.Nproj,self.N))
#    tmp_img =  clarray.zeros(self.queue,self.r_struct[1],DTYPE,"F")#clarray.to_device(self.queue,self.FTH(x.get()))#
    return (self.radon_ad(y,x,wait_for=wait_for))
#    tmp_result = clarray.zeros(self.queue,(self.unknowns,self.dimY,self.dimX),DTYPE,"C")
#    
#    tmp_result.add_event(self.eval_adj(tmp_result,tmp_img.T))
#    result = clarray.reshape(tmp_result,(self.unknowns,self.dimY,self.dimX))
#    return result

  def eval_adj(self,x,y,wait_for=None):

    return self.prg.operator_ad(x.queue, (self.dimY,self.dimX), None, 
                                 x.data, y.data, self.coil_buf, self.grad_buf, 
                                 np.int32(self.NC), np.int32(1),
                                 np.int32(self.NScan),np.int32(self.unknowns),
                                 wait_for=wait_for)      

  def eval_const(self, x, wait_for=None):
    num_const = (len(self.model.constraints))  
    min_const = np.zeros((num_const,1),dtype=np.float32)
    max_const = np.zeros((num_const,1),dtype=np.float32)
    real_const = np.zeros((num_const,1),dtype=np.int)
    for j in range(num_const):
        min_const[j] = self.model.constraints[j].min
        max_const[j] = self.model.constraints[j].max
        real_const[j] = self.model.constraints[j].real
        
#    print(x.shape[-3:])
        
    x.add_event(self.prg.box_con(x.queue, x.shape[-2:],None,
                                 x.data, min_const.data, max_const.data, real_const.data,
                                 np.float32(num_const),
                                 wait_for=wait_for))
  
   
#  
  def update_p(self,p, u, w, sigma, alpha, wait_for=[]):
    p.add_event(self.prg.update_p(p.queue, u.shape, None, p.data, u.data,
                w.data, np.float32(sigma), np.float32(1.0/alpha),
                wait_for=u.events + w.events + p.events + wait_for))

  def update_q(self,q, w, sigma, alpha, wait_for=[]):
    q.add_event(self.prg.update_q(q.queue, q.shape[1:], None, q.data, w.data,
                np.float32(sigma), np.float32(1.0/alpha),
                wait_for=q.events + w.events + wait_for))

  def update_lambda(self,lamb, Ku, f, sigma, normest, wait_for=[]):
    lamb.add_event(self.prg.update_lambda(lamb.queue, lamb.shape, None,
                   lamb.data, Ku.data, f.data, np.float32(sigma/normest),
                   np.float32(1.0/(sigma+1.0)),
                   wait_for=lamb.events + Ku.events + f.events + wait_for))

  def update_u(self,u, u_, p, Kstarlambda, tau, normest, wait_for=[]):
    u.add_event(self.prg.update_u(u.queue, u.shape, None, u.data, u_.data,
                p.data, Kstarlambda.data, np.float32(tau), np.float32(1.0/normest),
                wait_for=u.events + u_.events + p.events +
                Kstarlambda.events + wait_for))

  def update_w(self,w, w_, p, q, tau, wait_for=[]):
    w.add_event(self.prg.update_w(w.queue, w.shape[1:], None, w.data, w_.data,
                p.data, q.data, np.float32(tau),
                wait_for=w.events + w_.events + p.events + q.events +
                wait_for))


  def functional_value(self,accum_u, accum_f, u, w, Ku, f, alpha, r_struct):
    Ku.add_event(self.operator_forward_2D(Ku, u, wait_for=u.events))
    accum_f.add_event(self.prg.functional_discrepancy(Ku.queue, Ku.shape,
                      None, accum_f.data, Ku.data, f.data,
                      wait_for=Ku.events + f.events))
    accum_u.add_event(self.prg.functional_tgv(u.queue, u.shape, None,
                      accum_u.data, u.data, w.data, np.float32(alpha[0]),
                      np.float32(alpha[1]), wait_for=u.events + w.events))
    cl.wait_for_events(accum_f.events + accum_u.events)
    value = 0.5*abs(clarray.sum(accum_f).get()) \
            + abs(clarray.sum(accum_u).get())
    return value

#################
## main iteration

  def tgv_radon(self,f0, img_shape, angles, alpha, maxiter, record_values=True):
#    if angles is None:
#        angles = f0.shape[1]
#    r_struct = self.radon_struct(self.queue, img_shape, angles,
#                            n_detectors=f0.shape[0])
    
    f = clarray.to_device(self.queue, np.require(f0, DTYPE, 'F'))
    u = clarray.zeros(self.queue, img_shape, dtype=DTYPE, order='F')
    u_ = clarray.zeros_like(u)
    w = clarray.zeros(self.queue, (2,u.shape[0],u.shape[1]), dtype=DTYPE, order='F')
    w_ = clarray.zeros(self.queue, (2,u.shape[0],u.shape[1]), dtype=DTYPE, order='F')
    p = clarray.zeros(self.queue, (2,u.shape[0],u.shape[1]), dtype=DTYPE, order='F')
    q = clarray.zeros(self.queue, (4,u.shape[0],u.shape[1]), dtype=DTYPE, order='F')
    lamb = clarray.zeros_like(f)

    if record_values:
        accum_u = clarray.zeros_like(u)
        accum_f = clarray.zeros_like(f)
    
    Ku = clarray.zeros(self.queue, f0.shape, dtype=DTYPE, order='F')
    Kstarlambda = clarray.zeros_like(u)
    normest = 1#self.radon_normest()

    Lsqr = 0.5*(18.0 + np.sqrt(33))
    sigma = 1.0/np.sqrt(Lsqr)
    tau = 1.0/np.sqrt(Lsqr)
    alpha = np.array(alpha)/(self.scale**2)

    fig = None

    if record_values:
        values = []
    
    for i in range(maxiter):
        self.update_p(p, u_, w_, sigma, alpha[1])
        self.update_q(q, w_, sigma, alpha[0], wait_for=w_.events)
        Ku.add_event(self.operator_forward_2D(Ku, u_, wait_for=u_.events))
        self.update_lambda(lamb, Ku, f, sigma, normest)
        Kstarlambda.add_event(self.operator_adjoint_2D(Kstarlambda, lamb, wait_for=lamb.events))
        self.update_u(u_, u, p, Kstarlambda, tau, normest)
        self.update_w(w_, w, p, q, tau)

        if record_values:
            values.append(self.functional_value(accum_u, accum_f, u, w, Ku, f,
                                           alpha, self.r_struct))
            sys.stdout.write('Iteration %d, functional value=%f        \r' \
                             % (i, values[-1]))
            sys.stdout.flush()
        
        u.add_event(self.update_extra(u_, u, wait_for=u.events + u_.events))
        w.add_event(self.update_extra(w_, w, wait_for=w.events + w_.events))
        (u, u_, w, w_) = (u_, u, w_, w)

        if (i % 100 == 0):
            if fig == None:
                fig = plt.figure()
                disp_im = plt.imshow(np.abs(u.get()), cmap=plt.cm.gray)
                plt.title('iteration %d' %i)
                plt.draw()
                plt.pause(1e-10)
            else:
                u_cpu = np.abs(u.get())
                disp_im.set_data(u_cpu)
                disp_im.set_clim([u_cpu.min(), u_cpu.max()])
                plt.title('iteration %d' %i)
                plt.draw()
                plt.pause(1e-10)

    if record_values:
        sys.stdout.write('\n')
        return (u.get(), values)
    else:
        return u.get()

#  def nFT_2D(self, x):
#    result = np.zeros((self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE)  
#    for i in range(self.NScan):
#     for j in range(self.NC):
#         tmp_img = clarray.to_device(self.queue,np.require(np.reshape(x[i,j,...],(self.dimY,self.dimX)),DTYPE,"C"))
#         tmp_sino = clarray.zeros(self.queue,self.r_struct[i][2],DTYPE,"C")
#         (self.radon(tmp_sino,tmp_img,i))
#         result[i,j,...] = np.reshape(tmp_sino.get(),(self.Nproj,self.N))
#
#    return result
#
#
#
#  def nFTH_2D(self, x):
#    result = np.zeros((self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE)  
#    for i in range(self.NScan):
#     for j in range(self.NC):    
#        tmp_sino = clarray.to_device(self.queue,np.require(np.reshape(x[i,j,...],(self.Nproj,self.N)),DTYPE,"C"))
#        tmp_img = clarray.zeros(self.queue,self.r_struct[i][1],DTYPE,"C")
#        (self.radon_ad(tmp_img,tmp_sino,i))
#        result[i,j,...] = np.reshape(tmp_img.get(),(self.dimY,self.dimX))
#  
#    return result
  def nFT_2D(self, x):
    result = np.zeros((self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE)  
    for i in range(self.NScan):
       for j in range(self.NC):
          tmp_img = clarray.to_device(self.queue,np.require(np.reshape(x[i,j,...],(self.dimY,self.dimX)),DTYPE,"C"))
          tmp_sino = clarray.zeros(self.queue,self.r_struct[2],DTYPE,"F")
          (self.radon(tmp_sino,tmp_img.T))
          result[i,j,...] = np.reshape(tmp_sino.get().T,(self.NSlice,self.Nproj,self.N))

    return result



  def nFTH_2D(self, x):
    result = np.zeros((self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE)    
    for i in range(self.NScan):
       for j in range(self.NC):    
          tmp_sino = clarray.to_device(self.queue,np.require(np.reshape(x[i,j,...],(self.NSlice,self.Nproj,self.N)),DTYPE,"C"))
          tmp_img = clarray.zeros(self.queue,self.r_struct[1],DTYPE,"F")
          (self.radon_ad(tmp_img,tmp_sino.T))
          result[i,j,...] = np.reshape(tmp_img.get().T,(self.dimY,self.dimX))
  
    return result
            