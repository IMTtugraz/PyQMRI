
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
    self.C = par.C
    self.traj = traj
    self.model = model
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

__kernel void radon(__global float2 *sino, __global float2 *img,
                    __constant float4 *ofs, const int X,
                    const int Y, const int CS, const float scale)
{
  size_t I = get_global_size(2);
  size_t J = get_global_size(1);
  size_t i = get_global_id(2);
  size_t j = get_global_id(1);
  size_t k = get_global_id(0);
  
  size_t scan = k/CS;
    
  img += k*X*Y;

  float4 o = ofs[j+scan*J];
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
  sino[k*I*J+j*I + i] = acc/scale;
}

__kernel void radon_ad(__global float2 *img, __global float2 *sino,
                       __constant float4 *ofs, const int I,
                       const int J, const int CS, const float scale)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);
  
  size_t scan = k/CS;
  
  sino += k*I*J;

  float4 c = (float4)(x,y,1,0);
  float2 acc = 0.0f;
  for (int j=0; j < J; j++) {
    float i = dot(c, ofs[j+J*scan]);
    if ((i > -1) && (i < I)) {
      float i_floor;
      float2 w = fract(i, &i_floor);
      if (i_floor >= 0)   acc += (1.0f - w)*sino[(int)i_floor];
      if (i_floor <= I-2) acc += w*sino[(int)(i_floor+1)];
    }
    sino += I;
  }
  img[k*X*Y+y*X + x] = acc/scale;
}
__kernel void operator_fwd(__global float2 *out, __global float2 *in,
                       __global float2 *coils, __global float2 *grad, const int NCo,
                       const int NSl, const int NScan, const int Nuk)
{
  size_t X = get_global_size(0);
  size_t Y = get_global_size(1);
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  
  float2 tmp_in = 0.0f;
  float2 tmp_grad = 0.0f;
  float2 tmp_coil = 0.0f;
  float2 tmp_mul = 0.0f;

  for (int slice=0; slice < NSl; slice++)
  {
    for (int scan=0; scan<NScan; scan++)
    {
      for (int coil=0; coil < NCo; coil++)
      {
        tmp_coil = coils[coil*NSl*X*Y + slice*X*Y + y*X + x];
        float2 sum = 0.0f;
        for (int uk=0; uk<Nuk; uk++)
        {
          tmp_grad = grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + slice*X*Y + y*X + x];  
          tmp_in = in[uk*NSl*X*Y+slice*X*Y+ y*X + x];         
          
          tmp_mul = (float2)(tmp_in.x*tmp_grad.x-tmp_in.y*tmp_grad.y,tmp_in.x*tmp_grad.y+tmp_in.y*tmp_grad.x);          
          sum += (float2)(tmp_mul.x*tmp_coil.x-tmp_mul.y*tmp_coil.y,
                                                    tmp_mul.x*tmp_coil.y+tmp_mul.y*tmp_coil.x);

        }
        out[scan*NCo*NSl*X*Y+coil*NSl*X*Y+slice*X*Y + y*X + x] = sum;
      }
    }
  }
  
}
__kernel void operator_ad(__global float2 *out, __global float2 *in,
                       __global float2 *coils, __global float2 *grad, const int NCo,
                       const int NSl, const int NScan, const int Nuk)
{
  size_t X = get_global_size(0);
  size_t Y = get_global_size(1);
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  
  float2 tmp_in = 0.0f;
  float2 tmp_mul = 0.0f;
  float2 conj_grad = 0.0f;
  float2 conj_coils = 0.0f;
  
  for (int slice=0; slice < NSl; slice++)
  {
  
  for (int uk=0; uk<Nuk; uk++)
  {
  float2 sum = (float2) 0.0f;  
  for (int scan=0; scan<NScan; scan++)
  {
    conj_grad = (float2) (grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + slice*X*Y + y*X + x].x,
                          -grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + slice*X*Y + y*X + x].y);  
  for (int coil=0; coil < NCo; coil++)
  {
    conj_coils = (float2) (coils[coil*NSl*X*Y + slice*X*Y + y*X + x].x,
                                  -coils[coil*NSl*X*Y + slice*X*Y + y*X + x].y);

    tmp_in = in[scan*NCo*NSl*X*Y+coil*NSl*X*Y + slice*X*Y+ y*X + x];
    tmp_mul = (float2)(tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x);    


    sum += (float2)(tmp_mul.x*conj_coils.x-tmp_mul.y*conj_coils.y, 
                                     tmp_mul.x*conj_coils.y+tmp_mul.y*conj_coils.x);
  }
  }
  out[uk*NSl*X*Y+slice*X*Y+y*X+x] = sum; 
  }
  
  }
}  
""")

    self.r_struct=self.radon_struct()
    self.scale = (self.radon_normest())
    print("Radon Norm: %f" %(self.scale))


    print("Please Set Parameters, Data and Initial images")
      
  def radon_struct(self, n_detectors=None,
                   detector_width=1.0, detector_shift=0.0):

    angles = np.reshape(np.angle(self.traj[:,:,0]),(self.Nproj*self.NScan))
#    angles = np.linspace(0,np.pi,(self.Nproj))
    nd = self.N
    shift_read = np.array((-0.0322,-0.0314 ,  -0.0319  , -0.0315  , -0.0312   ,-0.0302   ,-0.0306   ,-0.0302  , -0.0302,       -0.0300))
    shift_phase = np.array((0.1280  ,  0.1283 ,   0.1282  ,  0.1307 ,   0.1333  ,  0.1327  ,  0.1379,    0.1395 ,   0.1293,    0.1290))
           
    midpoint_domain = np.zeros((self.NScan,2))
    for i in range(self.NScan):
#        midpoint_domain[i,:] = np.array([(self.dimX-1)/2.0-shift_read[i], (self.dimY-1)/2.0-shift_phase[i]])
      midpoint_domain[i,:] = np.array([(self.dimX-1)/2.0, (self.dimY-1)/2.0])
    midpoint_domain=np.repeat(midpoint_domain,self.Nproj,0)    
        
    midpoint_detectors = (nd-1.0)/2.0
   
    X = np.cos(angles)/detector_width
    Y = np.sin(angles)/detector_width
    Xinv = 1.0/X
   
    # set near vertical lines to vertical
    mask = np.abs(Xinv) > 10*nd
    X[mask] = 0
    Y[mask] = np.sin(angles[mask]).round()/detector_width
    Xinv[mask] = 0
   
    offset = midpoint_detectors - X*midpoint_domain[:,0] \
             - Y*midpoint_domain[:,1] + detector_shift/detector_width
             
    ofs = np.zeros((len(angles),4), dtype=np.float32, order='C')
    ofs[:,0] = X; ofs[:,1] = Y; ofs[:,2] = offset; ofs[:,3] = Xinv
   
    ofs_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ofs.data)
   #    cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()
    
    sinogram_shape = (self.NScan*self.NC*self.NSlice,self.Nproj,nd)
    
    return (ofs_buf, (self.NScan*self.NC*self.NSlice,self.dimY,self.dimX), sinogram_shape)
       
  
  def radon(self,sino, img, scan=0, wait_for=None):
      (ofs_buf, shape, sinogram_shape) = self.r_struct
      
      return self.prg.radon(sino.queue, sinogram_shape, None,
                       sino.data, img.data, ofs_buf,
                       np.int32(shape[-1]), np.int32(shape[-2]),
                       np.int32(self.NC*self.NSlice), np.float32(self.scale),
                       wait_for=wait_for)
  
  def radon_ad(self,img, sino, scan=0, wait_for=None):
      (ofs_buf, shape, sinogram_shape) = self.r_struct
  
      return self.prg.radon_ad(img.queue, shape, None,
                          img.data, sino.data, ofs_buf,
                          np.int32(sinogram_shape[-1]),
                          np.int32(sinogram_shape[-2]),
                          np.int32(self.NC*self.NSlice), np.float32(self.scale),
                          wait_for=wait_for)
  
  def radon_normest(self):
      img2 = np.require(np.random.randn(*(self.r_struct[1])), DTYPE, 'C')
      sino2 = np.require(np.random.randn(*(self.r_struct[2])), DTYPE, 'C')
      img = clarray.zeros(self.queue, self.r_struct[1], dtype=DTYPE, order='C')
      
      sino = clarray.to_device(self.queue, sino2)  
      img.add_event(self.radon_ad(img, sino))
      a = np.vdot(img2.flatten(),img.get().flatten())
      
      img = clarray.to_device(self.queue, img2)
      sino = clarray.zeros(self.queue, self.r_struct[2], dtype=DTYPE, order='C')
      self.radon(sino, img, wait_for=img.events)
      b = np.vdot(sino.get().flatten(),sino2.flatten())
      print("Ajointness test: %e" %(np.abs(a-b)))
      img = clarray.to_device(self.queue, np.require(np.random.randn(*self.r_struct[1]), DTYPE, 'C'))
      sino = clarray.zeros(self.queue, self.r_struct[2], dtype=DTYPE, order='C') 
      for i in range(10):
          normsqr = np.abs(clarray.sum(img).get())
          img /= normsqr
          sino.add_event(self.radon(sino, img, wait_for=img.events))
          img.add_event(self.radon_ad(img, sino, wait_for=sino.events))
  
      return np.sqrt(normsqr)

    
  def irgn_solve_2D(self, x, iters, data):
    

    ###################################
    ### Adjointness     
    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
    yy = np.random.random_sample(np.shape(data)).astype(DTYPE)
    a = np.vdot(xx.flatten(),self.operator_adjoint_2D(clarray.to_device(self.queue,yy)).get().flatten())
    b = np.vdot(self.operator_forward_2D(clarray.to_device(self.queue,xx)).get().flatten(),yy.flatten())
    test = np.abs(a-b)
    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))
    x_old = np.copy(x)
    
    x = clarray.to_device(self.queue,np.require(x,requirements="C"))
    a = np.squeeze(self.operator_forward_2D(x).get())
    b = np.squeeze(self.FT(self.step_val[:,None,...]*self.Coils))

    res = np.squeeze(data) - b + a
  
    x = self.tgv_solve_2D(x.get(),res,iters)      
    
    self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - np.squeeze(self.FT(self.model.execute_forward_2D(x,0)[:,None,:,:]*self.Coils)))**2
           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_1(x[:self.unknowns_TGV,...])-self.v))
           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_2(self.v))) 
           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)    
    print("-"*80)
    print ("Function value after GN-Step: %f" %(self.fval/self.irgn_par.lambd))

    return x
  
    
  def execute_2D(self):
     
      self.FT = self.nFT_2D
      self.FTH = self.nFTH_2D      
#      self.irgn_par.lambd *= self.scale

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
                        
                        
          result[:,islice,:,:] = self.irgn_solve_2D(result[:,islice,:,:], iters, self.data[:,:,islice,:,:])
          self.result[i,:,islice,:,:] = result[:,islice,:,:]
          
          iters = np.fmin(iters*2,self.irgn_par.max_iters)
          self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*0.5,self.irgn_par.gamma_min)
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
      
  def operator_forward_2D(self, x):
    
#    return self.FT(np.sum(x[:,None,...]*self.grad_x_2D,axis=0)[:,None,...]*self.Coils)
       
    tmp_img = clarray.to_device(self.queue,np.require(np.reshape(x.get(),(self.unknowns*self.NSlice,self.dimY,self.dimX)),DTYPE,"C"))
    tmp_result = clarray.zeros(self.queue,(self.NScan*self.NC,self.dimY,self.dimX),DTYPE,"C")
    tmp_result.add_event(self.eval_fwd(tmp_result,tmp_img))
    tmp_sino = clarray.zeros(self.queue,self.r_struct[2],DTYPE,"C")#self.FT(np.reshape(tmp_result.get(),(self.NScan,self.NC,self.dimY,self.dimX)))#
    (self.radon(tmp_sino,tmp_result,wait_for=tmp_result.events)).wait()
    result = clarray.reshape(tmp_sino,(self.NScan,self.NC,self.Nproj,self.N))
    return result
    
  def operator_adjoint_2D(self, x):
    
#    return np.squeeze(np.sum(np.squeeze(np.sum(self.FTH(x)*self.conjCoils,axis=1))*self.conj_grad_x_2D,axis=1)) 
    tmp_sino = clarray.to_device(self.queue,np.require(np.reshape(x.get(),(self.NScan*self.NC*self.NSlice,self.Nproj,self.N)),DTYPE,"C"))
#    tmp_sino = clarray.reshape(x,(self.NScan*self.NC,self.Nproj,self.N))
    tmp_img =  clarray.zeros(self.queue,self.r_struct[1],DTYPE,"C")#clarray.to_device(self.queue,self.FTH(x.get()))#
    (self.radon_ad(tmp_img,tmp_sino)).wait()
    tmp_result = clarray.zeros(self.queue,(self.unknowns,self.dimY,self.dimX),DTYPE,"C")
    
    tmp_result.add_event(self.eval_adj(tmp_result,tmp_img))
    result = clarray.reshape(tmp_result,(self.unknowns,self.dimY,self.dimX))
    return result

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
  
    
  def tgv_solve_2D(self, x,res, iters):
    alpha = self.irgn_par.gamma
    beta = self.irgn_par.gamma*2
    
    xx = np.zeros_like(x,dtype=DTYPE)
    yy = np.zeros_like(x,dtype=DTYPE)
    xx = np.random.random_sample(x.shape).astype(DTYPE)
    xxcl = clarray.to_device(self.queue,xx) 
    yy = self.operator_adjoint_2D(self.operator_forward_2D(xxcl)).get();
    for j in range(10):
       if not np.isclose(np.linalg.norm(yy.flatten()),0):
           xx = yy/np.linalg.norm(yy.flatten())
       else:
           xx = yy
       xxcl = clarray.to_device(self.queue,xx)    
       yy = self.operator_adjoint_2D(self.operator_forward_2D(xxcl)).get()
       l1 = np.vdot(yy.flatten(),xx.flatten());
    L = np.max(np.abs(l1)) ## Lipschitz constant estimate   
    L = 0.5*(18.0 + np.sqrt(33))
    print('L: %f'%(L))

    
    tau = 1/np.sqrt(L)
    tau_new = 0
    
    
    x = clarray.to_device(self.queue,x)
    xk = x.copy()
    x_new = clarray.zeros_like(x)
    
    r = clarray.to_device(self.queue,self.r)#np.zeros_like(res,dtype=DTYPE)
    z1 = clarray.to_device(self.queue,self.z1)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2 = clarray.to_device(self.queue,self.z2)#np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
   
    v = clarray.to_device(self.queue,self.v)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    res = clarray.to_device(self.queue, res)

    
    delta = self.irgn_par.delta
    mu = 1/delta
    
    theta_line = 1.0

    
    beta_line = 10
    beta_new = 0
    
    mu_line = 0.5
    delta_line = 1
    
    ynorm = 0.0
    lhs = 0.0

    primal = 0.0
    primal_new = 0
    dual = 0.0
    gap_min = 0.0
    gap = 0.0
        
    
    Axold = self.operator_forward_2D(x)
    
    Kyk1 = self.operator_adjoint_2D(r) - clarray.to_device(self.queue,gd.bdiv_1(z1.get()))
      
    Kyk2 = -z1 - clarray.to_device(self.queue,gd.fdiv_2(z2.get()))
    
    for i in range(iters):
        
      x_new = ((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta)
      x_new = x_new.get()      
      for j in range(len(self.model.constraints)):   
        x_new[j,...] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,x_new[j,...]))
        if self.model.constraints[j].real:
          x_new[j,...] = np.real(x_new[j,...])
      x_new = clarray.to_device(self.queue,x_new)    


      v_new = v-tau*Kyk2
      
      beta_new = beta_line*(1+mu*tau)
      
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      
      beta_line = beta_new
      
      gradx = clarray.to_device(self.queue,gd.fgrad_1(x_new.get()))
      gradx_xold = gradx - clarray.to_device(self.queue,gd.fgrad_1(x.get()))
      v_vold = v_new-v
      symgrad_v = clarray.to_device(self.queue,gd.sym_bgrad_2(v_new.get()))
      symgrad_v_vold = symgrad_v - clarray.to_device(self.queue,gd.sym_bgrad_2(v.get()))
      Ax = self.operator_forward_2D(x_new)
      Ax_Axold = Ax-Axold
    
      while True:
        
        theta_line = tau_new/tau
        
        z1_new = (z1 + beta_line*tau_new*( gradx[:self.unknowns_TGV] + theta_line*gradx_xold[:self.unknowns_TGV]
                                          - v_new - theta_line*v_vold  )).get()
        z1_new = clarray.to_device(self.queue,z1_new/np.maximum(1,(np.sqrt(np.sum(np.abs(z1_new)**2,axis=(0,1)))/alpha)))
     
        z2_new = z2 + beta_line*tau_new*( symgrad_v + theta_line*symgrad_v_vold )
        z2_new = z2_new.get()
        scal = np.sqrt( np.sum(np.abs(z2_new[:,0,:,:])**2 + np.abs(z2_new[:,1,:,:])**2 + 2*np.abs(z2_new[:,2,:,:])**2,axis=0) )

        scal = np.maximum(1,scal/(beta))

        z2_new = clarray.to_device(self.queue,z2_new/scal)
        
        tmp = Ax+theta_line*Ax_Axold

        r_new = ( r  + beta_line*tau_new*(tmp - res))/(1+beta_line*tau_new/self.irgn_par.lambd)    
        
        Kyk1_new = self.operator_adjoint_2D(r_new) - clarray.to_device(self.queue,gd.bdiv_1(z1_new.get()))
        ynorm = ((clarray.vdot(r_new-r,r_new-r)+clarray.vdot(z1_new-z1,z1_new-z1)+clarray.vdot(z2_new-z2,z2_new-z2))**(1/2)).real
        Kyk2_new = -z1_new -clarray.to_device(self.queue,gd.fdiv_2(z2_new.get()))

        
        lhs = np.sqrt(beta_line)*tau_new*((clarray.vdot(Kyk1_new-Kyk1,Kyk1_new-Kyk1)+clarray.vdot(Kyk2_new-Kyk2,Kyk2_new-Kyk2))**(1/2)).real
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
          
        self.model.plot_unknowns(x_new.get(),True)
        primal_new= (self.irgn_par.lambd/2*clarray.vdot(Ax-res,Ax-res)+alpha*clarray.sum(abs((gradx[:self.unknowns_TGV]-v))) +beta*clarray.sum(abs(symgrad_v)) + 1/(2*delta)*clarray.vdot(x_new-xk,x_new-xk)).real
      
        dual = (-delta/2*clarray.vdot(-Kyk1_new,-Kyk1_new)- clarray.vdot(xk,(-Kyk1_new)) + clarray.sum(Kyk2_new) 
                  - 1/(2*self.irgn_par.lambd)*clarray.vdot(r,r) - clarray.vdot(res,r)).real
            
        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<self.irgn_par.lambd*self.irgn_par.tol:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,abs(primal-primal_new).get()/self.irgn_par.lambd))
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          return x_new.get()
        if (gap > gap_min*self.irgn_par.stag) and i>1:
          self.v = v.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x.get()
        if np.abs(gap - gap_min)<self.irgn_par.lambd*self.irgn_par.tol and i>1:
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,abs(gap - gap_min).get()/self.irgn_par.lambd))
          return x_new.get()        
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal.get()/self.irgn_par.lambd,dual.get()/self.irgn_par.lambd,gap.get()/self.irgn_par.lambd))
#        print("Norm of primal gradient: %.3e"%(np.linalg.norm(Kyk1)+np.linalg.norm(Kyk2)))
#        print("Norm of dual gradient: %.3e"%(np.linalg.norm(tmp)+np.linalg.norm(gradx[:self.unknowns_TGV] + theta_line*gradx_xold[:self.unknowns_TGV]
#                                          - v_new - theta_line*v_vold)+np.linalg.norm( symgrad_v + theta_line*symgrad_v_vold)))
        
      x = (x_new)
      v = (v_new)
#      for j in range(self.par.unknowns_TGV):
#        self.scale_2D[j,...] = np.linalg.norm(x[j,...])
    self.v = v.get()
    self.r = r.get()
    self.z1 = z1.get()
    self.z2 = z2.get()    
    return x.get()
  


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
    tmp_img = clarray.to_device(self.queue,np.require(np.reshape(x,(self.NScan*self.NC*self.NSlice,self.dimY,self.dimX)),DTYPE,"C"))
    tmp_sino = clarray.zeros(self.queue,self.r_struct[2],DTYPE,"C")
    (self.radon(tmp_sino,tmp_img))
    result = np.reshape(tmp_sino.get(),(self.NScan,self.NC,self.NSlice,self.Nproj,self.N))

    return result



  def nFTH_2D(self, x):
    result = np.zeros((self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE)    
    tmp_sino = clarray.to_device(self.queue,np.require(np.reshape(x,(self.NScan*self.NC*self.NSlice,self.Nproj,self.N)),DTYPE,"C"))
    tmp_img = clarray.zeros(self.queue,self.r_struct[1],DTYPE,"C")
    (self.radon_ad(tmp_img,tmp_sino))
    result = np.reshape(tmp_img.get(),(self.NScan,self.NC,self.NSlice,self.dimY,self.dimX))
  
    return result