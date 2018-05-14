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
    self.C = par.C.T
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
    self.E1_scale = []         
    self.coil_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.C.data)   
    self.update_extra = cl.elementwise.ElementwiseKernel(self.queue.context, 'float2 *u_, float2 *u',
                                                'u[i] = 2.0f*u_[i] - u[i]')

    
    self.prg = Program(ctx, r"""
__kernel void update_p(__global float4 *p, __global float2 *u,
                       __global float4 *w,
                       const float sigma, const float alphainv, const int NUk) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t NSl = get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t k = get_global_id(2);
  size_t i = k*Nx*Ny+Nx*y + x;
  
  float4 val[2];
  float fac = 0.0f;
  
  for (int uk=0; uk<NUk; uk++)
  {
     // gradient 
     val[uk] = (float4)(-u[i].s0,-u[i].s0,-u[i].s1,-u[i].s1);
     if (x < Nx-1) 
     { val[uk].s0 += u[i+1].s0; val[uk].s2 += u[i+1].s1;}  
     else 
     { val[uk].s0 = 0.0f; val[uk].s2 = 0.0f;}
     
     if (y < Ny-1) 
     { val[uk].s1 += u[i+Nx].s0; val[uk].s3 += u[i+Nx].s1;} 
     else 
     { val[uk].s1 = 0.0f; val[uk].s3 = 0.0f; }
   
     // step
     val[uk] = p[i] + sigma*(val[uk] - w[i]);
   
     // reproject
     fac = hypot(fac,hypot(hypot(val[uk].s0,val[uk].s2), hypot(val[uk].s1,val[uk].s3))*alphainv);
     i += NSl*Nx*Ny;     
  }
  i = k*Nx*Ny+Nx*y + x;  
  for (int uk=0; uk<NUk; uk++)
  {
    if (fac > 1.0f) p[i] = val[uk]/fac; else p[i] = val[uk];
    i += NSl*Nx*Ny;    
  }
}
__kernel void gradient(__global float4 *grad, __global float2 *u, const int NUk) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t NSl = get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t k = get_global_id(2);
  size_t i = k*Nx*Ny+Nx*y + x;
  
  
  for (int uk=0; uk<NUk; uk++)
  {
     // gradient 
     grad[i] = (float4)(-u[i].s0,-u[i].s0,-u[i].s1,-u[i].s1);
     if (x < Nx-1) 
     { grad[i].s0 += u[i+1].s0; grad[i].s2 += u[i+1].s1;}  
     else 
     { grad[i].s0 = 0.0f; grad[i].s2 = 0.0f;}
     
     if (y < Ny-1) 
     { grad[i].s1 += u[i+Nx].s0; grad[i].s3 += u[i+Nx].s1;} 
     else 
     { grad[i].s1 = 0.0f; grad[i].s3 = 0.0f; }
     i += NSl*Nx*Ny;       
  }   
}

__kernel void update_q(__global float8 *q, __global float4 *w,
                       const float sigma, const float alphainv, const int NUk) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t NSl = get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t k = get_global_id(2);
  size_t i = k*Nx*Ny+Nx*y + x;
  
  float fac = 0.0f;
  float8 val2[2];
  
  for (int uk=0; uk<NUk; uk++)
  {
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
     
     val2[uk] = (float8)(val.s0, val.s3, 0.5f*(val.s1 + val.s2),0.0f,
                            val.s4, val.s7, 0.5f*(val.s5 + val.s6),0.0f);
   
     // step
     val2[uk] = q[i] + sigma*val2[uk];
   
     // reproject
     fac = hypot(fac,hypot(hypot(hypot(val2[uk].s0,val2[uk].s4), hypot(val2[uk].s1,val2[uk].s5)),
                           2.0f*hypot(val2[uk].s2,val2[uk].s6))*alphainv);
     i += NSl*Nx*Ny;  
   }
  
  i = k*Nx*Ny+Nx*y + x;  
  for (int uk=0; uk<NUk; uk++)
  {
    if (fac > 1.0f) q[i] = val2[uk]/fac; else q[i] = val2[uk];
    i += NSl*Nx*Ny;    
  }
}
__kernel void sym_grad(__global float8 *sym, __global float4 *w, const int NUk) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t NSl = get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t k = get_global_id(2);
  size_t i = k*Nx*Ny+Nx*y + x;
  
  
  for (int uk=0; uk<NUk; uk++)
  {
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
     
     sym[i] = (float8)(val.s0, val.s3, (val.s1 + val.s2),0.0f,
                            val.s4, val.s7, (val.s5 + val.s6),0.0f);
     i += NSl*Nx*Ny;       
   }
}
__kernel void update_lambda(__global float2 *lambda, __global float2 *Ku,
                            __global float2 *f, const float sigma,
                            const float sigmap1inv) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t k = get_global_id(2);
  size_t i = k*Nx*Ny+Nx*y + x;

  lambda[i] = (lambda[i] + sigma*(Ku[i] - f[i]))*sigmap1inv;
}

__kernel void update_u(__global float2 *u, __global float2 *u_,
                       __global float4 *p, __global float2 *Kstarlambda,
                       __global float2 *uk,
                       const float tau, const float norming, const float delta, const int NUk) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t NSl = get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t k = get_global_id(2);
  size_t i = k*Nx*Ny+Nx*y + x;
  
  for (int ukn=0; ukn<NUk; ukn++)
  {
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
     u[i] = (u_[i] + tau*(val.s02 + val.s13 - norming*Kstarlambda[i]+1/delta*uk[i]))/(1+tau/delta);
     i += NSl*Nx*Ny;     
  }

}

__kernel void update_w(__global float4 *w, __global float4 *w_,
                       __global float4 *p, __global float8 *q,
                       const float tau, const int NUk) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t NSl = get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t k = get_global_id(2);
  size_t i = k*Nx*Ny+Nx*y + x;
  
  for (int uk=0; uk<NUk; uk++)
  {
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
     i += NSl*Nx*Ny;     
  }
}

__kernel void functional_discrepancy(__global float *accum,
                                __global float2 *Ku, __global float2 *f) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t k = get_global_id(2);
  size_t i = k*Nx*Ny+Nx*y + x;

  float2 val = Ku[i] - f[i];
  accum[i] = hypot(val.x*val.x-val.y*val.y,2*val.x*val.y);
}

__kernel void functional_tgv(__global float *accum, __global float2 *u,
                        __global float4 *w,
                        const float alpha0, const float alpha1, const int NUk) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t NSl = get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t k = get_global_id(2);
  size_t i = k*Nx*Ny+Nx*y + x;
  
  for (int uk=0; uk<NUk; uk++)
  {
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
     i += NSl*Nx*Ny;           
  }
}
__kernel void radon(__global float2 *sino, __global float2 *img,
                    __constant float4 *ofs, const int X,
                    const int Y, const int CS, const float scale)
{
  size_t I = get_global_size(0);
  size_t J = get_global_size(1);
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k = get_global_id(2);
  
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
  size_t X = get_global_size(0);
  size_t Y = get_global_size(1);
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t k = get_global_id(2);
  
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
    shift_read = np.array((0.0294,0.0319,0.0301,0.0306,0.0311,0.0299,0.0318,0.0296,0.0281,0.0262))
    shift_phase = np.array((0.0904,0.0884,0.0884,0.0908,0.0912,0.0895,0.0910,0.0957,0.0968,0.0996))
           
    midpoint_domain = np.zeros((self.NScan,2))
    for i in range(self.NScan):
        midpoint_domain[i,:] = np.array([(self.dimX-1)/2.0-shift_read[i], (self.dimY-1)/2.0-shift_phase[i]])
#      midpoint_domain[i,:] = np.array([(self.dimX-1)/2.0, (self.dimY-1)/2.0])
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

    ofs = np.zeros((4, len(angles)), dtype=np.float32, order='F')
    ofs[0,:] = X; ofs[1,:] = Y; ofs[2,:] = offset; ofs[3,:] = Xinv

    ofs_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ofs.data)
#    cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()
    
    sinogram_shape = (nd, self.Nproj,self.NC*self.NScan*self.NSlice)
    
    return (ofs_buf, (self.dimY,self.dimX,self.NC*self.NScan*self.NSlice), sinogram_shape)     
  
  def radon(self,sino, img, scan=0, wait_for=None):
      (ofs_buf, shape, sinogram_shape) = self.r_struct
      
      return self.prg.radon(sino.queue, sinogram_shape, None,
                       sino.data, img.data, ofs_buf,
                       np.int32(shape[0]), np.int32(shape[1]),
                       np.int32(self.NC*self.NSlice),
                       np.float32(self.scale),
                       wait_for=wait_for)
  
  def radon_ad(self,img, sino, scan=0, wait_for=None):
      (ofs_buf, shape, sinogram_shape) = self.r_struct
  
      return self.prg.radon_ad(img.queue, shape, None,
                          img.data, sino.data, ofs_buf,
                          np.int32(sinogram_shape[0]),
                          np.int32(sinogram_shape[1]),
                          np.int32(self.NC*self.NSlice),
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
    xx = clarray.to_device(self.queue,np.random.random_sample((self.unknowns,self.NSlice,self.dimX,self.dimY)).T.astype(DTYPE))
    yy = clarray.to_device(self.queue,np.random.random_sample((self.NScan*self.NC*self.NSlice,self.Nproj,self.N)).T.astype(DTYPE))
    tmp1 = clarray.zeros_like(xx)
    tmp2 = clarray.zeros_like(yy)
    self.operator_adjoint_2D(tmp1,yy)
    self.operator_forward_2D(tmp2,xx)
    a = np.vdot(xx.get().flatten(),tmp1.get().flatten())
    b = np.vdot(tmp2.get().flatten(),yy.get().flatten())
    test = np.abs(a-b)
    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))
    x_old = np.squeeze(np.copy(x))
    x = clarray.to_device(self.queue,x)
    a = clarray.zeros(self.queue,(self.NScan*self.NC*self.NSlice,self.Nproj,self.N),dtype=DTYPE,order='F')
    self.operator_forward_2D(a,x.T)
    a = np.reshape(a.get().T,(self.NScan,self.NC,self.NSlice,self.Nproj,self.N))
    b = np.squeeze(self.FT(self.step_val[:,None,...]*self.Coils))

    res = np.squeeze(data) - b + a
    alpha = self.irgn_par.gamma
    del xx, yy, tmp1, tmp2, a, b
    res = np.reshape(res,(self.NScan*self.NC*self.NSlice,self.Nproj,self.N))
    (x,v, values) = self.tgv_radon(res.T, x.get().T, (self.dimX,self.dimY,self.NSlice), self.Nproj, (2*alpha, alpha), iters,
                              record_values=True)

    
    self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - np.squeeze(self.FT(self.model.execute_forward_3D(x)[:,None,...]*self.Coils)))**2
           +self.irgn_par.gamma*np.sum(np.abs(gd.fgrad_3(x[:self.unknowns_TGV,...])-v))
           +self.irgn_par.gamma*(2)*np.sum(np.abs(gd.sym_bgrad_3(v))) 
           +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)    
    print("-"*80)
    print ("Function value after GN-Step: %f" %(self.fval/self.irgn_par.lambd))

    return x
  
    
  def execute_2D(self):

      
      self.FT = self.nFT_2D
      self.FTH = self.nFTH_2D      

      gamma = self.irgn_par.gamma
      delta = self.irgn_par.delta
      
      self.result = np.zeros((self.irgn_par.max_GN_it,self.unknowns_TGV+self.unknowns_H1,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
      result = np.copy(self.model.guess)
      self.irgn_par.gamma = gamma
      self.irgn_par.delta = delta
      self.Coils = np.array(np.squeeze(self.par.C),order='C')
      self.conjCoils = np.conj(self.Coils)   
      self.v = np.zeros((2,self.dimX,self.dimY,self.NSlice,self.unknowns), dtype=DTYPE, order='F')
      self.r = np.zeros((self.NScan*self.NC*self.NSlice,self.Nproj,self.N),dtype=DTYPE, order='C').T
      self.z1 = np.zeros((2,self.dimX,self.dimY,self.NSlice,self.unknowns), dtype=DTYPE, order='F')
      self.z2 = np.zeros( (4,self.dimX,self.dimY,self.NSlice,self.unknowns), dtype=DTYPE, order='F')
 
      iters = self.irgn_par.start_iters          
      for i in range(self.irgn_par.max_GN_it):
       start = time.time()       
       self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_3D(result))
 
       scale = np.linalg.norm(np.abs(self.grad_x_2D[0,...]))/np.linalg.norm(np.abs(self.grad_x_2D[1,...]))
         
       for j in range(len(self.model.constraints)-1):
         self.model.constraints[j+1].update(scale)
           
       result[1,...] = result[1,...]*self.model.T1_sc        
       self.model.T1_sc = self.model.T1_sc*(scale)
       result[1,...] = result[1,...]/self.model.T1_sc          
       self.step_val = self.model.execute_forward_3D(result)
       self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_3D(result).astype(DTYPE))
       self.grad_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grad_x_2D.data)          
#          self.conj_grad_x_2D = np.conj(self.grad_x_2D)
                     
                     
       result= self.irgn_solve_2D(result, iters, self.data)
       self.E1_scale.append(self.model.T1_sc)
       self.result[i,...] = result[:,...]
       
       iters = np.fmin(iters*2,self.irgn_par.max_iters)
       self.irgn_par.gamma = np.maximum(self.irgn_par.gamma*0.5,self.irgn_par.gamma_min)
       self.irgn_par.delta = np.minimum(self.irgn_par.delta*self.irgn_par.delta_inc, self.irgn_par.delta_max)
       
       end = time.time()-start
       print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
       print("-"*80)
#          if np.abs(self.fval_min-self.fval) < self.irgn_par.lambd*self.irgn_par.tol:
#            print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-       self.fval)/self.irgn_par.lambd))            
#            return
       self.fval_min = np.minimum(self.fval,self.fval_min)
                 
         
  def eval_fwd(self,y,x,wait_for=None):
 
    return self.prg.operator_fwd(y.queue, (self.dimY,self.dimX), None, 
                                 y.data, x.data, self.coil_buf, self.grad_buf,
                                 np.int32(self.NC), np.int32(self.NSlice), np.int32(self.NScan), np.int32(self.unknowns),
                                 wait_for=wait_for)       
      
  def operator_forward_2D(self, outp, inp ,wait_for=None):
    
#    return self.FT(np.sum(x[:,None,...]*self.grad_x_2D,axis=0)[:,None,...]*self.Coils)
       
#    tmp_img = clarray.to_device(self.queue,np.require(np.reshape(x.get(),(self.unknowns*self.NSlice,self.dimY,self.dimX)),DTYPE,"C"))

    tmp_result = clarray.zeros(self.queue,(self.dimY,self.dimX,self.NC*self.NScan*self.NSlice),DTYPE,"F")
    tmp_result.add_event(self.eval_fwd(tmp_result,inp,wait_for=wait_for))
    
#    tmp_sino = clarray.zeros(self.queue,self.r_struct[2],DTYPE,"F")#self.FT(np.reshape(tmp_result.get(),(self.NScan,self.NC,self.dimY,self.dimX)))#
    return (self.radon(outp,tmp_result,wait_for=tmp_result.events))

    
  def operator_adjoint_2D(self,outp,inp,wait_for=None):
#     msv.imshow(np.abs(x.get()))
#    return np.squeeze(np.sum(np.squeeze(np.sum(self.FTH(x)*self.conjCoils,axis=1))*self.conj_grad_x_2D,axis=1)) 
#    tmp_sino = clarray.to_device(self.queue,np.require(np.reshape(x.get(),(self.NScan*self.NC*self.NSlice,self.Nproj,self.N)),DTYPE,"C"))
#    tmp_sino = clarray.reshape(x,(self.NScan*self.NC,self.Nproj,self.N))
     tmp_img =  clarray.zeros(self.queue,self.r_struct[1],DTYPE,"F")#clarray.to_device(self.queue,self.FTH(x.get()))#
     tmp_img.add_event(self.radon_ad(tmp_img,inp,wait_for=wait_for))
#    tmp_result = clarray.zeros(self.queue,(self.unknowns,self.dimY,self.dimX),DTYPE,"C")
#    
     return (self.eval_adj(outp,tmp_img,wait_for=tmp_img.events))
#    result = clarray.reshape(tmp_result,(self.unknowns,self.dimY,self.dimX))
#    return result

  def eval_adj(self,x,y,wait_for=None):

    return self.prg.operator_ad(x.queue, (self.dimY,self.dimX), None, 
                                 x.data, y.data, self.coil_buf, self.grad_buf,np.int32(self.NC), 
                                 np.int32(self.NSlice),  np.int32(self.NScan), np.int32(self.unknowns),
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
    p.add_event(self.prg.update_p(p.queue, u.shape[:-1], None, p.data, u.data,
                w.data, np.float32(sigma), np.float32(1.0/alpha), np.int32(self.unknowns),
                wait_for=u.events + w.events + p.events + wait_for))

  def update_q(self,q, w, sigma, alpha, wait_for=[]):
    q.add_event(self.prg.update_q(q.queue, q.shape[1:-1], None, q.data, w.data,
                np.float32(sigma), np.float32(1.0/alpha),np.int32(self.unknowns),
                wait_for=q.events + w.events + wait_for))
    
  def f_grad(self,grad, u, wait_for=[]):
    grad.add_event(self.prg.gradient(grad.queue, grad.shape[1:-1], None, grad.data, u.data,
                np.int32(self.unknowns),
                wait_for=u.events + wait_for))

  def sym_grad(self,sym, w, wait_for=[]):
    sym.add_event(self.prg.sym_grad(sym.queue, sym.shape[1:-1], None, sym.data, w.data,
                np.int32(self.unknowns),
                wait_for=sym.events + w.events + wait_for))    

  def update_lambda(self,lamb, Ku, f, sigma, normest, wait_for=[]):
    lamb.add_event(self.prg.update_lambda(lamb.queue, lamb.shape, None,
                   lamb.data, Ku.data, f.data, np.float32(sigma/normest),
                   np.float32(1.0/(sigma/self.irgn_par.lambd+1.0)),
                   wait_for=lamb.events + Ku.events + f.events + wait_for))

  def update_u(self,u, u_, p, Kstarlambda, uk, tau, normest, delta, wait_for=[]):
    u.add_event(self.prg.update_u(u.queue, u.shape[:-1], None, u.data, u_.data,
                p.data, Kstarlambda.data, uk.data, np.float32(tau), np.float32(1.0/normest),
                np.float32(delta),np.int32(self.unknowns),
                wait_for=u.events + u_.events + p.events +
                Kstarlambda.events + wait_for))

  def update_w(self,w, w_, p, q, tau, wait_for=[]):
    w.add_event(self.prg.update_w(w.queue, w.shape[1:-1], None, w.data, w_.data,
                p.data, q.data, np.float32(tau),np.int32(self.unknowns),
                wait_for=w.events + w_.events + p.events + q.events +
                wait_for))


  def functional_value(self,accum_u, accum_f, u, w, Ku, f, alpha, r_struct):
    Ku.add_event(self.operator_forward_2D(Ku, u, wait_for=u.events))
    accum_f.add_event(self.prg.functional_discrepancy(Ku.queue, Ku.shape,
                      None, accum_f.data, Ku.data, f.data,
                      wait_for=Ku.events + f.events))
    accum_u.add_event(self.prg.functional_tgv(u.queue, u.shape[:-1], None,
                      accum_u.data, u.data, w.data, np.float32(alpha[0]),
                      np.float32(alpha[1]), np.int32(self.unknowns),wait_for=u.events + w.events))
    cl.wait_for_events(accum_f.events + accum_u.events)
    value = 0.5*abs(np.abs(clarray.sum(accum_f).get()) \
            + abs(clarray.sum(accum_u).get()))
    return value

#################
## main iteration

  def tgv_radon(self,f0, x0, img_shape, angles, alpha, maxiter, record_values=True):
     
    f = clarray.to_device(self.queue, np.require(f0, DTYPE, 'F'))
    u = clarray.to_device(self.queue, np.require(x0, DTYPE, 'F'))
    uk = clarray.to_device(self.queue, np.require(np.copy(x0), DTYPE, 'F'))
    u_ = clarray.to_device(self.queue, np.require(np.copy(x0), DTYPE, 'F'))
    w = clarray.to_device(self.queue, self.v)
    w_ = clarray.to_device(self.queue, self.v)
    p = clarray.to_device(self.queue, self.z1)
    q = clarray.to_device(self.queue, self.z2)
    lamb = clarray.to_device(self.queue, self.r)
    
#    ## Guess OP norm
#    yy = clarray.zeros_like(u)
#    xx = np.random.random_sample(u.shape).astype(DTYPE)
#    xxcl = clarray.to_device(self.queue,xx) 
#    tmp = clarray.zeros_like(f)
#    self.operator_forward_2D(tmp,xxcl)
#    self.operator_adjoint_2D(yy,tmp)
#    for j in range(10):
#       if not np.isclose(np.linalg.norm(yy.get().flatten()),0):
#           xxcl = yy/np.linalg.norm(yy.get().flatten())
#       else:
#           xxcl = yy
#       self.operator_forward_2D(tmp,xxcl)
#       self.operator_adjoint_2D(yy,tmp)
#       l1 = clarray.vdot(yy,xxcl).get();
#    L = np.max(np.abs(l1)) ## Lipschitz constant estimate   
    L = ((0.5*(18.0 + np.sqrt(33)))+(300))
    print('L: %f'%(L))
#    del yy, xx, xxcl, tmp, l1
    
    p_d_ratio = 1
     


    if record_values:
        accum_u = clarray.zeros_like(u)
        accum_f = clarray.zeros_like(f)
    
    Ku = clarray.zeros(self.queue, f0.shape, dtype=DTYPE, order='F')
    Kstarlambda = clarray.zeros_like(u)
    normest = 1#self.radon_normest()

#    L = 0.5*(18.0 + np.sqrt(33))
    sigma = 1.0/np.sqrt(L)*p_d_ratio
    tau = 1.0/np.sqrt(L)/p_d_ratio
    alpha = np.array(alpha)

    if record_values:
        values = []
    for i in range(np.minimum(10,maxiter)):
        self.update_p(p, u_, w_, sigma, alpha[1])
        self.update_q(q, w_, sigma, alpha[0], wait_for=w_.events)
        Ku.add_event(self.operator_forward_2D(Ku, u_, wait_for=u_.events))
        self.update_lambda(lamb, Ku, f, sigma, normest)
        Kstarlambda.add_event(self.operator_adjoint_2D(Kstarlambda, lamb, wait_for=lamb.events))
        self.update_u(u_, u, p, Kstarlambda, uk, tau, normest, self.irgn_par.delta)
        self.update_w(w_, w, p, q, tau)
        
        u_= u_.get()
        for j in range(len(self.model.constraints)):   
          u_[...,j] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,u_[...,j]))
          if self.model.constraints[j].real:
            u_[...,j] = np.real(u_[...,j])   
        u_ = clarray.to_device(self.queue,u_)   
           
        if record_values:
            values.append(self.functional_value(accum_u, accum_f, u, w, Ku, f,
                                           alpha, self.r_struct))
            sys.stdout.write('Iteration %d, functional value=%f        \r' \
                             % (i, values[-1]))
            sys.stdout.flush()
        
        u.add_event(self.update_extra(u_, u, wait_for=u.events + u_.events))
        w.add_event(self.update_extra(w_, w, wait_for=w.events + w_.events))

        (u, u_, w, w_) = (u_, u, w_, w)
        #Adapt stepsize
        (sigma,tau) = self.step_update(u-u_,w-w_,sigma,tau,p_d_ratio)        
        
    for i in np.arange(10,maxiter):
        self.update_p(p, u_, w_, sigma, alpha[1])
        self.update_q(q, w_, sigma, alpha[0], wait_for=w_.events)
        Ku.add_event(self.operator_forward_2D(Ku, u_, wait_for=u_.events))
        self.update_lambda(lamb, Ku, f, sigma, normest)
        Kstarlambda.add_event(self.operator_adjoint_2D(Kstarlambda, lamb, wait_for=lamb.events))
        self.update_u(u_, u, p, Kstarlambda, uk, tau, normest, self.irgn_par.delta)
        self.update_w(w_, w, p, q, tau)
        
        u_= u_.get()
        for j in range(len(self.model.constraints)):   
          u_[...,j] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,u_[...,j]))
          if self.model.constraints[j].real:
            u_[...,j] = np.real(u_[...,j])   
        u_ = clarray.to_device(self.queue,u_)   
           
        if record_values:
            values.append(self.functional_value(accum_u, accum_f, u, w, Ku, f,
                                           alpha, self.r_struct))
            sys.stdout.write('Iteration %d, functional value=%f        \r' \
                             % (i, values[-1]))
            sys.stdout.flush()
        
        u.add_event(self.update_extra(u_, u, wait_for=u.events + u_.events))
        w.add_event(self.update_extra(w_, w, wait_for=w.events + w_.events))


        (u, u_, w, w_) = (u_, u, w_, w)
        
        if (i % 50 == 0):
            self.model.plot_unknowns(u.get().T)
            #Adapt stepsize      
            (sigma,tau) = self.step_update(u-u_,w-w_,sigma,tau,p_d_ratio)
                        

    if record_values:
        sys.stdout.write('\n')
        self.z1 = p.get()
        self.z2 = q.get()
        self.r = lamb.get()
        self.v = w.get()
        return (np.squeeze(u.get().T), np.transpose(np.squeeze(w.get().T),(0,-1,1,2,3)), values)
    else:
        return u.get()


  def nFT_2D(self, x):
    result = np.zeros((self.NScan,self.NC,self.NSlice,self.Nproj,self.N),dtype=DTYPE)  
#    for i in range(self.NScan):
#       for j in range(self.NC):
    tmp_img = clarray.to_device(self.queue,np.require(np.reshape(x,(self.NC*self.NScan*self.NSlice,self.dimY,self.dimX)),DTYPE,"C"))
    tmp_sino = clarray.zeros(self.queue,self.r_struct[2],DTYPE,"F")
    (self.radon(tmp_sino,tmp_img.T))
    result = np.reshape(tmp_sino.get().T,(self.NScan,self.NC,self.NSlice,self.Nproj,self.N))

    return result



  def nFTH_2D(self, x):
    result = np.zeros((self.NScan,self.NC,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)    
#    for i in range(self.NScan):
#       for j in range(self.NC):    
    tmp_sino = clarray.to_device(self.queue,np.require(np.reshape(x,(self.NScan*self.NC*self.NSlice,self.Nproj,self.N)),DTYPE,"C"))
    tmp_img = clarray.zeros(self.queue,self.r_struct[1],DTYPE,"F")
    (self.radon_ad(tmp_img,tmp_sino.T))
    result = np.reshape(tmp_img.get().T,(self.NScan,self.NC,self.NSlice,self.dimY,self.dimX))
  
    return result
            
 
  def step_update(self,u,v,sig,tau,s_t_ratio=1):
     sig = sig
     tau = tau
     
     Ku = clarray.zeros(self.queue, (self.N,self.Nproj,self.NC*self.NSlice*self.NScan),dtype=DTYPE,order='F')
     grad = clarray.zeros(self.queue, (2,self.dimX,self.dimY,self.NSlice,self.unknowns), dtype=DTYPE, order='F')
     sym_grad = clarray.zeros(self.queue, (4,self.dimX,self.dimY,self.NSlice,self.unknowns), dtype=DTYPE, order='F')
     Ku.add_event(self.operator_forward_2D(Ku, u, wait_for=u.events))
     self.f_grad(grad,u)
     grad = grad-v
     self.sym_grad(sym_grad,v)
     
     
     #Get |Kx|  2 bei sym grad berücksichtigt!
     sym_grad = sym_grad.get()

     nKx = np.real((clarray.vdot(Ku,Ku).get()+clarray.vdot(grad,grad).get()+np.sum(np.vdot(sym_grad[:2],sym_grad[:2]))+2*np.sum(np.vdot(sym_grad[2:],sym_grad[2:])))**(1/2))
     
     #Get |x|
     nx = np.real(((clarray.vdot(u,u)+clarray.vdot(v,v))**(1/2)).get())
     
    
     #Set |x| / |Kx|
     tmp = (nx/nKx);
     theta = 0.90;

    #Check convergence condition
     while sig*tau > tmp**2: #If stepsize is too large
        if theta**(2)*sig*tau < tmp**2: #Check if minimal decrease satisfies condition
            sig = theta*sig
            tau = theta*tau
        else:                 #If not, decrease further
            sig = tmp*s_t_ratio
            tau = tmp/s_t_ratio
            
     return (sig,tau)       
