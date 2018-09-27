
# cython: infer_types=True
# cython: profile=False

from __future__ import division

import numpy as np
import time
import sys
import gridroutines_slicefirst as NUFFT

import matplotlib.pyplot as plt
plt.ion()

DTYPE = np.complex64
DTYPE_real = np.float32


#import pynfft.nfft as nfft

import pyopencl as cl
import pyopencl.array as clarray



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
  def __init__(self,par,ctx,queue,traj=None,dcf=None,trafo=1,ksp_encoding='2D'):

    self.par = par
    self.C = np.require(np.transpose(par.C,[1,0,2,3]),requirements='C')
    self.traj = traj
    self.unknowns_TGV = par.unknowns_TGV
    self.unknowns_H1 = par.unknowns_H1
    self.unknowns = par.unknowns
    self.NSlice = par.NSlice
    self.NScan = par.NScan
    self.dimX = par.dimX
    self.dimY = par.dimY
    self.NC = par.NC
    self.N = par.N
    self.Nproj = par.Nproj
    self.fval_min = 0
    self.fval = 0
    self.ctx = ctx
    self.queue = queue


    self.gn_res = []
    self.num_dev = len(ctx)
    self.NUFFT = []
    self.par_slices = 1
    self.ukscale = []
    self.prg = []
    self.overlap = 1
    self.alloc=[]
    self.ratio = []
    for j in range(self.num_dev):
      self.ratio.append(clarray.to_device(self.queue[3*j],(100*np.ones(self.unknowns)).astype(dtype=DTYPE_real)))
      self.ratio[j][1] = 1
      if trafo:
        self.alloc.append(MyAllocator(ctx[j]))
        self.NUFFT.append(NUFFT.gridding(ctx[j],self.queue[3*j:3*(j+1)-1],4,2,par.N,par.NScan, (par.NScan*par.NC*(self.par_slices+self.overlap),par.N,par.N),(1,2),traj.astype(DTYPE),np.require(np.abs(dcf),DTYPE_real,requirements='C'),par.N,1000,DTYPE,DTYPE_real))
        self.ukscale.append(clarray.to_device(self.queue[3*j],np.ones(self.unknowns,dtype=DTYPE_real)))
      else:
        self.alloc.append(MyAllocator(ctx[j]))
        self.NUFFT.append(NUFFT.gridding(ctx[j],self.queue[3*j:3*(j+1)-1],4,2,par.N,par.NScan, (par.NScan*par.NC*(self.par_slices+self.overlap),par.N,par.N),(1,2),traj,dcf,par.N,1000,DTYPE,DTYPE_real,radial=trafo))
        self.ukscale.append(clarray.to_device(self.queue[3*j],np.ones(self.unknowns,dtype=DTYPE_real)))
      self.prg.append(Program(self.ctx[j], r"""
__kernel void update_v(__global float8 *v,__global float8 *v_, __global float8 *Kyk2, const float tau) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  v[i] = v_[i]-tau*Kyk2[i];
}

__kernel void update_r(__global float2 *r, __global float2 *r_, __global float2 *A, __global float2 *A_, __global float2 *res,
                          const float sigma, const float theta, const float lambdainv) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  r[i] = (r_[i]+sigma*((1+theta)*A[i]-theta*A_[i] - res[i]))*lambdainv;
}
__kernel void update_z2(__global float16 *z_new, __global float16 *z, __global float16 *gx,__global float16 *gx_,
                          const float sigma, const float theta, const float alphainv,
                          const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = 0;

  float fac = 0.0f;

  for (int uk=0; uk<NUk; uk++)
  {
     i = k*Nx*Ny*NUk+uk*Nx*Ny+Nx*y + x;
     z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]);

     // reproject
     fac = hypot(fac,hypot(
     hypot(hypot(hypot(z_new[i].s0,z_new[i].s1), hypot(z_new[i].s2,z_new[i].s3)),hypot(z_new[i].s4,z_new[i].s5)),
     hypot(hypot(2.0f*hypot(z_new[i].s6,z_new[i].s7),2.0f*hypot(z_new[i].s8,z_new[i].s9)),2.0f*hypot(z_new[i].sa,z_new[i].sb)))*alphainv);
   }

  for (int uk=0; uk<NUk; uk++)
  {
    i = k*Nx*Ny*NUk+uk*Nx*Ny+Nx*y + x;
    if (fac > 1.0f) {z_new[i] /=fac;}
  }
}
__kernel void update_z1(__global float8 *z_new, __global float8 *z, __global float8 *gx,__global float8 *gx_,
                          __global float8 *vx,__global float8 *vx_, const float sigma, const float theta, const float alphainv,
                          const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = 0;

  float fac = 0.0f;

  for (int uk=0; uk<NUk; uk++)
  {
     i = k*Nx*Ny*NUk+uk*Nx*Ny+Nx*y + x;
     z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]-(1+theta)*vx[i]+theta*vx_[i]);

     // reproject
     fac = hypot(fac,hypot(hypot(z_new[i].s0,z_new[i].s1), hypot(hypot(z_new[i].s2,z_new[i].s3),hypot(z_new[i].s4,z_new[i].s5)))*alphainv);
  }
  for (int uk=0; uk<NUk; uk++)
  {
    i = k*Nx*Ny*NUk+uk*Nx*Ny+Nx*y + x;
    if (fac > 1.0f) {z_new[i] /=fac;}
  }
}
  __kernel void update_z1_tv(__global float8 *z_new, __global float8 *z, __global float8 *gx,__global float8 *gx_,
                          const float sigma, const float theta, const float alphainv,
                          const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = 0;

  float fac = 0.0f;

  for (int uk=0; uk<NUk; uk++)
  {
     i = k*Nx*Ny*NUk+uk*Nx*Ny+Nx*y + x;
     z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]);

     // reproject
     fac = hypot(fac,hypot(hypot(z_new[i].s0,z_new[i].s1), hypot(hypot(z_new[i].s2,z_new[i].s3),hypot(z_new[i].s4,z_new[i].s5)))*alphainv);
  }
  for (int uk=0; uk<NUk; uk++)
  {
    i = k*Nx*Ny*NUk+uk*Nx*Ny+Nx*y + x;
    if (fac > 1.0f) z_new[i] /=fac;
  }
}
__kernel void update_primal(__global float2 *u_new, __global float2 *u, __global float2 *Kyk,__global float2 *u_k, const float tau, const float tauinv, float div, __global float* min, __global float* max, __global int* real, const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = 0;

  for (int uk=0; uk<NUk; uk++)
  {
     i = k*Nx*Ny*NUk+uk*Ny*Ny+Nx*y + x;
     u_new[i] = (u[i]-tau*Kyk[i]+tauinv*u_k[i])*div;

     if(real[uk]>0)
     {
         u_new[i].s1 = 0;
         if (u_new[i].s0<min[uk])
         {
             u_new[i].s0 = min[uk];
         }
         if(u_new[i].s0>max[uk])
         {
             u_new[i].s0 = max[uk];
         }
     }
     else
     {
         if (u_new[i].s0<min[uk])
         {
             u_new[i].s0 = min[uk];
         }
         if(u_new[i].s0>max[uk])
         {
             u_new[i].s0 = max[uk];
         }
         if (u_new[i].s1<min[uk])
         {
             u_new[i].s1 = min[uk];
         }
         if(u_new[i].s1>max[uk])
         {
             u_new[i].s1 = max[uk];
         }
     }

  }
}

__kernel void gradient(__global float8 *grad, __global float2 *u, const int NUk, __global float* scale, const float maxscal, __global float* ratio) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = 0;


  for (int uk=0; uk<NUk; uk++)
  {
     i = k*Nx*Ny*NUk+uk*Nx*Ny+Nx*y + x;
     // gradient
     grad[i] = (float8)(-u[i],-u[i],-u[i],0.0f,0.0f);
     if (x < Nx-1)
     { grad[i].s01 += u[i+1].s01;}
     else
     { grad[i].s01 = 0.0f;}

     if (y < Ny-1)
     { grad[i].s23 += u[i+Nx].s01;}
     else
     { grad[i].s23 = 0.0f;}
     if (k < NSl-1)
     { grad[i].s45 += u[i+Nx*Ny*NUk].s01;}
     else
     { grad[i].s45 = 0.0f;}
     // scale gradients
     if (uk>0)
     {grad[i]*=maxscal/(ratio[uk]*scale[uk]);}
     else
     {grad[i]*=(maxscal/(ratio[uk]*scale[uk]));}
  }
}

__kernel void sym_grad(__global float16 *sym, __global float8 *w, const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = 0;


  for (int uk=0; uk<NUk; uk++)
  {
     i = k*Nx*Ny*NUk+uk*Nx*Ny+Nx*y + x;
      // symmetrized gradient backward differences
     float16 val_real = (float16)(w[i].s024, w[i].s024, w[i].s024,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
     float16 val_imag = (float16)(w[i].s135, w[i].s135, w[i].s135,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
     if (x > 0)
     { val_real.s012 -= w[i-1].s024;  val_imag.s012 -= w[i-1].s135;}
     else
     { val_real.s012 = (float3) 0.0f; val_imag.s012 = (float3) 0.0f; }

     if (y > 0)
     {val_real.s345 -= w[i-Nx].s024;  val_imag.s345 -= w[i-Nx].s135;}
     else
     {val_real.s345 = (float3)  0.0f; val_imag.s345 = (float3) 0.0f;  }

     if (k > 0)
     {val_real.s678 -= w[i-Nx*Ny].s024;  val_imag.s678 -= w[i-Nx*Ny].s135;}
     else
     {val_real.s678 = (float3) 0.0f; val_imag.s678 = (float3) 0.0f;  }

     sym[i] = (float16)(val_real.s0, val_imag.s0, val_real.s4,val_imag.s4,val_real.s8,val_imag.s8,
                        0.5f*(val_real.s1 + val_real.s3), 0.5f*(val_imag.s1 + val_imag.s3),
                        0.5f*(val_real.s2 + val_real.s6), 0.5f*(val_imag.s2 + val_imag.s6),
                        0.5f*(val_real.s5 + val_real.s7), 0.5f*(val_imag.s5 + val_imag.s7),
                        0.0f,0.0f,0.0f,0.0f);
   }
}
__kernel void divergence(__global float2 *div, __global float8 *p, const int NUk,
                         __global float* scale, const float maxscal, __global float* ratio) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = 0;

  for (int ukn=0; ukn<NUk; ukn++)
  {
     i = k*Nx*Ny*NUk+ukn*Nx*Ny+Nx*y + x;
     // divergence
     float8 val = p[i];
     if (x == Nx-1)
     {
         //real
         val.s0 = 0.0f;
         //imag
         val.s1 = 0.0f;
     }
     if (x > 0)
     {
         //real
         val.s0 -= p[i-1].s0;
         //imag
         val.s1 -= p[i-1].s1;
     }
     if (y == Ny-1)
     {
         //real
         val.s2 = 0.0f;
         //imag
         val.s3 = 0.0f;
     }
     if (y > 0)
     {
         //real
         val.s2 -= p[i-Nx].s2;
         //imag
         val.s3 -= p[i-Nx].s3;
     }
     if (k == NSl-1)
     {
         //real
         val.s4 = 0.0f;
         //imag
         val.s5 = 0.0f;
     }
     if (k > 0)
     {
         //real
         val.s4 -= p[i-Nx*Ny*NUk].s4;
         //imag
         val.s5 -= p[i-Nx*Ny*NUk].s5;
     }
     div[i] = val.s01+val.s23+val.s45;
     // scale gradients
     if (ukn>0)
     {div[i]*=maxscal/(ratio[ukn]*scale[ukn]);}
     else
     {div[i]*=(maxscal/(ratio[ukn]*scale[ukn]));}
  }

}
__kernel void sym_divergence(__global float8 *w, __global float16 *q,
                       const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = 0;

  for (int uk=0; uk<NUk; uk++)
  {
     i = k*Nx*Ny*NUk+uk*Nx*Ny+Nx*y + x;
     // divergence forward differences adjoint to backward gradient
     float16 val0 = -q[i];
     float16 val_real = (float16)(val0.s0, val0.s6, val0.s8,
                                  val0.s6, val0.s2, val0.sa,
                                  val0.s8, val0.sa, val0.s4,
                                  0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
     float16 val_imag = (float16)(val0.s1, val0.s7, val0.s9,
                                  val0.s7, val0.s3, val0.sb,
                                  val0.s9, val0.sb, val0.s5,
                                  0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
     if (x == 0)
     {
         //real
         val_real.s012 = 0.0f;
         //imag
         val_imag.s012 = 0.0f;
     }
     if (x < Nx-1)
     {
         //real
         val_real.s012 += (float3)(q[i+1].s0, q[i+1].s68);
         //imag
         val_imag.s012 += (float3)(q[i+1].s1, q[i+1].s79);
     }
     if (y == 0)
     {
         //real
         val_real.s345 = 0.0f;
         //imag
         val_imag.s345 = 0.0f;
     }
     if (y < Ny-1)
     {
         //real
         val_real.s345 += (float3)(q[i+Nx].s6, q[i+Nx].s2, q[i+Nx].sa);
         //imag
         val_imag.s345 += (float3)(q[i+Nx].s7, q[i+Nx].s3, q[i+Nx].sb);
     }
     if (k == 0)
     {
         //real
         val_real.s678 = 0.0f;
         //imag
         val_imag.s678 = 0.0f;
     }
     if (k < NSl-1)
     {
         //real
         val_real.s678 += (float3)(q[i+Nx*Ny].s8a, q[i+Nx*Ny].s4);
         //imag
         val_imag.s678 += (float3)(q[i+Nx*Ny].s9b, q[i+Nx*Ny].s5);
     }
     // linear step
     //real
     w[i].s024 = val_real.s012 + val_real.s345 + val_real.s678;
     //imag
     w[i].s135 = val_imag.s012 + val_imag.s345 + val_imag.s678;
  }
}
__kernel void update_Kyk2(__global float8 *w, __global float16 *q, __global float8 *z,
                       const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = 0;

  for (int uk=0; uk<NUk; uk++)
  {
     i = k*Nx*Ny*NUk+uk*Nx*Ny+Nx*y + x;
     // divergence
     float16 val0 = -q[i];
     float16 val_real = (float16)(val0.s0, val0.s6, val0.s8,
                                  val0.s6, val0.s2, val0.sa,
                                  val0.s8, val0.sa, val0.s4,
                                  0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
     float16 val_imag = (float16)(val0.s1, val0.s7, val0.s9,
                                  val0.s7, val0.s3, val0.sb,
                                  val0.s9, val0.sb, val0.s5,
                                  0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
     if (x == 0)
     {
         //real
         val_real.s012 = 0.0f;
         //imag
         val_imag.s012 = 0.0f;
     }
     if (x < Nx-1)
     {
         //real
         val_real.s012 += (float3)(q[i+1].s0, q[i+1].s68);
         //imag
         val_imag.s012 += (float3)(q[i+1].s1, q[i+1].s79);
     }
     if (y == 0)
     {
         //real
         val_real.s345 = 0.0f;
         //imag
         val_imag.s345 = 0.0f;
     }
     if (y < Ny-1)
     {
         //real
         val_real.s345 += (float3)(q[i+Nx].s6, q[i+Nx].s2, q[i+Nx].sa);
         //imag
         val_imag.s345 += (float3)(q[i+Nx].s7, q[i+Nx].s3, q[i+Nx].sb);
     }
     if (k == 0)
     {
         //real
         val_real.s678 = 0.0f;
         //imag
         val_imag.s678 = 0.0f;
     }
     if (k < NSl-1)
     {
         //real
         val_real.s678 += (float3)(q[i+Nx*Ny].s8a, q[i+Nx*Ny].s4);
         //imag
         val_imag.s678 += (float3)(q[i+Nx*Ny].s9b, q[i+Nx*Ny].s5);
     }
     // linear step
     //real
     w[i].s024 = -val_real.s012 - val_real.s345 - val_real.s678 -z[i].s024;
     //imag
     w[i].s135 = -val_imag.s012 - val_imag.s345 - val_imag.s678 -z[i].s135;
  }
}

__kernel void operator_fwd(__global float2 *out, __global float2 *in,
                       __global float2 *coils, __global float2 *grad, const int NCo,
                       const int NSl, const int NScan, const int Nuk)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);

  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);

  float2 tmp_in = 0.0f;
  float2 tmp_grad = 0.0f;
  float2 tmp_coil = 0.0f;
  float2 tmp_mul = 0.0f;


    for (int scan=0; scan<NScan; scan++)
    {
      for (int coil=0; coil < NCo; coil++)
      {
        tmp_coil = coils[k*NCo*X*Y + coil*X*Y + y*X + x];
        float2 sum = 0.0f;
        for (int uk=0; uk<Nuk; uk++)
        {
          tmp_grad = grad[k*Nuk*NScan*X*Y+uk*NScan*X*Y+scan*X*Y + y*X + x];
          tmp_in = in[k*Nuk*X*Y+uk*X*Y+y*X+x];

          tmp_mul = (float2)(tmp_in.x*tmp_grad.x-tmp_in.y*tmp_grad.y,tmp_in.x*tmp_grad.y+tmp_in.y*tmp_grad.x);
          sum += (float2)(tmp_mul.x*tmp_coil.x-tmp_mul.y*tmp_coil.y,
                                                    tmp_mul.x*tmp_coil.y+tmp_mul.y*tmp_coil.x);

        }
        out[k*NScan*NCo*X*Y+scan*NCo*X*Y+coil*X*Y + y*X + x] = sum;
      }
    }


}
__kernel void operator_ad(__global float2 *out, __global float2 *in,
                       __global float2 *coils, __global float2 *grad, const int NCo,
                       const int NSl, const int NScan, const int Nuk)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);


  float2 tmp_in = 0.0f;
  float2 tmp_mul = 0.0f;
  float2 conj_grad = 0.0f;
  float2 conj_coils = 0.0f;


  for (int uk=0; uk<Nuk; uk++)
  {
  float2 sum = (float2) 0.0f;
  for (int scan=0; scan<NScan; scan++)
  {
    conj_grad = (float2) (grad[k*Nuk*NScan*X*Y+uk*NScan*X*Y+scan*X*Y + y*X + x].x,
                          -grad[k*Nuk*NScan*X*Y+uk*NScan*X*Y+scan*X*Y + y*X + x].y);
  for (int coil=0; coil < NCo; coil++)
  {
    conj_coils = (float2) (coils[k*NCo*X*Y + coil*X*Y + y*X + x].x,
                                  -coils[k*NCo*X*Y + coil*X*Y + y*X + x].y);

    tmp_in = in[k*NScan*NCo*X*Y+scan*NCo*X*Y+coil*X*Y + y*X + x];
    tmp_mul = (float2)(tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x);


    sum += (float2)(tmp_mul.x*conj_coils.x-tmp_mul.y*conj_coils.y,
                                     tmp_mul.x*conj_coils.y+tmp_mul.y*conj_coils.x);
  }
  }
  out[k*Nuk*X*Y+uk*X*Y+y*X+x] = sum;
  }

}



__kernel void update_Kyk1(__global float2 *out, __global float2 *in,
                       __global float2 *coils, __global float2 *grad, __global float8 *p, const int NCo,
                       const int NSl, const int NScan, __global float* scale, const float maxscal,__global float* ratio, const int Nuk)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = 0;

  float2 tmp_in = 0.0f;
  float2 tmp_mul = 0.0f;
  float2 conj_grad = 0.0f;
  float2 conj_coils = 0.0f;



  for (int uk=0; uk<Nuk; uk++)
  {
  i = k*X*Y*Nuk+uk*X*Y+X*y + x;

  float2 sum = (float2) 0.0f;
  for (int scan=0; scan<NScan; scan++)
  {
    conj_grad = (float2) (grad[k*Nuk*NScan*X*Y+uk*NScan*X*Y+scan*X*Y + y*X + x].x,
                          -grad[k*Nuk*NScan*X*Y+uk*NScan*X*Y+scan*X*Y + y*X + x].y);
  for (int coil=0; coil < NCo; coil++)
  {
    conj_coils = (float2) (coils[k*NCo*X*Y + coil*X*Y + y*X + x].x,
                                  -coils[k*NCo*X*Y + coil*X*Y + y*X + x].y);

    tmp_in = in[k*NScan*NCo*X*Y+scan*NCo*X*Y+coil*X*Y + y*X + x];
    tmp_mul = (float2)(tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x);


    sum += (float2)(tmp_mul.x*conj_coils.x-tmp_mul.y*conj_coils.y,
                                     tmp_mul.x*conj_coils.y+tmp_mul.y*conj_coils.x);
  }
  }

   // divergence
   float8 val = p[i];
   if (x == X-1)
   {
       //real
       val.s0 = 0.0f;
       //imag
       val.s1 = 0.0f;
   }
   if (x > 0)
   {
       //real
       val.s0 -= p[i-1].s0;
       //imag
       val.s1 -= p[i-1].s1;
   }
   if (y == Y-1)
   {
       //real
       val.s2 = 0.0f;
       //imag
       val.s3 = 0.0f;
   }
   if (y > 0)
   {
       //real
       val.s2 -= p[i-X].s2;
       //imag
       val.s3 -= p[i-X].s3;
   }
   if (k == NSl-1)
   {
       //real
       val.s4 = 0.0f;
       //imag
       val.s5 = 0.0f;
   }
   if (k > 0)
   {
       //real
       val.s4 -= p[i-X*Y*Nuk].s4;
       //imag
       val.s5 -= p[i-X*Y*Nuk].s5;
   }

   // scale gradients
   if (uk>0)
   {val*=maxscal/(ratio[uk]*scale[uk]);}
   else
   {val*=(maxscal/(ratio[uk]*scale[uk]));}

  out[k*Nuk*X*Y+uk*X*Y+y*X+x] = sum - (val.s01+val.s23+val.s45);

  }

}
/*__kernel void update_primal_explicit(__global float2 *u_new, __global float2 *u, __global float2 *Kyk, __global float2 *u_k,
__global float2* ATd, const float tau, const float delta_inv, const float lambd, __global float* mmin, __global float* mmax, __global int* real, const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);

  size_t i = k*Nx*Ny+Nx*y + x;


  for (int uk=0; uk<NUk; uk++)
  {
     u_new[i] = u[i]-tau*(lambd*u_new[i]-lambd*ATd[i]+delta_inv*u[i]-delta_inv*u_k[i]-Kyk[i]);

     if(real[uk]>0)
     {
         u_new[i].s1 = 0;
         if (u_new[i].s0<mmin[uk])
         {
             u_new[i].s0 = mmin[uk];
         }
         if(u_new[i].s0>mmax[uk])
         {
             u_new[i].s0 = mmax[uk];
         }
     }
     else
     {
         if (u_new[i].s0<mmin[uk])
         {
             u_new[i].s0 = mmin[uk];
         }
         if(u_new[i].s0>mmax[uk])
         {
             u_new[i].s0 = mmax[uk];
         }
         if (u_new[i].s1<mmin[uk])
         {
             u_new[i].s1 = mmin[uk];
         }
         if(u_new[i].s1>mmax[uk])
         {
             u_new[i].s1 = mmax[uk];
         }
     }
     i += NSl*Nx*Ny;
  }
}
*/
"""))
    self.tmp_img=[]
    for j in range(len(ctx)):
     self.tmp_img.append(clarray.zeros(self.queue[3*j],(self.par_slices+self.overlap,self.NScan,self.NC,self.dimY,self.dimX),DTYPE,"C"))
    self.tmp_FT = np.zeros((self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE)
    self.tmp_FTH = np.zeros((self.par_slices+self.overlap,self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE)
    self.tmp_adj = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE)
    self.tmp_out = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE)



  def operator_forward_full(self, out, x, idx=0,idxq=0,wait_for=[]):
    self.tmp_img[idx].add_event(self.eval_fwd_streamed(self.tmp_img[idx],x,idx,idxq,wait_for=self.tmp_img[idx].events+x.events))
    return  self.NUFFT[idx].fwd_NUFFT(out,self.tmp_img[idx],idxq,wait_for=out.events+wait_for+self.tmp_img[idx].events)

  def operator_adjoint_full(self, out, x,z,idx=0,idxq=0, wait_for=[]):
    (self.NUFFT[idx].adj_NUFFT(self.tmp_img[idx],x,idxq,wait_for=wait_for+x.events+self.tmp_img[idx].events)).wait()
#    print(out.shape)
    return self.prg[idx].update_Kyk1(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None,
                                 out.data, self.tmp_img[idx].data, self.coil_buf_part[idx+idxq*self.num_dev].data, self.grad_buf_part[idx+idxq*self.num_dev].data, z.data, np.int32(self.NC),
                                 np.int32(self.par_slices+self.overlap),  np.int32(self.NScan), self.ukscale[idx].data,
                                 np.float32(np.amax(self.ukscale[idx].get())),self.ratio[idx].data, np.int32(self.unknowns),
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
    return self.prg[idx].gradient(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None, grad.data, u.data,
                np.int32(self.unknowns),
                self.ukscale[idx].data,  np.float32(np.amax(self.ukscale[idx].get())),self.ratio[idx].data,
                wait_for=grad.events + u.events + wait_for)

  def bdiv(self,div, u, idx=0,idxq=0, wait_for=[]):
    return self.prg[idx].divergence(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None, div.data, u.data,
                np.int32(self.unknowns),
                self.ukscale[idx].data, np.float32(np.amax(self.ukscale[idx].get())),self.ratio[idx].data,
                wait_for=div.events + u.events + wait_for)

  def sym_grad(self,sym, w,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].sym_grad(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None, sym.data, w.data,
                np.int32(self.unknowns),
                wait_for=sym.events + w.events + wait_for)

  def sym_bdiv(self,div, u, idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].sym_divergence(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None, div.data, u.data,
                np.int32(self.unknowns),
                wait_for=div.events + u.events + wait_for)
  def update_Kyk2(self,div, u, z,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_Kyk2(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None, div.data, u.data, z.data,
                np.int32(self.unknowns),
                wait_for=div.events + u.events + z.events+wait_for)

  def update_primal(self, x_new, x, Kyk, xk, tau, delta,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_primal(self.queue[3*idx+idxq],(self.par_slices+self.overlap,self.dimY,self.dimX), None, x_new.data, x.data, Kyk.data, xk.data, np.float32(tau),
                                  np.float32(tau/delta), np.float32(1/(1+tau/delta)), self.min_const[idx].data, self.max_const[idx].data,
                                  self.real_const[idx].data, np.int32(self.unknowns),
                                  wait_for=x_new.events + x.events + Kyk.events+ xk.events+wait_for
                                  )
  def update_z1(self, z_new, z, gx, gx_, vx, vx_, sigma, theta, alpha,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_z1(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None, z_new.data, z.data, gx.data, gx_.data, vx.data, vx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/alpha), np.int32(self.unknowns),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ vx.events+ vx_.events+wait_for
                                  )
  def update_z1_tv(self, z_new, z, gx, gx_, sigma, theta, alpha,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_z1_tv(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None, z_new.data, z.data, gx.data, gx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/alpha), np.int32(self.unknowns),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ wait_for
                                  )
  def update_z2(self, z_new, z, gx, gx_, sigma, theta, beta,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_z2(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None, z_new.data, z.data, gx.data, gx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/beta),  np.int32(self.unknowns),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ wait_for
                                  )
  def update_r(self, r_new, r, A, A_, res, sigma, theta, lambd,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_r(self.queue[3*idx+idxq], (self.NScan*self.NC*(self.par_slices+self.overlap),self.Nproj,self.N), None, r_new.data, r.data, A.data, A_.data, res.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/(1+sigma/lambd)),
                                  wait_for= r_new.events + r.events + A.events+ A_.events+ wait_for
                                  )
  def update_v(self, v_new, v, Kyk2, tau,  idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_v(self.queue[3*idx+idxq], (self.unknowns*(self.par_slices+self.overlap),self.dimY,self.dimX), None,
                             v_new.data, v.data, Kyk2.data, np.float32(tau),
                                  wait_for= v_new.events + v.events + Kyk2.events+ wait_for
                                  )
  def update_primal_explicit(self, x_new, x, Kyk, xk, ATd, tau, delta, lambd, idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].update_primal_explicit(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None, x_new.data, x.data, Kyk.data, xk.data, ATd.data, np.float32(tau),
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
        print('scale %f at uk %i' %(self.ukscale[i][j].get(),j))

#  def scale_fwd(self,x):
#    y = np.copy(x)
#    for i in range(self.num_dev):
#      for j in range(self.unknowns):
#        y[:,j,...] /= self.ukscale[i][j].get()
#        if j==0:
#          y[:,j,...] *= np.max(self.ukscale[i].get())/self.ratio
#        else:
#          y[:,j,...] *= np.max(self.ukscale[i].get())
#    return y
#  def scale_adj(self,x):
#    y = np.copy(x)
#    for i in range(self.num_dev):
#      for j in range(self.unknowns):
#        y[:,j,...] /= self.ukscale[i][j].get()
#        if j==0:
#          y[:,j,...] *= np.max(self.ukscale[i].get())/self.ratio
#        else:
#          y[:,j,...] *= np.max(self.ukscale[i].get())
#    return y


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
   self.z1 = np.zeros(([self.NSlice,self.unknowns,self.par.dimX,self.par.dimY,4]),dtype=DTYPE)


   self.result = np.zeros((self.irgn_par["max_GN_it"]+1,self.unknowns,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
   self.result[0,:,:,:,:] = np.copy(self.model.guess)

   result = np.copy(self.model.guess)

   self.v = np.zeros(([self.NSlice,self.unknowns,self.par.dimX,self.par.dimY,4]),dtype=DTYPE)
   self.z2 = np.zeros(([self.NSlice,self.unknowns,self.par.dimX,self.par.dimY,8]),dtype=DTYPE)
   for i in range(self.irgn_par["max_GN_it"]):
    start = time.time()
    self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))

    for uk in range(self.unknowns-1):
      scale = np.linalg.norm(np.abs(self.grad_x[0,...]))/np.linalg.norm(np.abs(self.grad_x[uk+1,...]))
      self.model.constraints[uk+1].update(scale)
      result[uk+1,...] = result[uk+1,...]*self.model.uk_scale[uk+1]
      self.model.uk_scale[uk+1] = self.model.uk_scale[uk+1]*scale
      result[uk+1,...] = result[uk+1,...]/self.model.uk_scale[uk+1]

    self.step_val = np.nan_to_num(self.model.execute_forward_3D(result))
    self.step_val = np.require(np.transpose(self.step_val,[1,0,2,3]),requirements='C')
    self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))
    self.grad_x = np.require(np.transpose(self.grad_x,[2,0,1,3,4]),requirements='C')

    result = self.irgn_solve_3D(result, iters, self.data,TV)
    for uk in range(self.unknowns):
      self.result[i+1,uk,...] = result[uk,...]*self.model.uk_scale[uk]

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
    self.FT(b,np.require(np.transpose(self.model.execute_forward_3D(x),[1,0,2,3]),requirements='C')[:,:,None,...]*self.C[:,None,...])
    self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - b)**2
            +self.irgn_par["gamma"]*np.sum(np.abs(grad.get()-self.v))
            +self.irgn_par["gamma"]*(2)*np.sum(np.abs(sym_grad.get()))
            +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2)
    for knt in range(self.unknowns):
      print('Norm M0 grad: %f  norm T1 grad: %f' %(np.linalg.norm(grad.get()[0,...]),np.linalg.norm(grad.get()[knt,...])))
      scale = np.linalg.norm(grad.get()[knt,...])/np.linalg.norm(grad.get()[1,...])
      if scale == 0 or not np.isfinite(scale):
        pass
      else:
        print("Scale: %f" %scale)
        for j in range(self.num_dev):
          self.ratio[j][knt] *= scale


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

    self.set_scale(x)
    xk = x.copy()
    x_new = np.zeros_like(x)

    r = self.r#np.zeros_like(res,dtype=DTYPE)
    r_new = np.zeros_like(r)
    z1 = self.z1#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z1_new =  np.zeros_like(z1)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2 = self.z2#np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2_new =  np.zeros_like(z2)
    v = self.v#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    v_new =  np.zeros_like(v)
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
    Axold_tmp = np.zeros((self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE)
    Kyk1_part = []
    Kyk1_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE)
    Kyk2_part = []
    Kyk2_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE)
    for i in range(2*self.num_dev):
      Axold_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
      Kyk1_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
      Kyk2_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))


##### Warmup
    x_part = []
    r_part = []
    z1_part = []
    self.coil_buf_part = []
    self.grad_buf_part = []
    j=0
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
      Kyk1_part[i].add_event(self.operator_adjoint_full(Kyk1_part[i],r_part[i],z1_part[i],i,0))

    for i in range(self.num_dev):
      idx_start = (i+1+self.num_dev-1)*self.par_slices-self.overlap
      idx_stop = (i+2+self.num_dev-1)*self.par_slices
      x_part.append(clarray.to_device(self.queue[3*i+1], x[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
      r_part.append(clarray.to_device(self.queue[3*i+1], r[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
      z1_part.append(clarray.to_device(self.queue[3*i+1], z1[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
      self.coil_buf_part.append(clarray.to_device(self.queue[3*i+1], self.C[idx_start:idx_stop,...],allocator=self.alloc[i]))
      self.grad_buf_part.append(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start:idx_stop,...],allocator=self.alloc[i]))
    for i in range(self.num_dev):
      Axold_part[i+self.num_dev].add_event(self.operator_forward_full(Axold_part[i+self.num_dev],x_part[self.num_dev+i],i,1))
      Kyk1_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_part[i+self.num_dev],r_part[self.num_dev+i],z1_part[self.num_dev+i],i,1))
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
            Kyk1[idx_start:idx_stop,...] = Kyk1_tmp[:(self.par_slices),...]
            Axold[idx_start:idx_stop,...] = Axold_tmp[:(self.par_slices),...]
          else:
            Kyk1[idx_start:idx_stop,...] = Kyk1_tmp[self.overlap:,...]
            Axold[idx_start:idx_stop,...] = Axold_tmp[self.overlap:,...]
          ### Put Data
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)-self.overlap
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices
          x_part[i]=(clarray.to_device(self.queue[3*i], x[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
          r_part[i]=(clarray.to_device(self.queue[3*i], r[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
          z1_part[i]=(clarray.to_device(self.queue[3*i], z1[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
          self.coil_buf_part[i]=(clarray.to_device(self.queue[3*i], self.C[idx_start:idx_stop,...],allocator=self.alloc[i]))
          self.grad_buf_part[i]=(clarray.to_device(self.queue[3*i], self.grad_x[idx_start:idx_stop,...],allocator=self.alloc[i]))
        for i in range(self.num_dev):
          Axold_part[i].add_event(self.operator_forward_full(Axold_part[i],x_part[i],i,0))
          Kyk1_part[i].add_event(self.operator_adjoint_full(Kyk1_part[i],r_part[i],z1_part[i],i,0))
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
          x_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], x[idx_start:idx_stop,...] ,allocator=self.alloc[i]))# ))
          r_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], r[idx_start:idx_stop,...] ,allocator=self.alloc[i]))# ))
          z1_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], z1[idx_start:idx_stop,...] ,allocator=self.alloc[i]))# ))
          self.coil_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.C[idx_start:idx_stop,...] ,allocator=self.alloc[i]))
          self.grad_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start:idx_stop,...] ,allocator=self.alloc[i]))
        for i in range(self.num_dev):
          Axold_part[i+self.num_dev].add_event(self.operator_forward_full(Axold_part[i+self.num_dev],x_part[self.num_dev+i],i,1))
          Kyk1_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_part[i+self.num_dev],r_part[self.num_dev+i],z1_part[self.num_dev+i],i,1))
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
        Axold[idx_start:idx_stop,...] = Axold_tmp[:(self.par_slices),...]
        Kyk1[idx_start:idx_stop,...] = Kyk1_tmp[:(self.par_slices),...]
      else:
        Axold[idx_start:idx_stop,...] = Axold_tmp[self.overlap:,...]
        Kyk1[idx_start:idx_stop,...] = Kyk1_tmp[self.overlap:,...]
      self.queue[3*i+1].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      Axold_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=Axold_tmp)
      Kyk1_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=Kyk1_tmp)
      self.queue[3*i+2].finish()
      Axold[idx_start:idx_stop,...] = Axold_tmp[self.overlap:,...]
      Kyk1[idx_start:idx_stop,...] = Kyk1_tmp[self.overlap:,...]


##### Warmup
    j=0
    z2_part = []
    for i in range(self.num_dev):
      idx_start = i*self.par_slices
      idx_stop = (i+1)*self.par_slices+self.overlap
      z1_part[i].set(z1[idx_start:idx_stop,...],self.queue[3*i])
      z2_part.append(clarray.to_device(self.queue[3*i], z2[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
    for i in range(self.num_dev):
      Kyk2_part[i].add_event((self.update_Kyk2(Kyk2_part[i],z2_part[i],z1_part[i],i,0)))

    for i in range(self.num_dev):
      idx_start = (i+1+self.num_dev-1)*self.par_slices
      idx_stop = (i+2+self.num_dev-1)*self.par_slices
      if idx_stop == self.NSlice:
        idx_start -=self.overlap
      else:
        idx_stop +=self.overlap
      z1_part[i+self.num_dev].set(z1[idx_start:idx_stop,...],self.queue[3*i+1])# ))
      z2_part.append(clarray.to_device(self.queue[3*i+1], z2[idx_start:idx_stop,...],allocator=self.alloc[i]))# ))
    for i in range(self.num_dev):
      Kyk2_part[i+self.num_dev].add_event(self.update_Kyk2(Kyk2_part[i+self.num_dev],z2_part[self.num_dev+i],z1_part[self.num_dev+i],i,1))
#### Stream
    for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
        for i in range(self.num_dev):
          ### Get Data
          self.queue[3*i].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
          Kyk2_part[i].get(queue=self.queue[3*i+2],ary=Kyk2_tmp)
          self.queue[3*i+2].finish()
          Kyk2[idx_start:idx_stop,...] = Kyk2_tmp[:self.par_slices,...]
          ### Put Data
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices+self.overlap
          z1_part[i].set(z1[idx_start:idx_stop,...],self.queue[3*i])# ))
          z2_part[i].set(z2[idx_start:idx_stop,...],self.queue[3*i])# ))
        for i in range(self.num_dev):
          Kyk2_part[i].add_event(self.update_Kyk2(Kyk2_part[i],z2_part[i],z1_part[i],i,0))
        for i in range(self.num_dev):
          self.queue[3*i+1].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          Kyk2_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=Kyk2_tmp)
          self.queue[3*i+2].finish()
          Kyk2[idx_start:idx_stop,...] = Kyk2_tmp[:self.par_slices,...]
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
          if idx_stop == self.NSlice:
            idx_start -=self.overlap
          else:
            idx_stop +=self.overlap
          z1_part[self.num_dev+i].set(z1[idx_start:idx_stop,...],self.queue[3*i+1])# ))
          z2_part[self.num_dev+i].set(z2[idx_start:idx_stop,...],self.queue[3*i+1])# ))
        for i in range(self.num_dev):
          Kyk2_part[i+self.num_dev].add_event(self.update_Kyk2(Kyk2_part[i+self.num_dev],z2_part[i],z1_part[self.num_dev+i],i,1))
#### Collect last block
    if j<2*self.num_dev:
      j = 2*self.num_dev
    else:
      j+=1
    for i in range(self.num_dev):
      self.queue[3*i].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
      Kyk2_part[i].get(queue=self.queue[3*i+2],ary=Kyk2_tmp)
      self.queue[3*i+2].finish()
      Kyk2[idx_start:idx_stop,...] = Kyk2_tmp[:self.par_slices,...]
      self.queue[3*i+1].finish()
      idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
      Kyk2_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=Kyk2_tmp)
      self.queue[3*i+2].finish()
      if idx_stop == self.NSlice:
        Kyk2[idx_start:idx_stop,...] = Kyk2_tmp[self.overlap:,...]
      else:
        Kyk2[idx_start:idx_stop,...] = Kyk2_tmp[self.par_slices:,...]


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
    for i in range(self.num_dev):
      self.queue[3*i].finish()
      self.queue[3*i+1].finish()
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
        Ax_tmp = np.zeros((self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE)
        x_new_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE)
        v_new_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE)
        gradx_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE)
        gradx_xold_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE)
        symgrad_v_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,8),dtype=DTYPE)
        symgrad_v_vold_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,8),dtype=DTYPE)
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
            x_new[idx_start:idx_stop,...]=x_new_tmp[:(self.par_slices),...]
            gradx[idx_start:idx_stop,...] = gradx_tmp[:(self.par_slices),...]
            gradx_xold[idx_start:idx_stop,...] = gradx_xold_tmp[:(self.par_slices),...]
            Ax[idx_start:idx_stop,...]=Ax_tmp[:(self.par_slices),...]
            ### Put Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices+self.overlap
            x_part[i].set(x[idx_start:idx_stop,...],self.queue[3*i],)
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
            x_new[idx_start:idx_stop,...]=x_new_tmp[:(self.par_slices),...]
            gradx[idx_start:idx_stop,...] = gradx_tmp[:(self.par_slices),...]
            gradx_xold[idx_start:idx_stop,...] = gradx_xold_tmp[:(self.par_slices),...]
            Ax[idx_start:idx_stop,...]=Ax_tmp[:(self.par_slices),...]
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
        x_new[idx_start:idx_stop,...]=x_new_tmp[:(self.par_slices),...]
        gradx[idx_start:idx_stop,...] = gradx_tmp[:(self.par_slices),...]
        gradx_xold[idx_start:idx_stop,...] = gradx_xold_tmp[:(self.par_slices),...]
        Ax[idx_start:idx_stop,...]=Ax_tmp[:(self.par_slices),...]
        self.queue[3*i+1].finish()
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        x_new_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=x_new_tmp)
        gradx_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=gradx_tmp)
        gradx_xold_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=gradx_xold_tmp)
        Ax_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=Ax_tmp)
        self.queue[3*i+2].finish()
        if idx_stop == self.NSlice:
          x_new[idx_start:idx_stop,...]=x_new_tmp[self.overlap:,...]
          gradx[idx_start:idx_stop,...] = gradx_tmp[self.overlap:,...]
          gradx_xold[idx_start:idx_stop,...] = gradx_xold_tmp[self.overlap:,...]
          Ax[idx_start:idx_stop,...]=Ax_tmp[self.overlap:,...]
        else:
          x_new[idx_start:idx_stop,...]=x_new_tmp[:(self.par_slices),...]
          gradx[idx_start:idx_stop,...] = gradx_tmp[:(self.par_slices),...]
          gradx_xold[idx_start:idx_stop,...] = gradx_xold_tmp[:(self.par_slices),...]
          Ax[idx_start:idx_stop,...]=Ax_tmp[:(self.par_slices),...]

      j=0
      for i in range(self.num_dev):
        idx_start = i*self.par_slices
        idx_stop = (i+1)*self.par_slices
        if idx_start==0:
          idx_stop +=self.overlap
        else:
          idx_start-=self.overlap
        v_part[i].set(v[idx_start:idx_stop,...], self.queue[3*i])
        Kyk2_part[i].set(Kyk2[idx_start:idx_stop,...],self.queue[3*i])
      for i in range(self.num_dev):
        v_new_part[i].add_event(self.update_v(v_new_part[i],v_part[i],Kyk2_part[i],tau,i,0))
        symgrad_v_part[i].add_event(self.sym_grad(symgrad_v_part[i],v_new_part[i],i,0))
        symgrad_v_vold_part[i].add_event(self.sym_grad(symgrad_v_vold_part[i],v_part[i],i,0))
      for i in range(self.num_dev):
        idx_start = (i+1+self.num_dev-1)*self.par_slices-self.overlap
        idx_stop = (i+2+self.num_dev-1)*self.par_slices
        v_part[i+self.num_dev].set(v[idx_start:idx_stop,...], self.queue[3*i+1])
        Kyk2_part[i+self.num_dev].set(Kyk2[idx_start:idx_stop,...],self.queue[3*i+1])
      for i in range(self.num_dev):
        v_new_part[i+self.num_dev].add_event(self.update_v(v_new_part[i+self.num_dev],v_part[self.num_dev+i],Kyk2_part[self.num_dev+i],tau,i,1))
        symgrad_v_part[i+self.num_dev].add_event(self.sym_grad(symgrad_v_part[i+self.num_dev],v_new_part[i+self.num_dev],i,1))
        symgrad_v_vold_part[i+self.num_dev].add_event(self.sym_grad(symgrad_v_vold_part[i+self.num_dev],v_part[self.num_dev+i],i,1))
  #### Stream
      for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
#          tic = time.time()
          for i in range(self.num_dev):
            ### Get Data
            self.queue[3*i].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
            v_new_part[i].get(queue=self.queue[3*i+2],ary=v_new_tmp)
            symgrad_v_part[i].get(queue=self.queue[3*i+2],ary=symgrad_v_tmp)
            symgrad_v_vold_part[i].get(queue=self.queue[3*i+2],ary=symgrad_v_vold_tmp)
            self.queue[3*i+2].finish()
            if idx_start == 0:
              v_new[idx_start:idx_stop,...]=v_new_tmp[:(self.par_slices),...]
              symgrad_v[idx_start:idx_stop,...] = symgrad_v_tmp[:(self.par_slices),...]
              symgrad_v_vold[idx_start:idx_stop,...] = symgrad_v_vold_tmp[:(self.par_slices),...]
            else:
              v_new[idx_start:idx_stop,...]=v_new_tmp[self.overlap:,...]
              symgrad_v[idx_start:idx_stop,...] = symgrad_v_tmp[self.overlap:,...]
              symgrad_v_vold[idx_start:idx_stop,...] = symgrad_v_vold_tmp[self.overlap:,...]
            ### Put Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)-self.overlap
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices
            v_part[i].set(v[idx_start:idx_stop,...], self.queue[3*i])
            Kyk2_part[i].set(Kyk2[idx_start:idx_stop,...],self.queue[3*i])
          for i in range(self.num_dev):
            v_new_part[i].add_event(self.update_v(v_new_part[i],v_part[i],Kyk2_part[i],tau,i,0))
            symgrad_v_part[i].add_event(self.sym_grad(symgrad_v_part[i],v_new_part[i],i,0))
            symgrad_v_vold_part[i].add_event(self.sym_grad(symgrad_v_vold_part[i],v_part[i],i,0))
          for i in range(self.num_dev):
            self.queue[3*i+1].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            v_new_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=v_new_tmp)
            symgrad_v_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=symgrad_v_tmp)
            symgrad_v_vold_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=symgrad_v_vold_tmp)
            self.queue[3*i+2].finish()
            v_new[idx_start:idx_stop,...]=v_new_tmp[self.overlap:,...]
            symgrad_v[idx_start:idx_stop,...] = symgrad_v_tmp[self.overlap:,...]
            symgrad_v_vold[idx_start:idx_stop,...] = symgrad_v_vold_tmp[self.overlap:,...]
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices-self.overlap
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            v_part[i+self.num_dev].set(v[idx_start:idx_stop,...], self.queue[3*i+1])
            Kyk2_part[i+self.num_dev].set(Kyk2[idx_start:idx_stop,...],self.queue[3*i+1])
          for i in range(self.num_dev):
            v_new_part[i+self.num_dev].add_event(self.update_v(v_new_part[i+self.num_dev],v_part[self.num_dev+i],Kyk2_part[self.num_dev+i],tau,i,1))
            symgrad_v_part[i+self.num_dev].add_event(self.sym_grad(symgrad_v_part[i+self.num_dev],v_new_part[i+self.num_dev],i,1))
            symgrad_v_vold_part[i+self.num_dev].add_event(self.sym_grad(symgrad_v_vold_part[i+self.num_dev],v_part[self.num_dev+i],i,1))
  #### Collect last block
      if j<2*self.num_dev:
        j = 2*self.num_dev
      else:
        j+=1
      for i in range(self.num_dev):
        self.queue[3*i].finish()
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
        v_new_part[i].get(queue=self.queue[3*i+2],ary=v_new_tmp)
        symgrad_v_part[i].get(queue=self.queue[3*i+2],ary=symgrad_v_tmp)
        symgrad_v_vold_part[i].get(queue=self.queue[3*i+2],ary=symgrad_v_vold_tmp)
        self.queue[3*i+2].finish()
        if idx_start == 0:
          v_new[idx_start:idx_stop,...]=v_new_tmp[:(self.par_slices),...]
          symgrad_v[idx_start:idx_stop,...] = symgrad_v_tmp[:(self.par_slices),...]
          symgrad_v_vold[idx_start:idx_stop,...] = symgrad_v_vold_tmp[:(self.par_slices),...]
        else:
          v_new[idx_start:idx_stop,...]=v_new_tmp[self.overlap:,...]
          symgrad_v[idx_start:idx_stop,...] = symgrad_v_tmp[self.overlap:,...]
          symgrad_v_vold[idx_start:idx_stop,...] = symgrad_v_vold_tmp[self.overlap:,...]
        self.queue[3*i+1].finish()
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        v_new_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=v_new_tmp)
        symgrad_v_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=symgrad_v_tmp)
        symgrad_v_vold_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=symgrad_v_vold_tmp)
        self.queue[3*i+2].finish()
        v_new[idx_start:idx_stop,...]=v_new_tmp[self.overlap:,...]
        symgrad_v[idx_start:idx_stop,...] = symgrad_v_tmp[self.overlap:,...]
        symgrad_v_vold[idx_start:idx_stop,...] = symgrad_v_vold_tmp[self.overlap:,...]




      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new

      while True:
        theta_line = tau_new/tau
        #### Allocate temporary Arrays
        ynorm = 0
        lhs = 0
        j=0
        if myit == 0:
          z1_new_part = []
          z1_new_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE)
          z2_new_part = []
          z2_new_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,8),dtype=DTYPE)
          r_new_part = []
          r_new_tmp = np.zeros((self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE)
          Kyk1_new_part = []
          Kyk1_new_tmp = np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE)
          Kyk2_new_part = []
          Kyk2_new_tmp =np.zeros((self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE)
          for i in range(2*self.num_dev):
            z1_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
            z2_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,8),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
            r_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.NScan,self.NC,self.Nproj,self.N),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
            Kyk1_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))
            Kyk2_new_part.append(clarray.zeros(self.queue[3*int(np.mod(i,self.num_dev))],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX,4),dtype=DTYPE,allocator=self.alloc[int(np.mod(i,self.num_dev))]))

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
          v_new_part[i].set(v_new[idx_start:idx_stop,...] ,self.queue[3*i])
          v_part[i].set( v[idx_start:idx_stop,...] ,self.queue[3*i])
          r_part[i].set( r[idx_start:idx_stop,...] ,self.queue[3*i])
          Ax_part[i].set( Ax[idx_start:idx_stop,...] ,self.queue[3*i])
          Ax_old_part[i].set(  Axold[idx_start:idx_stop,...] ,self.queue[3*i])
          res_part[i].set(res[idx_start:idx_stop,...] ,self.queue[3*i])
          Kyk1_part[i].set( Kyk1[idx_start:idx_stop,...] ,self.queue[3*i])
          self.coil_buf_part[i].set(self.C[idx_start:idx_stop,...] ,self.queue[3*i])
          self.grad_buf_part[i].set(self.grad_x[idx_start:idx_stop,...],self.queue[3*i])
        for i in range(self.num_dev):
          z1_new_part[ i].add_event(self.update_z1(z1_new_part[ i],z1_part[i],gradx_part[i],gradx_xold_part[i],v_new_part[i],v_part[i], beta_line*tau_new, theta_line, alpha,i,0))
          r_new_part[ i].add_event(self.update_r(r_new_part[i],r_part[i],Ax_part[i],Ax_old_part[i],res_part[i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,0))
          Kyk1_new_part[ i].add_event(self.operator_adjoint_full(Kyk1_new_part[ i],r_new_part[ i],z1_new_part[ i],i,0))


        for i in range(self.num_dev):
          idx_start = (i+1+self.num_dev-1)*self.par_slices-self.overlap
          idx_stop = (i+2+self.num_dev-1)*self.par_slices
          z1_part[i+self.num_dev].set( z1[idx_start:idx_stop,...] ,self.queue[3*i+1])
          gradx_part[i+self.num_dev].set( gradx[idx_start:idx_stop,...] ,self.queue[3*i+1])
          gradx_xold_part[i+self.num_dev].set( gradx_xold[idx_start:idx_stop,...] ,self.queue[3*i+1])
          v_new_part[i+self.num_dev].set(v_new[idx_start:idx_stop,...] ,self.queue[3*i+1])
          v_part[i+self.num_dev].set( v[idx_start:idx_stop,...] ,self.queue[3*i+1])
          r_part[i+self.num_dev].set(r[idx_start:idx_stop,...] ,self.queue[3*i+1])
          Ax_part[i+self.num_dev].set( Ax[idx_start:idx_stop,...] ,self.queue[3*i+1])
          Ax_old_part[i+self.num_dev].set( Axold[idx_start:idx_stop,...] ,self.queue[3*i+1])
          res_part[i+self.num_dev].set( res[idx_start:idx_stop,...] ,self.queue[3*i+1])
          Kyk1_part[i+self.num_dev].set( Kyk1[idx_start:idx_stop,...] ,self.queue[3*i+1])
          self.coil_buf_part[i+self.num_dev].set(self.C[idx_start:idx_stop,...] ,self.queue[3*i+1])
          self.grad_buf_part[i+self.num_dev].set(self.grad_x[idx_start:idx_stop,...] ,self.queue[3*i+1])

        for i in range(self.num_dev):
          z1_new_part[i+self.num_dev].add_event(self.update_z1(z1_new_part[i+self.num_dev],z1_part[self.num_dev+i],gradx_part[self.num_dev+i],gradx_xold_part[self.num_dev+i],v_new_part[self.num_dev+i],v_part[self.num_dev+i], beta_line*tau_new, theta_line, alpha,i,1))
          r_new_part[i+self.num_dev].add_event(self.update_r(r_new_part[i+self.num_dev],r_part[self.num_dev+i],Ax_part[self.num_dev+i],Ax_old_part[self.num_dev+i],res_part[self.num_dev+i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,1))
          Kyk1_new_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_new_part[i+self.num_dev],r_new_part[i+self.num_dev],z1_new_part[i+self.num_dev],i,1))

      #### Stream
        for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
#          tic = time.time()
          for i in range(self.num_dev):
            self.queue[3*i].finish()
            ### Get Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
            z1_new_part[i].get(queue=self.queue[3*i+2],ary=z1_new_tmp)
            r_new_part[i].get(queue=self.queue[3*i+2],ary=r_new_tmp)
            Kyk1_new_part[i].get(queue=self.queue[3*i+2],ary=Kyk1_new_tmp)
            self.queue[3*i+2].finish()
            if idx_start == 0:
              z1_new[idx_start:idx_stop,...] = z1_new_tmp[:(self.par_slices),...]
              r_new[idx_start:idx_stop,...] = r_new_tmp[:(self.par_slices),...]
              Kyk1_new[idx_start:idx_stop,...] = Kyk1_new_tmp[:(self.par_slices),...]
              ynorm += ((clarray.vdot(r_new_part[i][:(self.par_slices),...]-r_part[i][:(self.par_slices),...],r_new_part[i][:(self.par_slices),...]-r_part[i][:(self.par_slices),...],queue=self.queue[3*i])+clarray.vdot(z1_new_part[i][:(self.par_slices),...]-z1_part[i][:(self.par_slices),...],z1_new_part[i][:(self.par_slices),...]-z1_part[i][:(self.par_slices),...],queue=self.queue[3*i]))).get()
              lhs += ((clarray.vdot(Kyk1_new_part[i][:(self.par_slices),...]-Kyk1_part[i][:(self.par_slices),...],Kyk1_new_part[i][:(self.par_slices),...]-Kyk1_part[i][:(self.par_slices),...],queue=self.queue[3*i]))).get()
            else:
              z1_new[idx_start:idx_stop,...] = z1_new_tmp[self.overlap:,...]
              r_new[idx_start:idx_stop,...] = r_new_tmp[self.overlap:,...]
              Kyk1_new[idx_start:idx_stop,...] = Kyk1_new_tmp[self.overlap:,...]
              ynorm += ((clarray.vdot(r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],queue=self.queue[3*i])+clarray.vdot(z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
              lhs += ((clarray.vdot(Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
            ### Put Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)-self.overlap
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices
            z1_part[i].set(  z1[idx_start:idx_stop,...] ,self.queue[3*i])
            gradx_part[i].set(gradx[idx_start:idx_stop,...] ,self.queue[3*i])
            gradx_xold_part[i].set( gradx_xold[idx_start:idx_stop,...] ,self.queue[3*i])
            v_new_part[i].set(v_new[idx_start:idx_stop,...] ,self.queue[3*i])
            v_part[i].set( v[idx_start:idx_stop,...] ,self.queue[3*i])
            r_part[i].set( r[idx_start:idx_stop,...] ,self.queue[3*i])
            Ax_part[i].set( Ax[idx_start:idx_stop,...] ,self.queue[3*i])
            Ax_old_part[i].set(  Axold[idx_start:idx_stop,...] ,self.queue[3*i])
            res_part[i].set(res[idx_start:idx_stop,...] ,self.queue[3*i])
            Kyk1_part[i].set( Kyk1[idx_start:idx_stop,...] ,self.queue[3*i])
            self.coil_buf_part[i].set(self.C[idx_start:idx_stop,...] ,self.queue[3*i])
            self.grad_buf_part[i].set(self.grad_x[idx_start:idx_stop,...],self.queue[3*i])
          for i in range(self.num_dev):
            z1_new_part[i].add_event(self.update_z1(z1_new_part[i],z1_part[i],gradx_part[i],gradx_xold_part[i],v_new_part[i],v_part[i], beta_line*tau_new, theta_line, alpha,i,0))
            r_new_part[i].add_event(self.update_r(r_new_part[i],r_part[i],Ax_part[i],Ax_old_part[i],res_part[i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,0))
            Kyk1_new_part[i].add_event(self.operator_adjoint_full(Kyk1_new_part[i],r_new_part[i],z1_new_part[i],i,0))
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
            ### Put Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices-self.overlap
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            z1_part[i+self.num_dev].set( z1[idx_start:idx_stop,...] ,self.queue[3*i+1])
            gradx_part[i+self.num_dev].set( gradx[idx_start:idx_stop,...] ,self.queue[3*i+1])
            gradx_xold_part[i+self.num_dev].set( gradx_xold[idx_start:idx_stop,...] ,self.queue[3*i+1])
            v_new_part[i+self.num_dev].set(v_new[idx_start:idx_stop,...] ,self.queue[3*i+1])
            v_part[i+self.num_dev].set( v[idx_start:idx_stop,...] ,self.queue[3*i+1])
            r_part[i+self.num_dev].set(r[idx_start:idx_stop,...] ,self.queue[3*i+1])
            Ax_part[i+self.num_dev].set( Ax[idx_start:idx_stop,...] ,self.queue[3*i+1])
            Ax_old_part[i+self.num_dev].set( Axold[idx_start:idx_stop,...] ,self.queue[3*i+1])
            res_part[i+self.num_dev].set( res[idx_start:idx_stop,...] ,self.queue[3*i+1])
            Kyk1_part[i+self.num_dev].set( Kyk1[idx_start:idx_stop,...] ,self.queue[3*i+1])
            self.coil_buf_part[i+self.num_dev].set(self.C[idx_start:idx_stop,...] ,self.queue[3*i+1])
            self.grad_buf_part[i+self.num_dev].set(self.grad_x[idx_start:idx_stop,...] ,self.queue[3*i+1])

          for i in range(self.num_dev):
            z1_new_part[i+self.num_dev].add_event(self.update_z1(z1_new_part[i+self.num_dev],z1_part[self.num_dev+i],gradx_part[self.num_dev+i],gradx_xold_part[self.num_dev+i],v_new_part[self.num_dev+i],v_part[self.num_dev+i], beta_line*tau_new, theta_line, alpha,i,1))
            r_new_part[i+self.num_dev].add_event(self.update_r(r_new_part[i+self.num_dev],r_part[self.num_dev+i],Ax_part[self.num_dev+i],Ax_old_part[self.num_dev+i],res_part[self.num_dev+i],beta_line*tau_new,theta_line,self.irgn_par["lambd"],i,1))
            Kyk1_new_part[i+self.num_dev].add_event(self.operator_adjoint_full(Kyk1_new_part[i+self.num_dev],r_new_part[i+self.num_dev],z1_new_part[i+self.num_dev],i,1))
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
            z1_new[idx_start:idx_stop,...] = z1_new_tmp[:(self.par_slices),...]
            r_new[idx_start:idx_stop,...] = r_new_tmp[:(self.par_slices),...]
            Kyk1_new[idx_start:idx_stop,...] = Kyk1_new_tmp[:(self.par_slices),...]
            ynorm += ((clarray.vdot(r_new_part[i][:(self.par_slices),...]-r_part[i][:(self.par_slices),...],r_new_part[i][:(self.par_slices),...]-r_part[i][:(self.par_slices),...],queue=self.queue[3*i])+clarray.vdot(z1_new_part[i][:(self.par_slices),...]-z1_part[i][:(self.par_slices),...],z1_new_part[i][:(self.par_slices),...]-z1_part[i][:(self.par_slices),...],queue=self.queue[3*i]))).get()
            lhs += ((clarray.vdot(Kyk1_new_part[i][:(self.par_slices),...]-Kyk1_part[i][:(self.par_slices),...],Kyk1_new_part[i][:(self.par_slices),...]-Kyk1_part[i][:(self.par_slices),...],queue=self.queue[3*i]))).get()
          else:
            z1_new[idx_start:idx_stop,...] = z1_new_tmp[self.overlap:,...]
            r_new[idx_start:idx_stop,...] = r_new_tmp[self.overlap:,...]
            Kyk1_new[idx_start:idx_stop,...] = Kyk1_new_tmp[self.overlap:,...]
            ynorm += ((clarray.vdot(r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],r_new_part[i][self.overlap:,...]-r_part[i][self.overlap:,...],queue=self.queue[3*i])+clarray.vdot(z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],z1_new_part[i][self.overlap:,...]-z1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
            lhs += ((clarray.vdot(Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],Kyk1_new_part[i][self.overlap:,...]-Kyk1_part[i][self.overlap:,...],queue=self.queue[3*i]))).get()
          self.queue[3*i+1].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          z1_new_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=z1_new_tmp)
          r_new_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=r_new_tmp)
          Kyk1_new_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=Kyk1_new_tmp)
          self.queue[3*i+2].finish()
          z1_new[idx_start:idx_stop,...] = z1_new_tmp[self.overlap:,...]
          r_new[idx_start:idx_stop,...] = r_new_tmp[self.overlap:,...]
          Kyk1_new[idx_start:idx_stop,...] = Kyk1_new_tmp[self.overlap:,...]
          ynorm += ((clarray.vdot(r_new_part[i+self.num_dev][self.overlap:,...]-r_part[i+self.num_dev][self.overlap:,...],r_new_part[i+self.num_dev][self.overlap:,...]-r_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1])+clarray.vdot(z1_new_part[i+self.num_dev][self.overlap:,...]-z1_part[i+self.num_dev][self.overlap:,...],z1_new_part[i+self.num_dev][self.overlap:,...]-z1_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
          lhs += ((clarray.vdot(Kyk1_new_part[i+self.num_dev][self.overlap:,...]-Kyk1_part[i+self.num_dev][self.overlap:,...],Kyk1_new_part[i+self.num_dev][self.overlap:,...]-Kyk1_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()

        j=0
        for i in range(self.num_dev):
          idx_start = i*self.par_slices
          idx_stop = (i+1)*self.par_slices+self.overlap #### +1 overlap
          z1_new_part[i].set(  z1_new[idx_start:idx_stop,...] ,self.queue[3*i])
          z2_part[i].set(  z2[idx_start:idx_stop,...] ,self.queue[3*i])
          symgrad_v_part[i].set(  symgrad_v[idx_start:idx_stop,...] ,self.queue[3*i])
          symgrad_v_vold_part[i].set( symgrad_v_vold[idx_start:idx_stop,...] ,self.queue[3*i])
          Kyk2_part[i].set( Kyk2[idx_start:idx_stop,...] ,self.queue[3*i])
        for i in range(self.num_dev):
          z2_new_part[ i].add_event(self.update_z2(z2_new_part[i],z2_part[i],symgrad_v_part[i],symgrad_v_vold_part[i],beta_line*tau_new,theta_line,beta,i,0))
          Kyk2_new_part[ i].add_event(self.update_Kyk2(Kyk2_new_part[ i],z2_new_part[ i],z1_new_part[ i],i,0))


        for i in range(self.num_dev):
          idx_start = (i+1+self.num_dev-1)*self.par_slices
          idx_stop = (i+2+self.num_dev-1)*self.par_slices
          if idx_stop == self.NSlice:
            idx_start -=self.overlap
          else:
            idx_stop +=self.overlap
          z1_new_part[i+self.num_dev].set( z1_new[idx_start:idx_stop,...] ,self.queue[3*i+1])
          z2_part[i+self.num_dev].set( z2[idx_start:idx_stop,...] ,self.queue[3*i+1])
          symgrad_v_part[i+self.num_dev].set( symgrad_v[idx_start:idx_stop,...] ,self.queue[3*i+1])
          symgrad_v_vold_part[i+self.num_dev].set(symgrad_v_vold[idx_start:idx_stop,...] ,self.queue[3*i+1])
          Kyk2_part[i+self.num_dev].set( Kyk2[idx_start:idx_stop,...] ,self.queue[3*i+1])

        for i in range(self.num_dev):
          z2_new_part[i+self.num_dev].add_event(self.update_z2(z2_new_part[i+self.num_dev],z2_part[self.num_dev+i],symgrad_v_part[self.num_dev+i],symgrad_v_vold_part[self.num_dev+i],beta_line*tau_new,theta_line,beta,i,1))
          Kyk2_new_part[i+self.num_dev].add_event(self.update_Kyk2(Kyk2_new_part[i+self.num_dev],z2_new_part[i+self.num_dev],z1_new_part[i+self.num_dev],i,1))

      #### Stream
        for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
#          tic = time.time()
          for i in range(self.num_dev):
            self.queue[3*i].finish()
            ### Get Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
            z2_new_part[i].get(queue=self.queue[3*i+2],ary=z2_new_tmp)
            Kyk2_new_part[i].get(queue=self.queue[3*i+2],ary=Kyk2_new_tmp)
            self.queue[3*i+2].finish()
            z2_new[idx_start:idx_stop,...] = z2_new_tmp[:(self.par_slices),...]
            Kyk2_new[idx_start:idx_stop,...] = Kyk2_new_tmp[:(self.par_slices),...]
            ynorm += ((clarray.vdot(z2_new_part[i][:(self.par_slices),...]-z2_part[i][:(self.par_slices),...],z2_new_part[i][:(self.par_slices),...]-z2_part[i][:(self.par_slices),...],queue=self.queue[3*i]))).get()
            lhs += ((clarray.vdot(Kyk2_new_part[i][:(self.par_slices),...]-Kyk2_part[i][:(self.par_slices),...],Kyk2_new_part[i][:(self.par_slices),...]-Kyk2_part[i][:(self.par_slices),...],queue=self.queue[3*i]))).get()
            ### Put Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices+self.overlap
            z1_new_part[i].set(  z1_new[idx_start:idx_stop,...] ,self.queue[3*i])
            z2_part[i].set(  z2[idx_start:idx_stop,...] ,self.queue[3*i])
            symgrad_v_part[i].set(  symgrad_v[idx_start:idx_stop,...] ,self.queue[3*i])
            symgrad_v_vold_part[i].set( symgrad_v_vold[idx_start:idx_stop,...] ,self.queue[3*i])
            Kyk2_part[i].set( Kyk2[idx_start:idx_stop,...] ,self.queue[3*i])
          for i in range(self.num_dev):
            z2_new_part[i].add_event(self.update_z2(z2_new_part[i],z2_part[i],symgrad_v_part[i],symgrad_v_vold_part[i],beta_line*tau_new,theta_line,beta,i,0))
            Kyk2_new_part[i].add_event(self.update_Kyk2(Kyk2_new_part[i],z2_new_part[i],z1_new_part[i],i,0))
          for i in range(self.num_dev):
            ### Get Data
            self.queue[3*i+1].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            z2_new_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=z2_new_tmp)
            Kyk2_new_part[i+self.num_dev].get(queue=self.queue[3*i+2],ary=Kyk2_new_tmp)
            self.queue[3*i+2].finish()
            z2_new[idx_start:idx_stop,...] = z2_new_tmp[:(self.par_slices),...]
            Kyk2_new[idx_start:idx_stop,...] = Kyk2_new_tmp[:(self.par_slices),...]
            ynorm += ((clarray.vdot(z2_new_part[i+self.num_dev][:(self.par_slices),...]-z2_part[i+self.num_dev][:(self.par_slices),...],z2_new_part[i+self.num_dev][:(self.par_slices),...]-z2_part[i+self.num_dev][:(self.par_slices),...],queue=self.queue[3*i+1]))).get()
            lhs += ((clarray.vdot(Kyk2_new_part[i+self.num_dev][:(self.par_slices),...]-Kyk2_part[i+self.num_dev][:(self.par_slices),...],Kyk2_new_part[i+self.num_dev][:(self.par_slices),...]-Kyk2_part[i+self.num_dev][:(self.par_slices),...],queue=self.queue[3*i+1]))).get()
            ### Put Data
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            if idx_stop == self.NSlice:
              idx_start-=self.overlap
            else:
              idx_stop+=self.overlap
            z1_new_part[i+self.num_dev].set( z1_new[idx_start:idx_stop,...] ,self.queue[3*i+1])
            z2_part[i+self.num_dev].set( z2[idx_start:idx_stop,...] ,self.queue[3*i+1])
            symgrad_v_part[i+self.num_dev].set( symgrad_v[idx_start:idx_stop,...] ,self.queue[3*i+1])
            symgrad_v_vold_part[i+self.num_dev].set(symgrad_v_vold[idx_start:idx_stop,...] ,self.queue[3*i+1])
            Kyk2_part[i+self.num_dev].set( Kyk2[idx_start:idx_stop,...] ,self.queue[3*i+1])

          for i in range(self.num_dev):
            z2_new_part[i+self.num_dev].add_event(self.update_z2(z2_new_part[i+self.num_dev],z2_part[self.num_dev+i],symgrad_v_part[self.num_dev+i],symgrad_v_vold_part[self.num_dev+i],beta_line*tau_new,theta_line,beta,i,1))
            Kyk2_new_part[i+self.num_dev].add_event(self.update_Kyk2(Kyk2_new_part[i+self.num_dev],z2_new_part[i+self.num_dev],z1_new_part[i+self.num_dev],i,1))
      #### Collect last block
        if j<2*self.num_dev:
          j = 2*self.num_dev
        else:
          j+=1
        for i in range(self.num_dev):
          self.queue[3*i].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
          z2_new_part[i].get(queue=self.queue[3*i+2],ary=z2_new_tmp)
          Kyk2_new_part[i].get(queue=self.queue[3*i+2],ary=Kyk2_new_tmp)
          self.queue[3*i+2].finish()
          z2_new[idx_start:idx_stop,...] = z2_new_tmp[:(self.par_slices),...]
          Kyk2_new[idx_start:idx_stop,...] = Kyk2_new_tmp[:(self.par_slices),...]
          ynorm += ((clarray.vdot(z2_new_part[i][:(self.par_slices),...]-z2_part[i][:(self.par_slices),...],z2_new_part[i][:(self.par_slices),...]-z2_part[i][:(self.par_slices),...],queue=self.queue[3*i]))).get()
          lhs += ((clarray.vdot(Kyk2_new_part[i][:(self.par_slices),...]-Kyk2_part[i][:(self.par_slices),...],Kyk2_new_part[i][:(self.par_slices),...]-Kyk2_part[i][:(self.par_slices),...],queue=self.queue[3*i]))).get()
          self.queue[3*i+1].finish()
          idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
          z2_new_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=z2_new_tmp)
          Kyk2_new_part[i+self.num_dev].get(queue=self.queue[3*i+1],ary=Kyk2_new_tmp)
          self.queue[3*i+2].finish()
          if idx_stop == self.NSlice:
            z2_new[idx_start:idx_stop,...] = z2_new_tmp[self.overlap:,...]
            Kyk2_new[idx_start:idx_stop,...] = Kyk2_new_tmp[self.overlap:,...]
            ynorm += ((clarray.vdot(z2_new_part[i+self.num_dev][self.overlap:,...]-z2_part[i+self.num_dev][self.overlap:,...],z2_new_part[i+self.num_dev][self.overlap:,...]-z2_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
            lhs += ((clarray.vdot(Kyk2_new_part[i+self.num_dev][self.overlap:,...]-Kyk2_part[i+self.num_dev][self.overlap:,...],Kyk2_new_part[i+self.num_dev][self.overlap:,...]-Kyk2_part[i+self.num_dev][self.overlap:,...],queue=self.queue[3*i+1]))).get()
          else:
            z2_new[idx_start:idx_stop,...] = z2_new_tmp[:(self.par_slices),...]
            Kyk2_new[idx_start:idx_stop,...] = Kyk2_new_tmp[:(self.par_slices),...]
            ynorm += ((clarray.vdot(z2_new_part[i+self.num_dev][:(self.par_slices),...]-z2_part[i+self.num_dev][:(self.par_slices),...],z2_new_part[i+self.num_dev][:(self.par_slices),...]-z2_part[i+self.num_dev][:(self.par_slices),...],queue=self.queue[3*i+1]))).get()
            lhs += ((clarray.vdot(Kyk2_new_part[i+self.num_dev][:(self.par_slices),...]-Kyk2_part[i+self.num_dev][:(self.par_slices),...],Kyk2_new_part[i+self.num_dev][:(self.par_slices),...]-Kyk2_part[i+self.num_dev][:(self.par_slices),...],queue=self.queue[3*i+1]))).get()

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

        dual = (-delta/2*np.vdot(-Kyk1,-Kyk1)- np.vdot(xk,(-Kyk1)) + np.sum(Kyk2)
                  - 1/(2*self.irgn_par["lambd"])*np.vdot(r,r) - np.vdot(res,r)).real

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
#        if (gap > gap_min*self.irgn_par["stag"]) and myit>1:
#          self.v = v_new
#          self.r = r
#          self.z1 = z1
#          self.z2 = z2
#          print("Terminated at iteration %d because the method stagnated"%(myit))
#          return x
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
        idx_stop = (i+1)*self.par_slices
        cl_data.append(clarray.to_device(self.queue[3*i], inp[idx_start:idx_stop+self.overlap,...]))
      for i in range(self.num_dev):
        self.queue[3*i].finish()
#        cl_out[2*i].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_data[i],0))
        (self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_data[i],0)).wait()
      for i in range(self.num_dev):
        idx_start = (i+1+self.num_dev-1)*self.par_slices
        idx_stop = (i+2+self.num_dev-1)*self.par_slices
        if idx_stop == self.NSlice:
          cl_data.append(clarray.to_device(self.queue[3*i+1], inp[idx_start-self.overlap:idx_stop,...]))
        else:
          cl_data.append(clarray.to_device(self.queue[3*i+1], inp[idx_start:idx_stop+self.overlap,...]))
      for i in range(self.num_dev):
        self.queue[3*i+1].finish()
        cl_out[2*i+1].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_data[self.num_dev+i],1))
#        (self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_data[self.num_dev+i],1)).wait()

      for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
          for i in range(self.num_dev):
            self.queue[3*i].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
            cl_out[2*i].get(queue=self.queue[3*i+2],ary=self.tmp_FT)
            self.queue[3*i+1].finish()
            outp[idx_start:idx_stop,...]=self.tmp_FT[:(self.par_slices),...]
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices
            cl_data[i] = clarray.to_device(self.queue[3*i], inp[idx_start:idx_stop+self.overlap,...])
          for i in range(self.num_dev):
            self.queue[3*i].finish()
            cl_out[2*i].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_data[i],0))
#            (self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_data[i],0)).wait()
          for i in range(self.num_dev):
            self.queue[3*i+1].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            cl_out[2*i+1].get(queue=self.queue[3*i+2],ary=self.tmp_FT)
            self.queue[3*i+2].finish()
            outp[idx_start:idx_stop,...]=self.tmp_FT[:(self.par_slices),...]
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            if idx_stop == self.NSlice:
              cl_data[i+self.num_dev] = clarray.to_device(self.queue[3*i+1],inp[idx_start-self.overlap:idx_stop,...])
            else:
              cl_data[i+self.num_dev] = clarray.to_device(self.queue[3*i+1],inp[idx_start:idx_stop+self.overlap,...])
          for i in range(self.num_dev):
            self.queue[3*i+1].finish()
            cl_out[2*i+1].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_data[i+self.num_dev],1))
#            (self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_data[self.num_dev+i],1)).wait()
      if j< 2*self.num_dev:
        j = 2*self.num_dev
      else:
        j+=1
      for i in range(self.num_dev):
        self.queue[3*i].finish()
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
        cl_out[2*i].get(queue=self.queue[3*i+2],ary=self.tmp_FT)
        self.queue[3*i+2].finish()
        outp[idx_start:idx_stop,...]=self.tmp_FT[:(self.par_slices),...]
        self.queue[3*i+1].finish()
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        cl_out[2*i+1].get(queue=self.queue[3*i+1],ary=self.tmp_FT)
        self.queue[3*i+2].finish()
        if idx_stop == self.NSlice:
          outp[idx_start:idx_stop,...]=self.tmp_FT[self.overlap:,...]
        else:
          outp[idx_start:idx_stop,...]=self.tmp_FT[:(self.par_slices),...]


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
        idx_stop = (i+1)*self.par_slices
        cl_data.append(clarray.to_device(self.queue[3*i], inp[idx_start:idx_stop+self.overlap,...]))
        self.coil_buf_part.append(clarray.to_device(self.queue[3*i], self.C[idx_start:idx_stop+self.overlap,...]))
        self.grad_buf_part.append(clarray.to_device(self.queue[3*i], self.grad_x[idx_start:idx_stop+self.overlap,...]))

      for i in range(self.num_dev):
#        cl_tmp[2*i].add_event(self.eval_fwd_streamed(cl_tmp[2*i],cl_data[i],idx=i,idxq=0,wait_for=cl_tmp[2*i].events+cl_data[i].events
#              +self.coil_buf_part[i].events+self.grad_buf_part[i].events))
#        cl_out[2*i].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_tmp[2*i],0,wait_for=cl_tmp[2*i].events+cl_out[2*i].events))
        (self.eval_fwd_streamed(cl_tmp[2*i],cl_data[i],idx=i,idxq=0,wait_for=cl_tmp[2*i].events+cl_data[i].events
              +self.coil_buf_part[i].events+self.grad_buf_part[i].events)).wait()
        (self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_tmp[2*i],0,wait_for=cl_tmp[2*i].events+cl_out[2*i].events)).wait()


      for i in range(self.num_dev):
        idx_start = (i+1+self.num_dev-1)*self.par_slices
        idx_stop = (i+2+self.num_dev-1)*self.par_slices

        if idx_stop==self.NSlice:
          cl_data.append(clarray.to_device(self.queue[3*i+1], inp[idx_start-self.overlap:idx_stop,...] ))
          self.coil_buf_part.append(clarray.to_device(self.queue[3*i+1], self.C[idx_start-self.overlap:idx_stop,...]))
          self.grad_buf_part.append(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start-self.overlap:idx_stop,...]))
        else:
          cl_data.append(clarray.to_device(self.queue[3*i+1], inp[idx_start:idx_stop+self.overlap,...]))
          self.coil_buf_part.append(clarray.to_device(self.queue[3*i+1], self.C[idx_start:idx_stop+self.overlap,...]))
          self.grad_buf_part.append(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start:idx_stop+self.overlap,...]))

      for i in range(self.num_dev):
        cl_tmp[2*i+1].add_event(self.eval_fwd_streamed(cl_tmp[2*i+1],cl_data[self.num_dev+i],idx=i,idxq=1,wait_for=cl_tmp[2*i+1].events+cl_data[self.num_dev+i].events
              +self.coil_buf_part[self.num_dev+i].events+self.grad_buf_part[self.num_dev+i].events))
        cl_out[2*i+1].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_tmp[2*i+1],1,wait_for=cl_tmp[2*i+1].events+cl_out[2*i+1].events))
#        (self.eval_fwd_streamed(cl_tmp[2*i+1],cl_data[self.num_dev+i],idx=i,idxq=1,wait_for=cl_tmp[2*i+1].events+cl_data[self.num_dev+i].events
#              +self.coil_buf_part[self.num_dev+i].events+self.grad_buf_part[self.num_dev+i].events)).wait()
#        (self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_tmp[2*i+1],1,wait_for=cl_tmp[2*i+1].events+cl_out[2*i+1].events)).wait()


      for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
          for i in range(self.num_dev):
            self.queue[3*i].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
            cl_out[2*i].get(queue=self.queue[3*i+2],ary=self.tmp_FT)
            self.queue[3*i+2].finish()
            outp[idx_start:idx_stop,...]=self.tmp_FT[:(self.par_slices),...]

            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices
            cl_data[i] = clarray.to_device(self.queue[3*i],inp[idx_start:idx_stop+self.overlap,...])
            self.coil_buf_part[i]=(clarray.to_device(self.queue[3*i], self.C[idx_start:idx_stop+self.overlap,...]))
            self.grad_buf_part[i]=(clarray.to_device(self.queue[3*i], self.grad_x[idx_start:idx_stop+self.overlap,...]))

          for i in range(self.num_dev):
            cl_tmp[2*i].add_event(self.eval_fwd_streamed(cl_tmp[2*i],cl_data[i],idx=i,idxq=0,wait_for=cl_tmp[2*i].events+cl_data[i].events
                  +self.coil_buf_part[i].events+self.grad_buf_part[i].events))
            cl_out[2*i].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_tmp[2*i],0,wait_for=cl_tmp[2*i].events+cl_out[2*i].events))
#            (self.eval_fwd_streamed(cl_tmp[2*i],cl_data[i],idx=i,idxq=0,wait_for=cl_tmp[2*i].events+cl_data[i].events
#                  +self.coil_buf_part[i].events+self.grad_buf_part[i].events)).wait()
#            (self.NUFFT[i].fwd_NUFFT(cl_out[2*i],cl_tmp[2*i],0,wait_for=cl_tmp[2*i].events+cl_out[2*i].events)).wait()
          for i in range(self.num_dev):
            self.queue[3*i+1].finish()
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            cl_out[2*i+1].get(queue=self.queue[3*i+2],ary=self.tmp_FT)
            self.queue[3*i+2].finish()
            outp[idx_start:idx_stop,...]=self.tmp_FT[:(self.par_slices),...]

            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            if idx_stop == self.NSlice:
              cl_data[i+self.num_dev] = clarray.to_device(self.queue[3*i+1], inp[idx_start-self.overlap:idx_stop,...])
              self.coil_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.C[idx_start-self.overlap:idx_stop,...]))
              self.grad_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start-self.overlap:idx_stop,...]))
            else:
              cl_data[i+self.num_dev] = clarray.to_device(self.queue[3*i+1], inp[idx_start:idx_stop+self.overlap,...])
              self.coil_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.C[idx_start:idx_stop+self.overlap,...]))
              self.grad_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start:idx_stop+self.overlap,...]))

          for i in range(self.num_dev):
            cl_tmp[2*i+1].add_event(self.eval_fwd_streamed(cl_tmp[2*i+1],cl_data[self.num_dev+i],idx=i,idxq=1,wait_for=cl_tmp[2*i+1].events+cl_data[self.num_dev+i].events
                  +self.coil_buf_part[self.num_dev+i].events+self.grad_buf_part[self.num_dev+i].events))
            cl_out[2*i+1].add_event(self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_tmp[2*i+1],1,wait_for=cl_tmp[2*i+1].events+cl_out[2*i+1].events))
#            (self.eval_fwd_streamed(cl_tmp[2*i+1],cl_data[self.num_dev+i],idx=i,idxq=1,wait_for=cl_tmp[2*i+1].events+cl_data[self.num_dev+i].events+self.coil_buf_part[self.num_dev+i].events+self.grad_buf_part[self.num_dev+i].events)).wait()
#            (self.NUFFT[i].fwd_NUFFT(cl_out[2*i+1],cl_tmp[2*i+1],1,wait_for=cl_tmp[2*i+1].events+cl_out[2*i+1].events)).wait()
      if j<2*self.num_dev:
        j = 2*self.num_dev
      else:
        j+=1
      for i in range(self.num_dev):
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
        self.queue[3*i].finish()
        cl_out[2*i].get(queue=self.queue[3*i+2],ary=self.tmp_FT)
        self.queue[3*i+2].finish()
        outp[idx_start:idx_stop,...]=self.tmp_FT[:(self.par_slices),...]
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        self.queue[3*i+1].finish()
        cl_out[2*i+1].get(queue=self.queue[3*i+1],ary=self.tmp_FT)
        self.queue[3*i+2].finish()
        if idx_stop==self.NSlice:
          outp[idx_start:idx_stop,...]=self.tmp_FT[self.overlap:,...]
        else:
          outp[idx_start:idx_stop,...]=self.tmp_FT[:(self.par_slices),...]

  def FTH_streamed(self,outp,inp):
      cl_out = []
#      par_slices = self.par_slices
      for i in range(self.num_dev):
        cl_out.append(clarray.zeros(self.queue[3*i],(self.par_slices+self.overlap,self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE))
        cl_out.append(clarray.zeros(self.queue[3*i+1],(self.par_slices+self.overlap,self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE))
      cl_data = []
      for i in range(self.num_dev):
#        print("Put Start: %i, Stop: %i" %(i*par_slices,(i+1)*par_slices))
        idx_start = i*self.par_slices
        idx_stop = (i+1)*self.par_slices
        cl_data.append(clarray.to_device(self.queue[3*i], inp[idx_start:idx_stop+self.overlap,...]))
      for i in range(self.num_dev):
        cl_out[2*i].add_event(self.NUFFT[i].adj_NUFFT(cl_out[2*i],cl_data[i],0))

#      for i in range(self.num_dev):
#        self.queue[3*i].finish()

      for i in range(self.num_dev):
#        print("Put Start: %i, Stop: %i" %((i+1+self.num_dev-1)*par_slices,(i+2+self.num_dev-1)*par_slices))
        idx_start = (i+1+self.num_dev-1)*self.par_slices
        idx_stop = (i+2+self.num_dev-1)*self.par_slices
        if idx_stop == self.NSlice:
          cl_data.append(clarray.to_device(self.queue[3*i+1], inp[idx_start-self.overlap:idx_stop,...]))
        else:
          cl_data.append(clarray.to_device(self.queue[3*i+1], inp[idx_start:idx_stop+self.overlap,...]))
      for i in range(self.num_dev):
        cl_out[2*i+1].add_event(self.NUFFT[i].adj_NUFFT(cl_out[2*i+1],cl_data[self.num_dev+i],1))

      for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
          for i in range(self.num_dev):
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
            cl_out[2*i].get(queue=self.queue[3*i+2],ary=self.tmp_FTH)
#            print("Get Start: %i, Stop: %i" %(i*par_slices+(2*self.num_dev*(j-2*self.num_dev)*par_slices),(i+1)*par_slices+(2*self.num_dev*(j-2*self.num_dev))*par_slices))
            outp[idx_start:idx_stop,...]=self.tmp_FTH[:(self.par_slices),...]
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
#            print("Put Start: %i, Stop: %i" %(i*par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*par_slices),(i+1)*par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*par_slices))
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices
            cl_data[i] = clarray.to_device(self.queue[3*i], inp[idx_start:idx_stop+self.overlap,...])
          for i in range(self.num_dev):
            cl_out[2*i].add_event(self.NUFFT[i].adj_NUFFT(cl_out[2*i],cl_data[i],0))
          for i in range(self.num_dev):
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            cl_out[2*i+1].get(queue=self.queue[3*i+2],ary=self.tmp_FTH)
#            print("Get Start: %i, Stop: %i" %(i*par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*par_slices,(i+1)*par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*par_slices))
            outp[idx_start:idx_stop,...]=self.tmp_FTH[:(self.par_slices),...]
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
#            print("Put Start: %i, Stop: %i" %(i*par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*par_slices,(i+1)*par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*par_slices))
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            if idx_stop == self.NSlice:
              cl_data[i+self.num_dev] = clarray.to_device(self.queue[3*i+1],inp[idx_start-self.overlap:idx_stop,...])
            else:
              cl_data[i+self.num_dev] = clarray.to_device(self.queue[3*i+1],inp[idx_start:idx_stop+self.overlap,...])
          for i in range(self.num_dev):
            cl_out[2*i+1].add_event(self.NUFFT[i].adj_NUFFT(cl_out[2*i+1],cl_data[i+self.num_dev],1))
      if j< 2*self.num_dev:
        j = 2*self.num_dev
      else:
        j+=1
      for i in range(self.num_dev):
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
        cl_out[2*i].get(queue=self.queue[3*i+2],ary=self.tmp_FTH)
#        print("Get Start: %i, Stop: %i" %(i*par_slices+(2*self.num_dev*(j-2*self.num_dev)*par_slices),(i+1)*par_slices+(2*self.num_dev*(j-2*self.num_dev))*par_slices))
        outp[idx_start:idx_stop,...]=self.tmp_FTH[:(self.par_slices),...]
#      for i in range(self.num_dev):
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        cl_out[2*i+1].get(queue=self.queue[3*i+1],ary=self.tmp_FTH)
#        print("Get Start: %i, Stop: %i" %(i*par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*par_slices,(i+1)*par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*par_slices))
        if idx_stop == self.NSlice:
          outp[idx_start:idx_stop,...]=self.tmp_FTH[self.overlap:,...]
        else:
          outp[idx_start:idx_stop,...]=self.tmp_FTH[:(self.par_slices),...]



  def eval_adj_streamed(self,y,x,idx=0,idxq=0,wait_for=[]):
    return self.prg[idx].operator_ad(self.queue[3*idx+idxq], (self.par_slices+self.overlap,self.dimY,self.dimX), None,
                                 y.data, x.data, self.coil_buf_part[idx+idxq*self.num_dev].data, self.grad_buf_part[idx+idxq*self.num_dev].data,
                                 np.int32(self.NC), np.int32(self.par_slices+self.overlap), np.int32(self.NScan), np.int32(self.unknowns),
                                 wait_for= y.events+x.events+wait_for)

  def operator_adjoint_streamed(self, outp, inp):
      cl_out = []
      cl_tmp = []
      self.coil_buf_part = []
      self.grad_buf_part = []


      for i in range(self.num_dev):
        cl_out.append(clarray.zeros(self.queue[3*i],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE))
        cl_out.append(clarray.zeros(self.queue[3*i+1],(self.par_slices+self.overlap,self.unknowns,self.dimY,self.dimX),dtype=DTYPE))
        cl_tmp.append(clarray.zeros(self.queue[3*i],(self.par_slices+self.overlap,self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE))
        cl_tmp.append(clarray.zeros(self.queue[3*i+1],(self.par_slices+self.overlap,self.NScan,self.NC,self.dimY,self.dimX),dtype=DTYPE))

      cl_data = []
      for i in range(self.num_dev):
        idx_start = i*self.par_slices
        idx_stop = (i+1)*self.par_slices
        cl_data.append(clarray.to_device(self.queue[3*i], inp[idx_start:idx_stop+self.overlap,...]))
        self.coil_buf_part.append(clarray.to_device(self.queue[3*i], self.C[idx_start:idx_stop+self.overlap,...]))
        self.grad_buf_part.append(clarray.to_device(self.queue[3*i], self.grad_x[idx_start:idx_stop+self.overlap,...]))

      for i in range(self.num_dev):
        cl_tmp[2*i].add_event(self.NUFFT[i].adj_NUFFT(cl_tmp[2*i],cl_data[i],0,wait_for=cl_tmp[2*i].events+cl_data[i].events))
        cl_out[2*i].add_event(self.eval_adj_streamed(cl_out[2*i],cl_tmp[2*i],idx=i,idxq=0,wait_for=cl_tmp[2*i].events+cl_out[2*i].events
              +self.coil_buf_part[i].events+self.grad_buf_part[i].events))



      for i in range(self.num_dev):
        idx_start = (i+1+self.num_dev-1)*self.par_slices
        idx_stop = (i+2+self.num_dev-1)*self.par_slices

        if idx_stop==self.NSlice:
          cl_data.append(clarray.to_device(self.queue[3*i+1], inp[idx_start-self.overlap:idx_stop,...] ))
          self.coil_buf_part.append(clarray.to_device(self.queue[3*i+1], self.C[idx_start-self.overlap:idx_stop,...]))
          self.grad_buf_part.append(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start-self.overlap:idx_stop,...]))
        else:
          cl_data.append(clarray.to_device(self.queue[3*i+1], inp[idx_start:idx_stop+self.overlap,...]))
          self.coil_buf_part.append(clarray.to_device(self.queue[3*i+1], self.C[idx_start:idx_stop+self.overlap,...]))
          self.grad_buf_part.append(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start:idx_stop+self.overlap,...]))

      for i in range(self.num_dev):
        cl_tmp[2*i+1].add_event(self.NUFFT[i].adj_NUFFT(cl_tmp[2*i+1],cl_data[self.num_dev+i],1,wait_for=cl_tmp[2*i+1].events+cl_data[self.num_dev+i].events))
        cl_out[2*i+1].add_event(self.eval_adj_streamed(cl_out[2*i+1],cl_tmp[2*i+1],idx=i,idxq=1,wait_for=cl_tmp[2*i+1].events+cl_out[2*i+1].events
              +self.coil_buf_part[self.num_dev+i].events+self.grad_buf_part[self.num_dev+i].events))



      for j in range(2*self.num_dev,int(self.NSlice/(2*self.par_slices*self.num_dev)+(2*self.num_dev-1))):
          for i in range(self.num_dev):
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
            cl_out[2*i].get(queue=self.queue[3*i+2],ary=self.tmp_out)
            outp[idx_start:idx_stop,...]=self.tmp_out[:(self.par_slices),...]
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)*self.par_slices)
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1))*self.par_slices
#            print("Put1 send Start: %i, stop: %i"%(idx_start,idx_stop))
            cl_data[i] = clarray.to_device(self.queue[3*i],inp[idx_start:idx_stop+self.overlap,...])
            self.coil_buf_part[i]=(clarray.to_device(self.queue[3*i], self.C[idx_start:idx_stop+self.overlap,...]))
            self.grad_buf_part[i]=(clarray.to_device(self.queue[3*i], self.grad_x[idx_start:idx_stop+self.overlap,...]))
          for i in range(self.num_dev):
            cl_tmp[2*i].add_event(self.NUFFT[i].adj_NUFFT(cl_tmp[2*i],cl_data[i],0,wait_for=cl_tmp[2*i].events+cl_data[i].events))
            cl_out[2*i].add_event(self.eval_adj_streamed(cl_out[2*i],cl_tmp[2*i],idx=i,idxq=0,wait_for=cl_tmp[2*i].events+cl_out[2*i].events
                  +self.coil_buf_part[i].events+self.grad_buf_part[i].events))


          for i in range(self.num_dev):
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
#            print("Get2 send Start: %i, stop: %i"%(idx_start,idx_stop))
            cl_out[2*i+1].get(queue=self.queue[3*i+2],ary=self.tmp_out)
            outp[idx_start:idx_stop,...]=self.tmp_out[:(self.par_slices),...]
            idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
            idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev+1)+self.num_dev)*self.par_slices
#            print("Put2 send Start: %i, stop: %i"%(idx_start,idx_stop))
            if idx_stop == self.NSlice:
              cl_data[i+self.num_dev] = clarray.to_device(self.queue[3*i+1], inp[idx_start-self.overlap:idx_stop,...])
              self.coil_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.C[idx_start-self.overlap:idx_stop,...]))
              self.grad_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start-self.overlap:idx_stop,...]))
            else:
              cl_data[i+self.num_dev] = clarray.to_device(self.queue[3*i+1], inp[idx_start:idx_stop+self.overlap,...])
              self.coil_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.C[idx_start:idx_stop+self.overlap,...]))
              self.grad_buf_part[self.num_dev+i]=(clarray.to_device(self.queue[3*i+1], self.grad_x[idx_start:idx_stop+self.overlap,...]))
          for i in range(self.num_dev):
            cl_tmp[2*i+1].add_event(self.NUFFT[i].adj_NUFFT(cl_tmp[2*i+1],cl_data[self.num_dev+i],1,wait_for=cl_tmp[2*i+1].events+cl_data[self.num_dev+i].events))
            cl_out[2*i+1].add_event(self.eval_adj_streamed(cl_out[2*i+1],cl_tmp[2*i+1],idx=i,idxq=1,wait_for=cl_tmp[2*i+1].events+cl_out[2*i+1].events
                  +self.coil_buf_part[self.num_dev+i].events+self.grad_buf_part[self.num_dev+i].events))
      if j<2*self.num_dev:
        j = 2*self.num_dev
      else:
        j+=1
      for i in range(self.num_dev):
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)*self.par_slices)
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev))*self.par_slices
#        print("Get1  Start: %i, stop: %i"%(idx_start,idx_stop))
        cl_out[2*i].get(queue=self.queue[3*i+2],ary=self.tmp_out)
        outp[idx_start:idx_stop,...]=self.tmp_out[:(self.par_slices),...]
        idx_start = i*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
        idx_stop = (i+1)*self.par_slices+(2*self.num_dev*(j-2*self.num_dev)+self.num_dev)*self.par_slices
#        print("Get2 Start: %i, stop: %i"%(idx_start,idx_stop))
        cl_out[2*i+1].get(queue=self.queue[3*i+1],ary=self.tmp_out)
        if idx_stop==self.NSlice:
          outp[idx_start:idx_stop,...]=self.tmp_out[self.overlap:,...]
        else:
          outp[idx_start:idx_stop,...]=self.tmp_out[:(self.par_slices),...]