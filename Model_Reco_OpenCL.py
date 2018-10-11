
# cython: infer_types=True
# cython: profile=False

from __future__ import division

import numpy as np
import time

import gridroutines as NUFFT

import matplotlib.pyplot as plt
plt.ion()

DTYPE = np.complex64
DTYPE_real = np.float32


#import pynfft.nfft as nfft

import pyopencl as cl
import pyopencl.array as clarray
import multislice_viewer as msv
import sys

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
    self.C = par.C
    self.traj = traj
    self.unknowns_TGV = par.unknowns_TGV
    self.unknowns_H1 = par.unknowns_H1
    self.unknowns = par.unknowns
    self.NSlice = par.NSlice
    self.NScan = par.NScan
    self.dimX = par.dimX
    self.dimY = par.dimY
    self.NC = par.NC
    self.fval_min = 0
    self.fval = 0
    self.ctx = ctx[0]
    self.queue = queue[0]
    self.coil_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.C.data)
    self.ratio = clarray.to_device(self.queue,(1*np.ones(self.unknowns)).astype(dtype=DTYPE_real))
    self.ratio[1] = 1
    self.ukscale =  clarray.to_device(self.queue,np.ones(self.unknowns,dtype=DTYPE_real))
    self.gn_res = []
    self.N = par.N
    self.Nproj = par.Nproj
    if trafo:
      self.NUFFT = NUFFT.gridding(self.ctx,queue,4,2,par.N,par.NScan,(par.NScan*par.NC*par.NSlice,par.N,par.N),(1,2),traj.astype(DTYPE),np.require(np.abs(dcf),DTYPE_real,requirements='C'),par.N,1000,DTYPE,DTYPE_real,radial=trafo)
    else:
      self.NUFFT = NUFFT.gridding(self.ctx,queue,4,2,par.N,par.NScan,(par.NScan*par.NC*par.NSlice,par.N,par.N),(1,2),traj,dcf,par.N,1000,DTYPE,DTYPE_real,radial=trafo)

    self.prg = Program(self.ctx, r"""
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
  size_t i = k*Nx*Ny+Nx*y + x;

  float fac = 0.0f;

  for (int uk=0; uk<NUk; uk++)
  {
     z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]);

     // reproject
     fac = hypot(fac,hypot(
     hypot(hypot(hypot(z_new[i].s0,z_new[i].s1), hypot(z_new[i].s2,z_new[i].s3)),hypot(z_new[i].s4,z_new[i].s5)),
     hypot(hypot(2.0f*hypot(z_new[i].s6,z_new[i].s7),2.0f*hypot(z_new[i].s8,z_new[i].s9)),2.0f*hypot(z_new[i].sa,z_new[i].sb)))*alphainv);
     i += NSl*Nx*Ny;
   }

  i = k*Nx*Ny+Nx*y + x;
  for (int uk=0; uk<NUk; uk++)
  {
    if (fac > 1.0f) z_new[i] /=fac;
    i += NSl*Nx*Ny;
  }
}
__kernel void update_z1(__global float8 *z_new, __global float8 *z, __global float8 *gx,__global float8 *gx_,
                          __global float8 *vx,__global float8 *vx_, const float sigma, const float theta, const float alphainv,
                          const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  float fac = 0.0f;

  for (int uk=0; uk<NUk; uk++)
  {
     z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]-(1+theta)*vx[i]+theta*vx_[i]);

     // reproject
     fac = hypot(fac,hypot(hypot(z_new[i].s0,z_new[i].s1), hypot(hypot(z_new[i].s2,z_new[i].s3),hypot(z_new[i].s4,z_new[i].s5)))*alphainv);
     i += NSl*Nx*Ny;
  }
  i = k*Nx*Ny+Nx*y + x;
  for (int uk=0; uk<NUk; uk++)
  {
    if (fac > 1.0f) z_new[i] /=fac;
    i += NSl*Nx*Ny;
  }
}
  __kernel void update_z1_tv(__global float8 *z_new, __global float8 *z, __global float8 *gx,__global float8 *gx_,
                          const float sigma, const float theta, const float alphainv,
                          const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  float fac = 0.0f;

  for (int uk=0; uk<NUk; uk++)
  {
     z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]);

     // reproject
     fac = hypot(fac,hypot(hypot(z_new[i].s0,z_new[i].s1), hypot(hypot(z_new[i].s2,z_new[i].s3),hypot(z_new[i].s4,z_new[i].s5)))*alphainv);
     i += NSl*Nx*Ny;
  }
  i = k*Nx*Ny+Nx*y + x;
  for (int uk=0; uk<NUk; uk++)
  {
    if (fac > 1.0f) z_new[i] /=fac;
    i += NSl*Nx*Ny;
  }
}
__kernel void update_primal(__global float2 *u_new, __global float2 *u, __global float2 *Kyk,__global float2 *u_k, const float tau, const float tauinv, float div, __global float* min, __global float* max, __global int* real, const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;


  for (int uk=0; uk<NUk; uk++)
  {
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
     i += NSl*Nx*Ny;
  }
}

__kernel void gradient(__global float8 *grad, __global float2 *u, const int NUk, __global float* scale, const float maxscal, __global float* ratio) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;


  for (int uk=0; uk<NUk; uk++)
  {
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
     { grad[i].s45 += u[i+Nx*Ny].s01;}
     else
     { grad[i].s45 = 0.0f;}
     // scale gradients
     if (uk>0)
     {grad[i]*=maxscal/(ratio[uk]*scale[uk]);}
     else
     {grad[i]*=(maxscal/(ratio[uk]*scale[uk]));}
     i += NSl*Nx*Ny;
  }
}

__kernel void sym_grad(__global float16 *sym, __global float8 *w, const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;


  for (int uk=0; uk<NUk; uk++)
  {
     // symmetrized gradient
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
     i += NSl*Nx*Ny;
   }
}
__kernel void divergence(__global float2 *div, __global float8 *p, const int NUk,
                         __global float* scale, const float maxscal, __global float* ratio) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  for (int ukn=0; ukn<NUk; ukn++)
  {
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
         val.s4 -= p[i-Nx*Ny].s4;
         //imag
         val.s5 -= p[i-Nx*Ny].s5;
     }
     div[i] = val.s01+val.s23+val.s45;
     // scale gradients
     if (ukn>0)
     {div[i]*=maxscal/(ratio[ukn]*scale[ukn]);}
     else
     {div[i]*=(maxscal/(ratio[ukn]*scale[ukn]));}
     i += NSl*Nx*Ny;
  }

}
__kernel void sym_divergence(__global float8 *w, __global float16 *q,
                       const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  for (int uk=0; uk<NUk; uk++)
  {
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
     w[i].s024 = val_real.s012 + val_real.s345 + val_real.s678;
     //imag
     w[i].s135 = val_imag.s012 + val_imag.s345 + val_imag.s678;
     i += NSl*Nx*Ny;
  }
}
__kernel void update_Kyk2(__global float8 *w, __global float16 *q, __global float8 *z,
                       const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  for (int uk=0; uk<NUk; uk++)
  {
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
     i += NSl*Nx*Ny;
  }
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
  int ii = i;

  float4 o = ofs[j+scan*J];
  float2 acc = 0.0f;

  for(int y = 0; y < Y; y++) {
    int x_low, x_high;
    float d = y*o.y + o.z;

    // compute bounds
    if (o.x == 0) {
      if ((d > ii-1) && (d < ii+1)) {
        x_low = 0; x_high = X-1;
      } else {
        img += X; continue;
      }
    } else if (o.x > 0) {
      x_low = (int)((ii-1 - d)*o.w);
      x_high = (int)((ii+1 - d)*o.w);
    } else {
      x_low = (int)((ii+1 - d)*o.w);
      x_high = (int)((ii-1 - d)*o.w);
    }
    x_low = max(x_low, 0);
    x_high = min(x_high, X-1);

    // integrate
    for(int x = x_low; x <= x_high; x++) {
      float weight = 1.0 - fabs(x*o.x + d - ii);
      if (weight > 0.0f) acc += weight*img[x];
    }
    img += X;
  }
  sino[k*I*J+j*I + ii] = acc/scale;
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
        tmp_coil = coils[coil*NSl*X*Y + k*X*Y + y*X + x];
        float2 sum = 0.0f;
        for (int uk=0; uk<Nuk; uk++)
        {
          tmp_grad = grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x];
          tmp_in = in[uk*NSl*X*Y+k*X*Y+ y*X + x];

          tmp_mul = (float2)(tmp_in.x*tmp_grad.x-tmp_in.y*tmp_grad.y,tmp_in.x*tmp_grad.y+tmp_in.y*tmp_grad.x);
          sum += (float2)(tmp_mul.x*tmp_coil.x-tmp_mul.y*tmp_coil.y,
                                                    tmp_mul.x*tmp_coil.y+tmp_mul.y*tmp_coil.x);

        }
        out[scan*NCo*NSl*X*Y+coil*NSl*X*Y+k*X*Y + y*X + x] = sum;
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
    conj_grad = (float2) (grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].x,
                          -grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].y);
  for (int coil=0; coil < NCo; coil++)
  {
    conj_coils = (float2) (coils[coil*NSl*X*Y + k*X*Y + y*X + x].x,
                                  -coils[coil*NSl*X*Y + k*X*Y + y*X + x].y);

    tmp_in = in[scan*NCo*NSl*X*Y+coil*NSl*X*Y + k*X*Y+ y*X + x];
    tmp_mul = (float2)(tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x);


    sum += (float2)(tmp_mul.x*conj_coils.x-tmp_mul.y*conj_coils.y,
                                     tmp_mul.x*conj_coils.y+tmp_mul.y*conj_coils.x);
  }
  }
  out[uk*NSl*X*Y+k*X*Y+y*X+x] = sum;
  }

}



__kernel void update_Kyk1(__global float2 *out, __global float2 *in,
                       __global float2 *coils, __global float2 *grad, __global float8 *p, const int NCo,
                       const int NSl, const int NScan, __global float* scale, const float maxscal, __global float* ratio, const int Nuk)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);

  size_t i = k*X*Y+X*y + x;

  float2 tmp_in = 0.0f;
  float2 tmp_mul = 0.0f;
  float2 conj_grad = 0.0f;
  float2 conj_coils = 0.0f;


  for (int uk=0; uk<Nuk; uk++)
  {
  float2 sum = (float2) 0.0f;
  for (int scan=0; scan<NScan; scan++)
  {
    conj_grad = (float2) (grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].x,
                          -grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].y);
  for (int coil=0; coil < NCo; coil++)
  {
    conj_coils = (float2) (coils[coil*NSl*X*Y + k*X*Y + y*X + x].x,
                                  -coils[coil*NSl*X*Y + k*X*Y + y*X + x].y);

    tmp_in = in[scan*NCo*NSl*X*Y+coil*NSl*X*Y + k*X*Y+ y*X + x];
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
       val.s4 -= p[i-X*Y].s4;
       //imag
       val.s5 -= p[i-X*Y].s5;
   }

   // scale gradients
   if (uk>0)
   {val*=maxscal/(ratio[uk]*scale[uk]);}
   else
   {val*=(maxscal/(ratio[uk]*scale[uk]));}

  out[uk*NSl*X*Y+k*X*Y+y*X+x] = sum - (val.s01+val.s23+val.s45);
  i += NSl*X*Y;
  }

}
__kernel void update_primal_explicit(__global float2 *u_new, __global float2 *u, __global float2 *Kyk, __global float2 *u_k,
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
""")
    self.tmp_result = clarray.zeros(self.queue,(self.NScan,self.NC,self.NSlice,self.dimY,self.dimX),DTYPE,"C")
    self.tmp_result2 = clarray.zeros(self.queue,(self.unknowns,self.NSlice,self.dimY,self.dimX),DTYPE,"C")
    self.tmp_img =  clarray.zeros(self.queue,(self.NScan,self.NC,self.NSlice,self.dimY,self.dimX),DTYPE,"C")
    self.tmp_sino = clarray.zeros(self.queue,(self.NScan,self.NC,self.NSlice,self.Nproj,self.N),DTYPE,"C")
    print("Please Set Parameters, Data and Initial images")


  def irgn_solve_2D(self, x, iters, data):


    ###################################
    ### Adjointness
    x= x[:,None,...]
    data= data[:,:,None,...]
    x_old = np.copy(x)

    x = clarray.to_device(self.queue,np.require(x,requirements="C"))
    a = (self.operator_forward(x).get())
    b = clarray.zeros(self.queue, data.shape,dtype=DTYPE)
    (self.FT(b,clarray.to_device(self.queue,self.step_val[:,None,...]*self.Coils)))
    res = (data) - b.get() + a

    x = self.tgv_solve_2D(x.get(),res,iters)
    x = clarray.to_device(self.queue,x)
    v = clarray.to_device(self.queue,self.v)
    grad = clarray.to_device(self.queue,np.zeros_like(self.z1))
    sym_grad = clarray.to_device(self.queue,np.zeros_like(self.z2))
    grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))
    sym_grad.add_event(self.sym_grad(sym_grad,v,wait_for=sym_grad.events+v.events))
    x = x.get()
    self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - (self.FT(self.model.execute_forward_2D(x,0)[:,None,None,...]*self.Coils)))**2
            +self.irgn_par["gamma"]*np.sum(np.abs(grad.get()-self.v))
            +self.irgn_par["gamma"]*(2)*np.sum(np.abs(sym_grad.get()))
            +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2)


    print('Norm M0 grad: %f  norm T1 grad: %f' %(np.linalg.norm(grad.get()[0,...]),
                                                  np.linalg.norm(grad.get()[1,...])))
    scale = np.linalg.norm(grad.get()[0,...])/np.linalg.norm(grad.get()[1,...])
    if scale == 0 or not np.isfinite(scale):
      self.ratio = self.ratio
    else:
      self.ratio *= scale
    return np.squeeze(x)


  def execute_2D(self):
      self.NSlice=1
      self.FT = self.NUFFT.fwd_NUFFT
      self.FTH = self.NUFFT.adj_NUFFT

      gamma = self.irgn_par["gamma"]
      delta = self.irgn_par["delta"]

      self.result = np.zeros((self.irgn_par["max_GN_it"],self.unknowns_TGV+self.unknowns_H1,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
      result = np.copy(self.model.guess)
      for islice in range(self.par.NSlice):
        self.irgn_par["gamma"] = gamma
        self.irgn_par["delta"] = delta
        self.Coils = np.array(np.squeeze(self.par.C[:,islice,:,:]),order='C')[:,None,...]
        self.coil_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Coils.data)
        self.conjCoils = np.conj(self.Coils)
        self.v = np.zeros(([self.unknowns_TGV,self.NSlice,self.par.dimX,self.par.dimY,4]),dtype=DTYPE)
        self.r = np.zeros(([self.NScan,self.NC,self.NSlice,self.Nproj,self.N]),dtype=DTYPE)
        self.z1 = np.zeros(([self.unknowns_TGV,self.NSlice,self.par.dimX,self.par.dimY,4]),dtype=DTYPE)
        self.z2 = np.zeros(([self.unknowns_TGV,self.NSlice,self.par.dimX,self.par.dimY,8]),dtype=DTYPE)
        iters = self.irgn_par["start_iters"]
        for i in range(self.irgn_par["max_GN_it"]):
          start = time.time()
          self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_2D(result[:,islice,:,:],islice))

          scale = np.linalg.norm(np.abs(self.grad_x_2D[0,...]))/np.linalg.norm(np.abs(self.grad_x_2D[1,...]))

          for j in range(len(self.model.constraints)-1):
            self.model.constraints[j+1].update(scale)

          result[1,islice,:,:] = result[1,islice,:,:]*self.model.T1_sc
          self.model.T1_sc = self.model.T1_sc*(scale)
          result[1,islice,:,:] = result[1,islice,:,:]/self.model.T1_sc
          self.step_val = self.model.execute_forward_2D(result[:,islice,:,:],islice)[:,None,...]
          self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_2D(result[:,islice,:,:],islice).astype(DTYPE))[:,:,None,...]
          self.grad_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grad_x_2D.data)
          self.conj_grad_x_2D = np.conj(self.grad_x_2D)


          result[:,islice,:,:] = self.irgn_solve_2D(result[:,islice,:,:], iters, self.data[:,:,islice,:,:])
          self.result[i,:,islice,:,:] = result[:,islice,:,:]

          iters = np.fmin(iters*2,self.irgn_par["max_iters"])
          self.irgn_par["gamma"] = np.maximum(self.irgn_par["gamma"]*0.5,self.irgn_par["gamma_min"])
          self.irgn_par["delta"] = np.minimum(self.irgn_par["delta"]*self.irgn_par["delta_inc"], self.irgn_par["delta_max"])

          end = time.time()-start
          print("GN-Iter: %d  Elapsed time: %f seconds" %(i,end))
          print("-"*80)
          if np.abs(self.fval_min-self.fval) < self.irgn_par["lambd"]*self.irgn_par["tol"]:
            print("Terminated at GN-iteration %d because the energy decrease was less than %.3e"%(i,np.abs(self.fval_min-       self.fval)/self.irgn_par["lambd"]))
            return
          self.fval_min = np.minimum(self.fval,self.fval_min)


  def eval_fwd(self,y,x,wait_for=[]):

    return self.prg.operator_fwd(self.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 y.data, x.data, self.coil_buf, self.grad_buf,
                                 np.int32(self.NC), np.int32(self.NSlice), np.int32(self.NScan), np.int32(self.unknowns),
                                 wait_for=wait_for)
  def operator_forward(self, x):

    self.tmp_result.add_event(self.eval_fwd(self.tmp_result,x,wait_for=self.tmp_result.events+x.events))
    self.tmp_sino.add_event(self.NUFFT.fwd_NUFFT(self.tmp_sino,self.tmp_result))
    return  self.tmp_sino

  def operator_forward_full(self, out, x, wait_for=[]):
    self.tmp_result.add_event(self.eval_fwd(self.tmp_result,x,wait_for=self.tmp_result.events+x.events))
    return  self.NUFFT.fwd_NUFFT(out,self.tmp_result,wait_for=wait_for+self.tmp_result.events)

  def operator_adjoint_full(self, out, x,z, wait_for=[]):

    self.tmp_img.add_event(self.NUFFT.adj_NUFFT(self.tmp_img,x,wait_for=wait_for+x.events))

    return self.prg.update_Kyk1(self.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 out.data, self.tmp_img.data, self.coil_buf, self.grad_buf, z.data, np.int32(self.NC),
                                 np.int32(self.NSlice),  np.int32(self.NScan), self.ukscale.data,
                                 np.float32(np.amax(self.ukscale.get())),self.ratio.data, np.int32(self.unknowns),
                                 wait_for=self.tmp_img.events+out.events+z.events+wait_for)

  def operator_adjoint(self, out, x, wait_for=[]):

    self.tmp_img.add_event(self.NUFFT.adj_NUFFT(self.tmp_img,x,wait_for=wait_for+x.events))

    return self.prg.operator_ad(out.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 out.data, self.tmp_img.data, self.coil_buf, self.grad_buf,np.int32(self.NC),
                                 np.int32(self.NSlice),  np.int32(self.NScan), np.int32(self.unknowns),
                                 wait_for=wait_for+self.tmp_img.events+out.events)


  def eval_adj(self,x,y,wait_for=[]):

    return self.prg.operator_ad(x.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 x.data, y.data, self.coil_buf, self.grad_buf,np.int32(self.NC),
                                 np.int32(self.NSlice),  np.int32(self.NScan), np.int32(self.unknowns),
                                 wait_for=wait_for)



  def eval_const(self):
    num_const = (len(self.model.constraints))
    self.min_const = np.zeros((num_const),dtype=np.float32)
    self.max_const = np.zeros((num_const),dtype=np.float32)
    self.real_const = np.zeros((num_const),dtype=np.int32)
    for j in range(num_const):
        self.min_const[j] = np.float32(self.model.constraints[j].min)
        self.max_const[j] = np.float32(self.model.constraints[j].max)
        self.real_const[j] = np.int32(self.model.constraints[j].real)
    self.min_const = clarray.to_device(self.queue, self.min_const)
    self.max_const = clarray.to_device(self.queue, self.max_const)
    self.real_const = clarray.to_device(self.queue, self.real_const)
#    print(x.shape[-3:])

#    x.add_event(self.prg.box_con(x.queue, x.shape[-2:],None,
#                                 x.data, min_const.data, max_const.data, real_const.data,
#                                 np.float32(num_const),
#                                 wait_for=wait_for))


  def tgv_solve_2D(self, x,res, iters):
    alpha = self.irgn_par["gamma"]
    beta = self.irgn_par["gamma"]*2

    L = np.float32(0.5*(18.0 + np.sqrt(33)))
    print('L: %f'%(L))


    tau = np.float32(1/np.sqrt(L))
    tau_new =np.float32(0)

    self.set_scale(x)
    x = clarray.to_device(self.queue,x)
    xk = x.copy()
    x_new = clarray.zeros_like(x)

    r = clarray.to_device(self.queue,self.r)#np.zeros_like(res,dtype=DTYPE)
    z1 = clarray.to_device(self.queue,self.z1)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z1_new = clarray.zeros_like(z1)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2 = clarray.to_device(self.queue,self.z2)#np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2_new = clarray.zeros_like(z2)
    v = clarray.to_device(self.queue,self.v)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    res = clarray.to_device(self.queue, res)


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

    Kyk1 = clarray.zeros_like(x)
    Kyk1_new = clarray.zeros_like(x)
    Kyk2 = clarray.zeros_like(z1)
    Kyk2_new = clarray.zeros_like(z1)
    gradx = clarray.zeros_like(z1)
    gradx_xold = clarray.zeros_like(z1)
    symgrad_v = clarray.zeros_like(z2)
    symgrad_v_vold = clarray.zeros_like(z2)

    Axold = clarray.zeros_like(res)
    Ax = clarray.zeros_like(res)

    Axold = self.operator_forward(x)
    Kyk1.add_event(self.operator_adjoint_full(Kyk1,r,z1))
    Kyk2.add_event(self.update_Kyk2(Kyk2,z2,z1))

    for i in range(iters):
      x_new.add_event(self.update_primal(x_new,x,Kyk1,xk,tau,delta))


      v_new = (v-tau*Kyk2).astype(DTYPE)

      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))

      beta_line = beta_new


      gradx.add_event(self.f_grad(gradx,x_new,wait_for=gradx.events+x_new.events))
      gradx_xold.add_event(self.f_grad(gradx_xold,x,wait_for=gradx_xold.events+x.events))

      symgrad_v.add_event(self.sym_grad(symgrad_v,v_new,wait_for=symgrad_v.events+v_new.events))
      symgrad_v_vold.add_event(self.sym_grad(symgrad_v_vold,v,wait_for=symgrad_v_vold.events+v.events))


      Ax = self.operator_forward(x_new)

      while True:

        theta_line = tau_new/tau
        z1_new.add_event(self.update_z1(z1_new,z1,gradx,gradx_xold,v_new,v, beta_line*tau_new, theta_line, alpha))
        z2_new.add_event(self.update_z2(z2_new,z2,symgrad_v,symgrad_v_vold,beta_line*tau_new,theta_line,beta))

        r_new = ( r  + beta_line*tau_new*((1+theta_line)*Ax-theta_line*Axold - res))/(1+beta_line*tau_new/self.irgn_par["lambd"])



        Kyk1_new.add_event(self.operator_adjoint_full(Kyk1_new,r_new,z1_new))
        Kyk2_new.add_event(self.update_Kyk2(Kyk2_new,z2_new,z1_new))

        ynorm = ((clarray.vdot(r_new-r,r_new-r)+clarray.vdot(z1_new-z1,z1_new-z1)+clarray.vdot(z2_new-z2,z2_new-z2))**(1/2)).real
        lhs = np.sqrt(beta_line)*tau_new*((clarray.vdot(Kyk1_new-Kyk1,Kyk1_new-Kyk1)+clarray.vdot(Kyk2_new-Kyk2,Kyk2_new-Kyk2))**(1/2)).real
        if lhs <= ynorm*delta_line:
            break
        else:
          tau_new = tau_new*mu_line

      (Kyk1, Kyk1_new, Kyk2, Kyk2_new, Axold, Ax, z1, z1_new, z2, z2_new, r, r_new) =\
      (Kyk1_new, Kyk1, Kyk2_new, Kyk2, Ax, Axold, z1_new, z1, z2_new, z2, r_new, r)
      tau =  (tau_new)

      if not np.mod(i,20):

        self.model.plot_unknowns(x_new.get()[:,0,...],True)
        primal_new= (self.irgn_par["lambd"]/2*clarray.vdot(Axold-res,Axold-res)+alpha*clarray.sum(abs((gradx[:self.unknowns_TGV]-v))) +beta*clarray.sum(abs(symgrad_v)) + 1/(2*delta)*clarray.vdot(x_new-xk,x_new-xk)).real

        dual = (-delta/2*clarray.vdot(-Kyk1,-Kyk1)- clarray.vdot(xk,(-Kyk1)) + clarray.sum(Kyk2)
                  - 1/(2*self.irgn_par["lambd"])*clarray.vdot(r,r) - clarray.vdot(res,r)).real

        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<self.irgn_par["lambd"]*self.irgn_par["tol"]:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,abs(primal-primal_new).get()/(self.irgn_par["lambd"]*self.NSlice)))
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          return x_new.get()
        if (gap > gap_min*self.irgn_par["stag"]) and i>1:
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x.get()
        if np.abs(gap - gap_min)<self.irgn_par["lambd"]*self.irgn_par["tol"] and i>1:
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,abs(gap - gap_min).get()/self.irgn_par["lambd"]))
          return x_new.get()
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal.get()/self.irgn_par["lambd"],dual.get()/self.irgn_par["lambd"],gap.get()/self.irgn_par["lambd"]))
#        print("Norm of primal gradient: %.3e"%(np.linalg.norm(Kyk1)+np.linalg.norm(Kyk2)))
#        print("Norm of dual gradient: %.3e"%(np.linalg.norm(tmp)+np.linalg.norm(gradx[:self.unknowns_TGV] + theta_line*gradx_xold[:self.unknowns_TGV]
#                                          - v_new - theta_line*v_vold)+np.linalg.norm( symgrad_v + theta_line*symgrad_v_vold)))

      (x, x_new) = (x_new, x)
      (v, v_new) = (v_new, v)
    self.v = v.get()
    self.r = r.get()
    self.z1 = z1.get()
    self.z2 = z2.get()
    return x.get()



  def f_grad(self,grad, u, wait_for=[]):
    return self.prg.gradient(self.queue, u.shape[1:], None, grad.data, u.data,
                np.int32(self.unknowns),
                self.ukscale.data,  np.float32(np.amax(self.ukscale.get())),self.ratio.data,
                wait_for=grad.events + u.events + wait_for)

  def bdiv(self,div, u, wait_for=[]):
    return self.prg.divergence(div.queue, u.shape[1:-1], None, div.data, u.data,
                np.int32(self.unknowns),
                self.ukscale.data, np.float32(np.amax(self.ukscale.get())),self.ratio.data,
                wait_for=div.events + u.events + wait_for)

  def sym_grad(self,sym, w, wait_for=[]):
    return self.prg.sym_grad(self.queue, w.shape[1:-1], None, sym.data, w.data,
                np.int32(self.unknowns),
                wait_for=sym.events + w.events + wait_for)

  def sym_bdiv(self,div, u, wait_for=[]):
    return self.prg.sym_divergence(self.queue, u.shape[1:-1], None, div.data, u.data,
                np.int32(self.unknowns),
                wait_for=div.events + u.events + wait_for)
  def update_Kyk2(self,div, u, z, wait_for=[]):
    return self.prg.update_Kyk2(self.queue, u.shape[1:-1], None, div.data, u.data, z.data,
                np.int32(self.unknowns),
                wait_for=div.events + u.events + z.events+wait_for)

  def update_primal(self, x_new, x, Kyk, xk, tau, delta, wait_for=[]):
    return self.prg.update_primal(self.queue, x.shape[1:], None, x_new.data, x.data, Kyk.data, xk.data, np.float32(tau),
                                  np.float32(tau/delta), np.float32(1/(1+tau/delta)), self.min_const.data, self.max_const.data,
                                  self.real_const.data, np.int32(self.unknowns),
                                  wait_for=x_new.events + x.events + Kyk.events+ xk.events+wait_for
                                  )
  def update_z1(self, z_new, z, gx, gx_, vx, vx_, sigma, theta, alpha, wait_for=[]):
    return self.prg.update_z1(self.queue, z.shape[1:-1], None, z_new.data, z.data, gx.data, gx_.data, vx.data, vx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/alpha), np.int32(self.unknowns),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ vx.events+ vx_.events+wait_for
                                  )
  def update_z1_tv(self, z_new, z, gx, gx_, sigma, theta, alpha, wait_for=[]):
    return self.prg.update_z1_tv(self.queue, z.shape[1:-1], None, z_new.data, z.data, gx.data, gx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/alpha), np.int32(self.unknowns),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ wait_for
                                  )
  def update_z2(self, z_new, z, gx, gx_, sigma, theta, beta, wait_for=[]):
    return self.prg.update_z2(self.queue, z.shape[1:-1], None, z_new.data, z.data, gx.data, gx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/beta),  np.int32(self.unknowns),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ wait_for
                                  )
  def update_r(self, r_new, r, A, A_, res, sigma, theta, lambd, wait_for=[]):
    return self.prg.update_r(self.queue, (self.NScan*self.NC*self.NSlice,self.Nproj,self.N), None, r_new.data, r.data, A.data, A_.data, res.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/(1+sigma/lambd)),
                                  wait_for= r_new.events + r.events + A.events+ A_.events+ wait_for
                                  )
  def update_v(self, v_new, v, Kyk2, tau, wait_for=[]):
    return self.prg.update_v(self.queue, (self.unknowns*self.NSlice,self.dimY,self.dimX), None,
                             v_new.data, v.data, Kyk2.data, np.float32(tau),
                                  wait_for= v_new.events + v.events + Kyk2.events+ wait_for
                                  )
  def update_primal_explicit(self, x_new, x, Kyk, xk, ATd, tau, delta, lambd,wait_for=[]):
    return self.prg.update_primal_explicit(self.queue, x.shape[1:], None, x_new.data, x.data, Kyk.data, xk.data, ATd.data, np.float32(tau),
                                  np.float32(1/delta), np.float32(lambd), self.min_const.data, self.max_const.data,
                                  self.real_const.data, np.int32(self.unknowns),
                                  wait_for=x_new.events + x.events + Kyk.events+ xk.events+ATd.events+wait_for
                                  )
################################################################################
### Scale before gradient ######################################################
################################################################################
  def set_scale(self,x):
    for j in range(x.shape[0]):
      self.ukscale[j] = np.linalg.norm(x[j,...])
      print('scale %f at uk %i' %(self.ukscale[j].get(),j))


################################################################################
### Start a 3D Reconstruction, set TV to True to perform TV instead of TGV######
### Precompute Model and Gradient values for xk ################################
### Call inner optimization ####################################################
### input: bool to switch between TV (1) and TGV (0) regularization ############
### output: optimal value of x #################################################
################################################################################
  def execute_3D(self, TV=0):
   self.FT = self.NUFFT.fwd_NUFFT# self.nFT
   self.FTH = self.NUFFT.adj_NUFFT#self.nFTH
   iters = self.irgn_par["start_iters"]


   self.r = np.zeros_like(self.data,dtype=DTYPE)
   self.z1 = np.zeros(([self.unknowns,self.NSlice,self.par.dimX,self.par.dimY,4]),dtype=DTYPE)


   self.result = np.zeros((self.irgn_par["max_GN_it"]+1,self.unknowns,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
   self.result[0,:,:,:,:] = np.copy(self.model.guess)

   self.Coils3D = (self.par.C)
   self.conjCoils3D = np.conj(self.Coils3D)
   result = np.copy(self.model.guess)

   if TV==1:
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
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))
        self.grad_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grad_x.data)
        self.conj_grad_x = np.nan_to_num(np.conj(self.grad_x))

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

   elif TV==0:
      self.v = np.zeros(([self.unknowns,self.NSlice,self.par.dimX,self.par.dimY,4]),dtype=DTYPE)
      self.z2 = np.zeros(([self.unknowns,self.NSlice,self.par.dimX,self.par.dimY,8]),dtype=DTYPE)
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
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))
        self.grad_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grad_x.data)
        self.conj_grad_x = np.nan_to_num(np.conj(self.grad_x))

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
   else:
      self.v = np.zeros(([self.unknowns,self.NSlice,self.par.dimX,self.par.dimY,4]),dtype=DTYPE)
      self.z2 = np.zeros(([self.unknowns,self.NSlice,self.par.dimX,self.par.dimY,8]),dtype=DTYPE)
      for i in range(self.irgn_par["max_GN_it"]):
        start = time.time()
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))


        for j in range(len(self.model.constraints)-1):
          scale = np.linalg.norm(np.abs(self.grad_x[0,...]))/np.linalg.norm(np.abs(self.grad_x[j+1,...]))
          self.model.constraints[j+1].update(scale)
          result[j+1,...] = result[j+1,...]*self.model.T1_sc
          self.model.T1_sc = self.model.T1_sc*(scale)
          result[j+1,...] = result[j+1,...]/self.model.T1_sc

        self.step_val = np.nan_to_num(self.model.execute_forward_3D(result))
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))
        self.grad_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grad_x.data)
        self.conj_grad_x = np.nan_to_num(np.conj(self.grad_x))

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
    x = clarray.to_device(self.queue,np.require(x,requirements="C"))
    b = clarray.zeros(self.queue, data.shape,dtype=DTYPE)
    self.FT(b,clarray.to_device(self.queue,self.step_val[:,None,...]*self.Coils3D)).wait()

    res = data - b.get() + self.operator_forward(x).get()


    if TV==1:
      x = self.tv_solve_3D(x.get(),res,iters)
      x = clarray.to_device(self.queue,x)
      grad = clarray.to_device(self.queue,np.zeros_like(self.z1))
      grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))
      x = x.get()
      self.FT(b,clarray.to_device(self.queue,self.model.execute_forward_3D(x)[:,None,...]*self.Coils3D)).wait()
      self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - b.get())**2
              +self.irgn_par["gamma"]*np.sum(np.abs(grad.get()))
              +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2)
      for knt in range(self.unknowns):
         print('Norm uk%i grad: %f' %(knt,np.linalg.norm(grad.get()[knt,...])))
         scale = np.linalg.norm(grad.get()[knt,...])/np.linalg.norm(grad.get()[1,...])
         if scale == 0 or not np.isfinite(scale):
           pass
         else:
           print("Scale: %f" %scale)
           self.ratio[knt] *= scale
    elif TV==0:
       x = self.tgv_solve_3D(x.get(),res,iters)
       x = clarray.to_device(self.queue,x)
       v = clarray.to_device(self.queue,self.v)
       grad = clarray.to_device(self.queue,np.zeros_like(self.z1))
       sym_grad = clarray.to_device(self.queue,np.zeros_like(self.z2))
       grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))
       sym_grad.add_event(self.sym_grad(sym_grad,v,wait_for=sym_grad.events+v.events))
       x = x.get()
       self.FT(b,clarray.to_device(self.queue,self.model.execute_forward_3D(x)[:,None,...]*self.Coils3D)).wait()
       self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - b.get())**2
              +self.irgn_par["gamma"]*np.sum(np.abs(grad.get()-self.v))
              +self.irgn_par["gamma"]*(2)*np.sum(np.abs(sym_grad.get()))
              +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2)
       for knt in range(self.unknowns):
         print('Norm uk%i grad: %f' %(knt,np.linalg.norm(grad.get()[knt,...])))
         scale = np.linalg.norm(grad.get()[knt,...])/np.linalg.norm(grad.get()[1,...])
         if scale == 0 or not np.isfinite(scale):
           pass
         else:
           print("Scale: %f" %scale)
           self.ratio[knt] *= scale
    else:
       x = self.tgv_solve_3D_explicit(x.get(),res,iters)
       x = clarray.to_device(self.queue,x)
       v = clarray.to_device(self.queue,self.v)
       grad = clarray.to_device(self.queue,np.zeros_like(self.z1))
       sym_grad = clarray.to_device(self.queue,np.zeros_like(self.z2))
       grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))
       sym_grad.add_event(self.sym_grad(sym_grad,v,wait_for=sym_grad.events+v.events))
       x = x.get()
       self.FT(b,clarray.to_device(self.queue,self.model.execute_forward_3D(x)[:,None,...]*self.Coils3D)).wait()
       self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - b.get())**2
              +self.irgn_par["gamma"]*np.sum(np.abs(grad.get()-self.v))
              +self.irgn_par["gamma"]*(2)*np.sum(np.abs(sym_grad.get()))
              +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2)
       for knt in range(self.unknowns):
         print('Norm uk%i grad: %f' %(knt,np.linalg.norm(grad.get()[knt,...])))
         scale = np.linalg.norm(grad.get()[knt,...])/np.linalg.norm(grad.get()[1,...])
         if scale == 0 or not np.isfinite(scale):
           pass
         else:
           print("Scale: %f" %scale)
           self.ratio[knt] *= scale

    print("-"*80)
    print ("Function value after GN-Step: %f" %(self.fval/(self.irgn_par["lambd"]*self.NSlice)))

    return x

  def tgv_solve_3D(self, x,res, iters):
    alpha = self.irgn_par["gamma"]
    beta = self.irgn_par["gamma"]*2

    L = np.float32(0.5*(18.0 + np.sqrt(33)))
    print('L: %f'%(L))


    tau = np.float32(1/np.sqrt(L))
    tau_new =np.float32(0)

    self.set_scale(x)
    x = clarray.to_device(self.queue,x)
    xk = x.copy()
    x_new = clarray.zeros_like(x)

    r = clarray.to_device(self.queue,self.r)#np.zeros_like(res,dtype=DTYPE)
    r_new = clarray.zeros_like(r)
    z1 = clarray.to_device(self.queue,self.z1)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z1_new = clarray.zeros_like(z1)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2 = clarray.to_device(self.queue,self.z2)#np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2_new = clarray.zeros_like(z2)
    v = clarray.to_device(self.queue,self.v)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    v_new = clarray.zeros_like(v)
    res = clarray.to_device(self.queue, res.astype(DTYPE))


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

    Kyk1 = clarray.zeros_like(x)
    Kyk1_new = clarray.zeros_like(x)
    Kyk2 = clarray.zeros_like(z1)
    Kyk2_new = clarray.zeros_like(z1)
    gradx = clarray.zeros_like(z1)
    gradx_xold = clarray.zeros_like(z1)
    symgrad_v = clarray.zeros_like(z2)
    symgrad_v_vold = clarray.zeros_like(z2)

    Axold = clarray.zeros_like(res)
    Ax = clarray.zeros_like(res)

    Axold.add_event(self.operator_forward_full(Axold,x))   #### Q1
    Kyk1.add_event(self.operator_adjoint_full(Kyk1,r,z1))    #### Q2
    Kyk2.add_event(self.update_Kyk2(Kyk2,z2,z1)) #### Q1

    for i in range(iters):
      x_new.add_event(self.update_primal(x_new,x,Kyk1,xk,tau,delta)) ### Q2
#      x_new = x_new.get()
#      x_new[-1,...] = np.median(x_new[-1,...])*np.ones_like(x_new[-1,...])
#      x_new = clarray.to_device(self.queue,x_new)
      v_new.add_event(self.update_v(v_new,v,Kyk2,tau)) ### Q1

      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))

      beta_line = beta_new


      gradx.add_event(self.f_grad(gradx,x_new))  ### Q1
      gradx_xold.add_event(self.f_grad(gradx_xold,x))  ### Q1
      symgrad_v.add_event(self.sym_grad(symgrad_v,v_new))  ### Q2
      symgrad_v_vold.add_event(self.sym_grad(symgrad_v_vold,v))  ### Q2
      Ax.add_event(self.operator_forward_full(Ax,x_new))  ### Q1

      while True:

        theta_line = tau_new/tau
        z1_new.add_event(self.update_z1(z1_new,z1,gradx,gradx_xold,v_new,v, beta_line*tau_new, theta_line, alpha)) ### Q2
        z2_new.add_event(self.update_z2(z2_new,z2,symgrad_v,symgrad_v_vold,beta_line*tau_new,theta_line,beta)) ### Q1
        r_new.add_event(self.update_r(r_new,r,Ax,Axold,res,beta_line*tau_new,theta_line,self.irgn_par["lambd"])) ### Q1

        Kyk1_new.add_event(self.operator_adjoint_full(Kyk1_new,r_new,z1_new))### Q2
        Kyk2_new.add_event(self.update_Kyk2(Kyk2_new,z2_new,z1_new))### Q1

        ynorm = ((clarray.vdot(r_new-r,r_new-r)+clarray.vdot(z1_new-z1,z1_new-z1)+clarray.vdot(z2_new-z2,z2_new-z2))**(1/2)).real
        lhs = np.sqrt(beta_line)*tau_new*((clarray.vdot(Kyk1_new-Kyk1,Kyk1_new-Kyk1)+clarray.vdot(Kyk2_new-Kyk2,Kyk2_new-Kyk2))**(1/2)).real
        if lhs <= ynorm*delta_line:
            break
        else:
          tau_new = tau_new*mu_line

      (Kyk1, Kyk1_new, Kyk2, Kyk2_new, Axold, Ax, z1, z1_new, z2, z2_new, r, r_new) =\
      (Kyk1_new, Kyk1, Kyk2_new, Kyk2, Ax, Axold, z1_new, z1, z2_new, z2, r_new, r)
      tau =  (tau_new)


      if not np.mod(i,10):

        self.model.plot_unknowns(x_new.get())
        primal_new= (self.irgn_par["lambd"]/2*clarray.vdot(Axold-res,Axold-res)+alpha*clarray.sum(abs((gradx[:self.unknowns_TGV]-v))) +beta*clarray.sum(abs(symgrad_v)) + 1/(2*delta)*clarray.vdot(x_new-xk,x_new-xk)).real

        dual = (-delta/2*clarray.vdot(-Kyk1,-Kyk1)- clarray.vdot(xk,(-Kyk1)) + clarray.sum(Kyk2)
                  - 1/(2*self.irgn_par["lambd"])*clarray.vdot(r,r) - clarray.vdot(res,r)).real

        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"]:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,abs(primal-primal_new).get()/(self.irgn_par["lambd"]*self.NSlice)))
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          return x_new.get()
        if (gap > gap_min*self.irgn_par["stag"]) and i>1:
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x.get()
        if np.abs(gap - gap_min)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"] and i>1:
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,abs(gap - gap_min).get()/(self.irgn_par["lambd"]*self.NSlice)))
          return x_new.get()
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        sys.stdout.write("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f \r"%(i,primal.get()/(self.irgn_par["lambd"]*self.NSlice),dual.get()/(self.irgn_par["lambd"]*self.NSlice),gap.get()/(self.irgn_par["lambd"]*self.NSlice)))
        sys.stdout.flush()

      (x, x_new) = (x_new, x)
      (v, v_new) = (v_new, v)

    self.v = v.get()
    self.r = r.get()
    self.z1 = z1.get()
    self.z2 = z2.get()
    return x.get()

  def tv_solve_3D(self, x,res, iters):
    alpha = self.irgn_par["gamma"]

    L = np.float32(8)
    print('L: %f'%(L))


    tau = np.float32(1/np.sqrt(L))
    tau_new =np.float32(0)

    self.set_scale(x)
    x = clarray.to_device(self.queue,x)
    xk = x.copy()
    x_new = clarray.zeros_like(x)

    r = clarray.to_device(self.queue,self.r)#np.zeros_like(res,dtype=DTYPE)
    r_new  = clarray.zeros_like(r)
    z1 = clarray.to_device(self.queue,self.z1)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z1_new = clarray.zeros_like(z1)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    res = clarray.to_device(self.queue, res.astype(DTYPE))


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

    Kyk1 = clarray.zeros_like(x)
    Kyk1_new = clarray.zeros_like(x)


    gradx = clarray.zeros_like(z1)
    gradx_xold = clarray.zeros_like(z1)

    Axold = clarray.zeros_like(res)
    Ax = clarray.zeros_like(res)

    Axold.add_event(self.operator_forward_full(Axold,x))
    Kyk1.add_event(self.operator_adjoint_full(Kyk1,r,z1))


    for i in range(iters):

      x_new.add_event(self.update_primal(x_new,x,Kyk1,xk,tau,delta))


      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new


      gradx.add_event(self.f_grad(gradx,x_new,wait_for=gradx.events+x_new.events))
      gradx_xold.add_event(self.f_grad(gradx_xold,x,wait_for=gradx_xold.events+x.events))


      Ax.add_event(self.operator_forward_full(Ax,x_new))


      while True:

        theta_line = tau_new/tau
        z1_new.add_event(self.update_z1_tv(z1_new,z1,gradx,gradx_xold, beta_line*tau_new, theta_line,alpha))
        r_new.add_event(self.update_r(r_new,r,Ax,Axold,res,beta_line*tau_new,theta_line,self.irgn_par["lambd"]))

        Kyk1_new.add_event(self.operator_adjoint_full(Kyk1_new,r_new,z1_new))

        ynorm = ((clarray.vdot(r_new-r,r_new-r)+clarray.vdot(z1_new-z1,z1_new-z1))**(1/2)).real
        lhs = np.sqrt(beta_line)*tau_new*((clarray.vdot(Kyk1_new-Kyk1,Kyk1_new-Kyk1))**(1/2)).real
        if lhs <= ynorm*delta_line:
            break
        else:
          tau_new = tau_new*mu_line

      (Kyk1, Kyk1_new,  Axold, Ax, z1, z1_new, r, r_new) =\
      (Kyk1_new, Kyk1,  Ax, Axold, z1_new, z1, r_new, r)
      tau =  (tau_new)


      if not np.mod(i,1):
        self.model.plot_unknowns(x_new.get())
        primal_new= (self.irgn_par["lambd"]/2*clarray.vdot(Axold-res,Axold-res)+alpha*clarray.sum(abs((gradx[:self.unknowns_TGV]))) + 1/(2*delta)*clarray.vdot(x_new-xk,x_new-xk)).real

        dual = (-delta/2*clarray.vdot(-Kyk1,-Kyk1)- clarray.vdot(xk,(-Kyk1))
                  - 1/(2*self.irgn_par["lambd"])*clarray.vdot(r,r) - clarray.vdot(res,r)).real

        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"]:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,abs(primal-primal_new).get()/(self.irgn_par["lambd"]*self.NSlice)))
          self.r = r.get()
          self.z1 = z1.get()
          return x_new.get()
        if (gap > gap_min*self.irgn_par["stag"]) and i>1:
          self.r = r.get()
          self.z1 = z1.get()
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x_new.get()
        if np.abs(gap - gap_min)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"] and i>1:
          self.r = r.get()
          self.z1 = z1.get()
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,abs(gap - gap_min).get()/(self.irgn_par["lambd"]*self.NSlice)))
          return x_new.get()
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        sys.stdout.write("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f \r"%(i,primal.get()/(self.irgn_par["lambd"]*self.NSlice),dual.get()/(self.irgn_par["lambd"]*self.NSlice),gap.get()/(self.irgn_par["lambd"]*self.NSlice)))
        sys.stdout.flush()


      (x, x_new) = (x_new, x)
    self.r = r.get()
    self.z1 = z1.get()
    return x.get()


  def tgv_solve_3D_explicit(self, x,res, iters):
    alpha = self.irgn_par["gamma"]
    beta = self.irgn_par["gamma"]*2

    L = np.float32(0.5*(18.0 + np.sqrt(33)))
    print('L: %f'%(L))


    tau = np.float32(1/(L*1e1))
    tau_new =np.float32(0)

    self.set_scale(x)
    x = clarray.to_device(self.queue,x)
    xk = x.copy()
    x_new = clarray.zeros_like(x)
    ATd = clarray.zeros_like(x)

    z1 = clarray.to_device(self.queue,self.z1)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z1_new = clarray.zeros_like(z1)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2 = clarray.to_device(self.queue,self.z2)#np.zeros(([self.unknowns,3,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    z2_new = clarray.zeros_like(z2)
    v = clarray.to_device(self.queue,self.v)#np.zeros(([self.unknowns,2,self.par.dimX,self.par.dimY]),dtype=DTYPE)
    v_new = clarray.zeros_like(v)
    res = clarray.to_device(self.queue, res.astype(DTYPE))


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

    Kyk1 = clarray.zeros_like(x)
    Kyk1_new = clarray.zeros_like(x)
    Kyk2 = clarray.zeros_like(z1)
    Kyk2_new = clarray.zeros_like(z1)
    gradx = clarray.zeros_like(z1)
    gradx_xold = clarray.zeros_like(z1)
    symgrad_v = clarray.zeros_like(z2)
    symgrad_v_vold = clarray.zeros_like(z2)

    AT = clarray.zeros_like(res)

    AT.add_event(self.operator_forward_full(AT,x))   #### Q1
    ATd.add_event(self.operator_adjoint(ATd,res))


    Kyk1.add_event(self.bdiv(Kyk1,z1))    #### Q2
    Kyk2.add_event(self.update_Kyk2(Kyk2,z2,z1)) #### Q1

    for i in range(iters):

      x_new.add_event(self.operator_adjoint(x_new,AT))
      x_new.add_event(self.update_primal_explicit(x_new,x,Kyk1,xk,ATd,tau,delta,self.irgn_par["lambd"])) ### Q2
      v_new.add_event(self.update_v(v_new,v,Kyk2,tau)) ### Q1

      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))

      beta_line = beta_new


      gradx.add_event(self.f_grad(gradx,x_new))  ### Q1
      gradx_xold.add_event(self.f_grad(gradx_xold,x))  ### Q1
      symgrad_v.add_event(self.sym_grad(symgrad_v,v_new))  ### Q2
      symgrad_v_vold.add_event(self.sym_grad(symgrad_v_vold,v))  ### Q2

      AT.add_event(self.operator_forward_full(AT,x_new))   #### Q1

      while True:

        theta_line = tau_new/tau
        z1_new.add_event(self.update_z1(z1_new,z1,gradx,gradx_xold,v_new,v, beta_line*tau_new, theta_line, alpha)) ### Q2
        z2_new.add_event(self.update_z2(z2_new,z2,symgrad_v,symgrad_v_vold,beta_line*tau_new,theta_line,beta)) ### Q1

        Kyk1_new.add_event(self.bdiv(Kyk1_new,z1_new))
        Kyk2_new.add_event(self.update_Kyk2(Kyk2_new,z2_new,z1_new))### Q1

        ynorm = ((clarray.vdot(z1_new-z1,z1_new-z1)+clarray.vdot(z2_new-z2,z2_new-z2))**(1/2)).real
        lhs = 1e2*np.sqrt(beta_line)*tau_new*((clarray.vdot(Kyk1_new-Kyk1,Kyk1_new-Kyk1)+clarray.vdot(Kyk2_new-Kyk2,Kyk2_new-Kyk2))**(1/2)).real
        if lhs <= ynorm*delta_line:
            break
        else:
          tau_new = tau_new*mu_line

      (Kyk1, Kyk1_new, Kyk2, Kyk2_new, z1, z1_new, z2, z2_new) =\
      (Kyk1_new, Kyk1, Kyk2_new, Kyk2, z1_new, z1, z2_new, z2)
      tau =  (tau_new)


      if not np.mod(i,50):

        self.model.plot_unknowns(x_new.get())
        primal_new= (self.irgn_par["lambd"]/2*clarray.vdot(AT-res,AT-res)+alpha*clarray.sum(abs((gradx[:self.unknowns_TGV]-v))) +beta*clarray.sum(abs(symgrad_v)) + 1/(2*delta)*clarray.vdot(x_new-xk,x_new-xk)).real

        dual = (-delta/2*clarray.vdot(-Kyk1,-Kyk1)- clarray.vdot(xk,(-Kyk1)) + clarray.sum(Kyk2)
                  ).real

        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"]:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,abs(primal-primal_new).get()/(self.irgn_par["lambd"]*self.NSlice)))
          self.v = v_new.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          return x_new.get()
        if (gap > gap_min*self.irgn_par["stag"]) and i>1:
          self.v = v_new.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x.get()
        if np.abs(gap - gap_min)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"] and i>1:
          self.v = v_new.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,abs(gap - gap_min).get()/(self.irgn_par["lambd"]*self.NSlice)))
          return x_new.get()
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal.get()/(self.irgn_par["lambd"]*self.NSlice),dual.get()/(self.irgn_par["lambd"]*self.NSlice),gap.get()/(self.irgn_par["lambd"]*self.NSlice)))

      (x, x_new) = (x_new, x)
      (v, v_new) = (v_new, v)

    self.v = v.get()
    self.z1 = z1.get()
    self.z2 = z2.get()
    return x.get()



