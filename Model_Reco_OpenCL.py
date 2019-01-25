
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

import pyopencl as cl
import pyopencl.array as clarray
import sys

class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build("-cl-mad-enable -cl-fast-relaxed-math")
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel


class Model_Reco:
  def __init__(self,par,ctx,queue,trafo=1,ksp_encoding='2D',imagespace=False,SMS=0):

    self.C = par["C"]
    self.traj = par["traj"]
    self.unknowns_TGV = par["unknowns_TGV"]
    self.unknowns_H1 = par["unknowns_H1"]
    self.unknowns = par["unknowns"]
    self.NSlice = par["NSlice"]
    self.NScan = par["NScan"]
    self.dimX = par["dimX"]
    self.dimY = par["dimY"]
    self.NC = par["NC"]
    self.fval_min = 0
    self.fval = 0
    self.ctx = ctx[0]
    self.queue = queue[0]
    self.ratio = clarray.to_device(self.queue,(np.ones(self.unknowns)).astype(dtype=DTYPE_real))
    self.ukscale =  clarray.to_device(self.queue,np.ones(self.unknowns,dtype=DTYPE_real))
    self.gn_res = []
    self.N = par["N"]
    self.Nproj = par["Nproj"]
    self.dz = 1
    if imagespace:
      self.operator_forward = self.operator_forward_imagespace
      self.operator_adjoint = self.operator_adjoint_imagespace
      self.operator_forward_full = self.operator_forward_full_imagespace
      self.operator_adjoint_full = self.operator_adjoint_full_imagespace
      self.coil_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.C.data)
      self.tmp_result = clarray.zeros(self.queue,(self.NScan,self.NSlice,self.dimY,self.dimX),DTYPE,"C")
    else:
      self.coil_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.C.data)
      if trafo:
        self.NUFFT = NUFFT.gridding(self.ctx,queue,4,2,par["N"],par["NScan"],(par["NScan"]*par["NC"]*par["NSlice"],par["N"],par["N"]),(1,2),par["traj"].astype(DTYPE),np.require(np.abs(par["dcf"]),DTYPE_real,requirements='C'),par["N"],10000,DTYPE,DTYPE_real,radial=trafo)
        self.tmp_sino = clarray.zeros(self.queue,(self.NScan,self.NC,self.NSlice,self.Nproj,self.N),DTYPE,"C")
        self.tmp_result = clarray.zeros(self.queue,(self.NScan,self.NC,self.NSlice,self.dimY,self.dimX),DTYPE,"C")
      else:
        if SMS:
          self.NUFFT = NUFFT.gridding(self.ctx,queue,4,2,par["N"],par["NScan"],(par["NScan"]*par["NC"]*par["NSlice"],par["N"],par["N"]),(1,2),par["traj"],par["dcf"],par["N"],1000,DTYPE,DTYPE_real,radial=trafo,mask=par['mask'])
          self.tmp_sino = clarray.zeros(self.queue,(self.NScan,self.NC,par["packs"],self.Nproj,self.N),DTYPE,"C")
          self.tmp_result = clarray.zeros(self.queue,(self.NScan,self.NC,self.NSlice,self.dimY,self.dimX),DTYPE,"C")
        else:
          self.NUFFT = NUFFT.gridding(self.ctx,queue,4,2,par["N"],par["NScan"],(par["NScan"]*par["NC"]*par["NSlice"],par["N"],par["N"]),(1,2),par["traj"],par["dcf"],par["N"],1000,DTYPE,DTYPE_real,radial=trafo,mask=par['mask'])
          self.tmp_sino = clarray.zeros(self.queue,(self.NScan,self.NC,self.NSlice,self.Nproj,self.N),DTYPE,"C")
          self.tmp_result = clarray.zeros(self.queue,(self.NScan,self.NC,self.NSlice,self.dimY,self.dimX),DTYPE,"C")
      self.operator_forward = self.operator_forward_kspace
      self.operator_adjoint = self.operator_adjoint_kspace
      self.operator_forward_full = self.operator_forward_full_kspace
      self.operator_adjoint_full = self.operator_adjoint_full_kspace




    self.prg = Program(self.ctx, r"""
__kernel void update_v(__global float8 *v,__global float8 *v_, __global float8 *Kyk2, const float tau) {
  size_t i = get_global_id(0);
  v[i] = v_[i]-tau*Kyk2[i];
}

__kernel void update_r(__global float2 *r, __global float2 *r_, __global float2 *A, __global float2 *A_, __global float2 *res,
                          const float sigma, const float theta, const float lambdainv) {
  size_t i = get_global_id(0);

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
     hypot(hypot(2.0f*hypot(z_new[i].s6,z_new[i].s7),2.0f*hypot(z_new[i].s8,z_new[i].s9)),2.0f*hypot(z_new[i].sa,z_new[i].sb))));
     i += NSl*Nx*Ny;
   }
  fac *= alphainv;
  i = k*Nx*Ny+Nx*y + x;
  for (int uk=0; uk<NUk; uk++)
  {
    if (fac > 1.0f) z_new[i] /=fac;
    i += NSl*Nx*Ny;
  }
}
__kernel void update_z1(__global float8 *z_new, __global float8 *z, __global float8 *gx,__global float8 *gx_,
                          __global float8 *vx,__global float8 *vx_, const float sigma, const float theta, const float alphainv,
                          const int NUk_tgv, const int NUk_H1, const float h1inv) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  float fac = 0.0f;

  for (int uk=0; uk<NUk_tgv; uk++)
  {
     z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]-(1+theta)*vx[i]+theta*vx_[i]);

     // reproject
     fac = hypot(fac,hypot(hypot(z_new[i].s0,z_new[i].s1), hypot(hypot(z_new[i].s2,z_new[i].s3),hypot(z_new[i].s4,z_new[i].s5))));
     i += NSl*Nx*Ny;
  }
  fac *= alphainv;
  i = k*Nx*Ny+Nx*y + x;
  for (int uk=0; uk<NUk_tgv; uk++)
  {
    if (fac > 1.0f) z_new[i] /=fac;
    i += NSl*Nx*Ny;
  }
  i = NSl*Nx*Ny*NUk_tgv+k*Nx*Ny+Nx*y + x;
  for (int uk=NUk_tgv; uk<(NUk_tgv+NUk_H1); uk++)
  {
    z_new[i] = (z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]))*h1inv;
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
     fac = hypot(fac,hypot(hypot(z_new[i].s0,z_new[i].s1), hypot(hypot(z_new[i].s2,z_new[i].s3),hypot(z_new[i].s4,z_new[i].s5))));
     i += NSl*Nx*Ny;
  }
  fac *= alphainv;
  i = k*Nx*Ny+Nx*y + x;
  for (int uk=0; uk<NUk; uk++)
  {
    if (fac > 1.0f) z_new[i] /=fac;
    i += NSl*Nx*Ny;
  }
}
__kernel void update_primal(__global float2 *u_new, __global float2 *u, __global float2 *Kyk,__global float2 *u_k, const float tau, const float tauinv, float div, __global float* min, __global float* max, __global int* real, __global int* pos_real, const int NUk) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;
  float norm = 0;
  int idx, idx2, idx3, idx4, idx5;
  float2 tmp;



  for (int uk=0; uk<NUk; uk++)
  {
     u_new[i] = (u[i]-tau*Kyk[i]+tauinv*u_k[i])*div;

     if(pos_real[uk]>=1)
     {
       idx = Nx/2+Ny/2*Nx+NSl/2*Nx*Ny+Nx*Ny*NSl*uk;
       idx2 = Nx/2-10+Ny/2*Nx-10+NSl/2*Nx*Ny+Nx*Ny*NSl*uk;
       idx3 = Nx/2-10+Ny/2*Nx+10+NSl/2*Nx*Ny+Nx*Ny*NSl*uk;
       idx4 = Nx/2+10+Ny/2*Nx-10+NSl/2*Nx*Ny+Nx*Ny*NSl*uk;
       idx5 = Nx/2+10+Ny/2*Nx+10+NSl/2*Nx*Ny+Nx*Ny*NSl*uk;
       tmp = u[i]-tau*Kyk[i]+tauinv*u_k[i]*div;
       u_new[i] = 0.2f*(u[idx]-tau*Kyk[idx]+tauinv*u_k[idx]+u[idx2]-tau*Kyk[idx2]+tauinv*u_k[idx2]+u[idx3]-tau*Kyk[idx3]+tauinv*u_k[idx3]+
                   u[idx4]-tau*Kyk[idx4]+tauinv*u_k[idx4]+u[idx5]-tau*Kyk[idx5]+tauinv*u_k[idx5])*div*tmp;
     }

     if(real[uk]>=1)
     {
         u_new[i].s1 = 0.0f;
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
         norm =  sqrt(pow((float)(u_new[i].s0),(float)(2.0))+pow((float)(u_new[i].s1),(float)(2.0)));
         if (norm<min[uk])
         {
             u_new[i].s0 *= 1/norm*min[uk];
             u_new[i].s1 *= 1/norm*min[uk];
         }
         if(norm>max[uk])
         {
            u_new[i].s0 *= 1/norm*max[uk];
            u_new[i].s1 *= 1/norm*max[uk];
         }
//         if(u_new[i].s0 < 0 && pos_real[uk])
//           u_new[i] = -u_new[i];

     }

     i += NSl*Nx*Ny;
  }
}

__kernel void gradient(__global float8 *grad, __global float2 *u, const int NUk, __global float* scale, const float maxscal, __global float* ratio, const float dz) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;


  for (int uk=0; uk<NUk; uk++)
  {
     // gradient
     grad[i] = (float8)(-u[i],-u[i],-u[i]/dz,0.0f,0.0f);
     if (x < Nx-1)
     { grad[i].s01 += u[i+1].s01;}
     else
     { grad[i].s01 = 0.0f;}

     if (y < Ny-1)
     { grad[i].s23 += u[i+Nx].s01;}
     else
     { grad[i].s23 = 0.0f;}
     if (k < NSl-1)
     { grad[i].s45 += u[i+Nx*Ny].s01/dz;}
     else
     { grad[i].s45 = 0.0f;}
     // scale gradients
     //{grad[i]*=(maxscal/(scale[uk]))*ratio[uk];}
     {grad[i]/=ratio[uk];}
     i += NSl*Nx*Ny;
  }
}

__kernel void sym_grad(__global float16 *sym, __global float8 *w, const int NUk, const float dz) {
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

     sym[i] = (float16)(val_real.s0, val_imag.s0, val_real.s4,val_imag.s4,val_real.s8/dz,val_imag.s8/dz,
                        0.5f*(val_real.s1 + val_real.s3), 0.5f*(val_imag.s1 + val_imag.s3),
                        0.5f*(val_real.s2 + val_real.s6/dz), 0.5f*(val_imag.s2 + val_imag.s6/dz),
                        0.5f*(val_real.s5 + val_real.s7/dz), 0.5f*(val_imag.s5 + val_imag.s7/dz),
                        0.0f,0.0f,0.0f,0.0f);
     i += NSl*Nx*Ny;
   }
}
__kernel void divergence(__global float2 *div, __global float8 *p, const int NUk,
                         __global float* scale, const float maxscal, __global float* ratio, const float dz) {
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
     div[i] = val.s01+val.s23+val.s45/dz;
     // scale gradients
     //{div[i]*=(maxscal/(scale[ukn]))*ratio[ukn];}
     {div[i]/=ratio[ukn];}
     i += NSl*Nx*Ny;
  }

}
__kernel void sym_divergence(__global float8 *w, __global float16 *q,
                       const int NUk, const float dz) {
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
     w[i].s024 = val_real.s012 + val_real.s345 + val_real.s678/dz;
     //imag
     w[i].s135 = val_imag.s012 + val_imag.s345 + val_imag.s678/dz;

     i += NSl*Nx*Ny;
  }
}
__kernel void update_Kyk2(__global float8 *w, __global float16 *q, __global float8 *z,
                       const int NUk, const float dz) {
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
     w[i].s024 = -val_real.s012 - val_real.s345 - val_real.s678/dz -z[i].s024;
     //imag
     w[i].s135 = -val_imag.s012 - val_imag.s345 - val_imag.s678/dz -z[i].s135;
     i += NSl*Nx*Ny;
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
                       const int NSl, const int NScan, __global float* scale, const float maxscal, __global float* ratio, const int Nuk, const float dz)
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
   //{val*=(maxscal/(scale[uk]))*ratio[uk];}
   {val/=ratio[uk];}

  out[uk*NSl*X*Y+k*X*Y+y*X+x] = sum - (val.s01+val.s23+val.s45/dz);
  i += NSl*X*Y;
  }
}


__kernel void operator_fwd_imagespace(__global float2 *out, __global float2 *in, __global float2 *grad,
                       const int NSl, const int NScan, const int Nuk)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);

  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);

  float2 tmp_in = 0.0f;
  float2 tmp_grad = 0.0f;

    for (int scan=0; scan<NScan; scan++)
    {
        float2 sum = 0.0f;
        for (int uk=0; uk<Nuk; uk++)
        {
          tmp_grad = grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x];
          tmp_in = in[uk*NSl*X*Y+k*X*Y+ y*X + x];

          sum += (float2)(tmp_in.x*tmp_grad.x-tmp_in.y*tmp_grad.y,tmp_in.x*tmp_grad.y+tmp_in.y*tmp_grad.x);

        }
        out[scan*NSl*X*Y+k*X*Y + y*X + x] = sum;
    }


}
__kernel void operator_ad_imagespace(__global float2 *out, __global float2 *in,
                      __global float2 *grad,
                       const int NSl, const int NScan, const int Nuk)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);


  float2 tmp_in = 0.0f;
  float2 conj_grad = 0.0f;



  for (int uk=0; uk<Nuk; uk++)
  {
  float2 sum = (float2) 0.0f;
  for (int scan=0; scan<NScan; scan++)
  {
    conj_grad = (float2) (grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].x,
                          -grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].y);
    tmp_in = in[scan*NSl*X*Y+ k*X*Y+ y*X + x];
    sum += (float2)(tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x);

  }
  out[uk*NSl*X*Y+k*X*Y+y*X+x] = sum;
  }

}



__kernel void update_Kyk1_imagespace(__global float2 *out, __global float2 *in,
                       __global float2 *grad, __global float8 *p,
                       const int NSl, const int NScan, __global float* scale, const float maxscal, __global float* ratio, const int Nuk, const float dz)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);

  size_t i = k*X*Y+X*y + x;

  float2 tmp_in = 0.0f;
  float2 conj_grad = 0.0f;


  for (int uk=0; uk<Nuk; uk++)
  {
  float2 sum = (float2) 0.0f;
  for (int scan=0; scan<NScan; scan++)
  {
    conj_grad = (float2) (grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].x,
                          -grad[uk*NScan*NSl*X*Y+scan*NSl*X*Y + k*X*Y + y*X + x].y);
    tmp_in = in[scan*NSl*X*Y+ k*X*Y+ y*X + x];
    sum += (float2)(tmp_in.x*conj_grad.x-tmp_in.y*conj_grad.y,tmp_in.x*conj_grad.y+tmp_in.y*conj_grad.x);
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
   //{val*=(maxscal/(scale[uk]))*ratio[uk];}
   {val/=ratio[uk];}

  out[uk*NSl*X*Y+k*X*Y+y*X+x] = sum - (val.s01+val.s23+val.s45/dz);
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
    print("Please Set Parameters, Data and Initial images")

  def eval_fwd_kspace(self,y,x,wait_for=[]):

    return self.prg.operator_fwd(self.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 y.data, x.data, self.coil_buf, self.grad_buf,
                                 np.int32(self.NC), np.int32(self.NSlice), np.int32(self.NScan), np.int32(self.unknowns),
                                 wait_for=wait_for)
  def operator_forward_kspace(self, x):

    self.tmp_result.add_event(self.eval_fwd_kspace(self.tmp_result,x,wait_for=self.tmp_result.events+x.events))
    self.tmp_sino.add_event(self.NUFFT.fwd_NUFFT(self.tmp_sino,self.tmp_result))
    return  self.tmp_sino

  def operator_forward_full_kspace(self, out, x, wait_for=[]):
    self.tmp_result.add_event(self.eval_fwd_kspace(self.tmp_result,x,wait_for=self.tmp_result.events+x.events))
    return  self.NUFFT.fwd_NUFFT(out,self.tmp_result,wait_for=wait_for+self.tmp_result.events)

  def operator_adjoint_full_kspace(self, out, x,z, wait_for=[]):

    self.tmp_result.add_event(self.NUFFT.adj_NUFFT(self.tmp_result,x,wait_for=wait_for+x.events))

    return self.prg.update_Kyk1(self.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 out.data, self.tmp_result.data, self.coil_buf, self.grad_buf, z.data, np.int32(self.NC),
                                 np.int32(self.NSlice),  np.int32(self.NScan), self.ukscale.data,
                                 np.float32(np.amax(self.ukscale.get())),self.ratio.data, np.int32(self.unknowns),
                                 np.float32(self.dz),
                                 wait_for=self.tmp_result.events+out.events+z.events+wait_for)

  def operator_adjoint_kspace(self, out, x, wait_for=[]):

    self.tmp_result.add_event(self.NUFFT.adj_NUFFT(self.tmp_result,x,wait_for=wait_for+x.events))

    return self.prg.operator_ad(out.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 out.data, self.tmp_result.data, self.coil_buf, self.grad_buf,np.int32(self.NC),
                                 np.int32(self.NSlice),  np.int32(self.NScan), np.int32(self.unknowns),
                                 wait_for=wait_for+self.tmp_result.events+out.events)


  def eval_const(self):
    num_const = (len(self.model.constraints))
    self.min_const = np.zeros((num_const),dtype=np.float32)
    self.max_const = np.zeros((num_const),dtype=np.float32)
    self.real_const = np.zeros((num_const),dtype=np.int32)
    self.pos_real = np.zeros((num_const),dtype=np.int32)
    for j in range(num_const):
        self.min_const[j] = np.float32(self.model.constraints[j].min)
        self.max_const[j] = np.float32(self.model.constraints[j].max)
        self.real_const[j] = np.int32(self.model.constraints[j].real)
        self.pos_real[j] = np.int32(self.model.constraints[j].pos_real)
    self.min_const = clarray.to_device(self.queue, self.min_const)
    self.max_const = clarray.to_device(self.queue, self.max_const)
    self.real_const = clarray.to_device(self.queue, self.real_const)
    self.pos_real = clarray.to_device(self.queue, self.pos_real)


  def f_grad(self,grad, u, wait_for=[]):
    return self.prg.gradient(self.queue, u.shape[1:], None, grad.data, u.data,
                np.int32(self.unknowns),
                self.ukscale.data,  np.float32(np.amax(self.ukscale.get())),self.ratio.data,
                np.float32(self.dz),
                wait_for=grad.events + u.events + wait_for)

  def bdiv(self,div, u, wait_for=[]):
    return self.prg.divergence(div.queue, u.shape[1:-1], None, div.data, u.data,
                np.int32(self.unknowns),
                self.ukscale.data, np.float32(np.amax(self.ukscale.get())),self.ratio.data,
                np.float32(self.dz),
                wait_for=div.events + u.events + wait_for)
#  def f_grad(self,grad, u, wait_for=[]):
#    return self.prg.gradient(self.queue, u.shape[1:], None, grad.data, u.data,
#                np.int32(self.unknowns),
#                self.ukscale.data,  np.float32(1e3),self.ratio.data,
#                wait_for=grad.events + u.events + wait_for)
#
#  def bdiv(self,div, u, wait_for=[]):
#    return self.prg.divergence(div.queue, u.shape[1:-1], None, div.data, u.data,
#                np.int32(self.unknowns),
#                self.ukscale.data, np.float32(1e3),self.ratio.data,
#                wait_for=div.events + u.events + wait_for)

  def sym_grad(self,sym, w, wait_for=[]):
    return self.prg.sym_grad(self.queue, w.shape[1:-1], None, sym.data, w.data,
                np.int32(self.unknowns_TGV),np.float32(self.dz),
                wait_for=sym.events + w.events + wait_for)

  def sym_bdiv(self,div, u, wait_for=[]):
    return self.prg.sym_divergence(self.queue, u.shape[1:-1], None, div.data, u.data,
                np.int32(self.unknowns_TGV),np.float32(self.dz),
                wait_for=div.events + u.events + wait_for)
  def update_Kyk2(self,div, u, z, wait_for=[]):
    return self.prg.update_Kyk2(self.queue, u.shape[1:-1], None, div.data, u.data, z.data,
                np.int32(self.unknowns_TGV),np.float32(self.dz),
                wait_for=div.events + u.events + z.events+wait_for)

  def update_primal(self, x_new, x, Kyk, xk, tau, delta, wait_for=[]):
    return self.prg.update_primal(self.queue, x.shape[1:], None, x_new.data, x.data, Kyk.data, xk.data, np.float32(tau),
                                  np.float32(tau/delta), np.float32(1/(1+tau/delta)), self.min_const.data, self.max_const.data,
                                  self.real_const.data, self.pos_real.data, np.int32(self.unknowns),
                                  wait_for=x_new.events + x.events + Kyk.events+ xk.events+wait_for
                                  )
  def update_z1(self, z_new, z, gx, gx_, vx, vx_, sigma, theta, alpha, omega, wait_for=[]):
    return self.prg.update_z1(self.queue, z.shape[1:-1], None, z_new.data, z.data, gx.data, gx_.data, vx.data, vx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/alpha), np.int32(self.unknowns_TGV), np.int32(self.unknowns_H1), np.float32(1/(1+sigma/omega)),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ vx.events+ vx_.events+wait_for
                                  )
  def update_z1_tv(self, z_new, z, gx, gx_, sigma, theta, alpha, wait_for=[]):
    return self.prg.update_z1_tv(self.queue, z.shape[1:-1], None, z_new.data, z.data, gx.data, gx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/alpha), np.int32(self.unknowns),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ wait_for
                                  )
  def update_z2(self, z_new, z, gx, gx_, sigma, theta, beta, wait_for=[]):
    return self.prg.update_z2(self.queue, z.shape[1:-1], None, z_new.data, z.data, gx.data, gx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/beta),  np.int32(self.unknowns_TGV),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ wait_for
                                  )
  def update_r(self, r_new, r, A, A_, res, sigma, theta, lambd, wait_for=[]):
    return self.prg.update_r(self.queue, (r.size,), None, r_new.data, r.data, A.data, A_.data, res.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/(1+sigma/lambd)),
                                  wait_for= r_new.events + r.events + A.events+ A_.events+ wait_for
                                  )
  def update_v(self, v_new, v, Kyk2, tau, wait_for=[]):
    return self.prg.update_v(self.queue, (v[...,0].size,), None,
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
#      print('scale %f at uk %i' %(self.ukscale[j].get(),j))
#    for j in range(x.shape[0]):
#        self.ratio[j] = self.ukscale[j]/np.amax(self.ukscale.get())
#        print('ratio %f at uk %i' %( self.ratio[j].get(),j))


    grad = clarray.to_device(self.queue,np.zeros_like(self.z1))
    x = clarray.to_device(self.queue,x)
    grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))

    for j in range(x.shape[0]):
      scale = np.linalg.norm(grad[j,...].get())/np.linalg.norm(grad[0,...].get())
#      print(scale)
      if np.isfinite(scale) and scale>1e-4:
        self.ratio[j] = scale
#      print('ratio %f at uk %i' %(self.ratio[j].get(),j))


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
   self.z1 = np.zeros(([self.unknowns,self.NSlice,self.dimY,self.dimX,4]),dtype=DTYPE)


   self.result = np.zeros((self.irgn_par["max_gn_it"]+1,self.unknowns,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
   self.result[0,:,:,:,:] = np.copy(self.model.guess)

   self.Coils3D = self.C
   result = np.copy(self.model.guess)

   if TV==1:
      for i in range(self.irgn_par["max_gn_it"]):
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
        self.result[i+1,...] = self.model.rescale(result)

        iters = np.fmin(iters*2,self.irgn_par["max_iters"])
        self.irgn_par["gamma"] = np.maximum(self.irgn_par["gamma"]*self.irgn_par["gamma_dec"],self.irgn_par["gamma_min"])
        self.irgn_par["delta"] = np.minimum(self.irgn_par["delta"]*self.irgn_par["delta_inc"],self.irgn_par["delta_max"])
        self.irgn_par["omega"] = np.maximum(self.irgn_par["omega"]*self.irgn_par["omega_dec"],self.irgn_par["omega_min"])

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
      self.v = np.zeros(([self.unknowns_TGV,self.NSlice,self.dimY,self.dimX,4]),dtype=DTYPE)
      self.z2 = np.zeros(([self.unknowns_TGV,self.NSlice,self.dimY,self.dimX,8]),dtype=DTYPE)
      for i in range(self.irgn_par["max_gn_it"]):
        start = time.time()


        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))

        for uk in range(self.unknowns-1):
          scale = np.linalg.norm(np.abs(self.grad_x[0,...].flatten()))/np.linalg.norm(np.abs(self.grad_x[uk+1,...].flatten()))
          self.model.constraints[uk+1].update(scale)
          result[uk+1,...] *= self.model.uk_scale[uk+1]
          self.model.uk_scale[uk+1]*=scale
          result[uk+1,...] /= self.model.uk_scale[uk+1]

        self.step_val = np.nan_to_num(self.model.execute_forward_3D(result))
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))
        self.grad_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grad_x.data)

        self.set_scale(result)

        result = self.irgn_solve_3D(result, iters, self.data,TV)
        self.result[i+1,...] = self.model.rescale(result)


        iters = np.fmin(iters*2,self.irgn_par["max_iters"])
        self.irgn_par["gamma"] = np.maximum(self.irgn_par["gamma"]*self.irgn_par["gamma_dec"],self.irgn_par["gamma_min"])
        self.irgn_par["delta"] = np.minimum(self.irgn_par["delta"]*self.irgn_par["delta_inc"],self.irgn_par["delta_max"])
        self.irgn_par["omega"] = np.maximum(self.irgn_par["omega"]*self.irgn_par["omega_dec"],self.irgn_par["omega_min"])

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
      self.v = np.zeros(([self.unknowns,self.NSlice,self.dimY,self.dimX,4]),dtype=DTYPE)
      self.z2 = np.zeros(([self.unknowns,self.NSlice,self.dimY,self.dimX,8]),dtype=DTYPE)
      for i in range(self.irgn_par["max_gn_it"]):
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
        self.result[i+1,...] = self.model.rescale(result)

        iters = np.fmin(iters*2,self.irgn_par["max_iters"])
        self.irgn_par["gamma"] = np.maximum(self.irgn_par["gamma"]*self.irgn_par["gamma_dec"],self.irgn_par["gamma_min"])
        self.irgn_par["delta"] = np.minimum(self.irgn_par["delta"]*self.irgn_par["delta_inc"],self.irgn_par["delta_max"])
        self.irgn_par["omega"] = np.maximum(self.irgn_par["omega"]*self.irgn_par["omega_dec"],self.irgn_par["omega_min"])

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
      grad = grad.get()
      self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - b.get())**2
              +self.irgn_par["gamma"]*np.sum(np.abs(grad[:self.unknowns_TGV]))
              +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2
              +self.irgn_par["omega"]/2*np.linalg.norm(grad[self.unknowns_TGV:])**2)
    elif TV==0:
       x = self.tgv_solve_3D(x.get(),res,iters)

       x = clarray.to_device(self.queue,x)
       v = clarray.to_device(self.queue,self.v)

       grad = clarray.to_device(self.queue,np.zeros_like(self.z1))


       sym_grad = clarray.to_device(self.queue,np.zeros_like(self.z2))


       grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))


       sym_grad.add_event(self.sym_grad(sym_grad,v,wait_for=sym_grad.events+v.events))
       x = x.get()
       grad = grad.get()
       test = self.model.execute_forward_3D(x)[:,None,...]
       self.FT(b,clarray.to_device(self.queue,test*self.Coils3D)).wait()
       self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - b.get())**2
              +self.irgn_par["gamma"]*np.sum(np.abs(grad[:self.unknowns_TGV]-self.v))
              +self.irgn_par["gamma"]*(2)*np.sum(np.abs(sym_grad.get()))
              +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2
              +self.irgn_par["omega"]/2*np.linalg.norm(grad[self.unknowns_TGV:])**2)
    else:
       x = self.tgv_solve_3D_explicit(x.get(),res,iters)
       x = clarray.to_device(self.queue,x)
       v = clarray.to_device(self.queue,self.v)
       grad = clarray.to_device(self.queue,np.zeros_like(self.z1))
       sym_grad = clarray.to_device(self.queue,np.zeros_like(self.z2))
       grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))
       sym_grad.add_event(self.sym_grad(sym_grad,v,wait_for=sym_grad.events+v.events))
       x = x.get()
       grad = grad.get()
       self.FT(b,clarray.to_device(self.queue,self.model.execute_forward_3D(x)[:,None,...]*self.Coils3D)).wait()
       self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - b.get())**2
              +self.irgn_par["gamma"]*np.sum(np.abs(grad[:self.unknowns_TGV]-self.v))
              +self.irgn_par["gamma"]*(2)*np.sum(np.abs(sym_grad.get()))
              +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2
              +self.irgn_par["omega"]/2*np.linalg.norm(grad[self.unknowns_TGV:])**2)

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


    x = clarray.to_device(self.queue,x)
    xk = x.copy()
    x_new = clarray.zeros_like(x)#x.copy()

    r = clarray.to_device(self.queue,np.zeros_like(self.r))#np.zeros_like(res,dtype=DTYPE)
    r_new = r.copy()#clarray.zeros_like(r)
    z1 = clarray.to_device(self.queue,np.zeros_like(self.z1))#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    z1_new = z1.copy()#clarray.zeros_like(z1)#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    z2 = clarray.to_device(self.queue,np.zeros_like(self.z2))#np.zeros(([self.unknowns,3,self.dimY,self.dimX]),dtype=DTYPE)
    z2_new = z2.copy()#clarray.zeros_like(z2)
    v = clarray.to_device(self.queue,np.zeros_like(self.v))#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    v_new = v.copy()#clarray.zeros_like(v)
    res = clarray.to_device(self.queue, res.astype(DTYPE))


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
#      if np.sum(np.abs(x_new[3,...]<self.model.constraints[3].min)) or np.sum(np.abs(x_new[3,...]>self.model.constraints[3].max)):
#        import ipdb
#        import multislice_viewer as msv
#        import matplotlib.pyplot as plt
#        ipdb.set_trace()
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
        z1_new.add_event(self.update_z1(z1_new,z1,gradx,gradx_xold,v_new,v, beta_line*tau_new, theta_line, alpha,self.irgn_par["omega"])) ### Q2
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


      if not np.mod(i,50):

        self.model.plot_unknowns(x_new.get())
        if self.unknowns_H1>0:
          primal_new= (self.irgn_par["lambd"]/2*clarray.vdot(Axold-res,Axold-res)+alpha*clarray.sum(abs((gradx[:self.unknowns_TGV]-v))) +beta*clarray.sum(abs(symgrad_v)) + 1/(2*delta)*clarray.vdot(x_new-xk,x_new-xk)+self.irgn_par["omega"]/2*clarray.vdot(gradx[self.unknowns_TGV:],gradx[self.unknowns_TGV:])).real

          dual = (-delta/2*clarray.vdot(-Kyk1,-Kyk1)- clarray.vdot(xk,(-Kyk1)) + clarray.sum(Kyk2)
                    - 1/(2*self.irgn_par["lambd"])*clarray.vdot(r,r) - clarray.vdot(res,r)-
                    1/(2*self.irgn_par["omega"])*clarray.vdot(z1[self.unknowns_TGV:],z1[self.unknowns_TGV:])).real
        else:
          primal_new= (self.irgn_par["lambd"]/2*clarray.vdot(Axold-res,Axold-res)+alpha*clarray.sum(abs((gradx-v))) +beta*clarray.sum(abs(symgrad_v)) + 1/(2*delta)*clarray.vdot(x_new-xk,x_new-xk)).real

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
          return x_new.get()
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

    r = clarray.to_device(self.queue,np.zeros_like(self.r))#np.zeros_like(res,dtype=DTYPE)
    r_new  = clarray.zeros_like(r)
    z1 = clarray.to_device(self.queue,np.zeros_like(self.z1))#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    z1_new = clarray.zeros_like(z1)#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    res = clarray.to_device(self.queue, res.astype(DTYPE))


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


      if not np.mod(i,50):
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

    z1 = clarray.to_device(self.queue,self.z1)#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    z1_new = clarray.zeros_like(z1)#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    z2 = clarray.to_device(self.queue,self.z2)#np.zeros(([self.unknowns,3,self.dimY,self.dimX]),dtype=DTYPE)
    z2_new = clarray.zeros_like(z2)
    v = clarray.to_device(self.queue,self.v)#np.zeros(([self.unknowns,2,self.dimY,self.dimX]),dtype=DTYPE)
    v_new = clarray.zeros_like(v)
    res = clarray.to_device(self.queue, res.astype(DTYPE))


    delta = self.irgn_par["delta"]
    omega = self.irgn_par["omega"]
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
        z1_new.add_event(self.update_z1(z1_new,z1,gradx,gradx_xold,v_new,v, beta_line*tau_new, theta_line, alpha,omega)) ### Q2
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
        sys.stdout.write("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f \r"%(i,primal.get()/(self.irgn_par["lambd"]*self.NSlice),dual.get()/(self.irgn_par["lambd"]*self.NSlice),gap.get()/(self.irgn_par["lambd"]*self.NSlice)))
        sys.stdout.flush()


      (x, x_new) = (x_new, x)
      (v, v_new) = (v_new, v)

    self.v = v.get()
    self.z1 = z1.get()
    self.z2 = z2.get()
    return x.get()


################################################################################
### Start a 3D Reconstruction, set TV to True to perform TV instead of TGV######
### Precompute Model and Gradient values for xk ################################
### Call inner optimization ####################################################
### input: bool to switch between TV (1) and TGV (0) regularization ############
### output: optimal value of x #################################################
################################################################################
  def execute_3D_imagespace(self, TV=0):
   iters = self.irgn_par["start_iters"]
   self.NC=1
   self.N = self.dimX
   self.Nproj = self.dimY


   self.r = np.zeros_like(self.data,dtype=DTYPE)
   self.z1 = np.zeros(([self.unknowns,self.NSlice,self.dimY,self.dimX,4]),dtype=DTYPE)


   self.result = np.zeros((self.irgn_par["max_gn_it"]+1,self.unknowns,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
   self.result[0,:,:,:,:] = np.copy(self.model.guess)

   result = np.copy(self.model.guess)
   if TV==1:
      for i in range(self.irgn_par["max_gn_it"]):
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

        result = self.irgn_solve_3D_imagespace(result, iters, self.data,TV)
        self.result[i+1,...] = self.model.rescale(result)

        iters = np.fmin(iters*2,self.irgn_par["max_iters"])
        self.irgn_par["gamma"] = np.maximum(self.irgn_par["gamma"]*self.irgn_par["gamma_dec"],self.irgn_par["gamma_min"])
        self.irgn_par["delta"] = np.minimum(self.irgn_par["delta"]*self.irgn_par["delta_inc"],self.irgn_par["delta_max"])
        self.irgn_par["omega"] = np.maximum(self.irgn_par["omega"]*self.irgn_par["omega_dec"],self.irgn_par["omega_min"])

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
      self.v = np.zeros(([self.unknowns_TGV,self.NSlice,self.dimY,self.dimX,4]),dtype=DTYPE)
      self.z2 = np.zeros(([self.unknowns_TGV,self.NSlice,self.dimY,self.dimX,8]),dtype=DTYPE)
      for i in range(self.irgn_par["max_gn_it"]):
        start = time.time()


        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))

        for uk in range(self.unknowns-1):
          scale = np.linalg.norm(np.abs(self.grad_x[0,...].flatten()))/np.linalg.norm(np.abs(self.grad_x[uk+1,...].flatten()))
          self.model.constraints[uk+1].update(scale)
          result[uk+1,...] *= self.model.uk_scale[uk+1]
          self.model.uk_scale[uk+1]*=scale
          result[uk+1,...] /= self.model.uk_scale[uk+1]

        self.step_val = np.nan_to_num(self.model.execute_forward_3D(result))
        self.grad_x = np.nan_to_num(self.model.execute_gradient_3D(result))
        self.grad_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grad_x.data)

        self.set_scale(result)

        result = self.irgn_solve_3D_imagespace(result, iters, self.data,TV)
        self.result[i+1,...] = self.model.rescale(result)


        iters = np.fmin(iters*2,self.irgn_par["max_iters"])
        self.irgn_par["gamma"] = np.maximum(self.irgn_par["gamma"]*self.irgn_par["gamma_dec"],self.irgn_par["gamma_min"])
        self.irgn_par["delta"] = np.minimum(self.irgn_par["delta"]*self.irgn_par["delta_inc"],self.irgn_par["delta_max"])
        self.irgn_par["omega"] = np.maximum(self.irgn_par["omega"]*self.irgn_par["omega_dec"],self.irgn_par["omega_min"])

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
      self.v = np.zeros(([self.unknowns,self.NSlice,self.dimY,self.dimX,4]),dtype=DTYPE)
      self.z2 = np.zeros(([self.unknowns,self.NSlice,self.dimY,self.dimX,8]),dtype=DTYPE)
      for i in range(self.irgn_par["max_gn_it"]):
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

        result = self.irgn_solve_3D_imagespace(result, iters, self.data,TV)
        self.result[i+1,...] = self.model.rescale(result)

        iters = np.fmin(iters*2,self.irgn_par["max_iters"])
        self.irgn_par["gamma"] = np.maximum(self.irgn_par["gamma"]*self.irgn_par["gamma_dec"],self.irgn_par["gamma_min"])
        self.irgn_par["delta"] = np.minimum(self.irgn_par["delta"]*self.irgn_par["delta_inc"],self.irgn_par["delta_max"])
        self.irgn_par["omega"] = np.maximum(self.irgn_par["omega"]*self.irgn_par["omega_dec"],self.irgn_par["omega_min"])

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
  def irgn_solve_3D_imagespace(self,x,iters, data, TV=0):
    x_old = x
    x = clarray.to_device(self.queue,np.require(x,requirements="C"))

    res = data - self.step_val + self.operator_forward(x).get()


    if TV==1:
      x = self.tv_solve_3D(x.get(),res,iters)
      x = clarray.to_device(self.queue,x)
      grad = clarray.to_device(self.queue,np.zeros_like(self.z1))
      grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))
      x = x.get()
      grad = grad.get()
      self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - self.model.execute_forward_3D(x))**2
              +self.irgn_par["gamma"]*np.sum(np.abs(grad[:self.unknowns_TGV]))
              +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2+self.irgn_par["omega"]/2*np.linalg.norm(grad[self.unknowns_TGV:])**2)
    elif TV==0:
       x = self.tgv_solve_3D(x.get(),res,iters)

       x = clarray.to_device(self.queue,x)
       v = clarray.to_device(self.queue,self.v)

       grad = clarray.to_device(self.queue,np.zeros_like(self.z1))


       sym_grad = clarray.to_device(self.queue,np.zeros_like(self.z2))


       grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))


       sym_grad.add_event(self.sym_grad(sym_grad,v,wait_for=sym_grad.events+v.events))
       x = x.get()
       grad = grad.get()
       self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - self.model.execute_forward_3D(x))**2
              +self.irgn_par["gamma"]*np.sum(np.abs(grad[:self.unknowns_TGV]-self.v))
              +self.irgn_par["gamma"]*(2)*np.sum(np.abs(sym_grad.get()))
              +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2
              +self.irgn_par["omega"]/2*np.linalg.norm(grad[self.unknowns_TGV:])**2)
    else:
       x = self.tgv_solve_3D_explicit(x.get(),res,iters)
       x = clarray.to_device(self.queue,x)
       v = clarray.to_device(self.queue,self.v)
       grad = clarray.to_device(self.queue,np.zeros_like(self.z1))
       sym_grad = clarray.to_device(self.queue,np.zeros_like(self.z2))
       grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))
       sym_grad.add_event(self.sym_grad(sym_grad,v,wait_for=sym_grad.events+v.events))
       x = x.get()
       grad = grad.get()
       self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(data - self.model.execute_forward_3D(x))**2
              +self.irgn_par["gamma"]*np.sum(np.abs(grad[:self.unknowns_TGV]-self.v))
              +self.irgn_par["gamma"]*(2)*np.sum(np.abs(sym_grad.get()))
              +1/(2*self.irgn_par["delta"])*np.linalg.norm((x-x_old).flatten())**2
              ++self.irgn_par["omega"]/2*np.linalg.norm(grad[self.unknowns_TGV:])**2)

    print("-"*80)
    print ("Function value after GN-Step: %f" %(self.fval/(self.irgn_par["lambd"]*self.NSlice)))

    return x


  def eval_fwd_imagespace(self,y,x,wait_for=[]):

    return self.prg.operator_fwd_imagespace(self.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 y.data, x.data, self.grad_buf,
                                 np.int32(self.NSlice), np.int32(self.NScan), np.int32(self.unknowns),
                                 wait_for=wait_for)
  def operator_forward_imagespace(self, x):

    self.tmp_result.add_event(self.eval_fwd_imagespace(self.tmp_result,x,wait_for=self.tmp_result.events+x.events))
    return  self.tmp_result

  def operator_forward_full_imagespace(self, out, x, wait_for=[]):
    return  self.eval_fwd_imagespace(out,x,wait_for=out.events+x.events)

  def operator_adjoint_full_imagespace(self, out, x,z, wait_for=[]):


    return self.prg.update_Kyk1_imagespace(self.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 out.data, x.data, self.grad_buf, z.data,
                                 np.int32(self.NSlice),  np.int32(self.NScan), self.ukscale.data,
                                 np.float32(np.amax(self.ukscale.get())),self.ratio.data, np.int32(self.unknowns),
                                 np.float32(self.dz),
                                 wait_for=x.events+out.events+z.events+wait_for)

  def operator_adjoint_imagespace(self, out, x, wait_for=[]):

    return self.prg.operator_ad_imagespace(out.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 out.data, x.data, self.grad_buf,
                                 np.int32(self.NSlice),  np.int32(self.NScan), np.int32(self.unknowns),
                                 wait_for=wait_for+x.events+out.events)

  def execute(self,TV=0,imagespace=0,reco_2D=0):

    if reco_2D:
      print("2D currently not implemented, 3D can be used with a single slice.")
      return
    else:
      if imagespace:
        self.execute_3D_imagespace(TV)
      else:
        self.execute_3D(TV)
