
# cython: infer_types=True
# cython: profile=False

from __future__ import division

import numpy as np
import time

import gradients_divergences_old as gd

import matplotlib.pyplot as plt
plt.ion()

DTYPE = np.complex64
DTYPE_real = np.float32


#import pynfft.nfft as nfft

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
    self.ratio = 100
#    self.ukscale =  np.ones(self.unknowns,dtype=DTYPE_real)
    self.ukscale =  clarray.to_device(self.queue,np.ones(self.unknowns,dtype=DTYPE_real))
    self.gn_res = []

    self.prg = Program(self.ctx, r"""
__kernel void gradient(__global float8 *grad, __global float2 *u, const int NUk, __global float* scale, const float ratio) {
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
     {grad[i]/=scale[uk];}
     else
     {grad[i]*=(ratio/scale[uk]);}
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
                         __global float* scale, const float ratio) {
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
     {div[i]/=scale[ukn];}
     else
     {div[i]*=(ratio/scale[ukn]);}
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
#    shift_read = np.array((-0.0322,-0.0314 ,  -0.0319  , -0.0315  , -0.0312   ,-0.0302   ,-0.0306   ,-0.0302  , -0.0302,       -0.0300))
#    shift_phase = np.array((0.1280  ,  0.1283 ,   0.1282  ,  0.1307 ,   0.1333  ,  0.1327  ,  0.1379,    0.1395 ,   0.1293,    0.1290))

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
    x= x[:,None,...]
    data= data[:,:,None,...]
    xx = np.random.random_sample(np.shape(x)).astype(DTYPE)
    yy = np.random.random_sample(np.shape(data)).astype(DTYPE)
    a = np.vdot(xx.flatten(),self.operator_adjoint(clarray.to_device(self.queue,yy)).get().flatten())
    b = np.vdot(self.operator_forward(clarray.to_device(self.queue,xx)).get().flatten(),yy.flatten())
    test = np.abs(a-b)
    print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)))
    x_old = np.copy(x)

    x = clarray.to_device(self.queue,np.require(x,requirements="C"))
    a = (self.operator_forward(x).get())
    b = (self.FT(self.step_val[:,None,...]*self.Coils))
    res = (data) - b + a

    x = self.tgv_solve_2D(x.get(),res,iters)
    x = clarray.to_device(self.queue,x)
    v = clarray.to_device(self.queue,self.v)
    grad = clarray.to_device(self.queue,np.zeros_like(self.z1))
    sym_grad = clarray.to_device(self.queue,np.zeros_like(self.z2))
    grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))
    sym_grad.add_event(self.sym_grad(sym_grad,v,wait_for=sym_grad.events+v.events))
    x = x.get()
    self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - (self.FT(self.model.execute_forward_2D(x,0)[:,None,None,...]*self.Coils)))**2
            +self.irgn_par.gamma*np.sum(np.abs(grad.get()-self.v))
            +self.irgn_par.gamma*(2)*np.sum(np.abs(sym_grad.get()))
            +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)


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
      self.r_struct=self.radon_struct()
      self.FT = self.nFT
      self.FTH = self.nFTH

      gamma = self.irgn_par.gamma
      delta = self.irgn_par.delta

      self.result = np.zeros((self.irgn_par.max_GN_it,self.unknowns_TGV+self.unknowns_H1,self.par.NSlice,self.par.dimY,self.par.dimX),dtype=DTYPE)
      result = np.copy(self.model.guess)
      for islice in range(self.par.NSlice):
        self.irgn_par.gamma = gamma
        self.irgn_par.delta = delta
        self.Coils = np.array(np.squeeze(self.par.C[:,islice,:,:]),order='C')[:,None,...]
        self.coil_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Coils.data)
        self.conjCoils = np.conj(self.Coils)
        self.v = np.zeros(([self.unknowns_TGV,self.NSlice,self.par.dimX,self.par.dimY,4]),dtype=DTYPE)
        self.r = np.zeros(([self.NScan,self.NC,self.NSlice,self.Nproj,self.N]),dtype=DTYPE)
        self.z1 = np.zeros(([self.unknowns_TGV,self.NSlice,self.par.dimX,self.par.dimY,4]),dtype=DTYPE)
        self.z2 = np.zeros(([self.unknowns_TGV,self.NSlice,self.par.dimX,self.par.dimY,8]),dtype=DTYPE)
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
          self.step_val = self.model.execute_forward_2D(result[:,islice,:,:],islice)[:,None,...]
          self.grad_x_2D = np.nan_to_num(self.model.execute_gradient_2D(result[:,islice,:,:],islice).astype(DTYPE))[:,:,None,...]
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


  def eval_fwd(self,y,x,wait_for=[]):

    return self.prg.operator_fwd(y.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 y.data, x.data, self.coil_buf, self.grad_buf,
                                 np.int32(self.NC), np.int32(self.NSlice), np.int32(self.NScan), np.int32(self.unknowns),
                                 wait_for=wait_for)
  def operator_forward(self, x):

#    return self.FT(np.sum(x[:,None,...]*self.grad_x_2D,axis=0)[:,None,...]*self.Coils)
#    tmp_img = clarray.to_device(self.queue,np.require(np.reshape(x.get(),(self.unknowns*self.NSlice,self.dimY,self.dimX)),DTYPE,"C"))
    tmp_img = clarray.reshape(x.astype(DTYPE),(self.unknowns*self.NSlice,self.dimY,self.dimX))
    tmp_result = clarray.zeros(self.queue,(self.NScan*self.NC*self.NSlice,self.dimY,self.dimX),DTYPE,"C")
    tmp_result.add_event(self.eval_fwd(tmp_result,tmp_img,wait_for=tmp_result.events+tmp_img.events))
    tmp_sino = clarray.zeros(self.queue,self.r_struct[2],DTYPE,"C")#self.FT(np.reshape(tmp_result.get(),(self.NScan,self.NC,self.dimY,self.dimX)))#
    tmp_sino.add_event(self.radon(tmp_sino,tmp_result,wait_for=tmp_sino.events+tmp_result.events))
    return (clarray.reshape(tmp_sino,(self.NScan,self.NC,self.NSlice,self.Nproj,self.N)))

  def operator_adjoint(self, x):

#    return np.squeeze(np.sum(np.squeeze(np.sum(self.FTH(x)*self.conjCoils,axis=1))*self.conj_grad_x_2D,axis=1))
#    tmp_sino = clarray.to_device(self.queue,np.require(np.reshape(x.get(),(self.NScan*self.NC*self.NSlice,self.Nproj,self.N)),DTYPE,"C"))
    tmp_sino = clarray.reshape(x.astype(DTYPE),(self.NScan*self.NC*self.NSlice,self.Nproj,self.N))
#    tmp_sino = clarray.reshape(x,(self.NScan*self.NC,self.Nproj,self.N))
    tmp_img =  clarray.zeros(self.queue,self.r_struct[1],DTYPE,"C")#clarray.to_device(self.queue,self.FTH(x.get()))#
    tmp_img.add_event(self.radon_ad(tmp_img,tmp_sino,wait_for=tmp_img.events+tmp_sino.events))
    tmp_result = clarray.zeros(self.queue,(self.unknowns*self.NSlice,self.dimY,self.dimX),DTYPE,"C")

    tmp_result.add_event(self.eval_adj(tmp_result,tmp_img,wait_for=tmp_result.events+tmp_img.events))
    return (clarray.reshape(tmp_result,(self.unknowns,self.NSlice,self.dimY,self.dimX)))

  def eval_adj(self,x,y,wait_for=[]):

    return self.prg.operator_ad(x.queue, (self.NSlice,self.dimY,self.dimX), None,
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


  def tgv_solve_2D(self, x,res, iters):
    alpha = self.irgn_par.gamma
    beta = self.irgn_par.gamma*2

    xx = np.zeros_like(x,dtype=DTYPE)
    yy = np.zeros_like(x,dtype=DTYPE)
    xx = np.random.random_sample(x.shape).astype(DTYPE)
    xxcl = clarray.to_device(self.queue,xx)
    yy = self.operator_adjoint(self.operator_forward(xxcl)).get();
    for j in range(10):
       if not np.isclose(np.linalg.norm(yy.flatten()),0):
           xx = yy/np.linalg.norm(yy.flatten())
       else:
           xx = yy
       xxcl = clarray.to_device(self.queue,xx)
       yy = self.operator_adjoint(self.operator_forward(xxcl)).get()
       l1 = np.vdot(yy.flatten(),xx.flatten());
    L = np.max(np.abs(l1)) ## Lipschitz constant estimate
    L = 0.5*(18.0 + np.sqrt(33))
    print('L: %f'%(L))


    tau = 1/np.sqrt(L)
    tau_new = 0

    self.set_scale(x)
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

    Kyk1 = clarray.zeros_like(x)
    Kyk1_new = clarray.zeros_like(x)
    Kyk2 = clarray.zeros_like(z1)
    Kyk2_new = clarray.zeros_like(z1)
    gradx = clarray.zeros_like(z1)
    gradx_xold = clarray.zeros_like(z1)
    symgrad_v = clarray.zeros_like(z2)
    symgrad_v_vold = clarray.zeros_like(z2)

    Axold = self.operator_forward(x)

    Kyk1.add_event(self.bdiv(Kyk1,z1,wait_for=Kyk1.events+z1.events))
    Kyk2.add_event(self.sym_bdiv(Kyk2,z2,wait_for=Kyk2.events+z2.events))
    self.queue.finish()

    Kyk1 = self.operator_adjoint(r) - Kyk1#clarray.to_device(self.queue,self.scale_adj(gd.bdiv_1(z1.get())))
    Kyk2 = -z1 - Kyk2#clarray.to_device(self.queue,gd.fdiv_2(z2.get()))
#    print('Kyk1: %f' %(np.linalg.norm(Kyk1.get())))
#    print('Kyk2: %f' %(np.linalg.norm(Kyk2.get())))

#    self.queue.finish()
    for i in range(iters):
      x_new = ((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta)
      x_new = x_new.get()

      for j in range(len(self.model.constraints)):
        x_new[j,...] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,x_new[j,...]))
        if self.model.constraints[j].real:
          x_new[j,...] = np.real(x_new[j,...])
      x_new = clarray.to_device(self.queue,x_new.astype(DTYPE))


      v_new = (v-tau*Kyk2).astype(DTYPE)

      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))

      beta_line = beta_new

#      self.queue.finish()
#      print('xnew: %f' %(np.linalg.norm(x_new.get())))
#      print(x_new.dtype)
#      print(v_new.dtype)
#      gradx = clarray.to_device(self.queue,gd.fgrad_1(self.scale_fwd(x_new.get())))

      gradx.add_event(self.f_grad(gradx,x_new,wait_for=gradx.events+x_new.events))
      gradx_xold.add_event(self.f_grad(gradx_xold,x,wait_for=gradx_xold.events+x.events))



      v_vold = v_new-v
      symgrad_v.add_event(self.sym_grad(symgrad_v,v_new,wait_for=symgrad_v.events+v_new.events))
      symgrad_v_vold.add_event(self.sym_grad(symgrad_v_vold,v,wait_for=symgrad_v_vold.events+v.events))
#      symgrad_v = clarray.to_device(self.queue,gd.sym_bgrad_2(v_new.get()))
#      self.queue.finish()
      gradx_xold = gradx - gradx_xold#clarray.to_device(self.queue,gd.fgrad_1(self.scale_fwd(x.get())))
      symgrad_v_vold = symgrad_v - symgrad_v_vold#clarray.to_device(self.queue,gd.sym_bgrad_2(v.get()))

      Ax = self.operator_forward(x_new)
      Ax_Axold = Ax-Axold
#      print(gradx.dtype)
#      print(gradx_xold.dtype)
#      print(symgrad_v.dtype)
#      print(v_vold.dtype)
#      print(symgrad_v_vold.dtype)
#      self.queue.finish()
#
#      print('gradxnew: %f' %(np.linalg.norm(gradx.get())))
#      print('gradx_xold: %f' %(np.linalg.norm(gradx_xold.get())))
#      print('symgrad_v_vold: %f' %(np.linalg.norm(symgrad_v_vold.get())))

      while True:

        theta_line = tau_new/tau
#        print('z1: %f' %(np.linalg.norm(z1.get())))
        z1_new = (z1 + beta_line*tau_new*( gradx + theta_line*gradx_xold
                                          - v_new - theta_line*v_vold  )).get()
        z1_new = clarray.to_device(self.queue,(z1_new/np.maximum(1,(np.sqrt(np.sum(np.abs(z1_new)**2,axis=(0,-1),keepdims=True))/alpha))).astype(DTYPE))
#        print('z1_new: %f' %(np.linalg.norm(z1_new.get())))
#        print('Kyk1_new: %f' %(np.linalg.norm(Kyk1_new.get())))
#        print(z1_new.dtype)
        Kyk1_new.add_event(self.bdiv(Kyk1_new,z1_new,wait_for=Kyk1_new.events+z1_new.events))
#        self.queue.finish()
#        print('Kyk1_new: %f' %(np.linalg.norm(Kyk1_new.get())))
        z2_new = z2 + beta_line*tau_new*( symgrad_v + theta_line*symgrad_v_vold )
        z2_new = z2_new.get()
        scal = np.sqrt( np.sum(np.abs(z2_new[...,0])**2 + np.abs(z2_new[...,1])**2 + 2*np.abs(z2_new[...,3])**2,axis=0,keepdims=True) )
        scal = np.maximum(1,scal/(beta))[...,None]

        z2_new = clarray.to_device(self.queue,(z2_new/scal).astype(DTYPE))
        Kyk2_new.add_event(self.sym_bdiv(Kyk2_new,z2_new,wait_for=Kyk2_new.events+z2_new.events))

        tmp = Ax+theta_line*Ax_Axold
        self.queue.finish()
        r_new = ( r  + beta_line*tau_new*(tmp - res))/(1+beta_line*tau_new/self.irgn_par.lambd)

#        self.queue.finish()
#        print('Kyk1_new: %f' %(np.linalg.norm(Kyk1_new.get())))
#        print('Kyk2_new: %f' %(np.linalg.norm(Kyk2_new.get())))

        Kyk1_new = self.operator_adjoint(r_new) - Kyk1_new#clarray.to_device(self.queue,self.scale_adj(gd.bdiv_1(z1_new.get())))
        Kyk2_new = -z1_new -Kyk2_new#clarray.to_device(self.queue,gd.fdiv_2(z2_new.get()))
#        print(Kyk1_new.dtype)
#        print(Kyk2_new.dtype)
#        self.queue.finish()
        ynorm = ((clarray.vdot(r_new-r,r_new-r)+clarray.vdot(z1_new-z1,z1_new-z1)+clarray.vdot(z2_new-z2,z2_new-z2))**(1/2)).real
        lhs = np.sqrt(beta_line)*tau_new*((clarray.vdot(Kyk1_new-Kyk1,Kyk1_new-Kyk1)+clarray.vdot(Kyk2_new-Kyk2,Kyk2_new-Kyk2))**(1/2)).real
        self.queue.finish()
        if lhs <= ynorm*delta_line:
            break
        else:
#          print('lhs: %f \n rhs: %f' %(lhs.get(), ynorm.get()*delta_line))
          tau_new = tau_new*mu_line

      (Kyk1, Kyk1_new, Kyk2, Kyk2_new, Axold, Ax, z1, z1_new, z2, z2_new, r, r_new) =\
      (Kyk1_new, Kyk1, Kyk2_new, Kyk2, Ax, Axold, z1_new, z1, z2_new, z2, r_new, r)
      tau =  (tau_new)


      if not np.mod(i,20):

        self.model.plot_unknowns(x_new.get()[:,0,...],True)
        primal_new= (self.irgn_par.lambd/2*clarray.vdot(Axold-res,Axold-res)+alpha*clarray.sum(abs((gradx[:self.unknowns_TGV]-v))) +beta*clarray.sum(abs(symgrad_v)) + 1/(2*delta)*clarray.vdot(x_new-xk,x_new-xk)).real

        dual = (-delta/2*clarray.vdot(-Kyk1,-Kyk1)- clarray.vdot(xk,(-Kyk1)) + clarray.sum(Kyk2)
                  - 1/(2*self.irgn_par.lambd)*clarray.vdot(r,r) - clarray.vdot(res,r)).real

        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<self.irgn_par.lambd*self.irgn_par.tol:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,abs(primal-primal_new).get()/(self.irgn_par.lambd*self.NSlice)))
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          return x_new.get()
        if (gap > gap_min*self.irgn_par.stag) and i>1:
          self.v = v_new.get()
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

      (x, x_new) = (x_new, x)
      (v, v_new) = (v_new, v)
#      for j in range(self.par.unknowns_TGV):
#        self.scale_2D[j,...] = np.linalg.norm(x[j,...])
    self.v = v.get()
    self.r = r.get()
    self.z1 = z1.get()
    self.z2 = z2.get()
    return x.get()


  def nFT(self, x):
    result = np.zeros((self.NScan,self.NC,self.NSlice,self.Nproj,self.N),dtype=DTYPE)
    tmp_img = clarray.to_device(self.queue,np.require(np.reshape(x,(self.NScan*self.NC*self.NSlice,self.dimY,self.dimX)),DTYPE,"C"))
    tmp_sino = clarray.zeros(self.queue,self.r_struct[2],DTYPE,"C")
    (self.radon(tmp_sino,tmp_img))
    result = (np.reshape(tmp_sino.get(),(self.NScan,self.NC,self.NSlice,self.Nproj,self.N)))

    return result



  def nFTH(self, x):
    result = np.zeros((self.NScan,self.NC,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
    tmp_sino = clarray.to_device(self.queue,np.require(np.reshape(x,(self.NScan*self.NC*self.NSlice,self.Nproj,self.N)),DTYPE,"C"))
    tmp_img = clarray.zeros(self.queue,self.r_struct[1],DTYPE,"C")
    (self.radon_ad(tmp_img,tmp_sino))
    result = (np.reshape(tmp_img.get(),(self.NScan,self.NC,self.NSlice,self.dimY,self.dimX)))

    return result

  def f_grad(self,grad, u, wait_for=[]):
    return self.prg.gradient(grad.queue, u.shape[1:], None, grad.data, u.data,
                np.int32(self.unknowns),
                self.ukscale.data, np.float32(np.amax(self.ukscale.get())/self.ratio),
                wait_for=grad.events + u.events + wait_for)

  def bdiv(self,div, u, wait_for=[]):
    return self.prg.divergence(div.queue, u.shape[1:-1], None, div.data, u.data,
                np.int32(self.unknowns),
                self.ukscale.data, np.float32(np.amax(self.ukscale.get())/self.ratio),
                wait_for=div.events + u.events + wait_for)

  def sym_grad(self,sym, w, wait_for=[]):
    return self.prg.sym_grad(sym.queue, w.shape[1:-1], None, sym.data, w.data,
                np.int32(self.unknowns),
                wait_for=sym.events + w.events + wait_for)

  def sym_bdiv(self,div, u, wait_for=[]):
    return self.prg.sym_divergence(div.queue, u.shape[1:-1], None, div.data, u.data,
                np.int32(self.unknowns),
                wait_for=div.events + u.events + wait_for)
################################################################################
### Scale before gradient ######################################################
################################################################################
  def set_scale(self,x):
    for j in range(x.shape[0]):
      self.ukscale[j] = np.linalg.norm(x[j,...])
      print('scale %f at uk %i' %(self.ukscale[j].get(),j))
  def scale_fwd(self,x):
    y = np.copy(x)
    for j in range(x.shape[0]):
      y[j,...] /= self.ukscale[j].get()
      if j==0:
        y[j,...] *= np.max(self.ukscale.get())/self.ratio
      else:
        y[j,...] *= np.max(self.ukscale.get())
    return y
  def scale_adj(self,x):
    y = np.copy(x)
    for j in range(x.shape[0]):
      y[j,...] /= self.ukscale[j].get()
      if j==0:
        y[j,...] *= np.max(self.ukscale.get())/self.ratio
      else:
        y[j,...] *= np.max(self.ukscale.get())
    return y


################################################################################
### Start a 3D Reconstruction, set TV to True to perform TV instead of TGV######
### Precompute Model and Gradient values for xk ################################
### Call inner optimization ####################################################
### input: bool to switch between TV (1) and TGV (0) regularization ############
### output: optimal value of x #################################################
################################################################################
  def execute_3D(self, TV=0):
   self.FT = self.nFT
   self.FTH = self.nFTH
   iters = self.irgn_par.start_iters


   self.r = np.zeros_like(self.data,dtype=DTYPE)
   self.z1 = np.zeros(([self.unknowns,self.NSlice,self.par.dimX,self.par.dimY,4]),dtype=DTYPE)


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
        self.grad_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grad_x.data)
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
      self.v = np.zeros(([self.unknowns,self.NSlice,self.par.dimX,self.par.dimY,4]),dtype=DTYPE)
      self.z2 = np.zeros(([self.unknowns,self.NSlice,self.par.dimX,self.par.dimY,8]),dtype=DTYPE)
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
        self.grad_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grad_x.data)
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
        self.grad_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grad_x.data)
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
  def irgn_solve_3D(self,x,iters, data, TV=0):

    x_old = x
    x = clarray.to_device(self.queue,np.require(x,requirements="C"))

    res = data - self.FT(self.step_val[:,None,...]*self.Coils3D) + self.operator_forward(x).get()


    if TV==1:
      x = self.tv_solve_3D(x.get(),res,iters)
      grad = gd.fgrad_3(self.scale_fwd(x),1,1,self.dz)
      self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_3D(x)[:,None,...]*self.Coils3D))**2
              +self.irgn_par.gamma*np.sum(np.abs(grad))
              +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)
      print('Norm M0 grad: %f  norm T1 grad: %f' %(np.linalg.norm(grad[0,...]),np.linalg.norm(grad[1,...])))
      scale = np.linalg.norm(grad[0,...])/np.linalg.norm(grad[1,...])
      if scale == 0 or not np.isfinite(scale):
        self.ratio = self.ratio
      else:
        self.ratio *= scale
    elif TV==0:
       x = self.tgv_solve_3D(x.get(),res,iters)
       x = clarray.to_device(self.queue,x)
       v = clarray.to_device(self.queue,self.v)
       grad = clarray.to_device(self.queue,np.zeros_like(self.z1))
       sym_grad = clarray.to_device(self.queue,np.zeros_like(self.z2))
       grad.add_event(self.f_grad(grad,x,wait_for=grad.events+x.events))
       sym_grad.add_event(self.sym_grad(sym_grad,v,wait_for=sym_grad.events+v.events))
       x = x.get()
       self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_3D(x)[:,None,...]*self.Coils3D))**2
              +self.irgn_par.gamma*np.sum(np.abs(grad.get()-self.v))
              +self.irgn_par.gamma*(2)*np.sum(np.abs(sym_grad.get()))
              +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)
       print('Norm M0 grad: %f  norm T1 grad: %f' %(np.linalg.norm(grad.get()[0,...]),np.linalg.norm(grad.get()[1,...])))
       scale = np.linalg.norm(grad.get()[0,...])/np.linalg.norm(grad.get()[1,...])
       if scale == 0 or not np.isfinite(scale):
         self.ratio = self.ratio
       else:
         self.ratio *= scale
    else:
       x = self.wt_solve_3D(x.get(),res,iters)
       grad = pywt.wavedec2(self.scale_fwd(x),self.wavelet,self.border)
       print('Norm M0 grad: %f  norm T1 grad: %f' %(np.linalg.norm(np.array(grad[len(grad)-1])[:,0,...]),np.linalg.norm(np.array(grad[len(grad)-1])[:,1,...])))
       scale = np.linalg.norm(np.array(grad[len(grad)-1])[:,0,...])/np.linalg.norm(np.array(grad[len(grad)-1])[:,1,...])
       if scale == 0 or not np.isfinite(scale):
         self.ratio = self.ratio
       else:
         self.ratio *= scale
       for j in range(len(grad)):
         grad[j] = np.sum(np.abs(np.array(grad[j])))
       self.fval= (self.irgn_par.lambd/2*np.linalg.norm(data - self.FT(self.model.execute_forward_3D(x)[:,None,...]*self.Coils3D))**2
              +self.irgn_par.gamma*np.abs(np.sum(grad))
              +1/(2*self.irgn_par.delta)*np.linalg.norm((x-x_old).flatten())**2)

    print("-"*80)
    print ("Function value after GN-Step: %f" %(self.fval/(self.irgn_par.lambd*self.NSlice)))

    return x

  def tgv_solve_3D(self, x,res, iters):
    alpha = self.irgn_par.gamma
    beta = self.irgn_par.gamma*2

#    xx = np.zeros_like(x,dtype=DTYPE)
#    yy = np.zeros_like(x,dtype=DTYPE)
#    xx = np.random.random_sample(x.shape).astype(DTYPE)
#    xxcl = clarray.to_device(self.queue,xx)
#    yy = self.operator_adjoint(self.operator_forward(xxcl)).get();
#    for j in range(10):
#       if not np.isclose(np.linalg.norm(yy.flatten()),0):
#           xx = yy/np.linalg.norm(yy.flatten())
#       else:
#           xx = yy
#       xxcl = clarray.to_device(self.queue,xx)
#       yy = self.operator_adjoint(self.operator_forward(xxcl)).get()
#       l1 = np.vdot(yy.flatten(),xx.flatten());
#    L = np.max(np.abs(l1)) ## Lipschitz constant estimate
    L = 0.5*(18.0 + np.sqrt(33))
    print('L: %f'%(L))


    tau = 1/np.sqrt(L)
    tau_new = 0

    self.set_scale(x)
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


    beta_line = 400
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

    Kyk1 = clarray.zeros_like(x)
    Kyk1_new = clarray.zeros_like(x)
    Kyk2 = clarray.zeros_like(z1)
    Kyk2_new = clarray.zeros_like(z1)
    gradx = clarray.zeros_like(z1)
    gradx_xold = clarray.zeros_like(z1)
    symgrad_v = clarray.zeros_like(z2)
    symgrad_v_vold = clarray.zeros_like(z2)

    Axold = self.operator_forward(x)

    Kyk1.add_event(self.bdiv(Kyk1,z1,wait_for=Kyk1.events+z1.events))
    Kyk2.add_event(self.sym_bdiv(Kyk2,z2,wait_for=Kyk2.events+z2.events))

    Kyk1 = self.operator_adjoint(r) - Kyk1#clarray.to_device(self.queue,self.scale_adj(gd.bdiv_1(z1.get())))
    Kyk2 = -z1 - Kyk2#clarray.to_device(self.queue,gd.fdiv_2(z2.get()))
#    print('Kyk1: %f' %(np.linalg.norm(Kyk1.get())))
#    print('Kyk2: %f' %(np.linalg.norm(Kyk2.get())))

#    self.queue.finish()
    for i in range(iters):
      x_new = ((x - tau*(Kyk1))+(tau/delta)*xk)/(1+tau/delta)
      x_new = x_new.get()

      for j in range(len(self.model.constraints)):
        x_new[j,...] = np.maximum(self.model.constraints[j].min,np.minimum(self.model.constraints[j].max,x_new[j,...]))
        if self.model.constraints[j].real:
          x_new[j,...] = np.real(x_new[j,...])
      x_new = clarray.to_device(self.queue,x_new.astype(DTYPE))


      v_new = (v-tau*Kyk2).astype(DTYPE)

      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))

      beta_line = beta_new

#      self.queue.finish()
#      print('xnew: %f' %(np.linalg.norm(x_new.get())))
#      print(x_new.dtype)
#      print(v_new.dtype)
#      gradx = clarray.to_device(self.queue,gd.fgrad_1(self.scale_fwd(x_new.get())))

      gradx.add_event(self.f_grad(gradx,x_new,wait_for=gradx.events+x_new.events))
      gradx_xold.add_event(self.f_grad(gradx_xold,x,wait_for=gradx_xold.events+x.events))



      v_vold = v_new-v
      symgrad_v.add_event(self.sym_grad(symgrad_v,v_new,wait_for=symgrad_v.events+v_new.events))
      symgrad_v_vold.add_event(self.sym_grad(symgrad_v_vold,v,wait_for=symgrad_v_vold.events+v.events))
#      symgrad_v = clarray.to_device(self.queue,gd.sym_bgrad_2(v_new.get()))
#      self.queue.finish()
      gradx_xold = gradx - gradx_xold#clarray.to_device(self.queue,gd.fgrad_1(self.scale_fwd(x.get())))
      symgrad_v_vold = symgrad_v - symgrad_v_vold#clarray.to_device(self.queue,gd.sym_bgrad_2(v.get()))

      Ax = self.operator_forward(x_new)
      Ax_Axold = Ax-Axold
#      print(gradx.dtype)
#      print(gradx_xold.dtype)
#      print(symgrad_v.dtype)
#      print(v_vold.dtype)
#      print(symgrad_v_vold.dtype)
#      self.queue.finish()
#
#      print('gradxnew: %f' %(np.linalg.norm(gradx.get())))
#      print('gradx_xold: %f' %(np.linalg.norm(gradx_xold.get())))
#      print('symgrad_v_vold: %f' %(np.linalg.norm(symgrad_v_vold.get())))

      while True:

        theta_line = tau_new/tau
#        print('z1: %f' %(np.linalg.norm(z1.get())))
        z1_new = (z1 + beta_line*tau_new*( gradx + theta_line*gradx_xold
                                          - v_new - theta_line*v_vold  )).get()
        z1_new = clarray.to_device(self.queue,(z1_new/np.maximum(1,(np.sqrt(np.sum(np.abs(z1_new)**2,axis=(0,-1),keepdims=True))/alpha))).astype(DTYPE))
#        print('z1_new: %f' %(np.linalg.norm(z1_new.get())))
#        print('Kyk1_new: %f' %(np.linalg.norm(Kyk1_new.get())))
#        print(z1_new.dtype)
        Kyk1_new.add_event(self.bdiv(Kyk1_new,z1_new,wait_for=Kyk1_new.events+z1_new.events))
#        self.queue.finish()
#        print('Kyk1_new: %f' %(np.linalg.norm(Kyk1_new.get())))
        z2_new = z2 + beta_line*tau_new*( symgrad_v + theta_line*symgrad_v_vold )
        z2_new = z2_new.get()
        scal = np.sqrt( np.sum(np.sum(np.abs(z2_new[...,0:3])**2,-1) + np.sum(2*np.abs(z2_new[...,3:6])**2,-1),axis=0,keepdims=True) )
        scal = np.maximum(1,scal/(beta))[...,None]

        z2_new = clarray.to_device(self.queue,(z2_new/scal).astype(DTYPE))
        Kyk2_new.add_event(self.sym_bdiv(Kyk2_new,z2_new,wait_for=Kyk2_new.events+z2_new.events))

        tmp = Ax+theta_line*Ax_Axold
#        self.queue.finish()
        r_new = ( r  + beta_line*tau_new*(tmp - res))/(1+beta_line*tau_new/self.irgn_par.lambd)

#        self.queue.finish()
#        print('Kyk1_new: %f' %(np.linalg.norm(Kyk1_new.get())))
#        print('Kyk2_new: %f' %(np.linalg.norm(Kyk2_new.get())))

        Kyk1_new = self.operator_adjoint(r_new) - Kyk1_new#clarray.to_device(self.queue,self.scale_adj(gd.bdiv_1(z1_new.get())))
        Kyk2_new = -z1_new -Kyk2_new#clarray.to_device(self.queue,gd.fdiv_2(z2_new.get()))
#        print(Kyk1_new.dtype)
#        print(Kyk2_new.dtype)
#        self.queue.finish()
        ynorm = ((clarray.vdot(r_new-r,r_new-r)+clarray.vdot(z1_new-z1,z1_new-z1)+clarray.vdot(z2_new-z2,z2_new-z2))**(1/2)).real
        lhs = np.sqrt(beta_line)*tau_new*((clarray.vdot(Kyk1_new-Kyk1,Kyk1_new-Kyk1)+clarray.vdot(Kyk2_new-Kyk2,Kyk2_new-Kyk2))**(1/2)).real
        self.queue.finish()
        if lhs <= ynorm*delta_line:
            break
        else:
#          print('lhs: %f \n rhs: %f' %(lhs.get(), ynorm.get()*delta_line))
          tau_new = tau_new*mu_line

      (Kyk1, Kyk1_new, Kyk2, Kyk2_new, Axold, Ax, z1, z1_new, z2, z2_new, r, r_new) =\
      (Kyk1_new, Kyk1, Kyk2_new, Kyk2, Ax, Axold, z1_new, z1, z2_new, z2, r_new, r)
      tau =  (tau_new)


      if not np.mod(i,20):

        self.model.plot_unknowns(x_new.get()[:,0,...],True)
        primal_new= (self.irgn_par.lambd/2*clarray.vdot(Axold-res,Axold-res)+alpha*clarray.sum(abs((gradx[:self.unknowns_TGV]-v))) +beta*clarray.sum(abs(symgrad_v)) + 1/(2*delta)*clarray.vdot(x_new-xk,x_new-xk)).real

        dual = (-delta/2*clarray.vdot(-Kyk1,-Kyk1)- clarray.vdot(xk,(-Kyk1)) + clarray.sum(Kyk2)
                  - 1/(2*self.irgn_par.lambd)*clarray.vdot(r,r) - clarray.vdot(res,r)).real

        gap = np.abs(primal_new - dual)
        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<(self.irgn_par.lambd*self.NSlice)*self.irgn_par.tol:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,abs(primal-primal_new).get()/(self.irgn_par.lambd*self.NSlice)))
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          return x_new.get()
        if (gap > gap_min*self.irgn_par.stag) and i>1:
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2.get()
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x.get()
        if np.abs(gap - gap_min)<(self.irgn_par.lambd*self.NSlice)*self.irgn_par.tol and i>1:
          self.v = v_new.get()
          self.r = r.get()
          self.z1 = z1.get()
          self.z2 = z2
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,abs(gap - gap_min).get()/(self.irgn_par.lambd*self.NSlice)))
          return x_new.get()
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        print("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f "%(i,primal.get()/(self.irgn_par.lambd*self.NSlice),dual.get()/(self.irgn_par.lambd*self.NSlice),gap.get()/(self.irgn_par.lambd*self.NSlice)))
#        print("Norm of primal gradient: %.3e"%(np.linalg.norm(Kyk1)+np.linalg.norm(Kyk2)))
#        print("Norm of dual gradient: %.3e"%(np.linalg.norm(tmp)+np.linalg.norm(gradx[:self.unknowns_TGV] + theta_line*gradx_xold[:self.unknowns_TGV]
#                                          - v_new - theta_line*v_vold)+np.linalg.norm( symgrad_v + theta_line*symgrad_v_vold)))

      (x, x_new) = (x_new, x)
      (v, v_new) = (v_new, v)
#      for j in range(self.par.unknowns_TGV):
#        self.scale_2D[j,...] = np.linalg.norm(x[j,...])
    self.v = v.get()
    self.r = r.get()
    self.z1 = z1.get()
    self.z2 = z2.get()
    return x.get()
