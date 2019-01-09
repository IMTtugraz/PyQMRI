#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:30:22 2018

@author: omaier
"""
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")







import numpy as np

import pyopencl as cl
import pyopencl.array as clarray

DTYPE = np.complex128
################################################################################
### Create OpenCL Context and Queues ###########################################
################################################################################
platforms = cl.get_platforms()

##### platfroms includes CPU and GPU if OpenCL drivers are set up
ctx = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platforms[1])])


##### setting up a command queue for device 0 of platform 1 (GPU) with some additional flags
queue = (cl.CommandQueue(ctx,platforms[1].get_devices()[0],properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE | cl.command_queue_properties.PROFILING_ENABLE))




class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build("-cl-mad-enable -cl-fast-relaxed-math")
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel

class MyOpenCLObject:
  def __init__(self,ctx):
    if DTYPE == np.float32:
      self.prg = Program(ctx, r"""
      __kernel void gradient(__global float4 *grad, __global float *u) {

      // Global Grid Size
      size_t Nx = get_global_size(2), Ny = get_global_size(1);
      size_t NSl = get_global_size(0);

      // Current Index
      size_t x = get_global_id(2), y = get_global_id(1);
      size_t k = get_global_id(0);

      // Reconstructed Pixel index
      size_t i = k*Nx*Ny+Nx*y + x;


      // gradient using forward differences
      grad[i] = (float4)(-u[i],-u[i],-u[i],0.0f);

      // X Direction
      if (x < Nx-1)
      { grad[i].x += u[i+1];}
      // take care of boundary
      else
      { grad[i].x = 0.0f;}

      // Y Direction
      if (y < Ny-1)
      { grad[i].y += u[i+Nx];}
      // take care of boundary
      else
      { grad[i].y = 0.0f;}

      // Z Direction
      if (k < NSl-1)
      { grad[i].z += u[i+Nx*Ny];}
      else
      // take care of boundary
      { grad[i].z = 0.0f;}
      }



  __kernel void divergence(__global float *div, __global float4 *p) {
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);

    size_t i = k*Nx*Ny+Nx*y + x;


       // divergence
       float4 val = p[i];
       if (x == Nx-1)
       {
           //real
           val.x = 0.0f;
       }
       if (x > 0)
       {
           //real
           val.x -= p[i-1].x;
       }
       if (y == Ny-1)
       {
           //real
           val.y = 0.0f;
       }
       if (y > 0)
       {
           //real
           val.y -= p[i-Nx].y;
       }
       if (k == NSl-1)
       {
           //real
           val.z = 0.0f;
       }
       if (k > 0)
       {
           //real
           val.z -= p[i-Nx*Ny].z;
       }
       div[i] = val.x+val.y+val.z;
  }""")
    elif DTYPE==np.float64:
      self.prg = Program(ctx, r"""
__kernel void gradient(__global double4 *grad, __global double *u) {

    // Global Grid Size
    size_t Nx = get_global_size(2), Ny = get_global_size(1);
    size_t NSl = get_global_size(0);

    // Current Index
    size_t x = get_global_id(2), y = get_global_id(1);
    size_t k = get_global_id(0);

    // Reconstructed Pixel index
    size_t i = k*Nx*Ny+Nx*y + x;


    // gradient using forward differences
    grad[i] = (double4)(-u[i],-u[i],-u[i],0.0f);

    // X Direction
    if (x < Nx-1)
    { grad[i].x += u[i+1];}
    // take care of boundary
    else
    { grad[i].x = 0.0f;}

    // Y Direction
    if (y < Ny-1)
    { grad[i].y += u[i+Nx];}
    // take care of boundary
    else
    { grad[i].y = 0.0f;}

    // Z Direction
    if (k < NSl-1)
    { grad[i].z += u[i+Nx*Ny];}
    else
    // take care of boundary
    { grad[i].z = 0.0f;}
    }



__kernel void divergence(__global double *div, __global double4 *p) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);

  size_t i = k*Nx*Ny+Nx*y + x;


     // divergence
     double4 val = p[i];
     if (x == Nx-1)
     {
         //real
         val.x = 0.0f;
     }
     if (x > 0)
     {
         //real
         val.x -= p[i-1].x;
     }
     if (y == Ny-1)
     {
         //real
         val.y = 0.0f;
     }
     if (y > 0)
     {
         //real
         val.y -= p[i-Nx].y;
     }
     if (k == NSl-1)
     {
         //real
         val.z = 0.0f;
     }
     if (k > 0)
     {
         //real
         val.z -= p[i-Nx*Ny].z;
     }
     div[i] = val.x+val.y+val.z;
}
    """)
    elif DTYPE==np.complex64:
      self.prg = Program(ctx, r"""
__kernel void gradient(__global float8 *grad, __global float2 *u) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;



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

}
__kernel void divergence(__global float2 *div, __global float8 *p) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

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
}

                         """)
    else:
      self.prg = Program(ctx, r"""

__kernel void gradient(__global double8 *grad, __global double2 *u) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;



     // gradient
     grad[i] = (double8)(-u[i],-u[i],-u[i],0.0f,0.0f);
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

}
__kernel void divergence(__global double2 *div, __global double8 *p) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

     // divergence
     double8 val = p[i];
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
}

                         """)



  def f_grad(self,grad, u, wait_for=[]):
    return self.prg.gradient(grad.queue, u.shape, None, grad.data, u.data,
                  wait_for=grad.events + u.events + wait_for)
  def bdiv(self,div, u, wait_for=[]):
    return self.prg.divergence(div.queue, u.shape[:-1], None, div.data, u.data,
                wait_for=div.events + u.events + wait_for)

##### Create Object with Gradient and Divergence
oclobj = MyOpenCLObject(ctx)

x = np.ones((4,256,256))
data = np.ones((4,256,256,4))
##### Generate some randomely initialized arrays for image and gradient
xx = np.array((np.random.random_sample(np.shape(x))+1j*np.random.random_sample(np.shape(x)))).astype(DTYPE)
yy = np.array((np.random.random_sample(np.shape(data))+1j*np.random.random_sample(np.shape(data)))).astype(DTYPE)
#### Send arrays to GPU
xx = clarray.to_device(queue,xx)
yy = clarray.to_device(queue,yy)
#### Allocate arrays for Output
div = clarray.zeros_like(xx)
grad = clarray.zeros_like(yy)
#### Call GPU functions on GPU Arrays
oclobj.f_grad(grad,xx)
oclobj.bdiv(div,yy)

#### Get results back to CPU Memory
a = clarray.vdot(xx,-div[...]).get()
b = clarray.vdot(grad,yy).get()
adj = abs(a-b)
print("test deriv-op-adjointness:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj_absolute: %.2E \n adj_relative: %.2E"  % (a.real,a.imag,b.real,b.imag,adj,adj/yy.size))


