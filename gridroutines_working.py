import numpy as np
from calckbkernel import calckbkernel
import pyopencl as cl
import pyopencl.array as clarray

import reikna.cluda as cluda
from reikna.fft import FFT, FFTShift


class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build("-cl-mad-enable -cl-fast-relaxed-math")
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel


class gridding:
  def __init__(self, ctx,queue, kwidth,overgridfactor,G,fft_size,fft_dim,traj,dcf,gridsize,klength=400, DTYPE=np.complex128,DTYPE_real=np.float64):
    (self.kerneltable,self.kerneltable_FT,self.u) = calckbkernel(kwidth,overgridfactor,G,klength)
    self.kernelpoints = self.kerneltable.size
    self.ctx = ctx
    self.queue = queue
    self.api = cluda.ocl_api()
    self.fft_scale = DTYPE_real(fft_size[-1])
    self.deapo = 1/self.kerneltable_FT.astype(DTYPE_real)
    self.kwidth = kwidth/2
    self.cl_kerneltable = cl.Buffer(self.queue[0].context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.kerneltable.astype(DTYPE_real).data)
    self.deapo_cl = cl.Buffer(self.queue[0].context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.deapo.data)
    self.dcf = clarray.to_device(self.queue[0],dcf)
    self.traj = clarray.to_device(self.queue[0],traj)
    self.tmp_fft_array = []
    self.fft2 = []
    self.fftshift = []
    for j in range(len(queue)):
      self.thr = self.api.Thread(queue[j])
      self.tmp_fft_array.append(clarray.zeros(self.queue[j],fft_size,dtype=DTYPE))
      fft = FFT(self.tmp_fft_array[j], axes=fft_dim)
      fftshift = FFTShift(self.tmp_fft_array[j],axes=fft_dim)
      self.fftshift.append(fftshift.compile(self.thr))
      self.fft2.append(fft.compile(self.thr))
    self.gridsize = gridsize
    self.DTYPE = DTYPE
    self.DTYPE_real = DTYPE_real
    if DTYPE==np.complex128:
      print('Using double precission')
      self.prg = Program(self.ctx, r"""

      #pragma OPENCL EXTENSION cl_khr_fp64: enable
      #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
      void AtomicAdd(__global double *val, double delta) {
        union {
          double f;
          ulong  i;
        } old;
        union {
          double f;
          ulong  i;
        } new;
        do {
          old.f = *val;
          new.f = old.f + delta;
        } while (atom_cmpxchg ( (volatile __global ulong *)val, old.i, new.i) != old.i);
      }

  __kernel void make_complex(__global double2 *out,__global double *re, __global double* im)
  {
    size_t k = get_global_id(0);

    out[k].s0 = re[k];
    out[k].s1 = im[k];

  }
  __kernel void deapo_adj(__global double2 *out, __global double2 *in, __constant double *deapo, const int dim, const double scale)
  {
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+dim/4;
    size_t M = dim;
    size_t n = y+dim/4;
    size_t N = dim;

    out[k*X*Y+y*X+x] = in[k*N*M+n*M+m] * deapo[y] * deapo[x]* scale;

  }
  __kernel void deapo_fwd(__global double2 *out, __global double2 *in, __constant double *deapo, const int dim, const double scale)
  {
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+dim/4;
    size_t M = dim;
    size_t n = y+dim/4;
    size_t N = dim;


    out[k*N*M+n*M+m] = in[k*X*Y+y*X+x] * deapo[y] * deapo[x] * scale;
  }

  __kernel void zero_tmp(__global double2 *tmp)
  {
    size_t x = get_global_id(0);
    tmp[x] = 0.0f;

  }

  __kernel void grid_lut(__global double *sg, __global double2 *s, __global double2 *kpos, const int gridsize, const double kwidth, __global double *dcf, __constant double* kerneltable, const int nkernelpts )
  {
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind;
    double kx, ky;
    double fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;


    double* ptr, pti;
    double2 kdat = s[k+kDim*n+kDim*NDim*scan]*(double2)(dcf[k],dcf[k]);



    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;


    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    if (ixmin < 0)
      ixmin=0;
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    if (ixmax >= gridsize)
      ixmax=gridsize-1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    if (iymin < 0)
      iymin=0;
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;
    if (iymax >= gridsize)
      iymax=gridsize-1;

  	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
  		dkx = (double)(gcount1-gridcenter) / (double)gridsize  - kx;
  		for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
  			{
          dky = (double)(gcount2-gridcenter) / (double)gridsize - ky;

  			dk = sqrt(dkx*dkx+dky*dky);

  			if (dk < kwidth)
  			    {

  			    fracind = dk/kwidth*(double)(nkernelpts-1);
  			    kernelind = (int)fracind;
  			    fracdk = fracind-(double)kernelind;

  			    kern = kerneltable[(int)kernelind]*(1-fracdk)+
  			    		kerneltable[(int)kernelind+1]*fracdk;

             AtomicAdd(&(sg[2*(gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan)]),(kern * kdat.s0));
             AtomicAdd(&(sg[2*(gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan)+1]),(kern * kdat.s1));
  			    }
  			}
  		}
  }


  __kernel void invgrid_lut(__global double2 *s, __global double2 *sg, __global double2 *kpos, const int gridsize, const double kwidth, __global double *dcf, __constant double* kerneltable, const int nkernelpts )
  {
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind;
    double kx, ky;
    double fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;

    double2 tmp_dat = 0.0f;


    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;

    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    if (ixmin < 0)
      ixmin=0;
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    if (ixmax >= gridsize)
      ixmax=gridsize-1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    if (iymin < 0)
      iymin=0;
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;
    if (iymax >= gridsize)
      iymax=gridsize-1;

  	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
  		dkx = (double)(gcount1-gridcenter) / (double)gridsize  - kx;
  		for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
  			{
          dky = (double)(gcount2-gridcenter) / (double)gridsize - ky;

  			dk = sqrt(dkx*dkx+dky*dky);

  			if (dk < kwidth)
  			    {

  			    fracind = dk/kwidth*(double)(nkernelpts-1);
  			    kernelind = (int)fracind;
  			    fracdk = fracind-(double)kernelind;

  			    kern = kerneltable[(int)kernelind]*(1-fracdk)+
  			    		kerneltable[(int)kernelind+1]*fracdk;
             tmp_dat += (double2)(kern,kern)*sg[gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan];
  			    }
  			}
  		}
    s[k+kDim*n+kDim*NDim*scan]= tmp_dat*(double2)(dcf[k],dcf[k]);
  }
                         """)
    else:
      print('Using single precission')
      self.prg = Program(self.ctx, r"""
 void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

  __kernel void make_complex(__global float2 *out,__global float *re, __global float* im)
  {
    size_t k = get_global_id(0);

    out[k].s0 = re[k];
    out[k].s1 = im[k];

  }
  __kernel void deapo_adj(__global float2 *out, __global float2 *in, __constant float *deapo, const int dim, const float scale)
  {
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+dim/4;
    size_t M = dim;
    size_t n = y+dim/4;
    size_t N = dim;

    out[k*X*Y+y*X+x] = in[k*N*M+n*M+m] * deapo[y]* deapo[x] * scale;

  }
  __kernel void deapo_fwd(__global float2 *out, __global float2 *in, __constant float *deapo, const int dim, const float scale)
  {
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+dim/4;
    size_t M = dim;
    size_t n = y+dim/4;
    size_t N = dim;


    out[k*N*M+n*M+m] = in[k*X*Y+y*X+x] * deapo[y]* deapo[x] * scale;
  }

  __kernel void zero_tmp(__global float2 *tmp)
  {
    size_t x = get_global_id(0);
    tmp[x] = 0.0f;

  }

  __kernel void grid_lut(__global float *sg, __global float2 *s, __global float2 *kpos, const int gridsize, const float kwidth, __global float *dcf, __constant float* kerneltable, const int nkernelpts )
  {
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind;
    float kx, ky;
    float fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;


    float* ptr, pti;
    float2 kdat = s[k+kDim*n+kDim*NDim*scan]*(float2)(dcf[k],dcf[k]);



    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;


    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    if (ixmin < 0)
      ixmin=0;
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    if (ixmax >= gridsize)
      ixmax=gridsize-1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    if (iymin < 0)
      iymin=0;
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;
    if (iymax >= gridsize)
      iymax=gridsize-1;

  	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
  		dkx = (float)(gcount1-gridcenter) / (float)gridsize  - kx;
  		for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
  			{
          dky = (float)(gcount2-gridcenter) / (float)gridsize - ky;

  			dk = sqrt(dkx*dkx+dky*dky);

  			if (dk < kwidth)
  			    {

  			    fracind = dk/kwidth*(float)(nkernelpts-1);
  			    kernelind = (int)fracind;
  			    fracdk = fracind-(float)kernelind;

  			    kern = kerneltable[(int)kernelind]*(1-fracdk)+
  			    		kerneltable[(int)kernelind+1]*fracdk;

             AtomicAdd(&(sg[2*(gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan)]),(kern * kdat.s0));
             AtomicAdd(&(sg[2*(gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan)+1]),(kern * kdat.s1));
  			    }
  			}
  		}
  }


  __kernel void invgrid_lut(__global float2 *s, __global float2 *sg, __global float2 *kpos, const int gridsize, const float kwidth, __global float *dcf, __constant float* kerneltable, const int nkernelpts )
  {
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind;
    float kx, ky;
    float fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;

    float2 tmp_dat = 0.0;


    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;

    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    if (ixmin < 0)
      ixmin=0;
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    if (ixmax >= gridsize)
      ixmax=gridsize-1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    if (iymin < 0)
      iymin=0;
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;
    if (iymax >= gridsize)
      iymax=gridsize-1;

  	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
  		dkx = (float)(gcount1-gridcenter) / (float)gridsize  - kx;
  		for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
  			{
          dky = (float)(gcount2-gridcenter) / (float)gridsize - ky;

  			dk = sqrt(dkx*dkx+dky*dky);

  			if (dk < kwidth)
  			    {

  			    fracind = dk/kwidth*(float)(nkernelpts-1);
  			    kernelind = (int)fracind;
  			    fracdk = fracind-(float)kernelind;

  			    kern = kerneltable[(int)kernelind]*(1-fracdk)+
  			    		kerneltable[(int)kernelind+1]*fracdk;
             tmp_dat += (float2)(kern,kern)*sg[gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan];
  			    }
  			}
  		}
    s[k+kDim*n+kDim*NDim*scan]= tmp_dat*(float2)(dcf[k],dcf[k]);
  }
                         """)
  def __del__(self):
    del self.traj
    del self.dcf
    del self.tmp_fft_array
    del self.cl_kerneltable
    del self.fft2
    del self.fftshift
    del self.deapo_cl

  def adj_NUFFT(self,sg,s,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]
    ### Zero tmp arrays
    self.tmp_fft_array[idx].add_event(self.prg.zero_tmp(queue, (self.tmp_fft_array[idx].size,),None,self.tmp_fft_array[idx].data))
    ### Grid k-space
    self.tmp_fft_array[idx].add_event(self.prg.grid_lut(queue,(s.shape[0],s.shape[1]*s.shape[2],s.shape[-2]*self.gridsize),None,self.tmp_fft_array[idx].data,s.data,self.traj.data, np.int32(self.gridsize),
                             self.DTYPE_real(self.kwidth/self.gridsize),self.dcf.data,self.cl_kerneltable,np.int32(self.kernelpoints),
                             wait_for = sg.events+s.events+wait_for+self.tmp_fft_array[idx].events))
    ### FFT
    self.fftshift[idx](self.tmp_fft_array[idx],self.tmp_fft_array[idx])
    self.fft2[idx](self.tmp_fft_array[idx],self.tmp_fft_array[idx],True)
    self.fftshift[idx](self.tmp_fft_array[idx],self.tmp_fft_array[idx])
    ### Deapodization and Scaling
    return self.prg.deapo_adj(queue,(sg.shape[0]*sg.shape[1]*sg.shape[2],sg.shape[3],sg.shape[4]),None,sg.data,self.tmp_fft_array[idx].data,self.deapo_cl, np.int32(self.tmp_fft_array[idx].shape[-1]),self.DTYPE_real(self.fft_scale),wait_for=self.tmp_fft_array[idx].events)





  def fwd_NUFFT(self,s,sg,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]
    ### Zero tmp arrays
    self.tmp_fft_array[idx].add_event(self.prg.zero_tmp(queue, (self.tmp_fft_array[idx].size,),None,self.tmp_fft_array[idx].data))
    ### Deapodization and Scaling
    self.tmp_fft_array[idx].add_event(self.prg.deapo_fwd(queue,(sg.shape[0]*sg.shape[1]*sg.shape[2],sg.shape[3],sg.shape[4]),None,self.tmp_fft_array[idx].data,sg.data,self.deapo_cl, np.int32(self.tmp_fft_array[idx].shape[-1]),self.DTYPE_real(1/self.fft_scale),wait_for = s.events+sg.events+wait_for+self.tmp_fft_array[idx].events))
    ### FFT
    self.fftshift[idx](self.tmp_fft_array[idx],self.tmp_fft_array[idx])
    self.fft2[idx](self.tmp_fft_array[idx],self.tmp_fft_array[idx])
    self.fftshift[idx](self.tmp_fft_array[idx],self.tmp_fft_array[idx])
    ### Resample on Spoke
    return self.prg.invgrid_lut(queue,(s.shape[0],s.shape[1]*s.shape[2],s.shape[-2]*self.gridsize),None,s.data,self.tmp_fft_array[idx].data,self.traj.data, np.int32(self.gridsize), self.DTYPE_real(self.kwidth/self.gridsize),self.dcf.data,self.cl_kerneltable,np.int32(self.kernelpoints),
                             wait_for = s.events+sg.events+wait_for+self.tmp_fft_array[idx].events)

